"""Scores trained LightGBM classifiers with s2and's native Rust booster.

`RustLightGBMBooster` reads LightGBM's text model format and returns probabilities that are
bit-identical to LightGBM's own. It is about 15x faster on one thread and 10x faster on 20
threads for S2APLER's production model.

Many rows of a request fall in the same bins of every split feature. Such rows take the same
path through every tree, so `predict_proba` scores each distinct one once.
"""

import logging
import math
import weakref
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import lightgbm as lgb
import numpy as np
from s2and.runtime import load_s2and_rust_extension

from s2apler.runtime import load_s2apler_rust_extension

logger = logging.getLogger("s2apler")

# LightGBM's decision_type bits: 1 marks a categorical split; bits 2-3 hold the missing type.
_CATEGORICAL_SPLIT = 1
_MISSING_ZERO = 1


def split_thresholds(model_text: str) -> Optional[List[List[float]]]:
    """Lists each feature's distinct split thresholds, when rows can be grouped by them.

    Rows that fall in the same bins of every feature take the same path through every tree of
    a numerical model; `distinct_bin_rows` keys rows as s2and's lockstep booster does. That
    breaks for categorical splits, splits that treat zero as missing, and linear trees, whose
    leaves read raw values; for those, this returns None.

    Parameters
    ----------
    model_text: str
        the booster in LightGBM's text model format

    Returns
    -------
    Optional[List[List[float]]]: sorted thresholds per feature, or None
    """
    feature_count = None
    thresholds: Dict[int, Set[float]] = defaultdict(set)
    tree: Dict[str, str] = {}

    def add_tree() -> bool:
        if tree.get("num_cat", "0") != "0" or tree.get("is_linear", "0") != "0":
            return False
        if "split_feature" not in tree:  # a single-leaf tree
            return True
        features = tree["split_feature"].split()
        values = tree["threshold"].split()
        decisions = tree["decision_type"].split()
        for feature, value, decision in zip(features, values, decisions, strict=True):
            decision_type = int(decision)
            if decision_type & _CATEGORICAL_SPLIT or (decision_type >> 2) & 3 == _MISSING_ZERO:
                return False
            threshold = float(value)
            if math.isnan(threshold):  # s2and's key table needs numeric thresholds
                return False
            thresholds[int(feature)].add(threshold)
        return True

    in_trees = False
    for line in model_text.splitlines():
        if line.startswith("max_feature_idx="):
            feature_count = int(line.split("=", 1)[1]) + 1
        elif line.startswith("Tree="):
            if in_trees and not add_tree():
                return None
            in_trees, tree = True, {}
        elif line.startswith("end of trees"):
            break
        elif in_trees and "=" in line:
            key, value = line.split("=", 1)
            tree[key] = value
    if (in_trees and not add_tree()) or feature_count is None:
        return None
    return [sorted(thresholds[feature]) for feature in range(feature_count)]


class NativeLightGBMClassifier:
    """Drop-in `predict_proba` for a fitted binary `LGBMClassifier`, scored natively.

    Parameters
    ----------
    model_text: str
        the booster in LightGBM's text model format
    num_threads: int
        how many threads to score with
    """

    def __init__(self, model_text: str, num_threads: int):
        if num_threads < 1:
            raise ValueError(f"num_threads must be at least 1, got {num_threads}")
        self.model_text = model_text
        self.num_threads = num_threads
        # s2and's loader checks the extension version and raises clearly if it's missing.
        self._booster = load_s2and_rust_extension().RustLightGBMBooster.from_string(model_text)
        self._thresholds = split_thresholds(model_text)

    @classmethod
    def from_lgbm(cls, classifier: Any, num_threads: int) -> "NativeLightGBMClassifier":
        """Builds a native scorer from a fitted `LGBMClassifier`.

        Parameters
        ----------
        classifier: LGBMClassifier
            a fitted binary classifier
        num_threads: int
            how many threads to score with

        Returns
        -------
        NativeLightGBMClassifier: the native scorer
        """
        if list(classifier.classes_) != [0, 1]:
            raise ValueError(f"Expected binary classes [0, 1], got {list(classifier.classes_)}")
        # LightGBM can stop summing trees early at prediction time; the native booster always
        # sums every tree, so it would return different probabilities.
        params = {**(classifier.booster_.params or {}), **classifier.get_params()}
        if params.get("pred_early_stop"):
            raise ValueError("pred_early_stop is set; the native booster can't reproduce early-stopped scores")
        return cls(classifier.booster_.model_to_string(), num_threads)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities with columns [P(0), P(1)], like `LGBMClassifier`.

        Parameters
        ----------
        X: np.ndarray
            feature rows

        Returns
        -------
        np.ndarray: an (n, 2) array of class probabilities
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        grouped = None
        if self._thresholds is not None and X.ndim == 2 and X.shape[1] == len(self._thresholds) and len(X) > 1:
            grouped = load_s2apler_rust_extension().distinct_bin_rows(X, self._thresholds, num_threads=self.num_threads)
        if grouped is None:
            positive = self._booster.predict_proba_positive(X, num_threads=self.num_threads)
        else:
            first_rows, groups = grouped
            # Rows in one group take the same path through every tree, so they score the same.
            distinct = np.ascontiguousarray(X[first_rows])
            positive = self._booster.predict_proba_positive(distinct, num_threads=self.num_threads)[groups]
        # LGBMClassifier stacks 1 - p and p the same way, so both columns match it exactly.
        return np.vstack((1.0 - positive, positive)).transpose()

    def __getstate__(self) -> Dict[str, Any]:
        # The native booster can't be pickled; rebuild it from the model text instead.
        return {"model_text": self.model_text, "num_threads": self.num_threads}

    def __setstate__(self, state: Dict[str, Any]):
        self.__init__(state["model_text"], state["num_threads"])


# Native scorers by LightGBM booster. Converting dumps the whole model as text (about 18 MB for
# the production model), so do it once per fit. Each fit makes a new booster, so a refit
# classifier gets a fresh scorer, and the cache never keeps a booster alive.
_NATIVE_SCORERS: "weakref.WeakKeyDictionary[Any, Optional[NativeLightGBMClassifier]]" = weakref.WeakKeyDictionary()


def scoring_classifier(classifier: Any, num_threads: int) -> Any:
    """Returns what to call `predict_proba` on: the native booster when it is exact, else `classifier`.

    A fitted binary `LGBMClassifier` scores natively, with identical probabilities. Other
    classifiers, and LightGBM models the native booster can't reproduce (multiclass, or with
    prediction-time early stopping), score with their own `predict_proba`.

    Parameters
    ----------
    classifier: Any
        a fitted classifier
    num_threads: int
        how many threads the native booster scores with

    Returns
    -------
    Any: an object with `predict_proba`
    """
    if not isinstance(classifier, lgb.LGBMClassifier):
        return classifier
    booster = classifier.booster_
    if booster not in _NATIVE_SCORERS:
        try:
            _NATIVE_SCORERS[booster] = NativeLightGBMClassifier.from_lgbm(classifier, num_threads)
        except ValueError as err:
            logger.info(f"Scoring with LightGBM, not the native booster: {err}")
            _NATIVE_SCORERS[booster] = None
    native = _NATIVE_SCORERS[booster]
    if native is None:
        return classifier
    native.num_threads = num_threads
    return native
