"""Scores trained LightGBM classifiers with s2and's native Rust booster.

`RustLightGBMBooster` reads LightGBM's text model format and returns probabilities that are
bit-identical to LightGBM's own. It is about 15x faster on one thread and 10x faster on 20
threads for S2APLER's production model.
"""

from typing import Any, Dict

import numpy as np
from s2and.runtime import load_s2and_rust_extension


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
        positive = self._booster.predict_proba_positive(
            np.ascontiguousarray(X, dtype=np.float64), num_threads=self.num_threads
        )
        # LGBMClassifier stacks 1 - p and p the same way, so both columns match it exactly.
        return np.vstack((1.0 - positive, positive)).transpose()

    def __getstate__(self) -> Dict[str, Any]:
        # The native booster can't be pickled; rebuild it from the model text instead.
        return {"model_text": self.model_text, "num_threads": self.num_threads}

    def __setstate__(self, state: Dict[str, Any]):
        self.__init__(state["model_text"], state["num_threads"])
