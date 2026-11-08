import pickle

import lightgbm as lgb
import numpy as np
import pytest

from s2apler.native_scorer import NativeLightGBMClassifier


@pytest.fixture(scope="module")
def classifier():
    rng = np.random.default_rng(0)
    X = rng.random((2000, 18))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 3]) > 1).astype(int)
    return lgb.LGBMClassifier(n_estimators=50, num_leaves=104, random_state=0, verbose=-1).fit(X, y)


def rows():
    rng = np.random.default_rng(1)
    X = rng.random((5000, 18))
    X[rng.random(X.shape) < 0.2] = np.nan
    return X


@pytest.mark.parametrize("num_threads", [1, 4])
def test_probabilities_match_lightgbm_exactly(classifier, num_threads):
    native = NativeLightGBMClassifier.from_lgbm(classifier, num_threads=num_threads)
    X = rows()
    assert np.array_equal(native.predict_proba(X), classifier.predict_proba(X))


def test_accepts_float32_and_non_contiguous_rows(classifier):
    native = NativeLightGBMClassifier.from_lgbm(classifier, num_threads=1)
    X = rows()
    assert np.array_equal(native.predict_proba(X[::2]), classifier.predict_proba(X[::2]))
    X32 = X.astype(np.float32)
    assert np.array_equal(native.predict_proba(X32), classifier.predict_proba(X32.astype(np.float64)))


def test_survives_pickling(classifier):
    native = pickle.loads(pickle.dumps(NativeLightGBMClassifier.from_lgbm(classifier, num_threads=2)))
    assert native.num_threads == 2
    assert np.array_equal(native.predict_proba(rows()), classifier.predict_proba(rows()))


def test_rejects_non_binary_classifier():
    rng = np.random.default_rng(0)
    multi = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(rng.random((300, 4)), rng.integers(0, 3, 300))
    with pytest.raises(ValueError, match="binary"):
        NativeLightGBMClassifier.from_lgbm(multi, num_threads=1)
