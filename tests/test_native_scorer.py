import os
import pickle

import lightgbm as lgb
import numpy as np
import pytest

from s2apler.native_scorer import NativeLightGBMClassifier

PROD_MODEL = "data/prod_model.pickle"


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


def edge_rows(classifier, n_features, seed):
    """Rows at every split threshold and its neighbours, plus NaN, infinities and near-zero values."""
    thresholds = []
    for line in classifier.booster_.model_to_string().splitlines():
        if line.startswith("threshold="):
            thresholds += [float(t) for t in line.split("=", 1)[1].split()]
    thresholds = np.unique(np.array(thresholds))
    near = np.concatenate([thresholds, np.nextafter(thresholds, np.inf), np.nextafter(thresholds, -np.inf)])
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-36, -1e-36, 1e-35, -1e-35, 5e-324, 1e300, -1e300])
    values = np.concatenate([near, specials])
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(20000, n_features))


def test_matches_lightgbm_on_splits_without_missing_values():
    # Training without NaN gives missing-type-None splits, where LightGBM maps NaN to 0.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    classifier = lgb.LGBMClassifier(n_estimators=40, num_leaves=31, random_state=0, verbose=-1).fit(X, y)
    native = NativeLightGBMClassifier.from_lgbm(classifier, num_threads=2)
    rows = edge_rows(classifier, 6, seed=3)
    assert np.array_equal(native.predict_proba(rows), np.asarray(classifier.predict_proba(rows)))


def test_matches_lightgbm_on_edge_values_with_nan_splits(classifier):
    native = NativeLightGBMClassifier.from_lgbm(classifier, num_threads=2)
    rows = edge_rows(classifier, 18, seed=4)
    assert np.array_equal(native.predict_proba(rows), classifier.predict_proba(rows))


def test_rejects_prediction_time_early_stopping(classifier):
    early = pickle.loads(pickle.dumps(classifier))
    early.set_params(pred_early_stop=True, pred_early_stop_freq=1, pred_early_stop_margin=0.1)
    with pytest.raises(ValueError, match="pred_early_stop"):
        NativeLightGBMClassifier.from_lgbm(early, num_threads=1)


@pytest.mark.parametrize("num_threads", [0, -1])
def test_rejects_non_positive_thread_counts(classifier, num_threads):
    with pytest.raises(ValueError, match="num_threads"):
        NativeLightGBMClassifier.from_lgbm(classifier, num_threads=num_threads)


@pytest.mark.skipif(not os.path.exists(PROD_MODEL), reason="needs the tracked production model")
def test_matches_lightgbm_on_the_production_model():
    with open(PROD_MODEL, "rb") as f:
        production = pickle.load(f)["clusterer"].classifier
    native = NativeLightGBMClassifier.from_lgbm(production, num_threads=4)
    rows = edge_rows(production, 18, seed=5)
    assert np.array_equal(native.predict_proba(rows), production.predict_proba(rows))


def test_scoring_classifier_caches_per_fit_and_falls_back(classifier):
    from s2apler.native_scorer import scoring_classifier

    native = scoring_classifier(classifier, num_threads=2)
    assert isinstance(native, NativeLightGBMClassifier)
    assert scoring_classifier(classifier, num_threads=3) is native and native.num_threads == 3
    # A refit makes a new booster, so it must get a new scorer.
    refit = pickle.loads(pickle.dumps(classifier))
    refit.fit(rows()[:500], (np.nan_to_num(rows()[:500, 1]) > 0.5).astype(int))
    assert scoring_classifier(refit, num_threads=1) is not native
    # Models the native booster can't reproduce score with LightGBM itself.
    early = pickle.loads(pickle.dumps(classifier))
    early.set_params(pred_early_stop=True)
    assert scoring_classifier(early, num_threads=1) is early
    plain = object()
    assert scoring_classifier(plain, num_threads=1) is plain


def test_clusterer_distances_match_lightgbm(monkeypatch):
    import s2apler.model as model
    from s2apler.data import PDData
    from s2apler.featurizer import FeaturizationInfo

    dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference")
    rng = np.random.default_rng(0)
    lightgbm_model = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(
        rng.random((400, 8)), rng.integers(0, 2, 400)
    )
    clusterer = model.Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "title_similarity"]),
        classifier=lightgbm_model,
        n_jobs=1,
    )
    blocks = {key: papers for key, papers in dataset.get_blocks().items() if len(papers) > 2}
    native = clusterer.make_distance_matrices(blocks, dataset, disable_tqdm=True)
    monkeypatch.setattr(model, "scoring_classifier", lambda classifier, num_threads: classifier)
    reference = clusterer.make_distance_matrices(blocks, dataset, disable_tqdm=True)
    assert native.keys() == reference.keys()
    for key in native:
        assert np.array_equal(native[key], reference[key])
