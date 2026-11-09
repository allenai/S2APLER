"""Array-built distances and constraints must match the per-pair reference path exactly."""

import random
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pytest

from s2apler.consts import LARGE_INTEGER
from s2apler.data import PDData
from s2apler.featurizer import FeaturizationInfo, many_pairs_featurize
from s2apler.model import Clusterer, _condensed_pair_chunks
from s2apler.text import PUBLISHER_SOURCES


def crafted_dataset(seed=0):
    """Two blocks whose papers share IDs, URIs, titles and sources often, so rules collide."""
    rng = random.Random(seed)
    publishers = sorted(PUBLISHER_SOURCES)[:2]
    papers = {}
    for k in range(60):
        papers[str(k)] = {
            "title": rng.choice(["Editorial", "Front matter", "A study of things", "", "Another study"]),
            "abstract": rng.choice([None, "some abstract text", "other abstract"]),
            "authors": [{"first": rng.choice(["Ann", "Bo"]), "last": rng.choice(["Lee", "Kim"])}],
            "year": rng.choice([None, 2001, 2004]),
            "venue": rng.choice(["", "Nature", "Science"]),
            "doi": rng.choice([None, None, "10.1/a", "10.1/b", "https://doi.org/10.1/A"]),
            "pmid": rng.choice([None, None, "1", "2", " "]),
            "pdf_hash": rng.choice([None, "h1", "h2", "h3"]),
            "source": rng.choice(publishers + ["Grobid", None]),
            "source_id": rng.choice([None, "s1", "s2", ""]),
            "source_uris": rng.choice([None, ["u1"], ["u1", "u2"], ["u3"]]),
            "block": "b1" if k < 40 else "b2",
        }
    # A PDF that no two DOIs share, so its hash forces merges.
    for k in [10, 11, 12, 41, 42]:
        papers[str(k)].update(doi=None, pdf_hash="single")
    keys = sorted(papers, key=int)
    seeds = {keys[0]: {keys[1]: "require", keys[2]: "require", keys[3]: "disallow"}, keys[5]: {keys[6]: "require"}}
    seeds[keys[7]] = {keys[8]: "disallow", keys[45]: "require"}
    return PDData(papers, name="crafted", mode="inference", cluster_seeds=seeds)


def small_classifier():
    rng = np.random.default_rng(0)
    X = rng.random((2000, 18))
    X[rng.random(X.shape) < 0.2] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 7]) > 1).astype(int)
    return lgb.LGBMClassifier(n_estimators=30, num_leaves=31, random_state=0, verbose=-1).fit(X, y)


def reference_distances(clusterer, block_dict, dataset, partial_supervision, incremental):
    """The per-pair path: distance_matrix_helper, many_pairs_featurize, then one write per pair."""
    out = {key: np.zeros(len(p) * (len(p) - 1) // 2, dtype=np.float16) for key, p in block_dict.items()}
    rows = list(
        clusterer.distance_matrix_helper(
            block_dict, dataset, partial_supervision, incremental_dont_use_cluster_seeds=incremental
        )
    )
    if not rows:
        return out
    pairs = [row[0] for row in rows]
    features, _ = many_pairs_featurize(pairs, dataset, clusterer.featurizer_info, clusterer.n_jobs)
    labels = np.array([pair[2] for pair in pairs])
    flag = np.isnan(labels)
    predictions = np.zeros(len(pairs))
    if flag.any():
        predictions[flag] = clusterer._scorer(clusterer.classifier).predict_proba(features[flag])[:, 0]
    predictions[~flag] = labels[~flag] + LARGE_INTEGER
    for (_, (i, j), key), prediction in zip(rows, predictions):
        n = len(block_dict[key])
        out[key][i * n - i * (i + 1) // 2 + (j - i - 1)] = prediction
    return out


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 40])
@pytest.mark.parametrize("max_pairs", [1, 5, 1000])
def test_pair_chunks_cover_the_condensed_order(n, max_pairs):
    chunks = list(_condensed_pair_chunks(n, max_pairs))
    left = np.concatenate([c[0] for c in chunks]) if chunks else np.array([], dtype=int)
    right = np.concatenate([c[1] for c in chunks]) if chunks else np.array([], dtype=int)
    expected_left, expected_right = np.triu_indices(n, k=1)
    assert np.array_equal(left, expected_left) and np.array_equal(right, expected_right)
    offset = 0
    for chunk_left, _, first_offset in chunks:
        assert first_offset == offset
        assert len(chunk_left) <= max_pairs or len(set(chunk_left.tolist())) == 1
        offset += len(chunk_left)


@pytest.mark.parametrize("dont_merge", [True, False])
@pytest.mark.parametrize("incremental", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_constraint_values_match_get_constraint(dont_merge, incremental, seed):
    dataset = crafted_dataset(seed)
    for papers in dataset.get_blocks().values():
        n = len(papers)
        for left, right, first_offset in _condensed_pair_chunks(n, 37):
            values = dataset.constraint_values(papers, left, right, first_offset, dont_merge, incremental)
            for k, (i, j) in enumerate(zip(left, right)):
                expected = dataset.get_constraint(
                    papers[i],
                    papers[j],
                    dont_merge_cluster_seeds=dont_merge,
                    incremental_dont_use_cluster_seeds=incremental,
                )
                assert (np.isnan(values[k]) and expected is None) or values[k] == expected, (papers[i], papers[j])


@pytest.mark.parametrize("use_constraints", [True, False])
@pytest.mark.parametrize("incremental", [True, False])
@pytest.mark.parametrize("batch_size", [3, 50, 1000000])
def test_distances_match_the_per_pair_path(use_constraints, incremental, batch_size):
    dataset = crafted_dataset()
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(),
        classifier=small_classifier(),
        n_jobs=1,
        use_default_constraints_as_supervision=use_constraints,
        batch_size=batch_size,
    )
    blocks = dataset.get_blocks()
    papers, others = blocks["b1"], blocks["b2"]
    # Both orders of one pair, with different values: the pair given in block order wins.
    # Pairs across blocks and unknown papers are ignored.
    partial = {
        (papers[3], papers[9]): 0.3,
        (papers[9], papers[3]): 0.7,
        (papers[12], papers[4]): 1,
        (others[5], others[1]): 0.25,
        (papers[0], others[0]): 0,
        ("x", "y"): 0,
    }
    for supervision in ({}, partial):
        actual = clusterer.make_distance_matrices(blocks, dataset, supervision, True, incremental)
        expected = reference_distances(clusterer, blocks, dataset, supervision, incremental)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].tobytes() == expected[key].tobytes(), key


def test_distances_match_on_the_test_dataset():
    dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference")
    clusterer = Clusterer(FeaturizationInfo(), small_classifier(), n_jobs=2, batch_size=500)
    blocks = dataset.get_blocks()
    actual = clusterer.make_distance_matrices(blocks, dataset, disable_tqdm=True)
    expected = reference_distances(clusterer, blocks, dataset, {}, False)
    assert all(actual[key].tobytes() == expected[key].tobytes() for key in expected)
    assert sum(len(list(combinations(p, 2))) for p in blocks.values()) == sum(map(len, actual.values()))


def test_small_inputs_preprocess_without_a_process_pool(monkeypatch):
    import s2apler.data as data

    def no_pool(*args, **kwargs):
        raise AssertionError("small inputs must not start a process pool")

    monkeypatch.setattr(data.multiprocessing, "Pool", no_pool)
    dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference", n_jobs=8)
    assert len(dataset.papers) < data.PREPROCESS_POOL_MIN_PAPERS


def test_nan_ids_never_count_as_equal():
    import json

    from s2apler.text import PUBLISHER_SOURCES

    source = sorted(PUBLISHER_SOURCES)[0]
    # json.loads returns one shared NaN object; get_constraint still finds nan != nan.
    papers = json.loads(
        '{"1": {"title": "t", "authors": [], "block": "b", "source": "%s", "source_id": NaN},'
        ' "2": {"title": "t", "authors": [], "block": "b", "source": "%s", "source_id": NaN}}' % (source, source)
    )
    dataset = PDData(papers, name="nan", mode="inference")
    left, right, _ = next(_condensed_pair_chunks(2, 10))
    values = dataset.constraint_values(["1", "2"], left, right)
    assert values[0] == dataset.get_constraint("1", "2")
