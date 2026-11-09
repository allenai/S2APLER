"""The Rust pair featurizer must match `_single_pair_featurize` bit for bit."""

from collections import Counter
from itertools import combinations
from types import ModuleType

import numpy as np
import pytest

from s2apler import __version__
from s2apler.data import PDData
from s2apler.featurizer import (
    NUM_FEATURES,
    FeaturizationInfo,
    _set_global_dataset,
    _single_pair_featurize,
    many_pairs_featurize,
    rust_pair_features,
)
from s2apler.runtime import load_s2apler_rust_extension


def python_features(dataset, pairs):
    _set_global_dataset(dataset.papers)
    return np.array([_single_pair_featurize(pair)[0] for pair in pairs], dtype=float)


def assert_bit_identical(expected, actual):
    assert expected.shape == actual.shape
    same = (expected == actual) | (np.isnan(expected) & np.isnan(actual))
    bad = [tuple(int(i) for i in index) for index in np.argwhere(~same)[:5]]
    assert same.all(), [(index, expected[index], actual[index]) for index in bad]


def author(first, last):
    return {"author_info_first": first, "author_info_middle": None, "author_info_last": last}


EDGE_PAPERS = {
    # Numerals: "part i" past the end of a shorter title compares with zip_longest padding.
    "1": {"title": "Internet resources", "year": 2010},
    "2": {"title": "Internet resources: social networking, part I", "year": 2011},
    "3": {"title": "Internet resources: social networking, part II", "year": 1650},
    # Special publication words and years in titles.
    "4": {"title": "Erratum: Results from 1999 and 2004", "source": "Crossref"},
    "5": {"title": "Reply to comment on results from 1999", "source": "Crossref"},
    "6": {"title": "Results from 1999 and 2004", "source": "Elsevier"},
    # Short titles, PDF sources, abstracts, venues and authors.
    "7": {"title": "A", "source": "Grobid", "abstract": "A short abstract about results."},
    "8": {"title": "Ab", "source": "MergedPDFExtraction", "venue": "Nature", "year": 2010},
    "9": {
        "title": "Graph methods for clustering",
        "abstract": "We cluster graphs. We cluster graphs again.",
        "venue": "Proceedings of KDD",
        "authors": [author("Sergey", "Feldman"), author("Daniel", "King")],
        "year": 2019,
    },
    "10": {
        "title": "Graph methods for clustering",
        "abstract": "We cluster graphs.",
        "venue": "KDD",
        "authors": [author("S.", "Feldman"), author("D.", "King")],
        "year": 2021,
    },
    "11": {
        "title": "Graph methods for clustering, volume 2",
        "authors": [author("Sergey", "Feldmann")],
        "year": 2021,
        "source": "Elsevier",
    },
}


def edge_dataset():
    papers = {
        key: {"authors": [], "block": "b", **{k: v for k, v in paper.items() if k != "authors"}, **paper}
        for key, paper in EDGE_PAPERS.items()
    }
    return PDData(papers=papers, name="edge", mode="inference", balanced_pair_sample=False)


def test_matches_python_on_edge_cases():
    dataset = edge_dataset()
    paper_ids: list[str] = sorted(dataset.papers, key=lambda paper_id: int(paper_id))
    pairs = list(combinations(paper_ids, 2))
    pairs += [(b, a) for a, b in pairs] + [(a, a) for a in dataset.papers]
    assert_bit_identical(python_features(dataset, pairs), rust_pair_features(dataset.papers, pairs, 2))


def test_preprocessing_makes_every_title_ascii():
    papers = {
        "1": {"title": "\u00dcber caf\u00e9 \u2014 \u6f22\u5b57 na\u00efve \\alpha $x^2$", "authors": [], "block": "b"},
        "2": {"title": "\u00a0\u2003\u03a9mega\ttitle\x00", "authors": [], "block": "b"},
    }
    dataset = PDData(papers=papers, name="unicode", mode="inference")
    assert all(paper.title.isascii() and paper.title for paper in dataset.papers.values())
    rust_pair_features(dataset.papers, [("1", "2")], 1)


def test_rejects_titles_that_skipped_preprocessing():
    dataset = edge_dataset()
    paper_id = next(iter(dataset.papers))
    # Title features compare bytes, which matches Python only on ASCII.
    dataset.papers[paper_id] = dataset.papers[paper_id]._replace(title="caf\u00e9 au lait")
    pairs = list(combinations(sorted(dataset.papers), 2))
    with pytest.raises(ValueError, match="ASCII"):
        rust_pair_features(dataset.papers, pairs, 1)


def test_matches_python_on_every_test_dataset_pair():
    dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference")
    pairs = [(str(a), str(b)) for block in dataset.get_blocks().values() for a, b in combinations(block, 2)]
    assert len(pairs) > 500
    assert_bit_identical(python_features(dataset, pairs), rust_pair_features(dataset.papers, pairs, 4))


@pytest.mark.parametrize("nan_value", [np.nan, -1.0])
def test_many_pairs_featurize_backends_agree(nan_value):
    dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference")
    pairs = dataset.all_pairs()[:3000]
    # A negative label marks a supervised pair, which is skipped.
    pairs[5] = (pairs[5][0], pairs[5][1], -1)
    info = FeaturizationInfo()
    outputs = [
        many_pairs_featurize(pairs, dataset, info, 1, nan_value=nan_value, backend=backend)
        for backend in ("python", "rust")
    ]
    for python_array, rust_array in zip(outputs[0], outputs[1]):
        assert_bit_identical(python_array, rust_array)


def test_rejects_unknown_backend():
    dataset = edge_dataset()
    with pytest.raises(ValueError, match="backend"):
        many_pairs_featurize([("1", "2", np.nan)], dataset, FeaturizationInfo(), 1, backend="java")


def test_extension_reports_its_feature_count():
    assert load_s2apler_rust_extension().get_build_info()["num_features"] == NUM_FEATURES


def test_featurizer_rejects_out_of_range_positions():
    featurizer = load_s2apler_rust_extension().PaperFeaturizer([])
    with pytest.raises(ValueError, match="out of range"):
        featurizer.featurize(np.array([0], dtype=np.uint32), np.array([0], dtype=np.uint32))


def test_loader_rejects_a_mismatched_extension_version():
    stale = ModuleType("s2apler._s2apler_rust")
    setattr(stale, "__version__", "0.0.1")
    with pytest.raises(RuntimeError, match="does not match"):
        load_s2apler_rust_extension(import_module=lambda name: stale)


def test_loader_maps_cargo_prerelease_spelling():
    native = ModuleType("s2apler._s2apler_rust")
    setattr(native, "__version__", __version__.replace("b", "-beta.") if "b" in __version__ else __version__)
    assert load_s2apler_rust_extension(import_module=lambda name: native) is native


def test_loader_reports_a_missing_extension():
    def missing(name):
        raise ModuleNotFoundError(name=name)

    with pytest.raises(RuntimeError, match="not importable"):
        load_s2apler_rust_extension(import_module=missing)


def counter_papers(count):
    papers = {
        key: {"title": "Same title", "abstract": "x", "authors": [], "block": "b", "year": year}
        for key, year in [("1", 2001.0), ("2", np.float64(2003)), ("3", 10**20), ("4", None)]
    }
    dataset = PDData(papers=papers, name="counts", mode="inference")
    for paper_id, paper in dataset.papers.items():
        dataset.papers[paper_id] = paper._replace(abstract_ngrams_words=Counter({"x": count, "y": 3}))
    return dataset


@pytest.mark.parametrize("count", [2**24 + 1, 2**32 + 7])
def test_matches_python_for_huge_counts_and_float_years(count):
    dataset = counter_papers(count)
    pairs = list(combinations(sorted(dataset.papers), 2))
    assert_bit_identical(python_features(dataset, pairs), rust_pair_features(dataset.papers, pairs, 1))


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 3])
def test_matches_python_for_every_thread_setting(n_jobs):
    dataset = edge_dataset()
    pairs = list(combinations(sorted(dataset.papers), 2))
    assert_bit_identical(python_features(dataset, pairs), rust_pair_features(dataset.papers, pairs, n_jobs))
