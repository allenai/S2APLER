"""Cluster seeds that only disallow pairs pin a split durably (allenai/scholar#42255).

predict_incremental extends the clusters that "require" seeds define, and returned no clusters
at all when there were none. Disallow-only seeds now take the normal path as constraints.
"""

import copy
import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/prod_model.pickle"), reason="needs the tracked production model"
)

BLOCKS = {"preface", "themicrotubuleplusendtrackingprotei", "cloninganddevelopmentalexpressionof"}


@pytest.fixture(scope="module")
def predictor():
    from s2apler.timo.interface import Predictor, PredictorConfig

    return Predictor(PredictorConfig(n_jobs=1), "data")


@pytest.fixture(scope="module")
def papers():
    with open("tests/test_dataset/papers.json") as f:
        raw = json.load(f)
    fields = ["title", "abstract", "venue", "journal_name", "year", "doi", "block"]
    return {
        key: {
            **{field: paper.get(field) for field in fields},
            "sourced_paper_id": str(paper["sourced_paper_id"]),
            "authors": [
                {
                    "author_info_first": a["first"],
                    "author_info_middle": " ".join(a["middle"]) or None,
                    "author_info_last": a["last"],
                }
                for a in paper["authors"]
            ],
        }
        for key, paper in raw.items()
        if paper["block"] in BLOCKS
    }


def predict(predictor, papers, seeds=None):
    from s2apler.timo.interface import Instance

    prediction = predictor.predict_one(Instance(papers=copy.deepcopy(papers), cluster_seeds=seeds)).prediction
    return {pid: key for key, members in prediction.items() for pid in members}


def partition_without(cluster_of, excluded):
    groups = {}
    for pid, key in cluster_of.items():
        if pid not in excluded:
            groups.setdefault(key, set()).add(pid)
    return sorted(sorted(g) for g in groups.values())


def multi_paper_clusters(cluster_of):
    groups = {}
    for pid, key in cluster_of.items():
        groups.setdefault(key, []).append(pid)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g))


def test_disallow_only_seeds_keep_every_paper_and_split_the_pair(predictor, papers):
    base = predict(predictor, papers)
    clusters = multi_paper_clusters(base)
    assert clusters, "the test data must have a multi-paper cluster"
    a, b = clusters[0][:2]
    seeded = predict(predictor, papers, {a: {b: "disallow"}})
    assert sorted(seeded) == sorted(papers)
    assert seeded[a] != seeded[b]
    # Clusters outside the seeded pair's cluster are untouched.
    original = {pid for pid, key in base.items() if key == base[a]}
    assert partition_without(seeded, original) == partition_without(base, original)


def test_several_disallowed_pairs_all_split(predictor, papers):
    base = predict(predictor, papers)
    clusters = multi_paper_clusters(base)
    assert len(clusters) >= 2
    pairs = [(c[0], c[1]) for c in clusters[:2]]
    seeds = {}
    for a, b in pairs:
        seeds.setdefault(a, {})[b] = "disallow"
    seeded = predict(predictor, papers, seeds)
    assert sorted(seeded) == sorted(papers)
    assert all(seeded[a] != seeded[b] for a, b in pairs)


def test_a_disallowed_pair_stays_apart_despite_a_shared_doi(predictor, papers):
    # Two records with one DOI always merge by identifier, unless a seed pins them apart.
    key = next(k for k, p in papers.items() if p["doi"])
    copied = dict(papers, copy={**papers[key], "sourced_paper_id": "copy"})
    assert predict(predictor, copied)[key] == predict(predictor, copied)["copy"]
    seeded = predict(predictor, copied, {key: {"copy": "disallow"}})
    assert sorted(seeded) == sorted(copied)
    assert seeded[key] != seeded["copy"]


def test_require_seeds_still_cluster_incrementally(predictor, papers, monkeypatch):
    calls = []
    original = predictor.clusterer.predict_incremental
    monkeypatch.setattr(predictor.clusterer, "predict_incremental", lambda *args: calls.append(args) or original(*args))
    a, b = sorted(papers)[:2]
    predict(predictor, papers, {a: {b: "require"}})
    predict(predictor, papers, {a: {b: "disallow"}})
    assert len(calls) == 1
