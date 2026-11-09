"""Cluster seeds that only disallow pairs pin a split durably (allenai/scholar#42255).

predict_incremental extends the clusters that "require" seeds define, and returned no clusters
at all when there were none. Disallow-only seeds now take the normal path as constraints.
"""

import copy
import json
import os
import random

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


def join(records, labels, first_null_label, seeds=None, join_titled=True):
    """Runs the identifier join on minimal records; returns each paper's joined label."""
    from s2apler.data import PDData
    from s2apler.model import _join_clusters_by_ids

    base = {"title": None, "doi": None, "authors": [{"first": "Ann", "last": "Lee"}], "block": "b", "year": 2020}
    dataset = PDData(
        {pid: {**base, **record} for pid, record in records.items()},
        name="seeds",
        mode="inference",
        cluster_seeds=seeds,
    )
    return dict(zip(records, _join_clusters_by_ids(list(records), labels, dataset, first_null_label, join_titled)))


@pytest.mark.parametrize("order", ["abc", "bca", "cab"])
def test_copies_a_seed_keeps_from_one_paper_still_join_each_other(order):
    # Each copy is refused by "a"; they must still try, and join, each other.
    joined = join({pid: {"doi": "10.1/x"} for pid in order}, [0, 1, 2], 0, {"a": {"b": "disallow", "c": "disallow"}})
    assert joined["b"] == joined["c"] != joined["a"]


def test_random_blocks_never_join_a_disallowed_pair():
    # Small random blocks with shared DOIs and PMIDs, titled and null-title papers, and seeds.
    rng = random.Random(0)
    for _ in range(300):
        titled, nulls = rng.randint(0, 5), rng.randint(2, 5)
        names = [f"p{i}" for i in range(titled + nulls)]
        records = {
            name: {
                "title": "x" if i < titled else None,
                "doi": rng.choice([None, "10.1/a", "10.1/b"]),
                "pmid": rng.choice([None, "1", "2"]),
            }
            for i, name in enumerate(names)
        }
        labels = [rng.randrange(titled) for _ in range(titled)] + [titled + i for i in range(nulls)]
        label_of = dict(zip(names, labels))
        pairs = {tuple(sorted(rng.sample(names, 2))) for _ in range(rng.randint(1, 4))}
        pairs = [(a, b) for a, b in pairs if label_of[a] != label_of[b]]
        seeds = {}
        for a, b in pairs:
            seeds.setdefault(a, {})[b] = "disallow"
        joined = join(records, labels, titled, seeds or None, join_titled=rng.random() < 0.8)
        assert all(joined[a] != joined[b] for a, b in pairs), (records, labels, pairs)


@pytest.mark.parametrize("order", ["abc", "acb", "bac", "bca", "cab", "cba"])
def test_a_seed_blocks_a_join_through_an_earlier_join(order):
    # a and b share a DOI, b and c share a PMID; the seed keeps a from c.
    records = {"a": {"doi": "10.1/x"}, "b": {"doi": "10.1/x", "pmid": "7"}, "c": {"pmid": "7"}}
    joined = join({pid: records[pid] for pid in order}, [0, 1, 2], 0, {"a": {"c": "disallow"}})
    assert joined["a"] != joined["c"]
    assert joined["b"] in (joined["a"], joined["c"])


def test_a_seed_on_any_cluster_member_blocks_the_join():
    # a1 and a2 are one titled cluster; only a1 shares c's DOI, but a2 is seeded apart from c.
    records = {"a1": {"title": "x", "doi": "10.1/x"}, "a2": {"title": "x"}, "c": {"doi": "10.1/x"}}
    joined = join(records, [0, 0, 1], 1, {"a2": {"c": "disallow"}})
    assert joined["a1"] == joined["a2"] != joined["c"]


@pytest.mark.parametrize("seeds", [None, {"a": {"n": "disallow"}}])
def test_titled_clusters_stay_apart_when_titled_joins_are_off(seeds):
    records = {"a": {"title": "x", "doi": "10.1/x"}, "b": {"title": "x", "doi": "10.1/x"}, "n": {"doi": "10.1/x"}}
    joined = join(records, [0, 1, 2], 2, seeds, join_titled=False)
    assert joined["a"] != joined["b"]
    # A null-title paper attaches to at most one titled cluster, and never to a seeded partner.
    assert joined["n"] in ((joined["b"],) if seeds else (joined["a"], joined["b"]))


def test_require_seeds_still_cluster_incrementally(predictor, papers, monkeypatch):
    calls = []
    original = predictor.clusterer.predict_incremental
    monkeypatch.setattr(predictor.clusterer, "predict_incremental", lambda *args: calls.append(args) or original(*args))
    a, b = sorted(papers)[:2]
    predict(predictor, papers, {a: {b: "require"}})
    predict(predictor, papers, {a: {b: "disallow"}})
    assert len(calls) == 1
