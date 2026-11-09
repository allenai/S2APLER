import copy
import json
import os

import pytest

from s2apler.data import PDData
from s2apler.native_scorer import NativeLightGBMClassifier
from s2apler.timo.interface import Instance, Prediction, Predictor, PredictorConfig

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/prod_model.pickle"), reason="needs the tracked production model"
)

BLOCKS = {"preface", "themicrotubuleplusendtrackingprotei", "cloninganddevelopmentalexpressionof"}


@pytest.fixture(scope="module")
def predictor():
    return Predictor(PredictorConfig(n_jobs=1), "data")


@pytest.fixture(scope="module")
def raw_papers():
    with open("tests/test_dataset/papers.json") as f:
        papers = json.load(f)
    return {key: paper for key, paper in papers.items() if paper["block"] in BLOCKS}


def instance_papers(raw_papers):
    """Converts test-dataset papers to the request schema that Scholar sends."""
    fields = ["title", "abstract", "venue", "journal_name", "year", "doi", "block"]
    return {
        key: {
            **{field: paper.get(field) for field in fields},
            "sourced_paper_id": str(paper["sourced_paper_id"]),
            "authors": [
                {
                    "author_info_first": author["first"],
                    "author_info_middle": " ".join(author["middle"]) or None,
                    "author_info_last": author["last"],
                    "author_info_suffix": author["suffix"],
                }
                for author in paper["authors"]
            ],
        }
        for key, paper in raw_papers.items()
    }


def assert_partition(prediction, paper_ids):
    clustered = [paper_id for cluster in prediction.values() for paper_id in cluster]
    assert sorted(clustered) == sorted(paper_ids)


@pytest.mark.parametrize("n_jobs, threads", [(1, 1), (3, 3), (-1, os.cpu_count() or 1), (0, 1)])
def test_converts_to_the_native_scorer_at_load(n_jobs, threads):
    predictor = Predictor(PredictorConfig(n_jobs=n_jobs), "data")
    scorer = predictor.clusterer._scorer(predictor.clusterer.classifier)
    assert isinstance(scorer, NativeLightGBMClassifier)
    assert scorer.num_threads == threads


def test_predict_one_matches_clustering_the_raw_papers(predictor, raw_papers):
    prediction = predictor.predict_one(Instance(papers=instance_papers(raw_papers)))
    assert isinstance(prediction, Prediction)
    assert_partition(prediction.prediction, raw_papers)
    # The request schema must reach PDData as the training-data format does.
    # PDData replaces the dict's values with preprocessed papers, so give it a copy.
    dataset = PDData(copy.deepcopy(raw_papers), name="raw", mode="inference")
    expected, _ = predictor.clusterer.predict(dataset.get_blocks(), dataset)
    assert prediction.prediction == expected
    # Papers sharing a DOI are always merged.
    cluster_of = {paper_id: key for key, cluster in prediction.prediction.items() for paper_id in cluster}
    by_doi = {}
    for paper_id, paper in raw_papers.items():
        if paper.get("doi"):
            by_doi.setdefault(paper["doi"].lower(), set()).add(cluster_of[paper_id])
    assert all(len(clusters) == 1 for clusters in by_doi.values())


def test_predict_one_with_cluster_seeds_clusters_incrementally(predictor, raw_papers, monkeypatch):
    preface = sorted(key for key, paper in raw_papers.items() if paper["block"] == "preface")
    seeded_a, seeded_b = preface[0], preface[1]
    calls = []
    original = predictor.clusterer.predict_incremental
    monkeypatch.setattr(
        predictor.clusterer,
        "predict_incremental",
        lambda *args: calls.append(args) or original(*args),
    )
    instance = Instance(
        papers=instance_papers(raw_papers),
        cluster_seeds={seeded_a: {seeded_b: "require"}},
    )
    prediction = predictor.predict_one(instance).prediction
    assert len(calls) == 1
    assert_partition(prediction, raw_papers)
    cluster_of = {paper_id: key for key, cluster in prediction.items() for paper_id in cluster}
    assert cluster_of[seeded_a] == cluster_of[seeded_b]


def test_predict_batch_predicts_each_instance(predictor, raw_papers):
    papers = instance_papers(raw_papers)
    one_block = {key: paper for key, paper in papers.items() if paper["block"] == "preface"}
    predictions = predictor.predict_batch([Instance(papers=papers), Instance(papers=one_block)])
    assert len(predictions) == 2
    assert_partition(predictions[1].prediction, one_block)
