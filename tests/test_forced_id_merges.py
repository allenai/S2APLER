"""Forced merges on shared DOI / PMID / pdf_hash.

Covers three production bugs:
- a block whose papers all lack titles dropped every paper from the output;
- the post-clustering ID join lost track of earlier joins, so clusters sharing an ID stayed apart;
- a pdf_hash shared by items with different DOIs (a page of letters, an abstract supplement)
  forced those distinct items into one paper.
"""

from itertools import permutations

import numpy as np
import pytest

from s2apler.consts import CLUSTER_SEEDS_LOOKUP
from s2apler.data import PDData
from s2apler.featurizer import FeaturizationInfo
from s2apler.model import Clusterer


def make_dataset(papers):
    for paper in papers.values():
        paper.setdefault("authors", [])
        paper.setdefault("block", "b")
    return PDData(papers=papers, name="t", mode="inference", balanced_pair_sample=False)


def make_clusterer():
    # predict() gets precomputed distances, so the classifier is never called.
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        n_jobs=1,
        use_default_constraints_as_supervision=True,
    )


def far_apart(n):
    """Condensed distances that keep every titled paper in its own cluster."""
    return np.ones(n * (n - 1) // 2, dtype=np.float16)


def partition(clusters):
    return sorted(sorted(members) for members in clusters.values())


class TestPdfHashConstraint:
    def test_shared_pdf_hash_requires_merge(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "pdf_hash": "h", "doi": "10.1/a"},
                "2": {"title": "A", "pdf_hash": "h"},
            }
        )
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]

    def test_pdf_hash_with_different_dois_does_not_force_merge(self):
        dataset = make_dataset(
            {
                "1": {"title": "Fees for Certificates", "pdf_hash": "h", "doi": "10.1136/bmj.3.5724.712-c"},
                "2": {"title": "Improved Medical Services", "pdf_hash": "h", "doi": "10.1136/bmj.3.5724.712-b"},
                "3": {"title": "Fees for Certificates", "pdf_hash": "h"},
            }
        )
        assert dataset.multi_doi_pdf_hashes == {"h"}
        # A DOI-less record of the same PDF must not chain the two letters together either.
        assert dataset.get_constraint("1", "2") is None
        assert dataset.get_constraint("1", "3") is None
        assert dataset.get_constraint("2", "3") is None

    def test_doi_case_does_not_make_a_pdf_hash_ambiguous(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "pdf_hash": "h", "doi": "10.1/ABC"},
                "2": {"title": "A", "pdf_hash": "h", "doi": "10.1/abc"},
            }
        )
        assert dataset.multi_doi_pdf_hashes == set()


class TestPredictIdJoins:
    def test_all_null_title_block_keeps_every_paper(self):
        dataset = make_dataset(
            {
                "1": {"title": None, "pdf_hash": "h"},
                "2": {"title": "", "pdf_hash": "h"},
                "3": {"title": None},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={})
        assert partition(clusters) == [["1", "2"], ["3"]]

    def test_null_title_papers_join_each_other_and_titled_papers(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "10.1/a"},
                "2": {"title": "B"},
                "3": {"title": None, "doi": "10.1/a", "pdf_hash": "h"},
                "4": {"title": None, "pdf_hash": "h"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3", "4"]}, dataset, dists={"b": far_apart(2)})
        assert partition(clusters) == [["1", "3", "4"], ["2"]]

    @pytest.mark.parametrize("order", list(permutations("12345")))
    def test_id_chain_joins_transitively(self, order):
        # 1-4 share a DOI, 1-5 and 3-4 a pdf_hash, 2-5 a PMID: all five are one paper.
        # Order "12345" split this into two clusters before the union-find join.
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "10.1/c", "pdf_hash": "g"},
                "2": {"title": "B", "pmid": "9"},
                "3": {"title": "C", "pdf_hash": "h"},
                "4": {"title": "D", "doi": "10.1/c", "pdf_hash": "h"},
                "5": {"title": "E", "pmid": "9", "pdf_hash": "g"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": list(order)}, dataset, dists={"b": far_apart(5)})
        assert partition(clusters) == [["1", "2", "3", "4", "5"]]

    def test_multi_doi_pdf_hash_does_not_join_clusters(self):
        dataset = make_dataset(
            {
                "1": {
                    "title": "Frost forms Mars gullies",
                    "pdf_hash": "h",
                    "doi": "10.1038/511386a",
                },
                "2": {
                    "title": "Hotter summers despite hiatus",
                    "pdf_hash": "h",
                    "doi": "10.1038/511386b",
                },
                "3": {"title": None, "pdf_hash": "h"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={"b": far_apart(2)})
        assert partition(clusters) == [["1"], ["2"], ["3"]]
