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

from s2apler.consts import CLUSTER_SEEDS_LOOKUP, LARGE_INTEGER
from s2apler.data import PDData
from s2apler.featurizer import FeaturizationInfo
from s2apler.model import Clusterer


def make_dataset(papers):
    for paper in papers.values():
        paper.setdefault("authors", [])
        paper.setdefault("block", "b")
    return PDData(papers=papers, name="t", mode="inference", balanced_pair_sample=False)


def make_clusterer(use_default_constraints_as_supervision=True):
    # predict() gets precomputed distances, so the classifier is never called.
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        n_jobs=1,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
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

    def test_arxiv_doi_keeps_the_pdf_hash_usable(self):
        # A preprint record and its published record share one PDF: one paper, not a container.
        dataset = make_dataset(
            {
                "1": {"title": "A", "pdf_hash": "h", "doi": "10.48550/arXiv.2101.00001"},
                "2": {"title": "A", "pdf_hash": "h", "doi": "10.1/published"},
            }
        )
        assert dataset.multi_doi_pdf_hashes == set()
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

    @pytest.mark.parametrize(
        "variant",
        [
            "10.1/ABC",
            " 10.1/abc ",
            "https://doi.org/10.1/abc",
            "http://dx.doi.org/10.1/abc",
            "https://www.doi.org/10.1/abc",
            "10.1/abc.",
            "doi.org/10.1/abc",
            "doi:10.1/abc",
            "DOI: 10.1/abc",
            "urn:doi:10.1/abc",
            "https://doi.org/10.1%2Fabc",
        ],
    )
    def test_doi_variants_match_and_keep_the_pdf_hash_usable(self, variant):
        dataset = make_dataset(
            {
                "1": {"title": "A", "pdf_hash": "h", "doi": "10.1/abc"},
                "2": {"title": "A", "pdf_hash": "h", "doi": variant},
                "3": {"title": "A", "pdf_hash": "h", "doi": ""},
            }
        )
        assert dataset.multi_doi_pdf_hashes == set()
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]
        assert dataset.get_constraint("1", "3") == CLUSTER_SEEDS_LOOKUP["require"]

    def test_url_escapes_decode_but_bare_percent_signs_stay(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "https://doi.org/10.1000/res%23test"},
                "2": {"title": "A", "doi": "10.1000/res%23test"},
            }
        )
        assert dataset.papers["1"].doi == "10.1000/res#test"
        assert dataset.papers["2"].doi == "10.1000/res%23test"

    def test_int_and_padded_ids_match_their_string_forms(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "pmid": 12345},
                "2": {"title": "B", "pmid": " 12345 "},
            }
        )
        assert dataset.papers["1"].pmid == "12345"
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]

    @pytest.mark.parametrize("field", ["doi", "pmid", "pdf_hash"])
    def test_blank_ids_do_not_force_merges(self, field):
        dataset = make_dataset(
            {
                "1": {"title": "A", field: ""},
                "2": {"title": "B", field: "  "},
            }
        )
        assert dataset.get_constraint("1", "2") is None
        assert dataset.id_merge_keys("1") == []


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

    def test_arxiv_id_joins_clusters_the_model_keeps_apart(self):
        # The DBLP record gives the arXiv DOI as a resolver URL; it still yields the arXiv id.
        dataset = make_dataset(
            {
                "1": {"title": "Olmo 3", "source": "ArXiv", "source_id": "2512.13961"},
                "2": {"title": "Olmo 3", "source": "DBLP", "doi": "https://doi.org/10.48550/arXiv.2512.13961"},
                "3": {"title": "Olmo 2", "source": "ArXiv", "source_id": "2501.00656"},
                "4": {"title": None, "source": "ArXiv", "source_id": "2501.00656v3"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3", "4"]}, dataset, dists={"b": far_apart(3)})
        assert partition(clusters) == [["1", "2"], ["3", "4"]]

    @pytest.mark.parametrize("order", list(permutations("1234")))
    def test_arxiv_id_joins_conference_and_journal_versions(self, order):
        # One preprint links to both versions: one of its records shares the conference PDF,
        # and another carries the journal DOI. Preprint and both versions are one work.
        dataset = make_dataset(
            {
                "1": {"title": "A", "source": "IEEE", "doi": "10.1109/isit.1", "pdf_hash": "g"},
                "2": {"title": "A", "source": "ArXiv", "source_id": "1401.00001v1", "pdf_hash": "g"},
                "3": {"title": "A", "source": "ArXiv", "source_id": "1401.00001", "doi": "10.1109/tit.2"},
                "4": {"title": "A", "source": "Crossref", "doi": "10.1109/tit.2"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": list(order)}, dataset, dists={"b": far_apart(4)})
        assert partition(clusters) == [["1", "2", "3", "4"]]

    def test_arxiv_id_joins_a_preprint_to_its_one_published_version(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "source": "ArXiv", "source_id": "1401.00001"},
                "2": {"title": "A", "source": "DBLP", "doi": "10.48550/arXiv.1401.00001"},
                "3": {"title": "A", "source": "ArXiv", "source_id": "1401.00001v2", "doi": "10.1109/tit.2"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={"b": far_apart(3)})
        assert partition(clusters) == [["1", "2", "3"]]

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

    @pytest.mark.parametrize("field", ["doi", "pmid", "pdf_hash"])
    def test_blank_ids_do_not_join_null_title_papers(self, field):
        dataset = make_dataset(
            {
                "1": {"title": "A"},
                "2": {"title": None, field: ""},
                "3": {"title": None, field: ""},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={})
        assert partition(clusters) == [["1"], ["2"], ["3"]]

    def test_missing_distance_matrix_raises(self):
        dataset = make_dataset({"1": {"title": "A"}, "2": {"title": "B"}})
        with pytest.raises(ValueError, match="No distance matrix"):
            make_clusterer().predict({"b": ["1", "2"]}, dataset, dists={})

    def test_unattached_null_title_papers_keep_their_keys(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "10.1/a"},
                "2": {"title": None, "doi": "10.1/a"},
                "3": {"title": None},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={})
        assert clusters == {"b_0": ["1", "2"], "b_1": ["3"]}

    def test_null_title_paper_cannot_bridge_titled_clusters(self):
        # Prod case: a title-less record carries one abstract's DOI and the hash of a PDF that a
        # different abstract's record also points to. Bridging them would merge two papers.
        dataset = make_dataset(
            {
                "1": {"title": "The relationship between physical activity and fitness", "doi": "10.1249/b"},
                "2": {"title": "Lower aerobic reserve in older adults", "pdf_hash": "h"},
                "3": {"title": None, "doi": "10.1249/b", "pdf_hash": "h"},
            }
        )
        clusters, _ = make_clusterer().predict({"b": ["1", "2", "3"]}, dataset, dists={"b": far_apart(2)})
        assert partition(clusters) == [["1", "3"], ["2"]]

    def test_null_title_paper_cannot_bridge_titled_clusters_without_default_constraints(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "10.1/x"},
                "2": {"title": "B", "pmid": "7"},
                "3": {"title": None, "doi": "10.1/x", "pmid": "7"},
            }
        )
        clusterer = make_clusterer(use_default_constraints_as_supervision=False)
        clusters, _ = clusterer.predict({"b": ["1", "2", "3"]}, dataset, dists={"b": far_apart(2)})
        assert partition(clusters) == [["1", "3"], ["2"]]

    def test_all_null_title_block_survives_next_to_titled_block(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "block": "x", "doi": "10.1/a"},
                "2": {"title": "B", "block": "x"},
                "3": {"title": None, "block": "y", "pdf_hash": "h"},
                "4": {"title": None, "block": "y", "pdf_hash": "h"},
            }
        )
        clusters, _ = make_clusterer().predict({"x": ["1", "2"], "y": ["3", "4"]}, dataset, dists={"x": far_apart(2)})
        assert clusters == {"x_1": ["1"], "x_2": ["2"], "y_0": ["3", "4"]}

    def test_multi_doi_pdf_hash_scope_is_dataset_wide(self):
        # A PDF seen with two DOIs anywhere is a container, so it stops forcing merges in every block.
        dataset = make_dataset(
            {
                "1": {"title": "A", "block": "x", "doi": "10.1/a", "pdf_hash": "h"},
                "2": {"title": "A", "block": "x", "pdf_hash": "h"},
                "3": {"title": "C", "block": "y", "doi": "10.1/c", "pdf_hash": "h"},
            }
        )
        assert dataset.get_constraint("1", "2") is None


class TestDistanceLabels:
    def test_ambiguous_pdf_hash_pairs_go_to_the_model(self):
        dataset = make_dataset(
            {
                "1": {"title": "A", "doi": "10.1/a", "pdf_hash": "h"},
                "2": {"title": "B", "doi": "10.1/b", "pdf_hash": "h"},
                "3": {"title": "C", "pdf_hash": "g"},
                "4": {"title": "C", "pdf_hash": "g"},
            }
        )
        rows = make_clusterer().distance_matrix_helper({"b": ["1", "2", "3", "4"]}, dataset, {})
        labels = {(p1, p2): label for (p1, p2, label), _, _ in rows}
        # NaN tells make_distance_matrices to ask the classifier; a number is a forced distance.
        assert np.isnan(labels[("1", "2")])
        assert labels[("3", "4")] == CLUSTER_SEEDS_LOOKUP["require"] - LARGE_INTEGER
