import unittest
import pytest

from s2apler.consts import CLUSTER_SEEDS_LOOKUP
from s2apler.data import PDData


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            balanced_pair_sample=False,
        )

    def test_split_pairs_within_blocks(self):
        # Test random sampling within blocks
        self.dataset.train_pairs_size = 100
        self.dataset.val_pairs_size = 50
        self.dataset.test_pairs_size = 50
        self.dataset.random_seed = 1111
        (
            train_block_dict,
            val_block_dict,
            test_block_dict,
        ) = self.dataset.split_cluster_papers()
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(train_block_dict, val_block_dict, test_block_dict)

        assert len(train_pairs) == 100 and len(val_pairs) == 50 and len(test_pairs) == 50
        assert (
            train_pairs[0] == ("1409183546974670848", "1401326657880461312", 0)
            and val_pairs[0] == ("35668499", "212140021", 1)
            and test_pairs[0] == ("65517728", "1458733052", 0)
        )

        # Test adding the all test pairs flag to the test above
        self.dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(train_block_dict, val_block_dict, test_block_dict)
        assert len(train_pairs) == 100, len(val_pairs) == 50 and len(test_pairs) == 72

    def test_initialization(self):
        dataset = PDData(papers={}, name="", mode="inference")
        assert dataset.paper_to_cluster_id is None
        assert dataset.all_test_pairs_flag

    def test_construct_cluster_to_signatures(self):
        cluster_to_signatures = self.dataset.construct_cluster_to_papers(
            {"a": ["20797514", "51804247"], "b": ["7355243", "65911307"]}
        )
        expected_cluster_to_signatures = {
            "PM_51910": ["20797514"],
            "PM_114482": ["51804247"],
            "PM_221928": ["7355243"],
            "PM_146905": ["65911307"],
        }
        assert cluster_to_signatures == expected_cluster_to_signatures


class TestSourceUriConstraint(unittest.TestCase):
    """Same source_uris AND same title -> must-merge, even when pdf_hash differs (e.g. HAL
    serves byte-different PDFs per fetch). The title check guards against URLs that legitimately
    serve multiple papers (proceedings volumes, journal index pages). See allenai/scholar#41863
    and allenai/S2APLER#16."""

    @staticmethod
    def _make_dataset(papers):
        return PDData(papers=papers, name="t", mode="inference", balanced_pair_sample=False)

    def test_shared_source_uri_with_different_pdf_hash_requires_merge(self):
        url = "https://inria.hal.science/hal-01136686/file/chiaroscuro-sigmod-main-hal.pdf"
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "Chiaroscuro",
                    "authors": [],
                    "pdf_hash": "hash_a",
                    "source_uris": [url],
                    "block": "chiaroscuro",
                },
                "2": {
                    "title": "Chiaroscuro",
                    "authors": [],
                    "pdf_hash": "hash_b",
                    "source_uris": [url],
                    "block": "chiaroscuro",
                },
            }
        )
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]

    def test_disjoint_source_uris_does_not_force_merge(self):
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "A",
                    "authors": [],
                    "source_uris": ["https://example.com/a.pdf"],
                    "block": "a",
                },
                "2": {
                    "title": "A",
                    "authors": [],
                    "source_uris": ["https://example.com/b.pdf"],
                    "block": "a",
                },
            }
        )
        assert dataset.get_constraint("1", "2") is None

    def test_missing_source_uris_does_not_force_merge(self):
        dataset = self._make_dataset(
            {
                "1": {"title": "A", "authors": [], "block": "a"},
                "2": {"title": "A", "authors": [], "source_uris": ["https://example.com/a.pdf"], "block": "a"},
            }
        )
        assert dataset.get_constraint("1", "2") is None

    def test_shared_uri_but_different_titles_does_not_force_merge(self):
        # Models the proceedings/index-page case: one URL serves multiple distinct papers.
        # E.g. http://splc2010.postech.ac.kr/SPLC2010_second_volume.pdf has 51 distinct titles.
        proceedings_url = "https://example.org/proceedings_2010.pdf"
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "First paper in the proceedings volume",
                    "authors": [],
                    "source_uris": [proceedings_url],
                    "block": "first",
                },
                "2": {
                    "title": "Second paper in the proceedings volume",
                    "authors": [],
                    "source_uris": [proceedings_url],
                    "block": "second",
                },
            }
        )
        assert dataset.get_constraint("1", "2") is None

    def test_shared_uri_same_title_but_doi_disagrees_does_not_force_merge(self):
        # The "Front Matter from same publisher" case Sergey raised: each issue's Front Matter
        # has a distinct DOI but identical title and may sit at the same publisher domain URL.
        url = "https://example.org/journal_volume_landing.pdf"
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "Front Matter",
                    "authors": [],
                    "doi": "10.1/issue-2020",
                    "source_uris": [url],
                    "block": "frontmatter",
                },
                "2": {
                    "title": "Front Matter",
                    "authors": [],
                    "doi": "10.1/issue-2021",
                    "source_uris": [url],
                    "block": "frontmatter",
                },
            }
        )
        assert dataset.get_constraint("1", "2") is None

    def test_shared_uri_same_title_but_first_authors_disagree_does_not_force_merge(self):
        # The college-catalog case: one URL serves a directory page; "papers" mined from it
        # share a generic title but have distinct first-author last names per entry.
        url = "https://example.edu/department/catalog.pdf"
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "Linguistics",
                    "authors": [{"first": "Alice", "last": "Smith"}],
                    "source_uris": [url],
                    "block": "linguistics",
                },
                "2": {
                    "title": "Linguistics",
                    "authors": [{"first": "Bob", "last": "Jones"}],
                    "source_uris": [url],
                    "block": "linguistics",
                },
            }
        )
        assert dataset.get_constraint("1", "2") is None

    def test_shared_uri_same_title_with_matching_first_author_force_merges(self):
        # Sanity check that the author guard doesn't over-fire: when both have the same
        # first author last name the URL+title constraint should still trigger.
        url = "https://inria.hal.science/hal-01136686/file/chiaroscuro-sigmod-main-hal.pdf"
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "Chiaroscuro",
                    "authors": [{"first": "Tristan", "last": "Allard"}],
                    "pdf_hash": "hash_a",
                    "source_uris": [url],
                    "block": "chiaroscuro",
                },
                "2": {
                    "title": "Chiaroscuro",
                    "authors": [{"first": "Tristan", "last": "Allard"}],
                    "pdf_hash": "hash_b",
                    "source_uris": [url],
                    "block": "chiaroscuro",
                },
            }
        )
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]

    def test_doi_match_still_takes_precedence(self):
        dataset = self._make_dataset(
            {
                "1": {
                    "title": "A",
                    "authors": [],
                    "doi": "10.1/x",
                    "source_uris": ["https://example.com/a.pdf"],
                    "block": "a",
                },
                "2": {
                    "title": "A",
                    "authors": [],
                    "doi": "10.1/x",
                    "source_uris": ["https://example.com/b.pdf"],
                    "block": "a",
                },
            }
        )
        assert dataset.get_constraint("1", "2") == CLUSTER_SEEDS_LOOKUP["require"]
