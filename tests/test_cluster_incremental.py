import unittest
import numpy as np

from s2apler.data import PDData
from s2apler.model import Clusterer
from s2apler.featurizer import FeaturizationInfo
import lightgbm as lgb


class AlwaysSameClassifier:
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class AlwaysDifferentClassifier:
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class RaisingClassifier:
    def predict_proba(self, X):
        raise AssertionError(
            "null-title incremental assignment should not call the classifier"
        )


def cluster_sets(output):
    return {frozenset(cluster) for cluster in output.values()}


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            cluster_seeds={
                "84177344": {"49188235": "require"},
                "214237506": {"217917498": "require"},
            },
            name="test_dataset",
            balanced_pair_sample=False,
        )

        features_to_use = [
            "year_diff",
            "title_similarity",
        ]

        self.featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 8))
        y_random = np.random.randint(0, 8, 10)
        self.clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=lgb.LGBMClassifier(
                random_state=1, data_random_seed=1, feature_fraction_seed=1
            ).fit(X_random, y_random),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

    def test_predict_incremental(self):
        block_papers = ["49188235", "84177344", "214237506", "217917498", "1473469382"]
        output = self.clusterer.predict_incremental(block_papers, self.dataset)
        expected_output = {
            "0": ["84177344", "49188235", "1473469382"],
            "1": ["214237506", "217917498"],
        }
        assert output == expected_output

        self.dataset.cluster_seeds_disallow = {("84177344", "1473469382")}
        output = self.clusterer.predict_incremental(block_papers, self.dataset)
        expected_output = {
            "0": ["84177344", "49188235"],
            "1": ["214237506", "217917498", "1473469382"],
        }
        assert output == expected_output

    def test_predict_incremental_without_required_seeds(self):
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=AlwaysSameClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=False,
        )
        dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            balanced_pair_sample=False,
        )

        output = clusterer.predict_incremental(["49188235", "84177344"], dataset)

        assert cluster_sets(output) == {frozenset({"49188235", "84177344"})}

    def test_predict_incremental_with_disallow_only_seeds(self):
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=AlwaysSameClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )
        dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            cluster_seeds={"49188235": {"84177344": "disallow"}},
            name="test_dataset",
            balanced_pair_sample=False,
        )

        output = clusterer.predict_incremental(["49188235", "84177344"], dataset)

        assert cluster_sets(output) == {
            frozenset({"49188235"}),
            frozenset({"84177344"}),
        }

    def test_predict_incremental_does_not_merge_across_blocks(self):
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=AlwaysSameClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=False,
        )
        dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            cluster_seeds={"49188235": {"84177344": "require"}},
            name="test_dataset",
            balanced_pair_sample=False,
        )

        output = clusterer.predict_incremental(
            ["49188235", "84177344", "210102606", "1400649365030178816"],
            dataset,
        )

        assert output["0"] == ["49188235", "84177344"]
        assert frozenset({"210102606", "1400649365030178816"}) in cluster_sets(output)

    def test_predict_incremental_merges_cross_block_hard_ids(self):
        papers = {
            "1": {
                "title": "Seed paper",
                "authors": [],
                "sourced_paper_id": "1",
                "block": "a",
                "doi": "10/example",
            },
            "2": {
                "title": "Seed partner",
                "authors": [],
                "sourced_paper_id": "2",
                "block": "a",
            },
            "3": {
                "title": "Same DOI in another block",
                "authors": [],
                "sourced_paper_id": "3",
                "block": "b",
                "doi": "10/example",
            },
            "4": {
                "title": "Unmatched paper",
                "authors": [],
                "sourced_paper_id": "4",
                "block": "b",
            },
        }
        dataset = PDData(
            papers,
            cluster_seeds={"1": {"2": "require"}},
            name="cross_block_hard_ids",
            mode="inference",
        )
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=AlwaysDifferentClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

        output = clusterer.predict_incremental(["1", "2", "3", "4"], dataset)

        assert set(output["0"]) == {"1", "2", "3"}
        assert frozenset({"4"}) in cluster_sets(output)

    def test_predict_incremental_attaches_null_titles_by_id(self):
        papers = {
            "1": {
                "title": "Seed paper",
                "authors": [],
                "sourced_paper_id": "1",
                "block": "b",
                "doi": "10/example",
            },
            "2": {
                "title": "Seed partner",
                "authors": [],
                "sourced_paper_id": "2",
                "block": "b",
            },
            "3": {
                "title": None,
                "authors": [],
                "sourced_paper_id": "3",
                "block": "b",
                "doi": "10/example",
            },
            "4": {
                "title": None,
                "authors": [],
                "sourced_paper_id": "4",
                "block": "b",
            },
        }
        dataset = PDData(
            papers,
            cluster_seeds={"1": {"2": "require"}},
            name="null_title_incremental",
            mode="inference",
        )
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=RaisingClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

        output = clusterer.predict_incremental(["1", "2", "3", "4"], dataset)

        assert set(output["0"]) == {"1", "2", "3"}
        assert frozenset({"4"}) in cluster_sets(output)

    def test_predict_incremental_attaches_to_null_title_seed_by_id(self):
        papers = {
            "1": {
                "title": None,
                "authors": [],
                "sourced_paper_id": "1",
                "block": "b",
                "doi": "10/example",
            },
            "2": {
                "title": None,
                "authors": [],
                "sourced_paper_id": "2",
                "block": "b",
            },
            "3": {
                "title": "New paper",
                "authors": [],
                "sourced_paper_id": "3",
                "block": "b",
                "doi": "10/example",
            },
            "4": {
                "title": "Unmatched paper",
                "authors": [],
                "sourced_paper_id": "4",
                "block": "b",
            },
        }
        dataset = PDData(
            papers,
            cluster_seeds={"1": {"2": "require"}},
            name="null_title_seed_incremental",
            mode="inference",
        )
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=RaisingClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

        output = clusterer.predict_incremental(["1", "2", "3", "4"], dataset)

        assert set(output["0"]) == {"1", "2", "3"}
        assert frozenset({"4"}) in cluster_sets(output)

    def test_predict_incremental_blocks_disallowed_altered_cluster_assignment(self):
        papers = {
            "1": {
                "title": "Seed paper",
                "authors": [],
                "sourced_paper_id": "1",
                "block": "b",
                "doi": "10/example",
            },
            "2": {
                "title": "Different seed paper",
                "authors": [],
                "sourced_paper_id": "2",
                "block": "b",
            },
            "3": {
                "title": "New paper",
                "authors": [],
                "sourced_paper_id": "3",
                "block": "b",
                "doi": "10/example",
            },
        }
        dataset = PDData(
            papers,
            cluster_seeds={"1": {"2": "require"}, "3": {"2": "disallow"}},
            altered_cluster_papers=["1"],
            name="altered_cluster_incremental",
            mode="inference",
        )
        clusterer = Clusterer(
            featurizer_info=self.featurizer_info,
            classifier=AlwaysDifferentClassifier(),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

        output = clusterer.predict_incremental(["1", "2", "3"], dataset)

        assert set(output["0"]) == {"1", "2"}
        assert frozenset({"3"}) in cluster_sets(output)
