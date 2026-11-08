import unittest
import pytest
import numpy as np

from s2apler.data import PDData
from s2apler.model import Clusterer
from s2apler.featurizer import FeaturizationInfo
import lightgbm as lgb


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            cluster_seeds="tests/test_dataset/cluster_seeds.json",
            name="test_dataset",
            balanced_pair_sample=False,
        )

        features_to_use = [
            "year_diff",
            "title_similarity",
        ]
        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 8))
        y_random = np.random.randint(0, 8, 10)
        self.clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=False,
        )

    def test_make_distance_matrix_fastcluster(self):
        block = {"reviewerlistfor": ["84177344", "49188235", "214237506", "217917498", "1473469382"]}
        partial_supervision = {("84177344", "49188235"): 1.1, ("49188235", "214237506"): 1e-6}
        distance_matrices = self.clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dataset,
            partial_supervision=partial_supervision,
        )
        distance_matrix = distance_matrices["reviewerlistfor"]
        self.assertEqual(distance_matrix[0], np.float16(1.1))
        self.assertEqual(distance_matrix[1], np.float16(0.2))
        self.assertEqual(distance_matrix[4], np.float16(1e-6))

        distance_matrices = self.clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dataset,
            partial_supervision={},
        )
        distance_matrix = distance_matrices["reviewerlistfor"]
        self.assertEqual(distance_matrix[0], np.float16(0.2))
        self.assertEqual(distance_matrix[1], np.float16(0.2))
        self.assertEqual(distance_matrix[4], np.float16(0.2))


class TestClustererFit(unittest.TestCase):
    def make_dataset_and_clusterer(self):
        dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            balanced_pair_sample=False,
        )
        np.random.seed(1)
        clusterer = Clusterer(
            featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "title_similarity"]),
            classifier=lgb.LGBMClassifier(random_state=1, verbose=-1).fit(
                np.random.random((10, 8)), np.random.randint(0, 2, 10)
            ),
            n_jobs=1,
            use_cache=False,
            n_iter=3,
        )
        return dataset, clusterer

    def test_fit_tunes_eps(self):
        dataset, clusterer = self.make_dataset_and_clusterer()
        clusterer.fit(dataset)
        assert 0.2 <= clusterer.best_params["eps"] <= 0.8

    def test_fit_aligns_validation_distances_with_titled_papers(self):
        # predict sets null-title papers aside, so fit's distances must skip them too.
        # Misaligned distances fail predict's shape check.
        dataset, clusterer = self.make_dataset_and_clusterer()
        for i, paper_id in enumerate(sorted(dataset.papers)):
            if i % 3 == 0:
                dataset.papers[paper_id] = dataset.papers[paper_id]._replace(title="")
        clusterer.fit(dataset)
        assert "eps" in clusterer.best_params

    def test_predict_rejects_distances_for_other_papers(self):
        dataset, clusterer = self.make_dataset_and_clusterer()
        block = {"reviewerlistfor": ["84177344", "49188235", "214237506"]}
        with pytest.raises(ValueError, match="has shape"):
            clusterer.predict(block, dataset, dists={"reviewerlistfor": np.ones(6, dtype=np.float16)})
