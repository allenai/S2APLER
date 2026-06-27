import unittest
import numpy as np

from s2apler.data import PDData
from s2apler.featurizer import FeaturizationInfo, many_pairs_featurize


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            balanced_pair_sample=False,
        )

        features_to_use = [
            "author_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            "abstract_similarity",
            "paper_quality",
        ]
        self.featurizer = FeaturizationInfo(features_to_use=features_to_use)

    def check_features_array_equal(self, array_1, array_2):
        assert len(array_1) == len(array_2)
        for i in range(len(array_1)):
            both_nan = np.isnan(array_1[i]) and np.isnan(array_2[i])
            if not both_nan:
                self.assertAlmostEqual(array_1[i], array_2[i], msg=i)

    def test_featurizer(self):
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
            ("3", "2", 0),
            ("3", "2", -1),
        ]

        self.dataset.train_pairs_size = 100
        self.dataset.val_pairs_size = 50
        self.dataset.test_pairs_size = 3
        self.dataset.random_seed = 1111
        (
            train_block_dict,
            val_block_dict,
            test_block_dict,
        ) = self.dataset.split_cluster_papers()
        _, _, test_pairs = self.dataset.split_pairs(train_block_dict, val_block_dict, test_block_dict)
        test_pair_neg_1 = list(test_pairs[0])
        test_pair_neg_1[-1] = -1
        test_pairs.append(tuple(test_pair_neg_1))

        # single thread
        features, labels, _ = many_pairs_featurize(test_pairs, self.dataset, self.featurizer, 1, False, 1, nan_value=-1)
        features, _, _ = many_pairs_featurize(test_pairs, self.dataset, self.featurizer, 2, False, 1, nan_value=-1)

        expected_features_1 = [
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            -1.0,
        ]
        expected_features_2 = [
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            0.0,
            0.0,
            -1.0,
        ]
        expected_features_3 = [
            0.06296296296296296,
            0.0,
            -1.0,
            0.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            2.0,
            1.0,
            0.0,
            0.0,
            -1.0,
        ]
        self.check_features_array_equal(list(features[0, :]), expected_features_1)
        self.check_features_array_equal(list(features[1, :]), expected_features_2)
        self.check_features_array_equal(list(features[2, :]), expected_features_3)

    def test_many_pairs_featurize_multiprocessing_initializes_worker_dataset(self):
        papers = {
            str(i): {
                "title": f"Shared block paper {i}",
                "abstract": f"Abstract {i}",
                "authors": [{"first": "A", "last": f"Author{i}"}],
                "venue": "Venue",
                "journal_name": "Venue",
                "year": 2020,
                "source": "PubMed",
                "block": "shared",
            }
            for i in range(46)
        }
        dataset = PDData(papers=papers, name="multiprocessing_test", mode="inference", n_jobs=1)
        paper_ids = list(dataset.papers.keys())
        pairs = [
            (paper_ids[i], paper_ids[j], np.nan)
            for i in range(len(paper_ids))
            for j in range(i + 1, len(paper_ids))
        ]
        featurizer = FeaturizationInfo(features_to_use=["title_similarity"])

        serial_features, _, _ = many_pairs_featurize(pairs, dataset, featurizer, 1, False, 100, nan_value=-1)
        parallel_features, _, _ = many_pairs_featurize(pairs, dataset, featurizer, 2, False, 100, nan_value=-1)

        assert len(pairs) > 1000
        np.testing.assert_allclose(parallel_features, serial_features)
