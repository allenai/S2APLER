import unittest
import numpy as np

from s2apler.data import PDData
from s2apler.featurizer import NUM_FEATURES, FeaturizationInfo, featurize, many_pairs_featurize


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
        # A negative label marks a pair as partially supervised.
        test_pairs.append((test_pairs[0][0], test_pairs[0][1], -1))

        features, labels = many_pairs_featurize(test_pairs, self.dataset, self.featurizer, 1, nan_value=-1)
        threaded, _ = many_pairs_featurize(test_pairs, self.dataset, self.featurizer, 2, nan_value=-1)
        assert np.array_equal(features, threaded)

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

    def test_featurize_returns_features_for_the_block_split(self):
        self.dataset.train_pairs_size = 400
        self.dataset.val_pairs_size = 100
        self.dataset.test_pairs_size = 100
        train, val, test = featurize(self.dataset, self.featurizer, n_jobs=2, nan_value=-1)
        # featurize must featurize exactly the pairs that the block split samples.
        pairs = self.dataset.split_pairs(*self.dataset.split_cluster_papers())
        assert all(len(split_pairs) > 0 for split_pairs in pairs)
        for (features, labels), split_pairs in zip([train, val, test], pairs):
            expected_features, expected_labels = many_pairs_featurize(
                split_pairs, self.dataset, self.featurizer, 1, nan_value=-1
            )
            assert features.shape == (len(split_pairs), NUM_FEATURES)
            assert np.array_equal(features, expected_features)
            assert np.array_equal(labels, expected_labels)
            assert set(np.unique(labels)) <= {0, 1}
            assert not np.isnan(features).any()
        # Blocks don't straddle splits, so no pair appears in two of them.
        train_ids, val_ids, test_ids = ({p for pair in split for p in pair[:2]} for split in pairs)
        assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)

    def test_featurize_rejects_inference_datasets(self):
        dataset = PDData("tests/test_dataset/papers.json", name="test_dataset", mode="inference")
        with self.assertRaisesRegex(ValueError, "train-mode"):
            featurize(dataset, self.featurizer)

    def test_features_to_use_keeps_the_chosen_groups_columns(self):
        pairs = self.dataset.all_pairs()[:300]
        full, _ = many_pairs_featurize(pairs, self.dataset, self.featurizer, 1)
        subset_info = FeaturizationInfo(features_to_use=["year_diff", "title_similarity"])
        subset, _ = many_pairs_featurize(pairs, self.dataset, subset_info, 1)
        columns = [self.featurizer.feature_names.index(name) for name in subset_info.feature_names]
        assert subset_info.feature_names[0] == "year_diff"
        assert subset_info.number_of_features == len(columns) == subset.shape[1]
        assert np.array_equal(subset, full[:, columns], equal_nan=True)
