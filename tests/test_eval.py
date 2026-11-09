import unittest

from s2apler.eval import b3_precision_recall_fscore
from s2apler.consts import ORPHAN_CLUSTER_KEY


class TestClusterer(unittest.TestCase):
    def test_b3_precision_recall_score(self):
        cluster_homogeneity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5, 6, 7], "3": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_1[0], 0.59864)
        self.assertAlmostEqual(cluster_homogeneity_1[1], 0.69524)
        self.assertAlmostEqual(cluster_homogeneity_1[2], 0.64333)

        cluster_homogeneity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_2[0], 0.69388)
        self.assertAlmostEqual(cluster_homogeneity_2[1], 0.69524)
        self.assertAlmostEqual(cluster_homogeneity_2[2], 0.69456)

        size_v_quantity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4, 5], "2": [6], "3": [7], "4": [8], "5": [9], "6": [10], "7": [11], "8": [12], "9": [13]},
        )
        self.assertAlmostEqual(size_v_quantity_1[0], 1)
        self.assertAlmostEqual(size_v_quantity_1[1], 0.69231)
        self.assertAlmostEqual(size_v_quantity_1[2], 0.81818)

        size_v_quantity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13]},
        )
        self.assertAlmostEqual(size_v_quantity_2[0], 1)
        self.assertAlmostEqual(size_v_quantity_2[1], 0.87692)
        self.assertAlmostEqual(size_v_quantity_2[2], 0.93443)

        size_v_quantity_3 = b3_precision_recall_fscore(
            {
                "dark": [1, 2, 3, 4, 5],
                "light": [6, 7],
                "line1": [8, 9],
                "line2": [10, 11],
                "line3": [12, 13],
                "x_" + ORPHAN_CLUSTER_KEY: [14],
            },
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13], "7": [14]},
        )
        # nothing changes in the output since the orphan goes to its own cluster
        self.assertEqual(size_v_quantity_3, size_v_quantity_2)

        size_v_quantity_4 = b3_precision_recall_fscore(
            {
                "dark": [1, 2, 3, 4, 5],
                "light": [6, 7],
                "line1": [8, 9],
                "line2": [10, 11],
                "line3": [12, 13],
                "x_" + ORPHAN_CLUSTER_KEY: [14],
            },
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13, 14]},
        )
        # doesn't matter where the orphan goes, we ignore it during B3 construction
        self.assertAlmostEqual(size_v_quantity_4, size_v_quantity_3)


def test_pairwise_eval_with_shap(tmp_path):
    # shap 0.51 returns one (n, f) array for binary LightGBM models; 0.41 returned one per class.
    import lightgbm as lgb
    import numpy as np

    from s2apler.eval import pairwise_eval

    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    classifier = lgb.LGBMClassifier(n_estimators=20, random_state=0, verbose=-1).fit(X, y)
    metrics = pairwise_eval(X, y, classifier, str(tmp_path), "t", ["a", "b", "c", "d"], shap_plot_type="bar")
    assert 0.5 < metrics["AUROC"] <= 1
    assert (tmp_path / "t_shap.png").exists()
