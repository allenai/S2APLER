from typing import Dict, Optional, Any, List, Tuple, TYPE_CHECKING

import logging

if TYPE_CHECKING:  # need this for circular import issues
    from s2apler.model import Clusterer
    from s2apler.data import PDData

from s2apler.consts import ORPHAN_CLUSTER_KEY

import os
from os.path import join

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


logger = logging.getLogger("s2apler")

sns.set(context="talk")


def cluster_eval(
    dataset: "PDData",
    clusterer: "Clusterer",
    split: str = "test",
    use_s2_clusters: bool = False,
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: PDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?
    use_s2_clusters: bool
        Whether to use the original S2 clusters

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by paper.
    """
    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test}!")

    # block ground truth labels: cluster_to_papers
    cluster_to_papers = dataset.construct_cluster_to_papers(block_dict)

    # predict
    pred_clusters, dists = clusterer.predict(block_dict, dataset, use_s2_clusters=use_s2_clusters, dists=None)

    # get metrics
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_paper,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_papers, pred_clusters)
    metrics: Dict[str, Tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}

    metrics["Pred bigger ratio (mean, count)"] = (
        np.round(np.mean(pred_bigger_ratios), 2),
        len(pred_bigger_ratios),
    )
    metrics["True bigger ratio (mean, count)"] = (
        np.round(np.mean(true_bigger_ratios), 2),
        len(true_bigger_ratios),
    )

    return metrics, b3_metrics_per_paper


def _positive_class_shap_values(model: Any, X: Optional[np.ndarray]) -> np.ndarray:
    """Returns SHAP values for class 1 of a binary tree model, shaped (n_samples, n_features).

    shap 0.41 returned one array per class. shap 0.51 returns a single array for binary
    LightGBM models and an (n_samples, n_features, n_classes) array for some other models.

    Parameters
    ----------
    model: Any
        a fitted tree classifier
    X: np.ndarray
        the feature rows to explain

    Returns
    -------
    np.ndarray: the positive-class SHAP values
    """
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        return values[1]
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def pairwise_eval(
    X: np.ndarray,
    y: np.ndarray,
    classifier: Any,
    figs_path: str,
    title: str,
    shap_feature_names: List[str],
    thresh_for_f1: float = 0.5,
    shap_plot_type: Optional[str] = "dot",
    skip_shap: bool = False,
) -> Dict[str, float]:
    """
    Performs pairwise model evaluation, without using blocks.
    Also writes plots to the provided file path

    Parameters
    ----------
    X: np.ndarray
        Feature matrix of features to do eval on.
    y: np.ndarray
        Feature matrix of labels to do eval on.
    classifier: sklearn compatible classifier
        Classifier to do eval on.
    figs_path: string
        Where to put the resulting evaluation figures.
    title: string
        Title to stick on all the plots and use for file name.
    shap_feature_names: List[str]
        List of feature names for the SHAP plots.
    thresh_for_f1: float
        Threshold for F1 computation. Defaults to 0.5.
    shap_plot_type: str
        Type of shap plot. Defaults to 'dot'.
        Can also be: 'bar', 'violin', 'compact_dot'
    skip_shap: bool
        Whether to skip SHAP entirely.

    Returns
    -------
    Dict: A dictionary of common pairwise metrics.
    """
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # filename base will be title but lower and underscores
    base_name = title.lower().replace(" ", "_")
    if hasattr(classifier, "classifier"):
        classifier = classifier.classifier

    y_prob = classifier.predict_proba(X)[:, 1]

    # plot AUROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(0, figsize=(15, 15))
    plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.xlim((-0.01, 1.0))
    plt.ylim((0.0, 1.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {title}")
    plt.legend(loc="lower right")
    plt.savefig(join(figs_path, base_name + "_roc.png"))
    plt.clf()
    plt.close()

    # plot AUPR
    precision, recall, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(1, figsize=(15, 15))
    plt.plot(
        precision,
        recall,
        lw=2,
        label="PR curve (average precision = %0.2f)" % avg_precision,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"PR Curve for {title}")
    plt.legend(loc="lower left")
    plt.savefig(join(figs_path, base_name + "_pr.png"))
    plt.clf()
    plt.close()

    # plot SHAP
    if not skip_shap:
        shap_values = _positive_class_shap_values(classifier, X)
        plt.figure(2)
        shap.summary_plot(
            shap_values,
            X,
            plot_type=shap_plot_type,
            feature_names=shap_feature_names,
            show=False,
            max_display=len(shap_feature_names),
        )
        plt.tight_layout()
        plt.savefig(join(figs_path, base_name + "_shap.png"))
        plt.clf()
        plt.close()

    # collect metrics and return
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_prob > thresh_for_f1, beta=1.0, average="macro")
    metrics = {
        "AUROC": np.round(roc_auc, 3),
        "Average Precision": np.round(avg_precision, 3),
        "F1": np.round(f1, 3),
        "Precision": np.round(pr, 3),
        "Recall": np.round(rc, 3),
    }

    return metrics


def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def b3_precision_recall_fscore(true_clus, pred_clus, skip_papers=None):
    """
    Compute the B^3 variant of precision, recall and F-score.
    Modified from: https://github.com/glouppe/beard/blob/master/beard/metrics/clustering.py

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the ground-truth paper id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the predicted paper id assignments as values.
    skip_papers: List[string]
        papers to leave out of the evaluation, such as partially supervised ones

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    Dict: P/R/F1 per paper

    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """

    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set()
    for v in true_clusters.values():
        tcset.update(v)
    pcset = set()
    for v in pred_clusters.values():
        pcset.update(v)

    if tcset != pcset:
        raise ValueError("Predictions do not cover all the papers!")

    # skip_papers are observed papers, which we skip for b3 calc.
    if skip_papers is not None:
        tcset = tcset.difference(skip_papers)

    # anything from orphan clusters will also be skipped
    # find orphan clusters, which are clusters that endwith ORPHAN_CLUSTER_KEY
    orphan_clusters_names = [k for k in true_clusters.keys() if k.endswith(ORPHAN_CLUSTER_KEY)]
    orphan_paper_ids = set()
    for to_skip in orphan_clusters_names:
        orphan_paper_ids.update(true_clusters[to_skip])

    # need to remove orphans from true_clusters and pred_clusters
    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(set(cluster).difference(orphan_paper_ids))

    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(set(cluster).difference(orphan_paper_ids))

    # invert lookups
    reverse_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            reverse_true_clusters[vi] = k

    reverse_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            reverse_pred_clusters[vi] = k

    intersections = {}
    per_paper_metrics = {}
    precision = 0.0
    recall = 0.0
    n_samples = 0
    true_bigger_ratios, pred_bigger_ratios = [], []
    for item in sorted(tcset):
        if item in orphan_paper_ids:
            continue  # skipping orphans
        pred_cluster_i = pred_clusters[reverse_pred_clusters[item]]
        true_cluster_i = true_clusters[reverse_true_clusters[item]]

        if len(pred_cluster_i) >= len(true_cluster_i):
            pred_bigger_ratios.append(len(pred_cluster_i) / len(true_cluster_i))
        else:
            true_bigger_ratios.append(len(true_cluster_i) / len(pred_cluster_i))

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection
        _precision = len(intersection) / len(pred_cluster_i)
        _recall = len(intersection) / len(true_cluster_i)
        precision += _precision
        recall += _recall
        per_paper_metrics[item] = (
            _precision,
            _recall,
            f1_score(_precision, _recall),
        )
        n_samples += 1

    precision /= n_samples
    recall /= n_samples

    f_score = f1_score(precision, recall)

    return (
        np.round(precision, 5),
        np.round(recall, 5),
        np.round(f_score, 5),
        per_paper_metrics,
        pred_bigger_ratios,
        true_bigger_ratios,
    )
