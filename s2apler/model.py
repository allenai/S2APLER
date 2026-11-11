from __future__ import annotations

from s2apler.eval import b3_precision_recall_fscore
from s2apler.featurizer import FeaturizationInfo, many_pairs_featurize, paper_featurizer
from s2apler.data import PDData, condensed_offsets
from s2apler.consts import LARGE_INTEGER
from s2apler.native_scorer import scoring_classifier
from s2and.thread_config import resolve_n_jobs

from typing import Dict, Optional, Any, Union, List, Tuple, Set, cast
from collections import defaultdict
import warnings
from functools import partial
from tqdm import tqdm
import logging
import copy

import numpy as np
from scipy.cluster.hierarchy import fcluster
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
from fastcluster import linkage

import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import EfficiencyWarning

logger = logging.getLogger("s2apler")


def _split_null_title_papers(
    block_dict: Dict[str, List[str]], dataset: PDData
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Splits each block into titled papers, which the model clusters, and null-title papers.

    Parameters
    ----------
    block_dict: Dict[str, List[str]]
        block key to paper ids
    dataset: PDData
        the dataset, which supplies each paper's title

    Returns
    -------
    Dict[str, List[str]]: titled paper ids per block, for blocks that have any
    Dict[str, List[str]]: null-title paper ids per block, for blocks that have any
    """
    titled: Dict[str, List[str]] = defaultdict(list)
    null_titled: Dict[str, List[str]] = defaultdict(list)
    for block_key, paper_ids in block_dict.items():
        for paper_id in paper_ids:
            title = dataset.papers[paper_id].title
            if title is None or len(title) == 0:
                null_titled[block_key].append(paper_id)
            else:
                titled[block_key].append(paper_id)
    return dict(titled), dict(null_titled)


def _join_clusters_by_ids(
    paper_ids: List[str],
    labels: List[int],
    dataset: PDData,
    first_null_label: int,
    join_titled: bool,
    dont_merge_cluster_seeds: bool = True,
) -> List[int]:
    """Joins clusters whose papers share an identifier.

    Titled papers join first, transitively, by union-find. That replaces an in-place label
    remap that lost track of earlier joins and left clusters sharing an ID apart. Null-title
    papers then join each other and attach to at most one titled cluster. They never bridge
    two titled clusters: a title-less record often carries one item's DOI and a multi-item
    PDF's hash, and bridging would merge distinct papers.

    An arXiv id joins like any other identifier, across different published DOIs too. A
    preprint, its conference paper and its journal version are one work to a reader.

    Parameters
    ----------
    paper_ids: List[str]
        papers in the block, titled papers first
    labels: List[int]
        cluster label for each paper; titled papers' labels are below first_null_label,
        and each null-title paper has its own label from first_null_label up
    dataset: PDData
        the dataset, which supplies each paper's identifiers
    first_null_label: int
        the smallest label given to a null-title paper
    join_titled: bool
        whether titled clusters sharing an identifier join each other
    dont_merge_cluster_seeds: bool
        whether papers seeded into different "require" clusters stay apart, as in get_constraint

    Returns
    -------
    List[int]: labels where each joined group takes its smallest label, and groups holding
        only null-title papers are numbered from first_null_label in order of appearance
    """
    parent = {label: label for label in labels}
    # A pair the cluster seeds disallow never joins, even through a shared identifier, so a
    # split pinned by a seed stays split. Only seeds between papers of this block matter.
    label_of = dict(zip(paper_ids, labels))
    partners: Dict[str, List[str]] = {}
    if dataset.cluster_seeds_disallow:
        for paper_id, others in dataset._disallow_partners().items():
            inside = [other for other in others if other in label_of]
            if paper_id in label_of and inside:
                partners[paper_id] = inside
    seeded: Dict[int, List[str]] = defaultdict(list)
    for paper_id in partners:
        seeded[label_of[paper_id]].append(paper_id)
    # get_constraint keeps papers of different "require" seeds apart; the join must too.
    # Each group tracks the require seeds its papers carry.
    require_seeds: Dict[int, Set[int]] = defaultdict(set)
    if dont_merge_cluster_seeds:
        for paper_id, label in label_of.items():
            if paper_id in dataset.cluster_seeds_require:
                require_seeds[label].add(dataset.cluster_seeds_require[paper_id])
    constrained = bool(partners) or bool(require_seeds)

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(label_a: int, label_b: int, may_join_titled: bool):
        root_a, root_b = find(label_a), find(label_b)
        if root_a == root_b:
            return
        # A root below first_null_label means the group already holds a titled cluster.
        if not may_join_titled and root_a < first_null_label and root_b < first_null_label:
            return
        if partners:
            # Seeds are symmetric, so scanning the group with fewer seeded papers suffices.
            fewer, more = sorted((root_a, root_b), key=lambda root: len(seeded.get(root, ())))
            if any(find(label_of[other]) == more for paper_id in seeded.get(fewer, ()) for other in partners[paper_id]):
                return
        seeds_a, seeds_b = require_seeds.get(root_a), require_seeds.get(root_b)
        if seeds_a and seeds_b and len(seeds_a | seeds_b) > 1:
            return
        kept, merged = min(root_a, root_b), max(root_a, root_b)
        parent[merged] = kept
        if merged in seeded:
            seeded[kept].extend(seeded.pop(merged))
        if merged in require_seeds:
            require_seeds[kept] |= require_seeds.pop(merged)

    # A join refused by a seed leaves several groups behind one key. Then try each, so a
    # paper the first group refuses can still join another. Without seeds, retrying cannot
    # change the result: a refused join means both groups hold titled clusters, as does
    # every other group left behind the key.
    groups_for_key: Dict[Tuple[str, str], List[int]] = {}
    for paper_id, label in zip(paper_ids, labels):
        is_titled = label < first_null_label
        for key in dataset.id_merge_keys(paper_id):
            earlier = groups_for_key.get(key, [])
            for other in earlier:
                union(other, label, may_join_titled=join_titled and is_titled)
            if constrained:
                groups_for_key[key] = list(dict.fromkeys(find(group) for group in [*earlier, label]))
            elif not earlier:
                groups_for_key[key] = [label]

    # Number null-only groups consecutively, as before, so unchanged clusters keep their keys.
    renumbered: Dict[int, int] = {}
    joined = []
    for label in labels:
        root = find(label)
        if root >= first_null_label:
            root = renumbered.setdefault(root, first_null_label + len(renumbered))
        joined.append(root)
    return joined


def _supervision_by_block(
    block_dict: Dict[str, List[str]], partial_supervision: Dict[Tuple[str, str], Union[int, float]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Places partial supervision into blocks once, as condensed offsets and values.

    distance_matrix_helper checks a pair in block order before the reverse order, so an entry
    given in block order wins over one given in reverse.

    Parameters
    ----------
    block_dict: Dict
        block key to paper ids
    partial_supervision: Dict
        (paper id, paper id) to the value to use for that pair

    Returns
    -------
    Dict: block key to (offsets, values) for the blocks with any supervised pair
    """
    if not partial_supervision:
        return {}
    places: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for block_key, papers in block_dict.items():
        for k, paper_id in enumerate(papers):
            places[paper_id].append((block_key, k))
    found: Dict[str, Dict[int, Union[int, float]]] = defaultdict(dict)
    for in_block_order in (False, True):
        for (paper_a, paper_b), value in partial_supervision.items():
            for block_key, i in places.get(paper_a, ()):
                for other_key, j in places.get(paper_b, ()):
                    if other_key != block_key or i == j or (i < j) != in_block_order:
                        continue
                    n = len(block_dict[block_key])
                    found[block_key][int(condensed_offsets(np.array(min(i, j)), np.array(max(i, j)), n))] = value
    return {
        block_key: (np.fromiter(entries, dtype=np.int64), np.array(list(entries.values()), dtype=float))
        for block_key, entries in found.items()
    }


def _condensed_pair_chunks(n: int, max_pairs: int):
    """Yields an n-paper block's pairs in condensed order, a few rows of the matrix at a time.

    Parameters
    ----------
    n: int
        the number of papers in the block
    max_pairs: int
        the most pairs to yield at once, unless one row holds more

    Yields
    ------
    (left, right, first_offset): paper indices of each pair, and the first pair's offset
    """
    row = 0
    while row < n - 1:
        # Row i holds the n - 1 - i pairs (i, j) with j > i.
        end = row + 1
        total = n - 1 - row
        while end < n - 1 and total + (n - 1 - end) <= max_pairs:
            total += n - 1 - end
            end += 1
        rows = np.arange(row, end)
        counts = n - 1 - rows
        left = np.repeat(rows, counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        right = left + 1 + (np.arange(total) - starts)
        yield left, right, row * n - row * (row + 1) // 2
        row = end


class Clusterer:
    """
    A wrapper for learning a clusterer

    Args:
        featurizer_info: FeaturizationInfo
            Featurization information
        classifier: sklearn compatible model
            Classifier which uses pairwise features to make a distance matrix
        val_blocks_size: int
            How many blocks to use during hyperparam optimization.
            Defaults to None, which uses all of them.
        cluster_model: FastCluster
            Clusterer model
            Defaults to None, which uses FastCluster with average linking.
        search_space: Dict
            Search space for the hyperpamater optimization.
            Defaults to None, which uses a space appropriate to FastCluster.
        n_iter: int
            Number of hyperparameter evaluations
        n_jobs: int
            Parallelize each clusterer this many ways
        use_default_constraints_as_supervision: bool
            Whether to use the default constraints when constructing the distance matrices.
            These are high precision and can save a lot of compute/time.
        random_state: int
            Random state
        dont_merge_cluster_seeds: bool
            this flag controls whether to use cluster seeds to enforce "dont merge"
            as well as "must merge" constraints
        batch_size: int
            batch size for featurization, lower means less memory, but slower
    """

    def __init__(
        self,
        featurizer_info: FeaturizationInfo,
        classifier: Any,
        val_blocks_size: Optional[int] = None,
        cluster_model: Optional[FastCluster] = None,
        search_space: Optional[Dict[str, Any]] = None,
        n_iter: int = 25,
        n_jobs: int = 16,
        # Keyword-only: removed parameters used to sit here, so old positional calls must fail.
        *,
        use_default_constraints_as_supervision: bool = True,
        random_state: int = 42,
        dont_merge_cluster_seeds: bool = True,
        batch_size: int = 1000000,
    ):
        self.featurizer_info = featurizer_info
        self.classifier = copy.deepcopy(classifier)
        self.val_blocks_size = val_blocks_size
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_default_constraints_as_supervision = use_default_constraints_as_supervision
        self.dont_merge_cluster_seeds = dont_merge_cluster_seeds
        if cluster_model is None:
            self.cluster_model = FastCluster(linkage="average")
        elif isinstance(cluster_model, FastCluster):
            self.cluster_model = copy.deepcopy(cluster_model)
        else:
            # Distances are built as condensed vectors, which only FastCluster reads.
            raise TypeError(f"cluster_model must be a FastCluster, got {type(cluster_model).__name__}")

        if search_space is None:
            self.search_space = {"eps": hp.uniform("eps", 0.2, 0.8)}
        else:
            self.search_space = search_space

        self.hyperopt_trials_store: Optional[Union[Trials, List[Trials]]] = None
        self.best_params: Optional[Dict[Any, Any]] = None
        self.batch_size = batch_size

    def _scorer(self, classifier: Any) -> Any:
        """Returns what scores pairs for `classifier`: the native booster when it is exact.

        Parameters
        ----------
        classifier: Any
            the pairwise classifier

        Returns
        -------
        Any: an object with `predict_proba`
        """
        # n_jobs below 1 follows scikit-learn (-1 is all CPUs); 0 runs on one thread.
        num_threads = resolve_n_jobs(self.n_jobs) if self.n_jobs != 0 else 1
        return scoring_classifier(classifier, num_threads)

    @staticmethod
    def filter_blocks(block_dict: Dict[str, List[str]], num_to_keep: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Filter out blocks of size 1, as they are not useful or train/val

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        num_to_keep: int
            the number of blocks to keep, keeps all if None

        Returns
        -------
        either the loaded json, or the passed in object
        """
        # blocks with only 1 element are useless for train/val
        # and we can only keep as many as is specified
        out_dict = {}
        count = 0
        for block_key, paper_ids in block_dict.items():
            if len(paper_ids) > 1:
                out_dict[block_key] = paper_ids
                count += 1
                # early stopping if we have enough
                if num_to_keep is not None and count == num_to_keep:
                    return out_dict
        return out_dict

    def distance_matrix_helper(
        self,
        block_dict: Dict,
        dataset: PDData,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]],
        incremental_dont_use_cluster_seeds: bool = False,
    ):
        """
        Helper generator function to yield one pair for batch featurization on the fly

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        dataset: PDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        yields pairs of ((paper id 1, paper id 2, label), index pair into the distance matrix, block key)
        """
        for block_key, papers in block_dict.items():
            for i, j in zip(*np.triu_indices(len(papers), k=1)):
                # subtracting LARGE_INTEGER so many_pairs_featurize knows not to make features
                label = np.nan
                if (papers[i], papers[j]) in partial_supervision:
                    label = partial_supervision[(papers[i], papers[j])] - LARGE_INTEGER
                elif (papers[j], papers[i]) in partial_supervision:
                    label = partial_supervision[(papers[j], papers[i])] - LARGE_INTEGER
                elif self.use_default_constraints_as_supervision:
                    value = dataset.get_constraint(
                        papers[i],
                        papers[j],
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    )
                    if value is not None:
                        label = value - LARGE_INTEGER

                yield ((papers[i], papers[j], label), (i, j), block_key)

    def make_distance_matrices(
        self,
        block_dict: Dict[str, List[str]],
        dataset: PDData,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {},
        disable_tqdm: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Creates the distance matrices for the input blocks.

        Pairs are handled as arrays, not one Python object each: constraints come from
        PDData.constraint_values, the native featurizer takes pair positions, and at most
        batch_size pairs are featurized and scored at a time. The distances are the same as
        building each pair with distance_matrix_helper, which stays as the reference.

        Parameters
        ----------
        block_dict: Dict
            the block dictionary to make distances for
        dataset: PDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        disable_tqdm: bool
            whether to turn off the tqdm progress bars in this function
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        Dict: the distance matrix dictionary, keyed by block key
        """
        logger.debug(f"Making {len(block_dict)} distance matrices")
        paper_ids = list(dict.fromkeys(paper_id for papers in block_dict.values() for paper_id in papers))
        position = {paper_id: i for i, paper_id in enumerate(paper_ids)}
        num_threads = resolve_n_jobs(self.n_jobs) if self.n_jobs != 0 else 1
        featurizer = None
        # Scored pairs wait here until a batch is full: (distances, offsets, left, right).
        pending: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        pending_count = 0

        def score_pending():
            nonlocal featurizer, pending, pending_count
            if featurizer is None:
                featurizer = paper_featurizer(dataset.papers, paper_ids)
            left = np.concatenate([item[2] for item in pending])
            right = np.concatenate([item[3] for item in pending])
            features = featurizer.featurize(left, right, num_threads=num_threads)
            features = features[:, self.featurizer_info.indices_to_use]
            # index 0 is p(not the same)
            probabilities = self._scorer(self.classifier).predict_proba(features)[:, 0]
            start = 0
            for distances, offsets, _, _ in pending:
                distances[offsets] = probabilities[start : start + len(offsets)]
                start += len(offsets)
            pending, pending_count = [], 0

        supervision = _supervision_by_block(block_dict, partial_supervision)
        distance_matrices = {}
        for block_key, papers in tqdm(block_dict.items(), desc="Making distance matrices", disable=disable_tqdm):
            n = len(papers)
            # flattened pdist style
            distances = np.zeros(n * (n - 1) // 2, dtype=np.float16)
            distance_matrices[block_key] = distances
            block_positions = np.array([position[paper_id] for paper_id in papers], dtype=np.uint32)
            pair_rules = dataset.block_pair_rules(papers) if self.use_default_constraints_as_supervision else None
            for left, right, first_offset in _condensed_pair_chunks(n, self.batch_size):
                labels = self._pair_labels(
                    papers,
                    left,
                    right,
                    first_offset,
                    dataset,
                    supervision.get(block_key),
                    incremental_dont_use_cluster_seeds,
                    pair_rules,
                )
                offsets = np.arange(first_offset, first_offset + len(left))
                fixed = ~np.isnan(labels)
                # The same arithmetic as the per-pair path, which offset labels by LARGE_INTEGER.
                distances[offsets[fixed]] = (labels[fixed] - LARGE_INTEGER) + LARGE_INTEGER
                scored = ~fixed
                if np.any(scored):
                    pending.append(
                        (distances, offsets[scored], block_positions[left[scored]], block_positions[right[scored]])
                    )
                    pending_count += int(np.count_nonzero(scored))
                if pending_count >= self.batch_size:
                    score_pending()
        if pending:
            score_pending()

        logger.debug(f"{len(block_dict)} distance matrices made")
        return distance_matrices

    def _pair_labels(
        self,
        papers: List[str],
        left: np.ndarray,
        right: np.ndarray,
        first_offset: int,
        dataset: PDData,
        supervision: Optional[Tuple[np.ndarray, np.ndarray]],
        incremental_dont_use_cluster_seeds: bool,
        pair_rules: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Labels for pairs of one block, as distance_matrix_helper gives them, without its offset.

        Parameters
        ----------
        papers: List[str]
            the block's paper ids
        left: np.ndarray
            the first paper of each pair, as an index into papers
        right: np.ndarray
            the second paper of each pair
        first_offset: int
            the condensed offset of the first pair
        dataset: PDData
            the dataset
        supervision: Tuple[np.ndarray, np.ndarray]
            the block's partial supervision from _supervision_by_block, or None
        incremental_dont_use_cluster_seeds: bool
            whether to ignore the dataset's cluster seeds
        pair_rules: Tuple[np.ndarray, np.ndarray]
            dataset.block_pair_rules for the block

        Returns
        -------
        np.ndarray: each pair's partial supervision or constraint value, NaN where neither applies
        """
        if self.use_default_constraints_as_supervision:
            labels = dataset.constraint_values(
                papers,
                left,
                right,
                first_offset,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                pair_rules=pair_rules,
            )
        else:
            labels = np.full(len(left), np.nan)
        if supervision is not None:
            # Partial supervision overrides constraints.
            offsets, values = supervision
            in_chunk = (offsets >= first_offset) & (offsets < first_offset + len(labels))
            labels[offsets[in_chunk] - first_offset] = values[in_chunk]
        return labels

    def fit(
        self,
        datasets: Union[PDData, List[PDData]],
        val_dists_precomputed: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        metric_for_hyperopt: str = "b3",
    ) -> Clusterer:
        """
        Fits the clusterer

        Parameters
        ----------
        datasets: List[PDData]
            the list of datasets to use for validations
        val_dists_precomputed: Dict
            precomputed distance matrices
        metric_for_hyperopt: string
            the metric to use for hyperparamter optimization

        Returns
        -------
        Clusterer: a fit clusterer, also sets the best params
        """
        assert metric_for_hyperopt in {"b3", "ratio"}
        logger.debug("Fitting clusterer")
        if isinstance(datasets, PDData):
            datasets = [datasets]
        val_block_dict_list = []
        val_cluster_to_papers_list = []
        val_dists_list = []
        weights = []
        for dataset in datasets:
            # blocks
            # the block-wise split uses only the validation blocks to fit the clustering model
            _, val_block_dict, _ = dataset.split_cluster_papers()

            # we don't need val blocks with only a single element
            val_block_dict = self.filter_blocks(val_block_dict, self.val_blocks_size)
            val_block_dict_list.append(val_block_dict)

            # block ground truth labels: cluster_to_papers
            val_cluster_to_papers = dataset.construct_cluster_to_papers(val_block_dict)
            val_cluster_to_papers_list.append(val_cluster_to_papers)

            # distance matrix
            if val_dists_precomputed is None:
                # predict clusters titled papers only, so its distances must cover exactly those.
                titled_val_block_dict, _ = _split_null_title_papers(val_block_dict, dataset)
                val_dists = self.make_distance_matrices(titled_val_block_dict, dataset)
            else:
                val_dists = val_dists_precomputed[dataset.name]
            val_dists_list.append(val_dists)

            # weights for weighted F1 average: total # of paper_ids in dataset
            weights.append(np.sum([len(i) for i in val_block_dict.values()]))

        def obj(params):
            self.set_params(params)
            f1s = []
            ratios = []
            for dataset, val_block_dict, val_cluster_to_papers, val_dists in zip(
                datasets, val_block_dict_list, val_cluster_to_papers_list, val_dists_list
            ):
                # predict reads paper titles to set aside null-title papers, so it needs the dataset.
                pred_clusters, _ = self.predict(val_block_dict, dataset=dataset, dists=val_dists)
                (
                    _,
                    _,
                    f1,
                    _,
                    pred_bigger_ratios,
                    true_bigger_ratios,
                ) = b3_precision_recall_fscore(val_cluster_to_papers, pred_clusters)
                ratios.append(np.mean(pred_bigger_ratios + true_bigger_ratios))
                f1s.append(f1)
            if metric_for_hyperopt == "ratio":
                return np.average(ratios, weights=weights)
            elif metric_for_hyperopt == "b3":
                # minimize means we need to negate
                return -np.average(f1s, weights=weights)

        self.hyperopt_trials_store = Trials()
        _ = fmin(
            fn=obj,
            space=self.search_space,
            algo=partial(tpe.suggest, n_startup_jobs=5),
            max_evals=self.n_iter,
            trials=self.hyperopt_trials_store,
            # hyperopt calls rstate.integers, which only numpy Generators have.
            rstate=np.random.default_rng(self.random_state),
        )
        # hyperopt has some problems with hp.choice so we need to do this:
        best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
        self.best_params = {k: intify(v) for k, v in best_params.items()}
        self.set_params(self.best_params)

        logger.debug("Clusterer fit")
        return self

    def set_params(self, params: Optional[Dict[str, Any]], clone_flag: bool = False):
        """
        Sets params on the cluster model

        Parameters
        ----------
        params: Dict
            the params to set
        clone_flag: bool
            whether to return a clone of the cluster model
        """
        if params is None:
            params = {}
        else:
            params = {k: intify(v) for k, v in params.items()}
        if clone_flag:
            cluster_model = clone(self.cluster_model)
            cluster_model.set_params(**params)
            return cluster_model
        else:
            self.cluster_model.set_params(**params)

    def predict(
        self,
        block_dict: Dict[str, List[str]],
        dataset: PDData,
        dists: Optional[Dict[str, np.ndarray]] = None,
        cluster_model_params: Optional[Dict[str, Any]] = None,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {},
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Tuple[Dict[str, List[str]], Optional[Dict[str, np.ndarray]]]:
        """
        Predicts clusters

        Parameters
        ----------
        block_dict: Dict
            the block dict to predict clusters from
        dataset: PDData
            the dataset
        dists: Dict
            (optional) precomputed distance matrices
        cluster_model_params: Dict
            params to set on the cluster model
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        use_s2_clusters: bool
            whether to "predict" using the clusters from Semantic Scholar's old system
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        Dict: the predicted clusters
        Dict: the predicted distance matrices
        """

        pred_clusters = defaultdict(list)

        if use_s2_clusters:
            for _, papers_list in block_dict.items():
                for _paper in papers_list:
                    s2_cluster_key = dataset.papers[_paper].corpus_paper_id
                    if s2_cluster_key is None:
                        s2_cluster_key = _paper  # this means S2 didn't cluster it with anything, so it's by itself
                    pred_clusters[s2_cluster_key].append(_paper)

            return dict(pred_clusters), dists

        # we need to remove all the null titles from the block_dict and then reattach them later
        block_dict_no_null_titles, block_dict_null_titles = _split_null_title_papers(block_dict, dataset)
        block_dict = block_dict_no_null_titles

        if dists is None:
            dists = self.make_distance_matrices(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            )

        # Walk every input block, not just blocks with titled papers: a block whose papers all
        # lack titles used to vanish from the output.
        all_block_keys = list(dict.fromkeys(list(block_dict_no_null_titles) + list(block_dict_null_titles)))
        for block_key in all_block_keys:
            titled_paper_ids = block_dict.get(block_key, [])
            if len(titled_paper_ids) > 1:
                if block_key not in dists:
                    raise ValueError(f"No distance matrix for block {block_key!r} with {len(titled_paper_ids)} papers")
                # Distances are read by position, so a matrix built over other papers mislabels silently.
                n = len(titled_paper_ids)
                expected_shape = (n * (n - 1) // 2,)
                if dists[block_key].shape != expected_shape:
                    raise ValueError(
                        f"Distance matrix for block {block_key!r} has shape {dists[block_key].shape}, "
                        f"expected {expected_shape} for its {n} titled papers"
                    )
                cluster_model = self.set_params(cluster_model_params, clone_flag=True)
                with warnings.catch_warnings():
                    # annoying sparse matrix not sorted warning
                    warnings.simplefilter("ignore", category=EfficiencyWarning)
                    cluster_model.fit(dists[block_key])
                labels = [int(label) for label in cluster_model.labels_]
            else:
                labels = [0] * len(titled_paper_ids)

            # Null-title papers skip the model; each starts as a singleton and joins others by ID.
            null_title_paper_ids = block_dict_null_titles.get(block_key, [])
            next_label = max(labels, default=-1) + 1
            paper_ids_for_block = titled_paper_ids + null_title_paper_ids
            labels = labels + list(range(next_label, next_label + len(null_title_paper_ids)))

            # The enforced 0 distance can lose to many large distances under average linkage,
            # so clusters sharing a DOI, PMID, usable pdf_hash or arXiv id are joined explicitly here.
            labels = _join_clusters_by_ids(
                paper_ids_for_block,
                labels,
                dataset,
                first_null_label=next_label,
                join_titled=self.use_default_constraints_as_supervision,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
            )

            for paper_id, label in zip(paper_ids_for_block, labels):
                pred_clusters[str(block_key) + "_" + str(label)].append(paper_id)

        return dict(pred_clusters), dists

    def predict_incremental(self, block_papers: List[str], dataset: PDData):
        """
        Predict clustering in incremental mode. This assumes that the majority of the labels are passed
        in using the cluster_seeds parameter of the dataset class, and skips work by simply assigning each
        unassigned paper to the closest cluster if distance is less than eps, and then separately cluster all
        the unassigned papers that are not within eps of any existing cluster.

        Corrected, claimed profiles should be noted via the altered_cluster_papers parameter (in PDData).
        Then predict_incremental performs a pre-clustering step on each altered cluster to determine how
        S2APLER would divide it into clusters. Mentions are incrementally added to these new subclusters,
        then reassembled to restore the complete claimed profile when S2APLER returns results.

        Currently this would be useful in the following situation. We have a massive block, for which we want
        to cluster a small number of new papers into (block size * number of new papers should be less
        than the normal batch size).

        Note: this function was designed to work on a single block at a time.

        Parameters
        ----------
        block_papers: List[str]
            the paper_ids in the block to predict from
        dataset: PDData
            the dataset

        Returns
        -------
        Dict: the predicted clusters
        """
        recluster_map = {}
        cluster_seeds_require = copy.deepcopy(dataset.cluster_seeds_require)
        if dataset.altered_cluster_papers is not None:
            altered_cluster_nums = set(
                dataset.cluster_seeds_require[altered_paper_id] for altered_paper_id in dataset.altered_cluster_papers
            )
            if len(altered_cluster_nums) > 0:
                cluster_seeds_require_inverse: Dict[int, list] = {}
                for papers_id, cluster_num in dataset.cluster_seeds_require.items():
                    if cluster_num not in cluster_seeds_require_inverse:
                        cluster_seeds_require_inverse[cluster_num] = []
                    cluster_seeds_require_inverse[cluster_num].append(papers_id)
                for altered_cluster_num in altered_cluster_nums:
                    papers_ids_for_cluster_num = cluster_seeds_require_inverse[altered_cluster_num]

                    # Note: incremental_dont_use_cluster_seeds is set to True, because at this stage
                    # of incremental clustering we are splitting up the claimed profiles that we received
                    # from production so that they align with s2aplers's predictions. When doing this, we
                    # don't want to use the passed in cluster seeds, because they reflect the claimed profile, not
                    # s2apler's predictions
                    reclustered_output, _ = self.predict(
                        {"block": papers_ids_for_cluster_num},
                        dataset,
                        incremental_dont_use_cluster_seeds=True,
                    )
                    if len(reclustered_output) > 1:
                        for i, new_cluster_of_papers in enumerate(reclustered_output.values()):
                            new_cluster_num = str(altered_cluster_num) + f"_{i}"
                            recluster_map[new_cluster_num] = altered_cluster_num
                            for reclustered_papers_id in new_cluster_of_papers:
                                cluster_seeds_require[reclustered_papers_id] = new_cluster_num  # type: ignore

        all_pairs = []
        for possibly_unassigned_paper in block_papers:
            if possibly_unassigned_paper in cluster_seeds_require:
                continue
            unassigned_paper = possibly_unassigned_paper
            for paper_id in cluster_seeds_require.keys():
                label = np.nan
                if self.use_default_constraints_as_supervision:
                    value = dataset.get_constraint(
                        unassigned_paper,
                        paper_id,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                    )
                    if value is not None:
                        label = value - LARGE_INTEGER
                all_pairs.append((unassigned_paper, paper_id, label))

        batch_features, _ = many_pairs_featurize(all_pairs, dataset, self.featurizer_info, self.n_jobs)

        # get predictions where there isn't partial supervision
        # and fill the rest with partial supervision
        # undoing the offset by LARGE_INTEGER from above
        logger.debug("Making predict flags")
        batch_labels = np.array([i[2] for i in all_pairs])
        predict_flag = np.isnan(batch_labels)
        not_predict_flag = ~predict_flag
        batch_predictions = np.zeros(len(batch_features))
        # index 0 is p(not the same)
        logger.debug("Pairwise classification")
        if np.any(predict_flag):
            batch_predictions[predict_flag] = self._scorer(self.classifier).predict_proba(
                batch_features[predict_flag, :]
            )[:, 0]
        if np.any(not_predict_flag):
            batch_predictions[not_predict_flag] = batch_labels[not_predict_flag] + LARGE_INTEGER

        logger.debug("Computing average distances for unassigned papers")
        papers_to_cluster_to_average_dist: Dict[str, Dict[int, Tuple[float, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0, 0))
        )
        for papers_pair, dist in zip(all_pairs, batch_predictions):
            unassigned_paper, assigned_paper, _ = papers_pair
            cluster_id = cluster_seeds_require[assigned_paper]
            previous_average, previous_count = papers_to_cluster_to_average_dist[unassigned_paper][cluster_id]
            papers_to_cluster_to_average_dist[unassigned_paper][cluster_id] = (
                (previous_average * previous_count + dist) / (previous_count + 1),
                previous_count + 1,
            )

        logger.debug("Assigning unassigned papers")
        pred_clusters = defaultdict(list)
        singleton_papers = []
        for papers_id, cluster_id in dataset.cluster_seeds_require.items():
            pred_clusters[f"{cluster_id}"].append(papers_id)
        for (
            unassigned_paper,
            cluster_dists,
        ) in papers_to_cluster_to_average_dist.items():
            best_cluster_id = None
            best_dist = float("inf")
            for cluster_id, (average_dist, _) in cluster_dists.items():
                if average_dist < best_dist and average_dist < self.cluster_model.eps:
                    best_cluster_id = cluster_id
                    best_dist = average_dist
            if best_cluster_id is not None:
                # undo the reclustering step
                if best_cluster_id in recluster_map:
                    best_cluster_id = recluster_map[best_cluster_id]
                pred_clusters[f"{best_cluster_id}"].append(unassigned_paper)
            else:
                singleton_papers.append(unassigned_paper)

        reclustered_output, _ = self.predict({"block": singleton_papers}, dataset)
        new_cluster_id = dataset.max_seed_cluster_id or 0
        for new_cluster in reclustered_output.values():
            pred_clusters[str(new_cluster_id)] = new_cluster
            new_cluster_id += 1

        logger.debug("Returning incrementally predicted clusters")
        return dict(pred_clusters)


class PairwiseModeler:
    """
    Wrapper to learn the pairwise model + hyperparameter optimization

    Parameters
    ----------
    estimator: sklearn compatible classifier
        A binary classifier with fit/predict interface.
        Defaults to LGBMClassifier if not specified. Will be cloned.
    search_space: Dict:
            A hyperopt search space for hyperparam optimization.
            Defaults to an appropriate LGBMClassifier space if not specified.
    n_iter: int
        Number of iterations for hyperparam optimization.
    n_jobs: int
        Parallelization for the classifier.
        Note: the hyperopt is serial, but can be made semi-parallel with batch search.
    random_state: int
        Random state for classifier and hyperopt.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        search_space: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        n_jobs: int = 16,  # for the model, not the hyperopt
        random_state: int = 42,
    ):
        if estimator is None:
            self.estimator = lgb.LGBMClassifier(
                objective="binary",
                metric="average_precision",  # lightgbm doesn't do F1 directly
                n_jobs=n_jobs,
                verbose=-1,
                tree_learner="data",
                random_state=random_state,
            )
        else:
            self.estimator = clone(estimator)

        if search_space is None:
            self.search_space = {
                "learning_rate": hp.loguniform("learning_rate", -7, 0),
                "num_leaves": scope.int(hp.qloguniform("num_leaves", 2, 7, 1)),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                "subsample": hp.uniform("subsample", 0.5, 1),
                "min_child_samples": scope.int(hp.qloguniform("min_child_samples", 3, 9, 1)),
                "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
                "reg_alpha": hp.loguniform("reg_alpha", -16, 2),
                "reg_lambda": hp.loguniform("reg_lambda", -16, 2),
                "n_estimators": scope.int(hp.quniform("n_estimators", 1000, 2500, 1)),
                "max_depth": scope.int(hp.quniform("max_depth", 1, 100, 1)),
                "min_split_gain": hp.uniform("min_split_gain", 0, 2),
            }
        else:
            self.search_space = search_space

        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params: Optional[Dict] = None
        self.hyperopt_trials_store: Optional[Trials] = None
        self.classifier: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[Trials]:
        """
        Fits the classifier

        Parameters
        ----------
        X_train: np.ndarray
            feature matrix for the training set
        y_train: np.ndarray
            labels for the training set
        X_val: np.ndarray
            feature matrix for the validation set
        y_val: np.ndarray
            labels for the validation set

        Returns
        -------
        Trials: the Trials object from hyperparameter optimization
        """
        if len(self.search_space) > 0:

            def obj(params):
                params = {k: intify(v) for k, v in params.items()}
                self.estimator.set_params(**params)
                self.estimator.fit(X_train, y_train)
                y_pred_proba = np.asarray(self.estimator.predict_proba(X_val))[:, 1]
                return -precision_recall_fscore_support(y_val, y_pred_proba > 0.5, beta=1.0, average="macro")[2]

            self.hyperopt_trials_store = Trials()
            _ = fmin(
                fn=obj,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.n_iter,
                trials=self.hyperopt_trials_store,
                rstate=np.random.default_rng(self.random_state),
            )
            best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
            self.best_params = {k: intify(v) for k, v in best_params.items()}
            self.estimator.set_params(**self.best_params)
        else:
            self.best_params = {}
            # No search ran, so drop trials from any earlier fit (this was a no-op comparison).
            self.hyperopt_trials_store = None

        # refitting but only on training data so as not to leak anything
        self.classifier = self.estimator.fit(X_train, y_train)

        return self.hyperopt_trials_store

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classifier is not None, "You need to call fit first"
        return self.classifier.predict_proba(X)


def intify(x):
    """Hyperopt is bad at ints..."""
    if hasattr(x, "is_integer") and x.is_integer():
        return int(x)
    else:
        return x


class FastCluster(TransformerMixin, BaseEstimator):
    """
    A scikit-learn wrapper for fastcluster.
    Inputs:
        linkage: string (default="average")
            Agglomerative linkage method. Defaults to "average".
            Must be one of "'complete', 'average', 'single,
            'weighted', 'ward', 'centroid', 'median'."
        eps: float (default = 0.65)
            Cutoff used to determine number of clusters.
        preserve_input: bool (default=True)
            Whether to preserve the X input or modify in place.
            Defaults to False, which modifies in place.
        input_as_observation_matrix: bool (default=False)
            If True, the input to fit/transform must be a 2-D array
            of observation vectors (N by d). If False input to fit/transform
            must be a 1-D condensed distance matrix, then it must be a
            (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix, and
            d is the dimensionality of the vector space.

    Note: FastCluster does *not* support two-dimensional distance matrices
    as input. They *must* be flattened. For more details, please see:
    https://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf
    """

    def __init__(
        self,
        linkage: str = "average",
        eps: float = 0.65,
        preserve_input: bool = True,
        input_as_observation_matrix: bool = False,
    ):
        if linkage not in {
            "complete",
            "average",
            "weighted",
            "ward",
            "centroid",
            "median",
            "single",
        }:
            raise Exception(
                "The 'linkage' parameter has to be one of: "
                + "'single', complete', 'average', 'weighted', 'ward', 'centroid', 'median'."
            )

        self.linkage = linkage
        self.eps = eps
        self.preserve_input = preserve_input
        self.input_as_observation_matrix = input_as_observation_matrix
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "FastCluster":
        """
        Fit the estimator on input data. The results are stored in self.labels_.
        Parameters
        ----------
        X: np.ndarray
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix. If X is 2-D
            then the flag `input_as_observation_matrix` must be set to True in init.
        Returns
        -------
        self
        """
        X = np.asarray(X)
        if len(X.shape) == 1 and self.input_as_observation_matrix:
            raise Exception(
                "Input to fit is one-dimensional, but input_as_observation_matrix flag is set to True. "
                "If you intended to pass in an observation matrix, it must be 2-D (N x feature_dimension)."
            )
        elif len(X.shape) == 2 and not self.input_as_observation_matrix:
            raise Exception(
                "Input to fit is two-dimensional, but input_as_observation_matrix flag is set to False. "
                "If you intended to pass in a distance matrix, it must be flattened (1-D)."
            )
        elif len(X.shape) > 2:
            raise Exception("The input to fit can only be one-dimensional or two-dimensional.")
        Z = linkage(X, self.linkage, preserve_input=self.preserve_input)
        self.labels_ = fcluster(Z, t=self.eps, criterion="distance")
        return self

    def fit_transform(self, X: np.ndarray, y: Any = None, **fit_params: Any) -> np.ndarray:
        """
        Fit the estimator on input data, and returns results.
        Parameters
        ----------
        X: np.ndarray
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix.
        y: Any
            ignored; present for scikit-learn API compatibility
        **fit_params: Any
            ignored; present for scikit-learn API compatibility
        Returns
        -------
        np.ndarray: A N-length array of clustering labels.
        """
        return cast(np.ndarray, self.fit(X).labels_)

    def transform(self, X: np.ndarray):
        raise Exception("FastCluster has no inductive mode. Use 'fit' or 'fit_transform' instead.")
