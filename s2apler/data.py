from typing import Optional, Union, Dict, List, Any, Tuple, Set, NamedTuple

import random
import json
import numpy as np
import pandas as pd
import pickle
import logging
import multiprocessing
from tqdm import tqdm

from collections import defaultdict, Counter

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


from s2apler.consts import (
    NUMPY_NAN,
    LARGE_DISTANCE,
    CLUSTER_SEEDS_LOOKUP,
    ORPHAN_CLUSTER_KEY,
    DEFAULT_CHUNK_SIZE,
)
from s2apler.text import (
    normalize_text,
    get_text_ngrams,
    get_text_ngrams_words,
    normalize_venue_name,
    NAME_PREFIXES,
    A_THROUGH_Z,
    PUBLISHER_SOURCES,
)

logger = logging.getLogger("s2apler")


class Author(NamedTuple):
    author_info_first: str
    author_info_first_normalized_without_apostrophe: Optional[str]
    author_info_middle: str
    author_info_middle_normalized_without_apostrophe: Optional[str]
    author_info_last_normalized: Optional[str]
    author_info_last: str
    author_info_suffix_normalized: Optional[str]
    author_info_suffix: str
    author_info_full_name: Optional[str]
    author_info_first_letters: Optional[set]


class Paper(NamedTuple):
    title: str
    abstract: Optional[str]
    has_abstract: Optional[bool]
    title_ngrams_words: Optional[Counter]
    abstract_ngrams_words: Optional[Counter]
    authors: List[Author]
    venue: Optional[str]
    journal_name: Optional[str]
    title_ngrams_chars: Optional[Counter]
    venue_ngrams: Optional[Counter]
    journal_ngrams: Optional[Counter]
    author_info_coauthor_n_grams: Optional[Counter]
    author_info_coauthor_email_prefix_n_grams: Optional[Counter]
    author_info_coauthor_email_suffix_n_grams: Optional[Counter]
    author_info_coauthor_affiliations_n_grams: Optional[Counter]
    year: Optional[int]
    sourced_paper_id: int
    corpus_paper_id: Optional[int]  # this is the id of the paper's current mapping in the corpus
    doi: Optional[str]
    pmid: Optional[str]
    source_id: Optional[str]
    pdf_hash: Optional[str]
    source: Optional[str]
    block: Optional[str]


class PDData:
    """
    The main class for holding our representation of an paper disambiguation data

    Input:
        papers: path to the papers information json file (or the json object)
        name: name of the dataset, used for caching computed features
        mode: 'train' or 'inference'; if 'inference', everything related to splitting will be ignored
        clusters: path to the clusters json file (or the json object)
            - a cluster may span multiple blocks, but we will only consider in-block clusters
            - there will be individual papers that definitely do not belong to any of the known clusters
              but which may or may not cluster with each other.
              papers in these clusters will all appear in clusters.json under the key clusterid_ORPHAN_CLUSTER_KEY
        specter_embeddings: path to the specter embeddings pickle (or the dictionary object)
        cluster_seeds: path to the cluster seed json file (or the json object)
        altered_cluster_papers: path to the paper ids \n-separated txt file (or a list or set object)
            Clusters that these papers appear in will be marked as "altered"
        train_pairs: path to predefined train pairs csv (or the dataframe object)
        val_pairs: path to predefined val pairs csv (or the dataframe object)
        test_pairs: path to predefined test pairs csv (or the dataframe object)
        train_papers: path to predefined train papers (or the json object)
        val_papers: path to predefined val papers (or the json object)
        test_papers: path to predefined test papers (or the json object)
        unit_of_data_split: options are ("papers", "blocks", "time")
        num_clusters_for_block_size: probably leave as default,
            controls train/val/test splits based on block size
        train_ratio: training ratio of instances for clustering
        val_ratio: validation ratio of instances for clustering
        test_ratio: test ratio of instances for clustering
        train_pairs_size: number of training pairs for learning the linkage function
        val_pairs_size: number of validation pairs for fine-tuning the linkage function parameters
        test_pairs_size: number of test pairs for evaluating the linkage function
        balanced_pair_sample: whether to sample pairs in a class-balanced way
        all_test_pairs_flag: With blocking, for the linkage function evaluation task, should the test
            contain all possible pairs from test blocks, or the given number of pairs (test_pairs_size)
        random_seed: random seed
        n_jobs: number of cpus to use
    """

    def __init__(
        self,
        papers: Union[str, Dict],
        name: str,
        mode: str = "train",
        clusters: Optional[Union[str, Dict]] = None,
        specter_embeddings: Optional[Union[str, Dict]] = None,
        cluster_seeds: Optional[Union[str, Dict]] = None,
        altered_cluster_papers: Optional[Union[str, List, Set]] = None,
        train_pairs: Optional[Union[str, List]] = None,
        val_pairs: Optional[Union[str, List]] = None,
        test_pairs: Optional[Union[str, List]] = None,
        train_papers: Optional[Union[str, List]] = None,
        val_papers: Optional[Union[str, List]] = None,
        test_papers: Optional[Union[str, List]] = None,
        unit_of_data_split: str = "blocks",
        num_clusters_for_block_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        train_pairs_size: int = 30000,
        val_pairs_size: int = 5000,
        test_pairs_size: int = 5000,
        balanced_pair_sample: bool = True,
        all_test_pairs_flag: bool = False,
        random_seed: int = 1111,
        n_jobs: int = 1,
    ):
        logger.debug("loading papers")
        self.papers = self.maybe_load_json(papers)

        # convert dictionary to namedtuples for memory reduction
        for sourced_paper_id, paper in self.papers.items():
            sourced_paper_id = str(sourced_paper_id)
            self.papers[sourced_paper_id] = Paper(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", None),
                has_abstract=paper.get("abstract", None) not in {"", None},
                title_ngrams_chars=None,
                title_ngrams_words=None,
                abstract_ngrams_words=None,
                authors=[
                    Author(
                        author_info_first=author.get("first", None) or author.get("author_info_first", None),
                        author_info_first_normalized_without_apostrophe=None,
                        author_info_middle=" ".join(
                            author.get("middle", [])
                            + [i for i in [author.get("author_info_middle", None)] if i is not None]
                        ),
                        author_info_middle_normalized_without_apostrophe=None,
                        author_info_last_normalized=None,
                        author_info_last=author.get("last", None) or author.get("author_info_last", None),
                        author_info_suffix_normalized=None,
                        author_info_suffix=author.get("suffix", None) or author.get("author_info_suffix", None),
                        author_info_full_name=None,
                        author_info_first_letters=None,
                    )
                    for author in paper["authors"]
                ],
                venue=paper.get("venue", None),
                journal_name=paper.get("journal_name", None),
                venue_ngrams=None,
                journal_ngrams=None,
                author_info_coauthor_n_grams=None,
                author_info_coauthor_email_prefix_n_grams=None,
                author_info_coauthor_email_suffix_n_grams=None,
                author_info_coauthor_affiliations_n_grams=None,
                year=paper.get("year", None),
                sourced_paper_id=sourced_paper_id,
                doi=paper.get("doi", None),
                pmid=paper.get("pmid", None),
                source=paper.get("source", None),
                source_id=paper.get("source_id", None),
                pdf_hash=paper.get("pdf_hash", None),
                block=paper.get("block", None),
                corpus_paper_id=paper.get("corpus_paper_id", None),
            )
        logger.debug("loaded papers")

        self.name = name
        self.mode = mode
        logger.debug("loading clusters")
        self.clusters: Optional[Dict] = self.maybe_load_json(clusters)
        logger.debug("loaded clusters, loading specter")
        self.specter_embeddings = self.maybe_load_specter(specter_embeddings)
        logger.debug("loaded specter, loading cluster seeds")
        cluster_seeds_dict = self.maybe_load_json(cluster_seeds)
        self.altered_cluster_papers = self.maybe_load_list(altered_cluster_papers)
        self.cluster_seeds_disallow = set()
        self.cluster_seeds_require = {}
        self.max_seed_cluster_id = None
        if cluster_seeds_dict is not None:
            cluster_num = 0
            for paper_id_a, values in cluster_seeds_dict.items():
                root_added = False
                for paper_id_b, constraint_string in values.items():
                    if constraint_string == "disallow":
                        self.cluster_seeds_disallow.add((paper_id_a, paper_id_b))
                    elif constraint_string == "require":
                        if not root_added:
                            self.cluster_seeds_require[paper_id_a] = cluster_num
                            root_added = True
                        self.cluster_seeds_require[paper_id_b] = cluster_num
                cluster_num += 1
            self.max_seed_cluster_id = cluster_num
        logger.debug("loaded cluster seeds")
        self.train_pairs = self.maybe_load_dataframe(train_pairs)
        self.val_pairs = self.maybe_load_dataframe(val_pairs)
        self.test_pairs = self.maybe_load_dataframe(test_pairs)
        self.train_papers = self.maybe_load_json(train_papers)
        self.val_papers = self.maybe_load_json(val_papers)
        self.test_papers = self.maybe_load_json(test_papers)
        self.unit_of_data_split = unit_of_data_split
        self.num_clusters_for_block_size = num_clusters_for_block_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "train/val/test ratio should add to 1"
        self.train_pairs_size = train_pairs_size
        self.val_pairs_size = val_pairs_size
        self.test_pairs_size = test_pairs_size
        self.balanced_pair_sample = balanced_pair_sample
        self.all_test_pairs_flag = all_test_pairs_flag
        self.random_seed = random_seed

        if self.clusters is None:
            self.paper_to_cluster_id = None

        if self.mode == "train":
            if self.clusters is not None:
                self.paper_to_cluster_id = {}
                logger.debug("making paper to cluster id")
                for cluster_id, cluster_info in self.clusters.items():
                    for sourced_paper_id in cluster_info["sourced_paper_ids"]:
                        self.paper_to_cluster_id[str(sourced_paper_id)] = cluster_id
                logger.debug("made paper to cluster id")
        elif self.mode == "inference":
            self.all_test_pairs_flag = True
        else:
            raise Exception(f"Unknown mode: {self.mode}")

        self.n_jobs = n_jobs
        self.paper_to_block = self.get_papers_to_block()

        logger.debug("preprocessing papers")
        self.papers = preprocess_papers_parallel(self.papers, self.n_jobs)
        logger.debug("preprocessed papers")

    @staticmethod
    def maybe_load_json(path_or_json: Optional[Union[str, Union[List, Dict]]]) -> Any:
        """
        Either loads a dictionary from a json file or passes through the object

        Parameters
        ----------
        path_or_json: string or Dict
            the file path or the object

        Returns
        -------
        either the loaded json, or the passed in object
        """
        if isinstance(path_or_json, str):
            with open(path_or_json) as _json_file:
                output = json.load(_json_file)
            return output
        else:
            return path_or_json

    @staticmethod
    def maybe_load_list(path_or_list: Optional[Union[str, list, Set]]) -> Optional[Union[list, Set]]:
        """
        Either loads a list from a text file or passes through the object

        Parameters
        ----------
        path_or_list: string or list
            the file path or the object

        Returns
        -------
        either the loaded list, or the passed in object
        """
        if isinstance(path_or_list, str):
            with open(path_or_list, "r") as f:
                return f.read().strip().split("\n")
        else:
            return path_or_list

    @staticmethod
    def maybe_load_dataframe(path_or_dataframe: Optional[Union[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        """
        Either loads a dataframe from a csv file or passes through the object

        Parameters
        ----------
        path_or_dataframe: string or dataframe
            the file path or the object

        Returns
        -------
        either the loaded dataframe, or the passed in object
        """
        if type(path_or_dataframe) == str:
            return pd.read_csv(path_or_dataframe, sep=",")
        else:
            return path_or_dataframe

    @staticmethod
    def maybe_load_specter(path_or_pickle: Optional[Union[str, Dict]]) -> Optional[Dict]:
        """
        Either loads a dictionary from a pickle file or passes through the object

        Parameters
        ----------
        path_or_pickle: string or dictionary
            the file path or the object

        Returns
        -------
        either the loaded json, or the passed in object
        """
        if isinstance(path_or_pickle, str):
            with open(path_or_pickle, "rb") as _pickle_file:
                X, keys = pickle.load(_pickle_file)
            D = {}
            for i, key in enumerate(keys):
                D[key] = X[i, :]
            return D
        else:
            return path_or_pickle

    def get_constraint(
        self,
        paper_id_1: str,
        paper_id_2: str,
        low_value: Union[float, int] = 0,
        high_value: Union[float, int] = LARGE_DISTANCE,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Optional[float]:
        """Applies the passed-in cluster_seeds

        Parameters
        ----------
        paper_id_1: string
            one paper id in the pair
        paper_id_2: string
            the other paper id in the pair
        low_value: float
            value to assign to same person override
        high_value: float
            value to assign to different person overrid
        dont_merge_cluster_seeds: bool
            this flag controls whether to use cluster seeds to enforce "dont merge"
            as well as "must merge" constraints
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        float: the constraint value
        """

        paper_1 = self.papers[str(paper_id_1)]
        paper_2 = self.papers[str(paper_id_2)]

        # cluster seeds have precedence
        if (paper_id_1, paper_id_2) in self.cluster_seeds_disallow or (
            paper_id_2,
            paper_id_1,
        ) in self.cluster_seeds_disallow:
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        elif (self.cluster_seeds_require.get(paper_id_1, -1) == self.cluster_seeds_require.get(paper_id_2, -2)) and (
            not incremental_dont_use_cluster_seeds
        ):
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            dont_merge_cluster_seeds
            and (paper_id_1 in self.cluster_seeds_require and paper_id_2 in self.cluster_seeds_require)
            and (self.cluster_seeds_require[paper_id_1] != self.cluster_seeds_require[paper_id_2])
        ):
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        elif paper_1.doi is not None and paper_2.doi is not None and paper_1.doi == paper_2.doi:
            # if dois is the same - same paper
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif paper_1.pmid is not None and paper_2.pmid is not None and paper_1.pmid == paper_2.pmid:
            # if pmid is the same - same paper
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif paper_1.pdf_hash is not None and paper_2.pdf_hash is not None and paper_1.pdf_hash == paper_2.pdf_hash:
            # same pdf hash - same paper
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            paper_1.source_id is not None
            and paper_2.source_id is not None
            and paper_1.source == paper_2.source
            and paper_1.source in PUBLISHER_SOURCES
            and paper_1.source_id != paper_2.source_id
        ):
            # same source but different ids: can't be the same paper
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        else:
            return None

    def get_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict based on the blocks provided

        Returns
        -------
        Dict: mapping from block id to list of papers in the block
        """
        block: Dict[str, List[str]] = {}
        for sourced_paper_id, paper in self.papers.items():
            block_id = paper.block
            if block_id not in block:
                block[block_id] = [sourced_paper_id]
            else:
                block[block_id].append(sourced_paper_id)
        return block

    def get_papers_to_block(self) -> Dict[str, str]:
        """
        Creates a dictionary mapping paper id to block key

        Each paper can only belong to a single block

        Returns
        -------
        Dict: the papers to block dictionary
        """
        paper_to_block: Dict[str, str] = {}
        block_dict = self.get_blocks()
        for block_key, papers in block_dict.items():
            for sourced_paper_id in papers:
                if sourced_paper_id in paper_to_block:
                    raise ValueError(
                        f"Paper {sourced_paper_id} is in \
                            multiple blocks: {paper_to_block[sourced_paper_id]} and {block_key}"
                    )
                paper_to_block[sourced_paper_id] = block_key
        return paper_to_block

    def split_blocks_helper(
        self, blocks: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks, while trying to preserve
        the distribution of block sizes between the splits.

        Parameters
        ----------
        blocks: Dict
            the full block dictionary

        Returns
        -------
        train/val/test block dictionaries
        """
        x = []
        y = []
        for block_id, papers in blocks.items():
            x.append(block_id)
            y.append(len(papers))

        clustering_model = KMeans(
            n_clusters=self.num_clusters_for_block_size,
            random_state=self.random_seed,
        ).fit(np.array(y).reshape(-1, 1))
        y_group = clustering_model.labels_

        train_blocks, val_test_blocks, _, val_test_length = train_test_split(
            x,
            y_group,
            test_size=self.val_ratio + self.test_ratio,
            stratify=y_group,
            random_state=self.random_seed,
        )
        val_blocks, test_blocks = train_test_split(
            val_test_blocks,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            stratify=val_test_length,
            random_state=self.random_seed,
        )

        train_block_dict = {k: blocks[k] for k in train_blocks}
        val_block_dict = {k: blocks[k] for k in val_blocks}
        test_block_dict = {k: blocks[k] for k in test_blocks}

        return train_block_dict, val_block_dict, test_block_dict

    def group_paper_helper(self, paper_list: List[str]) -> Dict[str, List[str]]:
        """
        Creates a block dict containing a specific input paper list

        Parameters
        ----------
        paper_list: List
            the list of papers to include

        Returns
        -------
        Dict: the block dict for the input papers
        """
        block_to_papers: Dict[str, List[str]] = {}

        for s in paper_list:
            if self.paper_to_block[s] not in block_to_papers:
                block_to_papers[self.paper_to_block[s]] = [s]
            else:
                block_to_papers[self.paper_to_block[s]].append(s)
        return block_to_papers

    def split_cluster_papers(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on split type requested.
        Options for splitting are `papers`, `blocks`, and `time`

        Returns
        -------
        train/val/test block dictionaries
        """
        blocks = self.get_blocks()

        if self.unit_of_data_split == "papers":
            paper_keys = list(self.papers.keys())
            train_papers, val_test_papers = train_test_split(
                paper_keys,
                test_size=self.val_ratio + self.test_ratio,
                random_state=self.random_seed,
            )
            val_papers, test_papers = train_test_split(
                val_test_papers,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=self.random_seed,
            )
            train_block_dict = self.group_paper_helper(train_papers)
            val_block_dict = self.group_paper_helper(val_papers)
            test_block_dict = self.group_paper_helper(test_papers)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "blocks":
            (
                train_block_dict,
                val_block_dict,
                test_block_dict,
            ) = self.split_blocks_helper(blocks)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "time":
            paper_to_year = {}
            for sourced_paper_id, paper in self.papers.items():
                # sourced_paper_id should be kept as string, so it can be matched to papers.json
                sourced_paper_id = str(paper.sourced_paper_id)
                if paper.year is None:
                    paper_to_year[sourced_paper_id] = 0
                else:
                    paper_to_year[sourced_paper_id] = paper.year

            train_size = int(len(paper_to_year) * self.train_ratio)
            val_size = int(len(paper_to_year) * self.val_ratio)
            papers_sorted_by_year = [i[0] for i in (sorted(paper_to_year.items(), key=lambda x: x[1]))]

            train_papers = papers_sorted_by_year[0:train_size]
            val_papers = papers_sorted_by_year[train_size : train_size + val_size]
            test_papers = papers_sorted_by_year[train_size + val_size : len(papers_sorted_by_year)]

            train_block_dict = self.group_paper_helper(train_papers)
            val_block_dict = self.group_paper_helper(val_papers)
            test_block_dict = self.group_paper_helper(test_papers)
            return train_block_dict, val_block_dict, test_block_dict

        else:
            raise Exception(f"Unknown unit_of_data_split: {self.unit_of_data_split}")

    def split_data_papers_fixed(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed paper
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        train_block_dict: Dict[str, List[str]] = {}
        val_block_dict: Dict[str, List[str]] = {}
        test_block_dict: Dict[str, List[str]] = {}

        test_papers = self.test_papers
        logger.debug("fixed papers split")

        if self.val_papers is None:
            train_papers = []
            val_papers = []
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            np.random.seed(self.random_seed)
            split_prob = np.random.rand(len(self.train_papers))
            for paper, p in zip(self.train_papers, split_prob):
                if p < train_prob:
                    train_papers.append(paper)
                else:
                    val_papers.append(paper)
            logger.debug(f"size of papers {len(train_papers), len(val_papers)}")
        else:
            train_papers = self.train_papers
            val_papers = self.val_papers

        train_block_dict = self.group_paper_helper(train_papers)
        val_block_dict = self.group_paper_helper(val_papers)
        test_block_dict = self.group_paper_helper(test_papers)

        return train_block_dict, val_block_dict, test_block_dict

    def split_pairs(
        self,
        train_papers_dict: Dict[str, List[str]],
        val_papers_dict: Dict[str, List[str]],
        test_papers_dict: Dict[str, List[str]],
    ) -> Tuple[
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
    ]:
        """
        creates pairs for the pairwise classification task

        Parameters
        ----------
        train_papers_dict: Dict
            the train block dict
        val_papers_dict: Dict
            the val block dict
        test_papers_dict: Dict
            the test block dict

        Returns
        -------
        train/val/test pairs, where each pair is (paper_id_1, paper_id_2, label)
        """
        assert (
            isinstance(train_papers_dict, dict)
            and isinstance(val_papers_dict, dict)
            and isinstance(test_papers_dict, dict)
        )
        train_pairs = self.pair_sampling(
            self.train_pairs_size,
            train_papers_dict,
            all_pairs=False,
            balanced_pair_sample=self.balanced_pair_sample,
        )
        val_pairs = (
            self.pair_sampling(
                self.val_pairs_size,
                val_papers_dict,
                all_pairs=False,
                balanced_pair_sample=self.balanced_pair_sample,
            )
            if len(val_papers_dict) > 0
            else []
        )

        test_pairs = self.pair_sampling(
            self.test_pairs_size,
            test_papers_dict,
            self.all_test_pairs_flag,
            balanced_pair_sample=False,
        )

        return train_pairs, val_pairs, test_pairs

    def construct_cluster_to_papers(
        self,
        block_dict: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        creates a dictionary mapping cluster to papers

        Parameters
        ----------
        block_dict: Dict
            the block dict to construct cluster to papers for

        Returns
        -------
        Dict: the dictionary mapping cluster to papers
        """
        cluster_to_papers = defaultdict(list)
        for papers in block_dict.values():
            for paper in papers:
                true_cluster_id = self.paper_to_cluster_id[paper]
                cluster_to_papers[true_cluster_id].append(paper)

        return dict(cluster_to_papers)

    def fixed_pairs(
        self,
    ) -> Tuple[
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
    ]:
        """
        creates pairs for the pairwise classification task from a fixed train/val/test split

        Returns
        -------
        train/val/test pairs, where each pair is (paper_id_1, paper_id_2, label)
        """
        assert (
            self.train_pairs is not None and self.test_pairs is not None
        ), "You need to pass in train and test pairs to use this function"
        self.train_pairs.loc[:, "label"] = self.train_pairs["label"].map(
            {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}
        )
        if self.val_pairs is not None:
            self.val_pairs.loc[:, "label"] = self.val_pairs["label"].map(
                {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}
            )
            train_pairs = list(self.train_pairs.to_records(index=False))
            val_pairs = list(self.val_pairs.to_records(index=False))
        else:
            np.random.seed(self.random_seed)
            # split train into train/val
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            msk = np.random.rand(len(self.train_pairs)) < train_prob
            train_pairs = list(self.train_pairs[msk].to_records(index=False))
            val_pairs = list(self.train_pairs[~msk].to_records(index=False))
        self.test_pairs.loc[:, "label"] = self.test_pairs["label"].map({"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1})
        test_pairs = list(self.test_pairs.to_records(index=False))

        return train_pairs, val_pairs, test_pairs

    def all_pairs(self) -> List[Tuple[str, str, Union[int, float]]]:
        """
        creates all pairs within blocks, probably used for inference

        Returns
        -------
        all pairs, where each pair is (paper_id_1, paper_id_2, label)
        """
        all_pairs_output = self.pair_sampling(
            0,
            self.get_blocks(),
            self.all_test_pairs_flag,  # ignored when all_test_pairs_flag is True
        )
        return all_pairs_output

    def pair_sampling(
        self,
        sample_size: int,
        blocks: Dict[str, List[str]],
        all_pairs: bool = False,
        balanced_pair_sample: bool = True,
    ) -> List[Tuple[str, str, Union[int, float]]]:
        """
        Enumerates all pairs exhaustively, and samples pairs from each class.

        Note: we don't know the label when either of the papers have the cluster clusterid_ORPHAN_CLUSTER_KEY.

        Parameters
        ----------
        sample_size: integer
            The desired sample size
        blocks: dict
            It has block ids as keys, and list of paper ids under each block as values.
            Must be provided when blocking is used
        all_pairs: bool
            Whether or not to return all pairs
        balanced_pair_sample: bool
            Whether to sample class-balanced pairs or not

        Returns
        -------
        list: list of paper pairs
        """

        possible: List[Tuple[str, str, Union[int, float]]] = []

        for _, papers in blocks.items():
            for i, s1 in enumerate(papers):
                for s2 in papers[i + 1 :]:
                    if self.paper_to_cluster_id is not None:  # we have ground truth
                        s1_cluster = self.paper_to_cluster_id[s1]
                        s2_cluster = self.paper_to_cluster_id[s2]
                        # we have to exclude orphans entirely from the labeled data set
                        if not (s1_cluster.endswith(ORPHAN_CLUSTER_KEY) or s2_cluster.endswith(ORPHAN_CLUSTER_KEY)):
                            if s1_cluster == s2_cluster:
                                possible.append((s1, s2, 1))
                            else:
                                possible.append((s1, s2, 0))
                        else:  # will be removed later if not all_pairs
                            possible.append((s1, s2, NUMPY_NAN))
                    else:  # we don't have labels so we are just going to make everything
                        possible.append((s1, s2, NUMPY_NAN))

        if all_pairs:
            pairs = possible
        else:
            random.seed(self.random_seed)
            if balanced_pair_sample:
                # make a balanced dataset by sampling from each class
                pairs_1 = [i for i in possible if i[2] == 1]
                pairs_0 = [i for i in possible if i[2] == 0]
                pairs_1 = random.sample(pairs_1, min(len(pairs_1), sample_size // 2))
                pairs_0 = random.sample(pairs_0, min(len(pairs_0), sample_size // 2))
                pairs = pairs_1 + pairs_0
            else:
                possible = [i for i in possible if not np.isnan(i[2])]
                pairs = random.sample(possible, min(len(possible), sample_size))
        return pairs


def get_full_name_for_features(author: Author, include_last: bool = True, include_suffix: bool = True) -> str:
    """
    Creates the full name from the name parts.

    Parameters
    ----------
    authir: Author
        the author to create the full name for
    include_last: bool
        whether to include the last name
    include_suffix: bool
        whether to include the suffix

    Returns
    -------
    string: the full name
    """
    first = author.author_info_first_normalized_without_apostrophe or author.author_info_first
    middle = author.author_info_middle_normalized_without_apostrophe or author.author_info_middle
    last = author.author_info_last_normalized or author.author_info_last
    suffix = author.author_info_suffix_normalized or author.author_info_suffix
    list_of_parts = [first, middle]
    if include_last:
        list_of_parts.append(last)
    if include_suffix:
        list_of_parts.append(suffix)
    name_parts = [part.strip() for part in list_of_parts if part is not None and len(part) != 0]
    return " ".join(name_parts)


def preprocess_authors(author):
    """
    Preprocess the authors, doing lots of normalization and feature creation
    TODO: fix here as described in https://github.com/allenai/S2AND/issues/39
    TODO: middle="Le", last="Merdy" vs middle = "" last = "Le Merdy"
    """

    # our normalization scheme is to normalize first and middle separately,
    # join them, then take the first token of the combined join
    first_normed = normalize_text(author.author_info_first or "", special_case_apostrophes_and_dashes=True)
    middle_normed = normalize_text(author.author_info_middle or "", special_case_apostrophes_and_dashes=True)
    last_normed = normalize_text(author.author_info_last or "", special_case_apostrophes_and_dashes=True)
    suffix_normed = normalize_text(author.author_info_suffix or "", special_case_apostrophes_and_dashes=True)

    first_middle_normed = (first_normed + " " + middle_normed).split(" ")

    if first_middle_normed[0] in NAME_PREFIXES and len(first_middle_normed) > 1:
        first_middle_normed = first_middle_normed[1:]

    author = author._replace(
        author_info_first_normalized_without_apostrophe=first_middle_normed[0],
        author_info_middle_normalized_without_apostrophe=" ".join(first_middle_normed[1:]),
        author_info_last_normalized=last_normed,
        author_info_suffix_normalized=suffix_normed,
    )

    author_info_first_letters = set()
    if len(first_normed) > 0:
        author_info_first_letters.add(first_normed[0])
    if len(first_normed[1:]) > 0:
        for i in first_normed[1:]:
            author_info_first_letters.add(i[0])
    if len(last_normed) > 0:
        author_info_first_letters.add(last_normed[0])

    author = author._replace(
        author_info_full_name=get_full_name_for_features(author).strip(),
        author_info_first_letters=author_info_first_letters,
    )

    return author


def preprocess_paper_1(item: Tuple[str, Paper]) -> Tuple[str, Paper]:
    """
    helper function to perform most of the preprocessing of a paper

    Parameters
    ----------
    item: Tuple[str, Paper]
        tuple of paper id and Paper object

    Returns
    -------
    Tuple[str, Paper]: tuple of paper id and preprocessed Paper object
    """

    key, paper = item

    title = normalize_text(paper.title)
    abstract = ""
    if paper.has_abstract:
        abstract = normalize_text(paper.abstract)
    if paper.title is None:
        title_lower_simple = ""
    else:
        title_lower_simple = paper.title.lower().replace(" ", "")
    paper = paper._replace(
        title=title,
        # title_ngrams_words=get_text_ngrams_words(title, stopwords=None),
        title_ngrams_chars=get_text_ngrams(title_lower_simple, stopwords=None, use_bigrams=False),
        abstract_ngrams_words=get_text_ngrams_words(abstract, stopwords=None),
    )
    venue = normalize_venue_name(paper.venue)
    journal_name = normalize_venue_name(paper.journal_name)
    if venue != journal_name:
        combined_venue = (journal_name + " " + venue).strip()
    else:
        combined_venue = venue
    # sometimes venues have no letters a-z, in which case it's just page numbers
    if A_THROUGH_Z.search(combined_venue) is None:
        combined_venue = ""
    venue_ngrams = get_text_ngrams(combined_venue, stopwords=None, use_bigrams=True)
    paper = paper._replace(venue=venue, journal_name=journal_name, venue_ngrams=venue_ngrams)

    authors = [preprocess_authors(author) for author in paper.authors]
    author_info_coauthor_n_grams = get_text_ngrams(
        " ".join([i.author_info_full_name for i in authors]),
        stopwords=None,
        use_unigrams=True,
        use_bigrams=True,
    )
    doi = paper.doi.lower() if paper.doi is not None else None

    paper = paper._replace(
        authors=authors,
        author_info_coauthor_n_grams=author_info_coauthor_n_grams,
        doi=doi,
    )

    return (key, paper)


def preprocess_papers_parallel(papers_dict: Dict, n_jobs: int) -> Dict:
    """
    helper function to preprocess papers

    Parameters
    ----------
    papers_dict: Dict
        the papers dictionary
    n_jobs: int
        how many cpus to use

    Returns
    -------
    Dict: the preprocessed papers dictionary
    """

    output = {}
    if n_jobs > 1:
        with multiprocessing.Pool(processes=n_jobs) as p:
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Preprocessing papers") as pbar:
                for key, value in p.imap(preprocess_paper_1, papers_dict.items(), DEFAULT_CHUNK_SIZE):
                    output[key] = value
                    pbar.update()
    else:
        for item in tqdm(papers_dict.items(), total=len(papers_dict), desc="Preprocessing papers"):
            result = preprocess_paper_1(item)
            output[result[0]] = result[1]

    return output


if __name__ == "__main__":
    pddata = PDData(
        papers="tests/test_dataset/papers.json",
        clusters="tests/test_dataset/clusters.json",
        name="test",
    )
