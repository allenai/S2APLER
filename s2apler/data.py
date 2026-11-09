from typing import Optional, Union, Dict, List, Any, Tuple, Set, NamedTuple

import random
import re
import urllib.parse
import json
import numpy as np
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
    abstract_ngrams_words: Optional[Counter]
    authors: List[Author]
    venue: Optional[str]
    journal_name: Optional[str]
    title_ngrams_chars: Optional[Counter]
    venue_ngrams: Optional[Counter]
    author_info_coauthor_n_grams: Optional[Counter]
    year: Optional[int]
    sourced_paper_id: str
    corpus_paper_id: Optional[int]  # this is the id of the paper's current mapping in the corpus
    doi: Optional[str]
    pmid: Optional[str]
    source_id: Optional[str]
    pdf_hash: Optional[str]
    source: Optional[str]
    block: Optional[str]
    source_uris: Optional[List[str]] = None
    # The texts the n-gram counters above come from. Preprocessing keeps these and leaves the
    # counters empty: the native featurizer counts n-grams itself, and with_counters builds
    # the Counters when Python needs them.
    title_ngram_text: Optional[str] = None
    abstract_ngram_text: Optional[str] = None
    venue_ngram_text: Optional[str] = None
    coauthor_ngram_text: Optional[str] = None


# Resolver URLs, with or without scheme or host prefix: https://doi.org/, http://dx.doi.org/, doi.org/.
DOI_URL_PREFIX = re.compile(r"^(?:https?://)?(?:dx\.|www\.)?doi\.org/", re.IGNORECASE)
# Scheme labels: doi:10.1/a, urn:doi:10.1/a.
DOI_SCHEME_PREFIX = re.compile(r"^(?:urn:)?doi:\s*", re.IGNORECASE)


def _blank_to_none(value: Any) -> Optional[str]:
    """Trims an identifier to a string, treating blanks as missing.

    Blank IDs would otherwise force unrelated papers together. Training data stores some
    PMIDs as JSON ints, so non-strings are converted rather than rejected.
    """
    if value is None:
        return None
    return str(value).strip() or None


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Canonicalizes a DOI: trims, strips resolver prefixes, decodes URL escapes, lowercases.

    One work can arrive as "10.1/a#b" and "https://doi.org/10.1/A%23b". Without this, the two
    don't match, and a PDF they share looks like it holds two different DOI'd items.
    """
    if doi is None:
        return None
    doi = doi.strip()
    url_prefix = DOI_URL_PREFIX.match(doi)
    if url_prefix:
        # URLs percent-encode DOI characters such as "#"; a bare DOI may hold a literal "%".
        doi = urllib.parse.unquote(doi[url_prefix.end() :])
    else:
        doi = DOI_SCHEME_PREFIX.sub("", doi, count=1)
    return doi.strip().lower() or None


def _codes(values: List[Any]) -> np.ndarray:
    """Integer codes for hashable values, equal codes for equal values; None becomes -1."""
    table: Dict[Any, int] = {}
    return np.fromiter(
        (-1 if value is None else table.setdefault(value, len(table)) for value in values),
        dtype=np.int64,
        count=len(values),
    )


def condensed_offsets(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Positions of pairs (i, j), i < j, in a condensed n-by-n distance vector."""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def _strong_signals_disagree(p1: "Paper", p2: "Paper") -> bool:
    """True if any of {DOI, PMID, first-author last name} are populated on both
    papers and conflict. Used to gate URL+title-based hard merge: see allenai/scholar#41863."""
    if p1.doi and p2.doi and p1.doi != p2.doi:
        return True
    if p1.pmid and p2.pmid and p1.pmid != p2.pmid:
        return True
    if p1.authors and p2.authors:
        a1 = p1.authors[0].author_info_last_normalized
        a2 = p2.authors[0].author_info_last_normalized
        if a1 and a2 and a1 != a2:
            return True
    return False


class PDData:
    """
    The main class for holding our representation of an paper disambiguation data

    Input:
        papers: path to the papers information json file (or the json object)
        name: name of the dataset; Clusterer.fit looks up precomputed distances by it
        mode: 'train' or 'inference'; if 'inference', everything related to splitting will be ignored
        clusters: path to the clusters json file (or the json object)
            - a cluster may span multiple blocks, but we will only consider in-block clusters
            - there will be individual papers that definitely do not belong to any of the known clusters
              but which may or may not cluster with each other.
              papers in these clusters will all appear in clusters.json under the key clusterid_ORPHAN_CLUSTER_KEY
        cluster_seeds: path to the cluster seed json file (or the json object)
        altered_cluster_papers: path to the paper ids \n-separated txt file (or a list or set object)
            Clusters that these papers appear in will be marked as "altered"
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
        cluster_seeds: Optional[Union[str, Dict]] = None,
        altered_cluster_papers: Optional[Union[str, List, Set]] = None,
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
                author_info_coauthor_n_grams=None,
                year=paper.get("year", None),
                sourced_paper_id=sourced_paper_id,
                doi=normalize_doi(paper.get("doi", None)),
                pmid=_blank_to_none(paper.get("pmid", None)),
                source=paper.get("source", None),
                source_id=paper.get("source_id", None),
                pdf_hash=_blank_to_none(paper.get("pdf_hash", None)),
                block=paper.get("block", None),
                corpus_paper_id=paper.get("corpus_paper_id", None),
                source_uris=paper.get("source_uris", None),
            )
        logger.debug("loaded papers")

        self.name = name
        self.mode = mode
        logger.debug("loading clusters")
        self.clusters: Optional[Dict] = self.maybe_load_json(clusters)
        logger.debug("loaded clusters, loading cluster seeds")
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

        self.paper_to_cluster_id: Optional[Dict[str, str]] = None

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
        self.multi_doi_pdf_hashes = self.get_multi_doi_pdf_hashes()

    def get_multi_doi_pdf_hashes(self) -> Set[str]:
        """Finds pdf_hashes shared by papers that carry different DOIs.

        One PDF often holds several short items: a page of BMJ letters, Nature research
        highlights, a meeting-abstract supplement. Each item has its own DOI, so a pdf_hash
        seen with two DOIs names a container, not a paper, and must not force merges.
        This spans the whole dataset, not one block: a container PDF is one everywhere.

        Returns
        -------
        Set[str]: the pdf_hashes that map to more than one distinct DOI
        """
        dois_by_hash: Dict[str, Set[str]] = defaultdict(set)
        for paper in self.papers.values():
            if paper.pdf_hash is not None and paper.doi is not None:
                dois_by_hash[paper.pdf_hash].add(paper.doi)
        return {pdf_hash for pdf_hash, dois in dois_by_hash.items() if len(dois) > 1}

    def id_merge_keys(self, paper_id: str) -> List[Tuple[str, str]]:
        """Lists the identifiers that force a paper into the same cluster as others.

        Parameters
        ----------
        paper_id: string
            the paper id

        Returns
        -------
        List[Tuple[str, str]]: (id type, value) pairs for the DOI, PMID and usable pdf_hash
        """
        paper = self.papers[str(paper_id)]
        keys = []
        if paper.doi is not None:
            keys.append(("doi", paper.doi))
        if paper.pmid is not None:
            keys.append(("pmid", paper.pmid))
        if paper.pdf_hash is not None and paper.pdf_hash not in self.multi_doi_pdf_hashes:
            keys.append(("pdf_hash", paper.pdf_hash))
        return keys

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
        elif (
            paper_1.pdf_hash is not None
            and paper_1.pdf_hash == paper_2.pdf_hash
            and paper_1.pdf_hash not in self.multi_doi_pdf_hashes
        ):
            # same pdf hash - same paper, unless the PDF holds several DOI'd items
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            paper_1.source_uris
            and paper_2.source_uris
            and not set(paper_1.source_uris).isdisjoint(paper_2.source_uris)
            and paper_1.title
            and paper_2.title
            and paper_1.title == paper_2.title
            and not _strong_signals_disagree(paper_1, paper_2)
        ):
            # PDF was fetched from the same URL AND extracted titles match. pdf_hash alone
            # isn't sufficient because some hosts (e.g. inria.hal.science) serve byte-different
            # PDFs per fetch (added watermark/timestamp), producing a fresh pdf_hash on every
            # recrawl. The title-equality and strong-signal-agreement guards together rule out
            # URLs that legitimately serve multiple papers — proceedings volumes (different
            # titles), publisher domains hosting many issues' "Front Matter"/"Editorial" entries
            # (different DOIs), or college catalog pages (different first-authors per faculty).
            # Titles compared post-preprocessing — already lowercased/unidecoded/whitespace-normed.
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

    def constraint_values(
        self,
        paper_ids: List[str],
        left: np.ndarray,
        right: np.ndarray,
        first_offset: int = 0,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> np.ndarray:
        """Applies get_constraint to many pairs of one block at once.

        Pairs (left[k], right[k]) index into paper_ids, with left[k] < right[k], and are
        consecutive entries of the block's condensed distance vector from first_offset on.
        Each rule fills its pairs from the lowest precedence up, so the rule that
        get_constraint checks first wins.

        Parameters
        ----------
        paper_ids: List[str]
            the block's paper ids
        left: np.ndarray
            the first paper of each pair
        right: np.ndarray
            the second paper of each pair
        first_offset: int
            the condensed offset of the first pair
        dont_merge_cluster_seeds: bool
            as for get_constraint
        incremental_dont_use_cluster_seeds: bool
            as for get_constraint

        Returns
        -------
        np.ndarray: the constraint value of each pair, NaN where get_constraint returns None
        """
        papers = [self.papers[str(paper_id)] for paper_id in paper_ids]
        n = len(papers)
        values = np.full(len(left), np.nan)

        def fill_pairs(pairs: List[Tuple[int, int]], value: float):
            # Rules that match few pairs are found by index, then placed by condensed offset.
            if not pairs:
                return
            i, j = np.array(pairs).T
            offsets = condensed_offsets(np.minimum(i, j), np.maximum(i, j), n) - first_offset
            values[offsets[(offsets >= 0) & (offsets < len(values))]] = value

        source = _codes([paper.source for paper in papers])
        is_publisher = np.array([paper.source in PUBLISHER_SOURCES for paper in papers], dtype=bool)
        source_id = _codes([paper.source_id for paper in papers])
        same = (
            (source_id[left] >= 0)
            & (source_id[right] >= 0)
            & (source[left] == source[right])
            & is_publisher[left]
            & (source_id[left] != source_id[right])
        )
        values[same] = CLUSTER_SEEDS_LOOKUP["disallow"]

        # Same source URI and the same title: candidates come from a (URI, title) index.
        by_uri_and_title = defaultdict(list)
        for position, paper in enumerate(papers):
            if paper.source_uris and paper.title:
                for uri in set(paper.source_uris):
                    by_uri_and_title[(uri, paper.title)].append(position)
        candidates = {
            (members[a], members[b])
            for members in by_uri_and_title.values()
            for a in range(len(members))
            for b in range(a + 1, len(members))
        }
        fill_pairs(
            [(i, j) for i, j in sorted(candidates) if not _strong_signals_disagree(papers[i], papers[j])],
            CLUSTER_SEEDS_LOOKUP["require"],
        )

        pdf_hash = _codes(
            [paper.pdf_hash if paper.pdf_hash not in self.multi_doi_pdf_hashes else None for paper in papers]
        )
        for codes in (pdf_hash, _codes([paper.pmid for paper in papers]), _codes([paper.doi for paper in papers])):
            values[(codes[left] >= 0) & (codes[left] == codes[right])] = CLUSTER_SEEDS_LOOKUP["require"]

        seed = _codes([self.cluster_seeds_require.get(paper_id) for paper_id in paper_ids])
        both_seeded = (seed[left] >= 0) & (seed[right] >= 0)
        if dont_merge_cluster_seeds:
            values[both_seeded & (seed[left] != seed[right])] = CLUSTER_SEEDS_LOOKUP["disallow"]
        if not incremental_dont_use_cluster_seeds:
            values[both_seeded & (seed[left] == seed[right])] = CLUSTER_SEEDS_LOOKUP["require"]

        position = {paper_id: k for k, paper_id in enumerate(paper_ids)}
        fill_pairs(
            [
                (position[a], position[b])
                for a, b in self.cluster_seeds_disallow
                if a in position and b in position and a != b
            ],
            CLUSTER_SEEDS_LOOKUP["disallow"],
        )
        return values

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

        # scikit-learn 1.4 changed the default to n_init="auto" (one run); keep the old 10 runs
        # so the same random_seed gives the same block-size groups and splits.
        clustering_model = KMeans(
            n_clusters=self.num_clusters_for_block_size,
            n_init=10,
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

    def split_cluster_papers(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the blocks into train/val/test blocks, keeping each block whole

        Returns
        -------
        train/val/test block dictionaries
        """
        return self.split_blocks_helper(self.get_blocks())

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
        assert self.paper_to_cluster_id is not None, "needs a dataset loaded with clusters in train mode"
        cluster_to_papers = defaultdict(list)
        for papers in block_dict.values():
            for paper in papers:
                true_cluster_id = self.paper_to_cluster_id[paper]
                cluster_to_papers[true_cluster_id].append(paper)

        return dict(cluster_to_papers)

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
    paper = paper._replace(title=title, title_ngram_text=title_lower_simple, abstract_ngram_text=abstract)
    venue = normalize_venue_name(paper.venue)
    journal_name = normalize_venue_name(paper.journal_name)
    if venue != journal_name:
        combined_venue = (journal_name + " " + venue).strip()
    else:
        combined_venue = venue
    # sometimes venues have no letters a-z, in which case it's just page numbers
    if A_THROUGH_Z.search(combined_venue) is None:
        combined_venue = ""
    paper = paper._replace(venue=venue, journal_name=journal_name, venue_ngram_text=combined_venue)

    authors = [preprocess_authors(author) for author in paper.authors]
    paper = paper._replace(
        authors=authors,
        coauthor_ngram_text=" ".join([i.author_info_full_name for i in authors]),
    )

    return (key, paper)


def with_counters(paper: Paper) -> Paper:
    """Fills a preprocessed paper's empty n-gram counters from its n-gram texts.

    Parameters
    ----------
    paper: Paper
        a preprocessed paper; counters it already holds are kept

    Returns
    -------
    Paper: the paper with every n-gram counter set
    """
    return paper._replace(
        title_ngrams_chars=paper.title_ngrams_chars
        if paper.title_ngrams_chars is not None
        else get_text_ngrams(paper.title_ngram_text, stopwords=None, use_bigrams=False),
        abstract_ngrams_words=paper.abstract_ngrams_words
        if paper.abstract_ngrams_words is not None
        else get_text_ngrams_words(paper.abstract_ngram_text, stopwords=None),
        venue_ngrams=paper.venue_ngrams
        if paper.venue_ngrams is not None
        else get_text_ngrams(paper.venue_ngram_text, stopwords=None, use_bigrams=True),
        author_info_coauthor_n_grams=paper.author_info_coauthor_n_grams
        if paper.author_info_coauthor_n_grams is not None
        else get_text_ngrams(paper.coauthor_ngram_text, stopwords=None, use_unigrams=True, use_bigrams=True),
    )


# Below this many papers, starting a process pool costs more than it saves: about 4 s on
# Windows for a 1,500-paper request that preprocesses in under 1.5 s on one process. s2and
# uses the same threshold for its parallel builds.
PREPROCESS_POOL_MIN_PAPERS = 20000


def preprocess_papers_parallel(papers_dict: Dict, n_jobs: int) -> Dict:
    """
    helper function to preprocess papers

    Parameters
    ----------
    papers_dict: Dict
        the papers dictionary
    n_jobs: int
        how many cpus to use, for at least PREPROCESS_POOL_MIN_PAPERS papers

    Returns
    -------
    Dict: the preprocessed papers dictionary
    """

    output = {}
    if n_jobs > 1 and len(papers_dict) >= PREPROCESS_POOL_MIN_PAPERS:
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
