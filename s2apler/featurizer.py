from typing import Tuple, List, Union, Dict, Any

import numpy as np
import logging
from collections import Counter

from tqdm import tqdm

from s2apler.data import PDData, Paper
from s2apler.runtime import load_s2apler_rust_extension
from s2and.thread_config import resolve_n_jobs
from s2apler.consts import LARGE_INTEGER
from s2apler.text import (
    diff,
    counter_jaccard,
    numeral_similarity,
    special_publication_word_similarity,
    TEXT_FUNCTIONS,
    name_text_features,
    year_similarity,
    PUBLISHER_SOURCES,
    NUMERALS,
    NUMERAL_PRECEDING_WORDS,
    SPECIAL_PUBLICATION_WORDS,
    YEAR,
)

logger = logging.getLogger("s2apler")

FeaturesAndLabels = Tuple[np.ndarray, np.ndarray]

PDF_SOURCES = {"MergedPDFExtraction", "ScienceParseMerged", "ScienceParsePlus", "Grobid"}


class FeaturizationInfo:
    """
    Class to store which feature groups to use

    Inputs:
        features_to_use: List[str[]]
            list of feature types to use
    """

    def __init__(
        self,
        features_to_use: List[str] = [
            "author_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            "abstract_similarity",
            "paper_quality",
        ],
    ):
        self.features_to_use = features_to_use
        self.feature_names = self.get_feature_names()

    def get_feature_names(self) -> List[str]:
        """
        Gets all of the feature names, and sets the indices of the features to use

        Returns
        -------
        List[string]: List of all the features names
        """
        feature_names = []
        self.indices_to_use = []
        start_count = 0
        groups = [
            ("author_similarity", ["author_names_similarity", "author_first_letter_compatibility"]),
            ("venue_similarity", ["venue_similarity"]),
            ("year_diff", ["year_diff"]),
            (
                "title_similarity",
                [
                    "title_character_similarity",
                    "title_numeral_similarity",  # trying to catch when we have stuff like "part i vs part ii"
                    "title_special_publication_word_similarity",  # commentary, response, letter, editorial, etc
                    "title_year_similarity",  # whether years mentioned are the same
                ]
                + ["title_" + func[1] for func in TEXT_FUNCTIONS],
            ),
            ("abstract_similarity", ["abstract_word_similarity"]),
            (
                "paper_quality",
                [f"paper_field_count_{i}" for i in ["abstract", "authors", "venue"]]
                + ["source_count_pdf", "source_count_publisher", "sources_publisher_are_same"],
            ),
        ]
        for group, these_features in groups:
            # Every group takes columns in the full feature array; only the chosen ones are kept.
            if group in self.features_to_use:
                feature_names.extend(these_features)
                self.indices_to_use.extend(range(start_count, start_count + len(these_features)))
            start_count += len(these_features)
        self.number_of_features = len(feature_names)
        return feature_names


NUM_FEATURES = FeaturizationInfo().number_of_features


def compare_author_first_letters(auth_1, auth_2, check_same_len=True, strict_order=True):
    """Checks if two author lists are compatible based on (a) num of authors (optionally),
    (b) whether the first letters of their names are the same (but possibly in the wrong order)
    and (c) whether the authors are in the right order (optionally)

    Args:
        auth_1 (list of Authors): list of authors
            must have key "author_info_first_letters" (str)
        auth_2 (list of Authors): ditto
        check_same_len (bool, optional): whether to check if the two author lists have the same len
        strict_order (bool, optional): whether the authors have to be in the same order or not.
            If False, tries to find a match anywhere in the author list as opposed to the corresponding position.
            Defaults to True.
    """
    # if either one doesn't have any authors, then this feature should be nan
    if len(auth_1) == 0 or len(auth_2) == 0:
        return np.nan

    # check if the two author lists have the same len
    if check_same_len and (len(auth_1) != len(auth_2)):
        return False

    # extract first letters
    first_letters_1 = [auth.author_info_first_letters for auth in auth_1]
    first_letters_2 = [auth.author_info_first_letters for auth in auth_2]

    # check if the first letters are the same
    if strict_order:
        for i in range(len(first_letters_1)):
            # comes back as true for {S, P, T} and {S, T}
            # and also for {S, T} and {S}
            # but false for {S, P, T} and {S, K, L}
            # min is 2 if 2 are available for first/last name
            min_size = min(2, min(len(first_letters_1[i]), len(first_letters_2[i])))
            if len(first_letters_1[i].intersection(first_letters_2[i])) < min_size:
                return False
        return True
    else:
        # since we don't care about order, we do an all-pairs comparison
        found_match = [False] * len(first_letters_1)
        for i in range(len(first_letters_1)):
            for j in range(len(first_letters_2)):
                if first_letters_1[i] == first_letters_2[j]:
                    found_match[i] = True
                    break  # from inner loop only
        return all(found_match)


# Papers for _single_pair_featurize, the reference implementation that tests compare Rust to.
global_dataset: Dict[str, Any] = {}


def _set_global_dataset(papers: Dict[str, Any]):
    """Sets the papers that _single_pair_featurize reads."""
    global global_dataset
    global_dataset = papers


def _single_pair_featurize(work_input: Tuple[str, str], index: int = -1) -> Tuple[List[Union[int, float]], int]:
    """
    Creates the features array for a single paper pair
    NOTE: This function reads papers from a module global. Call it through many_pairs_featurize with
    backend="python", or set the global with _set_global_dataset first.

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of paper ids
    index: int
        the index of the pair in the list of all pairs, passed through

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    global global_dataset

    features = []

    paper_id_1 = work_input[0]
    paper_id_2 = work_input[1]

    paper_1 = global_dataset[str(paper_id_1)]
    paper_2 = global_dataset[str(paper_id_2)]

    # author-related features
    features.extend(
        [
            counter_jaccard(
                paper_1.author_info_coauthor_n_grams,
                paper_2.author_info_coauthor_n_grams,
                denominator_max=5000,
            ),
            compare_author_first_letters(
                paper_1.authors,
                paper_2.authors,
            ),
        ]
    )

    # venue features
    features.extend(
        [
            counter_jaccard(paper_1.venue_ngrams, paper_2.venue_ngrams),
        ]
    )

    # year feature
    features.append(
        np.minimum(
            diff(
                paper_1.year if paper_1.year is not None and paper_1.year > 1700 else None,
                paper_2.year if paper_2.year is not None and paper_2.year > 1700 else None,
            ),
            5,
        )
    )

    # title features
    features.extend(
        [
            counter_jaccard(paper_1.title_ngrams_chars, paper_2.title_ngrams_chars),
            numeral_similarity(paper_1.title, paper_2.title),
            special_publication_word_similarity(paper_1.title, paper_2.title),
            year_similarity(paper_1.title, paper_2.title),
        ]
    )
    # these features concat titles (no spaces)
    features.extend(
        name_text_features(
            paper_1.title.replace(" ", ""),  # + " ".join([i.author_info_full_name for i in paper_1.authors]),
            paper_2.title.replace(" ", ""),  # + " ".join([i.author_info_full_name for i in paper_2.authors]),
        )
    )

    # abstract features
    features.append(
        counter_jaccard(paper_1.abstract_ngrams_words, paper_2.abstract_ngrams_words),
    )

    # paper quality features
    # paper_1_num_present_fields = (
    #     int(paper_1.abstract is not None and len(paper_1.abstract) > 0)
    #     + int(len(paper_1.authors) > 0)
    #     # + int(paper_1.venue is not None or paper_1.journal_name is not None)
    #     + int(paper_1.year is not None)
    # )

    # paper_2_num_present_fields = (
    #     int(paper_2.abstract is not None and len(paper_2.abstract) > 0)
    #     + int(len(paper_2.authors) > 0)
    #     # + int(paper_2.venue is not None or paper_2.journal_name is not None)
    #     + int(paper_2.year is not None)
    # )
    # features.extend(
    #     [
    #         min(paper_1_num_present_fields, paper_2_num_present_fields),
    #         max(paper_1_num_present_fields, paper_2_num_present_fields),
    #     ]
    # )

    features.extend(
        [
            int(paper_1.abstract is not None and len(paper_1.abstract) > 0)
            + int(paper_2.abstract is not None and len(paper_2.abstract) > 0),
            int(len(paper_1.authors) > 0) + int(len(paper_2.authors) > 0),
            int(len(paper_1.venue) > 0) + int(len(paper_2.venue) > 0),
            # int(paper_1.year is not None and paper_1.year > 0) + int(paper_2.year is not None and paper_2.year > 0),
        ]
    )

    # how many are from PDFs?
    source_counts_pdf = (paper_1.source in PDF_SOURCES) or (paper_2.source in PDF_SOURCES)
    # how many are from trusted publishers?
    publisher_source_counts = int(paper_1.source in PUBLISHER_SOURCES) + int(paper_2.source in PUBLISHER_SOURCES)
    # if both are from trusted publishers, are they the same publisher or different?
    same_publisher_sources = int(paper_1.source == paper_2.source) if publisher_source_counts == 2 else np.nan
    features.extend([source_counts_pdf, publisher_source_counts, same_publisher_sources])

    # unifying feature type in features array
    features = [float(val) if type(val) in [np.float32, np.float64, float] else int(val) for val in features]

    return features, index


class _Interner:
    """Maps strings to dense integer ids, so Rust compares ids instead of strings."""

    def __init__(self):
        self.ids: Dict[str, int] = {}

    def __call__(self, key: str) -> int:
        return self.ids.setdefault(key, len(self.ids))


def _counter_input(counter: Counter, intern: _Interner) -> Tuple[List[int], List[int]]:
    return [intern(key) for key in counter], list(counter.values())


def rust_paper_record(paper: Any, intern: _Interner) -> Tuple:
    """Packs one preprocessed paper into the record `PaperFeaturizer` reads.

    Each field holds exactly what `_single_pair_featurize` reads from the paper, so the Rust
    features match the Python ones. Strings become interned ids; equal ids mean equal strings.

    Parameters
    ----------
    paper: Paper
        a preprocessed paper; preprocessing fills the counters and author letters that `Paper`
        types as optional, and `_single_pair_featurize` relies on them the same way
    intern: _Interner
        the interner shared by every paper in one featurizer

    Returns
    -------
    Tuple: the nested record that `s2apler._s2apler_rust.PaperFeaturizer` extracts
    """
    tokens = paper.title.split()
    counters = (
        _counter_input(paper.author_info_coauthor_n_grams, intern),
        _counter_input(paper.venue_ngrams, intern),
        _counter_input(paper.title_ngrams_chars, intern),
        _counter_input(paper.abstract_ngrams_words, intern),
    )
    title = (
        [intern(token) for token in tokens],
        [token in NUMERALS for token in tokens],
        [token in NUMERAL_PRECEDING_WORDS for token in tokens],
        [intern(word) for word in set(tokens) & SPECIAL_PUBLICATION_WORDS],
        [int(year) for year in YEAR.findall(paper.title)],
        paper.title.replace(" ", ""),
    )
    flags = (
        paper.abstract is not None and len(paper.abstract) > 0,
        len(paper.authors) > 0,
        len(paper.venue) > 0,
        paper.source in PDF_SOURCES,
        paper.source in PUBLISHER_SOURCES,
    )
    return (
        counters,
        [[intern(letter) for letter in author.author_info_first_letters] for author in paper.authors],
        # A float, as `diff` converts years; JSON training data can hold float years.
        float(paper.year) if paper.year is not None and paper.year > 1700 else None,
        title,
        flags,
        intern(paper.source) if paper.source is not None else None,
    )


def rust_pair_features(papers: Dict[str, Paper], pairs: List[Tuple[str, str]], n_jobs: int) -> np.ndarray:
    """Computes `_single_pair_featurize`'s features for many pairs with the Rust extension.

    Parameters
    ----------
    papers: Dict[str, Paper]
        preprocessed papers by id
    pairs: List[Tuple[str, str]]
        paper id pairs
    n_jobs: int
        threads to use; -1 means all CPUs, as for the scorer, and 0 means one thread

    Returns
    -------
    np.ndarray: an (n_pairs, NUM_FEATURES) float64 array, with NaN for missing features
    """
    paper_ids = list(dict.fromkeys(paper_id for pair in pairs for paper_id in pair))
    position = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    featurizer = paper_featurizer(papers, paper_ids)
    left = np.fromiter((position[a] for a, _ in pairs), dtype=np.uint32, count=len(pairs))
    right = np.fromiter((position[b] for _, b in pairs), dtype=np.uint32, count=len(pairs))
    return featurizer.featurize(left, right, num_threads=resolve_n_jobs(n_jobs) if n_jobs != 0 else 1)


def paper_featurizer(papers: Dict[str, Paper], paper_ids: List[str]) -> Any:
    """Packs papers once into the native featurizer, which then takes pairs by position.

    Parameters
    ----------
    papers: Dict[str, Paper]
        preprocessed papers by id
    paper_ids: List[str]
        the papers to pack; a paper's position in this list is its position in pairs

    Returns
    -------
    PaperFeaturizer: the native featurizer
    """
    intern = _Interner()
    return load_s2apler_rust_extension().PaperFeaturizer(
        [rust_paper_record(papers[str(paper_id)], intern) for paper_id in paper_ids]
    )


def many_pairs_featurize(
    paper_pairs: List[Tuple[str, str, Union[int, float]]],
    dataset: PDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int,
    *,
    nan_value: float = np.nan,
    backend: str = "rust",
) -> FeaturesAndLabels:
    """
    Featurizes many pairs

    Parameters
    ----------
    paper_pairs: List[pairs]
        the pairs to featurize, as (paper id 1, paper id 2, label); pairs with negative
        labels are partially supervised, and their features are left unset
    dataset: PDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeaturizationInfo object listing the features to use
    n_jobs: int
        threads to use; -1 means all CPUs, and 0 means one thread
    nan_value: float
        the value to replace nans with
    backend: str
        "rust" computes features in the native extension; "python" uses `_single_pair_featurize`,
        which stays as the reference implementation

    Returns
    -------
    np.ndarray: the features for all the pairs
    np.ndarray: the labels for all the pairs
    """
    if backend not in {"rust", "python"}:
        raise ValueError(f"backend must be 'rust' or 'python', got {backend!r}")

    features = np.ones((len(paper_pairs), NUM_FEATURES)) * (-LARGE_INTEGER)
    labels = np.zeros(len(paper_pairs))
    work_pairs = []
    work_indices = []
    for i, pair in tqdm(enumerate(paper_pairs), desc="Creating work", disable=len(paper_pairs) <= 100000):
        labels[i] = pair[2]
        # negative labels are an indication of partial supervision
        if pair[2] < 0:
            continue
        work_pairs.append((pair[0], pair[1]))
        work_indices.append(i)

    if work_pairs and backend == "rust":
        logger.debug(f"Featurizing {len(work_pairs)} pairs in Rust")
        features[work_indices, :] = rust_pair_features(dataset.papers, work_pairs, n_jobs)
    elif work_pairs:
        logger.debug(f"Featurizing {len(work_pairs)} pairs in Python")
        # Only this path reads the global, so the Rust path never keeps old papers alive.
        _set_global_dataset(dataset.papers)
        for pair, index in tqdm(zip(work_pairs, work_indices), total=len(work_pairs), desc="Doing work"):
            features[index, :] = _single_pair_featurize(pair)[0]

    features = features[:, featurizer_info.indices_to_use]
    features[np.isnan(features)] = nan_value
    return features, labels


def featurize(
    dataset: PDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int = 1,
    *,
    nan_value: float = np.nan,
) -> Tuple[FeaturesAndLabels, FeaturesAndLabels, FeaturesAndLabels]:
    """
    Splits a train-mode dataset by block and featurizes sampled train/val/test pairs

    Parameters
    ----------
    dataset: PDData
        a train-mode dataset with clusters
    featurizer_info: FeaturizationInfo
        the FeaturizationInfo object listing the features to use
    n_jobs: int
        threads to use; -1 means all CPUs, and 0 means one thread
    nan_value: float
        the value to replace nans with

    Returns
    -------
    (features, labels) for the train, val and test pairs
    """
    if dataset.mode != "train":
        raise ValueError("featurize needs a train-mode dataset; featurize dataset.all_pairs() for inference")
    train_blocks, val_blocks, test_blocks = dataset.split_cluster_papers()
    train_pairs, val_pairs, test_pairs = dataset.split_pairs(train_blocks, val_blocks, test_blocks)
    return (
        many_pairs_featurize(train_pairs, dataset, featurizer_info, n_jobs, nan_value=nan_value),
        many_pairs_featurize(val_pairs, dataset, featurizer_info, n_jobs, nan_value=nan_value),
        many_pairs_featurize(test_pairs, dataset, featurizer_info, n_jobs, nan_value=nan_value),
    )
