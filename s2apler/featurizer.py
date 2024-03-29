from typing import Tuple, List, Union, Dict, Callable, Any, Optional

import os
import multiprocessing
import json
import numpy as np
import functools
import logging

from tqdm import tqdm

from s2apler.data import PDData
from s2apler.consts import (
    CACHE_ROOT,
    FEATURIZER_VERSION,
    LARGE_INTEGER,
    DEFAULT_CHUNK_SIZE,
)
from s2apler.text import (
    diff,
    counter_jaccard,
    numeral_similarity,
    special_publication_word_similarity,
    TEXT_FUNCTIONS,
    name_text_features,
    year_similarity,
    PUBLISHER_SOURCES,
)

logger = logging.getLogger("s2apler")

TupleOfArrays = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]

CACHED_FEATURES: Dict[str, Dict[str, Any]] = {}

PDF_SOURCES = {"MergedPDFExtraction", "ScienceParseMerged", "ScienceParsePlus", "Grobid"}


class FeaturizationInfo:
    """
    Class to store information about how to generate and cache features

    Inputs:
        features_to_use: List[str[]]
            list of feature types to use
        featurizer_version: int
            What version of the featurizer we are on. This should be
            incremented when changing how features are computed so that a new cache
            is created
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
        features_excluded_for_nameless: List[str] = ["title_similarity", "abstract_similarity"],
        featurizer_version: int = FEATURIZER_VERSION,
    ):
        self.features_to_use = features_to_use
        self.features_excluded_for_nameless = set(features_excluded_for_nameless)
        self.feature_names = self.get_feature_names()
        # NOTE: Increment this anytime a change is made to the featurization logic
        self.featurizer_version = featurizer_version

    def get_feature_names(self) -> List[str]:
        """
        Gets all of the feature names

        Returns
        -------
        List[string]: List of all the features names
        """
        feature_names = []
        lightgbm_monotone_constraints = []
        nameless_lightgbm_monotone_constraints = []
        self.indices_to_use = []
        self.nameless_indices_to_use = []
        start_count = 0
        # affiliation features
        if "author_similarity" in self.features_to_use:
            these_features = [
                "author_names_similarity",
                "author_first_letter_compatibility",
                # "author_affiliations_similarity",
                # "author_email_prefix_similarity",
                # "author_email_suffix_similarity",
            ]
            feature_names.extend(these_features)
            lightgbm_monotone_constraints.extend(["1"] * len(these_features))
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            if "author_similarity" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["1"] * len(these_features))
                self.nameless_indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            start_count += len(these_features)

        # email features
        if "venue_similarity" in self.features_to_use:
            these_features = ["venue_similarity"]
            feature_names.extend(these_features)
            lightgbm_monotone_constraints.extend(["1"] * len(these_features))
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            if "venue_similarity" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["1"] * len(these_features))
            start_count += len(these_features)

        # year features
        if "year_diff" in self.features_to_use:
            these_features = ["year_diff"]
            feature_names.extend(these_features)
            lightgbm_monotone_constraints.extend(["-1"])
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            if "year_diff" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["-1"])
                self.nameless_indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            start_count += len(these_features)

        if "title_similarity" in self.features_to_use:
            these_features = [
                "title_character_similarity",
                "title_numeral_similarity",  # trying to catch when we have stuff like "part i vs part ii"
                "title_special_publication_word_similarity",  # containing commentary, response, letter, editorial, etc
                "title_year_similarity",  # whether years mentioned are the same
            ]
            these_features.extend(["title_" + func[1] for func in TEXT_FUNCTIONS])
            feature_names.extend(these_features)
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            lightgbm_monotone_constraints.extend(["1"] * len(these_features))
            if "title_similarity" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["1"] * len(these_features))
                self.nameless_indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            start_count += len(these_features)

        if "abstract_similarity" in self.features_to_use:
            these_features = ["abstract_word_similarity"]
            feature_names.extend(these_features)
            lightgbm_monotone_constraints.extend(["1"] * len(these_features))
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            if "abstract_similarity" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["1"] * len(these_features))
                self.nameless_indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            start_count += len(these_features)

        if "paper_quality" in self.features_to_use:
            these_features = [f"paper_field_count_{i}" for i in ["abstract", "authors", "venue"]]  # 'year'
            these_features.extend(["source_count_pdf", "source_count_publisher", "sources_publisher_are_same"])
            feature_names.extend(these_features)
            lightgbm_monotone_constraints.extend(["0"] * len(these_features))
            self.indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            if "paper_quality" not in self.features_excluded_for_nameless:
                nameless_lightgbm_monotone_constraints.extend(["0"] * len(these_features))
                self.nameless_indices_to_use.extend(list(range(start_count, start_count + len(these_features))))
            start_count += len(these_features)

        self.lightgbm_monotone_constraints = ",".join(lightgbm_monotone_constraints)
        self.nameless_lightgbm_monotone_constraints = ",".join(nameless_lightgbm_monotone_constraints)
        self.number_of_features = start_count

        return feature_names

    @staticmethod
    def feature_cache_key(paper_pair: Tuple) -> str:
        """
        returns the key in the feature cache dictionary for a signature pair

        Parameters
        ----------
        signature_pair: Tuple[string]
            pair of signature ids

        Returns
        -------
        string: the cache key
        """
        return paper_pair[0] + "___" + paper_pair[1]

    def cache_directory(self, dataset_name: str) -> str:
        """
        returns the cache directory for this dataset and featurizer version

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the cache directory
        """
        return os.path.join(CACHE_ROOT, dataset_name, str(self.featurizer_version))

    def cache_file_path(self, dataset_name: str) -> str:
        """
        returns the file path for the features cache

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the full file path for the features cache file
        """
        return os.path.join(
            self.cache_directory(dataset_name),
            "all_features.json",
        )

    def write_cache(self, cached_features: Dict, dataset_name: str):
        """
        Writes the cached features to the features cache file

        Parameters
        ----------
        cached_features: Dict
            the features, keyed by signature pair
        dataset_name: str
            the name of the dataset

        Returns
        -------
        nothing, writes the cache file
        """
        with open(
            self.cache_file_path(dataset_name),
            "w",
        ) as _json_file:
            json.dump(cached_features, _json_file)


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


def _single_pair_featurize(work_input: Tuple[str, str], index: int = -1) -> Tuple[List[Union[int, float]], int]:
    """
    Creates the features array for a single paper pair
    NOTE: This function uses a global variable to support faster multiprocessing. That means that this function
    should only be called from the many_pairs_featurize function below (or if you have carefully set your own global
    variable)

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of paper ids
    index: int
        the index of the pair in the list of all pairs,
        used to keep track of cached features

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    global global_dataset

    features = []

    paper_id_1 = work_input[0]
    paper_id_2 = work_input[1]

    paper_1 = global_dataset[str(paper_id_1)]  # type: ignore
    paper_2 = global_dataset[str(paper_id_2)]  # type: ignore

    # author-related features
    features.extend(
        [
            counter_jaccard(
                paper_1.author_info_coauthor_n_grams,
                paper_2.author_info_coauthor_n_grams,
                denominator_max=5000,
            ),
            # counter_jaccard(
            #     paper_1.author_info_coauthor_affiliations_n_grams,
            #     paper_2.author_info_coauthor_affiliations_n_grams,
            # ),
            # counter_jaccard(
            #     paper_1.author_info_coauthor_email_prefix_n_grams,
            #     paper_2.author_info_coauthor_email_prefix_n_grams,
            # ),
            # counter_jaccard(
            #     paper_1.author_info_coauthor_email_suffix_n_grams,
            #     paper_2.author_info_coauthor_email_suffix_n_grams,
            # ),
            compare_author_first_letters(
                paper_1.authors,
                paper_2.authors,
            ),
        ]
    )

    # venue features
    features.extend(
        [
            # counter_jaccard(paper_1.journal_ngrams, paper_2.journal_ngrams),
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


def parallel_helper(piece_of_work: Tuple, worker_func: Callable):
    """
    Helper function to explode tuple arguments

    Parameters
    ----------
    piece_of_work: Tuple
        the input for the worker func, in tuple form
    worker_func: Callable
        the function that will do the work

    Returns
    -------
    returns the result of calling the worker function
    """
    result = worker_func(*piece_of_work)
    return result


def many_pairs_featurize(
    paper_pairs: List[Tuple[str, str, Union[int, float]]],
    dataset: PDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int,
    use_cache: bool,
    chunk_size: int,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Featurizes many pairs

    Parameters
    ----------
    signature_pairs: List[pairs]
        the pairs to featurize
    dataset: PDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any title features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training rows

    Returns
    -------
    np.ndarray: the main features for all the pairs
    np.ndarray: the labels for all the pairs
    np.ndarray: the nameless features for all the pairs
    """
    global global_dataset
    global_dataset = dataset.papers  # type: ignore

    cached_features: Dict[str, Any] = {"features": {}}
    cache_changed = False
    if use_cache:
        logger.debug("Loading cache...")
        if not os.path.exists(featurizer_info.cache_directory(dataset.name)):
            os.makedirs(featurizer_info.cache_directory(dataset.name))
        if os.path.exists(featurizer_info.cache_file_path(dataset.name)):
            if featurizer_info.cache_file_path(dataset.name) in CACHED_FEATURES:
                cached_features = CACHED_FEATURES[featurizer_info.cache_file_path(dataset.name)]
            else:
                with open(featurizer_info.cache_file_path(dataset.name)) as _json_file:
                    cached_features = json.load(_json_file)
                logger.debug(f"Cache loaded with {len(cached_features['features'])} keys")
        else:
            logger.debug("Cache initiated.")
            cached_features = {}
            cached_features["features"] = {}
            cached_features["features_to_use"] = featurizer_info.features_to_use

    features = np.ones((len(paper_pairs), NUM_FEATURES)) * (-LARGE_INTEGER)
    labels = np.zeros(len(paper_pairs))
    pieces_of_work = []
    logger.debug(f"Creating {len(paper_pairs)} pieces of work")
    for i, pair in tqdm(enumerate(paper_pairs), desc="Creating work", disable=len(paper_pairs) <= 100000):
        labels[i] = pair[2]

        # negative labels are an indication of partial supervision
        if pair[2] < 0:
            continue

        cache_key = pair[0] + "___" + pair[1]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_key = pair[1] + "___" + pair[0]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_changed = True
        pieces_of_work.append(((pair[0], pair[1]), i))

    logger.debug("Created pieces of work")

    if cache_changed:
        if n_jobs > 1:
            logger.debug(f"Cached changed, doing {len(pieces_of_work)} work in parallel")
            with multiprocessing.Pool(processes=n_jobs if len(pieces_of_work) > 1000 else 1) as p:
                _max = len(pieces_of_work)
                with tqdm(total=_max, desc="Doing work", disable=_max <= 10000) as pbar:
                    for feature_output, index in p.imap(
                        functools.partial(parallel_helper, worker_func=_single_pair_featurize),
                        pieces_of_work,
                        min(chunk_size, max(1, int((_max / n_jobs) / 2))),
                    ):
                        # Write to in memory cache if we're not skipping
                        if use_cache:
                            cached_features["features"][
                                featurizer_info.feature_cache_key(paper_pairs[index])
                            ] = feature_output
                        features[index, :] = feature_output
                        pbar.update()
        else:
            logger.debug(f"Cached changed, doing {len(pieces_of_work)} work in serial")
            partial_func = functools.partial(parallel_helper, worker_func=_single_pair_featurize)
            for piece in tqdm(pieces_of_work, total=len(pieces_of_work), desc="Doing work"):
                result = partial_func(piece)
                if use_cache:
                    cached_features["features"][featurizer_info.feature_cache_key(paper_pairs[result[1]])] = result[0]
                features[result[1], :] = result[0]
        logger.debug("Work completed")

    if use_cache and cache_changed:
        logger.debug("Writing to on disk cache")
        featurizer_info.write_cache(cached_features, dataset.name)
        logger.debug(f"Cache written with {len(cached_features['features'])} keys.")

    if use_cache:
        logger.debug("Writing to in memory cache")
        CACHED_FEATURES[featurizer_info.cache_file_path(dataset.name)] = cached_features
        logger.debug("In memory cache written")

    if delete_training_data:
        logger.debug("Deleting some training rows")
        negative_label_indices = labels == 0
        high_coauthor_sim_indices = features[:, featurizer_info.get_feature_names().index("coauthor_similarity")] > 0.95
        indices_to_remove = negative_label_indices & high_coauthor_sim_indices
        logger.debug(f"Intending to remove {sum(indices_to_remove)} rows")
        original_size = len(labels)
        features = features[~indices_to_remove, :]
        labels = labels[~indices_to_remove]
        logger.debug(f"Removed {original_size - features.shape[0]} rows and {original_size - len(labels)} labels")

    logger.debug("Making numpy arrays for features and labels")
    # have to do this before subselecting features
    if nameless_featurizer_info is not None:
        nameless_features = features[:, featurizer_info.nameless_indices_to_use]
        nameless_features[np.isnan(nameless_features)] = nan_value
    else:
        nameless_features = None

    features = features[:, featurizer_info.indices_to_use]
    features[np.isnan(features)] = nan_value

    logger.debug("Numpy arrays made")
    return features, labels, nameless_features


def featurize(
    dataset: PDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int = 1,
    use_cache: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
) -> Union[Tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays], TupleOfArrays]:
    """
    Featurizes the input dataset

    Parameters
    ----------
    dataset: PDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training examples

    Returns
    -------
    train/val/test features and labels if mode is 'train',
    features and labels for all pairs if mode is 'inference'
    """
    if dataset.mode == "inference":
        logger.debug("featurizing all pairs")
        all_pairs = dataset.all_pairs()
        all_features = many_pairs_featurize(
            all_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.debug("featurized all pairs")
        return all_features
    else:
        if dataset.train_pairs is None:
            if dataset.train_papers is not None:
                (
                    train_papers_dict,
                    val_papers_dict,
                    test_papers_dict,
                ) = dataset.split_data_papers_fixed()
            else:
                (
                    train_papers_dict,
                    val_papers_dict,
                    test_papers_dict,
                ) = dataset.split_cluster_papers()  # type: ignore

            train_pairs, val_pairs, test_pairs = dataset.split_pairs(
                train_papers_dict, val_papers_dict, test_papers_dict
            )

        else:
            train_pairs, val_pairs, test_pairs = dataset.fixed_pairs()

        logger.debug("featurizing train")
        train_features = many_pairs_featurize(
            train_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            delete_training_data,
        )
        logger.debug("featurized train, featurizing val")
        val_features = many_pairs_featurize(
            val_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.debug("featurized val, featurizing test")
        test_features = many_pairs_featurize(
            test_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.debug("featurized test")
        return train_features, val_features, test_features
