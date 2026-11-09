"""Featurize synthetic papers with an installed ``s2apler`` and compare Rust with Python.

Run it from outside the checkout, so ``s2apler`` imports from the installed distribution.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from s2apler.data import PDData
from s2apler.featurizer import _set_global_dataset, _single_pair_featurize, rust_pair_features

PAPERS = {
    "1": {"title": "Graph methods for clustering, part I", "year": 2019, "venue": "KDD", "source": "Crossref"},
    "2": {"title": "Graph methods for clustering, part II", "year": 2020, "venue": "KDD", "source": "Crossref"},
    "3": {"title": "Erratum: graph methods for clustering", "year": 2021, "source": "Grobid"},
    "4": {"title": "Reply to comments from 1999", "abstract": "We reply.", "source": "Elsevier"},
}


def main() -> None:
    papers = {
        key: {"authors": [{"author_info_first": "Ada", "author_info_last": "Lovelace"}], "block": "b", **paper}
        for key, paper in PAPERS.items()
    }
    dataset = PDData(papers=papers, name="smoke", mode="inference")
    pairs = list(combinations(sorted(dataset.papers), 2))
    _set_global_dataset(dataset.papers)
    expected = np.array([_single_pair_featurize(pair)[0] for pair in pairs], dtype=float)
    actual = rust_pair_features(dataset.papers, pairs, 2)
    same = (expected == actual) | (np.isnan(expected) & np.isnan(actual))
    if not same.all():
        raise RuntimeError(f"Rust and Python features differ at {np.argwhere(~same)[:5].tolist()}")
    print(f"Rust and Python features match on {len(pairs)} synthetic pairs")


if __name__ == "__main__":
    main()
