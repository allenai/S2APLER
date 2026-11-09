import unittest
import numpy as np
from collections import Counter

from s2apler.text import (
    normalize_text,
    name_text_features,
    get_text_ngrams,
    get_text_ngrams_words,
    counter_jaccard,
    diff,
)
from s2apler.consts import NUMPY_NAN


class TestClusterer(unittest.TestCase):
    def test_normalize_text(self):
        assert "" == normalize_text(None)
        assert "" == normalize_text("")
        assert "text" == normalize_text("TeXt")
        assert "te han zi xt" == normalize_text("te'漢字xt")
        assert "text" == normalize_text("te'xt", True)

    def test_name_similarity_features(self):
        assert [NUMPY_NAN] * 3 == name_text_features("", None)
        assert [0.0, 0.0, 1.0] == name_text_features("text", "text")
        assert all([s >= 0.0 and s <= 1.0 for s in name_text_features("textual", "txt")])

    def test_get_text_ngrams(self):
        assert Counter() == get_text_ngrams(None)
        assert Counter() == get_text_ngrams("the")
        assert Counter(
            ["hell", "ello", "hel", "ell", "llo", "he", "el", "ll", "lo", "wor", "wo", "or"]
        ) == get_text_ngrams("hello wor")
        assert Counter(
            [
                "hell",
                "ello",
                "hel",
                "ell",
                "llo",
                "he",
                "el",
                "ll",
                "lo",
                "wor",
                "wo",
                "or",
                "h",
                "e",
                "l",
                "l",
                "o",
                "w",
                "o",
                "r",
            ]
        ) == get_text_ngrams("hello wor", use_unigrams=True)

    def test_get_text_ngrams_words(self):
        assert Counter() == get_text_ngrams_words(None)
        assert Counter() == get_text_ngrams_words("the")
        assert Counter(
            [
                "quick green fox",
                "green fox jumped",
                "quick green",
                "green fox",
                "fox jumped",
                "quick",
                "green",
                "fox",
                "jumped",
            ]
        ) == get_text_ngrams_words("the quick green fox jumped")

    def test_counter_jaccard(self):
        assert np.isnan(counter_jaccard(Counter(), Counter()))
        self.assertAlmostEqual(4 / 6, counter_jaccard(Counter([1, 2, 3, 4, 5]), Counter([1, 2, 3, 4, 6])))
        self.assertAlmostEqual(4 / 7, counter_jaccard(Counter([1, 2, 3, 4, 5, 5]), Counter([1, 2, 3, 4, 6])))

    def test_diff(self):
        assert np.isnan(diff(None, None))
        assert 5 == diff(10, 5)
        assert 5 == diff(5, 10)
