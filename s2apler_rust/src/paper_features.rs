//! S2APLER's 18 pair features, mirroring `s2apler.featurizer._single_pair_featurize`.
//!
//! Python preprocesses each paper once (LaTeX, transliteration, n-gram counters) and passes a
//! compact record; this module computes every pairwise feature from two records. Each feature
//! repeats the Python operation order so results are bit-identical.

use crate::counter::{counter_jaccard_data, CounterData};
use crate::text::{name_text_features, year_diff_capped};

/// Number of features per pair; equals `s2apler.featurizer.NUM_FEATURES`.
pub(crate) const NUM_FEATURES: usize = 18;

/// One preprocessed paper. Ids are interned per featurizer, so equal ids mean equal strings.
pub(crate) struct PaperRecord {
    pub(crate) coauthor_ngrams: Option<CounterData>,
    pub(crate) venue_ngrams: Option<CounterData>,
    pub(crate) title_char_ngrams: Option<CounterData>,
    pub(crate) abstract_word_ngrams: Option<CounterData>,
    /// Each author's first-letter set, as sorted unique code points.
    pub(crate) author_first_letters: Vec<Vec<u32>>,
    /// The publication year when it is above 1700, else None.
    pub(crate) year: Option<i64>,
    /// Title tokens (`title.split()`) with whether each is a numeral or a numeral-preceding word.
    pub(crate) title_tokens: Vec<u32>,
    pub(crate) title_token_is_numeral: Vec<bool>,
    pub(crate) title_token_precedes_numeral: Vec<bool>,
    /// Sorted unique special publication words in the title.
    pub(crate) title_special_words: Vec<u32>,
    /// Years found in the title by `YEAR.findall`, in order.
    pub(crate) title_years: Vec<u32>,
    /// The normalized title with spaces removed.
    pub(crate) title_without_spaces: String,
    pub(crate) has_abstract: bool,
    pub(crate) has_authors: bool,
    pub(crate) has_venue: bool,
    pub(crate) is_pdf_source: bool,
    pub(crate) is_publisher_source: bool,
    pub(crate) source: Option<u32>,
}

fn flag(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

/// `compare_author_first_letters` with its defaults (same length, strict order).
fn author_first_letter_compatibility(a: &[Vec<u32>], b: &[Vec<u32>]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    if a.len() != b.len() {
        return 0.0;
    }
    for (letters_a, letters_b) in a.iter().zip(b) {
        let min_size = 2.min(letters_a.len().min(letters_b.len()));
        if sorted_intersection_len(letters_a, letters_b) < min_size {
            return 0.0;
        }
    }
    1.0
}

fn sorted_intersection_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            count += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

/// `numeral_similarity`: do numerals at aligned title positions match?
///
/// Python walks `zip_longest(tokens_a, tokens_b, fillvalue="PAD")`, and `NUMERALS` contains
/// "PAD", so a position past the shorter title counts as a numeral there. A position counts
/// when both sides are numerals, it is past the first token, and either both titles continue
/// with numeral-preceding words before it or it is either title's last token. A padded
/// position never equals a real token.
fn numeral_similarity(a: &PaperRecord, b: &PaperRecord) -> f64 {
    let (len_a, len_b) = (a.title_tokens.len(), b.title_tokens.len());
    let mut any = false;
    let mut all_match = true;
    for i in 1..len_a.max(len_b) {
        let (in_a, in_b) = (i < len_a, i < len_b);
        let numeral_a = !in_a || a.title_token_is_numeral[i];
        let numeral_b = !in_b || b.title_token_is_numeral[i];
        if !(numeral_a && numeral_b) {
            continue;
        }
        let preceded = in_a
            && in_b
            && a.title_token_precedes_numeral[i - 1]
            && b.title_token_precedes_numeral[i - 1];
        if preceded || i + 1 == len_a || i + 1 == len_b {
            any = true;
            all_match &= in_a && in_b && a.title_tokens[i] == b.title_tokens[i];
        }
    }
    if any {
        flag(all_match)
    } else {
        f64::NAN
    }
}

/// `special_publication_word_similarity`: same set of words like "erratum" or "reply"?
fn special_publication_word_similarity(a: &PaperRecord, b: &PaperRecord) -> f64 {
    if a.title_special_words.is_empty() && b.title_special_words.is_empty() {
        f64::NAN
    } else {
        flag(a.title_special_words == b.title_special_words)
    }
}

/// `year_similarity`: the same years, in the same order, in both titles?
fn title_year_similarity(a: &PaperRecord, b: &PaperRecord) -> f64 {
    if a.title_years.is_empty() && b.title_years.is_empty() {
        f64::NAN
    } else {
        flag(a.title_years == b.title_years)
    }
}

/// Writes the 18 features of one pair into `out`, in `FeaturizationInfo.feature_names` order.
pub(crate) fn pair_features(a: &PaperRecord, b: &PaperRecord, out: &mut [f64]) {
    let [levenshtein, prefix, jaro] =
        name_text_features(&a.title_without_spaces, &b.title_without_spaces);
    let publisher_sources = flag(a.is_publisher_source) + flag(b.is_publisher_source);
    out[..NUM_FEATURES].copy_from_slice(&[
        counter_jaccard_data(&a.coauthor_ngrams, &b.coauthor_ngrams, 5000.0),
        author_first_letter_compatibility(&a.author_first_letters, &b.author_first_letters),
        counter_jaccard_data(&a.venue_ngrams, &b.venue_ngrams, f64::INFINITY),
        year_diff_capped(a.year, b.year, 5.0),
        counter_jaccard_data(&a.title_char_ngrams, &b.title_char_ngrams, f64::INFINITY),
        numeral_similarity(a, b),
        special_publication_word_similarity(a, b),
        title_year_similarity(a, b),
        levenshtein,
        prefix,
        jaro,
        counter_jaccard_data(
            &a.abstract_word_ngrams,
            &b.abstract_word_ngrams,
            f64::INFINITY,
        ),
        flag(a.has_abstract) + flag(b.has_abstract),
        flag(a.has_authors) + flag(b.has_authors),
        flag(a.has_venue) + flag(b.has_venue),
        flag(a.is_pdf_source || b.is_pdf_source),
        publisher_sources,
        if publisher_sources == 2.0 {
            flag(a.source == b.source)
        } else {
            f64::NAN
        },
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(tokens: &[&str], numerals: &[&str], preceding: &[&str]) -> PaperRecord {
        let id = |token: &&str| {
            token
                .bytes()
                .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32))
        };
        PaperRecord {
            coauthor_ngrams: None,
            venue_ngrams: None,
            title_char_ngrams: None,
            abstract_word_ngrams: None,
            author_first_letters: Vec::new(),
            year: None,
            title_tokens: tokens.iter().map(id).collect(),
            title_token_is_numeral: tokens.iter().map(|t| numerals.contains(t)).collect(),
            title_token_precedes_numeral: tokens.iter().map(|t| preceding.contains(t)).collect(),
            title_special_words: Vec::new(),
            title_years: Vec::new(),
            title_without_spaces: tokens.concat(),
            has_abstract: false,
            has_authors: false,
            has_venue: false,
            is_pdf_source: false,
            is_publisher_source: false,
            source: None,
        }
    }

    const NUMERALS: [&str; 3] = ["i", "ii", "2"];
    const PRECEDING: [&str; 2] = ["part", "volume"];

    #[test]
    fn numeral_past_the_shorter_title_compares_with_padding() {
        // Python: zip_longest pads with "PAD", which NUMERALS contains, and "PAD" != "i".
        let short = record(&["internet", "resources"], &NUMERALS, &PRECEDING);
        let long = record(
            &["internet", "resources", "part", "i"],
            &NUMERALS,
            &PRECEDING,
        );
        assert_eq!(numeral_similarity(&short, &long), 0.0);
    }

    #[test]
    fn matching_and_differing_numerals_after_preceding_words() {
        let a = record(&["results", "part", "ii", "methods"], &NUMERALS, &PRECEDING);
        let same = record(&["results", "part", "ii", "methods"], &NUMERALS, &PRECEDING);
        let other = record(&["results", "part", "i", "methods"], &NUMERALS, &PRECEDING);
        assert_eq!(numeral_similarity(&a, &same), 1.0);
        assert_eq!(numeral_similarity(&a, &other), 0.0);
    }

    #[test]
    fn numerals_without_context_are_missing() {
        let a = record(&["a", "2", "b", "c"], &NUMERALS, &PRECEDING);
        let b = record(&["x", "2", "y", "z"], &NUMERALS, &PRECEDING);
        assert!(numeral_similarity(&a, &b).is_nan());
    }

    #[test]
    fn first_letters_need_two_shared_letters_when_both_have_two() {
        assert_eq!(
            author_first_letter_compatibility(&[vec![1, 2, 3]], &[vec![1, 3]]),
            1.0
        );
        assert_eq!(
            author_first_letter_compatibility(&[vec![1, 2]], &[vec![1, 4]]),
            0.0
        );
        assert_eq!(
            author_first_letter_compatibility(&[vec![1]], &[vec![1, 4]]),
            1.0
        );
        assert_eq!(
            author_first_letter_compatibility(&[vec![1]], &[vec![1], vec![2]]),
            0.0
        );
        assert!(author_first_letter_compatibility(&[], &[vec![1]]).is_nan());
    }
}
