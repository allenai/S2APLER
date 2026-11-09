//! Counts n-grams the way S2APLER's Python `get_text_ngrams` and `get_text_ngrams_words` do.
//!
//! Python passes each text as UTF-32 code points, already normalized or lowercased there, so
//! Rust never decodes, lowercases or splits Unicode itself. Code points also carry any lone
//! surrogate a Python string can hold.
//!
//! - Character n-grams are windows of code points. A window holding a space is skipped, as
//!   Python's `" " not in x` filter skips it.
//! - Word n-grams are runs of 1 to 3 words. Python passes the words joined by single spaces,
//!   so a run is one contiguous slice, equal to Python's `" ".join(...)` key.
//!
//! N-grams of different lengths never share a key, so Python's `Counter` union is a plain
//! merge, and each count is the number of occurrences.

use crate::counter::{CounterData, MAX_EXACT_COUNT};

type HashMap<K, V> = std::collections::HashMap<K, V, ahash::RandomState>;

const SPACE: u32 = ' ' as u32;

/// Counter keys for counted n-grams start here, apart from the ids Python interns.
pub(crate) const TEXT_GRAM_ID_BASE: u64 = 1 << 62;

/// Interns n-grams of one featurizer's papers; equal ids mean equal code point sequences.
#[derive(Default)]
pub(crate) struct GramInterner<'a> {
    ids: HashMap<&'a [u32], u64>,
}

impl<'a> GramInterner<'a> {
    fn id(&mut self, gram: &'a [u32]) -> u64 {
        let next = TEXT_GRAM_ID_BASE + self.ids.len() as u64;
        *self.ids.entry(gram).or_insert(next)
    }
}

/// Decodes UTF-32-LE bytes, as Python's `str.encode("utf-32-le", "surrogatepass")` writes them.
pub(crate) fn code_points(bytes: &[u8]) -> Option<Vec<u32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return None;
    }
    Some(
        chunks
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Turns gram ids into a counter, or `None` when there are none, as for an empty `Counter`.
fn counter_from_ids(mut ids: Vec<u64>) -> Option<CounterData> {
    if ids.is_empty() {
        return None;
    }
    ids.sort_unstable();
    let mut entries: Vec<(u64, u64)> = Vec::new();
    for id in ids {
        match entries.last_mut() {
            Some(last) if last.0 == id => last.1 += 1,
            _ => entries.push((id, 1)),
        }
    }
    // A text holds fewer than 2**53 grams, so every count and the total stay exact.
    debug_assert!(entries.iter().map(|entry| entry.1).sum::<u64>() < MAX_EXACT_COUNT);
    Some(CounterData::new(entries))
}

/// Python's `get_text_ngrams(text, stopwords=None, ...)` for n-gram lengths `sizes`.
pub(crate) fn char_ngrams<'a>(
    text: &'a [u32],
    sizes: &[usize],
    interner: &mut GramInterner<'a>,
) -> Option<CounterData> {
    let mut ids = Vec::new();
    for &size in sizes {
        for gram in text.windows(size) {
            if !gram.contains(&SPACE) {
                ids.push(interner.id(gram));
            }
        }
    }
    counter_from_ids(ids)
}

/// Python's `get_text_ngrams_words(text, stopwords=None)`: word unigrams, bigrams and trigrams.
///
/// `text` is the words joined by single spaces, as `" ".join(text.split())` makes them.
pub(crate) fn word_ngrams<'a>(
    text: &'a [u32],
    interner: &mut GramInterner<'a>,
) -> Option<CounterData> {
    if text.is_empty() {
        return None;
    }
    // (start, end) of each word in `text`.
    let mut words = Vec::new();
    let mut start = 0;
    for (position, &point) in text.iter().enumerate() {
        if point == SPACE {
            words.push((start, position));
            start = position + 1;
        }
    }
    words.push((start, text.len()));
    let mut ids = Vec::new();
    for length in 1..=3 {
        for run in words.windows(length) {
            ids.push(interner.id(&text[run[0].0..run[length - 1].1]));
        }
    }
    counter_from_ids(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(text: &str) -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    }

    fn grams(counter: Option<CounterData>, interner: &GramInterner) -> Vec<(String, u64)> {
        let names: HashMap<u64, String> = interner
            .ids
            .iter()
            .map(|(gram, id)| {
                (
                    *id,
                    gram.iter().map(|p| char::from_u32(*p).unwrap()).collect(),
                )
            })
            .collect();
        let mut out: Vec<(String, u64)> = counter
            .map(|c| {
                c.iter()
                    .map(|(id, count)| (names[id].clone(), *count as u64))
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    #[test]
    fn character_grams_skip_spaces_and_count_repeats() {
        let text = points("ab abab");
        let mut interner = GramInterner::default();
        let counter = char_ngrams(&text, &[2, 3], &mut interner);
        let expected = [("ab", 3), ("aba", 1), ("ba", 1), ("bab", 1)];
        let expected: Vec<(String, u64)> =
            expected.iter().map(|(g, c)| (g.to_string(), *c)).collect();
        assert_eq!(grams(counter, &interner), expected);
    }

    #[test]
    fn word_grams_are_runs_of_words() {
        let text = points("a b a b");
        let mut interner = GramInterner::default();
        let counter = word_ngrams(&text, &mut interner);
        let expected = [
            ("a", 2),
            ("a b", 2),
            ("a b a", 1),
            ("b", 2),
            ("b a", 1),
            ("b a b", 1),
        ];
        let expected: Vec<(String, u64)> =
            expected.iter().map(|(g, c)| (g.to_string(), *c)).collect();
        assert_eq!(grams(counter, &interner), expected);
    }

    #[test]
    fn empty_and_short_texts_have_no_counter() {
        let mut interner = GramInterner::default();
        assert!(char_ngrams(&[], &[1, 2], &mut interner).is_none());
        let short = points("ab");
        assert!(char_ngrams(&short, &[3, 4], &mut interner).is_none());
        assert!(word_ngrams(&[], &mut interner).is_none());
    }

    #[test]
    fn decodes_utf32_including_lone_surrogates() {
        let bytes = [0x41, 0, 0, 0, 0x00, 0xd8, 0, 0];
        assert_eq!(code_points(&bytes), Some(vec![0x41, 0xd800]));
        assert_eq!(code_points(&bytes[..3]), None);
    }
}
