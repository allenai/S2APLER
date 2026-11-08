//! String similarity functions. Copied from s2and's `features.rs` and `lib.rs` unless noted.
//!
//! jellyfish 1.x, which S2APLER's Python featurizer calls, compares grapheme clusters; these
//! compare characters, as s2and's do. The two agree on ASCII, and S2APLER's normalized titles
//! are ASCII: `normalize_text` transliterates with unidecode and keeps only [a-z0-9] and spaces.

use crate::text_fast;

pub(crate) fn py_len(s: &str) -> usize {
    // Python len() semantics for str: count of Unicode scalar values
    s.chars().count()
}

/// `py_len(s) > 1`, decoding at most two characters instead of counting all.
pub(crate) fn py_len_above_one(s: &str) -> bool {
    s.chars().nth(1).is_some()
}

pub(crate) fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    if a.is_ascii() && b.is_ascii() {
        return levenshtein_distance_bytes(a.as_bytes(), b.as_bytes());
    }

    let b_chars: Vec<char> = b.chars().collect();
    let len_b = b_chars.len();
    if a.is_empty() {
        return len_b;
    }
    if len_b == 0 {
        return a.chars().count();
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for (i, a_char) in a.chars().enumerate() {
        cur[0] = i + 1;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a_char == b_chars[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

pub(crate) fn levenshtein_distance_bytes(a: &[u8], b: &[u8]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }
    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut cur: Vec<usize> = vec![0; len_b + 1];
    for i in 1..=len_a {
        cur[0] = i;
        for j in 1..=len_b {
            let deletion = prev[j] + 1;
            let insertion = cur[j - 1] + 1;
            let edit = prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = deletion.min(insertion).min(edit);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[len_b]
}

/// S2APLER's `prefix_dist`: the smaller of the forward and reversed prefix distances.
///
/// s2and's `prefix_dist` compares prefixes only; S2APLER's Python version also compares the
/// reversed strings, because titles sometimes differ only in their first characters.
pub(crate) fn prefix_dist(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (min_chars, max_chars) = if a_chars.len() < b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };
    let min_reversed: Vec<char> = min_chars.iter().rev().copied().collect();
    let max_reversed: Vec<char> = max_chars.iter().rev().copied().collect();
    prefix_dist_helper(min_chars, max_chars).min(prefix_dist_helper(&min_reversed, &max_reversed))
}

/// The prefix loop of s2and's `prefix_dist`.
fn prefix_dist_helper(min_chars: &[char], max_chars: &[char]) -> f64 {
    let min_len = min_chars.len();
    for i in (1..=min_len).rev() {
        if min_chars[..i] == max_chars[..i] {
            return 1.0 - (i as f64 / min_len as f64);
        }
    }
    1.0
}

pub(crate) fn jaro_winkler_similarity(a: &str, b: &str, long_tolerance: bool) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }
    let min_len = a_len.min(b_len);
    let mut search_range = a_len.max(b_len) / 2;
    if search_range > 0 {
        search_range -= 1;
    }
    let mut a_flags = vec![false; a_len];
    let mut b_flags = vec![false; b_len];
    let mut common_chars = 0usize;
    for i in 0..a_len {
        let low = if i > search_range {
            i - search_range
        } else {
            0
        };
        let mut hi = i + search_range;
        if hi >= b_len {
            hi = b_len.saturating_sub(1);
        }
        for j in low..=hi {
            if !b_flags[j] && a_chars[i] == b_chars[j] {
                a_flags[i] = true;
                b_flags[j] = true;
                common_chars += 1;
                break;
            }
        }
    }
    if common_chars == 0 {
        return 0.0;
    }
    let mut k = 0usize;
    let mut trans_count = 0usize;
    for i in 0..a_len {
        if a_flags[i] {
            while k < b_len {
                if b_flags[k] {
                    break;
                }
                k += 1;
            }
            if k < b_len && a_chars[i] != b_chars[k] {
                trans_count += 1;
            }
            k += 1;
        }
    }
    trans_count /= 2;
    let common_f = common_chars as f64;
    let weight = (common_f / a_len as f64
        + common_f / b_len as f64
        + (common_f - trans_count as f64) / common_f)
        / 3.0;
    let mut weight = weight;
    if weight > 0.7 {
        let j = min_len.min(4);
        let mut i = 0usize;
        while i < j && a_chars[i] == b_chars[i] {
            i += 1;
        }
        if i > 0 {
            weight += (i as f64) * 0.1 * (1.0 - weight);
        }
        if long_tolerance && min_len > 4 && common_chars > i + 1 && 2 * common_chars >= min_len + i
        {
            weight += (1.0 - weight)
                * ((common_chars - i - 1) as f64 / (a_len + b_len - i * 2 + 2) as f64);
        }
    }
    weight
}

pub(crate) fn year_diff_capped(year1: Option<f64>, year2: Option<f64>, cap: f64) -> f64 {
    // Adapted from s2and's `year_diff`: years arrive as floats, as S2APLER's `diff` converts
    // them, and the cap is a parameter.
    let (Some(y1), Some(y2)) = (year1, year2) else {
        return f64::NAN;
    };
    (y1 - y2).abs().min(cap)
}

/// S2APLER's `name_text_features`: normalized Levenshtein, prefix distance and Jaro-Winkler.
///
/// Adapted from s2and's version, which adds an LCS feature and returns NaN only for empty names.
/// S2APLER returns NaN when either name has at most one character, as its Python code does.
pub(crate) fn name_text_features(n1: &str, n2: &str) -> [f64; 3] {
    if !py_len_above_one(n1) || !py_len_above_one(n2) {
        return [f64::NAN; 3];
    }
    if n1.is_ascii() && n2.is_ascii() {
        // Normalized titles are ASCII; the byte versions return identical values much faster.
        let (b1, b2) = (n1.as_bytes(), n2.as_bytes());
        let lev = text_fast::levenshtein_ascii(b1, b2) as f64 / (b1.len().max(b2.len()) as f64);
        return [
            lev,
            text_fast::prefix_dist_ascii(b1, b2),
            text_fast::jaro_winkler_ascii(b1, b2),
        ];
    }
    let lev = levenshtein_distance(n1, n2) as f64 / (py_len(n1).max(py_len(n2)) as f64);
    let pref = prefix_dist(n1, n2);
    let jaro = jaro_winkler_similarity(n1, n2, false);
    [lev, pref, jaro]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_distance_takes_the_better_of_both_ends() {
        // Forward prefixes share nothing; the reversed strings share "yduts".
        assert_eq!(prefix_dist("xstudy", "astudy"), 1.0 - 5.0 / 6.0);
        assert_eq!(prefix_dist("abc", "abd"), 1.0 - 2.0 / 3.0);
        assert_eq!(prefix_dist("same", "same"), 0.0);
    }

    #[test]
    fn name_text_features_need_two_characters() {
        assert!(name_text_features("a", "abc").iter().all(|v| v.is_nan()));
        let [lev, prefix, jaro] = name_text_features("kitten", "sitting");
        assert_eq!(lev, 3.0 / 7.0);
        // Expected values come from S2APLER's Python `name_text_features`.
        assert_eq!(prefix, 1.0);
        assert_eq!(jaro, 0.746031746031746);
    }

    #[test]
    fn year_difference_is_capped() {
        assert_eq!(year_diff_capped(Some(2001.0), Some(2020.0), 5.0), 5.0);
        assert_eq!(year_diff_capped(Some(2019.0), Some(2020.5), 5.0), 1.5);
        assert!(year_diff_capped(None, Some(2020.0), 5.0).is_nan());
    }
}
