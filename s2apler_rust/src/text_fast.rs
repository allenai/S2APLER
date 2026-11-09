//! Fast exact versions of the title text features for ASCII strings.
//!
//! S2APLER's normalized titles are ASCII, and the character-by-character versions in `text.rs`
//! cost about 30 µs per pair on 90-character titles: most of a large request's featurization.
//! These return identical values. Levenshtein distance is an integer, so any correct algorithm
//! gives the same result. Jaro-Winkler makes the same greedy matches as jellyfish's loop (the
//! first unmatched position in the window, in order), so its match and transposition counts and
//! its floating-point formula are unchanged. Tests compare each function with `text.rs`.

/// Advances one 64-row block of Myers' bit-vector Levenshtein algorithm by one text character.
///
/// `hin` is the horizontal delta entering the block's top row (-1, 0 or +1); the return value is
/// the delta leaving its row marked by `high`. Adapted from edlib's `calculateBlock`.
fn advance_block(pv: &mut u64, mv: &mut u64, eq: u64, hin: i32, high: u64) -> i32 {
    let mut eq = eq;
    let xv = eq | *mv;
    if hin < 0 {
        eq |= 1;
    }
    let xh = ((eq & *pv).wrapping_add(*pv) ^ *pv) | eq;
    let mut ph = *mv | !(xh | *pv);
    let mut mh = *pv & xh;
    let hout = if ph & high != 0 {
        1
    } else if mh & high != 0 {
        -1
    } else {
        0
    };
    ph <<= 1;
    mh <<= 1;
    if hin < 0 {
        mh |= 1;
    } else if hin > 0 {
        ph |= 1;
    }
    *pv = mh | !(xv | ph);
    *mv = ph & xv;
    hout
}

/// Levenshtein distance between byte strings, by Myers' bit-vector algorithm (Hyyrö's blocks).
pub(crate) fn levenshtein_ascii(a: &[u8], b: &[u8]) -> usize {
    if a == b {
        return 0;
    }
    let (pattern, text) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let m = pattern.len();
    if m == 0 {
        return text.len();
    }
    let blocks = m.div_ceil(64);
    // Row positions where each byte occurs in the pattern, one bit per row.
    let mut peq = vec![0u64; 256 * blocks];
    for (row, &byte) in pattern.iter().enumerate() {
        peq[byte as usize * blocks + row / 64] |= 1u64 << (row % 64);
    }
    let mut pv = vec![!0u64; blocks];
    let mut mv = vec![0u64; blocks];
    let last_row = 1u64 << ((m - 1) % 64);
    let mut score = m as isize;
    for &byte in text {
        // The top row of the edit table grows by one per text character.
        let mut hin = 1;
        let base = byte as usize * blocks;
        for block in 0..blocks {
            let high = if block + 1 == blocks {
                last_row
            } else {
                1u64 << 63
            };
            hin = advance_block(&mut pv[block], &mut mv[block], peq[base + block], hin, high);
        }
        score += hin as isize;
    }
    score as usize
}

/// jellyfish's `jaro_winkler_similarity` (without long-tolerance) on byte strings, with bitsets.
pub(crate) fn jaro_winkler_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let (len1, len2) = (s1.len(), s2.len());
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }
    let min_len = len1.min(len2);
    let search_range = (len1.max(len2) / 2).saturating_sub(1);
    let words = len2.div_ceil(64);
    let mut positions = vec![0u64; 256 * words];
    for (j, &byte) in s2.iter().enumerate() {
        positions[byte as usize * words + j / 64] |= 1u64 << (j % 64);
    }
    let mut matched2 = vec![0u64; words];
    let mut matched1 = vec![false; len1];
    let mut common = 0usize;
    for (i, &byte) in s1.iter().enumerate() {
        let low = i.saturating_sub(search_range);
        let high = (i + search_range).min(len2 - 1);
        if low > high {
            continue;
        }
        // The first unmatched position in [low, high] holding this byte, as jellyfish takes.
        let base = byte as usize * words;
        for word in low / 64..=high / 64 {
            let mut candidates = positions[base + word] & !matched2[word];
            if word == low / 64 {
                candidates &= !0u64 << (low % 64);
            }
            if word == high / 64 && high % 64 != 63 {
                candidates &= (1u64 << (high % 64 + 1)) - 1;
            }
            if candidates != 0 {
                matched2[word] |= candidates & candidates.wrapping_neg();
                matched1[i] = true;
                common += 1;
                break;
            }
        }
    }
    if common == 0 {
        return 0.0;
    }
    // Pair the matches in order on both sides and count the mismatched pairs.
    let mut trans_count = 0usize;
    let mut word = 0usize;
    let mut bits = matched2[0];
    for (i, &is_matched) in matched1.iter().enumerate() {
        if !is_matched {
            continue;
        }
        while bits == 0 {
            word += 1;
            bits = matched2[word];
        }
        let j = word * 64 + bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if s1[i] != s2[j] {
            trans_count += 1;
        }
    }
    // The same formula, in the same operation order, as `text::jaro_winkler_similarity`.
    trans_count /= 2;
    let common_f = common as f64;
    let mut weight = (common_f / len1 as f64
        + common_f / len2 as f64
        + (common_f - trans_count as f64) / common_f)
        / 3.0;
    if weight > 0.7 {
        let j = min_len.min(4);
        let mut i = 0usize;
        while i < j && s1[i] == s2[i] {
            i += 1;
        }
        if i > 0 {
            weight += (i as f64) * 0.1 * (1.0 - weight);
        }
    }
    weight
}

/// `text::prefix_dist` on byte strings: common prefix and suffix lengths, without allocating.
///
/// The original loop returns at the longest equal prefix of the shorter string, which is the
/// common prefix length; on the reversed strings it is the common suffix length.
pub(crate) fn prefix_dist_ascii(a: &[u8], b: &[u8]) -> f64 {
    if a == b {
        return 0.0;
    }
    let (min_bytes, max_bytes) = if a.len() < b.len() { (a, b) } else { (b, a) };
    let min_len = min_bytes.len();
    let prefix = min_bytes
        .iter()
        .zip(max_bytes)
        .take_while(|(x, y)| x == y)
        .count();
    let suffix = min_bytes
        .iter()
        .rev()
        .zip(max_bytes.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let distance = |shared: usize| {
        if shared == 0 {
            1.0
        } else {
            1.0 - (shared as f64 / min_len as f64)
        }
    };
    distance(prefix).min(distance(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text;

    fn random_strings(seed: u64, count: usize, max_len: usize, alphabet: &[u8]) -> Vec<String> {
        let mut state = seed;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        (0..count)
            .map(|_| {
                let len = next(max_len + 1);
                (0..len)
                    .map(|_| alphabet[next(alphabet.len())] as char)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn match_the_reference_implementations_on_random_strings() {
        // Small alphabets force many matches and ties; lengths cross the 64- and 128-bit blocks.
        for (seed, alphabet, max_len) in [
            (1u64, &b"ab"[..], 20),
            (2, b"abc", 70),
            (3, b"abcdefghij", 140),
            (4, b"abcdefghijklmnopqrstuvwxyz0123456789", 200),
            (5, b"aaaaab", 130),
        ] {
            let strings = random_strings(seed, 120, max_len, alphabet);
            for a in &strings {
                for b in &strings {
                    let (x, y) = (a.as_bytes(), b.as_bytes());
                    assert_eq!(
                        levenshtein_ascii(x, y),
                        text::levenshtein_distance(a, b),
                        "{a:?} {b:?}"
                    );
                    assert_eq!(
                        jaro_winkler_ascii(x, y).to_bits(),
                        text::jaro_winkler_similarity(a, b, false).to_bits(),
                        "{a:?} {b:?}"
                    );
                    assert_eq!(
                        prefix_dist_ascii(x, y).to_bits(),
                        text::prefix_dist(a, b).to_bits(),
                        "{a:?} {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn handle_empty_and_block_boundary_strings() {
        let long = "x".repeat(64);
        let longer = "x".repeat(65);
        for (a, b) in [
            ("", ""),
            ("", "abc"),
            ("abc", ""),
            ("a", "a"),
            (long.as_str(), longer.as_str()),
        ] {
            assert_eq!(
                levenshtein_ascii(a.as_bytes(), b.as_bytes()),
                text::levenshtein_distance(a, b)
            );
            assert_eq!(
                jaro_winkler_ascii(a.as_bytes(), b.as_bytes()).to_bits(),
                text::jaro_winkler_similarity(a, b, false).to_bits()
            );
            assert_eq!(
                prefix_dist_ascii(a.as_bytes(), b.as_bytes()).to_bits(),
                text::prefix_dist(a, b).to_bits()
            );
        }
    }
}
