//! Finds the rows a numerical LightGBM model must score identically, so each is scored once.
//!
//! The keys come from s2and's lockstep booster (`lightgbm_booster/fast.rs`, `KeyTable`): a
//! row's key for a feature is its bin among the feature's split thresholds, with NaN apart and
//! LightGBM's zero window folded into 0. A split sends a row left exactly when the key is at or
//! below the split's bin, so rows with equal keys take the same path through every tree. They
//! get the same leaves in the same order and bit-identical scores.

use rayon::prelude::*;

/// Hash maps with aHash, as in s2and's extension: SipHash would dominate the grouping pass.
type HashMap<K, V> = std::collections::HashMap<K, V, ahash::RandomState>;

// Copied from s2and's `lightgbm_booster.rs`.
/// LightGBM's kZeroThreshold is the float literal 1e-35f widened to double
/// (include/LightGBM/meta.h); the same widened value appears verbatim as
/// zero-boundary split thresholds in saved model files.
const K_ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

#[inline]
fn is_zero(fval: f64) -> bool {
    (-K_ZERO_THRESHOLD..=K_ZERO_THRESHOLD).contains(&fval)
}

#[inline]
fn normalize_dense_zero(fval: f64) -> f64 {
    if is_zero(fval) {
        0.0
    } else {
        fval
    }
}

// Copied from s2and's `lightgbm_booster/fast.rs`.
const MAX_KEY_THRESHOLDS: usize = 255;
const NAN_LEFT_KEY: u8 = 0;
const NAN_RIGHT_KEY: u8 = u8::MAX;

/// Per-feature sorted thresholds and the row-to-key transform; s2and's `KeyTable`, built from
/// threshold lists rather than a parsed model.
struct KeyTable {
    /// Original feature indices that appear in at least one split.
    used_features: Vec<usize>,
    /// Per feature: strictly increasing, non-NaN, at most `MAX_KEY_THRESHOLDS`.
    thresholds: Vec<Vec<f64>>,
}

impl KeyTable {
    /// Collects thresholds; `None` if a feature exceeds the key capacity.
    /// Callers have already rejected NaN thresholds.
    fn build(mut thresholds: Vec<Vec<f64>>) -> Option<Self> {
        for list in &mut thresholds {
            // total_cmp places -0.0 next to 0.0, so dedup by == merges them.
            list.sort_by(f64::total_cmp);
            list.dedup_by(|next, kept| next == kept);
            if list.len() > MAX_KEY_THRESHOLDS {
                return None;
            }
        }
        let used_features = thresholds
            .iter()
            .enumerate()
            .filter_map(|(feature, list)| (!list.is_empty()).then_some(feature))
            .collect();
        Some(Self {
            used_features,
            thresholds,
        })
    }

    /// The (NaN-left, NaN-right) keys of one raw feature value.
    #[inline]
    fn keys(&self, feature: usize, value: f64) -> (u8, u8) {
        let value = normalize_dense_zero(value);
        if value.is_nan() {
            return (NAN_LEFT_KEY, NAN_RIGHT_KEY);
        }
        // At most MAX_KEY_THRESHOLDS = 255 entries, so the count fits.
        let bin = self.thresholds[feature].partition_point(|threshold| *threshold < value) as u8;
        (bin, bin)
    }
}

/// Groups rows by their keys on every split feature.
///
/// `values` holds rows of `thresholds.len()` columns, row-major. An empty threshold list means
/// no split uses the column. Thresholds must not be NaN.
///
/// Returns the first row of each group, in order of appearance, and each row's group; `None`
/// when a feature has more thresholds than the keys can hold.
pub(crate) fn distinct_rows(
    values: &[f64],
    thresholds: Vec<Vec<f64>>,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let column_count = thresholds.len();
    let table = KeyTable::build(thresholds)?;
    let row_count = if column_count == 0 {
        0
    } else {
        values.len() / column_count
    };
    // A NaN key pair (0, 255) never equals a number's (bin, bin), so NaN stays apart.
    let width = 2 * table.used_features.len();
    let mut keys = vec![0u8; row_count * width];
    if width > 0 {
        keys.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, key)| {
                let start = row * column_count;
                for (slot, &feature) in key.chunks_mut(2).zip(&table.used_features) {
                    let (nan_left, nan_right) = table.keys(feature, values[start + feature]);
                    slot[0] = nan_left;
                    slot[1] = nan_right;
                }
            });
    }
    let mut groups: HashMap<&[u8], u32> = HashMap::default();
    let mut first_rows = Vec::new();
    let mut inverse = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let key = &keys[row * width..(row + 1) * width];
        let next_group = first_rows.len() as u32;
        let group = *groups.entry(key).or_insert(next_group);
        if group == next_group {
            first_rows.push(row as u32);
        }
        inverse.push(group);
    }
    Some((first_rows, inverse))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(thresholds: Vec<Vec<f64>>) -> KeyTable {
        KeyTable::build(thresholds).unwrap()
    }

    #[test]
    fn keys_split_at_thresholds_and_keep_nan_apart() {
        let table = table(vec![vec![2.0, 0.5, 2.0]]);
        assert_eq!(table.keys(0, 0.5), (0, 0));
        assert_eq!(table.keys(0, f64::from_bits(0.5f64.to_bits() + 1)), (1, 1));
        assert_eq!(table.keys(0, 2.0), (1, 1));
        assert_eq!(table.keys(0, f64::INFINITY), (2, 2));
        assert_eq!(table.keys(0, f64::NEG_INFINITY), (0, 0));
        assert_eq!(table.keys(0, f64::NAN), (NAN_LEFT_KEY, NAN_RIGHT_KEY));
    }

    #[test]
    fn values_in_the_zero_window_act_as_zero() {
        let table = table(vec![vec![-1e-300, 1e-300]]);
        let zero = table.keys(0, 0.0);
        assert_eq!(table.keys(0, 1e-36), zero);
        assert_eq!(table.keys(0, -1e-36), zero);
        assert_eq!(table.keys(0, -0.0), zero);
        assert_ne!(table.keys(0, 1e-34), zero);
    }

    #[test]
    fn groups_rows_by_key_and_ignores_unsplit_columns() {
        // Column 1 has no splits, so only column 0's bins tell rows apart.
        let values = [0.0, 5.0, 0.5, 6.0, 3.0, 5.0, f64::NAN, 1.0, 2.0, 0.0];
        let (first, inverse) = distinct_rows(&values, vec![vec![1.0], vec![]]).unwrap();
        assert_eq!(first, vec![0, 2, 3]);
        assert_eq!(inverse, vec![0, 0, 1, 2, 1]);
    }

    #[test]
    fn rows_without_split_columns_form_one_group() {
        let (first, inverse) = distinct_rows(&[1.0, 2.0, 3.0], vec![vec![]]).unwrap();
        assert_eq!(first, vec![0]);
        assert_eq!(inverse, vec![0, 0, 0]);
    }

    #[test]
    fn too_many_thresholds_disable_grouping() {
        let many: Vec<f64> = (0..=MAX_KEY_THRESHOLDS).map(|t| t as f64).collect();
        assert!(distinct_rows(&[0.0], vec![many]).is_none());
    }
}
