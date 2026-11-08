//! Sparse n-gram counters and their Jaccard overlap. Copied from s2and's `CounterData`
//! (`lib.rs`) and `counter_jaccard_data` (`features.rs`).
//!
//! S2APLER changes: keys are dataset-local ids interned in Python, not FNV hashes, so keys
//! never collide. Counts and their total are exact `f64` integers (exact below 2^53), so any
//! count matches Python's integer arithmetic. s2and stores `f32` counts, which stop being exact
//! at 2^24; here the `f32` copy exists only for the AVX2 path, and only when it is exact.

#[derive(Clone)]
pub(crate) struct CounterData {
    // Interned n-gram ids, sorted ascending, and their counts at the same positions.
    // Binary search replaces HashMap lookup; separate arrays let intersections load keys
    // four at a time.
    keys: Box<[u64]>,
    values: Box<[f64]>,
    /// `values` as `f32`, for the AVX2 intersection; empty unless `simd` holds.
    simd_values: Box<[f32]>,
    sum: f64,
    /// Keys strictly increase, every value is an integer in [0, 2^24), and the
    /// sum is below 2^30, so an intersection's min-sum is an exact integer in
    /// any order.
    simd: bool,
}

/// Counts at or above 2^53 are not exact in `f64`.
pub(crate) const MAX_EXACT_COUNT: u64 = 1 << 53;

impl CounterData {
    /// Builds a counter from `(key, count)` entries sorted by unique key, with every count
    /// below `MAX_EXACT_COUNT` and a total below it too.
    pub(crate) fn new(entries: Vec<(u64, u64)>) -> Self {
        let sum = entries.iter().map(|entry| entry.1).sum::<u64>() as f64;
        let simd = sum < 1_073_741_824.0
            && entries.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && entries.iter().all(|entry| entry.1 < 16_777_216);
        let simd_values: Vec<f32> = if simd {
            entries.iter().map(|entry| entry.1 as f32).collect()
        } else {
            Vec::new()
        };
        let (keys, values): (Vec<u64>, Vec<f64>) = entries
            .into_iter()
            .map(|(key, count)| (key, count as f64))
            .unzip();
        Self {
            keys: keys.into_boxed_slice(),
            values: values.into_boxed_slice(),
            simd_values: simd_values.into_boxed_slice(),
            sum,
            simd,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// `(key, value)` pairs in key order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u64, &f64)> + '_ {
        self.keys.iter().zip(self.values.iter())
    }
}

/// Sum of `min(value_a, value_b)` over the keys two counters share, for
/// counters whose `simd` flag holds: four keys per side are compared all
/// against all per step, and integer counts sum exactly in any order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersection_min_sum_avx2(a: &CounterData, b: &CounterData) -> u64 {
    use std::arch::x86_64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut acc = _mm_setzero_si128();
    // Low 32-bit lanes of the four 64-bit compare lanes.
    let pack_lows = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    while i + 4 <= na && j + 4 <= nb {
        let ak = _mm256_loadu_si256(a.keys.as_ptr().add(i) as *const __m256i);
        let bk = _mm256_loadu_si256(b.keys.as_ptr().add(j) as *const __m256i);
        let av = _mm_cvttps_epi32(_mm_loadu_ps(a.simd_values.as_ptr().add(i)));
        let bv = _mm_cvttps_epi32(_mm_loadu_ps(b.simd_values.as_ptr().add(j)));
        macro_rules! rotation {
            ($keys:expr, $values:expr) => {{
                let equal = _mm256_cmpeq_epi64(ak, $keys);
                let equal32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(equal, pack_lows));
                acc = _mm_add_epi32(acc, _mm_and_si128(_mm_min_epi32(av, $values), equal32));
            }};
        }
        rotation!(bk, bv);
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b00_11_10_01),
            _mm_shuffle_epi32(bv, 0b00_11_10_01)
        );
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b01_00_11_10),
            _mm_shuffle_epi32(bv, 0b01_00_11_10)
        );
        rotation!(
            _mm256_permute4x64_epi64(bk, 0b10_01_00_11),
            _mm_shuffle_epi32(bv, 0b10_01_00_11)
        );
        let a_last = *a.keys.get_unchecked(i + 3);
        let b_last = *b.keys.get_unchecked(j + 3);
        i += 4 * usize::from(a_last <= b_last);
        j += 4 * usize::from(b_last <= a_last);
    }
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    let mut total: u64 = lanes.iter().map(|lane| *lane as u64).sum();
    while i < na && j < nb {
        let (ka, kb) = (*a.keys.get_unchecked(i), *b.keys.get_unchecked(j));
        if ka == kb {
            total += a
                .simd_values
                .get_unchecked(i)
                .min(*b.simd_values.get_unchecked(j)) as u64;
        }
        i += usize::from(ka <= kb);
        j += usize::from(kb <= ka);
    }
    total
}

pub(crate) fn counter_jaccard_data(
    counter1: &Option<CounterData>,
    counter2: &Option<CounterData>,
    denom_max: f64,
) -> f64 {
    let (Some(c1), Some(c2)) = (counter1.as_ref(), counter2.as_ref()) else {
        return f64::NAN;
    };
    if c1.is_empty() || c2.is_empty() {
        return f64::NAN;
    }
    let (small, large) = if c1.len() <= c2.len() {
        (c1, c2)
    } else {
        (c2, c1)
    };
    let merge = large.len() < small.len().saturating_mul(4);
    #[cfg(target_arch = "x86_64")]
    let simd_sum = (merge && c1.simd && c2.simd && std::arch::is_x86_feature_detected!("avx2"))
        // SAFETY: AVX2 was detected in the condition.
        .then(|| unsafe { intersection_min_sum_avx2(c1, c2) });
    #[cfg(not(target_arch = "x86_64"))]
    let simd_sum: Option<u64> = None;
    let mut intersection = 0.0f64;
    if let Some(sum) = simd_sum {
        // Integer counts: the exact sum equals the merge's f64 sum.
        intersection = sum as f64;
    } else if merge {
        let mut small_idx = 0usize;
        let mut large_idx = 0usize;
        while small_idx < small.len() && large_idx < large.len() {
            let (small_hash, small_value) = (small.keys[small_idx], small.values[small_idx]);
            let (large_hash, large_value) = (large.keys[large_idx], large.values[large_idx]);
            if small_hash == large_hash {
                intersection += small_value.min(large_value);
            }
            // Advance without a hash-dependent branch; equal hashes move both,
            // so the sum keeps its order.
            small_idx += usize::from(small_hash <= large_hash);
            large_idx += usize::from(large_hash <= small_hash);
        }
    } else {
        for (h, v1) in small.iter() {
            if let Ok(idx) = large.keys.binary_search(h) {
                intersection += (*v1).min(large.values[idx]);
            }
        }
    }
    let union = c1.sum + c2.sum - intersection;
    if union == 0.0 {
        return f64::NAN;
    }
    let denom = if denom_max.is_infinite() {
        union
    } else {
        union.min(denom_max)
    };
    let score = intersection / denom;
    if score > 1.0 {
        1.0
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's `counter_jaccard`, computed in exact integers.
    fn reference(a: &[(u64, u64)], b: &[(u64, u64)], denom_max: f64) -> f64 {
        let intersection: u64 = a
            .iter()
            .filter_map(|(key, count)| {
                b.iter()
                    .find(|(other, _)| other == key)
                    .map(|(_, other)| (*count).min(*other))
            })
            .sum();
        let union =
            a.iter().map(|e| e.1).sum::<u64>() + b.iter().map(|e| e.1).sum::<u64>() - intersection;
        let denom = if denom_max.is_infinite() {
            union as f64
        } else {
            (union as f64).min(denom_max)
        };
        (intersection as f64 / denom).min(1.0)
    }

    fn random_counter(
        next: &mut impl FnMut(u64) -> u64,
        len: usize,
        max_count: u64,
    ) -> Vec<(u64, u64)> {
        let mut keys: Vec<u64> = (0..len).map(|_| next(len as u64 * 3)).collect();
        keys.sort_unstable();
        keys.dedup();
        keys.into_iter()
            .map(|key| (key, 1 + next(max_count)))
            .collect()
    }

    #[test]
    fn every_intersection_path_matches_exact_integer_arithmetic() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound.max(1)
        };
        // Small counts take the AVX2 path where available, huge counts the scalar merge,
        // and lopsided sizes the binary search.
        for (len_a, len_b, max_count) in [
            (40, 40, 5),
            (40, 40, 1 << 40),
            (5, 200, 5),
            (5, 200, 1 << 40),
            (1, 1, (1 << 24) + 1),
        ] {
            for _ in 0..200 {
                let a = random_counter(&mut next, len_a, max_count);
                let b = random_counter(&mut next, len_b, max_count);
                for denom_max in [f64::INFINITY, 5000.0] {
                    let expected = reference(&a, &b, denom_max);
                    let actual = counter_jaccard_data(
                        &Some(CounterData::new(a.clone())),
                        &Some(CounterData::new(b.clone())),
                        denom_max,
                    );
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{a:?} {b:?} {denom_max}"
                    );
                }
            }
        }
    }

    #[test]
    fn counts_beyond_f32_precision_stay_exact() {
        let counter = || {
            Some(CounterData::new(vec![
                (1, (1 << 24) + 1),
                (2, 3),
                (3, 5),
                (4, 7),
            ]))
        };
        assert_eq!(
            counter_jaccard_data(&counter(), &counter(), f64::INFINITY),
            1.0
        );
    }
}
