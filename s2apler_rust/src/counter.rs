//! Sparse n-gram counters and their Jaccard overlap. Copied from s2and's `CounterData`
//! (`lib.rs`) and `counter_jaccard_data` (`features.rs`).
//!
//! S2APLER changes: keys are dataset-local ids interned in Python, not FNV hashes, so keys
//! never collide. The counter total is an exact `f64`, not `f32`: S2APLER's coauthor and
//! abstract counters can exceed 2^24, where an `f32` total stops matching Python's integer sum.

#[derive(Clone)]
pub(crate) struct CounterData {
    // Interned n-gram ids, sorted ascending, and their f32 counts at the same positions.
    // Binary search replaces HashMap lookup; separate arrays let intersections load keys
    // four at a time.
    keys: Box<[u64]>,
    values: Box<[f32]>,
    sum: f64,
    /// Keys strictly increase, every value is an integer in [0, 2^24), and the
    /// sum is below 2^30, so an intersection's min-sum is an exact integer in
    /// any order.
    simd: bool,
}

impl CounterData {
    pub(crate) fn new(entries: Vec<(u64, f32)>, sum: f64) -> Self {
        let simd = sum < 1_073_741_824.0
            && entries.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && entries
                .iter()
                .all(|entry| entry.1 >= 0.0 && entry.1 < 16_777_216.0 && entry.1.fract() == 0.0);
        let (keys, values): (Vec<u64>, Vec<f32>) = entries.into_iter().unzip();
        Self {
            keys: keys.into_boxed_slice(),
            values: values.into_boxed_slice(),
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
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u64, &f32)> + '_ {
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
        let av = _mm_cvttps_epi32(_mm_loadu_ps(a.values.as_ptr().add(i)));
        let bv = _mm_cvttps_epi32(_mm_loadu_ps(b.values.as_ptr().add(j)));
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
            total += a.values.get_unchecked(i).min(*b.values.get_unchecked(j)) as u64;
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
                intersection += small_value.min(large_value) as f64;
            }
            // Advance without a hash-dependent branch; equal hashes move both,
            // so the sum keeps its order.
            small_idx += usize::from(small_hash <= large_hash);
            large_idx += usize::from(large_hash <= small_hash);
        }
    } else {
        for (h, v1) in small.iter() {
            if let Ok(idx) = large.keys.binary_search(h) {
                intersection += (*v1).min(large.values[idx]) as f64;
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
