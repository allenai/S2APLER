//! S2APLER's native extension: Rust pair featurization, and grouping of rows that a LightGBM
//! model scores identically.
//!
//! The counter, string-similarity, thread-pool and build modules come from s2and's extension;
//! see THIRD_PARTY_NOTICES.md.

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyDict;
use rayon::prelude::*;

mod counter;
mod distinct_rows;
mod ngrams;
mod paper_features;
mod rayon_pool;
mod text;
mod text_fast;

use counter::{CounterData, MAX_EXACT_COUNT};
use paper_features::{pair_features, PaperRecord, NUM_FEATURES};
use rayon_pool::install_with_optional_rayon_pool;

// Parallel readers allocate on worker threads; with glibc malloc that slows
// later kernels through arena contention, and mimalloc avoids it.
// Cargo.toml explains the targets that keep the system allocator.
#[cfg(not(any(
    target_env = "musl",
    target_os = "macos",
    all(target_os = "linux", target_arch = "aarch64")
)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// An n-gram counter: interned ids and counts from Python, or the text for Rust to count.
///
/// The text is UTF-32-LE code points of the string Python's n-gram functions would read.
#[derive(FromPyObject)]
enum CounterInput {
    Counts(Vec<u64>, Vec<u64>),
    Text(PyBackedBytes),
}
/// Title tokens, numeral flags, numeral-preceding flags, special words, years, unspaced title.
type TitleInput = (Vec<u32>, Vec<bool>, Vec<bool>, Vec<u32>, Vec<u32>, String);

/// One paper as Python passes it; see `s2apler.featurizer.rust_paper_record` for each field.
/// PyO3 extracts tuples of at most 12 items, so related fields are grouped.
#[derive(FromPyObject)]
struct PaperInput(
    // Coauthor, venue, title-character and abstract-word n-gram counters.
    (CounterInput, CounterInput, CounterInput, CounterInput),
    // Author first-letter sets and the year above 1700, as a float like Python's `diff` uses.
    Vec<Vec<u32>>,
    Option<f64>,
    TitleInput,
    // Has abstract, has authors, has venue, PDF source, publisher source.
    (bool, bool, bool, bool, bool),
    Option<u32>,
);

fn counter_from_counts(ids: Vec<u64>, counts: Vec<u64>) -> PyResult<Option<CounterData>> {
    if ids.len() != counts.len() {
        return Err(PyValueError::new_err(
            "counter ids and counts differ in length",
        ));
    }
    if ids.is_empty() {
        return Ok(None);
    }
    let mut entries: Vec<(u64, u64)> = ids.into_iter().zip(counts).collect();
    entries.sort_unstable_by_key(|entry| entry.0);
    if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(PyValueError::new_err("counter ids must be unique"));
    }
    // Python counts are exact integers; refuse any total that f64 could not hold exactly.
    let total = entries
        .iter()
        .try_fold(0u64, |total, entry| total.checked_add(entry.1));
    if !matches!(total, Some(total) if total < MAX_EXACT_COUNT) {
        return Err(PyValueError::new_err(
            "counter counts must total less than 2**53",
        ));
    }
    Ok(Some(CounterData::new(entries)))
}

/// Each counter's text as code points, for the counters Python passed as text.
type CounterTexts = [Option<Vec<u32>>; 4];

fn counter_texts(input: &PaperInput) -> PyResult<CounterTexts> {
    let (coauthor, venue, title_chars, abstract_words) = &input.0;
    let decode = |counter: &CounterInput| match counter {
        CounterInput::Counts(..) => Ok(None),
        CounterInput::Text(bytes) => ngrams::code_points(bytes)
            .map(Some)
            .ok_or_else(|| PyValueError::new_err("counter text must be UTF-32-LE bytes")),
    };
    Ok([
        decode(coauthor)?,
        decode(venue)?,
        decode(title_chars)?,
        decode(abstract_words)?,
    ])
}

/// Builds one counter: from Python's counts, or by counting `text` with `count`.
fn counter_from_input<'a>(
    input: CounterInput,
    text: &'a Option<Vec<u32>>,
    count: impl FnOnce(&'a [u32]) -> Option<CounterData>,
) -> PyResult<Option<CounterData>> {
    match (input, text) {
        (CounterInput::Counts(ids, counts), _) => counter_from_counts(ids, counts),
        (CounterInput::Text(_), Some(text)) => Ok(count(text)),
        (CounterInput::Text(_), None) => unreachable!("counter_texts decodes every text counter"),
    }
}

fn record_from_input<'a>(
    input: PaperInput,
    texts: &'a CounterTexts,
    interner: &mut ngrams::GramInterner<'a>,
) -> PyResult<PaperRecord> {
    let PaperInput(
        (coauthor, venue, title_chars, abstract_words),
        mut author_first_letters,
        year,
        (
            title_tokens,
            title_token_is_numeral,
            title_token_precedes_numeral,
            mut title_special_words,
            title_years,
            title_without_spaces,
        ),
        (has_abstract, has_authors, has_venue, is_pdf_source, is_publisher_source),
        source,
    ) = input;
    if title_token_is_numeral.len() != title_tokens.len()
        || title_token_precedes_numeral.len() != title_tokens.len()
    {
        return Err(PyValueError::new_err(
            "title token flags must match the tokens",
        ));
    }
    // Title features compare bytes, which matches Python only on ASCII. PDData's preprocessing
    // makes every title ASCII, so anything else means the papers skipped it.
    if !title_without_spaces.is_ascii() {
        return Err(PyValueError::new_err(
            "titles must be ASCII; preprocess papers with PDData",
        ));
    }
    for letters in &mut author_first_letters {
        letters.sort_unstable();
        letters.dedup();
    }
    title_special_words.sort_unstable();
    title_special_words.dedup();
    // The n-gram lengths of `preprocess_paper_1`'s `get_text_ngrams` calls.
    Ok(PaperRecord {
        coauthor_ngrams: counter_from_input(coauthor, &texts[0], |text| {
            ngrams::char_ngrams(text, &[1, 2, 3, 4], interner)
        })?,
        venue_ngrams: counter_from_input(venue, &texts[1], |text| {
            ngrams::char_ngrams(text, &[2, 3, 4], interner)
        })?,
        title_char_ngrams: counter_from_input(title_chars, &texts[2], |text| {
            ngrams::char_ngrams(text, &[3, 4], interner)
        })?,
        abstract_word_ngrams: counter_from_input(abstract_words, &texts[3], |text| {
            ngrams::word_ngrams(text, interner)
        })?,
        author_first_letters,
        year,
        title_tokens,
        title_token_is_numeral,
        title_token_precedes_numeral,
        title_special_words,
        title_years,
        title_without_spaces,
        has_abstract,
        has_authors,
        has_venue,
        is_pdf_source,
        is_publisher_source,
        source,
    })
}

/// Computes S2APLER's pair features for papers preprocessed in Python.
#[pyclass(frozen, module = "s2apler._s2apler_rust")]
struct PaperFeaturizer {
    papers: Vec<PaperRecord>,
}

#[pymethods]
impl PaperFeaturizer {
    #[new]
    fn new(papers: Vec<PaperInput>) -> PyResult<Self> {
        // Decode every text first: the interner borrows n-grams from them.
        let texts = papers
            .iter()
            .map(counter_texts)
            .collect::<PyResult<Vec<_>>>()?;
        let mut interner = ngrams::GramInterner::default();
        let papers = papers
            .into_iter()
            .zip(&texts)
            .map(|(paper, texts)| record_from_input(paper, texts, &mut interner))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { papers })
    }

    fn __len__(&self) -> usize {
        self.papers.len()
    }

    /// Features for pairs `(left[k], right[k])` of paper positions, as an (n, 18) float64 array.
    #[pyo3(signature = (left, right, num_threads=None))]
    fn featurize<'py>(
        &self,
        py: Python<'py>,
        left: PyReadonlyArray1<'py, u32>,
        right: PyReadonlyArray1<'py, u32>,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Copy the positions while holding the GIL: another thread could change the NumPy
        // arrays once it is released.
        let left = left.as_slice()?.to_vec();
        let right = right.as_slice()?.to_vec();
        if left.len() != right.len() {
            return Err(PyValueError::new_err(
                "left and right must have the same length",
            ));
        }
        let paper_count = self.papers.len();
        if let Some(position) = left
            .iter()
            .chain(&right)
            .find(|position| **position as usize >= paper_count)
        {
            return Err(PyValueError::new_err(format!(
                "paper position {position} is out of range for {paper_count} papers"
            )));
        }
        let pair_count = left.len();
        let papers = &self.papers;
        let features = py.allow_threads(|| {
            let mut out = vec![0.0f64; pair_count * NUM_FEATURES];
            install_with_optional_rayon_pool(Some(num_threads.unwrap_or(1).max(1)), || {
                out.par_chunks_mut(NUM_FEATURES)
                    .enumerate()
                    .for_each(|(k, row)| {
                        pair_features(&papers[left[k] as usize], &papers[right[k] as usize], row);
                    });
            });
            out
        });
        let array = Array2::from_shape_vec((pair_count, NUM_FEATURES), features)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

/// Groups rows that a numerical LightGBM model scores identically; see `distinct_rows`.
///
/// `thresholds` holds one list per column of `rows`, empty for columns that no split uses. Returns the first row of each group and each row's group, as uint32 arrays, or
/// `None` when a column has too many thresholds.
#[pyfunction]
#[pyo3(signature = (rows, thresholds, num_threads=None))]
#[allow(clippy::type_complexity)]
fn distinct_bin_rows<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray2<'py, f64>,
    thresholds: Vec<Vec<f64>>,
    num_threads: Option<usize>,
) -> PyResult<Option<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)>> {
    let shape = rows.shape();
    if shape[1] != thresholds.len() {
        return Err(PyValueError::new_err(format!(
            "expected one threshold list per column: {} columns, {} lists",
            shape[1],
            thresholds.len()
        )));
    }
    if shape[0] > u32::MAX as usize {
        return Err(PyValueError::new_err("too many rows for uint32 indices"));
    }
    if thresholds
        .iter()
        .flatten()
        .any(|threshold| threshold.is_nan())
    {
        return Err(PyValueError::new_err("thresholds must not be NaN"));
    }
    // Keys are computed with the GIL held, so no Python thread can change the rows meanwhile;
    // copying them first would cost more than the key pass saves.
    let values = rows.as_slice()?;
    let grouped = install_with_optional_rayon_pool(Some(num_threads.unwrap_or(1).max(1)), || {
        distinct_rows::distinct_rows(values, thresholds)
    });
    Ok(grouped.map(|(first, inverse)| (first.into_pyarray(py), inverse.into_pyarray(py))))
}

/// Build metadata: the crate version and a digest of the native sources, for freshness checks.
#[pyfunction]
fn get_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let build_info = PyDict::new(py);
    build_info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    build_info.set_item("source_digest", env!("S2APLER_RUST_SOURCE_DIGEST"))?;
    build_info.set_item("debug_assertions", cfg!(debug_assertions))?;
    build_info.set_item("num_features", NUM_FEATURES)?;
    Ok(build_info.unbind())
}

#[pymodule]
fn _s2apler_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(distinct_bin_rows, m)?)?;
    m.add_class::<PaperFeaturizer>()?;
    Ok(())
}
