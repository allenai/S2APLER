//! S2APLER's native extension: Rust pair featurization.
//!
//! The counter, string-similarity, thread-pool and build modules come from s2and's extension;
//! see THIRD_PARTY_NOTICES.md.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

mod counter;
mod paper_features;
mod rayon_pool;
mod text;

use counter::CounterData;
use paper_features::{pair_features, PaperRecord, NUM_FEATURES};
use rayon_pool::install_with_optional_rayon_pool;

// Parallel readers allocate on worker threads; with glibc malloc that slows
// later kernels through arena contention, and mimalloc avoids it.
// Cargo.toml explains the targets that keep the system allocator.
#[cfg(not(any(target_env = "musl", all(target_os = "linux", target_arch = "aarch64"))))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type CounterInput = (Vec<u64>, Vec<u32>);
/// Title tokens, numeral flags, numeral-preceding flags, special words, years, unspaced title.
type TitleInput = (Vec<u32>, Vec<bool>, Vec<bool>, Vec<u32>, Vec<u32>, String);

/// One paper as Python passes it; see `s2apler.featurizer.rust_paper_record` for each field.
/// PyO3 extracts tuples of at most 12 items, so related fields are grouped.
#[derive(FromPyObject)]
struct PaperInput(
    // Coauthor, venue, title-character and abstract-word n-gram counters.
    (CounterInput, CounterInput, CounterInput, CounterInput),
    // Author first-letter sets and the year above 1700.
    Vec<Vec<u32>>,
    Option<i64>,
    TitleInput,
    // Has abstract, has authors, has venue, PDF source, publisher source.
    (bool, bool, bool, bool, bool),
    Option<u32>,
);

fn counter_from_input((ids, counts): CounterInput) -> PyResult<Option<CounterData>> {
    if ids.len() != counts.len() {
        return Err(PyValueError::new_err(
            "counter ids and counts differ in length",
        ));
    }
    if ids.is_empty() {
        return Ok(None);
    }
    let mut entries: Vec<(u64, u32)> = ids.into_iter().zip(counts).collect();
    entries.sort_unstable_by_key(|entry| entry.0);
    if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(PyValueError::new_err("counter ids must be unique"));
    }
    // Sum in integers so the total matches Python's exactly.
    let sum = entries.iter().map(|entry| entry.1 as u64).sum::<u64>() as f64;
    let entries = entries
        .into_iter()
        .map(|(id, count)| (id, count as f32))
        .collect();
    Ok(Some(CounterData::new(entries, sum)))
}

fn record_from_input(input: PaperInput) -> PyResult<PaperRecord> {
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
    for letters in &mut author_first_letters {
        letters.sort_unstable();
        letters.dedup();
    }
    title_special_words.sort_unstable();
    title_special_words.dedup();
    Ok(PaperRecord {
        coauthor_ngrams: counter_from_input(coauthor)?,
        venue_ngrams: counter_from_input(venue)?,
        title_char_ngrams: counter_from_input(title_chars)?,
        abstract_word_ngrams: counter_from_input(abstract_words)?,
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
        let papers = papers
            .into_iter()
            .map(record_from_input)
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
        let left = left.as_slice()?;
        let right = right.as_slice()?;
        if left.len() != right.len() {
            return Err(PyValueError::new_err(
                "left and right must have the same length",
            ));
        }
        let paper_count = self.papers.len();
        if let Some(position) = left
            .iter()
            .chain(right)
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

/// Build metadata: the crate version and a digest of the native sources, for freshness checks.
#[pyfunction]
fn get_build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let build_info = PyDict::new(py);
    build_info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    build_info.set_item("source_digest", env!("S2APLER_RUST_SOURCE_DIGEST"))?;
    build_info.set_item("profile", option_env!("PROFILE").unwrap_or("unknown"))?;
    build_info.set_item("debug_assertions", cfg!(debug_assertions))?;
    build_info.set_item("num_features", NUM_FEATURES)?;
    Ok(build_info.unbind())
}

#[pymodule]
fn _s2apler_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_class::<PaperFeaturizer>()?;
    Ok(())
}
