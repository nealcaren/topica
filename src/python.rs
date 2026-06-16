//! PyO3 bindings: a Pythonic `LDA` + `Corpus` surface over the SparseLDA core.
//!
//! The compiled module is exposed to Python as `topica._topica`
//! (see pyproject.toml). A thin pure-Python package re-exports it.
//!
//! Design notes:
//!  * The heavy Gibbs sampling runs inside `Python::allow_threads`, so other
//!    Python threads keep running during training.
//!  * `LDA.fit` ports the averaging loop from `src/bin/train.rs` (the only
//!    pipeline logic that lived in the binary rather than the library), so the
//!    Python results match the `train` CLI exactly for a given seed.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, ToPyArray};

use crate::detm;
use crate::dmr;
use crate::dtm;
use crate::gsdmm;
use crate::hdp;
use crate::infoctm;
use crate::keyatm;
use crate::seeded;
use crate::hlda;
use crate::lsa;
use crate::nmf;
use crate::pa;
use crate::prodlda;
use crate::pt;
use crate::slda;
use crate::top2vec;
use crate::bertopic;
use crate::etm;
use crate::etm_vae;
use crate::fastopic;
use crate::labeled;
use crate::sage;
use crate::variational::LogisticNormalModel;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use regex::Regex;

use crate::corpus::{self, InputFormat, LoadOptions};
use crate::model::TopicModel;
use crate::{coherence as coh, ctm, cvb0, lightlda, optimize, output, sampler, spectral, sts, warplda};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn io_err(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// Validate a count argument given as a signed Python int, so negatives raise a
/// clean `ValueError` instead of PyO3's raw "can't convert negative int to
/// unsigned" `OverflowError`. Accepting `i64` at the signature keeps the boundary
/// from rejecting negatives before our own message can run.
fn require_count(value: i64, min: i64, name: &str) -> PyResult<usize> {
    if value < min {
        return Err(PyValueError::new_err(format!(
            "{name} must be >= {min}, got {value}"
        )));
    }
    Ok(value as usize)
}

/// Guard that every element of a 2-D feature/covariate/embedding matrix is
/// finite. Called right after `parse_features` returns and the row-count check
/// passes, so the error names the parameter exactly as the user passed it.
fn check_all_finite_2d(name: &str, rows: &[Vec<f64>]) -> PyResult<()> {
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite values (NaN or inf) at row {i}, col {j}"
                )));
            }
        }
    }
    Ok(())
}

/// Guard that every element of a 2-D ndarray view is finite. Used when data
/// enters via `PyReadonlyArray2` rather than through `parse_features`.
fn check_all_finite_arr2(name: &str, arr: &numpy::ndarray::ArrayView2<f64>) -> PyResult<()> {
    for ((i, j), &v) in arr.indexed_iter() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite values (NaN or inf) at row {i}, col {j}"
            )));
        }
    }
    Ok(())
}

/// Guard that every element of a 1-D numeric sequence (e.g. timestamps) is
/// finite.
fn check_all_finite_1d(name: &str, vals: &[f64]) -> PyResult<()> {
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite values (NaN or inf) at index {i}"
            )));
        }
    }
    Ok(())
}

// `from_py_with` hooks for count constructor arguments. They take the int as a
// signed `i64` so a negative value yields a clean `ValueError` here rather than
// PyO3's raw `OverflowError`. Per-model minimums above 1 (e.g. CTM/STM need >= 2)
// stay enforced by the existing guards inside each constructor body.
fn py_num_topics(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_topics")
}
fn py_num_pseudo(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_pseudo")
}
fn py_num_super(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_super")
}
fn py_num_sub(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "num_sub")
}
fn py_depth(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    require_count(ob.extract()?, 1, "depth")
}
fn py_num_topics_opt(ob: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    if ob.is_none() {
        return Ok(None);
    }
    Ok(Some(require_count(ob.extract()?, 1, "num_topics")?))
}

// ---------------------------------------------------------------------------
// Model serialization (save / load)
// ---------------------------------------------------------------------------

/// Serializable form of an ndarray `Array2` (shape + row-major data).
#[derive(serde::Serialize, serde::Deserialize)]
struct Arr2 {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
/// Serializable form of an ndarray `Array3` (f64).
#[derive(serde::Serialize, serde::Deserialize)]
struct Arr3 {
    d0: usize,
    d1: usize,
    d2: usize,
    data: Vec<f64>,
}
/// Serializable form of an ndarray `Array3<f32>` (used for theta_draws).
#[derive(serde::Serialize, serde::Deserialize)]
struct Arr3f32 {
    d0: usize,
    d1: usize,
    d2: usize,
    data: Vec<f32>,
}

fn arr2_opt(a: &Option<Array2<f64>>) -> Option<Arr2> {
    a.as_ref().map(|m| Arr2 { rows: m.nrows(), cols: m.ncols(), data: m.iter().copied().collect() })
}
fn arr2_back(s: Option<Arr2>) -> Option<Array2<f64>> {
    s.map(|a| Array2::from_shape_vec((a.rows, a.cols), a.data).unwrap())
}
fn arr3_opt(a: &Option<Array3<f64>>) -> Option<Arr3> {
    a.as_ref().map(|m| {
        let d = m.dim();
        Arr3 { d0: d.0, d1: d.1, d2: d.2, data: m.iter().copied().collect() }
    })
}
fn arr3_back(s: Option<Arr3>) -> Option<Array3<f64>> {
    s.map(|a| Array3::from_shape_vec((a.d0, a.d1, a.d2), a.data).unwrap())
}

/// Run `f` on a rayon pool of `num_threads` workers, or on the global pool (all
/// cores) when `num_threads` is `None`/0. The variational fits are deterministic
/// regardless of worker count, so this controls only resource use, not output.
/// Call inside `py.allow_threads`.
fn run_with_threads<T: Send, F: FnOnce() -> T + Send>(num_threads: Option<usize>, f: F) -> T {
    match num_threads {
        Some(n) if n >= 1 => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(pool) => pool.install(f),
            Err(_) => f(),
        },
        _ => f(),
    }
}
fn arr3f32_opt(a: &Option<Array3<f32>>) -> Option<Arr3f32> {
    a.as_ref().map(|m| {
        let d = m.dim();
        Arr3f32 { d0: d.0, d1: d.1, d2: d.2, data: m.iter().copied().collect() }
    })
}
fn arr3f32_back(s: Option<Arr3f32>) -> Option<Array3<f32>> {
    s.map(|a| Array3::from_shape_vec((a.d0, a.d1, a.d2), a.data).unwrap())
}
fn arr1_opt(a: &Option<Array1<f64>>) -> Option<Vec<f64>> {
    a.as_ref().map(|m| m.to_vec())
}
fn arr1_back(s: Option<Vec<f64>>) -> Option<Array1<f64>> {
    s.map(Array1::from)
}

// ---------------------------------------------------------------------------
// Save-file header: magic + format version + model tag
//
// Layout (8 bytes prepended before the bincode payload):
//   bytes 0..6  : b"TOPICA"   (magic, 6 bytes)
//   byte  6     : format version u8 = 1
//   byte  7     : model tag u8 (see MODEL_TAG_* constants below)
//   bytes 8..   : bincode payload
//
// Header encode/decode logic lives in src/saveformat.rs (always compiled, so
// it can have Rust unit tests without the `python` feature gate or libpython).
// Old (headerless) files produce a clear "not a topica model file" error
// rather than a bincode panic.
// ---------------------------------------------------------------------------

// One tag per concrete model type that calls write_state / read_state.
const MODEL_TAG_LDA:       u8 = 1;
const MODEL_TAG_DMR:       u8 = 2;
const MODEL_TAG_LABELED:   u8 = 3;
const MODEL_TAG_SAGE:      u8 = 4;
const MODEL_TAG_CTM:       u8 = 5;
const MODEL_TAG_STM:       u8 = 6;
const MODEL_TAG_STS:       u8 = 7;
const MODEL_TAG_HDP:       u8 = 8;
const MODEL_TAG_DTM:       u8 = 9;
const MODEL_TAG_SLDA:      u8 = 10;
const MODEL_TAG_PT:        u8 = 11;
const MODEL_TAG_GSDMM:     u8 = 12;
const MODEL_TAG_SEEDED:    u8 = 13;
const MODEL_TAG_TOP2VEC:   u8 = 14;
const MODEL_TAG_BERTOPIC:  u8 = 15;
const MODEL_TAG_ETM:       u8 = 16;
const MODEL_TAG_PRODLDA:   u8 = 17;
const MODEL_TAG_FASTOPIC:  u8 = 18;
const MODEL_TAG_KEYATM:    u8 = 19;
const MODEL_TAG_PA:        u8 = 20;
const MODEL_TAG_HLDA:      u8 = 21;
const MODEL_TAG_NMF:       u8 = 22;
const MODEL_TAG_LSA:       u8 = 23;
const MODEL_TAG_COMBINEDTM: u8 = 24;
const MODEL_TAG_ZEROSHOTTM: u8 = 25;
const MODEL_TAG_DETM:      u8 = 26;
const MODEL_TAG_ECTM:      u8 = 27;

fn model_tag_name(tag: u8) -> &'static str {
    match tag {
        MODEL_TAG_LDA      => "LDA",
        MODEL_TAG_DMR      => "DMR",
        MODEL_TAG_LABELED  => "LabeledLDA",
        MODEL_TAG_SAGE     => "SAGE",
        MODEL_TAG_CTM      => "CTM",
        MODEL_TAG_STM      => "STM",
        MODEL_TAG_STS      => "STS",
        MODEL_TAG_HDP      => "HDP",
        MODEL_TAG_DTM      => "DTM",
        MODEL_TAG_SLDA     => "SupervisedLDA",
        MODEL_TAG_PT       => "PT",
        MODEL_TAG_GSDMM    => "GSDMM",
        MODEL_TAG_SEEDED   => "SeededLDA",
        MODEL_TAG_TOP2VEC  => "Top2Vec",
        MODEL_TAG_BERTOPIC => "BERTopic",
        MODEL_TAG_ETM      => "ETM",
        MODEL_TAG_PRODLDA  => "ProdLDA",
        MODEL_TAG_FASTOPIC => "FASTopic",
        MODEL_TAG_KEYATM   => "KeyATM",
        MODEL_TAG_PA       => "PA",
        MODEL_TAG_HLDA     => "HLDA",
        MODEL_TAG_NMF      => "NMF",
        MODEL_TAG_LSA      => "LSA",
        MODEL_TAG_COMBINEDTM => "CombinedTM",
        MODEL_TAG_ZEROSHOTTM => "ZeroShotTM",
        MODEL_TAG_DETM     => "DETM",
        MODEL_TAG_ECTM     => "ECTM",
        _                  => "unknown",
    }
}

fn write_state<S: serde::Serialize>(path: &str, model_tag: u8, state: &S) -> PyResult<()> {
    let buf = crate::saveformat::encode_state(model_tag, state)
        .map_err(PyValueError::new_err)?;
    std::fs::write(path, buf).map_err(io_err)
}
fn read_state<S: serde::de::DeserializeOwned>(path: &str, expected_tag: u8) -> PyResult<S> {
    let bytes = std::fs::read(path).map_err(io_err)?;
    crate::saveformat::decode_state(&bytes, expected_tag, model_tag_name)
        .map_err(PyValueError::new_err)
}

// Per-model serializable snapshots (ndarray fields stored as Arr2/Arr3/Vec).
#[derive(serde::Serialize, serde::Deserialize)]
struct LdaState {
    num_topics: usize, alpha_sum: Option<f64>, beta: f64, optimize_interval: usize,
    burn_in: usize, seed: u64, num_threads: usize, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, model: Option<TopicModel>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] use_symmetric_alpha: bool,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
    #[serde(default)] init_spectral: bool,
    // Sampler backend flags (persisted so a reloaded model is behaviorally identical).
    #[serde(default)] light: bool,
    #[serde(default)] warp: bool,
    #[serde(default)] cvb0: bool,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DmrState {
    num_topics: usize, beta: f64, optimize_interval: usize, burn_in: usize, seed: u64,
    prior_variance: f64, lbfgs_iters: usize, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, feature_effects: Option<Arr2>,
    feature_names: Vec<String>, corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct LabeledState {
    alpha: f64, beta: f64, seed: u64, fitted: bool, num_topics: usize,
    phi: Option<Arr2>, theta: Option<Arr2>, label_vocab: Vec<String>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SageState {
    num_topics: usize, alpha: f64, prior_variance: f64, optimize_interval: usize,
    burn_in: usize, seed: u64, lbfgs_iters: usize, fitted: bool, num_groups: usize,
    beta: Vec<Vec<f64>>, theta: Option<Arr2>, group_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
/// serde default for the bound of a model saved before convergence tracking
/// existed: NaN signals "unknown", distinct from a real bound of 0.
fn nan() -> f64 {
    f64::NAN
}
/// serde default for the variational-covariance mode on models saved before the
/// `variational` field existed: "laplace" (the original full-covariance E-step).
fn default_variational() -> String {
    "laplace".to_string()
}
#[derive(serde::Serialize, serde::Deserialize)]
struct CtmState {
    num_topics: usize, sigma_shrink: f64, seed: u64, init_spectral: bool, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>, corr: Option<Arr2>,
    eta_mean: Option<Arr2>, eta_cov: Option<Arr3>,
    #[serde(default)] mu: Vec<f64>, #[serde(default)] sigma: Vec<f64>,
    corpus: Option<corpus::Corpus>,
    #[serde(default = "nan")] bound: f64,
    #[serde(default)] bound_history: Vec<f64>,
    #[serde(default)] converged: bool,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default = "default_variational")] variational: String,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct StmState {
    num_topics: usize, sigma_shrink: f64, seed: u64, init_spectral: bool, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>, corr: Option<Arr2>,
    eta_mean: Option<Arr2>, eta_cov: Option<Arr3>, gamma: Option<Arr2>,
    feature_names: Vec<String>, content_beta: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(default)] mu: Vec<f64>, #[serde(default)] sigma: Vec<f64>,
    group_names: Vec<String>, corpus: Option<corpus::Corpus>,
    #[serde(default = "nan")] bound: f64,
    #[serde(default)] bound_history: Vec<f64>,
    #[serde(default)] converged: bool,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default = "default_variational")] variational: String,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct EctmState {
    num_topics: usize, sigma_shrink: f64, seed: u64, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>,
    eta_mean: Option<Arr2>, eta_cov: Option<Arr3>, gamma: Option<Arr2>,
    feature_names: Vec<String>,
    content_beta: Vec<Vec<Vec<f64>>>,
    num_groups: usize, num_periods: usize,
    group_names: Vec<String>, period_names: Vec<String>,
    #[serde(default)] mu: Vec<f64>, #[serde(default)] sigma: Vec<f64>,
    corpus: Option<corpus::Corpus>,
    #[serde(default = "nan")] bound: f64,
    #[serde(default)] bound_history: Vec<f64>,
    #[serde(default)] converged: bool,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default = "default_variational")] variational: String,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct StsState {
    num_topics: usize, seed: u64, init_spectral: bool, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>, sentiment: Option<Arr2>,
    gamma: Option<Arr2>, eta_mean: Option<Arr2>, eta_cov: Option<Arr3>,
    feature_names: Vec<String>,
    kappa_t: Vec<Vec<f64>>, kappa_s: Vec<Vec<f64>>, mv: Vec<f64>, sigma: Vec<f64>,
    corpus: Option<corpus::Corpus>,
    #[serde(default = "nan")] bound: f64,
    #[serde(default)] bound_history: Vec<f64>,
    #[serde(default)] converged: bool,
    #[serde(default)] topic_names: Vec<String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct HdpState {
    alpha: f64, gamma: f64, eta: f64, seed: u64, resample_conc: bool, fitted: bool,
    num_topics: usize, learned_alpha: f64, learned_gamma: f64,
    beta: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
    #[serde(default)] trace: Vec<(usize, usize, f64, f64, f64)>,
    #[serde(default)] topic_names: Vec<String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DtmState {
    num_topics: usize, alpha: f64, chain_variance: f64, obs_variance: f64, seed: u64,
    fitted: bool, num_times: usize, bound: f64,
    topic_words: Option<Vec<Vec<Vec<f64>>>>, corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SldaState {
    num_topics: usize, alpha: f64, seed: u64, fitted: bool, sigma2: f64,
    eta: Option<Vec<f64>>, beta: Option<Arr2>, theta: Option<Arr2>,
    log_beta: Option<Vec<Vec<f64>>>, corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct PtState {
    num_topics: usize, num_pseudo: usize, alpha: f64, beta: f64, seed: u64, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct GsdmmState {
    k_max: usize, alpha: f64, beta: f64, seed: u64, fitted: bool, num_used: usize,
    phi: Option<Arr2>, theta: Option<Arr2>, doc_cluster: Vec<usize>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] trace: Vec<(usize, usize, f64)>,
    #[serde(default)] topic_names: Vec<String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SeededState {
    num_topics: usize, alpha: f64, beta: f64, weight: f64, seed: u64, fitted: bool,
    topic_names: Vec<String>, phi: Option<Arr2>, theta: Option<Arr2>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
    // Seed metadata: persisted so load() restores the model faithfully.
    // seed_names / seed_words allow re-fit without re-supplying the keyword dict;
    // residual is the count of unseeded fallback topics.
    #[serde(default)] seed_names: Vec<String>,
    #[serde(default)] seed_words: Vec<Vec<String>>,
    #[serde(default)] residual: usize,
    // Sampler backend flags.
    #[serde(default)] warp: bool,
    #[serde(default)] cvb0: bool,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyAtmState {
    num_topics: usize, alpha: f64, beta: f64, beta_keyword: f64, gamma1: f64, gamma2: f64,
    seed: u64, fitted: bool, topic_names: Vec<String>, keyword_rate: Vec<f64>,
    phi: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64, f64)>,
    #[serde(default)] converged: bool,
    #[serde(default)] alpha_history: Vec<(usize, Vec<f64>)>,
    #[serde(default)] pi_history: Vec<(usize, Vec<f64>)>,
    #[serde(default)] alpha_vec: Option<Vec<f64>>,
    #[serde(default = "default_num_threads")] num_threads: usize,
    // Thinned MCMC theta draws (num_draws, num_docs, num_topics), f32.
    #[serde(default)] theta_draws: Option<Arr3f32>,
}
fn default_num_threads() -> usize { 1 }
#[derive(serde::Serialize, serde::Deserialize)]
struct PaState {
    num_super: usize, num_sub: usize, alpha: f64, beta: f64, seed: u64, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, super_sub: Option<Arr2>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
    #[serde(default)] log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)] converged: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct HldaState {
    depth: usize, gamma: f64, eta: f64, alpha: f64, seed: u64, fitted: bool,
    num_nodes: usize, node_topic_word: Option<Arr2>, node_levels: Vec<usize>,
    node_parents: Vec<i64>, doc_paths: Vec<Vec<usize>>, corpus: Option<corpus::Corpus>,
    #[serde(default)] topic_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Corpus building from in-memory tokenised documents
// ---------------------------------------------------------------------------

/// Build a `corpus::Corpus` from already-tokenised documents.
///
/// Mirrors the vocab-construction and frequency-filtering logic of
/// `corpus::load_text_file`, minus the regex tokenisation/lowercasing — the
/// caller owns tokenisation here.
#[allow(clippy::too_many_arguments)]
/// True iff `x` is finite and strictly positive. Used to validate float
/// hyperparameters: a plain `x <= 0.0` check lets NaN/Inf through and silently
/// corrupts the fit, so constructors route positivity checks through this.
#[inline]
fn finite_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

fn build_corpus_from_docs(
    docs_in: Vec<Vec<String>>,
    doc_names_in: Option<Vec<String>>,
    doc_labels_in: Option<Vec<String>>,
    stopwords: HashSet<String>,
    min_doc_freq: u32,
    max_doc_fraction: f64,
    min_cf: u32,
    rm_top: usize,
) -> PyResult<(corpus::Corpus, Vec<usize>)> {
    let n = docs_in.len();
    if let Some(names) = &doc_names_in {
        if names.len() != n {
            return Err(PyValueError::new_err(format!(
                "doc_names has {} entries but there are {} documents",
                names.len(),
                n
            )));
        }
    }
    if let Some(labels) = &doc_labels_in {
        if labels.len() != n {
            return Err(PyValueError::new_err(format!(
                "doc_labels has {} entries but there are {} documents",
                labels.len(),
                n
            )));
        }
    }

    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: Vec<String> = Vec::new();
    let mut total_freqs: Vec<u32> = Vec::new();
    let mut docs: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut per_doc_type_sets: Vec<HashSet<usize>> = Vec::with_capacity(n);

    for tokens in &docs_in {
        let mut token_ids: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut seen: HashSet<usize> = HashSet::new();
        for tok in tokens {
            if stopwords.contains(tok) {
                continue;
            }
            let id = if let Some(&eid) = vocab.get(tok) {
                eid
            } else {
                let new_id = id_to_word.len();
                vocab.insert(tok.clone(), new_id);
                id_to_word.push(tok.clone());
                total_freqs.push(0);
                new_id
            };
            total_freqs[id] += 1;
            token_ids.push(id as u32);
            seen.insert(id);
        }
        docs.push(token_ids);
        per_doc_type_sets.push(seen);
    }

    let doc_names: Vec<String> = doc_names_in
        .unwrap_or_else(|| (0..n).map(|i| format!("doc_{}", i)).collect());
    let doc_labels: Vec<String> = doc_labels_in.unwrap_or_else(|| vec![String::new(); n]);

    let num_types = id_to_word.len();
    if num_types == 0 {
        return Err(PyValueError::new_err(
            "corpus has no words after tokenization (all documents are empty)",
        ));
    }
    let num_docs = docs.len();

    let mut doc_freqs = vec![0u32; num_types];
    for set in &per_doc_type_sets {
        for &id in set {
            doc_freqs[id] += 1;
        }
    }

    // Frequency filtering. `min_doc_freq`/`max_doc_fraction` prune by document
    // frequency; `min_cf` prunes by collection (total) frequency; `rm_top` drops
    // the most frequent words by collection frequency (tomotopy's min_df/min_cf/
    // rm_top). The top-`rm_top` set is by total frequency, ties broken by id.
    let max_df = (num_docs as f64 * max_doc_fraction).ceil() as u32;
    let drop_top: HashSet<usize> = if rm_top > 0 {
        let mut order: Vec<usize> = (0..num_types).collect();
        order.sort_by(|&a, &b| total_freqs[b].cmp(&total_freqs[a]).then(a.cmp(&b)));
        order.into_iter().take(rm_top).collect()
    } else {
        HashSet::new()
    };
    let keep: Vec<bool> = (0..num_types)
        .map(|id| {
            doc_freqs[id] >= min_doc_freq
                && doc_freqs[id] <= max_df
                && total_freqs[id] >= min_cf
                && !drop_top.contains(&id)
        })
        .collect();

    if keep.iter().all(|&k| k) {
        let n = docs.len();
        return Ok((
            corpus::Corpus {
                id_to_word,
                docs,
                doc_names,
                doc_labels,
                doc_freqs,
                total_freqs,
            },
            (0..n).collect(),
        ));
    }

    // Remap surviving vocabulary to a dense id range.
    let mut remap: Vec<Option<usize>> = vec![None; num_types];
    let mut new_id_to_word: Vec<String> = Vec::new();
    let mut new_doc_freqs: Vec<u32> = Vec::new();
    let mut new_total_freqs: Vec<u32> = Vec::new();
    for id in 0..num_types {
        if keep[id] {
            remap[id] = Some(new_id_to_word.len());
            new_id_to_word.push(id_to_word[id].clone());
            new_doc_freqs.push(doc_freqs[id]);
            new_total_freqs.push(total_freqs[id]);
        }
    }

    let new_docs: Vec<Vec<u32>> = docs
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .filter_map(|id| remap[id as usize].map(|r| r as u32))
                .collect()
        })
        .collect();

    // Drop documents emptied by pruning, keeping names/labels aligned and
    // recording which original document indices survived (so callers can align
    // external covariates/metadata).
    let mut final_docs: Vec<Vec<u32>> = Vec::new();
    let mut final_names: Vec<String> = Vec::new();
    let mut final_labels: Vec<String> = Vec::new();
    let mut kept_indices: Vec<usize> = Vec::new();
    for (orig_idx, ((doc, name), label)) in new_docs
        .into_iter()
        .zip(doc_names.into_iter())
        .zip(doc_labels.into_iter())
        .enumerate()
    {
        if !doc.is_empty() {
            final_docs.push(doc);
            final_names.push(name);
            final_labels.push(label);
            kept_indices.push(orig_idx);
        }
    }

    if new_id_to_word.is_empty() {
        return Err(PyValueError::new_err(
            "corpus has no words after frequency filtering (min_doc_freq / rm_top too aggressive)",
        ));
    }
    let corpus = corpus::Corpus {
        id_to_word: new_id_to_word,
        docs: final_docs,
        doc_names: final_names,
        doc_labels: final_labels,
        doc_freqs: new_doc_freqs,
        total_freqs: new_total_freqs,
    };
    Ok((corpus, kept_indices))
}

// ---------------------------------------------------------------------------
// Corpus pyclass
// ---------------------------------------------------------------------------

/// A preprocessed, integer-encoded document collection.
///
/// Build one from already-tokenised documents with
/// :meth:`Corpus.from_documents`, from a raw text file with
/// :meth:`Corpus.from_text_file`, or load a binary corpus written by the
/// ``preprocess`` CLI with :meth:`Corpus.load`.
#[pyclass(module = "topica")]
pub struct Corpus {
    inner: corpus::Corpus,
    // Original document indices that survived pruning (parallel to the rows of
    // the corpus). Lets callers realign external covariate/metadata arrays.
    kept_indices: Vec<usize>,
    // Optional per-document metadata (e.g. a pandas DataFrame), already filtered
    // to the surviving rows. Round-tripped as a plain Python object.
    metadata: Option<PyObject>,
}

// Manual Clone: PyObject needs the GIL to bump its refcount, so it can't derive.
impl Clone for Corpus {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Corpus {
            inner: self.inner.clone(),
            kept_indices: self.kept_indices.clone(),
            metadata: self.metadata.as_ref().map(|m| m.clone_ref(py)),
        })
    }
}

#[pymethods]
impl Corpus {
    /// Build a corpus from pre-tokenised documents.
    ///
    /// `documents` is a sequence of token lists. Optional `doc_names` /
    /// `doc_labels` (each the same length as `documents`) attach an id and a
    /// label to every document. `stopwords` are dropped. Vocabulary is pruned by
    /// `min_doc_freq` (minimum document frequency) and `max_doc_fraction`
    /// (maximum fraction of documents), by `min_cf` (minimum collection/total
    /// frequency), and by `rm_top` (drop the N most frequent words) — matching
    /// tomotopy's `min_df` / `min_cf` / `rm_top`.
    ///
    /// A document left with no tokens by pruning is dropped, so `num_docs` can be
    /// smaller than `len(documents)`. The surviving original indices are in
    /// `kept_indices`; realign any external covariate matrix with
    /// `X[corpus.kept_indices]`. (An input document that is empty before any
    /// pruning is retained.)
    #[staticmethod]
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None,
                        stopwords=None, min_doc_freq=1, max_doc_fraction=1.0,
                        min_cf=0, rm_top=0))]
    #[allow(clippy::too_many_arguments)]
    fn from_documents(
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        min_cf: u32,
        rm_top: usize,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let (inner, kept_indices) = build_corpus_from_docs(
            documents,
            doc_names,
            doc_labels,
            stop,
            min_doc_freq,
            max_doc_fraction,
            min_cf,
            rm_top,
        )?;
        Ok(Corpus { inner, kept_indices, metadata: None })
    }

    /// Load and tokenise a raw text file (MALLET-style), matching the
    /// ``preprocess`` CLI.
    ///
    /// `format` is ``"plain"`` (one document per line) or ``"tsv"``. In plain
    /// mode, `id_field=True` treats the first whitespace token as the doc id.
    /// In tsv mode, `id_column`/`label_column`/`text_column` select columns
    /// (`label_column=None` disables labels).
    #[staticmethod]
    #[pyo3(signature = (path, *, format="plain", id_field=false,
                        id_column=0, label_column=1, text_column=2,
                        token_regex=None, stopwords=None,
                        min_doc_freq=1, max_doc_fraction=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn from_text_file(
        path: &str,
        format: &str,
        id_field: bool,
        id_column: usize,
        label_column: Option<usize>,
        text_column: usize,
        token_regex: Option<String>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
    ) -> PyResult<Self> {
        let fmt = match format {
            "plain" => InputFormat::Plain { id_field },
            "tsv" => InputFormat::Tsv {
                id_column,
                label_column,
                text_column,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown format {:?} (use 'plain' or 'tsv')",
                    other
                )))
            }
        };
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let opts = LoadOptions {
            format: fmt,
            token_regex: token_regex.unwrap_or_else(|| corpus::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
        };
        let inner = corpus::load_text_file(Path::new(path), &opts).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus { inner, kept_indices, metadata: None })
    }

    /// Load a binary corpus file written by the ``preprocess`` CLI or
    /// :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = corpus::load_corpus(Path::new(path)).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus { inner, kept_indices, metadata: None })
    }

    /// Write this corpus to a binary file (the ``preprocess`` format), so it
    /// can be reused by the CLI tools or reloaded with :meth:`load`.
    fn save(&self, path: &str) -> PyResult<()> {
        corpus::save_corpus(&self.inner, Path::new(path)).map_err(io_err)
    }

    #[getter]
    fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    #[getter]
    fn num_words(&self) -> usize {
        self.inner.num_types()
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    /// Tokens per document in the pruned vocabulary, one entry per kept document
    /// (parallel to the rows of a fitted model's ``doc_topic``). This is the
    /// document length ``N_d`` that :func:`topica.dirichlet_theta_samples` needs to
    /// recover each document's Dirichlet posterior for method-of-composition
    /// standard errors.
    #[getter]
    fn doc_lengths(&self) -> Vec<usize> {
        self.inner.docs.iter().map(|d| d.len()).collect()
    }

    #[getter]
    fn vocabulary(&self) -> Vec<String> {
        self.inner.id_to_word.clone()
    }

    /// The corpus as token lists — one list of word strings per document, in the
    /// pruned vocabulary and the kept-document order. The inverse of
    /// ``from_documents``: use it to recover tokens for ``prepare_pyldavis``,
    /// ``coherence``, or any function that wants ``list[list[str]]`` after you have
    /// committed to a ``Corpus``.
    fn documents(&self) -> Vec<Vec<String>> {
        self.inner
            .docs
            .iter()
            .map(|d| d.iter().map(|&w| self.inner.id_to_word[w as usize].clone()).collect())
            .collect()
    }

    /// Original document indices that survived pruning, parallel to the rows of
    /// this corpus. Use it to realign an external covariate array or DataFrame
    /// to the documents the corpus actually kept: ``X = X[corpus.kept_indices]``.
    #[getter]
    fn kept_indices(&self) -> Vec<usize> {
        self.kept_indices.clone()
    }

    /// Optional per-document metadata, already aligned to the surviving rows
    /// (set by :func:`topica.from_dataframe`, or assign your own). ``None`` if
    /// unset.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Option<PyObject> {
        self.metadata.as_ref().map(|m| m.clone_ref(py))
    }

    #[setter]
    fn set_metadata(&mut self, value: Option<PyObject>) {
        self.metadata = value;
    }

    #[getter]
    fn doc_names(&self) -> Vec<String> {
        self.inner.doc_names.clone()
    }

    #[getter]
    fn doc_labels(&self) -> Vec<String> {
        self.inner.doc_labels.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Corpus(num_docs={}, num_words={}, total_tokens={})",
            self.inner.num_docs(),
            self.inner.num_types(),
            self.inner.total_tokens()
        )
    }
}

// ---------------------------------------------------------------------------
// LDA pyclass
// ---------------------------------------------------------------------------

/// SparseLDA topic model (the MALLET algorithm).
///
/// Construct with the hyperparameters, then call :meth:`fit` on a
/// :class:`Corpus` or a list of token lists. After fitting, the estimated
/// distributions are available as :attr:`topic_word` (φ) and
/// :attr:`doc_topic` (θ).
#[pyclass(module = "topica")]
pub struct LDA {
    num_topics: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    num_threads: usize,
    // Sampling backend: false = SparseLDA (MALLET), true = LightLDA alias-MH.
    light: bool,
    // WarpLDA cache-efficient two-pass MH sampler (mutually exclusive with light).
    warp: bool,
    // CVB0 deterministic collapsed-variational inference (no MCMC draws).
    cvb0: bool,
    mh_steps: usize,
    // MALLET's --use-symmetric-alpha: optimize only the alpha concentration,
    // keeping every alpha[t] equal, instead of learning the per-topic shape.
    use_symmetric_alpha: bool,
    // Seed the initial token→topic assignment from a spectral anchor-word β
    // instead of a uniform random draw. Opt-in (default random) so the CLI
    // byte-parity (binding == bundled train CLI) and existing determinism
    // baselines are unchanged.
    init_spectral: bool,

    // Populated after fit().
    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,   // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
    model: Option<TopicModel>,
    corpus: Option<corpus::Corpus>,
    // Convergence tracking (issue #46 uniform interface).
    log_likelihood_history: Vec<(usize, f64)>, // (iteration, log_likelihood)
    converged: bool,   // true only when convergence_tol criterion was met
}

impl LDA {
    /// Transpose the accumulated φ/θ snapshots into the conventional matrix
    /// orientation and store the fitted state. Shared by both sampler paths.
    #[allow(clippy::too_many_arguments)]
    fn finalize_fit(
        &mut self,
        num_topics: usize,
        num_types: usize,
        num_docs: usize,
        acc_phi: Vec<Vec<f64>>,
        acc_theta: Vec<Vec<f64>>,
        model: TopicModel,
        corpus: corpus::Corpus,
        log_likelihood_history: Vec<(usize, f64)>,
        converged: bool,
    ) {
        // phi: transpose (word, topic) -> (topic, word).
        let mut phi = Array2::<f64>::zeros((num_topics, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                phi[[t, w]] = v;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, num_topics));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                theta[[d, t]] = v;
            }
        }
        self.topic_names = (0..num_topics).map(|i| format!("topic_{i}")).collect();
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.log_likelihood_history = log_likelihood_history;
        self.converged = converged;
        self.fitted = true;
    }

    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    /// Top-`n` word ids per topic, descending by φ.
    fn top_word_ids(&self, n: usize) -> Vec<Vec<usize>> {
        let phi = self.phi.as_ref().unwrap();
        let num_words = phi.shape()[1];
        (0..self.num_topics)
            .map(|t| {
                let mut idx: Vec<usize> = (0..num_words).collect();
                idx.sort_by(|&a, &b| f64::total_cmp(&phi[[t, b]], &phi[[t, a]]));
                idx.truncate(n);
                idx
            })
            .collect()
    }

    /// Map held-out documents (a `Corpus` or `list[list[str]]`) to trained
    /// vocabulary ids, dropping out-of-vocabulary tokens. Returns
    /// `(docs_as_ids, num_tokens_scored, num_oov_dropped)`.
    fn map_heldout(
        &self,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<Vec<usize>>, usize, usize)> {
        let trained = self.corpus.as_ref().unwrap();
        let index: HashMap<&str, usize> = trained
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let str_docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
                .docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&wid| c.inner.id_to_word[wid as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract::<Vec<Vec<String>>>().map_err(|_| {
                PyValueError::new_err("expected a Corpus or a list of token lists (list[list[str]])")
            })?
        };

        let mut out = Vec::with_capacity(str_docs.len());
        let mut n_tokens = 0usize;
        let mut n_oov = 0usize;
        for doc in &str_docs {
            let mut ids = Vec::with_capacity(doc.len());
            for tok in doc {
                match index.get(tok.as_str()) {
                    Some(&id) => {
                        ids.push(id);
                        n_tokens += 1;
                    }
                    None => n_oov += 1,
                }
            }
            out.push(ids);
        }
        Ok((out, n_tokens, n_oov))
    }
}

#[pymethods]
impl LDA {
    /// Create an unfitted model.
    ///
    /// `alpha_sum` is the total document-topic Dirichlet mass (default:
    /// `num_topics`, i.e. 1.0 per topic). `beta` is the per-word topic-word
    /// prior. With `optimize_interval > 0`, α and β are re-estimated every
    /// that-many iterations once past `burn_in`.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_sum=None, beta=0.01,
                        optimize_interval=50, burn_in=200, seed=42, num_threads=1,
                        sampler="sparse", mh_steps=2, use_symmetric_alpha=false,
                        init="random"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha_sum: Option<f64>,
        beta: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        num_threads: usize,
        sampler: &str,
        mh_steps: usize,
        use_symmetric_alpha: bool,
        init: &str,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if !finite_pos(beta) {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        if let Some(a) = alpha_sum {
            if !finite_pos(a) {
                return Err(PyValueError::new_err(
                    "alpha_sum must be a positive, finite number",
                ));
            }
        }
        let (light, warp, cvb0) = match sampler {
            "sparse" | "mallet" => (false, false, false),
            "lightlda" | "light" | "alias" => (true, false, false),
            "warp" | "warplda" => (false, true, false),
            "cvb0" | "cvb" => (false, false, true),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\", \"lightlda\", \"warp\", or \"cvb0\""
                )))
            }
        };
        if light && mh_steps == 0 {
            return Err(PyValueError::new_err("mh_steps must be >= 1 for the lightlda sampler"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        Ok(LDA {
            num_topics,
            alpha_sum,
            beta,
            optimize_interval,
            burn_in,
            seed,
            num_threads: num_threads.max(1),
            light,
            warp,
            cvb0,
            mh_steps,
            use_symmetric_alpha,
            init_spectral,
            fitted: false,
            topic_names: Vec::new(),
            phi: None,
            theta: None,
            theta_draws: None,
            model: None,
            corpus: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Run Gibbs sampling on `data`, then average `num_samples` snapshots
    /// (taken `sample_interval` iterations apart) into the final φ/θ estimates.
    ///
    /// `data` may be a :class:`Corpus` or a list of token lists (each a list of
    /// strings). When a token-list is passed, an internal corpus is built with
    /// no frequency filtering — build a :class:`Corpus` explicitly for that.
    ///
    /// `progress`, if given, is called as ``progress(iteration, ll_per_token)``
    /// every `progress_interval` iterations during the main loop.
    ///
    /// `convergence_tol` (default 0.0, disabled) enables early stopping: after
    /// each `check_every` sweeps the relative change in a smoothed log-likelihood
    /// is compared; if it falls below `convergence_tol` the loop stops and
    /// :attr:`converged` is set to ``True``. When 0 (default), the full `iters`
    /// sweeps always run (default behavior is unchanged, bit-for-bit identical).
    ///
    /// `turbo_merge_every` (default 1, exact) is an opt-in approximate-speed knob
    /// for multi-threaded runs only. The parallel sampler partitions documents
    /// across workers and reconciles the shared topic-word counts after every
    /// sweep; that per-sweep merge is the thread-scaling ceiling. Setting this to
    /// ``m > 1`` lets each worker run ``m`` sweeps against its own counts before
    /// one merge, so the table is synchronized once per ``m`` sweeps. This is
    /// approximate (workers sample against staler cross-partition counts the
    /// deeper into a batch they go), so results differ from the exact path and
    /// are not bit-reproducible against it; with ``m = 1`` (or single-threaded,
    /// or the LightLDA/WarpLDA/CVB0 samplers) the exact per-sweep path runs and
    /// is unchanged. We measured the tradeoff on a large wide-vocabulary corpus
    /// (30k docs, 30k vocabulary, K=400, 8 threads): ``m = 3`` ran 1.55x faster
    /// for a 0.010 drop in c_npmi topic coherence. The win appears only when the
    /// merge actually dominates (large corpus, wide vocabulary, high K, many
    /// threads); on smaller corpora it does not help and can run slower, so leave
    /// it at the default unless profiling shows the merge is your bottleneck.
    /// Recommended range when it helps: 3 to 4.
    #[pyo3(signature = (data, *, iters=1000, num_samples=5, sample_interval=25,
                        progress=None, progress_interval=50,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize, num_threads=None,
                        turbo_merge_every=1_usize))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
        num_threads: Option<usize>,
        turbo_merge_every: usize,
    ) -> PyResult<()> {
        // Accept either a Corpus or a list[list[str]].
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err(
                    "fit() expects a Corpus or a list of token lists (list[list[str]])",
                )
            })?;
            build_corpus_from_docs(docs, None, None, HashSet::new(), 1, 1.0, 0, 0)?.0
        };

        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let num_topics = self.num_topics;
        let num_types = corpus.num_types();
        let num_docs = corpus.num_docs();
        let alpha_sum = self.alpha_sum.unwrap_or(num_topics as f64);
        let total_tokens = corpus.total_tokens().max(1) as f64;

        // When check_every=0 the caller explicitly disabled trace recording.
        // When convergence_tol > 0 and check_every was given a positive value,
        // enforce at least 1 so the modulo never divides by zero.
        let check_every = if check_every == 0 {
            0_usize
        } else if convergence_tol > 0.0 {
            check_every.max(1)
        } else {
            check_every
        };

        // Thinned θ-draw retention (issue #31): keep the last `draw_cap` snapshots
        // taken every `draw_thin` sweeps of the main loop. 0 ⇒ collection off.
        let draw_cap = if keep_theta_draws { num_theta_draws } else { 0 };
        // draw_thin is computed against `iters`; under early stop we apply the
        // same schedule (any iteration that passes draw_thin mod-check gets a draw).
        let draw_thin = theta_draw_thin(iters, draw_cap);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, num_topics)?;

        let mut model = TopicModel::new(num_topics, alpha_sum, self.beta, num_types);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        // Spectral anchor-word init is opt-in; it falls back to the random draw
        // when the corpus is too small for anchor recovery (spectral_init -> None).
        if self.init_spectral {
            match spectral::spectral_init(&corpus.docs, num_topics, num_types) {
                Some(beta) => model.initialize_spectral(&corpus, &beta, &mut rng),
                None => model.initialize(&corpus, &mut rng),
            }
        } else {
            model.initialize(&corpus, &mut rng);
        }

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        // num_threads from fit() overrides the constructor default for this run.
        let num_threads = num_threads.unwrap_or(self.num_threads).max(1);
        // turbo_merge_every (default 1): in the approximate-parallel (num_threads
        // > 1) path, defer the per-sweep count-table reconciliation, merging once
        // every this-many sweeps instead. 1 keeps the exact per-sweep path
        // (bit-identical). Has no effect single-threaded (the sequential sampler
        // is always exact) or on the LightLDA/WarpLDA/CVB0 samplers.
        if turbo_merge_every == 0 {
            return Err(PyValueError::new_err("turbo_merge_every must be >= 1"));
        }
        let merge_every = turbo_merge_every.max(1);
        let seed_base = self.seed;
        let light = self.light;
        let warp = self.warp;
        let cvb0_flag = self.cvb0;
        let mh_steps = self.mh_steps;
        let beta = self.beta;
        let use_symmetric_alpha = self.use_symmetric_alpha;

        // CVB0 path: deterministic collapsed-variational inference. No MCMC, so
        // no θ-draws; convergence_tol early-stops on the mean |Δγ| per sweep.
        if cvb0_flag {
            let (acc_phi, acc_theta, ll_history, converged, model, corpus) =
                py.allow_threads(move || {
                    let alpha0 = vec![alpha_sum / num_topics as f64; num_topics];
                    let mut cv = cvb0::Cvb0::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                    let mut ll_history: Vec<(usize, f64)> = Vec::new();
                    let mut converged = false;
                    for iter in 1..=iters {
                        let change = cv.sweep();
                        if let Some(cb) = &progress {
                            if progress_interval > 0 && iter % progress_interval == 0 {
                                let m = cv.to_topic_model(&corpus);
                                let ll = output::model_log_likelihood(&m, &corpus) / total_tokens;
                                Python::with_gil(|py| {
                                    let _ = cb.call1(py, (iter, ll));
                                });
                            }
                        }
                        if check_every > 0 && iter % check_every == 0 {
                            let m = cv.to_topic_model(&corpus);
                            ll_history.push((iter, output::model_log_likelihood(&m, &corpus)));
                        }
                        if convergence_tol > 0.0 && change < convergence_tol {
                            converged = true;
                            break;
                        }
                    }
                    let mut acc_phi = vec![vec![0.0f64; num_topics]; num_types];
                    let mut acc_theta = vec![vec![0.0f64; num_topics]; num_docs];
                    cv.phi_into(&mut acc_phi);
                    cv.theta_into(&mut acc_theta);
                    let model = cv.to_topic_model(&corpus);
                    (acc_phi, acc_theta, ll_history, converged, model, corpus)
                });
            self.theta_draws = None;
            self.finalize_fit(num_topics, num_types, num_docs, acc_phi, acc_theta, model, corpus,
                ll_history, converged);
            return Ok(());
        }

        // Metropolis-Hastings backends (WarpLDA, LightLDA): each owns its dense
        // state and is driven through the shared `run_mh_training` loop, then
        // packed back into a TopicModel. Construction is the only per-sampler
        // difference. Unlike the SparseLDA path below, these compute no inline
        // log_likelihood, so convergence_tol is unsupported (full iters, empty
        // trace, converged=false). The SparseLDA path stays separate to keep its
        // convergence trace, parallel sweep, and CLI byte-parity untouched.
        if warp || light {
            let (acc_phi, acc_theta, theta_draw_buf, model, corpus) =
                py.allow_threads(move || {
                    let alpha0 = vec![alpha_sum / num_topics as f64; num_topics];
                    if warp {
                        let ws = warplda::WarpLda::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                        run_mh_training(
                            ws, corpus, num_topics, num_types, num_docs, iters, num_samples,
                            sample_interval, burn_in, optimize_interval, use_symmetric_alpha,
                            draw_thin, draw_cap, total_tokens, &mut rng, &progress, progress_interval,
                        )
                    } else {
                        let mut ls =
                            lightlda::LightLda::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                        ls.mh_steps = mh_steps;
                        run_mh_training(
                            ls, corpus, num_topics, num_types, num_docs, iters, num_samples,
                            sample_interval, burn_in, optimize_interval, use_symmetric_alpha,
                            draw_thin, draw_cap, total_tokens, &mut rng, &progress, progress_interval,
                        )
                    }
                });
            self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, num_topics, None);
            self.finalize_fit(num_topics, num_types, num_docs, acc_phi, acc_theta, model, corpus,
                Vec::new(), false);
            return Ok(());
        }

        // Heavy loop runs with the GIL released; the progress callback briefly
        // re-acquires it. allow_threads returns the owned model + accumulators.
        let (acc_phi, acc_theta, theta_draw_buf, ll_history, converged, model) =
            py.allow_threads(move || {
            // One logical Gibbs sweep: exact sequential path when
            // single-threaded, approximate parallel sampling otherwise. `sweep`
            // seeds the per-worker RNGs so parallel runs are deterministic.
            //
            // In turbo mode (merge_every > 1, parallel only) sampling proceeds in
            // batches: each `do_sweep` call runs `merge_every` worker sweeps and
            // reconciles once, returning a globally consistent model. The caller
            // therefore steps the iteration counter by `merge_every` per call and
            // runs the per-iteration bookkeeping (θ-draws, α/β optimization,
            // convergence checks) at those batch boundaries, where the model is
            // consistent. With merge_every == 1 this is the exact per-sweep path.
            let mut sweep: u64 = 0;
            // logical iterations advanced by one do_sweep call.
            let step = if num_threads <= 1 { 1 } else { merge_every };
            // `batch` is the number of worker sweeps to run before reconciling;
            // it is `step` for full windows and the remainder for a short tail
            // window so the run never samples more than `iters` total sweeps.
            let mut do_sweep =
                |model: &mut TopicModel, rng: &mut Pcg64Mcg, batch: usize| {
                    if num_threads <= 1 {
                        sweep += 1;
                        sampler::run_iteration(model, &corpus, rng);
                    } else {
                        sweep += 1;
                        let s = seed_base
                            .wrapping_add(sweep.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        parallel_sweep_batched(model, &corpus.docs, num_threads, s, batch.max(1));
                    }
                };
            let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut ll_history: Vec<(usize, f64)> = Vec::new();
            let mut converged = false;

            // ---- main training loop (ports src/bin/train.rs) ----
            // `step` is 1 except in turbo mode, where each do_sweep advances
            // `step` logical iterations. We sample at the start of each window
            // and run the per-iteration bookkeeping at its end (where the model
            // is freshly reconciled), so θ-draws/optimization/convergence always
            // observe a consistent global state.
            let mut iter = 0usize;
            while iter < iters {
                let batch = (iters - iter).min(step);
                do_sweep(&mut model, &mut rng, batch);
                iter += batch;

                if draw_thin > 0 && iter % draw_thin == 0 {
                    push_capped(
                        &mut theta_draw_buf,
                        theta_snapshot_f32(&model, &corpus),
                        draw_cap,
                    );
                }

                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    if use_symmetric_alpha {
                        optimize::optimize_alpha_symmetric(&mut model, &corpus);
                    } else {
                        optimize::optimize_alpha(&mut model, &corpus);
                    }
                    optimize::optimize_beta(&mut model);
                }

                // Trace recording and optional convergence check (never alters RNG).
                if convergence_tol > 0.0 && check_every > 0 && iter % check_every == 0 {
                    let ll = output::model_log_likelihood(&model, &corpus);
                    ll_history.push((iter, ll));
                    // Relative change criterion: compare the current ll to the
                    // one recorded one window back (window = check_every sweeps).
                    if ll_history.len() >= 2 {
                        let prev = ll_history[ll_history.len() - 2].1;
                        let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                        if rel < convergence_tol {
                            converged = true;
                            break;
                        }
                    }
                } else if convergence_tol == 0.0 && check_every > 0 && iter % check_every == 0 {
                    // When tol is disabled, still record the trace so fit_history
                    // is non-empty, but never break early.
                    let ll = output::model_log_likelihood(&model, &corpus);
                    ll_history.push((iter, ll));
                }

                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let ll = output::model_log_likelihood(&model, &corpus) / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, ll));
                        });
                    }
                }
            }

            // Under early stop, draw_thin was computed against the nominal `iters`
            // but the loop ended at `actual_iters` sweeps; remaining draws are
            // whatever was already collected (ring-buffered), which is correct.

            // ---- sampling phase: average num_samples smoothed snapshots ----
            let mut acc_phi = vec![vec![0.0f64; num_topics]; num_types];
            let mut acc_theta = vec![vec![0.0f64; num_topics]; num_docs];

            for _ in 0..num_samples {
                // Step through `sample_interval` sweeps; in turbo mode each
                // do_sweep covers a batch of `step`, reconciling once per batch.
                let mut done = 0usize;
                while done < sample_interval {
                    let batch = (sample_interval - done).min(step);
                    do_sweep(&mut model, &mut rng, batch);
                    done += batch;
                }
                accumulate_phi(&model, &mut acc_phi);
                accumulate_theta(&model, &corpus, &mut acc_theta);
            }

            let n = (num_samples.max(1)) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }

            // Return the corpus too (move it back out for storage).
            (acc_phi, acc_theta, theta_draw_buf, ll_history, converged, (model, corpus))
        });
        let (model, corpus) = model;
        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, num_topics, None);
        self.finalize_fit(num_topics, num_types, num_docs, acc_phi, acc_theta, model, corpus,
            ll_history, converged);
        Ok(())
    }

    /// Topic-word probability matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic probability matrix θ, shape ``(num_docs, num_topics)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Thinned MCMC θ draws, shape ``(num_draws, num_docs, num_topics)``, or
    /// ``None`` when fit with ``keep_theta_draws=False``. These are real
    /// cross-sweep posterior samples; :func:`topica.composition_theta` prefers
    /// them over the within-document Dirichlet approximation.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Per-document token counts (length D), in :attr:`doc_topic` row order. Lets
    /// :func:`topica.composition_theta` recover the Dirichlet concentration N_d
    /// without re-threading the original :class:`Corpus`.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self
            .corpus
            .as_ref()
            .map(|c| c.docs.iter().map(|d| d.len()).collect())
            .unwrap_or_default())
    }

    /// The vocabulary: word for each column of :attr:`topic_word`.
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Document ids, parallel to the rows of :attr:`doc_topic`.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Per-topic α after (optional) optimisation, shape ``(num_topics,)``.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let a = Array1::from(self.model.as_ref().unwrap().alpha.clone());
        Ok(a.to_pyarray_bound(py))
    }

    /// The (optimised) symmetric β.
    #[getter]
    fn beta(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.model.as_ref().unwrap().beta)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Top `n` words per topic as ``(word, probability)`` pairs.
    ///
    /// Returns a list of `n`-length lists (one per topic), or — when `topic`
    /// is given — just that topic's list.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let num_words = vocab.len();

        let one_topic = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let mut idx: Vec<usize> = (0..num_words).collect();
            idx.sort_by(|&a, &b| f64::total_cmp(&phi[[t, b]], &phi[[t, a]]));
            let items: Vec<Bound<'py, PyTuple>> = idx
                .iter()
                .take(n)
                .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)]))
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one_topic(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one_topic).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// Per-iteration log-likelihood trace: ``(iteration, log_likelihood)`` pairs
    /// recorded every ``check_every`` sweeps during :meth:`fit`. Non-empty for
    /// the SparseLDA path; empty for the LightLDA path.
    #[getter]
    fn log_likelihood_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// Uniform convergence trace: ``(iteration, log_likelihood)`` pairs, one per
    /// trace checkpoint. Equivalent to :attr:`log_likelihood_history` for LDA.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// ``True`` if fit stopped early because the convergence tolerance criterion
    /// was met (``convergence_tol > 0``); ``False`` if the full ``iters``
    /// sweeps ran (the default).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// MALLET-formula model log-likelihood of the final sampler state.
    fn log_likelihood(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(output::model_log_likelihood(
            self.model.as_ref().unwrap(),
            self.corpus.as_ref().unwrap(),
        ))
    }

    /// Write topic-word probabilities to a TSV file (the ``train`` CLI format).
    fn save_topic_word(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        // Re-orient to [word][topic] as write_topic_word_matrix expects.
        let phi_wt: Vec<Vec<f64>> = (0..corpus.num_types())
            .map(|w| (0..self.num_topics).map(|t| phi[[t, w]]).collect())
            .collect();
        output::write_topic_word_matrix(&phi_wt, corpus, Path::new(path)).map_err(io_err)
    }

    /// Write document-topic probabilities to a TSV file (the ``train`` CLI format).
    fn save_doc_topic(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        let theta = self.theta.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        let theta_dt: Vec<Vec<f64>> = (0..corpus.num_docs())
            .map(|d| (0..self.num_topics).map(|t| theta[[d, t]]).collect())
            .collect();
        output::write_doc_topic_matrix(&theta_dt, corpus, Path::new(path)).map_err(io_err)
    }

    /// Write the token-level Gibbs state to a gzipped file in MALLET's
    /// ``--output-state`` format: a header, the ``#alpha``/``#beta`` hyperparameter
    /// lines, then one row per token — ``doc source pos typeindex type topic`` —
    /// giving the final topic assignment of every token in the training corpus.
    /// Researchers pipe this into custom visualizations (e.g. pyLDAvis) or
    /// corpus metrics. The file is gzip-compressed, as MALLET writes it.
    fn save_state(&self, path: &str) -> PyResult<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        self.require_fitted()?;
        let model = self.model.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no token-level state available; refit the model")
        })?;
        let corpus = self.corpus.as_ref().unwrap();

        let mut buf = String::new();
        buf.push_str("#doc source pos typeindex type topic\n");
        buf.push_str("#alpha : ");
        buf.push_str(
            &model.alpha.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(" "),
        );
        buf.push('\n');
        buf.push_str(&format!("#beta : {}\n", model.beta));

        for (d, doc) in corpus.docs.iter().enumerate() {
            let source = match corpus.doc_names.get(d) {
                Some(s) if !s.is_empty() => s.as_str(),
                _ => "NA",
            };
            let z = &model.doc_topics[d];
            for (pos, &w) in doc.iter().enumerate() {
                let word = &corpus.id_to_word[w as usize];
                buf.push_str(&format!("{} {} {} {} {} {}\n", d, source, pos, w, word, z[pos]));
            }
        }

        let file = std::fs::File::create(path).map_err(io_err)?;
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(buf.as_bytes()).map_err(io_err)?;
        enc.finish().map_err(io_err)?;
        Ok(())
    }

    /// Reconstruct a fitted model from a MALLET-format Gibbs state file (the
    /// inverse of :meth:`save_state`; MALLET's ``--input-state``). The file may
    /// be gzip-compressed or plain text. The vocabulary, documents, per-token
    /// topic assignments, and the ``#alpha``/``#beta`` hyperparameters are read
    /// back, so the loaded model supports the full read-only surface
    /// (``topic_word``, ``doc_topic``, ``top_words``, …) and ``transform`` on new
    /// documents, and can re-emit the state with :meth:`save_state`.
    #[staticmethod]
    fn load_state(path: &str) -> PyResult<Self> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let raw = std::fs::read(path).map_err(io_err)?;
        // Detect gzip by magic bytes; fall back to plain text otherwise.
        let text = if raw.starts_with(&[0x1f, 0x8b]) {
            let mut s = String::new();
            GzDecoder::new(&raw[..]).read_to_string(&mut s).map_err(io_err)?;
            s
        } else {
            String::from_utf8(raw).map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        let mut alpha: Vec<f64> = Vec::new();
        let mut beta = 0.01f64;
        let mut id_to_word: Vec<String> = Vec::new();
        // doc id -> (pos, word id, topic); BTreeMap keeps documents in id order.
        let mut docs_tokens: std::collections::BTreeMap<usize, Vec<(usize, u32, u32)>> =
            std::collections::BTreeMap::new();
        let mut doc_source: std::collections::BTreeMap<usize, String> =
            std::collections::BTreeMap::new();
        let mut max_topic = 0u32;

        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("#alpha") {
                alpha = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                continue;
            }
            if let Some(rest) = line.strip_prefix("#beta") {
                if let Some(b) = rest.split_whitespace().find_map(|s| s.parse().ok()) {
                    beta = b;
                }
                continue;
            }
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            // doc source pos typeindex type topic
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() < 6 {
                return Err(PyValueError::new_err(format!("malformed state row: {line:?}")));
            }
            let parse_err = || PyValueError::new_err(format!("malformed state row: {line:?}"));
            let doc: usize = p[0].parse().map_err(|_| parse_err())?;
            let pos: usize = p[2].parse().map_err(|_| parse_err())?;
            let typeindex: usize = p[3].parse().map_err(|_| parse_err())?;
            let topic: u32 = p[5].parse().map_err(|_| parse_err())?;
            if typeindex >= id_to_word.len() {
                id_to_word.resize(typeindex + 1, String::new());
            }
            id_to_word[typeindex] = p[4].to_string();
            max_topic = max_topic.max(topic);
            docs_tokens.entry(doc).or_default().push((pos, typeindex as u32, topic));
            doc_source.entry(doc).or_insert_with(|| p[1].to_string());
        }

        if docs_tokens.is_empty() {
            return Err(PyValueError::new_err("state file contains no token rows"));
        }
        let num_topics = if alpha.is_empty() { max_topic as usize + 1 } else { alpha.len() };
        let num_types = id_to_word.len();

        let mut docs_v: Vec<Vec<u32>> = Vec::new();
        let mut doc_topics: Vec<Vec<u32>> = Vec::new();
        let mut doc_names: Vec<String> = Vec::new();
        for (doc_id, mut toks) in docs_tokens {
            toks.sort_by_key(|&(pos, _, _)| pos);
            docs_v.push(toks.iter().map(|&(_, w, _)| w).collect());
            doc_topics.push(toks.iter().map(|&(_, _, t)| t).collect());
            let src = doc_source.remove(&doc_id).unwrap_or_default();
            doc_names.push(if src.is_empty() || src == "NA" { format!("doc_{doc_id}") } else { src });
        }
        let num_docs = docs_v.len();

        // Word frequencies for the reconstructed corpus.
        let mut total_freqs = vec![0u32; num_types];
        let mut doc_freqs = vec![0u32; num_types];
        for doc in &docs_v {
            let mut seen = vec![false; num_types];
            for &w in doc {
                total_freqs[w as usize] += 1;
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    doc_freqs[w as usize] += 1;
                }
            }
        }

        let corpus = corpus::Corpus {
            id_to_word,
            docs: docs_v,
            doc_names,
            doc_labels: vec![String::new(); num_docs],
            doc_freqs,
            total_freqs,
        };

        let alpha_sum: f64 = if alpha.is_empty() {
            num_topics as f64
        } else {
            alpha.iter().sum()
        };
        let mut model = TopicModel::new(num_topics, alpha_sum, beta, num_types);
        model.initialize_from_assignments(&corpus, doc_topics);
        if !alpha.is_empty() {
            model.alpha = alpha;
            model.alpha_sum = alpha_sum;
        }

        // φ and θ from the restored counts (smoothed point estimates).
        let mut phi = Array2::<f64>::zeros((num_topics, num_types));
        let mut theta = Array2::<f64>::zeros((num_docs, num_topics));
        for (d, (doc, topics)) in corpus.docs.iter().zip(model.doc_topics.iter()).enumerate() {
            for (&w, &t) in doc.iter().zip(topics) {
                phi[[t as usize, w as usize]] += 1.0;
                theta[[d, t as usize]] += 1.0;
            }
            let denom = doc.len() as f64 + model.alpha_sum;
            for t in 0..num_topics {
                theta[[d, t]] = (theta[[d, t]] + model.alpha[t]) / denom;
            }
        }
        for t in 0..num_topics {
            let denom = model.tokens_per_topic[t] as f64 + beta * num_types as f64;
            for w in 0..num_types {
                phi[[t, w]] = (phi[[t, w]] + beta) / denom;
            }
        }

        Ok(LDA {
            num_topics,
            alpha_sum: Some(model.alpha_sum),
            beta,
            optimize_interval: 50,
            burn_in: 200,
            seed: 42,
            num_threads: 1,
            light: false,
            warp: false,
            cvb0: false,
            mh_steps: 2,
            use_symmetric_alpha: false,
            init_spectral: false,
            fitted: true,
            topic_names: (0..num_topics).map(|i| format!("topic_{i}")).collect(),
            phi: Some(phi),
            theta: Some(theta),
            theta_draws: None,
            model: Some(model),
            corpus: Some(corpus),
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Held-out evaluation via the Wallach et al. (2009) left-to-right
    /// estimator (the method MALLET's ``evaluate-topics`` uses).
    ///
    /// `data` is a held-out :class:`Corpus` or `list[list[str]]`; its tokens are
    /// matched to the training vocabulary by string (out-of-vocabulary tokens
    /// are dropped). Returns a dict with `log_likelihood` (total held-out log
    /// P(data)), `perplexity` (``exp(-LL / num_tokens)``, lower is better),
    /// `num_tokens` (scored), and `num_oov` (dropped). Cost grows with the
    /// square of document length, so keep `num_particles` modest.
    #[pyo3(signature = (data, *, num_particles=10, seed=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        num_particles: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        if num_particles == 0 {
            return Err(PyValueError::new_err("num_particles must be >= 1"));
        }
        let (docs, n_tokens, n_oov) = self.map_heldout(data)?;
        let model = self.model.as_ref().unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(seed.unwrap_or(self.seed));

        let ll = py.allow_threads(move || {
            let mut total = 0.0;
            for doc in &docs {
                total += left_to_right_doc(model, doc, num_particles, &mut rng);
            }
            total
        });

        let perplexity = if n_tokens > 0 {
            (-ll / n_tokens as f64).exp()
        } else {
            f64::NAN
        };

        let d = PyDict::new_bound(py);
        d.set_item("log_likelihood", ll)?;
        d.set_item("perplexity", perplexity)?;
        d.set_item("num_tokens", n_tokens)?;
        d.set_item("num_oov", n_oov)?;
        Ok(d)
    }

    /// Held-out perplexity (lower is better) — convenience wrapper over
    /// :meth:`evaluate`. See `evaluate` for `data`/`num_particles` semantics.
    #[pyo3(signature = (data, *, num_particles=10, seed=None))]
    fn perplexity<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        num_particles: usize,
        seed: Option<u64>,
    ) -> PyResult<f64> {
        let d = self.evaluate(py, data, num_particles, seed)?;
        d.get_item("perplexity")?.unwrap().extract()
    }

    /// UMass topic coherence for each topic, shape ``(num_topics,)``.
    ///
    /// Intrinsic (no external corpus): for each topic's top-`n` words,
    /// `Σ_{i>j} log[(codoc(w_i,w_j)+1)/docfreq(w_j)]` over the training corpus.
    /// Higher (closer to 0) is more coherent. `numpy.mean(...)` gives the
    /// usual single-number summary.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = self.top_word_ids(n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Per-topic diagnostics (MALLET-style), one dict per topic, suitable for
    /// `pandas.DataFrame(model.diagnostics())`.
    ///
    /// Keys mirror MALLET's topic diagnostics: `topic`, `tokens` (assignments to
    /// the topic), `coherence` (UMass), `exclusivity` (mean top-word share of φ
    /// vs. other topics; higher = more distinctive), `effective_words`
    /// (`exp(H(φ_t))`, MALLET's `eff_num_words`; lower = more focused),
    /// `document_entropy` (entropy of the topic's token allocation across
    /// documents), `uniform_dist` (KL of φ_t from uniform) and `corpus_dist`
    /// (KL of φ_t from the corpus word distribution), `rank1_docs` (documents
    /// whose dominant topic is this one), `alpha`, and `top_words`.
    #[pyo3(signature = (n=10))]
    fn diagnostics<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let theta = self.theta.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        let model = self.model.as_ref().unwrap();
        let vocab = &corpus.id_to_word;
        let num_words = phi.shape()[1];
        let num_docs = theta.shape()[0];

        let tops = self.top_word_ids(n);
        let coh = umass_coherence(corpus, &tops);

        // Column sums of φ for the exclusivity denominator.
        let mut col_sum = vec![0.0f64; num_words];
        for t in 0..self.num_topics {
            for (w, c) in col_sum.iter_mut().enumerate() {
                *c += phi[[t, w]];
            }
        }

        // Rank-1 (dominant-topic) document counts.
        let mut rank1 = vec![0usize; self.num_topics];
        for d in 0..num_docs {
            let mut best = 0usize;
            let mut best_v = theta[[d, 0]];
            for t in 1..self.num_topics {
                if theta[[d, t]] > best_v {
                    best_v = theta[[d, t]];
                    best = t;
                }
            }
            rank1[best] += 1;
        }

        // Document-entropy accumulator: H_t = ln(T_t) - (1/T_t) Σ_d n_dt ln n_dt,
        // the entropy of each topic's token allocation across documents (MALLET's
        // `document_entropy`; lower = concentrated in few documents).
        let mut doc_ent_s = vec![0.0f64; self.num_topics];
        let mut tc = vec![0u32; self.num_topics];
        for topics in &model.doc_topics {
            for &t in topics {
                tc[t as usize] += 1;
            }
            for (t, c) in tc.iter_mut().enumerate() {
                if *c > 0 {
                    let cf = *c as f64;
                    doc_ent_s[t] += cf * cf.ln();
                    *c = 0;
                }
            }
        }

        // Corpus word distribution (for the corpus-distance diagnostic).
        let total_tokens: f64 = corpus.total_freqs.iter().map(|&c| c as f64).sum::<f64>().max(1.0);

        let list = PyList::empty_bound(py);
        for t in 0..self.num_topics {
            let topn = &tops[t];

            let mut excl = 0.0;
            for &w in topn {
                if col_sum[w] > 0.0 {
                    excl += phi[[t, w]] / col_sum[w];
                }
            }
            if !topn.is_empty() {
                excl /= topn.len() as f64;
            }

            // One pass over the vocabulary for the φ-entropy and the two
            // distribution distances (from uniform and from the corpus).
            let rowsum: f64 = (0..num_words).map(|w| phi[[t, w]]).sum();
            let mut h = 0.0;
            let mut uniform_dist = 0.0;
            let mut corpus_dist = 0.0;
            if rowsum > 0.0 {
                for w in 0..num_words {
                    let p = phi[[t, w]] / rowsum;
                    if p > 0.0 {
                        h -= p * p.ln();
                        uniform_dist += p * (p * num_words as f64).ln();
                        let q = corpus.total_freqs[w] as f64 / total_tokens;
                        if q > 0.0 {
                            corpus_dist += p * (p / q).ln();
                        }
                    }
                }
            }
            let effective_words = h.exp();

            let tt = model.tokens_per_topic[t] as f64;
            let document_entropy = if tt > 0.0 { tt.ln() - doc_ent_s[t] / tt } else { 0.0 };

            let words: Vec<String> = topn.iter().map(|&w| vocab[w].clone()).collect();

            let d = PyDict::new_bound(py);
            d.set_item("topic", t)?;
            d.set_item("tokens", model.tokens_per_topic[t])?;
            d.set_item("coherence", coh[t])?;
            d.set_item("exclusivity", excl)?;
            d.set_item("effective_words", effective_words)?;
            d.set_item("document_entropy", document_entropy)?;
            d.set_item("uniform_dist", uniform_dist)?;
            d.set_item("corpus_dist", corpus_dist)?;
            d.set_item("rank1_docs", rank1[t])?;
            d.set_item("alpha", model.alpha[t])?;
            d.set_item("top_words", words)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Infer document-topic distributions for *new, unseen* documents under the
    /// fitted model (sklearn-style `transform`). `data` is a :class:`Corpus` or
    /// `list[list[str]]`; tokens are matched to the training vocabulary by
    /// string (OOV dropped). A document with no in-vocabulary tokens gets the
    /// prior θ. Returns an array of shape ``(num_new_docs, num_topics)`` whose
    /// rows sum to 1.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iterations = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let (docs, _n, _oov) = self.map_heldout(data)?;
        let model = self.model.as_ref().unwrap();
        let k = self.num_topics;
        let mut rng = Pcg64Mcg::seed_from_u64(seed.unwrap_or(self.seed));

        let thetas: Vec<Vec<f64>> = py.allow_threads(move || {
            docs.iter()
                .map(|d| {
                    infer_doc(model, d, iterations, burn_in, num_samples, sample_interval, &mut rng)
                })
                .collect()
        });

        let mut arr = Array2::<f64>::zeros((thetas.len(), k));
        for (i, row) in thetas.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                arr[[i, t]] = v;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// The `n` training documents most strongly associated with `topic`, as
    /// ``(doc_name, weight)`` pairs sorted by descending θ for that topic.
    #[pyo3(signature = (topic, n=10))]
    fn top_documents<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic {} out of range (num_topics={})",
                topic, self.num_topics
            )));
        }
        let theta = self.theta.as_ref().unwrap();
        let names = &self.corpus.as_ref().unwrap().doc_names;
        let num_docs = theta.shape()[0];

        let mut idx: Vec<usize> = (0..num_docs).collect();
        idx.sort_by(|&a, &b| f64::total_cmp(&theta[[b, topic]], &theta[[a, topic]]));
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&d| {
                PyTuple::new_bound(
                    py,
                    &[names[d].clone().into_py(py), theta[[d, topic]].into_py(py)],
                )
            })
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Pairwise Jensen-Shannon divergence between topic-word distributions,
    /// shape ``(num_topics, num_topics)`` (base 2, in [0, 1]; 0 on the diagonal).
    /// Low off-diagonal values flag near-duplicate topics.
    #[getter]
    fn topic_divergence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let k = self.num_topics;
        let w = phi.shape()[1];

        // Normalize each topic row to a distribution over words.
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|t| {
                let s: f64 = (0..w).map(|i| phi[[t, i]]).sum();
                (0..w).map(|i| phi[[t, i]] / s).collect()
            })
            .collect();

        let mut arr = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            for b in (a + 1)..k {
                let d = js_divergence(&rows[a], &rows[b]);
                arr[[a, b]] = d;
                arr[[b, a]] = d;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// The `n` training documents most similar to document `doc` (by index),
    /// as ``(doc_name, divergence)`` pairs sorted by ascending Jensen-Shannon
    /// divergence of their document-topic distributions.
    #[pyo3(signature = (doc, n=10))]
    fn similar_documents<'py>(
        &self,
        py: Python<'py>,
        doc: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let theta = self.theta.as_ref().unwrap();
        let names = &self.corpus.as_ref().unwrap().doc_names;
        let num_docs = theta.shape()[0];
        let k = self.num_topics;
        if doc >= num_docs {
            return Err(PyValueError::new_err(format!(
                "doc {} out of range (num_docs={})",
                doc, num_docs
            )));
        }

        let target: Vec<f64> = (0..k).map(|t| theta[[doc, t]]).collect();
        let mut scored: Vec<(usize, f64)> = (0..num_docs)
            .filter(|&d| d != doc)
            .map(|d| {
                let q: Vec<f64> = (0..k).map(|t| theta[[d, t]]).collect();
                (d, js_divergence(&target, &q))
            })
            .collect();
        scored.sort_by(|a, b| f64::total_cmp(&a.1, &b.1));

        let items: Vec<Bound<'py, PyTuple>> = scored
            .iter()
            .take(n)
            .map(|&(d, div)| {
                PyTuple::new_bound(py, &[names[d].clone().into_py(py), div.into_py(py)])
            })
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Save the fitted model to `path` (compact binary). Reload with `LDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_LDA, &LdaState {
            num_topics: self.num_topics, alpha_sum: self.alpha_sum, beta: self.beta,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            num_threads: self.num_threads, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            model: self.model.clone(), corpus: self.corpus.clone(),
            use_symmetric_alpha: self.use_symmetric_alpha,
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            init_spectral: self.init_spectral,
            light: self.light, warp: self.warp, cvb0: self.cvb0,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: LdaState = read_state(path, MODEL_TAG_LDA)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(LDA {
            num_topics: s.num_topics, alpha_sum: s.alpha_sum, beta: s.beta,
            optimize_interval: s.optimize_interval, burn_in: s.burn_in, seed: s.seed,
            num_threads: s.num_threads, light: s.light, warp: s.warp, cvb0: s.cvb0,
            mh_steps: 2, fitted: s.fitted,
            use_symmetric_alpha: s.use_symmetric_alpha,
            init_spectral: s.init_spectral,
            topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            theta_draws: arr3f32_back(s.theta_draws),
            model: s.model, corpus: s.corpus,
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LDA(num_topics={}, beta={}, fitted={})",
            self.num_topics, self.beta, self.fitted
        )
    }
}

// ---------------------------------------------------------------------------
// Averaging helpers (ported from src/bin/train.rs)
// ---------------------------------------------------------------------------

/// Snapshot the current smoothed topic-word distribution into `acc[word][topic]`.
fn accumulate_phi(m: &TopicModel, acc: &mut [Vec<f64>]) {
    for word_id in 0..m.num_types {
        for topic in 0..m.num_topics {
            let count = m.get_type_topic_count(word_id, topic);
            let denom = m.tokens_per_topic[topic] as f64 + m.beta_sum;
            acc[word_id][topic] += (count as f64 + m.beta) / denom;
        }
    }
}

/// Snapshot the current smoothed document-topic distribution into `acc[doc][topic]`.
fn accumulate_theta(m: &TopicModel, c: &corpus::Corpus, acc: &mut [Vec<f64>]) {
    let mut counts = vec![0u32; m.num_topics];
    for doc_idx in 0..c.num_docs() {
        for t in counts.iter_mut() {
            *t = 0;
        }
        for &t in &m.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        let doc_len = c.docs[doc_idx].len() as f64;
        let denom = doc_len + m.alpha_sum;
        for t in 0..m.num_topics {
            acc[doc_idx][t] += (counts[t] as f64 + m.alpha[t]) / denom;
        }
    }
}

// ---------------------------------------------------------------------------
// Thinned MCMC theta-draw retention (issue #31)
// ---------------------------------------------------------------------------
//
// A single normalized θ snapshot from the current sampler state, kept as f32 to
// halve the (num_draws × D × K) store. Collected on a ring buffer during the
// main sweep loop so the retained draws are the converged tail of the chain;
// `composition_theta` then propagates real cross-sweep variance instead of the
// within-document Dirichlet approximation.

/// Thinning period given the run length and how many draws we want to keep:
/// take ~`2·cap` snapshots over the run so the kept `cap` (ring-buffered) sit in
/// the back half. `0` disables collection.
fn theta_draw_thin(iters: usize, cap: usize) -> usize {
    if cap == 0 {
        return 0;
    }
    (iters / (2 * cap)).max(1)
}

/// One normalized θ snapshot (D×K) as f32 from a `TopicModel`'s current counts.
fn theta_snapshot_f32(m: &TopicModel, c: &corpus::Corpus) -> Vec<Vec<f32>> {
    let mut counts = vec![0u32; m.num_topics];
    let mut out = Vec::with_capacity(c.num_docs());
    for doc_idx in 0..c.num_docs() {
        for t in counts.iter_mut() {
            *t = 0;
        }
        for &t in &m.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        let denom = c.docs[doc_idx].len() as f64 + m.alpha_sum;
        out.push(
            (0..m.num_topics)
                .map(|t| ((counts[t] as f64 + m.alpha[t]) / denom) as f32)
                .collect(),
        );
    }
    out
}

/// Push a draw onto a ring buffer that keeps only the last `cap`.
fn push_capped(buf: &mut Vec<Vec<Vec<f32>>>, draw: Vec<Vec<f32>>, cap: usize) {
    buf.push(draw);
    if buf.len() > cap {
        buf.remove(0);
    }
}

/// Generic training loop for any [`crate::mh::MhSampler`] backend (LightLDA,
/// WarpLDA, and future MH samplers). Runs `iters` sweeps with periodic θ-draw
/// thinning, hyperparameter optimization, and an optional progress callback,
/// then averages `num_samples` post-burn snapshots into φ/θ. Returns the
/// accumulators, thinned draws, packed model, and the (moved-through) corpus.
///
/// This is the single place the MH samplers' fit loop lives; the per-sampler
/// `LDA::fit` branches collapse to "construct the sampler, call this". Call it
/// inside `py.allow_threads`; the progress closure re-acquires the GIL itself.
#[allow(clippy::too_many_arguments)]
fn run_mh_training<S: crate::mh::MhSampler>(
    mut sampler: S,
    corpus: corpus::Corpus,
    num_topics: usize,
    num_types: usize,
    num_docs: usize,
    iters: usize,
    num_samples: usize,
    sample_interval: usize,
    burn_in: usize,
    optimize_interval: usize,
    use_symmetric_alpha: bool,
    draw_thin: usize,
    draw_cap: usize,
    total_tokens: f64,
    rng: &mut Pcg64Mcg,
    progress: &Option<PyObject>,
    progress_interval: usize,
) -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<f32>>>,
    TopicModel,
    corpus::Corpus,
) {
    let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();

    for iter in 1..=iters {
        sampler.sweep(&corpus, rng);

        if draw_thin > 0 && iter % draw_thin == 0 {
            let mut tmp = vec![vec![0.0f64; num_topics]; num_docs];
            sampler.theta_into(&corpus, &mut tmp);
            let snap = tmp
                .iter()
                .map(|r| r.iter().map(|&v| v as f32).collect())
                .collect();
            push_capped(&mut theta_draw_buf, snap, draw_cap);
        }

        if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
            let mut m = sampler.to_topic_model();
            if use_symmetric_alpha {
                optimize::optimize_alpha_symmetric(&mut m, &corpus);
            } else {
                optimize::optimize_alpha(&mut m, &corpus);
            }
            optimize::optimize_beta(&mut m);
            sampler.set_hyper(&m.alpha, m.beta);
        }

        if let Some(cb) = progress {
            if progress_interval > 0 && iter % progress_interval == 0 {
                let m = sampler.to_topic_model();
                let ll = output::model_log_likelihood(&m, &corpus) / total_tokens;
                Python::with_gil(|py| {
                    let _ = cb.call1(py, (iter, ll));
                });
            }
        }
    }

    let mut acc_phi = vec![vec![0.0f64; num_topics]; num_types];
    let mut acc_theta = vec![vec![0.0f64; num_topics]; num_docs];
    for _ in 0..num_samples {
        for _ in 0..sample_interval {
            sampler.sweep(&corpus, rng);
        }
        sampler.phi_into(&mut acc_phi);
        sampler.theta_into(&corpus, &mut acc_theta);
    }
    let n = (num_samples.max(1)) as f64;
    for row in acc_phi.iter_mut() {
        for v in row.iter_mut() { *v /= n; }
    }
    for row in acc_theta.iter_mut() {
        for v in row.iter_mut() { *v /= n; }
    }
    let model = sampler.to_topic_model();
    (acc_phi, acc_theta, theta_draw_buf, model, corpus)
}

/// Warn (once, before the heavy loop) when retaining θ draws would cost more
/// than ~512 MB, so a large corpus does not silently balloon memory. The user
/// can pass `keep_theta_draws=False` or a smaller `num_theta_draws`.
fn warn_theta_draw_memory(
    py: Python<'_>,
    keep: bool,
    num_draws: usize,
    num_docs: usize,
    num_topics: usize,
) -> PyResult<()> {
    if !keep || num_draws == 0 {
        return Ok(());
    }
    const THRESHOLD: usize = 512 * 1024 * 1024; // 512 MB of f32
    let bytes = num_draws
        .saturating_mul(num_docs)
        .saturating_mul(num_topics)
        .saturating_mul(4);
    if bytes > THRESHOLD {
        let mb = bytes / (1024 * 1024);
        let msg = format!(
            "keep_theta_draws will retain ~{mb} MB of MCMC theta draws \
             ({num_draws} draws x {num_docs} docs x {num_topics} topics, f32). \
             Pass keep_theta_draws=False, or a smaller num_theta_draws, to avoid this."
        );
        let warnings = py.import_bound("warnings")?;
        warnings.call_method1("warn", (msg,))?;
    }
    Ok(())
}

/// Stack the collected draws into an `(S, D, K)` array, or `None` if empty.
/// When `order` is given, row `i` of each draw is scattered to document
/// `order[i]` (the keyATM dynamic model fits on time-sorted documents, so its
/// draws come back sorted and must be unsorted to match the other outputs).
fn draws_to_array3(
    buf: &[Vec<Vec<f32>>],
    num_docs: usize,
    num_topics: usize,
    order: Option<&[usize]>,
) -> Option<Array3<f32>> {
    if buf.is_empty() {
        return None;
    }
    let mut arr = Array3::<f32>::zeros((buf.len(), num_docs, num_topics));
    for (s, draw) in buf.iter().enumerate() {
        for (i, row) in draw.iter().enumerate() {
            let d = order.map_or(i, |o| o[i]);
            for (t, &v) in row.iter().enumerate() {
                arr[[s, d, t]] = v;
            }
        }
    }
    Some(arr)
}

// ---------------------------------------------------------------------------
// Parallel Gibbs sampling (MALLET-style approximate distributed sampling)
// ---------------------------------------------------------------------------

/// Split `n` items into up to `parts` contiguous, balanced ranges.
fn partition_ranges(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(n.max(1));
    let base = n / parts;
    let rem = n % parts;
    let mut ranges = Vec::with_capacity(parts);
    let mut start = 0;
    for i in 0..parts {
        let len = base + if i < rem { 1 } else { 0 };
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

struct WorkerOut {
    ttc: Vec<Vec<u32>>,
    tpt: Vec<u32>,
    start: usize,
    dt: Vec<Vec<u32>>,
}

/// One approximate-parallel Gibbs sweep. Documents are partitioned across
/// `num_threads` workers; each samples its slice against a private copy of the
/// topic-word counts (so workers don't see each other's within-sweep updates),
/// then the per-worker count changes are reconciled exactly into the global
/// model. Token bookkeeping stays consistent (each token belongs to exactly one
/// worker); only the sampling distribution is approximated. `sweep_seed` makes
/// the result deterministic for a fixed `num_threads`.
/// Approximate-parallel sampling with a deferred merge (turbo mode). Each worker
/// runs `batch` consecutive Gibbs sweeps over its document partition against its
/// own private count tables before any reconciliation, so the global topic-word
/// counts are synchronized once per `batch` sweeps instead of every sweep. This
/// trades a small amount of mixing accuracy (workers sample against staler
/// cross-partition counts the deeper they are into a batch) for `batch`× fewer
/// O(V·K) reconciliations and per-worker table clones — the per-sweep merge is
/// the thread-scaling ceiling, so deferring it lifts that ceiling. With
/// `batch == 1` this is the exact per-sweep path and is bit-identical to it.
///
/// The reconciliation is the same exact additive formula used by the per-sweep
/// path: each worker only ever touches its own documents' tokens relative to the
/// shared `original` snapshot, so summing the workers and subtracting (W−1)
/// copies of the snapshot recovers the combined state regardless of how many
/// sweeps each worker ran. Determinism for a fixed `num_threads`/`batch` is
/// preserved: each worker seeds one RNG from `sweep_seed` and draws from it
/// across all `batch` internal sweeps.
fn parallel_sweep_batched(
    model: &mut TopicModel,
    docs: &[Vec<u32>],
    num_threads: usize,
    sweep_seed: u64,
    batch: usize,
) {
    let batch = batch.max(1);
    let k = model.num_topics;
    let mask = model.topic_mask;
    let bits = model.topic_bits;
    let beta = model.beta;
    let beta_sum = model.beta_sum;
    let v = model.num_types;
    let ranges = partition_ranges(docs.len(), num_threads);

    // Snapshot for reconciliation, and clone the shared read-only inputs.
    let original_ttc = model.type_topic_counts.clone();
    let original_tpt = model.tokens_per_topic.clone();
    let alpha = model.alpha.clone();
    let dt_all = &model.doc_topics;

    // --- Workers: each samples its document partition independently. ---
    let outs: Vec<WorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut ttc = original_ttc.clone();
            let mut tpt = original_tpt.clone();
            let mut dt: Vec<Vec<u32>> = dt_all[start..end].to_vec();
            let mut rng = Pcg64Mcg::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            for _ in 0..batch {
                sampler::run_sweep(
                    &mut ttc,
                    &mut tpt,
                    &mut dt,
                    &docs[start..end],
                    &alpha,
                    beta,
                    beta_sum,
                    mask,
                    bits,
                    k,
                    &mut rng,
                );
            }
            WorkerOut { ttc, tpt, start, dt }
        })
        .collect();

    // --- Reconcile topic-word counts. Each worker started from `original` and
    // only changed its own documents' tokens, so the exact global state is
    //     final = Σ_w worker_w − (W−1)·original
    // computed densely per word (in parallel, reusing a per-thread accumulator
    // to avoid per-word allocation). Re-encode into the packed layout. ---
    let wm1 = (outs.len() as i64) - 1;
    let new_ttc: Vec<Vec<u32>> = (0..v)
        .into_par_iter()
        .map_init(
            || vec![0i64; k],
            |acc, w| {
                for a in acc.iter_mut() {
                    *a = 0;
                }
                for out in &outs {
                    for &e in &out.ttc[w] {
                        if e == 0 {
                            break;
                        }
                        acc[(e & mask) as usize] += (e >> bits) as i64;
                    }
                }
                for &e in &original_ttc[w] {
                    if e == 0 {
                        break;
                    }
                    acc[(e & mask) as usize] -= wm1 * (e >> bits) as i64;
                }
                let mut entries: Vec<u32> = (0..k)
                    .filter(|&t| acc[t] > 0)
                    .map(|t| ((acc[t] as u32) << bits) | (t as u32))
                    .collect();
                entries.sort_unstable_by(|a, b| b.cmp(a));
                let len = original_ttc[w].len();
                entries.resize(len.max(entries.len()), 0);
                entries
            },
        )
        .collect();
    model.type_topic_counts = new_ttc;

    // --- Reconcile tokens-per-topic the same way. ---
    let mut tpt: Vec<i64> = original_tpt.iter().map(|&c| c as i64).collect();
    for out in &outs {
        for t in 0..k {
            tpt[t] += out.tpt[t] as i64 - original_tpt[t] as i64;
        }
    }
    model.tokens_per_topic = tpt.iter().map(|&c| c.max(0) as u32).collect();

    // --- Write each worker's updated topic assignments back into place. ---
    for out in outs {
        let start = out.start;
        for (i, row) in out.dt.into_iter().enumerate() {
            model.doc_topics[start + i] = row;
        }
    }
}

// ---------------------------------------------------------------------------
// Held-out evaluation: Wallach et al. (2009) left-to-right estimator
// ---------------------------------------------------------------------------

/// Dense per-topic φ(word | topic) vectors for each distinct word in `doc`,
/// under the fixed trained model. Decoding each word's packed sparse entries
/// once removes the repeated O(K) linear scans of `get_type_topic_count` from
/// the inference/eval inner loops — an exact (value-identical) speedup that
/// matters once K is large.
fn build_phi_cache(model: &TopicModel, doc: &[usize]) -> HashMap<usize, Vec<f64>> {
    let k = model.num_topics;
    let denom: Vec<f64> = (0..k)
        .map(|t| model.beta_sum + model.tokens_per_topic[t] as f64)
        .collect();

    let mut cache: HashMap<usize, Vec<f64>> = HashMap::new();
    for &w in doc {
        cache.entry(w).or_insert_with(|| {
            let mut dense = vec![0u32; k];
            for &entry in &model.type_topic_counts[w] {
                if entry == 0 {
                    break;
                }
                let t = (entry & model.topic_mask) as usize;
                dense[t] = entry >> model.topic_bits;
            }
            (0..k)
                .map(|t| (model.beta + dense[t] as f64) / denom[t])
                .collect()
        });
    }
    cache
}

/// Sample an index in proportion to non-negative `weights` summing to `total`.
fn sample_categorical<R: Rng>(weights: &[f64], total: f64, rng: &mut R) -> usize {
    let mut u = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// Estimate log P(doc) under a fixed trained model with the left-to-right
/// estimator. `doc` holds trained-vocabulary word ids (OOV already dropped).
///
/// Per token position, the probability is averaged across `num_particles`
/// particles (each a left-to-right pass that resamples earlier positions);
/// the document log-likelihood is the sum of the logs of those averages.
/// The trained topic-word counts stay fixed — only per-document topic counts
/// evolve.
fn left_to_right_doc<R: Rng>(
    model: &TopicModel,
    doc: &[usize],
    num_particles: usize,
    rng: &mut R,
) -> f64 {
    let k = model.num_topics;
    let n = doc.len();
    if n == 0 {
        return 0.0;
    }
    let alpha = &model.alpha;
    let alpha_sum = model.alpha_sum;
    let phi_cache = build_phi_cache(model, doc);

    let mut word_prob = vec![0.0f64; n]; // accumulated across particles
    let mut weights = vec![0.0f64; k];

    for _ in 0..num_particles {
        let mut local = vec![0u32; k]; // per-document topic counts
        let mut z = vec![0usize; n]; // topic assigned to each position this pass

        for pos in 0..n {
            // Resample the topic of every earlier token given the current state.
            for prev in 0..pos {
                let phi = &phi_cache[&doc[prev]];
                local[z[prev]] -= 1;
                let mut total = 0.0;
                for t in 0..k {
                    let val = (alpha[t] + local[t] as f64) * phi[t];
                    weights[t] = val;
                    total += val;
                }
                let t_new = sample_categorical(&weights, total, rng);
                z[prev] = t_new;
                local[t_new] += 1;
            }

            // Score the current token: p(w_pos) = (Σ_t weight_t)/(alpha_sum+pos).
            let phi = &phi_cache[&doc[pos]];
            let mut total = 0.0;
            for t in 0..k {
                let val = (alpha[t] + local[t] as f64) * phi[t];
                weights[t] = val;
                total += val;
            }
            word_prob[pos] += total / (alpha_sum + pos as f64);

            // Sample this token's topic and fold it into the local counts.
            let t_new = sample_categorical(&weights, total, rng);
            z[pos] = t_new;
            local[t_new] += 1;
        }
    }

    let mut ll = 0.0;
    let r = num_particles as f64;
    for p in &word_prob {
        ll += (p / r).ln();
    }
    ll
}

/// UMass coherence for each topic's top-word list (descending by probability).
/// `C = Σ_{i>j} log[(codoc(w_i,w_j)+1) / docfreq(w_j)]`, intrinsic to the
/// training corpus.
fn umass_coherence(corpus: &corpus::Corpus, tops: &[Vec<usize>]) -> Vec<f64> {
    use std::collections::HashSet;

    let relevant: HashSet<usize> = tops.iter().flatten().copied().collect();
    let mut codoc: HashMap<(usize, usize), u32> = HashMap::new();

    for doc in &corpus.docs {
        let mut present: Vec<usize> = doc
            .iter()
            .map(|&w| w as usize)
            .filter(|w| relevant.contains(w))
            .collect();
        present.sort_unstable();
        present.dedup();
        for a in 0..present.len() {
            for b in (a + 1)..present.len() {
                *codoc.entry((present[a], present[b])).or_insert(0) += 1;
            }
        }
    }

    tops.iter()
        .map(|top| {
            let mut score = 0.0;
            for i in 1..top.len() {
                for j in 0..i {
                    let (wi, wj) = (top[i], top[j]); // wj is the more probable word
                    let key = if wi < wj { (wi, wj) } else { (wj, wi) };
                    let co = *codoc.get(&key).unwrap_or(&0) as f64;
                    let dfj = corpus.doc_freqs[wj].max(1) as f64;
                    score += ((co + 1.0) / dfj).ln();
                }
            }
            score
        })
        .collect()
}

/// Infer a document-topic distribution for a *new* document under a fixed
/// trained model (the MALLET TopicInferencer approach): run Gibbs over the
/// document's tokens, sampling each topic from
/// `(alpha_t + n_{t,doc}) * (beta + N_{w,t})/(beta_sum + tokens_per_topic_t)`
/// while the trained topic-word counts stay frozen, then average θ snapshots.
fn infer_doc<R: Rng>(
    model: &TopicModel,
    doc: &[usize],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    rng: &mut R,
) -> Vec<f64> {
    let k = model.num_topics;
    let alpha = &model.alpha;
    let alpha_sum = model.alpha_sum;
    let n = doc.len();

    let mut theta = vec![0.0f64; k];
    if n == 0 {
        // No in-vocabulary tokens: fall back to the prior.
        for t in 0..k {
            theta[t] = alpha[t] / alpha_sum;
        }
        return theta;
    }

    let phi_cache = build_phi_cache(model, doc);

    let mut local = vec![0u32; k];
    let mut z = vec![0usize; n];
    for i in 0..n {
        let t = rng.gen_range(0..k);
        z[i] = t;
        local[t] += 1;
    }

    let mut weights = vec![0.0f64; k];
    let mut samples_taken = 0usize;
    for iter in 1..=iterations {
        for i in 0..n {
            let phi = &phi_cache[&doc[i]];
            local[z[i]] -= 1;
            let mut total = 0.0;
            for t in 0..k {
                let v = (alpha[t] + local[t] as f64) * phi[t];
                weights[t] = v;
                total += v;
            }
            let t_new = sample_categorical(&weights, total, rng);
            z[i] = t_new;
            local[t_new] += 1;
        }
        if iter > burn_in
            && samples_taken < num_samples
            && (iter - burn_in) % sample_interval.max(1) == 0
        {
            let denom = n as f64 + alpha_sum;
            for t in 0..k {
                theta[t] += (local[t] as f64 + alpha[t]) / denom;
            }
            samples_taken += 1;
        }
    }

    if samples_taken == 0 {
        let denom = n as f64 + alpha_sum;
        for t in 0..k {
            theta[t] = (local[t] as f64 + alpha[t]) / denom;
        }
    } else {
        for t in theta.iter_mut() {
            *t /= samples_taken as f64;
        }
    }
    theta
}

/// Collapsed-Gibbs inference of a held-out document's topic proportions θ
/// against a *fixed* normalized topic-word matrix `phi` (K rows, each a
/// distribution over the vocabulary) and a Dirichlet prior `alpha` (length K).
/// This is the model-agnostic counterpart to [`infer_doc`]: any Gibbs model
/// that exposes a normalized topic-word matrix can reuse it for `transform`.
fn infer_theta_gibbs<R: Rng>(
    phi: &[Vec<f64>],
    alpha: &[f64],
    doc: &[usize],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    rng: &mut R,
) -> Vec<f64> {
    let k = phi.len();
    let alpha_sum: f64 = alpha.iter().sum();
    let n = doc.len();
    let mut theta = vec![0.0f64; k];
    if n == 0 {
        for t in 0..k {
            theta[t] = alpha[t] / alpha_sum;
        }
        return theta;
    }

    // Per-token phi column (probability of this word under each topic).
    let cols: Vec<Vec<f64>> = doc.iter().map(|&w| (0..k).map(|t| phi[t][w]).collect()).collect();

    let mut local = vec![0u32; k];
    let mut z = vec![0usize; n];
    for i in 0..n {
        let t = rng.gen_range(0..k);
        z[i] = t;
        local[t] += 1;
    }

    let mut weights = vec![0.0f64; k];
    let mut samples_taken = 0usize;
    for iter in 1..=iterations {
        for i in 0..n {
            let col = &cols[i];
            local[z[i]] -= 1;
            let mut total = 0.0;
            for t in 0..k {
                let v = (alpha[t] + local[t] as f64) * col[t];
                weights[t] = v;
                total += v;
            }
            let t_new = sample_categorical(&weights, total, rng);
            z[i] = t_new;
            local[t_new] += 1;
        }
        if iter > burn_in
            && samples_taken < num_samples
            && (iter - burn_in) % sample_interval.max(1) == 0
        {
            let denom = n as f64 + alpha_sum;
            for t in 0..k {
                theta[t] += (local[t] as f64 + alpha[t]) / denom;
            }
            samples_taken += 1;
        }
    }

    if samples_taken == 0 {
        let denom = n as f64 + alpha_sum;
        for t in 0..k {
            theta[t] = (local[t] as f64 + alpha[t]) / denom;
        }
    } else {
        for t in theta.iter_mut() {
            *t /= samples_taken as f64;
        }
    }
    theta
}

/// Resolve the `iters`/`iterations` deprecation for `transform()`: if the
/// caller passed the old `iterations` keyword, emit a `DeprecationWarning` and
/// use that value; otherwise use `iters`. When both are supplied `iters` wins
/// (the caller is already using the new name) but the warning still fires.
fn resolve_iters_deprecated(
    py: Python<'_>,
    iters: usize,
    iterations: Option<usize>,
) -> PyResult<usize> {
    if let Some(old_val) = iterations {
        let warnings = py.import_bound("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "transform(iterations=) is deprecated; use iters= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ),
        )?;
        // iters wins when both are supplied (caller already migrated), but we
        // still fire the warning because `iterations` was passed explicitly.
        if iters != 100 {
            Ok(iters)
        } else {
            Ok(old_val)
        }
    } else {
        Ok(iters)
    }
}

/// Batch wrapper for [`infer_theta_gibbs`]: maps new docs to ids, runs the
/// sampler per document (parallel, seeded deterministically per doc), and
/// returns a ``(num_docs, K)`` array. `alpha` is the length-K prior.
fn transform_gibbs<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    id_to_word: &[String],
    phi: &Array2<f64>,
    alpha: &[f64],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    base_seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let docs = docs_to_ids(data, id_to_word)?;
    let docs_usize: Vec<Vec<usize>> =
        docs.iter().map(|d| d.iter().map(|&w| w as usize).collect()).collect();
    let phi_rows: Vec<Vec<f64>> = phi.outer_iter().map(|r| r.to_vec()).collect();
    let alpha_v = alpha.to_vec();
    let k = phi_rows.len();

    let rows: Vec<Vec<f64>> = py.allow_threads(|| {
        docs_usize
            .par_iter()
            .enumerate()
            .map(|(i, d)| {
                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                infer_theta_gibbs(
                    &phi_rows, &alpha_v, d, iterations, burn_in, num_samples, sample_interval,
                    &mut rng,
                )
            })
            .collect()
    });

    let mut arr = Array2::<f64>::zeros((rows.len(), k));
    for (i, row) in rows.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            arr[[i, t]] = v;
        }
    }
    Ok(arr.to_pyarray_bound(py))
}

/// Jensen-Shannon divergence (base 2, in [0, 1]) between two distributions.
fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let mut d = 0.0;
    for i in 0..p.len() {
        let m = 0.5 * (p[i] + q[i]);
        if p[i] > 0.0 && m > 0.0 {
            d += 0.5 * p[i] * (p[i] / m).log2();
        }
        if q[i] > 0.0 && m > 0.0 {
            d += 0.5 * q[i] * (q[i] / m).log2();
        }
    }
    d.max(0.0)
}

// ---------------------------------------------------------------------------
// DMR: Dirichlet-Multinomial Regression topic model
// ---------------------------------------------------------------------------

/// Per-document topic-count vectors `[num_docs][num_topics]`.
fn doc_topic_counts(doc_topics: &[Vec<u32>], k: usize) -> Vec<Vec<f64>> {
    doc_topics
        .iter()
        .map(|topics| {
            let mut c = vec![0.0f64; k];
            for &t in topics {
                c[t as usize] += 1.0;
            }
            c
        })
        .collect()
}

/// Convert a `Vec<Vec<f64>>` (rows) into an `Array2`.
fn vecs_to_arr2(rows: &[Vec<f64>]) -> Array2<f64> {
    let r = rows.len();
    let c = if r > 0 { rows[0].len() } else { 0 };
    let mut a = Array2::<f64>::zeros((r, c));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    a
}

/// Shared `top_words(n, topic=None)` implementation over a φ matrix: returns a
/// list of `(word, prob)` for one topic, or a list of those lists for all.
fn topic_words_helper<'py>(
    py: Python<'py>,
    beta: &Array2<f64>,
    vocab: &[String],
    num_topics: usize,
    n: usize,
    topic: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let tops = top_word_ids_phi(beta, num_topics, n);
    let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
        if t >= num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let items: Vec<Bound<'py, PyTuple>> = tops[t]
            .iter()
            .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    };
    match topic {
        Some(t) => Ok(one(t)?.into_any()),
        None => {
            let all: Vec<Bound<'py, PyList>> = (0..num_topics).map(one).collect::<PyResult<_>>()?;
            Ok(PyList::new_bound(py, all).into_any())
        }
    }
}

/// Top-`n` word ids per topic from a (num_topics, num_words) φ matrix.
fn top_word_ids_phi(phi: &Array2<f64>, num_topics: usize, n: usize) -> Vec<Vec<usize>> {
    let w = phi.shape()[1];
    (0..num_topics)
        .map(|t| {
            let mut idx: Vec<usize> = (0..w).collect();
            idx.sort_by(|&a, &b| f64::total_cmp(&phi[[t, b]], &phi[[t, a]]));
            idx.truncate(n);
            idx
        })
        .collect()
}

/// Parse a feature matrix (a 2D numpy float array or a list of float lists)
/// into `[num_docs][num_features]`.
fn parse_features(data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let a = arr.as_array();
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        return Ok((0..rows)
            .map(|i| (0..cols).map(|j| a[[i, j]]).collect())
            .collect());
    }
    data.extract::<Vec<Vec<f64>>>().map_err(|_| {
        PyValueError::new_err("features must be a 2D float array or a list of float lists")
    })
}

/// Map a per-document timestamp sequence to contiguous 0-based time-segment
/// indices plus the sorted, distinct labels. Accepts numbers or strings; the
/// distinct values are sorted to define the time order. Returns
/// `(time_index_per_doc, labels)`.
fn build_time_index(
    data: &Bound<'_, PyAny>,
    num_docs: usize,
) -> PyResult<(Vec<usize>, Vec<String>)> {
    // Numeric timestamps (e.g. years) — sort numerically.
    if let Ok(vals) = data.extract::<Vec<f64>>() {
        if vals.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "timestamps has {} entries but corpus has {} documents",
                vals.len(),
                num_docs
            )));
        }
        check_all_finite_1d("timestamps", &vals)?;
        let mut uniq = vals.clone();
        uniq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        uniq.dedup();
        let idx: Vec<usize> = vals
            .iter()
            .map(|v| uniq.iter().position(|u| u == v).unwrap())
            .collect();
        let labels = uniq
            .iter()
            .map(|&u| {
                if u.fract() == 0.0 {
                    format!("{}", u as i64)
                } else {
                    format!("{u}")
                }
            })
            .collect();
        return Ok((idx, labels));
    }
    // String timestamps — sort lexicographically.
    if let Ok(vals) = data.extract::<Vec<String>>() {
        if vals.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "timestamps has {} entries but corpus has {} documents",
                vals.len(),
                num_docs
            )));
        }
        let mut uniq = vals.clone();
        uniq.sort();
        uniq.dedup();
        let idx: Vec<usize> = vals
            .iter()
            .map(|v| uniq.iter().position(|u| u == v).unwrap())
            .collect();
        return Ok((idx, uniq));
    }
    Err(PyValueError::new_err(
        "timestamps must be a sequence of numbers or strings, one per document",
    ))
}

/// Dirichlet-Multinomial Regression topic model (Mimno & McCallum, 2008).
///
/// Like :class:`LDA`, but the per-document topic prior is a log-linear function
/// of document features: ``α_{d,t} = exp(λ_t · x_d)``. After fitting, the
/// learned weights are available as :attr:`feature_effects` — how each covariate
/// shifts each topic's prevalence.
#[pyclass(module = "topica")]
pub struct DMR {
    num_topics: usize,
    beta: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    prior_variance: f64,
    lbfgs_iters: usize,
    // WarpLDA cache-efficient sampler (per-document-α doc phase) instead of the
    // default SparseLDA DMR sweep. Recommended for large K.
    warp: bool,
    // CVB0 deterministic collapsed-variational inference (per-document α).
    cvb0: bool,

    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,            // (num_topics, num_words)
    theta: Option<Array2<f64>>,          // (num_docs, num_topics)
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
    feature_effects: Option<Array2<f64>>, // (num_topics, num_features)
    feature_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl DMR {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl DMR {
    /// Create an unfitted DMR model. `prior_variance` is the Gaussian prior
    /// variance σ² on the feature weights λ (smaller = stronger shrinkage);
    /// `lbfgs_iters` caps the L-BFGS steps per optimization round.
    #[new]
    #[pyo3(signature = (num_topics, *, beta=0.01, optimize_interval=50,
                        burn_in=200, seed=42, prior_variance=1.0, lbfgs_iters=20,
                        sampler="sparse"))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        beta: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        prior_variance: f64,
        lbfgs_iters: usize,
        sampler: &str,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if !finite_pos(beta) {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        if !finite_pos(prior_variance) {
            return Err(PyValueError::new_err("prior_variance must be > 0"));
        }
        let (warp, cvb0) = match sampler {
            "sparse" => (false, false),
            "warp" | "warplda" => (true, false),
            "cvb0" | "cvb" => (false, true),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\", \"warp\", or \"cvb0\""
                )))
            }
        };
        Ok(DMR {
            num_topics,
            beta,
            optimize_interval,
            burn_in,
            seed,
            prior_variance,
            lbfgs_iters,
            warp,
            cvb0,
            fitted: false,
            topic_names: Vec::new(),
            phi: None,
            theta: None,
            theta_draws: None,
            feature_effects: None,
            feature_names: Vec::new(),
            corpus: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `features` is a `(num_docs, F)` numpy array or list of float lists (an
    /// intercept column is prepended automatically). `feature_names` (length F)
    /// names the columns; an "intercept" name is prepended.
    /// `covariates` is accepted as a no-deprecation alias for `features`.
    #[pyo3(signature = (data, features=None, *, feature_names=None, iters=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize, covariates=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        features: Option<&Bound<'_, PyAny>>,
        feature_names: Option<Vec<String>>,
        iters: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
        covariates: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // covariates= is a no-deprecation alias for features=
        let features: &Bound<'_, PyAny> = match (features, covariates) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "DMR.fit: pass either features= or covariates=, not both",
                ));
            }
            (Some(f), None) => f,
            (None, Some(c)) => c,
            (None, None) => {
                return Err(PyValueError::new_err(
                    "DMR.fit: features (or covariates) is required",
                ));
            }
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let raw = parse_features(features)?;
        if raw.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "features has {} rows but corpus has {} documents",
                raw.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("features", &raw)?;
        let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
        if raw.iter().any(|r| r.len() != f_in) {
            return Err(PyValueError::new_err("all feature rows must have the same length"));
        }
        if let Some(names) = &feature_names {
            if names.len() != f_in {
                return Err(PyValueError::new_err(format!(
                    "feature_names has {} entries but features has {} columns",
                    names.len(),
                    f_in
                )));
            }
        }

        // Prepend an intercept column.
        let nf = f_in + 1;
        let feats: Vec<Vec<f64>> = raw
            .iter()
            .map(|x| {
                let mut v = Vec::with_capacity(nf);
                v.push(1.0);
                v.extend_from_slice(x);
                v
            })
            .collect();
        let mut names = vec!["intercept".to_string()];
        names.extend(
            feature_names.unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
        );

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let num_docs = corpus.num_docs();
        let total_tokens = corpus.total_tokens().max(1) as f64;

        // λ starts at zero -> α ≡ 1 (symmetric) before optimization kicks in.
        let mut lambda = vec![vec![0.0f64; nf]; k];
        let mut model = TopicModel::new(k, k as f64, self.beta, num_types);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        // The sparse path initializes `model` inside its branch below; the warp
        // path builds its own WarpLda state, so the shared init is deferred.

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        let prior_variance = self.prior_variance;
        let lbfgs_iters = self.lbfgs_iters;
        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let beta = self.beta;
        let warp = self.warp;
        let cvb0_flag = self.cvb0;
        let (acc_phi, acc_theta, theta_draw_buf, feat_eff, ll_history, converged_flag, model, corpus) =
          if cvb0_flag {
            // CVB0 DMR: deterministic; per-document α is fed to the CVB0 sweep,
            // and the soft expected counts E[n_dk] feed the λ optimizer directly.
            // No MCMC, so no θ-draws and no convergence trace.
            py.allow_threads(move || {
                let alpha0 = vec![1.0f64; k];
                let mut cv = cvb0::Cvb0::new(&corpus, k, &alpha0, beta, &mut rng);
                for iter in 1..=iters {
                    let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats, None);
                    cv.set_doc_alpha(doc_alpha);
                    cv.sweep();
                    if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                        dmr::optimize_lambda(
                            &mut lambda, &feats, cv.doc_topic_expected(), k, nf,
                            prior_variance, lbfgs_iters, None,
                        );
                    }
                }
                let mut acc_phi = vec![vec![0.0f64; k]; num_types];
                let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
                cv.phi_into(&mut acc_phi);
                cv.theta_into(&mut acc_theta);
                let model = cv.to_topic_model(&corpus);
                (acc_phi, acc_theta, Vec::new(), lambda, Vec::new(), false, model, corpus)
            })
          } else if warp {
            // WarpLDA DMR path: same λ-optimization loop, but the per-document
            // prior α is fed to the WarpLDA per-doc doc phase each sweep. Like the
            // LDA WarpLDA path it computes no inline log_likelihood, so the
            // convergence trace / convergence_tol are not recorded here.
            py.allow_threads(move || {
                let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut ws = warplda::WarpLda::new(&corpus, k, &vec![1.0f64; k], beta, &mut rng);

                for iter in 1..=iters {
                    let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats, None);
                    ws.set_doc_alpha(doc_alpha);
                    ws.sweep(&corpus, &mut rng);

                    if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                        let dtc = doc_topic_counts(ws.doc_topics(), k);
                        dmr::optimize_lambda(
                            &mut lambda, &feats, &dtc, k, nf, prior_variance, lbfgs_iters, None,
                        );
                    }

                    if draws_opts.thin > 0 && iter % draws_opts.thin == 0 {
                        let doc_alpha_snap = dmr::compute_doc_alpha(&lambda, &feats, None);
                        let snap: Vec<Vec<f32>> = ws.doc_topics().iter()
                            .enumerate()
                            .map(|(d, topics)| {
                                let mut c = vec![0.0f64; k];
                                for &t in topics { c[t as usize] += 1.0; }
                                let asum: f64 = doc_alpha_snap[d].iter().sum();
                                let denom = c.iter().sum::<f64>() + asum;
                                (0..k).map(|t| ((c[t] + doc_alpha_snap[d][t]) / denom) as f32).collect()
                            })
                            .collect();
                        push_capped(&mut theta_draw_buf, snap, draws_opts.cap);
                    }

                    if let Some(cb) = &progress {
                        if progress_interval > 0 && iter % progress_interval == 0 {
                            let dtc = doc_topic_counts(ws.doc_topics(), k);
                            let (ll, _) = dmr::dmr_objective_and_gradient(
                                &lambda, &feats, &dtc, k, nf, prior_variance, None,
                            );
                            let llpt = ll / total_tokens;
                            Python::with_gil(|py| {
                                let _ = cb.call1(py, (iter, llpt));
                            });
                        }
                    }
                }

                // Sampling phase: λ (and thus α per doc) fixed.
                let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats, None);
                ws.set_doc_alpha(doc_alpha.clone());
                let mut acc_phi = vec![vec![0.0f64; k]; num_types];
                let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
                for _ in 0..num_samples {
                    for _ in 0..sample_interval {
                        ws.sweep(&corpus, &mut rng);
                    }
                    ws.phi_into(&mut acc_phi);
                    let counts = doc_topic_counts(ws.doc_topics(), k);
                    for d in 0..num_docs {
                        let asum: f64 = doc_alpha[d].iter().sum();
                        let denom = corpus.docs[d].len() as f64 + asum;
                        for t in 0..k {
                            acc_theta[d][t] += (counts[d][t] as f64 + doc_alpha[d][t]) / denom;
                        }
                    }
                }
                let n = num_samples.max(1) as f64;
                for row in acc_phi.iter_mut() {
                    for v in row.iter_mut() { *v /= n; }
                }
                for row in acc_theta.iter_mut() {
                    for v in row.iter_mut() { *v /= n; }
                }
                let model = ws.to_topic_model();
                (acc_phi, acc_theta, theta_draw_buf, lambda, Vec::new(), false, model, corpus)
            })
          } else {
            model.initialize(&corpus, &mut rng);
            py.allow_threads(move || {
            let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut ll_history: Vec<(usize, f64)> = Vec::new();
            let mut converged_flag = false;

            'outer: for iter in 1..=iters {
                let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats, None);
                dmr::run_sweep_dmr(
                    &mut model.type_topic_counts,
                    &mut model.tokens_per_topic,
                    &mut model.doc_topics,
                    &corpus.docs,
                    &doc_alpha,
                    model.beta,
                    model.beta_sum,
                    model.topic_mask,
                    model.topic_bits,
                    k,
                    &mut rng,
                );

                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    let dtc = doc_topic_counts(&model.doc_topics, k);
                    dmr::optimize_lambda(
                        &mut lambda, &feats, &dtc, k, nf, prior_variance, lbfgs_iters, None,
                    );
                }

                // Snapshot θ = (n_dk + α_dk) / (N_d + Σα_d) every thin sweeps.
                if draws_opts.thin > 0 && iter % draws_opts.thin == 0 {
                    let doc_alpha_snap = dmr::compute_doc_alpha(&lambda, &feats, None);
                    let snap: Vec<Vec<f32>> = model.doc_topics.iter()
                        .enumerate()
                        .map(|(d, topics)| {
                            let mut c = vec![0.0f64; k];
                            for &t in topics { c[t as usize] += 1.0; }
                            let asum: f64 = doc_alpha_snap[d].iter().sum();
                            let denom = c.iter().sum::<f64>() + asum;
                            (0..k).map(|t| ((c[t] + doc_alpha_snap[d][t]) / denom) as f32).collect()
                        })
                        .collect();
                    push_capped(&mut theta_draw_buf, snap, draws_opts.cap);
                }

                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let dtc = doc_topic_counts(&model.doc_topics, k);
                        let (ll, _) = dmr::dmr_objective_and_gradient(
                            &lambda, &feats, &dtc, k, nf, prior_variance, None,
                        );
                        let llpt = ll / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, llpt));
                        });
                    }
                }

                // Trace recording and optional convergence check (never alters RNG).
                if check_every > 0 && iter % check_every == 0 {
                    let ll = output::model_log_likelihood(&model, &corpus);
                    ll_history.push((iter, ll));
                    if convergence_tol > 0.0 && ll_history.len() >= 2 {
                        let prev = ll_history[ll_history.len() - 2].1;
                        let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                        if rel < convergence_tol {
                            converged_flag = true;
                            break 'outer;
                        }
                    }
                }
            }

            // Sampling phase: λ is now fixed, so α per doc is fixed too.
            let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats, None);
            let mut acc_phi = vec![vec![0.0f64; k]; num_types];
            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];

            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    dmr::run_sweep_dmr(
                        &mut model.type_topic_counts,
                        &mut model.tokens_per_topic,
                        &mut model.doc_topics,
                        &corpus.docs,
                        &doc_alpha,
                        model.beta,
                        model.beta_sum,
                        model.topic_mask,
                        model.topic_bits,
                        k,
                        &mut rng,
                    );
                }
                accumulate_phi(&model, &mut acc_phi);
                // DMR θ uses the per-document prior.
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let asum: f64 = doc_alpha[d].iter().sum();
                    let denom = corpus.docs[d].len() as f64 + asum;
                    for t in 0..k {
                        acc_theta[d][t] += (counts[d][t] as f64 + doc_alpha[d][t]) / denom;
                    }
                }
            }

            let n = num_samples.max(1) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }

            (acc_phi, acc_theta, theta_draw_buf, lambda, ll_history, converged_flag, model, corpus)
            })
          };
        let _ = model;

        let mut phi = Array2::<f64>::zeros((k, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                phi[[t, w]] = val;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }
        let mut fe = Array2::<f64>::zeros((k, nf));
        for (t, row) in feat_eff.iter().enumerate() {
            for (f, &val) in row.iter().enumerate() {
                fe[[t, f]] = val;
            }
        }

        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, k, None);
        self.feature_effects = Some(fe);
        self.feature_names = names;
        self.log_likelihood_history = ll_history;
        self.converged = converged_flag;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Thinned MCMC θ draws, shape ``(num_draws, num_docs, num_topics)``, or
    /// ``None`` when fit with ``keep_theta_draws=False``. These are real
    /// cross-sweep posterior samples; :func:`topica.composition_theta` prefers
    /// them over the within-document Dirichlet approximation.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Per-document token counts (length D), in :attr:`doc_topic` row order.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self
            .corpus
            .as_ref()
            .map(|c| c.docs.iter().map(|d| d.len()).collect())
            .unwrap_or_default())
    }

    /// The baseline document-topic Dirichlet prior α, shape ``(num_topics,)``:
    /// ``exp(λ_intercept)``, the per-topic prior at covariates = 0. DMR's prior is
    /// per-document (``α_{d,k} = exp(λ_k · x_d)``), so this is the baseline; it
    /// marks DMR as a Dirichlet model for :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        // feature_effects is (num_topics, num_features); column 0 is the intercept.
        let lam = self.feature_effects.as_ref().unwrap();
        let a: Vec<f64> = lam.column(0).iter().map(|&l| l.exp()).collect();
        Ok(Array1::from(a).to_pyarray_bound(py))
    }

    /// Learned feature weights λ, shape ``(num_topics, num_features)`` — how
    /// each feature (column 0 is the intercept) shifts each topic's log-prior.
    /// Positive ⇒ the feature raises that topic's prevalence.
    #[getter]
    fn feature_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.feature_effects.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Feature names aligned with the columns of :attr:`feature_effects`
    /// (``"intercept"`` first).
    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    /// Per-iteration log-likelihood trace. Returns one ``(iter, ll)`` pair for
    /// every ``check_every`` sweeps (empty when ``check_every=0``, the default).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// ``True`` if the relative-change convergence criterion was satisfied before
    /// all iterations completed. Always ``False`` when ``convergence_tol=0``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Top `n` words per topic as ``(word, probability)`` pairs (all topics, or
    /// one when `topic` is given).
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(phi, self.num_topics, n);

        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by collapsed Gibbs against
    /// the fitted topic-word matrix. `data` is a :class:`Corpus` or
    /// `list[list[str]]`; OOV tokens are dropped. `features` (optional, a
    /// ``(num_docs, F)`` covariate array matching training, no intercept) sets
    /// each document's Dirichlet prior `α_d = exp(Xγ)`; if omitted the
    /// intercept-only baseline prior is used. Returns ``(num_docs, num_topics)``.
    #[pyo3(signature = (data, features=None, *, iters=100, burn_in=10,
                        num_samples=10, sample_interval=5, seed=None, iterations=None))]
    #[allow(clippy::too_many_arguments)]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        features: Option<PyReadonlyArray2<f64>>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iterations = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let k = self.num_topics;
        let eff = self.feature_effects.as_ref().unwrap(); // (K, F) incl. intercept at col 0
        let nf = eff.shape()[1];
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let docs = docs_to_ids(data, id_to_word)?;
        let docs_usize: Vec<Vec<usize>> =
            docs.iter().map(|d| d.iter().map(|&w| w as usize).collect()).collect();
        let phi_rows: Vec<Vec<f64>> =
            self.phi.as_ref().unwrap().outer_iter().map(|r| r.to_vec()).collect();

        // Per-document Dirichlet prior α_d = exp(Xγ); intercept is column 0.
        let alphas: Vec<Vec<f64>> = match &features {
            Some(x) => {
                let x = x.as_array();
                if x.shape()[0] != docs_usize.len() {
                    return Err(PyValueError::new_err(
                        "features rows must match number of documents",
                    ));
                }
                if x.shape()[1] + 1 != nf {
                    return Err(PyValueError::new_err(format!(
                        "features must have {} columns (the {} training covariates, no intercept)",
                        nf - 1,
                        nf - 1
                    )));
                }
                check_all_finite_arr2("features", &x)?;
                (0..docs_usize.len())
                    .map(|d| {
                        (0..k)
                            .map(|t| {
                                let mut s = eff[[t, 0]];
                                for f in 1..nf {
                                    s += eff[[t, f]] * x[[d, f - 1]];
                                }
                                s.exp()
                            })
                            .collect()
                    })
                    .collect()
            }
            None => {
                let base: Vec<f64> = (0..k).map(|t| eff[[t, 0]].exp()).collect();
                vec![base; docs_usize.len()]
            }
        };

        let base_seed = seed.unwrap_or(self.seed);
        let rows: Vec<Vec<f64>> = py.allow_threads(|| {
            docs_usize
                .par_iter()
                .zip(alphas.par_iter())
                .enumerate()
                .map(|(i, (d, alpha))| {
                    let mut rng = Pcg64Mcg::seed_from_u64(base_seed.wrapping_add(i as u64));
                    infer_theta_gibbs(
                        &phi_rows, alpha, d, iterations, burn_in, num_samples, sample_interval,
                        &mut rng,
                    )
                })
                .collect()
        });
        let mut arr = Array2::<f64>::zeros((rows.len(), k));
        for (i, row) in rows.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                arr[[i, t]] = v;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (compact binary). Reload with `DMR.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_DMR, &DmrState {
            num_topics: self.num_topics, beta: self.beta,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            prior_variance: self.prior_variance, lbfgs_iters: self.lbfgs_iters, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            feature_effects: arr2_opt(&self.feature_effects),
            feature_names: self.feature_names.clone(), corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: DmrState = read_state(path, MODEL_TAG_DMR)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(DMR {
            num_topics: s.num_topics, beta: s.beta, optimize_interval: s.optimize_interval,
            burn_in: s.burn_in, seed: s.seed, prior_variance: s.prior_variance,
            lbfgs_iters: s.lbfgs_iters, warp: false, cvb0: false, fitted: s.fitted,
            topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            feature_effects: arr2_back(s.feature_effects),
            feature_names: s.feature_names, corpus: s.corpus,
            theta_draws: arr3f32_back(s.theta_draws),
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!("DMR(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// Labeled LDA
// ---------------------------------------------------------------------------

/// Supervised topic model (Ramage et al., 2009): each document carries a set of
/// labels, each label is a topic, and a document's tokens are constrained to its
/// labels' topics. The number of topics is the number of distinct labels.
///
/// Documents with an empty label set are treated as unconstrained (all topics).
#[pyclass(module = "topica")]
pub struct LabeledLDA {
    alpha: f64,
    beta: f64,
    seed: u64,
    // CVB0 deterministic collapsed-variational inference (masked γ per document)
    // instead of the default restricted SparseLDA sweep.
    cvb0: bool,

    fitted: bool,
    num_topics: usize,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    label_vocab: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl LabeledLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl LabeledLDA {
    /// Create an unfitted model. `alpha` is the (symmetric) per-topic prior
    /// over a document's allowed topics.
    #[new]
    #[pyo3(signature = (*, alpha=0.1, beta=0.01, seed=42, sampler="sparse"))]
    fn new(alpha: f64, beta: f64, seed: u64, sampler: &str) -> PyResult<Self> {
        if !finite_pos(alpha) {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        if !finite_pos(beta) {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        let cvb0 = match sampler {
            "sparse" => false,
            "cvb0" | "cvb" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\" or \"cvb0\""
                )))
            }
        };
        Ok(LabeledLDA {
            alpha,
            beta,
            seed,
            cvb0,
            fitted: false,
            num_topics: 0,
            topic_names: Vec::new(),
            phi: None,
            theta: None,
            label_vocab: Vec::new(),
            corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `labels` is a list (one per document) of label lists. The topic set is
    /// the union of all labels (or `label_names`, which also fixes topic order).
    /// An empty label list leaves that document unconstrained.
    ///
    /// `convergence_tol` (default 0.0, disabled) enables early stopping based
    /// on the relative change in log-likelihood every `check_every` sweeps.
    #[pyo3(signature = (data, labels, *, label_names=None, iters=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        labels: Vec<Vec<String>>,
        label_names: Option<Vec<String>>,
        iters: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if labels.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "labels has {} entries but corpus has {} documents",
                labels.len(),
                num_docs
            )));
        }

        // Topic vocabulary: provided order, or the sorted union of all labels.
        let label_vocab: Vec<String> = match label_names {
            Some(n) => n,
            None => {
                let mut set: HashSet<String> = HashSet::new();
                for ls in &labels {
                    for l in ls {
                        set.insert(l.clone());
                    }
                }
                let mut v: Vec<String> = set.into_iter().collect();
                v.sort();
                v
            }
        };
        if label_vocab.is_empty() {
            return Err(PyValueError::new_err("no labels found; provide labels or label_names"));
        }
        let k = label_vocab.len();
        let index: HashMap<&str, usize> = label_vocab
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        let allowed: Vec<Vec<usize>> = labels
            .iter()
            .map(|ls| {
                let mut v: Vec<usize> =
                    ls.iter().filter_map(|l| index.get(l.as_str()).copied()).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        let num_types = corpus.num_types();
        let total_tokens = corpus.total_tokens().max(1) as f64;
        let alpha_sum = self.alpha * k as f64;
        let mut model = TopicModel::new(k, alpha_sum, self.beta, num_types);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        labeled::initialize_labeled(&mut model, &corpus.docs, &allowed, &mut rng);

        let check_every_labeled = if check_every == 0 { 0 } else if convergence_tol > 0.0 { check_every.max(1) } else { check_every };
        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        if self.cvb0 {
            // CVB0 LabeledLDA: deterministic; the per-document label set masks the
            // responsibilities (γ is zero off the allowed topics — free in CVB0,
            // unlike a sampler's proposal rejection). No MCMC, so no θ-draws.
            let beta = self.beta;
            let alpha = self.alpha;
            let (acc_phi, acc_theta, model, corpus) = py.allow_threads(move || {
                let alpha0 = vec![alpha; k];
                let mut cv = cvb0::Cvb0::new(&corpus, k, &alpha0, beta, &mut rng);
                cv.set_allowed(allowed);
                for _ in 0..iters {
                    cv.sweep();
                }
                let mut acc_phi = vec![vec![0.0f64; k]; num_types];
                let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
                cv.phi_into(&mut acc_phi);
                cv.theta_into(&mut acc_theta);
                let model = cv.to_topic_model(&corpus);
                (acc_phi, acc_theta, model, corpus)
            });
            let _ = &model; // packed CVB0 state (argmax γ) backs coherence/save
            let mut phi = Array2::<f64>::zeros((k, num_types));
            for (w, row) in acc_phi.iter().enumerate() {
                for (t, &val) in row.iter().enumerate() {
                    phi[[t, w]] = val;
                }
            }
            let theta = vecs_to_arr2(&acc_theta);
            self.num_topics = k;
            self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
            self.label_vocab = label_vocab;
            self.phi = Some(phi);
            self.theta = Some(theta);
            self.theta_draws = None;
            self.corpus = Some(corpus);
            self.log_likelihood_history = Vec::new();
            self.converged = false;
            self.fitted = true;
            return Ok(());
        }

        let (acc_phi, acc_theta, theta_draw_buf, ll_history, converged, model, corpus) = py.allow_threads(move || {
            let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
            let all_topics: Vec<usize> = (0..k).collect();
            let mut ll_history: Vec<(usize, f64)> = Vec::new();
            let mut converged = false;

            'outer: for iter in 1..=iters {
                labeled::run_sweep_labeled(&mut model, &corpus.docs, &allowed, &mut rng);
                if draws_opts.thin > 0 && iter % draws_opts.thin == 0 {
                    let counts = doc_topic_counts(&model.doc_topics, k);
                    let snap: Vec<Vec<f32>> = (0..num_docs).map(|d| {
                        let allow: &[usize] = if allowed[d].is_empty() { &all_topics } else { &allowed[d] };
                        let asum: f64 = allow.iter().map(|&t| model.alpha[t]).sum();
                        let denom = corpus.docs[d].len() as f64 + asum;
                        let mut row = vec![0.0f32; k];
                        for &t in allow {
                            row[t] = ((counts[d][t] as f64 + model.alpha[t]) / denom) as f32;
                        }
                        row
                    }).collect();
                    push_capped(&mut theta_draw_buf, snap, draws_opts.cap);
                }
                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let ll = output::model_log_likelihood(&model, &corpus) / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, ll));
                        });
                    }
                }
                // Trace recording and optional convergence check (never alters RNG).
                if check_every_labeled > 0 && iter % check_every_labeled == 0 {
                    let ll = output::model_log_likelihood(&model, &corpus);
                    ll_history.push((iter, ll));
                    if convergence_tol > 0.0 && ll_history.len() >= 2 {
                        let prev = ll_history[ll_history.len() - 2].1;
                        let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                        if rel < convergence_tol {
                            converged = true;
                            break 'outer;
                        }
                    }
                }
            }

            let mut acc_phi = vec![vec![0.0f64; k]; num_types];
            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    labeled::run_sweep_labeled(&mut model, &corpus.docs, &allowed, &mut rng);
                }
                accumulate_phi(&model, &mut acc_phi);
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let allow: &[usize] = if allowed[d].is_empty() {
                        &all_topics
                    } else {
                        &allowed[d]
                    };
                    let asum: f64 = allow.iter().map(|&t| model.alpha[t]).sum();
                    let denom = corpus.docs[d].len() as f64 + asum;
                    for &t in allow {
                        acc_theta[d][t] += (counts[d][t] as f64 + model.alpha[t]) / denom;
                    }
                }
            }

            let n = num_samples.max(1) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            (acc_phi, acc_theta, theta_draw_buf, ll_history, converged, model, corpus)
        });
        let _ = model;

        let mut phi = Array2::<f64>::zeros((k, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                phi[[t, w]] = val;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, k, None);
        self.num_topics = k;
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.label_vocab = label_vocab;
        self.corpus = Some(corpus);
        self.log_likelihood_history = ll_history;
        self.converged = converged;
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; for each
    /// document only its label topics are non-zero, and rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The symmetric document-topic Dirichlet prior α, shape ``(num_topics,)``.
    /// Marks LabeledLDA as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_topics]).to_pyarray_bound(py))
    }

    /// Thinned MCMC θ snapshots, shape ``(num_draws, num_docs, num_topics)``,
    /// dtype ``float32``. ``None`` when fit with ``keep_theta_draws=False``. These
    /// are real cross-sweep draws; use them with
    /// :func:`topica.effects.composition_theta` for uncertainty quantification.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }

    /// The label name for each topic, in topic (column) order.
    #[getter]
    fn labels(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.label_vocab.clone())
    }

    /// Per-iteration log-likelihood trace recorded every ``check_every`` sweeps.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// ``True`` if the convergence criterion was met; ``False`` otherwise.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_topics)
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Top `n` words for one topic (by label name or index) or all topics.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(phi, self.num_topics, n);

        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer label (topic) proportions θ for *new* documents by collapsed Gibbs
    /// against the fitted topic-word matrix, treating every label as available
    /// (unsupervised inference). `data` is a :class:`Corpus` or
    /// `list[list[str]]`; OOV tokens are dropped. Returns ``(num_docs,
    /// num_topics)``; columns align with :attr:`labels`.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.phi.as_ref().unwrap(),
            &alpha, iters, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// Save the fitted model to `path`. Reload with `LabeledLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_LABELED, &LabeledState {
            alpha: self.alpha, beta: self.beta, seed: self.seed, fitted: self.fitted,
            num_topics: self.num_topics, phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            label_vocab: self.label_vocab.clone(), corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: LabeledState = read_state(path, MODEL_TAG_LABELED)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(LabeledLDA {
            alpha: s.alpha, beta: s.beta, seed: s.seed, cvb0: false, fitted: s.fitted,
            num_topics: s.num_topics, topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            label_vocab: s.label_vocab, corpus: s.corpus,
            theta_draws: arr3f32_back(s.theta_draws),
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LabeledLDA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}

// ---------------------------------------------------------------------------
// SAGE: content-covariate topic model
// ---------------------------------------------------------------------------

/// Parse a per-document group covariate (list of strings or ints) to strings.
fn parse_groups(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(v) = obj.extract::<Vec<String>>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<Vec<i64>>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    Err(PyValueError::new_err("groups must be a list of strings or ints"))
}

/// Content-covariate topic model (SAGE / the STM content model).
///
/// Topics are shared, but each topic's word distribution varies by a
/// document-level **group** covariate, so you can read how a topic is worded
/// differently across groups. Construct, then :meth:`fit` on documents plus a
/// per-document group label.
#[pyclass(module = "topica")]
pub struct SAGE {
    num_topics: usize,
    alpha: f64,
    prior_variance: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    lbfgs_iters: usize,

    fitted: bool,
    topic_names: Vec<String>,
    num_groups: usize,
    beta: Vec<Vec<f64>>, // [K*G][V]
    theta: Option<Array2<f64>>,
    group_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl SAGE {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    /// β for (topic, group) averaged over groups → a plain (K, V) topic-word.
    fn topic_marginal(&self) -> Array2<f64> {
        let k = self.num_topics;
        let g = self.num_groups;
        let v = self.corpus.as_ref().unwrap().num_types();
        let mut out = Array2::<f64>::zeros((k, v));
        for kk in 0..k {
            for gg in 0..g {
                let cell = &self.beta[kk * g + gg];
                for vv in 0..v {
                    out[[kk, vv]] += cell[vv] / g as f64;
                }
            }
        }
        out
    }
}

#[pymethods]
impl SAGE {
    /// Create an unfitted model. `alpha` is the symmetric document-topic prior;
    /// `prior_variance` is the Gaussian prior on the κ content deviations.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, prior_variance=1.0,
                        optimize_interval=50, burn_in=200, seed=42, lbfgs_iters=20))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: f64,
        prior_variance: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        lbfgs_iters: usize,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if !finite_pos(prior_variance) {
            return Err(PyValueError::new_err("prior_variance must be > 0"));
        }
        Ok(SAGE {
            num_topics,
            alpha,
            prior_variance,
            optimize_interval,
            burn_in,
            seed,
            lbfgs_iters,
            fitted: false,
            topic_names: Vec::new(),
            num_groups: 0,
            beta: Vec::new(),
            theta: None,
            group_names: Vec::new(),
            corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `groups` is a per-document group label (strings or ints), one per
    /// document. `group_names` fixes the group order (defaults to sorted union).
    #[pyo3(signature = (data, groups, *, group_names=None, iters=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        groups: &Bound<'_, PyAny>,
        group_names: Option<Vec<String>>,
        iters: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let groups_str = parse_groups(groups)?;
        if groups_str.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "groups has {} entries but corpus has {} documents",
                groups_str.len(),
                num_docs
            )));
        }

        let group_vocab: Vec<String> = match group_names {
            Some(n) => n,
            None => {
                let mut set: HashSet<String> = groups_str.iter().cloned().collect();
                let mut v: Vec<String> = set.drain().collect();
                v.sort();
                v
            }
        };
        let gindex: HashMap<&str, usize> = group_vocab
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_str(), i))
            .collect();
        let groups_idx: Vec<usize> = groups_str
            .iter()
            .map(|g| {
                gindex
                    .get(g.as_str())
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("group {:?} not in group_names", g)))
            })
            .collect::<PyResult<_>>()?;

        let k = self.num_topics;
        let group_n = group_vocab.len();
        let num_types = corpus.num_types();
        let alpha = self.alpha;
        let alpha_sum = alpha * k as f64;
        let total_tokens = corpus.total_tokens().max(1) as f64;

        let mut model = sage::SageModel::new(k, group_n, num_types, alpha, self.prior_variance);
        model.set_background(&corpus.docs);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        model.initialize(&corpus.docs, &groups_idx, &mut rng);

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        let lbfgs_iters = self.lbfgs_iters;

        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let (beta, acc_theta, theta_draw_buf, ll_history, converged_flag, corpus) = py.allow_threads(move || {
            let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut ll_history: Vec<(usize, f64)> = Vec::new();
            let mut converged_flag = false;

            // Inline LL for SAGE: sum_c sum_v n_cv * ln(beta_cv).
            let compute_ll = |model: &sage::SageModel| -> f64 {
                let mut ll = 0.0f64;
                for c in 0..(k * group_n) {
                    for v in 0..num_types {
                        let n = model.counts[c][v] as f64;
                        if n > 0.0 {
                            ll += n * model.beta[c][v].max(1e-300).ln();
                        }
                    }
                }
                ll
            };

            'outer: for iter in 1..=iters {
                sage::run_sweep_sage(&mut model, &corpus.docs, &groups_idx, &mut rng);
                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    sage::optimize_kappa(&mut model, lbfgs_iters);
                }
                if draws_opts.thin > 0 && iter % draws_opts.thin == 0 {
                    let counts = doc_topic_counts(&model.doc_topics, k);
                    let snap: Vec<Vec<f32>> = (0..num_docs).map(|d| {
                        let denom = corpus.docs[d].len() as f64 + alpha_sum;
                        (0..k).map(|t| ((counts[d][t] as f64 + alpha) / denom) as f32).collect()
                    }).collect();
                    push_capped(&mut theta_draw_buf, snap, draws_opts.cap);
                }
                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let llpt = compute_ll(&model) / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, llpt));
                        });
                    }
                }
                // Trace recording and optional convergence check (never alters RNG).
                if check_every > 0 && iter % check_every == 0 {
                    let ll = compute_ll(&model);
                    ll_history.push((iter, ll));
                    if convergence_tol > 0.0 && ll_history.len() >= 2 {
                        let prev = ll_history[ll_history.len() - 2].1;
                        let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                        if rel < convergence_tol {
                            converged_flag = true;
                            break 'outer;
                        }
                    }
                }
            }
            sage::optimize_kappa(&mut model, lbfgs_iters); // final β refresh

            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    sage::run_sweep_sage(&mut model, &corpus.docs, &groups_idx, &mut rng);
                }
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let denom = corpus.docs[d].len() as f64 + alpha_sum;
                    for t in 0..k {
                        acc_theta[d][t] += (counts[d][t] as f64 + alpha) / denom;
                    }
                }
            }
            let n = num_samples.max(1) as f64;
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            (model.beta.clone(), acc_theta, theta_draw_buf, ll_history, converged_flag, corpus)
        });

        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, k, None);
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.num_groups = group_n;
        self.beta = beta;
        self.theta = Some(theta);
        self.group_names = group_vocab;
        self.corpus = Some(corpus);
        self.log_likelihood_history = ll_history;
        self.converged = converged_flag;
        self.fitted = true;
        Ok(())
    }

    /// Topic-word distributions per group, shape ``(num_topics, num_groups, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        let k = self.num_topics;
        let g = self.num_groups;
        let v = self.corpus.as_ref().unwrap().num_types();
        let mut arr = Array3::<f64>::zeros((k, g, v));
        for kk in 0..k {
            for gg in 0..g {
                let cell = &self.beta[kk * g + gg];
                for vv in 0..v {
                    arr[[kk, gg, vv]] = cell[vv];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Group-averaged topic-word matrix, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word_marginal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.topic_marginal().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The symmetric document-topic Dirichlet prior α, shape ``(num_topics,)``.
    /// SAGE's sparse additive parameterization is on the word side; the
    /// document side is an ordinary Dirichlet, so this marks SAGE as a Dirichlet
    /// model for :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_topics]).to_pyarray_bound(py))
    }

    /// Thinned MCMC θ snapshots, shape ``(num_draws, num_docs, num_topics)``,
    /// dtype ``float32``. ``None`` when fit with ``keep_theta_draws=False``. These
    /// are real cross-sweep draws; use them with
    /// :func:`topica.effects.composition_theta` for uncertainty quantification.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }

    /// Group names, in the index order used by :attr:`topic_word`'s second axis.
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.group_names.clone())
    }

    /// Per-iteration log-likelihood trace. Returns one ``(iter, ll)`` pair for
    /// every ``check_every`` sweeps (empty when ``check_every=0``, the default).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// ``True`` if the relative-change convergence criterion was satisfied before
    /// all iterations completed. Always ``False`` when ``convergence_tol=0``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    #[getter]
    fn num_groups(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_groups)
    }

    /// Top `n` words per topic. `topic=None` (default) returns a list of lists
    /// (one per topic); `topic=k` returns the list for topic k. With `group`
    /// (name or index) given, uses that group's word distribution; otherwise the
    /// group-averaged distribution is used.
    #[pyo3(signature = (n=10, *, topic=None, group=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        group: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;

        // Helper: top-n words for a single topic index.
        let top_for = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let dist: Vec<f64> = match group {
                Some(gobj) => {
                    let gi = self.resolve_group(gobj)?;
                    self.beta[t * self.num_groups + gi].clone()
                }
                None => {
                    let m = self.topic_marginal();
                    (0..vocab.len()).map(|v| m[[t, v]]).collect()
                }
            };
            let mut idx: Vec<usize> = (0..vocab.len()).collect();
            idx.sort_by(|&a, &b| f64::total_cmp(&dist[b], &dist[a]));
            let items: Vec<Bound<'py, PyTuple>> = idx
                .iter()
                .take(n)
                .map(|&v| {
                    PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), dist[v].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(top_for(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(top_for).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// Words that most distinguish how `topic` is worded in `group_a` vs
    /// `group_b`, by log-ratio of the two groups' word probabilities. Returns
    /// ``(word, log_ratio)`` — positive favours `group_a`.
    #[pyo3(signature = (topic, group_a, group_b, n=10))]
    fn word_contrast<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        group_a: &Bound<'py, PyAny>,
        group_b: &Bound<'py, PyAny>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let ga = self.resolve_group(group_a)?;
        let gb = self.resolve_group(group_b)?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let a = &self.beta[topic * self.num_groups + ga];
        let b = &self.beta[topic * self.num_groups + gb];
        let ratio: Vec<f64> = (0..vocab.len())
            .map(|v| (a[v].max(1e-300) / b[v].max(1e-300)).ln())
            .collect();
        let mut idx: Vec<usize> = (0..vocab.len()).collect();
        idx.sort_by(|&x, &y| f64::total_cmp(&ratio[y], &ratio[x]));
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), ratio[v].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// UMass topic coherence per topic (group-averaged), shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(&self.topic_marginal(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `SAGE.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_SAGE, &SageState {
            num_topics: self.num_topics, alpha: self.alpha, prior_variance: self.prior_variance,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            lbfgs_iters: self.lbfgs_iters, fitted: self.fitted, num_groups: self.num_groups,
            beta: self.beta.clone(), theta: arr2_opt(&self.theta),
            group_names: self.group_names.clone(), corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SageState = read_state(path, MODEL_TAG_SAGE)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(SAGE {
            num_topics: s.num_topics, alpha: s.alpha, prior_variance: s.prior_variance,
            optimize_interval: s.optimize_interval, burn_in: s.burn_in, seed: s.seed,
            lbfgs_iters: s.lbfgs_iters, fitted: s.fitted, num_groups: s.num_groups,
            topic_names,
            beta: s.beta, theta: arr2_back(s.theta), group_names: s.group_names, corpus: s.corpus,
            theta_draws: arr3f32_back(s.theta_draws),
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    /// Infer document-topic distributions for new, unseen documents under the
    /// fitted model (sklearn-style ``transform``). Holds the fitted
    /// group-averaged topic-word distributions fixed and runs collapsed Gibbs
    /// to infer θ for each document. Returns shape
    /// ``(num_new_docs, num_topics)`` with rows summing to 1.
    ///
    /// **Approximation:** held-out inference uses the group-averaged
    /// topic-word matrix (the marginal over groups) and does not condition on
    /// a group covariate for new documents. This is a baseline projection;
    /// the group-specific word distributions are a training-time device and
    /// cannot be recovered for documents whose group label is unknown.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let phi = self.topic_marginal();
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(py, data, id_to_word, &phi, &alpha, iters, burn_in,
                        num_samples, sample_interval, seed.unwrap_or(self.seed))
    }

    fn __repr__(&self) -> String {
        format!(
            "SAGE(num_topics={}, num_groups={}, fitted={})",
            self.num_topics, self.num_groups, self.fitted
        )
    }
}

impl SAGE {
    /// Resolve a group given as a name (str) or an index (int) to its index.
    fn resolve_group(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.num_groups {
                return Ok(i);
            }
            return Err(PyValueError::new_err(format!(
                "group index {} out of range (num_groups={})",
                i, self.num_groups
            )));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .group_names
                .iter()
                .position(|g| g == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown group {:?}", s)));
        }
        Err(PyValueError::new_err("group must be a name (str) or index (int)"))
    }
}

// ---------------------------------------------------------------------------
// CTM: Correlated Topic Model (STM's logistic-normal variational core)
// ---------------------------------------------------------------------------

/// Extract the per-document variational posterior of η from a fitted CTM/STM:
/// the means λ (D × K-1) and covariances ν (D × K-1 × K-1). These define the
/// logistic-normal posterior `η_d ~ N(λ_d, ν_d)` used for sampling θ draws
/// (method-of-composition uncertainty).
/// Map new documents (a `Corpus` or `list[list[str]]`) onto the training
/// vocabulary, dropping out-of-vocabulary tokens. Tokens are lowercased to
/// match the corpus loader. Returns one `Vec<u32>` of word-ids per document.
fn docs_to_ids(
    data: &Bound<'_, PyAny>,
    id_to_word: &[String],
) -> PyResult<Vec<Vec<u32>>> {
    let word_to_id: HashMap<&str, u32> = id_to_word
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as u32))
        .collect();
    let str_docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
        c.inner
            .docs
            .iter()
            .map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect())
            .collect()
    } else {
        data.extract().map_err(|_| {
            PyValueError::new_err("transform() expects a Corpus or a list of token lists")
        })?
    };
    Ok(str_docs
        .into_iter()
        .map(|doc| {
            doc.iter()
                .filter_map(|t| word_to_id.get(t.to_lowercase().as_str()).copied())
                .collect()
        })
        .collect())
}

/// Run the CTM/STM variational E-step inference for a batch of documents,
/// returning their topic proportions θ as a ``(num_docs, K)`` array. Parallel
/// over documents; the per-doc result is independent so order is preserved.
fn infer_theta_batch(
    py: Python<'_>,
    beta: &[Vec<f64>],
    mu: &[f64],
    sigma: &[f64],
    docs: &[Vec<u32>],
) -> Array2<f64> {
    let k = mu.len() + 1;
    let km1 = mu.len();
    let siginv = crate::linalg::spd_inverse(sigma, km1).unwrap_or_else(|| {
        let mut s = sigma.to_vec();
        crate::linalg::make_diagonally_dominant(&mut s, km1);
        crate::linalg::spd_inverse(&s, km1).unwrap()
    });
    let rows: Vec<Vec<f64>> = py.allow_threads(|| {
        docs.par_iter()
            .map(|doc| {
                let (words, counts) = crate::variational::doc_sparse(doc);
                ctm::infer_theta(beta, mu, &siginv, &words, &counts)
            })
            .collect()
    });
    let mut out = Array2::<f64>::zeros((rows.len(), k));
    for (d, row) in rows.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            out[[d, t]] = v;
        }
    }
    out
}

/// Run the CTM/STM variational E-step with a PER-DOCUMENT prior mean.
///
/// `mu_per_doc` has shape `(num_docs, K-1)`: row `d` is the prior mean for
/// document `d` (e.g. `X_d γ` from the prevalence regression). The prior
/// covariance `sigma` is shared across all documents (the global fitted Σ).
/// Precomputes `siginv` once, then maps each document independently.
fn infer_theta_batch_per_doc(
    py: Python<'_>,
    beta: &[Vec<f64>],
    mu_per_doc: &Array2<f64>,
    sigma: &[f64],
    docs: &[Vec<u32>],
) -> Array2<f64> {
    let nd = docs.len();
    let km1 = mu_per_doc.ncols();
    let k = km1 + 1;
    let siginv = crate::linalg::spd_inverse(sigma, km1).unwrap_or_else(|| {
        let mut s = sigma.to_vec();
        crate::linalg::make_diagonally_dominant(&mut s, km1);
        crate::linalg::spd_inverse(&s, km1).unwrap()
    });
    // Collect per-doc prior means as owned Vec<f64> for thread-safety.
    let mus: Vec<Vec<f64>> = (0..nd)
        .map(|d| (0..km1).map(|j| mu_per_doc[[d, j]]).collect())
        .collect();
    let rows: Vec<Vec<f64>> = py.allow_threads(|| {
        docs.par_iter()
            .zip(mus.par_iter())
            .map(|(doc, mu_d)| {
                let (words, counts) = crate::variational::doc_sparse(doc);
                ctm::infer_theta(beta, mu_d, &siginv, &words, &counts)
            })
            .collect()
    });
    let mut out = Array2::<f64>::zeros((rows.len(), k));
    for (d, row) in rows.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            out[[d, t]] = v;
        }
    }
    out
}


/// Correlated Topic Model (Blei & Lafferty; the STM core). Topics are drawn
/// from a logistic-normal prior with a full covariance, so they can correlate —
/// unlike LDA's Dirichlet. Fit by variational EM (STM's Laplace E-step).
///
/// This is the engine STM builds on; prevalence/content covariates layer on top.
///
/// The per-document E-step runs in parallel on all cores by default; cap it with
/// ``fit(num_threads=...)`` (results are identical regardless). ``variational=``
/// chooses the covariance approximation (``"laplace"`` full, or ``"diagonal"``
/// for a faster mean-field one at high K), and ``fit(keep_eta_cov=False)`` trades
/// stored covariance for far less memory at large K.
#[pyclass(module = "topica")]
pub struct CTM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,
    /// Variational-covariance mode: "laplace" (full ν = H⁻¹) or "diagonal"
    /// (mean-field ν = diag(1/H_ii)).
    variational: String,

    fitted: bool,
    topic_names: Vec<String>,
    beta: Option<Array2<f64>>,  // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    corr: Option<Array2<f64>>,  // (num_topics, num_topics)
    eta_mean: Option<Array2<f64>>, // (num_docs, num_topics-1) variational means λ
    eta_cov: Option<Array3<f32>>,  // (num_docs, K-1, K-1) variational covariances ν — stored as f32 to halve memory
    mu: Vec<f64>,                  // K-1 logistic-normal prior mean (for inference)
    sigma: Vec<f64>,               // (K-1)² logistic-normal prior covariance
    /// Sigma from the last E-step (may differ from sigma when the final M-step
    /// updated sigma after the last E-step). Used by `_recompute_eta_cov`.
    sigma_estep: Vec<f64>,
    /// Topic-word matrix β (K×V) used in the last E-step (before the final
    /// M-step updated `beta`). Used by `_recompute_eta_cov` to reproduce ν.
    beta_estep: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
    bound: f64,                    // final variational bound (ELBO)
    bound_history: Vec<f64>,       // bound after each EM iteration
    converged: bool,               // hit em_tol (true) or em_iters cap (false)
}

impl CTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl CTM {
    /// Create an unfitted model. `sigma_shrink` ∈ [0,1] shrinks the topic
    /// covariance toward its diagonal each M-step (stabilizes Σ). `init` is
    /// ``"spectral"`` (default; deterministic anchor-word init, matching STM's
    /// default — `seed` is then irrelevant) or ``"random"`` (seeded).
    /// `variational` selects the per-document variational-covariance mode:
    /// ``"laplace"`` (default; full posterior covariance ν = H⁻¹) or
    /// ``"diagonal"`` (mean-field ν = diag(1/H_ii), which skips the per-document
    /// Cholesky/inverse for a large E-step speedup at high K, at the cost of the
    /// off-diagonal posterior covariance — topic-correlation/SE precision is lower).
    #[new]
    #[pyo3(signature = (num_topics, *, sigma_shrink=0.0, seed=42, init="spectral", variational="laplace"))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, sigma_shrink: f64, seed: u64, init: &str, variational: &str) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !(0.0..=1.0).contains(&sigma_shrink) {
            return Err(PyValueError::new_err("sigma_shrink must be in [0, 1]"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        if variational != "laplace" && variational != "diagonal" {
            return Err(PyValueError::new_err("variational must be 'laplace' or 'diagonal'"));
        }
        Ok(CTM {
            num_topics,
            sigma_shrink,
            seed,
            init_spectral,
            variational: variational.to_string(),
            fitted: false,
            topic_names: Vec::new(),
            beta: None,
            theta: None,
            corr: None,
            eta_mean: None,
            eta_cov: None,
            mu: Vec::new(),
            sigma: Vec::new(),
            sigma_estep: Vec::new(),
            beta_estep: None,
            corpus: None,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit by variational EM. `data` is a :class:`Corpus` or `list[list[str]]`.
    /// EM runs until the relative change in the variational bound drops below
    /// `convergence_tol` (R `stm`'s `emtol`) or `iters` iterations are reached,
    /// whichever comes first. Pass ``convergence_tol=0`` to always run `iters` steps.
    /// Check :attr:`converged` and :attr:`bound` afterward.
    /// `inference="svi"` switches from full-batch variational EM to stochastic
    /// variational inference (online VB): documents are processed in minibatches
    /// of `batch_size`, taking a stochastic step on the global parameters with a
    /// decaying learning rate `(tau + t)^(-kappa)`, for `iters` epochs. SVI is
    /// for very large corpora; on moderate corpora the default `"batch"` EM is
    /// preferable. SVI uses the base logistic-normal model only.
    /// `num_threads` caps the worker pool for the parallel per-document E-step;
    /// the default ``None`` uses all available cores. The fit is bit-for-bit
    /// identical regardless of the worker count, so this only trades resource use
    /// (set it to 1 for a fully serial run). `keep_eta_cov=False` does not store
    /// the per-document variational covariance (an O(N*K^2) array), cutting memory
    /// sharply at large K; `posterior_theta_samples` / `estimate_effect` with
    /// draws transparently recompute it on demand when needed.
    #[pyo3(signature = (data, *, iters=500, convergence_tol=1e-5, inference="batch",
                        batch_size=256, tau=64.0, kappa=0.7, em_tol=None, keep_eta_cov=true,
                        num_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        convergence_tol: f64,
        inference: &str,
        batch_size: usize,
        tau: f64,
        kappa: f64,
        em_tol: Option<f64>,
        keep_eta_cov: bool,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "CTM.fit(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            // convergence_tol wins if explicitly set (not the default 1e-5); else deprecated.
            if (convergence_tol - 1e-5_f64).abs() > f64::EPSILON { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let svi = match inference {
            "batch" => false,
            "svi" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown inference {other:?}; expected \"batch\" or \"svi\""
                )))
            }
        };

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let shrink = self.sigma_shrink;
        let spectral = self.init_spectral;
        let diagonal = self.variational == "diagonal";
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = run_with_threads(num_threads, || {
                if svi {
                    ctm::fit_ctm_svi(
                        &corpus.docs, k, num_types, iters, batch_size, tau, kappa, shrink, spectral,
                        keep_eta_cov, diagonal, &mut rng,
                    )
                } else {
                    ctm::fit_ctm(
                        &corpus.docs, k, num_types, iters, convergence_tol, shrink, None, None,
                        spectral, ctm::GammaPrior::Pooled, keep_eta_cov, diagonal, &mut rng,
                    )
                }
            });
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        for t in 0..k {
            for v in 0..num_types {
                beta[[t, v]] = model.beta[t][v];
            }
        }
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (d, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }
        let corr_v = model.topic_correlation();
        let mut corr = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                corr[[i, j]] = corr_v[i][j];
            }
        }

        // Always build eta_mean; only build eta_cov when keep_eta_cov=True
        // (when keep_nu=false the nu array is empty; only build eta_cov when kept).
        let mean_rows = model.eta_mean();
        let d_docs = mean_rows.len();
        let dim = k - 1;
        let mut eta_mean_arr = Array2::<f64>::zeros((d_docs, dim));
        for di in 0..d_docs {
            for i in 0..dim {
                eta_mean_arr[[di, i]] = mean_rows[di][i];
            }
        }
        let stored_eta_cov: Option<Array3<f32>> = if keep_eta_cov {
            let cov_rows = model.eta_cov();
            let mut cov = Array3::<f32>::zeros((d_docs, dim, dim));
            for di in 0..d_docs {
                for i in 0..dim {
                    for j in 0..dim {
                        cov[[di, i, j]] = cov_rows[di][i * dim + j] as f32;
                    }
                }
            }
            Some(cov)
        } else {
            None
        };

        // Store beta from the last E-step so _recompute_eta_cov uses the same
        // beta that was active when nu was computed (pre-final-M-step).
        let beta_estep_arr: Array2<f64> = {
            let rows = &model.beta_estep;
            let v = rows[0].len();
            let mut arr = Array2::<f64>::zeros((k, v));
            for (t, row) in rows.iter().enumerate() {
                for (vi, &val) in row.iter().enumerate() {
                    arr[[t, vi]] = val;
                }
            }
            arr
        };

        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
        self.eta_mean = Some(eta_mean_arr);
        self.eta_cov = stored_eta_cov;
        self.mu = model.mu.clone();
        self.sigma = model.sigma.clone();
        // Retain the E-step snapshots only when eta_cov was NOT kept, so the
        // default path carries no extra state (recompute uses the stored eta_cov).
        if !keep_eta_cov {
            self.sigma_estep = model.sigma_estep.clone();
            self.beta_estep = Some(beta_estep_arr);
        }
        self.corpus = Some(corpus);
        self.bound = model.bound;
        self.bound_history = model.bound_history.clone();
        self.converged = model.converged;
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Final variational bound (approximate ELBO) at convergence — the quantity
    /// R `stm` reports as `convergence$bound`.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    /// The variational bound after each EM iteration (the convergence
    /// trajectory). Its length is the number of iterations actually run.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        self.require_fitted()?;
        Ok(self.bound_history.clone())
    }

    /// ``True`` if EM stopped on the `em_tol` criterion; ``False`` if it hit the
    /// `iters` cap first (the fit may not have converged).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Variational-covariance mode: ``"laplace"`` (full ν = H⁻¹) or
    /// ``"diagonal"`` (mean-field ν = diag(1/H_ii)).
    #[getter]
    fn variational(&self) -> String {
        self.variational.clone()
    }

    /// Uniform convergence trace: ``(iteration, bound)`` pairs, one per EM
    /// iteration. The objective is the variational ELBO (same as
    /// :attr:`bound_history`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Topic-correlation matrix from the logistic-normal Σ, shape
    /// ``(num_topics, num_topics)``. Off-diagonal entries are genuine topic
    /// correlations (the whole point of CTM vs. LDA).
    #[getter]
    fn topic_correlation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.corr.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior means λ of the logistic-normal η,
    /// shape ``(num_docs, num_topics-1)``. Pairs with :attr:`eta_cov` to sample
    /// θ draws (method-of-composition uncertainty).
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, num_topics-1, num_topics-1)``. Stored as float32 in memory
    /// to halve the dominant memory term; cast to float64 with
    /// ``np.asarray(model.eta_cov, dtype=np.float64)`` when full precision is needed.
    /// Raises RuntimeError if the model was fit with ``keep_eta_cov=False``; use
    /// :meth:`_recompute_eta_cov` to regenerate on demand.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        self.eta_cov.as_ref()
            .map(|c| c.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err(
                "model was fit with keep_eta_cov=False; refit with keep_eta_cov=True, \
                 or use posterior_theta_samples/_recompute_eta_cov which recompute it on demand"
            ))
    }

    /// Recompute the per-document variational covariance ν on demand.
    /// Use this when the model was fit with ``keep_eta_cov=False`` to save memory.
    /// Returns the same ``(num_docs, K-1, K-1)`` float32 array as :attr:`eta_cov`.
    fn _recompute_eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        let corpus = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no corpus retained; cannot recompute eta_cov")
        })?;
        let k = self.num_topics;
        let km1 = k - 1;
        let sparse: Vec<(Vec<usize>, Vec<f64>)> = corpus.docs.iter()
            .map(|doc| crate::variational::doc_sparse(doc))
            .collect();
        // Build a minimal CtmModel stub using only the fields recompute_nu needs.
        // Use beta_estep (the topic-word matrix from the last E-step, before the
        // final M-step updated beta) so the Hessian computation is bit-identical
        // to what was used when nu was originally stored.
        let beta_src = self.beta_estep.as_ref().unwrap_or_else(|| self.beta.as_ref().unwrap());
        let beta_v: Vec<Vec<f64>> = beta_src
            .outer_iter().map(|r| r.to_vec()).collect();
        let lambda_v: Vec<Vec<f64>> = self.eta_mean.as_ref().unwrap()
            .outer_iter().map(|r| r.to_vec()).collect();
        let d = lambda_v.len();
        // ν is independent of the prior mean μ, so recompute_nu uses self.mu for
        // every document. Fall back to self.sigma for loaded models (sigma_estep
        // is not persisted).
        let sigma_for_recompute = if !self.sigma_estep.is_empty() {
            self.sigma_estep.clone()
        } else {
            self.sigma.clone()
        };
        let model_stub = ctm::CtmModel {
            num_topics: k,
            num_types: corpus.num_types(),
            beta: beta_v.clone(),
            beta_estep: beta_v,
            mu: self.mu.clone(),
            sigma: self.sigma.clone(),
            sigma_estep: sigma_for_recompute,
            lambda: lambda_v,
            nu: Vec::new(),
            gamma: None,
            content_beta: None,
            num_groups: 1,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
            em_iters_run: 0,
            // Recompute ν in the same mode the fit used (laplace/diagonal).
            diagonal: self.variational == "diagonal",
        };
        let nu = py.allow_threads(|| ctm::recompute_nu(&model_stub, &sparse));
        let mut out = Array3::<f32>::zeros((d, km1, km1));
        for di in 0..d {
            for i in 0..km1 {
                for j in 0..km1 {
                    out[[di, i, j]] = nu[di][i * km1 + j] as f32;
                }
            }
        }
        Ok(out.to_pyarray_bound(py))
    }

    /// The fitted logistic-normal prior covariance Σ over η, shape
    /// ``(num_topics-1, num_topics-1)`` (the last topic is the softmax reference,
    /// so it is dropped). This is the model's own topic covariance — unlike
    /// :attr:`topic_correlation`, which is an across-document θ correlation.
    #[getter]
    fn topic_covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let km1 = self.num_topics.saturating_sub(1);
        if self.sigma.len() != km1 * km1 {
            return Err(PyRuntimeError::new_err(
                "this model was fit before topic_covariance was stored; refit to use it",
            ));
        }
        let arr = Array2::from_shape_vec((km1, km1), self.sigma.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by the variational E-step
    /// against the fitted globals (β, logistic-normal prior μ, Σ). `data` is a
    /// :class:`Corpus` or `list[list[str]]`; tokens outside the training
    /// vocabulary are dropped. Returns a ``(num_docs, num_topics)`` array.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let beta = self.beta.as_ref().unwrap();
        let beta_v: Vec<Vec<f64>> = beta.outer_iter().map(|r| r.to_vec()).collect();
        let theta = infer_theta_batch(py, &beta_v, &self.mu, &self.sigma, &docs);
        Ok(theta.to_pyarray_bound(py))
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `CTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        // eta_cov is stored as f32 in memory; upcast to f64 for the on-disk format
        // so existing saved models remain compatible.
        let eta_cov_f64 = self.eta_cov.as_ref().map(|c| c.mapv(|x| x as f64));
        write_state(path, MODEL_TAG_CTM, &CtmState {
            num_topics: self.num_topics, sigma_shrink: self.sigma_shrink, seed: self.seed,
            init_spectral: self.init_spectral, fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corr: arr2_opt(&self.corr),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&eta_cov_f64),
            mu: self.mu.clone(), sigma: self.sigma.clone(),
            corpus: self.corpus.clone(),
            bound: self.bound, bound_history: self.bound_history.clone(),
            converged: self.converged,
            topic_names: self.topic_names.clone(),
            variational: self.variational.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: CtmState = read_state(path, MODEL_TAG_CTM)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        // eta_cov is saved as f64 for format compatibility; downcast to f32 in memory.
        let eta_cov = arr3_back(s.eta_cov).map(|c| c.mapv(|x| x as f32));
        Ok(CTM {
            num_topics: s.num_topics, sigma_shrink: s.sigma_shrink, seed: s.seed,
            init_spectral: s.init_spectral, variational: s.variational, fitted: s.fitted,
            topic_names,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corr: arr2_back(s.corr),
            eta_mean: arr2_back(s.eta_mean), eta_cov,
            mu: s.mu, sigma: s.sigma,
            sigma_estep: Vec::new(),  // not persisted; falls back to sigma in _recompute_eta_cov
            beta_estep: None,         // not persisted; falls back to self.beta
            corpus: s.corpus,
            bound: s.bound, bound_history: s.bound_history, converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!("CTM(num_topics={}, variational={:?}, fitted={})", self.num_topics, self.variational, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// STM: Structural Topic Model (CTM core + prevalence covariates)
// ---------------------------------------------------------------------------

/// Structural Topic Model (Roberts, Stewart & Tingley). The correlated-topic
/// core (:class:`CTM`) with **prevalence covariates**: a document's prior topic
/// mean is a regression on its covariates, `μ_d = X_d γ`, so covariates shift
/// which topics a document discusses. After fitting, `prevalence_effects` holds
/// the learned γ; pair it with `topica.stm.estimate_effect` for inference.
///
/// The per-document E-step runs in parallel on all cores by default; cap it with
/// ``fit(num_threads=...)`` (results are identical regardless). ``variational=``
/// chooses the covariance approximation (``"laplace"`` full, or ``"diagonal"``
/// for a faster mean-field one at high K), and ``fit(keep_eta_cov=False)`` trades
/// stored covariance for far less memory at large K.
#[pyclass(module = "topica")]
pub struct STM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,
    /// Variational-covariance mode: "laplace" (full ν = H⁻¹) or "diagonal"
    /// (mean-field ν = diag(1/H_ii)).
    variational: String,

    fitted: bool,
    topic_names: Vec<String>,
    beta: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corr: Option<Array2<f64>>,
    eta_mean: Option<Array2<f64>>, // (num_docs, num_topics-1) variational means λ
    eta_cov: Option<Array3<f32>>,  // (num_docs, K-1, K-1) variational covariances ν — stored as f32 to halve memory
    gamma: Option<Array2<f64>>, // (num_features, num_topics-1); None if no prevalence
    feature_names: Vec<String>,
    content_beta: Option<Vec<Vec<Vec<f64>>>>, // G×K×V; None if no content
    group_names: Vec<String>,
    mu: Vec<f64>,    // K-1 logistic-normal prior mean (covariate-free baseline)
    sigma: Vec<f64>, // (K-1)² logistic-normal prior covariance
    /// Sigma from the last E-step; may differ from sigma when the final M-step
    /// runs after the last E-step. Used by `_recompute_eta_cov`.
    sigma_estep: Vec<f64>,
    /// Topic-word matrix β (K×V) used in the last E-step (before the final
    /// M-step updated `beta`). Used by `_recompute_eta_cov` to reproduce ν.
    beta_estep: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
    bound: f64,                // final variational bound (ELBO)
    bound_history: Vec<f64>,   // bound after each EM iteration
    converged: bool,           // hit em_tol (true) or em_iters cap (false)
}

impl STM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    fn resolve_group(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.group_names.len() {
                return Ok(i);
            }
            return Err(PyValueError::new_err("group index out of range"));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .group_names
                .iter()
                .position(|g| g == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown group {:?}", s)));
        }
        Err(PyValueError::new_err("group must be a name (str) or index (int)"))
    }
}

#[pymethods]
impl STM {
    /// Create an unfitted model. `sigma_shrink` ∈ [0,1] shrinks Σ toward its
    /// diagonal each M-step. `init` is ``"spectral"`` (default; deterministic
    /// anchor-word init, matching STM's default — `seed` is then irrelevant for
    /// β init) or ``"random"`` (seeded). Spectral init applies to the
    /// topic-word β; with a content model the per-group β is always random.
    /// `variational` selects the per-document variational-covariance mode:
    /// ``"laplace"`` (default; full posterior covariance ν = H⁻¹) or
    /// ``"diagonal"`` (mean-field ν = diag(1/H_ii), which skips the per-document
    /// Cholesky/inverse for a large E-step speedup at high K, at the cost of the
    /// off-diagonal posterior covariance — topic-correlation/SE precision is lower).
    #[new]
    #[pyo3(signature = (num_topics, *, sigma_shrink=0.0, seed=42, init="spectral", variational="laplace"))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, sigma_shrink: f64, seed: u64, init: &str, variational: &str) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !(0.0..=1.0).contains(&sigma_shrink) {
            return Err(PyValueError::new_err("sigma_shrink must be in [0, 1]"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        if variational != "laplace" && variational != "diagonal" {
            return Err(PyValueError::new_err("variational must be 'laplace' or 'diagonal'"));
        }
        Ok(STM {
            num_topics,
            sigma_shrink,
            seed,
            init_spectral,
            variational: variational.to_string(),
            fitted: false,
            topic_names: Vec::new(),
            beta: None,
            theta: None,
            corr: None,
            eta_mean: None,
            eta_cov: None,
            gamma: None,
            feature_names: Vec::new(),
            content_beta: None,
            group_names: Vec::new(),
            mu: Vec::new(),
            sigma: Vec::new(),
            sigma_estep: Vec::new(),
            beta_estep: None,
            corpus: None,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit. `data` is a :class:`Corpus` or `list[list[str]]`. `prevalence`
    /// (optional, `(num_docs, F)` covariates) makes topic prevalence depend on
    /// covariates (`μ_d = X_d γ`); an intercept is prepended. `content`
    /// (optional, one group label per document) makes the topic-word
    /// distributions vary by group (the SAGE content model). At least one of
    /// `prevalence`/`content` should be given (else use :class:`CTM`).
    ///
    /// EM runs until the relative change in the variational bound drops below
    /// `em_tol` (R `stm`'s `emtol`) or `iters` iterations are reached,
    /// whichever comes first. Pass ``em_tol=0`` to always run `iters`
    /// steps. Inspect :attr:`converged` and :attr:`bound` after fitting.
    ///
    /// `gamma_prior` controls the prevalence-coefficient (γ) regression in the
    /// M-step. ``"pooled"`` (default) uses ridge regression, matching R `stm`'s
    /// ``gamma.prior="Pooled"`` path. ``"l1"`` fits an elastic-net path by
    /// coordinate descent with the penalty selected by AIC — recommended when the
    /// prevalence design is high-dimensional (many one-hot levels). `gamma_enet`
    /// is the elastic-net mix: 1.0 is pure lasso, values in (0, 1) add a ridge
    /// component (R `stm`'s ``gamma.enet``). `gamma_enet` is ignored when
    /// `gamma_prior="pooled"`.
    /// `num_threads` caps the worker pool for the parallel per-document E-step;
    /// the default ``None`` uses all available cores, and the fit is bit-for-bit
    /// identical regardless of the worker count (set it to 1 for a fully serial
    /// run). `keep_eta_cov=False` does not store the per-document variational
    /// covariance (an O(N*K^2) array), cutting memory sharply at large K;
    /// `posterior_theta_samples` / `estimate_effect` with draws recompute it on
    /// demand. The covariance approximation is set on the constructor via
    /// `variational=` (``"laplace"`` default, or ``"diagonal"`` for a faster,
    /// lower-precision mean-field covariance).
    #[pyo3(signature = (data, prevalence=None, *, prevalence_names=None,
                        content=None, content_names=None, iters=500, convergence_tol=1e-5,
                        gamma_prior="pooled", gamma_enet=1.0, em_tol=None, covariates=None,
                        keep_eta_cov=true, num_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        prevalence: Option<&Bound<'_, PyAny>>,
        prevalence_names: Option<Vec<String>>,
        content: Option<&Bound<'_, PyAny>>,
        content_names: Option<Vec<String>>,
        iters: usize,
        convergence_tol: f64,
        gamma_prior: &str,
        gamma_enet: f64,
        em_tol: Option<f64>,
        covariates: Option<&Bound<'_, PyAny>>,
        keep_eta_cov: bool,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "STM.fit(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (convergence_tol - 1e-5_f64).abs() > f64::EPSILON { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        // covariates= is a no-deprecation alias for prevalence=
        let prevalence = match (prevalence, covariates) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "STM.fit: pass either prevalence= or covariates=, not both",
                ));
            }
            (Some(p), None) => Some(p),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if prevalence.is_none() && content.is_none() {
            return Err(PyValueError::new_err(
                "STM needs prevalence and/or content covariates; use CTM for neither",
            ));
        }

        // --- Prevalence design (optional) ---
        let mut prevalence_x: Option<Vec<Vec<f64>>> = None;
        let mut feat_names: Vec<String> = Vec::new();
        if let Some(prev) = prevalence {
            let raw = parse_features(prev)?;
            if raw.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "prevalence has {} rows but corpus has {} documents",
                    raw.len(),
                    num_docs
                )));
            }
            check_all_finite_2d("prevalence", &raw)?;
            let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
            if raw.iter().any(|r| r.len() != f_in) {
                return Err(PyValueError::new_err("all prevalence rows must have the same length"));
            }
            if let Some(names) = &prevalence_names {
                if names.len() != f_in {
                    return Err(PyValueError::new_err(
                        "prevalence_names length must match the number of covariate columns",
                    ));
                }
            }
            let nf = f_in + 1;
            prevalence_x = Some(
                raw.iter()
                    .map(|r| {
                        let mut v = Vec::with_capacity(nf);
                        v.push(1.0);
                        v.extend_from_slice(r);
                        v
                    })
                    .collect(),
            );
            feat_names.push("intercept".to_string());
            feat_names.extend(
                prevalence_names
                    .unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
            );
        }

        // --- Content groups (optional) ---
        let mut content_groups: Option<(Vec<usize>, usize)> = None;
        let mut group_vocab: Vec<String> = Vec::new();
        if let Some(cont) = content {
            let groups_str = parse_groups(cont)?;
            if groups_str.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "content has {} entries but corpus has {} documents",
                    groups_str.len(),
                    num_docs
                )));
            }
            group_vocab = match content_names {
                Some(n) => n,
                None => {
                    let mut set: HashSet<String> = groups_str.iter().cloned().collect();
                    let mut v: Vec<String> = set.drain().collect();
                    v.sort();
                    v
                }
            };
            let gindex: HashMap<&str, usize> = group_vocab
                .iter()
                .enumerate()
                .map(|(i, g)| (g.as_str(), i))
                .collect();
            let idx: Vec<usize> = groups_str
                .iter()
                .map(|g| {
                    gindex.get(g.as_str()).copied().ok_or_else(|| {
                        PyValueError::new_err(format!("content group {:?} not in content_names", g))
                    })
                })
                .collect::<PyResult<_>>()?;
            content_groups = Some((idx, group_vocab.len()));
        }

        let gprior = match gamma_prior {
            "pooled" => ctm::GammaPrior::Pooled,
            "l1" => {
                if !(gamma_enet > 0.0 && gamma_enet <= 1.0) {
                    return Err(PyValueError::new_err("gamma_enet must be in (0, 1]"));
                }
                ctm::GammaPrior::L1 { alpha: gamma_enet }
            }
            other => return Err(PyValueError::new_err(format!(
                "gamma_prior must be \"pooled\" or \"l1\", got {:?}", other
            ))),
        };

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let shrink = self.sigma_shrink;
        let spectral = self.init_spectral;
        let diagonal = self.variational == "diagonal";
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let prev_ref = prevalence_x.as_deref();
            let cont_ref = content_groups.as_ref().map(|(g, n)| (g.as_slice(), *n));
            let m = run_with_threads(num_threads, || {
                ctm::fit_ctm(
                    &corpus.docs, k, num_types, iters, convergence_tol, shrink, prev_ref, cont_ref,
                    spectral, gprior, keep_eta_cov, diagonal, &mut rng,
                )
            });
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        for t in 0..k {
            for v in 0..num_types {
                beta[[t, v]] = model.beta[t][v];
            }
        }
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (di, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }
        let corr_v = model.topic_correlation();
        let mut corr = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                corr[[i, j]] = corr_v[i][j];
            }
        }
        self.gamma = model.gamma.as_ref().map(|g| {
            let nf = g.len();
            let mut arr = Array2::<f64>::zeros((nf, k - 1));
            for ff in 0..nf {
                for t in 0..(k - 1) {
                    arr[[ff, t]] = g[ff][t];
                }
            }
            arr
        });

        // Build eta_mean; only build eta_cov when keep_eta_cov=True.
        let mean_rows = model.eta_mean();
        let d_docs = mean_rows.len();
        let dim = k - 1;
        let mut eta_mean_arr = Array2::<f64>::zeros((d_docs, dim));
        for di in 0..d_docs {
            for i in 0..dim {
                eta_mean_arr[[di, i]] = mean_rows[di][i];
            }
        }
        let stored_eta_cov: Option<Array3<f32>> = if keep_eta_cov {
            let cov_rows = model.eta_cov();
            let mut cov = Array3::<f32>::zeros((d_docs, dim, dim));
            for di in 0..d_docs {
                for i in 0..dim {
                    for j in 0..dim {
                        cov[[di, i, j]] = cov_rows[di][i * dim + j] as f32;
                    }
                }
            }
            Some(cov)
        } else {
            None
        };

        // Store beta from the last E-step so _recompute_eta_cov uses the same
        // beta that was active when nu was computed (pre-final-M-step).
        let beta_estep_arr: Array2<f64> = {
            let rows = &model.beta_estep;
            let v = rows[0].len();
            let mut arr = Array2::<f64>::zeros((k, v));
            for (t, row) in rows.iter().enumerate() {
                for (vi, &val) in row.iter().enumerate() {
                    arr[[t, vi]] = val;
                }
            }
            arr
        };

        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
        self.eta_mean = Some(eta_mean_arr);
        self.eta_cov = stored_eta_cov;
        // E-step β snapshot: retained only when eta_cov was NOT kept (see below).
        if !keep_eta_cov {
            self.beta_estep = Some(beta_estep_arr);
        }
        self.feature_names = feat_names;
        self.content_beta = model.content_beta;
        self.group_names = group_vocab;
        self.mu = model.mu.clone();
        self.sigma = model.sigma.clone();
        // E-step Σ snapshot: retained only when eta_cov was NOT kept.
        if !keep_eta_cov {
            self.sigma_estep = model.sigma_estep.clone();
        }
        self.corpus = Some(corpus);
        self.bound = model.bound;
        self.bound_history = model.bound_history.clone();
        self.converged = model.converged;
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Final variational bound (approximate ELBO) at convergence — the quantity
    /// R `stm` reports as `convergence$bound`.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    /// The variational bound after each EM iteration (the convergence
    /// trajectory). Its length is the number of iterations actually run.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        self.require_fitted()?;
        Ok(self.bound_history.clone())
    }

    /// ``True`` if EM stopped on the `em_tol` criterion; ``False`` if it hit the
    /// `iters` cap first (the fit may not have converged).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Variational-covariance mode: ``"laplace"`` (full ν = H⁻¹) or
    /// ``"diagonal"`` (mean-field ν = diag(1/H_ii)).
    #[getter]
    fn variational(&self) -> String {
        self.variational.clone()
    }

    /// Uniform convergence trace: ``(iteration, bound)`` pairs, one per EM
    /// iteration. The objective is the variational ELBO (same as
    /// :attr:`bound_history`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Topic-correlation matrix, shape ``(num_topics, num_topics)``.
    #[getter]
    fn topic_correlation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.corr.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior means λ of η, shape
    /// ``(num_docs, num_topics-1)``. With :attr:`eta_cov` this is the
    /// logistic-normal posterior used to draw θ samples for
    /// method-of-composition uncertainty in ``estimate_effect``.
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, num_topics-1, num_topics-1)``. Stored as float32 in memory
    /// to halve the dominant memory term; cast to float64 with
    /// ``np.asarray(model.eta_cov, dtype=np.float64)`` when full precision is needed.
    /// Raises RuntimeError if the model was fit with ``keep_eta_cov=False``; use
    /// :meth:`_recompute_eta_cov` to regenerate on demand.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        self.eta_cov.as_ref()
            .map(|c| c.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err(
                "model was fit with keep_eta_cov=False; refit with keep_eta_cov=True, \
                 or use posterior_theta_samples/_recompute_eta_cov which recompute it on demand"
            ))
    }

    /// Recompute the per-document variational covariance ν on demand.
    /// Use this when the model was fit with ``keep_eta_cov=False`` to save memory.
    /// Returns the same ``(num_docs, K-1, K-1)`` float32 array as :attr:`eta_cov`.
    fn _recompute_eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        let corpus = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no corpus retained; cannot recompute eta_cov")
        })?;
        let k = self.num_topics;
        let km1 = k - 1;
        let sparse: Vec<(Vec<usize>, Vec<f64>)> = corpus.docs.iter()
            .map(|doc| crate::variational::doc_sparse(doc))
            .collect();
        // Use beta_estep (the topic-word matrix from the last E-step, before the
        // final M-step updated beta) so the Hessian computation is bit-identical
        // to what was used when nu was originally stored.
        let beta_src = self.beta_estep.as_ref().unwrap_or_else(|| self.beta.as_ref().unwrap());
        let beta_v: Vec<Vec<f64>> = beta_src
            .outer_iter().map(|r| r.to_vec()).collect();
        let lambda_v: Vec<Vec<f64>> = self.eta_mean.as_ref().unwrap()
            .outer_iter().map(|r| r.to_vec()).collect();
        let d = lambda_v.len();
        // ν is independent of the prior mean μ, so recompute_nu uses self.mu for
        // every document.
        let sigma_for_recompute = if !self.sigma_estep.is_empty() {
            self.sigma_estep.clone()
        } else {
            self.sigma.clone()
        };
        let model_stub = ctm::CtmModel {
            num_topics: k,
            num_types: corpus.num_types(),
            beta: beta_v.clone(),
            beta_estep: beta_v,
            mu: self.mu.clone(),
            sigma: self.sigma.clone(),
            sigma_estep: sigma_for_recompute,
            lambda: lambda_v,
            nu: Vec::new(),
            gamma: None,
            content_beta: None,
            num_groups: 1,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
            em_iters_run: 0,
            // Recompute ν in the same mode the fit used (laplace/diagonal).
            diagonal: self.variational == "diagonal",
        };
        let nu = py.allow_threads(|| ctm::recompute_nu(&model_stub, &sparse));
        let mut out = Array3::<f32>::zeros((d, km1, km1));
        for di in 0..d {
            for i in 0..km1 {
                for j in 0..km1 {
                    out[[di, i, j]] = nu[di][i * km1 + j] as f32;
                }
            }
        }
        Ok(out.to_pyarray_bound(py))
    }

    /// The fitted logistic-normal prior covariance Σ over η, shape
    /// ``(num_topics-1, num_topics-1)`` (the last topic is the softmax reference,
    /// so it is dropped). This is the model's own topic covariance — unlike
    /// :attr:`topic_correlation`, which is an across-document θ correlation.
    #[getter]
    fn topic_covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let km1 = self.num_topics.saturating_sub(1);
        if self.sigma.len() != km1 * km1 {
            return Err(PyRuntimeError::new_err(
                "this model was fit before topic_covariance was stored; refit to use it",
            ));
        }
        let arr = Array2::from_shape_vec((km1, km1), self.sigma.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.to_pyarray_bound(py))
    }

    /// Prevalence coefficients γ, shape ``(num_features, num_topics-1)`` — how
    /// each covariate (row 0 is the intercept) shifts each topic's log-prior.
    /// The last topic is the softmax reference. For inference, prefer
    /// ``topica.stm.estimate_effect(model.doc_topic, X)``.
    #[getter]
    fn prevalence_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let g = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without prevalence covariates"))?;
        Ok(g.to_pyarray_bound(py))
    }

    /// Per-group topic-word distributions, shape ``(num_topics, num_groups,
    /// num_words)`` — only available when fit with `content` covariates.
    #[getter]
    fn topic_word_by_group<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        let cb = self.content_beta.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("model was fit without content covariates")
        })?;
        let g = cb.len();
        let k = self.num_topics;
        let v = self.corpus.as_ref().unwrap().num_types();
        // cb is G×K×V; expose as (topics, groups, words).
        let mut arr = Array3::<f64>::zeros((k, g, v));
        for gg in 0..g {
            for t in 0..k {
                for w in 0..v {
                    arr[[t, gg, w]] = cb[gg][t][w];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Content-covariate group names (axis-1 order of :attr:`topic_word_by_group`).
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        if self.group_names.is_empty() {
            return Err(PyRuntimeError::new_err("model was fit without content covariates"));
        }
        Ok(self.group_names.clone())
    }

    /// Words that most distinguish how `topic` is worded in `group_a` vs
    /// `group_b` (log word-probability ratio; positive favours `group_a`).
    /// Requires content covariates.
    #[pyo3(signature = (topic, group_a, group_b, n=10))]
    fn word_contrast<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        group_a: &Bound<'py, PyAny>,
        group_b: &Bound<'py, PyAny>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let cb = self.content_beta.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("model was fit without content covariates")
        })?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let ga = self.resolve_group(group_a)?;
        let gb = self.resolve_group(group_b)?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let a = &cb[ga][topic];
        let b = &cb[gb][topic];
        let ratio: Vec<f64> = (0..vocab.len())
            .map(|v| (a[v].max(1e-300) / b[v].max(1e-300)).ln())
            .collect();
        let mut idx: Vec<usize> = (0..vocab.len()).collect();
        idx.sort_by(|&x, &y| f64::total_cmp(&ratio[y], &ratio[x]));
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), ratio[v].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Covariate names aligned with the rows of :attr:`prevalence_effects`
    /// (``"intercept"`` first).
    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by the variational E-step
    /// against the fitted globals (β and the logistic-normal prior). `data` is a
    /// :class:`Corpus` or `list[list[str]]`; out-of-vocabulary tokens are dropped.
    /// Returns a ``(num_docs, num_topics)`` array.
    ///
    /// When `eta_prior_mean` is ``None`` (the default), the covariate-free
    /// baseline μ learned at fit time is used for every document — the same
    /// inference that ``stm``'s ``fitNewDocuments`` performs when no new
    /// covariate design is supplied.
    ///
    /// When `eta_prior_mean` is a ``(num_docs, num_topics-1)`` array, each
    /// document's prior mean is set to the corresponding row.  This is the
    /// low-level hook used by :func:`topica.stm.transform` to apply the
    /// prevalence-covariate prior ``μ_d = X_d γ`` to held-out documents.
    #[pyo3(signature = (data, *, eta_prior_mean=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        eta_prior_mean: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let beta = self.beta.as_ref().unwrap();
        let beta_v: Vec<Vec<f64>> = beta.outer_iter().map(|r| r.to_vec()).collect();
        let theta = if let Some(mu_arr) = eta_prior_mean {
            let mu_nd = mu_arr.as_array();
            let nd = docs.len();
            let km1 = self.mu.len();
            if mu_nd.shape() != [nd, km1] {
                return Err(PyValueError::new_err(format!(
                    "eta_prior_mean must have shape ({nd}, {km1}); got {:?}",
                    mu_nd.shape()
                )));
            }
            let owned = mu_nd.to_owned();
            infer_theta_batch_per_doc(py, &beta_v, &owned, &self.sigma, &docs)
        } else {
            infer_theta_batch(py, &beta_v, &self.mu, &self.sigma, &docs)
        };
        Ok(theta.to_pyarray_bound(py))
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `STM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        // eta_cov is stored as f32 in memory; upcast to f64 for the on-disk format
        // so existing saved models remain compatible.
        let eta_cov_f64 = self.eta_cov.as_ref().map(|c| c.mapv(|x| x as f64));
        write_state(path, MODEL_TAG_STM, &StmState {
            num_topics: self.num_topics, sigma_shrink: self.sigma_shrink, seed: self.seed,
            init_spectral: self.init_spectral, fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corr: arr2_opt(&self.corr),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&eta_cov_f64),
            gamma: arr2_opt(&self.gamma), feature_names: self.feature_names.clone(),
            content_beta: self.content_beta.clone(),
            mu: self.mu.clone(), sigma: self.sigma.clone(),
            group_names: self.group_names.clone(),
            corpus: self.corpus.clone(),
            bound: self.bound, bound_history: self.bound_history.clone(),
            converged: self.converged,
            topic_names: self.topic_names.clone(),
            variational: self.variational.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: StmState = read_state(path, MODEL_TAG_STM)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        // eta_cov is saved as f64 for format compatibility; downcast to f32 in memory.
        let eta_cov = arr3_back(s.eta_cov).map(|c| c.mapv(|x| x as f32));
        Ok(STM {
            num_topics: s.num_topics, sigma_shrink: s.sigma_shrink, seed: s.seed,
            init_spectral: s.init_spectral, variational: s.variational, fitted: s.fitted,
            topic_names,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corr: arr2_back(s.corr),
            eta_mean: arr2_back(s.eta_mean), eta_cov,
            gamma: arr2_back(s.gamma), feature_names: s.feature_names,
            content_beta: s.content_beta,
            mu: s.mu, sigma: s.sigma,
            sigma_estep: Vec::new(),  // not persisted; falls back to sigma in _recompute_eta_cov
            beta_estep: None,         // not persisted; falls back to self.beta
            group_names: s.group_names, corpus: s.corpus,
            bound: s.bound, bound_history: s.bound_history, converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!("STM(num_topics={}, variational={:?}, fitted={})", self.num_topics, self.variational, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// ECTM: Evolving Content Topic Model (STM core + dynamic content covariates)
// ---------------------------------------------------------------------------

/// Evolving Content Topic Model. An STM whose **content** (topic-word) model
/// carries a group-by-time interaction: the same stable topic is worded
/// differently across a document **group** covariate, and that difference
/// **evolves** across discrete time **periods**. Where :class:`STM`'s content
/// model gives one word distribution per group, ECTM gives one per
/// (group, period) cell, tied across adjacent periods by a random-walk prior so
/// sparse cells borrow strength from their temporal neighbours.
///
/// After fitting, :meth:`content_word_dist` returns the topic-word matrix for
/// any (group, period) cell; the helpers in :mod:`topica.ectm`
/// (``content_words`` / ``content_contrast`` / ``content_trajectory``) read
/// those to report how a group's vocabulary on a topic shifts over time.
///
/// The prevalence side is the standard STM regression ``μ_d = X_d γ`` and is
/// optional; ``content`` (a per-document group label) and ``times`` (a
/// per-document period label) are required.
#[pyclass(module = "topica")]
pub struct ECTM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    variational: String,

    fitted: bool,
    topic_names: Vec<String>,
    beta: Option<Array2<f64>>,        // K×V, cell-averaged topic-word
    theta: Option<Array2<f64>>,       // D×K
    content_beta: Vec<Vec<Vec<f64>>>, // [cell][k][v], cell = g*num_periods + t
    num_groups: usize,
    num_periods: usize,
    group_names: Vec<String>,
    period_names: Vec<String>,
    eta_mean: Option<Array2<f64>>,
    eta_cov: Option<Array3<f32>>,
    gamma: Option<Array2<f64>>, // (num_features, K-1); None if no prevalence
    feature_names: Vec<String>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    corpus: Option<corpus::Corpus>,
    bound: f64,
    bound_history: Vec<f64>,
    converged: bool,
}

impl ECTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    fn resolve_group(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.group_names.len() {
                return Ok(i);
            }
            return Err(PyValueError::new_err("group index out of range"));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .group_names
                .iter()
                .position(|g| g == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown group {:?}", s)));
        }
        Err(PyValueError::new_err("group must be a name (str) or index (int)"))
    }

    fn resolve_period(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.period_names.len() {
                return Ok(i);
            }
            return Err(PyValueError::new_err("period index out of range"));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .period_names
                .iter()
                .position(|p| p == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown period {:?}", s)));
        }
        Err(PyValueError::new_err("period must be a label (str) or index (int)"))
    }
}

#[pymethods]
impl ECTM {
    /// Create an unfitted model. `sigma_shrink` ∈ [0,1] shrinks Σ toward its
    /// diagonal each M-step. `variational` selects the per-document
    /// variational-covariance mode: ``"laplace"`` (default, full ν = H⁻¹) or
    /// ``"diagonal"`` (mean-field). ECTM always uses a seeded random init for the
    /// content model, so `seed` matters for reproducibility.
    #[new]
    #[pyo3(signature = (num_topics, *, sigma_shrink=0.0, seed=42, variational="laplace"))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        sigma_shrink: f64,
        seed: u64,
        variational: &str,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !(0.0..=1.0).contains(&sigma_shrink) {
            return Err(PyValueError::new_err("sigma_shrink must be in [0, 1]"));
        }
        if variational != "laplace" && variational != "diagonal" {
            return Err(PyValueError::new_err("variational must be 'laplace' or 'diagonal'"));
        }
        Ok(ECTM {
            num_topics,
            sigma_shrink,
            seed,
            variational: variational.to_string(),
            fitted: false,
            topic_names: Vec::new(),
            beta: None,
            theta: None,
            content_beta: Vec::new(),
            num_groups: 0,
            num_periods: 0,
            group_names: Vec::new(),
            period_names: Vec::new(),
            eta_mean: None,
            eta_cov: None,
            gamma: None,
            feature_names: Vec::new(),
            mu: Vec::new(),
            sigma: Vec::new(),
            corpus: None,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit. `data` is a :class:`Corpus` or `list[list[str]]`. `times` is one
    /// period label per document (numbers or strings; the distinct values, sorted,
    /// define the period order). `content` is one group label per document. Both
    /// are required — together they form the (group, period) content cells.
    /// `prevalence` (optional, `(num_docs, F)`) adds the STM prevalence regression
    /// `μ_d = X_d γ` (an intercept is prepended).
    ///
    /// The content κ are regularized by: an L2 prior with variance
    /// `content_prior_var`; a first-order random-walk across periods with
    /// precision `period_smooth` (larger ⇒ smoother, more pooling across adjacent
    /// periods); and an extra L2 factor `interaction_shrink` on the group×time
    /// term (larger ⇒ the changing contrast is pulled harder toward zero unless
    /// the data demand it). EM runs until the relative change in the variational
    /// bound drops below `convergence_tol` or `iters` iterations are reached.
    #[pyo3(signature = (data, times, content, *, prevalence=None, prevalence_names=None,
                        content_names=None, period_names=None, iters=500, convergence_tol=1e-5,
                        content_prior_var=1.0, period_smooth=5.0, interaction_shrink=2.0,
                        keep_eta_cov=true, num_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        times: &Bound<'_, PyAny>,
        content: &Bound<'_, PyAny>,
        prevalence: Option<&Bound<'_, PyAny>>,
        prevalence_names: Option<Vec<String>>,
        content_names: Option<Vec<String>>,
        period_names: Option<Vec<String>>,
        iters: usize,
        convergence_tol: f64,
        content_prior_var: f64,
        period_smooth: f64,
        interaction_shrink: f64,
        keep_eta_cov: bool,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        if content_prior_var <= 0.0 {
            return Err(PyValueError::new_err("content_prior_var must be > 0"));
        }
        if period_smooth < 0.0 {
            return Err(PyValueError::new_err("period_smooth must be >= 0"));
        }
        if interaction_shrink <= 0.0 {
            return Err(PyValueError::new_err("interaction_shrink must be > 0"));
        }

        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        // --- Period index from times ---
        let (period_idx, mut period_vocab) = build_time_index(times, num_docs)?;
        if let Some(names) = period_names {
            if names.len() != period_vocab.len() {
                return Err(PyValueError::new_err(format!(
                    "period_names has {} entries but the data has {} distinct periods",
                    names.len(),
                    period_vocab.len()
                )));
            }
            period_vocab = names;
        }
        let num_periods = period_vocab.len();

        // --- Group index from content ---
        let groups_str = parse_groups(content)?;
        if groups_str.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "content has {} entries but corpus has {} documents",
                groups_str.len(),
                num_docs
            )));
        }
        let group_vocab = match content_names {
            Some(n) => n,
            None => {
                let mut set: HashSet<String> = groups_str.iter().cloned().collect();
                let mut v: Vec<String> = set.drain().collect();
                v.sort();
                v
            }
        };
        let gindex: HashMap<&str, usize> =
            group_vocab.iter().enumerate().map(|(i, g)| (g.as_str(), i)).collect();
        let group_idx: Vec<usize> = groups_str
            .iter()
            .map(|g| {
                gindex.get(g.as_str()).copied().ok_or_else(|| {
                    PyValueError::new_err(format!("content group {:?} not in content_names", g))
                })
            })
            .collect::<PyResult<_>>()?;
        let num_groups = group_vocab.len();

        // --- Prevalence design (optional) ---
        let mut prevalence_x: Option<Vec<Vec<f64>>> = None;
        let mut feat_names: Vec<String> = Vec::new();
        if let Some(prev) = prevalence {
            let raw = parse_features(prev)?;
            if raw.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "prevalence has {} rows but corpus has {} documents",
                    raw.len(),
                    num_docs
                )));
            }
            check_all_finite_2d("prevalence", &raw)?;
            let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
            if raw.iter().any(|r| r.len() != f_in) {
                return Err(PyValueError::new_err("all prevalence rows must have the same length"));
            }
            if let Some(names) = &prevalence_names {
                if names.len() != f_in {
                    return Err(PyValueError::new_err(
                        "prevalence_names length must match the number of covariate columns",
                    ));
                }
            }
            let nf = f_in + 1;
            prevalence_x = Some(
                raw.iter()
                    .map(|r| {
                        let mut v = Vec::with_capacity(nf);
                        v.push(1.0);
                        v.extend_from_slice(r);
                        v
                    })
                    .collect(),
            );
            feat_names.push("intercept".to_string());
            feat_names.extend(
                prevalence_names.unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
            );
        }

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let shrink = self.sigma_shrink;
        let diagonal = self.variational == "diagonal";
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let prev_ref = prevalence_x.as_deref();
            let m = run_with_threads(num_threads, || {
                crate::ectm::fit_ectm(
                    &corpus.docs, k, num_types, &group_idx, num_groups, &period_idx, num_periods,
                    iters, convergence_tol, shrink, prev_ref, content_prior_var, period_smooth,
                    period_smooth, interaction_shrink, keep_eta_cov, diagonal, &mut rng,
                )
            });
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        for t in 0..k {
            for v in 0..num_types {
                beta[[t, v]] = model.beta[t][v];
            }
        }
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (di, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }
        self.gamma = model.gamma.as_ref().map(|g| {
            let nf = g.len();
            let mut arr = Array2::<f64>::zeros((nf, k - 1));
            for ff in 0..nf {
                for t in 0..(k - 1) {
                    arr[[ff, t]] = g[ff][t];
                }
            }
            arr
        });

        let dim = k - 1;
        let d_docs = model.lambda.len();
        let mut eta_mean_arr = Array2::<f64>::zeros((d_docs, dim));
        for di in 0..d_docs {
            for i in 0..dim {
                eta_mean_arr[[di, i]] = model.lambda[di][i];
            }
        }
        let stored_eta_cov: Option<Array3<f32>> = if keep_eta_cov && !model.nu.is_empty() {
            let mut cov = Array3::<f32>::zeros((d_docs, dim, dim));
            for di in 0..d_docs {
                for i in 0..dim {
                    for j in 0..dim {
                        cov[[di, i, j]] = model.nu[di][i * dim + j] as f32;
                    }
                }
            }
            Some(cov)
        } else {
            None
        };

        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.content_beta = model.content_beta;
        self.num_groups = num_groups;
        self.num_periods = num_periods;
        self.group_names = group_vocab;
        self.period_names = period_vocab;
        self.eta_mean = Some(eta_mean_arr);
        self.eta_cov = stored_eta_cov;
        self.feature_names = feat_names;
        self.mu = model.mu.clone();
        self.sigma = model.sigma.clone();
        self.corpus = Some(corpus);
        self.bound = model.bound;
        self.bound_history = model.bound_history.clone();
        self.converged = model.converged;
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)`` — averaged over all
    /// (group, period) cells. Use :meth:`content_word_dist` for a specific cell.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// The content groups, in index order (the column order of cells).
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.group_names.clone())
    }

    /// The time periods, sorted into the order used for the random-walk prior.
    #[getter]
    fn periods(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.period_names.clone())
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.num_groups
    }

    #[getter]
    fn num_periods(&self) -> usize {
        self.num_periods
    }

    /// Topic-word matrix β for one (group, period) cell, shape
    /// ``(num_topics, num_words)``. `group` and `period` accept either a label
    /// (str) or an index (int). This is the content surface that the
    /// :mod:`topica.ectm` helpers read.
    fn content_word_dist<'py>(
        &self,
        py: Python<'py>,
        group: &Bound<'py, PyAny>,
        period: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let g = self.resolve_group(group)?;
        let t = self.resolve_period(period)?;
        let c = g * self.num_periods + t;
        let v = self.content_beta[c][0].len();
        let mut arr = Array2::<f64>::zeros((self.num_topics, v));
        for k in 0..self.num_topics {
            for w in 0..v {
                arr[[k, w]] = self.content_beta[c][k][w];
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Vocabulary, the column order of :attr:`topic_word`.
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Prevalence coefficients γ, shape ``(num_features, num_topics-1)``; raises
    /// if the model was fit without a prevalence design.
    #[getter]
    fn prevalence_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        self.gamma
            .as_ref()
            .map(|g| g.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without prevalence covariates"))
    }

    /// Prevalence feature names (``["intercept", ...]``), aligned with
    /// :attr:`prevalence_effects` rows. Empty if no prevalence design.
    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    /// Per-document variational posterior means λ of η, shape
    /// ``(num_docs, num_topics-1)``.
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, num_topics-1, num_topics-1)`` as float32. Raises if fit with
    /// ``keep_eta_cov=False``.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        self.eta_cov.as_ref().map(|c| c.to_pyarray_bound(py)).ok_or_else(|| {
            PyRuntimeError::new_err("model was fit with keep_eta_cov=False; refit with keep_eta_cov=True")
        })
    }

    /// Final variational bound (approximate ELBO) at convergence.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    /// The variational bound after each EM iteration.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        self.require_fitted()?;
        Ok(self.bound_history.clone())
    }

    /// ``True`` if EM stopped on the `convergence_tol` criterion; ``False`` if it
    /// hit the `iters` cap first.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Uniform convergence trace: ``(iteration, bound)`` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.bound_history.iter().enumerate().map(|(i, &b)| (i + 1, b)).collect())
    }

    /// Variational-covariance mode (``"laplace"`` or ``"diagonal"``).
    #[getter]
    fn variational(&self) -> String {
        self.variational.clone()
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs,
    /// from the cell-averaged β.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)]))
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> = (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// One label per topic, in topic order.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `ECTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        let eta_cov_f64 = self.eta_cov.as_ref().map(|c| c.mapv(|x| x as f64));
        write_state(path, MODEL_TAG_ECTM, &EctmState {
            num_topics: self.num_topics, sigma_shrink: self.sigma_shrink, seed: self.seed,
            fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&eta_cov_f64),
            gamma: arr2_opt(&self.gamma), feature_names: self.feature_names.clone(),
            content_beta: self.content_beta.clone(),
            num_groups: self.num_groups, num_periods: self.num_periods,
            group_names: self.group_names.clone(), period_names: self.period_names.clone(),
            mu: self.mu.clone(), sigma: self.sigma.clone(),
            corpus: self.corpus.clone(),
            bound: self.bound, bound_history: self.bound_history.clone(),
            converged: self.converged,
            topic_names: self.topic_names.clone(),
            variational: self.variational.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: EctmState = read_state(path, MODEL_TAG_ECTM)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        let eta_cov = arr3_back(s.eta_cov).map(|c| c.mapv(|x| x as f32));
        Ok(ECTM {
            num_topics: s.num_topics, sigma_shrink: s.sigma_shrink, seed: s.seed,
            variational: s.variational, fitted: s.fitted, topic_names,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta),
            content_beta: s.content_beta,
            num_groups: s.num_groups, num_periods: s.num_periods,
            group_names: s.group_names, period_names: s.period_names,
            eta_mean: arr2_back(s.eta_mean), eta_cov,
            gamma: arr2_back(s.gamma), feature_names: s.feature_names,
            mu: s.mu, sigma: s.sigma, corpus: s.corpus,
            bound: s.bound, bound_history: s.bound_history, converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ECTM(num_topics={}, num_groups={}, num_periods={}, fitted={})",
            self.num_topics, self.num_groups, self.num_periods, self.fitted
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Window/document co-occurrence counts for coherence scoring.
///
/// `docs` holds relevant-word ids per token (`4294967295` marks a non-relevant
/// token). `pairs` are `(a, b)` with `a < b`. `window == 0` requests
/// document-level co-occurrence (one window per document, for UMass); a positive
/// width slides a window one token at a time. Returns
/// `(occ[num_relevant], co[len(pairs)], n_windows)`.
#[pyfunction]
fn window_cooccurrence(
    py: Python<'_>,
    docs: Vec<Vec<u32>>,
    num_relevant: usize,
    pairs: Vec<(u32, u32)>,
    window: u32,
) -> (Vec<f64>, Vec<f64>, f64) {
    py.allow_threads(move || coh::cooccurrence(&docs, num_relevant, &pairs, window))
}

/// Warn that a neighbor-preserving projection (UMAP / t-SNE) distorts global
/// geometry and is not reproducible across runs, so PCA stays the honest default.
fn warn_stochastic(py: Python<'_>, method: &str) -> PyResult<()> {
    let warnings = py.import_bound("warnings")?;
    warnings.call_method1(
        "warn",
        (format!(
            "method='{method}' preserves local neighborhoods but distorts global \
             geometry (between-cluster distances and cluster sizes are not meaningful) \
             and is not reproducible across runs. Use method='pca' for a deterministic, \
             distance-faithful projection."
        ),),
    )?;
    Ok(())
}

/// Project a high-dimensional array to a low-dimensional layout (for plotting or
/// clustering). `method` is "pca" (default, deterministic, distance-faithful),
/// "umap", or "tsne"; the latter two preserve local neighborhoods but distort
/// global geometry and are not reproducible (a warning is issued). `data` is a 2D
/// float array or a list of float lists. Returns an `(n_rows, n_components)` array.
#[pyfunction]
#[pyo3(signature = (data, n_components=2, *, method="pca", n_neighbors=15, perplexity=30.0, seed=0))]
fn project<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    n_components: usize,
    method: &str,
    n_neighbors: usize,
    perplexity: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = parse_features(data)?;
    match method {
        "pca" => {}
        "umap" => {
            if !crate::reduce::umap_available() {
                return Err(PyRuntimeError::new_err(
                    "method='umap' is not available in this build; rebuild with the \
                     `umap` feature, or use method='pca' (the default)",
                ));
            }
            warn_stochastic(py, "umap")?;
        }
        "tsne" => {
            if !crate::reduce::tsne_available() {
                return Err(PyRuntimeError::new_err(
                    "method='tsne' is not available in this build; rebuild with the \
                     `tsne` feature, or use method='pca' (the default)",
                ));
            }
            warn_stochastic(py, "tsne")?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected 'pca', 'umap', or 'tsne'"
            )));
        }
    }
    if n_components == 0 {
        return Err(PyValueError::new_err("n_components must be >= 1"));
    }
    let n = rows.len();
    let method = method.to_string(); // own it so the GIL can be released
    let out = py.allow_threads(move || {
        crate::reduce::project(&rows, n_components, &method, n_neighbors, perplexity, 0.5, 1000, seed)
    });
    let mut arr = Array2::<f64>::zeros((n, n_components));
    for (i, r) in out.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            if j < n_components {
                arr[[i, j]] = v;
            }
        }
    }
    Ok(arr.to_pyarray_bound(py))
}

/// Tokenize a string the way the corpus loader does: find regex tokens,
/// optionally lowercase, drop short tokens and stopwords. Handy for building
/// `list[list[str]]` input outside of `Corpus.from_text_file`.
#[pyfunction]
#[pyo3(signature = (text, *, lowercase=true, stopwords=None, token_regex=None, min_length=1))]
fn tokenize(
    text: &str,
    lowercase: bool,
    stopwords: Option<&Bound<'_, PyAny>>,
    token_regex: Option<String>,
    min_length: usize,
) -> PyResult<Vec<String>> {
    let pattern = token_regex.unwrap_or_else(|| corpus::DEFAULT_TOKEN_REGEX.to_string());
    let re = Regex::new(&pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Accept any iterable of strings (list, tuple, set, frozenset) so a
    // `ENGLISH_STOPWORDS` frozenset can be passed directly.
    let stop: HashSet<String> = match stopwords {
        Some(obj) => {
            let mut s = HashSet::new();
            for item in obj.iter()? {
                s.insert(item?.extract::<String>()?);
            }
            s
        }
        None => HashSet::new(),
    };

    let mut out = Vec::new();
    for m in re.find_iter(text) {
        let tok = if lowercase {
            m.as_str().to_lowercase()
        } else {
            m.as_str().to_string()
        };
        if tok.chars().count() < min_length {
            continue;
        }
        if stop.contains(&tok) {
            continue;
        }
        out.push(tok);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// STS: Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024)
// ---------------------------------------------------------------------------

/// β_{k,·} at a given sentiment level: softmax over the vocabulary of
/// `m_v + κ^(t)_{k,v} + κ^(s)_{k,v}·level`.
fn sts_beta_at(
    kappa_t: &[Vec<f64>],
    kappa_s: &[Vec<f64>],
    mv: &[f64],
    k: usize,
    v: usize,
    level: f64,
) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((k, v));
    for t in 0..k {
        let mut mx = f64::NEG_INFINITY;
        let mut lin = vec![0.0f64; v];
        for i in 0..v {
            lin[i] = mv[i] + kappa_t[t][i] + kappa_s[t][i] * level;
            if lin[i] > mx {
                mx = lin[i];
            }
        }
        let mut s = 0.0;
        for x in lin.iter_mut() {
            *x = (*x - mx).exp();
            s += *x;
        }
        for i in 0..v {
            b[[t, i]] = lin[i] / s;
        }
    }
    b
}

/// Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024, *Management
/// Science*). STS extends STM with a per-document, per-topic **continuous
/// sentiment-discourse** latent `α^(s)` that modulates the topic-word
/// distribution, with both topic prevalence and sentiment-discourse driven by
/// document covariates. Fit by Laplace variational EM (a faithful port of the
/// authors' R ``sts`` package).
#[pyclass(module = "topica")]
pub struct STS {
    num_topics: usize,
    seed: u64,
    init_spectral: bool,

    fitted: bool,
    topic_names: Vec<String>,
    beta: Option<Array2<f64>>,      // K×V baseline topic-word (α^(s)=0)
    theta: Option<Array2<f64>>,     // D×K prevalence
    sentiment: Option<Array2<f64>>, // D×K topic sentiment-discourse α^(s)
    gamma: Option<Array2<f64>>,     // F×(2K-1) prevalence+sentiment regression
    feature_names: Vec<String>,
    kappa_t: Vec<Vec<f64>>, // K×V (final, after last κ M-step)
    kappa_s: Vec<Vec<f64>>, // K×V (final, after last κ M-step)
    mv: Vec<f64>,           // V
    sigma: Vec<f64>,        // (2K-1)²
    eta_mean: Option<Array2<f64>>,  // D×(2K-1)
    eta_cov: Option<Array3<f32>>,   // D×(2K-1)×(2K-1) — stored as f32 to halve memory; None when fit with keep_eta_cov=False
    /// Sigma from the last E-step (before the final Σ M-step). Used by
    /// `_recompute_eta_cov` to reproduce ν exactly. Empty when not needed (loaded
    /// models) — falls back to `sigma` in that case.
    sigma_estep: Vec<f64>,
    /// kappa from the last E-step (before the final κ M-step). Used by
    /// `_recompute_eta_cov` to reproduce ν exactly. Empty when not retained
    /// (loaded models) — falls back to kappa_t/kappa_s in that case.
    kappa_t_estep: Vec<Vec<f64>>,
    kappa_s_estep: Vec<Vec<f64>>,
    corpus: Option<corpus::Corpus>,
    bound: f64,
    bound_history: Vec<f64>,
    converged: bool,
}

impl STS {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl STS {
    /// Create an unfitted model. `init` is ``"spectral"`` (default; deterministic
    /// anchor-word β init) or ``"random"`` (seeded).
    #[new]
    #[pyo3(signature = (num_topics, *, seed=42, init="spectral"))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        seed: u64,
        init: &str,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        Ok(STS {
            num_topics,
            seed,
            init_spectral,
            fitted: false,
            topic_names: Vec::new(),
            beta: None,
            theta: None,
            sentiment: None,
            gamma: None,
            feature_names: Vec::new(),
            kappa_t: Vec::new(),
            kappa_s: Vec::new(),
            mv: Vec::new(),
            sigma: Vec::new(),
            sigma_estep: Vec::new(),
            kappa_t_estep: Vec::new(),
            kappa_s_estep: Vec::new(),
            eta_mean: None,
            eta_cov: None,
            corpus: None,
            bound: f64::NAN,
            bound_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit. `data` is a :class:`Corpus` or ``list[list[str]]``. `sentiment_seed`
    /// (required, one value per document) defines the discrete aggregation groups
    /// for the κ Poisson M-step and seeds the initial sentiment — typically a
    /// document attribute the sentiment should track (e.g. a star rating).
    /// `prevalence` (optional, ``(num_docs, F)`` covariates) makes both topic
    /// prevalence and sentiment-discourse depend on covariates (`α_d ~ N(X_d Γ,
    /// Σ)`); an intercept is prepended.
    ///
    /// EM runs until the relative change in the variational bound drops below
    /// `convergence_tol` or `iters` iterations are reached.
    ///
    /// `kappa_estimation` chooses the topic-word (κ) estimator: ``"ridge"``
    /// (default) is a fast ridge-penalized Poisson fit (`kappa_ridge` sets the
    /// ridge); ``"lasso"`` is an L1 Poisson path with AIC-selected penalty,
    /// matching the reference R `sts` exactly (sparser κ) at a higher cost. The
    /// two give the same topics on well-conditioned corpora.
    #[pyo3(signature = (data, sentiment_seed, prevalence=None, *,
                        prevalence_names=None, iters=30, convergence_tol=1e-5,
                        kappa_estimation="ridge", kappa_ridge=1e-3, em_tol=None, covariates=None,
                        keep_eta_cov=true))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        sentiment_seed: Vec<f64>,
        prevalence: Option<&Bound<'_, PyAny>>,
        prevalence_names: Option<Vec<String>>,
        iters: usize,
        convergence_tol: f64,
        kappa_estimation: &str,
        kappa_ridge: f64,
        em_tol: Option<f64>,
        covariates: Option<&Bound<'_, PyAny>>,
        keep_eta_cov: bool,
    ) -> PyResult<()> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "STS.fit(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (convergence_tol - 1e-5_f64).abs() > f64::EPSILON { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        // covariates= is a no-deprecation alias for prevalence=
        let prevalence = match (prevalence, covariates) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "STS.fit: pass either prevalence= or covariates=, not both",
                ));
            }
            (Some(p), None) => Some(p),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };
        let kappa_est = match kappa_estimation {
            "lasso" => sts::KappaEst::Lasso { nlambda: 100, lambda_min_ratio: 0.001 },
            "ridge" => sts::KappaEst::Ridge(kappa_ridge),
            other => {
                return Err(PyValueError::new_err(format!(
                    "kappa_estimation must be \"lasso\" or \"ridge\", got {:?}", other
                )))
            }
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if sentiment_seed.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "sentiment_seed has {} values but corpus has {} documents",
                sentiment_seed.len(),
                num_docs
            )));
        }

        // Prevalence design (optional): prepend an intercept column.
        let mut prevalence_x: Option<Vec<Vec<f64>>> = None;
        let mut feat_names: Vec<String> = Vec::new();
        if let Some(prev) = prevalence {
            let raw = parse_features(prev)?;
            if raw.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "prevalence has {} rows but corpus has {} documents",
                    raw.len(),
                    num_docs
                )));
            }
            check_all_finite_2d("prevalence", &raw)?;
            let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
            if raw.iter().any(|r| r.len() != f_in) {
                return Err(PyValueError::new_err("all prevalence rows must have the same length"));
            }
            if let Some(names) = &prevalence_names {
                if names.len() != f_in {
                    return Err(PyValueError::new_err(
                        "prevalence_names length must match the number of covariate columns",
                    ));
                }
            }
            let x: Vec<Vec<f64>> = raw
                .iter()
                .map(|r| {
                    let mut row = Vec::with_capacity(f_in + 1);
                    row.push(1.0);
                    row.extend_from_slice(r);
                    row
                })
                .collect();
            feat_names.push("(Intercept)".to_string());
            match &prevalence_names {
                Some(names) => feat_names.extend(names.iter().cloned()),
                None => feat_names.extend((0..f_in).map(|i| format!("x{}", i + 1))),
            }
            prevalence_x = Some(x);
        }

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let spectral = self.init_spectral;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let prev_ref = prevalence_x.as_deref();
            let m = sts::fit_sts(
                &corpus.docs, k, num_types, iters, convergence_tol, prev_ref, Some(&sentiment_seed),
                kappa_est, spectral, keep_eta_cov, &mut rng,
            );
            (m, corpus)
        });

        let n = 2 * k - 1;
        // Baseline topic-word (α^(s)=0).
        let beta = sts_beta_at(&model.kappa_t, &model.kappa_s, &model.mv, k, num_types, 0.0);
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (di, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }
        let sent_v = model.doc_sentiment();
        let mut sentiment = Array2::<f64>::zeros((sent_v.len(), k));
        for (di, row) in sent_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                sentiment[[di, t]] = val;
            }
        }
        self.gamma = model.gamma.as_ref().map(|g| {
            let nf = g.len();
            let mut arr = Array2::<f64>::zeros((nf, n));
            for ff in 0..nf {
                for t in 0..n {
                    arr[[ff, t]] = g[ff][t];
                }
            }
            arr
        });

        // Always build eta_mean; only build eta_cov when keep_eta_cov=True
        // (when keep_nu=false the nu array is empty; only build eta_cov when kept).
        let d = model.alpha.len();
        let dim = 2 * k - 1;
        let mut eta_mean_arr = Array2::<f64>::zeros((d, dim));
        for (di, row) in model.alpha.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                eta_mean_arr[[di, i]] = v;
            }
        }
        let stored_eta_cov: Option<Array3<f32>> = if keep_eta_cov {
            let cov_rows = model.eta_cov();
            let mut cov = Array3::<f32>::zeros((d, dim, dim));
            for di in 0..d {
                for i in 0..dim {
                    for j in 0..dim {
                        cov[[di, i, j]] = cov_rows[di][i * dim + j] as f32;
                    }
                }
            }
            Some(cov)
        } else {
            None
        };
        self.eta_mean = Some(eta_mean_arr);
        self.eta_cov = stored_eta_cov;
        // Retain the E-step snapshots only when eta_cov was NOT kept, so the
        // default path carries no extra state; they let _recompute_eta_cov
        // reproduce ν exactly. When kept, the stored eta_cov is used directly.
        if !keep_eta_cov {
            self.sigma_estep = model.sigma_estep.clone();
            self.kappa_t_estep = model.kappa_t_estep.clone();
            self.kappa_s_estep = model.kappa_s_estep.clone();
        }
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.sentiment = Some(sentiment);
        self.feature_names = feat_names;
        self.kappa_t = model.kappa_t;
        self.kappa_s = model.kappa_s;
        self.mv = model.mv;
        self.sigma = model.sigma;
        self.corpus = Some(corpus);
        self.bound = model.bound_history.last().copied().unwrap_or(f64::NAN);
        self.bound_history = model.bound_history;
        self.converged = model.converged;
        self.fitted = true;
        Ok(())
    }

    /// Baseline topic-word matrix β at neutral sentiment, shape ``(num_topics,
    /// num_words)``. Use :meth:`topic_word_at` for other sentiment levels.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Topic-word matrix β at sentiment level `level` (the same value applied to
    /// every topic), shape ``(num_topics, num_words)``. Inspect the wording at
    /// positive vs. negative sentiment by passing percentiles of :attr:`sentiment`.
    #[pyo3(signature = (level))]
    fn topic_word_at<'py>(&self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let v = self.mv.len();
        Ok(sts_beta_at(&self.kappa_t, &self.kappa_s, &self.mv, self.num_topics, v, level).to_pyarray_bound(py))
    }

    /// Document-topic prevalence matrix θ, shape ``(num_docs, num_topics)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document topic sentiment-discourse α^(s), shape ``(num_docs,
    /// num_topics)``. Positive values mean the document discussed that topic with
    /// wording shifted along the κ^(s) (sentiment-discourse) direction.
    #[getter]
    fn sentiment<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.sentiment.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Prevalence regression coefficients Γ^(p), shape ``(num_features,
    /// num_topics-1)`` — covariate effects on topic prevalence. Requires a
    /// prevalence design at fit time.
    #[getter]
    fn prevalence_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let g = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without a prevalence design"))?;
        let km1 = self.num_topics - 1;
        let nf = g.nrows();
        let mut out = Array2::<f64>::zeros((nf, km1));
        for ff in 0..nf {
            for t in 0..km1 {
                out[[ff, t]] = g[[ff, t]];
            }
        }
        Ok(out.to_pyarray_bound(py))
    }

    /// Sentiment-discourse regression coefficients Γ^(s), shape ``(num_features,
    /// num_topics)`` — covariate effects on topic sentiment-discourse. Requires a
    /// prevalence design at fit time.
    #[getter]
    fn sentiment_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let g = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without a prevalence design"))?;
        let k = self.num_topics;
        let km1 = k - 1;
        let nf = g.nrows();
        let mut out = Array2::<f64>::zeros((nf, k));
        for ff in 0..nf {
            for t in 0..k {
                out[[ff, t]] = g[[ff, km1 + t]];
            }
        }
        Ok(out.to_pyarray_bound(py))
    }

    /// Per-document variational posterior means λ of the logistic-normal latent η
    /// = [α^(p)_{1..K-1}, α^(s)_{1..K}], shape ``(num_docs, 2*num_topics-1)``.
    /// Pairs with :attr:`eta_cov` as the joint prevalence/sentiment posterior for
    /// method-of-composition uncertainty.
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, 2*num_topics-1, 2*num_topics-1)``. Stored as float32 in memory
    /// to halve the dominant memory term; cast to float64 with
    /// ``np.asarray(model.eta_cov, dtype=np.float64)`` when full precision is needed.
    /// Raises RuntimeError if the model was fit with ``keep_eta_cov=False``; use
    /// :meth:`_recompute_eta_cov` to regenerate on demand.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        self.eta_cov.as_ref()
            .map(|c| c.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err(
                "model was fit with keep_eta_cov=False; refit with keep_eta_cov=True, \
                 or use _recompute_eta_cov which recomputes it on demand"
            ))
    }

    /// Recompute the per-document variational covariance ν on demand.
    /// Use this when the model was fit with ``keep_eta_cov=False`` to save memory.
    /// Returns the same ``(num_docs, 2*num_topics-1, 2*num_topics-1)`` float32
    /// array as :attr:`eta_cov`.
    fn _recompute_eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.require_fitted()?;
        let corpus = self.corpus.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no corpus retained; cannot recompute eta_cov"))?;
        let sparse: Vec<(Vec<usize>, Vec<f64>)> =
            corpus.docs.iter().map(|d| crate::variational::doc_sparse(d)).collect();

        // Build a minimal StsModel stub with only the fields recompute_nu_sts needs.
        let alpha = self.eta_mean.as_ref().unwrap()
            .rows().into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        // ν is independent of the prior mean μ — use sigma_estep (the sigma that was
        // active during the last E-step). Fall back to sigma if sigma_estep was not
        // persisted (e.g. loaded from disk).
        let sigma_for_recompute = if self.sigma_estep.is_empty() {
            self.sigma.clone()
        } else {
            self.sigma_estep.clone()
        };
        // Use kappa from the last E-step. Fall back to kappa_t/kappa_s if estep
        // snapshots were not persisted (e.g. loaded from disk).
        let kt_recompute = if self.kappa_t_estep.is_empty() {
            self.kappa_t.clone()
        } else {
            self.kappa_t_estep.clone()
        };
        let ks_recompute = if self.kappa_s_estep.is_empty() {
            self.kappa_s.clone()
        } else {
            self.kappa_s_estep.clone()
        };
        let model_stub = sts::StsModel {
            k: self.num_topics,
            num_types: self.mv.len(),
            alpha,
            nu: Vec::new(),
            gamma: None,
            sigma: sigma_for_recompute.clone(),
            sigma_estep: sigma_for_recompute,
            kappa_t: self.kappa_t.clone(),
            kappa_s: self.kappa_s.clone(),
            kappa_t_estep: kt_recompute,
            kappa_s_estep: ks_recompute,
            mv: self.mv.clone(),
            beta: Vec::new(),
            bound_history: Vec::new(),
            converged: false,
            em_iters_run: 0,
        };

        let nu = py.allow_threads(|| sts::recompute_nu_sts(&model_stub, &sparse));

        // Convert to (D, 2K-1, 2K-1) f32 array.
        let d = nu.len();
        let dim = 2 * self.num_topics - 1;
        let mut cov = Array3::<f32>::zeros((d, dim, dim));
        for di in 0..d {
            for i in 0..dim {
                for j in 0..dim {
                    cov[[di, i, j]] = nu[di][i * dim + j] as f32;
                }
            }
        }
        Ok(cov.to_pyarray_bound(py))
    }

    /// Final variational bound (approximate ELBO).
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    /// The variational bound after each EM iteration.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        self.require_fitted()?;
        Ok(self.bound_history.clone())
    }

    /// ``True`` if EM stopped on the `em_tol` criterion, ``False`` if it hit the
    /// `iters` cap.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Uniform convergence trace: ``(iteration, bound)`` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.bound_history.iter().enumerate().map(|(i, &b)| (i + 1, b)).collect())
    }

    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Document labels (row order of :attr:`doc_topic`), default the document
    /// indices as strings.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Infer topic prevalence θ for *new* documents by the Laplace E-step against
    /// the fitted globals (κ, m, Σ) with a zero prior mean (held-out documents
    /// carry no covariates). `data` is a :class:`Corpus` or `list[list[str]]`;
    /// tokens outside the training vocabulary are dropped. Returns a
    /// ``(num_docs, num_topics)`` array of prevalence proportions.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let theta = py.allow_threads(|| {
            sts::sts_infer(&docs, &self.kappa_t, &self.kappa_s, &self.mv, &self.sigma, self.num_topics)
        });
        Ok(vecs_to_arr2(&theta).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with :meth:`STS.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        // eta_cov is stored as f32 in memory; upcast to f64 for the on-disk format
        // so existing saved models remain compatible.
        let eta_cov_f64 = self.eta_cov.as_ref().map(|c| c.mapv(|x| x as f64));
        write_state(path, MODEL_TAG_STS, &StsState {
            num_topics: self.num_topics, seed: self.seed, init_spectral: self.init_spectral,
            fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta),
            sentiment: arr2_opt(&self.sentiment), gamma: arr2_opt(&self.gamma),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&eta_cov_f64),
            feature_names: self.feature_names.clone(),
            kappa_t: self.kappa_t.clone(), kappa_s: self.kappa_s.clone(),
            mv: self.mv.clone(), sigma: self.sigma.clone(),
            corpus: self.corpus.clone(),
            bound: self.bound, bound_history: self.bound_history.clone(),
            converged: self.converged, topic_names: self.topic_names.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: StsState = read_state(path, MODEL_TAG_STS)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        // eta_cov is saved as f64 for format compatibility; downcast to f32 in memory.
        let eta_cov = arr3_back(s.eta_cov).map(|c| c.mapv(|x| x as f32));
        Ok(STS {
            num_topics: s.num_topics, seed: s.seed, init_spectral: s.init_spectral,
            fitted: s.fitted, topic_names,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta),
            sentiment: arr2_back(s.sentiment), gamma: arr2_back(s.gamma),
            eta_mean: arr2_back(s.eta_mean), eta_cov,
            feature_names: s.feature_names,
            kappa_t: s.kappa_t, kappa_s: s.kappa_s, mv: s.mv, sigma: s.sigma,
            sigma_estep: Vec::new(),         // not persisted; falls back to sigma in _recompute_eta_cov
            kappa_t_estep: Vec::new(),       // not persisted; falls back to kappa_t/kappa_s
            kappa_s_estep: Vec::new(),
            corpus: s.corpus, bound: s.bound, bound_history: s.bound_history,
            converged: s.converged,
        })
    }

    /// Top `n` words per topic (or one topic) at neutral sentiment, as
    /// ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)]))
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> = (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "expected {} topic names, got {}",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// HDP: Hierarchical Dirichlet Process (nonparametric LDA — infers K)
// ---------------------------------------------------------------------------

/// Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei 2006):
/// LDA that **infers the number of topics** rather than fixing it. Fit by the
/// direct-assignment Gibbs sampler (the Chinese Restaurant Franchise). The two
/// concentration parameters `alpha` (document level) and `gamma` (corpus level)
/// govern how readily new topics appear; by default both are resampled from the
/// data (a faithful port of blei-lab/hdp), so you typically don't tune them.
#[pyclass(module = "topica")]
pub struct HDP {
    alpha: f64,
    gamma: f64,
    eta: f64,
    seed: u64,
    resample_conc: bool,

    fitted: bool,
    num_topics: usize,
    topic_names: Vec<String>,
    learned_alpha: f64,
    learned_gamma: f64,
    beta: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
    // Discovery/convergence trace: (iteration, num_topics, log-likelihood, alpha, gamma).
    trace: Vec<(usize, usize, f64, f64, f64)>,
    // Thinned θ draws (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Because HDP's K varies during training, these draws
    // are sampled from the final Dirichlet posterior Dirichlet(njk[d]+alpha*beta[k])
    // after the Gibbs chain ends, using the stabilized topic count.
    theta_draws: Option<Array3<f32>>,
}

impl HDP {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl HDP {
    /// Create an unfitted model. `alpha`/`gamma` are the document- and
    /// corpus-level DP concentrations; `eta` is the topic-word Dirichlet (base
    /// measure). `gamma` is the dominant lever on the inferred topic count:
    /// larger values find more topics (`0.1` is conservative, like tomotopy's
    /// default; raise it for finer granularity).
    ///
    /// `resample_conc` controls whether `alpha`/`gamma` are resampled each sweep.
    /// It defaults to ``False`` (fixed concentrations), which gives a stable,
    /// reproducible topic count. Resampling (`resample_conc=True`) lets the model
    /// adapt the concentrations to the data, but the corpus-level update is a
    /// positive-feedback loop, more topics raise gamma, which creates more
    /// topics, that ran the topic count away to the hundreds on real corpora
    /// (issue #68). The resampled concentrations are now capped to keep that
    /// bounded, but fixed concentrations remain the recommended default; set
    /// `gamma` to choose the granularity directly.
    #[new]
    #[pyo3(signature = (*, alpha=0.1, gamma=0.1, beta=0.01, seed=42, resample_conc=false, eta=None))]
    fn new(py: Python<'_>, alpha: f64, gamma: f64, beta: f64, seed: u64, resample_conc: bool, eta: Option<f64>) -> PyResult<Self> {
        let beta = if let Some(old_val) = eta {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "HDP(eta=) is deprecated; use beta= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (beta - 0.01_f64).abs() > f64::EPSILON { beta } else { old_val }
        } else {
            beta
        };
        if !finite_pos(alpha) || !finite_pos(gamma) {
            return Err(PyValueError::new_err("alpha and gamma must be > 0"));
        }
        if !finite_pos(beta) {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        Ok(HDP {
            alpha,
            gamma,
            eta: beta,
            seed,
            resample_conc,
            fitted: false,
            num_topics: 0,
            topic_names: Vec::new(),
            learned_alpha: alpha,
            learned_gamma: gamma,
            beta: None,
            theta: None,
            corpus: None,
            trace: Vec::new(),
            theta_draws: None,
        })
    }

    /// Fit by Gibbs sampling for `iters` sweeps. `data` is a :class:`Corpus` or
    /// `list[list[str]]`. The inferred topic count is available as `num_topics`.
    #[pyo3(signature = (data, *, iters=150, progress_interval=0,
                        keep_theta_draws=true, num_theta_draws=25, report_interval=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        progress_interval: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        report_interval: Option<usize>,
    ) -> PyResult<()> {
        let progress_interval = if let Some(old_val) = report_interval {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "HDP.fit(report_interval=) is deprecated; use progress_interval= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            // progress_interval wins if explicitly set (non-zero); else deprecated value.
            if progress_interval != 0 { progress_interval } else { old_val }
        } else {
            progress_interval
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        let (alpha, gamma, eta, conc) = (self.alpha, self.gamma, self.eta, self.resample_conc);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        // 0 = auto: ~50 evenly spaced trace points across the run.
        let ll_interval = if progress_interval == 0 { (iters / 50).max(1) } else { progress_interval };

        // HDP's K varies during training, so theta_draws are sampled from the final
        // Dirichlet posterior Dirichlet(njk[d]+alpha*beta[k]) after the chain ends.
        let draw_cap = if keep_theta_draws { num_theta_draws } else { 0 };

        let (model, corpus) = py.allow_threads(move || {
            let m = hdp::fit_hdp(
                &corpus.docs, num_types, alpha, gamma, eta, iters, conc, ll_interval, &mut rng,
            );
            (m, corpus)
        });

        let k = model.num_topics();
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let tw = model.topic_word();
        let mut beta = Array2::<f64>::zeros((k, num_types));
        for (t, row) in tw.iter().enumerate() {
            for (v, &val) in row.iter().enumerate() {
                beta[[t, v]] = val;
            }
        }
        let th = model.doc_topic();
        let mut theta = Array2::<f64>::zeros((th.len(), k));
        for (d, row) in th.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        // Draw from Dirichlet(njk[d] + alpha*beta[k]) for each draw request.
        let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
        if draw_cap > 0 {
            let mut draw_rng = Pcg64Mcg::seed_from_u64(self.seed.wrapping_add(1));
            for _ in 0..draw_cap {
                let snap: Vec<Vec<f32>> = model.njk.iter().map(|counts| {
                    let mut gammas: Vec<f64> = (0..k)
                        .map(|t| {
                            let shape = counts[t] as f64 + model.alpha * model.beta[t];
                            hdp::sample_gamma(shape.max(1e-12), &mut draw_rng)
                        })
                        .collect();
                    let s: f64 = gammas.iter().sum();
                    if s > 0.0 {
                        for g in gammas.iter_mut() { *g /= s; }
                    }
                    gammas.iter().map(|&g| g as f32).collect()
                }).collect();
                theta_draw_buf.push(snap);
            }
        }

        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, k, None);
        self.num_topics = k;
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.learned_alpha = model.alpha;
        self.learned_gamma = model.gamma;
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corpus = Some(corpus);
        self.trace = model.trace.clone();
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The inferred number of topics K.
    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_topics)
    }

    /// The topic-discovery trajectory: ``(iteration, num_topics)`` pairs sampled
    /// during fit. Watching K stabilize is the nonparametric model's headline
    /// convergence check (it grows and shrinks before settling). Sampled every
    /// ``report_interval`` sweeps (auto ≈ 50 points); empty if disabled.
    #[getter]
    fn topic_count_history(&self) -> PyResult<Vec<(usize, usize)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, k, _, _, _)| (it, k)).collect())
    }

    /// The convergence trace: ``(iteration, per-token log-likelihood)`` pairs
    /// sampled during fit. Empty if tracing was disabled.
    #[getter]
    fn log_likelihood_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, _, ll, _, _)| (it, ll)).collect())
    }

    /// Uniform convergence trace: ``(iteration, log_likelihood)`` pairs (same as
    /// :attr:`log_likelihood_history`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, _, ll, _, _)| (it, ll)).collect())
    }

    /// HDP does not implement an early-stop criterion; always ``False``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(false)
    }

    /// The learned-concentration trace: ``(iteration, alpha, gamma)`` triples
    /// sampled during fit (only informative when ``resample_conc=True``). Empty
    /// if tracing was disabled.
    #[getter]
    fn concentration_history(&self) -> PyResult<Vec<(usize, f64, f64)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, _, _, a, g)| (it, a, g)).collect())
    }

    /// The fitted document-level concentration α0 (resampled if enabled).
    #[getter]
    fn alpha(&self) -> f64 {
        self.learned_alpha
    }

    /// The fitted corpus-level concentration γ (resampled if enabled).
    #[getter]
    fn gamma(&self) -> f64 {
        self.learned_gamma
    }

    /// Thinned θ draws, shape ``(num_draws, num_docs, num_topics)``, dtype
    /// ``float32``. ``None`` when fit with ``keep_theta_draws=False``. Because
    /// HDP's K changes during training, these draws are sampled from the final
    /// Dirichlet posterior after the Gibbs chain ends.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents over the discovered topics,
    /// by collapsed Gibbs against the fixed topic-word matrix. `data` is a
    /// :class:`Corpus` or `list[list[str]]`; OOV tokens are dropped. The
    /// document-level prior is symmetric with total mass equal to the learned
    /// concentration α. Returns a ``(num_docs, num_topics)`` array.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let k = self.num_topics;
        let alpha = vec![self.learned_alpha / k as f64; k];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.beta.as_ref().unwrap(),
            &alpha, iters, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `HDP.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_HDP, &HdpState {
            alpha: self.alpha, gamma: self.gamma, eta: self.eta, seed: self.seed,
            resample_conc: self.resample_conc, fitted: self.fitted, num_topics: self.num_topics,
            learned_alpha: self.learned_alpha, learned_gamma: self.learned_gamma,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
            trace: self.trace.clone(),
            topic_names: self.topic_names.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: HdpState = read_state(path, MODEL_TAG_HDP)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(HDP {
            alpha: s.alpha, gamma: s.gamma, eta: s.eta, seed: s.seed,
            resample_conc: s.resample_conc, fitted: s.fitted, num_topics: s.num_topics,
            topic_names,
            learned_alpha: s.learned_alpha, learned_gamma: s.learned_gamma,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corpus: s.corpus,
            trace: s.trace,
            theta_draws: None,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!("HDP(num_topics={} [inferred], fitted=true)", self.num_topics)
        } else {
            format!("HDP(alpha={}, gamma={}, fitted=false)", self.alpha, self.gamma)
        }
    }
}

// ---------------------------------------------------------------------------
// DTM: Dynamic Topic Model (topics that evolve over time)
// ---------------------------------------------------------------------------

/// Dynamic Topic Model (Blei & Lafferty 2006): topics whose word distributions
/// **evolve across time slices**. Each topic-word chain follows a Gaussian
/// state-space model; inference is variational with Kalman smoothing, a faithful
/// port of Blei's C `dtm` / gensim's `LdaSeqModel`. After fitting, query a
/// topic's word distribution at any slice with `topic_word(time)` and trace a
/// word's trajectory with `word_evolution(topic, word)`.
#[pyclass(module = "topica")]
pub struct DTM {
    num_topics: usize,
    alpha: f64,
    chain_variance: f64,
    obs_variance: f64,
    seed: u64,

    fitted: bool,
    topic_names: Vec<String>,
    num_times: usize,
    bound: f64,
    // (num_times, num_topics, num_words): p(word | topic, time).
    topic_words: Option<Vec<Vec<Vec<f64>>>>,
    corpus: Option<corpus::Corpus>,
}

impl DTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl DTM {
    /// Create an unfitted model. `chain_variance` controls how much a topic may
    /// drift between adjacent slices (larger = freer to change; gensim's default
    /// is 0.005). `obs_variance` is the observation noise; `alpha` the Dirichlet
    /// concentration on document-topic proportions.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.01, chain_variance=0.005, obs_variance=0.5, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: f64,
        chain_variance: f64,
        obs_variance: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !finite_pos(alpha) || !finite_pos(chain_variance) || !finite_pos(obs_variance) {
            return Err(PyValueError::new_err(
                "alpha, chain_variance, obs_variance must be > 0",
            ));
        }
        Ok(DTM {
            num_topics,
            alpha,
            chain_variance,
            obs_variance,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            num_times: 0,
            bound: 0.0,
            topic_words: None,
            corpus: None,
        })
    }

    /// Fit by variational EM. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `times` gives each document's integer time-slice index (0-based,
    /// contiguous). The number of slices is inferred as ``max(times) + 1``.
    #[pyo3(signature = (data, times, *, iters=20))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        times: Vec<i64>,
        iters: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if times.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "times has length {} but there are {} documents",
                times.len(),
                corpus.num_docs()
            )));
        }
        if times.iter().any(|&t| t < 0) {
            return Err(PyValueError::new_err("time-slice indices must be >= 0"));
        }
        let times_u: Vec<usize> = times.iter().map(|&t| t as usize).collect();
        let num_times = times_u.iter().copied().max().unwrap() + 1;
        // Require every slice to be populated (contiguous 0..num_times).
        let mut seen = vec![false; num_times];
        for &t in &times_u {
            seen[t] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(PyValueError::new_err(
                "time slices must be contiguous 0..max; some slice has no documents",
            ));
        }

        let num_types = corpus.num_types();
        let k = self.num_topics;
        let (alpha, cv, ov) = (self.alpha, self.chain_variance, self.obs_variance);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = dtm::fit_dtm(
                &corpus.docs, &times_u, num_types, k, num_times, alpha, cv, ov, iters, &mut rng,
            );
            (m, corpus)
        });

        // Precompute p(word | topic, time) for every slice.
        let tw: Vec<Vec<Vec<f64>>> =
            (0..num_times).map(|t| model.topic_word_matrix(t)).collect();

        self.num_times = num_times;
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.bound = model.bound;
        self.topic_words = Some(tw);
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix at time slice `time`, shape ``(num_topics, num_words)``;
    /// rows sum to 1.
    fn topic_word<'py>(&self, py: Python<'py>, time: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        if time >= self.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        let tw = &self.topic_words.as_ref().unwrap()[time];
        let mut arr = Array2::<f64>::zeros((self.num_topics, tw[0].len()));
        for (k, row) in tw.iter().enumerate() {
            for (w, &val) in row.iter().enumerate() {
                arr[[k, w]] = val;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Trajectory of a word's probability in a topic across slices, shape
    /// ``(num_times,)``. `word` is a vocabulary string or its integer id.
    fn word_evolution<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        word: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let wid = if let Ok(i) = word.extract::<usize>() {
            i
        } else {
            let s = word.extract::<String>()?;
            vocab.iter().position(|w| w == &s).ok_or_else(|| {
                PyValueError::new_err(format!("word {:?} not in vocabulary", s))
            })?
        };
        if wid >= vocab.len() {
            return Err(PyValueError::new_err("word id out of range"));
        }
        let tw = self.topic_words.as_ref().unwrap();
        let traj: Vec<f64> = (0..self.num_times).map(|t| tw[t][topic][wid]).collect();
        Ok(Array1::from(traj).to_pyarray_bound(py))
    }

    /// Top `n` words for a topic at one time slice as ``(word, probability)``.
    #[pyo3(signature = (topic, time, n=10))]
    fn top_words(
        &self,
        topic: usize,
        time: usize,
        n: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        if time >= self.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let row = &self.topic_words.as_ref().unwrap()[time][topic];
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| f64::total_cmp(&row[b], &row[a]));
        Ok(idx.into_iter().take(n).map(|w| (vocab[w].clone(), row[w])).collect())
    }

    /// Which words inside `topic` drift most between two time slices.
    ///
    /// For each word, the change in its probability within the topic from
    /// `from_time` to `to_time` (defaults: the first and last slices) is
    /// computed. Returns a dict with two keys, ``"rising"`` and ``"falling"``,
    /// each a list of ``(word, delta)`` pairs (largest gain first; largest drop
    /// first). This is how you see *what* makes a topic's vocabulary evolve, not
    /// just that it does.
    #[pyo3(signature = (topic, *, n=10, from_time=0, to_time=None))]
    fn word_drift<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        n: usize,
        from_time: usize,
        to_time: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let to = to_time.unwrap_or(self.num_times - 1);
        if from_time >= self.num_times || to >= self.num_times {
            return Err(PyValueError::new_err("time slice out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tw = self.topic_words.as_ref().unwrap();
        let a = &tw[from_time][topic];
        let b = &tw[to][topic];
        let mut deltas: Vec<(usize, f64)> = (0..a.len()).map(|w| (w, b[w] - a[w])).collect();
        deltas.sort_by(|x, y| f64::total_cmp(&y.1, &x.1)); // descending by delta

        let to_pairs = |items: Vec<(usize, f64)>| -> Vec<(String, f64)> {
            items.into_iter().map(|(w, d)| (vocab[w].clone(), d)).collect()
        };
        let rising = to_pairs(
            deltas.iter().filter(|&&(_, d)| d > 0.0).take(n).copied().collect(),
        );
        let falling = to_pairs(
            deltas.iter().rev().filter(|&&(_, d)| d < 0.0).take(n).copied().collect(),
        );

        let out = PyDict::new_bound(py);
        out.set_item("rising", rising)?;
        out.set_item("falling", falling)?;
        Ok(out)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// The number of time slices (available after fit).
    #[getter]
    fn num_times(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_times)
    }

    /// The final variational bound (ELBO) reached during fitting.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// DTM has no per-iteration ELBO trace yet; always returns ``[]``.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(Vec::new())
    }

    /// DTM does not implement an early-stop criterion; always ``False``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(false)
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `DTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_DTM, &DtmState {
            num_topics: self.num_topics, alpha: self.alpha, chain_variance: self.chain_variance,
            obs_variance: self.obs_variance, seed: self.seed, fitted: self.fitted,
            num_times: self.num_times, bound: self.bound,
            topic_words: self.topic_words.clone(), corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: DtmState = read_state(path, MODEL_TAG_DTM)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(DTM {
            num_topics: s.num_topics, alpha: s.alpha, chain_variance: s.chain_variance,
            obs_variance: s.obs_variance, seed: s.seed, fitted: s.fitted,
            topic_names,
            num_times: s.num_times, bound: s.bound, topic_words: s.topic_words, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "DTM(num_topics={}, num_times={}, fitted=true)",
                self.num_topics, self.num_times
            )
        } else {
            format!("DTM(num_topics={}, fitted=false)", self.num_topics)
        }
    }
}

// ---------------------------------------------------------------------------
// SupervisedLDA: sLDA (topics shaped to predict a per-document response)
// ---------------------------------------------------------------------------

/// Supervised LDA (Blei & McAuliffe 2007): LDA in which each document carries a
/// real-valued response `y_d ~ N(ηᵀ z̄_d, σ²)` regressed on its topic usage.
/// Fitting is supervised by the response, so topics are shaped to be predictive
/// and the coefficients `η` report how each topic moves `y`. Fit by variational
/// EM; `predict` returns ŷ for new documents.
#[pyclass(module = "topica")]
pub struct SupervisedLDA {
    num_topics: usize,
    alpha: f64,
    seed: u64,

    fitted: bool,
    topic_names: Vec<String>,
    sigma2: f64,
    eta: Option<Array1<f64>>,
    beta: Option<Array2<f64>>,  // K × V
    theta: Option<Array2<f64>>, // D × K
    log_beta: Option<Vec<Vec<f64>>>,
    corpus: Option<corpus::Corpus>,
    // Thinned θ draws (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Each draw samples from Dirichlet(gamma_d).
    theta_draws: Option<Array3<f32>>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl SupervisedLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl SupervisedLDA {
    /// Create an unfitted model. `alpha` is the symmetric Dirichlet
    /// concentration on document-topic proportions.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, seed=42))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, alpha: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !finite_pos(alpha) {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        Ok(SupervisedLDA {
            num_topics,
            alpha,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            sigma2: 0.0,
            eta: None,
            beta: None,
            theta: None,
            log_beta: None,
            corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit by variational EM. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `y` is the per-document real-valued response (length = number of docs).
    #[pyo3(signature = (data, y, *, iters=25, var_iters=15,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=1_usize))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        y: Vec<f64>,
        iters: usize,
        var_iters: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if y.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "y has length {} but there are {} documents",
                y.len(),
                corpus.num_docs()
            )));
        }

        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        let (k, alpha) = (self.num_topics, self.alpha);

        let draw_cap = if keep_theta_draws { num_theta_draws } else { 0 };
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, ll_history, converged_flag, corpus) = py.allow_threads(move || {
            let (m, hist, conv) = slda::fit_slda(
                &corpus.docs, &y, num_types, k, alpha, iters, var_iters,
                convergence_tol, check_every, &mut rng,
            );
            (m, hist, conv, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        let tw = model.topic_word();
        for (t, row) in tw.iter().enumerate() {
            for (w, &val) in row.iter().enumerate() {
                beta[[t, w]] = val;
            }
        }
        let th = model.doc_topic();
        let mut theta = Array2::<f64>::zeros((th.len(), k));
        for (di, row) in th.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }

        // Draw from Dirichlet(gamma_d) for each requested draw.
        let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
        if draw_cap > 0 {
            let mut draw_rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(1));
            for _ in 0..draw_cap {
                let snap: Vec<Vec<f32>> = model.gamma.iter().map(|gd| {
                    let mut gammas: Vec<f64> = gd.iter()
                        .map(|&g| hdp::sample_gamma(g.max(1e-12), &mut draw_rng))
                        .collect();
                    let s: f64 = gammas.iter().sum();
                    if s > 0.0 { for x in gammas.iter_mut() { *x /= s; } }
                    gammas.iter().map(|&g| g as f32).collect()
                }).collect();
                theta_draw_buf.push(snap);
            }
        }
        self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, k, None);

        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.sigma2 = model.sigma2;
        self.eta = Some(Array1::from(model.eta.clone()));
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.log_beta = Some(model.log_beta.clone());
        self.corpus = Some(corpus);
        self.log_likelihood_history = ll_history;
        self.converged = converged_flag;
        self.fitted = true;
        Ok(())
    }

    /// Predict the response ŷ for new documents (`list[list[str]]` or a
    /// :class:`Corpus`). Out-of-vocabulary words are ignored. Returns a 1-D array
    /// of length = number of documents.
    #[pyo3(signature = (data, *, var_iters=20))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        var_iters: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let word_id: std::collections::HashMap<&str, u32> =
            vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i as u32)).collect();

        let docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner.docs.iter().map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect()).collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("predict() expects a Corpus or a list of token lists")
            })?
        };

        let log_beta = self.log_beta.as_ref().unwrap();
        let model = slda::SldaModel {
            num_topics: self.num_topics,
            num_types: vocab.len(),
            alpha: self.alpha,
            log_beta: log_beta.clone(),
            eta: self.eta.as_ref().unwrap().to_vec(),
            sigma2: self.sigma2,
            gamma: Vec::new(),
        };

        let preds: Vec<f64> = docs
            .iter()
            .map(|doc| {
                let ids: Vec<u32> = doc.iter().filter_map(|w| word_id.get(w.as_str()).copied()).collect();
                slda::predict_one(&model, &ids, var_iters)
            })
            .collect();
        Ok(Array1::from(preds).to_pyarray_bound(py))
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The symmetric document-topic Dirichlet prior α, shape ``(num_topics,)``.
    /// Marks SupervisedLDA as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_topics]).to_pyarray_bound(py))
    }

    /// Thinned θ draws, shape ``(num_draws, num_docs, num_topics)``, dtype
    /// ``float32``. ``None`` when fit with ``keep_theta_draws=False``. Each draw
    /// samples from the variational posterior Dirichlet(γ_d).
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }

    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }

    /// Regression coefficients η, shape ``(num_topics,)`` — how each topic moves
    /// the response (in the response's units, per unit of topic frequency).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(self.eta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The fitted response variance σ².
    #[getter]
    fn sigma2(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.sigma2)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Per-EM-iteration response log-likelihood trace. Returns one ``(iter, ll)``
    /// pair per ``check_every`` EM iterations (empty when ``check_every=0``).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// ``True`` if the relative-change convergence criterion was satisfied before
    /// all EM iterations completed. Always ``False`` when ``convergence_tol=0``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by collapsed Gibbs against
    /// the fitted topic-word matrix (the response is not used — this is the
    /// unsupervised E-step). `data` is a :class:`Corpus` or `list[list[str]]`;
    /// OOV tokens are dropped. Returns ``(num_docs, num_topics)``. To predict the
    /// response for new documents, take ``transform(data) @ eta``.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.beta.as_ref().unwrap(),
            &alpha, iters, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Save the fitted model to `path`. Reload with `SupervisedLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_SLDA, &SldaState {
            num_topics: self.num_topics, alpha: self.alpha, seed: self.seed, fitted: self.fitted,
            sigma2: self.sigma2, eta: arr1_opt(&self.eta), beta: arr2_opt(&self.beta),
            theta: arr2_opt(&self.theta), log_beta: self.log_beta.clone(),
            corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SldaState = read_state(path, MODEL_TAG_SLDA)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(SupervisedLDA {
            num_topics: s.num_topics, alpha: s.alpha, seed: s.seed, fitted: s.fitted,
            topic_names,
            sigma2: s.sigma2, eta: arr1_back(s.eta), beta: arr2_back(s.beta),
            theta: arr2_back(s.theta), log_beta: s.log_beta, corpus: s.corpus,
            theta_draws: None,
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    fn __repr__(&self) -> String {
        format!("SupervisedLDA(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// PT: Pseudo-document Topic Model (short texts)
// ---------------------------------------------------------------------------

/// Pseudo-document Topic Model (Zuo et al. 2016) for **short texts**. Documents
/// are aggregated into `num_pseudo` pseudo-documents that carry the topic
/// distributions, so the topic structure is estimated from richer aggregated
/// statistics than individual short documents would provide. Collapsed Gibbs.
#[pyclass(module = "topica")]
pub struct PT {
    num_topics: usize,
    num_pseudo: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Each doc inherits its pseudo-doc's topic distribution.
    theta_draws: Option<Array3<f32>>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl PT {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl PT {
    /// Create an unfitted model. `num_pseudo` is the number of pseudo-documents
    /// short texts are aggregated into (more = finer, fewer = more aggregation).
    #[new]
    #[pyo3(signature = (num_topics, *, num_pseudo=100, alpha=0.1, beta=0.01, seed=42))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, #[pyo3(from_py_with = "py_num_pseudo")] num_pseudo: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if num_pseudo < 1 {
            return Err(PyValueError::new_err("num_pseudo must be >= 1"));
        }
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(PT {
            num_topics, num_pseudo, alpha, beta, seed,
            fitted: false, topic_names: Vec::new(), phi: None, theta: None, corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit by collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=1000, keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        let (k, p, a, b) = (self.num_topics, self.num_pseudo, self.alpha, self.beta);

        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let (model, ll_history, converged_flag, corpus) = py.allow_threads(move || {
            let (m, hist, conv) = pt::fit_ptm_with_draws(
                &corpus.docs, num_types, k, p, a, b, iters, draws_opts,
                convergence_tol, check_every, &mut rng,
            );
            (m, hist, conv, corpus)
        });
        self.theta_draws = draws_to_array3(&model.theta_draws, num_docs, k, None);
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.phi = Some(vecs_to_arr2(&model.topic_word()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.log_likelihood_history = ll_history;
        self.converged = converged_flag;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// The symmetric document-topic Dirichlet prior α, shape ``(num_topics,)``.
    /// Marks PT as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_topics]).to_pyarray_bound(py))
    }
    /// Thinned MCMC θ snapshots, shape ``(num_draws, num_docs, num_topics)``,
    /// dtype ``float32``. ``None`` when fit with ``keep_theta_draws=False``.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }
    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Per-iteration log-likelihood trace. Returns one ``(iter, ll)`` pair for
    /// every ``check_every`` sweeps (empty when ``check_every=0``, the default).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }
    /// ``True`` if the relative-change convergence criterion was satisfied before
    /// all iterations completed. Always ``False`` when ``convergence_tol=0``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }
    /// One label per topic, in topic order. Defaults to ``["topic_0", ...]``
    /// after fit; assign a list of the same length to override.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `PT.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_PT, &PtState {
            num_topics: self.num_topics, num_pseudo: self.num_pseudo, alpha: self.alpha,
            beta: self.beta, seed: self.seed, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
            topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: PtState = read_state(path, MODEL_TAG_PT)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(PT {
            num_topics: s.num_topics, num_pseudo: s.num_pseudo, alpha: s.alpha, beta: s.beta,
            seed: s.seed, fitted: s.fitted, topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            corpus: s.corpus,
            theta_draws: None,
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    /// Infer document-topic distributions for new, unseen documents under the
    /// fitted model (sklearn-style ``transform``). Holds the fitted topic-word
    /// distributions fixed and runs collapsed Gibbs to infer θ for each
    /// document. Returns shape ``(num_new_docs, num_topics)`` with rows
    /// summing to 1.
    ///
    /// **Approximation:** the pseudo-document layer is a training-time
    /// aggregation device. Held-out documents infer θ over the K topics
    /// directly under the fitted topic-word matrix, without pseudo-document
    /// assignment.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let phi = self.phi.as_ref().unwrap();
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(py, data, id_to_word, phi, &alpha, iters, burn_in,
                        num_samples, sample_interval, seed.unwrap_or(self.seed))
    }

    fn __repr__(&self) -> String {
        format!("PT(num_topics={}, num_pseudo={}, fitted={})", self.num_topics, self.num_pseudo, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// GSDMM: Gibbs Sampling Dirichlet Multinomial Mixture (short-text clustering)
// ---------------------------------------------------------------------------

/// GSDMM — the "Movie Group Process" (Yin & Wang 2014). A mixture model for
/// **short texts** (tweets, survey answers, headlines) where each document
/// belongs to exactly *one* topic, not a mixture. You set an upper bound `K` on
/// the number of clusters; empty clusters die out during sampling, so the
/// effective `num_topics` is inferred from the data (≤ K). Handles the sparsity
/// of short documents far better than LDA.
#[pyclass(module = "topica")]
pub struct GSDMM {
    k_max: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    num_used: usize,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,        // num_used × V (used clusters only)
    theta: Option<Array2<f64>>,      // num_docs × num_used (soft assignment)
    doc_cluster: Vec<usize>,         // hard assignment per doc, remapped to 0..num_used
    corpus: Option<corpus::Corpus>,
    // Discovery/convergence trace: (iteration, num_clusters, log-likelihood).
    trace: Vec<(usize, usize, f64)>,
}

impl GSDMM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl GSDMM {
    /// Create an unfitted model. `num_topics` is the *maximum* number of clusters
    /// `K`; the number actually used (non-empty after fitting) is reported by the
    /// `num_topics` getter and is usually smaller. `alpha` controls the pull
    /// toward populous clusters; `beta` is the word-Dirichlet smoothing.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, beta=0.1, seed=42))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics (max clusters) must be >= 2"));
        }
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(GSDMM {
            k_max: num_topics, alpha, beta, seed,
            fitted: false, num_used: 0, topic_names: Vec::new(),
            phi: None, theta: None,
            doc_cluster: Vec::new(), corpus: None, trace: Vec::new(),
        })
    }

    /// Fit by the Movie Group Process (collapsed Gibbs) for `iters` sweeps.
    /// `progress_interval` controls the cluster-discovery trace
    /// (`cluster_count_history` / `log_likelihood_history`): 0 = auto (~50
    /// points), a positive value records every that-many sweeps.
    #[pyo3(signature = (data, *, iters=30, progress_interval=0, report_interval=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        progress_interval: usize,
        report_interval: Option<usize>,
    ) -> PyResult<()> {
        let progress_interval = if let Some(old_val) = report_interval {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "GSDMM.fit(report_interval=) is deprecated; use progress_interval= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if progress_interval != 0 { progress_interval } else { old_val }
        } else {
            progress_interval
        };
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (k, a, b) = (self.k_max, self.alpha, self.beta);
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let ll_interval = if progress_interval == 0 { (iters / 50).max(1) } else { progress_interval };
        let (model, corpus) = py.allow_threads(move || {
            let m = gsdmm::fit_gsdmm(&corpus.docs, num_types, k, a, b, iters, ll_interval, &mut rng);
            (m, corpus)
        });

        // Keep only non-empty clusters; remap their ids to a dense 0..num_used.
        let used = model.used_clusters();
        let mut remap = vec![usize::MAX; self.k_max];
        for (new_i, &old) in used.iter().enumerate() {
            remap[old] = new_i;
        }
        let num_used = used.len();

        let phi_rows: Vec<Vec<f64>> = used.iter().map(|&k| model.cluster_word(k)).collect();
        self.phi = Some(vecs_to_arr2(&phi_rows));

        // Soft per-doc distribution restricted to the used clusters, renormalized.
        let dist = model.doc_cluster_dist(&corpus.docs);
        let d = dist.len();
        let mut theta = Array2::<f64>::zeros((d, num_used));
        for (di, row) in dist.iter().enumerate() {
            let mut s = 0.0;
            for &old in &used {
                s += row[old];
            }
            let s = if s > 0.0 { s } else { 1.0 };
            for (&old, ni) in used.iter().zip(0..) {
                theta[[di, ni]] = row[old] / s;
            }
        }
        self.theta = Some(theta);
        self.doc_cluster = model.doc_cluster().iter().map(|&c| remap[c]).collect();
        self.num_used = num_used;
        self.topic_names = (0..num_used).map(|i| format!("topic_{i}")).collect();
        self.corpus = Some(corpus);
        self.trace = model.trace.clone();
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)`` (used clusters only).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The cluster-discovery trajectory: ``(iteration, num_clusters)`` pairs over
    /// the fit. The Movie Group Process starts from `num_topics` clusters and
    /// empties most of them; watching the count collapse to a stable value is
    /// its headline convergence check. Sampled every ``report_interval`` sweeps
    /// (auto ≈ 50 points); empty if disabled.
    #[getter]
    fn cluster_count_history(&self) -> PyResult<Vec<(usize, usize)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, k, _)| (it, k)).collect())
    }

    /// The convergence trace: ``(iteration, per-token log-likelihood)`` pairs
    /// (each document scored under its assigned cluster). Empty if disabled.
    #[getter]
    fn log_likelihood_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, _, ll)| (it, ll)).collect())
    }
    /// Uniform convergence trace: ``(iteration, log_likelihood)`` pairs (same as
    /// :attr:`log_likelihood_history`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.trace.iter().map(|&(it, _, ll)| (it, ll)).collect())
    }
    /// GSDMM does not implement an early-stop criterion; always ``False``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(false)
    }
    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Hard cluster assignment of each document, shape ``(num_docs,)``; values in
    /// ``0..num_topics``. GSDMM gives each document a single cluster.
    #[getter]
    fn doc_cluster<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.require_fitted()?;
        let v: Vec<i64> = self.doc_cluster.iter().map(|&c| c as i64).collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }
    /// The number of *non-empty* clusters discovered (≤ the `K` you set).
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_used
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_used {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_used,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_used, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_used, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `GSDMM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_GSDMM, &GsdmmState {
            k_max: self.k_max, alpha: self.alpha, beta: self.beta, seed: self.seed,
            fitted: self.fitted, num_used: self.num_used,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            doc_cluster: self.doc_cluster.clone(), corpus: self.corpus.clone(),
            trace: self.trace.clone(), topic_names: self.topic_names.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: GsdmmState = read_state(path, MODEL_TAG_GSDMM)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_used).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(GSDMM {
            k_max: s.k_max, alpha: s.alpha, beta: s.beta, seed: s.seed, fitted: s.fitted,
            num_used: s.num_used, topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            doc_cluster: s.doc_cluster, corpus: s.corpus, trace: s.trace,
        })
    }

    fn __repr__(&self) -> String {
        format!("GSDMM(num_topics={}, k_max={}, fitted={})", self.num_used, self.k_max, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// SeededLDA: guided topics via seed-word priors
// ---------------------------------------------------------------------------

/// Parse a ``{topic_name: [words]}`` dict into ordered (names, word-lists),
/// preserving insertion order.
fn parse_seed_dict(d: &Bound<'_, PyDict>) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    let mut names = Vec::new();
    let mut words = Vec::new();
    for (k, v) in d.iter() {
        names.push(k.extract::<String>().map_err(|_| {
            PyValueError::new_err("seed/keyword dict keys must be strings (topic names)")
        })?);
        words.push(v.extract::<Vec<String>>().map_err(|_| {
            PyValueError::new_err("seed/keyword dict values must be lists of strings")
        })?);
    }
    if names.is_empty() {
        return Err(PyValueError::new_err("provide at least one seeded/keyword topic"));
    }
    Ok((names, words))
}

/// Map per-topic seed/keyword *words* to vocabulary ids (dropping out-of-vocab),
/// padding with empty lists up to `num_topics` total topics.
fn seed_word_ids(
    word_strings: &[Vec<String>],
    id_to_word: &[String],
    num_topics: usize,
) -> Vec<Vec<usize>> {
    let index: HashMap<&str, usize> =
        id_to_word.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();
    let mut out: Vec<Vec<usize>> = word_strings
        .iter()
        .map(|ws| ws.iter().filter_map(|w| index.get(w.as_str()).copied()).collect())
        .collect();
    out.resize(num_topics, Vec::new());
    out
}

/// Seeded LDA (guided topic modeling): you supply a few **seed words** per topic
/// and the model is steered so those topics form around them, while the rest of
/// each topic's vocabulary (and any `residual` unseeded topics) is still learned.
/// Useful when theory tells you which themes to expect (Jagarlamudi et al. 2012;
/// the seeding follows koheiw/seededlda — seed words get a `weight × 100`
/// prior pseudocount in their topic).
#[pyclass(module = "topica")]
pub struct SeededLDA {
    seed_names: Vec<String>,
    seed_words: Vec<Vec<String>>,
    residual: usize,
    alpha: f64,
    beta: f64,
    weight: f64,
    seed: u64,
    // WarpLDA cache-efficient sampler (seeded word phase) instead of the default
    // SparseLDA seeded sweep. Recommended for large K.
    warp: bool,
    // CVB0 deterministic collapsed-variational inference (seeded β).
    cvb0: bool,
    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
    corpus: Option<corpus::Corpus>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl SeededLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
    fn num_topics_val(&self) -> usize {
        self.seed_names.len() + self.residual
    }
}

#[pymethods]
impl SeededLDA {
    /// Create an unfitted model. `seed_words` is ``{topic_name: [words]}``;
    /// `residual` adds that many extra unseeded topics. `weight` (default 0.01,
    /// matching the seededlda package) scales the seed prior. `alpha` is the
    /// per-topic Dirichlet, `beta` the base topic-word smoothing.
    #[new]
    #[pyo3(signature = (seed_words, *, residual=0, alpha=0.1, beta=0.01, weight=0.01, seed=42,
                        sampler="sparse"))]
    fn new(
        seed_words: &Bound<'_, PyDict>,
        residual: usize,
        alpha: f64,
        beta: f64,
        weight: f64,
        seed: u64,
        sampler: &str,
    ) -> PyResult<Self> {
        let (names, words) = parse_seed_dict(seed_words)?;
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        if names.len() + residual < 2 {
            return Err(PyValueError::new_err("need at least 2 topics (seeded + residual)"));
        }
        let (warp, cvb0) = match sampler {
            "sparse" => (false, false),
            "warp" | "warplda" => (true, false),
            "cvb0" | "cvb" => (false, true),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\", \"warp\", or \"cvb0\""
                )))
            }
        };
        Ok(SeededLDA {
            seed_names: names, seed_words: words, residual, alpha, beta, weight, seed, warp, cvb0,
            fitted: false, topic_names: Vec::new(), phi: None, theta: None,
            theta_draws: None, corpus: None,
            log_likelihood_history: Vec::new(), converged: false,
        })
    }

    /// Fit by collapsed Gibbs for `iters` sweeps. Seeded topics come first (in
    /// the order given), then the residual topics.
    ///
    /// `doc_topic_prior` (optional, `(num_docs, num_topics)`) supplies a
    /// per-document asymmetric Dirichlet prior `α_{d,k}` that replaces the
    /// symmetric `alpha`, biasing each document's topic mixture toward chosen
    /// topics (e.g. from a document embedding). It is a prior, so the sampler
    /// can still move a document away from it.
    ///
    /// `convergence_tol` (default 0.0, disabled) enables early stopping: after
    /// each `check_every` sweeps the relative change in the log-likelihood is
    /// compared; if it falls below `convergence_tol` the loop stops and
    /// :attr:`converged` is set to ``True``. When 0 (default), the full `iters`
    /// run exactly as before.
    #[pyo3(signature = (data, *, iters=2000, doc_topic_prior=None,
                        keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        doc_topic_prior: Option<&Bound<'_, PyAny>>,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_topics = self.num_topics_val();
        let num_types = corpus.num_types();
        let seeds = seed_word_ids(&self.seed_words, &corpus.id_to_word, num_topics);
        let (alpha, beta, seed_weight) = (self.alpha, self.beta, self.weight * 100.0);

        let doc_alpha: Option<Vec<Vec<f64>>> = match doc_topic_prior {
            Some(p) => {
                let rows = parse_features(p)?;
                if rows.len() != corpus.num_docs() {
                    return Err(PyValueError::new_err(format!(
                        "doc_topic_prior has {} rows but corpus has {} documents",
                        rows.len(),
                        corpus.num_docs()
                    )));
                }
                if rows.iter().any(|r| r.len() != num_topics) {
                    return Err(PyValueError::new_err(
                        "each doc_topic_prior row must have num_topics entries",
                    ));
                }
                if rows.iter().any(|r| r.iter().any(|&a| a <= 0.0)) {
                    return Err(PyValueError::new_err("doc_topic_prior entries must be > 0"));
                }
                Some(rows)
            }
            None => None,
        };

        let check_every = if check_every == 0 { 0 } else if convergence_tol > 0.0 { check_every.max(1) } else { check_every };
        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, corpus.num_docs(), num_topics)?;
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);

        if self.cvb0 {
            // CVB0 seeded path: deterministic, asymmetric β via set_seeds. The
            // per-document prior is not threaded through CVB0's θ output yet.
            if doc_alpha.is_some() {
                return Err(PyValueError::new_err(
                    "sampler=\"cvb0\" does not support doc_topic_prior yet; use sampler=\"sparse\"",
                ));
            }
            let (phi_tw, theta_dk, corpus) = py.allow_threads(move || {
                let alpha0 = vec![alpha; num_topics];
                let mut cv = cvb0::Cvb0::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                cv.set_seeds(&seeds, seed_weight);
                for _ in 0..iters {
                    cv.sweep();
                }
                (cv.topic_word(), cv.doc_topic(), corpus)
            });
            self.phi = Some(vecs_to_arr2(&phi_tw));
            self.theta = Some(vecs_to_arr2(&theta_dk));
            self.theta_draws = None;
            let mut names = self.seed_names.clone();
            for i in 0..self.residual {
                names.push(format!("residual_{}", i + 1));
            }
            self.topic_names = names;
            self.corpus = Some(corpus);
            self.log_likelihood_history = Vec::new();
            self.converged = false;
            self.fitted = true;
            return Ok(());
        }

        if self.warp {
            // WarpLDA seeded path. The per-document prior (doc_topic_prior) is not
            // yet wired through the warp θ output, so require the symmetric case.
            if doc_alpha.is_some() {
                return Err(PyValueError::new_err(
                    "sampler=\"warp\" does not support doc_topic_prior yet; use sampler=\"sparse\"",
                ));
            }
            let num_docs = corpus.num_docs();
            let (phi_tw, theta_dk, theta_draw_buf, corpus) = py.allow_threads(move || {
                let alpha0 = vec![alpha; num_topics];
                let mut ws = warplda::WarpLda::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                ws.set_seeds(&seeds, seed_weight);
                let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
                for iter in 1..=iters {
                    ws.sweep(&corpus, &mut rng);
                    if draws_opts.thin > 0 && iter % draws_opts.thin == 0 {
                        let mut tmp = vec![vec![0.0f64; num_topics]; num_docs];
                        ws.theta_into(&corpus, &mut tmp);
                        let snap = tmp.iter()
                            .map(|r| r.iter().map(|&v| v as f32).collect())
                            .collect();
                        push_capped(&mut theta_draw_buf, snap, draws_opts.cap);
                    }
                }
                let phi_tw = ws.topic_word();
                let mut theta_dk = vec![vec![0.0f64; num_topics]; num_docs];
                ws.theta_into(&corpus, &mut theta_dk);
                (phi_tw, theta_dk, theta_draw_buf, corpus)
            });
            self.phi = Some(vecs_to_arr2(&phi_tw));
            self.theta = Some(vecs_to_arr2(&theta_dk));
            self.theta_draws = draws_to_array3(&theta_draw_buf, num_docs, num_topics, None);
            let mut names = self.seed_names.clone();
            for i in 0..self.residual {
                names.push(format!("residual_{}", i + 1));
            }
            self.topic_names = names;
            self.corpus = Some(corpus);
            self.log_likelihood_history = Vec::new();
            self.converged = false;
            self.fitted = true;
            return Ok(());
        }

        let (model, ll_history, converged, corpus) = py.allow_threads(move || {
            let (m, ll, conv) = seeded::fit_seeded_lda(
                &corpus.docs, num_types, num_topics, &seeds, alpha, beta, seed_weight, doc_alpha,
                iters, draws_opts, convergence_tol, check_every, &mut rng,
            );
            (m, ll, conv, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word_all()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.theta_draws = draws_to_array3(&model.theta_draws, corpus.num_docs(), num_topics, None);
        let mut names = self.seed_names.clone();
        for i in 0..self.residual {
            names.push(format!("residual_{}", i + 1));
        }
        self.topic_names = names;
        self.corpus = Some(corpus);
        self.log_likelihood_history = ll_history;
        self.converged = converged;
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Thinned MCMC θ draws, shape ``(num_draws, num_docs, num_topics)``, or
    /// ``None`` when fit with ``keep_theta_draws=False``. Real cross-sweep
    /// posterior samples that :func:`topica.composition_theta` prefers over the
    /// within-document Dirichlet approximation.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }
    /// Per-document token counts (length D), in ``doc_topic`` row order, so
    /// ``composition_theta`` can recover N_d without re-threading the Corpus.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self
            .corpus
            .as_ref()
            .map(|c| c.docs.iter().map(|d| d.len()).collect())
            .unwrap_or_default())
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics_val()
    }
    /// Per-iteration log-likelihood trace. Each entry is ``(iteration, log_likelihood)``
    /// recorded every ``check_every`` sweeps during :meth:`fit`. Non-empty after fitting.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }
    /// ``True`` if the convergence criterion was met (``convergence_tol > 0``);
    /// ``False`` if the full ``iters`` ran.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }
    /// The symmetric document-topic Dirichlet prior α, broadcast to
    /// ``(num_topics,)``. Marks SeededLDA as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_topics_val()]).to_pyarray_bound(py))
    }
    /// The topic labels: the seed names you gave, then ``residual_1`` … for any
    /// unseeded topics. Settable after fit; length must equal ``num_topics``.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.num_topics_val();
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                k,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics_val(), n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics_val(), n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `SeededLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_SEEDED, &SeededState {
            num_topics: self.num_topics_val(), alpha: self.alpha, beta: self.beta,
            weight: self.weight, seed: self.seed, fitted: self.fitted,
            topic_names: self.topic_names.clone(), phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            seed_names: self.seed_names.clone(),
            seed_words: self.seed_words.clone(),
            residual: self.residual,
            warp: self.warp, cvb0: self.cvb0,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SeededState = read_state(path, MODEL_TAG_SEEDED)?;
        // num_topics is the total topic count (seeded + residual); use it directly.
        Ok(SeededLDA {
            seed_names: s.seed_names,
            seed_words: s.seed_words,
            residual: s.residual,
            alpha: s.alpha, beta: s.beta, weight: s.weight, seed: s.seed,
            warp: s.warp, cvb0: s.cvb0, fitted: s.fitted,
            topic_names: s.topic_names, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            theta_draws: arr3f32_back(s.theta_draws),
            corpus: s.corpus,
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    /// Infer document-topic distributions for new, unseen documents under the
    /// fitted model (sklearn-style ``transform``). Holds the fitted topic-word
    /// distributions fixed and runs collapsed Gibbs to infer θ for each
    /// document. Returns shape ``(num_new_docs, num_topics)`` with rows
    /// summing to 1.
    ///
    /// **Approximation:** the seed-word boost is baked into the fitted
    /// topic-word matrix. New documents infer θ under those distributions
    /// without re-estimating the seed prior.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let phi = self.phi.as_ref().unwrap();
        let k = self.num_topics_val();
        let alpha = vec![self.alpha; k];
        transform_gibbs(py, data, id_to_word, phi, &alpha, iters, burn_in,
                        num_samples, sample_interval, seed.unwrap_or(self.seed))
    }

    fn __repr__(&self) -> String {
        format!("SeededLDA(seeded={}, residual={}, fitted={})", self.seed_names.len(), self.residual, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// Top2Vec: embedding-clustering topic model (Angelov 2020)
// ---------------------------------------------------------------------------

/// Parse the `reducer` choice for the embedding models into a `use_umap` flag.
fn parse_reducer(reducer: &str) -> PyResult<bool> {
    match reducer {
        "pca" => Ok(false),
        "umap" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "unknown reducer {other:?}; expected 'pca' or 'umap'"
        ))),
    }
}

/// Validate the `clusterer` choice for the embedding models. `"hdbscan"` (the
/// default) discovers the topic count and leaves a `-1` noise bucket; `"kmeans"`
/// and `"agglomerative"` assign every document to `num_clusters` clusters, so they
/// require `num_clusters >= 1`.
fn parse_clusterer(clusterer: &str, num_clusters: Option<i64>) -> PyResult<(String, Option<usize>)> {
    match clusterer {
        "hdbscan" => Ok(("hdbscan".to_string(), None)),
        "kmeans" | "agglomerative" => {
            let k = num_clusters.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "clusterer={clusterer:?} needs num_clusters (the number of clusters to form)"
                ))
            })?;
            if k < 1 {
                return Err(PyValueError::new_err(format!(
                    "num_clusters must be >= 1, got {k}"
                )));
            }
            Ok((clusterer.to_string(), Some(k as usize)))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown clusterer {other:?}; expected 'hdbscan', 'kmeans', or 'agglomerative'"
        ))),
    }
}

/// For `reducer='umap'`, warn that the topic-discovery fit is not reproducible
/// (the transform / prediction phase is deterministic regardless), so the default
/// `reducer='pca'` stays the reproducible choice. Also guards the case (not present
/// in a standard wheel) where the `umap` feature was not compiled in.
fn umap_notice(py: Python<'_>, use_umap: bool) -> PyResult<()> {
    if !use_umap {
        return Ok(());
    }
    if !crate::reduce::umap_available() {
        return Err(PyRuntimeError::new_err(
            "reducer='umap' is not available in this build; rebuild with the `umap` \
             feature, or use reducer='pca' (the default)",
        ));
    }
    let warnings = py.import_bound("warnings")?;
    warnings.call_method1(
        "warn",
        ("reducer='umap': the UMAP topic-discovery fit is not reproducible across runs \
          (the Rust UMAP optimizer's negative sampling is unseeded). The transform / \
          prediction phase is deterministic regardless. Use reducer='pca' (the default) \
          for a reproducible fit.",),
    )?;
    Ok(())
}

/// Top2Vec: topics by clustering document embeddings. We reduce the document
/// embeddings (randomized PCA), density-cluster them (HDBSCAN), and read each
/// topic off its cluster: the topic vector is the mean of its documents'
/// embeddings, and its words are the vocabulary terms nearest that vector.
///
/// You bring the embeddings. `fit(data, doc_embeddings)` needs one embedding row
/// per document; pass `word_embeddings` with the aligned `vocabulary` (same
/// space) to also get `topic_neighbors`. The topic count is discovered, not set.
///
/// `Top2Vec` and `BERTopic` share the class-based TF-IDF `topic_word` matrix, so
/// their `topic_word` / `topic_table` are the same given the same clusters. What
/// makes Top2Vec distinct is the **centroid** representation — the vocabulary
/// nearest the cluster centroid in embedding space — which `top_words` returns by
/// default when `word_embeddings` are present (pass `representation="c-tf-idf"`
/// for the shared view, or read it from `topic_neighbors`).
///
/// No embedder of your own? `topica.llm_embed(texts, model=...)` builds the
/// matrix (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct Top2Vec {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    has_word_vectors: bool,
    topic_names: Vec<String>,
    model: Option<top2vec::Top2VecModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Top2VecState {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    has_word_vectors: bool,
    #[serde(default)] topic_names: Vec<String>,
    model: Option<top2vec::Top2VecModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

impl Top2Vec {
    fn fitted_model(&self) -> PyResult<&top2vec::Top2VecModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl Top2Vec {
    /// Create an unfitted model. `n_components` is the reduced dimensionality
    /// before clustering. `clusterer` is `"hdbscan"` (default; discovers the topic
    /// count, leaves a `-1` noise bucket — `min_cluster_size`/`min_samples` are
    /// its knobs), or `"kmeans"` / `"agglomerative"`, which assign every document
    /// to `num_clusters` clusters (no noise). `min_samples` defaults to
    /// `min_cluster_size`.
    #[new]
    #[pyo3(signature = (*, n_components=5, min_cluster_size=15, min_samples=None,
                        reducer="pca", n_neighbors=15, clusterer="hdbscan",
                        num_clusters=None, seed=42))]
    fn new(
        n_components: usize,
        min_cluster_size: usize,
        min_samples: Option<usize>,
        reducer: &str,
        n_neighbors: usize,
        clusterer: &str,
        num_clusters: Option<i64>,
        seed: u64,
    ) -> PyResult<Self> {
        if min_cluster_size < 2 {
            return Err(PyValueError::new_err("min_cluster_size must be >= 2"));
        }
        let use_umap = parse_reducer(reducer)?;
        let (clusterer, num_clusters) = parse_clusterer(clusterer, num_clusters)?;
        Ok(Top2Vec {
            n_components,
            use_umap,
            n_neighbors,
            min_cluster_size,
            min_samples: min_samples.unwrap_or(min_cluster_size),
            clusterer,
            num_clusters,
            seed,
            fitted: false,
            has_word_vectors: false,
            topic_names: Vec::new(),
            model: None,
            id_to_word: Vec::new(),
            docs: Vec::new(),
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`), one row per document. Pass `word_embeddings`
    /// (`(len(vocabulary), E)`) and `vocabulary` together to enable
    /// `topic_neighbors`; the word embeddings are realigned to topica's vocabulary.
    #[pyo3(signature = (data, doc_embeddings, *, word_embeddings=None, vocabulary=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        word_embeddings: Option<&Bound<'_, PyAny>>,
        vocabulary: Option<Vec<String>>,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let num_types = corpus.num_types();

        // Realign user word embeddings to topica's vocabulary order; words topica
        // kept but the user did not supply get a zero vector (no neighbors there).
        let word_vecs: Vec<Vec<f64>> = match word_embeddings {
            Some(we) => {
                let vocab = vocabulary.ok_or_else(|| {
                    PyValueError::new_err("word_embeddings requires `vocabulary` to align them")
                })?;
                let rows = parse_features(we)?;
                if rows.len() != vocab.len() {
                    return Err(PyValueError::new_err(format!(
                        "word_embeddings has {} rows but vocabulary has {} words",
                        rows.len(),
                        vocab.len()
                    )));
                }
                check_all_finite_2d("word_embeddings", &rows)?;
                let e = rows.first().map(|r| r.len()).unwrap_or(0);
                let map: std::collections::HashMap<&str, usize> =
                    vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();
                corpus
                    .id_to_word
                    .iter()
                    .map(|w| match map.get(w.as_str()) {
                        Some(&i) => rows[i].clone(),
                        None => vec![0.0; e],
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        self.has_word_vectors = !word_vecs.is_empty();
        self.id_to_word = corpus.id_to_word.clone();
        self.docs = corpus.docs.clone();

        umap_notice(py, self.use_umap)?;
        let (nc, uu, nn, mcs, ms, seed) = (
            self.n_components, self.use_umap, self.n_neighbors,
            self.min_cluster_size, self.min_samples, self.seed,
        );
        let clusterer = self.clusterer.clone();
        let num_clusters = self.num_clusters;
        let model = py.allow_threads(move || {
            top2vec::fit_top2vec(
                &corpus.docs, &doc_emb, &word_vecs, num_types, nc, uu, nn, mcs, ms,
                &clusterer, num_clusters, seed,
            )
        });
        if model.num_topics == 0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "Top2Vec: clustering found no clusters (num_topics=0). Lower \
                 min_cluster_size, add data, or check the scale of your embeddings.",
            ))?;
        }
        let k = model.num_topics;
        self.model = Some(model);
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    /// Number of topics discovered (HDBSCAN clusters found).
    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_topics)
    }
    /// Topic-word distribution from class-based TF-IDF, row-normalized
    /// (`(num_topics, vocab)`).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Soft document-topic membership (`(num_docs, num_topics)`), rows sum to one.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Each topic's vector in the embedding space (`(num_topics, E)`).
    #[getter]
    fn topic_vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_vectors).to_pyarray_bound(py))
    }
    /// Hard cluster assignment per document; `-1` is a noise document with no topic.
    #[getter]
    fn labels(&self) -> PyResult<Vec<i64>> {
        Ok(self.fitted_model()?.labels.clone())
    }
    /// Topic labels (``topic_0`` … by default; settable after fit).
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.fitted_model()?.num_topics;
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                k,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// Top `n` words of `topic` (or every topic when `topic` is None) by
    /// class-TF-IDF weight, as `(word, weight)` pairs.
    /// Top words per topic. Top2Vec's distinctive view is the **centroid**
    /// representation: the vocabulary words nearest the cluster centroid in
    /// embedding space. When fit with `word_embeddings`, `top_words` returns that
    /// by default (so `summary`/`top_words` show Top2Vec's identity, not just the
    /// class-based TF-IDF it shares with `BERTopic`). Pass
    /// `representation="c-tf-idf"` for the c-TF-IDF words, or `"centroid"`
    /// explicitly. `topic_word` and `topic_table` always stay c-TF-IDF.
    #[pyo3(signature = (n=10, *, topic=None, representation=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        representation: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let rep = match representation {
            Some(r) => r,
            None if self.has_word_vectors => "centroid",
            None => "c-tf-idf",
        };
        match rep {
            "centroid" => {
                if !self.has_word_vectors {
                    return Err(PyValueError::new_err(
                        "representation='centroid' requires fitting with word_embeddings (and vocabulary)",
                    ));
                }
                let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
                    if t >= m.num_topics {
                        return Err(PyValueError::new_err("topic out of range"));
                    }
                    let items: Vec<Bound<'py, PyTuple>> = m
                        .topic_neighbors(n, t)
                        .into_iter()
                        .map(|(w, s)| {
                            PyTuple::new_bound(
                                py,
                                &[self.id_to_word[w].clone().into_py(py), s.into_py(py)],
                            )
                        })
                        .collect();
                    Ok(PyList::new_bound(py, items))
                };
                match topic {
                    Some(t) => Ok(one(t)?.into_any()),
                    None => {
                        let all: Vec<Bound<'py, PyList>> =
                            (0..m.num_topics).map(one).collect::<PyResult<_>>()?;
                        Ok(PyList::new_bound(py, all).into_any())
                    }
                }
            }
            "c-tf-idf" | "ctfidf" | "c_tf_idf" => {
                let phi = vecs_to_arr2(&m.topic_word);
                topic_words_helper(py, &phi, &self.id_to_word, m.num_topics, n, topic)
            }
            other => Err(PyValueError::new_err(format!(
                "representation must be 'centroid' or 'c-tf-idf', got {other:?}"
            ))),
        }
    }

    /// The `n` vocabulary words whose embeddings are nearest `topic`'s vector by
    /// cosine, as `(word, cosine)` pairs. Requires fitting with `word_embeddings`.
    /// `topic` is the first argument, so `topic_neighbors(0, n=8)` reads naturally.
    #[pyo3(signature = (topic, *, n=10))]
    fn topic_neighbors(&self, topic: usize, n: usize) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to use topic_neighbors",
            ));
        }
        if topic >= m.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        Ok(m.topic_neighbors(n, topic)
            .into_iter()
            .map(|(w, s)| (self.id_to_word[w].clone(), s))
            .collect())
    }

    /// Soft topic membership for new documents from their embeddings (cosine to
    /// each topic vector, normalized). `data` is accepted for API symmetry but
    /// Top2Vec assigns by embedding only. Returns `(num_docs, num_topics)`.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        let corpus = corpus::Corpus {
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
            doc_names: (0..self.docs.len()).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); self.docs.len()],
            doc_freqs: {
                let v = self.id_to_word.len();
                let mut df = vec![0u32; v];
                for doc in &self.docs {
                    let mut seen = std::collections::HashSet::new();
                    for &w in doc { seen.insert(w as usize); }
                    for w in seen { if w < v { df[w] += 1; } }
                }
                df
            },
            total_freqs: {
                let v = self.id_to_word.len();
                let mut tf = vec![0u32; v];
                for doc in &self.docs { for &w in doc { if (w as usize) < v { tf[w as usize] += 1; } } }
                tf
            },
        };
        Ok(Array1::from(umass_coherence(&corpus, &tops)).to_pyarray_bound(py))
    }

    #[pyo3(signature = (data, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let de_obj = doc_embeddings.ok_or_else(|| {
            PyValueError::new_err("Top2Vec.transform requires doc_embeddings")
        })?;
        let _ = data;
        let de = parse_features(de_obj)?;
        Ok(vecs_to_arr2(&m.assign(&de)).to_pyarray_bound(py))
    }

    /// Fit, then return the document-topic proportions (`fit_transform`).
    #[pyo3(signature = (data, doc_embeddings, *, word_embeddings=None, vocabulary=None))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: &Bound<'py, PyAny>,
        word_embeddings: Option<&Bound<'py, PyAny>>,
        vocabulary: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(py, data, doc_embeddings, word_embeddings, vocabulary)?;
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Merge groups of topics into single topics, e.g. ``[[3, 7], [1, 2]]``. The
    /// topic vectors, document-topic, and topic-word are rebuilt and topic ids
    /// renumbered to a dense range.
    fn merge_topics(&mut self, groups: Vec<Vec<usize>>) -> PyResult<()> {
        let vocab = self.id_to_word.len();
        let m = self.model.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("model is not fitted yet; call fit() first")
        })?;
        m.merge_topics(&self.docs, &groups, vocab);
        Ok(())
    }

    /// Reassign noise documents (label ``-1``) to their nearest topic and rebuild
    /// the topic-word matrix. Returns how many documents were reassigned.
    fn reduce_outliers(&mut self) -> PyResult<usize> {
        let vocab = self.id_to_word.len();
        let m = self.model.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("model is not fitted yet; call fit() first")
        })?;
        let before = m.labels.iter().filter(|&&l| l < 0).count();
        m.reduce_outliers(&self.docs, vocab);
        Ok(before - m.labels.iter().filter(|&&l| l < 0).count())
    }

    /// Save the fitted model to `path` (topica's binary format), so a discovered
    /// fit can be reloaded and reused without refitting (useful with the stochastic
    /// `reducer="umap"` discovery).
    fn save(&self, path: &str) -> PyResult<()> {
        self.fitted_model()?;
        write_state(path, MODEL_TAG_TOP2VEC, &Top2VecState {
            n_components: self.n_components,
            use_umap: self.use_umap,
            n_neighbors: self.n_neighbors,
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            clusterer: self.clusterer.clone(),
            num_clusters: self.num_clusters,
            seed: self.seed,
            fitted: self.fitted,
            has_word_vectors: self.has_word_vectors,
            topic_names: self.topic_names.clone(),
            model: self.model.clone(),
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: Top2VecState = read_state(path, MODEL_TAG_TOP2VEC)?;
        let num_topics = s.model.as_ref().map_or(0, |m| m.num_topics);
        let topic_names = if s.topic_names.is_empty() {
            (0..num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(Top2Vec {
            n_components: s.n_components,
            use_umap: s.use_umap,
            n_neighbors: s.n_neighbors,
            min_cluster_size: s.min_cluster_size,
            min_samples: s.min_samples,
            clusterer: s.clusterer,
            num_clusters: s.num_clusters,
            seed: s.seed,
            fitted: s.fitted,
            has_word_vectors: s.has_word_vectors,
            topic_names,
            model: s.model,
            id_to_word: s.id_to_word,
            docs: s.docs,
        })
    }

    /// Top2Vec has no iterative objective; fit_history is always ``[]``.
    #[getter]
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// Top2Vec is not an iterative sampler (UMAP + clustering); converged is always ``None``.
    #[getter]
    fn converged(&self) -> Option<bool> {
        None
    }

    fn __repr__(&self) -> String {
        let k = self.model.as_ref().map_or(0, |m| m.num_topics);
        format!("Top2Vec(fitted={}, num_topics={})", self.fitted, k)
    }
}

// ---------------------------------------------------------------------------
// BERTopic: embedding-clustering topic model with c-TF-IDF (Grootendorst 2022)
// ---------------------------------------------------------------------------

/// BERTopic: the same reduce/cluster pipeline as `Top2Vec`, but topics are
/// defined by class-based TF-IDF over their documents' words, so no word
/// embeddings are needed. `nr_topics` merges the most similar topics down to a
/// target count; `doc_topic` is the approximate distribution (a sliding window's
/// c-TF-IDF compared to each topic). You bring the document embeddings.
///
/// No embedder of your own? `topica.llm_embed(texts, model=...)` builds the
/// matrix (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct BERTopic {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    nr_topics: Option<usize>,
    window: usize,
    stride: usize,
    bm25: bool,
    reduce_frequent: bool,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<bertopic::BertopicModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BertopicState {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    nr_topics: Option<usize>,
    window: usize,
    stride: usize,
    bm25: bool,
    reduce_frequent: bool,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    #[serde(default)] topic_names: Vec<String>,
    model: Option<bertopic::BertopicModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

impl BERTopic {
    fn fitted_model(&self) -> PyResult<&bertopic::BertopicModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
    /// Map token-list documents to id documents over the fitted vocabulary,
    /// dropping out-of-vocabulary words (used for `approximate_distribution`).
    fn to_ids(&self, docs: &[Vec<String>]) -> Vec<Vec<u32>> {
        let map: std::collections::HashMap<&str, u32> =
            self.id_to_word.iter().enumerate().map(|(i, w)| (w.as_str(), i as u32)).collect();
        docs.iter()
            .map(|d| d.iter().filter_map(|w| map.get(w.as_str()).copied()).collect())
            .collect()
    }
}

#[pymethods]
impl BERTopic {
    /// Create an unfitted model. `nr_topics` (optional) reduces the discovered
    /// topics to that many by merging the most c-TF-IDF-similar; `window`/`stride`
    /// parameterize the soft `doc_topic` distribution.
    #[new]
    #[pyo3(signature = (*, n_components=5, min_cluster_size=15, min_samples=None,
                        nr_topics=None, window=4, stride=1, reducer="pca", n_neighbors=15,
                        bm25=false, reduce_frequent=false, clusterer="hdbscan",
                        num_clusters=None, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        min_cluster_size: usize,
        min_samples: Option<usize>,
        nr_topics: Option<usize>,
        window: usize,
        stride: usize,
        reducer: &str,
        n_neighbors: usize,
        bm25: bool,
        reduce_frequent: bool,
        clusterer: &str,
        num_clusters: Option<i64>,
        seed: u64,
    ) -> PyResult<Self> {
        if min_cluster_size < 2 {
            return Err(PyValueError::new_err("min_cluster_size must be >= 2"));
        }
        let use_umap = parse_reducer(reducer)?;
        let (clusterer, num_clusters) = parse_clusterer(clusterer, num_clusters)?;
        Ok(BERTopic {
            n_components,
            use_umap,
            n_neighbors,
            min_cluster_size,
            min_samples: min_samples.unwrap_or(min_cluster_size),
            nr_topics,
            window: window.max(1),
            stride: stride.max(1),
            bm25,
            reduce_frequent,
            clusterer,
            num_clusters,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            id_to_word: Vec::new(),
            docs: Vec::new(),
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`), one row per document. No word embeddings are needed.
    #[pyo3(signature = (data, doc_embeddings))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let num_types = corpus.num_types();
        self.id_to_word = corpus.id_to_word.clone();
        self.docs = corpus.docs.clone();
        umap_notice(py, self.use_umap)?;
        let (nc, uu, nn, mcs, ms, nr, win, st, b25, rf, seed) = (
            self.n_components, self.use_umap, self.n_neighbors, self.min_cluster_size,
            self.min_samples, self.nr_topics, self.window, self.stride,
            self.bm25, self.reduce_frequent, self.seed,
        );
        let clusterer = self.clusterer.clone();
        let num_clusters = self.num_clusters;
        let model = py.allow_threads(move || {
            bertopic::fit_bertopic(
                &corpus.docs, &doc_emb, num_types, nc, uu, nn, mcs, ms, nr, win, st, b25, rf,
                &clusterer, num_clusters, seed,
            )
        });
        if model.num_topics == 0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "BERTopic: clustering found no clusters (num_topics=0). Lower \
                 min_cluster_size, add data, or check the scale of your embeddings.",
            ))?;
        }
        let k = model.num_topics;
        self.model = Some(model);
        self.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_topics)
    }
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    #[getter]
    fn labels(&self) -> PyResult<Vec<i64>> {
        Ok(self.fitted_model()?.labels.clone())
    }
    /// Topic labels (``topic_0`` … by default; settable after fit).
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.fitted_model()?.num_topics;
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                k,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// Top `n` words of `topic` (or every topic when None) by c-TF-IDF weight.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let phi = vecs_to_arr2(&m.topic_word);
        topic_words_helper(py, &phi, &self.id_to_word, m.num_topics, n, topic)
    }

    /// UMass coherence for each topic's top-`n` words, over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        let corpus = corpus::Corpus {
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
            doc_names: (0..self.docs.len()).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); self.docs.len()],
            doc_freqs: {
                let v = self.id_to_word.len();
                let mut df = vec![0u32; v];
                for doc in &self.docs {
                    let mut seen = std::collections::HashSet::new();
                    for &w in doc { seen.insert(w as usize); }
                    for w in seen { if w < v { df[w] += 1; } }
                }
                df
            },
            total_freqs: {
                let v = self.id_to_word.len();
                let mut tf = vec![0u32; v];
                for doc in &self.docs { for &w in doc { if (w as usize) < v { tf[w as usize] += 1; } } }
                tf
            },
        };
        Ok(Array1::from(umass_coherence(&corpus, &tops)).to_pyarray_bound(py))
    }

    /// The soft topic distribution for `data` (Corpus or token lists), as
    /// `(num_docs, num_topics)`. Words outside the fitted vocabulary are dropped;
    /// `window`/`stride` default to the values set on the model.
    #[pyo3(signature = (data, *, window=None, stride=None))]
    fn approximate_distribution<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let docs_str: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner.docs.iter().map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect()).collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("approximate_distribution expects a Corpus or token lists")
            })?
        };
        let ids = self.to_ids(&docs_str);
        let dist = m.approximate_distribution(
            &ids,
            window.unwrap_or(self.window),
            stride.unwrap_or(self.stride),
        );
        Ok(vecs_to_arr2(&dist).to_pyarray_bound(py))
    }

    /// Soft topic distribution for new documents (the approximate distribution
    /// over their words). BERTopic reads topics from text; `doc_embeddings` is
    /// accepted but not used (for API consistency with the other embedding
    /// models). Returns `(num_docs, num_topics)`.
    #[pyo3(signature = (data, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = doc_embeddings;
        self.approximate_distribution(py, data, None, None)
    }

    /// Fit, then return the document-topic distribution (`fit_transform`).
    #[pyo3(signature = (data, doc_embeddings))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(py, data, doc_embeddings)?;
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Merge groups of topics into single topics, e.g. ``[[3, 7], [1, 2]]``,
    /// rebuilding the c-TF-IDF representation and the document-topic distribution.
    fn merge_topics(&mut self, groups: Vec<Vec<usize>>) -> PyResult<()> {
        let (vocab, b25, rf, win, st) =
            (self.id_to_word.len(), self.bm25, self.reduce_frequent, self.window, self.stride);
        let m = self.model.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("model is not fitted yet; call fit() first")
        })?;
        m.merge_topics(&self.docs, &groups, vocab, b25, rf, win, st);
        Ok(())
    }

    /// Reassign noise documents (label ``-1``) to their nearest topic by c-TF-IDF
    /// fit and rebuild. Returns how many documents were reassigned.
    fn reduce_outliers(&mut self) -> PyResult<usize> {
        let (vocab, b25, rf, win, st) =
            (self.id_to_word.len(), self.bm25, self.reduce_frequent, self.window, self.stride);
        let m = self.model.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("model is not fitted yet; call fit() first")
        })?;
        let before = m.labels.iter().filter(|&&l| l < 0).count();
        m.reduce_outliers(&self.docs, vocab, b25, rf, win, st);
        Ok(before - m.labels.iter().filter(|&&l| l < 0).count())
    }

    /// Save the fitted model to `path` (topica's binary format), so a discovered
    /// fit can be reloaded and reused without refitting (useful with the stochastic
    /// `reducer="umap"` discovery).
    fn save(&self, path: &str) -> PyResult<()> {
        self.fitted_model()?;
        write_state(path, MODEL_TAG_BERTOPIC, &BertopicState {
            n_components: self.n_components,
            use_umap: self.use_umap,
            n_neighbors: self.n_neighbors,
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            nr_topics: self.nr_topics,
            window: self.window,
            stride: self.stride,
            bm25: self.bm25,
            reduce_frequent: self.reduce_frequent,
            clusterer: self.clusterer.clone(),
            num_clusters: self.num_clusters,
            seed: self.seed,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            model: self.model.clone(),
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: BertopicState = read_state(path, MODEL_TAG_BERTOPIC)?;
        let num_topics = s.model.as_ref().map_or(0, |m| m.num_topics);
        let topic_names = if s.topic_names.is_empty() {
            (0..num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(BERTopic {
            n_components: s.n_components,
            use_umap: s.use_umap,
            n_neighbors: s.n_neighbors,
            min_cluster_size: s.min_cluster_size,
            min_samples: s.min_samples,
            nr_topics: s.nr_topics,
            window: s.window,
            stride: s.stride,
            bm25: s.bm25,
            reduce_frequent: s.reduce_frequent,
            clusterer: s.clusterer,
            num_clusters: s.num_clusters,
            seed: s.seed,
            fitted: s.fitted,
            topic_names,
            model: s.model,
            id_to_word: s.id_to_word,
            docs: s.docs,
        })
    }

    /// BERTopic has no iterative objective; fit_history is always ``[]``.
    #[getter]
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// BERTopic is not an iterative sampler (UMAP + clustering); converged is always ``None``.
    #[getter]
    fn converged(&self) -> Option<bool> {
        None
    }

    fn __repr__(&self) -> String {
        let k = self.model.as_ref().map_or(0, |m| m.num_topics);
        format!("BERTopic(fitted={}, num_topics={})", self.fitted, k)
    }
}

// ---------------------------------------------------------------------------
// ETM: Embedded Topic Model (Dieng, Ruiz & Blei 2020)
// ---------------------------------------------------------------------------

/// Embedded Topic Model: LDA with the topic-word matrix factored through
/// embeddings, ``beta_{k,v} = softmax_v(rho_v . alpha_k)``, and a logistic-normal
/// document prior. You bring the word embeddings ``rho``; topica fits the topic
/// embeddings ``alpha`` and the prior by the same variational EM as ``CTM`` (no
/// VAE, no PyTorch). Semantically related words share topic mass even when a
/// topic never saw them.
///
/// No embedder of your own? `topica.llm_embed(vocabulary, model=...)` builds the
/// word embeddings `rho` (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct ETM {
    num_topics: usize,
    inference: String,
    em_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    max_inner: usize,
    hidden_size: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    seed: u64,
    // VAE-path flags (#174, #176); ignored on the EM path.
    prior: String,
    contrastive: bool,
    contrastive_weight: f64,
    contrastive_temp: f64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<etm::EtmModel>,
    vae: Option<etm_vae::EtmVaeModel>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted ETM.
#[derive(serde::Serialize, serde::Deserialize)]
struct EtmState {
    num_topics: usize,
    inference: String,
    em_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    max_inner: usize,
    hidden_size: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    seed: u64,
    #[serde(default = "default_prior")]
    prior: String,
    #[serde(default)]
    contrastive: bool,
    #[serde(default = "default_contrastive_weight")]
    contrastive_weight: f64,
    #[serde(default = "default_contrastive_temp")]
    contrastive_temp: f64,
    fitted: bool,
    topic_names: Vec<String>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // EM path fields (None when inference=="vae")
    beta_em: Option<Vec<Vec<f64>>>,
    alpha_em: Option<Vec<Vec<f64>>>,
    mu_em: Option<Vec<f64>>,
    sigma_em: Option<Vec<f64>>,
    lambda_em: Option<Vec<Vec<f64>>>,
    bound_em: Option<f64>,
    converged_em: Option<bool>,
    // VAE path fields (None when inference=="em")
    beta_vae: Option<Vec<Vec<f64>>>,
    alpha_vae: Option<Vec<Vec<f64>>>,
    doc_topic_vae: Option<Vec<Vec<f64>>>,
    bound_vae: Option<f64>,
    converged_vae: Option<bool>,
    // VAE encoder weights (None when inference=="em")
    enc_v: Option<usize>,
    enc_hidden: Option<usize>,
    enc_w1: Option<Vec<f64>>,
    enc_b1: Option<Vec<f64>>,
    enc_w2: Option<Vec<f64>>,
    enc_b2: Option<Vec<f64>>,
    enc_w_mu: Option<Vec<f64>>,
    enc_b_mu: Option<Vec<f64>>,
    enc_w_ls: Option<Vec<f64>>,
    enc_b_ls: Option<Vec<f64>>,
}

impl ETM {
    fn fitted_model(&self) -> PyResult<&etm::EtmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn ensure_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    /// Topic-word matrix beta, from whichever inference path was fit.
    fn surf_beta(&self) -> PyResult<&Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(&m.beta),
            (_, Some(m)) => Ok(&m.beta),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }

    /// Topic embeddings alpha.
    fn surf_alpha(&self) -> PyResult<&Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(&m.alpha),
            (_, Some(m)) => Ok(&m.alpha),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }

    /// Document-topic proportions theta (computed fresh for the EM path).
    fn surf_doc_topic(&self) -> PyResult<Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.doc_topics()),
            (_, Some(m)) => Ok(m.doc_topic.clone()),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }

    fn surf_bound(&self) -> PyResult<f64> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.bound),
            (_, Some(m)) => Ok(m.bound),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }

    fn surf_converged(&self) -> PyResult<bool> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.converged),
            (_, Some(m)) => Ok(m.converged),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }

    fn surf_bound_history(&self) -> PyResult<Vec<f64>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.bound_history.clone()),
            (_, Some(m)) => Ok(m.bound_history.clone()),
            _ => Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")),
        }
    }
}

#[pymethods]
impl ETM {
    /// Create an unfitted model. `inference` selects the engine: `"em"` (default)
    /// is per-document variational EM, accurate but not minibatched; `"vae"` is the
    /// reference's amortized autoencoder, which scales to large corpora and maps new
    /// documents with a single encoder pass. `convergence_tol`/`prior_variance`/
    /// `max_inner`/`sigma_shrink` govern the EM path; `hidden_size`/
    /// `batch_size`/`lr`/`wdecay`/`convergence_tol` govern the VAE path.
    /// Pass `iters` to :meth:`fit` to set the iteration count.
    #[new]
    #[pyo3(signature = (num_topics, *, inference="em", convergence_tol=1e-4,
                        sigma_shrink=0.0, prior_variance=1e6, max_inner=25,
                        hidden_size=800, batch_size=1000, lr=0.005,
                        wdecay=1.2e-6, seed=42, prior="laplace".to_string(),
                        contrastive=false, contrastive_weight=0.5, contrastive_temp=0.5,
                        em_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        inference: &str,
        convergence_tol: f64,
        sigma_shrink: f64,
        prior_variance: f64,
        max_inner: usize,
        hidden_size: usize,
        batch_size: usize,
        lr: f64,
        wdecay: f64,
        seed: u64,
        prior: String,
        contrastive: bool,
        contrastive_weight: f64,
        contrastive_temp: f64,
        em_tol: Option<f64>,
    ) -> PyResult<Self> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "ETM(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (convergence_tol - 1e-4_f64).abs() > f64::EPSILON { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !finite_pos(prior_variance) {
            return Err(PyValueError::new_err("prior_variance must be > 0"));
        }
        if inference != "em" && inference != "vae" {
            return Err(PyValueError::new_err("inference must be \"em\" or \"vae\""));
        }
        // Validate the VAE flags eagerly (they only take effect on the vae path).
        build_avitm_options(&prior, contrastive, contrastive_weight, contrastive_temp)?;
        Ok(ETM {
            num_topics,
            inference: inference.to_string(),
            em_tol: convergence_tol,
            sigma_shrink,
            prior_variance,
            max_inner,
            hidden_size,
            batch_size,
            lr,
            wdecay,
            seed,
            prior,
            contrastive,
            contrastive_weight,
            contrastive_temp,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            vae: None,
            id_to_word: Vec::new(),
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `word_embeddings`
    /// (`(len(vocabulary), E)`) and the aligned `vocabulary`. The vocabulary
    /// defines the word ids; tokens outside it are dropped.
    /// `iters` sets the number of training iterations (EM iterations or VAE epochs).
    /// `convergence_tol` overrides the constructor value for this run (when given).
    #[pyo3(signature = (data, word_embeddings, vocabulary, *, iters=None, convergence_tol=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        word_embeddings: &Bound<'_, PyAny>,
        vocabulary: Vec<String>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
        // Use fit()-level convergence_tol if given, else fall back to constructor value.
        let tol = convergence_tol.unwrap_or(self.em_tol);
        let (docs_str, corpus_opt): (Vec<Vec<String>>, Option<corpus::Corpus>) =
            if let Ok(c) = data.extract::<Corpus>() {
                let strings = c.inner.docs.iter()
                    .map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect())
                    .collect();
                (strings, Some(c.inner.clone()))
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                (docs, None)
            };
        let rho = parse_features(word_embeddings)?;
        if rho.len() != vocabulary.len() {
            return Err(PyValueError::new_err(format!(
                "word_embeddings has {} rows but vocabulary has {} words",
                rho.len(),
                vocabulary.len()
            )));
        }
        check_all_finite_2d("word_embeddings", &rho)?;
        if vocabulary.len() < self.num_topics {
            return Err(PyValueError::new_err("vocabulary must have at least num_topics words"));
        }
        let map: std::collections::HashMap<&str, u32> =
            vocabulary.iter().enumerate().map(|(i, w)| (w.as_str(), i as u32)).collect();
        let docs_ids: Vec<Vec<u32>> = docs_str
            .iter()
            .map(|d| d.iter().filter_map(|w| map.get(w.as_str()).copied()).collect())
            .collect();
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err("no in-vocabulary tokens in the documents"));
        }
        let num_types = vocabulary.len();
        self.id_to_word = vocabulary.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        if self.inference == "vae" {
            let ep = iters.unwrap_or(150);
            let opts = build_avitm_options(
                &self.prior, self.contrastive, self.contrastive_weight, self.contrastive_temp,
            )?;
            let (k, h, bs, lr, wd, et) = (
                self.num_topics, self.hidden_size, self.batch_size,
                self.lr, self.wdecay, tol,
            );
            let m = py.allow_threads(move || {
                etm_vae::fit_etm_vae(&docs_ids, k, num_types, &rho, h, ep, bs, lr, wd, et, opts, &mut rng)
            });
            self.vae = Some(m);
            self.model = None;
        } else {
            let ei = iters.unwrap_or(100);
            let (k, et, ss, pv, mi) = (
                self.num_topics, tol, self.sigma_shrink,
                self.prior_variance, self.max_inner,
            );
            let model = py.allow_threads(move || {
                etm::fit_etm(&docs_ids, k, num_types, &rho, ei, et, ss, pv, mi, &mut rng)
            });
            self.model = Some(model);
            self.vae = None;
        }
        // Retain the corpus for coherence/doc_names; build a minimal one if raw docs were given.
        self.corpus = Some(corpus_opt.unwrap_or_else(|| {
            let n = docs_str.len();
            let vocab_clone = vocabulary.clone();
            let v = vocab_clone.len();
            let mut df = vec![0u32; v];
            let mut tf = vec![0u32; v];
            let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(n);
            for doc in &docs_str {
                let ids: Vec<u32> = doc.iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect();
                let mut seen = std::collections::HashSet::new();
                for &id in &ids {
                    tf[id as usize] += 1;
                    seen.insert(id as usize);
                }
                for id in seen { df[id] += 1; }
                id_docs.push(ids);
            }
            corpus::Corpus {
                id_to_word: vocab_clone,
                docs: id_docs,
                doc_names: (0..n).map(|i| format!("doc_{i}")).collect(),
                doc_labels: vec![String::new(); n],
                doc_freqs: df,
                total_freqs: tf,
            }
        }));
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn inference(&self) -> String {
        self.inference.clone()
    }
    /// Topic-word matrix beta (num_topics, vocab), each row a distribution.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(self.surf_beta()?).to_pyarray_bound(py))
    }
    /// Document-topic proportions theta (num_docs, num_topics).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.surf_doc_topic()?).to_pyarray_bound(py))
    }
    /// Topic embeddings alpha (num_topics, E).
    #[getter]
    fn topic_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(self.surf_alpha()?).to_pyarray_bound(py))
    }
    /// The variational evidence bound (EM) or the ELBO (VAE) at convergence.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.surf_bound()
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.surf_converged()
    }
    /// Uniform convergence trace: ``(iteration, bound)`` pairs, one per EM or
    /// VAE epoch. The objective is the variational ELBO. Empty after
    /// :meth:`load` (bound_history is not persisted in the saved state).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.surf_bound_history()?
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.ensure_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.ensure_fitted()?;
        Ok(self.id_to_word.clone())
    }
    /// Document names from the training corpus, in corpus order.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.ensure_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(self.surf_beta()?);
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }
    /// UMass coherence for each topic's top-`n` words, over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(self.surf_beta()?);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        self.ensure_fitted()?;
        let (beta_em, alpha_em, mu_em, sigma_em, lambda_em, bound_em, converged_em) =
            if let Some(m) = &self.model {
                (Some(m.beta.clone()), Some(m.alpha.clone()), Some(m.mu.clone()),
                 Some(m.sigma.clone()), Some(m.lambda.clone()), Some(m.bound), Some(m.converged))
            } else {
                (None, None, None, None, None, None, None)
            };
        let (beta_vae, alpha_vae, doc_topic_vae, bound_vae, converged_vae,
             enc_v, enc_hidden, enc_w1, enc_b1, enc_w2, enc_b2,
             enc_w_mu, enc_b_mu, enc_w_ls, enc_b_ls) =
            if let Some(m) = &self.vae {
                let enc = &m.encoder;
                (Some(m.beta.clone()), Some(m.alpha.clone()), Some(m.doc_topic.clone()),
                 Some(m.bound), Some(m.converged),
                 Some(enc.v), Some(enc.hidden),
                 Some(enc.w1.clone()), Some(enc.b1.clone()),
                 Some(enc.w2.clone()), Some(enc.b2.clone()),
                 Some(enc.w_mu.clone()), Some(enc.b_mu.clone()),
                 Some(enc.w_ls.clone()), Some(enc.b_ls.clone()))
            } else {
                (None, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None)
            };
        write_state(path, MODEL_TAG_ETM, &EtmState {
            num_topics: self.num_topics,
            inference: self.inference.clone(),
            em_tol: self.em_tol,
            sigma_shrink: self.sigma_shrink,
            prior_variance: self.prior_variance,
            max_inner: self.max_inner,
            hidden_size: self.hidden_size,
            batch_size: self.batch_size,
            lr: self.lr,
            wdecay: self.wdecay,
            seed: self.seed,
            prior: self.prior.clone(),
            contrastive: self.contrastive,
            contrastive_weight: self.contrastive_weight,
            contrastive_temp: self.contrastive_temp,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            id_to_word: self.id_to_word.clone(),
            corpus: self.corpus.clone(),
            beta_em, alpha_em, mu_em, sigma_em, lambda_em, bound_em, converged_em,
            beta_vae, alpha_vae, doc_topic_vae, bound_vae, converged_vae,
            enc_v, enc_hidden, enc_w1, enc_b1, enc_w2, enc_b2,
            enc_w_mu, enc_b_mu, enc_w_ls, enc_b_ls,
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: EtmState = read_state(path, MODEL_TAG_ETM)?;
        let model = if s.inference == "em" {
            s.beta_em.map(|beta| etm::EtmModel {
                num_topics: s.num_topics,
                num_types: s.id_to_word.len(),
                beta,
                alpha: s.alpha_em.unwrap_or_default(),
                mu: s.mu_em.unwrap_or_default(),
                sigma: s.sigma_em.unwrap_or_default(),
                lambda: s.lambda_em.unwrap_or_default(),
                bound: s.bound_em.unwrap_or(f64::NAN),
                bound_history: Vec::new(),
                converged: s.converged_em.unwrap_or(false),
                em_iters_run: 0,
            })
        } else { None };
        let vae = if s.inference == "vae" {
            s.beta_vae.map(|beta| etm_vae::EtmVaeModel {
                num_topics: s.num_topics,
                num_types: s.id_to_word.len(),
                beta,
                alpha: s.alpha_vae.unwrap_or_default(),
                doc_topic: s.doc_topic_vae.unwrap_or_default(),
                bound: s.bound_vae.unwrap_or(f64::NAN),
                bound_history: Vec::new(),
                converged: s.converged_vae.unwrap_or(false),
                epochs_run: 0,
                encoder: etm_vae::Encoder {
                    v: s.enc_v.unwrap_or(0),
                    hidden: s.enc_hidden.unwrap_or(0),
                    k: s.num_topics,
                    w1: s.enc_w1.unwrap_or_default(),
                    b1: s.enc_b1.unwrap_or_default(),
                    w2: s.enc_w2.unwrap_or_default(),
                    b2: s.enc_b2.unwrap_or_default(),
                    w_mu: s.enc_w_mu.unwrap_or_default(),
                    b_mu: s.enc_b_mu.unwrap_or_default(),
                    w_ls: s.enc_w_ls.unwrap_or_default(),
                    b_ls: s.enc_b_ls.unwrap_or_default(),
                },
                prior: prior_from_str(&s.prior),
            })
        } else { None };
        Ok(ETM {
            num_topics: s.num_topics,
            inference: s.inference,
            em_tol: s.em_tol,
            sigma_shrink: s.sigma_shrink,
            prior_variance: s.prior_variance,
            max_inner: s.max_inner,
            hidden_size: s.hidden_size,
            batch_size: s.batch_size,
            lr: s.lr,
            wdecay: s.wdecay,
            seed: s.seed,
            prior: s.prior,
            contrastive: s.contrastive,
            contrastive_weight: s.contrastive_weight,
            contrastive_temp: s.contrastive_temp,
            fitted: s.fitted,
            topic_names: s.topic_names,
            id_to_word: s.id_to_word,
            corpus: s.corpus,
            model,
            vae,
        })
    }

    /// Held-out topic proportions for new documents. For the EM path this is the
    /// logistic-normal E-step with the fitted `beta` and prior held fixed; for the
    /// VAE path it is a single encoder forward pass (`theta = softmax(mu)`). Tokens
    /// outside the vocabulary are dropped. `doc_embeddings` is accepted but not
    /// used (for API consistency with the other embedding models). Returns
    /// `(num_docs, num_topics)`.
    #[pyo3(signature = (data, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = doc_embeddings;
        self.ensure_fitted()?;
        let docs = docs_to_ids(data, &self.id_to_word)?;
        if let Some(m) = &self.vae {
            Ok(vecs_to_arr2(&m.transform(&docs)).to_pyarray_bound(py))
        } else {
            let m = self.fitted_model()?;
            Ok(infer_theta_batch(py, &m.beta, &m.mu, &m.sigma, &docs).to_pyarray_bound(py))
        }
    }

    /// Fit, then return the document-topic proportions (`fit_transform`).
    #[pyo3(signature = (data, word_embeddings, vocabulary, *, iters=None))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        word_embeddings: &Bound<'py, PyAny>,
        vocabulary: Vec<String>,
        iters: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(py, data, word_embeddings, vocabulary, iters, None)?;
        Ok(vecs_to_arr2(&self.surf_doc_topic()?).to_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ETM(num_topics={}, inference={}, fitted={})",
            self.num_topics, self.inference, self.fitted
        )
    }
}

/// DETM: the Dynamic Embedded Topic Model (Dieng, Ruiz & Blei 2019,
/// arXiv:1907.05545). DETM extends ``ETM`` to time-stamped corpora: the topic
/// embeddings ``alpha`` and the per-time topic prior ``eta`` each follow a Gaussian
/// random walk, so a topic's words drift smoothly across time slices. The headline
/// output is the time-varying topic-word tensor ``beta`` (``num_times`` x
/// ``num_topics`` x ``vocab``); ``topic_word`` is its mean over time, and
/// :meth:`topic_word_at` / :meth:`top_words_at` read a single slice.
///
/// You bring the word embeddings ``rho`` like ``ETM``; topica fits the topic
/// embeddings, the per-time prior, and an amortized encoder for the document-topic
/// proportions by minibatch Adam on the ELBO (hand-coded gradients, no PyTorch).
///
/// The variational posterior over ``eta`` follows the reference: a multi-layer LSTM
/// (sized by ``eta_hidden_size`` / ``eta_nlayers``) over the per-time bag of words
/// amortizes the per-slice mean and log-variance, with the same random-walk KL. The
/// LSTM forward and its backprop-through-time are hand-coded (no PyTorch). See
/// ``src/detm.rs`` for the full account.
///
/// No embedder of your own? ``topica.llm_embed(vocabulary, model=...)`` builds the
/// word embeddings ``rho`` (OpenAI, or offline ``sentence-transformers``).
#[pyclass(module = "topica")]
pub struct DETM {
    num_topics: usize,
    delta: f64,
    hidden_size: usize,
    eta_hidden_size: usize,
    eta_nlayers: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    grad_clip: Option<f64>,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    num_times: usize,
    model: Option<detm::DetmModel>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted DETM.
#[derive(serde::Serialize, serde::Deserialize)]
struct DetmState {
    num_topics: usize,
    delta: f64,
    hidden_size: usize,
    eta_hidden_size: usize,
    eta_nlayers: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    #[serde(default)]
    grad_clip: Option<f64>,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    num_times: usize,
    num_types: usize,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    beta_over_time: Vec<Vec<Vec<f64>>>,
    doc_topic: Vec<Vec<f64>>,
    alpha: Vec<Vec<Vec<f64>>>,
    eta: Vec<Vec<f64>>,
    bound: f64,
    converged: bool,
}

impl DETM {
    fn fitted_model(&self) -> PyResult<&detm::DetmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl DETM {
    /// Create an unfitted model. ``delta`` is the random-walk standard-deviation
    /// knob on the topic-embedding and topic-prior trajectories (smaller = smoother
    /// drift; reference default 0.005). ``hidden_size`` is the document encoder
    /// width. ``eta_hidden_size``/``eta_nlayers`` size the LSTM that amortizes the
    /// per-time topic prior q(eta) (reference defaults 200 / 3).
    /// ``batch_size``/``lr``/``wdecay`` drive Adam; ``convergence_tol`` stops on the
    /// relative change in the epoch ELBO (0 disables early stop). ``grad_clip`` is an
    /// optional global gradient-norm clip (the reference's ``--clip``), off by default
    /// (``None``); set it to a positive float to rescale each minibatch's gradients so
    /// their global L2 norm does not exceed it before the Adam step, which stabilizes
    /// training on large vocabularies at higher learning rates. (The variational
    /// log-variances are additionally clamped before every ``exp`` for stability; that
    /// clamp is internal and never reached on a well-behaved fit.) Pass ``iters`` to
    /// :meth:`fit` for the epoch count.
    #[new]
    #[pyo3(signature = (num_topics, *, delta=0.005, hidden_size=800,
                        eta_hidden_size=200, eta_nlayers=3, batch_size=1000,
                        lr=0.005, wdecay=1.2e-6, grad_clip=None, convergence_tol=0.0, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        delta: f64,
        hidden_size: usize,
        eta_hidden_size: usize,
        eta_nlayers: usize,
        batch_size: usize,
        lr: f64,
        wdecay: f64,
        grad_clip: Option<f64>,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !finite_pos(delta) {
            return Err(PyValueError::new_err("delta must be > 0"));
        }
        if eta_nlayers < 1 {
            return Err(PyValueError::new_err("eta_nlayers must be >= 1"));
        }
        if eta_hidden_size < 1 {
            return Err(PyValueError::new_err("eta_hidden_size must be >= 1"));
        }
        if let Some(c) = grad_clip {
            if !(c > 0.0) || !c.is_finite() {
                return Err(PyValueError::new_err("grad_clip must be a positive finite float or None"));
            }
        }
        Ok(DETM {
            num_topics,
            delta,
            hidden_size,
            eta_hidden_size,
            eta_nlayers,
            batch_size,
            lr,
            wdecay,
            grad_clip,
            convergence_tol,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            num_times: 0,
            model: None,
            id_to_word: Vec::new(),
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `word_embeddings`
    /// (`(len(vocabulary), L)`) and the aligned `vocabulary`. `times` is each
    /// document's integer time-slice index (0-based, contiguous; alias
    /// `timestamps`). `iters` sets the number of training epochs.
    #[pyo3(signature = (data, word_embeddings, vocabulary, *, times=None, timestamps=None,
                        iters=100, convergence_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        word_embeddings: &Bound<'_, PyAny>,
        vocabulary: Vec<String>,
        times: Option<Vec<i64>>,
        timestamps: Option<Vec<i64>>,
        iters: usize,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
        // times is canonical; timestamps is the accepted alias.
        let times = match (times, timestamps) {
            (Some(t), None) => t,
            (None, Some(t)) => t,
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass either times= or timestamps=, not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "fit() requires times= (the per-document time-slice index)",
                ))
            }
        };
        let tol = convergence_tol.unwrap_or(self.convergence_tol);

        let (docs_str, corpus_opt): (Vec<Vec<String>>, Option<corpus::Corpus>) =
            if let Ok(c) = data.extract::<Corpus>() {
                let strings = c.inner.docs.iter()
                    .map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect())
                    .collect();
                (strings, Some(c.inner.clone()))
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                (docs, None)
            };
        if docs_str.is_empty() {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if times.len() != docs_str.len() {
            return Err(PyValueError::new_err(format!(
                "times has length {} but there are {} documents",
                times.len(),
                docs_str.len()
            )));
        }
        if times.iter().any(|&t| t < 0) {
            return Err(PyValueError::new_err("time-slice indices must be >= 0"));
        }
        let times_u: Vec<usize> = times.iter().map(|&t| t as usize).collect();
        let num_times = times_u.iter().copied().max().unwrap() + 1;
        let mut seen = vec![false; num_times];
        for &t in &times_u {
            seen[t] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(PyValueError::new_err(
                "time slices must be contiguous 0..max; some slice has no documents",
            ));
        }

        let rho = parse_features(word_embeddings)?;
        if rho.len() != vocabulary.len() {
            return Err(PyValueError::new_err(format!(
                "word_embeddings has {} rows but vocabulary has {} words",
                rho.len(),
                vocabulary.len()
            )));
        }
        check_all_finite_2d("word_embeddings", &rho)?;
        if vocabulary.len() < self.num_topics {
            return Err(PyValueError::new_err("vocabulary must have at least num_topics words"));
        }

        let map: std::collections::HashMap<&str, u32> =
            vocabulary.iter().enumerate().map(|(i, w)| (w.as_str(), i as u32)).collect();
        // Per-document sparse (tokens, counts) over the vocabulary ids.
        let mut tokens: Vec<Vec<u32>> = Vec::with_capacity(docs_str.len());
        let mut counts: Vec<Vec<u32>> = Vec::with_capacity(docs_str.len());
        for doc in &docs_str {
            let mut m: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
            for w in doc {
                if let Some(&id) = map.get(w.as_str()) {
                    *m.entry(id).or_insert(0) += 1;
                }
            }
            tokens.push(m.keys().copied().collect());
            counts.push(m.values().copied().collect());
        }
        if tokens.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err("no in-vocabulary tokens in the documents"));
        }

        let num_types = vocabulary.len();
        self.id_to_word = vocabulary.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (k, delta, h, eh, enl, bs, lr, wd, gc) = (
            self.num_topics, self.delta, self.hidden_size, self.eta_hidden_size,
            self.eta_nlayers, self.batch_size, self.lr, self.wdecay, self.grad_clip,
        );
        let model = py.allow_threads(move || {
            detm::fit_detm(
                &tokens, &counts, &times_u, k, num_types, num_times, &rho, delta, h, eh, enl,
                iters, bs, lr, wd, tol, gc, &mut rng,
            )
        });

        self.num_times = num_times;
        self.model = Some(model);
        // Retain a corpus for coherence/doc_names; build a minimal one if raw docs were given.
        self.corpus = Some(corpus_opt.unwrap_or_else(|| {
            let n = docs_str.len();
            let v = num_types;
            let mut df = vec![0u32; v];
            let mut tf = vec![0u32; v];
            let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(n);
            for doc in &docs_str {
                let ids: Vec<u32> =
                    doc.iter().filter_map(|w| map.get(w.as_str()).copied()).collect();
                let mut s = std::collections::HashSet::new();
                for &id in &ids {
                    tf[id as usize] += 1;
                    s.insert(id as usize);
                }
                for id in s {
                    df[id] += 1;
                }
                id_docs.push(ids);
            }
            corpus::Corpus {
                id_to_word: vocabulary.clone(),
                docs: id_docs,
                doc_names: (0..n).map(|i| format!("doc_{i}")).collect(),
                doc_labels: vec![String::new(); n],
                doc_freqs: df,
                total_freqs: tf,
            }
        }));
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// The number of time slices (available after fit).
    #[getter]
    fn num_times(&self) -> PyResult<usize> {
        self.fitted_model()?;
        Ok(self.num_times)
    }

    /// Time-collapsed topic-word matrix beta (num_topics, vocab): the mean of the
    /// per-time beta over the slices. Each row is a distribution.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word_mean()).to_pyarray_bound(py))
    }

    /// Time-varying topic-word tensor beta (num_times, num_topics, vocab); every
    /// ``beta[t, k]`` is a distribution over the vocabulary. This is DETM's headline
    /// output.
    #[getter]
    fn beta_over_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let m = self.fitted_model()?;
        let (t, k, v) = (m.num_times, m.num_topics, m.num_types);
        let mut arr = Array3::<f64>::zeros((t, k, v));
        for tt in 0..t {
            for kk in 0..k {
                for vv in 0..v {
                    arr[[tt, kk, vv]] = m.beta_over_time[tt][kk][vv];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Alias of :attr:`beta_over_time`: the per-time topic-word tensor
    /// (num_times, num_topics, vocab).
    #[getter]
    fn topic_word_over_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.beta_over_time(py)
    }

    /// Topic-word matrix at a single time slice ``t``, shape (num_topics, vocab);
    /// each row a distribution.
    fn topic_word_at<'py>(&self, py: Python<'py>, t: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if t >= m.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        Ok(vecs_to_arr2(&m.beta_over_time[t]).to_pyarray_bound(py))
    }

    /// Document-topic proportions theta (num_docs, num_topics).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Topic-embedding trajectories alpha (num_times, num_topics, L), the smooth
    /// latent the random walk regularizes (variational means).
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let m = self.fitted_model()?;
        let (t, k, l) = (m.num_times, m.num_topics, if m.num_topics > 0 { m.alpha[0][0].len() } else { 0 });
        let mut arr = Array3::<f64>::zeros((t, k, l));
        for tt in 0..t {
            for kk in 0..k {
                for ll in 0..l {
                    arr[[tt, kk, ll]] = m.alpha[tt][kk][ll];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// The time-varying topic prevalence prior eta (num_times, num_topics).
    #[getter]
    fn eta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.eta).to_pyarray_bound(py))
    }

    /// The final ELBO reached during fitting.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.bound)
    }

    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    /// Uniform convergence trace: ``(epoch, ELBO)`` pairs, one per training epoch.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        let m = self.fitted_model()?;
        Ok(m.bound_history.iter().enumerate().map(|(i, &b)| (i + 1, b)).collect())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// Document names from the training corpus, in corpus order.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Top `n` words per topic from the time-collapsed ``topic_word``. Pass
    /// ``topic=`` for a single topic. Use :meth:`top_words_at` for a single slice.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word_mean());
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }

    /// Top `n` words for the topics at a single time slice ``t``. Pass ``topic=``
    /// for one topic. This is the per-slice diagnostic: watch how a topic's words
    /// change from one slice to the next.
    #[pyo3(signature = (t, n=10, *, topic=None))]
    fn top_words_at<'py>(
        &self,
        py: Python<'py>,
        t: usize,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        if t >= m.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        let phi = vecs_to_arr2(&m.beta_over_time[t]);
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }

    /// UMass coherence for each topic's top-`n` words (time-collapsed topic_word),
    /// over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word_mean());
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `DETM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(path, MODEL_TAG_DETM, &DetmState {
            num_topics: self.num_topics,
            delta: self.delta,
            hidden_size: self.hidden_size,
            eta_hidden_size: self.eta_hidden_size,
            eta_nlayers: self.eta_nlayers,
            batch_size: self.batch_size,
            lr: self.lr,
            wdecay: self.wdecay,
            grad_clip: self.grad_clip,
            convergence_tol: self.convergence_tol,
            seed: self.seed,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            num_times: self.num_times,
            num_types: m.num_types,
            id_to_word: self.id_to_word.clone(),
            corpus: self.corpus.clone(),
            beta_over_time: m.beta_over_time.clone(),
            doc_topic: m.doc_topic.clone(),
            alpha: m.alpha.clone(),
            eta: m.eta.clone(),
            bound: m.bound,
            converged: m.converged,
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: DetmState = read_state(path, MODEL_TAG_DETM)?;
        let model = detm::DetmModel {
            num_topics: s.num_topics,
            num_times: s.num_times,
            num_types: s.num_types,
            beta_over_time: s.beta_over_time,
            doc_topic: s.doc_topic,
            alpha: s.alpha,
            eta: s.eta,
            bound: s.bound,
            bound_history: Vec::new(),
            converged: s.converged,
            epochs_run: 0,
        };
        Ok(DETM {
            num_topics: s.num_topics,
            delta: s.delta,
            hidden_size: s.hidden_size,
            eta_hidden_size: s.eta_hidden_size,
            eta_nlayers: s.eta_nlayers,
            batch_size: s.batch_size,
            lr: s.lr,
            wdecay: s.wdecay,
            grad_clip: s.grad_clip,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            num_times: s.num_times,
            model: Some(model),
            id_to_word: s.id_to_word,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "DETM(num_topics={}, num_times={}, fitted=true)",
                self.num_topics, self.num_times
            )
        } else {
            format!("DETM(num_topics={}, fitted=false)", self.num_topics)
        }
    }
}

/// InfoCTM (Wu et al. 2023), a cross-lingual neural topic model. Two ProdLDA/AVITM
/// models -- one per language, over independent vocabularies, sharing the topic
/// index -- are fit jointly and aligned by a Topic-Alignment Mutual-Information
/// term: a masked cross-lingual InfoNCE over the topic-word columns, with positive
/// word pairs taken from a bilingual ``dictionary`` (optionally densified by
/// per-language word ``embeddings``). After fitting, topic ``k`` denotes the same
/// theme in both languages, so ``topic_word(lang=...)`` and ``top_words(lang=...)``
/// return aligned topics for comparative cross-lingual analysis. This is the
/// dictionary-grounded alternative to the embedding-based ``ZeroShotTM`` path.
///
/// Training follows the InfoCTM reference (Adam ``beta1=0.9``) at a constant learning
/// rate; the reference's ``StepLR`` schedule is not applied, so an exact numerical
/// match to a reference run is not expected (the model and objective are unchanged).
#[pyclass(module = "topica")]
pub struct InfoCTM {
    num_topics: usize,
    mi_weight: f64,
    mi_temperature: f64,
    pos_threshold: f64,
    hidden_size: usize,
    dropout: f64,
    lr: f64,
    em_tol: f64,
    seed: u64,
    languages: (String, String),
    model: Option<infoctm::InfoctmModel>,
    corpus_a: Option<corpus::Corpus>,
    corpus_b: Option<corpus::Corpus>,
    fitted: bool,
}

impl InfoCTM {
    fn lang_index(&self, lang: &str) -> PyResult<usize> {
        if lang == self.languages.0 || lang == "a" || lang == "0" {
            Ok(0)
        } else if lang == self.languages.1 || lang == "b" || lang == "1" {
            Ok(1)
        } else {
            Err(PyValueError::new_err(format!(
                "lang must be one of {:?}, {:?}, \"a\", or \"b\"; got {lang:?}",
                self.languages.0, self.languages.1
            )))
        }
    }
    fn model_for(&self, lang: &str) -> PyResult<&prodlda::ProdldaModel> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        Ok(if self.lang_index(lang)? == 0 { &m.model_a } else { &m.model_b })
    }
    fn corpus_for(&self, lang: &str) -> PyResult<&corpus::Corpus> {
        let c = if self.lang_index(lang)? == 0 { &self.corpus_a } else { &self.corpus_b };
        c.as_ref().ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl InfoCTM {
    /// Create an unfitted model. `mi_weight` scales the alignment term (reference
    /// 30-50); `mi_temperature` is the InfoNCE temperature (0.2); `pos_threshold`
    /// is the cosine cutoff for the embedding-densified positive mask (0.4, used
    /// only when embeddings are given). `languages` names the two corpora for the
    /// `lang=` selector (default `("a", "b")`). Pass `iters`/`batch_size` to `fit`.
    #[new]
    #[pyo3(signature = (num_topics, *, mi_weight=30.0, mi_temperature=0.2,
                        pos_threshold=0.4, hidden_size=100, dropout=0.0, lr=0.002,
                        convergence_tol=0.0, seed=42, languages=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        mi_weight: f64,
        mi_temperature: f64,
        pos_threshold: f64,
        hidden_size: usize,
        dropout: f64,
        lr: f64,
        convergence_tol: f64,
        seed: u64,
        languages: Option<(String, String)>,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be in [0, 1)"));
        }
        if !(mi_temperature > 0.0 && mi_temperature.is_finite()) {
            return Err(PyValueError::new_err("mi_temperature must be > 0"));
        }
        Ok(InfoCTM {
            num_topics,
            mi_weight,
            mi_temperature,
            pos_threshold,
            hidden_size,
            dropout,
            lr,
            em_tol: convergence_tol,
            seed,
            languages: languages.unwrap_or_else(|| ("a".to_string(), "b".to_string())),
            model: None,
            corpus_a: None,
            corpus_b: None,
            fitted: false,
        })
    }

    /// Fit both languages jointly. `data_a`/`data_b` are Corpora or lists of token
    /// lists (independent vocabularies). `dictionary` is an iterable of
    /// `(word_a, word_b)` pairs (a bilingual lexicon). `embeddings_a`/`embeddings_b`
    /// are optional `{word: vector}` maps that densify the alignment mask; absent,
    /// the positives are the direct dictionary pairs. `iters` is the number of
    /// epochs (reference 500).
    #[pyo3(signature = (data_a, data_b, *, dictionary, embeddings_a=None,
                        embeddings_b=None, iters=None, batch_size=128))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data_a: &Bound<'_, PyAny>,
        data_b: &Bound<'_, PyAny>,
        dictionary: Vec<(String, String)>,
        embeddings_a: Option<std::collections::HashMap<String, Vec<f64>>>,
        embeddings_b: Option<std::collections::HashMap<String, Vec<f64>>>,
        iters: Option<usize>,
        batch_size: usize,
    ) -> PyResult<()> {
        let to_corpus = |data: &Bound<'_, PyAny>| -> PyResult<corpus::Corpus> {
            if let Ok(c) = data.extract::<Corpus>() {
                Ok(c.inner)
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                Ok(build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0)
            }
        };
        let corpus_a = to_corpus(data_a)?;
        let corpus_b = to_corpus(data_b)?;
        let (va, vb) = (corpus_a.num_types(), corpus_b.num_types());
        if corpus_a.num_docs() == 0 || corpus_b.num_docs() == 0 {
            return Err(PyValueError::new_err("both corpora must contain documents"));
        }
        if va < self.num_topics || vb < self.num_topics {
            return Err(PyValueError::new_err("each vocabulary must have at least num_topics words"));
        }

        // word -> id maps for each language.
        let wid = |c: &corpus::Corpus| -> std::collections::HashMap<String, usize> {
            c.id_to_word.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect()
        };
        let wa = wid(&corpus_a);
        let wb = wid(&corpus_b);

        // Bilingual dictionary matrix trans_ab (Va x Vb).
        let mut trans_ab = vec![vec![0.0f64; vb]; va];
        for (word_a, word_b) in &dictionary {
            if let (Some(&ia), Some(&ib)) = (wa.get(word_a), wb.get(word_b)) {
                trans_ab[ia][ib] = 1.0;
            }
        }

        // Optional embedding matrices aligned to each vocabulary (rows of 0 for
        // out-of-embedding words; those simply do not densify).
        let emb_matrix = |emb: Option<std::collections::HashMap<String, Vec<f64>>>,
                          c: &corpus::Corpus|
         -> Option<Vec<Vec<f64>>> {
            emb.map(|map| {
                let dim = map.values().next().map(|v| v.len()).unwrap_or(0);
                c.id_to_word
                    .iter()
                    .map(|w| map.get(w).cloned().unwrap_or_else(|| vec![0.0; dim]))
                    .collect()
            })
        };
        let emb_a = emb_matrix(embeddings_a, &corpus_a);
        let emb_b = emb_matrix(embeddings_b, &corpus_b);

        let ep = iters.unwrap_or(500);
        let (k, h, dp, lr, et) =
            (self.num_topics, self.hidden_size, self.dropout, self.lr, self.em_tol);
        let (mw, mt, pt) = (self.mi_weight, self.mi_temperature, self.pos_threshold);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let docs_a = corpus_a.docs.clone();
        let docs_b = corpus_b.docs.clone();
        let model = py.allow_threads(move || {
            infoctm::fit_infoctm(
                &docs_a, &docs_b, va, vb, &trans_ab, emb_a.as_deref(), emb_b.as_deref(),
                k, h, dp, ep, batch_size, lr, mw, mt, pt, et, &mut rng,
            )
        });
        self.model = Some(model);
        self.corpus_a = Some(corpus_a);
        self.corpus_b = Some(corpus_b);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Topic-word matrix for one language (num_topics, vocab); each row is
    /// ``softmax(beta_k)``. `lang` selects the language (its name or "a"/"b").
    #[pyo3(signature = (lang="a"))]
    fn topic_word<'py>(&self, py: Python<'py>, lang: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.model_for(lang)?.topic_word()).to_pyarray_bound(py))
    }

    /// Document-topic proportions for one language (num_docs, num_topics).
    #[pyo3(signature = (lang="a"))]
    fn doc_topic<'py>(&self, py: Python<'py>, lang: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.model_for(lang)?.doc_topic).to_pyarray_bound(py))
    }

    /// Vocabulary for one language, in id order.
    #[pyo3(signature = (lang="a"))]
    fn vocabulary(&self, lang: &str) -> PyResult<Vec<String>> {
        Ok(self.corpus_for(lang)?.id_to_word.clone())
    }

    /// Top-`n` words per topic for one language as `(word, weight)` pairs.
    #[pyo3(signature = (n=10, *, lang="a"))]
    fn top_words(&self, n: usize, lang: &str) -> PyResult<Vec<Vec<(String, f64)>>> {
        let model = self.model_for(lang)?;
        let vocab = &self.corpus_for(lang)?.id_to_word;
        let tw = model.topic_word();
        Ok(tw
            .iter()
            .map(|row| {
                let mut idx: Vec<usize> = (0..row.len()).collect();
                idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                idx.into_iter().take(n).map(|j| (vocab[j].clone(), row[j])).collect()
            })
            .collect())
    }

    /// Assign held-out documents of one language to the discovered topics
    /// (num_docs, num_topics) via a single encoder pass. Words outside that
    /// language's vocabulary are dropped.
    #[pyo3(signature = (data, *, lang="a"))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        lang: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self.model_for(lang)?;
        let docs = docs_to_ids(data, &self.corpus_for(lang)?.id_to_word)?;
        Ok(vecs_to_arr2(&model.transform(&docs)).to_pyarray_bound(py))
    }

    /// The per-epoch training ELBO (negative joint loss) trace.
    #[getter]
    fn fit_history(&self) -> Vec<f64> {
        self.model.as_ref().map(|m| m.bound_history.clone()).unwrap_or_default()
    }

    #[getter]
    fn converged(&self) -> Option<bool> {
        self.model.as_ref().map(|m| m.converged)
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!("InfoCTM(num_topics={}, fitted)", self.num_topics)
        } else {
            format!("InfoCTM(num_topics={}, unfitted)", self.num_topics)
        }
    }
}

/// ProdLDA (Srivastava & Sutton 2017), the AVITM autoencoding-variational topic
/// model. ProdLDA is LDA with the word-level mixture replaced by a *product of
/// experts*: each topic is an unnormalized expert and the word distribution is
/// ``softmax(beta . theta)`` rather than ``softmax(beta) . theta``, which yields
/// noticeably more coherent topics. Inference is amortized -- an encoder network
/// maps a document's bag of words to a logistic-normal posterior over ``theta``,
/// trained by minibatch Adam on the ELBO -- so new documents transform with a
/// single forward pass. Batch normalization and high-momentum Adam guard against
/// the component collapse that otherwise afflicts this model. Unlike ``ETM`` you
/// bring no embeddings: ``beta`` is learned directly.
#[pyclass(module = "topica")]
pub struct ProdLDA {
    num_topics: usize,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    em_tol: f64,
    seed: u64,
    // #176 prior: "laplace" (default) or "dirichlet" (Weibull-reparameterized).
    prior: String,
    // #174 contrastive (InfoNCE) regularization on the topic vectors.
    contrastive: bool,
    contrastive_weight: f64,
    contrastive_temp: f64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<prodlda::ProdldaModel>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted ProdLDA.
// Serde defaults for the VAE flags so pre-change saved models load unchanged.
fn default_prior() -> String {
    "laplace".to_string()
}
fn default_contrastive_weight() -> f64 {
    0.5
}
fn default_contrastive_temp() -> f64 {
    0.5
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ProdldaState {
    num_topics: usize,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    em_tol: f64,
    seed: u64,
    // VAE objective/prior flags (#174, #176). `#[serde(default)]` so models saved
    // before these fields existed still load with the pre-change behavior.
    #[serde(default = "default_prior")]
    prior: String,
    #[serde(default)]
    contrastive: bool,
    #[serde(default = "default_contrastive_weight")]
    contrastive_weight: f64,
    #[serde(default = "default_contrastive_temp")]
    contrastive_temp: f64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Fitted model fields
    doc_topic: Option<Vec<Vec<f64>>>,
    bound: Option<f64>,
    bound_history: Option<Vec<f64>>,
    converged: Option<bool>,
    epochs_run: Option<usize>,
    // Weights
    w_v: Option<usize>,
    w_hidden: Option<usize>,
    w_k: Option<usize>,
    w_w1: Option<Vec<f64>>,
    w_b1: Option<Vec<f64>>,
    w_w2: Option<Vec<f64>>,
    w_b2: Option<Vec<f64>>,
    w_w_mu: Option<Vec<f64>>,
    w_b_mu: Option<Vec<f64>>,
    w_w_ls: Option<Vec<f64>>,
    w_b_ls: Option<Vec<f64>>,
    w_beta: Option<Vec<f64>>,
    // BN mu running stats
    bn_running_mean: Option<Vec<f64>>,
    bn_running_var: Option<Vec<f64>>,
}

/// Validate the VAE-model flags (#174, #176) and build the [`prodlda::AvitmOptions`]
/// passed into `fit_avitm`. With `prior == "laplace"` and `contrastive == false`
/// (the defaults) this yields `AvitmOptions::default()`, the pre-flag code path.
/// Map a stored/validated prior string to the enum. Lenient (unknown -> laplace);
/// used on the `load` path where the string was already validated at construction.
/// Construction itself validates strictly via [`build_avitm_options`].
fn prior_from_str(prior: &str) -> prodlda::Prior {
    match prior {
        "dirichlet" => prodlda::Prior::Dirichlet,
        "stick_breaking" => prodlda::Prior::StickBreaking,
        _ => prodlda::Prior::Laplace,
    }
}

fn build_avitm_options(
    prior: &str,
    contrastive: bool,
    contrastive_weight: f64,
    contrastive_temp: f64,
) -> PyResult<prodlda::AvitmOptions> {
    let prior_enum = match prior {
        "laplace" => prodlda::Prior::Laplace,
        "dirichlet" => prodlda::Prior::Dirichlet,
        "stick_breaking" => prodlda::Prior::StickBreaking,
        other => {
            return Err(PyValueError::new_err(format!(
                "prior must be \"laplace\", \"dirichlet\", or \"stick_breaking\", got {other:?}"
            )))
        }
    };
    if contrastive {
        if !(contrastive_weight >= 0.0 && contrastive_weight.is_finite()) {
            return Err(PyValueError::new_err("contrastive_weight must be >= 0 and finite"));
        }
        if !(contrastive_temp > 0.0 && contrastive_temp.is_finite()) {
            return Err(PyValueError::new_err("contrastive_temp must be > 0 and finite"));
        }
    }
    Ok(prodlda::AvitmOptions {
        prior: prior_enum,
        contrastive,
        contrastive_weight,
        contrastive_temp,
    })
}

impl ProdLDA {
    fn fitted_model(&self) -> PyResult<&prodlda::ProdldaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl ProdLDA {
    /// Create an unfitted model. `alpha` is the symmetric Dirichlet prior
    /// concentration (reference 1.0); `hidden_size` is the encoder width (reference
    /// 100); `dropout` is the dropout rate on the hidden layer and on `theta`;
    /// `batch_size`/`lr` drive Adam (reference 200/0.002, with `beta1 = 0.99`);
    /// `convergence_tol > 0` stops early on the relative change in the epoch ELBO (0 runs
    /// all epochs). Pass `iters` to :meth:`fit` to set the number of epochs.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=1.0, hidden_size=100, dropout=0.2,
                        batch_size=200, lr=0.002, convergence_tol=0.0, seed=42,
                        prior="laplace".to_string(), contrastive=false,
                        contrastive_weight=0.5, contrastive_temp=0.5, em_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: f64,
        hidden_size: usize,
        dropout: f64,
        batch_size: usize,
        lr: f64,
        convergence_tol: f64,
        seed: u64,
        prior: String,
        contrastive: bool,
        contrastive_weight: f64,
        contrastive_temp: f64,
        em_tol: Option<f64>,
    ) -> PyResult<Self> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "ProdLDA(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            // ProdLDA default is 0.0; if unchanged, use the deprecated value.
            if convergence_tol != 0.0 { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !finite_pos(alpha) {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be in [0, 1)"));
        }
        // Validate the flags eagerly so a bad prior/weight fails at construction.
        build_avitm_options(&prior, contrastive, contrastive_weight, contrastive_temp)?;
        Ok(ProdLDA {
            num_topics,
            hidden_size,
            alpha,
            dropout,
            batch_size,
            lr,
            em_tol: convergence_tol,
            seed,
            prior,
            contrastive,
            contrastive_weight,
            contrastive_temp,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists).
    /// `iters` sets the number of training epochs (default 200).
    /// `convergence_tol` overrides the constructor value for this run (when given).
    #[pyo3(signature = (data, *, iters=None, convergence_tol=None))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: Option<usize>, convergence_tol: Option<f64>) -> PyResult<()> {
        let tol = convergence_tol.unwrap_or(self.em_tol);
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        if num_types < self.num_topics {
            return Err(PyValueError::new_err("vocabulary must have at least num_topics words"));
        }
        let ep = iters.unwrap_or(200);
        let opts = build_avitm_options(
            &self.prior, self.contrastive, self.contrastive_weight, self.contrastive_temp,
        )?;
        let (k, h, a, dp, bs, lr, et) = (
            self.num_topics, self.hidden_size, self.alpha, self.dropout,
            self.batch_size, self.lr, tol,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let empty: Vec<Vec<f64>> = vec![Vec::new(); corpus.docs.len()];
        let (model, corpus) = py.allow_threads(move || {
            let m = prodlda::fit_avitm(
                &corpus.docs, &empty, prodlda::InputMode::BowOnly, k, num_types, 0, h, a, dp,
                ep, bs, lr, et, opts, &mut rng,
            );
            (m, corpus)
        });
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix (num_topics, vocab); each row is ``softmax(beta_k)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word()).to_pyarray_bound(py))
    }
    /// Document-topic proportions theta (num_docs, num_topics); rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// The ELBO (negative training loss) at the final epoch.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.bound)
    }
    /// Per-epoch ELBO trajectory.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.bound_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Uniform convergence trace: ``(epoch, elbo)`` pairs, one per training
    /// epoch (same as :attr:`bound_history` but indexed).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?
            .bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }
    #[getter]
    fn epochs_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.epochs_run)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        topic_words_helper(py, &phi, &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Held-out topic proportions for new documents: one encoder forward pass each
    /// (`theta = softmax(mu)`, running batchnorm statistics, no sampling). Tokens
    /// outside the vocabulary are dropped. Returns `(num_docs, num_topics)`.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        Ok(vecs_to_arr2(&m.transform(&docs)).to_pyarray_bound(py))
    }

    /// Fit, then return the document-topic proportions (`fit_transform`).
    #[pyo3(signature = (data))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(py, data, None, None)?;
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(path, MODEL_TAG_PRODLDA, &ProdldaState {
            num_topics: self.num_topics,
            hidden_size: self.hidden_size,
            alpha: self.alpha,
            dropout: self.dropout,
            batch_size: self.batch_size,
            lr: self.lr,
            em_tol: self.em_tol,
            seed: self.seed,
            prior: self.prior.clone(),
            contrastive: self.contrastive,
            contrastive_weight: self.contrastive_weight,
            contrastive_temp: self.contrastive_temp,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            corpus: self.corpus.clone(),
            doc_topic: Some(m.doc_topic.clone()),
            bound: Some(m.bound),
            bound_history: Some(m.bound_history.clone()),
            converged: Some(m.converged),
            epochs_run: Some(m.epochs_run),
            w_v: Some(m.weights.v),
            w_hidden: Some(m.weights.hidden),
            w_k: Some(m.weights.k),
            w_w1: Some(m.weights.w1.clone()),
            w_b1: Some(m.weights.b1.clone()),
            w_w2: Some(m.weights.w2.clone()),
            w_b2: Some(m.weights.b2.clone()),
            w_w_mu: Some(m.weights.w_mu.clone()),
            w_b_mu: Some(m.weights.b_mu.clone()),
            w_w_ls: Some(m.weights.w_ls.clone()),
            w_b_ls: Some(m.weights.b_ls.clone()),
            w_beta: Some(m.weights.beta.clone()),
            bn_running_mean: Some(m.bn_mu.running_mean.clone()),
            bn_running_var: Some(m.bn_mu.running_var.clone()),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: ProdldaState = read_state(path, MODEL_TAG_PRODLDA)?;
        let model = if s.fitted && s.w_v.is_some() {
            let v = s.w_v.unwrap();
            let hidden = s.w_hidden.unwrap();
            let k = s.w_k.unwrap();
            Some(prodlda::ProdldaModel {
                num_topics: s.num_topics,
                num_types: v,
                doc_topic: s.doc_topic.unwrap_or_default(),
                bound: s.bound.unwrap_or(f64::NAN),
                bound_history: s.bound_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                epochs_run: s.epochs_run.unwrap_or(0),
                weights: prodlda::Weights {
                    v, e: 0, hidden, k, mode: prodlda::InputMode::BowOnly,
                    w1: s.w_w1.unwrap_or_default(),
                    b1: s.w_b1.unwrap_or_default(),
                    w2: s.w_w2.unwrap_or_default(),
                    b2: s.w_b2.unwrap_or_default(),
                    w_mu: s.w_w_mu.unwrap_or_default(),
                    b_mu: s.w_b_mu.unwrap_or_default(),
                    w_ls: s.w_w_ls.unwrap_or_default(),
                    b_ls: s.w_b_ls.unwrap_or_default(),
                    beta: s.w_beta.unwrap_or_default(),
                },
                bn_mu: prodlda::BatchNorm {
                    running_mean: s.bn_running_mean.unwrap_or_else(|| vec![0.0; k]),
                    running_var: s.bn_running_var.unwrap_or_else(|| vec![1.0; k]),
                    momentum: 0.1,
                },
                prior: prior_from_str(&s.prior),
            })
        } else { None };
        Ok(ProdLDA {
            num_topics: s.num_topics,
            hidden_size: s.hidden_size,
            alpha: s.alpha,
            dropout: s.dropout,
            batch_size: s.batch_size,
            lr: s.lr,
            em_tol: s.em_tol,
            seed: s.seed,
            prior: s.prior,
            contrastive: s.contrastive,
            contrastive_weight: s.contrastive_weight,
            contrastive_temp: s.contrastive_temp,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    /// The document-topic prior: ``"laplace"`` (default) or ``"dirichlet"``.
    #[getter]
    fn prior(&self) -> String {
        self.prior.clone()
    }
    /// Whether contrastive (InfoNCE) regularization is enabled.
    #[getter]
    fn contrastive(&self) -> bool {
        self.contrastive
    }

    fn __repr__(&self) -> String {
        format!("ProdLDA(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// --- CombinedTM / ZeroShotTM (Bianchi et al. 2021) ---------------------------
//
// Both models are ProdLDA with a different encoder *input*: CombinedTM
// concatenates the normalized bag of words with a caller-supplied document
// embedding (`InputMode::BowEmb`), and ZeroShotTM uses the embedding alone
// (`InputMode::EmbOnly`). The decoder, prior, KL, reparameterization, batchnorm,
// Adam, and BoW reconstruction loss are identical to ProdLDA; see
// `crate::prodlda::fit_avitm`. The two pyclasses share their whole surface, so we
// generate them with one macro and only vary the input mode, the model tag, and
// the class name. Embeddings are caller-supplied (sentence-transformers / API /
// ollama), matching ETM's caller-supplied-vectors pattern; ZeroShotTM's
// embedding-only encoder is what enables cross-lingual transfer at `transform`.

/// Serializable snapshot of a fitted CombinedTM / ZeroShotTM. The encoder `w1` is
/// `hidden x (V + E)`; `emb_dim` records `E` and `mode` the encoder input.
#[derive(serde::Serialize, serde::Deserialize)]
struct CtmEmbState {
    num_topics: usize,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    convergence_tol: f64,
    seed: u64,
    #[serde(default = "default_prior")]
    prior: String,
    #[serde(default)]
    contrastive: bool,
    #[serde(default = "default_contrastive_weight")]
    contrastive_weight: f64,
    #[serde(default = "default_contrastive_temp")]
    contrastive_temp: f64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    emb_dim: usize,
    mode: u8, // 1 = BowEmb (CombinedTM), 2 = EmbOnly (ZeroShotTM)
    doc_topic: Option<Vec<Vec<f64>>>,
    bound: Option<f64>,
    bound_history: Option<Vec<f64>>,
    converged: Option<bool>,
    epochs_run: Option<usize>,
    w_v: Option<usize>,
    w_e: Option<usize>,
    w_hidden: Option<usize>,
    w_k: Option<usize>,
    w_w1: Option<Vec<f64>>,
    w_b1: Option<Vec<f64>>,
    w_w2: Option<Vec<f64>>,
    w_b2: Option<Vec<f64>>,
    w_w_mu: Option<Vec<f64>>,
    w_b_mu: Option<Vec<f64>>,
    w_w_ls: Option<Vec<f64>>,
    w_b_ls: Option<Vec<f64>>,
    w_beta: Option<Vec<f64>>,
    bn_running_mean: Option<Vec<f64>>,
    bn_running_var: Option<Vec<f64>>,
}

fn mode_to_u8(m: prodlda::InputMode) -> u8 {
    match m {
        prodlda::InputMode::BowOnly => 0,
        prodlda::InputMode::BowEmb => 1,
        prodlda::InputMode::EmbOnly => 2,
    }
}

fn u8_to_mode(m: u8) -> prodlda::InputMode {
    match m {
        2 => prodlda::InputMode::EmbOnly,
        _ => prodlda::InputMode::BowEmb,
    }
}

/// Parse `doc_embeddings` into dense rows and check the row count matches the
/// document count.
fn parse_doc_embeddings(
    data: &Bound<'_, PyAny>,
    num_docs: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let embs = parse_features(data)?;
    if embs.len() != num_docs {
        return Err(PyValueError::new_err(format!(
            "doc_embeddings has {} rows but corpus has {} documents",
            embs.len(),
            num_docs
        )));
    }
    check_all_finite_2d("doc_embeddings", &embs)?;
    Ok(embs)
}

macro_rules! ctm_embedding_model {
    ($name:ident, $tag:expr, $mode:expr, $repr:expr, $doc:expr) => {
        #[doc = $doc]
        #[pyclass(module = "topica")]
        pub struct $name {
            num_topics: usize,
            hidden_size: usize,
            alpha: f64,
            dropout: f64,
            batch_size: usize,
            lr: f64,
            convergence_tol: f64,
            seed: u64,
            prior: String,
            contrastive: bool,
            contrastive_weight: f64,
            contrastive_temp: f64,
            fitted: bool,
            topic_names: Vec<String>,
            model: Option<prodlda::ProdldaModel>,
            corpus: Option<corpus::Corpus>,
            emb_dim: usize,
        }

        impl $name {
            fn fitted_model(&self) -> PyResult<&prodlda::ProdldaModel> {
                self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("model is not fitted yet; call fit() first")
                })
            }
        }

        #[pymethods]
        impl $name {
            /// Create an unfitted model. `alpha` is the symmetric Dirichlet prior
            /// concentration (reference 1.0); `hidden_size` is the encoder width
            /// (reference 100); `dropout` is the dropout rate on the hidden layer
            /// and on `theta`; `batch_size`/`lr` drive Adam (reference 200/0.002,
            /// with `beta1 = 0.99`); `convergence_tol > 0` stops early on the
            /// relative change in the epoch ELBO (0 runs all epochs). Pass `iters`
            /// to :meth:`fit` to set the number of epochs.
            #[new]
            #[pyo3(signature = (num_topics, *, alpha=1.0, hidden_size=100, dropout=0.2,
                                batch_size=200, lr=0.002, convergence_tol=0.0, seed=42,
                                prior="laplace".to_string(), contrastive=false,
                                contrastive_weight=0.5, contrastive_temp=0.5))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
                alpha: f64,
                hidden_size: usize,
                dropout: f64,
                batch_size: usize,
                lr: f64,
                convergence_tol: f64,
                seed: u64,
                prior: String,
                contrastive: bool,
                contrastive_weight: f64,
                contrastive_temp: f64,
            ) -> PyResult<Self> {
                if num_topics < 2 {
                    return Err(PyValueError::new_err("need at least 2 topics"));
                }
                if !finite_pos(alpha) {
                    return Err(PyValueError::new_err("alpha must be > 0"));
                }
                if !(0.0..1.0).contains(&dropout) {
                    return Err(PyValueError::new_err("dropout must be in [0, 1)"));
                }
                build_avitm_options(&prior, contrastive, contrastive_weight, contrastive_temp)?;
                Ok($name {
                    num_topics,
                    hidden_size,
                    alpha,
                    dropout,
                    batch_size,
                    lr,
                    convergence_tol,
                    seed,
                    prior,
                    contrastive,
                    contrastive_weight,
                    contrastive_temp,
                    fitted: false,
                    topic_names: Vec::new(),
                    model: None,
                    corpus: None,
                    emb_dim: 0,
                })
            }

            /// Fit on `data` (a Corpus or list of token lists) with
            /// `doc_embeddings`, a `(num_docs, E)` dense array (one row per
            /// document, in corpus order). The decoder reconstructs the bag of
            /// words; the encoder reads the embedding (and, for CombinedTM, the
            /// bag of words too). `iters` sets the number of training epochs
            /// (default 200). `convergence_tol` overrides the constructor value
            /// for this run (when given).
            #[pyo3(signature = (data, doc_embeddings, *, iters=None, convergence_tol=None))]
            fn fit(
                &mut self,
                py: Python<'_>,
                data: &Bound<'_, PyAny>,
                doc_embeddings: &Bound<'_, PyAny>,
                iters: Option<usize>,
                convergence_tol: Option<f64>,
            ) -> PyResult<()> {
                let tol = convergence_tol.unwrap_or(self.convergence_tol);
                let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
                    c.inner
                } else {
                    let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                        PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                    })?;
                    build_corpus_from_docs(
                        docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0,
                    )?
                    .0
                };
                if corpus.num_docs() == 0 {
                    return Err(PyValueError::new_err("corpus contains no documents"));
                }
                let num_types = corpus.num_types();
                if num_types < self.num_topics {
                    return Err(PyValueError::new_err(
                        "vocabulary must have at least num_topics words",
                    ));
                }
                let embs = parse_doc_embeddings(doc_embeddings, corpus.num_docs())?;
                let emb_dim = embs.first().map(|r| r.len()).unwrap_or(0);
                if emb_dim == 0 {
                    return Err(PyValueError::new_err("doc_embeddings must have at least one column"));
                }
                self.emb_dim = emb_dim;
                let ep = iters.unwrap_or(200);
                let opts = build_avitm_options(
                    &self.prior, self.contrastive, self.contrastive_weight,
                    self.contrastive_temp,
                )?;
                let (k, h, a, dp, bs, lr) = (
                    self.num_topics, self.hidden_size, self.alpha, self.dropout,
                    self.batch_size, self.lr,
                );
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                let (model, corpus) = py.allow_threads(move || {
                    let m = prodlda::fit_avitm(
                        &corpus.docs, &embs, $mode, k, num_types, emb_dim, h, a, dp, ep, bs,
                        lr, tol, opts, &mut rng,
                    );
                    (m, corpus)
                });
                self.model = Some(model);
                self.corpus = Some(corpus);
                self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
                self.fitted = true;
                Ok(())
            }

            #[getter]
            fn num_topics(&self) -> usize {
                self.num_topics
            }
            /// Topic-word matrix (num_topics, vocab); each row is ``softmax(beta_k)``.
            #[getter]
            fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                Ok(vecs_to_arr2(&self.fitted_model()?.topic_word()).to_pyarray_bound(py))
            }
            /// Document-topic proportions theta (num_docs, num_topics); rows sum to 1.
            #[getter]
            fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
            }
            /// The ELBO (negative training loss) at the final epoch.
            #[getter]
            fn bound(&self) -> PyResult<f64> {
                Ok(self.fitted_model()?.bound)
            }
            /// Per-epoch ELBO trajectory.
            #[getter]
            fn bound_history(&self) -> PyResult<Vec<f64>> {
                Ok(self.fitted_model()?.bound_history.clone())
            }
            #[getter]
            fn converged(&self) -> PyResult<bool> {
                Ok(self.fitted_model()?.converged)
            }
            /// Uniform convergence trace: ``(epoch, elbo)`` pairs, one per epoch.
            #[getter]
            fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
                Ok(self
                    .fitted_model()?
                    .bound_history
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| (i + 1, b))
                    .collect())
            }
            #[getter]
            fn epochs_run(&self) -> PyResult<usize> {
                Ok(self.fitted_model()?.epochs_run)
            }
            #[getter]
            fn topic_names(&self) -> PyResult<Vec<String>> {
                self.fitted_model()?;
                Ok(self.topic_names.clone())
            }
            #[setter]
            fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
                if names.len() != self.num_topics {
                    return Err(PyValueError::new_err(format!(
                        "topic_names must have length {} (got {})",
                        self.num_topics,
                        names.len()
                    )));
                }
                self.topic_names = names;
                Ok(())
            }
            #[getter]
            fn vocabulary(&self) -> PyResult<Vec<String>> {
                self.fitted_model()?;
                Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
            }
            #[getter]
            fn doc_names(&self) -> PyResult<Vec<String>> {
                self.fitted_model()?;
                Ok(self.corpus.as_ref().unwrap().doc_names.clone())
            }
            #[pyo3(signature = (n=10, *, topic=None))]
            fn top_words<'py>(
                &self,
                py: Python<'py>,
                n: usize,
                topic: Option<usize>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
                topic_words_helper(
                    py,
                    &phi,
                    &self.corpus.as_ref().unwrap().id_to_word,
                    self.num_topics,
                    n,
                    topic,
                )
            }
            #[pyo3(signature = (n=10))]
            fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
                let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
                let tops = top_word_ids_phi(&phi, self.num_topics, n);
                Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                    .to_pyarray_bound(py))
            }

            /// Held-out topic proportions for new documents: one encoder forward
            /// pass each (no sampling, running batchnorm statistics). Pass the same
            /// `data`/`doc_embeddings` shape as :meth:`fit`. For ZeroShotTM the
            /// encoder reads the embeddings alone, so a multilingual encoder maps
            /// documents in a new language to the trained topics; for CombinedTM
            /// the bag of words is read as well. Tokens outside the vocabulary are
            /// dropped. Returns `(num_docs, num_topics)`.
            #[pyo3(signature = (data, doc_embeddings))]
            fn transform<'py>(
                &self,
                py: Python<'py>,
                data: &Bound<'py, PyAny>,
                doc_embeddings: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let m = self.fitted_model()?;
                let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
                let embs = parse_doc_embeddings(doc_embeddings, docs.len())?;
                Ok(vecs_to_arr2(&m.transform_with_emb(&docs, &embs)).to_pyarray_bound(py))
            }

            /// Fit, then return the document-topic proportions (`fit_transform`).
            #[pyo3(signature = (data, doc_embeddings, *, iters=None))]
            fn fit_transform<'py>(
                &mut self,
                py: Python<'py>,
                data: &Bound<'py, PyAny>,
                doc_embeddings: &Bound<'py, PyAny>,
                iters: Option<usize>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                self.fit(py, data, doc_embeddings, iters, None)?;
                Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
            }

            /// Save the fitted model to `path` (topica's binary format).
            fn save(&self, path: &str) -> PyResult<()> {
                let m = self.fitted_model()?;
                write_state(path, $tag, &CtmEmbState {
                    num_topics: self.num_topics,
                    hidden_size: self.hidden_size,
                    alpha: self.alpha,
                    dropout: self.dropout,
                    batch_size: self.batch_size,
                    lr: self.lr,
                    convergence_tol: self.convergence_tol,
                    seed: self.seed,
                    prior: self.prior.clone(),
                    contrastive: self.contrastive,
                    contrastive_weight: self.contrastive_weight,
                    contrastive_temp: self.contrastive_temp,
                    fitted: self.fitted,
                    topic_names: self.topic_names.clone(),
                    corpus: self.corpus.clone(),
                    emb_dim: self.emb_dim,
                    mode: mode_to_u8($mode),
                    doc_topic: Some(m.doc_topic.clone()),
                    bound: Some(m.bound),
                    bound_history: Some(m.bound_history.clone()),
                    converged: Some(m.converged),
                    epochs_run: Some(m.epochs_run),
                    w_v: Some(m.weights.v),
                    w_e: Some(m.weights.e),
                    w_hidden: Some(m.weights.hidden),
                    w_k: Some(m.weights.k),
                    w_w1: Some(m.weights.w1.clone()),
                    w_b1: Some(m.weights.b1.clone()),
                    w_w2: Some(m.weights.w2.clone()),
                    w_b2: Some(m.weights.b2.clone()),
                    w_w_mu: Some(m.weights.w_mu.clone()),
                    w_b_mu: Some(m.weights.b_mu.clone()),
                    w_w_ls: Some(m.weights.w_ls.clone()),
                    w_b_ls: Some(m.weights.b_ls.clone()),
                    w_beta: Some(m.weights.beta.clone()),
                    bn_running_mean: Some(m.bn_mu.running_mean.clone()),
                    bn_running_var: Some(m.bn_mu.running_var.clone()),
                })
            }

            /// Load a model previously written by :meth:`save`.
            #[staticmethod]
            fn load(path: &str) -> PyResult<Self> {
                let s: CtmEmbState = read_state(path, $tag)?;
                let model = if s.fitted && s.w_v.is_some() {
                    let v = s.w_v.unwrap();
                    let e = s.w_e.unwrap_or(0);
                    let hidden = s.w_hidden.unwrap();
                    let k = s.w_k.unwrap();
                    Some(prodlda::ProdldaModel {
                        num_topics: s.num_topics,
                        num_types: v,
                        doc_topic: s.doc_topic.unwrap_or_default(),
                        bound: s.bound.unwrap_or(f64::NAN),
                        bound_history: s.bound_history.unwrap_or_default(),
                        converged: s.converged.unwrap_or(false),
                        epochs_run: s.epochs_run.unwrap_or(0),
                        weights: prodlda::Weights {
                            v,
                            e,
                            hidden,
                            k,
                            mode: u8_to_mode(s.mode),
                            w1: s.w_w1.unwrap_or_default(),
                            b1: s.w_b1.unwrap_or_default(),
                            w2: s.w_w2.unwrap_or_default(),
                            b2: s.w_b2.unwrap_or_default(),
                            w_mu: s.w_w_mu.unwrap_or_default(),
                            b_mu: s.w_b_mu.unwrap_or_default(),
                            w_ls: s.w_w_ls.unwrap_or_default(),
                            b_ls: s.w_b_ls.unwrap_or_default(),
                            beta: s.w_beta.unwrap_or_default(),
                        },
                        bn_mu: prodlda::BatchNorm {
                            running_mean: s.bn_running_mean.unwrap_or_else(|| vec![0.0; k]),
                            running_var: s.bn_running_var.unwrap_or_else(|| vec![1.0; k]),
                            momentum: 0.1,
                        },
                        prior: prior_from_str(&s.prior),
                    })
                } else {
                    None
                };
                Ok($name {
                    num_topics: s.num_topics,
                    hidden_size: s.hidden_size,
                    alpha: s.alpha,
                    dropout: s.dropout,
                    batch_size: s.batch_size,
                    lr: s.lr,
                    convergence_tol: s.convergence_tol,
                    seed: s.seed,
                    prior: s.prior,
                    contrastive: s.contrastive,
                    contrastive_weight: s.contrastive_weight,
                    contrastive_temp: s.contrastive_temp,
                    fitted: s.fitted,
                    topic_names: s.topic_names,
                    model,
                    corpus: s.corpus,
                    emb_dim: s.emb_dim,
                })
            }

            /// The document-topic prior: ``"laplace"`` (default) or ``"dirichlet"``.
            #[getter]
            fn prior(&self) -> String {
                self.prior.clone()
            }
            /// Whether contrastive (InfoNCE) regularization is enabled.
            #[getter]
            fn contrastive(&self) -> bool {
                self.contrastive
            }

            fn __repr__(&self) -> String {
                format!(concat!($repr, "(num_topics={}, fitted={})"), self.num_topics, self.fitted)
            }
        }
    };
}

ctm_embedding_model!(
    CombinedTM,
    MODEL_TAG_COMBINEDTM,
    prodlda::InputMode::BowEmb,
    "CombinedTM",
    "CombinedTM (Bianchi, Terragni & Hovy 2021), a contextualized topic model. \
CombinedTM is ProdLDA whose encoder reads the normalized bag of words \
*concatenated with* a caller-supplied document embedding (e.g. from a \
sentence-transformer); the product-of-experts decoder still reconstructs the bag \
of words. Mixing contextual embeddings into the encoder yields more coherent \
topics than bag-of-words ProdLDA. Bring the embeddings at :meth:`fit` as a \
`(num_docs, E)` array, aligned to the documents. The reference implementation is \
`contextualized-topic-models` (Bianchi et al., MIT)."
);

ctm_embedding_model!(
    ZeroShotTM,
    MODEL_TAG_ZEROSHOTTM,
    prodlda::InputMode::EmbOnly,
    "ZeroShotTM",
    "ZeroShotTM (Bianchi, Nozza & Hovy 2021), a contextualized topic model. \
ZeroShotTM is ProdLDA whose encoder reads *only* a caller-supplied document \
embedding (no bag of words); the product-of-experts decoder still reconstructs \
the bag of words. Because topics are inferred from the embedding alone, a \
document embedded with a multilingual encoder maps to the trained topics without \
any bag of words at all, which enables cross-lingual transfer: fit on one \
language, :meth:`transform` documents in another. Bring the embeddings at \
:meth:`fit` as a `(num_docs, E)` array, aligned to the documents. The reference \
implementation is `contextualized-topic-models` (Bianchi et al., MIT)."
);

/// NMF, non-negative matrix factorization for topic modeling (Lee & Seung 2001;
/// Boutsidis & Gallopoulos 2008). We factor the non-negative document-term matrix
/// ``X (D x V)`` as ``X ~ W H`` with ``W (D x K) >= 0`` and ``H (K x V) >= 0`` by
/// multiplicative updates. Two divergences are available through ``beta_loss``:
/// the squared Frobenius loss (default) and the generalized Kullback-Leibler
/// divergence. The reference implementation is scikit-learn's
/// ``sklearn.decomposition.NMF`` (BSD-3-Clause); we read it to match the
/// initialization and update detail. The topic-word matrix is each row of ``H``
/// normalized to sum 1, and the document-topic matrix is each row of ``W``
/// normalized to sum 1.
#[pyclass(module = "topica")]
pub struct NMF {
    num_topics: usize,
    beta_loss: nmf::BetaLoss,
    init: nmf::Init,
    weighting_tfidf: bool,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<nmf::NmfModel>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted NMF.
#[derive(serde::Serialize, serde::Deserialize)]
struct NmfState {
    num_topics: usize,
    beta_loss: u8,
    init: u8,
    weighting_tfidf: bool,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    h: Option<Vec<Vec<f64>>>,
    w: Option<Vec<Vec<f64>>>,
    reconstruction_error: Option<f64>,
    error_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
}

fn parse_beta_loss(s: &str) -> PyResult<nmf::BetaLoss> {
    match s.to_ascii_lowercase().as_str() {
        "frobenius" => Ok(nmf::BetaLoss::Frobenius),
        "kullback-leibler" | "kl" => Ok(nmf::BetaLoss::KullbackLeibler),
        other => Err(PyValueError::new_err(format!(
            "beta_loss must be 'frobenius' or 'kullback-leibler' (got {other:?})"
        ))),
    }
}

fn parse_nmf_init(s: &str) -> PyResult<nmf::Init> {
    match s.to_ascii_lowercase().as_str() {
        "nndsvd" => Ok(nmf::Init::Nndsvd),
        "random" => Ok(nmf::Init::Random),
        other => Err(PyValueError::new_err(format!(
            "init must be 'nndsvd' or 'random' (got {other:?})"
        ))),
    }
}

fn parse_weighting(s: &str) -> PyResult<bool> {
    match s.to_ascii_lowercase().as_str() {
        "count" => Ok(false),
        "tfidf" | "tf-idf" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "weighting must be 'count' or 'tfidf' (got {other:?})"
        ))),
    }
}

impl NMF {
    fn fitted_model(&self) -> PyResult<&nmf::NmfModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl NMF {
    /// Create an unfitted model. `num_topics` is K (2 <= K <= vocabulary size).
    /// `beta_loss` is `"frobenius"` (default) or `"kullback-leibler"` (alias
    /// `"kl"`). `init` is `"nndsvd"` (default, deterministic SVD-based init) or
    /// `"random"` (seeded by `seed`). `weighting` is `"count"` (default, raw term
    /// counts) or `"tfidf"`. `convergence_tol` stops early on the relative
    /// reconstruction-error decrease. `seed` affects only `init="random"`.
    #[new]
    #[pyo3(signature = (num_topics, *, beta_loss="frobenius", init="nndsvd",
                        weighting="count", convergence_tol=1e-4, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        beta_loss: &str,
        init: &str,
        weighting: &str,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        Ok(NMF {
            num_topics,
            beta_loss: parse_beta_loss(beta_loss)?,
            init: parse_nmf_init(init)?,
            weighting_tfidf: parse_weighting(weighting)?,
            convergence_tol,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists). `iters` is the maximum
    /// number of multiplicative-update iterations (default 200). `convergence_tol`
    /// overrides the constructor value for this run (when given).
    #[pyo3(signature = (data, *, iters=None, convergence_tol=None))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: Option<usize>, convergence_tol: Option<f64>) -> PyResult<()> {
        let tol = convergence_tol.unwrap_or(self.convergence_tol);
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        if num_types < self.num_topics {
            return Err(PyValueError::new_err("vocabulary must have at least num_topics words"));
        }
        let it = iters.unwrap_or(200);
        let (k, bl, ini, tfidf, seed) =
            (self.num_topics, self.beta_loss, self.init, self.weighting_tfidf, self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = nmf::fit_nmf(&corpus.docs, k, num_types, bl, ini, tfidf, it, tol, seed);
            (m, corpus)
        });
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix (num_topics, vocab); each row is H normalized to sum 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word()).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics); each row is W normalized to sum 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// The reconstruction error at the final iteration (Frobenius loss or KL
    /// divergence, depending on `beta_loss`).
    #[getter]
    fn reconstruction_error(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.reconstruction_error)
    }
    /// Per-iteration reconstruction-error trajectory (the initial error first).
    #[getter]
    fn error_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.error_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Uniform convergence trace: `(iter, reconstruction_error)` pairs. The first
    /// entry (`iter = 1`) is the initial error before any update.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?
            .error_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
            .collect())
    }
    #[getter]
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.iters_run)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        topic_words_helper(py, &phi, &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(path, MODEL_TAG_NMF, &NmfState {
            num_topics: self.num_topics,
            beta_loss: match self.beta_loss {
                nmf::BetaLoss::Frobenius => 0,
                nmf::BetaLoss::KullbackLeibler => 1,
            },
            init: match self.init {
                nmf::Init::Nndsvd => 0,
                nmf::Init::Random => 1,
            },
            weighting_tfidf: self.weighting_tfidf,
            convergence_tol: self.convergence_tol,
            seed: self.seed,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            corpus: self.corpus.clone(),
            num_types: Some(m.num_types),
            topic_word: Some(m.topic_word.clone()),
            doc_topic: Some(m.doc_topic.clone()),
            h: Some(m.h.clone()),
            w: Some(m.w.clone()),
            reconstruction_error: Some(m.reconstruction_error),
            error_history: Some(m.error_history.clone()),
            converged: Some(m.converged),
            iters_run: Some(m.iters_run),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: NmfState = read_state(path, MODEL_TAG_NMF)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(nmf::NmfModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                h: s.h.unwrap_or_default(),
                w: s.w.unwrap_or_default(),
                reconstruction_error: s.reconstruction_error.unwrap_or(f64::NAN),
                error_history: s.error_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
            })
        } else {
            None
        };
        let beta_loss = if s.beta_loss == 1 {
            nmf::BetaLoss::KullbackLeibler
        } else {
            nmf::BetaLoss::Frobenius
        };
        let init = if s.init == 1 { nmf::Init::Random } else { nmf::Init::Nndsvd };
        Ok(NMF {
            num_topics: s.num_topics,
            beta_loss,
            init,
            weighting_tfidf: s.weighting_tfidf,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("NMF(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

/// LSA / LSI, latent semantic analysis (Deerwester et al. 1990; the randomized
/// truncated SVD follows Halko et al. 2011). We take a truncated SVD of the
/// weighted document-term matrix ``X (D x V) ~ U_k Sigma_k V_k^T``. The reference
/// implementation we validate against is ``sklearn.decomposition.TruncatedSVD``
/// (BSD-3-Clause).
///
/// Unlike topica's probabilistic models, LSA outputs are SIGNED latent
/// coordinates, not probabilities. ``topic_word (K x V)`` is the right singular
/// vectors ``V_k`` (signed term loadings; ``top_words`` ranks by absolute value).
/// ``doc_topic (D x K)`` is ``U_k Sigma_k`` (signed document coordinates; rows do
/// not sum to 1). ``singular_values (K)`` is ``Sigma_k``. A deterministic
/// ``svd_flip`` sign convention (largest-magnitude entry of each right singular
/// vector made positive) matches scikit-learn's output.
#[pyclass(module = "topica")]
pub struct LSA {
    num_topics: usize,
    weighting_tfidf: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<lsa::LsaModel>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted LSA.
#[derive(serde::Serialize, serde::Deserialize)]
struct LsaState {
    num_topics: usize,
    weighting_tfidf: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    singular_values: Option<Vec<f64>>,
}

impl LSA {
    fn fitted_model(&self) -> PyResult<&lsa::LsaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl LSA {
    /// Create an unfitted model. `num_topics` is K (2 <= K <= min(num_docs,
    /// vocabulary size)). `weighting` is `"tfidf"` (default, classic LSI) or
    /// `"count"` (raw term counts). `seed` seeds the randomized-SVD sketch.
    #[new]
    #[pyo3(signature = (num_topics, *, weighting="tfidf", seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        weighting: &str,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        Ok(LSA {
            num_topics,
            weighting_tfidf: parse_weighting(weighting)?,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists). The SVD is a direct solve,
    /// so there is no `iters` argument.
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        // K must not exceed min(num_docs, vocabulary size): the truncated SVD has
        // at most min(D, V) nonzero singular triplets.
        let max_k = corpus.num_docs().min(num_types);
        if self.num_topics > max_k {
            return Err(PyValueError::new_err(format!(
                "num_topics ({}) must be <= min(num_docs, vocab) = {}",
                self.num_topics, max_k
            )));
        }
        let (k, tfidf, seed) = (self.num_topics, self.weighting_tfidf, self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = lsa::fit_lsa(&corpus.docs, k, num_types, tfidf, seed);
            (m, corpus)
        });
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix (num_topics, vocab): the signed right singular vectors
    /// ``V_k``. These are term loadings, NOT probabilities (rows are not a
    /// simplex).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word()).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics): the signed document
    /// coordinates ``U_k Sigma_k``. Rows do NOT sum to 1 (LSA is not
    /// mixed-membership).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// The truncated singular values ``Sigma_k`` (length num_topics), the energy
    /// of each component (non-increasing, non-negative).
    #[getter]
    fn singular_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.singular_values.clone()).to_pyarray_bound(py))
    }
    /// No iterative trace: the SVD is a direct solve. Returns an empty list to
    /// keep the uniform fitted surface.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.fitted_model()?;
        Ok(Vec::new())
    }
    /// `None`: "converged" is not meaningful for a one-shot SVD.
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        self.fitted_model()?;
        Ok(None)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    /// Top-`n` words per topic, ranked by ABSOLUTE loading (a large negative
    /// loading is as defining of a component as a large positive one). Each entry
    /// is `(word, signed_loading)`.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word());
        // Rank by |loading| via an abs-valued matrix, but report the SIGNED value.
        let absphi = phi.mapv(f64::abs);
        let tops = top_word_ids_phi(&absphi, self.num_topics, n);
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }
    /// Topic coherence (UMass) computed on the ABSOLUTE top-loading words per
    /// component. Read with the caveat that LSA loadings are signed, not P(w|t).
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let absphi = phi.mapv(f64::abs);
        let tops = top_word_ids_phi(&absphi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(path, MODEL_TAG_LSA, &LsaState {
            num_topics: self.num_topics,
            weighting_tfidf: self.weighting_tfidf,
            seed: self.seed,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            corpus: self.corpus.clone(),
            num_types: Some(m.num_types),
            topic_word: Some(m.topic_word.clone()),
            doc_topic: Some(m.doc_topic.clone()),
            singular_values: Some(m.singular_values.clone()),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: LsaState = read_state(path, MODEL_TAG_LSA)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(lsa::LsaModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                singular_values: s.singular_values.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(LSA {
            num_topics: s.num_topics,
            weighting_tfidf: s.weighting_tfidf,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("LSA(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

/// FASTopic (Wu et al. 2024): a topic model with no encoder and no neural
/// network. The topic proportions ``theta`` and the topic-word matrix ``beta`` are
/// read off two entropic optimal-transport plans between embedding sets. You bring
/// the document embeddings ``D``; topica learns the topic embeddings, the word
/// embeddings (in the same space), and the transport marginals, minimizing a
/// bag-of-words reconstruction plus the two transport costs. New documents are
/// mapped to topics by a distance-softmax over the fitted topic embeddings, so
/// ``transform`` needs only their embeddings.
///
/// No embedder of your own? `topica.llm_embed(texts, model=...)` builds the
/// matrix (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct FASTopic {
    num_topics: usize,
    lr: f64,
    dt_alpha: f64,
    tw_alpha: f64,
    theta_temp: f64,
    em_tol: f64,
    sinkhorn_iters: usize,
    sinkhorn_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<fastopic::FastopicModel>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted FASTopic.
#[derive(serde::Serialize, serde::Deserialize)]
struct FastopicState {
    num_topics: usize,
    lr: f64,
    dt_alpha: f64,
    tw_alpha: f64,
    theta_temp: f64,
    em_tol: f64,
    sinkhorn_iters: usize,
    sinkhorn_tol: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Fitted model fields
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    topic_embeddings: Option<Vec<Vec<f64>>>,
    word_embeddings: Option<Vec<Vec<f64>>>,
    train_doc_embeddings: Option<Vec<Vec<f64>>>,
    loss_history: Option<Vec<f64>>,
    converged: Option<bool>,
    epochs_run: Option<usize>,
}

impl FASTopic {
    fn fitted_model(&self) -> PyResult<&fastopic::FastopicModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl FASTopic {
    /// Create an unfitted model. `lr` drives the full-batch Adam optimizer;
    /// `dt_alpha`/`tw_alpha` are the inverse entropic regularizations for the
    /// doc-topic and topic-word transport (reference defaults 3.0 and 2.0);
    /// `theta_temp` is the inference temperature; `convergence_tol` stops on the relative
    /// loss change. `sinkhorn_iters`/`sinkhorn_tol` cap each Sinkhorn solve.
    /// Pass `iters` to :meth:`fit` to set the number of training epochs.
    #[new]
    #[pyo3(signature = (num_topics, *, lr=0.002, dt_alpha=3.0, tw_alpha=2.0,
                        theta_temp=1.0, convergence_tol=1e-6, sinkhorn_iters=50, sinkhorn_tol=1e-4, seed=42, em_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        lr: f64,
        dt_alpha: f64,
        tw_alpha: f64,
        theta_temp: f64,
        convergence_tol: f64,
        sinkhorn_iters: usize,
        sinkhorn_tol: f64,
        seed: u64,
        em_tol: Option<f64>,
    ) -> PyResult<Self> {
        let convergence_tol = if let Some(old_val) = em_tol {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "FASTopic(em_tol=) is deprecated; use convergence_tol= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (convergence_tol - 1e-6_f64).abs() > f64::EPSILON { convergence_tol } else { old_val }
        } else {
            convergence_tol
        };
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if theta_temp <= 0.0 {
            return Err(PyValueError::new_err("theta_temp must be > 0"));
        }
        Ok(FASTopic {
            num_topics,
            lr,
            dt_alpha,
            tw_alpha,
            theta_temp,
            em_tol: convergence_tol,
            sinkhorn_iters,
            sinkhorn_tol,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            id_to_word: Vec::new(),
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`), one frozen row per document. The vocabulary is taken from
    /// the corpus; FASTopic learns the word embeddings itself, so none are passed.
    /// `iters` sets the number of training epochs (default 200).
    /// `convergence_tol` overrides the constructor value for this run (when given).
    #[pyo3(signature = (data, doc_embeddings, *, iters=None, convergence_tol=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
        let tol = convergence_tol.unwrap_or(self.em_tol);
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let num_types = corpus.num_types();
        if num_types < self.num_topics {
            return Err(PyValueError::new_err("vocabulary must have at least num_topics words"));
        }
        self.id_to_word = corpus.id_to_word.clone();
        let docs_ids = corpus.docs.clone();
        let ep = iters.unwrap_or(200);

        let (k, lr, dta, twa, tt, et, si, st) = (
            self.num_topics, self.lr, self.dt_alpha, self.tw_alpha,
            self.theta_temp, tol, self.sinkhorn_iters, self.sinkhorn_tol,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let model = py.allow_threads(move || {
            fastopic::fit_fastopic(
                &docs_ids, &doc_emb, k, num_types, ep, lr, dta, twa, tt, et, si, st, &mut rng,
            )
        });
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix beta (num_topics, vocab), each row a distribution.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Document-topic proportions theta (num_docs, num_topics).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Topic embeddings (num_topics, E), the learned topic points.
    #[getter]
    fn topic_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_embeddings).to_pyarray_bound(py))
    }
    /// Word embeddings (vocab, E), learned in the document-embedding space.
    #[getter]
    fn word_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.word_embeddings).to_pyarray_bound(py))
    }
    /// The training loss at each epoch.
    #[getter]
    fn loss_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.loss_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Uniform convergence trace: ``(epoch, negative_loss)`` pairs. The
    /// objective is the negated OT loss (so higher = better), indexed from 1.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?
            .loss_history
            .iter()
            .enumerate()
            .map(|(i, &l)| (i + 1, -l))
            .collect())
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }
    /// Document names from the training corpus, in corpus order.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }
    /// UMass coherence for each topic's top-`n` words, over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Held-out topic proportions for new documents from their embeddings
    /// (`(n, E)`): the reference's distance-softmax over the fitted topic
    /// embeddings, normalized by the training documents. `data` is accepted but
    /// not used (for API consistency with the other embedding models);
    /// `doc_embeddings` is required. Returns `(n, num_topics)`.
    #[pyo3(signature = (data=None, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: Option<&Bound<'py, PyAny>>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = data;
        let de_obj = doc_embeddings.ok_or_else(|| {
            PyValueError::new_err("FASTopic.transform requires doc_embeddings")
        })?;
        let m = self.fitted_model()?;
        let doc_emb = parse_features(de_obj)?;
        Ok(vecs_to_arr2(&m.transform(&doc_emb)).to_pyarray_bound(py))
    }

    /// Fit, then return the document-topic proportions (`fit_transform`).
    #[pyo3(signature = (data, doc_embeddings))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fit(py, data, doc_embeddings, None, None)?;
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(path, MODEL_TAG_FASTOPIC, &FastopicState {
            num_topics: self.num_topics,
            lr: self.lr,
            dt_alpha: self.dt_alpha,
            tw_alpha: self.tw_alpha,
            theta_temp: self.theta_temp,
            em_tol: self.em_tol,
            sinkhorn_iters: self.sinkhorn_iters,
            sinkhorn_tol: self.sinkhorn_tol,
            seed: self.seed,
            fitted: self.fitted,
            topic_names: self.topic_names.clone(),
            id_to_word: self.id_to_word.clone(),
            corpus: self.corpus.clone(),
            topic_word: Some(m.topic_word.clone()),
            doc_topic: Some(m.doc_topic.clone()),
            topic_embeddings: Some(m.topic_embeddings.clone()),
            word_embeddings: Some(m.word_embeddings.clone()),
            train_doc_embeddings: Some(m.train_doc_embeddings.clone()),
            loss_history: Some(m.loss_history.clone()),
            converged: Some(m.converged),
            epochs_run: Some(m.epochs_run),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: FastopicState = read_state(path, MODEL_TAG_FASTOPIC)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(fastopic::FastopicModel {
                num_topics: s.num_topics,
                num_types: s.id_to_word.len(),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                topic_embeddings: s.topic_embeddings.unwrap_or_default(),
                word_embeddings: s.word_embeddings.unwrap_or_default(),
                train_doc_embeddings: s.train_doc_embeddings.unwrap_or_default(),
                theta_temp: s.theta_temp,
                loss_history: s.loss_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                epochs_run: s.epochs_run.unwrap_or(0),
            })
        } else { None };
        Ok(FASTopic {
            num_topics: s.num_topics,
            lr: s.lr,
            dt_alpha: s.dt_alpha,
            tw_alpha: s.tw_alpha,
            theta_temp: s.theta_temp,
            em_tol: s.em_tol,
            sinkhorn_iters: s.sinkhorn_iters,
            sinkhorn_tol: s.sinkhorn_tol,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            id_to_word: s.id_to_word,
            corpus: s.corpus,
            model,
        })
    }

    fn __repr__(&self) -> String {
        format!("FASTopic(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// KeyATM: keyword-assisted topic model (Eshima, Imai & Sasaki 2024)
// ---------------------------------------------------------------------------

/// Keyword-Assisted Topic Model (keyATM Base). Like LDA, but some topics carry a
/// researcher-supplied **keyword** list; a token in a keyword topic comes either
/// from a distribution over only that topic's keywords or from the topic's full
/// distribution. This anchors keyword topics to their keywords while still
/// learning the rest of the vocabulary. Faithful to keyATM/keyATM.
#[pyclass(module = "topica")]
pub struct KeyATM {
    key_names: Vec<String>,
    keywords: Vec<Vec<String>>,
    num_topics: usize,
    alpha: f64,
    beta: f64,
    beta_keyword: f64,
    gamma1: f64,
    gamma2: f64,
    seed: u64,
    estimate_alpha: bool,
    // Default thread count for fit(); can be overridden per-call via fit(num_threads=).
    num_threads: usize,
    // CVB0 deterministic collapsed-variational inference for the base model
    // (optional, non-R-parity; covariate/dynamic variants stay Gibbs-only).
    cvb0: bool,
    fitted: bool,
    topic_names: Vec<String>,
    keyword_rate: Vec<f64>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
    // Covariate model only: learned λ (K × F+1, intercept first) and column names.
    feature_effects: Option<Array2<f64>>,
    feature_names: Vec<String>,
    // Dynamic model only: the HMM state of each time segment (length T), the
    // smoothed prevalence per segment (T × K), the segment labels, and the
    // left-to-right transition matrix (S × S).
    time_state: Vec<usize>,
    time_prevalence: Option<Array2<f64>>,
    time_labels: Vec<String>,
    transition_matrix: Option<Array2<f64>>,
    // Convergence trace: (iteration, log-likelihood, perplexity) — keyATM's model_fit.
    log_likelihood_history: Vec<(usize, f64, f64)>,
    // Whether the Gibbs run early-stopped on convergence_tol (opt-in; false by default).
    converged: bool,
    // (iteration, alpha vector) and (iteration, pi vector) — plot_alpha / plot_pi.
    alpha_history: Vec<(usize, Vec<f64>)>,
    pi_history: Vec<(usize, Vec<f64>)>,
    // Base model: the estimated asymmetric document-topic Dirichlet prior α_k
    // (length K). None for the covariate model (which uses the DMR λ) and the
    // dynamic model (per-state α); the `alpha` getter then falls back to the
    // symmetric prior.
    alpha_vec: Option<Vec<f64>>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_topics), f32; None when
    // keep_theta_draws=False. Feeds composition_theta's cross-sweep uncertainty.
    theta_draws: Option<Array3<f32>>,
}

impl KeyATM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl KeyATM {
    /// Create an unfitted model. `keywords` is ``{topic_name: [words]}`` (the
    /// keyword topics, in order). `num_topics` (default = number of keyword
    /// topics) may be larger to add regular, no-keyword topics. `alpha` is the
    /// per-topic Dirichlet; it defaults to ``1 / num_topics``, matching R keyATM's
    /// base prior (this is the starting point when `estimate_alpha` is on).
    /// `beta`/`beta_keyword` are the regular and keyword topic-word smoothing, and
    /// `gamma1`/`gamma2` the Beta prior on the keyword-vs-regular switch.
    #[new]
    #[pyo3(signature = (keywords, *, num_topics=None, alpha=None, beta=0.01, beta_keyword=0.1, gamma1=1.0, gamma2=1.0, seed=42, estimate_alpha=true, sampler="sparse", num_threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        keywords: &Bound<'_, PyDict>,
        #[pyo3(from_py_with = "py_num_topics_opt")] num_topics: Option<usize>,
        alpha: Option<f64>,
        beta: f64,
        beta_keyword: f64,
        gamma1: f64,
        gamma2: f64,
        seed: u64,
        estimate_alpha: bool,
        sampler: &str,
        num_threads: usize,
    ) -> PyResult<Self> {
        let (names, words) = parse_seed_dict(keywords)?;
        let k = num_topics.unwrap_or(names.len());
        if k < names.len() {
            return Err(PyValueError::new_err(
                "num_topics must be >= the number of keyword topics",
            ));
        }
        if k < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        // Default to R keyATM's base prior 1/K.
        let alpha = alpha.unwrap_or(1.0 / k as f64);
        if !finite_pos(alpha) || !finite_pos(beta) || !finite_pos(beta_keyword) || !finite_pos(gamma1) || !finite_pos(gamma2) {
            return Err(PyValueError::new_err("alpha, beta, beta_keyword, gamma1, gamma2 must be > 0"));
        }
        let cvb0 = match sampler {
            "sparse" => false,
            "cvb0" | "cvb" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\" or \"cvb0\""
                )))
            }
        };
        Ok(KeyATM {
            key_names: names, keywords: words, num_topics: k, alpha, beta, beta_keyword,
            gamma1, gamma2, seed, estimate_alpha, num_threads: num_threads.max(1),
            cvb0, fitted: false, topic_names: Vec::new(),
            keyword_rate: Vec::new(), phi: None, theta: None, corpus: None,
            feature_effects: None, feature_names: Vec::new(),
            time_state: Vec::new(), time_prevalence: None, time_labels: Vec::new(),
            transition_matrix: None, log_likelihood_history: Vec::new(), converged: false,
            alpha_history: Vec::new(), pi_history: Vec::new(), alpha_vec: None,
            theta_draws: None,
        })
    }

    /// Weighted LDA — keyATM's ``weightedLDA``: a keyword-free model with no
    /// keyword topics, so it is plain LDA fit with keyATM's token weighting and
    /// estimated asymmetric α (collapsed Gibbs). Use it as the unsupervised
    /// baseline next to a keyword-assisted :class:`KeyATM`. `fit` it the same
    /// way (the `weights` argument controls the token weighting); the
    /// keyword-specific outputs (``keyword_rate``, ``pi_history``) are empty.
    #[staticmethod]
    #[pyo3(signature = (num_topics, *, alpha=0.1, beta=0.01, seed=42))]
    fn weighted_lda(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(KeyATM {
            key_names: Vec::new(), keywords: Vec::new(), num_topics, alpha, beta,
            beta_keyword: 0.1, gamma1: 1.0, gamma2: 1.0, seed, estimate_alpha: true,
            num_threads: 1,
            cvb0: false,
            fitted: false,
            topic_names: Vec::new(), keyword_rate: Vec::new(), phi: None, theta: None,
            corpus: None, feature_effects: None, feature_names: Vec::new(),
            time_state: Vec::new(), time_prevalence: None, time_labels: Vec::new(),
            transition_matrix: None, log_likelihood_history: Vec::new(), converged: false,
            alpha_history: Vec::new(), pi_history: Vec::new(), alpha_vec: None,
            theta_draws: None,
        })
    }

    /// Fit by collapsed Gibbs for `iters` sweeps. Keyword topics come first (in
    /// the order given), then any regular topics.
    ///
    /// Pass `covariates` (a ``(num_docs, F)`` array or list of float lists) for
    /// the **covariate** keyATM: the document-topic prior becomes a
    /// Dirichlet-multinomial regression, ``α_{d,k} = exp(x_d · λ_k)`` (an
    /// intercept is prepended). `feature_names` (length F) labels the columns;
    /// the learned `λ` is exposed as `feature_effects`. With no `covariates`,
    /// this is the base symmetric-α keyATM.
    ///
    /// Pass `times` (one value per document) for the **dynamic** keyATM: a
    /// Chib (1998) change-point HMM lets topic prevalence shift over time across
    /// `num_states` latent regimes. Documents are sorted by time internally;
    /// the smoothed prevalence path is exposed as `time_prevalence` (aligned with
    /// `time_labels`) and the per-segment regime as `time_state`. `times`
    /// and `covariates` are mutually exclusive. `timestamps=` is an accepted
    /// alias for `times=` (the canonical cross-model name, as in DTM).
    #[pyo3(signature = (data, *, iters=1500, covariates=None, feature_names=None,
                        times=None, timestamps=None, num_states=5, weights="information-theory",
                        num_threads=None, optimize_interval=50, burn_in=200, prior_variance=1.0,
                        lbfgs_iters=20, progress_interval=0, prior_offset=None,
                        keep_theta_draws=true, num_theta_draws=25, convergence_tol=0.0_f64,
                        report_interval=None, turbo_alpha_stride=1))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        covariates: Option<&Bound<'_, PyAny>>,
        feature_names: Option<Vec<String>>,
        times: Option<&Bound<'_, PyAny>>,
        timestamps: Option<&Bound<'_, PyAny>>,
        num_states: usize,
        weights: &str,
        num_threads: Option<usize>,
        optimize_interval: usize,
        burn_in: usize,
        prior_variance: f64,
        lbfgs_iters: usize,
        progress_interval: usize,
        prior_offset: Option<&Bound<'_, PyAny>>,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        report_interval: Option<usize>,
        turbo_alpha_stride: usize,
    ) -> PyResult<()> {
        if turbo_alpha_stride < 1 {
            return Err(PyValueError::new_err(
                "turbo_alpha_stride must be >= 1 (1 = exact; >1 = approximate, subsample documents in the alpha sampler)",
            ));
        }
        // `times` is the canonical cross-model name for a per-document time index
        // (as in DTM); `timestamps` is accepted as an alias. Exactly one.
        let timestamps: Option<&Bound<'_, PyAny>> = match (times, timestamps) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "KeyATM.fit: pass either times= or timestamps=, not both",
                ));
            }
            (Some(t), None) | (None, Some(t)) => Some(t),
            (None, None) => None,
        };
        let progress_interval = if let Some(old_val) = report_interval {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "KeyATM.fit(report_interval=) is deprecated; use progress_interval= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if progress_interval != 0 { progress_interval } else { old_val }
        } else {
            progress_interval
        };
        // num_threads: fit()-level value overrides the constructor default.
        let nthreads_fit = num_threads.unwrap_or(self.num_threads).max(1);
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_topics = self.num_topics;
        let num_types = corpus.num_types();
        // Thinned θ-draw retention schedule (issue #31), shared by all three fits.
        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, corpus.num_docs(), num_topics)?;
        // Warn about keywords absent from the (pruned) vocabulary: a "seeded"
        // topic whose keywords were all dropped was never actually seeded, and
        // pruning (rm_top / min_doc_freq) or a typo/stemming mismatch silently
        // causes it.
        {
            let vocab: HashSet<&str> = corpus.id_to_word.iter().map(|s| s.as_str()).collect();
            let mut notes: Vec<String> = Vec::new();
            for (name, words) in self.key_names.iter().zip(self.keywords.iter()) {
                let oov: Vec<&str> =
                    words.iter().map(|w| w.as_str()).filter(|w| !vocab.contains(w)).collect();
                if !oov.is_empty() {
                    notes.push(format!(
                        "'{}' ({} of {} not in vocabulary, ignored: {})",
                        name, oov.len(), words.len(), oov.join(", ")
                    ));
                }
            }
            if !notes.is_empty() {
                let warnings = py.import_bound("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!("KeyATM: some keywords were dropped — {}", notes.join("; ")),),
                )?;
            }
        }
        let keys = seed_word_ids(&self.keywords, &corpus.id_to_word, num_topics);
        let (alpha, beta, beta_key, g1, g2) =
            (self.alpha, self.beta, self.beta_keyword, self.gamma1, self.gamma2);
        let estimate_alpha = self.estimate_alpha;
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let nthreads = nthreads_fit;
        let weight_scheme = match weights {
            "information-theory" | "info" => keyatm::WeightScheme::InfoTheory,
            "inv-freq" | "inverse-frequency" => keyatm::WeightScheme::InvFreq,
            "none" => keyatm::WeightScheme::None,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown weights={other:?}; expected 'information-theory', 'inv-freq', or 'none'"
                )))
            }
        };
        // Convergence trace cadence (keyATM's model_fit). 0 = auto: ~50 evenly
        // spaced points across the run.
        let ll_interval = if progress_interval == 0 {
            (iters / 50).max(1)
        } else {
            progress_interval
        };

        // The CVB0 backend covers only the base model.
        if self.cvb0 && (timestamps.is_some() || covariates.is_some() || prior_offset.is_some()) {
            return Err(PyValueError::new_err(
                "sampler=\"cvb0\" supports only the base keyATM (no timestamps, covariates, or prior_offset)",
            ));
        }
        // The turbo alpha subsampling only applies to the base model's α
        // slice-sampler; the covariate model learns λ and the dynamic model learns
        // per-state α, so it has no effect there.
        if turbo_alpha_stride > 1 && (timestamps.is_some() || covariates.is_some()) {
            return Err(PyValueError::new_err(
                "turbo_alpha_stride > 1 applies only to the base keyATM (not the covariate or dynamic model)",
            ));
        }
        let cvb0 = self.cvb0;

        // --- Dynamic model: timestamps drive a change-point HMM on prevalence. ---
        if let Some(ts) = timestamps {
            if covariates.is_some() {
                return Err(PyValueError::new_err(
                    "`timestamps` (dynamic) and `covariates` are mutually exclusive",
                ));
            }
            if prior_offset.is_some() {
                return Err(PyValueError::new_err(
                    "`prior_offset` (embedding anchor) is not supported with `timestamps`",
                ));
            }
            if num_states < 1 {
                return Err(PyValueError::new_err("num_states must be >= 1"));
            }
            let (time_raw, labels) = build_time_index(ts, corpus.num_docs())?;
            let num_time = labels.len();
            if num_time < num_states {
                return Err(PyValueError::new_err(format!(
                    "num_states ({num_states}) cannot exceed the number of distinct timestamps ({num_time})"
                )));
            }
            // keyATM requires documents ordered by time; sort, fit, then unsort θ.
            let mut order: Vec<usize> = (0..corpus.num_docs()).collect();
            order.sort_by_key(|&d| time_raw[d]);
            let sorted_docs: Vec<Vec<u32>> = order.iter().map(|&d| corpus.docs[d].clone()).collect();
            let sorted_time: Vec<usize> = order.iter().map(|&d| time_raw[d]).collect();

            let model = py.allow_threads(move || {
                keyatm::fit_keyatm_dynamic(
                    &sorted_docs, num_types, num_topics, &keys, &sorted_time, num_states,
                    beta, beta_key, g1, g2,
                    1.0, 1.0, 2.0, 1.0, // keyATM α-prior defaults: eta_1, eta_2, eta_1_reg, eta_2_reg
                    iters, ll_interval, weight_scheme, nthreads, draws_opts, convergence_tol, &mut rng,
                )
            });

            // θ comes back in sorted order; scatter it to the original doc order.
            let theta_sorted = model.doc_topic();
            let mut theta = vec![vec![0.0f64; num_topics]; corpus.num_docs()];
            for (i, &d) in order.iter().enumerate() {
                theta[d] = theta_sorted[i].clone();
            }
            self.theta = Some(vecs_to_arr2(&theta));
            // θ draws are also sorted; unsort their rows via `order` to match θ.
            self.theta_draws =
                draws_to_array3(&model.theta_draws, corpus.num_docs(), num_topics, Some(&order));
            self.phi = Some(vecs_to_arr2(&model.topic_word_all()));
            self.keyword_rate = model.keyword_rate();
            self.time_prevalence = model.time_prevalence().map(|tp| vecs_to_arr2(&tp));
            if let Some(d) = &model.dynamic {
                self.time_state = d.r_est.clone();
                self.transition_matrix = Some(vecs_to_arr2(&d.p_est));
            }
            self.log_likelihood_history = model.log_likelihood_history.clone();
            self.converged = model.converged;
            self.alpha_history = model.alpha_history.clone();
            self.pi_history = model.pi_history.clone();
            self.alpha_vec = model.alpha_vec.clone();
            self.time_labels = labels;

            let mut names = self.key_names.clone();
            for i in self.key_names.len()..num_topics {
                names.push(format!("topic_{}", i));
            }
            self.topic_names = names;
            self.corpus = Some(corpus);
            self.fitted = true;
            return Ok(());
        }

        // Build the (intercept-prepended) feature matrix if covariates were given.
        let (feats, cov_names): (Option<Vec<Vec<f64>>>, Vec<String>) = match covariates {
            Some(c) => {
                let raw = parse_features(c)?;
                if raw.len() != corpus.num_docs() {
                    return Err(PyValueError::new_err(format!(
                        "covariates has {} rows but corpus has {} documents",
                        raw.len(),
                        corpus.num_docs()
                    )));
                }
                check_all_finite_2d("covariates", &raw)?;
                let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
                if raw.iter().any(|r| r.len() != f_in) {
                    return Err(PyValueError::new_err("all covariate rows must have the same length"));
                }
                if let Some(n) = &feature_names {
                    if n.len() != f_in {
                        return Err(PyValueError::new_err(format!(
                            "feature_names has {} entries but covariates has {} columns",
                            n.len(), f_in
                        )));
                    }
                }
                let feats: Vec<Vec<f64>> = raw
                    .iter()
                    .map(|x| {
                        let mut v = Vec::with_capacity(f_in + 1);
                        v.push(1.0);
                        v.extend_from_slice(x);
                        v
                    })
                    .collect();
                let mut names = vec!["intercept".to_string()];
                names.extend(
                    feature_names.unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
                );
                (Some(feats), names)
            }
            None => (None, Vec::new()),
        };

        // Embedding anchor: a fixed (num_docs, num_topics) offset added inside the
        // DMR exponent. It needs the covariate (DMR) path, so when it is supplied
        // without covariates we synthesize an intercept-only design (the intercept
        // then learns each topic's baseline prevalence on top of the anchor).
        let offset: Option<Vec<Vec<f64>>> = match prior_offset {
            Some(o) => {
                let off = parse_features(o)?;
                if off.len() != corpus.num_docs() {
                    return Err(PyValueError::new_err(format!(
                        "prior_offset has {} rows but corpus has {} documents",
                        off.len(),
                        corpus.num_docs()
                    )));
                }
                if off.iter().any(|r| r.len() != num_topics) {
                    return Err(PyValueError::new_err(format!(
                        "prior_offset must have {num_topics} columns (one per topic)"
                    )));
                }
                check_all_finite_2d("prior_offset", &off)?;
                Some(off)
            }
            None => None,
        };
        let (feats, cov_names) = match (feats, &offset) {
            (None, Some(_)) => {
                let intercept = vec![vec![1.0f64]; corpus.num_docs()];
                (Some(intercept), vec!["intercept".to_string()])
            }
            (f, _) => (f, cov_names),
        };

        let (model, corpus) = py.allow_threads(move || {
            let m = match &feats {
                Some(f) => keyatm::fit_keyatm_cov(
                    &corpus.docs, num_types, num_topics, &keys, f, f[0].len(),
                    beta, beta_key, g1, g2, iters, optimize_interval, burn_in,
                    prior_variance, lbfgs_iters, ll_interval, weight_scheme, nthreads,
                    offset.as_deref(), draws_opts, convergence_tol, &mut rng,
                ),
                None if cvb0 => keyatm::fit_keyatm_cvb0(
                    &corpus.docs, num_types, num_topics, &keys, alpha, beta, beta_key, g1, g2,
                    iters, weight_scheme, &mut rng,
                ),
                None => keyatm::fit_keyatm(
                    &corpus.docs, num_types, num_topics, &keys, alpha, beta, beta_key, g1, g2,
                    iters, ll_interval, estimate_alpha, weight_scheme, nthreads, draws_opts, convergence_tol, turbo_alpha_stride, &mut rng,
                ),
            };
            (m, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word_all()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.theta_draws =
            draws_to_array3(&model.theta_draws, corpus.num_docs(), num_topics, None);
        self.keyword_rate = model.keyword_rate();
        self.log_likelihood_history = model.log_likelihood_history.clone();
        self.converged = model.converged;
        self.alpha_history = model.alpha_history.clone();
        self.pi_history = model.pi_history.clone();
        self.alpha_vec = model.alpha_vec.clone();
        if let Some(lam) = &model.lambda {
            self.feature_effects = Some(vecs_to_arr2(lam));
            self.feature_names = cov_names;
        }
        let mut names = self.key_names.clone();
        for i in self.key_names.len()..num_topics {
            names.push(format!("topic_{}", i));
        }
        self.topic_names = names;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Covariate model: learned DMR coefficients λ, shape ``(num_topics, F+1)``;
    /// column 0 is the intercept. Raises if the model was fit without covariates.
    #[getter]
    fn feature_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        self.feature_effects
            .as_ref()
            .map(|e| e.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without covariates"))
    }

    /// Covariate model: names aligned with `feature_effects` columns
    /// (``"intercept"`` first). Empty for the base model.
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }

    /// Dynamic model: smoothed topic prevalence per time segment, shape
    /// ``(T, num_topics)``, rows sum to 1, aligned with `time_labels`. Raises if
    /// the model was fit without `timestamps`.
    #[getter]
    fn time_prevalence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        self.time_prevalence
            .as_ref()
            .map(|t| t.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without timestamps"))
    }

    /// Dynamic model: the latent HMM state (regime) of each time segment, length
    /// T, aligned with `time_labels`. Empty for non-dynamic models.
    #[getter]
    fn time_state(&self) -> Vec<usize> {
        self.time_state.clone()
    }

    /// Dynamic model: the distinct, sorted timestamp labels, one per time
    /// segment (length T). Empty for non-dynamic models.
    #[getter]
    fn time_labels(&self) -> Vec<String> {
        self.time_labels.clone()
    }

    /// Dynamic model: the left-to-right state transition matrix, shape
    /// ``(num_states, num_states)``. Raises if fit without `timestamps`.
    #[getter]
    fn transition_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        self.transition_matrix
            .as_ref()
            .map(|t| t.to_pyarray_bound(py))
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without timestamps"))
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Thinned MCMC θ draws, shape ``(num_draws, num_docs, num_topics)``, or
    /// ``None`` when fit with ``keep_theta_draws=False``. Real cross-sweep
    /// posterior samples that :func:`topica.composition_theta` prefers over the
    /// within-document Dirichlet approximation.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }
    /// Per-document token counts (length D), in ``doc_topic`` row order, so
    /// ``composition_theta`` can recover N_d without re-threading the Corpus.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self
            .corpus
            .as_ref()
            .map(|c| c.docs.iter().map(|d| d.len()).collect())
            .unwrap_or_default())
    }
    /// Per-topic keyword switch rate ``π_k`` (the share of a keyword topic's mass
    /// drawn from its keyword distribution); 0 for regular topics.
    #[getter]
    fn keyword_rate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(self.keyword_rate.clone()).to_pyarray_bound(py))
    }

    /// The document-topic Dirichlet prior α, shape ``(num_topics,)``. For the base
    /// model this is the estimated asymmetric prior (R keyATM's ``alpha``); the
    /// covariate and dynamic models use a per-document prior, so this falls back to
    /// the symmetric base value. Marks keyATM as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let a = match &self.alpha_vec {
            Some(v) => v.clone(),
            None => vec![self.alpha; self.num_topics],
        };
        Ok(Array1::from(a).to_pyarray_bound(py))
    }

    /// Convergence trace as a list of ``(iteration, log_likelihood, perplexity)``
    /// triples — the three columns of keyATM's ``model_fit`` (``plot_modelfit``).
    /// ``log_likelihood`` is the collapsed marginal log-likelihood and
    /// ``perplexity`` is ``exp(-log_likelihood / total_weighted_tokens)``, both on
    /// R keyATM's scale. Sampled every ``report_interval`` sweeps during
    /// :meth:`fit` (auto ≈ 50 points). Empty if tracing was disabled.
    #[getter]
    fn log_likelihood_history(&self) -> PyResult<Vec<(usize, f64, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }

    /// Uniform convergence trace: ``(iteration, log_likelihood)`` pairs (the
    /// first two columns of :attr:`log_likelihood_history`; perplexity column
    /// dropped for cross-model uniformity).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history
            .iter()
            .map(|&(it, ll, _)| (it, ll))
            .collect())
    }

    /// ``True`` if the Gibbs run early-stopped because the relative change in the
    /// recorded ``model_fit`` log-likelihood fell below ``convergence_tol``;
    /// ``False`` when the full ``iters`` sweeps ran (the default, and always for
    /// the CVB0 backend, which keeps no trace).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Trace of the estimated document-topic prior α as ``(iteration, alpha)``
    /// pairs, where ``alpha`` is the length-K asymmetric prior at that sweep —
    /// keyATM's ``plot_alpha`` / ``values_iter$alpha_iter``. Base model only;
    /// empty for the covariate model (which traces λ) and dynamic model.
    #[getter]
    fn alpha_history(&self) -> PyResult<Vec<(usize, Vec<f64>)>> {
        self.require_fitted()?;
        Ok(self.alpha_history.clone())
    }

    /// Trace of the per-topic keyword switch rate π as ``(iteration, pi)`` pairs
    /// (``pi`` length K, 0 for regular topics) — keyATM's ``plot_pi`` /
    /// ``values_iter$pi_iter``. Empty for a keyword-free model.
    #[getter]
    fn pi_history(&self) -> PyResult<Vec<(usize, Vec<f64>)>> {
        self.require_fitted()?;
        Ok(self.pi_history.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// The keyword topic labels (then any regular topic labels). Settable after
    /// fit; length must equal ``num_topics``.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `KeyATM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_KEYATM, &KeyAtmState {
            num_topics: self.num_topics, alpha: self.alpha, beta: self.beta,
            beta_keyword: self.beta_keyword, gamma1: self.gamma1, gamma2: self.gamma2,
            seed: self.seed, fitted: self.fitted, topic_names: self.topic_names.clone(),
            keyword_rate: self.keyword_rate.clone(), phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
            alpha_history: self.alpha_history.clone(),
            pi_history: self.pi_history.clone(),
            alpha_vec: self.alpha_vec.clone(),
            num_threads: self.num_threads,
            theta_draws: arr3f32_opt(&self.theta_draws),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: KeyAtmState = read_state(path, MODEL_TAG_KEYATM)?;
        Ok(KeyATM {
            key_names: Vec::new(), keywords: Vec::new(), num_topics: s.num_topics,
            alpha: s.alpha, beta: s.beta, beta_keyword: s.beta_keyword, gamma1: s.gamma1,
            gamma2: s.gamma2, seed: s.seed, estimate_alpha: true, cvb0: false, fitted: s.fitted,
            topic_names: s.topic_names, num_threads: s.num_threads,
            keyword_rate: s.keyword_rate, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            corpus: s.corpus, feature_effects: None, feature_names: Vec::new(),
            time_state: Vec::new(), time_prevalence: None, time_labels: Vec::new(),
            transition_matrix: None, log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
            alpha_history: s.alpha_history, pi_history: s.pi_history,
            alpha_vec: s.alpha_vec, theta_draws: arr3f32_back(s.theta_draws),
        })
    }

    /// Infer document-topic distributions for new, unseen documents under the
    /// fitted model (sklearn-style ``transform``). Holds the fitted effective
    /// topic-word distributions fixed and runs collapsed Gibbs to infer θ for
    /// each document. Returns shape ``(num_new_docs, num_topics)`` with rows
    /// summing to 1.
    ///
    /// **Approximation:** held-out inference uses the fitted effective P(w |
    /// topic), which already marginalizes over the keyword switch, and the
    /// estimated asymmetric document-topic prior α (falling back to the
    /// symmetric base value when α was not estimated). The keyword switch
    /// variable is not re-estimated for new tokens.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let phi = self.phi.as_ref().unwrap();
        let alpha: Vec<f64> = match &self.alpha_vec {
            Some(v) => v.clone(),
            None => vec![self.alpha; self.num_topics],
        };
        transform_gibbs(py, data, id_to_word, phi, &alpha, iters, burn_in,
                        num_samples, sample_interval, seed.unwrap_or(self.seed))
    }

    fn __repr__(&self) -> String {
        format!("KeyATM(keyword_topics={}, num_topics={}, fitted={})", self.key_names.len(), self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// PA: Pachinko Allocation Model (super-/sub-topic hierarchy)
// ---------------------------------------------------------------------------

/// Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
/// super-topics over `num_sub` shared sub-topics over words, capturing topic
/// *correlations* — `super_sub` reports which sub-topics each super-topic groups
/// together. Collapsed Gibbs over (super, sub) pairs.
#[pyclass(module = "topica")]
pub struct PA {
    num_super: usize,
    num_sub: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,       // num_sub × V
    theta: Option<Array2<f64>>,     // num_docs × num_sub
    super_sub: Option<Array2<f64>>, // num_super × num_sub
    corpus: Option<corpus::Corpus>,
    // Thinned MCMC θ snapshots (num_draws, num_docs, num_sub), f32; None when
    // keep_theta_draws=False. Sub-topic proportions marginalized over super-topics.
    theta_draws: Option<Array3<f32>>,
    log_likelihood_history: Vec<(usize, f64)>,
    converged: bool,
}

impl PA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl PA {
    /// Create an unfitted model with `num_super` super-topics and `num_sub`
    /// sub-topics (the sub-topics are the word-level topics).
    #[new]
    #[pyo3(signature = (num_super, num_sub, *, alpha=0.1, beta=0.01, seed=42))]
    fn new(#[pyo3(from_py_with = "py_num_super")] num_super: usize, #[pyo3(from_py_with = "py_num_sub")] num_sub: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_super < 1 || num_sub < 2 {
            return Err(PyValueError::new_err("num_super must be >= 1 and num_sub >= 2"));
        }
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(PA {
            num_super, num_sub, alpha, beta, seed,
            fitted: false, topic_names: Vec::new(),
            phi: None, theta: None, super_sub: None, corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit by collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=1000, keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        let (s, k, a, b) = (self.num_super, self.num_sub, self.alpha, self.beta);

        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let (model, ll_history, converged_flag, corpus) = py.allow_threads(move || {
            let (m, hist, conv) = pa::fit_pam_with_draws(
                &corpus.docs, num_types, s, k, a, b, iters, draws_opts,
                convergence_tol, check_every, &mut rng,
            );
            (m, hist, conv, corpus)
        });
        self.theta_draws = draws_to_array3(&model.theta_draws, num_docs, k, None);
        self.phi = Some(vecs_to_arr2(&model.topic_word()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.super_sub = Some(vecs_to_arr2(&model.super_sub()));
        self.topic_names = (0..self.num_sub).map(|i| format!("topic_{i}")).collect();
        self.log_likelihood_history = ll_history;
        self.converged = converged_flag;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Sub-topic word distributions, shape ``(num_sub, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Document × sub-topic proportions, shape ``(num_docs, num_sub)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// The symmetric sub-topic Dirichlet prior α, broadcast to the columns of
    /// :attr:`doc_topic`, shape ``(num_sub,)``. Marks PA as a Dirichlet model for
    /// :func:`topica.effects.composition_theta`.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(vec![self.alpha; self.num_sub]).to_pyarray_bound(py))
    }
    /// Thinned MCMC θ snapshots, shape ``(num_draws, num_docs, num_sub)``,
    /// dtype ``float32``. ``None`` when fit with ``keep_theta_draws=False``.
    #[getter]
    fn theta_draws<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        self.theta_draws.as_ref().map(|a| a.to_pyarray_bound(py))
    }
    /// Number of tokens in each training document, shape ``(num_docs,)``.
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().map(|c| c.docs.iter().map(|d| d.len()).collect()).unwrap_or_default())
    }
    /// Super-topic → sub-topic association, shape ``(num_super, num_sub)``; row s
    /// shows which sub-topics super-topic s groups together (the correlations).
    #[getter]
    fn super_sub<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.super_sub.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn num_super(&self) -> usize {
        self.num_super
    }
    #[getter]
    fn num_sub(&self) -> usize {
        self.num_sub
    }
    /// Alias for `num_sub` (the word-level topics).
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_sub
    }
    /// Per-iteration log-likelihood trace. Returns one ``(iter, ll)`` pair for
    /// every ``check_every`` sweeps (empty when ``check_every=0``, the default).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self.log_likelihood_history.clone())
    }
    /// ``True`` if the relative-change convergence criterion was satisfied before
    /// all iterations completed. Always ``False`` when ``convergence_tol=0``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_sub {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_sub,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_sub, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_sub, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `PA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_PA, &PaState {
            num_super: self.num_super, num_sub: self.num_sub, alpha: self.alpha, beta: self.beta,
            seed: self.seed, fitted: self.fitted, phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), super_sub: arr2_opt(&self.super_sub),
            corpus: self.corpus.clone(), topic_names: self.topic_names.clone(),
            log_likelihood_history: self.log_likelihood_history.clone(),
            converged: self.converged,
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: PaState = read_state(path, MODEL_TAG_PA)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_sub).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(PA {
            num_super: s.num_super, num_sub: s.num_sub, alpha: s.alpha, beta: s.beta,
            seed: s.seed, fitted: s.fitted, topic_names,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            super_sub: arr2_back(s.super_sub), corpus: s.corpus,
            theta_draws: None,
            log_likelihood_history: s.log_likelihood_history,
            converged: s.converged,
        })
    }

    /// Infer sub-topic proportions for new, unseen documents under the fitted
    /// model (sklearn-style ``transform``). Holds the fitted sub-topic–word
    /// distributions fixed and runs collapsed Gibbs to infer θ over the
    /// ``num_sub`` sub-topics for each document. Returns shape
    /// ``(num_new_docs, num_sub)`` with rows summing to 1.
    ///
    /// **Approximation:** held-out inference projects directly onto the
    /// fitted sub-topics, marginalizing the super-topic layer. The
    /// super-topic assignments are a training-time device and are not
    /// re-estimated for new documents.
    #[pyo3(signature = (data, *, iters=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None, iterations=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iters: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
        iterations: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let iters = resolve_iters_deprecated(py, iters, iterations)?;
        self.require_fitted()?;
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let phi = self.phi.as_ref().unwrap();
        let alpha = vec![self.alpha; self.num_sub];
        transform_gibbs(py, data, id_to_word, phi, &alpha, iters, burn_in,
                        num_samples, sample_interval, seed.unwrap_or(self.seed))
    }

    fn __repr__(&self) -> String {
        format!("PA(num_super={}, num_sub={}, fitted={})", self.num_super, self.num_sub, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// HLDA: Hierarchical LDA (nested CRP topic tree)
// ---------------------------------------------------------------------------

/// Hierarchical LDA (Blei, Griffiths & Jordan): topics organized in a tree of
/// fixed `depth`, inferred by the nested Chinese Restaurant Process. The root is
/// the shared (general) topic; deeper nodes are progressively more specific.
/// Each document follows a root-to-leaf path. Inspect the tree with
/// `topic_word`/`node_levels`/`node_parents`/`doc_paths`.
#[pyclass(module = "topica")]
pub struct HLDA {
    depth: usize,
    gamma: f64,
    eta: f64,
    alpha: f64,
    seed: u64,
    fitted: bool,
    num_nodes: usize,
    topic_names: Vec<String>,
    node_topic_word: Option<Array2<f64>>, // num_nodes × V
    node_levels: Vec<usize>,
    node_parents: Vec<i64>, // -1 for the root
    doc_paths: Vec<Vec<usize>>,
    corpus: Option<corpus::Corpus>,
}

impl HLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl HLDA {
    /// Create an unfitted model. `depth` is the (fixed) tree depth; `gamma` is
    /// the nested-CRP concentration (larger => more child topics); `beta` the
    /// topic-word Dirichlet; `alpha` the per-document level distribution.
    #[new]
    #[pyo3(signature = (*, depth=3, gamma=1.0, beta=0.01, alpha=0.1, seed=42, eta=None))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = "py_depth")] depth: usize, gamma: f64, beta: f64, alpha: f64, seed: u64, eta: Option<f64>) -> PyResult<Self> {
        let beta = if let Some(old_val) = eta {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1("warn", (
                "HLDA(eta=) is deprecated; use beta= instead",
                py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                2_i32,
            ))?;
            if (beta - 0.01_f64).abs() > f64::EPSILON { beta } else { old_val }
        } else {
            beta
        };
        if depth < 2 {
            return Err(PyValueError::new_err("depth must be >= 2"));
        }
        if !finite_pos(gamma) || !finite_pos(beta) || !finite_pos(alpha) {
            return Err(PyValueError::new_err("gamma, beta, alpha must be > 0"));
        }
        Ok(HLDA {
            depth, gamma, eta: beta, alpha, seed,
            fitted: false, num_nodes: 0, topic_names: Vec::new(),
            node_topic_word: None,
            node_levels: Vec::new(), node_parents: Vec::new(), doc_paths: Vec::new(), corpus: None,
        })
    }

    /// Fit by nested-CRP collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=500))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (depth, gamma, eta, alpha) = (self.depth, self.gamma, self.eta, self.alpha);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = hlda::fit_hlda(&corpus.docs, num_types, depth, gamma, eta, alpha, iters, &mut rng);
            (m, corpus)
        });

        let nn = model.num_nodes();
        let mut tw = Array2::<f64>::zeros((nn, num_types));
        for i in 0..nn {
            for (w, &val) in model.topic_word(i).iter().enumerate() {
                tw[[i, w]] = val;
            }
        }
        self.num_nodes = nn;
        self.topic_names = (0..nn).map(|i| format!("topic_{i}")).collect();
        self.node_topic_word = Some(tw);
        self.node_levels = (0..nn).map(|i| model.node_level(i)).collect();
        self.node_parents = (0..nn).map(|i| model.node_parent(i).map(|p| p as i64).unwrap_or(-1)).collect();
        self.doc_paths = (0..corpus.num_docs()).map(|d| model.doc_path(d)).collect();
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// The number of topic nodes in the inferred tree.
    #[getter]
    fn num_nodes(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_nodes)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_nodes {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (num_nodes, got {})",
                self.num_nodes,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    /// Per-node word distributions, shape ``(num_nodes, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.node_topic_word.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// The tree level (0 = root) of each node, length ``num_nodes``.
    #[getter]
    fn node_levels(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.node_levels.clone())
    }
    /// The parent node id of each node (``-1`` for the root), length ``num_nodes``.
    #[getter]
    fn node_parents(&self) -> PyResult<Vec<i64>> {
        self.require_fitted()?;
        Ok(self.node_parents.clone())
    }
    /// Each document's root-to-leaf path (a list of node ids), length ``num_docs``.
    #[getter]
    fn doc_paths(&self) -> PyResult<Vec<Vec<usize>>> {
        self.require_fitted()?;
        Ok(self.doc_paths.clone())
    }
    /// The leaf node ids (nodes that are no node's parent).
    #[getter]
    fn leaves(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        let parents: HashSet<i64> = self.node_parents.iter().copied().collect();
        Ok((0..self.num_nodes).filter(|&i| !parents.contains(&(i as i64))).collect())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// HLDA has no per-iteration trace yet (part B); always returns ``[]``.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(Vec::new())
    }

    /// HLDA does not implement an early-stop criterion; always ``False``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(false)
    }

    /// Top `n` words for one topic node as ``(word, probability)`` pairs.
    #[pyo3(signature = (node, n=10))]
    fn top_words(&self, node: usize, n: usize) -> PyResult<Vec<(String, f64)>> {
        self.require_fitted()?;
        if node >= self.num_nodes {
            return Err(PyValueError::new_err("node out of range"));
        }
        let tw = self.node_topic_word.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let v = tw.shape()[1];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| f64::total_cmp(&tw[[node, b]], &tw[[node, a]]));
        Ok(idx.into_iter().take(n).map(|w| (vocab[w].clone(), tw[[node, w]])).collect())
    }

    /// Save the fitted model to `path`. Reload with `HLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, MODEL_TAG_HLDA, &HldaState {
            depth: self.depth, gamma: self.gamma, eta: self.eta, alpha: self.alpha,
            seed: self.seed, fitted: self.fitted, num_nodes: self.num_nodes,
            node_topic_word: arr2_opt(&self.node_topic_word), node_levels: self.node_levels.clone(),
            node_parents: self.node_parents.clone(), doc_paths: self.doc_paths.clone(),
            corpus: self.corpus.clone(), topic_names: self.topic_names.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: HldaState = read_state(path, MODEL_TAG_HLDA)?;
        let topic_names = if s.topic_names.is_empty() {
            (0..s.num_nodes).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(HLDA {
            depth: s.depth, gamma: s.gamma, eta: s.eta, alpha: s.alpha, seed: s.seed,
            fitted: s.fitted, num_nodes: s.num_nodes, topic_names,
            node_topic_word: arr2_back(s.node_topic_word),
            node_levels: s.node_levels, node_parents: s.node_parents, doc_paths: s.doc_paths,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!("HLDA(depth={}, num_nodes={}, fitted=true)", self.depth, self.num_nodes)
        } else {
            format!("HLDA(depth={}, fitted=false)", self.depth)
        }
    }
}

#[pymodule]
fn _topica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LDA>()?;
    m.add_class::<DMR>()?;
    m.add_class::<LabeledLDA>()?;
    m.add_class::<SAGE>()?;
    m.add_class::<CTM>()?;
    m.add_class::<STM>()?;
    m.add_class::<ECTM>()?;
    m.add_class::<STS>()?;
    m.add_class::<HDP>()?;
    m.add_class::<DTM>()?;
    m.add_class::<SupervisedLDA>()?;
    m.add_class::<PT>()?;
    m.add_class::<GSDMM>()?;
    m.add_class::<SeededLDA>()?;
    m.add_class::<KeyATM>()?;
    m.add_class::<Top2Vec>()?;
    m.add_class::<BERTopic>()?;
    m.add_class::<ETM>()?;
    m.add_class::<ProdLDA>()?;
    m.add_class::<InfoCTM>()?;
    m.add_class::<FASTopic>()?;
    m.add_class::<PA>()?;
    m.add_class::<HLDA>()?;
    m.add_class::<NMF>()?;
    m.add_class::<LSA>()?;
    m.add_class::<CombinedTM>()?;
    m.add_class::<ZeroShotTM>()?;
    m.add_class::<DETM>()?;
    m.add_class::<Corpus>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(window_cooccurrence, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;
    m.add("DEFAULT_TOKEN_REGEX", corpus::DEFAULT_TOKEN_REGEX)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    #[test]
    fn check_all_finite_2d_accepts_clean_data() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(check_all_finite_2d("x", &rows).is_ok());
    }

    #[test]
    fn check_all_finite_2d_rejects_nan() {
        let rows = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        let err = check_all_finite_2d("prevalence", &rows).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("prevalence"), "message should name the parameter");
        assert!(msg.contains("non-finite"), "message should mention non-finite");
        assert!(msg.contains("row 0"), "message should include row number");
    }

    #[test]
    fn check_all_finite_2d_rejects_inf() {
        let rows = vec![vec![0.0, f64::INFINITY]];
        assert!(check_all_finite_2d("features", &rows).is_err());
    }

    #[test]
    fn check_all_finite_1d_accepts_clean_data() {
        assert!(check_all_finite_1d("timestamps", &[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn check_all_finite_1d_rejects_nan() {
        let vals = vec![1.0, f64::NAN, 3.0];
        let err = check_all_finite_1d("timestamps", &vals).unwrap_err();
        assert!(err.to_string().contains("timestamps"));
    }

    #[test]
    fn check_all_finite_arr2_accepts_clean() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(check_all_finite_arr2("features", &arr.view()).is_ok());
    }

    #[test]
    fn check_all_finite_arr2_rejects_nan() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();
        let err = check_all_finite_arr2("features", &arr.view()).unwrap_err();
        assert!(err.to_string().contains("features"));
    }

    #[test]
    fn total_cmp_sort_with_nan_does_not_panic() {
        // A NaN in a float slice should sort without panicking when using total_cmp.
        let data = vec![3.0f64, f64::NAN, 1.0, 2.0];
        let mut idx: Vec<usize> = (0..data.len()).collect();
        // Descending sort: using total_cmp equivalent pattern.
        idx.sort_by(|&a, &b| f64::total_cmp(&data[b], &data[a]));
        // NaN is larger than any finite value under total_cmp, so it sorts to position 0.
        assert_eq!(idx[0], 1); // NaN is "greatest" under total_cmp
    }
}
