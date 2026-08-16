//! Python bindings for Factorial LDA (fLDA).
//!
//! Factorial LDA (Paul & Dredze, NIPS 2012) assigns each token a K-tuple — one
//! component from each of K factors (e.g. topic x sentiment) — rather than a single
//! topic. Structured log-linear priors tie tuples that share a component, and a
//! relaxed sparsity mask can switch whole tuples off. See `src/factorial_lda.rs`.

use super::*;
use crate::factorial_lda::{FactorialLdaConfig, OmegaPriors};
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyType;
use pyo3::types::{PyDict, PyList};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Factorial LDA: a sparse multi-dimensional topic model. Each token is drawn from
/// a K-tuple of latent factors (for example (topic, sentiment) or (topic,
/// perspective, focus)); structured word priors tie tuples that share a component,
/// and a sparsity prior can deactivate unsupported tuples.
///
/// `factor_sizes` gives the number of components per factor, e.g. `[20, 2]` for 20
/// topics x 2 sentiments; the model has `prod(factor_sizes)` tuples, which is
/// `num_topics`. Fit is Monte Carlo EM (collapsed Gibbs + gradient ascent on the
/// log-linear weights).
#[pyclass(module = "topica")]
pub struct FactorialLDA {
    factor_sizes: Vec<usize>,
    num_topics: usize, // prod(factor_sizes)
    sigma_alpha: f64,
    sigma_alpha_bias: f64,
    sigma_omega: f64,
    sigma_omega_bias: f64,
    delta0: f64,
    delta1: f64,
    alpha_bias_init: f64,
    omega_bias_init: f64,
    step_alpha_doc: f64,
    step_alpha_corpus: Option<f64>,
    step_alpha_bias: Option<f64>,
    step_omega: f64,
    step_omega_bias: Option<f64>,
    step_beta: f64,
    block_freq: usize,
    weight_burnin: usize,
    word_priors: bool,
    sparsity: bool,
    symmetric_word_prior: bool,
    seed: u64,
    fitted: bool,
    model: Option<crate::factorial_lda::FactorialLDAModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FldaState {
    factor_sizes: Vec<usize>,
    num_topics: usize,
    sigma_alpha: f64,
    sigma_alpha_bias: f64,
    sigma_omega: f64,
    sigma_omega_bias: f64,
    delta0: f64,
    delta1: f64,
    alpha_bias_init: f64,
    omega_bias_init: f64,
    step_alpha_doc: f64,
    step_alpha_corpus: Option<f64>,
    step_alpha_bias: Option<f64>,
    step_omega: f64,
    step_omega_bias: Option<f64>,
    step_beta: f64,
    block_freq: usize,
    weight_burnin: usize,
    word_priors: bool,
    sparsity: bool,
    symmetric_word_prior: bool,
    seed: u64,
    fitted: bool,
    corpus: Option<corpus::Corpus>,
    tuples: Option<Vec<Vec<usize>>>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    omega_b: Option<f64>,
    omega_w: Option<Vec<f64>>,
    omega_zw: Option<Vec<Vec<Vec<f64>>>>,
    tuple_activity: Option<Vec<f64>>,
    fit_history: Option<Vec<(usize, f64)>>,
}

impl FactorialLDA {
    fn fitted_model(&self) -> PyResult<&crate::factorial_lda::FactorialLDAModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn build_config(&self, iters: usize, samples: usize, eval_every: usize) -> FactorialLdaConfig {
        FactorialLdaConfig {
            factor_sizes: self.factor_sizes.clone(),
            iters,
            samples,
            sigma_alpha: self.sigma_alpha,
            sigma_alpha_bias: self.sigma_alpha_bias,
            sigma_omega: self.sigma_omega,
            sigma_omega_bias: self.sigma_omega_bias,
            delta0: self.delta0,
            delta1: self.delta1,
            alpha_bias_init: self.alpha_bias_init,
            omega_bias_init: self.omega_bias_init,
            step_alpha_doc: self.step_alpha_doc,
            step_alpha_corpus: self
                .step_alpha_corpus
                .unwrap_or(self.step_alpha_doc / 100.0),
            step_alpha_bias: self.step_alpha_bias.unwrap_or(self.step_alpha_doc / 100.0),
            step_omega: self.step_omega,
            step_omega_bias: self.step_omega_bias.unwrap_or(self.step_omega / 100.0),
            step_beta: self.step_beta,
            block_freq: self.block_freq,
            weight_burnin: self.weight_burnin,
            word_priors: self.word_priors,
            sparsity: self.sparsity,
            symmetric_word_prior: self.symmetric_word_prior,
            eval_every,
        }
    }
}

/// Parse the `omega_priors` dict into per-word / per-component mean arrays, mapping
/// words through the fitted corpus vocabulary. Accepts:
///   {"background": {word: weight} | [float; V],
///    "components": {(k, z): {word: weight} | [float; V]}}
fn parse_omega_priors(
    obj: &Bound<'_, PyAny>,
    vocab: &[String],
    factor_sizes: &[usize],
) -> PyResult<OmegaPriors> {
    let v = vocab.len();
    let index: HashMap<&str, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i))
        .collect();
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("omega_priors must be a dict"))?;

    let read_vec = |val: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
        if let Ok(map) = val.downcast::<PyDict>() {
            let mut arr = vec![0.0; v];
            for (kw, wv) in map.iter() {
                let word: String = kw.extract()?;
                let weight: f64 = wv.extract()?;
                if let Some(&wi) = index.get(word.as_str()) {
                    arr[wi] = weight;
                }
            }
            Ok(arr)
        } else {
            let arr: Vec<f64> = val.extract().map_err(|_| {
                PyValueError::new_err(
                    "omega_priors values must be a {word: float} dict or a length-V list",
                )
            })?;
            if arr.len() != v {
                return Err(PyValueError::new_err(format!(
                    "omega_priors array length {} != vocabulary size {v}",
                    arr.len()
                )));
            }
            Ok(arr)
        }
    };

    let mut eta_w = vec![0.0; v];
    let mut eta_zw: Vec<Vec<Vec<f64>>> = factor_sizes
        .iter()
        .map(|&zk| vec![vec![0.0; v]; zk])
        .collect();

    if let Some(bg) = dict.get_item("background")? {
        eta_w = read_vec(&bg)?;
    }
    if let Some(comps) = dict.get_item("components")? {
        let cmap = comps
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("omega_priors['components'] must be a dict"))?;
        for (key, val) in cmap.iter() {
            let (k, z): (usize, usize) = key.extract().map_err(|_| {
                PyValueError::new_err(
                    "omega_priors['components'] keys must be (factor, component) tuples",
                )
            })?;
            if k >= factor_sizes.len() || z >= factor_sizes[k] {
                return Err(PyValueError::new_err(format!(
                    "omega_priors component ({k}, {z}) is out of range for factor_sizes"
                )));
            }
            eta_zw[k][z] = read_vec(&val)?;
        }
    }
    Ok(OmegaPriors { eta_w, eta_zw })
}

/// Parse `observed_factors` into a `[num_docs][K]` matrix of optional labels.
/// Accepts `{factor_index: labels}` where `labels` is a length-`num_docs` sequence
/// of ints (the observed component for that document) or `None` (latent for that
/// document — semi-supervised). Only the named factors are constrained.
fn parse_observed_factors(
    obj: &Bound<'_, PyAny>,
    num_docs: usize,
    factor_sizes: &[usize],
) -> PyResult<Vec<Vec<Option<usize>>>> {
    let k = factor_sizes.len();
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("observed_factors must be a dict {factor: labels}"))?;
    let mut out = vec![vec![None; k]; num_docs];
    for (key, val) in dict.iter() {
        let kf: usize = key
            .extract()
            .map_err(|_| PyValueError::new_err("observed_factors keys must be factor indices"))?;
        if kf >= k {
            return Err(PyValueError::new_err(format!(
                "observed factor {kf} out of range (K = {k})"
            )));
        }
        let labels: Vec<Option<i64>> = val.extract().map_err(|_| {
            PyValueError::new_err(
                "observed_factors[factor] must be a sequence of ints or None, one per document",
            )
        })?;
        if labels.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "observed_factors[{kf}] has {} labels but the corpus has {num_docs} documents",
                labels.len()
            )));
        }
        for (d, lab) in labels.into_iter().enumerate() {
            if let Some(y) = lab {
                if y < 0 || y as usize >= factor_sizes[kf] {
                    return Err(PyValueError::new_err(format!(
                        "observed label {y} for factor {kf} is out of range [0, {})",
                        factor_sizes[kf]
                    )));
                }
                out[d][kf] = Some(y as usize);
            }
        }
    }
    Ok(out)
}

#[pymethods]
impl FactorialLDA {
    #[new]
    #[pyo3(signature = (
        factor_sizes,
        *,
        sigma_alpha=1.0,
        sigma_alpha_bias=1.0,
        sigma_omega=0.5,
        sigma_omega_bias=10.0,
        delta0=0.1,
        delta1=0.1,
        alpha_bias_init=-5.0,
        omega_bias_init=-5.0,
        step_alpha_doc=1e-2,
        step_alpha_corpus=None,
        step_alpha_bias=None,
        step_omega=1e-3,
        step_omega_bias=None,
        step_beta=1e-3,
        block_freq=1,
        weight_burnin=100,
        word_priors=true,
        sparsity=true,
        symmetric_word_prior=false,
        seed=13
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        factor_sizes: Vec<usize>,
        sigma_alpha: f64,
        sigma_alpha_bias: f64,
        sigma_omega: f64,
        sigma_omega_bias: f64,
        delta0: f64,
        delta1: f64,
        alpha_bias_init: f64,
        omega_bias_init: f64,
        step_alpha_doc: f64,
        step_alpha_corpus: Option<f64>,
        step_alpha_bias: Option<f64>,
        step_omega: f64,
        step_omega_bias: Option<f64>,
        step_beta: f64,
        block_freq: usize,
        weight_burnin: usize,
        word_priors: bool,
        sparsity: bool,
        symmetric_word_prior: bool,
        seed: u64,
    ) -> PyResult<Self> {
        if factor_sizes.is_empty() {
            return Err(PyValueError::new_err(
                "factor_sizes must list at least one factor",
            ));
        }
        if factor_sizes.iter().any(|&z| z < 1) {
            return Err(PyValueError::new_err(
                "every factor must have at least one component",
            ));
        }
        // Checked product so an enormous factor_sizes returns a clean error rather
        // than panicking on overflow (Gate B).
        let num_topics: usize = factor_sizes
            .iter()
            .try_fold(1usize, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| {
                PyValueError::new_err(
                    "factor_sizes product overflows usize; use fewer/smaller factors",
                )
            })?;
        if block_freq < 1 {
            return Err(PyValueError::new_err(
                "block_freq must be >= 1 (block-sample every k-th iteration; \
                 a large value approximates always-independent sampling)",
            ));
        }
        for (name, s) in [
            ("sigma_alpha", sigma_alpha),
            ("sigma_alpha_bias", sigma_alpha_bias),
            ("sigma_omega", sigma_omega),
            ("sigma_omega_bias", sigma_omega_bias),
        ] {
            if !(s.is_finite() && s > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and > 0"
                )));
            }
        }
        if !(delta0 > 0.0 && delta1 > 0.0) {
            return Err(PyValueError::new_err("delta0 and delta1 must be > 0"));
        }
        Ok(FactorialLDA {
            factor_sizes,
            num_topics,
            sigma_alpha,
            sigma_alpha_bias,
            sigma_omega,
            sigma_omega_bias,
            delta0,
            delta1,
            alpha_bias_init,
            omega_bias_init,
            step_alpha_doc,
            step_alpha_corpus,
            step_alpha_bias,
            step_omega,
            step_omega_bias,
            step_beta,
            block_freq,
            weight_burnin,
            word_priors,
            sparsity,
            symmetric_word_prior,
            seed,
            fitted: false,
            model: None,
            corpus: None,
        })
    }

    #[pyo3(signature = (data, *, iters=2000, samples=100, eval_every=0, omega_priors=None, observed_factors=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        samples: usize,
        eval_every: usize,
        omega_priors: Option<&Bound<'_, PyAny>>,
        observed_factors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(
                docs,
                None,
                None,
                std::collections::HashSet::new(),
                1,
                1.0,
                0,
                0,
            )?
            .0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        // Guard the sample-collection window: with iters=0 the Gibbs loop never
        // runs, no samples are collected, and topic_word/doc_topic would be all
        // zeros (rows not summing to 1) while reporting converged=True. Require at
        // least one sweep, at least one collected sample, and enough sweeps to
        // hold the requested sample tail.
        if iters < 1 {
            return Err(PyValueError::new_err("iters must be >= 1"));
        }
        if samples < 1 {
            return Err(PyValueError::new_err("samples must be >= 1"));
        }
        if samples > iters {
            return Err(PyValueError::new_err(
                "samples must be <= iters (samples are collected from the final sweeps)",
            ));
        }
        let priors = match omega_priors {
            Some(obj) => parse_omega_priors(obj, &corpus.id_to_word, &slf.factor_sizes)?,
            None => OmegaPriors::default(),
        };
        let observed = match observed_factors {
            Some(obj) => parse_observed_factors(obj, corpus.num_docs(), &slf.factor_sizes)?,
            None => Vec::new(),
        };
        let cfg = slf.build_config(iters, samples, eval_every);
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let model = crate::factorial_lda::fit_flda(&corpus, &cfg, &priors, &observed, &mut rng);
            (model, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("factor_sizes", self.factor_sizes.clone())?;
        d.set_item("sigma_alpha", self.sigma_alpha)?;
        d.set_item("sigma_alpha_bias", self.sigma_alpha_bias)?;
        d.set_item("sigma_omega", self.sigma_omega)?;
        d.set_item("sigma_omega_bias", self.sigma_omega_bias)?;
        d.set_item("delta0", self.delta0)?;
        d.set_item("delta1", self.delta1)?;
        d.set_item("alpha_bias_init", self.alpha_bias_init)?;
        d.set_item("omega_bias_init", self.omega_bias_init)?;
        d.set_item("step_alpha_doc", self.step_alpha_doc)?;
        d.set_item("step_alpha_corpus", self.step_alpha_corpus)?;
        d.set_item("step_alpha_bias", self.step_alpha_bias)?;
        d.set_item("step_omega", self.step_omega)?;
        d.set_item("step_omega_bias", self.step_omega_bias)?;
        d.set_item("step_beta", self.step_beta)?;
        d.set_item("block_freq", self.block_freq)?;
        d.set_item("weight_burnin", self.weight_burnin)?;
        d.set_item("word_priors", self.word_priors)?;
        d.set_item("sparsity", self.sparsity)?;
        d.set_item("symmetric_word_prior", self.symmetric_word_prior)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    // --- Required analysis surface ---
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn early_stopped(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    // --- fLDA-specific factorial surface ---
    /// Number of components per factor.
    #[getter]
    fn factor_sizes(&self) -> Vec<usize> {
        self.factor_sizes.clone()
    }
    /// Number of tuples (= num_topics = prod(factor_sizes)).
    #[getter]
    fn num_tuples(&self) -> usize {
        self.num_topics
    }
    /// Tuple index -> per-factor component vector, as a list of lists.
    #[getter]
    fn tuples(&self) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.fitted_model()?.tuples.clone())
    }
    /// Per-tuple activity b_x = sigma(beta_x) in (0, 1); a tuple is "inactive" when
    /// b_x <= 0.5. All ones when `sparsity=False`.
    #[getter]
    fn tuple_activity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.tuple_activity.clone()).to_pyarray_bound(py))
    }
    /// Corpus-wide word-prior bias omega_B.
    #[getter]
    fn omega_background(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.omega_b)
    }
    /// Background per-word weights omega_w (length V).
    #[getter]
    fn omega_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.omega_w.clone()).to_pyarray_bound(py))
    }
    /// Per-factor-component word weights for factor `k`, shape (Z_k, V). These are
    /// the "overview" weights that summarize each component (paper Fig. 3).
    fn factor_word<'py>(
        &self,
        py: Python<'py>,
        factor: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if factor >= m.omega_zw.len() {
            return Err(PyValueError::new_err(format!(
                "factor {factor} out of range (K = {})",
                m.omega_zw.len()
            )));
        }
        Ok(vecs_to_arr2(&m.omega_zw[factor]).to_pyarray_bound(py))
    }
    /// Top `n` words (by omega weight) for component `component` of factor `factor`.
    #[pyo3(signature = (factor, component, n=10))]
    fn factor_top_words<'py>(
        &self,
        py: Python<'py>,
        factor: usize,
        component: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let m = self.fitted_model()?;
        if factor >= m.omega_zw.len() || component >= m.omega_zw[factor].len() {
            return Err(PyValueError::new_err("factor/component out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let weights = &m.omega_zw[factor][component];
        let mut idx: Vec<usize> = (0..weights.len()).collect();
        idx.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap());
        let out = PyList::empty_bound(py);
        for &wi in idx.iter().take(n) {
            out.append((vocab[wi].clone(), weights[wi]))?;
        }
        Ok(out)
    }
    /// (iter, log-likelihood) trace. Empty unless `eval_every > 0`; always ends with
    /// the final-iteration value.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }

    // --- Conventional extras ---
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
            weights,
        )
    }
    /// Per-topic topic coherence, shape ``(num_topics,)``, aligned to topic index.
    /// Scores each topic's top-``n`` words. ``coherence_type`` selects the measure
    /// (``"u_mass"`` default, or ``"c_v"`` / ``"c_uci"`` / ``"c_npmi"``); ``texts``
    /// supplies the reference corpus for the windowed measures (defaults to the
    /// training corpus). Higher is more coherent (``u_mass`` is <= 0, nearer 0 is
    /// better; ``c_v`` in [0, 1]). Compare topics within one fit, not across corpora.
    #[pyo3(signature = (n=TopN(10), *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: TopN,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n = n.0;
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        coherence_dispatch(
            py,
            self.corpus.as_ref().unwrap(),
            &tops,
            n,
            &coherence_type,
            texts,
        )
    }

    /// Save the fitted model to `path`. Reload with `FactorialLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_FLDA,
            &FldaState {
                factor_sizes: self.factor_sizes.clone(),
                num_topics: self.num_topics,
                sigma_alpha: self.sigma_alpha,
                sigma_alpha_bias: self.sigma_alpha_bias,
                sigma_omega: self.sigma_omega,
                sigma_omega_bias: self.sigma_omega_bias,
                delta0: self.delta0,
                delta1: self.delta1,
                alpha_bias_init: self.alpha_bias_init,
                omega_bias_init: self.omega_bias_init,
                step_alpha_doc: self.step_alpha_doc,
                step_alpha_corpus: self.step_alpha_corpus,
                step_alpha_bias: self.step_alpha_bias,
                step_omega: self.step_omega,
                step_omega_bias: self.step_omega_bias,
                step_beta: self.step_beta,
                block_freq: self.block_freq,
                weight_burnin: self.weight_burnin,
                word_priors: self.word_priors,
                sparsity: self.sparsity,
                symmetric_word_prior: self.symmetric_word_prior,
                seed: self.seed,
                fitted: self.fitted,
                corpus: self.corpus.clone(),
                tuples: Some(m.tuples.clone()),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                omega_b: Some(m.omega_b),
                omega_w: Some(m.omega_w.clone()),
                omega_zw: Some(m.omega_zw.clone()),
                tuple_activity: Some(m.tuple_activity.clone()),
                fit_history: Some(m.fit_history.clone()),
            },
        )
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: FldaState = read_state(path, MODEL_TAG_FLDA)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(crate::factorial_lda::FactorialLDAModel {
                num_topics: s.num_topics,
                factor_sizes: s.factor_sizes.clone(),
                tuples: s.tuples.unwrap_or_default(),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                omega_b: s.omega_b.unwrap_or(0.0),
                omega_w: s.omega_w.unwrap_or_default(),
                omega_zw: s.omega_zw.unwrap_or_default(),
                tuple_activity: s.tuple_activity.unwrap_or_default(),
                fit_history: s.fit_history.unwrap_or_default(),
                converged: true,
            })
        } else {
            None
        };
        Ok(FactorialLDA {
            factor_sizes: s.factor_sizes,
            num_topics: s.num_topics,
            sigma_alpha: s.sigma_alpha,
            sigma_alpha_bias: s.sigma_alpha_bias,
            sigma_omega: s.sigma_omega,
            sigma_omega_bias: s.sigma_omega_bias,
            delta0: s.delta0,
            delta1: s.delta1,
            alpha_bias_init: s.alpha_bias_init,
            omega_bias_init: s.omega_bias_init,
            step_alpha_doc: s.step_alpha_doc,
            step_alpha_corpus: s.step_alpha_corpus,
            step_alpha_bias: s.step_alpha_bias,
            step_omega: s.step_omega,
            step_omega_bias: s.step_omega_bias,
            step_beta: s.step_beta,
            block_freq: s.block_freq,
            weight_burnin: s.weight_burnin,
            word_priors: s.word_priors,
            sparsity: s.sparsity,
            symmetric_word_prior: s.symmetric_word_prior,
            seed: s.seed,
            fitted: s.fitted,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FactorialLDA(factor_sizes={:?}, num_topics={}, fitted={})",
            self.factor_sizes, self.num_topics, self.fitted
        )
    }
}
