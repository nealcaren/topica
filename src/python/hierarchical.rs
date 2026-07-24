//! Python bindings for the hierarchy models: Pachinko Allocation (PA) and
//! hierarchical LDA (HLDA). Extracted from `mod.rs` (issue #385); shared helpers,
//! type aliases, and the save-format tags stay in `mod.rs` and are reached via
//! `use super::*`. No public API change: both classes are still registered in the
//! `#[pymodule]` fn and imported as `topica.PA` / `topica.HLDA`.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(serde::Serialize, serde::Deserialize)]
struct PaState {
    num_super: usize,
    num_sub: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    phi: Option<Arr2>,
    theta: Option<Arr2>,
    super_sub: Option<Arr2>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)]
    topic_names: Vec<String>,
    #[serde(default)]
    log_likelihood_history: Vec<(usize, f64)>,
    #[serde(default)]
    converged: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct HldaState {
    depth: usize,
    gamma: f64,
    eta: f64,
    alpha: f64,
    seed: u64,
    fitted: bool,
    num_nodes: usize,
    node_topic_word: Option<Arr2>,
    node_levels: Vec<usize>,
    node_parents: Vec<i64>,
    doc_paths: Vec<Vec<usize>>,
    corpus: Option<corpus::Corpus>,
    #[serde(default)]
    topic_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// PA: Pachinko Allocation Model (super-/sub-topic hierarchy)
// ---------------------------------------------------------------------------

/// Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
/// super-topics over `num_sub` shared sub-topics over words, capturing topic
/// *correlations* — `super_sub` reports which sub-topics each super-topic groups
/// together. Collapsed Gibbs over (super, sub) pairs.
///
/// Behavioral differences from MALLET's PAM, worth knowing (#497): the default
/// `alpha = 0.1` is much smaller than MALLET's effective ~50/num_super super
/// prior, which (with the single-super-topic commitment at init) makes documents
/// commit hard to their initial super-topic early — the intended "give the super
/// layer something to specialize on" design, but a real difference. α_s is
/// re-estimated only over the final quarter of sweeps (vs MALLET's periodic
/// post-burn-in optimization), so short runs adapt little. `doc_topic` is the
/// doc->sub marginal, floored by `alpha` (the doc->super prior; there is no
/// canonical PAM doc->sub prior).
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
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }
}

#[pymethods]
impl PA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_super", self.num_super)?;
        d.set_item("num_sub", self.num_sub)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model with `num_super` super-topics and `num_sub`
    /// sub-topics (the sub-topics are the word-level topics).
    /// `alpha` is the symmetric Dirichlet prior on each document's distribution
    /// over super-topics; `beta` is the topic-word Dirichlet smoothing; `seed`
    /// seeds the Gibbs RNG.
    #[new]
    #[pyo3(signature = (num_super, num_sub, *, alpha=0.1, beta=0.01, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_super")] num_super: usize,
        #[pyo3(from_py_with = "py_num_sub")] num_sub: usize,
        alpha: f64,
        beta: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_super < 1 || num_sub < 2 {
            return Err(PyValueError::new_err(
                "num_super must be >= 1 and num_sub >= 2",
            ));
        }
        if !finite_pos(alpha) || !finite_pos(beta) {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(PA {
            num_super,
            num_sub,
            alpha,
            beta,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            phi: None,
            theta: None,
            super_sub: None,
            corpus: None,
            theta_draws: None,
            log_likelihood_history: Vec::new(),
            converged: false,
        })
    }

    /// Fit by collapsed Gibbs sampling for `iters` sweeps.
    /// `keep_theta_draws` (default True) retains `num_theta_draws` thinned MCMC θ
    /// snapshots in `theta_draws`, the cross-sweep posterior samples
    /// `composition_theta` prefers over the Dirichlet approximation; set it False to
    /// save memory.
    /// `convergence_tol` (default 0.0, disabled) enables opt-in early stopping: the
    /// run stops once the relative change in the recorded log-likelihood between the
    /// last two trace points, |ΔLL| / |LL|, falls below it, setting `converged`. The
    /// monitored quantity is the collapsed model-fit log-likelihood; the comparison
    /// window is the trace cadence (`check_every` / `progress_interval`), so a coarser
    /// cadence compares more widely spaced sweeps. This is a pragmatic early-stop
    /// heuristic on the log-likelihood trace, not a guarantee the Gibbs chain has
    /// mixed. `check_every` is how often, in sweeps, the log-likelihood is recorded
    /// and the `convergence_tol` test is applied.
    #[pyo3(signature = (data, *, iters=1000, keep_theta_draws=true, num_theta_draws=25,
                        convergence_tol=0.0_f64, check_every=10_usize))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        keep_theta_draws: bool,
        num_theta_draws: usize,
        convergence_tol: f64,
        check_every: usize,
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
        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        let (s, k, a, b) = (slf.num_super, slf.num_sub, slf.alpha, slf.beta);

        let draws_opts = keyatm::ThetaDrawOpts::new(keep_theta_draws, num_theta_draws, iters);
        warn_theta_draw_memory(py, keep_theta_draws, num_theta_draws, num_docs, k)?;

        let mut rng = Pcg64Mcg::seed_from_u64(slf.seed);
        let (model, ll_history, converged_flag, corpus) = py.allow_threads(move || {
            let (m, hist, conv) = pa::fit_pam_with_draws(
                &corpus.docs,
                num_types,
                s,
                k,
                a,
                b,
                iters,
                draws_opts,
                convergence_tol,
                check_every,
                &mut rng,
            );
            (m, hist, conv, corpus)
        });
        slf.theta_draws = draws_to_array3(&model.theta_draws, num_docs, k, None);
        slf.phi = Some(vecs_to_arr2(&model.topic_word()));
        slf.theta = Some(vecs_to_arr2(&model.doc_topic()));
        slf.super_sub = Some(vecs_to_arr2(&model.super_sub()));
        slf.topic_names = (0..slf.num_sub).map(|i| format!("topic_{i}")).collect();
        slf.log_likelihood_history = ll_history;
        slf.converged = converged_flag;
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
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
        Ok(self
            .corpus
            .as_ref()
            .map(|c| c.docs.iter().map(|d| d.len()).collect())
            .unwrap_or_default())
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
        topic_words_helper(
            py,
            self.phi.as_ref().unwrap(),
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_sub,
            n,
            topic,
        )
    }
    /// UMass topic coherence per topic, shape ``(num_topics,)``. `n` is the number
    /// of top words per topic scored.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_sub, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path`. Reload with `PA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(
            path,
            MODEL_TAG_PA,
            &PaState {
                num_super: self.num_super,
                num_sub: self.num_sub,
                alpha: self.alpha,
                beta: self.beta,
                seed: self.seed,
                fitted: self.fitted,
                phi: arr2_opt(&self.phi),
                theta: arr2_opt(&self.theta),
                super_sub: arr2_opt(&self.super_sub),
                corpus: self.corpus.clone(),
                topic_names: self.topic_names.clone(),
                log_likelihood_history: self.log_likelihood_history.clone(),
                converged: self.converged,
            },
        )
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
            num_super: s.num_super,
            num_sub: s.num_sub,
            alpha: s.alpha,
            beta: s.beta,
            seed: s.seed,
            fitted: s.fitted,
            topic_names,
            phi: arr2_back(s.phi)?,
            theta: arr2_back(s.theta)?,
            super_sub: arr2_back(s.super_sub)?,
            corpus: s.corpus,
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
    ///
    /// The collapsed-Gibbs controls are per-document: `iters` sweeps each new
    /// document, discarding the first `burn_in`, then averaging `num_samples` θ
    /// snapshots taken `sample_interval` sweeps apart; `seed` seeds the inference
    /// RNG. `iterations` is a deprecated alias for `iters`.
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
        transform_gibbs(
            py,
            data,
            id_to_word,
            phi,
            &alpha,
            iters,
            burn_in,
            num_samples,
            sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PA(num_super={}, num_sub={}, fitted={})",
            self.num_super, self.num_sub, self.fitted
        )
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
///
/// Simplifications vs the hlda-c reference, worth knowing when comparing (#496):
/// the level prior is a symmetric Dirichlet `alpha`, not the GEM stick-breaking
/// prior, so there is no built-in bias of general words toward shallower levels
/// (recovery relies on the likelihood); `beta` is a single scalar topic-word
/// Dirichlet, not hlda-c's per-level (typically decreasing) vector, so deeper
/// topics are not sharpened by the prior; the hyperparameters are held fixed (no
/// per-sweep `eta`/`gamma`/GEM resampling); and the default `beta = 0.01` is
/// topica's own sharp calibration, below hlda-c's ~0.1-1.0 norm — expect more,
/// sparser nodes at the default.
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
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }
}

#[pymethods]
impl HLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``beta`` is the effective topic-word
    /// Dirichlet in force (the internal ``eta`` field, after resolving the
    /// deprecated ``eta=`` alias); ``eta`` is the deprecated alias and is not
    /// retained, so it always reports ``None``.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("depth", self.depth)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("beta", self.eta)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("seed", self.seed)?;
        d.set_item("eta", None::<f64>)?;
        Ok(d)
    }

    /// Create an unfitted model. `depth` is the (fixed) tree depth; `gamma` is
    /// the nested-CRP concentration (larger => more child topics); `beta` the
    /// topic-word Dirichlet; `alpha` the per-document level distribution.
    /// `seed` seeds the Gibbs RNG. `eta` is a deprecated alias for `beta` (kept
    /// for back-compat; pass `beta` instead).
    #[new]
    #[pyo3(signature = (*, depth=3, gamma=1.0, beta=0.01, alpha=0.1, seed=42, eta=None))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = "py_depth")] depth: usize,
        gamma: f64,
        beta: f64,
        alpha: f64,
        seed: u64,
        eta: Option<f64>,
    ) -> PyResult<Self> {
        let beta = if let Some(old_val) = eta {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "HLDA(eta=) is deprecated; use beta= instead",
                    py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                    2_i32,
                ),
            )?;
            if (beta - 0.01_f64).abs() > f64::EPSILON {
                beta
            } else {
                old_val
            }
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
            depth,
            gamma,
            eta: beta,
            alpha,
            seed,
            fitted: false,
            num_nodes: 0,
            topic_names: Vec::new(),
            node_topic_word: None,
            node_levels: Vec::new(),
            node_parents: Vec::new(),
            doc_paths: Vec::new(),
            corpus: None,
        })
    }

    /// Fit by nested-CRP collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=500))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
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
        let num_types = corpus.num_types();
        let (depth, gamma, eta, alpha) = (slf.depth, slf.gamma, slf.eta, slf.alpha);
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = hlda::fit_hlda(
                &corpus.docs,
                num_types,
                depth,
                gamma,
                eta,
                alpha,
                iters,
                &mut rng,
            );
            (m, corpus)
        });

        let nn = model.num_nodes();
        let mut tw = Array2::<f64>::zeros((nn, num_types));
        for i in 0..nn {
            for (w, &val) in model.topic_word(i).iter().enumerate() {
                tw[[i, w]] = val;
            }
        }
        slf.num_nodes = nn;
        slf.topic_names = (0..nn).map(|i| format!("topic_{i}")).collect();
        slf.node_topic_word = Some(tw);
        slf.node_levels = (0..nn).map(|i| model.node_level(i)).collect();
        slf.node_parents = (0..nn)
            .map(|i| model.node_parent(i).map(|p| p as i64).unwrap_or(-1))
            .collect();
        slf.doc_paths = (0..corpus.num_docs()).map(|d| model.doc_path(d)).collect();
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
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
        Ok((0..self.num_nodes)
            .filter(|&i| !parents.contains(&(i as i64)))
            .collect())
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
        Ok(idx
            .into_iter()
            .take(n)
            .map(|w| (vocab[w].clone(), tw[[node, w]]))
            .collect())
    }

    /// Save the fitted model to `path`. Reload with `HLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(
            path,
            MODEL_TAG_HLDA,
            &HldaState {
                depth: self.depth,
                gamma: self.gamma,
                eta: self.eta,
                alpha: self.alpha,
                seed: self.seed,
                fitted: self.fitted,
                num_nodes: self.num_nodes,
                node_topic_word: arr2_opt(&self.node_topic_word),
                node_levels: self.node_levels.clone(),
                node_parents: self.node_parents.clone(),
                doc_paths: self.doc_paths.clone(),
                corpus: self.corpus.clone(),
                topic_names: self.topic_names.clone(),
            },
        )
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
            depth: s.depth,
            gamma: s.gamma,
            eta: s.eta,
            alpha: s.alpha,
            seed: s.seed,
            fitted: s.fitted,
            num_nodes: s.num_nodes,
            topic_names,
            node_topic_word: arr2_back(s.node_topic_word)?,
            node_levels: s.node_levels,
            node_parents: s.node_parents,
            doc_paths: s.doc_paths,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "HLDA(depth={}, num_nodes={}, fitted=true)",
                self.depth, self.num_nodes
            )
        } else {
            format!("HLDA(depth={}, fitted=false)", self.depth)
        }
    }
}
