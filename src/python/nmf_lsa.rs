//! NMF and LSA pyclasses (matrix-factorization topic models, validated vs
//! scikit-learn). Carved out of python/mod.rs; `use super::*` pulls in the shared
//! bindings helpers (Corpus, build_corpus_from_docs, save/load, array adapters, …).

use super::*;
use pyo3::types::PyDict;

/// NMF, non-negative matrix factorization for topic modeling (Lee & Seung 2001;
/// Boutsidis & Gallopoulos 2008). We factor the non-negative document-term matrix
/// ``X (D x V)`` as ``X ~ W H`` with ``W (D x K) >= 0`` and ``H (K x V) >= 0`` by
/// multiplicative updates. Two divergences are available through ``beta_loss``:
/// the squared Frobenius loss (default) and the generalized Kullback-Leibler
/// divergence. The reference implementation is scikit-learn's
/// ``sklearn.decomposition.NMF`` (BSD-3-Clause); we read it to match the
/// initialization and the multiplicative-update formulas. The update order and
/// the convergence-check cadence differ from sklearn (see the notes in
/// ``nmf.rs``), so the guarantee is eventual close agreement of the fitted
/// factors, not iteration-for-iteration parity. The topic-word matrix is each row of ``H``
/// normalized to sum 1. The document-topic matrix weights each topic by its
/// ``H``-row mass before row-normalizing (``W_{d,k} * rowsum(H_k)``, then rows to
/// sum 1), so the reported proportion tracks each topic's share of the
/// reconstructed term mass. ``weighting`` builds ``X`` from topica's TF-IDF
/// (default, the classic NMF recipe) or raw counts (``weighting="count"``).
///
/// Constructor: ``NMF(num_topics, *, beta_loss="frobenius", init="nndsvd",
/// weighting="tfidf", convergence_tol=1e-4, seed=13)``. ``convergence_tol`` stops
/// early on the relative reconstruction-error decrease; ``0.0`` disables the
/// early-stop check, so the fit runs the full ``iters`` and reports
/// ``converged=False`` by design. ``seed`` affects only ``init="random"`` -- the
/// default ``init="nndsvd"`` is deterministic and ignores it, so a stability check
/// that varies only the seed under nndsvd compares identical fits; use
/// ``init="random"`` to vary across seeds.
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
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). Enum-ish params are reported under
    /// their public strings (``beta_loss``, ``init``, ``weighting``).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        let beta_loss = match self.beta_loss {
            nmf::BetaLoss::Frobenius => "frobenius",
            nmf::BetaLoss::KullbackLeibler => "kullback-leibler",
        };
        d.set_item("beta_loss", beta_loss)?;
        let init = match self.init {
            nmf::Init::Nndsvd => "nndsvd",
            nmf::Init::Random => "random",
        };
        d.set_item("init", init)?;
        let weighting = if self.weighting_tfidf {
            "tfidf"
        } else {
            "count"
        };
        d.set_item("weighting", weighting)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `num_topics` is K (2 <= K <= vocabulary size).
    /// `beta_loss` is `"frobenius"` (default) or `"kullback-leibler"` (alias
    /// `"kl"`). `init` is `"nndsvd"` (default, deterministic SVD-based init) or
    /// `"random"` (seeded by `seed`). The `"nndsvd"` init fills exact-zero entries
    /// of the SVD factors with the data mean (scikit-learn's NNDSVDa variant), so
    /// the initial factors are dense; it requires `num_topics <= min(num_documents,
    /// num_words)` (use `"random"` above that rank). `weighting` is `"tfidf"` (default)
    /// or `"count"`. `convergence_tol` stops early on the relative
    /// reconstruction-error decrease; `0.0` disables the early-stop check, so the
    /// fit runs the full `iters` and reports `converged=False` by design. `seed`
    /// affects only `init="random"` (the default `"nndsvd"` init is deterministic
    /// and ignores it).
    #[new]
    #[pyo3(signature = (num_topics, *, beta_loss="frobenius", init="nndsvd",
                        weighting="tfidf", convergence_tol=1e-4, seed=13))]
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
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
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
    /// overrides the constructor value for this run (when given). `num_threads` caps
    /// the worker pool for the parallel multiplicative-update matmuls (`None`/`0` =
    /// all cores); output is deterministic regardless of the worker count, so it
    /// controls only resource use, not results.
    #[pyo3(signature = (data, *, iters=None, convergence_tol=None, num_threads=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        ensure_finite_nonneg("convergence_tol", tol)?;
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
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        // NNDSVD-family init needs `num_topics` leading singular triplets, which
        // only exist up to rank min(num_documents, num_words). Above that,
        // `nndsvd_init` would index past the truncated SVD and panic. Reject it
        // here with a clear error (matching sklearn, which raises ValueError for
        // NNDSVD above min(n_samples, n_features)); `init="random"` has no such
        // rank limit, so this guard is init-specific (#448).
        if matches!(slf.init, nmf::Init::Nndsvd) {
            let max_rank = corpus.num_docs().min(num_types);
            if slf.num_topics > max_rank {
                return Err(PyValueError::new_err(format!(
                    "init=\"nndsvd\" requires num_topics <= min(num_documents, num_words) \
                     = {max_rank} (got num_topics={} with {} documents and {} words). \
                     Reduce num_topics, or use init=\"random\", which supports a larger rank.",
                    slf.num_topics,
                    corpus.num_docs(),
                    num_types,
                )));
            }
        }
        let it = iters.unwrap_or(200);
        let (k, bl, ini, tfidf, seed) = (
            slf.num_topics,
            slf.beta_loss,
            slf.init,
            slf.weighting_tfidf,
            slf.seed,
        );
        let (model, corpus) = py.allow_threads(move || {
            let m = run_with_threads(num_threads, || {
                nmf::fit_nmf(&corpus.docs, k, num_types, bl, ini, tfidf, it, tol, seed)
            });
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
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
    /// Document-topic matrix (num_docs, num_topics); each row is W with columns
    /// scaled by their H-row mass (W_{d,k} * rowsum(H_k)) and then normalized to
    /// sum 1, so the proportion reflects each topic's share of the reconstructed
    /// term mass.
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
        Ok(self
            .fitted_model()?
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
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
        )
    }
    /// Per-topic topic coherence, shape ``(num_topics,)``, aligned to topic index.
    /// Scores each topic's top-``n`` words. ``coherence_type`` selects the measure
    /// (``"u_mass"`` default, or ``"c_v"`` / ``"c_uci"`` / ``"c_npmi"``); ``texts``
    /// supplies the reference corpus for the windowed measures (defaults to the
    /// training corpus). Higher is more coherent (``u_mass`` is <= 0, nearer 0 is
    /// better; ``c_v`` in [0, 1]). Compare topics within one fit, not across corpora.
    #[pyo3(signature = (n=10, *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
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

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_NMF,
            &NmfState {
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
            },
        )
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
        let init = if s.init == 1 {
            nmf::Init::Random
        } else {
            nmf::Init::Nndsvd
        };
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
        format!(
            "NMF(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
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
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``weighting`` is reported under its
    /// public string.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        let weighting = if self.weighting_tfidf {
            "tfidf"
        } else {
            "count"
        };
        d.set_item("weighting", weighting)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `num_topics` is K (2 <= K <= min(num_docs,
    /// vocabulary size)). `weighting` is `"tfidf"` (default, classic LSI) or
    /// `"count"` (raw term counts). `seed` seeds the randomized-SVD sketch.
    #[new]
    #[pyo3(signature = (num_topics, *, weighting="tfidf", seed=13))]
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
    /// so there is no `iters` argument. `num_threads` caps the worker pool for the
    /// parallel matmuls in the truncated SVD (`None`/`0` = all cores); output is
    /// deterministic regardless of the worker count, so it controls only resource
    /// use, not results.
    #[pyo3(signature = (data, *, num_threads=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        num_threads: Option<usize>,
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
        // K must not exceed min(num_docs, vocabulary size): the truncated SVD has
        // at most min(D, V) nonzero singular triplets.
        let max_k = corpus.num_docs().min(num_types);
        if slf.num_topics > max_k {
            return Err(PyValueError::new_err(format!(
                "num_topics ({}) must be <= min(num_docs, vocab) = {}",
                slf.num_topics, max_k
            )));
        }
        let (k, tfidf, seed) = (slf.num_topics, slf.weighting_tfidf, slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = run_with_threads(num_threads, || {
                lsa::fit_lsa(&corpus.docs, k, num_types, tfidf, seed)
            });
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
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
    /// Per-topic topic coherence, shape ``(num_topics,)``, aligned to topic index.
    /// Scores each topic's top-``n`` words. ``coherence_type`` selects the measure
    /// (``"u_mass"`` default, or ``"c_v"`` / ``"c_uci"`` / ``"c_npmi"``); ``texts``
    /// supplies the reference corpus for the windowed measures (defaults to the
    /// training corpus). Higher is more coherent (``u_mass`` is <= 0, nearer 0 is
    /// better; ``c_v`` in [0, 1]). Compare topics within one fit, not across corpora.
    #[pyo3(signature = (n=10, *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let absphi = phi.mapv(f64::abs);
        let tops = top_word_ids_phi(&absphi, self.num_topics, n);
        coherence_dispatch(
            py,
            self.corpus.as_ref().unwrap(),
            &tops,
            n,
            &coherence_type,
            texts,
        )
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_LSA,
            &LsaState {
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
            },
        )
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
        format!(
            "LSA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
