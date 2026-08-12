//! OnlineLDA pyclass: online (stochastic) variational-Bayes Dirichlet LDA
//! (Hoffman, Blei & Bach 2010), the streaming counterpart of :class:`LDA` and the
//! analogue of gensim's `LdaModel`. `use super::*` pulls in the shared binding
//! helpers (Corpus, build_corpus_from_docs, save/load, array adapters, …); the
//! algorithm lives in `crate::online_lda`.

use super::*;
use crate::online_lda::{self, OnlineLDAModel};
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Online (streaming) variational-Bayes LDA — minibatch stochastic VB on the
/// Dirichlet LDA model (Hoffman, Blei & Bach, *NeurIPS* 2010). The direct
/// analogue of gensim's ``LdaModel``: it fits in minibatches with a decaying
/// learning rate ``rho_t = (tau + t)^(-kappa)`` without holding the whole corpus
/// in memory, and supports streaming :meth:`partial_fit` to fold new documents
/// into an already-fitted model.
///
/// Prefer this over the default batch-Gibbs :class:`LDA` for very large or
/// streaming corpora; for moderate corpora the batch samplers (or CVB0) usually
/// give better topics per unit compute.
///
/// Constructor arguments map onto gensim's ``LdaModel`` as: ``batch_size`` ↔
/// ``chunksize``, ``tau`` ↔ ``offset``, ``kappa`` ↔ ``decay``, ``beta`` ↔
/// ``eta``, ``inner_iters`` ↔ ``iterations``, and ``fit(iters=)`` ↔ ``passes``.
#[pyclass(module = "topica")]
pub struct OnlineLDA {
    num_topics: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    tau: f64,
    kappa: f64,
    batch_size: usize,
    inner_iters: usize,
    mean_change_tol: f64,
    total_docs: Option<f64>,
    seed: u64,

    fitted: bool,
    topic_names: Vec<String>,
    model: Option<OnlineLDAModel>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted OnlineLDA. The streaming state (`lambda`,
/// `updates`, the schedule hyperparameters) is persisted so a loaded model can
/// resume :meth:`partial_fit` on the exact same Robbins-Monro trajectory.
#[derive(serde::Serialize, serde::Deserialize)]
struct OnlineLdaState {
    num_topics: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    tau: f64,
    kappa: f64,
    batch_size: usize,
    inner_iters: usize,
    mean_change_tol: f64,
    total_docs: Option<f64>,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Fitted state.
    num_types: Option<usize>,
    alpha: Option<Vec<f64>>,
    eta: Option<f64>,
    lambda: Option<Vec<Vec<f64>>>,
    updates: Option<usize>,
    model_tau: Option<f64>,
    model_kappa: Option<f64>,
    model_batch_size: Option<usize>,
    model_inner_iters: Option<usize>,
    model_mean_change_tol: Option<f64>,
    model_total_docs: Option<f64>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    doc_lengths: Option<Vec<usize>>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
}

impl OnlineLDA {
    fn fitted_model(&self) -> PyResult<&OnlineLDAModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn fitted_model_mut(&mut self) -> PyResult<&mut OnlineLDAModel> {
        self.model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// The symmetric per-topic α implied by `alpha_sum` (default: 1.0 per topic).
    fn alpha_vec(&self) -> Vec<f64> {
        let sum = self.alpha_sum.unwrap_or(self.num_topics as f64);
        vec![sum / self.num_topics as f64; self.num_topics]
    }

    /// Map held-out documents (a `Corpus` or `list[list[str]]`) onto the trained
    /// vocabulary, dropping out-of-vocabulary tokens. Requires a fitted model
    /// (the vocabulary is fixed by the first `fit`, exactly as gensim fixes it via
    /// the `id2word` dictionary at construction).
    fn map_to_ids(&self, data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
        let trained = self
            .corpus
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        let index: HashMap<&str, u32> = trained
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
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
                PyValueError::new_err(
                    "expected a Corpus or a list of token lists (list[list[str]])",
                )
            })?
        };

        Ok(str_docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|tok| index.get(tok.as_str()).copied())
                    .collect()
            })
            .collect())
    }
}

#[pymethods]
impl OnlineLDA {
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
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha_sum", self.alpha_sum)?;
        d.set_item("beta", self.beta)?;
        d.set_item("tau", self.tau)?;
        d.set_item("kappa", self.kappa)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("inner_iters", self.inner_iters)?;
        d.set_item("mean_change_tol", self.mean_change_tol)?;
        d.set_item("total_docs", self.total_docs)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model.
    ///
    /// `alpha_sum` is the total document-topic Dirichlet mass (default:
    /// `num_topics`, i.e. 1.0 per topic); the symmetric per-topic α is
    /// `alpha_sum / num_topics`. `beta` is the symmetric topic-word Dirichlet
    /// prior (gensim's `eta`). `tau` (offset ≥ 0) down-weights early, noisy
    /// minibatches and `kappa` (decay, in (0.5, 1]) sets the forgetting rate of
    /// the learning schedule `rho_t = (tau + t)^(-kappa)`. `batch_size` is the
    /// minibatch size (gensim's `chunksize`). `inner_iters` caps the per-document
    /// E-step fixed-point iterations (gensim's `iterations`), stopping early once
    /// the mean change in γ falls below `mean_change_tol`. `total_docs` is the
    /// assumed corpus size D used for the D/|batch| gradient scaling — set it when
    /// streaming a corpus larger than the first `fit` batch (default: the `fit`
    /// corpus size). `seed` seeds the initial λ and the per-pass shuffle.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_sum=None, beta=0.01, tau=1.0, kappa=0.7,
                        batch_size=256, inner_iters=100, mean_change_tol=1e-3,
                        total_docs=None, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha_sum: Option<f64>,
        beta: f64,
        tau: f64,
        kappa: f64,
        batch_size: usize,
        inner_iters: usize,
        mean_change_tol: f64,
        total_docs: Option<f64>,
        seed: u64,
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
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(PyValueError::new_err("tau must be finite and >= 0"));
        }
        // Robbins-Monro convergence requires kappa in (0.5, 1].
        if !(kappa > 0.5 && kappa <= 1.0) {
            return Err(PyValueError::new_err("kappa must be in (0.5, 1.0]"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        if inner_iters == 0 {
            return Err(PyValueError::new_err("inner_iters must be >= 1"));
        }
        if let Some(td) = total_docs {
            if !(td.is_finite() && td >= 1.0) {
                return Err(PyValueError::new_err("total_docs must be finite and >= 1"));
            }
        }
        Ok(OnlineLDA {
            num_topics,
            alpha_sum,
            beta,
            tau,
            kappa,
            batch_size,
            inner_iters,
            mean_change_tol,
            total_docs,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit by online VB: sweep the corpus for `iters` passes, one stochastic λ
    /// update per minibatch. `data` is a :class:`Corpus` or a list of token
    /// lists. `convergence_tol` (default 0.0, disabled) early-stops on the
    /// relative change in the per-pass evidence lower bound. Returns `self`.
    #[pyo3(signature = (data, *, iters=100, convergence_tol=0.0))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
        convergence_tol: f64,
    ) -> PyResult<Py<Self>> {
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
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

        let num_topics = slf.num_topics;
        let alpha = slf.alpha_vec();
        let (beta, tau, kappa, batch_size, inner_iters, mct, seed) = (
            slf.beta,
            slf.tau,
            slf.kappa,
            slf.batch_size,
            slf.inner_iters,
            slf.mean_change_tol,
            slf.seed,
        );
        let total_docs_override = slf.total_docs;

        let (model, corpus) = py.allow_threads(move || {
            let mut rng = Pcg64Mcg::seed_from_u64(seed);
            // A user-declared streaming corpus size scales the natural gradient for
            // BOTH the batch fit and later partial_fit; otherwise D = fit-corpus size.
            let m = online_lda::fit(
                &corpus,
                num_topics,
                alpha,
                beta,
                tau,
                kappa,
                batch_size,
                inner_iters,
                mct,
                iters,
                convergence_tol,
                total_docs_override,
                &mut rng,
            );
            (m, corpus)
        });

        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Fold one fresh minibatch of documents into the fitted model: a single
    /// stochastic λ update that advances the Robbins-Monro schedule. `data` is a
    /// :class:`Corpus` or list of token lists; tokens outside the trained
    /// vocabulary are dropped (the vocabulary is fixed at the first `fit`, as
    /// gensim fixes it via its dictionary). Updates :attr:`topic_word` and returns
    /// the minibatch's document-topic matrix (rows = documents, columns = topics).
    /// Requires a prior `fit` to establish the vocabulary.
    fn partial_fit<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "partial_fit requires a fitted model: call fit() on an initial batch first \
                 to establish the vocabulary, then stream further minibatches with partial_fit()",
            ));
        }
        let docs = self.map_to_ids(data)?;
        if docs.is_empty() {
            return Err(PyValueError::new_err("partial_fit received no documents"));
        }
        let model = self.fitted_model_mut()?;
        py.allow_threads(|| model.partial_fit(&docs));
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Infer the document-topic matrix for held-out `data` at the current λ,
    /// **without** updating the model (a pure E-step). Out-of-vocabulary tokens
    /// are dropped. Returns a (num_docs, num_topics) row-stochastic array.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let docs = self.map_to_ids(data)?;
        let model = self.fitted_model()?;
        let theta = py.allow_threads(|| model.transform(&docs));
        // Preserve the (0, num_topics) shape on an empty input rather than the
        // (0, 0) that `vecs_to_arr2` returns for an empty row set.
        if theta.is_empty() {
            return Ok(Array2::<f64>::zeros((0, self.num_topics)).to_pyarray_bound(py));
        }
        Ok(vecs_to_arr2(&theta).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix φ (num_topics, vocab); each row is λ normalized to sum 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Document-topic matrix θ (num_docs, num_topics), each row summing to 1.
    /// After `fit` this covers the training corpus; after `partial_fit` it is the
    /// most recently processed minibatch.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// The number of stochastic λ updates applied so far (the Robbins-Monro step
    /// index); advanced by every minibatch of `fit` and each `partial_fit`.
    #[getter]
    fn updates(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.updates)
    }
    /// Per-topic document-topic Dirichlet prior α, shape (num_topics,).
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.alpha.clone()).to_pyarray_bound(py))
    }
    /// The symmetric topic-word Dirichlet prior (scalar).
    #[getter]
    fn beta(&self) -> f64 {
        self.beta
    }
    /// Per-document token counts underlying the current `doc_topic` (Tier-2
    /// Dirichlet surface; row order matches `doc_topic`).
    #[getter]
    fn doc_lengths(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.doc_lengths.clone())
    }
    /// Online VB keeps a single variational posterior, not MCMC samples, so there
    /// are no retained θ draws (always `None`). Present for the Dirichlet-family
    /// contract; `standard_errors(..., method="composition")` therefore falls back
    /// to the Dirichlet approximation rather than cross-draw samples.
    #[getter]
    fn theta_draws(&self, _py: Python<'_>) -> Option<PyObject> {
        None
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Per-pass evidence-lower-bound trace as `(pass, elbo)` pairs (from `fit`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
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
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
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

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_ONLINE_LDA,
            &OnlineLdaState {
                num_topics: self.num_topics,
                alpha_sum: self.alpha_sum,
                beta: self.beta,
                tau: self.tau,
                kappa: self.kappa,
                batch_size: self.batch_size,
                inner_iters: self.inner_iters,
                mean_change_tol: self.mean_change_tol,
                total_docs: self.total_docs,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                num_types: Some(m.num_types),
                alpha: Some(m.alpha.clone()),
                eta: Some(m.eta),
                lambda: Some(m.lambda.clone()),
                updates: Some(m.updates),
                model_tau: Some(m.tau),
                model_kappa: Some(m.kappa),
                model_batch_size: Some(m.batch_size),
                model_inner_iters: Some(m.inner_iters),
                model_mean_change_tol: Some(m.mean_change_tol),
                model_total_docs: Some(m.total_docs),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                doc_lengths: Some(m.doc_lengths.clone()),
                fit_history: Some(m.fit_history.clone()),
                converged: Some(m.converged),
            },
        )
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: OnlineLdaState = read_state(path, MODEL_TAG_ONLINE_LDA)?;
        let model = if s.fitted && s.lambda.is_some() {
            Some(OnlineLDAModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                alpha: s.alpha.unwrap_or_default(),
                eta: s.eta.unwrap_or(s.beta),
                tau: s.model_tau.unwrap_or(s.tau),
                kappa: s.model_kappa.unwrap_or(s.kappa),
                batch_size: s.model_batch_size.unwrap_or(s.batch_size),
                inner_iters: s.model_inner_iters.unwrap_or(s.inner_iters),
                mean_change_tol: s.model_mean_change_tol.unwrap_or(s.mean_change_tol),
                total_docs: s.model_total_docs.unwrap_or(1.0),
                lambda: s.lambda.unwrap_or_default(),
                updates: s.updates.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                doc_lengths: s.doc_lengths.unwrap_or_default(),
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
            })
        } else {
            None
        };
        Ok(OnlineLDA {
            num_topics: s.num_topics,
            alpha_sum: s.alpha_sum,
            beta: s.beta,
            tau: s.tau,
            kappa: s.kappa,
            batch_size: s.batch_size,
            inner_iters: s.inner_iters,
            mean_change_tol: s.mean_change_tol,
            total_docs: s.total_docs,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineLDA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
