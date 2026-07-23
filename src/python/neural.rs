//! Python bindings for the neural / VAE topic models: ETM, DETM, InfoCTM,
//! ProdLDA, and its contextualized variants CombinedTM and ZeroShotTM. Extracted
//! from `mod.rs` (issue #385). These share the AVITM prior/option helpers
//! (`prior_from_str`, `build_avitm_options`, the encoder-input `mode` codec, and
//! `parse_doc_embeddings`), which move with them; the models themselves are built
//! on `src/prodlda.rs` / `src/etm.rs` / `src/detm.rs` / `src/infoctm.rs`. Other
//! shared helpers, type aliases, and save-format tags stay in `mod.rs`, reached
//! via `use super::*`. No public API change: the classes are still registered in
//! the `#[pymodule]` fn and imported as `topica.ETM`, `topica.ProdLDA`, etc.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }

    /// Topic-word matrix beta, from whichever inference path was fit.
    fn surf_beta(&self) -> PyResult<&Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(&m.beta),
            (_, Some(m)) => Ok(&m.beta),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }

    /// Topic embeddings alpha.
    fn surf_alpha(&self) -> PyResult<&Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(&m.alpha),
            (_, Some(m)) => Ok(&m.alpha),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }

    /// Document-topic proportions theta (computed fresh for the EM path).
    fn surf_doc_topic(&self) -> PyResult<Vec<Vec<f64>>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.doc_topics()),
            (_, Some(m)) => Ok(m.doc_topic.clone()),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }

    fn surf_bound(&self) -> PyResult<f64> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.bound),
            (_, Some(m)) => Ok(m.bound),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }

    fn surf_converged(&self) -> PyResult<bool> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.converged),
            (_, Some(m)) => Ok(m.converged),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }

    fn surf_bound_history(&self) -> PyResult<Vec<f64>> {
        self.ensure_fitted()?;
        match (&self.model, &self.vae) {
            (Some(m), _) => Ok(m.bound_history.clone()),
            (_, Some(m)) => Ok(m.bound_history.clone()),
            _ => Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            )),
        }
    }
}

#[pymethods]
impl ETM {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). `inference`/`prior` are reported under
    /// their public strings; `convergence_tol` is the effective tolerance in force
    /// (the deprecated `em_tol` alias is folded into it and reported as ``None``).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("inference", self.inference.as_str())?;
        d.set_item("convergence_tol", self.em_tol)?;
        d.set_item("sigma_shrink", self.sigma_shrink)?;
        d.set_item("prior_variance", self.prior_variance)?;
        d.set_item("max_inner", self.max_inner)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("lr", self.lr)?;
        d.set_item("wdecay", self.wdecay)?;
        d.set_item("seed", self.seed)?;
        d.set_item("prior", self.prior.as_str())?;
        d.set_item("contrastive", self.contrastive)?;
        d.set_item("contrastive_weight", self.contrastive_weight)?;
        d.set_item("contrastive_temp", self.contrastive_temp)?;
        d.set_item("em_tol", None::<f64>)?;
        Ok(d)
    }

    /// Create an unfitted model. `inference` selects the engine: `"em"` (default)
    /// is per-document variational EM, accurate but not minibatched; `"vae"` is the
    /// reference's amortized autoencoder, which scales to large corpora and maps new
    /// documents with a single encoder pass. `convergence_tol`/`prior_variance`/
    /// `max_inner`/`sigma_shrink` govern the EM path; `hidden_size`/
    /// `batch_size`/`lr`/`wdecay`/`convergence_tol` govern the VAE path.
    /// Pass `iters` to :meth:`fit` to set the iteration count.
    ///
    /// `num_topics` is the number of topics K; `seed` seeds the RNG. `prior` sets
    /// the document-topic prior on the VAE path: ``"laplace"`` (default,
    /// logistic-normal Laplace approximation to a Dirichlet), ``"dirichlet"`` (true
    /// Dirichlet via a Weibull reparameterization) or ``"stick_breaking"`` (Gaussian
    /// stick-breaking). `contrastive` adds an InfoNCE contrastive term on the topic
    /// vectors, scaled by `contrastive_weight` with InfoNCE temperature
    /// `contrastive_temp`. `em_tol` is a deprecated alias for `convergence_tol`.
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
            warnings.call_method1(
                "warn",
                (
                    "ETM(em_tol=) is deprecated; use convergence_tol= instead",
                    py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                    2_i32,
                ),
            )?;
            if (convergence_tol - 1e-4_f64).abs() > f64::EPSILON {
                convergence_tol
            } else {
                old_val
            }
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
                let strings = c
                    .inner
                    .docs
                    .iter()
                    .map(|d| {
                        d.iter()
                            .map(|&w| c.inner.id_to_word[w as usize].clone())
                            .collect()
                    })
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
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        let map: std::collections::HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let docs_ids: Vec<Vec<u32>> = docs_str
            .iter()
            .map(|d| {
                d.iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens in the documents",
            ));
        }
        let num_types = vocabulary.len();
        self.id_to_word = vocabulary.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        if self.inference == "vae" {
            let ep = iters.unwrap_or(150);
            let opts = build_avitm_options(
                &self.prior,
                self.contrastive,
                self.contrastive_weight,
                self.contrastive_temp,
            )?;
            let (k, h, bs, lr, wd, et) = (
                self.num_topics,
                self.hidden_size,
                self.batch_size,
                self.lr,
                self.wdecay,
                tol,
            );
            let m = py.allow_threads(move || {
                etm_vae::fit_etm_vae(
                    &docs_ids, k, num_types, &rho, h, ep, bs, lr, wd, et, opts, &mut rng,
                )
            });
            self.vae = Some(m);
            self.model = None;
        } else {
            let ei = iters.unwrap_or(100);
            let (k, et, ss, pv, mi) = (
                self.num_topics,
                tol,
                self.sigma_shrink,
                self.prior_variance,
                self.max_inner,
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
                let ids: Vec<u32> = doc
                    .iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect();
                let mut seen = std::collections::HashSet::new();
                for &id in &ids {
                    tf[id as usize] += 1;
                    seen.insert(id as usize);
                }
                for id in seen {
                    df[id] += 1;
                }
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
        Ok(self
            .surf_bound_history()?
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
        let phi = vecs_to_arr2(self.surf_beta()?);
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }
    /// UMass coherence for each topic's top-`n` words, over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(self.surf_beta()?);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        self.ensure_fitted()?;
        let (beta_em, alpha_em, mu_em, sigma_em, lambda_em, bound_em, converged_em) =
            if let Some(m) = &self.model {
                (
                    Some(m.beta.clone()),
                    Some(m.alpha.clone()),
                    Some(m.mu.clone()),
                    Some(m.sigma.clone()),
                    Some(m.lambda.clone()),
                    Some(m.bound),
                    Some(m.converged),
                )
            } else {
                (None, None, None, None, None, None, None)
            };
        let (
            beta_vae,
            alpha_vae,
            doc_topic_vae,
            bound_vae,
            converged_vae,
            enc_v,
            enc_hidden,
            enc_w1,
            enc_b1,
            enc_w2,
            enc_b2,
            enc_w_mu,
            enc_b_mu,
            enc_w_ls,
            enc_b_ls,
        ) = if let Some(m) = &self.vae {
            let enc = &m.encoder;
            (
                Some(m.beta.clone()),
                Some(m.alpha.clone()),
                Some(m.doc_topic.clone()),
                Some(m.bound),
                Some(m.converged),
                Some(enc.v),
                Some(enc.hidden),
                Some(enc.w1.clone()),
                Some(enc.b1.clone()),
                Some(enc.w2.clone()),
                Some(enc.b2.clone()),
                Some(enc.w_mu.clone()),
                Some(enc.b_mu.clone()),
                Some(enc.w_ls.clone()),
                Some(enc.b_ls.clone()),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None,
            )
        };
        write_state(
            path,
            MODEL_TAG_ETM,
            &EtmState {
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
                beta_em,
                alpha_em,
                mu_em,
                sigma_em,
                lambda_em,
                bound_em,
                converged_em,
                beta_vae,
                alpha_vae,
                doc_topic_vae,
                bound_vae,
                converged_vae,
                enc_v,
                enc_hidden,
                enc_w1,
                enc_b1,
                enc_w2,
                enc_b2,
                enc_w_mu,
                enc_b_mu,
                enc_w_ls,
                enc_b_ls,
            },
        )
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
        } else {
            None
        };
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
        } else {
            None
        };
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
    ///
    /// `word_embeddings` is the ``(len(vocabulary), E)`` dense embedding matrix
    /// aligned to `vocabulary`, the list of word strings; tokens outside it are
    /// dropped. `iters` is the number of training iterations (EM iterations or VAE
    /// epochs).
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
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``grad_clip`` is ``None`` when unset.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("delta", self.delta)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("eta_hidden_size", self.eta_hidden_size)?;
        d.set_item("eta_nlayers", self.eta_nlayers)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("lr", self.lr)?;
        d.set_item("wdecay", self.wdecay)?;
        d.set_item("grad_clip", self.grad_clip)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

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
    /// `num_topics` is the number of topics K; `seed` seeds the RNG.
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
            if c <= 0.0 || !c.is_finite() {
                return Err(PyValueError::new_err(
                    "grad_clip must be a positive finite float or None",
                ));
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
    /// `convergence_tol` is the relative-bound tolerance for EM early stopping —
    /// the run stops when the relative change in the variational evidence bound
    /// falls below it.
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
                let strings = c
                    .inner
                    .docs
                    .iter()
                    .map(|d| {
                        d.iter()
                            .map(|&w| c.inner.id_to_word[w as usize].clone())
                            .collect()
                    })
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
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }

        let map: std::collections::HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
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
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens in the documents",
            ));
        }

        let num_types = vocabulary.len();
        self.id_to_word = vocabulary.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (k, delta, h, eh, enl, bs, lr, wd, gc) = (
            self.num_topics,
            self.delta,
            self.hidden_size,
            self.eta_hidden_size,
            self.eta_nlayers,
            self.batch_size,
            self.lr,
            self.wdecay,
            self.grad_clip,
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
                let ids: Vec<u32> = doc
                    .iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect();
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
        let (t, k, l) = (
            m.num_times,
            m.num_topics,
            if m.num_topics > 0 {
                m.alpha[0][0].len()
            } else {
                0
            },
        );
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
        Ok(m.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
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
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path`. Reload with `DETM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_DETM,
            &DetmState {
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
            },
        )
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
        Ok(if self.lang_index(lang)? == 0 {
            &m.model_a
        } else {
            &m.model_b
        })
    }
    fn corpus_for(&self, lang: &str) -> PyResult<&corpus::Corpus> {
        let c = if self.lang_index(lang)? == 0 {
            &self.corpus_a
        } else {
            &self.corpus_b
        };
        c.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl InfoCTM {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). `convergence_tol` is the effective
    /// tolerance in force; `languages` is the normalized ``(a, b)`` pair.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("mi_weight", self.mi_weight)?;
        d.set_item("mi_temperature", self.mi_temperature)?;
        d.set_item("pos_threshold", self.pos_threshold)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("dropout", self.dropout)?;
        d.set_item("lr", self.lr)?;
        d.set_item("convergence_tol", self.em_tol)?;
        d.set_item("seed", self.seed)?;
        d.set_item("languages", self.languages.clone())?;
        Ok(d)
    }

    /// Create an unfitted model. `mi_weight` scales the alignment term (reference
    /// 30-50); `mi_temperature` is the InfoNCE temperature (0.2); `pos_threshold`
    /// is the cosine cutoff for the embedding-densified positive mask (0.4, used
    /// only when embeddings are given). `languages` names the two corpora for the
    /// `lang=` selector (default `("a", "b")`). Pass `iters`/`batch_size` to `fit`.
    ///
    /// `num_topics` is the number of topics K; `seed` seeds the RNG.
    /// `hidden_size` is the encoder hidden-layer width, `dropout` the encoder
    /// dropout rate, and `lr` the Adam learning rate. `convergence_tol` is the
    /// relative-bound tolerance for EM early stopping (0 disables it).
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
    /// `batch_size` is the number of documents per minibatch.
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
                Ok(build_corpus_from_docs(
                    docs,
                    None,
                    None,
                    std::collections::HashSet::new(),
                    1,
                    1.0,
                    0,
                    0,
                )?
                .0)
            }
        };
        let corpus_a = to_corpus(data_a)?;
        let corpus_b = to_corpus(data_b)?;
        let (va, vb) = (corpus_a.num_types(), corpus_b.num_types());
        if corpus_a.num_docs() == 0 || corpus_b.num_docs() == 0 {
            return Err(PyValueError::new_err("both corpora must contain documents"));
        }
        if va < self.num_topics || vb < self.num_topics {
            return Err(PyValueError::new_err(
                "each vocabulary must have at least num_topics words",
            ));
        }

        // word -> id maps for each language.
        let wid = |c: &corpus::Corpus| -> std::collections::HashMap<String, usize> {
            c.id_to_word
                .iter()
                .enumerate()
                .map(|(i, w)| (w.clone(), i))
                .collect()
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
        let (k, h, dp, lr, et) = (
            self.num_topics,
            self.hidden_size,
            self.dropout,
            self.lr,
            self.em_tol,
        );
        let (mw, mt, pt) = (self.mi_weight, self.mi_temperature, self.pos_threshold);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let docs_a = corpus_a.docs.clone();
        let docs_b = corpus_b.docs.clone();
        let model = py.allow_threads(move || {
            infoctm::fit_infoctm(
                &docs_a,
                &docs_b,
                va,
                vb,
                &trans_ab,
                emb_a.as_deref(),
                emb_b.as_deref(),
                k,
                h,
                dp,
                ep,
                batch_size,
                lr,
                mw,
                mt,
                pt,
                et,
                &mut rng,
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
    /// `lang` selects which language's output to return (the language name, or
    /// ``"a"``/``"b"`` / ``"0"``/``"1"`` for the first/second corpus).
    #[pyo3(signature = (lang="a"))]
    fn doc_topic<'py>(&self, py: Python<'py>, lang: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.model_for(lang)?.doc_topic).to_pyarray_bound(py))
    }

    /// Vocabulary for one language, in id order.
    /// `lang` selects which language's vocabulary to return (the language name, or
    /// ``"a"``/``"b"`` / ``"0"``/``"1"`` for the first/second corpus).
    #[pyo3(signature = (lang="a"))]
    fn vocabulary(&self, lang: &str) -> PyResult<Vec<String>> {
        Ok(self.corpus_for(lang)?.id_to_word.clone())
    }

    /// Top-`n` words per topic for one language as `(word, weight)` pairs.
    /// `lang` selects which language's output to return (the language name, or
    /// ``"a"``/``"b"`` / ``"0"``/``"1"`` for the first/second corpus).
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
                idx.into_iter()
                    .take(n)
                    .map(|j| (vocab[j].clone(), row[j]))
                    .collect()
            })
            .collect())
    }

    /// Assign held-out documents of one language to the discovered topics
    /// (num_docs, num_topics) via a single encoder pass. Words outside that
    /// language's vocabulary are dropped.
    /// `lang` selects which language's output to return (the language name, or
    /// ``"a"``/``"b"`` / ``"0"``/``"1"`` for the first/second corpus).
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
        self.model
            .as_ref()
            .map(|m| m.bound_history.clone())
            .unwrap_or_default()
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
            return Err(PyValueError::new_err(
                "contrastive_weight must be >= 0 and finite",
            ));
        }
        if !(contrastive_temp > 0.0 && contrastive_temp.is_finite()) {
            return Err(PyValueError::new_err(
                "contrastive_temp must be > 0 and finite",
            ));
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
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). `prior` is reported as its public
    /// string; `convergence_tol` is the effective tolerance in force (the
    /// deprecated `em_tol` alias is folded into it and reported as ``None``).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("dropout", self.dropout)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("lr", self.lr)?;
        d.set_item("convergence_tol", self.em_tol)?;
        d.set_item("seed", self.seed)?;
        d.set_item("prior", self.prior.as_str())?;
        d.set_item("contrastive", self.contrastive)?;
        d.set_item("contrastive_weight", self.contrastive_weight)?;
        d.set_item("contrastive_temp", self.contrastive_temp)?;
        d.set_item("em_tol", None::<f64>)?;
        Ok(d)
    }

    /// Create an unfitted model. `alpha` is the symmetric Dirichlet prior
    /// concentration (reference 1.0); `hidden_size` is the encoder width (reference
    /// 100); `dropout` is the dropout rate on the hidden layer and on `theta`;
    /// `batch_size`/`lr` drive Adam (reference 200/0.002, with `beta1 = 0.99`);
    /// `convergence_tol > 0` stops early on the relative change in the epoch ELBO (0 runs
    /// all epochs). Pass `iters` to :meth:`fit` to set the number of epochs.
    ///
    /// `num_topics` is the number of topics K; `seed` seeds the RNG. `contrastive`
    /// adds an InfoNCE contrastive term on the topic vectors, scaled by
    /// `contrastive_weight` with InfoNCE temperature `contrastive_temp`. `em_tol`
    /// is a deprecated alias for `convergence_tol`.
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
            warnings.call_method1(
                "warn",
                (
                    "ProdLDA(em_tol=) is deprecated; use convergence_tol= instead",
                    py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
                    2_i32,
                ),
            )?;
            // ProdLDA default is 0.0; if unchanged, use the deprecated value.
            if convergence_tol != 0.0 {
                convergence_tol
            } else {
                old_val
            }
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
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
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
        if num_types < self.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        let ep = iters.unwrap_or(200);
        let opts = build_avitm_options(
            &self.prior,
            self.contrastive,
            self.contrastive_weight,
            self.contrastive_temp,
        )?;
        let (k, h, a, dp, bs, lr, et) = (
            self.num_topics,
            self.hidden_size,
            self.alpha,
            self.dropout,
            self.batch_size,
            self.lr,
            tol,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let empty: Vec<Vec<f64>> = vec![Vec::new(); corpus.docs.len()];
        let (model, corpus) = py.allow_threads(move || {
            let m = prodlda::fit_avitm(
                &corpus.docs,
                &empty,
                prodlda::InputMode::BowOnly,
                k,
                num_types,
                0,
                h,
                a,
                dp,
                ep,
                bs,
                lr,
                et,
                opts,
                &mut rng,
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
    /// UMass topic coherence per topic, shape ``(num_topics,)``. `n` is the number
    /// of top words per topic scored.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
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
        write_state(
            path,
            MODEL_TAG_PRODLDA,
            &ProdldaState {
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
            },
        )
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: ProdldaState = read_state(path, MODEL_TAG_PRODLDA)?;
        let model = if let (true, Some(v)) = (s.fitted, s.w_v) {
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
                    e: 0,
                    hidden,
                    k,
                    mode: prodlda::InputMode::BowOnly,
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
        format!(
            "ProdLDA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
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
fn parse_doc_embeddings(data: &Bound<'_, PyAny>, num_docs: usize) -> PyResult<Vec<Vec<f64>>> {
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
            /// The random seed the model was constructed with.
            #[getter]
            fn seed(&self) -> u64 {
                self.seed
            }

            /// The constructor configuration as a JSON-serialisable dict,
            /// keyword-named to match ``__init__`` (issue #400). `prior` is
            /// reported as its public string.
            #[getter]
            fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let d = PyDict::new_bound(py);
                d.set_item("num_topics", self.num_topics)?;
                d.set_item("alpha", self.alpha)?;
                d.set_item("hidden_size", self.hidden_size)?;
                d.set_item("dropout", self.dropout)?;
                d.set_item("batch_size", self.batch_size)?;
                d.set_item("lr", self.lr)?;
                d.set_item("convergence_tol", self.convergence_tol)?;
                d.set_item("seed", self.seed)?;
                d.set_item("prior", self.prior.as_str())?;
                d.set_item("contrastive", self.contrastive)?;
                d.set_item("contrastive_weight", self.contrastive_weight)?;
                d.set_item("contrastive_temp", self.contrastive_temp)?;
                Ok(d)
            }

            /// Create an unfitted model. `alpha` is the symmetric Dirichlet prior
            /// concentration (reference 1.0); `hidden_size` is the encoder width
            /// (reference 100); `dropout` is the dropout rate on the hidden layer
            /// and on `theta`; `batch_size`/`lr` drive Adam (reference 200/0.002,
            /// with `beta1 = 0.99`); `convergence_tol > 0` stops early on the
            /// relative change in the epoch ELBO (0 runs all epochs). Pass `iters`
            /// to :meth:`fit` to set the number of epochs.
            ///
            /// `num_topics` is the number of topics K; `seed` seeds the RNG. `contrastive`
            /// adds an InfoNCE contrastive term on the topic vectors, scaled by
            /// `contrastive_weight` with InfoNCE temperature `contrastive_temp`.
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
                if num_types < self.num_topics {
                    return Err(PyValueError::new_err(
                        "vocabulary must have at least num_topics words",
                    ));
                }
                let embs = parse_doc_embeddings(doc_embeddings, corpus.num_docs())?;
                let emb_dim = embs.first().map(|r| r.len()).unwrap_or(0);
                if emb_dim == 0 {
                    return Err(PyValueError::new_err(
                        "doc_embeddings must have at least one column",
                    ));
                }
                self.emb_dim = emb_dim;
                let ep = iters.unwrap_or(200);
                let opts = build_avitm_options(
                    &self.prior,
                    self.contrastive,
                    self.contrastive_weight,
                    self.contrastive_temp,
                )?;
                let (k, h, a, dp, bs, lr) = (
                    self.num_topics,
                    self.hidden_size,
                    self.alpha,
                    self.dropout,
                    self.batch_size,
                    self.lr,
                );
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                let (model, corpus) = py.allow_threads(move || {
                    let m = prodlda::fit_avitm(
                        &corpus.docs,
                        &embs,
                        $mode,
                        k,
                        num_types,
                        emb_dim,
                        h,
                        a,
                        dp,
                        ep,
                        bs,
                        lr,
                        tol,
                        opts,
                        &mut rng,
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
            /// UMass topic coherence per topic, shape ``(num_topics,)``. `n` is the number
            /// of top words per topic scored.
            #[pyo3(signature = (n=10))]
            fn coherence<'py>(
                &self,
                py: Python<'py>,
                n: usize,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
                let tops = top_word_ids_phi(&phi, self.num_topics, n);
                Ok(
                    Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                        .to_pyarray_bound(py),
                )
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
            ///
            /// `doc_embeddings` is the ``(num_docs, E)`` dense embedding matrix, one row
            /// per document in corpus order. `iters` is the number of training epochs.
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
                write_state(
                    path,
                    $tag,
                    &CtmEmbState {
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
                    },
                )
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
                format!(
                    concat!($repr, "(num_topics={}, fitted={})"),
                    self.num_topics, self.fitted
                )
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
