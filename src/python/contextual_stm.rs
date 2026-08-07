//! Python bindings for ContextualSTM (experimental): a contextual sentence-embedding
//! topic model with STM/SCHOLAR-style prevalence covariates. See
//! `crate::contextual_stm` for the model.

use super::*;
use crate::contextual_stm::{CovariateMode, EncoderKind};
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn parse_encoder(s: &str) -> PyResult<EncoderKind> {
    match s {
        "combined" => Ok(EncoderKind::Combined),
        "zeroshot" => Ok(EncoderKind::ZeroShot),
        other => Err(PyValueError::new_err(format!(
            "encoder must be 'combined' or 'zeroshot', got {other:?}"
        ))),
    }
}

fn parse_covariate_mode(s: &str) -> PyResult<CovariateMode> {
    match s {
        "encoder_prior" => Ok(CovariateMode::EncoderPrior),
        "prior_only" => Ok(CovariateMode::PriorOnly),
        other => Err(PyValueError::new_err(format!(
            "covariate_mode must be 'encoder_prior' or 'prior_only', got {other:?}"
        ))),
    }
}

/// ContextualSTM (experimental): CombinedTM/ZeroShotTM's contextual sentence-embedding
/// encoder extended with SCHOLAR's prevalence-covariate prior. `covariates` shift the
/// per-document logistic-normal prior mean, so a covariate that co-occurs with a topic
/// raises its prevalence — the neural analog of STM/DMR prevalence covariates,
/// estimated inside the fit rather than post-hoc. `covariate_effects` is a *point*
/// estimate on the standardized-logit scale (no uncertainty); for proportion-scale
/// prevalence effects run ``topica.estimate_effect(model.doc_topic, X=covariates)``.
/// Experimental: enable with ``topica.enable_experimental()``.
#[pyclass(module = "topica")]
pub struct ContextualSTM {
    num_topics: usize,
    encoder: String,
    covariate_mode: String,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    l2_prior_reg: f64,
    convergence_tol: f64,
    seed: u64,
    // Covariates optionally supplied at construction (else at fit).
    covariates: Option<Vec<Vec<f64>>>,
    covariate_names: Option<Vec<String>>,
    // Standardization learned at fit (per covariate column), applied at transform.
    cov_mean: Option<Vec<f64>>,
    cov_std: Option<Vec<f64>>,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::contextual_stm::ContextualStmModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ContextualStmState {
    num_topics: usize,
    encoder: String,
    covariate_mode: String,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    l2_prior_reg: f64,
    convergence_tol: f64,
    seed: u64,
    covariate_names: Vec<String>,
    cov_mean: Option<Vec<f64>>,
    cov_std: Option<Vec<f64>>,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    n_covariates: Option<usize>,
    emb_dim: Option<usize>,
    // Fitted model.
    doc_topic: Option<Vec<Vec<f64>>>,
    bound: Option<f64>,
    bound_history: Option<Vec<f64>>,
    converged: Option<bool>,
    epochs_run: Option<usize>,
    prior_w: Option<Vec<f64>>,
    // Base VAE weights.
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
    w_w_adapt: Option<Vec<f64>>,
    w_b_adapt: Option<Vec<f64>>,
    bn_running_mean: Option<Vec<f64>>,
    bn_running_var: Option<Vec<f64>>,
}

impl ContextualSTM {
    fn fitted_model(&self) -> PyResult<&crate::contextual_stm::ContextualStmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Parse + validate a raw covariate matrix to `(matrix, width)`, preferring the
    /// one passed here over the constructor's.
    fn resolve_covariates(
        &self,
        provided: Option<&Bound<'_, PyAny>>,
        num_docs: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let cov = match provided {
            Some(obj) => parse_features(obj)?,
            None => self.covariates.clone().ok_or_else(|| {
                PyValueError::new_err(
                    "covariates are required: pass covariates= to ContextualSTM(...) or to fit()",
                )
            })?,
        };
        if cov.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "covariates has {} rows but there are {num_docs} documents",
                cov.len()
            )));
        }
        if cov.is_empty() || cov[0].is_empty() {
            return Err(PyValueError::new_err(
                "covariates must have at least one column",
            ));
        }
        let width = cov[0].len();
        if cov.iter().any(|r| r.len() != width) {
            return Err(PyValueError::new_err(
                "all covariate rows must have the same number of columns",
            ));
        }
        Ok(cov)
    }
}

#[pymethods]
impl ContextualSTM {
    /// Create an unfitted ContextualSTM. `encoder` selects the contextual encoder:
    /// ``"combined"`` (CombinedTM: adapt-projected embedding + bag of words) or
    /// ``"zeroshot"`` (ZeroShotTM: embedding only). `covariate_mode` selects how
    /// covariates enter: ``"encoder_prior"`` (SCHOLAR-style — prior and encoder;
    /// default, best recovery) or ``"prior_only"`` (STM-purist — prior only).
    /// `covariates` (a `(num_docs, n_covars)` numeric matrix) may be given here or at
    /// :meth:`fit`; `covariate_names` labels the columns. Covariates are standardized
    /// (z-scored) internally, so effects are on the standardized scale. `alpha` is the
    /// Dirichlet concentration behind the prior variance; `l2_prior_reg` is the L2
    /// penalty on the covariate weights (raise it if covariates are collinear).
    #[new]
    #[pyo3(signature = (num_topics, *, encoder="combined".to_string(),
                        covariate_mode="encoder_prior".to_string(), covariates=None,
                        covariate_names=None, alpha=1.0, hidden_size=100, dropout=0.2,
                        batch_size=200, lr=0.002, l2_prior_reg=0.0, convergence_tol=0.0,
                        seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        encoder: String,
        covariate_mode: String,
        covariates: Option<&Bound<'_, PyAny>>,
        covariate_names: Option<Vec<String>>,
        alpha: f64,
        hidden_size: usize,
        dropout: f64,
        batch_size: usize,
        lr: f64,
        l2_prior_reg: f64,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("ContextualSTM")?;
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !finite_pos(alpha) {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be in [0, 1)"));
        }
        if !(l2_prior_reg >= 0.0 && l2_prior_reg.is_finite()) {
            return Err(PyValueError::new_err(
                "l2_prior_reg must be >= 0 and finite",
            ));
        }
        parse_encoder(&encoder)?;
        parse_covariate_mode(&covariate_mode)?;
        let covariates = match covariates {
            Some(obj) => Some(parse_features(obj)?),
            None => None,
        };
        if let (Some(cov), Some(names)) = (&covariates, &covariate_names) {
            if !cov.is_empty() && names.len() != cov[0].len() {
                return Err(PyValueError::new_err(format!(
                    "covariate_names has length {} but covariates has {} columns",
                    names.len(),
                    cov[0].len()
                )));
            }
        }
        Ok(ContextualSTM {
            num_topics,
            encoder,
            covariate_mode,
            hidden_size,
            alpha,
            dropout,
            batch_size,
            lr,
            l2_prior_reg,
            convergence_tol,
            seed,
            covariates,
            covariate_names,
            cov_mean: None,
            cov_std: None,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// The constructor configuration as a JSON-serialisable dict (issue #400). The
    /// numeric `covariates` array is excluded; `covariate_names` is kept.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("encoder", self.encoder.clone())?;
        d.set_item("covariate_mode", self.covariate_mode.clone())?;
        d.set_item("covariate_names", self.covariate_names.clone())?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("dropout", self.dropout)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("lr", self.lr)?;
        d.set_item("l2_prior_reg", self.l2_prior_reg)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (a `(num_docs, E)` dense array, one row per document in corpus order) and
    /// `covariates` (a `(num_docs, n_covars)` numeric matrix, given here or at
    /// construction). `iters` sets the number of epochs (default 200).
    #[pyo3(signature = (data, doc_embeddings, *, covariates=None, iters=None,
                        convergence_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        covariates: Option<&Bound<'_, PyAny>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
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
        let num_docs = corpus.docs.len();
        let num_types = corpus.num_types();
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }

        // Document (sentence) embeddings, one row per document.
        let embs = parse_features(doc_embeddings)?;
        if embs.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but there are {num_docs} documents",
                embs.len()
            )));
        }
        let emb_dim = embs.first().map(|r| r.len()).unwrap_or(0);
        if emb_dim == 0 {
            return Err(PyValueError::new_err(
                "doc_embeddings must have at least one column",
            ));
        }
        if embs.iter().any(|r| r.len() != emb_dim) {
            return Err(PyValueError::new_err(
                "all doc_embeddings rows must have the same width",
            ));
        }

        // Covariates: validate, guard against constant/collinear columns, standardize.
        let raw_cov = slf.resolve_covariates(covariates, num_docs)?;
        let n_covariates = raw_cov[0].len();
        let (mean, std) = crate::contextual_stm::column_stats(&raw_cov);
        let names = match &slf.covariate_names {
            Some(n) if n.len() == n_covariates => n.clone(),
            _ => (0..n_covariates)
                .map(|c| format!("covariate_{c}"))
                .collect(),
        };
        for (j, &s) in std.iter().enumerate() {
            if s <= 1e-12 {
                return Err(PyValueError::new_err(format!(
                    "covariate '{}' is constant across documents; drop it (a constant \
                     covariate carries no prevalence signal and is unidentified)",
                    names[j]
                )));
            }
        }
        let cov = crate::contextual_stm::standardize_with(&raw_cov, &mean, &std);
        let rank = crate::contextual_stm::covariate_rank(&cov);
        if rank < n_covariates && slf.l2_prior_reg == 0.0 {
            return Err(PyValueError::new_err(format!(
                "covariates are collinear (rank {rank} < {n_covariates} columns): the \
                 covariate effects are unidentified. Drop a redundant column (e.g. a \
                 reference level under full dummy coding) or set l2_prior_reg > 0."
            )));
        }

        let encoder = parse_encoder(&slf.encoder)?;
        let covariate_mode = parse_covariate_mode(&slf.covariate_mode)?;
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        let ep = iters.unwrap_or(200);
        let (k, h, a, dp, bs, lr, l2) = (
            slf.num_topics,
            slf.hidden_size,
            slf.alpha,
            slf.dropout,
            slf.batch_size,
            slf.lr,
            slf.l2_prior_reg,
        );
        let seed = slf.seed;
        let (model, corpus) = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let m = crate::contextual_stm::fit_contextual_stm(
                &corpus.docs,
                &embs,
                &cov,
                k,
                num_types,
                n_covariates,
                emb_dim,
                h,
                a,
                dp,
                ep,
                bs,
                lr,
                l2,
                tol,
                encoder,
                covariate_mode,
                &mut rng,
            );
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.covariate_names = Some(names);
        slf.cov_mean = Some(mean);
        slf.cov_std = Some(std);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
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

    /// Document-topic proportions theta (num_docs, num_topics); rows sum to 1. This is
    /// ``q(theta | embedding)`` and is not covariate-adjusted.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.base.doc_topic).to_pyarray_bound(py))
    }

    /// Covariate-by-topic prevalence effects `(n_covars, num_topics)`. Entry ``[c][t]``
    /// is how much (standardized) covariate `c` shifts the log-prior mean of topic `t`;
    /// positive raises that topic's prevalence for documents high on covariate `c`. A
    /// *point* estimate on the standardized-logit latent scale — not a proportion
    /// change, and magnitudes are not directly comparable across topics. For
    /// proportion-scale effects use ``topica.estimate_effect(model.doc_topic, X=cov)``.
    #[getter]
    fn covariate_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.covariate_effects()).to_pyarray_bound(py))
    }

    /// The covariate column names, in the order of `covariate_effects` rows.
    #[getter]
    fn covariate_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.covariate_names.clone().unwrap_or_default())
    }

    /// The ELBO (negative training loss) at the final epoch.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.base.bound)
    }

    /// Per-epoch ELBO history.
    #[getter]
    fn fit_history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.base.bound_history.clone()).to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Held-out topic proportions for new documents given their embeddings and
    /// covariates (standardized with the fit-time mean/std).
    #[pyo3(signature = (data, doc_embeddings, *, covariates=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        covariates: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("transform() expects a Corpus or a list of token lists")
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
        let num_docs = corpus.docs.len();
        let embs = parse_features(doc_embeddings)?;
        if embs.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but there are {num_docs} documents",
                embs.len()
            )));
        }
        let raw_cov = self.resolve_covariates(covariates, num_docs)?;
        if raw_cov[0].len() != m.n_covariates {
            return Err(PyValueError::new_err(format!(
                "covariates has {} columns but the model was fit with {}",
                raw_cov[0].len(),
                m.n_covariates
            )));
        }
        let (mean, std) = (
            self.cov_mean.as_ref().unwrap(),
            self.cov_std.as_ref().unwrap(),
        );
        let cov = crate::contextual_stm::standardize_with(&raw_cov, mean, std);
        Ok(vecs_to_arr2(&m.transform(&corpus.docs, &embs, &cov)).to_pyarray_bound(py))
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
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path`. Reload with `ContextualSTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_CONTEXTUAL_STM,
            &ContextualStmState {
                num_topics: self.num_topics,
                encoder: self.encoder.clone(),
                covariate_mode: self.covariate_mode.clone(),
                hidden_size: self.hidden_size,
                alpha: self.alpha,
                dropout: self.dropout,
                batch_size: self.batch_size,
                lr: self.lr,
                l2_prior_reg: self.l2_prior_reg,
                convergence_tol: self.convergence_tol,
                seed: self.seed,
                covariate_names: self.covariate_names.clone().unwrap_or_default(),
                cov_mean: self.cov_mean.clone(),
                cov_std: self.cov_std.clone(),
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                n_covariates: Some(m.n_covariates),
                emb_dim: Some(m.emb_dim),
                doc_topic: Some(m.base.doc_topic.clone()),
                bound: Some(m.base.bound),
                bound_history: Some(m.base.bound_history.clone()),
                converged: Some(m.base.converged),
                epochs_run: Some(m.base.epochs_run),
                prior_w: Some(m.prior_w.clone()),
                w_v: Some(m.base.weights.v),
                w_e: Some(m.base.weights.e),
                w_hidden: Some(m.base.weights.hidden),
                w_k: Some(m.base.weights.k),
                w_w1: Some(m.base.weights.w1.clone()),
                w_b1: Some(m.base.weights.b1.clone()),
                w_w2: Some(m.base.weights.w2.clone()),
                w_b2: Some(m.base.weights.b2.clone()),
                w_w_mu: Some(m.base.weights.w_mu.clone()),
                w_b_mu: Some(m.base.weights.b_mu.clone()),
                w_w_ls: Some(m.base.weights.w_ls.clone()),
                w_b_ls: Some(m.base.weights.b_ls.clone()),
                w_beta: Some(m.base.weights.beta.clone()),
                w_w_adapt: Some(m.base.weights.w_adapt.clone()),
                w_b_adapt: Some(m.base.weights.b_adapt.clone()),
                bn_running_mean: Some(m.base.bn_mu.running_mean.clone()),
                bn_running_var: Some(m.base.bn_mu.running_var.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        require_experimental("ContextualSTM")?;
        let s: ContextualStmState = read_state(path, MODEL_TAG_CONTEXTUAL_STM)?;
        let encoder_kind = parse_encoder(&s.encoder)?;
        let covariate_mode = parse_covariate_mode(&s.covariate_mode)?;
        let model = if let (true, Some(beta)) = (s.fitted, s.w_beta) {
            let k = s.w_k.unwrap();
            let v = s.w_v.unwrap();
            let e = s.w_e.unwrap();
            let hidden = s.w_hidden.unwrap();
            let weights = prodlda::Weights {
                v,
                e,
                hidden,
                k,
                mode: encoder_kind.input_mode(),
                w1: s.w_w1.unwrap(),
                b1: s.w_b1.unwrap(),
                w2: s.w_w2.unwrap(),
                b2: s.w_b2.unwrap(),
                w_mu: s.w_w_mu.unwrap(),
                b_mu: s.w_b_mu.unwrap(),
                w_ls: s.w_w_ls.unwrap(),
                b_ls: s.w_b_ls.unwrap(),
                beta,
                w_adapt: s.w_w_adapt.unwrap_or_default(),
                b_adapt: s.w_b_adapt.unwrap_or_default(),
            };
            let bn_mu = prodlda::BatchNorm {
                running_mean: s.bn_running_mean.clone().unwrap_or_else(|| vec![0.0; k]),
                running_var: s.bn_running_var.clone().unwrap_or_else(|| vec![1.0; k]),
                ..prodlda::BatchNorm::new(k)
            };
            let base = prodlda::ProdldaModel {
                num_topics: k,
                num_types: v,
                doc_topic: s.doc_topic.unwrap_or_default(),
                bound: s.bound.unwrap_or(f64::NAN),
                bound_history: s.bound_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                epochs_run: s.epochs_run.unwrap_or(0),
                weights,
                bn_mu,
                bn_lv: None,
                prior: prodlda::Prior::Laplace,
            };
            Some(crate::contextual_stm::ContextualStmModel {
                base,
                prior_w: s.prior_w.unwrap_or_default(),
                n_covariates: s.n_covariates.unwrap_or(0),
                l2_prior_reg: s.l2_prior_reg,
                encoder: encoder_kind,
                covariate_mode,
                emb_dim: s.emb_dim.unwrap_or(e),
            })
        } else {
            None
        };
        Ok(ContextualSTM {
            num_topics: s.num_topics,
            encoder: s.encoder,
            covariate_mode: s.covariate_mode,
            hidden_size: s.hidden_size,
            alpha: s.alpha,
            dropout: s.dropout,
            batch_size: s.batch_size,
            lr: s.lr,
            l2_prior_reg: s.l2_prior_reg,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            covariates: None,
            covariate_names: if s.covariate_names.is_empty() {
                None
            } else {
                Some(s.covariate_names)
            },
            cov_mean: s.cov_mean,
            cov_std: s.cov_std,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ContextualSTM(num_topics={}, encoder={:?}, covariate_mode={:?}, fitted={})",
            self.num_topics, self.encoder, self.covariate_mode, self.fitted
        )
    }
}
