//! Python bindings for SCHOLAR (Card, Tan & Smith 2018) — prior-covariate path.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyType;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// SCHOLAR with prior (prevalence) covariates: a ProdLDA/AVITM VAE whose
/// document-topic prior mean is shifted by document metadata, `mu_0 = W . covariates`.
/// A covariate that co-occurs with a topic raises that topic's prevalence — the
/// neural analog of STM/DMR prevalence covariates, learned jointly with the topics
/// (not post-hoc). `covariate_effects` reads as a covariate-by-topic prevalence
/// matrix. Built on topica's ProdLDA backbone; the covariates also enter the encoder.
#[pyclass(module = "topica")]
pub struct Scholar {
    num_topics: usize,
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
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::scholar::ScholarModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScholarState {
    num_topics: usize,
    hidden_size: usize,
    alpha: f64,
    dropout: f64,
    batch_size: usize,
    lr: f64,
    l2_prior_reg: f64,
    convergence_tol: f64,
    seed: u64,
    covariate_names: Vec<String>,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    n_prior_covars: Option<usize>,
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
    bn_running_mean: Option<Vec<f64>>,
    bn_running_var: Option<Vec<f64>>,
}

impl Scholar {
    fn fitted_model(&self) -> PyResult<&crate::scholar::ScholarModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Resolve the covariate matrix for this call: prefer the one passed to `fit`,
    /// else the one given at construction. Validate row count against the corpus and
    /// that every row has the same width, and return `(matrix, n_prior_covars)`.
    fn resolve_covariates(
        &self,
        provided: Option<&Bound<'_, PyAny>>,
        num_docs: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let pcs = match provided {
            Some(obj) => parse_features(obj)?,
            None => self.covariates.clone().ok_or_else(|| {
                PyValueError::new_err(
                    "covariates are required: pass covariates= to Scholar(...) or to fit()",
                )
            })?,
        };
        if pcs.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "covariates has {} rows but there are {} documents",
                pcs.len(),
                num_docs
            )));
        }
        if pcs.is_empty() {
            return Err(PyValueError::new_err("covariates matrix is empty"));
        }
        let width = pcs[0].len();
        if width == 0 {
            return Err(PyValueError::new_err(
                "covariates must have at least one column",
            ));
        }
        if pcs.iter().any(|r| r.len() != width) {
            return Err(PyValueError::new_err(
                "all covariate rows must have the same number of columns",
            ));
        }
        Ok(pcs)
    }
}

#[pymethods]
impl Scholar {
    /// Create an unfitted SCHOLAR. `covariates` (a `(num_docs, n_covars)` numeric
    /// matrix — numpy array, list of lists, or a numeric pandas/Polars frame) may be
    /// given here or at :meth:`fit`. `covariate_names` labels the covariate columns.
    /// `alpha` is the Dirichlet concentration behind the prior variance (reference
    /// 1.0); `hidden_size` the encoder width (100); `dropout` the encoder/`theta`
    /// dropout (0.2); `batch_size`/`lr` drive Adam (200/0.002, `beta1 = 0.99`);
    /// `l2_prior_reg` is the L2 penalty on the covariate weights (reference 0.0);
    /// `convergence_tol > 0` stops early on the relative epoch-ELBO change. Pass
    /// `iters` to :meth:`fit` for the number of epochs.
    #[new]
    #[pyo3(signature = (num_topics, *, covariates=None, covariate_names=None, alpha=1.0,
                        hidden_size=100, dropout=0.2, batch_size=200, lr=0.002,
                        l2_prior_reg=0.0, convergence_tol=0.0, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
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
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
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
        let covariates = match covariates {
            Some(obj) => Some(parse_features(obj)?),
            None => None,
        };
        if let (Some(pcs), Some(names)) = (&covariates, &covariate_names) {
            if !pcs.is_empty() && names.len() != pcs[0].len() {
                return Err(PyValueError::new_err(format!(
                    "covariate_names has length {} but covariates has {} columns",
                    names.len(),
                    pcs[0].len()
                )));
            }
        }
        Ok(Scholar {
            num_topics,
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
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with prior `covariates`
    /// (one row per document). `iters` sets the number of epochs (default 200).
    /// `convergence_tol` overrides the constructor value for this run.
    #[pyo3(signature = (data, *, covariates=None, iters=None, convergence_tol=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        covariates: Option<&Bound<'_, PyAny>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
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
        let pcs = self.resolve_covariates(covariates, corpus.docs.len())?;
        let n_prior_covars = pcs[0].len();
        let num_types = corpus.num_types();
        if num_types < self.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        // Default names if none were supplied and the width now differs.
        let names = match &self.covariate_names {
            Some(n) if n.len() == n_prior_covars => n.clone(),
            _ => (0..n_prior_covars)
                .map(|c| format!("covariate_{c}"))
                .collect(),
        };
        let tol = convergence_tol.unwrap_or(self.convergence_tol);
        let ep = iters.unwrap_or(200);
        let (k, h, a, dp, bs, lr, l2) = (
            self.num_topics,
            self.hidden_size,
            self.alpha,
            self.dropout,
            self.batch_size,
            self.lr,
            self.l2_prior_reg,
        );
        let seed = self.seed;
        let (model, corpus) = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let m = crate::scholar::fit_scholar(
                &corpus.docs,
                &pcs,
                k,
                num_types,
                n_prior_covars,
                h,
                a,
                dp,
                ep,
                bs,
                lr,
                l2,
                tol,
                &mut rng,
            );
            (m, corpus)
        });
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.covariate_names = Some(names);
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
        Ok(vecs_to_arr2(&self.fitted_model()?.base.doc_topic).to_pyarray_bound(py))
    }

    /// Covariate-by-topic prevalence effects `(n_covars, num_topics)`. Entry
    /// ``[c][t]`` is how much covariate `c` shifts the log-prior mean of topic `t`
    /// (positive raises that topic's prevalence for documents high on covariate `c`).
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

    /// Per-epoch ELBO trajectory.
    #[getter]
    fn bound_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.base.bound_history.clone())
    }

    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.base.converged)
    }

    /// Fit history: ``(epoch, elbo)`` pairs, one per training epoch.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .base
            .bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }

    #[getter]
    fn epochs_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.base.epochs_run)
    }

    #[getter]
    fn model_family(&self) -> &'static str {
        "none"
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

    /// Top `n` words per topic as ``(word, probability)`` pairs (or one topic's list
    /// when `topic` is given).
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

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Held-out topic proportions for new documents. `covariates` (one row per
    /// document) enter the encoder exactly as at fit time and must have the same
    /// number of columns. Returns `(num_docs, num_topics)`.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        covariates: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let pcs = self.resolve_covariates(Some(covariates), docs.len())?;
        if pcs[0].len() != m.n_prior_covars {
            return Err(PyValueError::new_err(format!(
                "covariates has {} columns but the model was fit with {}",
                pcs[0].len(),
                m.n_prior_covars
            )));
        }
        Ok(vecs_to_arr2(&m.transform(&docs, &pcs)).to_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        let ncov = self
            .model
            .as_ref()
            .map(|m| m.n_prior_covars)
            .or_else(|| {
                self.covariates
                    .as_ref()
                    .and_then(|c| c.first())
                    .map(|r| r.len())
            })
            .unwrap_or(0);
        format!(
            "Scholar(num_topics={}, n_covariates={}, fitted={})",
            self.num_topics, ncov, self.fitted
        )
    }

    /// Save the fitted model to `path`. Reload with `Scholar.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_SCHOLAR,
            &ScholarState {
                num_topics: self.num_topics,
                hidden_size: self.hidden_size,
                alpha: self.alpha,
                dropout: self.dropout,
                batch_size: self.batch_size,
                lr: self.lr,
                l2_prior_reg: self.l2_prior_reg,
                convergence_tol: self.convergence_tol,
                seed: self.seed,
                covariate_names: self.covariate_names.clone().unwrap_or_default(),
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                n_prior_covars: Some(m.n_prior_covars),
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
                bn_running_mean: Some(m.base.bn_mu.running_mean.clone()),
                bn_running_var: Some(m.base.bn_mu.running_var.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: ScholarState = read_state(path, MODEL_TAG_SCHOLAR)?;
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
                mode: prodlda::InputMode::BowEmb,
                w1: s.w_w1.unwrap(),
                b1: s.w_b1.unwrap(),
                w2: s.w_w2.unwrap(),
                b2: s.w_b2.unwrap(),
                w_mu: s.w_w_mu.unwrap(),
                b_mu: s.w_b_mu.unwrap(),
                w_ls: s.w_w_ls.unwrap(),
                b_ls: s.w_b_ls.unwrap(),
                beta,
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
                prior: prodlda::Prior::Laplace,
            };
            Some(crate::scholar::ScholarModel {
                base,
                prior_w: s.prior_w.unwrap_or_default(),
                n_prior_covars: s.n_prior_covars.unwrap_or(0),
                l2_prior_reg: s.l2_prior_reg,
            })
        } else {
            None
        };
        Ok(Scholar {
            num_topics: s.num_topics,
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
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }
}
