//! Python bindings for SCHOLAR (Card, Tan & Smith 2018) — prior-covariate path.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Extract per-document class labels (str or int) as strings, one per document.
fn extract_labels(y: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(v) = y.extract::<Vec<String>>() {
        return Ok(v);
    }
    if let Ok(v) = y.extract::<Vec<i64>>() {
        return Ok(v.into_iter().map(|i| i.to_string()).collect());
    }
    Err(PyValueError::new_err(
        "labels must be a sequence of class labels (str or int), one per document",
    ))
}

/// SCHOLAR (Card, Tan & Smith 2018): a ProdLDA/AVITM VAE with document metadata in
/// three roles. Prior (prevalence) `covariates` shift the topic-prior mean, so a
/// covariate that co-occurs with a topic raises its prevalence (neural STM/DMR
/// prevalence; `covariate_effects`). Supervised `labels` add a softmax classifier off
/// `theta`, shaping the topics to be predictive (neural sLDA;
/// `predict`/`predict_proba`/`classes`). Content (topic-covariate) `content` adds
/// per-covariate word deviations to the decoder, so the same topic is worded
/// differently across groups (neural SAGE; `content_effects`). All three compose.
/// Built on topica's ProdLDA backbone.
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
    l1_content_reg: f64,
    interactions: bool,
    // Covariates optionally supplied at construction (else at fit).
    covariates: Option<Vec<Vec<f64>>>,
    covariate_names: Option<Vec<String>>,
    // Content (topic-covariate) covariates optionally supplied at construction.
    content: Option<Vec<Vec<f64>>>,
    content_names: Option<Vec<String>>,
    // Sorted unique class labels (empty when fit without labels), the order of
    // `predict_proba` columns and `classes`.
    classes: Vec<String>,
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
    #[serde(default)]
    content_names: Vec<String>,
    #[serde(default)]
    l1_content_reg: f64,
    #[serde(default)]
    interactions: bool,
    #[serde(default)]
    classes: Vec<String>,
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
    // Label classifier head (empty when fit without labels).
    #[serde(default)]
    wc: Option<Vec<f64>>,
    #[serde(default)]
    bc: Option<Vec<f64>>,
    #[serde(default)]
    n_labels: Option<usize>,
    // Content decoder deviations (empty when fit without content covariates).
    #[serde(default)]
    beta_c: Option<Vec<f64>>,
    #[serde(default)]
    beta_ci: Option<Vec<f64>>,
    #[serde(default)]
    n_topic_covars: Option<usize>,
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

    /// Resolve covariates for a prediction call: empty rows when the model was fit
    /// without covariates, else the parsed matrix validated to the fitted width.
    fn pred_covariates(
        &self,
        covariates: Option<&Bound<'_, PyAny>>,
        num_docs: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n_pc = self.model.as_ref().map(|m| m.n_prior_covars).unwrap_or(0);
        if n_pc == 0 {
            return Ok(vec![Vec::new(); num_docs]);
        }
        let pcs = self.resolve_covariates(covariates, num_docs)?;
        if pcs[0].len() != n_pc {
            return Err(PyValueError::new_err(format!(
                "covariates has {} columns but the model was fit with {n_pc}",
                pcs[0].len()
            )));
        }
        Ok(pcs)
    }

    /// Resolve the content (topic-covariate) matrix for this call: prefer the one
    /// passed here, else the one given at construction; validate row count and width.
    fn resolve_content(
        &self,
        provided: Option<&Bound<'_, PyAny>>,
        num_docs: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let tcs = match provided {
            Some(obj) => parse_features(obj)?,
            None => self.content.clone().ok_or_else(|| {
                PyValueError::new_err(
                    "content is required: pass content= to Scholar(...) or to fit()",
                )
            })?,
        };
        if tcs.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "content has {} rows but there are {num_docs} documents",
                tcs.len()
            )));
        }
        if tcs.is_empty() || tcs[0].is_empty() {
            return Err(PyValueError::new_err(
                "content must have at least one column",
            ));
        }
        let width = tcs[0].len();
        if tcs.iter().any(|r| r.len() != width) {
            return Err(PyValueError::new_err(
                "all content rows must have the same number of columns",
            ));
        }
        Ok(tcs)
    }

    /// Resolve content for a prediction call: empty rows when the model was fit
    /// without content, else the parsed matrix validated to the fitted width.
    fn pred_content(
        &self,
        content: Option<&Bound<'_, PyAny>>,
        num_docs: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n_tc = self.model.as_ref().map(|m| m.n_topic_covars).unwrap_or(0);
        if n_tc == 0 {
            return Ok(vec![Vec::new(); num_docs]);
        }
        let tcs = self.resolve_content(content, num_docs)?;
        if tcs[0].len() != n_tc {
            return Err(PyValueError::new_err(format!(
                "content has {} columns but the model was fit with {n_tc}",
                tcs[0].len()
            )));
        }
        Ok(tcs)
    }
}

#[pymethods]
impl Scholar {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). The numeric data arrays ``covariates``
    /// and ``content`` are excluded; the config-name lists ``covariate_names`` and
    /// ``content_names`` (a list of strings, or ``None`` when unset) are kept.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("covariate_names", self.covariate_names.clone())?;
        d.set_item("content_names", self.content_names.clone())?;
        d.set_item("interactions", self.interactions)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("hidden_size", self.hidden_size)?;
        d.set_item("dropout", self.dropout)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("lr", self.lr)?;
        d.set_item("l2_prior_reg", self.l2_prior_reg)?;
        d.set_item("l1_content_reg", self.l1_content_reg)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

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
    #[pyo3(signature = (num_topics, *, covariates=None, covariate_names=None, content=None,
                        content_names=None, interactions=false, alpha=1.0, hidden_size=100,
                        dropout=0.2, batch_size=200, lr=0.002, l2_prior_reg=0.0,
                        l1_content_reg=0.0, convergence_tol=0.0, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        covariates: Option<&Bound<'_, PyAny>>,
        covariate_names: Option<Vec<String>>,
        content: Option<&Bound<'_, PyAny>>,
        content_names: Option<Vec<String>>,
        interactions: bool,
        alpha: f64,
        hidden_size: usize,
        dropout: f64,
        batch_size: usize,
        lr: f64,
        l2_prior_reg: f64,
        l1_content_reg: f64,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if !finite_pos(alpha) {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
        if !(0.0..1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be in [0, 1)"));
        }
        if !(l2_prior_reg >= 0.0 && l2_prior_reg.is_finite()) {
            return Err(PyValueError::new_err(
                "l2_prior_reg must be >= 0 and finite",
            ));
        }
        if !(l1_content_reg >= 0.0 && l1_content_reg.is_finite()) {
            return Err(PyValueError::new_err(
                "l1_content_reg must be >= 0 and finite",
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
        let content = match content {
            Some(obj) => Some(parse_features(obj)?),
            None => None,
        };
        if let (Some(tcs), Some(names)) = (&content, &content_names) {
            if !tcs.is_empty() && names.len() != tcs[0].len() {
                return Err(PyValueError::new_err(format!(
                    "content_names has length {} but content has {} columns",
                    names.len(),
                    tcs[0].len()
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
            l1_content_reg,
            interactions,
            convergence_tol,
            seed,
            covariates,
            covariate_names,
            content,
            content_names,
            classes: Vec::new(),
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with prior `covariates`,
    /// supervised `labels` (str/int, one per document), and/or topic-covariate
    /// `content` (a numeric matrix). At least one of covariates, labels, or content
    /// must be given. `iters` sets the number of epochs (default 200);
    /// `convergence_tol` overrides the constructor value for this run.
    #[pyo3(signature = (data, *, covariates=None, labels=None, content=None, iters=None,
                        convergence_tol=None, progress=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        covariates: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
        progress: Option<PyObject>,
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

        // Labels (optional): map to sorted-unique class indices.
        let (label_idx, classes) = match labels {
            Some(y) => {
                let labels_str = extract_labels(y)?;
                if labels_str.len() != num_docs {
                    return Err(PyValueError::new_err(format!(
                        "labels has {} entries but there are {num_docs} documents",
                        labels_str.len()
                    )));
                }
                let mut classes: Vec<String> = labels_str.clone();
                classes.sort();
                classes.dedup();
                if classes.len() < 2 {
                    return Err(PyValueError::new_err("need at least 2 distinct labels"));
                }
                let cindex: std::collections::HashMap<&str, usize> = classes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.as_str(), i))
                    .collect();
                let idx: Vec<usize> = labels_str.iter().map(|l| cindex[l.as_str()]).collect();
                (Some(idx), classes)
            }
            None => (None, Vec::new()),
        };
        let n_labels = classes.len();

        // Covariates / content (optional): empty rows when absent. Require at least
        // one of covariates / labels / content so the model has metadata to condition on.
        let has_covars = covariates.is_some() || slf.covariates.is_some();
        let has_content = content.is_some() || slf.content.is_some();
        if !has_covars && label_idx.is_none() && !has_content {
            return Err(PyValueError::new_err(
                "Scholar needs covariates, labels, and/or content: pass covariates=, labels=, \
                 and/or content=",
            ));
        }
        let pcs = if has_covars {
            slf.resolve_covariates(covariates, num_docs)?
        } else {
            vec![Vec::new(); num_docs]
        };
        let n_prior_covars = pcs[0].len();
        let tcs = if has_content {
            slf.resolve_content(content, num_docs)?
        } else {
            vec![Vec::new(); num_docs]
        };
        let n_topic_covars = tcs[0].len();

        let num_types = corpus.num_types();
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        let names = match &slf.covariate_names {
            Some(n) if n.len() == n_prior_covars => n.clone(),
            _ => (0..n_prior_covars)
                .map(|c| format!("covariate_{c}"))
                .collect(),
        };
        let content_names = match &slf.content_names {
            Some(n) if n.len() == n_topic_covars => n.clone(),
            _ => (0..n_topic_covars)
                .map(|c| format!("content_{c}"))
                .collect(),
        };
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        let ep = iters.unwrap_or(200);
        let interactions = slf.interactions && n_topic_covars > 0;
        let (k, h, a, dp, bs, lr, l2, l1c) = (
            slf.num_topics,
            slf.hidden_size,
            slf.alpha,
            slf.dropout,
            slf.batch_size,
            slf.lr,
            slf.l2_prior_reg,
            slf.l1_content_reg,
        );
        let seed = slf.seed;
        let progress = resolve_progress(py, progress, "Scholar");
        let (model, corpus) = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut on_progress = |it: usize, total: usize, ll: f64| {
                if let Some(cb) = &progress {
                    Python::with_gil(|py| emit_progress(py, cb, it, total, ll));
                }
            };
            let m = crate::scholar::fit_scholar(
                &corpus.docs,
                &pcs,
                label_idx.as_deref(),
                n_labels,
                &tcs,
                n_topic_covars,
                interactions,
                l1c,
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
                &mut on_progress,
                &mut rng,
            );
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.covariate_names = Some(names);
        slf.content_names = Some(content_names);
        slf.classes = classes;
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

    /// Content (topic-covariate) word deviations `(n_content, vocab)`. Entry
    /// ``[c][j]`` is how much content covariate `c` shifts the unnormalized log-word
    /// weight of word `j` — the SAGE "same topic, worded differently across groups"
    /// deviations. Empty if fit without content covariates.
    #[getter]
    fn content_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.content_effects()).to_pyarray_bound(py))
    }

    /// The content covariate column names, in the order of `content_effects` rows.
    #[getter]
    fn content_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.content_names.clone().unwrap_or_default())
    }

    /// The class labels (sorted), in the order of `predict_proba` columns. Empty if
    /// the model was fit without labels.
    #[getter]
    fn classes(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.classes.clone())
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

    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.base.converged)
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn early_stopped(&self) -> PyResult<bool> {
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

    /// Top `n` words per topic (bare word strings), or one topic's list when
    /// `topic` is given. Pass ``weights=True`` for ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
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

    /// Held-out topic proportions for new documents. `covariates` and `content` (one
    /// row per document) enter the encoder exactly as at fit time and must have the
    /// same number of columns; pass `None` for either the model was not fit with.
    /// Returns `(num_docs, num_topics)`.
    #[pyo3(signature = (data, covariates=None, content=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        covariates: Option<&Bound<'py, PyAny>>,
        content: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let pcs = self.pred_covariates(covariates, docs.len())?;
        let tcs = self.pred_content(content, docs.len())?;
        Ok(vecs_to_arr2(&m.transform(&docs, &pcs, &tcs)).to_pyarray_bound(py))
    }

    /// Class-probability predictions `softmax(wc . theta + bc)` for new documents,
    /// shape `(num_docs, n_classes)`, columns in `classes` order. Requires a
    /// label-trained model. `covariates`/`content` as in `transform`.
    #[pyo3(signature = (data, covariates=None, content=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        covariates: Option<&Bound<'py, PyAny>>,
        content: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.n_labels == 0 {
            return Err(PyValueError::new_err(
                "model was fit without labels; predict_proba is unavailable",
            ));
        }
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let pcs = self.pred_covariates(covariates, docs.len())?;
        let tcs = self.pred_content(content, docs.len())?;
        Ok(vecs_to_arr2(&m.predict_proba(&docs, &pcs, &tcs)).to_pyarray_bound(py))
    }

    /// Predicted class label per document (argmax of `predict_proba`), as strings in
    /// `classes` space. Requires a label-trained model.
    #[pyo3(signature = (data, covariates=None, content=None))]
    fn predict(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        covariates: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<String>> {
        let m = self.fitted_model()?;
        if m.n_labels == 0 {
            return Err(PyValueError::new_err(
                "model was fit without labels; predict is unavailable",
            ));
        }
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let pcs = self.pred_covariates(covariates, docs.len())?;
        let tcs = self.pred_content(content, docs.len())?;
        let _ = py;
        let proba = m.predict_proba(&docs, &pcs, &tcs);
        Ok(proba
            .iter()
            .map(|p| {
                let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
                self.classes[best].clone()
            })
            .collect())
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
                content_names: self.content_names.clone().unwrap_or_default(),
                l1_content_reg: self.l1_content_reg,
                interactions: self.interactions,
                classes: self.classes.clone(),
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
                wc: Some(m.wc.clone()),
                bc: Some(m.bc.clone()),
                n_labels: Some(m.n_labels),
                beta_c: Some(m.beta_c.clone()),
                beta_ci: m.beta_ci.clone(),
                n_topic_covars: Some(m.n_topic_covars),
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
                // SCHOLAR uses the simple-concat `BowEmb` encoder (no `adapt_bert`).
                w_adapt: Vec::new(),
                b_adapt: Vec::new(),
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
                // SCHOLAR uses the Laplace transform (softmax(mu)); bn_lv is unused.
                bn_lv: None,
                prior: prodlda::Prior::Laplace,
            };
            Some(crate::scholar::ScholarModel {
                base,
                prior_w: s.prior_w.unwrap_or_default(),
                n_prior_covars: s.n_prior_covars.unwrap_or(0),
                l2_prior_reg: s.l2_prior_reg,
                wc: s.wc.unwrap_or_default(),
                bc: s.bc.unwrap_or_default(),
                n_labels: s.n_labels.unwrap_or(0),
                beta_c: s.beta_c.unwrap_or_default(),
                beta_ci: s.beta_ci,
                n_topic_covars: s.n_topic_covars.unwrap_or(0),
                interactions: s.interactions,
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
            l1_content_reg: s.l1_content_reg,
            interactions: s.interactions,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            covariates: None,
            covariate_names: if s.covariate_names.is_empty() {
                None
            } else {
                Some(s.covariate_names)
            },
            content: None,
            content_names: if s.content_names.is_empty() {
                None
            } else {
                Some(s.content_names)
            },
            classes: s.classes,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }
}
