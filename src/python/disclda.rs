//! Python bindings for DiscLDA (Lacoste-Julien, Sha & Jordan 2008).

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use pyo3::types::PyType;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// DiscLDA (Lacoste-Julien, Sha & Jordan 2008): a discriminative topic model. The
/// actual topics partition into `k_class` topics specific to each class (one block
/// per class) plus `k_shared` shared topics; a document of a given class uses only
/// its class block and the shared block. This yields topics that separate what each
/// class talks about distinctively from the common ground, and a document
/// representation that carries the class signal (`transform`/`predict`). Fixed
/// block-transform variant (paper §4.1).
#[pyclass(module = "topica")]
pub struct DiscLDA {
    k_class: usize,
    k_shared: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    infer_sweeps: usize,
    seed: u64,
    // Class-prior policy for the direct classifier: "empirical" (default,
    // observed class frequencies), "uniform", or "custom" (per-class weights in
    // `class_prior_custom`, sorted-class order). Resolved to a log-prior at fit.
    class_prior_mode: String,
    class_prior_custom: Option<Vec<f64>>,
    // Observed class document counts (sorted-class order), set at fit.
    class_counts: Vec<usize>,
    fitted: bool,
    classes: Vec<String>,
    topic_names: Vec<String>,
    model: Option<crate::disclda::DiscLdaModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DiscLdaState {
    k_class: usize,
    k_shared: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    infer_sweeps: usize,
    seed: u64,
    fitted: bool,
    classes: Vec<String>,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    resolved_alpha: Option<f64>,
    num_topics: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    // Class-prior policy + the resolved log-prior (older saves default to uniform).
    #[serde(default = "disclda_prior_mode_legacy")]
    class_prior_mode: String,
    #[serde(default)]
    class_prior_custom: Option<Vec<f64>>,
    #[serde(default)]
    class_counts: Vec<usize>,
    #[serde(default)]
    class_log_prior: Option<Vec<f64>>,
}

fn disclda_prior_mode_legacy() -> String {
    "uniform".to_string()
}

/// Parse the `class_prior` constructor argument into (mode, custom weights).
fn parse_class_prior(spec: Option<&Bound<'_, PyAny>>) -> PyResult<(String, Option<Vec<f64>>)> {
    let Some(obj) = spec else {
        return Ok(("empirical".to_string(), None));
    };
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "empirical" | "uniform" => Ok((s, None)),
            other => Err(PyValueError::new_err(format!(
                "class_prior must be \"empirical\", \"uniform\", or a sequence of \
                 per-class weights, got {other:?}"
            ))),
        };
    }
    let v: Vec<f64> = obj.extract().map_err(|_| {
        PyValueError::new_err(
            "class_prior must be \"empirical\", \"uniform\", or a sequence of floats",
        )
    })?;
    if v.is_empty() || v.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(PyValueError::new_err(
            "class_prior weights must be non-empty, finite, and > 0",
        ));
    }
    Ok(("custom".to_string(), Some(v)))
}

/// Resolve the class-prior policy to a log-prior (length `num_classes`) given the
/// observed per-class document counts (sorted-class order).
fn resolve_log_prior(
    mode: &str,
    custom: &Option<Vec<f64>>,
    counts: &[usize],
    num_classes: usize,
) -> PyResult<Vec<f64>> {
    match mode {
        "uniform" => Ok(vec![-(num_classes as f64).ln(); num_classes]),
        "empirical" => {
            let total: usize = counts.iter().sum();
            let total = total.max(1) as f64;
            Ok(counts
                .iter()
                .map(|&c| ((c as f64).max(1e-300) / total).ln())
                .collect())
        }
        "custom" => {
            let w = custom
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("custom class_prior weights missing"))?;
            if w.len() != num_classes {
                return Err(PyValueError::new_err(format!(
                    "class_prior has {} weights but there are {num_classes} classes \
                     (weights are in sorted-class order)",
                    w.len(),
                )));
            }
            // Normalise stably. Dividing by the max weight before summing keeps the
            // denominator in `(0, num_classes]`, so an extreme-but-valid weight (e.g.
            // 1e308) cannot overflow the raw sum to +inf and collapse every log-prior
            // to -inf — which would make predict_proba NaN (#460). The max cancels, so
            // `(x/wmax) / Σ(x/wmax)` == `x / Σx` exactly; the largest-weight class
            // always keeps a finite log-prior, so the softmax stays well-defined.
            let wmax = w.iter().cloned().fold(0.0f64, f64::max);
            let s: f64 = w.iter().map(|&x| x / wmax).sum();
            Ok(w.iter().map(|&x| ((x / wmax) / s).ln()).collect())
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown class_prior mode {mode:?}"
        ))),
    }
}

impl DiscLDA {
    fn fitted_model(&self) -> PyResult<&crate::disclda::DiscLdaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn class_index(&self, label: &str) -> PyResult<usize> {
        self.classes.iter().position(|c| c == label).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown class {label:?}; known classes: {:?}",
                self.classes
            ))
        })
    }
}

/// Extract per-document class labels as strings from a Python sequence (accepts
/// str or int entries).
fn extract_labels(y: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(v) = y.extract::<Vec<String>>() {
        return Ok(v);
    }
    if let Ok(v) = y.extract::<Vec<i64>>() {
        return Ok(v.into_iter().map(|i| i.to_string()).collect());
    }
    Err(PyValueError::new_err(
        "y must be a sequence of class labels (str or int), one per document",
    ))
}

fn map_to_vocab(corpus: &corpus::Corpus, data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
    let index: HashMap<&str, u32> = corpus
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
            PyValueError::new_err("expected a Corpus or a list of token lists (list[list[str]])")
        })?
    };
    Ok(str_docs
        .into_iter()
        .map(|doc| {
            doc.iter()
                .filter_map(|w| index.get(w.as_str()).copied())
                .collect()
        })
        .collect())
}

#[pymethods]
impl DiscLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``alpha`` is ``None`` when left at its
    /// ``0.1`` default (resolved only at fit).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("k_class", self.k_class)?;
        d.set_item("k_shared", self.k_shared)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("iters", self.iters)?;
        d.set_item("infer_sweeps", self.infer_sweeps)?;
        match (&self.class_prior_mode, &self.class_prior_custom) {
            (mode, Some(w)) if mode == "custom" => d.set_item("class_prior", w.clone())?,
            (mode, _) => d.set_item("class_prior", mode)?,
        }
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted DiscLDA. `k_class` is the number of class-specific topics
    /// per class, `k_shared` the number of shared topics; the total topic count is
    /// `num_classes * k_class + k_shared`, with `num_classes` taken from the labels
    /// at fit. `alpha` defaults to 0.1 (per allowed topic), `beta` to 0.01.
    /// `infer_sweeps` is the restricted-Gibbs passes used per class in
    /// `transform`/`predict`.
    ///
    /// `class_prior` sets the prior the direct classifier combines with each
    /// document's plug-in likelihood: ``"empirical"`` (default) uses the observed
    /// class frequencies from fit, so `predict_proba` is calibrated to class
    /// prevalence; ``"uniform"`` gives every class equal prior; or pass a sequence
    /// of positive per-class weights (in the sorted-class order of `classes`),
    /// which is normalised.
    #[new]
    #[pyo3(signature = (k_class, k_shared, *, alpha=None, beta=0.01, iters=1000,
                        infer_sweeps=100, class_prior=None, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_class: usize,
        k_shared: usize,
        alpha: Option<f64>,
        beta: f64,
        iters: usize,
        infer_sweeps: usize,
        class_prior: Option<&Bound<'_, PyAny>>,
        seed: u64,
    ) -> PyResult<Self> {
        let (class_prior_mode, class_prior_custom) = parse_class_prior(class_prior)?;
        if k_class == 0 {
            return Err(PyValueError::new_err("k_class must be >= 1"));
        }
        if k_shared == 0 {
            return Err(PyValueError::new_err("k_shared must be >= 1"));
        }
        // `<= 0.0` is false for NaN and +inf, so those slipped through and produced
        // NaN topic-word / doc-topic / predict_proba. Require finite and positive (#460).
        if let Some(a) = alpha {
            if !a.is_finite() || a <= 0.0 {
                return Err(PyValueError::new_err("alpha must be finite and > 0.0"));
            }
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be finite and > 0.0"));
        }
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be > 0"));
        }
        if infer_sweeps == 0 {
            return Err(PyValueError::new_err("infer_sweeps must be > 0"));
        }
        Ok(DiscLDA {
            k_class,
            k_shared,
            alpha,
            beta,
            iters,
            infer_sweeps,
            class_prior_mode,
            class_prior_custom,
            class_counts: Vec::new(),
            seed,
            fitted: false,
            classes: Vec::new(),
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on documents with a per-document class label `y` (str or int, one per
    /// document). Classes are sorted to a fixed order.
    #[pyo3(signature = (data, y, *, iters=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        iters: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let labels_str = extract_labels(y)?;
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
        if labels_str.len() != corpus.docs.len() {
            return Err(PyValueError::new_err(format!(
                "y has {} labels but there are {} documents",
                labels_str.len(),
                corpus.docs.len()
            )));
        }
        // Sorted unique classes -> index map.
        let mut classes: Vec<String> = labels_str.clone();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(PyValueError::new_err("need at least 2 distinct classes"));
        }
        let cindex: HashMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        let labels: Vec<usize> = labels_str.iter().map(|l| cindex[l.as_str()]).collect();

        let num_types = corpus.num_types();
        let num_classes = classes.len();
        let l = num_classes * slf.k_class + slf.k_shared;
        if num_types < l {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_classes*k_class + k_shared words",
            ));
        }
        let iters = iters.unwrap_or(slf.iters);
        // The constructor rejects iters == 0, but a per-fit override bypassed it.
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be > 0"));
        }
        let alpha = slf.alpha.unwrap_or(0.1);
        let (k_class, k_shared, beta, seed) = (slf.k_class, slf.k_shared, slf.beta, slf.seed);

        // Observed per-class document counts (sorted-class order) — the empirical
        // prior and the reported `class_counts`.
        let mut class_counts = vec![0usize; num_classes];
        for &c in &labels {
            class_counts[c] += 1;
        }
        let class_log_prior = resolve_log_prior(
            &slf.class_prior_mode,
            &slf.class_prior_custom,
            &class_counts,
            num_classes,
        )?;

        let (mut model, corpus) = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let m = crate::disclda::fit_disclda(
                &corpus.docs,
                &labels,
                num_classes,
                k_class,
                k_shared,
                num_types,
                alpha,
                beta,
                iters,
                &mut rng,
            );
            (m, corpus)
        });
        model.class_log_prior = class_log_prior;
        slf.class_counts = class_counts;
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.classes = classes;
        slf.topic_names = slf.build_topic_names();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// The class-marginalized discriminative representation Σ_c p(c|w)·θ_c for new
    /// documents (num_docs, num_topics) -- the feature vector for a downstream
    /// classifier or visualization. Cost is O(num_classes · infer_sweeps) per
    /// document (one restricted-Gibbs inference per class), single-threaded; for
    /// many-class corpora inference can dominate fitting -- lower `infer_sweeps` if
    /// it is too slow.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let mapped = map_to_vocab(self.corpus.as_ref().unwrap(), data)?;
        let sweeps = self.infer_sweeps;
        let seed = self.seed;
        let rep: Vec<Vec<f64>> = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            mapped
                .iter()
                .map(|doc| crate::disclda::predict_doc(doc, m, sweeps, &mut rng).1)
                .collect()
        });
        Ok(vecs_to_arr2(&rep).to_pyarray_bound(py))
    }

    /// Predict the class label of each document (argmax of `predict_proba`). A
    /// document with no in-vocabulary tokens carries no likelihood signal, so its
    /// prediction is the most probable class under `class_prior` (the majority
    /// class for the default empirical prior).
    fn predict(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let m = self.fitted_model()?;
        let mapped = map_to_vocab(self.corpus.as_ref().unwrap(), data)?;
        let sweeps = self.infer_sweeps;
        let seed = self.seed;
        let classes = self.classes.clone();
        Ok(py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            mapped
                .iter()
                .map(|doc| {
                    let (post, _) = crate::disclda::predict_doc(doc, m, sweeps, &mut rng);
                    let best = (0..post.len())
                        .max_by(|&a, &b| post[a].total_cmp(&post[b]))
                        .unwrap();
                    classes[best].clone()
                })
                .collect()
        }))
    }

    /// Approximate class posterior probabilities for each document
    /// (num_docs, num_classes), columns in `classes` order.
    ///
    /// This is a topica-native **plug-in** classifier, not a fully marginalized
    /// DiscLDA evidence: each class score is the posterior-mean-θ likelihood of the
    /// document under that class, combined with `class_prior` (empirical class
    /// frequencies by default) and softmaxed. Treat it as a well-behaved,
    /// prior-calibrated score rather than an exact p(class | words).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let mapped = map_to_vocab(self.corpus.as_ref().unwrap(), data)?;
        let sweeps = self.infer_sweeps;
        let seed = self.seed;
        let proba: Vec<Vec<f64>> = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            mapped
                .iter()
                .map(|doc| crate::disclda::predict_doc(doc, m, sweeps, &mut rng).0)
                .collect()
        });
        Ok(vecs_to_arr2(&proba).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_topics)
    }

    /// The class labels, in the fixed (sorted) order used for topic blocks and
    /// `predict_proba` columns.
    #[getter]
    fn classes(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.classes.clone())
    }

    /// The resolved class prior p(c) used by `predict`/`predict_proba`, length
    /// `num_classes` in `classes` order (sums to 1). Empirical class frequencies
    /// by default; uniform or the normalised custom weights otherwise.
    #[getter]
    fn class_prior(&self) -> PyResult<Vec<f64>> {
        let m = self.fitted_model()?;
        Ok(m.class_log_prior.iter().map(|&lp| lp.exp()).collect())
    }

    /// The observed per-class document counts from fit, in `classes` order.
    #[getter]
    fn class_counts(&self) -> PyResult<Vec<usize>> {
        self.fitted_model()?;
        Ok(self.class_counts.clone())
    }

    /// Topic-word matrix φ (num_topics, vocab); rows sum to 1. Topics are ordered
    /// class-0 block, class-1 block, ..., then the shared block.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }

    /// Document-topic matrix θ (num_docs, num_topics); rows sum to 1, mass only on
    /// each document's class block and the shared block.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// The topic indices specific to a class (its block), by class label.
    fn class_topic_ids(&self, label: &str) -> PyResult<Vec<usize>> {
        let m = self.fitted_model()?;
        let c = self.class_index(label)?;
        Ok(m.class_block(c).collect())
    }

    /// The shared topic indices.
    fn shared_topic_ids(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.shared_block().collect())
    }

    /// Top-`n` words for the topics specific to a class.
    #[pyo3(signature = (label, n=10))]
    fn class_topics<'py>(
        &self,
        py: Python<'py>,
        label: &str,
        n: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let c = self.class_index(label)?;
        self.topics_subset(py, m, m.class_block(c).collect(), n)
    }

    /// Top-`n` words for the shared topics.
    #[pyo3(signature = (n=10))]
    fn shared_topics<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        self.topics_subset(py, m, m.shared_block().collect(), n)
    }

    #[getter]
    fn model_family(&self) -> &'static str {
        "none"
    }

    /// Fit history (iteration, objective); empty -- DiscLDA keeps no bound trace.
    #[getter]
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// Convergence flag; `None` -- DiscLDA runs a fixed number of Gibbs sweeps.
    #[getter]
    fn converged(&self) -> Option<bool> {
        None
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    #[getter]
    fn early_stopped(&self) -> Option<bool> {
        None
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

    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let l = self.fitted_model()?.num_topics;
        if names.len() != l {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {l} (got {})",
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Top-`n` words per topic (all topics; class blocks then shared block).
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            m.num_topics,
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
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        coherence_dispatch(
            py,
            self.corpus.as_ref().unwrap(),
            &tops,
            n,
            &coherence_type,
            texts,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscLDA(k_class={}, k_shared={}, classes={}, fitted={})",
            self.k_class,
            self.k_shared,
            self.classes.len(),
            self.fitted
        )
    }

    /// Save the fitted model to `path`. Reload with `DiscLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_DISCLDA,
            &DiscLdaState {
                k_class: self.k_class,
                k_shared: self.k_shared,
                alpha: self.alpha,
                beta: self.beta,
                iters: self.iters,
                infer_sweeps: self.infer_sweeps,
                seed: self.seed,
                fitted: self.fitted,
                classes: self.classes.clone(),
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                num_types: Some(m.num_types),
                resolved_alpha: Some(m.alpha),
                num_topics: Some(m.num_topics),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                class_prior_mode: self.class_prior_mode.clone(),
                class_prior_custom: self.class_prior_custom.clone(),
                class_counts: self.class_counts.clone(),
                class_log_prior: Some(m.class_log_prior.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: DiscLdaState = read_state(path, MODEL_TAG_DISCLDA)?;
        let num_classes = s.classes.len();
        let model = if s.fitted && s.topic_word.is_some() {
            // Older saves have no class_log_prior; fall back to uniform.
            let clp = s
                .class_log_prior
                .clone()
                .unwrap_or_else(|| vec![-(num_classes as f64).ln(); num_classes]);
            Some(crate::disclda::DiscLdaModel {
                num_classes,
                k_class: s.k_class,
                k_shared: s.k_shared,
                num_types: s.num_types.unwrap_or(0),
                alpha: s.resolved_alpha.unwrap_or(0.1),
                beta: s.beta,
                num_topics: s.num_topics.unwrap_or(num_classes * s.k_class + s.k_shared),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                class_log_prior: clp,
            })
        } else {
            None
        };
        Ok(DiscLDA {
            k_class: s.k_class,
            k_shared: s.k_shared,
            alpha: s.alpha,
            beta: s.beta,
            iters: s.iters,
            infer_sweeps: s.infer_sweeps,
            class_prior_mode: s.class_prior_mode,
            class_prior_custom: s.class_prior_custom,
            class_counts: s.class_counts,
            seed: s.seed,
            fitted: s.fitted,
            classes: s.classes,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }
}

impl DiscLDA {
    /// Default topic names: `<class>_0`, ... for class blocks; `shared_0`, ... for
    /// the shared block.
    fn build_topic_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for c in &self.classes {
            for i in 0..self.k_class {
                names.push(format!("{c}_{i}"));
            }
        }
        for i in 0..self.k_shared {
            names.push(format!("shared_{i}"));
        }
        names
    }

    fn topics_subset<'py>(
        &self,
        py: Python<'py>,
        m: &crate::disclda::DiscLdaModel,
        ids: Vec<usize>,
        n: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let sub: Vec<Vec<f64>> = ids.iter().map(|&t| m.topic_word[t].clone()).collect();
        let phi = vecs_to_arr2(&sub);
        topic_words_helper(py, &phi, vocab, ids.len(), n, None, true)
    }
}
