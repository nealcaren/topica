//! IdealPointTM pyclass: a topic model with a latent ideal-point head. Each author
//! gets a low-dimensional position that displaces within-topic word choice, with a
//! per-topic discrimination. The position is latent and estimated.
//!
//! IdealPointTM consumes word tokens in one of two representations, selected by
//! whether you pass `word_embeddings` to `fit`:
//!   - omit them and the topic-word matrix is parameterized directly over the
//!     vocabulary (counts; "Wordfish with topics"), the count core
//!     `crate::idealpoint_lda`;
//!   - pass them and the matrix is factored through word embeddings, exactly as
//!     ETM, the embedded core `crate::idealpoint`.
//! Both are the same model; the embedding is a low-rank factorization of the same
//! displaced multinomial. The two cores keep separate, optimized inner loops, so
//! the pyclass holds either behind the `IpInner` enum. Experimental-tier, gated.
//! `use super::*` pulls in the shared bindings (parse_features, vecs_to_arr2,
//! coherence helpers, save/load, the experimental gate).

use super::*;
use pyo3::types::PyDict;

use crate::idealpoint::{self, IdealPointModel};
use crate::idealpoint_lda::{self, IdealPointLdaModel};
use std::collections::{HashMap, HashSet};

/// A fitted model in one of its two representations. Both cores expose the same
/// author/topic getters; only the topic-word parameterization differs.
enum IpInner {
    /// Word-embedding (ETM) representation, fitted with `word_embeddings`.
    Embedded(IdealPointModel),
    /// Count (direct-over-vocabulary) representation, fitted without embeddings.
    Count(IdealPointLdaModel),
}

impl IpInner {
    fn num_authors(&self) -> usize {
        match self {
            IpInner::Embedded(m) => m.num_authors,
            IpInner::Count(m) => m.num_authors,
        }
    }
    fn beta0(&self) -> &Vec<Vec<f64>> {
        match self {
            IpInner::Embedded(m) => &m.beta0,
            IpInner::Count(m) => &m.beta0,
        }
    }
    fn doc_topics(&self) -> Vec<Vec<f64>> {
        match self {
            IpInner::Embedded(m) => m.doc_topics(),
            IpInner::Count(m) => m.doc_topics(),
        }
    }
    fn x(&self) -> &Vec<Vec<f64>> {
        match self {
            IpInner::Embedded(m) => &m.x,
            IpInner::Count(m) => &m.x,
        }
    }
    fn group(&self) -> &Vec<usize> {
        match self {
            IpInner::Embedded(m) => &m.group,
            IpInner::Count(m) => &m.group,
        }
    }
    fn position_se(&self, ntot: &[Vec<f64>], x_prior_variance: f64) -> Vec<Vec<f64>> {
        match self {
            IpInner::Embedded(m) => m.position_se(ntot, x_prior_variance),
            IpInner::Count(m) => m.position_se(ntot, x_prior_variance),
        }
    }
    fn topic_discrimination(&self) -> Vec<f64> {
        match self {
            IpInner::Embedded(m) => m.topic_discrimination(),
            IpInner::Count(m) => m.topic_discrimination(),
        }
    }
    fn w(&self) -> &Vec<Vec<Vec<f64>>> {
        match self {
            IpInner::Embedded(m) => &m.w,
            IpInner::Count(m) => &m.w,
        }
    }
    fn position_topic_beta(&self, k: usize, x: &[f64]) -> Vec<f64> {
        match self {
            IpInner::Embedded(m) => m.position_topic_beta(k, x),
            IpInner::Count(m) => m.position_topic_beta(k, x),
        }
    }
    fn bound(&self) -> f64 {
        match self {
            IpInner::Embedded(m) => m.bound,
            IpInner::Count(m) => m.bound,
        }
    }
    fn bound_history(&self) -> &Vec<f64> {
        match self {
            IpInner::Embedded(m) => &m.bound_history,
            IpInner::Count(m) => &m.bound_history,
        }
    }
    fn converged(&self) -> bool {
        match self {
            IpInner::Embedded(m) => m.converged,
            IpInner::Count(m) => m.converged,
        }
    }
    fn em_iters_run(&self) -> usize {
        match self {
            IpInner::Embedded(m) => m.em_iters_run,
            IpInner::Count(m) => m.em_iters_run,
        }
    }
    /// `"word2vec"` for the embedded representation, `"counts"` for the count one.
    fn representation(&self) -> &'static str {
        match self {
            IpInner::Embedded(_) => "word2vec",
            IpInner::Count(_) => "counts",
        }
    }
}

#[pyclass(module = "topica")]
pub struct IdealPointTM {
    num_topics: usize,
    num_dims: usize,
    convergence_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    w_prior_variance: f64,
    x_prior_variance: f64,
    max_inner: usize,
    min_count: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    model: Option<IpInner>,
    corpus: Option<corpus::Corpus>,
}

// Tag 31: the word-embedding representation (carries the embedded core's base/disc).
#[derive(serde::Serialize, serde::Deserialize)]
struct IdealPointState {
    num_topics: usize,
    num_dims: usize,
    convergence_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    w_prior_variance: f64,
    x_prior_variance: f64,
    max_inner: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    // Fitted model fields (None when unfitted).
    num_types: Option<usize>,
    num_authors: Option<usize>,
    alpha: Option<Vec<Vec<f64>>>,
    w: Option<Vec<Vec<Vec<f64>>>>,
    x: Option<Vec<Vec<f64>>>,
    group: Option<Vec<usize>>,
    beta0: Option<Vec<Vec<f64>>>,
    base: Option<Vec<Vec<f64>>>,
    disc: Option<Vec<Vec<Vec<f64>>>>,
    mu: Option<Vec<f64>>,
    sigma: Option<Vec<f64>>,
    lambda: Option<Vec<Vec<f64>>>,
    bound: Option<f64>,
    bound_history: Option<Vec<f64>>,
    converged: Option<bool>,
    em_iters_run: Option<usize>,
}

// Tag 33: the count representation (no base/disc; carries min_count).
#[derive(serde::Serialize, serde::Deserialize)]
struct IdealPointLdaState {
    num_topics: usize,
    num_dims: usize,
    convergence_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    w_prior_variance: f64,
    x_prior_variance: f64,
    max_inner: usize,
    min_count: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    num_authors: Option<usize>,
    alpha: Option<Vec<Vec<f64>>>,
    w: Option<Vec<Vec<Vec<f64>>>>,
    x: Option<Vec<Vec<f64>>>,
    group: Option<Vec<usize>>,
    beta0: Option<Vec<Vec<f64>>>,
    mu: Option<Vec<f64>>,
    sigma: Option<Vec<f64>>,
    lambda: Option<Vec<Vec<f64>>>,
    bound: Option<f64>,
    bound_history: Option<Vec<f64>>,
    converged: Option<bool>,
    em_iters_run: Option<usize>,
}

impl IdealPointTM {
    fn fitted_model(&self) -> PyResult<&IpInner> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Per-author expected topic-token counts `n_{a,k}` (`A x K`), reconstructed
    /// from the stored corpus and document topic proportions:
    /// `n_{a,k} = sum_{d in a} L_d theta_{d,k}`, where `L_d` is the doc's in-vocab
    /// length. This is the data weight each author's position SE conditions on, and
    /// it is recoverable from saved state, so the SE is available after `load`.
    fn author_topic_counts(&self) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted_model()?;
        let theta = m.doc_topics();
        let group = m.group();
        let a_n = m.num_authors();
        let corpus = self
            .corpus
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        let mut ntot = vec![vec![0.0f64; self.num_topics]; a_n];
        for (d, theta_d) in theta.iter().enumerate() {
            let a = group[d];
            let len = corpus.docs[d].len() as f64;
            for (kk, &t) in theta_d.iter().enumerate() {
                ntot[a][kk] += len * t;
            }
        }
        Ok(ntot)
    }

    /// Shared front matter for both fit paths: resolve docs, author grouping, and
    /// anchors. Returns the token-string docs, doc names, the per-doc author index,
    /// the author label list, and the resolved anchor pairs.
    #[allow(clippy::type_complexity)]
    fn prepare_fit(
        data: &Bound<'_, PyAny>,
        group: &Option<Vec<String>>,
        anchors: &Option<HashMap<String, f64>>,
    ) -> PyResult<(
        Vec<Vec<String>>,
        Vec<String>,
        Vec<usize>,
        Vec<String>,
        Vec<(usize, f64)>,
    )> {
        let (docs_str, doc_names): (Vec<Vec<String>>, Vec<String>) =
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
                (strings, c.inner.doc_names.clone())
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                let names = (0..docs.len()).map(|i| format!("doc_{i}")).collect();
                (docs, names)
            };
        let num_docs = docs_str.len();
        if num_docs == 0 {
            return Err(PyValueError::new_err("data contains no documents"));
        }

        let (group_idx, author_names): (Vec<usize>, Vec<String>) = match group {
            Some(labels) => {
                if labels.len() != num_docs {
                    return Err(PyValueError::new_err(format!(
                        "group must have length num_docs ({num_docs}), got {}",
                        labels.len()
                    )));
                }
                let mut names: Vec<String> = labels.clone();
                names.sort();
                names.dedup();
                let index: HashMap<&str, usize> = names
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect();
                let idx: Vec<usize> = labels.iter().map(|l| index[l.as_str()]).collect();
                (idx, names)
            }
            None => ((0..num_docs).collect(), doc_names.clone()),
        };

        let anchor_pairs: Vec<(usize, f64)> = match anchors {
            None => Vec::new(),
            Some(m) => {
                let mut pairs = Vec::with_capacity(m.len());
                for (label, &target) in m {
                    let i = author_names
                        .iter()
                        .position(|a| a == label)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "anchor label {label:?} is not an author label"
                            ))
                        })?;
                    pairs.push((i, target));
                }
                pairs
            }
        };

        Ok((docs_str, doc_names, group_idx, author_names, anchor_pairs))
    }
}

#[pymethods]
impl IdealPointTM {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). Values are the effective ones actually
    /// in force (e.g. ``min_count`` after the ``.max(1)`` floor).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("num_dims", self.num_dims)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("sigma_shrink", self.sigma_shrink)?;
        d.set_item("prior_variance", self.prior_variance)?;
        d.set_item("w_prior_variance", self.w_prior_variance)?;
        d.set_item("x_prior_variance", self.x_prior_variance)?;
        d.set_item("max_inner", self.max_inner)?;
        d.set_item("min_count", self.min_count)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `num_topics` is K (>= 2); `num_dims` is the
    /// dimensionality d of the latent ideal point (default 1). For `num_dims > 1`
    /// the positions are identified only up to an orthogonal rotation (and a per-
    /// dimension sign), so read them through the loadings, not coordinate-by-
    /// coordinate; see `author_positions`. `prior_variance` is
    /// the Gaussian prior on the topic profiles (weak by default, as ETM);
    /// `w_prior_variance` regularizes the position loadings W (smaller = more
    /// shrinkage toward neutral topics); `x_prior_variance` is the prior on the
    /// positions (1.0 matches the unit-variance standardization). `convergence_tol`
    /// stops EM on the relative change in the bound; `max_inner` caps the L-BFGS
    /// steps per M-step. `min_count` drops words below that corpus frequency (count
    /// representation only; ignored when `word_embeddings` is passed to fit, where
    /// the vocabulary is supplied). `seed` seeds the RNG.
    #[new]
    #[pyo3(signature = (num_topics, *, num_dims=1, convergence_tol=1e-4,
                        sigma_shrink=0.0, prior_variance=1e6, w_prior_variance=10.0,
                        x_prior_variance=1.0, max_inner=15, min_count=1, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        num_dims: usize,
        convergence_tol: f64,
        sigma_shrink: f64,
        prior_variance: f64,
        w_prior_variance: f64,
        x_prior_variance: f64,
        max_inner: usize,
        min_count: usize,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("IdealPointTM")?;
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if num_dims < 1 {
            return Err(PyValueError::new_err("num_dims must be >= 1"));
        }
        if !finite_pos(prior_variance)
            || !finite_pos(w_prior_variance)
            || !finite_pos(x_prior_variance)
        {
            return Err(PyValueError::new_err(
                "prior_variance, w_prior_variance and x_prior_variance must be > 0",
            ));
        }
        // #481-class guards: a non-finite sigma_shrink feeds `1.0 - sigma_shrink`
        // into the prior covariance -> NaN topics; it is a mixing ratio in [0, 1).
        if !(sigma_shrink.is_finite() && (0.0..1.0).contains(&sigma_shrink)) {
            return Err(PyValueError::new_err(format!(
                "sigma_shrink must be in [0, 1) (got {sigma_shrink})"
            )));
        }
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
        Ok(IdealPointTM {
            num_topics,
            num_dims,
            convergence_tol,
            sigma_shrink,
            prior_variance,
            w_prior_variance,
            x_prior_variance,
            max_inner,
            min_count: min_count.max(1),
            seed,
            fitted: false,
            topic_names: Vec::new(),
            author_names: Vec::new(),
            id_to_word: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists). The representation is
    /// selected by `word_embeddings`:
    ///   - omit it (the default) and the topic-word matrix is parameterized directly
    ///     over the vocabulary (counts); the vocabulary is built from the corpus by
    ///     `min_count`. Do not pass `vocabulary` in this case.
    ///   - pass `word_embeddings` (`(len(vocabulary), E)`) with the aligned
    ///     `vocabulary` and the matrix is factored through word embeddings, as ETM.
    ///
    /// `group` is an optional list of author labels (length num_docs): documents
    /// sharing a label share one latent position; if omitted, each document is its
    /// own author. `anchors` is an optional `{author_label: value}` mapping used to
    /// orient the sign of the first latent dimension so positions align with the
    /// supplied direction. `iters` sets the EM iteration count (default 50).
    #[pyo3(signature = (data, *, word_embeddings=None, vocabulary=None, group=None,
                        anchors=None, iters=None, convergence_tol=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        word_embeddings: Option<&Bound<'_, PyAny>>,
        vocabulary: Option<Vec<String>>,
        group: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<Py<Self>> {
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        let it = iters.unwrap_or(50);
        match word_embeddings {
            Some(emb) => slf.fit_embedded(py, data, emb, vocabulary, group, anchors, it, tol)?,
            None => {
                if vocabulary.is_some() {
                    return Err(PyValueError::new_err(
                        "vocabulary is only used with word_embeddings; omit both for the count \
                         representation (the vocabulary is built from the corpus by min_count)",
                    ));
                }
                slf.fit_counts(py, data, group, anchors, it, tol)?
            }
        }
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn num_dims(&self) -> usize {
        self.num_dims
    }
    /// `"word2vec"` if fitted with `word_embeddings`, `"counts"` if fitted without,
    /// `None` if unfitted.
    #[getter]
    fn representation(&self) -> Option<&'static str> {
        self.model.as_ref().map(|m| m.representation())
    }
    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors())
    }
    /// Topic-word matrix at the neutral position x=0 (num_topics, vocab); the
    /// topics before any positional displacement. Rows are simplices.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(self.fitted_model()?.beta0()).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topics()).to_pyarray_bound(py))
    }
    /// Author positions (num_authors, num_dims): the latent ideal points,
    /// standardized to mean 0 / unit variance per dimension.
    ///
    /// Identifiability: the scale is fixed but the axis is identified only up to
    /// **sign** per dimension — and, for `num_dims > 1`, up to an arbitrary
    /// **rotation** of the axes (the likelihood is invariant under
    /// `x -> x @ R`, `W -> R^-1 @ W`). Pass `anchors` to `fit()` to fix the sign of
    /// dimension 0; without them the orientation is deterministic for a given seed
    /// but otherwise arbitrary (it can flip across seeds/corpora), and
    /// multi-dimensional positions are best read through the loadings, not
    /// coordinate-by-coordinate.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(self.fitted_model()?.x()).to_pyarray_bound(py))
    }
    /// Asymptotic standard error of each author position (num_authors, num_dims),
    /// from the observed information of the penalized position objective at the fit.
    /// Conditions on the fitted topic content (alpha/W), the multinomial-content
    /// analog of Wordfish's Hessian-based `se.theta`; an author's SE shrinks with
    /// the number of tokens they contribute. Aligned to `author_positions` /
    /// `author_names`.
    #[getter]
    fn position_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ntot = self.author_topic_counts()?;
        let se = self
            .fitted_model()?
            .position_se(&ntot, self.x_prior_variance);
        Ok(vecs_to_arr2(&se).to_pyarray_bound(py))
    }
    /// The author labels, in the row order of `author_positions`.
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// Per-topic discrimination ||W_k|| (num_topics): large where the topic sharply
    /// separates positions, ~0 where the topic is neutral. With the count
    /// representation the full-vocabulary loadings make this less concentrated on a
    /// single topic than the word-embedding representation's.
    #[getter]
    fn topic_discrimination<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.topic_discrimination()).to_pyarray_bound(py))
    }
    /// The words whose use within `topic` rises at the positive vs negative end of
    /// latent dimension `dim`. Returns `(positive, negative)`, each a list of
    /// `(word, score)` for the top-`n` words at `+magnitude` vs `-magnitude`.
    ///
    /// `weighting` sets the score and the ranking. `"prob"` (default) uses the
    /// probability difference `beta(+) - beta(-)`, which favors words the topic
    /// actually uses and keeps the contrast inside the topic's own vocabulary.
    /// `"logratio"` uses `log beta(+) - log beta(-)`, which is more sensitive but
    /// dominated by rare words, so it can pull in off-topic vocabulary.
    #[pyo3(signature = (topic, *, n=10, magnitude=1.0, dim=0, weighting="prob"))]
    fn position_shift(
        &self,
        topic: usize,
        n: usize,
        magnitude: f64,
        dim: usize,
        weighting: &str,
    ) -> PyResult<(Vec<(String, f64)>, Vec<(String, f64)>)> {
        let m = self.fitted_model()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        if dim >= self.num_dims {
            return Err(PyValueError::new_err("dim out of range"));
        }
        if weighting != "prob" && weighting != "logratio" {
            return Err(PyValueError::new_err(
                "weighting must be \"prob\" or \"logratio\"",
            ));
        }
        let mut xp = vec![0.0; self.num_dims];
        let mut xn = vec![0.0; self.num_dims];
        xp[dim] = magnitude;
        xn[dim] = -magnitude;
        let bp = m.position_topic_beta(topic, &xp);
        let bn = m.position_topic_beta(topic, &xn);
        let score: Vec<(usize, f64)> = (0..bp.len())
            .map(|v| {
                let s = if weighting == "logratio" {
                    (bp[v].max(1e-300)).ln() - (bn[v].max(1e-300)).ln()
                } else {
                    bp[v] - bn[v]
                };
                (v, s)
            })
            .collect();
        let mut by_pos = score.clone();
        by_pos.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut by_neg = score;
        by_neg.sort_by(|a, b| a.1.total_cmp(&b.1));
        let pos: Vec<(String, f64)> = by_pos
            .iter()
            .take(n)
            .map(|&(v, s)| (self.id_to_word[v].clone(), s))
            .collect();
        let neg: Vec<(String, f64)> = by_neg
            .iter()
            .take(n)
            .map(|&(v, s)| (self.id_to_word[v].clone(), s))
            .collect();
        Ok((pos, neg))
    }
    /// Position loadings W as a `(num_topics, num_dims * feature_dim)` array,
    /// row-major over `(dim, feature)`, where the feature dimension is the embedding
    /// dimension (word-embedding representation) or the vocabulary (count
    /// representation). These are the per-topic discrimination directions; their
    /// pairwise cosine alignment shows whether topics share one latent axis.
    #[getter]
    fn loadings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat: Vec<Vec<f64>> = self
            .fitted_model()?
            .w()
            .iter()
            .map(|wk| wk.iter().flatten().copied().collect())
            .collect();
        Ok(vecs_to_arr2(&flat).to_pyarray_bound(py))
    }
    /// Uniform convergence trace: `(iter, bound)` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .bound_history()
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged()))
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn early_stopped(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged()))
    }
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.bound())
    }
    #[getter]
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.em_iters_run())
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
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(self.fitted_model()?.beta0());
        topic_words_helper(
            py,
            &phi,
            &self.id_to_word,
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
        let phi = vecs_to_arr2(self.fitted_model()?.beta0());
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

    fn save(&self, path: &str) -> PyResult<()> {
        match &self.model {
            Some(IpInner::Count(m)) => self.save_counts(path, Some(m)),
            // Embedded fitted, or unfitted (default to the word-embedding format).
            Some(IpInner::Embedded(m)) => self.save_embedded(path, Some(m)),
            None => self.save_embedded(path, None),
        }
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        require_experimental("IdealPointTM")?;
        let tag = peek_model_tag(path)?;
        match tag {
            MODEL_TAG_IDEALPOINT_LDA => Self::load_counts(path),
            _ => Self::load_embedded(path),
        }
    }

    fn __repr__(&self) -> String {
        let rep = self.representation().unwrap_or("unset");
        format!(
            "IdealPointTM(num_topics={}, num_dims={}, representation={}, fitted={})",
            self.num_topics, self.num_dims, rep, self.fitted
        )
    }
}

// Representation-specific fit/save/load helpers. Not exposed to Python.
impl IdealPointTM {
    #[allow(clippy::too_many_arguments)]
    fn fit_embedded(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        word_embeddings: &Bound<'_, PyAny>,
        vocabulary: Option<Vec<String>>,
        group: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        it: usize,
        tol: f64,
    ) -> PyResult<()> {
        let vocabulary = vocabulary.ok_or_else(|| {
            PyValueError::new_err("word_embeddings requires the aligned vocabulary= argument")
        })?;
        let (docs_str, doc_names, group_idx, author_names, anchor_pairs) =
            Self::prepare_fit(data, &group, &anchors)?;
        let num_docs = docs_str.len();

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
        let map: HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_authors = author_names.len();

        let mut df = vec![0u32; vocabulary.len()];
        let mut tf = vec![0u32; vocabulary.len()];
        let mut docs_ids: Vec<Vec<u32>> = Vec::with_capacity(num_docs);
        for doc in &docs_str {
            let ids: Vec<u32> = doc
                .iter()
                .filter_map(|w| map.get(w.as_str()).copied())
                .collect();
            let mut seen = HashSet::new();
            for &id in &ids {
                tf[id as usize] += 1;
                seen.insert(id as usize);
            }
            for id in seen {
                df[id] += 1;
            }
            docs_ids.push(ids);
        }
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens in the documents",
            ));
        }
        let coherence_corpus = corpus::Corpus {
            id_to_word: vocabulary.clone(),
            docs: docs_ids.clone(),
            doc_names: doc_names.clone(),
            doc_labels: vec![String::new(); num_docs],
            doc_freqs: df,
            total_freqs: tf,
        };

        let num_types = vocabulary.len();
        let (k, dd, ss, pv, wpv, xpv, mi, seed) = (
            self.num_topics,
            self.num_dims,
            self.sigma_shrink,
            self.prior_variance,
            self.w_prior_variance,
            self.x_prior_variance,
            self.max_inner,
            self.seed,
        );
        let model = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            idealpoint::fit_idealpoint(
                &docs_ids,
                &group_idx,
                num_authors,
                k,
                num_types,
                dd,
                &rho,
                &anchor_pairs,
                it,
                tol,
                ss,
                pv,
                wpv,
                xpv,
                mi,
                &mut rng,
            )
        });

        self.model = Some(IpInner::Embedded(model));
        self.corpus = Some(coherence_corpus);
        self.id_to_word = vocabulary;
        self.author_names = author_names;
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    fn fit_counts(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        group: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        it: usize,
        tol: f64,
    ) -> PyResult<()> {
        let (docs_str, doc_names, group_idx, author_names, anchor_pairs) =
            Self::prepare_fit(data, &group, &anchors)?;
        let num_docs = docs_str.len();
        let num_authors = author_names.len();

        // Vocabulary by corpus frequency >= min_count (desc freq then word).
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for doc in &docs_str {
            for w in doc {
                *freq.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut vocab_pairs: Vec<(&str, usize)> = freq
            .iter()
            .filter(|&(_, &c)| c >= self.min_count)
            .map(|(&w, &c)| (w, c))
            .collect();
        vocab_pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let vocabulary: Vec<String> = vocab_pairs.iter().map(|&(w, _)| w.to_string()).collect();
        if vocabulary.len() < self.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words after min_count pruning",
            ));
        }
        let word_id: HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_types = vocabulary.len();

        let mut df = vec![0u32; num_types];
        let mut tf = vec![0u32; num_types];
        let mut docs_ids: Vec<Vec<u32>> = Vec::with_capacity(num_docs);
        for doc in &docs_str {
            let ids: Vec<u32> = doc
                .iter()
                .filter_map(|w| word_id.get(w.as_str()).copied())
                .collect();
            let mut seen = HashSet::new();
            for &id in &ids {
                tf[id as usize] += 1;
                seen.insert(id as usize);
            }
            for id in seen {
                df[id] += 1;
            }
            docs_ids.push(ids);
        }
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens in the documents",
            ));
        }
        let coherence_corpus = corpus::Corpus {
            id_to_word: vocabulary.clone(),
            docs: docs_ids.clone(),
            doc_names: doc_names.clone(),
            doc_labels: vec![String::new(); num_docs],
            doc_freqs: df,
            total_freqs: tf,
        };

        let (k, dd, ss, pv, wpv, xpv, mi, seed) = (
            self.num_topics,
            self.num_dims,
            self.sigma_shrink,
            self.prior_variance,
            self.w_prior_variance,
            self.x_prior_variance,
            self.max_inner,
            self.seed,
        );
        let model = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            idealpoint_lda::fit_idealpoint_lda(
                &docs_ids,
                &group_idx,
                num_authors,
                k,
                num_types,
                dd,
                &anchor_pairs,
                it,
                tol,
                ss,
                pv,
                wpv,
                xpv,
                mi,
                &mut rng,
            )
        });

        self.model = Some(IpInner::Count(model));
        self.corpus = Some(coherence_corpus);
        self.id_to_word = vocabulary;
        self.author_names = author_names;
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    fn save_embedded(&self, path: &str, m: Option<&IdealPointModel>) -> PyResult<()> {
        write_state(
            path,
            MODEL_TAG_IDEALPOINT,
            &IdealPointState {
                num_topics: self.num_topics,
                num_dims: self.num_dims,
                convergence_tol: self.convergence_tol,
                sigma_shrink: self.sigma_shrink,
                prior_variance: self.prior_variance,
                w_prior_variance: self.w_prior_variance,
                x_prior_variance: self.x_prior_variance,
                max_inner: self.max_inner,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                author_names: self.author_names.clone(),
                id_to_word: self.id_to_word.clone(),
                corpus: self.corpus.clone(),
                num_types: m.map(|m| m.num_types),
                num_authors: m.map(|m| m.num_authors),
                alpha: m.map(|m| m.alpha.clone()),
                w: m.map(|m| m.w.clone()),
                x: m.map(|m| m.x.clone()),
                group: m.map(|m| m.group.clone()),
                beta0: m.map(|m| m.beta0.clone()),
                base: m.map(|m| m.base.clone()),
                disc: m.map(|m| m.disc.clone()),
                mu: m.map(|m| m.mu.clone()),
                sigma: m.map(|m| m.sigma.clone()),
                lambda: m.map(|m| m.lambda.clone()),
                bound: m.map(|m| m.bound),
                bound_history: m.map(|m| m.bound_history.clone()),
                converged: m.map(|m| m.converged),
                em_iters_run: m.map(|m| m.em_iters_run),
            },
        )
    }

    fn save_counts(&self, path: &str, m: Option<&IdealPointLdaModel>) -> PyResult<()> {
        write_state(
            path,
            MODEL_TAG_IDEALPOINT_LDA,
            &IdealPointLdaState {
                num_topics: self.num_topics,
                num_dims: self.num_dims,
                convergence_tol: self.convergence_tol,
                sigma_shrink: self.sigma_shrink,
                prior_variance: self.prior_variance,
                w_prior_variance: self.w_prior_variance,
                x_prior_variance: self.x_prior_variance,
                max_inner: self.max_inner,
                min_count: self.min_count,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                author_names: self.author_names.clone(),
                id_to_word: self.id_to_word.clone(),
                corpus: self.corpus.clone(),
                num_types: m.map(|m| m.num_types),
                num_authors: m.map(|m| m.num_authors),
                alpha: m.map(|m| m.alpha.clone()),
                w: m.map(|m| m.w.clone()),
                x: m.map(|m| m.x.clone()),
                group: m.map(|m| m.group.clone()),
                beta0: m.map(|m| m.beta0.clone()),
                mu: m.map(|m| m.mu.clone()),
                sigma: m.map(|m| m.sigma.clone()),
                lambda: m.map(|m| m.lambda.clone()),
                bound: m.map(|m| m.bound),
                bound_history: m.map(|m| m.bound_history.clone()),
                converged: m.map(|m| m.converged),
                em_iters_run: m.map(|m| m.em_iters_run),
            },
        )
    }

    fn load_embedded(path: &str) -> PyResult<Self> {
        let s: IdealPointState = read_state(path, MODEL_TAG_IDEALPOINT)?;
        let model = if s.fitted && s.alpha.is_some() {
            Some(IpInner::Embedded(IdealPointModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                num_dims: s.num_dims,
                num_authors: s.num_authors.unwrap_or(0),
                alpha: s.alpha.unwrap_or_default(),
                w: s.w.unwrap_or_default(),
                x: s.x.unwrap_or_default(),
                group: s.group.unwrap_or_default(),
                beta0: s.beta0.unwrap_or_default(),
                base: s.base.unwrap_or_default(),
                disc: s.disc.unwrap_or_default(),
                mu: s.mu.unwrap_or_default(),
                sigma: s.sigma.unwrap_or_default(),
                lambda: s.lambda.unwrap_or_default(),
                bound: s.bound.unwrap_or(f64::NAN),
                bound_history: s.bound_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                em_iters_run: s.em_iters_run.unwrap_or(0),
            }))
        } else {
            None
        };
        Ok(IdealPointTM {
            num_topics: s.num_topics,
            num_dims: s.num_dims,
            convergence_tol: s.convergence_tol,
            sigma_shrink: s.sigma_shrink,
            prior_variance: s.prior_variance,
            w_prior_variance: s.w_prior_variance,
            x_prior_variance: s.x_prior_variance,
            max_inner: s.max_inner,
            min_count: 1,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            author_names: s.author_names,
            id_to_word: s.id_to_word,
            model,
            corpus: s.corpus,
        })
    }

    fn load_counts(path: &str) -> PyResult<Self> {
        let s: IdealPointLdaState = read_state(path, MODEL_TAG_IDEALPOINT_LDA)?;
        let model = if s.fitted && s.alpha.is_some() {
            Some(IpInner::Count(IdealPointLdaModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                num_dims: s.num_dims,
                num_authors: s.num_authors.unwrap_or(0),
                alpha: s.alpha.unwrap_or_default(),
                w: s.w.unwrap_or_default(),
                x: s.x.unwrap_or_default(),
                group: s.group.unwrap_or_default(),
                beta0: s.beta0.unwrap_or_default(),
                mu: s.mu.unwrap_or_default(),
                sigma: s.sigma.unwrap_or_default(),
                lambda: s.lambda.unwrap_or_default(),
                bound: s.bound.unwrap_or(f64::NAN),
                bound_history: s.bound_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                em_iters_run: s.em_iters_run.unwrap_or(0),
            }))
        } else {
            None
        };
        Ok(IdealPointTM {
            num_topics: s.num_topics,
            num_dims: s.num_dims,
            convergence_tol: s.convergence_tol,
            sigma_shrink: s.sigma_shrink,
            prior_variance: s.prior_variance,
            w_prior_variance: s.w_prior_variance,
            x_prior_variance: s.x_prior_variance,
            max_inner: s.max_inner,
            min_count: s.min_count,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            author_names: s.author_names,
            id_to_word: s.id_to_word,
            model,
            corpus: s.corpus,
        })
    }
}
