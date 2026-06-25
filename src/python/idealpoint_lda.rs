//! IdealPointLDA pyclass: the count-based twin of IdealPointTM. A topic model whose
//! per-topic word distributions are displaced by a latent author position, in
//! vocabulary space (no embeddings) — "Wordfish with topics". Experimental, gated.
//! `use super::*` pulls in the shared bindings.

use super::*;
use crate::idealpoint_lda::{self, IdealPointLdaModel};
use std::collections::{HashMap, HashSet};

#[pyclass(module = "topica")]
pub struct IdealPointLDA {
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
    model: Option<IdealPointLdaModel>,
    corpus: Option<corpus::Corpus>,
}

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

impl IdealPointLDA {
    fn fitted_model(&self) -> PyResult<&IdealPointLdaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl IdealPointLDA {
    /// Create an unfitted model. `num_topics` is K (>= 2); `num_dims` the latent
    /// ideal-point dimensionality (default 1). `prior_variance` regularizes the topic
    /// log-profiles, `w_prior_variance` the position loadings (smaller = more
    /// shrinkage toward neutral topics), `x_prior_variance` the positions.
    /// `min_count` drops words below that corpus frequency. `convergence_tol` stops
    /// EM on the relative change in the bound; `max_inner` caps the L-BFGS M-step.
    #[new]
    #[pyo3(signature = (num_topics, *, num_dims=1, convergence_tol=1e-4,
                        sigma_shrink=0.0, prior_variance=1e6, w_prior_variance=10.0,
                        x_prior_variance=1.0, max_inner=15, min_count=1, seed=42))]
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
        require_experimental("IdealPointLDA")?;
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
        Ok(IdealPointLDA {
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

    /// Fit on `data` (a Corpus or list of token lists). `group` is an optional list
    /// of author labels (length num_docs): documents sharing a label share one latent
    /// position; if omitted, each document is its own author. `anchors` is an optional
    /// `{author_label: value}` mapping orienting the sign of the first latent
    /// dimension. `iters` sets the EM iteration count (default 50).
    #[pyo3(signature = (data, *, group=None, anchors=None, iters=None,
                        convergence_tol=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        group: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
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

        // Author grouping.
        let (group_idx, author_names): (Vec<usize>, Vec<String>) = match &group {
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

        // Map tokens to ids; build the coherence corpus over the same id space.
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

        // Anchors.
        let anchor_pairs: Vec<(usize, f64)> = match &anchors {
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

        let tol = convergence_tol.unwrap_or(self.convergence_tol);
        let it = iters.unwrap_or(50);
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

        self.model = Some(model);
        self.corpus = Some(coherence_corpus);
        self.id_to_word = vocabulary;
        self.author_names = author_names;
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn num_dims(&self) -> usize {
        self.num_dims
    }
    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors)
    }
    /// Topic-word matrix at the neutral position x=0 (num_topics, vocab).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.beta0).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topics()).to_pyarray_bound(py))
    }
    /// Author positions (num_authors, num_dims), standardized to mean 0 / unit var.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.x).to_pyarray_bound(py))
    }
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// Per-topic discrimination ||W_k|| (num_topics). Note: with full-vocabulary
    /// loadings this is less concentrated on a single topic than IdealPointTM's.
    #[getter]
    fn topic_discrimination<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.topic_discrimination()).to_pyarray_bound(py))
    }
    /// The words whose use within `topic` rises at the positive vs negative end of
    /// latent dimension `dim`. Returns `(positive, negative)` lists of `(word, score)`.
    /// `weighting` is `"prob"` (beta(+) - beta(-), default) or `"logratio"`.
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
    /// Position loadings W as (num_topics, num_dims * vocab), row-major over (dim, v).
    #[getter]
    fn loadings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let flat: Vec<Vec<f64>> =
            m.w.iter()
                .map(|wk| wk.iter().flatten().copied().collect())
                .collect();
        Ok(vecs_to_arr2(&flat).to_pyarray_bound(py))
    }
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
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged))
    }
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.bound)
    }
    #[getter]
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.em_iters_run)
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
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.beta0);
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.beta0);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
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

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        require_experimental("IdealPointLDA")?;
        let s: IdealPointLdaState = read_state(path, MODEL_TAG_IDEALPOINT_LDA)?;
        let model = if s.fitted && s.alpha.is_some() {
            Some(IdealPointLdaModel {
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
            })
        } else {
            None
        };
        Ok(IdealPointLDA {
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

    fn __repr__(&self) -> String {
        format!(
            "IdealPointLDA(num_topics={}, num_dims={}, fitted={})",
            self.num_topics, self.num_dims, self.fitted
        )
    }
}
