//! Wordfish pyclass: the Slapin & Proksch (2008) Poisson scaling model — a
//! word-frequency ideal-point estimator with no topics and no embeddings. The
//! baseline companion to `IdealPointTM`. `use super::*` pulls in the shared
//! bindings (Corpus, arrays, save/load).

use super::*;
use crate::wordfish::{self, WordfishModel};
use std::collections::HashMap;

#[pyclass(module = "topica")]
pub struct Wordfish {
    beta_prior_sd: f64,
    theta_prior_sd: f64,
    min_count: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    model: Option<WordfishModel>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WordfishState {
    beta_prior_sd: f64,
    theta_prior_sd: f64,
    min_count: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    num_authors: Option<usize>,
    num_types: Option<usize>,
    theta: Option<Vec<f64>>,
    alpha: Option<Vec<f64>>,
    psi: Option<Vec<f64>>,
    beta: Option<Vec<f64>>,
    log_likelihood: Option<f64>,
    ll_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
}

impl Wordfish {
    fn fitted_model(&self) -> PyResult<&WordfishModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl Wordfish {
    /// Create an unfitted Wordfish model. `beta_prior_sd` / `theta_prior_sd` are the
    /// standard deviations of the weak Gaussian priors regularizing the word
    /// discriminations and the positions (pass `inf` for none); `min_count` drops
    /// words occurring fewer than that many times across the whole corpus;
    /// `convergence_tol` stops the EM on the relative change in the log-likelihood.
    /// `seed` is accepted for API uniformity — the fit is deterministic.
    #[new]
    #[pyo3(signature = (*, beta_prior_sd=3.0, theta_prior_sd=1.0, min_count=1,
                        convergence_tol=1e-6, seed=42))]
    fn new(
        beta_prior_sd: f64,
        theta_prior_sd: f64,
        min_count: usize,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Wordfish {
            beta_prior_sd,
            theta_prior_sd,
            min_count: min_count.max(1),
            convergence_tol,
            seed,
            fitted: false,
            author_names: Vec::new(),
            id_to_word: Vec::new(),
            model: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists). `group` is an optional list
    /// of author labels (length num_docs): documents sharing a label are pooled into
    /// one unit with one position; if omitted, each document is its own unit.
    /// `anchors` is an optional `{author_label: value}` mapping used to orient the
    /// sign of the axis so positions align with the supplied direction. `iters` sets
    /// the EM iteration cap (default 100).
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

        // Resolve the author grouping.
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
        if num_authors < 2 {
            return Err(PyValueError::new_err(
                "Wordfish needs at least 2 authors/documents to scale",
            ));
        }

        // Build the vocabulary (corpus frequency >= min_count), deterministically
        // ordered by descending frequency then word.
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
        let id_to_word: Vec<String> = vocab_pairs.iter().map(|&(w, _)| w.to_string()).collect();
        if id_to_word.len() < 2 {
            return Err(PyValueError::new_err(
                "vocabulary has fewer than 2 words after min_count pruning",
            ));
        }
        let word_id: HashMap<&str, u32> = id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_types = id_to_word.len();

        // Aggregate counts per author.
        let mut author_counts: Vec<HashMap<u32, f64>> = vec![HashMap::new(); num_authors];
        for (d, doc) in docs_str.iter().enumerate() {
            let a = group_idx[d];
            for w in doc {
                if let Some(&wid) = word_id.get(w.as_str()) {
                    *author_counts[a].entry(wid).or_insert(0.0) += 1.0;
                }
            }
        }
        let counts: Vec<Vec<(u32, f64)>> = author_counts
            .into_iter()
            .map(|m| {
                let mut v: Vec<(u32, f64)> = m.into_iter().collect();
                v.sort_by_key(|&(w, _)| w);
                v
            })
            .collect();
        if counts.iter().all(|c| c.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens after pruning",
            ));
        }

        // Resolve anchors into (author_index, target).
        let anchor_pairs: Vec<(usize, f64)> = match &anchors {
            None => Vec::new(),
            Some(m) => {
                let mut pairs = Vec::with_capacity(m.len());
                for (label, &target) in m {
                    let i = author_names
                        .iter()
                        .position(|x| x == label)
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
        let it = iters.unwrap_or(100);
        let (bsd, tsd) = (self.beta_prior_sd, self.theta_prior_sd);
        let model = py.allow_threads(move || {
            wordfish::fit_wordfish(&counts, num_types, &anchor_pairs, it, tol, bsd, tsd)
        });

        self.model = Some(model);
        self.id_to_word = id_to_word;
        self.author_names = author_names;
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors)
    }
    /// Author positions as a (num_authors, 1) matrix, standardized to mean 0 / unit
    /// variance. The latent left-right scale.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.positions()).to_pyarray_bound(py))
    }
    /// Asymptotic standard error of each author position (num_authors,), from the
    /// observed information of the penalized Poisson log-likelihood — the same
    /// Hessian-based SE R quanteda reports as `se.theta`. Aligned to `author_names`.
    #[getter]
    fn position_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.position_se()).to_pyarray_bound(py))
    }
    /// The author labels, in the row order of `author_positions`.
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// Per-word discrimination beta (num_types): how strongly each word's rate moves
    /// along the latent axis. Large |beta| = a strongly polarizing word.
    #[getter]
    fn word_discrimination<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.beta.clone()).to_pyarray_bound(py))
    }
    /// Per-word intercept psi (num_types): the word's baseline log-rate.
    #[getter]
    fn word_intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.psi.clone()).to_pyarray_bound(py))
    }
    /// The words most associated with the positive vs negative end of the axis,
    /// ranked by discrimination beta. Returns `(positive, negative)`, each a list of
    /// `(word, beta)` for the top-`n`.
    #[pyo3(signature = (n=10))]
    fn discriminating_words(&self, n: usize) -> PyResult<(Vec<(String, f64)>, Vec<(String, f64)>)> {
        let m = self.fitted_model()?;
        let mut idx: Vec<usize> = (0..m.num_types).collect();
        idx.sort_by(|&a, &b| m.beta[b].total_cmp(&m.beta[a]));
        let pos: Vec<(String, f64)> = idx
            .iter()
            .take(n)
            .map(|&j| (self.id_to_word[j].clone(), m.beta[j]))
            .collect();
        let neg: Vec<(String, f64)> = idx
            .iter()
            .rev()
            .take(n)
            .map(|&j| (self.id_to_word[j].clone(), m.beta[j]))
            .collect();
        Ok((pos, neg))
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.log_likelihood)
    }
    /// Convergence trace: `(iter, log_likelihood)` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .ll_history
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
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.iters_run)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_WORDFISH,
            &WordfishState {
                beta_prior_sd: self.beta_prior_sd,
                theta_prior_sd: self.theta_prior_sd,
                min_count: self.min_count,
                convergence_tol: self.convergence_tol,
                seed: self.seed,
                fitted: self.fitted,
                author_names: self.author_names.clone(),
                id_to_word: self.id_to_word.clone(),
                num_authors: m.map(|m| m.num_authors),
                num_types: m.map(|m| m.num_types),
                theta: m.map(|m| m.theta.clone()),
                alpha: m.map(|m| m.alpha.clone()),
                psi: m.map(|m| m.psi.clone()),
                beta: m.map(|m| m.beta.clone()),
                log_likelihood: m.map(|m| m.log_likelihood),
                ll_history: m.map(|m| m.ll_history.clone()),
                converged: m.map(|m| m.converged),
                iters_run: m.map(|m| m.iters_run),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: WordfishState = read_state(path, MODEL_TAG_WORDFISH)?;
        let model = if s.fitted && s.theta.is_some() {
            Some(WordfishModel {
                num_authors: s.num_authors.unwrap_or(0),
                num_types: s.num_types.unwrap_or(0),
                theta: s.theta.unwrap_or_default(),
                alpha: s.alpha.unwrap_or_default(),
                psi: s.psi.unwrap_or_default(),
                beta: s.beta.unwrap_or_default(),
                log_likelihood: s.log_likelihood.unwrap_or(f64::NAN),
                ll_history: s.ll_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
                theta_prior_sd: s.theta_prior_sd,
            })
        } else {
            None
        };
        Ok(Wordfish {
            beta_prior_sd: s.beta_prior_sd,
            theta_prior_sd: s.theta_prior_sd,
            min_count: s.min_count,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            fitted: s.fitted,
            author_names: s.author_names,
            id_to_word: s.id_to_word,
            model,
        })
    }

    fn __repr__(&self) -> String {
        format!("Wordfish(fitted={})", self.fitted)
    }
}
