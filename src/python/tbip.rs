//! TBIP pyclass: Text-Based Ideal Points (Vafa, Naidu & Blei 2020). A Poisson
//! factorization whose neutral topic-word intensities are rescaled by a per-word
//! ideological factor `exp(x_s * eta_kv)`, with the author position `x_s` latent.
//! Fit by the paper's mean-field VI (reparameterized SVI, Adam). `use super::*`
//! pulls in the shared bindings.

use super::*;
use pyo3::types::PyDict;

use crate::tbip::{self, TbipConfig, TbipModel, TbipParams};
use std::collections::{HashMap, HashSet};

#[pyclass(module = "topica")]
pub struct TBIP {
    num_topics: usize,
    a_gamma: f64,
    b_gamma: f64,
    iters: usize,
    batch_size: usize,
    learning_rate: f64,
    min_count: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    model: Option<TbipModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TbipState {
    num_topics: usize,
    a_gamma: f64,
    b_gamma: f64,
    iters: usize,
    batch_size: usize,
    learning_rate: f64,
    min_count: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    num_authors: Option<usize>,
    num_docs: Option<usize>,
    mu_theta: Option<Vec<f64>>,
    rs_theta: Option<Vec<f64>>,
    mu_beta: Option<Vec<f64>>,
    rs_beta: Option<Vec<f64>>,
    mu_eta: Option<Vec<f64>>,
    rs_eta: Option<Vec<f64>>,
    mu_x: Option<Vec<f64>>,
    rs_x: Option<Vec<f64>>,
    group: Option<Vec<usize>>,
    elbo_history: Option<Vec<f64>>,
    iters_run: Option<usize>,
}

impl TBIP {
    fn fitted_model(&self) -> PyResult<&TbipModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl TBIP {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). Values are the effective ones actually
    /// in force (e.g. ``batch_size``/``min_count`` after the ``.max(1)`` floor).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("a_gamma", self.a_gamma)?;
        d.set_item("b_gamma", self.b_gamma)?;
        d.set_item("iters", self.iters)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("learning_rate", self.learning_rate)?;
        d.set_item("min_count", self.min_count)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `num_topics` is K (>= 2). `a_gamma`/`b_gamma` are
    /// the sparse-Gamma prior hyperparameters on theta and beta (default 0.3 each,
    /// as in the paper). `iters` is the number of SVI steps; `batch_size` the
    /// document minibatch; `learning_rate` the Adam step (halved at 50% and 80% of
    /// the schedule). `min_count` drops words below that corpus frequency.
    #[new]
    #[pyo3(signature = (num_topics, *, a_gamma=0.3, b_gamma=0.3, iters=7000,
                        batch_size=512, learning_rate=0.05, min_count=1, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        a_gamma: f64,
        b_gamma: f64,
        iters: usize,
        batch_size: usize,
        learning_rate: f64,
        min_count: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !finite_pos(a_gamma) || !finite_pos(b_gamma) {
            return Err(PyValueError::new_err("a_gamma and b_gamma must be > 0"));
        }
        if !finite_pos(learning_rate) {
            return Err(PyValueError::new_err("learning_rate must be > 0"));
        }
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be >= 1"));
        }
        Ok(TBIP {
            num_topics,
            a_gamma,
            b_gamma,
            iters,
            batch_size: batch_size.max(1),
            learning_rate,
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
    /// ideal point; if omitted, each document is its own author. `iters`,
    /// `batch_size`, and `learning_rate` override the constructor values when given.
    #[pyo3(signature = (data, *, group=None, iters=None, batch_size=None,
                        learning_rate=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        group: Option<Vec<String>>,
        iters: Option<usize>,
        batch_size: Option<usize>,
        learning_rate: Option<f64>,
    ) -> PyResult<Py<Self>> {
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

        // Author grouping (same semantics as the ideal-point family).
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
            .filter(|&(_, &c)| c >= slf.min_count)
            .map(|(&w, &c)| (w, c))
            .collect();
        vocab_pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let vocabulary: Vec<String> = vocab_pairs.iter().map(|&(w, _)| w.to_string()).collect();
        if vocabulary.len() < slf.num_topics {
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

        let cfg = TbipConfig {
            a_gamma: slf.a_gamma,
            b_gamma: slf.b_gamma,
            iters: iters.unwrap_or(slf.iters),
            batch_size: batch_size.unwrap_or(slf.batch_size).max(1),
            learning_rate: learning_rate.unwrap_or(slf.learning_rate),
        };
        let (k, seed) = (slf.num_topics, slf.seed);
        let model = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            tbip::fit_tbip(
                &docs_ids,
                &group_idx,
                num_authors,
                k,
                num_types,
                &cfg,
                &mut rng,
            )
        });

        slf.model = Some(model);
        slf.corpus = Some(coherence_corpus);
        slf.id_to_word = vocabulary;
        slf.author_names = author_names;
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors)
    }
    /// Author ideal points (num_authors,), the posterior-mean positions mu_x.
    ///
    /// Identifiability: identified only up to **sign** — the model is invariant
    /// under `x -> -x, eta -> -eta`, so the direction is arbitrary and determined by
    /// the seed (TBIP has no anchoring). Compare runs by absolute correlation, or
    /// flip to a chosen reference author. The scale is only softly pinned by the
    /// N(0, 1) prior.
    #[getter]
    fn ideal_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.ideal_points()).to_pyarray_bound(py))
    }
    /// Standard error of each author ideal point (num_authors,): the standard
    /// deviation of the Gaussian variational posterior q(x_s) estimated jointly with
    /// the mean. As with any mean-field VI this can understate the true posterior
    /// spread. Aligned to `ideal_points` / `author_names`.
    #[getter]
    fn position_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.position_se()).to_pyarray_bound(py))
    }
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// Neutral topic-word matrix (num_topics, vocab), exp(mu_beta) row-normalized.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word()).to_pyarray_bound(py))
    }
    /// Ideological topics eta (num_topics, vocab), real-valued mu_eta. A positive
    /// entry pushes the word up at the positive end of the ideal-point axis.
    #[getter]
    fn ideological_topics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.ideological_topics()).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics), exp(mu_theta) row-normalized.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic()).to_pyarray_bound(py))
    }
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .elbo_history
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
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word());
        topic_words_helper(py, &phi, &self.id_to_word, self.num_topics, n, topic)
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

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_TBIP,
            &TbipState {
                num_topics: self.num_topics,
                a_gamma: self.a_gamma,
                b_gamma: self.b_gamma,
                iters: self.iters,
                batch_size: self.batch_size,
                learning_rate: self.learning_rate,
                min_count: self.min_count,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                author_names: self.author_names.clone(),
                id_to_word: self.id_to_word.clone(),
                corpus: self.corpus.clone(),
                num_types: m.map(|m| m.num_types),
                num_authors: m.map(|m| m.num_authors),
                num_docs: m.map(|m| m.params.num_docs),
                mu_theta: m.map(|m| m.params.mu_theta.clone()),
                rs_theta: m.map(|m| m.params.rs_theta.clone()),
                mu_beta: m.map(|m| m.params.mu_beta.clone()),
                rs_beta: m.map(|m| m.params.rs_beta.clone()),
                mu_eta: m.map(|m| m.params.mu_eta.clone()),
                rs_eta: m.map(|m| m.params.rs_eta.clone()),
                mu_x: m.map(|m| m.params.mu_x.clone()),
                rs_x: m.map(|m| m.params.rs_x.clone()),
                group: m.map(|m| m.group.clone()),
                elbo_history: m.map(|m| m.elbo_history.clone()),
                iters_run: m.map(|m| m.iters_run),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: TbipState = read_state(path, MODEL_TAG_TBIP)?;
        let model = if s.fitted && s.mu_x.is_some() {
            let k = s.num_topics;
            let v = s.num_types.unwrap_or(0);
            let a = s.num_authors.unwrap_or(0);
            let dn = s.num_docs.unwrap_or(0);
            Some(TbipModel {
                num_topics: k,
                num_types: v,
                num_authors: a,
                params: TbipParams {
                    num_docs: dn,
                    num_topics: k,
                    num_types: v,
                    num_authors: a,
                    mu_theta: s.mu_theta.unwrap_or_default(),
                    rs_theta: s.rs_theta.unwrap_or_default(),
                    mu_beta: s.mu_beta.unwrap_or_default(),
                    rs_beta: s.rs_beta.unwrap_or_default(),
                    mu_eta: s.mu_eta.unwrap_or_default(),
                    rs_eta: s.rs_eta.unwrap_or_default(),
                    mu_x: s.mu_x.unwrap_or_default(),
                    rs_x: s.rs_x.unwrap_or_default(),
                },
                group: s.group.unwrap_or_default(),
                elbo_history: s.elbo_history.unwrap_or_default(),
                iters_run: s.iters_run.unwrap_or(0),
            })
        } else {
            None
        };
        Ok(TBIP {
            num_topics: s.num_topics,
            a_gamma: s.a_gamma,
            b_gamma: s.b_gamma,
            iters: s.iters,
            batch_size: s.batch_size,
            learning_rate: s.learning_rate,
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
            "TBIP(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
