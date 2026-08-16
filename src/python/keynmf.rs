//! KeyNMF pyclass: Kristensen-McLachlan et al. (2024) embedding-keyword NMF. You
//! bring the document and word embeddings and the aligned vocabulary; the model
//! scores each document's words by embedding similarity, keeps the top-N positive,
//! and factors that keyword-importance matrix with topica's NMF. `use super::*`
//! pulls in the shared bindings.

use super::*;
use pyo3::types::PyDict;

use crate::keynmf::{self, KeyNmfModel, Metric};
use std::collections::HashMap;

/// KeyNMF (Kristensen-McLachlan et al. 2024): an NMF topic model over an
/// embedding-derived keyword-importance matrix. For each document, every candidate
/// word (a token present in the document and in the supplied vocabulary) is scored by
/// the similarity between the document embedding and the word embedding; the top-N
/// positive words form a sparse doc-by-word matrix that is factored by NMF. You bring
/// the document and word embeddings and the aligned vocabulary. The fit is
/// deterministic (NNDSVD init).
#[pyclass(module = "topica")]
pub struct KeyNMF {
    num_topics: usize,
    top_n: usize,
    metric: String,
    iters: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    id_to_word: Vec<String>,
    doc_names: Vec<String>,
    model: Option<KeyNmfModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyNmfState {
    num_topics: usize,
    top_n: usize,
    metric: String,
    iters: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    id_to_word: Vec<String>,
    doc_names: Vec<String>,
    doc_freqs: Vec<u32>,
    total_freqs: Vec<u32>,
    docs: Vec<Vec<u32>>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    h: Option<Vec<Vec<f64>>>,
    w: Option<Vec<Vec<f64>>>,
    reconstruction_error: Option<f64>,
    error_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
    keywords: Option<Vec<Vec<(u32, f64)>>>,
}

impl KeyNMF {
    fn fitted_model(&self) -> PyResult<&KeyNmfModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl KeyNMF {
    /// Create an unfitted KeyNMF model. `top_n` keeps that many highest-similarity
    /// positive keywords per document; `metric` is `"cosine"` (default) or `"dot"`.
    /// `seed` is accepted for API uniformity — with the default NNDSVD init the fit is
    /// deterministic. The NMF iteration cap and tolerance are `fit()` arguments.
    #[new]
    #[pyo3(signature = (num_topics, *, top_n=25, metric="cosine".to_string(), seed=13))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        top_n: usize,
        metric: String,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if top_n < 1 {
            return Err(PyValueError::new_err("top_n must be >= 1"));
        }
        if metric != "cosine" && metric != "dot" {
            return Err(PyValueError::new_err(
                "metric must be \"cosine\" or \"dot\"",
            ));
        }
        Ok(KeyNMF {
            num_topics,
            top_n,
            metric,
            iters: 200,
            convergence_tol: 0.0,
            seed,
            fitted: false,
            id_to_word: Vec::new(),
            doc_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("top_n", self.top_n)?;
        d.set_item("metric", &self.metric)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a list of token lists or a Corpus) with `doc_embeddings`
    /// (`(num_docs, E)`), `word_embeddings` (`(len(vocabulary), E)`), and the aligned
    /// `vocabulary`. The **vocabulary is the model's vocabulary**: a document's
    /// candidate words are its tokens that appear in `vocabulary`, scored by the
    /// similarity between the document embedding and the word embedding. Token order
    /// is irrelevant (a document is a set of words here). `num_topics` must be
    /// `<= min(num_docs, len(vocabulary))`.
    #[pyo3(signature = (data, doc_embeddings, *, word_embeddings, vocabulary,
                        iters=None, convergence_tol=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        word_embeddings: &Bound<'_, PyAny>,
        vocabulary: Vec<String>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<Py<Self>> {
        if let Some(t) = convergence_tol {
            ensure_finite_nonneg("convergence_tol", t)?;
        }
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

        // The supplied vocabulary IS the model vocabulary (aligned to word_embeddings).
        if vocabulary.len() < 2 {
            return Err(PyValueError::new_err(
                "vocabulary must have at least 2 words",
            ));
        }
        let word_id: HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_types = vocabulary.len();

        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but data has {num_docs} documents",
                doc_emb.len()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let word_emb = parse_features(word_embeddings)?;
        if word_emb.len() != num_types {
            return Err(PyValueError::new_err(format!(
                "word_embeddings has {} rows but vocabulary has {num_types} words",
                word_emb.len()
            )));
        }
        check_all_finite_2d("word_embeddings", &word_emb)?;
        let emb_dim = doc_emb.first().map(|r| r.len()).unwrap_or(0);
        if emb_dim == 0 {
            return Err(PyValueError::new_err("doc_embeddings has zero width"));
        }
        if doc_emb.iter().any(|r| r.len() != emb_dim) || word_emb.iter().any(|r| r.len() != emb_dim)
        {
            return Err(PyValueError::new_err(
                "doc_embeddings and word_embeddings must be rectangular with the same width",
            ));
        }

        // Present words per document = its tokens that are in the vocabulary (deduped,
        // ascending). Also build in-vocab id sequences for the coherence corpus.
        let mut doc_words: Vec<Vec<u32>> = Vec::with_capacity(num_docs);
        let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(num_docs);
        // Dedup with a reused per-vocabulary flag buffer (one allocation total) rather
        // than a fresh HashSet per document, so this stays cheap on large corpora.
        let mut seen = vec![false; num_types];
        for doc in &docs_str {
            let mut present: Vec<u32> = Vec::new();
            let mut ids: Vec<u32> = Vec::new();
            for w in doc {
                if let Some(&wid) = word_id.get(w.as_str()) {
                    ids.push(wid);
                    if !seen[wid as usize] {
                        seen[wid as usize] = true;
                        present.push(wid);
                    }
                }
            }
            for &wid in &present {
                seen[wid as usize] = false; // reset only the flags we set
            }
            present.sort_unstable();
            doc_words.push(present);
            id_docs.push(ids);
        }

        let k = slf.num_topics;
        let max_rank = num_docs.min(num_types);
        if k > max_rank {
            return Err(PyValueError::new_err(format!(
                "num_topics ({k}) must be <= min(num_docs, vocab) = {max_rank} \
                 (the NNDSVD initialization requires it)"
            )));
        }

        // Flatten embeddings row-major for the core.
        let doc_flat: Vec<f64> = doc_emb.iter().flatten().copied().collect();
        let word_flat: Vec<f64> = word_emb.iter().flatten().copied().collect();
        let metric = if slf.metric == "dot" {
            Metric::Dot
        } else {
            Metric::Cosine
        };
        let it = iters.unwrap_or(200);
        // Default 0.0 = run the full `iters` budget (no early stop) unless the caller
        // opts in, matching GuidedNMF and keeping a default fit deterministic in its
        // iteration count.
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        let (top_n, seed) = (slf.top_n, slf.seed);
        let model = py.allow_threads(move || {
            keynmf::fit_keynmf(
                &doc_words, &doc_flat, &word_flat, num_docs, num_types, emb_dim, k, top_n, metric,
                it, tol, seed,
            )
        });
        slf.iters = it;
        slf.convergence_tol = tol;

        // Coherence corpus over the same vocabulary.
        let mut doc_freqs = vec![0u32; num_types];
        let mut total_freqs = vec![0u32; num_types];
        for ids in &id_docs {
            let mut seen = std::collections::HashSet::new();
            for &w in ids {
                total_freqs[w as usize] += 1;
                if seen.insert(w) {
                    doc_freqs[w as usize] += 1;
                }
            }
        }
        slf.corpus = Some(corpus::Corpus {
            id_to_word: vocabulary.clone(),
            docs: id_docs,
            doc_names: doc_names.clone(),
            doc_labels: Vec::new(),
            doc_freqs,
            total_freqs,
        });
        slf.model = Some(model);
        slf.id_to_word = vocabulary;
        slf.doc_names = doc_names;
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.nmf.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.nmf.doc_topic).to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok((0..self.num_topics).map(|k| format!("topic_{k}")).collect())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.doc_names.clone())
    }
    /// Per-iteration NMF reconstruction error as `(iter, error)` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .nmf
            .error_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
            .collect())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.nmf.converged)
    }
    /// The NMF reconstruction error (Frobenius) at convergence.
    #[getter]
    fn reconstruction_error(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.nmf.reconstruction_error)
    }

    /// The extracted keywords for a document as `(word, importance)` pairs, sorted by
    /// descending importance — the sparse row of the factored keyword matrix.
    #[pyo3(signature = (doc, n=None))]
    fn keywords(&self, doc: usize, n: Option<usize>) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        let row = m
            .keywords
            .get(doc)
            .ok_or_else(|| PyValueError::new_err(format!("document index {doc} out of range")))?;
        let mut out: Vec<(String, f64)> = row
            .iter()
            .map(|&(w, s)| (self.id_to_word[w as usize].clone(), s))
            .collect();
        if let Some(n) = n {
            out.truncate(n);
        }
        Ok(out)
    }

    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.nmf.topic_word);
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

    #[pyo3(signature = (n=10, *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.nmf.topic_word);
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
        let m = self.model.as_ref();
        let c = self.corpus.as_ref();
        write_state(
            path,
            MODEL_TAG_KEYNMF,
            &KeyNmfState {
                num_topics: self.num_topics,
                top_n: self.top_n,
                metric: self.metric.clone(),
                iters: self.iters,
                convergence_tol: self.convergence_tol,
                seed: self.seed,
                fitted: self.fitted,
                id_to_word: self.id_to_word.clone(),
                doc_names: self.doc_names.clone(),
                doc_freqs: c.map(|c| c.doc_freqs.clone()).unwrap_or_default(),
                total_freqs: c.map(|c| c.total_freqs.clone()).unwrap_or_default(),
                docs: c.map(|c| c.docs.clone()).unwrap_or_default(),
                topic_word: m.map(|m| m.nmf.topic_word.clone()),
                doc_topic: m.map(|m| m.nmf.doc_topic.clone()),
                h: m.map(|m| m.nmf.h.clone()),
                w: m.map(|m| m.nmf.w.clone()),
                reconstruction_error: m.map(|m| m.nmf.reconstruction_error),
                error_history: m.map(|m| m.nmf.error_history.clone()),
                converged: m.map(|m| m.nmf.converged),
                iters_run: m.map(|m| m.nmf.iters_run),
                keywords: m.map(|m| m.keywords.clone()),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: KeyNmfState = read_state(path, MODEL_TAG_KEYNMF)?;
        let corpus = if s.fitted {
            Some(corpus::Corpus {
                id_to_word: s.id_to_word.clone(),
                docs: s.docs.clone(),
                doc_names: s.doc_names.clone(),
                doc_labels: Vec::new(),
                doc_freqs: s.doc_freqs.clone(),
                total_freqs: s.total_freqs.clone(),
            })
        } else {
            None
        };
        let model = if s.fitted && s.topic_word.is_some() {
            Some(KeyNmfModel {
                nmf: crate::nmf::NmfModel {
                    num_topics: s.num_topics,
                    num_types: s.id_to_word.len(),
                    topic_word: s.topic_word.unwrap_or_default(),
                    doc_topic: s.doc_topic.unwrap_or_default(),
                    h: s.h.unwrap_or_default(),
                    w: s.w.unwrap_or_default(),
                    reconstruction_error: s.reconstruction_error.unwrap_or(f64::NAN),
                    error_history: s.error_history.unwrap_or_default(),
                    converged: s.converged.unwrap_or(false),
                    iters_run: s.iters_run.unwrap_or(0),
                },
                keywords: s.keywords.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(KeyNMF {
            num_topics: s.num_topics,
            top_n: s.top_n,
            metric: s.metric,
            iters: s.iters,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            fitted: s.fitted,
            id_to_word: s.id_to_word,
            doc_names: s.doc_names,
            model,
            corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KeyNMF(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
