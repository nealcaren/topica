//! Python bindings for Online Tensor LDA (TensorLDA) topic model.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use pyo3::types::PyType;
use std::collections::HashMap;

/// Online Tensor LDA (TensorLDA) topic model.
/// Gated behind `topica.enable_experimental()`.
#[pyclass(module = "topica")]
pub struct TensorLDA {
    num_topics: usize,
    alpha_0: f64,
    n_iter_train: usize,
    n_iter_test: usize,
    learning_rate: f64,
    batch_size: usize,
    smoothing: f64,
    theta: f64,
    n_eigenvec: Option<usize>,
    pca_batch_size: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::tlda::TensorLdaModel>,
    corpus: Option<corpus::Corpus>,
    /// Streaming state (set by `partial_fit`, consumed by `finalize`).
    stream: Option<crate::tlda::TldaStream>,
    /// word -> id map fixed on the first `partial_fit` call.
    stream_index: Option<HashMap<String, u32>>,
    /// vocabulary (id -> word) fixed on the first `partial_fit` call.
    stream_vocab: Option<Vec<String>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TensorLdaState {
    num_topics: usize,
    alpha_0: f64,
    n_iter_train: usize,
    n_iter_test: usize,
    learning_rate: f64,
    batch_size: usize,
    smoothing: f64,
    theta: f64,
    n_eigenvec: Option<usize>,
    #[serde(default)]
    pca_batch_size: Option<usize>,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    weights: Option<Vec<f64>>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
    unwhitened_raw: Option<Vec<f64>>,
}

impl TensorLDA {
    fn fitted_model(&self) -> PyResult<&crate::tlda::TensorLdaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn require_fitted(&self) -> PyResult<()> {
        if !self.fitted || self.model.is_none() || self.corpus.is_none() {
            return Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ));
        }
        Ok(())
    }
}

fn map_heldout(corpus: &corpus::Corpus, data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
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

    let mut out = Vec::with_capacity(str_docs.len());
    for s_doc in str_docs {
        let mut doc = Vec::new();
        for word in s_doc {
            if let Some(&id) = index.get(word.as_str()) {
                doc.push(id);
            }
        }
        out.push(doc);
    }
    Ok(out)
}

#[pymethods]
impl TensorLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``n_eigenvec`` is ``None`` when left
    /// unset (defaults to ``num_topics`` at fit).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha_0", self.alpha_0)?;
        d.set_item("n_iter_train", self.n_iter_train)?;
        d.set_item("n_iter_test", self.n_iter_test)?;
        d.set_item("learning_rate", self.learning_rate)?;
        d.set_item("batch_size", self.batch_size)?;
        d.set_item("smoothing", self.smoothing)?;
        d.set_item("theta", self.theta)?;
        d.set_item("n_eigenvec", self.n_eigenvec)?;
        d.set_item("pca_batch_size", self.pca_batch_size)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted TensorLDA model.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_0=1.0, n_iter_train=100, n_iter_test=30,
                        learning_rate=0.01, batch_size=10, smoothing=0.01,
                        theta=1.0, n_eigenvec=None, pca_batch_size=128, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_topics: usize,
        alpha_0: f64,
        n_iter_train: usize,
        n_iter_test: usize,
        learning_rate: f64,
        batch_size: usize,
        smoothing: f64,
        theta: f64,
        n_eigenvec: Option<usize>,
        pca_batch_size: usize,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("TensorLDA")?;

        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if alpha_0 <= 0.0 {
            return Err(PyValueError::new_err("alpha_0 must be > 0.0"));
        }
        if n_iter_train == 0 {
            return Err(PyValueError::new_err("n_iter_train must be > 0"));
        }
        if n_iter_test == 0 {
            return Err(PyValueError::new_err("n_iter_test must be > 0"));
        }
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("learning_rate must be > 0.0"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        if !(0.0..1.0).contains(&smoothing) {
            return Err(PyValueError::new_err("smoothing must be in [0.0, 1.0)"));
        }
        if theta <= 0.0 {
            return Err(PyValueError::new_err("theta must be > 0.0"));
        }
        if let Some(ne) = n_eigenvec {
            if ne < num_topics {
                return Err(PyValueError::new_err("n_eigenvec must be >= num_topics"));
            }
        }
        if pca_batch_size == 0 {
            return Err(PyValueError::new_err("pca_batch_size must be > 0"));
        }
        Ok(TensorLDA {
            num_topics,
            alpha_0,
            n_iter_train,
            n_iter_test,
            learning_rate,
            batch_size,
            smoothing,
            theta,
            n_eigenvec,
            pca_batch_size,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
            stream: None,
            stream_index: None,
            stream_vocab: None,
        })
    }

    /// Fit the model on the given corpus or token lists.
    #[pyo3(signature = (data, *, iters=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: Option<usize>,
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
        let num_docs = corpus.num_docs();
        let num_types = corpus.num_types();
        if num_docs < slf.num_topics {
            return Err(PyValueError::new_err(format!(
                "corpus must have at least num_topics={} documents, got {}",
                slf.num_topics, num_docs
            )));
        }
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }

        let max_rank = num_docs.min(num_types);
        let n_eigen = slf.n_eigenvec.unwrap_or(slf.num_topics);
        if n_eigen > max_rank {
            return Err(PyValueError::new_err(format!(
                "whitening rank n_eigenvec={} cannot exceed min(num_docs, vocab_size)={}",
                n_eigen, max_rank
            )));
        }

        let n_iter_train = iters.unwrap_or(slf.n_iter_train);
        let (k, alpha_0, n_iter_test, lr, bs, smoothing, theta, n_eigen, seed) = (
            slf.num_topics,
            slf.alpha_0,
            slf.n_iter_test,
            slf.learning_rate,
            slf.batch_size,
            slf.smoothing,
            slf.theta,
            slf.n_eigenvec,
            slf.seed,
        );

        let (model, corpus) = py.allow_threads(move || {
            let m = crate::tlda::fit_tlda(
                &corpus.docs,
                k,
                num_types,
                alpha_0,
                n_iter_train,
                n_iter_test,
                lr,
                bs,
                smoothing,
                theta,
                n_eigen,
                seed,
            );
            (m, corpus)
        });

        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        // A batch fit supersedes any half-built stream.
        slf.stream = None;
        slf.stream_index = None;
        slf.stream_vocab = None;
        Ok(slf.into())
    }

    /// Update the model with a batch of documents (streaming / online fit).
    ///
    /// The FIRST time a `batch_index` is seen this updates the running mean and
    /// the incremental whitening only; every LATER sighting whitens the batch
    /// and runs the CP factor SGD. So the usual driver makes one pass over the
    /// batches to build the whitening, then several passes to train the factors,
    /// never holding the whole corpus in memory:
    ///
    /// ```python
    /// m = topica.TensorLDA(num_topics=10, seed=0)
    /// for _ in range(1 + n_iter_train):
    ///     for i, batch in enumerate(batches):
    ///         m.partial_fit(batch, i, vocabulary=vocab)
    /// m.finalize()
    /// ```
    ///
    /// Fix the vocabulary once with `vocabulary=` (recommended for streaming) or
    /// let the first batch define it; out-of-vocabulary tokens in later batches
    /// are dropped. Call `finalize()` when the stream is done.
    #[pyo3(signature = (data, batch_index, *, vocabulary=None))]
    fn partial_fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        batch_index: i64,
        vocabulary: Option<Vec<String>>,
    ) -> PyResult<()> {
        if self.fitted {
            return Err(PyRuntimeError::new_err(
                "model is already finalized; construct a new TensorLDA to stream again",
            ));
        }
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
                PyValueError::new_err("partial_fit() expects a Corpus or a list of token lists")
            })?
        };
        if str_docs.is_empty() {
            return Err(PyValueError::new_err("partial_fit() got an empty batch"));
        }

        // Establish the fixed vocabulary + streaming state on the first call.
        if self.stream.is_none() {
            let vocab = if let Some(v) = vocabulary {
                if v.is_empty() {
                    return Err(PyValueError::new_err("vocabulary must be non-empty"));
                }
                // Duplicate entries would inflate the vocab size while the
                // word->id map collapses them, leaving dead count columns and
                // letting the n_eigenvec rank check pass above the true rank.
                let unique: std::collections::HashSet<&String> = v.iter().collect();
                if unique.len() != v.len() {
                    return Err(PyValueError::new_err(
                        "vocabulary must not contain duplicate words",
                    ));
                }
                v
            } else {
                let mut seen = std::collections::BTreeSet::new();
                for doc in &str_docs {
                    for w in doc {
                        seen.insert(w.clone());
                    }
                }
                seen.into_iter().collect::<Vec<String>>()
            };
            let vsize = vocab.len();
            if vsize < self.num_topics {
                return Err(PyValueError::new_err(
                    "vocabulary must have at least num_topics words",
                ));
            }
            let n_eigen = self.n_eigenvec.unwrap_or(self.num_topics);
            if n_eigen > vsize {
                return Err(PyValueError::new_err(format!(
                    "whitening rank n_eigenvec={n_eigen} cannot exceed vocabulary size {vsize}",
                )));
            }
            let index: HashMap<String, u32> = vocab
                .iter()
                .enumerate()
                .map(|(i, w)| (w.clone(), i as u32))
                .collect();
            self.stream = Some(crate::tlda::TldaStream::new(
                self.num_topics,
                vsize,
                self.n_eigenvec,
                self.alpha_0,
                self.theta,
                self.learning_rate,
                self.pca_batch_size,
                self.batch_size,
                self.smoothing,
                self.seed,
            ));
            self.stream_index = Some(index);
            self.stream_vocab = Some(vocab);
        }

        let v = self.stream_vocab.as_ref().unwrap().len();
        let index = self.stream_index.as_ref().unwrap();
        let n = str_docs.len();
        let mut counts = vec![0.0; n * v];
        for (i, doc) in str_docs.iter().enumerate() {
            for w in doc {
                if let Some(&id) = index.get(w) {
                    counts[i * v + id as usize] += 1.0;
                }
            }
        }
        let stream = self.stream.as_mut().unwrap();
        py.allow_threads(move || stream.partial_fit_batch(&counts, n, batch_index));
        Ok(())
    }

    /// Recover the topic-word matrix and weights from the streamed batches and
    /// mark the model fitted. Errors unless every batch was fed at least twice
    /// (one whitening pass, then at least one training pass). After this,
    /// `topic_word`, `weights`, `top_words`, and `transform` are available;
    /// `doc_topic` is not (streaming does not retain the documents -- use
    /// `transform` on the docs whose topics you want).
    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        let stream = match self.stream.as_ref() {
            Some(s) => s,
            None => {
                return Err(PyRuntimeError::new_err(
                    "no streamed batches; call partial_fit() before finalize()",
                ))
            }
        };
        if !stream.trained() {
            return Err(PyRuntimeError::new_err(
                "each batch must be seen at least twice: one pass to build the whitening, \
                 then one or more passes to train the factors",
            ));
        }
        // The whitening rank cannot exceed the number of documents seen during
        // the whitening pass -- the streaming counterpart of the batch fit's
        // `n_eigenvec <= min(num_docs, vocab)` guard. Below that, the small
        // singular values are clamped and the whitening blows up.
        if stream.n_documents() < stream.n_eigen() {
            return Err(PyRuntimeError::new_err(format!(
                "whitening rank n_eigenvec={} exceeds the {} documents streamed; \
                 feed more documents or lower n_eigenvec/num_topics",
                stream.n_eigen(),
                stream.n_documents(),
            )));
        }
        let model = py.allow_threads(move || stream.finalize());

        let vocab = self.stream_vocab.take().unwrap();
        let vsize = vocab.len();
        let corpus = corpus::Corpus {
            id_to_word: vocab,
            docs: Vec::new(),
            doc_names: Vec::new(),
            doc_labels: Vec::new(),
            doc_freqs: vec![0; vsize],
            total_freqs: vec![0; vsize],
        };
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        self.stream = None;
        self.stream_index = None;
        Ok(())
    }

    /// Transform new documents using the fitted model to get their document-topic distributions.
    #[pyo3(signature = (data, *, seed=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let trained_corpus = self.corpus.as_ref().unwrap();
        let mapped_docs = map_heldout(trained_corpus, data)?;

        let d = mapped_docs.len();
        let v = trained_corpus.num_types();
        let mut x_test = vec![0.0; d * v];
        for (i, doc) in mapped_docs.iter().enumerate() {
            for &w in doc {
                if (w as usize) < v {
                    x_test[i * v + w as usize] += 1.0;
                }
            }
        }

        let m = self.model.as_ref().unwrap();
        let doc_topic = py.allow_threads(move || {
            crate::tlda::predict_doc_topics(
                &x_test,
                &m.unwhitened_raw,
                &m.weights,
                self.num_topics,
                v,
                self.n_iter_test,
                seed.unwrap_or(self.seed),
            )
        });

        Ok(vecs_to_arr2(&doc_topic).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Topic-word matrix (num_topics, vocab); rows sum to 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }

    /// Document-topic matrix (num_docs, num_topics); rows sum to 1. Not available
    /// for a model built by streaming `partial_fit` (the documents are not
    /// retained) -- call `transform(docs)` on the documents whose topics you want.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.doc_topic.is_empty() {
            // Raise AttributeError (not RuntimeError) so `hasattr(m, "doc_topic")`
            // dispatch guards in topica.ensemble / effects / coherence return
            // False for a streamed model instead of crashing.
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "doc_topic is unavailable for a streamed (partial_fit) model; \
                 call transform(docs) to infer document topics",
            ));
        }
        Ok(vecs_to_arr2(&m.doc_topic).to_pyarray_bound(py))
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.weights.clone()).to_pyarray_bound(py))
    }

    /// Unwhitened, raw factor matrix (vocab_size, num_topics) before normalization.
    #[getter]
    fn unwhitened_raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let v = self.corpus.as_ref().unwrap().num_types();
        let k = self.num_topics;
        let mut raw_matrix = vec![vec![0.0; k]; v];
        for w in 0..v {
            for j in 0..k {
                raw_matrix[w][j] = m.unwhitened_raw[w * k + j];
            }
        }
        Ok(vecs_to_arr2(&raw_matrix).to_pyarray_bound(py))
    }

    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }

    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
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

    #[getter]
    fn model_family(&self) -> &'static str {
        "none"
    }

    /// Top-`n` words per topic.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
        )
    }

    /// Topic coherence (UMass).
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.fitted_model()?;
        // UMass coherence reads document co-occurrences; a streamed model does not
        // retain the documents, so it would silently return all-zeros. Refuse it.
        if self.corpus.as_ref().is_some_and(|c| c.docs.is_empty()) {
            return Err(PyRuntimeError::new_err(
                "coherence is unavailable for a streamed (partial_fit) model; \
                 it needs the training documents, which streaming does not retain",
            ));
        }
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_TLDA,
            &TensorLdaState {
                num_topics: self.num_topics,
                alpha_0: self.alpha_0,
                n_iter_train: self.n_iter_train,
                n_iter_test: self.n_iter_test,
                learning_rate: self.learning_rate,
                batch_size: self.batch_size,
                smoothing: self.smoothing,
                theta: self.theta,
                n_eigenvec: self.n_eigenvec,
                pca_batch_size: Some(self.pca_batch_size),
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                num_types: Some(m.num_types),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                weights: Some(m.weights.clone()),
                fit_history: Some(m.fit_history.clone()),
                converged: Some(m.converged),
                unwhitened_raw: Some(m.unwhitened_raw.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let _py = cls.py();
        require_experimental("TensorLDA")?;
        let s: TensorLdaState = read_state(path, MODEL_TAG_TLDA)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(crate::tlda::TensorLdaModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                weights: s.weights.unwrap_or_default(),
                alpha_0: s.alpha_0,
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                unwhitened_raw: s.unwhitened_raw.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(TensorLDA {
            num_topics: s.num_topics,
            alpha_0: s.alpha_0,
            n_iter_train: s.n_iter_train,
            n_iter_test: s.n_iter_test,
            learning_rate: s.learning_rate,
            batch_size: s.batch_size,
            smoothing: s.smoothing,
            theta: s.theta,
            n_eigenvec: s.n_eigenvec,
            pca_batch_size: s.pca_batch_size.unwrap_or(128),
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
            stream: None,
            stream_index: None,
            stream_vocab: None,
        })
    }
}
