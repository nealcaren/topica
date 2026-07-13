//! Python bindings for Online Tensor LDA (TensorLDA) topic model.

use super::*;
use numpy::{PyArray1, PyArray2};
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
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::tlda::TensorLdaModel>,
    corpus: Option<corpus::Corpus>,
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
    /// Create an unfitted TensorLDA model.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_0=1.0, n_iter_train=100, n_iter_test=30,
                        learning_rate=0.01, batch_size=10, smoothing=0.01, seed=42))]
    fn new(
        num_topics: usize,
        alpha_0: f64,
        n_iter_train: usize,
        n_iter_test: usize,
        learning_rate: f64,
        batch_size: usize,
        smoothing: f64,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("TensorLDA")?;

        if num_topics < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        Ok(TensorLDA {
            num_topics,
            alpha_0,
            n_iter_train,
            n_iter_test,
            learning_rate,
            batch_size,
            smoothing,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit the model on the given corpus or token lists.
    #[pyo3(signature = (data, *, iters=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: Option<usize>,
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
        let num_types = corpus.num_types();
        if num_types < self.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }

        let n_iter_train = iters.unwrap_or(self.n_iter_train);
        let (k, alpha_0, n_iter_test, lr, bs, smoothing, seed) = (
            self.num_topics,
            self.alpha_0,
            self.n_iter_test,
            self.learning_rate,
            self.batch_size,
            self.smoothing,
            self.seed,
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
                seed,
            );
            (m, corpus)
        });

        self.model = Some(model);
        self.corpus = Some(corpus);
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
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

    /// Document-topic matrix (num_docs, num_topics); rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.weights.clone()).to_pyarray_bound(py))
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
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }
}
