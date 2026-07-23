//! Python bindings for the Biterm Topic Model (BTM).

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use pyo3::types::PyType;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Biterm Topic Model (Yan et al. 2013): a word co-occurrence topic model for
/// short text. Instead of a per-document topic mixture (which short texts are too
/// sparse to estimate), BTM learns one global topic distribution and per-topic
/// word distributions from the corpus's **biterms** -- unordered word pairs
/// co-occurring within a window.
#[pyclass(module = "topica")]
pub struct BTM {
    num_topics: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    window: usize,
    background: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::btm::BtmModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BtmState {
    num_topics: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    window: usize,
    background: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    resolved_alpha: Option<f64>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    theta: Option<Vec<f64>>,
    num_biterms: Option<usize>,
}

impl BTM {
    fn fitted_model(&self) -> PyResult<&crate::btm::BtmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

/// Map new documents onto the training vocabulary (dropping out-of-vocabulary
/// tokens), for `transform`. Mirrors the same helper the other models use.
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
impl BTM {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``alpha`` is ``None`` when left at its
    /// ``50 / num_topics`` default (resolved only at fit).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("iters", self.iters)?;
        d.set_item("window", self.window)?;
        d.set_item("background", self.background)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted BTM model. `alpha` defaults to `50 / num_topics`
    /// (the reference default), `beta` to `0.01`, `window` to 15.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=None, beta=0.01, iters=1000,
                        window=15, background=false, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_topics: usize,
        alpha: Option<f64>,
        beta: f64,
        iters: usize,
        window: usize,
        background: bool,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if let Some(a) = alpha {
            if a <= 0.0 {
                return Err(PyValueError::new_err("alpha must be > 0.0"));
            }
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0.0"));
        }
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be > 0"));
        }
        if window < 2 {
            return Err(PyValueError::new_err(
                "window must be >= 2 (biterms need word pairs)",
            ));
        }
        Ok(BTM {
            num_topics,
            alpha,
            beta,
            iters,
            window,
            background,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit the model on a corpus or list of token lists.
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
        let num_types = corpus.num_types();
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }
        let iters = iters.unwrap_or(slf.iters);
        let alpha = slf.alpha.unwrap_or(50.0 / slf.num_topics as f64);
        let (k, beta, window, background, seed) = (
            slf.num_topics,
            slf.beta,
            slf.window,
            slf.background,
            slf.seed,
        );

        let (model, corpus) = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let m = crate::btm::fit_btm(
                &corpus.docs,
                k,
                num_types,
                alpha,
                beta,
                iters,
                window,
                background,
                &mut rng,
            );
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Infer document-topic distributions for new documents (the `sum_b` scheme).
    /// Out-of-vocabulary tokens are dropped before biterms are formed (the
    /// reference keeps them as window fillers and drops only the biterms that
    /// contain them); documents left with no in-vocabulary words return a uniform
    /// simplex.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let corpus = self.corpus.as_ref().unwrap();
        let mapped = map_to_vocab(corpus, data)?;
        let dt: Vec<Vec<f64>> = py.allow_threads(|| {
            mapped
                .iter()
                .map(|doc| {
                    crate::btm::infer_doc(
                        doc,
                        &m.theta,
                        &m.topic_word,
                        self.num_topics,
                        self.window,
                    )
                })
                .collect()
        });
        Ok(vecs_to_arr2(&dt).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Topic-word matrix φ (num_topics, vocab); rows sum to 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }

    /// Document-topic matrix (num_docs, num_topics); rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Global topic distribution θ (num_topics); sums to 1. BTM's corpus-level
    /// topic prevalence, the counterpart to a per-document mixture.
    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.theta.clone()).to_pyarray_bound(py))
    }

    /// Number of biterms extracted from the training corpus.
    #[getter]
    fn num_biterms(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_biterms)
    }

    #[getter]
    fn model_family(&self) -> &'static str {
        "none"
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

    fn __repr__(&self) -> String {
        format!(
            "BTM(num_topics={}, window={}, fitted={})",
            self.num_topics, self.window, self.fitted
        )
    }

    /// Save the fitted model to `path`. Reload with `BTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_BTM,
            &BtmState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                beta: self.beta,
                iters: self.iters,
                window: self.window,
                background: self.background,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                num_types: Some(m.num_types),
                resolved_alpha: Some(m.alpha),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                theta: Some(m.theta.clone()),
                num_biterms: Some(m.num_biterms),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: BtmState = read_state(path, MODEL_TAG_BTM)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(crate::btm::BtmModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                alpha: s
                    .resolved_alpha
                    .unwrap_or(s.alpha.unwrap_or(50.0 / s.num_topics as f64)),
                beta: s.beta,
                window: s.window,
                background: s.background,
                topic_word: s.topic_word.unwrap_or_default(),
                theta: s.theta.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                num_biterms: s.num_biterms.unwrap_or(0),
            })
        } else {
            None
        };
        Ok(BTM {
            num_topics: s.num_topics,
            alpha: s.alpha,
            beta: s.beta,
            iters: s.iters,
            window: s.window,
            background: s.background,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }
}
