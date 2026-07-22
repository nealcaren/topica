//! Python bindings for DiscLDA (Lacoste-Julien, Sha & Jordan 2008).

use super::*;
use numpy::{PyArray1, PyArray2};
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

    /// Create an unfitted DiscLDA. `k_class` is the number of class-specific topics
    /// per class, `k_shared` the number of shared topics; the total topic count is
    /// `num_classes * k_class + k_shared`, with `num_classes` taken from the labels
    /// at fit. `alpha` defaults to 0.1 (per allowed topic), `beta` to 0.01.
    /// `infer_sweeps` is the restricted-Gibbs passes used per class in
    /// `transform`/`predict`.
    #[new]
    #[pyo3(signature = (k_class, k_shared, *, alpha=None, beta=0.01, iters=1000,
                        infer_sweeps=100, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_class: usize,
        k_shared: usize,
        alpha: Option<f64>,
        beta: f64,
        iters: usize,
        infer_sweeps: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if k_class == 0 {
            return Err(PyValueError::new_err("k_class must be >= 1"));
        }
        if k_shared == 0 {
            return Err(PyValueError::new_err("k_shared must be >= 1"));
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
        Ok(DiscLDA {
            k_class,
            k_shared,
            alpha,
            beta,
            iters,
            infer_sweeps,
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
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        iters: Option<usize>,
    ) -> PyResult<()> {
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
        let l = num_classes * self.k_class + self.k_shared;
        if num_types < l {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_classes*k_class + k_shared words",
            ));
        }
        let iters = iters.unwrap_or(self.iters);
        let alpha = self.alpha.unwrap_or(0.1);
        let (k_class, k_shared, beta, seed) = (self.k_class, self.k_shared, self.beta, self.seed);

        let (model, corpus) = py.allow_threads(move || {
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
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.classes = classes;
        self.topic_names = self.build_topic_names();
        self.fitted = true;
        Ok(())
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

    /// Predict the class label of each document (MAP under p(class|words)). A
    /// document with no in-vocabulary tokens has a uniform posterior and resolves to
    /// the first class.
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

    /// Class posterior probabilities p(class|words) for each document
    /// (num_docs, num_classes), columns in `classes` order.
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
            m.num_topics,
            n,
            topic,
        )
    }

    /// Topic coherence (UMass) for all topics.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
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
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: DiscLdaState = read_state(path, MODEL_TAG_DISCLDA)?;
        let model = if s.fitted && s.topic_word.is_some() {
            let num_classes = s.classes.len();
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
        topic_words_helper(py, &phi, vocab, ids.len(), n, None)
    }
}
