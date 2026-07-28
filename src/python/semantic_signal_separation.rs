//! Python bindings for SemanticSignalSeparation (S³).
//!
//! S³ is embedding-native: the caller brings a document-embedding matrix and an
//! aligned vocabulary-embedding matrix, exactly like [`crate::top2vec`]. The
//! binding is thin plumbing over `crate::semantic_signal_separation::fit`.

use super::*;
use crate::semantic_signal_separation::{
    fit as s3_fit, FeatureImportance, SemanticSignalSeparationModel,
};
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyList, PyTuple};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Semantic Signal Separation (S³): topics as independent axes of semantic space.
///
/// S³ (Kardos et al., "S³ - Semantic Signal Separation", ACL 2025) decomposes the
/// document embeddings with FastICA; each independent component is a topic axis,
/// and a word's importance to a topic comes from projecting the vocabulary
/// embeddings onto that axis. It works directly in the embedding space, with no
/// bag-of-words modelling, and is fully deterministic from ``seed``.
///
/// ICA axes are signed: each topic has a positive and a negative pole. ``components``
/// (K x V) and ``source_scores`` (D x K) are the signed native outputs; ``topic_word``
/// and ``doc_topic`` are the nonnegative, row-normalized positive poles that the rest
/// of topica's analysis surface consumes. ``top_words(..., pole="negative")`` reaches
/// the negative pole.
///
/// You bring the embeddings. ``fit(data, doc_embeddings, vocab_embeddings, *,
/// vocabulary=None)`` needs one ``doc_embeddings`` row per document and one
/// ``vocab_embeddings`` row per vocabulary term, in the same embedding space. Pass
/// ``vocabulary`` (the words matching ``vocab_embeddings`` rows) to realign them to
/// topica's vocabulary; omit it only when the rows are already in the corpus
/// vocabulary order.
///
/// Reference: turftopic ``SemanticSignalSeparation`` (MIT, Márton Kardos).
#[pyclass(module = "topica")]
pub struct SemanticSignalSeparation {
    num_topics: usize,
    feature_importance: String,
    iters: usize,
    tol: f64,
    seed: u64,
    fitted: bool,
    model: Option<SemanticSignalSeparationModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

impl SemanticSignalSeparation {
    fn fitted_model(&self) -> PyResult<&SemanticSignalSeparationModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Build a minimal `Corpus` from the retained tokens for UMass coherence
    /// (mirrors the embedding-cluster models; S³ itself works on embeddings).
    fn coherence_corpus(&self) -> corpus::Corpus {
        let v = self.id_to_word.len();
        let mut doc_freqs = vec![0u32; v];
        let mut total_freqs = vec![0u32; v];
        for doc in &self.docs {
            let mut seen = std::collections::HashSet::new();
            for &w in doc {
                let wi = w as usize;
                if wi < v {
                    total_freqs[wi] += 1;
                    seen.insert(wi);
                }
            }
            for wi in seen {
                doc_freqs[wi] += 1;
            }
        }
        corpus::Corpus {
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
            doc_names: (0..self.docs.len()).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); self.docs.len()],
            doc_freqs,
            total_freqs,
        }
    }
}

/// Serialized fitted state.
#[derive(serde::Serialize, serde::Deserialize)]
struct S3State {
    num_topics: usize,
    feature_importance: String,
    iters: usize,
    tol: f64,
    seed: u64,
    fitted: bool,
    model: Option<SemanticSignalSeparationModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

#[pymethods]
impl SemanticSignalSeparation {
    #[new]
    #[pyo3(signature = (num_topics, *, feature_importance="combined".to_string(), iters=200, convergence_tol=1e-4, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        feature_importance: String,
        iters: usize,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if FeatureImportance::parse(&feature_importance).is_none() {
            return Err(PyValueError::new_err(
                "feature_importance must be one of 'combined', 'axial', 'angular'",
            ));
        }
        if iters < 1 {
            return Err(PyValueError::new_err("iters must be >= 1"));
        }
        if !convergence_tol.is_finite() || convergence_tol <= 0.0 {
            return Err(PyValueError::new_err(
                "convergence_tol must be a finite value > 0",
            ));
        }
        Ok(SemanticSignalSeparation {
            num_topics,
            feature_importance,
            iters,
            tol: convergence_tol,
            seed,
            fitted: false,
            model: None,
            id_to_word: Vec::new(),
            docs: Vec::new(),
        })
    }

    /// Fit S³ on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`) and `vocab_embeddings` (`(len(vocabulary), E)`), in one
    /// shared embedding space. Pass `vocabulary` to realign `vocab_embeddings` to
    /// topica's vocabulary; omit it when the rows already match the corpus order.
    #[pyo3(signature = (data, doc_embeddings, vocab_embeddings, *, vocabulary=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        vocab_embeddings: &Bound<'_, PyAny>,
        vocabulary: Option<Vec<String>>,
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
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let e = doc_emb.first().map(|r| r.len()).unwrap_or(0);
        if e == 0 {
            return Err(PyValueError::new_err("doc_embeddings has zero-width rows"));
        }
        if doc_emb.iter().any(|r| r.len() != e) {
            return Err(PyValueError::new_err(
                "doc_embeddings has ragged rows; every document vector must have the same length",
            ));
        }

        let raw_vocab = parse_features(vocab_embeddings)?;
        check_all_finite_2d("vocab_embeddings", &raw_vocab)?;
        if raw_vocab.iter().any(|r| r.len() != e) {
            return Err(PyValueError::new_err(format!(
                "vocab_embeddings dim must equal doc_embeddings dim ({e}); word and document \
                 vectors must live in the same embedding space"
            )));
        }
        let num_types = corpus.num_types();
        // Realign vocab embeddings to topica's vocabulary order. With `vocabulary`,
        // map by word; a corpus term the caller supplied no embedding for gets a
        // placeholder row AND a `missing` flag, so the core zeroes its importance
        // (a placeholder vector would otherwise project to a spurious score).
        // Without `vocabulary`, the rows must already be in corpus order (none
        // missing).
        let (vocab_emb, missing): (Vec<Vec<f64>>, Vec<bool>) = match vocabulary {
            Some(vocab) => {
                if raw_vocab.len() != vocab.len() {
                    return Err(PyValueError::new_err(format!(
                        "vocab_embeddings has {} rows but vocabulary has {} words",
                        raw_vocab.len(),
                        vocab.len()
                    )));
                }
                let mut map: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::with_capacity(vocab.len());
                for (i, w) in vocab.iter().enumerate() {
                    if map.insert(w.as_str(), i).is_some() {
                        return Err(PyValueError::new_err(format!(
                            "vocabulary has a duplicate word '{w}'; every word must be unique"
                        )));
                    }
                }
                let mut emb = Vec::with_capacity(num_types);
                let mut miss = Vec::with_capacity(num_types);
                for w in &corpus.id_to_word {
                    match map.get(w.as_str()) {
                        Some(&i) => {
                            emb.push(raw_vocab[i].clone());
                            miss.push(false);
                        }
                        None => {
                            emb.push(vec![0.0; e]);
                            miss.push(true);
                        }
                    }
                }
                if miss.iter().all(|&x| x) {
                    return Err(PyValueError::new_err(
                        "no corpus term has a vocab_embeddings row; check that `vocabulary` \
                         covers the corpus vocabulary",
                    ));
                }
                (emb, miss)
            }
            None => {
                if raw_vocab.len() != num_types {
                    return Err(PyValueError::new_err(format!(
                        "vocab_embeddings has {} rows but the corpus vocabulary has {} terms; \
                         pass vocabulary= to align them by word",
                        raw_vocab.len(),
                        num_types
                    )));
                }
                (raw_vocab, Vec::new())
            }
        };

        let k = slf.num_topics;
        let max_k = doc_emb.len().min(e);
        if k > max_k {
            return Err(PyValueError::new_err(format!(
                "num_topics ({k}) must be <= min(num_docs, embedding_dim) = {max_k} for FastICA"
            )));
        }
        let fi = FeatureImportance::parse(&slf.feature_importance).unwrap();
        let (iters, tol, seed) = (slf.iters, slf.tol, slf.seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let model = py.allow_threads(move || {
            s3_fit(&doc_emb, &vocab_emb, k, fi, iters, tol, &missing, &mut rng)
        });

        // A non-finite fit (e.g. embeddings with pathological scale/conditioning)
        // must not be silently rectified into a "valid" uniform topic_word; surface
        // it as an error the caller can act on.
        let finite = |rows: &[Vec<f64>]| rows.iter().all(|r| r.iter().all(|x| x.is_finite()));
        if !finite(&model.components) || !finite(&model.source_scores) {
            return Err(PyValueError::new_err(
                "S³ fit produced non-finite values; check that the embeddings are finite and \
                 reasonably scaled (e.g. L2-normalized)",
            ));
        }

        slf.id_to_word = corpus.id_to_word.clone();
        slf.docs = corpus.docs.clone();
        slf.model = Some(model);
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("feature_importance", &self.feature_importance)?;
        d.set_item("iters", self.iters)?;
        d.set_item("convergence_tol", self.tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    // --- Required analysis surface ---
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
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

    // --- Signed S³-native outputs ---
    /// Signed per-word importance under `feature_importance` (K x V): the positive
    /// and negative poles of each topic axis.
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.components).to_pyarray_bound(py))
    }
    /// The raw axial projection of the vocabulary onto each axis (K x V),
    /// turftopic's `axial_components_`, before the angular/combined reweighting.
    #[getter]
    fn axial_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.axial).to_pyarray_bound(py))
    }
    /// Signed document loadings on each axis, the raw ICA sources (D x K).
    #[getter]
    fn source_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.source_scores).to_pyarray_bound(py))
    }
    /// Whether FastICA reached `convergence_tol` before `iters`.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// `(iteration, convergence measure)` for the FastICA fixed-point, one entry
    /// per iteration until convergence.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }

    // --- Conventional extras ---
    /// The top `n` words of a topic. `pole="positive"` (default) returns the
    /// highest-loading words on the axis; `pole="negative"` returns the opposite
    /// pole (the most negatively loaded words), with their signed importance.
    #[pyo3(signature = (n=10, *, topic=None, pole="positive"))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        pole: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.fitted_model()?;
        let negative = match pole {
            "positive" => false,
            "negative" => true,
            _ => {
                return Err(PyValueError::new_err(
                    "pole must be 'positive' or 'negative'",
                ))
            }
        };
        let one = |t: usize| -> Bound<'py, PyList> {
            let items: Vec<Bound<'py, PyTuple>> = model
                .top_words(n, t, negative)
                .into_iter()
                .map(|(i, w)| {
                    PyTuple::new_bound(py, &[self.id_to_word[i].clone().into_py(py), w.into_py(py)])
                })
                .collect();
            PyList::new_bound(py, items)
        };
        match topic {
            Some(t) => {
                if t >= self.num_topics {
                    return Err(PyValueError::new_err(format!(
                        "topic {t} out of range (num_topics={})",
                        self.num_topics
                    )));
                }
                Ok(one(t).into_any())
            }
            None => {
                let all: Vec<Bound<'py, PyList>> = (0..self.num_topics).map(one).collect();
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        let corpus = self.coherence_corpus();
        Ok(Array1::from(umass_coherence(&corpus, &tops)).to_pyarray_bound(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_S3,
            &S3State {
                num_topics: self.num_topics,
                feature_importance: self.feature_importance.clone(),
                iters: self.iters,
                tol: self.tol,
                seed: self.seed,
                fitted: self.fitted,
                model: self.model.clone(),
                id_to_word: self.id_to_word.clone(),
                docs: self.docs.clone(),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: S3State = read_state(path, MODEL_TAG_S3)?;
        Ok(SemanticSignalSeparation {
            num_topics: s.num_topics,
            feature_importance: s.feature_importance,
            iters: s.iters,
            tol: s.tol,
            seed: s.seed,
            fitted: s.fitted,
            model: s.model,
            id_to_word: s.id_to_word,
            docs: s.docs,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticSignalSeparation(num_topics={}, feature_importance='{}', fitted={})",
            self.num_topics, self.feature_importance, self.fitted
        )
    }
}
