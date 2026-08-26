//! Python bindings for CSATM (Conversational Structure Aware and Context
//! Sensitive Topic Model). Mirrors the GSDMM/BTM binding shape; see
//! .github/CONTRIBUTING-MODELS.md section B2.

use super::*;
use crate::csatm::{CsatmParams, WeightSeq};
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// CSATM: a collapsed-Gibbs topic model for threaded forum discussions (posts +
/// nested reply trees, one comment per document). It weights each comment's
/// tokens by a reply-tree "popularity" score and, after inference, smooths each
/// comment's topic distribution toward its ancestors along the reply path
/// ("transitivity"). Reference: Sun, Loparo & Kolacinski, IEEE ICSC 2020,
/// arXiv:2002.02353.
#[pyclass(module = "topica")]
pub struct CSATM {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    lambda_: f64,
    weight_seq: String,
    weight_c: f64,
    weight_d: f64,
    weight_g: f64,
    seed: u64,
    fitted: bool,
    model: Option<crate::csatm::CSATMModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CsatmState {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    lambda_: f64,
    weight_seq: String,
    weight_c: f64,
    weight_d: f64,
    weight_g: f64,
    seed: u64,
    fitted: bool,
    corpus: Option<corpus::Corpus>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    doc_topic_raw: Option<Vec<Vec<f64>>>,
    popularity: Option<Vec<f64>>,
    doc_lengths: Option<Vec<usize>>,
}

impl CSATM {
    fn fitted_model(&self) -> PyResult<&crate::csatm::CSATMModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Build the level-weight sequence from the constructor settings.
    fn weight_sequence(&self) -> WeightSeq {
        match self.weight_seq.as_str() {
            "geometric" => WeightSeq::Geometric {
                c: self.weight_c,
                r: self.weight_d,
            },
            "harmonic" => WeightSeq::Harmonic {
                c: self.weight_c,
                b: self.weight_d,
                g: self.weight_g,
            },
            // "arithmetic" (default) and anything else validated in `new`.
            _ => WeightSeq::Arithmetic {
                c: self.weight_c,
                d: self.weight_d,
            },
        }
    }
}

#[pymethods]
impl CSATM {
    /// Create an unfitted model. `num_topics` is the number of topics K.
    /// `alpha`/`beta` are the symmetric doc-topic / topic-word Dirichlet priors
    /// (paper defaults 0.1 / 0.01). `lambda_` scales the popularity weight applied
    /// to token counts (not paper-specified for CSATM; 0.1 is a documented
    /// default). `weight_seq` selects the decreasing level-weight sequence used by
    /// both popularity and transitivity: "arithmetic" (default, `w_l =
    /// max(weight_c - (l-1)*weight_d, 0)`), "geometric" (`weight_c *
    /// weight_d^(l-1)`), or "harmonic" (`(weight_c + (l-1)*weight_d)^(-weight_g)`).
    #[new]
    #[pyo3(signature = (
        num_topics, *, alpha=0.1, beta=0.01, lambda_=0.1,
        weight_seq="arithmetic".to_string(),
        weight_c=1.0, weight_d=0.5, weight_g=1.0, seed=13
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: f64,
        beta: f64,
        lambda_: f64,
        weight_seq: String,
        weight_c: f64,
        weight_d: f64,
        weight_g: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(PyValueError::new_err("alpha must be finite and > 0"));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(PyValueError::new_err("beta must be finite and > 0"));
        }
        if !(lambda_.is_finite() && lambda_ > 0.0) {
            return Err(PyValueError::new_err("lambda_ must be finite and > 0"));
        }
        if !matches!(weight_seq.as_str(), "arithmetic" | "geometric" | "harmonic") {
            return Err(PyValueError::new_err(
                "weight_seq must be 'arithmetic', 'geometric', or 'harmonic'",
            ));
        }
        for (name, val) in [("weight_c", weight_c), ("weight_d", weight_d), ("weight_g", weight_g)] {
            if !val.is_finite() {
                return Err(PyValueError::new_err(format!("{name} must be finite")));
            }
        }
        Ok(CSATM {
            num_topics,
            alpha,
            beta,
            lambda_,
            weight_seq,
            weight_c,
            weight_d,
            weight_g,
            seed,
            fitted: false,
            model: None,
            corpus: None,
        })
    }

    /// Fit CSATM. `data` is a `Corpus` or a list of token lists. `parents` is an
    /// optional per-document list of parent document indices in the reply tree
    /// (`-1` for a thread root / post); when omitted, every document is treated as
    /// a root, so with `lambda_=1` the fit reduces to ordinary LDA. `iters` is the
    /// number of collapsed-Gibbs sweeps.
    #[pyo3(signature = (data, parents=None, *, iters=1000))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        parents: Option<Vec<i64>>,
        iters: usize,
    ) -> PyResult<Py<Self>> {
        // Build (or accept) the corpus; keep the surviving-document indices so the
        // parent tree stays aligned when empty documents are pruned.
        let (corpus, kept, expected_len): (corpus::Corpus, Vec<usize>, usize) =
            if let Ok(c) = data.extract::<Corpus>() {
                let inner = c.inner;
                let n = inner.num_docs();
                (inner, (0..n).collect(), n)
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                let orig = docs.len();
                let (cp, kept) = build_corpus_from_docs(
                    docs,
                    None,
                    None,
                    std::collections::HashSet::new(),
                    1,
                    1.0,
                    0,
                    0,
                )?;
                (cp, kept, orig)
            };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        // Remap parents to the surviving-document index space. A parent that was
        // pruned (or an out-of-range index) becomes a root (-1).
        let parents_remapped: Vec<i64> = match parents {
            None => Vec::new(),
            Some(p) => {
                if p.len() != expected_len {
                    return Err(PyValueError::new_err(format!(
                        "parents has {} entries but there are {} documents",
                        p.len(),
                        expected_len
                    )));
                }
                let mut old_to_new = vec![-1i64; expected_len];
                for (new_idx, &old) in kept.iter().enumerate() {
                    old_to_new[old] = new_idx as i64;
                }
                kept.iter()
                    .map(|&old| {
                        let par = p[old];
                        if par >= 0 && (par as usize) < expected_len {
                            old_to_new[par as usize]
                        } else {
                            -1
                        }
                    })
                    .collect()
            }
        };

        let params = CsatmParams {
            num_topics: slf.num_topics,
            alpha: slf.alpha,
            beta: slf.beta,
            lambda: slf.lambda_,
            weight: slf.weight_sequence(),
        };
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let model = crate::csatm::fit(&corpus, &parents_remapped, &params, iters, &mut rng);
            (model, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Constructor config as a JSON-serialisable dict (#400). `parents` is
    /// guidance data (supplied to `fit`), not reported here.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("lambda_", self.lambda_)?;
        d.set_item("weight_seq", &self.weight_seq)?;
        d.set_item("weight_c", self.weight_c)?;
        d.set_item("weight_d", self.weight_d)?;
        d.set_item("weight_g", self.weight_g)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    // --- Required analysis surface (B3) ---
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// The raw Gibbs doc-topic distribution, before the transitivity smoothing
    /// that produces `doc_topic`.
    #[getter]
    fn doc_topic_raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic_raw).to_pyarray_bound(py))
    }
    /// Per-document reply-tree popularity score `p_c` (a fit diagnostic).
    #[getter]
    fn popularity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.popularity.clone()).to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    // --- Conventional extras ---
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
            weights,
        )
    }
    /// Per-topic topic coherence, shape ``(num_topics,)``, aligned to topic index.
    /// ``coherence_type`` selects the measure (``"u_mass"`` default, or ``"c_v"`` /
    /// ``"c_uci"`` / ``"c_npmi"``); ``texts`` supplies the reference corpus for the
    /// windowed measures (defaults to the training corpus). Higher is more coherent.
    #[pyo3(signature = (n=TopN(10), *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: TopN,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n = n.0;
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
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

    /// Save the fitted model to `path`. Reload with `CSATM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_CSATM,
            &CsatmState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                beta: self.beta,
                lambda_: self.lambda_,
                weight_seq: self.weight_seq.clone(),
                weight_c: self.weight_c,
                weight_d: self.weight_d,
                weight_g: self.weight_g,
                seed: self.seed,
                fitted: self.fitted,
                corpus: self.corpus.clone(),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                doc_topic_raw: Some(m.doc_topic_raw.clone()),
                popularity: Some(m.popularity.clone()),
                doc_lengths: Some(m.doc_lengths.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: CsatmState = read_state(path, MODEL_TAG_CSATM)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(crate::csatm::CSATMModel {
                num_topics: s.num_topics,
                alpha: s.alpha,
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                doc_topic_raw: s.doc_topic_raw.unwrap_or_default(),
                popularity: s.popularity.unwrap_or_default(),
                doc_lengths: s.doc_lengths.unwrap_or_default(),
                fit_history: Vec::new(),
                converged: false,
            })
        } else {
            None
        };
        Ok(CSATM {
            num_topics: s.num_topics,
            alpha: s.alpha,
            beta: s.beta,
            lambda_: s.lambda_,
            weight_seq: s.weight_seq,
            weight_c: s.weight_c,
            weight_d: s.weight_d,
            weight_g: s.weight_g,
            seed: s.seed,
            fitted: s.fitted,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CSATM(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
