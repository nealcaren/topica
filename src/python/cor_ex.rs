//! Python bindings for CorEx (information-theoretic Correlation Explanation).
//!
//! `use super::*` pulls in the shared binding helpers (Corpus, build_corpus_from_docs,
//! save/load, array adapters, run_with_threads, the SeededLDA anchor matcher, …).

use super::*;
use crate::cor_ex::{corex_transform, fit_corex, CorExModel};
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// CorEx: Correlation Explanation topic model (Gallagher et al., TACL 2017). An
/// information-theoretic, non-generative model that learns binary latent topics
/// maximizing the total correlation they explain about the words, with optional
/// anchor words for semi-supervision. Unlike LDA/NMF, ``doc_topic`` is a matrix of
/// independent per-topic probabilities (rows do NOT sum to 1) and ``topic_word`` is
/// ``alpha * mis`` (mutual information weighted by membership), not a distribution.
/// Reference: the ``corextopic`` package (Apache-2.0).
#[pyclass(module = "topica")]
pub struct CorEx {
    num_topics: usize,
    anchor_names: Vec<String>,
    anchor_words: Vec<Vec<String>>,
    anchor_strength: f64,
    count: String,
    convergence_tol: f64,
    seed_match: String,
    case_insensitive: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<CorExModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CorExState {
    num_topics: usize,
    anchor_names: Vec<String>,
    anchor_words: Vec<Vec<String>>,
    anchor_strength: f64,
    count: String,
    convergence_tol: f64,
    seed_match: String,
    case_insensitive: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_types: Option<usize>,
    num_groups: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    mis: Option<Vec<Vec<f64>>>,
    alpha: Option<Vec<Vec<f64>>>,
    labels: Option<Vec<Vec<u8>>>,
    clusters: Option<Vec<usize>>,
    sign: Option<Vec<Vec<i8>>>,
    tcs: Option<Vec<f64>>,
    total_correlation: Option<f64>,
    tc_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
    log_p_y: Option<Vec<f64>>,
    theta: Option<[Vec<Vec<f64>>; 4]>,
    lp0: Option<Vec<f64>>,
    px_frac: Option<Vec<f64>>,
}

impl CorEx {
    fn fitted_model(&self) -> PyResult<&CorExModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl CorEx {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted model. ``num_topics`` is the number of binary latent
    /// topics. ``anchor_words`` is ``{topic_name: [words]}`` — one anchored topic
    /// per group (in insertion order; #groups <= num_topics). ``anchor_strength``
    /// is the alpha value written for anchored (word, topic) pairs (the reference
    /// default 1.0; higher pins harder). ``count`` is ``"binarize"`` (default;
    /// counts>1 become presence). ``convergence_tol`` is the total-correlation
    /// change that signals convergence. ``seed_match``/``case_insensitive`` control
    /// anchor-word matching exactly as in :class:`SeededLDA`.
    #[new]
    #[pyo3(signature = (num_topics=2, *, anchor_words=None, anchor_strength=1.0,
                        count="binarize", convergence_tol=1e-5, seed_match="fixed",
                        case_insensitive=false, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        anchor_words: Option<&Bound<'_, PyDict>>,
        anchor_strength: f64,
        count: &str,
        convergence_tol: f64,
        seed_match: &str,
        case_insensitive: bool,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("need at least 1 topic"));
        }
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
        let count_l = count.to_ascii_lowercase();
        match count_l.as_str() {
            "binarize" => {}
            "fraction" => {
                return Err(PyValueError::new_err(
                    "count='fraction' is not yet implemented; use count='binarize'",
                ))
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "count must be 'binarize' (got {other:?})"
                )))
            }
        }
        let _ = SeedMatch::parse(seed_match)?;
        let (names, words) = match anchor_words {
            Some(d) => parse_seed_dict(d)?,
            None => (Vec::new(), Vec::new()),
        };
        if names.len() > num_topics {
            return Err(PyValueError::new_err(format!(
                "num_topics ({num_topics}) must be >= the number of anchor groups ({})",
                names.len()
            )));
        }
        if !(anchor_strength.is_finite() && anchor_strength > 0.0) {
            return Err(PyValueError::new_err(
                "anchor_strength must be finite and > 0",
            ));
        }
        Ok(CorEx {
            num_topics,
            anchor_names: names,
            anchor_words: words,
            anchor_strength,
            count: count_l,
            convergence_tol,
            seed_match: seed_match.to_string(),
            case_insensitive,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Constructor config as a JSON-serialisable dict (issue #400). ``anchor_words``
    /// is guidance data, not reported.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("anchor_strength", self.anchor_strength)?;
        d.set_item("count", &self.count)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed_match", &self.seed_match)?;
        d.set_item("case_insensitive", self.case_insensitive)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a Corpus or list of token lists). `iters` is the maximum
    /// number of update iterations (default 200). `num_threads` caps the worker pool
    /// for the parallel sparse products; output is deterministic regardless.
    #[pyo3(signature = (data, *, iters=None, convergence_tol=None, num_threads=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
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
        if num_types < slf.num_topics {
            return Err(PyValueError::new_err(
                "vocabulary must have at least num_topics words",
            ));
        }

        // Build anchor word-id lists per topic (length num_topics; groups first).
        let mode = SeedMatch::parse(&slf.seed_match)?;
        let mut anchors: Vec<Vec<usize>> = vec![Vec::new(); slf.num_topics];
        if !slf.anchor_names.is_empty() {
            let ids = seed_word_ids(
                &slf.anchor_words,
                &corpus.id_to_word,
                slf.anchor_names.len(),
                mode,
                slf.case_insensitive,
            )?;
            for (g, group) in ids.iter().enumerate() {
                if group.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "anchor group {:?} matched no vocabulary words",
                        slf.anchor_names[g]
                    )));
                }
                anchors[g] = group.clone();
            }
        }

        let (k, strength, seed) = (slf.num_topics, slf.anchor_strength, slf.seed);
        let it = iters.unwrap_or(200);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = run_with_threads(num_threads, || {
                fit_corex(
                    &corpus.docs,
                    num_types,
                    k,
                    &anchors,
                    strength,
                    it,
                    tol,
                    &mut rng,
                )
            });
            (m, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Label held-out documents: p(y_j=1 | doc) for each topic (n_docs, num_topics).
    /// Input tokens are always remapped onto the TRAINING vocabulary (a held-out
    /// Corpus with its own id_to_word is translated through its word strings, not its
    /// raw ids); out-of-vocabulary tokens are dropped.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let train = self.corpus.as_ref().unwrap();
        let word_to_id: std::collections::HashMap<&str, u32> = train
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        // Extract the input as token lists (word strings), whatever the form, so we
        // can map onto the training vocabulary by word — never by raw column id.
        let token_docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            let inner = c.inner;
            inner
                .docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&id| inner.id_to_word[id as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("transform() expects a Corpus or a list of token lists")
            })?
        };
        let mapped: Vec<Vec<u32>> = token_docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| word_to_id.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        let pygx = corex_transform(m, &mapped);
        Ok(vecs_to_arr2(&pygx).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix (num_topics, vocab) = alpha * mis (membership-weighted MI).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics) = p(y_j=1|doc); independent
    /// per-topic probabilities (rows do NOT sum to 1).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Raw mutual-information matrix (num_topics, vocab), in bits.
    #[getter]
    fn mis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.mis).to_pyarray_bound(py))
    }
    /// Word->topic soft membership alpha (num_topics, vocab).
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.alpha).to_pyarray_bound(py))
    }
    /// Binary topic labels per document (num_docs, num_topics), p(y|x) > 0.5.
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let m = self.fitted_model()?;
        let rows: Vec<Vec<f64>> = m
            .labels
            .iter()
            .map(|r| r.iter().map(|&x| x as f64).collect())
            .collect();
        let arr = vecs_to_arr2(&rows).mapv(|x| x as i64);
        Ok(arr.to_pyarray_bound(py))
    }
    /// Word cluster assignment = argmax topic of alpha (length vocab).
    #[getter]
    fn clusters(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.clusters.clone())
    }
    /// Per-topic total correlation (nats).
    #[getter]
    fn topic_tc(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.tcs.clone())
    }
    /// Total correlation explained (sum of per-topic TC), nats.
    #[getter]
    fn total_correlation(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.total_correlation)
    }
    /// Sum-TC per iteration (nats); the convergence trace.
    #[getter]
    fn tc_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.tc_history.clone())
    }
    /// Uniform convergence trace: (iter, total_tc) pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .tc_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
            .collect())
    }
    /// True only if the total-correlation early stop fired before the iter budget.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
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
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    /// Top (word, alpha*mis) pairs per topic, ranked by membership-weighted MI.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
        )
    }
    /// Per-topic topic coherence, shape ``(num_topics,)``, aligned to topic index.
    /// Scores each topic's top-``n`` words. ``coherence_type`` selects the measure
    /// (``"u_mass"`` default, or ``"c_v"`` / ``"c_uci"`` / ``"c_npmi"``); ``texts``
    /// supplies the reference corpus for the windowed measures (defaults to the
    /// training corpus). Higher is more coherent (``u_mass`` is <= 0, nearer 0 is
    /// better; ``c_v`` in [0, 1]). Compare topics within one fit, not across corpora.
    #[pyo3(signature = (n=10, *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
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

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_COREX,
            &CorExState {
                num_topics: self.num_topics,
                anchor_names: self.anchor_names.clone(),
                anchor_words: self.anchor_words.clone(),
                anchor_strength: self.anchor_strength,
                count: self.count.clone(),
                convergence_tol: self.convergence_tol,
                seed_match: self.seed_match.clone(),
                case_insensitive: self.case_insensitive,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                num_types: Some(m.num_types),
                num_groups: Some(m.num_groups),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                mis: Some(m.mis.clone()),
                alpha: Some(m.alpha.clone()),
                labels: Some(m.labels.clone()),
                clusters: Some(m.clusters.clone()),
                sign: Some(m.sign.clone()),
                tcs: Some(m.tcs.clone()),
                total_correlation: Some(m.total_correlation),
                tc_history: Some(m.tc_history.clone()),
                converged: Some(m.converged),
                iters_run: Some(m.iters_run),
                log_p_y: Some(m.log_p_y.clone()),
                theta: Some(m.theta.clone()),
                lp0: Some(m.lp0.clone()),
                px_frac: Some(m.px_frac.clone()),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: CorExState = read_state(path, MODEL_TAG_COREX)?;
        let model = if s.fitted {
            Some(CorExModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                num_groups: s.num_groups.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                mis: s.mis.unwrap_or_default(),
                alpha: s.alpha.unwrap_or_default(),
                labels: s.labels.unwrap_or_default(),
                clusters: s.clusters.unwrap_or_default(),
                sign: s.sign.unwrap_or_default(),
                tcs: s.tcs.unwrap_or_default(),
                total_correlation: s.total_correlation.unwrap_or(0.0),
                tc_history: s.tc_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
                log_p_y: s.log_p_y.unwrap_or_default(),
                theta: s.theta.unwrap_or_default(),
                lp0: s.lp0.unwrap_or_default(),
                px_frac: s.px_frac.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(CorEx {
            num_topics: s.num_topics,
            anchor_names: s.anchor_names,
            anchor_words: s.anchor_words,
            anchor_strength: s.anchor_strength,
            count: s.count,
            convergence_tol: s.convergence_tol,
            seed_match: s.seed_match,
            case_insensitive: s.case_insensitive,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CorEx(num_topics={}, anchors={}, fitted={})",
            self.num_topics,
            self.anchor_names.len(),
            self.fitted
        )
    }
}
