//! Python bindings for TopicsOverTime (ToT; Wang & McCallum, KDD 2006).
//!
//! `use super::*` pulls in the shared binding helpers (Corpus, build_corpus_from_docs,
//! save/load, array adapters, topic_words_helper, …).

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Timestamps are clipped this far from the {0,1} boundaries after normalization, so the
/// log-Beta factor `ln t`/`ln(1-t)` stays finite at the extreme dates.
const TIME_EPS: f64 = 1e-6;

/// TopicsOverTime: Wang & McCallum's continuous-time topic model (KDD 2006). LDA with a
/// per-topic Beta density over each document's timestamp, so a topic carries both a word
/// distribution and a temporal profile (when it rises and falls). The timestamp
/// influences the topic assignment jointly with the words, so topics are pulled to be
/// coherent in time as well as in vocabulary. Unlike DTM/DETM (discrete time slices with
/// drifting word distributions) the topic vocabulary is fixed and time is continuous;
/// this is descriptive continuous-time *prevalence*, not vocabulary drift. Reference: the
/// paper (no single maintained library); validated by planted continuous-time recovery
/// and a numerical Beta reference.
#[pyclass(module = "topica")]
pub struct TopicsOverTime {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    time_min: f64,
    time_max: f64,
    model: Option<crate::topics_over_time::TopicsOverTimeModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TopicsOverTimeState {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    time_min: f64,
    time_max: f64,
    corpus: Option<corpus::Corpus>,
    alpha_vec: Option<Vec<f64>>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    psi: Option<Vec<[f64; 2]>>,
    peak_norm: Option<Vec<f64>>,
    mean_norm: Option<Vec<f64>>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
}

impl TopicsOverTime {
    fn fitted_model(&self) -> PyResult<&crate::topics_over_time::TopicsOverTimeModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Map a normalized [0,1] value back to the original timestamp scale. NaN (an
    /// undefined peak) passes through unchanged.
    fn to_original(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        self.time_min + x * (self.time_max - self.time_min)
    }
}

#[pymethods]
impl TopicsOverTime {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted model. ``num_topics`` is K. ``alpha`` is the symmetric
    /// doc-topic Dirichlet concentration (default ``50/K``, the paper's value);
    /// ``beta`` is the symmetric topic-word Dirichlet (default 0.1, the paper's value).
    /// ``seed`` is the RNG seed.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=None, beta=0.1, seed=13))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: Option<f64>,
        beta: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        let alpha = alpha.unwrap_or(50.0 / num_topics as f64);
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(PyValueError::new_err("alpha must be finite and > 0"));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(PyValueError::new_err("beta must be finite and > 0"));
        }
        Ok(TopicsOverTime {
            num_topics,
            alpha,
            beta,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            time_min: 0.0,
            time_max: 1.0,
            model: None,
            corpus: None,
        })
    }

    /// Constructor config as a JSON-serialisable dict (#400). ``times`` is guidance
    /// data (supplied to ``fit``), not reported here.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a Corpus or a list of token lists) with per-document numeric
    /// timestamps. `times` is the canonical argument (any numeric scale — year, decade,
    /// ordinal date, unix time); `timestamps` is an accepted alias. Timestamps are
    /// min-max normalized to [0,1] internally (the original range is kept for reporting
    /// peaks/means in the input units). `iters` is the number of collapsed-Gibbs sweeps
    /// (default 1000, a topica default). A constant-timestamp corpus reduces to LDA (the
    /// per-topic Beta collapses to uniform).
    ///
    /// `progress` takes a ``(iteration, total, info)`` callback (see
    /// ``topica.progress``); pass ``True``/``False`` to force the bar on or off, and a
    /// ``KeyboardInterrupt`` raised from the callback aborts the fit.
    #[pyo3(signature = (data, times=None, *, timestamps=None, iters=1000, progress=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        times: Option<Vec<f64>>,
        timestamps: Option<Vec<f64>>,
        iters: usize,
        progress: Option<PyObject>,
    ) -> PyResult<Py<Self>> {
        let raw_times = match (times, timestamps) {
            (Some(t), None) | (None, Some(t)) => t,
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass either times= or timestamps=, not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "fit() requires per-document times= (numeric timestamps)",
                ))
            }
        };
        for &t in &raw_times {
            if !t.is_finite() {
                return Err(PyValueError::new_err(
                    "times must all be finite (no NaN/inf)",
                ));
            }
        }

        // Build (or accept) the corpus; keep surviving-document indices so timestamps
        // stay aligned when empty documents are pruned.
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
        if raw_times.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "times has {} entries but there are {} documents",
                raw_times.len(),
                expected_len
            )));
        }

        // Realign timestamps to the surviving documents.
        let doc_times: Vec<f64> = kept.iter().map(|&i| raw_times[i]).collect();

        // Min-max normalize to [0,1], clipped away from the boundaries. A degenerate
        // (constant) range maps everything to 0.5 → the Beta collapses to uniform → LDA.
        let tmin = doc_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let tmax = doc_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = tmax - tmin;
        let norm: Vec<f64> = if span <= 0.0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "TopicsOverTime: all timestamps are identical; the temporal factor is \
                 uniform and the model reduces to LDA",
                ),
            )?;
            vec![0.5; doc_times.len()]
        } else {
            doc_times
                .iter()
                .map(|&t| ((t - tmin) / span).clamp(TIME_EPS, 1.0 - TIME_EPS))
                .collect()
        };

        let num_types = corpus.num_types();
        let k = slf.num_topics;
        let alpha = vec![slf.alpha; k];
        let beta = slf.beta;
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);

        let progress = resolve_progress(py, progress, "TopicsOverTime")?;
        let (model, corpus) = py.allow_threads(move || {
            let mut on_progress = on_progress_bare(&progress);
            let m = crate::topics_over_time::fit(
                &corpus.docs,
                num_types,
                &norm,
                k,
                alpha,
                beta,
                iters,
                &mut on_progress,
                &mut rng,
            );
            (m, corpus)
        });
        reraise_if_interrupted(py)?;

        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.time_min = tmin;
        slf.time_max = tmax;
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Topic-word matrix φ (num_topics, vocab). Each row sums to 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }

    /// Document-topic matrix θ_d (num_docs, num_topics): the smoothed proportions of
    /// each document's token→topic assignments in the terminal Gibbs draw. Each row
    /// sums to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Per-topic Beta parameters (num_topics, 2): columns are (ψ1, ψ2) of the topic's
    /// Beta density over normalized [0,1] time. Three regimes: both ψ>1 is a single
    /// interior peak (see :attr:`topic_time_peak`); exactly one ψ<1 is a monotone
    /// rising/falling topic (peak at a boundary); both ψ<1 is a U-shaped topic with mass
    /// at both ends of the range and NO single peak (`topic_time_peak` is NaN). ψ1=ψ2=1
    /// is the explicit uniform fallback for a topic with no usable temporal signal (or a
    /// constant-timestamp corpus). Use :attr:`topic_time_peak` / :attr:`topic_time_mean`
    /// for human-readable values in the original timestamp units.
    #[getter]
    fn topic_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let psi = &self.fitted_model()?.psi;
        let rows: Vec<Vec<f64>> = psi.iter().map(|p| vec![p[0], p[1]]).collect();
        Ok(vecs_to_arr2(&rows).to_pyarray_bound(py))
    }

    /// Per-topic peak time (num_topics,) in the ORIGINAL timestamp units: the mode of
    /// the topic's Beta density where one exists (both ψ>1), the earliest/latest date
    /// for a monotone (rising/falling) topic, and NaN for a U-shaped or uniform topic
    /// that has no single peak. Prefer this over the mean for "when did this topic
    /// peak"; a skewed topic's mean and mode differ.
    #[getter]
    fn topic_time_peak<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let peaks: Vec<f64> = self
            .fitted_model()?
            .peak_norm
            .iter()
            .map(|&x| self.to_original(x))
            .collect();
        Ok(Array1::from(peaks).to_pyarray_bound(py))
    }

    /// Per-topic mean time (num_topics,) in the ORIGINAL timestamp units: ψ1/(ψ1+ψ2)
    /// mapped back to the input scale. Always defined (a uniform/no-signal topic falls
    /// back to ψ=(1,1) → the midpoint of the range). IMPORTANT: when
    /// :attr:`topic_time_peak` is NaN (a topic with no single peak — U-shaped or
    /// no-temporal-signal), this mean is NOT a peak; do not report it as "when the topic
    /// peaked." Inspect :attr:`topic_time` and the topic's document dates instead. For a
    /// skewed but single-peaked topic, prefer :attr:`topic_time_peak` for the peak.
    #[getter]
    fn topic_time_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let means: Vec<f64> = self
            .fitted_model()?
            .mean_norm
            .iter()
            .map(|&x| self.to_original(x))
            .collect();
        Ok(Array1::from(means).to_pyarray_bound(py))
    }

    /// The (min, max) of the input timestamps, the range peaks/means are reported on.
    #[getter]
    fn time_range(&self) -> PyResult<(f64, f64)> {
        self.fitted_model()?;
        Ok((self.time_min, self.time_max))
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
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
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }

    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn early_stopped(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    /// Top `n` words per topic (bare word strings). Pass ``weights=True`` for
    /// ``(word, φ)`` pairs.
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
    /// Scores each topic's top-``n`` words. ``coherence_type`` selects the measure
    /// (``"u_mass"`` default, or ``"c_v"`` / ``"c_uci"`` / ``"c_npmi"``); ``texts``
    /// supplies the reference corpus for the windowed measures (defaults to the
    /// training corpus). Higher is more coherent (``u_mass`` is <= 0, nearer 0 is
    /// better; ``c_v`` in [0, 1]). Compare topics within one fit, not across corpora.
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

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_TOPICS_OVER_TIME,
            &TopicsOverTimeState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                beta: self.beta,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                time_min: self.time_min,
                time_max: self.time_max,
                corpus: self.corpus.clone(),
                alpha_vec: Some(m.alpha.clone()),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                psi: Some(m.psi.clone()),
                peak_norm: Some(m.peak_norm.clone()),
                mean_norm: Some(m.mean_norm.clone()),
                fit_history: Some(m.fit_history.clone()),
                converged: Some(m.converged),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: TopicsOverTimeState = read_state(path, MODEL_TAG_TOPICS_OVER_TIME)?;
        let model = if s.fitted {
            Some(crate::topics_over_time::TopicsOverTimeModel {
                num_topics: s.num_topics,
                alpha: s.alpha_vec.unwrap_or_default(),
                beta: s.beta,
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                psi: s.psi.unwrap_or_default(),
                peak_norm: s.peak_norm.unwrap_or_default(),
                mean_norm: s.mean_norm.unwrap_or_default(),
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
            })
        } else {
            None
        };
        Ok(TopicsOverTime {
            num_topics: s.num_topics,
            alpha: s.alpha,
            beta: s.beta,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            time_min: s.time_min,
            time_max: s.time_max,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TopicsOverTime(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
