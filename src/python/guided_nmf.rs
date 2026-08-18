//! Python bindings for GuidedNMF (seed-word-guided semi-supervised NMF).
//!
//! Mirrors the NMF binding (`nmf_lsa.rs`) for the factorization surface and reuses
//! the shared seed-word matcher (`parse_seed_dict`, `seed_word_ids`, `SeedMatch`)
//! for the guidance dictionary, as SeededLDA does. `use super::*` pulls in the
//! shared binding helpers (Corpus, build_corpus_from_docs, save/load, array
//! adapters, run_with_threads, …).

use super::*;
use crate::guided_nmf::{fit_guided_nmf, GnmfInit, GuidedNMFModel};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// GuidedNMF: seed-word-guided semi-supervised NMF (Vendrow, Haddock, Rebrova &
/// Needell, ICASSP 2021). Factors the nonnegative document-term matrix
/// ``X (D x V) ~ A S`` (``A`` document-topic, ``S`` topic-word) while a supervision
/// term ``guidance * ||Y - B S||_F^2`` steers designated topics toward user-supplied
/// seed-word groups (``Y`` is the seed matrix, one row per group). It is the NMF
/// analogue of :class:`SeededLDA`; reach for it when NMF alone yields redundant or
/// off-theme topics and you can name a few words per theme you expect. The
/// operational reference is the ``ssnmf`` package (MIT) in supervised Frobenius
/// mode; ``topic_word`` is each row of ``S`` normalized to sum 1, ``doc_topic`` is
/// the scale-corrected ``A`` row-normalized.
#[pyclass(module = "topica")]
pub struct GuidedNMF {
    num_topics: usize,
    seed_names: Vec<String>,
    seed_words: Vec<Vec<String>>,
    guidance: f64,
    seed_weight: f64,
    init: String,
    weighting_tfidf: bool,
    convergence_tol: f64,
    seed_match: String,
    case_insensitive: bool,
    init_a: Option<Vec<Vec<f64>>>,
    init_s: Option<Vec<Vec<f64>>>,
    init_b: Option<Vec<Vec<f64>>>,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<GuidedNMFModel>,
    corpus: Option<corpus::Corpus>,
}

/// Serializable snapshot of a fitted GuidedNMF.
#[derive(serde::Serialize, serde::Deserialize)]
struct GuidedNmfState {
    num_topics: usize,
    seed_names: Vec<String>,
    seed_words: Vec<Vec<String>>,
    guidance: f64,
    seed_weight: f64,
    init: String,
    weighting_tfidf: bool,
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
    factor_a: Option<Vec<Vec<f64>>>,
    factor_s: Option<Vec<Vec<f64>>>,
    factor_b: Option<Vec<Vec<f64>>>,
    seed_topic_indices: Option<Vec<usize>>,
    reconstruction_error: Option<f64>,
    error_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
}

fn parse_gnmf_weighting(s: &str) -> PyResult<bool> {
    match s.to_ascii_lowercase().as_str() {
        "count" => Ok(false),
        "tfidf" | "tf-idf" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "weighting must be 'count' or 'tfidf' (got {other:?})"
        ))),
    }
}

/// Validate a caller-supplied init factor: nonnegative and finite. Shape is
/// checked at fit (it needs the corpus dimensions).
fn check_init_factor(name: &str, m: &[Vec<f64>]) -> PyResult<()> {
    for row in m {
        for &v in row {
            if !v.is_finite() || v < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "init={name:?} entries must be finite and nonnegative"
                )));
            }
        }
    }
    Ok(())
}

fn arr2_to_vecs(a: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let a = a.as_array();
    let (r, c) = (a.shape()[0], a.shape()[1]);
    (0..r)
        .map(|i| (0..c).map(|j| a[[i, j]]).collect())
        .collect()
}

impl GuidedNMF {
    fn fitted_model(&self) -> PyResult<&GuidedNMFModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

/// Warn when two or more seed groups steer the SAME learned topic (#686): their
/// document prevalence is then indistinguishable, so a user could report one
/// group's `doc_topic` share as another's. Silent when every group owns a distinct
/// topic. Collisions are sorted so the message is deterministic.
fn guided_nmf_collapse_warning(
    py: Python<'_>,
    seed_topic_indices: &[usize],
    seed_names: &[String],
) {
    let mut by_topic: std::collections::HashMap<usize, Vec<&str>> =
        std::collections::HashMap::new();
    for (i, &k) in seed_topic_indices.iter().enumerate() {
        let name = seed_names.get(i).map(String::as_str).unwrap_or("?");
        by_topic.entry(k).or_default().push(name);
    }
    let mut collisions: Vec<String> = by_topic
        .iter()
        .filter(|(_, names)| names.len() > 1)
        .map(|(k, names)| format!("topic {k}: {}", names.join(", ")))
        .collect();
    if collisions.is_empty() {
        return;
    }
    collisions.sort();
    if let Ok(warnings) = py.import_bound("warnings") {
        let _ = warnings.call_method1(
            "warn",
            (format!(
                "GuidedNMF: seed groups collapsed onto a shared topic ({}). Their \
                 document prevalence is indistinguishable — do not report one group's \
                 doc_topic share as another's (see seed_topic_map). Try a larger \
                 num_topics, a higher guidance, or more distinctive seed words.",
                collisions.join("; ")
            ),),
        );
    }
}

#[pymethods]
impl GuidedNMF {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted model. ``num_topics`` is K. ``seed_words`` is
    /// ``{group_name: [words]}`` — one guided topic per group (G groups, G <= K).
    /// ``guidance`` (alias ``lam``) is the supervision weight lambda. topica
    /// defaults to 3.0, lower than the reference's rarely-used 20: at 20 the guided
    /// topics are pinned so tightly to their seed words that they carry near-zero
    /// document prevalence (their `doc_topic` share collapses), while 3 keeps the
    /// same on-theme top words with interpretable prevalence. Raise it toward 20 to
    /// reproduce the reference / hold topics closer to the seeds; lower it for more
    /// data-driven topics. With count weighting, scale it down further.
    /// ``seed_weight`` is the value written into the seed matrix at each matched
    /// seed word. ``init`` is ``"random"`` (default, seeded Uniform[0,1], matching
    /// the reference), ``"nndsvd"`` (deterministic SVD init for A,S — a topica
    /// extension that ignores the seeds and may settle a different basin), or
    /// ``"none"`` (supply ``init_a``/``init_s``/``init_b`` directly). ``weighting``
    /// is ``"tfidf"`` (default, the reference regime) or ``"count"``.
    /// ``convergence_tol`` defaults to 0.0 (no early stop — the reference runs a
    /// fixed iteration budget); set it > 0 to stop on the relative objective
    /// decrease. ``seed_match``/``case_insensitive`` control seed-word matching
    /// exactly as in :class:`SeededLDA`. ``seed`` affects only ``init="random"``.
    #[new]
    #[pyo3(signature = (num_topics, seed_words, *, guidance=3.0, lam=None,
                        seed_weight=1.0, init="random", weighting="tfidf",
                        convergence_tol=0.0, seed_match="fixed", case_insensitive=false,
                        init_a=None, init_s=None, init_b=None, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        seed_words: &Bound<'_, PyDict>,
        guidance: f64,
        lam: Option<f64>,
        seed_weight: f64,
        init: &str,
        weighting: &str,
        convergence_tol: f64,
        seed_match: &str,
        case_insensitive: bool,
        init_a: Option<PyReadonlyArray2<f64>>,
        init_s: Option<PyReadonlyArray2<f64>>,
        init_b: Option<PyReadonlyArray2<f64>>,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("need at least 1 topic"));
        }
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
        let (names, words) = parse_seed_dict(seed_words)?;
        if names.len() > num_topics {
            return Err(PyValueError::new_err(format!(
                "num_topics ({num_topics}) must be >= the number of seed groups ({}); \
                 GuidedNMF cannot guide more groups than there are topics",
                names.len()
            )));
        }
        // Validate seed_match up front (reuse SeededLDA's parser).
        let _ = SeedMatch::parse(seed_match)?;
        let init_l = init.to_ascii_lowercase();
        if !matches!(init_l.as_str(), "random" | "nndsvd" | "none") {
            return Err(PyValueError::new_err(format!(
                "init must be 'random', 'nndsvd', or 'none' (got {init:?})"
            )));
        }
        let guidance = lam.unwrap_or(guidance);
        if !(guidance.is_finite() && guidance >= 0.0) {
            return Err(PyValueError::new_err(
                "guidance (lam) must be finite and >= 0",
            ));
        }
        if !(seed_weight.is_finite() && seed_weight > 0.0) {
            return Err(PyValueError::new_err("seed_weight must be finite and > 0"));
        }
        let (ia, is, ib) = (
            init_a.as_ref().map(arr2_to_vecs),
            init_s.as_ref().map(arr2_to_vecs),
            init_b.as_ref().map(arr2_to_vecs),
        );
        if init_l == "none" {
            match (&ia, &is, &ib) {
                (Some(a), Some(s), Some(b)) => {
                    check_init_factor("none:init_a", a)?;
                    check_init_factor("none:init_s", s)?;
                    check_init_factor("none:init_b", b)?;
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "init='none' requires init_a, init_s, and init_b (all three factors)",
                    ))
                }
            }
        }
        Ok(GuidedNMF {
            num_topics,
            seed_names: names,
            seed_words: words,
            guidance,
            seed_weight,
            init: init_l,
            weighting_tfidf: parse_gnmf_weighting(weighting)?,
            convergence_tol,
            seed_match: seed_match.to_string(),
            case_insensitive,
            init_a: ia,
            init_s: is,
            init_b: ib,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named to
    /// match ``__init__`` (issue #400). ``seed_words`` is guidance data, not a
    /// hyperparameter, so it is not reported (as in SeededLDA).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("guidance", self.guidance)?;
        d.set_item("seed_weight", self.seed_weight)?;
        d.set_item("init", &self.init)?;
        let weighting = if self.weighting_tfidf {
            "tfidf"
        } else {
            "count"
        };
        d.set_item("weighting", weighting)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed_match", &self.seed_match)?;
        d.set_item("case_insensitive", self.case_insensitive)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a Corpus or list of token lists). `iters` is the number of
    /// multiplicative-update iterations (default 50, the reference budget).
    /// `convergence_tol` overrides the constructor value for this run. `num_threads`
    /// caps the worker pool for the parallel matmuls; output is deterministic
    /// regardless of the worker count.
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
        // No blanket K <= V restriction: the supervised-Frobenius factorization
        // (and ssnmf) support overcomplete K > V. Only NNDSVD needs the rank guard
        // (checked in its init branch below).
        let g = slf.seed_names.len();

        // Build the seed matrix Y (G x V) via the shared matcher.
        let mode = SeedMatch::parse(&slf.seed_match)?;
        let seed_ids = seed_word_ids(
            &slf.seed_words,
            &corpus.id_to_word,
            g,
            mode,
            slf.case_insensitive,
        )?;
        let mut y = vec![vec![0.0f64; num_types]; g];
        for (gg, ids) in seed_ids.iter().enumerate() {
            if ids.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "seed group {:?} matched no vocabulary words; adjust the seed words, \
                     seed_match, or case_insensitive",
                    slf.seed_names[gg]
                )));
            }
            for &j in ids {
                y[gg][j] = slf.seed_weight;
            }
        }

        // Resolve the initialization.
        let init = match slf.init.as_str() {
            "nndsvd" => {
                let max_rank = corpus.num_docs().min(num_types);
                if slf.num_topics > max_rank {
                    return Err(PyValueError::new_err(format!(
                        "init=\"nndsvd\" requires num_topics <= min(num_documents, num_words) \
                         = {max_rank} (got {}). Use init=\"random\".",
                        slf.num_topics
                    )));
                }
                GnmfInit::Nndsvd
            }
            "none" => {
                // Init factors may be absent if this model was loaded from disk
                // (save/load restores the fitted factors, not the one-shot init):
                // fail clearly instead of panicking on unwrap.
                let missing = || {
                    PyValueError::new_err(
                        "init='none' requires init_a, init_s, init_b, but none are set \
                         (e.g. this model was loaded from disk); re-supply them to re-fit, \
                         or construct a fresh GuidedNMF with the init factors",
                    )
                };
                let a = slf.init_a.clone().ok_or_else(missing)?;
                let s = slf.init_s.clone().ok_or_else(missing)?;
                let b = slf.init_b.clone().ok_or_else(missing)?;
                let (d, k) = (corpus.num_docs(), slf.num_topics);
                check_init_shape("init_a", &a, d, k)?;
                check_init_shape("init_s", &s, k, num_types)?;
                check_init_shape("init_b", &b, g, k)?;
                GnmfInit::Explicit { a, s, b }
            }
            _ => GnmfInit::Random,
        };

        let (k, lam, tfidf, seed) = (slf.num_topics, slf.guidance, slf.weighting_tfidf, slf.seed);
        let it = iters.unwrap_or(50);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = run_with_threads(num_threads, || {
                fit_guided_nmf(
                    &corpus.docs,
                    num_types,
                    &y,
                    k,
                    lam,
                    tfidf,
                    init,
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
        guided_nmf_collapse_warning(
            py,
            &slf.model.as_ref().unwrap().seed_topic_indices,
            &slf.seed_names,
        );
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// Topic-word matrix (num_topics, vocab): each row of S normalized to sum 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Document-topic matrix (num_docs, num_topics): scale-corrected A row-normalized.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Raw factor A (num_docs, num_topics), before scale correction / normalization.
    #[getter]
    fn factor_a<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.a).to_pyarray_bound(py))
    }
    /// Raw factor S (num_topics, vocab), before row normalization.
    #[getter]
    fn factor_s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.s).to_pyarray_bound(py))
    }
    /// Raw mixing factor B (num_groups, num_topics): each seed group as a
    /// nonnegative combination of the learned topics.
    #[getter]
    fn factor_b<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.b).to_pyarray_bound(py))
    }
    /// For each seed group (in constructor order), the learned topic it most
    /// steers: ``argmax_k B_gk ||S_k||_1`` (scale-invariant).
    #[getter]
    fn seed_topic_indices(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.seed_topic_indices.clone())
    }
    /// The seed-group names, in the order that indexes ``seed_topic_indices``.
    #[getter]
    fn seed_group_names(&self) -> Vec<String> {
        self.seed_names.clone()
    }

    /// Convenience map ``{seed_group_name: learned_topic_index}`` — the same pairing
    /// as ``dict(zip(seed_group_names, seed_topic_indices))``. When two group names
    /// point at the *same* index they collapsed onto a shared topic (:meth:`fit`
    /// warns), and their document prevalence cannot be told apart.
    #[getter]
    fn seed_topic_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let idx = &self.fitted_model()?.seed_topic_indices;
        let d = PyDict::new_bound(py);
        for (name, &k) in self.seed_names.iter().zip(idx.iter()) {
            d.set_item(name, k)?;
        }
        Ok(d)
    }
    /// The guidance weight lambda.
    #[getter]
    fn guidance(&self) -> f64 {
        self.guidance
    }
    #[getter]
    fn reconstruction_error(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.reconstruction_error)
    }
    /// Per-iteration value of the FULL objective
    /// ``||X - A S||_F^2 + guidance * ||Y - B S||_F^2`` (reconstruction plus the
    /// guidance term), initial value (before any update) first.
    #[getter]
    fn error_history(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_model()?.error_history.clone())
    }
    /// True only if an early stop fired (relative objective decrease <
    /// ``convergence_tol``). With the default ``convergence_tol=0.0`` there is no
    /// early stop, so a completed fit reports ``False`` — it means "ran the full
    /// iters budget", not a failure.
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
    /// Uniform convergence trace: `(iter, objective)` pairs (iter 1 = initial).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .error_history
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
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    /// Top ``n`` words per topic (bare word strings). With ``topic=None`` returns
    /// one list per topic (a list of lists); with ``topic=k`` returns the single
    /// list for topic k. Pass ``weights=True`` for ``(word, weight)`` pairs.
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
            MODEL_TAG_GUIDED_NMF,
            &GuidedNmfState {
                num_topics: self.num_topics,
                seed_names: self.seed_names.clone(),
                seed_words: self.seed_words.clone(),
                guidance: self.guidance,
                seed_weight: self.seed_weight,
                init: self.init.clone(),
                weighting_tfidf: self.weighting_tfidf,
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
                factor_a: Some(m.a.clone()),
                factor_s: Some(m.s.clone()),
                factor_b: Some(m.b.clone()),
                seed_topic_indices: Some(m.seed_topic_indices.clone()),
                reconstruction_error: Some(m.reconstruction_error),
                error_history: Some(m.error_history.clone()),
                converged: Some(m.converged),
                iters_run: Some(m.iters_run),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: GuidedNmfState = read_state(path, MODEL_TAG_GUIDED_NMF)?;
        let model = if s.fitted {
            Some(GuidedNMFModel {
                num_topics: s.num_topics,
                num_types: s.num_types.unwrap_or(0),
                num_groups: s.num_groups.unwrap_or(s.seed_names.len()),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                a: s.factor_a.unwrap_or_default(),
                s: s.factor_s.unwrap_or_default(),
                b: s.factor_b.unwrap_or_default(),
                seed_topic_indices: s.seed_topic_indices.unwrap_or_default(),
                reconstruction_error: s.reconstruction_error.unwrap_or(0.0),
                error_history: s.error_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
            })
        } else {
            None
        };
        Ok(GuidedNMF {
            num_topics: s.num_topics,
            seed_names: s.seed_names,
            seed_words: s.seed_words,
            guidance: s.guidance,
            seed_weight: s.seed_weight,
            init: s.init,
            weighting_tfidf: s.weighting_tfidf,
            convergence_tol: s.convergence_tol,
            seed_match: s.seed_match,
            case_insensitive: s.case_insensitive,
            init_a: None,
            init_s: None,
            init_b: None,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GuidedNMF(num_topics={}, seed_groups={}, guidance={}, fitted={})",
            self.num_topics,
            self.seed_names.len(),
            self.guidance,
            self.fitted
        )
    }
}

/// Validate a caller-supplied init factor's shape (rows x cols).
fn check_init_shape(name: &str, m: &[Vec<f64>], rows: usize, cols: usize) -> PyResult<()> {
    if m.len() != rows || m.iter().any(|r| r.len() != cols) {
        return Err(PyValueError::new_err(format!(
            "init={name:?} must have shape ({rows}, {cols})"
        )));
    }
    Ok(())
}
