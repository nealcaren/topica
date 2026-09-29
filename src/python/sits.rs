//! Python bindings for SITS (parametric Speaker Identity for Topic Segmentation;
//! Nguyen, Boyd-Graber & Resnik 2012; Rossiter 2022).
//!
//! `use super::*` pulls in the shared binding helpers (Corpus, build_corpus_from_docs,
//! save/load, array adapters, topic_words_helper, …).

use super::*;
use crate::sits::{Compat, Sampler, SitsConfig, SitsData, SitsModel};
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Share of non-first turns that may be ineligible before `fit` warns.
const INELIGIBLE_WARN_SHARE: f64 = 0.25;

/// SITS: parametric Speaker Identity for Topic Segmentation (Nguyen, Boyd-Graber &
/// Resnik, ACL 2012; Nguyen et al., Machine Learning 2014), the agenda-setting
/// measure of Rossiter (2022, AJPS). Conversations are sequences of speaking turns;
/// each turn either continues the current topic segment or shifts to a new one, and
/// the probability that speaker ``m`` shifts the topic, ``π_m``, is the speaker's
/// agenda-setting propensity. Collapsed Gibbs, ported from the reference Java
/// sampler (``vietansegan/sits`` and Rossiter's fork ``erossiter/sits``).
#[pyclass(module = "topica")]
pub struct SITS {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    min_shift_tokens: usize,
    init_shift_rate: f64,
    compat: Option<String>,
    sampler: String,
    seed: u64,
    fitted: bool,
    speaker_names: Vec<String>,
    speaker_is_int: bool,
    turn_speaker: Vec<u32>,
    conv_start: Vec<bool>,
    topic_names: Vec<String>,
    burn_in: usize,
    iters: usize,
    geweke: Option<f64>,
    model: Option<SitsModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SitsState {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    min_shift_tokens: usize,
    init_shift_rate: f64,
    compat: Option<String>,
    sampler: String,
    seed: u64,
    fitted: bool,
    speaker_names: Vec<String>,
    speaker_is_int: bool,
    turn_speaker: Vec<u32>,
    conv_start: Vec<bool>,
    topic_names: Vec<String>,
    burn_in: usize,
    iters: usize,
    geweke: Option<f64>,
    model: Option<SitsModel>,
    corpus: Option<corpus::Corpus>,
}

/// `init_every` (the fork's `I`) implied by an initial shift rate: 0 → 1 (never),
/// else 1/rate, which must be an integer >= 2.
fn compat_init_every(rate: f64) -> PyResult<u32> {
    if rate == 0.0 {
        return Ok(1);
    }
    let i = (1.0 / rate).round();
    if i < 2.0 || ((1.0 / rate) - i).abs() > 1e-6 {
        return Err(PyValueError::new_err(format!(
            "compat='rossiter2022' draws initial shifts as nextInt(I) == 1, so \
             init_shift_rate must be 0 or 1/I for an integer I >= 2 (her runs used \
             I = 3 to 7, e.g. init_shift_rate=1/3); got {rate}"
        )));
    }
    Ok(i as u32)
}

/// Per-turn labels: all integers (Python or numpy ints) or all strings. Returns the
/// labels as strings plus whether they were integers. Mixed types, bools, `None`
/// and floats are rejected, so `1` and `"1"` can never merge into one speaker.
fn labels_checked(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<String>, bool)> {
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyValueError::new_err(format!(
            "{what} must be a sequence of labels, one per turn, not a single string"
        )));
    }
    let items: Vec<Bound<'_, PyAny>> = obj
        .iter()
        .map_err(|_| {
            PyValueError::new_err(format!("{what} must be a sequence (one entry per turn)"))
        })?
        .collect::<PyResult<_>>()?;
    let mut out = Vec::with_capacity(items.len());
    let (mut n_int, mut n_str) = (0usize, 0usize);
    for (t, x) in items.iter().enumerate() {
        if x.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyValueError::new_err(format!(
                "{what}[{t}] is a bool; use integer or string labels"
            )));
        }
        if x.is_instance_of::<pyo3::types::PyString>() {
            out.push(x.extract::<String>()?);
            n_str += 1;
        } else if x.hasattr("__index__")? {
            let v: i64 = x.call_method0("__index__")?.extract()?;
            out.push(v.to_string());
            n_int += 1;
        } else {
            return Err(PyValueError::new_err(format!(
                "{what}[{t}] = {} is neither an integer nor a string label",
                x.repr()?
            )));
        }
    }
    if n_int > 0 && n_str > 0 {
        return Err(PyValueError::new_err(format!(
            "{what} mixes integer and string labels ({n_int} integers, {n_str} strings); \
             use one type so that 1 and \"1\" cannot be confused"
        )));
    }
    Ok((out, n_int > 0))
}

/// Quantile of a sorted slice by linear interpolation.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

impl SITS {
    fn fitted_model(&self) -> PyResult<&SitsModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Per-draw agenda-setting scores (S×M): (γ + Σ l)/(2γ + n_m).
    fn propensity_draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted_model()?;
        let g = self.gamma;
        Ok(m.speaker_shift_draws
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&m.speaker_turns)
                    .map(|(&s, &n)| (g + s as f64) / (2.0 * g + n as f64))
                    .collect()
            })
            .collect())
    }

    /// Per-draw eligible-turn shift rates (S×M): Σ_{eligible} l / n_eligible (NaN
    /// for a speaker with no eligible turns).
    fn eligible_rate_draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let m = self.fitted_model()?;
        Ok(m.speaker_eligible_shift_draws
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&m.speaker_eligible)
                    .map(|(&s, &n)| {
                        if n == 0 {
                            f64::NAN
                        } else {
                            s as f64 / n as f64
                        }
                    })
                    .collect()
            })
            .collect())
    }

    fn interval(draws: &[Vec<f64>], num_cols: usize, level: f64) -> PyResult<Array2<f64>> {
        if !(level > 0.0 && level < 1.0) {
            return Err(PyValueError::new_err("level must be in (0, 1)"));
        }
        let mut out = Array2::<f64>::from_elem((num_cols, 2), f64::NAN);
        for j in 0..num_cols {
            let mut col: Vec<f64> = draws
                .iter()
                .map(|r| r[j])
                .filter(|x| x.is_finite())
                .collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out[[j, 0]] = quantile(&col, (1.0 - level) / 2.0);
            out[[j, 1]] = quantile(&col, 1.0 - (1.0 - level) / 2.0);
        }
        Ok(out)
    }
}

#[pymethods]
impl SITS {
    /// Create an unfitted model.
    ///
    /// ``num_topics`` is K. ``alpha`` is the symmetric segment-topic Dirichlet
    /// (default ``1/K``, Rossiter's setting); ``beta`` the topic-word Dirichlet
    /// (default 0.1); ``gamma`` the symmetric Beta prior on each speaker's shift
    /// probability (default 1.0, Rossiter's setting; the Java command line defaults
    /// to 0.25). ``min_shift_tokens`` (default 5, Rossiter's threshold): a turn with
    /// fewer tokens is never sampled as a shift and always continues the current
    /// segment. Tokens are counted on what you pass to ``fit``, after your own
    /// stopword removal and pruning (Rossiter counted after hers), so the share of
    /// ineligible turns depends on preprocessing. ``min_shift_tokens=0`` makes every
    /// non-first turn eligible, empty turns included. ``init_shift_rate`` (default
    /// 0.1) is the probability an eligible turn starts the chain as a shift.
    ///
    /// ``compat="rossiter2022"`` reproduces the behaviour of Rossiter's fork,
    /// including a bookkeeping defect: short turns drawn as initial shifts stay
    /// segment boundaries and stay counted as shifts for the whole chain, which
    /// raises every speaker's shift rate. Use it only to replicate published
    /// results. It requires an explicit ``init_shift_rate = 1/I`` for the run being
    /// replicated (or 0).
    ///
    /// ``sampler`` chooses how shift indicators are resampled. ``"single"``
    /// (default) is the reference sampler: one turn at a time, π integrated out.
    /// ``"block"`` draws each speaker's π from its Beta posterior and then each
    /// conversation's whole segmentation at once from its exact conditional given
    /// the topic assignments. Both target the same posterior; the block sampler
    /// moves whole segment boundaries in one step and so mixes much faster.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=None, beta=0.1, gamma=1.0, min_shift_tokens=5,
                        init_shift_rate=None, compat=None, sampler="single".to_string(), seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: Option<f64>,
        beta: f64,
        gamma: f64,
        min_shift_tokens: usize,
        init_shift_rate: Option<f64>,
        compat: Option<String>,
        sampler: String,
        seed: u64,
    ) -> PyResult<Self> {
        match (sampler.as_str(), compat.as_deref()) {
            ("single", _) | ("block", None) => {}
            ("block", Some(_)) => {
                return Err(PyValueError::new_err(
                    "compat='rossiter2022' replicates the reference sampler; use \
                     sampler='single' with it",
                ))
            }
            (other, _) => {
                return Err(PyValueError::new_err(format!(
                    "sampler must be 'single' or 'block', got {other:?}"
                )))
            }
        }
        let init_shift_rate =
            match (init_shift_rate, compat.as_deref()) {
                (Some(r), _) => r,
                (None, None) => 0.1,
                (None, Some(_)) => return Err(PyValueError::new_err(
                    "compat='rossiter2022' needs an explicit init_shift_rate = 1/I for the run \
                     being replicated (Rossiter used I = 3 to 7, e.g. init_shift_rate=1/3); \
                     the default 0.1 would replicate an I = 10 run she never made",
                )),
            };
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        let alpha = alpha.unwrap_or(1.0 / num_topics as f64);
        for (name, x) in [("alpha", alpha), ("beta", beta), ("gamma", gamma)] {
            if !(x.is_finite() && x > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and > 0"
                )));
            }
        }
        if !(0.0..=1.0).contains(&init_shift_rate) {
            return Err(PyValueError::new_err("init_shift_rate must be in [0, 1]"));
        }
        match compat.as_deref() {
            None => {}
            Some("rossiter2022") => {
                compat_init_every(init_shift_rate)?;
            }
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "compat must be None or 'rossiter2022', got {other:?}"
                )))
            }
        }
        Ok(SITS {
            num_topics,
            alpha,
            beta,
            gamma,
            min_shift_tokens,
            init_shift_rate,
            compat,
            sampler,
            seed,
            fitted: false,
            speaker_names: Vec::new(),
            speaker_is_int: false,
            turn_speaker: Vec::new(),
            conv_start: Vec::new(),
            topic_names: Vec::new(),
            burn_in: 0,
            iters: 0,
            geweke: None,
            model: None,
            corpus: None,
        })
    }

    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Constructor config as a JSON-serialisable dict (#400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("min_shift_tokens", self.min_shift_tokens)?;
        d.set_item("init_shift_rate", self.init_shift_rate)?;
        d.set_item("compat", self.compat.clone())?;
        d.set_item("sampler", self.sampler.clone())?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on ``data`` (a Corpus or a list of token lists, one per speaking turn, in
    /// conversation order). ``speakers`` gives each turn's speaker and
    /// ``conversations`` each turn's conversation id; both are per-turn sequences of
    /// labels (strings or ints). A conversation's turns must be contiguous and in
    /// speaking order: an id that reappears after another conversation raises.
    /// ``authors=`` is accepted as an alias of ``speakers`` (the reference's term).
    /// Every turn is kept, including empty ones: they count toward their speaker.
    ///
    /// ``iters`` is the number of Gibbs sweeps (default 50,000; the chain mixes
    /// slowly, and Rossiter ran 100,000 to 500,000). ``burn_in`` sweeps (default
    /// ``iters // 2``) are discarded; every later sweep is a draw (no thinning), as
    /// in Rossiter's ``readSits``. ``sample_interval`` thins only the stored
    /// per-speaker draws used for intervals (default: keep at most 2,000); posterior
    /// means always use every draw.
    ///
    /// Warns when more than a quarter of the non-first turns are too short to be
    /// shifts, and when the eligible-shift trace fails a Geweke check (see
    /// :attr:`geweke_z`).
    #[pyo3(signature = (data, speakers=None, *, conversations=None, iters=50_000, burn_in=None,
                        sample_interval=None, progress=None, authors=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        speakers: Option<&Bound<'_, PyAny>>,
        conversations: Option<&Bound<'_, PyAny>>,
        iters: usize,
        burn_in: Option<usize>,
        sample_interval: Option<usize>,
        progress: Option<PyObject>,
        authors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let speakers = match (speakers, authors) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass speakers= or its alias authors=, not both",
                ))
            }
            (Some(s), None) | (None, Some(s)) => s,
            (None, None) => {
                return Err(PyValueError::new_err(
                    "SITS.fit needs speakers (one label per turn)",
                ))
            }
        };
        let conversations = conversations.ok_or_else(|| {
            PyValueError::new_err(
                "SITS.fit needs conversations= (one conversation id per turn, turns in order)",
            )
        })?;
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be >= 1"));
        }
        let burn_in = burn_in.unwrap_or(iters / 2);
        if burn_in >= iters {
            return Err(PyValueError::new_err(format!(
                "burn_in ({burn_in}) must be smaller than iters ({iters})"
            )));
        }

        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            if c.kept_indices.iter().enumerate().any(|(i, &k)| i != k) {
                return Err(PyValueError::new_err(
                    "this Corpus dropped turns during pruning (see corpus.kept_indices); SITS \
                     needs every turn in order, because a dropped turn changes its speaker's \
                     counts and the segment structure. Do not realign speakers to \
                     kept_indices: rebuild the Corpus so empty turns are kept, or pass the \
                     token lists (filter tokens to corpus.vocabulary yourself)",
                ));
            }
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            let n = docs.len();
            let (cp, kept) =
                build_corpus_from_docs(docs, None, None, HashSet::new(), 1, 1.0, 0, 0)?;
            if kept.len() != n {
                return Err(PyValueError::new_err(
                    "internal error: the corpus builder dropped turns; SITS needs every turn",
                ));
            }
            cp
        };
        let nt = corpus.num_docs();
        if nt == 0 {
            return Err(PyValueError::new_err("corpus contains no turns"));
        }
        let (spk_labels, speaker_is_int) = labels_checked(speakers, "speakers")?;
        let (conv_labels, _) = labels_checked(conversations, "conversations")?;
        for (name, len) in [
            ("speakers", spk_labels.len()),
            ("conversations", conv_labels.len()),
        ] {
            if len != nt {
                return Err(PyValueError::new_err(format!(
                    "{name} has {len} entries but there are {nt} turns; pass one label per \
                     turn, keeping empty turns (if a Corpus pruned turns, rebuild it rather \
                     than realigning the labels)"
                )));
            }
        }
        // Conversation starts; ids must be contiguous.
        let mut conv_start = vec![false; nt];
        let mut seen: HashSet<&str> = HashSet::new();
        for t in 0..nt {
            if t == 0 || conv_labels[t] != conv_labels[t - 1] {
                if !seen.insert(conv_labels[t].as_str()) {
                    return Err(PyValueError::new_err(format!(
                        "conversation {:?} reappears at turn {t} after another conversation; \
                         SITS needs each conversation's turns contiguous and in speaking order",
                        conv_labels[t]
                    )));
                }
                conv_start[t] = true;
            }
        }
        // Speaker vocabulary: sorted numerically for integer labels, else as strings.
        let mut names: Vec<String> = spk_labels
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if speaker_is_int {
            names.sort_by_key(|s| s.parse::<i64>().unwrap_or(0));
        }
        let id: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let spk_ids: Vec<u32> = spk_labels.iter().map(|s| id[s.as_str()]).collect();

        let compat = match slf.compat.as_deref() {
            Some("rossiter2022") => Compat::Rossiter2022 {
                init_every: compat_init_every(slf.init_shift_rate)?,
            },
            _ => Compat::None,
        };
        let num_draws = iters - burn_in;
        if sample_interval == Some(0) {
            return Err(PyValueError::new_err("sample_interval must be >= 1"));
        }
        let sample_interval = sample_interval.unwrap_or(num_draws.div_ceil(2000)).max(1);
        let cfg = SitsConfig {
            num_topics: slf.num_topics,
            alpha: slf.alpha,
            beta: slf.beta,
            gamma: slf.gamma,
            min_shift_tokens: slf.min_shift_tokens,
            init_shift_rate: slf.init_shift_rate,
            compat,
            sampler: if slf.sampler == "block" {
                Sampler::Block
            } else {
                Sampler::Single
            },
            warmup: std::env::var("TOPICA_SITS_WARMUP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            iters,
            burn_in,
            sample_interval,
        };

        // Warn when the length threshold removes a large share of turns.
        let non_first = conv_start.iter().filter(|&&s| !s).count();
        let short = (0..nt)
            .filter(|&t| !conv_start[t] && corpus.docs[t].len() < slf.min_shift_tokens)
            .count();
        if non_first > 0 && short as f64 / non_first as f64 > INELIGIBLE_WARN_SHARE {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "SITS: {short} of {non_first} non-first turns ({:.0}%) have fewer than \
                     min_shift_tokens={} tokens and can never be topic shifts; they count \
                     toward their speaker as non-shifts. Report this share with the \
                     agenda-setting scores, and check sensitivity to min_shift_tokens.",
                    100.0 * short as f64 / non_first as f64,
                    slf.min_shift_tokens
                ),),
            )?;
        }

        let num_types = corpus.num_types();
        let num_speakers = names.len();
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let progress = resolve_progress(py, progress, "SITS")?;
        let turn_speaker = spk_ids.clone();
        let (model, corpus, conv_start) = py.allow_threads(move || {
            let mut on_progress = on_progress_bare(&progress);
            let data = SitsData {
                turns: &corpus.docs,
                speakers: &spk_ids,
                conv_start: &conv_start,
                num_speakers,
                num_types,
            };
            let m = crate::sits::fit(&data, &cfg, &mut on_progress, &mut rng);
            (m, corpus, conv_start)
        });
        reraise_if_interrupted(py)?;

        // Geweke check on the post-burn-in eligible-shift trace.
        let kept: Vec<f64> = model
            .shift_trace
            .iter()
            .skip(burn_in)
            .map(|&x| x as f64)
            .collect();
        let gz = crate::sits::geweke_z(&kept, 0.1, 0.5);
        if gz.is_none() {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "SITS: only {} post-burn-in sweeps, too few to check convergence \
                     (need at least 200); these results are not usable estimates",
                    kept.len()
                ),),
            )?;
        }
        if let Some(z) = gz {
            if z.abs() > 2.0 {
                let warnings = py.import_bound("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "SITS: the number of topic shifts is still drifting after burn-in \
                         (Geweke z = {z:.2} on the eligible-shift trace), so the chain has \
                         probably not converged. Increase iters (and burn_in), and compare \
                         several seeds before reporting agenda-setting scores."
                    ),),
                )?;
            }
        }
        if compat != Compat::None && model.num_phantom > 0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "SITS(compat='rossiter2022'): {} short turns started as shifts and will \
                     remain segment boundaries counted as shifts for the whole chain \
                     (the fork's bookkeeping defect). This reproduces published results but \
                     inflates shift rates; use the default mode for new analyses.",
                    model.num_phantom
                ),),
            )?;
        }

        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.speaker_names = names;
        slf.speaker_is_int = speaker_is_int;
        slf.turn_speaker = turn_speaker;
        slf.conv_start = conv_start;
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.burn_in = burn_in;
        slf.iters = iters;
        slf.geweke = gz;
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Topic-word matrix φ (num_topics, vocab) from the terminal Gibbs state,
    /// (n_kw + β)/(n_k + Vβ), as the reference's ``phi.txt``. Rows sum to 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }

    /// Turn-topic matrix θ (num_turns, num_topics): each turn's *segment* mixture in
    /// the terminal Gibbs state, (n_seg,k + α)/(n_seg + Kα); turns in one terminal
    /// segment share a row. The terminal segmentation can differ from
    /// :attr:`segments` (the posterior-majority segmentation), and in compat mode it
    /// includes the phantom boundaries. For each turn's own topic mix (the
    /// reference's ``theta.txt``, and the right input for ``find_thoughts``) see
    /// :attr:`turn_topic`.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Each turn's own smoothed topic proportions (num_turns, num_topics),
    /// (n_t,k + α)/(n_t + Kα), in the terminal state: the reference's ``theta.txt``.
    #[getter]
    fn turn_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.turn_topic).to_pyarray_bound(py))
    }

    /// Posterior probability that each turn shifts the topic (num_turns,): the
    /// mean of the sampled shift indicator over every post-burn-in sweep. A
    /// conversation's first turn is 1; a turn shorter than ``min_shift_tokens`` is 0.
    #[getter]
    fn shift_prob<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.shift_prob.clone()).to_pyarray_bound(py))
    }

    /// Whether each turn is sampled as a possible shift (num_turns,): not the first
    /// turn of its conversation and at least ``min_shift_tokens`` tokens long.
    #[getter]
    fn eligible<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        Ok(Array1::from(self.fitted_model()?.eligible.clone()).to_pyarray_bound(py))
    }

    /// Segment id of each turn (num_turns,), numbered from 0 across the corpus: a
    /// new segment starts at every conversation's first turn and at every turn with
    /// :attr:`shift_prob` >= 0.5 (the posterior-majority segmentation).
    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let m = self.fitted_model()?;
        let mut id = -1i64;
        let seg: Vec<i64> = (0..m.shift_prob.len())
            .map(|t| {
                if self.conv_start[t] || m.shift_prob[t] >= 0.5 {
                    id += 1;
                }
                id
            })
            .collect();
        Ok(Array1::from(seg).to_pyarray_bound(py))
    }

    /// Conversation index of each turn (num_turns,), numbered from 0 in order.
    #[getter]
    fn conversation_index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fitted_model()?;
        let mut id = -1i64;
        let v: Vec<i64> = self
            .conv_start
            .iter()
            .map(|&s| {
                if s {
                    id += 1;
                }
                id
            })
            .collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }

    /// The speaker labels indexing every per-speaker array: integers sorted
    /// numerically when the speakers were integers, else strings sorted.
    #[getter]
    fn speakers(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.fitted_model()?;
        if self.speaker_is_int {
            let v: Vec<i64> = self
                .speaker_names
                .iter()
                .map(|s| s.parse::<i64>().unwrap_or(0))
                .collect();
            Ok(v.into_py(py))
        } else {
            Ok(self.speaker_names.clone().into_py(py))
        }
    }

    /// Each turn's speaker as a position in :attr:`speakers` (num_turns,).
    #[getter]
    fn speaker_index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fitted_model()?;
        let v: Vec<i64> = self.turn_speaker.iter().map(|&x| x as i64).collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }

    /// Share of non-first turns too short to be sampled as shifts (fewer than
    /// ``min_shift_tokens`` tokens, counted on the tokens passed to ``fit``). Report
    /// it with the agenda-setting scores.
    #[getter]
    fn short_turn_share(&self) -> PyResult<f64> {
        let m = self.fitted_model()?;
        let non_first = self.conv_start.iter().filter(|&&s| !s).count();
        let elig = m.eligible.iter().filter(|&&e| e).count();
        Ok(if non_first == 0 {
            f64::NAN
        } else {
            (non_first - elig) as f64 / non_first as f64
        })
    }

    /// Number of turns per speaker, aligned to :attr:`speakers`.
    #[getter]
    fn speaker_turn_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let c: Vec<i64> = self
            .fitted_model()?
            .speaker_turns
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(Array1::from(c).to_pyarray_bound(py))
    }

    /// Number of eligible (sampled) turns per speaker, aligned to :attr:`speakers`.
    #[getter]
    fn speaker_eligible_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let c: Vec<i64> = self
            .fitted_model()?
            .speaker_eligible
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(Array1::from(c).to_pyarray_bound(py))
    }

    /// Agenda-setting score per speaker (num_speakers,), Rossiter's (2022) measure
    /// as computed by ``sitsr::readSits``: the posterior mean over post-burn-in
    /// draws of (γ + number of the speaker's turns that shift) / (2γ + number of
    /// the speaker's turns). It counts *all* of a speaker's turns: conversation
    /// openers always shift and turns shorter than ``min_shift_tokens`` never do,
    /// so the score mixes the sampled shifts with each speaker's share of openers
    /// and short turns. Report :attr:`eligible_shift_rate` next to it.
    #[getter]
    fn shift_propensity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let g = self.gamma;
        let v: Vec<f64> = m
            .speaker_shift_mean
            .iter()
            .zip(&m.speaker_turns)
            .map(|(&s, &n)| (g + s) / (2.0 * g + n as f64))
            .collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }

    /// Stored posterior draws of :attr:`shift_propensity` (num_stored_draws,
    /// num_speakers), thinned by ``sample_interval``.
    #[getter]
    fn shift_propensity_draws<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.propensity_draws()?;
        let ncol = self.speaker_names.len();
        Ok(Array2::from_shape_vec((d.len(), ncol), d.concat())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray_bound(py))
    }

    /// Equal-tailed posterior interval of :attr:`shift_propensity` (num_speakers, 2)
    /// from the stored draws. Reflects within-chain uncertainty only: fit several
    /// seeds to see between-chain variation.
    #[pyo3(signature = (level=0.9))]
    fn shift_propensity_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.propensity_draws()?;
        Ok(Self::interval(&d, self.speaker_names.len(), level)?.to_pyarray_bound(py))
    }

    /// Share of each speaker's *eligible* turns that shift the topic, posterior mean
    /// (num_speakers,); NaN for a speaker with no eligible turns. Unlike
    /// :attr:`shift_propensity`, this leaves out conversation openers and short
    /// turns, so it isolates what the sampler inferred.
    #[getter]
    fn eligible_shift_rate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let v: Vec<f64> = m
            .speaker_eligible_shift_mean
            .iter()
            .zip(&m.speaker_eligible)
            .map(|(&s, &n)| if n == 0 { f64::NAN } else { s / n as f64 })
            .collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }

    /// Stored posterior draws of :attr:`eligible_shift_rate` (num_stored_draws,
    /// num_speakers).
    #[getter]
    fn eligible_shift_rate_draws<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.eligible_rate_draws()?;
        let ncol = self.speaker_names.len();
        Ok(Array2::from_shape_vec((d.len(), ncol), d.concat())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray_bound(py))
    }

    /// Equal-tailed posterior interval of :attr:`eligible_shift_rate`
    /// (num_speakers, 2).
    #[pyo3(signature = (level=0.9))]
    fn eligible_shift_rate_interval<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.eligible_rate_draws()?;
        Ok(Self::interval(&d, self.speaker_names.len(), level)?.to_pyarray_bound(py))
    }

    /// Number of eligible turns sampled as shifts after each sweep (sweeps,): the
    /// convergence trace. It should be flat after burn-in.
    #[getter]
    fn shift_trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let v: Vec<i64> = self
            .fitted_model()?
            .shift_trace
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }

    /// Geweke z-score of the post-burn-in :attr:`shift_trace` (first 10% vs last
    /// 50%, batch-means variances). ``|z| > 2`` suggests the chain is still
    /// drifting. None when the trace is too short.
    #[getter]
    fn geweke_z(&self) -> PyResult<Option<f64>> {
        self.fitted_model()?;
        Ok(self.geweke)
    }

    /// Number of post-burn-in sweeps averaged into the posterior means.
    #[getter]
    fn num_draws(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_draws)
    }

    /// Burn-in sweeps discarded by the last fit.
    #[getter]
    fn burn_in(&self) -> PyResult<usize> {
        self.fitted_model()?;
        Ok(self.burn_in)
    }

    /// The sampler's internal per-speaker counts (num_speakers, 2) of
    /// ``(non-shifts, shifts)`` in the terminal state, the reference's ``pi.txt``
    /// numerators. In the default mode they equal the recorded shifts; with
    /// ``compat='rossiter2022'`` they also carry the phantom shifts.
    #[getter]
    fn sampler_shift_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let m = self.fitted_model()?;
        let mut a = Array2::<i64>::zeros((m.sampler_speaker_counts.len(), 2));
        for (i, c) in m.sampler_speaker_counts.iter().enumerate() {
            a[[i, 0]] = c[0] as i64;
            a[[i, 1]] = c[1] as i64;
        }
        Ok(a.to_pyarray_bound(py))
    }

    /// Number of phantom segment boundaries (``compat='rossiter2022'`` only; 0 in
    /// the default mode).
    #[getter]
    fn num_phantom(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_phantom)
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

    /// Collapsed log joint (speaker Beta-Bernoulli + topic-word + segment
    /// Dirichlet-multinomials) of the state after each recorded sweep, as
    /// ``(sweep, value)``; the reference's ``loglikelihood.txt`` logs the same
    /// quantity before each sweep.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }

    /// Always False: SITS runs the requested ``iters`` (no early stop). Use
    /// :attr:`geweke_z` and several seeds to judge convergence.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Alias of :attr:`converged` (issue #755): True only on an early stop.
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

    /// Per-topic coherence, shape ``(num_topics,)``. ``coherence_type`` is
    /// ``"u_mass"`` (default), ``"c_v"``, ``"c_uci"`` or ``"c_npmi"``; ``texts``
    /// supplies the reference corpus for the windowed measures.
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
        self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_SITS,
            &SitsState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                beta: self.beta,
                gamma: self.gamma,
                min_shift_tokens: self.min_shift_tokens,
                init_shift_rate: self.init_shift_rate,
                compat: self.compat.clone(),
                sampler: self.sampler.clone(),
                seed: self.seed,
                fitted: self.fitted,
                speaker_names: self.speaker_names.clone(),
                speaker_is_int: self.speaker_is_int,
                turn_speaker: self.turn_speaker.clone(),
                conv_start: self.conv_start.clone(),
                topic_names: self.topic_names.clone(),
                burn_in: self.burn_in,
                iters: self.iters,
                geweke: self.geweke,
                model: self.model.clone(),
                corpus: self.corpus.clone(),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SitsState = read_state(path, MODEL_TAG_SITS)?;
        Ok(SITS {
            num_topics: s.num_topics,
            alpha: s.alpha,
            beta: s.beta,
            gamma: s.gamma,
            min_shift_tokens: s.min_shift_tokens,
            init_shift_rate: s.init_shift_rate,
            compat: s.compat,
            sampler: s.sampler,
            seed: s.seed,
            fitted: s.fitted,
            speaker_names: s.speaker_names,
            speaker_is_int: s.speaker_is_int,
            turn_speaker: s.turn_speaker,
            conv_start: s.conv_start,
            topic_names: s.topic_names,
            burn_in: s.burn_in,
            iters: s.iters,
            geweke: s.geweke,
            model: s.model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SITS(num_topics={}, speakers={}, compat={:?}, fitted={})",
            self.num_topics,
            self.speaker_names.len(),
            self.compat,
            self.fitted
        )
    }
}
