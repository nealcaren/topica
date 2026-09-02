//! Python binding for ThreadTM — a reply-threaded topic model (CTM/STM logistic-normal topics
//! with a reply-tree structured prior; see `crate::thread_tm`). Experimental tier: topica-original,
//! no published reference yet, so `fit` is gated behind `topica.enable_experimental()`.
//!
//! The class exposes the validated core — fit on a `Corpus` or token lists + a reply tree + an
//! optional categorical covariate, with topic/proportion/prevalence readouts (prevalence carries a
//! cluster-robust method-of-composition SE), coherence, and save/load. Reply persistence is best
//! read from `persistence()` — an identifiable reduced-form estimate (observed slope + reliability
//! gate + attenuation-corrected structural κ) — rather than the ML `kappa` getter, which collapses
//! to the σ² floor on real corpora. The covariate story lives entirely in
//! `group_prevalence`/`prevalence_se`; ThreadTM is outside the `effects` namespace. `transform`
//! infers proportions for new reply forests (a single topological pass, topics/field/anchors held
//! fixed); formula covariates remain a follow-up.

use super::*;
use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::types::{PyBool, PyDict, PyList, PyType};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// ThreadTM: a reply-threaded topic model. A reply's topic prior is coupled to the comment it
/// answers (a persistence-smoothing prior along reply edges), reverting toward its covariate-group
/// baseline; `kappa` measures the reversion (on real corpora it is typically ~0, i.e. persistence-
/// dominated). Reduces to a plain logistic-normal topic model when the reply tree is flat.
/// `num_topics` is K; `em_iters` the variational-EM iteration cap; `seed` makes the fit deterministic.
/// `coupling` chooses the prior structure: `"parent"` (default; shrink toward the immediate parent,
/// the reply-chain prior), `"root"` (shrink toward the thread root, a broadcast / topic-around-the-
/// root prior), or `"blend"` (shrink toward both, `α·parent + β·root + (1-α-β)·anchor`, with the
/// weights estimated or pinned via `blend_alpha`/`blend_beta`).
#[pyclass(module = "topica")]
pub struct ThreadTM {
    num_topics: usize,
    em_iters: usize,
    seed: u64,
    // Reply coupling structure: "parent" (a node shrinks toward its immediate parent, the OU
    // reply-chain prior) or "root" (a node shrinks toward its thread root, a broadcast /
    // topic-around-the-root prior). "root" is fit by running the SAME kernel on a reparented
    // depth-2 star topology (every non-root points at its thread root), so the whole field / kappa
    // / transform / persistence stack is unchanged; only the coupling neighbor differs. "blend"
    // couples each node to BOTH its parent and its thread root (α·parent + β·root + rest·anchor);
    // its weights are estimated (or pinned by blend_alpha_fixed/blend_beta_fixed).
    coupling: String,
    // Fixed blend weights from the constructor (None = estimate in the M-step). Retained so
    // `settings` round-trips and refits reproduce the same pinning.
    blend_alpha_fixed: Option<f64>,
    blend_beta_fixed: Option<f64>,
    // Effective (fitted or pinned) blend weights after fit; NaN unless coupling="blend" and fitted.
    blend_alpha: f64,
    blend_beta: f64,
    // Cluster-robust (thread-root) sandwich SEs of the fitted blend weights and the anchor share
    // (issue #863); NaN unless blend fitted with >=2 threads, 0 for a pinned weight, and the
    // individual weight SEs NaN (anchor SE kept) when the α/β split is not identified.
    blend_alpha_se: f64,
    blend_beta_se: f64,
    blend_anchor_se: f64,
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
    // Per-topic names (length K). Positional `topic_i` unless seeded with a string-keyed
    // `seed_words` dict, whose keys name the seeded topics (issue #854). Empty until fitted.
    topic_names: Vec<String>,
    // Which fitted-vocabulary words each seeded topic's patterns actually matched (issue #856):
    // `seed_matches[t]` is topic `t`'s matched words, empty for an unseeded topic. Lets a user
    // audit what `seed_match="glob"`/`"regex"` resolved to. Empty when the fit was unseeded.
    seed_matches: Vec<Vec<String>>,
    beta: Vec<Vec<f64>>,
    doc_topic: Vec<Vec<f64>>,
    doc_eta: Vec<Vec<f64>>,
    doc_topic_var: Vec<Vec<f64>>,
    group_prevalence: Vec<Vec<f64>>,
    prevalence_se: Vec<Vec<f64>>,
    kappa: f64,
    kappa_ci: (f64, f64),
    sigma2: f64,
    p0: f64,
    // Full root-prior covariance Σ_root (K-1 × K-1, row-major); the base logistic-normal prior for
    // root documents and for `transform`'s root nodes (#834).
    sigma_root: Vec<f64>,
    // Full edge (OU step) covariance Σ_edge; the reply edges' prior covariance, on the same footing
    // as sigma_root so tree and no-tree share the base covariance.
    sigma_edge: Vec<f64>,
    bound_history: Vec<f64>,
    // Whether the variational EM bound converged within `em_iters` (bound change below tol) rather
    // than hitting the iteration cap. Computed by the kernel; exposed via `converged`.
    converged: bool,
    // Content covariate (issue #841): per-level topic-word matrices (G_content × K × V) and the SAGE
    // κ deviations, empty unless fit with `content=`. `num_content_groups` is 1 when no content
    // covariate was fit. `content_names` labels the levels.
    content_beta: Vec<Vec<Vec<f64>>>,
    content_kappa_m: Vec<f64>,
    content_kappa_topic: Vec<Vec<f64>>,
    content_kappa_cov: Vec<Vec<f64>>,
    content_kappa_interaction: Vec<Vec<f64>>,
    content_names: Vec<String>,
    num_content_groups: usize,
    // The depth-bin lower edges used when fit with content="depth" (empty otherwise), so transform
    // can re-bin a new forest's depths the same way.
    content_depth_edges: Vec<usize>,
    // Training reply tree + covariate groups (document-aligned), retained so `persistence()` can
    // re-fit an uncoupled pass and regress child η on parent η.
    fit_parents: Vec<i64>,
    fit_groups: Vec<usize>,
    // Training corpus (all documents kept, including any emptied by min_count, so the reply-tree
    // node indices stay valid). Backs `coherence()` and save/load.
    corpus: Option<corpus::Corpus>,
}

/// Serialisable snapshot of a fitted ThreadTM (see `save`/`load`).
#[derive(serde::Serialize, serde::Deserialize)]
struct ThreadTmState {
    num_topics: usize,
    em_iters: usize,
    seed: u64,
    // Defaults a missing `coupling` to the original parent coupling. NOTE this is inert for the
    // current positional-bincode save format (a genuinely older ThreadTM save, which predates this
    // field, cannot round-trip and will fail to load) — it only migrates a self-describing format.
    // Acceptable under the pre-v1.0 save-compat policy: ThreadTM is new and experimental. Kept for
    // consistency with the other states (see mod.rs / neural.rs).
    #[serde(default = "default_coupling")]
    coupling: String,
    #[serde(default = "default_none_f64")]
    blend_alpha_fixed: Option<f64>,
    #[serde(default = "default_none_f64")]
    blend_beta_fixed: Option<f64>,
    #[serde(default = "default_nan")]
    blend_alpha: f64,
    #[serde(default = "default_nan")]
    blend_beta: f64,
    #[serde(default = "default_nan")]
    blend_alpha_se: f64,
    #[serde(default = "default_nan")]
    blend_beta_se: f64,
    #[serde(default = "default_nan")]
    blend_anchor_se: f64,
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
    #[serde(default)]
    topic_names: Vec<String>,
    #[serde(default)]
    seed_matches: Vec<Vec<String>>,
    beta: Vec<Vec<f64>>,
    doc_topic: Vec<Vec<f64>>,
    doc_eta: Vec<Vec<f64>>,
    doc_topic_var: Vec<Vec<f64>>,
    group_prevalence: Vec<Vec<f64>>,
    prevalence_se: Vec<Vec<f64>>,
    kappa: f64,
    kappa_ci: (f64, f64),
    sigma2: f64,
    p0: f64,
    #[serde(default)]
    sigma_root: Vec<f64>,
    #[serde(default)]
    sigma_edge: Vec<f64>,
    bound_history: Vec<f64>,
    #[serde(default)]
    converged: bool,
    #[serde(default)]
    content_beta: Vec<Vec<Vec<f64>>>,
    #[serde(default)]
    content_kappa_m: Vec<f64>,
    #[serde(default)]
    content_kappa_topic: Vec<Vec<f64>>,
    #[serde(default)]
    content_kappa_cov: Vec<Vec<f64>>,
    #[serde(default)]
    content_kappa_interaction: Vec<Vec<f64>>,
    #[serde(default)]
    content_names: Vec<String>,
    #[serde(default = "default_one_usize")]
    num_content_groups: usize,
    #[serde(default)]
    content_depth_edges: Vec<usize>,
    fit_parents: Vec<i64>,
    fit_groups: Vec<usize>,
    corpus: Option<corpus::Corpus>,
}

fn default_coupling() -> String {
    "parent".to_string()
}

fn default_none_f64() -> Option<f64> {
    None
}

fn default_nan() -> f64 {
    f64::NAN
}

fn default_one_usize() -> usize {
    1
}

/// Each node's thread root (itself for a root). Used to build the blend `BlendConfig`.
fn thread_roots(parents: &[i64]) -> Vec<usize> {
    (0..parents.len())
        .map(|start| {
            let mut cur = start as i64;
            while parents[cur as usize] >= 0 {
                cur = parents[cur as usize];
            }
            cur as usize
        })
        .collect()
}

/// Reparent a forest to its thread-root star: every non-root node points at its thread root, roots
/// stay roots. This is the topology the `coupling="root"` variant fits and infers on. `parents`
/// must already be validated acyclic (each parent chain reaches a root).
fn root_star_parents(parents: &[i64]) -> Vec<i64> {
    let n = parents.len();
    let mut root = vec![-1i64; n];
    for start in 0..n {
        let mut cur = start as i64;
        while parents[cur as usize] >= 0 {
            cur = parents[cur as usize];
        }
        root[start] = if cur == start as i64 { -1 } else { cur };
    }
    root
}

/// Linear-interpolated percentile of an already-ascending-sorted slice (`q` in [0, 1]).
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Coerce a Python `covariates` argument for `fit` into dense integer group ids `0..G` plus, for a
/// factorized categorical input, the distinct labels in id order. Integer input (a list or a pandas
/// Series of ints) keeps its values, validated non-negative. String/categorical input is factorized
/// in first-seen order and its labels become the default group names. `None` here is handled by the
/// caller (a single global anchor).
fn coerce_fit_covariates(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Option<Vec<String>>)> {
    // Integer path first: preserve the original behavior (a Series of numpy ints extracts here too).
    if let Ok(ints) = obj.extract::<Vec<i64>>() {
        let mut out = Vec::with_capacity(ints.len());
        for (d, &v) in ints.iter().enumerate() {
            if v < 0 {
                return Err(PyValueError::new_err(format!(
                    "covariates[{d}] = {v} is negative; pass dense integer group ids 0..num_groups, \
                     or string/categorical labels to have them auto-encoded"
                )));
            }
            out.push(v as usize);
        }
        return Ok((out, None));
    }
    // Whole-valued floats: a common pandas artifact (an integer group column that ever held a NaN
    // is stored as float64), so accept 0.0/1.0/... as integer ids rather than rejecting them.
    if let Ok(floats) = obj.extract::<Vec<f64>>() {
        let mut out = Vec::with_capacity(floats.len());
        for (d, &v) in floats.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(PyValueError::new_err(format!(
                    "covariates[{d}] = {v} is not a non-negative whole number; pass integer group \
                     ids 0..num_groups (e.g. df['group'].astype(int)) or string/categorical labels"
                )));
            }
            out.push(v as usize);
        }
        return Ok((out, None));
    }
    // String / categorical path: factorize in first-seen order.
    let labels: Vec<String> = obj.extract::<Vec<String>>().map_err(|_| {
        PyValueError::new_err(
            "covariates must be a sequence of dense integer group ids (0..num_groups) or of \
             string/categorical labels (auto-encoded to 0..num_groups); got an unsupported type",
        )
    })?;
    let mut cats: Vec<String> = Vec::new();
    let mut idx: HashMap<String, usize> = HashMap::new();
    let mut groups = Vec::with_capacity(labels.len());
    for s in labels {
        let g = *idx.entry(s.clone()).or_insert_with(|| {
            cats.push(s.clone());
            cats.len() - 1
        });
        groups.push(g);
    }
    Ok((groups, Some(cats)))
}

/// Coerce a Python `covariates` argument for `transform` into integer group ids aligned to the
/// FITTED groups. Integer input passes through (validated against the group count by the caller);
/// string/categorical input is mapped through the fitted labels, and an unseen label is an error.
fn coerce_transform_covariates(obj: &Bound<'_, PyAny>, fitted: &[String]) -> PyResult<Vec<i64>> {
    if let Ok(ints) = obj.extract::<Vec<i64>>() {
        return Ok(ints);
    }
    // Whole-valued floats (pandas float64 group column) map to integer ids, as in fit.
    if let Ok(floats) = obj.extract::<Vec<f64>>() {
        let mut out = Vec::with_capacity(floats.len());
        for (d, &v) in floats.iter().enumerate() {
            if !v.is_finite() || v.fract() != 0.0 {
                return Err(PyValueError::new_err(format!(
                    "covariates[{d}] = {v} is not a whole number; pass integer group ids or \
                     string/categorical labels matching the fitted groups"
                )));
            }
            out.push(v as i64);
        }
        return Ok(out);
    }
    let labels: Vec<String> = obj.extract::<Vec<String>>().map_err(|_| {
        PyValueError::new_err(
            "covariates must be integer group ids or string/categorical labels matching the \
             fitted groups",
        )
    })?;
    let idx: HashMap<&str, i64> = fitted
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut out = Vec::with_capacity(labels.len());
    for (d, s) in labels.iter().enumerate() {
        match idx.get(s.as_str()) {
            Some(&g) => out.push(g),
            None => {
                return Err(PyValueError::new_err(format!(
                    "covariates[{d}] = {s:?} is not one of the fitted group labels {fitted:?}"
                )))
            }
        }
    }
    Ok(out)
}

/// Bin each document's depth (steps to its thread root along `parents`) into content levels by a
/// list of ascending lower edges: level = the number of edges `<= depth`, minus 1. Default edges
/// `[0, 1, 3]` give root (0) / shallow (1-2) / deep (3+). Returns the per-document levels, the
/// level count, and default level names.
fn depth_content_groups(parents: &[i64], edges: &[usize]) -> (Vec<usize>, usize, Vec<String>) {
    let n_levels = edges.len();
    let groups: Vec<usize> = (0..parents.len())
        .map(|start| {
            let mut steps = 0usize;
            let mut cur = start as i64;
            while cur >= 0 && parents[cur as usize] >= 0 {
                cur = parents[cur as usize];
                steps += 1;
            }
            // largest level i with edges[i] <= steps (edges ascending, edges[0] == 0)
            let mut b = 0usize;
            for (i, &e) in edges.iter().enumerate() {
                if steps >= e {
                    b = i;
                }
            }
            b
        })
        .collect();
    // Default names: the canonical 3-bin scheme reads as root/shallow/deep; otherwise "depth>=e".
    let names = if edges == [0, 1, 3] {
        vec![
            "root".to_string(),
            "shallow".to_string(),
            "deep".to_string(),
        ]
    } else {
        edges.iter().map(|e| format!("depth>={e}")).collect()
    };
    (groups, n_levels, names)
}

impl ThreadTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }

    /// Resolve a content level given either its integer index or its string label.
    fn resolve_content_level(&self, level: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = level.extract::<usize>() {
            if i >= self.num_content_groups {
                return Err(PyValueError::new_err(format!(
                    "content level {i} out of range; the model has {} level(s)",
                    self.num_content_groups
                )));
            }
            return Ok(i);
        }
        if let Ok(s) = level.extract::<String>() {
            if let Some(i) = self.content_names.iter().position(|n| n == &s) {
                return Ok(i);
            }
            return Err(PyValueError::new_err(format!(
                "content level {s:?} is not one of the fitted levels {:?}",
                self.content_names
            )));
        }
        Err(PyValueError::new_err(
            "content level must be an integer index or a string label",
        ))
    }
}

#[pymethods]
impl ThreadTM {
    #[new]
    #[pyo3(signature = (num_topics, *, em_iters=150, seed=13, coupling="parent".to_string(), blend_alpha=None, blend_beta=None))]
    fn new(
        num_topics: usize,
        em_iters: usize,
        seed: u64,
        coupling: String,
        blend_alpha: Option<f64>,
        blend_beta: Option<f64>,
    ) -> PyResult<Self> {
        // Gate at construction, not only at fit (issue #856): a first-timer who builds the model
        // before enabling the experimental tier gets the requirement immediately, not many lines
        // into a fit later.
        require_experimental("ThreadTM")?;
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if em_iters == 0 {
            return Err(PyValueError::new_err("em_iters must be >= 1"));
        }
        if coupling != "parent" && coupling != "root" && coupling != "blend" {
            return Err(PyValueError::new_err(format!(
                "coupling must be \"parent\" (immediate parent), \"root\" (thread root), or \
                 \"blend\" (both); got {coupling:?}"
            )));
        }
        // blend weights are only meaningful under blend coupling; a pinned weight must be a valid
        // convex share (α, β ≥ 0 and α + β ≤ 1 so the anchor keeps the remainder).
        if coupling != "blend" && (blend_alpha.is_some() || blend_beta.is_some()) {
            return Err(PyValueError::new_err(
                "blend_alpha/blend_beta are only valid with coupling=\"blend\"",
            ));
        }
        for (name, w) in [("blend_alpha", blend_alpha), ("blend_beta", blend_beta)] {
            if let Some(v) = w {
                if !(0.0..=1.0).contains(&v) {
                    return Err(PyValueError::new_err(format!("{name} must be in [0, 1]")));
                }
            }
        }
        if let (Some(a), Some(b)) = (blend_alpha, blend_beta) {
            if a + b > 1.0 {
                return Err(PyValueError::new_err(
                    "blend_alpha + blend_beta must be <= 1 (the anchor takes the remaining weight)",
                ));
            }
        }
        Ok(ThreadTM {
            num_topics,
            em_iters,
            seed,
            coupling,
            blend_alpha_fixed: blend_alpha,
            blend_beta_fixed: blend_beta,
            blend_alpha: f64::NAN,
            blend_beta: f64::NAN,
            blend_alpha_se: f64::NAN,
            blend_beta_se: f64::NAN,
            blend_anchor_se: f64::NAN,
            fitted: false,
            vocab: Vec::new(),
            group_names: Vec::new(),
            topic_names: Vec::new(),
            seed_matches: Vec::new(),
            beta: Vec::new(),
            doc_topic: Vec::new(),
            doc_eta: Vec::new(),
            doc_topic_var: Vec::new(),
            group_prevalence: Vec::new(),
            prevalence_se: Vec::new(),
            kappa: f64::NAN,
            kappa_ci: (f64::NAN, f64::NAN),
            sigma2: f64::NAN,
            p0: f64::NAN,
            sigma_root: Vec::new(),
            sigma_edge: Vec::new(),
            bound_history: Vec::new(),
            converged: false,
            content_beta: Vec::new(),
            content_kappa_m: Vec::new(),
            content_kappa_topic: Vec::new(),
            content_kappa_cov: Vec::new(),
            content_kappa_interaction: Vec::new(),
            content_names: Vec::new(),
            num_content_groups: 1,
            content_depth_edges: Vec::new(),
            fit_parents: Vec::new(),
            fit_groups: Vec::new(),
            corpus: None,
        })
    }

    /// Fit ThreadTM. `data` is either a `topica.Corpus` or a list of token lists (already
    /// tokenized). `parents[d]` is `d`'s parent **document index** in the reply tree (`-1` for a
    /// thread root); build it in the SAME order as the documents. `covariates` is an optional
    /// per-document categorical group id in a DENSE range `0..num_groups` (the reversion anchor
    /// becomes that group's baseline prevalence); omit for a single global anchor. String or
    /// categorical labels (a list, or a pandas Series) are accepted too and auto-encoded to
    /// `0..num_groups` in first-seen order, with the distinct labels becoming the group names.
    /// `covariate_names` names the groups for the readouts (overriding auto-encoded labels).
    /// `min_count` drops words rarer than it (the same vocabulary knob `Corpus` spells `min_cf`).
    ///
    /// Two INDEPENDENT supervision axes, both orthogonal to the reply tree:
    /// (A) WORD SEEDING — `seed_words` (+ `weight`/`seed_strength`, `seed_prior`, `seed_match`) shapes
    /// what seeded topics MEAN; (B) PREVALENCE ANCHORING — `prevalence_anchor` (+ `prevalence_strength`)
    /// steers a covariate group's topic MIX. They compose and never share a knob.
    ///
    /// (A) `seed_words` biases seeded topics' word distributions toward the keywords and pins them to
    /// fixed slots, while unseeded topics are learned freely. Its keys are EITHER int topic
    /// indices (`{0: [...]}`, explicit slots) OR string topic names (`{"space": [...]}`, which name
    /// the seeded topics and take the leading slots in insertion order, as SeededLDA/KeyATM do, and
    /// populate `topic_names`); keys must be all-int or all-string, not mixed. Audit which vocabulary
    /// words each pattern matched with the `seed_matches` property after fitting.
    /// `weight` matches SeededLDA's `weight` (a `[0, 1]` fraction, default `0.01`).
    /// `seed_prior="frequency"` (default) gives each matched seed word a pseudocount of
    /// `corpus_count(word) * weight * 100`; `"uniform"` a flat `weight * 100` per word;
    /// `seed_strength` (if set) overrides both with a flat RAW per-word pseudocount (so it silently
    /// supersedes `seed_prior`/`weight`). `seed_match` is `"fixed"` (default), `"glob"`, or `"regex"`, one
    /// strategy for the whole dict, with `case_insensitive`. Seeding is rejected together with a
    /// `content` covariate. `prevalence_anchor={group_index: [K-length mix]}` shrinks a group's
    /// baseline topic mix toward the target by `prevalence_strength` (0..1); the key is the ENCODED
    /// integer group id (covariates are encoded `0..num_groups` in FIRST-SEEN order, not the string
    /// label and not alphabetical — check `group_names`/`covariate_names`), and the mix need not sum
    /// to 1. NOTE: the fit is deterministic given the inputs; `seed` does NOT vary the fit (the
    /// variational EM uses a fixed spectral init), so refitting across seeds is not a robustness
    /// check — resample threads instead. Experimental: requires `topica.enable_experimental()`.
    #[pyo3(signature = (data, parents=None, covariates=None, covariate_names=None, *, min_count=1,
                        content=None, content_names=None, content_prior="l2".to_string(),
                        content_prior_var=0.5, content_smooth=0.0, depth_bins=None,
                        seed_words=None, seed_prior="frequency".to_string(),
                        weight=0.01, seed_strength=None,
                        seed_match="fixed".to_string(), case_insensitive=false,
                        prevalence_anchor=None, prevalence_strength=0.5))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        parents: Option<Vec<i64>>,
        covariates: Option<Bound<'_, PyAny>>,
        covariate_names: Option<Vec<String>>,
        min_count: usize,
        content: Option<Bound<'_, PyAny>>,
        content_names: Option<Vec<String>>,
        content_prior: String,
        content_prior_var: f64,
        content_smooth: f64,
        depth_bins: Option<Vec<usize>>,
        // User supervision (issue #854). seed_words maps a topic index to keyword strings
        // (Dirichlet β seeding). `weight` matches SeededLDA's `weight` (a [0, 1] fraction, default
        // 0.01): seed_prior="frequency" (default) sets each seed word's pseudocount to
        // corpus_count(word) * weight * 100, so seeding is scale-robust and does not collapse a
        // topic to its seeds (SeededLDA's scheme); "uniform" uses a flat weight * 100. Passing
        // seed_strength overrides both with a flat RAW per-word pseudocount (unscaled).
        // seed_match/case_insensitive select how patterns match the vocabulary (shared with
        // SeededLDA). prevalence_anchor maps a covariate-group index to a length-K target topic mix.
        seed_words: Option<Bound<'_, PyAny>>,
        seed_prior: String,
        weight: f64,
        seed_strength: Option<f64>,
        seed_match: String,
        case_insensitive: bool,
        prevalence_anchor: Option<Bound<'_, PyAny>>,
        prevalence_strength: f64,
    ) -> PyResult<Py<Self>> {
        require_experimental("ThreadTM")?;
        // Accept either a topica.Corpus (materialise its token strings) or raw token lists, so the
        // reply tree can be built in the same document order the corpus was ingested in.
        let docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
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
                PyValueError::new_err(
                    "expected a Corpus or a list of token lists (list[list[str]])",
                )
            })?
        };
        let n = docs.len();
        if n == 0 {
            return Err(PyValueError::new_err("docs is empty"));
        }

        // vocab: words with >= min_count occurrences, in first-seen order
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for doc in &docs {
            for w in doc {
                *counts.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut vocab: Vec<String> = Vec::new();
        let mut wid: HashMap<&str, u32> = HashMap::new();
        for doc in &docs {
            for w in doc {
                if counts[w.as_str()] >= min_count && !wid.contains_key(w.as_str()) {
                    wid.insert(w.as_str(), vocab.len() as u32);
                    vocab.push(w.clone());
                }
            }
        }
        if vocab.is_empty() {
            return Err(PyValueError::new_err(
                "no words survive min_count; lower min_count",
            ));
        }
        // Documents emptied by min_count filtering: they stay as tree nodes (their latent η is
        // still coupled to parent/children) but contribute no tokens, so their proportions are
        // prior-only. Warn — a silently emptied node quietly changes the tree the user built.
        let docs_id: Vec<Vec<u32>> = docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| wid.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        let emptied = docs_id
            .iter()
            .zip(&docs)
            .filter(|(ids, orig)| ids.is_empty() && !orig.is_empty())
            .count();
        if emptied > 0 {
            PyErr::warn_bound(
                py,
                &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                &format!(
                    "min_count={min_count} emptied {emptied} document(s) of all tokens; they \
                     remain as reply-tree nodes but contribute no words (their topic proportions \
                     are prior-only). Lower min_count to keep them.",
                ),
                1,
            )?;
        }

        // parents: validate length + range + acyclicity (indices into the doc list; -1 = root)
        let par: Vec<i64> = match &parents {
            None => {
                // No reply tree: the model degenerates to a plain logistic-normal topic model
                // and the reply-structure parameters are undefined. Warn rather than silently
                // report a meaningless kappa/sigma2 (they come back NaN with no edges).
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    "ThreadTM.fit called with parents=None: no reply tree, so the model reduces to \
                     a plain logistic-normal topic model and kappa/sigma2 are undefined (NaN). \
                     Pass parents to use the reply structure.",
                    1,
                )?;
                vec![-1; n]
            }
            Some(p) => {
                if p.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "parents has {} entries but there are {n} documents",
                        p.len()
                    )));
                }
                for (d, &pd) in p.iter().enumerate() {
                    if pd < -1 || pd >= n as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] = {pd} out of range; must be -1 or in [0, {n})"
                        )));
                    }
                    if pd == d as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] points at itself"
                        )));
                    }
                }
                // acyclicity: walking parent links from any node must reach a root within n steps
                for start in 0..n {
                    let (mut cur, mut steps) = (start as i64, 0usize);
                    while cur >= 0 {
                        cur = p[cur as usize];
                        steps += 1;
                        if steps > n {
                            return Err(PyValueError::new_err(format!(
                                "parents contains a cycle reachable from document {start}; the \
                                 reply tree must be acyclic"
                            )));
                        }
                    }
                }
                p.clone()
            }
        };

        // covariate groups (integer ids, or string/categorical labels auto-encoded to 0..num_groups)
        let (groups, num_groups, group_names) = match &covariates {
            None => (vec![0usize; n], 1usize, vec!["all".to_string()]),
            Some(obj) => {
                let (g, derived_names) = coerce_fit_covariates(obj)?;
                if g.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "covariates has {} entries but there are {n} documents",
                        g.len()
                    )));
                }
                let ng = g.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                // A factorized categorical input is dense by construction; only integer input can
                // have a gap (a missing id creates an empty phantom group), so warn only there.
                if derived_names.is_none() {
                    for gid in 0..ng {
                        if !g.contains(&gid) {
                            PyErr::warn_bound(
                                py,
                                &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                                "covariates has a gap: group ids are not dense 0..num_groups, so an \
                                 empty phantom group will appear in group_prevalence. Re-code the \
                                 covariate to consecutive ids.",
                                1,
                            )?;
                            break;
                        }
                    }
                }
                // Explicit covariate_names win; otherwise the auto-encoded labels; otherwise generic.
                let names = covariate_names
                    .clone()
                    .or(derived_names)
                    .unwrap_or_else(|| (0..ng).map(|i| format!("group{i}")).collect());
                if names.len() != ng {
                    return Err(PyValueError::new_err(format!(
                        "covariate_names has {} names but the covariate has {ng} groups",
                        names.len()
                    )));
                }
                (g, ng, names)
            }
        };

        // Build a corpus snapshot (all documents kept, in tree-index order) for coherence()
        // and save/load. Uses the SAME vocab ids as `beta`'s columns.
        let corpus_snapshot = {
            let mut doc_freqs = vec![0u32; vocab.len()];
            let mut total_freqs = vec![0u32; vocab.len()];
            for doc in &docs_id {
                let mut seen = vec![false; vocab.len()];
                for &w in doc {
                    total_freqs[w as usize] += 1;
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        doc_freqs[w as usize] += 1;
                    }
                }
            }
            corpus::Corpus {
                id_to_word: vocab.clone(),
                docs: docs_id.clone(),
                doc_names: (0..n).map(|i| format!("doc{i}")).collect(),
                doc_labels: vec![String::new(); n],
                doc_freqs,
                total_freqs,
            }
        };

        // The coupling topology the kernel actually fits on. Parent coupling uses the reply tree
        // itself; root coupling uses the reparented thread-root star; blend keeps the real tree
        // (it couples to parent AND root via a BlendConfig, not a reparented topology). Everything
        // downstream (field fit, kappa, persistence, transform) operates on `fit_parents`, so
        // storing the right topology keeps them consistent with how the model was fit.
        let coupling_par = if slf.coupling == "root" {
            root_star_parents(&par)
        } else {
            par.clone()
        };
        // Blend configuration: the real tree's thread roots + any pinned weights. Warn when the tree
        // lacks depth-3 structure, where a node's parent equals its root and α vs β is unidentified.
        let blend_cfg = if slf.coupling == "blend" {
            let roots = thread_roots(&par); // compute once, reused for the check and BlendConfig
            let n_deep = (0..n)
                .filter(|&c| par[c] >= 0 && roots[c] != par[c] as usize)
                .count();
            if slf.blend_alpha_fixed.is_none() && slf.blend_beta_fixed.is_none() && n_deep < 2 {
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    "coupling=\"blend\": the reply tree has almost no depth-3+ structure (a node's \
                     parent is its thread root), so the parent weight α and root weight β are not \
                     separately identified and the split is arbitrary. Pin blend_alpha/blend_beta, \
                     or use coupling=\"parent\"/\"root\".",
                    1,
                )?;
            }
            Some(crate::thread_tm::BlendConfig {
                root: roots,
                fixed_alpha: slf.blend_alpha_fixed,
                fixed_beta: slf.blend_beta_fixed,
            })
        } else {
            None
        };

        // Retain the coupling topology + groups for persistence() before the move-closure consumes them.
        slf.fit_parents = coupling_par.clone();
        slf.fit_groups = groups.clone();

        // Content covariate (issue #841): a second covariate that shapes topic WORDS by level via
        // the SAGE κ channel, orthogonal to the tree prevalence prior. Either "depth" (auto-binned
        // position-in-thread from the REAL reply tree `par`, not the coupling topology) or an explicit
        // per-document categorical (int/str/float labels, auto-encoded like `covariates`).
        let content_l1 = match content_prior.as_str() {
            "l2" => 0.0,
            "l1" => 1.0 / content_prior_var,
            other => {
                return Err(PyValueError::new_err(format!(
                    "content_prior must be \"l2\" or \"l1\", got {other:?}"
                )))
            }
        };
        if !(content_prior_var.is_finite() && content_prior_var > 0.0) {
            return Err(PyValueError::new_err(
                "content_prior_var must be finite and > 0",
            ));
        }
        if !(content_smooth.is_finite() && content_smooth >= 0.0) {
            return Err(PyValueError::new_err(
                "content_smooth must be finite and >= 0 (0 disables adjacent-level smoothing)",
            ));
        }
        let mut content_depth_edges_used: Vec<usize> = Vec::new();
        let (content_groups, n_content, content_group_names): (
            Option<Vec<usize>>,
            usize,
            Vec<String>,
        ) = match &content {
            None => (None, 1, Vec::new()),
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    if s != "depth" {
                        return Err(PyValueError::new_err(format!(
                            "content must be \"depth\" or a per-document label sequence, got {s:?}"
                        )));
                    }
                    let edges = depth_bins.clone().unwrap_or_else(|| vec![0, 1, 3]);
                    if edges.is_empty() || edges[0] != 0 || edges.windows(2).any(|w| w[0] >= w[1]) {
                        return Err(PyValueError::new_err(
                            "depth_bins must be strictly ascending lower edges starting at 0, \
                                 e.g. [0, 1, 3] for root/shallow/deep",
                        ));
                    }
                    content_depth_edges_used = edges.clone();
                    let (g, ncg, names) = depth_content_groups(&par, &edges);
                    let names = match &content_names {
                        Some(cn) if cn.len() == ncg => cn.clone(),
                        Some(cn) => {
                            return Err(PyValueError::new_err(format!(
                                "content_names has {} names but depth_bins define {ncg} levels",
                                cn.len()
                            )))
                        }
                        None => names,
                    };
                    (Some(g), ncg, names)
                } else {
                    let (g, derived) = coerce_fit_covariates(obj)?;
                    if g.len() != n {
                        return Err(PyValueError::new_err(format!(
                            "content has {} entries but there are {n} documents",
                            g.len()
                        )));
                    }
                    let ncg = g.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                    let names = content_names
                        .clone()
                        .or(derived)
                        .unwrap_or_else(|| (0..ncg).map(|i| format!("content{i}")).collect());
                    if names.len() != ncg {
                        return Err(PyValueError::new_err(format!(
                            "content_names has {} names but the content covariate has {ncg} levels",
                            names.len()
                        )));
                    }
                    (Some(g), ncg, names)
                }
            }
        };

        let k = slf.num_topics;
        let v = vocab.len();
        let iters = slf.em_iters;
        let seed = slf.seed;

        // Build the seed/anchor supervision config (issue #854) from the fit-time vocabulary and
        // covariate groups. seed_words -> β pseudocounts on the matched (topic, word) cells;
        // prevalence_anchor -> per-group η targets (additive-log-ratio of the supplied mix).
        let (seed_cfg, seed_matches_out, topic_names_out) = {
            // Seeding the topic-word channel is unsupported alongside a content covariate (SAGE):
            // with content the β M-step is replaced by per-level sparse deviations, so the seed
            // pseudocounts would only reach the initializer, not the fit. Reject the combination.
            if seed_words.is_some() && content.is_some() {
                return Err(PyValueError::new_err(
                    "seed_words is not supported together with a content covariate; fit the seeds \
                     without content, or drop seed_words (prevalence_anchor works with content).",
                ));
            }
            let mut beta_pseudo: Vec<(usize, usize, f64)> = Vec::new();
            let mut seed_matches: Vec<Vec<String>> = Vec::new();
            // Topic names default to positional `topic_i`, overridden per slot when seed_words is
            // string-keyed (a {name: [words]} dict names the seeded topics, as SeededLDA/KeyATM do).
            // Always length-K so `topic_names` is defined for every fit, seeded or not.
            let mut topic_names: Vec<String> = (0..k).map(|i| format!("topic_{i}")).collect();
            // Warn on an orphaned supervision knob: a strength set without its companion collection
            // is a silent no-op, and usually a forgotten arg. Two independent axes — word seeding
            // (weight/seed_strength ↔ seed_words) and prevalence anchoring (prevalence_strength ↔
            // prevalence_anchor). `!= default` via abs-diff to dodge the float_cmp lint.
            if seed_words.is_none() && ((weight - 0.01).abs() > 1e-12 || seed_strength.is_some()) {
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    "weight/seed_strength was set but seed_words is None; the word-seeding knobs \
                     are ignored (pass seed_words to seed topics).",
                    1,
                )?;
            }
            if prevalence_anchor.is_none() && (prevalence_strength - 0.5).abs() > 1e-12 {
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    "prevalence_strength was set but prevalence_anchor is None; prevalence \
                     anchoring is ignored (pass prevalence_anchor to steer prevalence).",
                    1,
                )?;
            }
            if let Some(obj) = &seed_words {
                // Validate the seeding knobs only on the path that uses them. Guard finiteness and
                // sign so a NaN/negative pseudocount cannot silently poison β (a negative pseudocount
                // makes a normalized β entry negative; f64::clamp would let a NaN through untouched).
                if seed_prior != "frequency" && seed_prior != "uniform" {
                    return Err(PyValueError::new_err(
                        "seed_prior must be \"frequency\" or \"uniform\"",
                    ));
                }
                if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                    return Err(PyValueError::new_err(
                        "weight must be finite and in [0, 1] (as in SeededLDA / the seededlda package)",
                    ));
                }
                if let Some(s) = seed_strength {
                    if !s.is_finite() || s < 0.0 {
                        return Err(PyValueError::new_err(
                            "seed_strength must be finite and >= 0",
                        ));
                    }
                }
                let mode = crate::python::SeedMatch::parse(&seed_match)?;
                // seed_words keys are EITHER int topic indices (explicit positional slots) OR string
                // topic names (which name the seeded topics and take the leading slots 0..G in
                // insertion order, as SeededLDA/KeyATM do). Keys must be all-int or all-string, not
                // mixed. Values resolve against the vocabulary with the shared matcher (fixed/glob/
                // regex, dedup within a topic) that SeededLDA/CorEx use. Dict iteration is
                // insertion-ordered (Python 3.7+), so string-keyed slots are assigned deterministically.
                let dict = obj.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "seed_words must be a dict mapping a topic (an int index or a string name) \
                         to a list of keyword strings, e.g. {0: [\"orbit\", \"planet\"]} or \
                         {\"space\": [\"orbit\", \"planet\"]}",
                    )
                })?;
                let mut per_topic: Vec<Vec<String>> = vec![Vec::new(); k];
                let (mut saw_int, mut saw_str, mut str_slot) = (false, false, 0usize);
                for (key, val) in dict.iter() {
                    let words: Vec<String> = val.extract().map_err(|_| {
                        PyValueError::new_err("seed_words values must be a list of keyword strings")
                    })?;
                    // A Python bool is a subclass of int, so it would slip into the usize branch and
                    // be read as a topic index (True->1, False->0). Reject it explicitly.
                    if key.is_instance_of::<PyBool>() {
                        return Err(PyValueError::new_err(
                            "seed_words keys must be an int topic index or a string topic name, \
                             not a bool",
                        ));
                    }
                    if let Ok(name) = key.extract::<String>() {
                        saw_str = true;
                        if str_slot >= k {
                            return Err(PyValueError::new_err(format!(
                                "seed_words names {} topics but num_topics={k}; pass at most K names",
                                str_slot + 1
                            )));
                        }
                        // A string key NAMES a topic and takes the next leading slot, unlike an int
                        // key that selects a slot. Warn when the name looks like a valid index, since
                        // {"2": ...} and {2: ...} then diverge silently (the string seeds the next
                        // leading slot, named "2", not index 2).
                        if let Ok(as_idx) = name.parse::<usize>() {
                            if as_idx < k {
                                PyErr::warn_bound(
                                    py,
                                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                                    &format!(
                                        "seed_words key {name:?} is a string, so it NAMES a topic and \
                                         takes the next leading slot (slot {str_slot}), not index \
                                         {as_idx}; pass the int {as_idx} for a positional slot."
                                    ),
                                    1,
                                )?;
                            }
                        }
                        per_topic[str_slot] = words;
                        topic_names[str_slot] = name;
                        str_slot += 1;
                    } else if let Ok(t) = key.extract::<usize>() {
                        saw_int = true;
                        if t >= k {
                            return Err(PyValueError::new_err(format!(
                                "seed_words topic index {t} is out of range for num_topics={k}"
                            )));
                        }
                        per_topic[t] = words;
                    } else {
                        return Err(PyValueError::new_err(
                            "seed_words keys must be an int topic index or a string topic name",
                        ));
                    }
                }
                if saw_int && saw_str {
                    return Err(PyValueError::new_err(
                        "seed_words keys must be all int topic indices or all string topic names, \
                         not a mix",
                    ));
                }
                let matched =
                    crate::python::seed_word_ids(&per_topic, &vocab, k, mode, case_insensitive)?;
                // Record the resolved vocabulary words per topic so a user can audit what glob/regex
                // seeding matched (exposed as `seed_matches`, issue #856).
                seed_matches = matched
                    .iter()
                    .map(|ids| ids.iter().map(|&id| vocab[id].clone()).collect())
                    .collect();
                let mut n_seeded_words = 0usize;
                for (t, ids) in matched.iter().enumerate() {
                    for &id in ids {
                        // Pseudocount per matched seed word. `seed_strength` (if given) overrides the
                        // scheme with a flat RAW count (unscaled). Otherwise `weight` matches
                        // SeededLDA's `weight`: "frequency" = corpus_count(word) * weight * 100
                        // (scale-robust, so a common seed word is trusted more), "uniform" = a flat
                        // weight * 100 per word; the two schemes relate as frequency =
                        // uniform * corpus_count(word). The *100 is SeededLDA's unit convention that
                        // lets `weight` be a tidy [0, 1] fraction while landing at a count-magnitude
                        // pseudocount (weight=0.01 default => 1x corpus count).
                        let pc = match seed_strength {
                            Some(s) => s,
                            None if seed_prior == "frequency" => {
                                counts[vocab[id].as_str()] as f64 * weight * 100.0
                            }
                            None => weight * 100.0,
                        };
                        beta_pseudo.push((t, id, pc));
                        n_seeded_words += 1;
                    }
                }
                if n_seeded_words == 0 {
                    PyErr::warn_bound(
                        py,
                        &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                        "no seed words matched the fitted vocabulary (all dropped by min_count or \
                         absent); the fit is unseeded.",
                        1,
                    )?;
                }
            }
            let mut anchor_target: Vec<(usize, Vec<f64>, f64)> = Vec::new();
            if let Some(obj) = &prevalence_anchor {
                if !prevalence_strength.is_finite() || !(0.0..=1.0).contains(&prevalence_strength) {
                    return Err(PyValueError::new_err(
                        "prevalence_strength must be finite and in [0, 1]",
                    ));
                }
                let dict = obj.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err(
                        "prevalence_anchor must be a dict mapping a group (an int index or a string \
                         label) to a length-K topic mix",
                    )
                })?;
                // A group key may be the encoded integer index OR the string covariate label; resolve
                // labels through the fitted group names so callers can use the same labels they passed
                // to covariates= (issue #856).
                let label_to_idx: HashMap<&str, usize> = group_names
                    .iter()
                    .enumerate()
                    .map(|(i, nm)| (nm.as_str(), i))
                    .collect();
                let s = prevalence_strength;
                for (key, val) in dict.iter() {
                    let g: usize = if let Ok(lbl) = key.extract::<String>() {
                        *label_to_idx.get(lbl.as_str()).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "prevalence_anchor label {lbl:?} is not one of the covariate groups \
                                 {group_names:?}"
                            ))
                        })?
                    } else if let Ok(i) = key.extract::<usize>() {
                        i
                    } else {
                        return Err(PyValueError::new_err(
                            "prevalence_anchor keys must be an int group index or a string group label",
                        ));
                    };
                    if g >= num_groups {
                        return Err(PyValueError::new_err(format!(
                            "prevalence_anchor group index {g} is out of range for {num_groups} group(s)"
                        )));
                    }
                    let mix: Vec<f64> = val.extract().map_err(|_| {
                        PyValueError::new_err(
                            "prevalence_anchor values must be a length-K list of numbers",
                        )
                    })?;
                    if mix.len() != k {
                        return Err(PyValueError::new_err(format!(
                            "prevalence_anchor[{g}] has length {}, expected num_topics={k}",
                            mix.len()
                        )));
                    }
                    if mix.iter().any(|&p| !p.is_finite() || p < 0.0) {
                        return Err(PyValueError::new_err(format!(
                            "prevalence_anchor[{g}] must be a non-negative topic mix (got a negative \
                             or non-finite entry); it need not sum to 1"
                        )));
                    }
                    // additive-log-ratio with the last topic as reference (η is K-1 dimensional).
                    let floor = 1e-9;
                    let last = mix[k - 1].max(floor);
                    let eta: Vec<f64> = (0..k - 1)
                        .map(|i| (mix[i].max(floor) / last).ln())
                        .collect();
                    anchor_target.push((g, eta, s));
                }
            }
            let cfg = if beta_pseudo.is_empty() && anchor_target.is_empty() {
                None
            } else {
                Some(crate::thread_tm::SeedConfig {
                    beta_pseudo,
                    anchor_target,
                })
            };
            (cfg, seed_matches, topic_names)
        };

        let m = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let content_cfg = content_groups
                .as_ref()
                .map(|g| crate::thread_tm::ContentConfig {
                    groups: g,
                    num_groups: n_content,
                    prior_var: content_prior_var,
                    l1: content_l1,
                    smooth: content_smooth,
                });
            crate::thread_tm::fit_thread_tm(
                &docs_id,
                &coupling_par,
                &groups,
                num_groups,
                k,
                v,
                iters,
                1e-6,
                false, // kappa_ci computed lazily by the getter, not in fit
                blend_cfg.as_ref(),
                content_cfg.as_ref(),
                seed_cfg.as_ref(),
                |_, _, _| true,
                &mut rng,
            )
        });

        let dt = m.doc_topic();
        let gp = m.group_prevalence();
        slf.vocab = vocab;
        slf.group_names = group_names;
        slf.topic_names = topic_names_out;
        slf.seed_matches = seed_matches_out;
        slf.doc_eta = m.lambda.clone();
        slf.doc_topic_var = m.doc_topic_var.clone();
        slf.beta = m.beta;
        slf.doc_topic = dt;
        slf.group_prevalence = gp;
        slf.prevalence_se = m.anchor_se;
        slf.kappa = m.kappa;
        slf.kappa_ci = m.kappa_ci;
        slf.blend_alpha = m.blend_alpha;
        slf.blend_beta = m.blend_beta;
        slf.blend_alpha_se = m.blend_alpha_se;
        slf.blend_beta_se = m.blend_beta_se;
        slf.blend_anchor_se = m.blend_anchor_se;
        slf.sigma2 = m.sigma2;
        slf.p0 = m.p0;
        slf.sigma_root = m.sigma_root;
        slf.sigma_edge = m.sigma_edge;
        slf.bound_history = m.bound_history;
        slf.converged = m.converged;
        // Content covariate outputs (empty/1 when no content covariate was fit).
        slf.num_content_groups = m.num_content_groups;
        slf.content_names = content_group_names;
        slf.content_depth_edges = content_depth_edges_used;
        slf.content_beta = m.content_beta.unwrap_or_default();
        if let Some(ck) = m.content_kappa {
            slf.content_kappa_m = ck.m;
            slf.content_kappa_topic = ck.kappa_topic;
            slf.content_kappa_cov = ck.kappa_cov;
            slf.content_kappa_interaction = ck.kappa_interaction;
        }
        slf.corpus = Some(corpus_snapshot);
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Infer topic proportions for a NEW reply forest, holding the fitted topics, reversion `kappa`,
    /// step/root variances, and per-group anchors fixed. `data` is a `topica.Corpus` or a list of
    /// token lists (mapped to the training vocabulary; out-of-vocabulary tokens are dropped, exactly
    /// as at fit). `parents[d]` is `d`'s parent **document index** in the new forest (`-1` for a
    /// thread root), in the SAME order as the documents; omit it to treat every document as a root
    /// (a plain logistic-normal inference against the group anchor, ignoring reply structure).
    /// `covariates` is the per-document group id; omit to anchor every document at the unweighted mean
    /// of the fitted group anchors (a neutral baseline, not the size-weighted marginal). Returns an
    /// N×K proportions matrix.
    ///
    /// The reply coupling is directed (a document's prior mean depends only on its parent's η), so a
    /// single topological pass is the structured mean-field fixed point, no iteration needed; on a
    /// tree of token-bearing nodes it reproduces the converged fit. Requires a model fit **with** a
    /// reply tree (the step/root variances are otherwise undefined).
    #[pyo3(signature = (data, parents=None, covariates=None, content=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        parents: Option<Vec<i64>>,
        covariates: Option<Bound<'py, PyAny>>,
        content: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        // Accept integer group ids, or string/categorical labels mapped through the fitted groups.
        let covariates: Option<Vec<i64>> = match &covariates {
            None => None,
            Some(obj) => Some(coerce_transform_covariates(obj, &self.group_names)?),
        };
        if content.is_some() && self.content_beta.is_empty() {
            return Err(PyValueError::new_err(
                "content= was passed to transform, but the model was not fit with a content \
                 covariate; refit with content= or drop it here",
            ));
        }
        if !self.sigma2.is_finite() || !self.p0.is_finite() {
            return Err(PyValueError::new_err(
                "transform needs a model fit with a reply tree; this model was fit with \
                 parents=None (the step/root variances are undefined). Refit with parents, or use \
                 CTM for tree-free inference.",
            ));
        }

        // Map new tokens to the training vocabulary (raw words, matching fit — NO lowercasing).
        let wid: HashMap<&str, u32> = self
            .vocab
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
                        .map(|&w| c.inner.id_to_word[w as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract::<Vec<Vec<String>>>().map_err(|_| {
                PyValueError::new_err(
                    "expected a Corpus or a list of token lists (list[list[str]])",
                )
            })?
        };
        let docs_id: Vec<Vec<u32>> = str_docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| wid.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        let n = docs_id.len();
        if n == 0 {
            return Err(PyValueError::new_err("data is empty"));
        }

        // parents: default to an all-root forest; validate length, range, and acyclicity otherwise.
        let par: Vec<i64> = match parents {
            None => vec![-1; n],
            Some(p) => {
                if p.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "parents has {} entries but there are {n} documents",
                        p.len()
                    )));
                }
                for (d, &pd) in p.iter().enumerate() {
                    if pd < -1 || pd >= n as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] = {pd} out of range; must be -1 or in [0, {n})"
                        )));
                    }
                    if pd == d as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] points at itself"
                        )));
                    }
                }
                for start in 0..n {
                    let (mut cur, mut steps) = (start as i64, 0usize);
                    while cur >= 0 {
                        cur = p[cur as usize];
                        steps += 1;
                        if steps > n {
                            return Err(PyValueError::new_err(format!(
                                "parents contains a cycle reachable from document {start}; the \
                                 reply tree must be acyclic"
                            )));
                        }
                    }
                }
                p
            }
        };

        // Reconstruct the η-space anchor per training group from the stored softmax prevalence
        // (exact inverse: anchor[g][k] = ln(prevalence[g][k] / prevalence[g][K-1])), as in kappa_ci.
        let km1 = self.num_topics - 1;
        let ng = self.group_names.len().max(1);
        let anchors: Vec<Vec<f64>> = self
            .group_prevalence
            .iter()
            .map(|gp| {
                let ref_p = gp[km1].max(1e-12);
                (0..km1).map(|k| (gp[k].max(1e-12) / ref_p).ln()).collect()
            })
            .collect();

        // groups: an explicit covariate, or a single synthetic anchor at the across-group mean.
        let (groups, anchor_rows) = match covariates {
            Some(g) => {
                if g.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "covariates has {} entries but there are {n} documents",
                        g.len()
                    )));
                }
                if let Some(&bad) = g.iter().find(|&&gid| gid < 0 || gid >= ng as i64) {
                    return Err(PyValueError::new_err(format!(
                        "covariates has group id {bad}, but the model was fit with {ng} group(s); \
                         valid ids are 0 through {}",
                        ng - 1
                    )));
                }
                (
                    g.iter().map(|&gid| gid as usize).collect::<Vec<usize>>(),
                    anchors,
                )
            }
            None => {
                let mut mean = vec![0.0f64; km1];
                for a in &anchors {
                    for (m, &v) in mean.iter_mut().zip(a) {
                        *m += v / ng as f64;
                    }
                }
                (vec![0usize; n], vec![mean])
            }
        };

        // Keep the raw new-forest parents for content="depth" re-binning (the coupling match below
        // consumes `par` into the coupling topology, which may be reparented for root/blend).
        let par_for_content = par.clone();

        // Couple the new forest the same way the model was fit: toward the immediate parent; (root)
        // toward each node's thread root via the reparented star; or (blend) toward both, with the
        // fitted weights passed through so transform_thread_tm builds α·parent + β·root + rest·anchor.
        let (coupling_par, blend) = if self.coupling == "root" {
            (root_star_parents(&par), None)
        } else if self.coupling == "blend" {
            (par, Some((self.blend_alpha, self.blend_beta)))
        } else {
            (par, None)
        };

        let kappa = self.kappa;
        // Full precisions for roots (Σ_root⁻¹) and edges (Σ_edge⁻¹). The isotropic diagonal fallback
        // covers a freshly-constructed/degenerate model with an empty covariance; note it does NOT
        // rescue genuinely old saves — the positional-bincode format cannot load a pre-covariance
        // file at all (see the serde-default note on the state struct), per the pre-v1.0 policy.
        let full_or_iso = |sig: &[f64], var: f64| -> Vec<f64> {
            if sig.len() == km1 * km1 {
                crate::linalg::spd_inverse_and_half_logdet(sig, km1).0
            } else {
                let mut s = vec![0.0f64; km1 * km1];
                for i in 0..km1 {
                    s[i * km1 + i] = 1.0 / var;
                }
                s
            }
        };
        let root_siginv = full_or_iso(&self.sigma_root, self.p0);
        let edge_siginv = full_or_iso(&self.sigma_edge, self.sigma2);
        let beta = self.beta.clone();
        // Content covariate for the new forest: "depth" re-bins the new tree with the fitted edges;
        // labels map through the fitted content levels. Validated against the fitted level count.
        let content_groups: Option<Vec<usize>> = match &content {
            None => None,
            Some(obj) => {
                let g: Vec<usize> = if let Ok(s) = obj.extract::<String>() {
                    if s != "depth" {
                        return Err(PyValueError::new_err(format!(
                            "content must be \"depth\" or a per-document label sequence, got {s:?}"
                        )));
                    }
                    if self.content_depth_edges.is_empty() {
                        return Err(PyValueError::new_err(
                            "the model was not fit with content=\"depth\"; pass the content labels \
                             it was fit with instead",
                        ));
                    }
                    depth_content_groups(&par_for_content, &self.content_depth_edges).0
                } else {
                    coerce_transform_covariates(obj, &self.content_names)?
                        .iter()
                        .map(|&x| x as usize)
                        .collect()
                };
                if let Some(&bad) = g.iter().find(|&&x| x >= self.num_content_groups) {
                    return Err(PyValueError::new_err(format!(
                        "content level {bad} is out of range; the model has {} level(s)",
                        self.num_content_groups
                    )));
                }
                Some(g)
            }
        };
        let content_beta = self.content_beta.clone();
        let theta = py.allow_threads(move || {
            let content_arg = content_groups
                .as_ref()
                .map(|g| (content_beta.as_slice(), g.as_slice()));
            crate::thread_tm::transform_thread_tm(
                &docs_id,
                &coupling_par,
                &groups,
                &beta,
                &anchor_rows,
                kappa,
                &edge_siginv,
                &root_siginv,
                blend,
                content_arg,
            )
        });
        Ok(vecs_to_arr2(&theta).to_pyarray_bound(py))
    }

    /// K×V topic-word probability matrix.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.beta).to_pyarray_bound(py))
    }

    /// Number of topics K.
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// D×(K-1) per-document variational mean η (softmax basis, reference topic K-1 fixed at 0):
    /// the latent topic coordinate. NOTE this is the **tree-coupled** posterior — the reply prior
    /// already pulls a child's η toward its parent's — so regressing these directly to measure
    /// persistence is CIRCULAR. Use `persistence()`, which refits an uncoupled pass for that.
    #[getter]
    fn doc_eta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_eta).to_pyarray_bound(py))
    }

    /// D×(K-1) per-document posterior variance ν of η (the Laplace curvature): the measurement-error
    /// variance of each `doc_eta` row. Exposed for diagnostics; `persistence()` uses it (with an
    /// uncoupled η) to correct its child-on-parent regression for attenuation.
    #[getter]
    fn doc_topic_var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_topic_var).to_pyarray_bound(py))
    }

    /// D×K document-topic proportions θ.
    ///
    /// This is the PLUG-IN `softmax([mean η, 0])`, which discards the per-document posterior
    /// variance `ν` (`doc_topic_var`). For a logistic-normal model `softmax(E[η]) != E[softmax(η)]`:
    /// the plug-in is an overconfident point estimate that ignores ν, sharpest exactly where ν is
    /// large (thin documents whose η is barely identified). A collapsed-Gibbs model's `doc_topic`
    /// (e.g. LDA) is instead a sample-averaged, hedged posterior mean, so to compare the two on the
    /// same estimator footing (e.g. for held-out token prediction) use `posterior_doc_topic`, the
    /// posterior-predictive `E[softmax(η)]` (issue #838).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_topic).to_pyarray_bound(py))
    }

    /// D×K posterior-predictive document-topic proportions `E[softmax([η, 0])]`, a Monte-Carlo
    /// average of `n_samples` draws of η from each document's Gaussian posterior
    /// `N(doc_eta, diag(doc_topic_var))`. Unlike the plug-in `doc_topic`, this integrates over the
    /// posterior variance ν, so it hedges thin, high-ν documents instead of committing to an
    /// overconfident point estimate. That puts it on the same estimator footing as a collapsed-Gibbs
    /// model's sample-averaged θ (e.g. LDA), which is what makes a held-out token comparison a model
    /// comparison rather than an estimator artifact (issue #838). Deterministic given `seed`. Note
    /// the draws use only the diagonal of ν (the stored marginal variances), not its full
    /// off-diagonal covariance.
    #[pyo3(signature = (*, n_samples=400, seed=13))]
    fn posterior_doc_topic<'py>(
        &self,
        py: Python<'py>,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        if n_samples == 0 {
            return Err(PyValueError::new_err("n_samples must be >= 1"));
        }
        let eta = &self.doc_eta;
        let var = &self.doc_topic_var;
        let km1 = if eta.is_empty() { 0 } else { eta[0].len() };
        let k = km1 + 1;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Box-Muller standard normal (no rand_distr dependency; matches the fit's convention).
        let gauss = |rng: &mut ChaCha8Rng| -> f64 {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut out = vec![vec![0.0f64; k]; eta.len()];
        let inv_s = 1.0 / n_samples as f64;
        for (di, mean) in eta.iter().enumerate() {
            let sd: Vec<f64> = var[di].iter().map(|v| v.max(0.0).sqrt()).collect();
            let acc = &mut out[di];
            for _ in 0..n_samples {
                // draw η_s = mean + sd ⊙ z, then softmax([η_s, 0]) with reference topic K-1 at 0.
                let mut logits = vec![0.0f64; k];
                let mut mx = 0.0f64; // reference logit is 0, so the running max starts at 0
                for i in 0..km1 {
                    let e = mean[i] + sd[i] * gauss(&mut rng);
                    logits[i] = e;
                    if e > mx {
                        mx = e;
                    }
                }
                let mut z = 0.0f64;
                for i in 0..k {
                    logits[i] = (logits[i] - mx).exp();
                    z += logits[i];
                }
                for i in 0..k {
                    acc[i] += logits[i] / z;
                }
            }
            for a in acc.iter_mut() {
                *a *= inv_s;
            }
        }
        Ok(vecs_to_arr2(&out).to_pyarray_bound(py))
    }

    /// G×K per-group baseline topic prevalence (softmax of the covariate anchor): the descriptive
    /// mean topic mix of each covariate group's documents. See `prevalence_se` for uncertainty.
    #[getter]
    fn group_prevalence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.group_prevalence).to_pyarray_bound(py))
    }

    /// G×(K-1) cluster-robust method-of-composition standard error of the group prevalence anchor
    /// (in η space): combines a between-THREAD sampling variance (clustered on the reply-tree root,
    /// so within-thread correlation does not deflate it) with the mean per-document posterior
    /// variance. `NaN` for a group with fewer than two threads (variance unidentified). To get an
    /// interval on the probability-scale `group_prevalence`, apply the delta method.
    #[getter]
    fn prevalence_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.prevalence_se).to_pyarray_bound(py))
    }

    /// Monte-Carlo primitive backing `topica.inspect.group_prevalence_ci`: returns a `G×K×4` array
    /// whose last axis is `[mean, ci_low, ci_high, sd]` on the probability scale. Per group it draws
    /// η from `N(anchor, diag(prevalence_se²))` (the anchor reconstructed from `group_prevalence` by
    /// inverse softmax), softmaxes each draw, and summarizes per topic; `ci` is the coverage. Uses
    /// only the diagonal (marginal) η SE. A group with fewer than two threads (NaN `prevalence_se`)
    /// yields NaN. The Python wrapper attaches labels and a `to_frame()` (issue #843).
    #[pyo3(signature = (*, ci=0.95, n_samples=2000, seed=13))]
    fn _group_prevalence_ci_mc<'py>(
        &self,
        py: Python<'py>,
        ci: f64,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        if !(0.0 < ci && ci < 1.0) {
            return Err(PyValueError::new_err("ci must be in (0, 1)"));
        }
        if n_samples == 0 {
            return Err(PyValueError::new_err("n_samples must be >= 1"));
        }
        let ng = self.group_prevalence.len();
        let k = if ng == 0 {
            0
        } else {
            self.group_prevalence[0].len()
        };
        let km1 = k.saturating_sub(1);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let gauss = |rng: &mut ChaCha8Rng| -> f64 {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let lo_q = 0.5 * (1.0 - ci);
        let hi_q = 0.5 * (1.0 + ci);
        let mut out = numpy::ndarray::Array3::<f64>::zeros((ng, k, 4));
        for g in 0..ng {
            let gp = &self.group_prevalence[g];
            let ref_p = gp[km1].max(1e-12);
            let anchor: Vec<f64> = (0..km1).map(|i| (gp[i].max(1e-12) / ref_p).ln()).collect();
            let se = &self.prevalence_se[g];
            if se.iter().any(|v| !v.is_finite()) {
                for kk in 0..k {
                    out[[g, kk, 0]] = gp[kk]; // mean is still the plug-in point estimate
                    out[[g, kk, 1]] = f64::NAN;
                    out[[g, kk, 2]] = f64::NAN;
                    out[[g, kk, 3]] = f64::NAN;
                }
                continue;
            }
            let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); k];
            for _ in 0..n_samples {
                let mut logits = vec![0.0f64; k];
                let mut mx = 0.0f64; // reference logit is 0
                for i in 0..km1 {
                    let e = anchor[i] + se[i] * gauss(&mut rng);
                    logits[i] = e;
                    if e > mx {
                        mx = e;
                    }
                }
                let mut z = 0.0f64;
                for i in 0..k {
                    logits[i] = (logits[i] - mx).exp();
                    z += logits[i];
                }
                for i in 0..k {
                    cols[i].push(logits[i] / z);
                }
            }
            for (kk, col) in cols.iter_mut().enumerate() {
                let mean = col.iter().sum::<f64>() / col.len() as f64;
                let var = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
                    / col.len().max(1) as f64;
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out[[g, kk, 0]] = gp[kk]; // report the exact plug-in mean as the point estimate
                out[[g, kk, 1]] = percentile_sorted(col, lo_q);
                out[[g, kk, 2]] = percentile_sorted(col, hi_q);
                out[[g, kk, 3]] = var.sqrt();
            }
        }
        Ok(out.to_pyarray_bound(py))
    }

    /// The covariate group labels (order matches `group_prevalence` rows).
    fn group_labels(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.group_names.clone())
    }

    /// The vocabulary (index order matches the `topic_word` columns).
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.vocab.clone())
    }

    /// Per-topic names (length K), in `topic_word` row order. Positional `topic_i` unless the fit
    /// used a string-keyed `seed_words` dict, whose keys name the seeded topics (issue #854).
    /// Assignable (a full-length list) to rename topics, as in SeededLDA/AnchorLDA.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }

    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        self.require_fitted()?;
        if names.len() != self.beta.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} topic names, got {}",
                self.beta.len(),
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    /// Which fitted-vocabulary words each seeded topic's patterns actually matched (issue #856), as
    /// `{topic_index: [words]}` over the seeded topics only. Lets you audit what `seed_words` with
    /// `seed_match="glob"`/`"regex"` resolved to (e.g. confirm `"planet*"` caught `planet`,
    /// `planets` and nothing unintended). Empty dict when the fit was unseeded.
    #[getter]
    fn seed_matches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        let d = PyDict::new_bound(py);
        for (t, words) in self.seed_matches.iter().enumerate() {
            if !words.is_empty() {
                d.set_item(t, words.clone())?;
            }
        }
        Ok(d)
    }

    /// The training `Corpus` the model retained (all documents in reply-tree index order, including
    /// any emptied by `min_count`). Lets `record_fit`/`coherence` recover it without re-passing the
    /// corpus; rows align to `doc_topic`.
    #[getter]
    fn corpus(&self) -> PyResult<Corpus> {
        self.require_fitted()?;
        let inner = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no training corpus retained; refit the model")
        })?;
        Ok(Corpus::from_inner(inner.clone()))
    }

    /// Top-`n` words per topic. With `topic=None` (default) returns a list of lists for all K
    /// topics; with an integer `topic`, returns that one topic's words. `weights=True` returns
    /// `(word, probability)` pairs instead of bare words.
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = vecs_to_arr2(&self.beta);
        topic_words_helper(py, &phi, &self.vocab, self.num_topics, n, topic, weights)
    }

    /// The content-covariate level labels (order matches the group axis of `topic_word_by_group`);
    /// empty unless the model was fit with `content=`. Also exposed as `groups` for the shared
    /// `topica.content` diagnostics.
    #[getter]
    fn content_labels(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.content_names.clone())
    }

    /// Alias of `content_labels` under the name the cross-model `topica.content` helpers key on.
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.content_names.clone())
    }

    /// K × G_content × V per-content-level topic-word distributions, `None` unless fit with
    /// `content=`. `topic_word_by_group[k, g]` is topic `k`'s word distribution among documents at
    /// content level `g` (e.g. depth bin), so `topic_word_by_group[:, deep] - topic_word_by_group[:,
    /// root]` shows how each topic's vocabulary shifts downstream. Same `(K, G, V)` layout STM's
    /// `topic_word_by_group` uses, so the `topica.content` diagnostics (`group_topic_word`,
    /// `topic_polarization`, `group_exclusivity`) work on it. The plain `topic_word` is the
    /// level-averaged marginal.
    #[getter]
    fn topic_word_by_group<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray3<f64>>>> {
        self.require_fitted()?;
        if self.content_beta.is_empty() {
            return Ok(None);
        }
        let g = self.content_beta.len();
        let k = self.num_topics;
        let v = self.vocab.len();
        let mut out = numpy::ndarray::Array3::<f64>::zeros((k, g, v));
        // content_beta is stored (G, K, V); expose the codebase-canonical (K, G, V).
        for (gi, level) in self.content_beta.iter().enumerate() {
            for (ti, row) in level.iter().enumerate() {
                for (wi, &p) in row.iter().enumerate() {
                    out[[ti, gi, wi]] = p;
                }
            }
        }
        Ok(Some(out.to_pyarray_bound(py)))
    }

    /// The level-averaged marginal topic-word matrix (K × V), i.e. `topic_word`; exposed under the
    /// name the `topica.content` `group_exclusivity` helper reads. `None` without a content covariate.
    #[getter]
    fn topic_word_marginal<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.require_fitted()?;
        if self.content_beta.is_empty() {
            return Ok(None);
        }
        Ok(Some(vecs_to_arr2(&self.beta).to_pyarray_bound(py)))
    }

    /// The fitted SAGE content deviations `κ` as a dict of numpy arrays (`None` unless fit with
    /// `content=`), matching STM's `content_kappa`: `"m"` (V, the log word-frequency background),
    /// `"kappa_topic"` (K×V), `"kappa_cov"` (G×V, the per-level deviation), and `"kappa_interaction"`
    /// (K×G×V, the topic×level deviation). A deviation near zero means that level does not shift the
    /// topic's words from the marginal.
    #[getter]
    fn content_kappa<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.require_fitted()?;
        if self.content_kappa_topic.is_empty() {
            return Ok(None);
        }
        let k = self.num_topics;
        let g = self.num_content_groups;
        let v = self.vocab.len();
        // Reshape the flat (K*G, V) interaction (indexed topic*G + level) to STM's (K, G, V).
        let mut inter = numpy::ndarray::Array3::<f64>::zeros((k, g, v));
        for t in 0..k {
            for gi in 0..g {
                let row = &self.content_kappa_interaction[t * g + gi];
                for (wi, &x) in row.iter().enumerate() {
                    inter[[t, gi, wi]] = x;
                }
            }
        }
        let d = PyDict::new_bound(py);
        d.set_item("m", PyArray1::from_slice_bound(py, &self.content_kappa_m))?;
        d.set_item(
            "kappa_topic",
            vecs_to_arr2(&self.content_kappa_topic).to_pyarray_bound(py),
        )?;
        d.set_item(
            "kappa_cov",
            vecs_to_arr2(&self.content_kappa_cov).to_pyarray_bound(py),
        )?;
        d.set_item("kappa_interaction", inter.to_pyarray_bound(py))?;
        Ok(Some(d))
    }

    /// Top-`n` words per content level for one topic: `{level_label: [word, ...]}`, the "how does
    /// topic `topic`'s language shift by level" readout (e.g. root vs deep). Requires a content fit.
    #[pyo3(signature = (topic, n=10))]
    fn content_top_words<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        if self.content_beta.is_empty() {
            return Err(PyValueError::new_err(
                "no content covariate was fit; pass content= to fit for per-level words",
            ));
        }
        if topic >= self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic {topic} out of range 0..{}",
                self.num_topics
            )));
        }
        let out = PyDict::new_bound(py);
        for (gi, level) in self.content_beta.iter().enumerate() {
            let row = &level[topic];
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let words: Vec<String> = idx.iter().take(n).map(|&w| self.vocab[w].clone()).collect();
            let label = self
                .content_names
                .get(gi)
                .cloned()
                .unwrap_or_else(|| format!("level{gi}"));
            out.set_item(label, words)?;
        }
        Ok(out)
    }

    /// The words that most separate one topic's language between two content levels: the top-`n`
    /// `(word, log_ratio)` pairs by descending `ln(β[level_a] / β[level_b])` for `topic` (STM's
    /// `word_contrast`, on content levels). `level_a`/`level_b` are level indices or labels (e.g.
    /// `"deep"`, `"root"`). Positive log-ratio = more characteristic of `level_a`. The natural
    /// threaded readout is `content_word_contrast(k, "deep", "root")`. Requires a content fit.
    #[pyo3(signature = (topic, level_a, level_b, n=10))]
    fn content_word_contrast<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        level_a: &Bound<'py, PyAny>,
        level_b: &Bound<'py, PyAny>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if self.content_beta.is_empty() {
            return Err(PyValueError::new_err(
                "no content covariate was fit; pass content= to fit for the level contrast",
            ));
        }
        if topic >= self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic {topic} out of range 0..{}",
                self.num_topics
            )));
        }
        let la = self.resolve_content_level(level_a)?;
        let lb = self.resolve_content_level(level_b)?;
        let a = &self.content_beta[la][topic];
        let b = &self.content_beta[lb][topic];
        let ratio: Vec<f64> = (0..self.vocab.len())
            .map(|v| (a[v].max(1e-300) / b[v].max(1e-300)).ln())
            .collect();
        let mut idx: Vec<usize> = (0..self.vocab.len()).collect();
        idx.sort_by(|&x, &y| f64::total_cmp(&ratio[y], &ratio[x]));
        let items: Vec<Bound<'py, pyo3::types::PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| {
                pyo3::types::PyTuple::new_bound(
                    py,
                    &[self.vocab[v].clone().into_py(py), ratio[v].into_py(py)],
                )
            })
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Topic coherence (one score per topic). `coherence_type` is `"u_mass"` (default, uses the
    /// training corpus) or a windowed measure (`"c_v"`, `"c_uci"`, `"c_npmi"`); `texts` supplies
    /// an alternative reference corpus for the windowed measures. Higher is more coherent.
    #[pyo3(signature = (n=TopN(10), *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: TopN,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let n = n.0;
        let phi = vecs_to_arr2(&self.beta);
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

    /// Save the fitted model to `path`. Reload with `ThreadTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(
            path,
            MODEL_TAG_THREADTM,
            &ThreadTmState {
                num_topics: self.num_topics,
                em_iters: self.em_iters,
                seed: self.seed,
                coupling: self.coupling.clone(),
                blend_alpha_fixed: self.blend_alpha_fixed,
                blend_beta_fixed: self.blend_beta_fixed,
                blend_alpha: self.blend_alpha,
                blend_beta: self.blend_beta,
                blend_alpha_se: self.blend_alpha_se,
                blend_beta_se: self.blend_beta_se,
                blend_anchor_se: self.blend_anchor_se,
                fitted: self.fitted,
                vocab: self.vocab.clone(),
                group_names: self.group_names.clone(),
                topic_names: self.topic_names.clone(),
                seed_matches: self.seed_matches.clone(),
                beta: self.beta.clone(),
                doc_topic: self.doc_topic.clone(),
                doc_eta: self.doc_eta.clone(),
                doc_topic_var: self.doc_topic_var.clone(),
                group_prevalence: self.group_prevalence.clone(),
                prevalence_se: self.prevalence_se.clone(),
                kappa: self.kappa,
                kappa_ci: self.kappa_ci,
                sigma2: self.sigma2,
                p0: self.p0,
                sigma_root: self.sigma_root.clone(),
                sigma_edge: self.sigma_edge.clone(),
                bound_history: self.bound_history.clone(),
                converged: self.converged,
                content_beta: self.content_beta.clone(),
                content_kappa_m: self.content_kappa_m.clone(),
                content_kappa_topic: self.content_kappa_topic.clone(),
                content_kappa_cov: self.content_kappa_cov.clone(),
                content_kappa_interaction: self.content_kappa_interaction.clone(),
                content_names: self.content_names.clone(),
                num_content_groups: self.num_content_groups,
                content_depth_edges: self.content_depth_edges.clone(),
                fit_parents: self.fit_parents.clone(),
                fit_groups: self.fit_groups.clone(),
                corpus: self.corpus.clone(),
            },
        )
    }

    /// Load a model saved with `save`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let mut s: ThreadTmState = read_state(path, MODEL_TAG_THREADTM)?;
        // Preserve the "topic_names is length K when fitted" invariant for a state that predates the
        // field (serde default -> empty): backfill positional names from the topic-word row count.
        if s.fitted && s.topic_names.is_empty() {
            s.topic_names = (0..s.beta.len()).map(|i| format!("topic_{i}")).collect();
        }
        Ok(ThreadTM {
            num_topics: s.num_topics,
            em_iters: s.em_iters,
            seed: s.seed,
            coupling: s.coupling,
            blend_alpha_fixed: s.blend_alpha_fixed,
            blend_beta_fixed: s.blend_beta_fixed,
            blend_alpha: s.blend_alpha,
            blend_beta: s.blend_beta,
            blend_alpha_se: s.blend_alpha_se,
            blend_beta_se: s.blend_beta_se,
            blend_anchor_se: s.blend_anchor_se,
            fitted: s.fitted,
            vocab: s.vocab,
            group_names: s.group_names,
            topic_names: s.topic_names,
            seed_matches: s.seed_matches,
            beta: s.beta,
            doc_topic: s.doc_topic,
            doc_eta: s.doc_eta,
            doc_topic_var: s.doc_topic_var,
            group_prevalence: s.group_prevalence,
            prevalence_se: s.prevalence_se,
            kappa: s.kappa,
            kappa_ci: s.kappa_ci,
            sigma2: s.sigma2,
            p0: s.p0,
            sigma_root: s.sigma_root,
            sigma_edge: s.sigma_edge,
            bound_history: s.bound_history,
            converged: s.converged,
            content_beta: s.content_beta,
            content_kappa_m: s.content_kappa_m,
            content_kappa_topic: s.content_kappa_topic,
            content_kappa_cov: s.content_kappa_cov,
            content_kappa_interaction: s.content_kappa_interaction,
            content_names: s.content_names,
            num_content_groups: s.num_content_groups,
            content_depth_edges: s.content_depth_edges,
            fit_parents: s.fit_parents,
            fit_groups: s.fit_groups,
            corpus: s.corpus,
        })
    }

    /// Reduced-form reply **persistence** — the honest, identifiable replacement for the
    /// boundary-prone `kappa`. The ML `kappa` collapses to the σ² floor on real corpora; this instead
    /// fits an internal NO-TREE pass (plain logistic-normal η, so a parent and child are estimated
    /// **independently** and the estimate is not circular), then regresses each reply's η on its
    /// coupling neighbor's η (centered on the covariate-group mean), pooled across topics with a
    /// thread-clustered bootstrap. The coupling neighbor is the immediate parent under the default
    /// `coupling="parent"` and the thread ROOT under `coupling="root"`, so persistence reads as
    /// child-tracks-parent or child-tracks-root respectively. Returns a dict:
    ///   `observed_persistence` — the raw slope `a` (how much a reply's topic mix tracks its
    ///     parent's); identified whenever parents vary (NaN on a degenerate corpus where every
    ///     eligible parent equals its group anchor). `observed_ci` is its 95% bootstrap interval,
    ///     or `(NaN, NaN)` when too many resamples are unidentifiable for an honest interval.
    ///   `reliability` — `Var(η) / (Var(η) + mean ν)` of the parent, the signal share and the
    ///     identifiability gate: `<= 0` means the per-document η are mostly noise, so the structural
    ///     value cannot be recovered.
    ///   `structural_kappa` — `1 - a/reliability`, the measurement-error-corrected reversion, with
    ///     `structural_kappa_ci`; `NaN` (and NaN bounds) when `reliability <= 0`.
    #[pyo3(signature = (*, bootstrap=400))]
    fn persistence<'py>(&self, py: Python<'py>, bootstrap: usize) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        let corpus = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no training corpus retained; refit the model")
        })?;
        let docs_id = &corpus.docs;
        let n = docs_id.len();
        let n_edges = self.fit_parents.iter().filter(|&&p| p >= 0).count();
        if n_edges == 0 {
            return Err(PyValueError::new_err(
                "persistence needs a reply tree; this model was fit with parents=None",
            ));
        }
        let k = self.num_topics;
        let km1 = k - 1;
        let v = self.vocab.len();
        let ng = self.group_names.len().max(1);
        let groups = self.fit_groups.clone();
        let iters = self.em_iters;
        let seed = self.seed;
        // Uncoupled η: fit the same corpus with every document a root (no parent coupling).
        let all_roots = vec![-1i64; n];
        let docs_owned: Vec<Vec<u32>> = docs_id.clone();
        let m0 = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            crate::thread_tm::fit_thread_tm(
                &docs_owned,
                &all_roots,
                &groups,
                ng,
                k,
                v,
                iters,
                1e-6,
                false, // uncoupled pass for persistence(); no kappa_ci needed
                None,  // uncoupled: every doc a root, no blend
                None,  // persistence ignores the content channel (prevalence only)
                None,  // persistence ignores seed/anchor supervision
                |_, _, _| true,
                &mut rng,
            )
        });
        let eta = &m0.lambda;
        let nu = &m0.doc_topic_var;

        // group anchors = per-group mean of the uncoupled η (documents with tokens)
        let has_tok: Vec<bool> = docs_id.iter().map(|d| !d.is_empty()).collect();
        let mut anchor = vec![vec![0.0f64; km1]; ng];
        let mut gcnt = vec![0usize; ng];
        for i in 0..n {
            if !has_tok[i] {
                continue;
            }
            let g = self.fit_groups[i];
            gcnt[g] += 1;
            for kk in 0..km1 {
                anchor[g][kk] += eta[i][kk];
            }
        }
        for g in 0..ng {
            if gcnt[g] > 0 {
                for kk in 0..km1 {
                    anchor[g][kk] /= gcnt[g] as f64;
                }
            }
        }

        // thread root of each document
        let mut root = vec![0usize; n];
        for start in 0..n {
            let mut cur = start as i64;
            while self.fit_parents[cur as usize] >= 0 {
                cur = self.fit_parents[cur as usize];
            }
            root[start] = cur as usize;
        }
        // per-thread aggregates over reply edges: (Sxy, Sxx, Snu, count)
        let mut agg: HashMap<usize, [f64; 4]> = HashMap::new();
        for ch in 0..n {
            let p = self.fit_parents[ch];
            if p < 0 || !has_tok[ch] || !has_tok[p as usize] {
                continue;
            }
            let g = self.fit_groups[ch];
            let (pe, ce) = (&eta[p as usize], &eta[ch]);
            let pnu = &nu[p as usize];
            let e = agg.entry(root[ch]).or_insert([0.0; 4]);
            for kk in 0..km1 {
                let x = pe[kk] - anchor[g][kk];
                let y = ce[kk] - anchor[g][kk];
                e[0] += x * y;
                e[1] += x * x;
                e[2] += pnu[kk];
                e[3] += 1.0;
            }
        }
        // Sort by thread root so the order (hence float-sum order and bootstrap draws) is
        // deterministic — HashMap iteration order is randomized.
        let mut items: Vec<(usize, [f64; 4])> = agg.into_iter().collect();
        items.sort_by_key(|(root, _)| *root);
        let threads: Vec<[f64; 4]> = items.into_iter().map(|(_, v)| v).collect();
        if threads.len() < 2 {
            return Err(PyValueError::new_err(
                "not enough reply edges across threads to estimate persistence",
            ));
        }
        // point estimate from a set of thread aggregates. Guards the degenerate case where the
        // pooled parent variance is zero (all eligible parents equal their group anchor) — the
        // regression is then unidentified, so the slope/reliability are NaN, not 0/0.
        let est = |sel: &[[f64; 4]]| -> (f64, f64, f64) {
            let mut s = [0.0f64; 4];
            for a in sel {
                for j in 0..4 {
                    s[j] += a[j];
                }
            }
            if !s[1].is_finite() || s[1] <= 0.0 || s[3] <= 0.0 {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let a = s[0] / s[1]; // slope
            let var_x = s[1] / s[3];
            let mean_nu = s[2] / s[3];
            let rel = 1.0 - mean_nu / var_x;
            let a_corr = if rel > 0.0 { a / rel } else { f64::NAN };
            (a, rel, 1.0 - a_corr) // (observed a, reliability, structural kappa)
        };
        let (a_obs, rel, kappa_s) = est(&threads);

        // thread-clustered bootstrap
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (mut a_bs, mut k_bs): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let t = threads.len();
        let reps = bootstrap.max(1);
        for _ in 0..reps {
            let sel: Vec<[f64; 4]> = (0..t)
                .map(|_| threads[(rng.gen::<f64>() * t as f64) as usize % t])
                .collect();
            let (a, _, kap) = est(&sel);
            a_bs.push(a);
            k_bs.push(kap);
        }
        // Honest percentile CI: report it ONLY if >=90% of resamples are identifiable. Silently
        // dropping non-finite resamples would make the interval conditional on the identifiable
        // ones and look precise even when the parameter is badly non-identified (e.g. many
        // resamples hit reliability<=0), so below that threshold we return NaN bounds.
        let ci = |v: &[f64]| -> (f64, f64) {
            let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            if f.len() < 2 || (f.len() as f64) < 0.9 * reps as f64 {
                return (f64::NAN, f64::NAN);
            }
            f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let at = |q: f64| f[((q * (f.len() - 1) as f64).round() as usize).min(f.len() - 1)];
            (at(0.025), at(0.975))
        };
        let obs_ci = ci(&a_bs);
        let kap_ci = ci(&k_bs);

        let d = PyDict::new_bound(py);
        d.set_item("observed_persistence", a_obs)?;
        d.set_item("observed_ci", obs_ci)?;
        d.set_item("reliability", rel)?;
        d.set_item("structural_kappa", kappa_s)?;
        d.set_item("structural_kappa_ci", kap_ci)?;
        Ok(d)
    }

    /// 95% profile-likelihood CI for the reversion `κ` as a `(lower, upper)` tuple, re-optimizing
    /// (σ², p0) at each κ so the a↔σ² ridge is reflected; `(nan, nan)` when there are no reply
    /// edges or the field was not fit. Conditional on the topic fit; the point estimate is biased
    /// toward κ→0 (persistence) by topic-model shrinkage and the interval does not correct that.
    /// Computed lazily on access (it is the dominant per-fit cost and usually not needed —
    /// `persistence()` supersedes it), from the stored fit; expect ~1s per call. The profile uses
    /// the isotropic scalar (σ², p0) tree-field, an approximation to the full Σ_edge/Σ_root the fit
    /// actually estimates — the CI reflects the reversion ridge, not the anisotropy of the priors.
    #[getter]
    fn kappa_ci<'py>(&self, py: Python<'py>) -> PyResult<(f64, f64)> {
        self.require_fitted()?;
        let a = 1.0 - self.kappa;
        if !a.is_finite() {
            return Ok((f64::NAN, f64::NAN));
        }
        let corpus = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no training corpus retained; refit the model")
        })?;
        let has_tokens: Vec<bool> = corpus.docs.iter().map(|doc| !doc.is_empty()).collect();
        // Reconstruct the η-space group anchor from the softmax group prevalence (exact inverse:
        // anchor[g][k] = ln(prevalence[g][k] / prevalence[g][K-1])).
        let km1 = self.num_topics - 1;
        let anchor: Vec<Vec<f64>> = self
            .group_prevalence
            .iter()
            .map(|gp| {
                let ref_p = gp[km1].max(1e-12);
                (0..km1).map(|k| (gp[k].max(1e-12) / ref_p).ln()).collect()
            })
            .collect();
        let doc_eta = self.doc_eta.clone();
        let doc_var = self.doc_topic_var.clone();
        let parents = self.fit_parents.clone();
        let groups = self.fit_groups.clone();
        let (s2, p0) = (self.sigma2, self.p0);
        let (lo, hi) = py.allow_threads(move || {
            crate::thread_tm::kappa_profile_ci(
                &parents,
                &doc_eta,
                &anchor,
                &groups,
                &doc_var,
                &has_tokens,
                s2,
                p0,
                a,
            )
        });
        // Boundary peg: on a strongly-persistent corpus the profile is maximized at the reversion
        // clamp (κ→0), so the 95% region collapses to a single grid point at the floor. Reporting
        // that as a zero-width `(0.001, 0.001)` reads as false precision, so flag it: warn and
        // return a one-sided `(lo, nan)` (the upper bound is not identified), matching the spirit of
        // the `(nan, nan)` unidentified case (issue #830).
        if lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-6 && lo <= 0.0015 {
            PyErr::warn_bound(
                py,
                &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                "kappa_ci collapsed to a zero-width interval at the persistence floor \
                 (kappa is pegged at the reversion clamp): the profile likelihood is maximized at \
                 the boundary, so the interval is not identified. Returning (lower, nan) rather than \
                 a false-precision zero-width CI; read this as strong persistence, not a tight CI.",
                1,
            )?;
            return Ok((lo, f64::NAN));
        }
        Ok((lo, hi))
    }

    /// Reversion strength `κ = 1 - a` (0 = pure persistence / parent-copy, 1 = no memory). `NaN`
    /// under `coupling="blend"`, where the mix is described by `blend_alpha`/`blend_beta` instead.
    #[getter]
    fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Blend parent weight `α` (how much a node tracks its immediate parent), `NaN` unless the model
    /// was fit with `coupling="blend"`. A reliability-corrected (errors-in-variables) hard-EM
    /// estimate, so it is de-attenuated for the η measurement error but still conditional on the
    /// topic fit; read the held-out `reply_completion` delta for the model-vs-model comparison.
    #[getter]
    fn blend_alpha(&self) -> f64 {
        self.blend_alpha
    }

    /// Blend root weight `β` (how much a node tracks its thread root), `NaN` unless the model was fit
    /// with `coupling="blend"`. The anchor takes the remaining `1 - α - β`. Same estimator caveat as
    /// `blend_alpha`.
    #[getter]
    fn blend_beta(&self) -> f64 {
        self.blend_beta
    }

    /// Blend anchor weight `1 - α - β` (how much a node reverts to its covariate-group baseline
    /// rather than to its parent or root), `NaN` unless the model was fit with `coupling="blend"`.
    /// Together with `blend_alpha`/`blend_beta` these three shares are a hand-coding-free structural
    /// readout of how a discourse space is organised: parent-chained (α large), broadcast-around-the-
    /// root (β large), or context-free (anchor large).
    #[getter]
    fn blend_anchor(&self) -> f64 {
        1.0 - self.blend_alpha - self.blend_beta
    }

    /// Asymptotic cluster-robust (sandwich) standard error of `blend_alpha`, clustered on the thread
    /// root (issue #863). `NaN` unless `coupling="blend"` was fitted with at least two threads; `0`
    /// when `blend_alpha` was pinned in the constructor (it was fixed, not estimated); and `NaN` when
    /// the tree lacks depth-3 structure so the α-vs-β split is not identified (the fit warns) — read
    /// `blend_anchor_se` there, which stays finite because `α+β` is identified. Conditional on the
    /// topic fit; like any Wald SE it is not strictly valid at a boundary (a weight at 0 or on the
    /// `α+β=1` edge). For a paired parent-vs-root contrast on the held-out scale, use
    /// `evaluate.reply_completion` with the `"root"`/`"blend"` baselines instead.
    #[getter]
    fn blend_alpha_se(&self) -> f64 {
        self.blend_alpha_se
    }

    /// Asymptotic cluster-robust (sandwich) SE of `blend_beta`, clustered on the thread root; same
    /// conventions and caveats as `blend_alpha_se`.
    #[getter]
    fn blend_beta_se(&self) -> f64 {
        self.blend_beta_se
    }

    /// Cluster-robust (sandwich) SE of the anchor weight `1 - α - β` — the SE of the identified
    /// combined share `α + β`. `NaN` unless fitted with `coupling="blend"` and ≥2 threads; `0` when
    /// both weights are pinned. Unlike the individual weight SEs, this stays finite even when the
    /// α-vs-β split is unidentified (shallow trees), since the anchor share does not depend on the
    /// split.
    #[getter]
    fn blend_anchor_se(&self) -> f64 {
        self.blend_anchor_se
    }

    /// The estimated blend mix as a labelled dict: `{"alpha", "beta", "anchor", "alpha_se", "beta_se",
    /// "anchor_se"}` — the parent, root, and anchor shares of each node's prior mean, each with its
    /// thread-root-clustered SE (issue #863). All `NaN` unless the model was fit with
    /// `coupling="blend"`. A pinned weight reads its fixed value with SE `0`; on a shallow tree where
    /// the α-vs-β split is unidentified, `alpha_se`/`beta_se` are `NaN` but `anchor_se` is finite.
    /// This is the structural characterisation a discourse space earns from `coupling="blend"`:
    /// whether topics track the reply chain (α), the thread root (β), or neither (anchor).
    fn blend_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        let d = PyDict::new_bound(py);
        d.set_item("alpha", self.blend_alpha)?;
        d.set_item("beta", self.blend_beta)?;
        d.set_item("anchor", 1.0 - self.blend_alpha - self.blend_beta)?;
        d.set_item("alpha_se", self.blend_alpha_se)?;
        d.set_item("beta_se", self.blend_beta_se)?;
        d.set_item("anchor_se", self.blend_anchor_se)?;
        Ok(d)
    }

    /// Per-edge (OU step) variance: the mean marginal variance of the fitted full edge covariance
    /// Σ_edge (a scalar summary; the edge prior is the full Σ_edge, on the same footing as the root
    /// Σ_root). `NaN` when there are no reply edges / the field was not fit.
    #[getter]
    fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Root prior variance: the mean marginal variance of the fitted full root covariance Σ_root (a
    /// scalar summary; the base logistic-normal prior is the full Σ_root, not this isotropic value).
    /// Defined whenever the corpus has token-bearing roots (including the no-tree/CTM-equivalent
    /// case); `NaN` otherwise.
    #[getter]
    fn p0(&self) -> f64 {
        self.p0
    }

    /// The per-iteration variational-objective trace (sum of the per-document CTM bounds with the
    /// tree coupling plugged in as a fixed mean). This is a monitoring free energy, NOT a true
    /// ELBO for the joint tree model, so it is not guaranteed monotone.
    #[getter]
    fn bound_history(&self) -> Vec<f64> {
        self.bound_history.clone()
    }

    /// Whether variational EM converged (the monitoring bound's change fell below tolerance) before
    /// reaching the `em_iters` cap. `False` means the fit stopped at the cap and may benefit from
    /// more iterations; check it before trusting a fit rather than inferring convergence from
    /// `len(bound_history)`.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// Alias of `converged` under the name that says what the flag means: `True` only if the fit
    /// early-stopped on the convergence tolerance, `False` when the full `em_iters` ran.
    /// `topica.stop_reason` turns it into a plain-language summary (issue #755).
    #[getter]
    fn early_stopped(&self) -> PyResult<bool> {
        self.require_fitted()?;
        Ok(self.converged)
    }

    /// The fit trace as `(iteration, objective)` pairs — `bound_history` in the shape
    /// `topica.stop_reason` reads. The objective is the monitoring free energy (see `bound_history`),
    /// not a true ELBO.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.require_fitted()?;
        Ok(self
            .bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i, b))
            .collect())
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named to match
    /// `__init__` (issue #400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("em_iters", self.em_iters)?;
        d.set_item("seed", self.seed)?;
        d.set_item("coupling", self.coupling.clone())?;
        d.set_item("blend_alpha", self.blend_alpha_fixed)?;
        d.set_item("blend_beta", self.blend_beta_fixed)?;
        Ok(d)
    }

    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The reply coupling structure (`"parent"`, `"root"`, or `"blend"`).
    #[getter]
    fn coupling(&self) -> String {
        self.coupling.clone()
    }

    fn __repr__(&self) -> String {
        if self.fitted && self.coupling == "blend" {
            return format!(
                "ThreadTM(num_topics={}, coupling=\"blend\", fitted, alpha={:.3}±{:.3}, \
                 beta={:.3}±{:.3}, sigma2={:.3})",
                self.num_topics,
                self.blend_alpha,
                self.blend_alpha_se,
                self.blend_beta,
                self.blend_beta_se,
                self.sigma2
            );
        }
        if self.fitted {
            format!(
                "ThreadTM(num_topics={}, coupling={:?}, fitted, kappa={:.3}, sigma2={:.3})",
                self.num_topics, self.coupling, self.kappa, self.sigma2
            )
        } else {
            format!(
                "ThreadTM(num_topics={}, coupling={:?}, unfitted)",
                self.num_topics, self.coupling
            )
        }
    }
}
