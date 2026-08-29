//! Python binding for ReplyTM — a reply-threaded topic model (CTM/STM logistic-normal topics
//! with a reply-tree structured prior; see `crate::reply_tm`). Experimental tier: topica-original,
//! no published reference yet, so `fit` is gated behind `topica.enable_experimental()`.
//!
//! The class exposes the validated core — fit on a `Corpus` or token lists + a reply tree + an
//! optional categorical covariate, with topic/proportion/prevalence readouts (prevalence carries a
//! cluster-robust method-of-composition SE), coherence, and save/load. Reply persistence is best
//! read from `persistence()` — an identifiable reduced-form estimate (observed slope + reliability
//! gate + attenuation-corrected structural κ) — rather than the ML `kappa` getter, which collapses
//! to the σ² floor on real corpora. The covariate story lives entirely in
//! `group_prevalence`/`prevalence_se`; ReplyTM is outside the `effects` namespace. `transform`
//! infers proportions for new reply forests (a single topological pass, topics/field/anchors held
//! fixed); formula covariates remain a follow-up.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// ReplyTM: a reply-threaded topic model. A reply's topic prior is coupled to the comment it
/// answers (a persistence-smoothing prior along reply edges), reverting toward its covariate-group
/// baseline; `kappa` measures the reversion (on real corpora it is typically ~0, i.e. persistence-
/// dominated). Reduces to a plain logistic-normal topic model when the reply tree is flat.
/// `num_topics` is K; `em_iters` the variational-EM iteration cap; `seed` makes the fit deterministic.
/// `coupling` chooses the prior structure: `"parent"` (default; shrink toward the immediate parent,
/// the reply-chain prior), `"root"` (shrink toward the thread root, a broadcast / topic-around-the-
/// root prior), or `"blend"` (shrink toward both, `α·parent + β·root + (1-α-β)·anchor`, with the
/// weights estimated or pinned via `blend_alpha`/`blend_beta`).
#[pyclass(module = "topica")]
pub struct ReplyTM {
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
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
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
    // Training reply tree + covariate groups (document-aligned), retained so `persistence()` can
    // re-fit an uncoupled pass and regress child η on parent η.
    fit_parents: Vec<i64>,
    fit_groups: Vec<usize>,
    // Training corpus (all documents kept, including any emptied by min_count, so the reply-tree
    // node indices stay valid). Backs `coherence()` and save/load.
    corpus: Option<corpus::Corpus>,
}

/// Serialisable snapshot of a fitted ReplyTM (see `save`/`load`).
#[derive(serde::Serialize, serde::Deserialize)]
struct ReplyTmState {
    num_topics: usize,
    em_iters: usize,
    seed: u64,
    // Defaults a missing `coupling` to the original parent coupling. NOTE this is inert for the
    // current positional-bincode save format (a genuinely older ReplyTM save, which predates this
    // field, cannot round-trip and will fail to load) — it only migrates a self-describing format.
    // Acceptable under the pre-v1.0 save-compat policy: ReplyTM is new and experimental. Kept for
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
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
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

impl ReplyTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }
}

#[pymethods]
impl ReplyTM {
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
        Ok(ReplyTM {
            num_topics,
            em_iters,
            seed,
            coupling,
            blend_alpha_fixed: blend_alpha,
            blend_beta_fixed: blend_beta,
            blend_alpha: f64::NAN,
            blend_beta: f64::NAN,
            fitted: false,
            vocab: Vec::new(),
            group_names: Vec::new(),
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
            fit_parents: Vec::new(),
            fit_groups: Vec::new(),
            corpus: None,
        })
    }

    /// Fit ReplyTM. `data` is either a `topica.Corpus` or a list of token lists (already
    /// tokenized). `parents[d]` is `d`'s parent **document index** in the reply tree (`-1` for a
    /// thread root); build it in the SAME order as the documents. `covariates` is an optional
    /// per-document categorical group id in a DENSE range `0..num_groups` (the reversion anchor
    /// becomes that group's baseline prevalence); omit for a single global anchor.
    /// `covariate_names` names the groups for the readouts. `min_count` drops words rarer than it.
    /// Experimental: requires `topica.enable_experimental()`.
    #[pyo3(signature = (data, parents=None, covariates=None, covariate_names=None, *, min_count=1))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        parents: Option<Vec<i64>>,
        covariates: Option<Vec<usize>>,
        covariate_names: Option<Vec<String>>,
        min_count: usize,
    ) -> PyResult<Py<Self>> {
        require_experimental("ReplyTM")?;
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
                    "ReplyTM.fit called with parents=None: no reply tree, so the model reduces to \
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

        // covariate groups
        let (groups, num_groups, group_names) = match &covariates {
            None => (vec![0usize; n], 1usize, vec!["all".to_string()]),
            Some(g) => {
                if g.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "covariates has {} entries but there are {n} documents",
                        g.len()
                    )));
                }
                let ng = g.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                // warn on a non-dense covariate (a gap creates an empty phantom group)
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
                let names = covariate_names
                    .clone()
                    .unwrap_or_else(|| (0..ng).map(|i| format!("group{i}")).collect());
                if names.len() != ng {
                    return Err(PyValueError::new_err(format!(
                        "covariate_names has {} names but the covariate has {ng} groups",
                        names.len()
                    )));
                }
                (g.clone(), ng, names)
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
            Some(crate::reply_tm::BlendConfig {
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

        let k = slf.num_topics;
        let v = vocab.len();
        let iters = slf.em_iters;
        let seed = slf.seed;
        let m = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            crate::reply_tm::fit_reply_tm(
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
                |_, _, _| true,
                &mut rng,
            )
        });

        let dt = m.doc_topic();
        let gp = m.group_prevalence();
        slf.vocab = vocab;
        slf.group_names = group_names;
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
        slf.sigma2 = m.sigma2;
        slf.p0 = m.p0;
        slf.sigma_root = m.sigma_root;
        slf.sigma_edge = m.sigma_edge;
        slf.bound_history = m.bound_history;
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
    #[pyo3(signature = (data, parents=None, covariates=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        parents: Option<Vec<i64>>,
        covariates: Option<Vec<i64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
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

        // Couple the new forest the same way the model was fit: toward the immediate parent; (root)
        // toward each node's thread root via the reparented star; or (blend) toward both, with the
        // fitted weights passed through so transform_reply_tm builds α·parent + β·root + rest·anchor.
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
        let theta = py.allow_threads(move || {
            crate::reply_tm::transform_reply_tm(
                &docs_id,
                &coupling_par,
                &groups,
                &beta,
                &anchor_rows,
                kappa,
                &edge_siginv,
                &root_siginv,
                blend,
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
    /// variance `ν` (`doc_topic_var`). For a logistic-normal model `softmax(E[η]) != E[softmax(η)]`,
    /// and the plug-in is biased toward the center exactly where `ν` is large (thin documents). For
    /// held-out token prediction use `posterior_doc_topic`, the posterior-predictive `E[softmax(η)]`,
    /// which puts ReplyTM on the same footing as a Gibbs model's sample-averaged θ (issue #838).
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_topic).to_pyarray_bound(py))
    }

    /// D×K posterior-predictive document-topic proportions `E[softmax([η, 0])]`, a Monte-Carlo
    /// average of `n_samples` draws of η from each document's Gaussian posterior
    /// `N(doc_eta, diag(doc_topic_var))`. Unlike the plug-in `doc_topic`, this integrates over the
    /// posterior variance ν, so it does not collapse thin, high-ν documents toward a uniform mix.
    /// It is the θ to score held-out tokens with when comparing against a sample-averaged Gibbs
    /// model such as LDA (issue #838). Deterministic given `seed`. Note the draws use only the
    /// diagonal of ν (the stored marginal variances), not its full off-diagonal covariance.
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

    /// Save the fitted model to `path`. Reload with `ReplyTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(
            path,
            MODEL_TAG_REPLYTM,
            &ReplyTmState {
                num_topics: self.num_topics,
                em_iters: self.em_iters,
                seed: self.seed,
                coupling: self.coupling.clone(),
                blend_alpha_fixed: self.blend_alpha_fixed,
                blend_beta_fixed: self.blend_beta_fixed,
                blend_alpha: self.blend_alpha,
                blend_beta: self.blend_beta,
                fitted: self.fitted,
                vocab: self.vocab.clone(),
                group_names: self.group_names.clone(),
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
                fit_parents: self.fit_parents.clone(),
                fit_groups: self.fit_groups.clone(),
                corpus: self.corpus.clone(),
            },
        )
    }

    /// Load a model saved with `save`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: ReplyTmState = read_state(path, MODEL_TAG_REPLYTM)?;
        Ok(ReplyTM {
            num_topics: s.num_topics,
            em_iters: s.em_iters,
            seed: s.seed,
            coupling: s.coupling,
            blend_alpha_fixed: s.blend_alpha_fixed,
            blend_beta_fixed: s.blend_beta_fixed,
            blend_alpha: s.blend_alpha,
            blend_beta: s.blend_beta,
            fitted: s.fitted,
            vocab: s.vocab,
            group_names: s.group_names,
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
            crate::reply_tm::fit_reply_tm(
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
        Ok(py.allow_threads(move || {
            crate::reply_tm::kappa_profile_ci(
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
        }))
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
                "ReplyTM(num_topics={}, coupling=\"blend\", fitted, alpha={:.3}, beta={:.3}, \
                 sigma2={:.3})",
                self.num_topics, self.blend_alpha, self.blend_beta, self.sigma2
            );
        }
        if self.fitted {
            format!(
                "ReplyTM(num_topics={}, coupling={:?}, fitted, kappa={:.3}, sigma2={:.3})",
                self.num_topics, self.coupling, self.kappa, self.sigma2
            )
        } else {
            format!(
                "ReplyTM(num_topics={}, coupling={:?}, unfitted)",
                self.num_topics, self.coupling
            )
        }
    }
}
