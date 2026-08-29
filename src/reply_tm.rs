//! ReplyTM — a reply-threaded topic model.
//!
//! ReplyTM is CTM's logistic-normal topic model with **one structural change**: a document's
//! prior mean is coupled to its parent in the reply tree (a persistence-smoothing prior along
//! reply edges), reverting toward a per-covariate-group anchor:
//!
//! ```text
//! root d:      η_d ~ N(anchor_g, Σ_root)                        (full covariance, CTM-style; #834)
//! non-root d:  η_d ~ N((1-κ)·η_{parent} + κ·anchor_g, Σ_edge)   (κ = reversion; Σ_edge full covariance)
//! tokens:      w ~ softmax([η_d, 0]) · β                        (CTM logistic-normal likelihood)
//! ```
//!
//! Both the root prior (Σ_root) and the reply-edge step (Σ_edge) carry a FULL covariance (the same
//! correlated logistic-normal prior CTM/STM fit). Roots get it so with no reply tree ReplyTM reduces
//! to CTM — up to empty-document handling and a small ridge — rather than a weaker isotropic model
//! (#834); edges get it so a reply leaf is on the SAME covariance footing as a root, so
//! `reply_completion`'s tree-vs-no_tree comparison reflects the reply coupling and not a covariance
//! downgrade. It reduces to a plain (correlated) logistic-normal topic model when the tree is flat. On real corpora the
//! fit drives κ toward 0, i.e. **persistence** (a reply ≈ its parent), so the "reversion" reading
//! is usually vacuous — κ is reported with a profile-likelihood CI that reflects this.
//!
//! The per-document machinery is reused verbatim from [`crate::ctm`]: the variational
//! η-optimization (`lbfgs_minimize` over `ctm_lhood_grad`) and the Laplace bound (`ctm_hpb`,
//! giving the posterior variance `ν` and expected counts `φ`). The E-step coupling uses the
//! parent's current variational mean λ_parent (a structured mean-field on point estimates, NOT
//! the smoothed tree posterior); the [`crate::tree_field`] kernel supplies the exact marginal
//! likelihood used to fit `(κ, σ²)` and to profile κ for its CI. Prevalence carries a
//! cluster-robust (on the thread) method-of-composition SE; κ carries a profile-likelihood CI.
//!
//! The per-iteration objective (`bound_history`) is the sum of the per-document CTM conditional
//! bounds with the parent coupling plugged in as a fixed mean — a variational free energy, NOT a
//! true ELBO for the joint tree model (the parent's posterior variance does not enter the coupling
//! term), so it is monitored, not guaranteed monotone. Ships experimental (topica-original, no
//! published reference).

#![allow(dead_code)] // several public entry points are exercised only by tests / the binding

use crate::ctm::{ctm_hpb, ctm_lhood_grad};
use crate::tree_field::{self, TreeFieldParams};
use crate::variational::{doc_sparse, lbfgs_minimize};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// A fitted ReplyTM.
pub struct ReplyTmModel {
    pub num_topics: usize,
    pub num_types: usize,
    /// K×V topic-word distributions (rows sum to 1).
    pub beta: Vec<Vec<f64>>,
    /// Per-document variational means `η` (each length K-1; reference topic K-1 fixed at 0).
    pub lambda: Vec<Vec<f64>>,
    /// Per-group anchor `μ_g` (num_groups × K-1): the per-topic baseline each covariate group
    /// reverts toward. With one group this is the global mean; with a categorical covariate
    /// (subreddit, verdict, submission-type) it is that group's prevalence in η-space.
    pub anchor: Vec<Vec<f64>>,
    /// Cluster-robust method-of-composition standard error of each anchor entry (num_groups × K-1):
    /// combines a between-THREAD sampling variance of η (clustered on the reply-tree root, so the
    /// within-thread correlation the model fits does not deflate it) with the mean per-document
    /// posterior variance ν. `NaN` for a group with fewer than two threads (variance unidentified).
    pub anchor_se: Vec<Vec<f64>>,
    /// Profile-likelihood 95% CI for the reversion `κ` (lower, upper), re-optimizing (σ², p0) at
    /// each κ; `(NaN, NaN)` when there are no reply edges or the field was not fit (κ unidentified).
    /// Conditional on the topic fit; the point estimate is biased toward κ→0 (persistence).
    pub kappa_ci: (f64, f64),
    /// Reversion strength `κ = 1 - a` toward the anchor. `NaN` under blend coupling (where the mix
    /// is described by `alpha`/`beta` instead of a single reversion).
    pub kappa: f64,
    /// Blend coupling parent weight `α` (`NaN` unless `coupling="blend"`): how much a node's prior
    /// mean tracks its immediate parent.
    pub blend_alpha: f64,
    /// Blend coupling root weight `β` (`NaN` unless `coupling="blend"`): how much a node's prior mean
    /// tracks its thread root. The anchor gets the remaining `1 - α - β`.
    pub blend_beta: f64,
    /// Reported per-edge variance: the mean marginal variance of the fitted full edge covariance
    /// `sigma_edge` (a scalar summary; the edge prior is the full `sigma_edge`).
    pub sigma2: f64,
    /// Root prior variance (mean marginal variance of `sigma_root`, a scalar summary of the full
    /// root covariance).
    pub p0: f64,
    /// Root prior FULL covariance Σ_root (K-1 × K-1, row-major), estimated CTM-style so the base
    /// logistic-normal model captures topic correlation (#834). Used as the root prior in the E-step
    /// and in `transform`.
    pub sigma_root: Vec<f64>,
    /// Edge (OU step) FULL covariance Σ_edge (K-1 × K-1, row-major): the reply edges' prior
    /// covariance, on the same full-covariance footing as `sigma_root` so tree and no-tree share the
    /// base. Identity (unfit) when there are no reply edges.
    pub sigma_edge: Vec<f64>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
    /// Per-document posterior variance of `η` (D × K-1), the final-iteration Laplace `ν`. Exposes
    /// the measurement-error variance of each `lambda` row so a reduced-form persistence estimator
    /// can correct child-on-parent regression for attenuation.
    pub doc_topic_var: Vec<Vec<f64>>,
}

/// Numerically-stable softmax of `[eta, 0]` (reference topic K-1 fixed at 0). Subtracts the max
/// (including the implicit 0) before exponentiating, so large `eta` cannot overflow to NaN.
fn softmax_ref(eta: &[f64]) -> Vec<f64> {
    let mx = eta.iter().copied().fold(0.0_f64, f64::max); // includes the reference-topic 0
    let mut e: Vec<f64> = eta.iter().map(|&x| (x - mx).exp()).collect();
    e.push((-mx).exp());
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

impl ReplyTmModel {
    /// Topic proportions `θ_d = softmax([η_d, 0])` per document.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.lambda.iter().map(|eta| softmax_ref(eta)).collect()
    }

    /// Per-group baseline topic prevalence `θ_g = softmax([μ_g, 0])` — the descriptive mean topic
    /// mix of each covariate group's documents (which topics dominate group g's threads).
    pub fn group_prevalence(&self) -> Vec<Vec<f64>> {
        self.anchor.iter().map(|mu| softmax_ref(mu)).collect()
    }
}

/// Fit ReplyTM by variational EM. `docs` are token-id lists, `parents[d]` the index of `d`'s
/// parent in the reply forest (any negative value marks a root). `num_types` is the vocabulary
/// size. Returns the fitted model; deterministic given `rng` (the E-step runs in parallel but
/// folds sufficient statistics in document order, as in `fit_ctm`).
/// Blend coupling configuration (issue #831). Under blend a non-root node's prior mean is
/// `α·η_parent + β·η_root + (1-α-β)·anchor`, coupling each node to BOTH its immediate parent and its
/// thread root. `root[c]` is node `c`'s thread root (itself for a root). `fixed_alpha`/`fixed_beta`
/// pin the weights; when `None` they are estimated in the M-step by a ridge-regularized regression
/// of each node's centered η on its parent's and root's centered η (a hard-EM point estimate).
pub struct BlendConfig {
    pub root: Vec<usize>,
    pub fixed_alpha: Option<f64>,
    pub fixed_beta: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn fit_reply_tm<R: Rng, F: FnMut(usize, usize, f64) -> bool>(
    docs: &[Vec<u32>],
    parents: &[i64],
    groups: &[usize],
    num_groups: usize,
    num_topics: usize,
    num_types: usize,
    em_iters: usize,
    em_tol: f64,
    compute_ci: bool,
    blend: Option<&BlendConfig>,
    mut on_progress: F,
    rng: &mut R,
) -> ReplyTmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();
    // Empty documents (no tokens, or all tokens dropped by the vocabulary) carry no evidence:
    // their η is never updated by the E-step, so they must be excluded from the anchor mean and
    // treated as UNOBSERVED (latent-only) in the field fit. Feeding their frozen η=0 in as a
    // near-exact pseudo-observation (the old bug) pinned the field and biased (κ, σ²) and the
    // anchor toward the origin.
    let has_tokens: Vec<bool> = sparse.iter().map(|(w, _)| !w.is_empty()).collect();
    let n_edges = parents.iter().filter(|&&p| p >= 0).count();

    // Initialize the topic-word matrix by the SAME spectral (anchor-word) init STM/CTM use
    // (`crate::spectral`), so ReplyTM's logistic-normal base starts from an STM-quality point rather
    // than a random document (issue #834: the weaker random seed left the base ~0.1-0.26 nats/token
    // behind STM, masking the tree's gain). Falls back to the previous per-document random seed when
    // spectral init is unavailable (it returns None on a degenerate/too-small corpus).
    let nonempty: Vec<usize> = (0..d).filter(|&i| !sparse[i].0.is_empty()).collect();
    let mut beta = crate::spectral::spectral_init_with_threshold(
        docs,
        k,
        num_types,
        crate::spectral::DEFAULT_PROJ_THRESHOLD,
    )
    .unwrap_or_else(|| {
        // Fallback: seed each topic from a random non-empty document's word distribution, smoothed.
        let mut beta = vec![vec![1.0f64 / num_types as f64; num_types]; k];
        if !nonempty.is_empty() {
            for row in beta.iter_mut() {
                let di =
                    nonempty[(rng.gen::<f64>() * nonempty.len() as f64) as usize % nonempty.len()];
                let (words, counts) = &sparse[di];
                let mut r = vec![1.0f64; num_types]; // Laplace smoothing
                for (wi, &w) in words.iter().enumerate() {
                    r[w] += 10.0 * counts[wi];
                }
                let s: f64 = r.iter().sum();
                for (v, &ri) in row.iter_mut().zip(&r) {
                    *v = ri / s;
                }
            }
        }
        beta
    });

    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut anchor = vec![vec![0.0f64; km1]; num_groups.max(1)];
    let mut last_nu_diag: Vec<Vec<f64>> = vec![vec![0.0f64; km1]; d]; // final-iter posterior var
                                                                      // field hyperparameters; a = 1 - kappa
    let mut a = 0.7f64;
    let mut sigma2 = 1.0f64;
    let mut p0 = 1.0f64;
    // Root prior FULL covariance Σ_root (K-1 × K-1), estimated CTM-style from the root documents so
    // the base logistic-normal model captures topic correlation (issue #834: an isotropic root prior
    // left the base ~0.1-0.26 nats/token behind STM at scale). With no reply tree every document is a
    // root, so this makes ReplyTM's base equivalent to CTM. Init to the identity (≡ the old p0 = 1).
    let mut sigma_root = vec![0.0f64; km1 * km1];
    // Edge (OU step) FULL covariance Σ_edge, the analogue of Σ_root for reply edges. Keeping edges
    // isotropic while roots were full made a thin leaf's prior RICHER with the tree off than on, so
    // `reply_completion`'s tree-vs-no_tree delta conflated the reply coupling with a covariance
    // downgrade. A full Σ_edge (both sides full) restores a fair comparison. Warm-up-gated (identity
    // until the topics separate) to avoid the collapse trap the isotropic σ² field also guarded.
    let mut sigma_edge = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma_root[i * km1 + i] = 1.0;
        sigma_edge[i * km1 + i] = 1.0;
    }
    // Blend coupling weights (α = parent, β = root); seeded to a parent-leaning mix and updated in
    // the M-step unless pinned. Unused (and returned as NaN) when `blend` is None.
    let mut alpha = blend.and_then(|b| b.fixed_alpha).unwrap_or(0.5);
    let mut beta_w = blend.and_then(|b| b.fixed_beta).unwrap_or(0.25);

    let mut bound_history: Vec<f64> = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;
    // Whether Σ_edge was ever estimated from at least one token-bearing edge (so the reported sigma2
    // is a real estimate, not the identity init on a degenerate tree whose non-roots are all empty).
    let mut edge_cov_fit = false;
    // Whether the tree field (a, σ², p0) was ever actually fit. It is gated behind a warm-up, so a
    // corpus that converges inside the warm-up window would otherwise return the INIT constants
    // (κ=0.3, σ²=1, p0=1) dressed up as estimates. We refuse to break before it has run once, and
    // null the field params if the iteration budget never reached it.
    let mut field_fit_ran = false;

    for em in 0..em_iters {
        em_iters_run = em + 1;
        let kappa = 1.0 - a;

        // Both root and edge priors use their FULL covariance (inverse + half-logdet), like CTM, so
        // roots and reply edges are on the same covariance footing.
        let (siginv_edge, ent_edge) = crate::linalg::spd_inverse_and_half_logdet(&sigma_edge, km1);
        let (siginv_root, ent_root) = crate::linalg::spd_inverse_and_half_logdet(&sigma_root, km1);

        // E-step: per-document logistic-normal inference with a tree-coupled prior mean.
        // Reads the previous iteration's `lambda` for the parent coupling (structured
        // mean-field); parallel map, folded in document order for determinism.
        let results: Vec<(usize, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64)> = sparse
            .par_iter()
            .enumerate()
            .filter(|(_, (w, _))| !w.is_empty())
            .map(|(di, (words, counts))| {
                let par = parents[di];
                let ag = &anchor[groups[di]]; // the document's covariate-group anchor
                let (mu_d, siginv, entropy) = if par < 0 {
                    (ag.clone(), &siginv_root, ent_root)
                } else if let Some(bc) = blend {
                    // Blend: couple to BOTH parent and thread root, reverting toward the anchor by
                    // the residual weight. Reads the previous iteration's λ for both (structured
                    // mean-field); parent and root are ancestors, so this is a directed coupling.
                    let lp = &lambda[par as usize];
                    let lr = &lambda[bc.root[di]];
                    let mu: Vec<f64> = (0..km1)
                        .map(|i| alpha * lp[i] + beta_w * lr[i] + (1.0 - alpha - beta_w) * ag[i])
                        .collect();
                    (mu, &siginv_edge, ent_edge)
                } else {
                    let lp = &lambda[par as usize];
                    let mu: Vec<f64> = (0..km1)
                        .map(|i| (1.0 - kappa) * lp[i] + kappa * ag[i])
                        .collect();
                    (mu, &siginv_edge, ent_edge)
                };
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| ctm_lhood_grad(eta, &beta, words, counts, &mu_d, siginv),
                    40,
                    7,
                    1e-5,
                );
                // Both roots and edges now re-estimate a FULL covariance, so every document needs the
                // full Laplace covariance ν (diagonal=false).
                let res = ctm_hpb(&opt, &beta, words, counts, &mu_d, siginv, entropy, false);
                (di, opt, res.nu, res.phi, res.bound)
            })
            .collect();

        let total_bound: f64 = results.iter().map(|(_, _, _, _, b)| b).sum();
        bound_history.push(total_bound);
        if !on_progress(em + 1, em_iters, total_bound) {
            break;
        }

        // fold sufficient statistics in document order
        let mut beta_ss = vec![vec![1e-8f64; num_types]; k];
        let mut nu_diag_store = vec![vec![0.0f64; km1]; d];
        // Full ν for every token-bearing document (Σ_root uses the roots' ν, Σ_edge the edges').
        let mut nu_full: Vec<Vec<f64>> = vec![Vec::new(); d];
        for (di, opt, nu, phi, _) in &results {
            let di = *di;
            lambda[di] = opt.clone();
            nu_diag_store[di] = (0..km1).map(|i| nu[i * km1 + i]).collect();
            nu_full[di] = nu.clone();
            let words = &sparse[di].0;
            for (wi, &w) in words.iter().enumerate() {
                for (t, brow) in beta_ss.iter_mut().enumerate() {
                    brow[w] += phi[t][wi];
                }
            }
        }
        last_nu_diag.clone_from(&nu_diag_store); // keep the final-iteration posterior variances

        // Only honor convergence once the field has been fit at least once (or there are no edges,
        // so there is no field to fit). Otherwise a fast-converging corpus would return the field
        // initializers as if estimated (see `field_fit_ran`).
        if em_tol > 0.0 && bound_history.len() >= 2 && (field_fit_ran || n_edges == 0) {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                let _ = on_progress(em + 1, em + 1, total_bound);
                converged = true;
                break;
            }
        }

        // M-step (β): normalize expected token counts
        for brow in beta_ss.iter_mut() {
            let s: f64 = brow.iter().sum();
            for v in brow.iter_mut() {
                *v /= s;
            }
        }
        beta = beta_ss;

        // M-step (anchor): per-GROUP per-topic baseline = mean η within the covariate group
        // (categorical prevalence; equals the global mean when there is a single group).
        let ng = num_groups.max(1);
        let mut gsum = vec![vec![0.0f64; km1]; ng];
        let mut gcnt = vec![0usize; ng];
        for (di, l) in lambda.iter().enumerate() {
            if !has_tokens[di] {
                continue; // empty docs have a frozen η=0, not a real estimate
            }
            let g = groups[di];
            gcnt[g] += 1;
            for i in 0..km1 {
                gsum[g][i] += l[i];
            }
        }
        for g in 0..ng {
            if gcnt[g] > 0 {
                for i in 0..km1 {
                    anchor[g][i] = gsum[g][i] / gcnt[g] as f64;
                }
            }
        }

        // M-step (Σ_root): full root-prior covariance, CTM's update
        // Σ = (1/n_root)[ Σ ν_root + Σ (η_root − anchor)(η_root − anchor)ᵀ ], over token-bearing
        // roots. Gives the base logistic-normal model topic correlation (#834). A small diagonal
        // ridge keeps it positive-definite for the inverse/logdet next iteration.
        {
            let mut ss = vec![0.0f64; km1 * km1];
            let mut n_root = 0usize;
            for di in 0..d {
                if parents[di] >= 0 || !has_tokens[di] {
                    continue;
                }
                let ag = &anchor[groups[di]];
                let nu = &nu_full[di];
                for i in 0..km1 {
                    let ci = lambda[di][i] - ag[i];
                    for j in 0..km1 {
                        let cj = lambda[di][j] - ag[j];
                        ss[i * km1 + j] += nu[i * km1 + j] + ci * cj;
                    }
                }
                n_root += 1;
            }
            if n_root > 0 {
                for v in ss.iter_mut() {
                    *v /= n_root as f64;
                }
                for i in 0..km1 {
                    ss[i * km1 + i] += 1e-6; // PD ridge
                }
                sigma_root = ss;
            }
        }

        // M-step (field): fit (a, σ², p0) with the tree-field kernel on centered η. Centering
        // by the anchor lets the K dims share isotropic (a, σ²) with a per-topic baseline.
        //
        // WARM-UP: hold the field fixed (moderate σ²) for the first `warmup` iterations so the
        // topics separate first. Fitting it too early is a degenerate trap — before topics
        // separate, η is collapsed, so the fit drives σ²→0, whose infinite prior precision then
        // PINS each reply's η to its parent and prevents topics from ever separating. A σ² FLOOR
        // keeps a reply's own tokens able to move it off the parent even once the field is fit.
        let warmup = 15;
        if em + 1 > warmup {
            if let Some(bc) = blend {
                // Blend M-step: estimate (α, β) by a ridge-regularized, errors-in-variables
                // regression of each node's anchor-centered η on its parent's and root's
                // anchor-centered η, pooled over topics and edges. The EIV term (below) subtracts the
                // Laplace ν so the weights are reliability-corrected rather than attenuated (the
                // two-regressor analogue of persistence()'s structural correction); still a hard-EM
                // estimate conditional on the topic fit, so the held-out reply_completion delta is
                // the model-vs-model readout. σ² is the residual variance; p0 the root variance.
                let (mut spp, mut spr, mut srr, mut spy, mut sry, mut syy, mut ne) =
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
                // Measurement-error (Laplace ν) sums for the errors-in-variables correction below:
                // the parent/root regressors are noisy estimates of the latent η, so their observed
                // second moments are inflated by ν; subtracting it de-attenuates α/β (the same
                // reliability correction persistence() applies, generalized to two regressors).
                let (mut snu_p, mut snu_r, mut snu_pr) = (0.0, 0.0, 0.0f64);
                for c in 0..d {
                    let par = parents[c];
                    if par < 0 || !has_tokens[c] {
                        continue;
                    }
                    let (p, r) = (par as usize, bc.root[c]);
                    if !has_tokens[p] || !has_tokens[r] {
                        continue;
                    }
                    let ag = &anchor[groups[c]];
                    for i in 0..km1 {
                        let xp = lambda[p][i] - ag[i];
                        let xr = lambda[r][i] - ag[i];
                        let y = lambda[c][i] - ag[i];
                        spp += xp * xp;
                        spr += xp * xr;
                        srr += xr * xr;
                        spy += xp * y;
                        sry += xr * y;
                        syy += y * y;
                        snu_p += nu_diag_store[p][i];
                        snu_r += nu_diag_store[r][i];
                        // When parent == root (depth-2 nodes) the two regressors are the SAME noisy
                        // node, so their measurement errors are perfectly shared and inflate the
                        // cross term too; otherwise the errors are independent (cross-correction 0).
                        if p == r {
                            snu_pr += nu_diag_store[p][i];
                        }
                        ne += 1.0;
                    }
                }
                if ne > 0.0 {
                    // Errors-in-variables normal equations (de-attenuated), with a ridge for
                    // stability / collinearity (depth-2 threads, where parent == root).
                    let ridge = 1e-6 * (spp + srr) + 1e-9;
                    let a11 = (spp - snu_p).max(0.0) + ridge;
                    let a22 = (srr - snu_r).max(0.0) + ridge;
                    let a12 = spr - snu_pr;
                    // The free (unpinned) weight's conditional 1-D EIV minimizer, clamped so it never
                    // drives α+β past 1 (a negative anchor weight would break the convex blend).
                    let free_a = |b: f64, hi: f64| ((spy - b * a12) / a11).clamp(0.0, hi);
                    let free_b = |a: f64, hi: f64| ((sry - a * a12) / a22).clamp(0.0, hi);
                    let (al, be) = match (bc.fixed_alpha, bc.fixed_beta) {
                        (Some(fa), Some(fb)) => (fa, fb),
                        // Pin one, estimate the other conditional on the pin (not the biased joint
                        // value) and cap it at 1 - pin so the invariant always holds.
                        (Some(fa), None) => (fa, free_b(fa, 1.0 - fa)),
                        (None, Some(fb)) => (free_a(fb, 1.0 - fb), fb),
                        // Both free: the joint solve, projected onto {α,β ≥ 0, α+β ≤ 1} (clamp then
                        // rescale) so the prior mean stays a convex blend of parent, root, anchor.
                        (None, None) => {
                            let det = a11 * a22 - a12 * a12;
                            if det.abs() <= 1e-12 {
                                (alpha, beta_w) // keep the previous iterate if degenerate
                            } else {
                                let mut a = ((a22 * spy - a12 * sry) / det).max(0.0);
                                let mut b = ((a11 * sry - a12 * spy) / det).max(0.0);
                                if a + b > 1.0 {
                                    let s = a + b;
                                    a /= s;
                                    b /= s;
                                }
                                (a, b)
                            }
                        }
                    };
                    alpha = al;
                    beta_w = be;
                    // σ² = residual variance of y around α·xp + β·xr (floored). Left with the
                    // measurement error folded in, as the σ² floor already bounds the edge prior;
                    // de-convolving ν here destabilized the fit through the E-step feedback.
                    let resid = (syy - 2.0 * (alpha * spy + beta_w * sry)
                        + alpha * alpha * spp
                        + 2.0 * alpha * beta_w * spr
                        + beta_w * beta_w * srr)
                        / ne;
                    sigma2 = resid.max(0.1);
                }
                // p0 = root variance around the anchor.
                let (mut rss, mut nr) = (0.0, 0.0f64);
                for c in 0..d {
                    if parents[c] < 0 && has_tokens[c] {
                        let ag = &anchor[groups[c]];
                        for i in 0..km1 {
                            let e = lambda[c][i] - ag[i];
                            rss += e * e;
                            nr += 1.0;
                        }
                    }
                }
                if nr > 0.0 {
                    p0 = (rss / nr).max(0.1);
                }
                field_fit_ran = true;
            } else {
                let obs: Vec<Vec<f64>> = (0..km1)
                    .map(|i| {
                        lambda
                            .iter()
                            .enumerate()
                            .map(|(di, l)| l[i] - anchor[groups[di]][i])
                            .collect()
                    })
                    .collect();
                // Empty nodes stay in the tree (so their children still couple through them) but
                // carry NO observation: r = +inf makes them latent-only, so they never pin the field.
                let r: Vec<f64> = (0..d)
                    .map(|di| {
                        if !has_tokens[di] {
                            return 1e12;
                        }
                        let nd = &nu_diag_store[di];
                        (nd.iter().sum::<f64>() / km1 as f64).max(1e-6)
                    })
                    .collect();
                // fit with the anchor mean held at 0 (obs are anchor-centered): this keeps the fitted
                // `a` consistent with the m=0 profile used for kappa_ci, and avoids double-counting the
                // mean the anchor already removed.
                let fit = tree_field::fit_fixed_mean(
                    parents,
                    &obs,
                    &r,
                    TreeFieldParams {
                        a,
                        q: sigma2,
                        m: 0.0,
                        p0,
                    },
                );
                // clamp a < 1 so the next iteration's logit init stays finite (a=1 → +inf → NaN in
                // the Nelder-Mead simplex); floor σ²/p0 (a clamp, not an estimate — see kappa_ci).
                // This scalar isotropic σ² seeds only the tree-field BP that estimates the reversion
                // `a` (κ); the REPORTED sigma2 is reassigned from mean-diag Σ_edge after the loop.
                a = fit.a.min(0.999);
                sigma2 = fit.q.max(0.1);
                p0 = fit.p0.max(0.1);
                field_fit_ran = true;
            }
        }

        // M-step (Σ_edge): full edge (OU step) covariance, CTM's update over the reply edges around
        // the coupled mean μ_c = (1-κ)·parent + κ·anchor (or the blend mean). Warm-up-gated: held at
        // identity until the topics separate, since (as with the isotropic σ² field) estimating the
        // edge covariance too early can collapse it and pin children to their parents. This puts the
        // tree edges on the same full-covariance footing as the roots, so `reply_completion`'s
        // tree-vs-no_tree delta reflects the reply coupling, not a covariance downgrade.
        if em + 1 > warmup {
            let mut ss = vec![0.0f64; km1 * km1];
            let mut n_edge = 0usize;
            for di in 0..d {
                let par = parents[di];
                if par < 0 || !has_tokens[di] {
                    continue;
                }
                let ag = &anchor[groups[di]];
                let p = par as usize;
                // The prior mean used for this edge (matches the E-step coupling with the updated κ).
                let mu: Vec<f64> = if let Some(bc) = blend {
                    let lr = &lambda[bc.root[di]];
                    (0..km1)
                        .map(|i| {
                            alpha * lambda[p][i] + beta_w * lr[i] + (1.0 - alpha - beta_w) * ag[i]
                        })
                        .collect()
                } else {
                    let kappa = 1.0 - a;
                    (0..km1)
                        .map(|i| (1.0 - kappa) * lambda[p][i] + kappa * ag[i])
                        .collect()
                };
                let nu = &nu_full[di];
                for i in 0..km1 {
                    let ci = lambda[di][i] - mu[i];
                    for j in 0..km1 {
                        let cj = lambda[di][j] - mu[j];
                        ss[i * km1 + j] += nu[i * km1 + j] + ci * cj;
                    }
                }
                n_edge += 1;
            }
            if n_edge > 0 {
                for v in ss.iter_mut() {
                    *v /= n_edge as f64;
                }
                for i in 0..km1 {
                    ss[i * km1 + i] += 1e-6; // PD ridge
                }
                sigma_edge = ss;
                edge_cov_fit = true;
            }
        }
    }

    // The field params are only real estimates when there ARE reply edges AND the field was
    // actually fit (it is warm-up-gated, so a short/fast run may never reach it). Otherwise they
    // are still the init constants; report NaN (unidentified) rather than dress them up as fitted.
    if n_edges == 0 || !field_fit_ran {
        a = f64::NAN;
    }
    // sigma2 summarizes the fitted full edge covariance (mean marginal variance) — but only once
    // Σ_edge was actually estimated from a token-bearing edge; otherwise it is the identity init on a
    // degenerate tree (all non-roots empty), so report NaN rather than a spurious 1.0.
    sigma2 = if edge_cov_fit {
        (0..km1).map(|i| sigma_edge[i * km1 + i]).sum::<f64>() / km1 as f64
    } else {
        f64::NAN
    };
    // p0 now summarizes the fitted full root covariance (mean marginal variance), so it is defined
    // whenever there are token-bearing roots — including the no-tree (CTM-equivalent) case — rather
    // than the old scalar root variance. NaN only if Σ_root was never estimated (no roots).
    p0 = {
        let m: f64 = (0..km1).map(|i| sigma_root[i * km1 + i]).sum::<f64>() / km1 as f64;
        if m > 0.0 && m.is_finite() {
            m
        } else {
            f64::NAN
        }
    };
    // Blend reports its mix via (alpha, beta), not a single reversion; kappa is undefined. When the
    // field never ran (no edges / warm-up not reached) the weights are still the init constants, so
    // report them as NaN like the other field params.
    let (kappa_out, alpha_out, beta_out) = if blend.is_some() {
        if n_edges == 0 || !field_fit_ran {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (f64::NAN, alpha, beta_w)
        }
    } else {
        (1.0 - a, f64::NAN, f64::NAN)
    };

    // ---- uncertainty ----
    // Thread root of each document (walk parents to the root). The reply tree induces WITHIN-THREAD
    // correlation — that is the whole model — so the group-prevalence SE must cluster on the thread,
    // not treat documents as i.i.d. (which would understate the variance by the design effect,
    // worst exactly in the κ≈0 persistence regime the model is sold for).
    let thread_root: Vec<usize> = (0..d)
        .map(|start| {
            let mut cur = start as i64;
            while parents[cur as usize] >= 0 {
                cur = parents[cur as usize];
            }
            cur as usize
        })
        .collect();

    // Anchor SE (method of composition, cluster-robust): the group prevalence is a mean of
    // per-document η. Its variance combines (a) a CLUSTER-robust between-thread term — sum the
    // centered η within each thread, square, sum over threads, with a G/(G-1) small-cluster
    // correction — and (b) the composition term for latent uncertainty, mean per-document posterior
    // variance ν. NaN when a group has fewer than two threads (the between-thread variance is then
    // unidentified — we decline to report a spuriously tight interval).
    let ng = num_groups.max(1);
    let mut anchor_se = vec![vec![f64::NAN; km1]; ng];
    for (g, se_g) in anchor_se.iter_mut().enumerate() {
        let members: Vec<usize> = (0..d)
            .filter(|&di| has_tokens[di] && groups[di] == g)
            .collect();
        let n = members.len();
        // distinct threads represented in this group
        let mut roots: Vec<usize> = members.iter().map(|&di| thread_root[di]).collect();
        roots.sort_unstable();
        roots.dedup();
        let n_clusters = roots.len();
        if n < 2 || n_clusters < 2 {
            continue; // leaves se = NaN: not enough independent threads for an honest interval
        }
        let cf = n_clusters as f64 / (n_clusters as f64 - 1.0); // small-cluster correction
        for (i, se) in se_g.iter_mut().enumerate() {
            let mean = anchor[g][i];
            // between-thread: (G/(G-1)) * Σ_c (Σ_{d in c} (η_d - mean))² / n²
            let mut csum: HashMap<usize, f64> = HashMap::new();
            for &di in &members {
                *csum.entry(thread_root[di]).or_insert(0.0) += lambda[di][i] - mean;
            }
            let between: f64 =
                cf * csum.values().map(|s| s * s).sum::<f64>() / (n as f64 * n as f64);
            // composition term for latent uncertainty: Σ_d ν_d / n²
            let nu_term: f64 =
                members.iter().map(|&di| last_nu_diag[di][i]).sum::<f64>() / (n as f64 * n as f64);
            *se = (between + nu_term).sqrt();
        }
    }

    // κ CI is the dominant per-fit cost (a 99-point profile, each an inner Nelder-Mead), and it is
    // usually not read (persistence() supersedes it). Compute it here only when explicitly asked;
    // otherwise the binding computes it lazily from the stored fit via `kappa_profile_ci`.
    // kappa_ci is a tree-field profile of the reversion; it has no meaning under blend coupling.
    let kappa_ci = if compute_ci && blend.is_none() {
        kappa_profile_ci(
            parents,
            &lambda,
            &anchor,
            groups,
            &last_nu_diag,
            &has_tokens,
            sigma2,
            p0,
            a,
        )
    } else {
        (f64::NAN, f64::NAN)
    };

    ReplyTmModel {
        num_topics: k,
        num_types,
        beta,
        lambda,
        anchor,
        anchor_se,
        kappa_ci,
        kappa: kappa_out,
        blend_alpha: alpha_out,
        blend_beta: beta_out,
        sigma2,
        p0,
        sigma_root,
        sigma_edge,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
        doc_topic_var: last_nu_diag,
    }
}

/// Infer topic proportions θ for a NEW reply forest under a fitted model, holding the topics `beta`,
/// the reversion `kappa`, the full edge and root precisions `edge_siginv` (= Σ_edge⁻¹) and
/// `root_siginv` (= Σ_root⁻¹), and the per-group `anchor` all fixed. This is `transform` for ReplyTM.
///
/// The E-step coupling is directed — a document's prior mean depends only on its PARENT's η, never on
/// its children (see `fit_reply_tm`). So a single sweep in topological order (every parent before its
/// children) is the structured mean-field fixed point: each node is inferred against a prior mean
/// built from its parent's already-finalized η, and nothing downstream feeds back, so there is no need
/// to iterate. On a tree of token-bearing nodes this reproduces the converged fit E-step. `parents[d]`
/// indexes into `docs` (negative = root); `groups[d]` selects the anchor row. Documents with no
/// in-vocabulary tokens carry no evidence, so their posterior mode is the prior mean (θ = softmax of
/// that mean) and they still pass persistence on to their children. NOTE this differs from
/// `fit_reply_tm`, which excludes empty documents from its E-step and leaves their η at the init 0
/// (θ uniform); transform's prior-mode is the principled value (it matches `ctm::infer_theta`), so on
/// a tree with an empty interior or root node transform and the stored fit `doc_topic` diverge for
/// that node and its subtree. Returns D×K proportions in `docs` order.
#[allow(clippy::too_many_arguments)]
pub fn transform_reply_tm(
    docs: &[Vec<u32>],
    parents: &[i64],
    groups: &[usize],
    beta: &[Vec<f64>],
    anchor: &[Vec<f64>],
    kappa: f64,
    edge_siginv: &[f64],
    root_siginv: &[f64],
    blend: Option<(f64, f64)>,
) -> Vec<Vec<f64>> {
    let k = beta.len();
    let km1 = k - 1;
    let d = docs.len();
    // Blend needs each node's thread root (parent and root are both ancestors, so the same
    // topological pass still finalizes both before a node is inferred).
    let root: Vec<usize> = (0..d)
        .map(|start| {
            let mut cur = start as i64;
            while parents[cur as usize] >= 0 {
                cur = parents[cur as usize];
            }
            cur as usize
        })
        .collect();

    // Both edges (Σ_edge⁻¹) and roots (Σ_root⁻¹) use their FULL fitted precision.
    let siginv_edge = edge_siginv;
    let siginv_root = root_siginv;

    // Topological order (roots first, every parent before its children). The tree is acyclic (the
    // binding validates this), so a BFS from the roots down the children lists visits each node once.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); d];
    let mut order: Vec<usize> = Vec::with_capacity(d);
    for (c, &p) in parents.iter().enumerate() {
        if p < 0 {
            order.push(c);
        } else {
            children[p as usize].push(c);
        }
    }
    let mut head = 0;
    while head < order.len() {
        let node = order[head];
        head += 1;
        for &c in &children[node] {
            order.push(c);
        }
    }

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();
    let mut lambda = vec![vec![0.0f64; km1]; d];
    for &di in &order {
        let ag = &anchor[groups[di]];
        let par = parents[di];
        let (mu_d, siginv): (Vec<f64>, &[f64]) = if par < 0 {
            (ag.clone(), siginv_root)
        } else if let Some((alpha, beta_w)) = blend {
            let lp = &lambda[par as usize];
            let lr = &lambda[root[di]];
            let mu: Vec<f64> = (0..km1)
                .map(|i| alpha * lp[i] + beta_w * lr[i] + (1.0 - alpha - beta_w) * ag[i])
                .collect();
            (mu, siginv_edge)
        } else {
            let lp = &lambda[par as usize];
            let mu: Vec<f64> = (0..km1)
                .map(|i| (1.0 - kappa) * lp[i] + kappa * ag[i])
                .collect();
            (mu, siginv_edge)
        };
        let (words, counts) = &sparse[di];
        lambda[di] = if words.is_empty() {
            // No token evidence: the objective reduces to the prior, whose mode is η = μ.
            mu_d
        } else {
            lbfgs_minimize(
                mu_d.clone(),
                |eta| ctm_lhood_grad(eta, beta, words, counts, &mu_d, siginv),
                40,
                7,
                1e-5,
            )
        };
    }

    lambda.iter().map(|eta| softmax_ref(eta)).collect()
}

/// 95% profile-likelihood CI for the reversion κ, factored out of `fit_reply_tm` because it is the
/// dominant per-fit cost and is usually not read. At each candidate `a` on a 99-point grid it
/// re-optimizes the nuisance variances (σ², p0) via `profile_loglik_at_a` and keeps the a's within
/// a χ²(1)/2 = 1.92 log-likelihood drop of the profile max. The grid points are independent, so
/// they run in PARALLEL. `lambda`/`nu` are per-document η and its posterior variance, `anchor` the
/// per-group η mean, `has_tokens` marks documents that carry evidence. Returns `(NaN, NaN)` when
/// there are no reply edges or the field was not fit (`a` non-finite). κ = 1 - a flips the interval.
#[allow(clippy::too_many_arguments)]
pub fn kappa_profile_ci(
    parents: &[i64],
    lambda: &[Vec<f64>],
    anchor: &[Vec<f64>],
    groups: &[usize],
    nu: &[Vec<f64>],
    has_tokens: &[bool],
    sigma2: f64,
    p0: f64,
    a: f64,
) -> (f64, f64) {
    let d = lambda.len();
    if d == 0 || !a.is_finite() || parents.iter().all(|&p| p < 0) {
        return (f64::NAN, f64::NAN);
    }
    let km1 = lambda[0].len();
    let obs: Vec<Vec<f64>> = (0..km1)
        .map(|i| {
            (0..d)
                .map(|di| lambda[di][i] - anchor[groups[di]][i])
                .collect()
        })
        .collect();
    let r: Vec<f64> = (0..d)
        .map(|di| {
            if has_tokens[di] {
                (nu[di].iter().sum::<f64>() / km1 as f64).max(1e-6)
            } else {
                1e12
            }
        })
        .collect();
    let init = TreeFieldParams {
        a: 0.0,
        q: sigma2.max(1e-6),
        m: 0.0,
        p0: p0.max(1e-6),
    };
    let prof = |av: f64| tree_field::profile_loglik_at_a(parents, &obs, &r, av, init);
    // 99-point grid (plus the fitted â), evaluated in parallel — each profile is independent.
    let grid: Vec<f64> = (1..=99).map(|j| j as f64 / 100.0).chain([a]).collect();
    let profs: Vec<(f64, f64)> = grid.par_iter().map(|&av| (av, prof(av))).collect();
    let ll_max = profs
        .iter()
        .map(|&(_, ll)| ll)
        .fold(f64::NEG_INFINITY, f64::max);
    // Seed lo/hi ONLY from admissible points (do not force â in — an â below the cutoff would be
    // wrongly included). ll_max is attained on the grid, so at least that point is admissible.
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(av, ll) in &profs {
        if ll >= ll_max - 1.92 {
            lo = lo.min(av);
            hi = hi.max(av);
        }
    }
    (1.0 - hi, 1.0 - lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn gauss(rng: &mut impl Rng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-12);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate a threaded corpus with a known reversion `κ`, disjoint block topics, and an OU
    /// prevalence field; fit ReplyTM; recover the topics and a non-degenerate κ.
    #[test]
    fn recovers_topics_and_reversion() {
        let (k, wpt) = (4usize, 5usize); // 4 topics, 5 words each -> V = 20, disjoint blocks
        let v = k * wpt;
        let km1 = k - 1;
        let (a_true, q_true, p0_true) = (0.75f64, 0.40f64, 1.0f64); // κ = 0.25
        let (n_trees, depth, doc_len) = (80usize, 15usize, 50usize);

        let mut rng = StdRng::seed_from_u64(13);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut parents: Vec<i64> = Vec::new();
        let mut eta: Vec<Vec<f64>> = Vec::new();
        for _ in 0..n_trees {
            for t in 0..depth {
                let idx = docs.len();
                let e: Vec<f64> = if t == 0 {
                    parents.push(-1);
                    (0..km1).map(|_| p0_true.sqrt() * gauss(&mut rng)).collect()
                } else {
                    parents.push((idx - 1) as i64);
                    let ep = &eta[idx - 1];
                    (0..km1)
                        .map(|i| a_true * ep[i] + q_true.sqrt() * gauss(&mut rng))
                        .collect()
                };
                // theta = softmax([eta, 0])
                let mut ex: Vec<f64> = e.iter().map(|&x| x.exp()).collect();
                ex.push(1.0);
                let s: f64 = ex.iter().sum();
                let theta: Vec<f64> = ex.iter().map(|&x| x / s).collect();
                eta.push(e);
                // sample doc: z ~ theta, word ~ uniform over z's block
                let mut doc = Vec::with_capacity(doc_len);
                for _ in 0..doc_len {
                    let u: f64 = rng.gen();
                    let mut acc = 0.0;
                    let mut z = k - 1;
                    for (t2, &p) in theta.iter().enumerate() {
                        acc += p;
                        if u <= acc {
                            z = t2;
                            break;
                        }
                    }
                    let w = z * wpt + (rng.gen::<f64>() * wpt as f64) as usize;
                    doc.push(w.min(v - 1) as u32);
                }
                docs.push(doc);
            }
        }

        let mut fit_rng = StdRng::seed_from_u64(7);
        let groups = vec![0usize; docs.len()]; // single covariate group (global anchor)
        let model = fit_reply_tm(
            &docs,
            &parents,
            &groups,
            1,
            k,
            v,
            150,
            1e-7,
            true, // exercise the eager kappa_ci path
            None, // parent coupling
            |_, _, _| true,
            &mut fit_rng,
        );

        // topic recovery: each true block should be captured by a distinct fitted topic with
        // most of its mass on that block (greedy injective match).
        let block_mass = |f: usize, blk: usize| -> f64 {
            (blk * wpt..blk * wpt + wpt).map(|w| model.beta[f][w]).sum()
        };
        let mut used = vec![false; k];
        let mut worst_match = 1.0f64;
        for blk in 0..k {
            let mut best = (0usize, -1.0f64);
            for f in 0..k {
                if !used[f] {
                    let m = block_mass(f, blk);
                    if m > best.1 {
                        best = (f, m);
                    }
                }
            }
            used[best.0] = true;
            worst_match = worst_match.min(best.1);
        }
        assert!(
            worst_match > 0.8,
            "topic recovery weak: worst block mass {worst_match:.3}"
        );
        // Reversion is recovered non-degenerate and in a sane band. We do NOT assert exact κ:
        // through-model κ is biased low by topic-model shrinkage (a base-invariant confound the
        // permutation-null diagnostic handles, not this test). The milestone here is end-to-end
        // recovery — separated topics plus a non-collapsed, non-saturated reversion.
        assert!(
            model.kappa > 0.02 && model.kappa < 0.6,
            "kappa degenerate: {}",
            model.kappa
        );
    }
}
