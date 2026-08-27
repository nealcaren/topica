//! ReplyTM — a reply-threaded topic model.
//!
//! ReplyTM is CTM/STM's logistic-normal topic model with **one structural change**: a
//! document's prior mean is not independent but *tree-coupled* to its parent in the reply
//! tree. Topic prevalence `η_d` diffuses along reply edges as an Ornstein–Uhlenbeck process,
//! so a reply starts near the comment it answers and reverts toward a community anchor:
//!
//! ```text
//! root d:      η_d ~ N(anchor, p0·I)
//! non-root d:  η_d ~ N((1-κ)·η_{parent} + κ·anchor, σ²·I)      (κ = reversion, σ² = diffusion)
//! tokens:      w ~ softmax([η_d, 0]) · β                        (CTM/STM likelihood)
//! ```
//!
//! It reduces exactly to a plain logistic-normal topic model when `κ = 1` (no memory). The
//! per-document machinery is reused verbatim from [`crate::ctm`]: the variational η-optimization
//! (`lbfgs_minimize` over `ctm_lhood_grad`) and the Laplace bound (`ctm_hpb`, giving the
//! posterior variance `ν` and the expected token counts `φ`). The only new pieces are the
//! tree-coupled prior mean in the E-step and an M-step that fits `(κ, σ²)` with the exact
//! Gaussian tree-field kernel [`crate::tree_field`].
//!
//! This is the diffusion model that replaced the shelved directed-transition "ReplyTM"; the name
//! now denotes the model for its purpose (topic-modeling reply threads). Ships experimental.

// Binding lands in a follow-up commit; the fit entry point is exercised by the planted-recovery
// test until then.
#![allow(dead_code)]

use crate::ctm::{ctm_hpb, ctm_lhood_grad};
use crate::tree_field::{self, TreeFieldParams};
use crate::variational::{doc_sparse, lbfgs_minimize};
use rand::Rng;
use rayon::prelude::*;

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
    /// Reversion strength `κ = 1 - a` toward the anchor.
    pub kappa: f64,
    /// Per-edge diffusion variance `σ²`.
    pub sigma2: f64,
    /// Root prior variance.
    pub p0: f64,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl ReplyTmModel {
    /// Topic proportions `θ_d = softmax([η_d, 0])` per document.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.lambda
            .iter()
            .map(|eta| {
                let mut e: Vec<f64> = eta.iter().map(|&x| x.exp()).collect();
                e.push(1.0);
                let s: f64 = e.iter().sum();
                e.iter().map(|&x| x / s).collect()
            })
            .collect()
    }

    /// Per-group topic prevalence `θ_g = softmax([μ_g, 0])` — the STM-style baseline topic mix
    /// of each covariate group (the answer to "which topics dominate group g's threads").
    pub fn group_prevalence(&self) -> Vec<Vec<f64>> {
        self.anchor
            .iter()
            .map(|mu| {
                let mut e: Vec<f64> = mu.iter().map(|&x| x.exp()).collect();
                e.push(1.0);
                let s: f64 = e.iter().sum();
                e.iter().map(|&x| x / s).collect()
            })
            .collect()
    }
}

/// Fit ReplyTM by variational EM. `docs` are token-id lists, `parents[d]` the index of `d`'s
/// parent in the reply forest (any negative value marks a root). `num_types` is the vocabulary
/// size. Returns the fitted model; deterministic given `rng` (the E-step runs in parallel but
/// folds sufficient statistics in document order, as in `fit_ctm`).
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
    mut on_progress: F,
    rng: &mut R,
) -> ReplyTmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // seed each topic from a random (non-empty) document's word distribution, smoothed — an
    // LDA/CTM-style init that breaks symmetry toward real word clusters (random-uniform init
    // gets stuck in merged-topic optima).
    let nonempty: Vec<usize> = (0..d).filter(|&i| !sparse[i].0.is_empty()).collect();
    let mut beta = vec![vec![1.0f64 / num_types as f64; num_types]; k];
    if !nonempty.is_empty() {
        for row in beta.iter_mut() {
            let di = nonempty[(rng.gen::<f64>() * nonempty.len() as f64) as usize % nonempty.len()];
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

    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut anchor = vec![vec![0.0f64; km1]; num_groups.max(1)];
    // field hyperparameters; a = 1 - kappa
    let mut a = 0.7f64;
    let mut sigma2 = 1.0f64;
    let mut p0 = 1.0f64;

    let mut bound_history: Vec<f64> = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;

    for em in 0..em_iters {
        em_iters_run = em + 1;
        let kappa = 1.0 - a;

        // per-document-type inverse prior covariance (diagonal) and its half-logdet
        let siginv_diag = |var: f64| -> (Vec<f64>, f64) {
            let mut s = vec![0.0f64; km1 * km1];
            for i in 0..km1 {
                s[i * km1 + i] = 1.0 / var;
            }
            (s, 0.5 * km1 as f64 * var.ln())
        };
        let (siginv_edge, ent_edge) = siginv_diag(sigma2);
        let (siginv_root, ent_root) = siginv_diag(p0);

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
                let res = ctm_hpb(&opt, &beta, words, counts, &mu_d, siginv, entropy, true);
                let nu_diag: Vec<f64> = (0..km1).map(|i| res.nu[i * km1 + i]).collect();
                (di, opt, nu_diag, res.phi, res.bound)
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
        for (di, opt, nu_diag, phi, _) in &results {
            let di = *di;
            lambda[di] = opt.clone();
            nu_diag_store[di] = nu_diag.clone();
            let words = &sparse[di].0;
            for (wi, &w) in words.iter().enumerate() {
                for (t, brow) in beta_ss.iter_mut().enumerate() {
                    brow[w] += phi[t][wi];
                }
            }
        }

        if em_tol > 0.0 && bound_history.len() >= 2 {
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
            let obs: Vec<Vec<f64>> = (0..km1)
                .map(|i| {
                    lambda
                        .iter()
                        .enumerate()
                        .map(|(di, l)| l[i] - anchor[groups[di]][i])
                        .collect()
                })
                .collect();
            let r: Vec<f64> = (0..d)
                .map(|di| {
                    let nd = &nu_diag_store[di];
                    (nd.iter().sum::<f64>() / km1 as f64).max(1e-6)
                })
                .collect();
            let fit = tree_field::fit(
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
            a = fit.a;
            sigma2 = fit.q.max(0.1);
            p0 = fit.p0.max(0.1);
        }
    }

    ReplyTmModel {
        num_topics: k,
        num_types,
        beta,
        lambda,
        anchor,
        kappa: 1.0 - a,
        sigma2,
        p0,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
    }
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
