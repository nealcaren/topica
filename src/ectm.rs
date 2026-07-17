//! ECTM: the Evolving Content Topic Model.
//!
//! An STM (logistic-normal variational) whose **content** (topic-word) model
//! carries a group-by-time interaction: the same stable topic is worded
//! differently across a document-level *group* covariate, and that difference
//! *evolves* across a discrete *time period*. Where SAGE/STM content covariates
//! give one β per group, ECTM gives one β per (group, period) cell, with a
//! first-order random-walk prior tying adjacent periods so sparse cells borrow
//! strength from their temporal neighbours instead of fragmenting the topic.
//!
//! ## Content model
//!
//! For topic `k`, word `v`, group `g`, period `t`, an unnormalized log word rate
//!
//! ```text
//!   η_{k,g,t,v} = m_v + κT_{k,v} + κKP_{k,t,v} + κKG_{k,g,v} + κKGP_{k,g,t,v}
//!   β_{k,g,t,v} = softmax_v( η_{k,g,t,v} )
//! ```
//!
//! mirrors the proposal's `m_v + κ_kv + f_kv(t) + γ_kvg + h_kvg(t)`:
//!
//! - `m_v`   — corpus background log word-frequency (fixed);
//! - `κT`    — topic baseline vocabulary (K×V);
//! - `κKP`   — the topic's shared temporal trajectory `f_kv(t)` (K×P×V),
//!             smoothed by a random walk across periods;
//! - `κKG`   — the average group deviation `γ_kvg` (K×G×V);
//! - `κKGP`  — the group-by-time deviation `h_kvg(t)` (K×G×P×V), the headline
//!             "changing lexical contrast", random-walk-smoothed across periods
//!             and shrunk harder toward zero.
//!
//! The κ are MAP-estimated (Gaussian L2 priors plus the random-walk difference
//! penalties) from the variational expected (topic×group×period×word) counts
//! between E-steps, by the same L-BFGS used for the SAGE content model. Levels
//! are softly identified by the L2 priors rather than hard reference/sum-to-zero
//! constraints; interpretable contrasts are taken on the normalized β scale.
//!
//! ## Prevalence
//!
//! The prevalence side is unchanged from STM: a logistic-normal regression
//! `μ_d = X_d γ` on document covariates, reusing the shared variational E-step
//! (`ctm_hpb`) and the ridge prevalence M-step (`fit_gamma_ridge`).

use rand::Rng;

use crate::ctm::{ctm_hpb, ctm_lhood_grad, HpbResult};
use crate::estimator::{Estimator, ModelFamily};
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse};
use crate::variational::{
    doc_sparse, fit_gamma_ridge, fit_gamma_ridge_from_ss, gamma_ss, laplace_estep, lbfgs_minimize,
    svi, LogisticNormalModel,
};

/// A fitted ECTM model.
pub struct EctmModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_groups: usize,
    pub num_periods: usize,
    /// Group-and-period-averaged topic-word β (K×V) — the headline `topic_word`.
    pub beta: Vec<Vec<f64>>,
    /// Per-cell topic-word distributions, indexed `[cell][k][v]` with
    /// `cell = g * num_periods + t`. This is the full content surface that
    /// `content_words` / `content_contrast` / `content_trajectory` read.
    pub content_beta: Vec<Vec<Vec<f64>>>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub lambda: Vec<Vec<f64>>,
    pub nu: Vec<Vec<f64>>,
    /// Prevalence coefficients γ (num_features × (K-1)), `Some` when prevalence
    /// covariates were supplied.
    pub gamma: Option<Vec<Vec<f64>>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
    pub diagonal: bool,
    /// SVI only: the relative L2 change of the content deviations κ at each
    /// content M-step solve. A run whose last entries are still large has not
    /// converged its content model (the headline divergences may be understated);
    /// empty for the batch fit. See `content_converged`.
    pub content_shift_history: Vec<f64>,
}

/// `cell = g * P + t`.
#[inline]
fn cell(g: usize, t: usize, num_periods: usize) -> usize {
    g * num_periods + t
}

/// Per-document topic proportions θ = softmax([η, 0]).
fn expeta(eta: &[f64]) -> Vec<f64> {
    let mut e: Vec<f64> = eta.iter().map(|x| x.exp()).collect();
    e.push(1.0);
    e
}

/// Build per-cell topic-word β (indexed `[cell][k][v]`) from the content
/// Seed-anchor the E-step content β: for a seeded (topic, word) the per-cell word
/// probability is inflated by `exp(seed_mean[topic*V+w])`, which directly raises
/// that topic's responsibility for the seed word (SeededLDA-style), forcing seed
/// tokens onto the seeded topic. Applied only to the β used for the E-step; the
/// M-step counts that result then reinforce the anchor, and the reported content
/// surface is rebuilt un-boosted from κ. Unseeded entries (and seed_mean all-zero)
/// return β unchanged. `ctm_lhood_grad`/`ctm_hpb` are column-wise in β[t][w], so a
/// non-row-normalized boost is safe.
fn seed_boost_beta(
    content_beta: &[Vec<Vec<f64>>],
    seed_mean: &[f64],
    v: usize,
) -> Vec<Vec<Vec<f64>>> {
    content_beta
        .iter()
        .map(|cell| {
            cell.iter()
                .enumerate()
                .map(|(topic, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(w, &b)| {
                            let s = seed_mean.get(topic * v + w).copied().unwrap_or(0.0);
                            if s != 0.0 {
                                b * s.exp()
                            } else {
                                b
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// deviations: `β_{g,t,k,v} = softmax_v(m_v + κT_k + κKP_{k,t} + κKG_{k,g} + κKGP_{k,g,t})`.
#[allow(clippy::too_many_arguments)]
fn build_content_beta(
    m: &[f64],
    kt: &[Vec<f64>],   // [K][V]
    kkp: &[Vec<f64>],  // [K*P][V]
    kkg: &[Vec<f64>],  // [K*G][V]
    kkgp: &[Vec<f64>], // [K*G*P][V]
    k: usize,
    g: usize,
    p: usize,
    v: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut out = vec![vec![vec![0.0f64; v]; k]; g * p];
    for topic in 0..k {
        for grp in 0..g {
            for per in 0..p {
                let c = cell(grp, per, p);
                let kp = topic * p + per;
                let kg = topic * g + grp;
                let kgp = (topic * g + grp) * p + per;
                let mut max = f64::NEG_INFINITY;
                let mut eta = vec![0.0f64; v];
                for w in 0..v {
                    let e = m[w] + kt[topic][w] + kkp[kp][w] + kkg[kg][w] + kkgp[kgp][w];
                    eta[w] = e;
                    if e > max {
                        max = e;
                    }
                }
                let mut z = 0.0;
                for w in 0..v {
                    z += (eta[w] - max).exp();
                }
                for w in 0..v {
                    out[c][topic][w] = (eta[w] - max).exp() / z;
                }
            }
        }
    }
    out
}

/// Convex blend of a content-deviation block toward its previous value:
/// `cur ← (1-rho)·old + rho·cur`. Used by the SVI path to damp the per-minibatch
/// κ solve toward the running global (the natural-parameter SVI average).
fn blend_kappa(cur: &mut [Vec<f64>], old: &[Vec<f64>], rho: f64) {
    for (c_row, o_row) in cur.iter_mut().zip(old) {
        for (c, &o) in c_row.iter_mut().zip(o_row) {
            *c = (1.0 - rho) * o + rho * *c;
        }
    }
}

/// One SVI content-κ update: scale the window's accumulated soft counts to corpus
/// magnitude (`D / window docs`), re-solve κ warm-started from the current global
/// (a few inner iterations is enough — the move is rho-damped), and blend the
/// candidate toward the previous κ. Snapshots κ before the solve so the blend uses
/// the pre-solve global.
#[allow(clippy::too_many_arguments)]
fn solve_and_blend_content(
    m: &[f64],
    kt: &mut [Vec<f64>],
    kkp: &mut [Vec<f64>],
    kkg: &mut [Vec<f64>],
    kkgp: &mut [Vec<f64>],
    content_ss: &[Vec<f64>],
    win_docs: f64,
    d: usize,
    k: usize,
    g: usize,
    p: usize,
    v: usize,
    sigma2: f64,
    rw_kp: f64,
    rw_kgp: f64,
    shrink_kgp: f64,
    content_l1: f64,
    rho: f64,
    inner_iters: usize,
    seed_mean: &[f64],
) -> f64 {
    let scale = d as f64 / win_docs.max(1.0);
    let counts: Vec<Vec<f64>> = content_ss
        .iter()
        .map(|r| r.iter().map(|&c| c * scale + 1e-8).collect())
        .collect();
    let kt_old = kt.to_vec();
    let kkp_old = kkp.to_vec();
    let kkg_old = kkg.to_vec();
    let kkgp_old = kkgp.to_vec();
    optimize_content(
        m,
        kt,
        kkp,
        kkg,
        kkgp,
        &counts,
        k,
        g,
        p,
        v,
        sigma2,
        rw_kp,
        rw_kgp,
        shrink_kgp,
        content_l1,
        inner_iters,
        seed_mean,
    );
    blend_kappa(kt, &kt_old, rho);
    blend_kappa(kkp, &kkp_old, rho);
    blend_kappa(kkg, &kkg_old, rho);
    blend_kappa(kkgp, &kkgp_old, rho);
    // Relative L2 change of the full κ vector (new vs pre-solve), the
    // content-model convergence signal: still-large at the end => not converged.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (cur, old) in [
        (&*kt, &kt_old),
        (&*kkp, &kkp_old),
        (&*kkg, &kkg_old),
        (&*kkgp, &kkgp_old),
    ] {
        for (cr, orow) in cur.iter().zip(old) {
            for (&c, &o) in cr.iter().zip(orow) {
                num += (c - o) * (c - o);
                den += o * o;
            }
        }
    }
    (num.sqrt()) / (den.sqrt() + 1e-12)
}

/// Soft-thresholding operator `sign(z)·max(|z|-g, 0)` — the proximal map of the
/// L1 penalty and the source of exact zeros under the sparse content prior.
fn soft_threshold(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// Minimize `f(x) + lam·Σ_i |x_i − anchor_i|` by FISTA with backtracking line
/// search (Beck & Teboulle 2009). `f_and_grad` supplies the smooth part's value
/// and gradient (already in minimization form). The L1 term is non-smooth at its
/// anchor, so plain L-BFGS cannot produce exact zeros; the proximal step
/// (soft-thresholding around `anchor`) does. Only invoked for the L1 content
/// prior (`lam > 0`); the L2 default keeps the original L-BFGS solve bit-exact.
fn fista_l1<F>(x0: Vec<f64>, f_and_grad: F, lam: f64, anchor: &[f64], max_iter: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    // Proximal-gradient step from `y` with inverse-Lipschitz step `t`.
    let prox_step = |y: &[f64], g: &[f64], t: f64| -> Vec<f64> {
        let thr = lam * t;
        (0..n)
            .map(|i| {
                let a = anchor.get(i).copied().unwrap_or(0.0);
                a + soft_threshold(y[i] - t * g[i] - a, thr)
            })
            .collect::<Vec<f64>>()
    };
    let mut x = x0.clone();
    let mut y = x0;
    let mut t_step = 1.0f64;
    let mut theta = 1.0f64;
    for _ in 0..max_iter {
        let (fy, gy) = f_and_grad(&y);
        // Backtracking: shrink the step until the quadratic upper bound holds.
        let mut step = t_step * 2.0;
        let x_new = loop {
            let cand = prox_step(&y, &gy, step);
            let (fc, _) = f_and_grad(&cand);
            let mut dot = 0.0;
            let mut sq = 0.0;
            for i in 0..n {
                let d = cand[i] - y[i];
                dot += gy[i] * d;
                sq += d * d;
            }
            if fc <= fy + dot + sq / (2.0 * step) + 1e-12 || step < 1e-12 {
                break cand;
            }
            step *= 0.5;
        };
        t_step = step;
        // Nesterov momentum.
        let theta_next = (1.0 + (1.0 + 4.0 * theta * theta).sqrt()) / 2.0;
        let beta = (theta - 1.0) / theta_next;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let d = x_new[i] - x[i];
            num += d * d;
            den += x[i] * x[i];
            y[i] = x_new[i] + beta * d;
        }
        x = x_new;
        theta = theta_next;
        if num.sqrt() / (den.sqrt() + 1e-12) < 1e-5 {
            break;
        }
    }
    x
}

/// MAP-update the content deviations κ from the variational expected
/// (topic×group×period×word) counts, then rebuild per-cell β.
///
/// `counts[k*(G*P) + cell][v]` are the soft token counts. The objective is the
/// multinomial-logit log-likelihood plus: an L2 prior on every κ (variance
/// `sigma2`); a first-order random-walk penalty on κKP across periods (within a
/// topic); and a random-walk penalty plus a tighter L2 on κKGP across periods
/// (within a topic×group). `rw_kp` / `rw_kgp` are the random-walk precisions
/// (1/τ²); `shrink_kgp` multiplies the κKGP L2 precision.
#[allow(clippy::too_many_arguments)]
fn optimize_content(
    m: &[f64],
    kt: &mut [Vec<f64>],
    kkp: &mut [Vec<f64>],
    kkg: &mut [Vec<f64>],
    kkgp: &mut [Vec<f64>],
    counts: &[Vec<f64>],
    k: usize,
    g: usize,
    p: usize,
    v: usize,
    sigma2: f64,
    rw_kp: f64,
    rw_kgp: f64,
    shrink_kgp: f64,
    content_l1: f64,
    max_iter: usize,
    seed_mean: &[f64],
) -> Vec<Vec<Vec<f64>>> {
    let n_t = k * v;
    let n_kp = k * p * v;
    let n_kg = k * g * v;
    let n_kgp = k * g * p * v;
    let off_kp = n_t;
    let off_kg = n_t + n_kp;
    let off_kgp = n_t + n_kp + n_kg;

    let totals: Vec<f64> = counts.iter().map(|row| row.iter().sum()).collect();

    let mut x0 = Vec::with_capacity(n_t + n_kp + n_kg + n_kgp);
    for r in kt.iter() {
        x0.extend_from_slice(r);
    }
    for r in kkp.iter() {
        x0.extend_from_slice(r);
    }
    for r in kkg.iter() {
        x0.extend_from_slice(r);
    }
    for r in kkgp.iter() {
        x0.extend_from_slice(r);
    }

    let inv_var = 1.0 / sigma2;

    // Smooth part of the negative log-posterior (to *minimize*): the
    // multinomial-logit NLL, the L2 priors, and the random-walk penalties. Under
    // the L1 (sparse Laplace) content prior the L2 stays as a small ridge for
    // identifiability and conditioning while FISTA adds the non-smooth |κ| term.
    let smooth = |flat: &[f64]| -> (f64, Vec<f64>) {
        {
            let kt_i = |t: usize, w: usize| flat[t * v + w];
            let kkp_i = |t: usize, per: usize, w: usize| flat[off_kp + (t * p + per) * v + w];
            let kkg_i = |t: usize, grp: usize, w: usize| flat[off_kg + (t * g + grp) * v + w];
            let kkgp_i = |t: usize, grp: usize, per: usize, w: usize| {
                flat[off_kgp + ((t * g + grp) * p + per) * v + w]
            };

            let mut value = 0.0f64;
            let mut grad = vec![0.0f64; flat.len()];

            // Multinomial-logit log-likelihood over (topic, group, period) cells.
            for topic in 0..k {
                for grp in 0..g {
                    for per in 0..p {
                        let c = cell(grp, per, p);
                        let cidx = topic * (g * p) + c;
                        let nkt = totals[cidx];
                        let mut max = f64::NEG_INFINITY;
                        let mut eta = vec![0.0f64; v];
                        for w in 0..v {
                            let e = m[w]
                                + kt_i(topic, w)
                                + kkp_i(topic, per, w)
                                + kkg_i(topic, grp, w)
                                + kkgp_i(topic, grp, per, w);
                            eta[w] = e;
                            if e > max {
                                max = e;
                            }
                        }
                        let mut z = 0.0;
                        for w in 0..v {
                            z += (eta[w] - max).exp();
                        }
                        let log_z = max + z.ln();
                        for w in 0..v {
                            let n = counts[cidx][w];
                            value += n * (eta[w] - log_z);
                            let beta = (eta[w] - log_z).exp();
                            let resid = n - nkt * beta; // ∂LL/∂η
                            grad[topic * v + w] += resid;
                            grad[off_kp + (topic * p + per) * v + w] += resid;
                            grad[off_kg + (topic * g + grp) * v + w] += resid;
                            grad[off_kgp + ((topic * g + grp) * p + per) * v + w] += resid;
                        }
                    }
                }
            }

            // L2 priors. κT (the topic baseline, indices 0..n_t) is pulled toward
            // `seed_mean` -- zero everywhere except seeded (topic, word) entries,
            // which anchor a topic's shared vocabulary. The L2 defends the shift so
            // the M-step cannot cancel it. κKP/κKG keep a plain zero-mean L2.
            for i in 0..n_t {
                let resid = flat[i] - seed_mean.get(i).copied().unwrap_or(0.0);
                value -= 0.5 * inv_var * resid * resid;
                grad[i] -= inv_var * resid;
            }
            for i in n_t..off_kgp {
                let xi = flat[i];
                value -= 0.5 * inv_var * xi * xi;
                grad[i] -= inv_var * xi;
            }
            let inv_var_kgp = inv_var * shrink_kgp;
            for i in off_kgp..flat.len() {
                let xi = flat[i];
                value -= 0.5 * inv_var_kgp * xi * xi;
                grad[i] -= inv_var_kgp * xi;
            }

            // Random-walk penalty on κKP across periods (within a topic):
            //   (rw_kp/2) Σ_{t=1}^{P-1} (x_{k,t} - x_{k,t-1})².
            if p >= 2 {
                for topic in 0..k {
                    for per in 1..p {
                        for w in 0..v {
                            let a = off_kp + (topic * p + per) * v + w;
                            let b = off_kp + (topic * p + (per - 1)) * v + w;
                            let diff = flat[a] - flat[b];
                            value -= 0.5 * rw_kp * diff * diff;
                            grad[a] -= rw_kp * diff;
                            grad[b] += rw_kp * diff;
                        }
                    }
                }
                // Random-walk penalty on κKGP across periods (within topic×group).
                for topic in 0..k {
                    for grp in 0..g {
                        for per in 1..p {
                            for w in 0..v {
                                let a = off_kgp + ((topic * g + grp) * p + per) * v + w;
                                let b = off_kgp + ((topic * g + grp) * p + (per - 1)) * v + w;
                                let diff = flat[a] - flat[b];
                                value -= 0.5 * rw_kgp * diff * diff;
                                grad[a] -= rw_kgp * diff;
                                grad[b] += rw_kgp * diff;
                            }
                        }
                    }
                }
            }

            (-value, grad.iter().map(|gv| -gv).collect())
        }
    };

    let x = if content_l1 > 0.0 {
        // L1 (sparse Laplace) prior on the content deviations: κT is pulled
        // toward `seed_mean` (0 except at seeded entries), every other block
        // toward 0. Solved by FISTA so the deviations reach exact zeros — sparse,
        // readable top-word lists and sharp per-cell Δ contrasts (SAGE-style).
        let mut anchor = vec![0.0f64; n_t + n_kp + n_kg + n_kgp];
        for (i, a) in anchor.iter_mut().enumerate().take(n_t) {
            *a = seed_mean.get(i).copied().unwrap_or(0.0);
        }
        fista_l1(x0, smooth, content_l1, &anchor, max_iter.max(200))
    } else {
        lbfgs_minimize(x0, smooth, max_iter, 7, 1e-4)
    };

    for t in 0..k {
        kt[t].copy_from_slice(&x[t * v..(t + 1) * v]);
    }
    for i in 0..(k * p) {
        let off = off_kp + i * v;
        kkp[i].copy_from_slice(&x[off..off + v]);
    }
    for i in 0..(k * g) {
        let off = off_kg + i * v;
        kkg[i].copy_from_slice(&x[off..off + v]);
    }
    for i in 0..(k * g * p) {
        let off = off_kgp + i * v;
        kkgp[i].copy_from_slice(&x[off..off + v]);
    }
    build_content_beta(m, kt, kkp, kkg, kkgp, k, g, p, v)
}

/// Fit ECTM by variational EM.
///
/// `groups[d]` ∈ 0..num_groups and `periods[d]` ∈ 0..num_periods give the
/// content covariate cell of document `d`. `prevalence` (optional, D×F with an
/// intercept column already prepended) is the STM prevalence design. `sigma2` is
/// the L2 prior variance on the content κ; `rw_kp` / `rw_kgp` are the
/// random-walk precisions; `shrink_kgp` tightens the κKGP L2.
#[allow(clippy::too_many_arguments)]
pub fn fit_ectm<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    groups: &[usize],
    num_groups: usize,
    periods: &[usize],
    num_periods: usize,
    em_iters: usize,
    em_tol: f64,
    sigma_shrink: f64,
    prevalence: Option<&[Vec<f64>]>,
    sigma2: f64,
    rw_kp: f64,
    rw_kgp: f64,
    shrink_kgp: f64,
    content_l1: f64,
    keep_nu: bool,
    diagonal: bool,
    init_spectral: bool,
    seed_mean: &[f64],
    rng: &mut R,
) -> EctmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let g = num_groups;
    let p = num_periods;
    let nf = prevalence.map(|x| x[0].len());

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Base per-topic β to break across-topic symmetry of κT. With `init_spectral`
    // (the default) we use the same deterministic anchor-word spectral init as
    // STM/CTM/STS (issue #216/#220): a random base β leaves the content model
    // multimodal, and most seeds collapse to flat/no-group content. Falls back to
    // a seeded random β if the spectral solve is unavailable (e.g. K > vocab).
    let random_beta = |rng: &mut R| {
        let mut b = vec![vec![0.0f64; num_types]; k];
        for row in b.iter_mut() {
            let mut s = 0.0;
            for x in row.iter_mut() {
                *x = 1.0 + rng.gen::<f64>();
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        b
    };
    let mut beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, num_types).unwrap_or_else(|| random_beta(rng))
    } else {
        random_beta(rng)
    };

    // Background m_v = corpus log word-frequency (+1 smoothing).
    let mut m_bg = vec![0.0f64; num_types];
    {
        let mut freq = vec![1.0f64; num_types];
        let mut total = num_types as f64;
        for doc in docs {
            for &w in doc {
                freq[w as usize] += 1.0;
                total += 1.0;
            }
        }
        for v in 0..num_types {
            m_bg[v] = (freq[v] / total).ln();
        }
    }

    // Content deviations. Seed κT from the random β (as STM does) so the topics
    // start differentiated; all higher-order deviations start at zero.
    let mut kt = vec![vec![0.0f64; num_types]; k];
    let mut kkp = vec![vec![0.0f64; num_types]; k * p];
    let mut kkg = vec![vec![0.0f64; num_types]; k * g];
    let mut kkgp = vec![vec![0.0f64; num_types]; k * g * p];
    for t in 0..k {
        for v in 0..num_types {
            kt[t][v] = beta[t][v].max(1e-12).ln() - m_bg[v];
        }
    }
    let mut content_beta = build_content_beta(&m_bg, &kt, &kkp, &kkg, &kkgp, k, g, p, num_types);

    let mut mu_shared = vec![0.0f64; km1];
    let mut gamma: Option<Vec<Vec<f64>>> = nf.map(|f| vec![vec![0.0f64; km1]; f]);
    let mut sigma = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; km1 * km1]; d]
    } else {
        Vec::new()
    };

    let doc_mu = |di: usize, gamma: &Option<Vec<Vec<f64>>>, mu_shared: &[f64]| -> Vec<f64> {
        match (prevalence, gamma) {
            (Some(x), Some(gm)) => (0..km1)
                .map(|t| x[di].iter().zip(gm).map(|(xi, gr)| xi * gr[t]).sum())
                .collect(),
            _ => mu_shared.to_vec(),
        }
    };

    let mut bound_history: Vec<f64> = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;
    let num_cells = g * p;

    for em in 0..em_iters {
        em_iters_run = em + 1;
        let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, km1);
            spd_inverse(&s, km1).unwrap()
        });
        let entropy = match cholesky(&sigma, km1) {
            Some(l) => half_logdet(&l, km1),
            None => 0.0,
        };

        // Soft expected counts per (topic × cell, word).
        let mut content_ss = vec![vec![1e-8f64; num_types]; k * num_cells];
        let mut sigma_ss = vec![0.0f64; km1 * km1];
        let mut lambda_sum = vec![0.0f64; km1];

        let chunk = (128 * 1024 * 1024 / (km1 * km1 * 8).max(1))
            .max(256)
            .min(d.max(1));
        // Seed-anchored E-step: boost seed-word responsibilities toward their
        // seeded topic (no-op when there are no seeds; preserves bit-exactness).
        let has_seed = seed_mean.iter().any(|&x| x != 0.0);
        let estep_beta = if has_seed {
            Some(seed_boost_beta(&content_beta, seed_mean, num_types))
        } else {
            None
        };
        let beta_src: &[Vec<Vec<f64>>] = estep_beta.as_deref().unwrap_or(&content_beta);
        let mut total_bound = 0.0f64;
        let mut base = 0usize;
        while base < d {
            let end = (base + chunk).min(d);
            let chunk_results: Vec<(usize, (Vec<f64>, HpbResult))> =
                laplace_estep(&sparse[base..end], |local_di, words, counts| {
                    let di = base + local_di;
                    let mu_d = doc_mu(di, &gamma, &mu_shared);
                    let c = cell(groups[di], periods[di], p);
                    let beta_doc: &[Vec<f64>] = &beta_src[c];
                    let opt = lbfgs_minimize(
                        lambda[di].clone(),
                        |eta| ctm_lhood_grad(eta, beta_doc, words, counts, &mu_d, &siginv),
                        40,
                        7,
                        1e-5,
                    );
                    let res = ctm_hpb(
                        &opt, beta_doc, words, counts, &mu_d, &siginv, entropy, diagonal,
                    );
                    (opt, res)
                });

            for (local_di, (opt, res)) in &chunk_results {
                let di = base + *local_di;
                total_bound += res.bound;
                let words = &sparse[di].0;
                lambda[di] = opt.clone();
                let c = cell(groups[di], periods[di], p);
                for (wi, &w) in words.iter().enumerate() {
                    for t in 0..k {
                        content_ss[t * num_cells + c][w] += res.phi[t][wi];
                    }
                }
                if keep_nu {
                    nu_store[di] = res.nu.clone();
                }
                for i in 0..km1 {
                    lambda_sum[i] += opt[i];
                    for j in 0..km1 {
                        sigma_ss[i * km1 + j] += res.nu[i * km1 + j];
                    }
                }
            }
            base = end;
        }

        bound_history.push(total_bound);
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }

        // Prevalence M-step (ridge) or shared prior mean.
        if let Some(x) = prevalence {
            gamma = Some(fit_gamma_ridge(x, &lambda, nf.unwrap(), km1, 1e-6));
        } else {
            for i in 0..km1 {
                mu_shared[i] = lambda_sum[i] / d as f64;
            }
        }

        // Σ update with shrinkage toward the diagonal.
        let mus: Vec<Vec<f64>> = (0..d).map(|di| doc_mu(di, &gamma, &mu_shared)).collect();
        {
            use rayon::prelude::*;
            let df = d as f64;
            sigma.par_chunks_mut(km1).enumerate().for_each(|(i, row)| {
                for (j, sij) in row.iter_mut().enumerate() {
                    let mut cross = 0.0;
                    for (di, li) in lambda.iter().enumerate() {
                        cross += (li[i] - mus[di][i]) * (li[j] - mus[di][j]);
                    }
                    *sij = (sigma_ss[i * km1 + j] + cross) / df;
                }
            });
        }
        if sigma_shrink > 0.0 {
            for i in 0..km1 {
                for j in 0..km1 {
                    if i != j {
                        sigma[i * km1 + j] *= 1.0 - sigma_shrink;
                    }
                }
            }
        }

        // Content M-step.
        content_beta = optimize_content(
            &m_bg,
            &mut kt,
            &mut kkp,
            &mut kkg,
            &mut kkgp,
            &content_ss,
            k,
            g,
            p,
            num_types,
            sigma2,
            rw_kp,
            rw_kgp,
            shrink_kgp,
            content_l1,
            20,
            seed_mean,
        );
    }

    // Reported β is the cell-averaged topic-word.
    for t in 0..k {
        for v in 0..num_types {
            let mut s = 0.0;
            for c in 0..num_cells {
                s += content_beta[c][t][v];
            }
            beta[t][v] = s / num_cells as f64;
        }
    }

    EctmModel {
        num_topics: k,
        num_types,
        num_groups: g,
        num_periods: p,
        beta,
        content_beta,
        mu: mu_shared,
        sigma,
        lambda,
        nu: nu_store,
        gamma,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
        diagonal,
        content_shift_history: Vec::new(),
    }
}

/// Fit ECTM by **stochastic** variational EM (minibatch SVI) for corpora too
/// large for the full-batch `fit_ectm`. Each step subsamples `batch_size`
/// documents, runs the warm-started Laplace E-step on them, scales the minibatch
/// sufficient statistics to corpus magnitude (`D/|B|`), and moves every global
/// toward the minibatch estimate with the Robbins-Monro rate
/// `rho_t = (tau + t)^(-kappa)`. The closed-form globals (μ/Σ, and γ via blended
/// ridge sufficient statistics) blend directly; the non-conjugate content κ is
/// re-solved by `optimize_content` on the scaled minibatch counts (warm-started
/// from the current global) and the result blended toward the previous κ. A final
/// full E-step gives every document its λ/ν posterior and the corpus bound.
///
/// `epochs` is the number of passes over the corpus; the remaining arguments
/// match `fit_ectm`. The minibatch order is drawn from `rng`, so the fit is
/// seed-reproducible (not bit-exact across seeds, unlike the spectral batch fit).
#[allow(clippy::too_many_arguments)]
pub fn fit_ectm_svi<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    groups: &[usize],
    num_groups: usize,
    periods: &[usize],
    num_periods: usize,
    epochs: usize,
    batch_size: usize,
    tau: f64,
    kappa: f64,
    content_every: usize,
    sigma_shrink: f64,
    prevalence: Option<&[Vec<f64>]>,
    sigma2: f64,
    rw_kp: f64,
    rw_kgp: f64,
    shrink_kgp: f64,
    content_l1: f64,
    keep_nu: bool,
    diagonal: bool,
    init_spectral: bool,
    seed_mean: &[f64],
    rng: &mut R,
) -> EctmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let g = num_groups;
    let p = num_periods;
    let num_cells = g * p;
    let nf = prevalence.map(|x| x[0].len());

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // --- initialization: identical to fit_ectm (spectral base, m_v, κ seed) ---
    let random_beta = |rng: &mut R| {
        let mut b = vec![vec![0.0f64; num_types]; k];
        for row in b.iter_mut() {
            let mut s = 0.0;
            for x in row.iter_mut() {
                *x = 1.0 + rng.gen::<f64>();
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        b
    };
    let mut beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, num_types).unwrap_or_else(|| random_beta(rng))
    } else {
        random_beta(rng)
    };

    let mut m_bg = vec![0.0f64; num_types];
    {
        let mut freq = vec![1.0f64; num_types];
        let mut total = num_types as f64;
        for doc in docs {
            for &w in doc {
                freq[w as usize] += 1.0;
                total += 1.0;
            }
        }
        for v in 0..num_types {
            m_bg[v] = (freq[v] / total).ln();
        }
    }

    let mut kt = vec![vec![0.0f64; num_types]; k];
    let mut kkp = vec![vec![0.0f64; num_types]; k * p];
    let mut kkg = vec![vec![0.0f64; num_types]; k * g];
    let mut kkgp = vec![vec![0.0f64; num_types]; k * g * p];
    for t in 0..k {
        for v in 0..num_types {
            kt[t][v] = beta[t][v].max(1e-12).ln() - m_bg[v];
        }
    }
    let mut content_beta = build_content_beta(&m_bg, &kt, &kkp, &kkg, &kkgp, k, g, p, num_types);

    let mut mu_shared = vec![0.0f64; km1];
    let mut gamma: Option<Vec<Vec<f64>>> = nf.map(|f| vec![vec![0.0f64; km1]; f]);
    let mut sigma = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; km1 * km1]; d]
    } else {
        Vec::new()
    };

    let doc_mu = |di: usize, gamma: &Option<Vec<Vec<f64>>>, mu_shared: &[f64]| -> Vec<f64> {
        match (prevalence, gamma) {
            (Some(x), Some(gm)) => (0..km1)
                .map(|t| x[di].iter().zip(gm).map(|(xi, gr)| xi * gr[t]).sum())
                .collect(),
            _ => mu_shared.to_vec(),
        }
    };

    // Running ridge sufficient statistics for γ, blended across minibatches.
    let mut s_xx = nf.map(|f| vec![0.0f64; f * f]);
    let mut s_xl = nf.map(|f| vec![0.0f64; f * km1]);

    let batch = batch_size.clamp(1, d.max(1));
    // The content κ-solve (optimize_content over all K·G·P·V cells) costs the same
    // regardless of minibatch size, so re-solving it every minibatch dominates the
    // ECTM SVI cost. Instead accumulate the content sufficient statistics across
    // `content_every` minibatches and re-solve κ once per window, while μ/Σ/γ (cheap,
    // closed-form) still update every minibatch. `content_every == 0` means "once per
    // epoch". The κ counts persist across the window and are scaled by D/(window docs)
    // at solve time.
    let mb_per_epoch = d.div_ceil(batch).max(1);
    // Safer default (content_every == 0): target a floor of ~TARGET_KSOLVES content
    // M-step solves so the content model develops even at a low epoch count, while
    // never solving less often than once per epoch. A naive low-epoch call would
    // otherwise under-develop κ and silently understate the divergences.
    const TARGET_KSOLVES: usize = 10;
    let solve_every = if content_every == 0 {
        let total_mb = mb_per_epoch.saturating_mul(epochs).max(1);
        (total_mb / TARGET_KSOLVES).clamp(1, mb_per_epoch)
    } else {
        content_every
    };
    // The κ schedule is indexed by the number of κ-solves, not minibatches: κ
    // updates rarely (once per window), each on a near-full-corpus estimate, so it
    // takes a strong early step (rho_k from a small tau) and does a fuller inner
    // solve when it solves rarely. Indexing on t_step (as the cheap globals do)
    // would creep at rho ~ 0.04 and never develop the content deviations.
    let kiters = if solve_every >= 8 { 20 } else { 6 };
    let mut content_ss = vec![vec![0.0f64; num_types]; k * num_cells];
    let mut content_shift_history: Vec<f64> = Vec::new();
    let mut win_docs = 0.0f64;
    let mut steps_since_solve = 0usize;
    let mut n_ksolve = 0usize;
    let mut t_step: usize = 0;

    for _epoch in 0..epochs {
        let order = svi::shuffled_order(d, rng);
        for chunk in order.chunks(batch) {
            t_step += 1;
            let rho = svi::rho(tau, kappa, t_step);

            let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
                let mut s = sigma.clone();
                make_diagonally_dominant(&mut s, km1);
                spd_inverse(&s, km1).unwrap()
            });
            let entropy = match cholesky(&sigma, km1) {
                Some(l) => half_logdet(&l, km1),
                None => 0.0,
            };

            // E-step over the minibatch (warm-started from the stored λ). Content
            // sufficient statistics accumulate into the persistent window buffer.
            let mut sigma_ss = vec![0.0f64; km1 * km1];
            let mut lambda_sum = vec![0.0f64; km1];
            let mut etas: Vec<Vec<f64>> = Vec::with_capacity(chunk.len());
            let has_seed = seed_mean.iter().any(|&x| x != 0.0);
            let estep_beta = if has_seed {
                Some(seed_boost_beta(&content_beta, seed_mean, num_types))
            } else {
                None
            };
            let beta_src: &[Vec<Vec<f64>>] = estep_beta.as_deref().unwrap_or(&content_beta);
            for &di in chunk {
                let mu_d = doc_mu(di, &gamma, &mu_shared);
                let c = cell(groups[di], periods[di], p);
                let beta_doc: &[Vec<f64>] = &beta_src[c];
                let words = &sparse[di].0;
                let counts = &sparse[di].1;
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| ctm_lhood_grad(eta, beta_doc, words, counts, &mu_d, &siginv),
                    40,
                    7,
                    1e-5,
                );
                let res = ctm_hpb(
                    &opt, beta_doc, words, counts, &mu_d, &siginv, entropy, diagonal,
                );
                for (wi, &w) in words.iter().enumerate() {
                    for t in 0..k {
                        content_ss[t * num_cells + c][w] += res.phi[t][wi];
                    }
                }
                for i in 0..km1 {
                    lambda_sum[i] += opt[i];
                    for j in 0..km1 {
                        sigma_ss[i * km1 + j] += res.nu[i * km1 + j];
                    }
                }
                lambda[di] = opt.clone();
                if keep_nu {
                    nu_store[di] = res.nu.clone();
                }
                etas.push(opt);
            }
            let bsz = chunk.len() as f64;
            let scale = d as f64 / bsz;
            win_docs += bsz;
            steps_since_solve += 1;

            // Prevalence γ (blend ridge sufficient stats) or shared mean μ.
            if let (Some(x), Some(f)) = (prevalence, nf) {
                let xmb: Vec<Vec<f64>> = chunk.iter().map(|&di| x[di].clone()).collect();
                let (xx_b, xl_b) = gamma_ss(&xmb, &etas, f, km1);
                let sxx = s_xx.as_mut().unwrap();
                let sxl = s_xl.as_mut().unwrap();
                for (s, &b) in sxx.iter_mut().zip(&xx_b) {
                    *s = (1.0 - rho) * *s + rho * scale * b;
                }
                for (s, &b) in sxl.iter_mut().zip(&xl_b) {
                    *s = (1.0 - rho) * *s + rho * scale * b;
                }
                gamma = Some(fit_gamma_ridge_from_ss(sxx, sxl, f, km1, 1e-6));
            } else {
                for i in 0..km1 {
                    mu_shared[i] = (1.0 - rho) * mu_shared[i] + rho * (lambda_sum[i] / bsz);
                }
            }

            // Σ: blend toward the minibatch covariance estimate.
            let mus: Vec<Vec<f64>> = chunk
                .iter()
                .map(|&di| doc_mu(di, &gamma, &mu_shared))
                .collect();
            for i in 0..km1 {
                for j in 0..km1 {
                    let mut cross = 0.0;
                    for (eta, mu) in etas.iter().zip(&mus) {
                        cross += (eta[i] - mu[i]) * (eta[j] - mu[j]);
                    }
                    let shat = (sigma_ss[i * km1 + j] + cross) / bsz;
                    let mut newv = (1.0 - rho) * sigma[i * km1 + j] + rho * shat;
                    if sigma_shrink > 0.0 && i != j {
                        newv *= 1.0 - sigma_shrink;
                    }
                    sigma[i * km1 + j] = newv;
                }
            }

            // Content κ: once per `solve_every` minibatches, re-solve on the
            // window's accumulated counts (scaled to corpus magnitude, warm-started
            // from the current global), then blend the candidate toward the old
            // global and reset the window.
            if steps_since_solve >= solve_every {
                n_ksolve += 1;
                let rho_k = svi::rho(1.0, kappa, n_ksolve);
                let shift = solve_and_blend_content(
                    &m_bg,
                    &mut kt,
                    &mut kkp,
                    &mut kkg,
                    &mut kkgp,
                    &content_ss,
                    win_docs,
                    d,
                    k,
                    g,
                    p,
                    num_types,
                    sigma2,
                    rw_kp,
                    rw_kgp,
                    shrink_kgp,
                    content_l1,
                    rho_k,
                    kiters,
                    seed_mean,
                );
                content_shift_history.push(shift);
                content_beta =
                    build_content_beta(&m_bg, &kt, &kkp, &kkg, &kkgp, k, g, p, num_types);
                for row in content_ss.iter_mut() {
                    for c in row.iter_mut() {
                        *c = 0.0;
                    }
                }
                win_docs = 0.0;
                steps_since_solve = 0;
            }
        }
    }

    // Flush any partial window so the last minibatches inform the content model.
    if steps_since_solve > 0 {
        n_ksolve += 1;
        let rho_k = svi::rho(1.0, kappa, n_ksolve);
        let shift = solve_and_blend_content(
            &m_bg,
            &mut kt,
            &mut kkp,
            &mut kkg,
            &mut kkgp,
            &content_ss,
            win_docs,
            d,
            k,
            g,
            p,
            num_types,
            sigma2,
            rw_kp,
            rw_kgp,
            shrink_kgp,
            content_l1,
            rho_k,
            kiters,
            seed_mean,
        );
        content_shift_history.push(shift);
        content_beta = build_content_beta(&m_bg, &kt, &kkp, &kkg, &kkgp, k, g, p, num_types);
    }

    // Final full E-step with the converged globals: populate every doc's λ/ν and
    // the corpus bound. Chunked + parallel like fit_ectm, summed in document order.
    let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
        let mut s = sigma.clone();
        make_diagonally_dominant(&mut s, km1);
        spd_inverse(&s, km1).unwrap()
    });
    let entropy = match cholesky(&sigma, km1) {
        Some(l) => half_logdet(&l, km1),
        None => 0.0,
    };
    let chunk = (128 * 1024 * 1024 / (km1 * km1 * 8).max(1))
        .max(256)
        .min(d.max(1));
    let mut total_bound = 0.0f64;
    let has_seed = seed_mean.iter().any(|&x| x != 0.0);
    let final_estep_beta = if has_seed {
        Some(seed_boost_beta(&content_beta, seed_mean, num_types))
    } else {
        None
    };
    let beta_src: &[Vec<Vec<f64>>] = final_estep_beta.as_deref().unwrap_or(&content_beta);
    let mut base = 0usize;
    while base < d {
        let end = (base + chunk).min(d);
        let chunk_results: Vec<(usize, (Vec<f64>, HpbResult))> =
            laplace_estep(&sparse[base..end], |local_di, words, counts| {
                let di = base + local_di;
                let mu_d = doc_mu(di, &gamma, &mu_shared);
                let c = cell(groups[di], periods[di], p);
                let beta_doc: &[Vec<f64>] = &beta_src[c];
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| ctm_lhood_grad(eta, beta_doc, words, counts, &mu_d, &siginv),
                    40,
                    7,
                    1e-5,
                );
                let res = ctm_hpb(
                    &opt, beta_doc, words, counts, &mu_d, &siginv, entropy, diagonal,
                );
                (opt, res)
            });
        for (local_di, (opt, res)) in &chunk_results {
            let di = base + *local_di;
            total_bound += res.bound;
            lambda[di] = opt.clone();
            if keep_nu {
                nu_store[di] = res.nu.clone();
            }
        }
        base = end;
    }

    // Reported β is the cell-averaged topic-word.
    for t in 0..k {
        for v in 0..num_types {
            let mut s = 0.0;
            for c in 0..num_cells {
                s += content_beta[c][t][v];
            }
            beta[t][v] = s / num_cells as f64;
        }
    }

    EctmModel {
        num_topics: k,
        num_types,
        num_groups: g,
        num_periods: p,
        beta,
        content_beta,
        mu: mu_shared,
        sigma,
        lambda,
        nu: nu_store,
        gamma,
        bound: total_bound,
        bound_history: vec![total_bound],
        converged: true,
        em_iters_run: epochs,
        diagonal,
        content_shift_history,
    }
}

impl EctmModel {
    /// `cell = g * num_periods + t`.
    #[inline]
    pub fn cell(&self, g: usize, t: usize) -> usize {
        g * self.num_periods + t
    }

    /// Per-document topic proportions θ = softmax([η, 0]).
    pub fn doc_topics(&self) -> Vec<Vec<f64>> {
        self.lambda
            .iter()
            .map(|eta| {
                let e = expeta(eta);
                let s: f64 = e.iter().sum();
                e.iter().map(|x| x / s).collect()
            })
            .collect()
    }
}

impl Estimator for EctmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.beta.clone()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topics()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect()
    }

    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::LogisticNormal
    }
}

impl LogisticNormalModel for EctmModel {
    fn eta_dim(&self) -> usize {
        self.num_topics - 1
    }

    fn eta_mean(&self) -> &[Vec<f64>] {
        &self.lambda
    }

    fn eta_cov(&self) -> &[Vec<f64>] {
        &self.nu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // One topic, two groups, three periods, vocab {0,1,2,3}. Both groups start
    // sharing words {0,1}; over periods group 1 drifts toward words {2,3} while
    // group 0 stays on {0,1}. ECTM should recover a group contrast on {2,3} that
    // GROWS across periods (scenario 4: a difference that grows over time).
    fn growing_contrast_corpus() -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        let mut periods = Vec::new();
        // period 0: groups identical on {0,1}; period 1: group1 mixes; period 2: group1 all {2,3}.
        let g1_by_period: [&[u32]; 3] = [&[0, 1, 0, 1], &[0, 1, 2, 3], &[2, 3, 2, 3]];
        for rep in 0..80 {
            for per in 0..3 {
                docs.push(vec![0u32, 1, 0, 1]);
                groups.push(0usize);
                periods.push(per);
                let _ = rep;
                docs.push(g1_by_period[per].to_vec());
                groups.push(1usize);
                periods.push(per);
            }
        }
        (docs, groups, periods)
    }

    fn fit_small_init(seed: u64, init_spectral: bool) -> EctmModel {
        let (docs, groups, periods) = growing_contrast_corpus();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        fit_ectm(
            &docs,
            2,
            4,
            &groups,
            2,
            &periods,
            3,
            60,
            0.0,
            0.0,
            None,
            1.0,
            5.0,
            5.0,
            2.0,
            0.0,
            true,
            false,
            init_spectral,
            &[],
            &mut rng,
        )
    }

    // Default path is spectral init.
    fn fit_small(seed: u64) -> EctmModel {
        fit_small_init(seed, true)
    }

    #[test]
    fn recovers_growing_group_contrast() {
        let m = fit_small(1);
        // Find the topic that group 1 uses for words {2,3} in the last period.
        // The group-1-on-{2,3} mass should grow from period 0 to period 2.
        // Use the topic whose period-2 group-1 mass on {2,3} is largest.
        let mut best_topic = 0;
        let mut best_mass = -1.0;
        for k in 0..m.num_topics {
            let c2 = m.cell(1, 2);
            let mass = m.content_beta[c2][k][2] + m.content_beta[c2][k][3];
            if mass > best_mass {
                best_mass = mass;
                best_topic = k;
            }
        }
        let k = best_topic;
        let contrast = |per: usize| -> f64 {
            let cg1 = m.cell(1, per);
            let cg0 = m.cell(0, per);
            // group1 minus group0 mass on words {2,3}.
            (m.content_beta[cg1][k][2] + m.content_beta[cg1][k][3])
                - (m.content_beta[cg0][k][2] + m.content_beta[cg0][k][3])
        };
        let c0 = contrast(0);
        let c2 = contrast(2);
        assert!(
            c2 > c0 + 0.2,
            "contrast should grow across periods: period0={c0:.3}, period2={c2:.3}"
        );
        assert!(
            c2 > 0.3,
            "final-period contrast should be substantial: {c2:.3}"
        );
    }

    #[test]
    fn spectral_init_is_seed_independent() {
        // The spectral base (issue #220) is deterministic, so the whole fit is
        // bit-exact: different seeds give identical content β (no random-base
        // multimodal collapse). This is the ECTM half of #216.
        let a = fit_small_init(7, true);
        let b = fit_small_init(8, true);
        for c in 0..a.content_beta.len() {
            for k in 0..a.num_topics {
                assert_eq!(
                    a.content_beta[c][k], b.content_beta[c][k],
                    "spectral init must be seed-independent (cell {c}, topic {k})"
                );
            }
        }
    }

    #[test]
    fn random_init_is_seeded_not_fixed() {
        // With init_spectral=false the base β is seeded random: same seed
        // reproduces, different seeds differ (the pre-#220 behavior, still
        // available via init="random").
        let a = fit_small_init(7, false);
        let b = fit_small_init(7, false);
        for c in 0..a.content_beta.len() {
            for k in 0..a.num_topics {
                assert_eq!(a.content_beta[c][k], b.content_beta[c][k]);
            }
        }
        let cc = fit_small_init(8, false);
        let any_diff = (0..a.content_beta.len())
            .any(|c| (0..a.num_topics).any(|k| a.content_beta[c][k] != cc.content_beta[c][k]));
        assert!(any_diff, "different seeds should differ under random init");
    }

    #[test]
    fn ectm_conforms() {
        let m = fit_small(3);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {base:?}");
        let ln = crate::conformance::check_logistic_normal(&m);
        assert!(ln.is_empty(), "check_logistic_normal: {ln:?}");
    }

    // FISTA recovers the closed-form lasso solution of a separable quadratic:
    // minimize 0.5‖x − target‖² + λ‖x‖₁ has solution soft_threshold(target, λ),
    // with entries below λ driven to *exact* zero. This directly exercises the
    // sparse-content solver independent of the ECTM likelihood.
    #[test]
    fn fista_l1_recovers_soft_threshold() {
        let target = [3.0f64, -2.0, 0.5, -0.3, 0.05];
        let lam = 1.0;
        let f_and_grad = |x: &[f64]| {
            let mut val = 0.0;
            let mut grad = vec![0.0; x.len()];
            for i in 0..x.len() {
                let d = x[i] - target[i];
                val += 0.5 * d * d;
                grad[i] = d;
            }
            (val, grad)
        };
        let anchor = vec![0.0; target.len()];
        let x = fista_l1(vec![0.0; target.len()], f_and_grad, lam, &anchor, 500);
        let expected: Vec<f64> = target.iter().map(|&t| soft_threshold(t, lam)).collect();
        for (xi, ei) in x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-4, "got {x:?}, expected {expected:?}");
        }
        // entries with |target| <= lam are pinned to exact zero (sparsity).
        assert_eq!(x[2], 0.0);
        assert_eq!(x[3], 0.0);
        assert_eq!(x[4], 0.0);
    }

    // The L1 content prior drives the between-group content deviations toward
    // zero: with a strong penalty the group κ collapse, so the two groups' fitted
    // word distributions coincide (no spurious wording contrast). The L2 default
    // keeps a nonzero contrast. Verifies the flag reaches the estimator and the
    // proximal solve sparsifies end-to-end.
    #[test]
    fn l1_content_prior_sparsifies_group_contrast() {
        let (docs, groups, periods) = growing_contrast_corpus();
        let fit = |content_l1: f64| {
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            fit_ectm(
                &docs, 2, 4, &groups, 2, &periods, 3, 60, 0.0, 0.0, None, 1.0, 5.0, 5.0, 2.0,
                content_l1, true, false, true, &[], &mut rng,
            )
        };
        // Total between-group L1 wording distance over cells.
        let contrast = |m: &EctmModel| -> f64 {
            let mut tot = 0.0;
            for t in 0..m.num_periods {
                let a = &m.content_beta[m.cell(0, t)];
                let b = &m.content_beta[m.cell(1, t)];
                for k in 0..m.num_topics {
                    for w in 0..a[k].len() {
                        tot += (a[k][w] - b[k][w]).abs();
                    }
                }
            }
            tot
        };
        // Sparsity increases monotonically with the penalty; a strong penalty
        // collapses even this (deliberately strong) 4-word group signal.
        let c_l2 = contrast(&fit(0.0));
        let c_mid = contrast(&fit(50.0));
        let c_strong = contrast(&fit(1000.0));
        assert!(c_mid < c_l2, "L1 should reduce the contrast: L2={c_l2}, L1={c_mid}");
        assert!(
            c_strong < 0.25 * c_l2,
            "strong L1 should largely collapse the group contrast: L2={c_l2}, strong={c_strong}"
        );
    }
}
