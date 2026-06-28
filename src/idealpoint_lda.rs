//! The count representation of IdealPointTM (the binding merges both behind one
//! pyclass; see [`crate::idealpoint`] for the word-embedding representation).
//!
//! Where the embedded core factors the topic-word matrix through word *embeddings*,
//! this one parameterizes it directly over the vocabulary, so it is the
//! identity-embedding (every word its own dimension) case of the same model:
//!
//! ```text
//!   eta_{a,k,v} = alpha_{k,v} + sum_j x_{a,j} W_{k,j,v}
//!   beta_{a,k,v} = softmax_v( eta_{a,k} )
//! ```
//!
//! `alpha_k` is the topic-word log-profile at the neutral position `x = 0`;
//! `W_{k,j,v}` is the per-word position loading; `x_a` the author's latent ideal
//! point. So an author's position shifts word choice *within* each topic, and
//! `||W_k||` is the topic's discrimination. It is "Wordfish with topics": each
//! topic carries its own Wordfish-style discrimination over the vocabulary, and a
//! document mixes the topics. The position is latent and estimated.
//!
//! Inference is variational EM, identical in skeleton to IdealPointTM (logistic-
//! normal document topics, the same Laplace E-step) but with the embedding
//! projection removed: `base = alpha`, `disc = W`, the sufficient statistic is the
//! plain expected count `S_{a,k,v}` and the expected word distribution is `beta`
//! itself. Identification (standardize positions each iteration, absorbed losslessly
//! into `alpha`/`W`; orient to anchors) matches IdealPointTM.

use crate::ctm::{ctm_grad, ctm_hpb, ctm_lhood, HpbResult};
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse};
use crate::variational::{doc_sparse, lbfgs_minimize};
use rand::Rng;
use rayon::prelude::*;

/// `beta_{a,k} = softmax_v(alpha_k[v] + sum_j x_{a,j} W_{k,j}[v])` over the
/// vocabulary (length V).
fn topic_beta(alpha_k: &[f64], w_k: &[Vec<f64>], x_a: &[f64]) -> Vec<f64> {
    let vsz = alpha_k.len();
    let mut eta = vec![0.0f64; vsz];
    let mut max = f64::NEG_INFINITY;
    for v in 0..vsz {
        let mut e = alpha_k[v];
        for (j, &xj) in x_a.iter().enumerate() {
            e += xj * w_k[j][v];
        }
        eta[v] = e;
        if e > max {
            max = e;
        }
    }
    let mut z = 0.0;
    for e in &eta {
        z += (e - max).exp();
    }
    eta.iter().map(|e| (e - max).exp() / z).collect()
}

/// A fitted count-representation model. `beta0` (K x V) is the topic-word matrix at `x = 0`;
/// `alpha` (K x V) the topic log-profiles, `w` (K x d x V) the position loadings,
/// `x` (A x d) the author positions.
pub struct IdealPointLdaModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_dims: usize,
    pub num_authors: usize,
    pub alpha: Vec<Vec<f64>>,
    pub w: Vec<Vec<Vec<f64>>>,
    pub x: Vec<Vec<f64>>,
    pub group: Vec<usize>,
    pub beta0: Vec<Vec<f64>>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub lambda: Vec<Vec<f64>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl IdealPointLdaModel {
    /// Per-document topic proportions theta = softmax([eta, 0]) (D x K).
    pub fn doc_topics(&self) -> Vec<Vec<f64>> {
        self.lambda
            .iter()
            .map(|eta| {
                let mut full = eta.clone();
                full.push(0.0);
                let max = full.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = full.iter().map(|e| (e - max).exp()).collect();
                let s: f64 = exps.iter().sum();
                exps.iter().map(|e| e / s).collect()
            })
            .collect()
    }

    /// Topic-word distribution for topic `k` at an arbitrary position `x`.
    pub fn position_topic_beta(&self, k: usize, x: &[f64]) -> Vec<f64> {
        topic_beta(&self.alpha[k], &self.w[k], x)
    }

    /// Per-topic discrimination `||W_k||` (Frobenius over the d x V loading).
    pub fn topic_discrimination(&self) -> Vec<f64> {
        self.w
            .iter()
            .map(|wk| {
                wk.iter()
                    .flat_map(|wkj| wkj.iter())
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    }

    /// Asymptotic standard error of each author position, `(A x d)`. Same observed-
    /// information derivation as the embedded core; here the discrimination logits
    /// are the per-word loadings `W_{k,j,v}` directly. See
    /// [`crate::idealpoint::position_ses`].
    pub fn position_se(&self, ntot: &[Vec<f64>], x_prior_variance: f64) -> Vec<Vec<f64>> {
        crate::idealpoint::position_ses(
            self.num_authors,
            self.num_dims,
            ntot,
            &self.w,
            &|a, k| self.position_topic_beta(k, &self.x[a]),
            x_prior_variance,
        )
    }
}

/// Fit the count representation by variational EM. `group[d]` maps document `d` to its author.
/// `anchors` are `(author, target)` pairs orienting the sign of the first latent
/// dimension. The prior variances are Gaussian priors on `alpha`, the loadings `W`,
/// and the positions `x`.
#[allow(clippy::too_many_arguments)]
pub fn fit_idealpoint_lda<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    num_authors: usize,
    num_topics: usize,
    num_types: usize,
    num_dims: usize,
    anchors: &[(usize, f64)],
    em_iters: usize,
    em_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    w_prior_variance: f64,
    x_prior_variance: f64,
    max_inner: usize,
    rng: &mut R,
) -> IdealPointLdaModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let a_n = num_authors;
    let dd = num_dims;
    let v = num_types;
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Topic log-profiles initialized near a flat distribution plus small noise, with
    // K random words boosted so topics start differentiated (LDA-style).
    let mut alpha = vec![vec![0.0f64; v]; k];
    for ak in alpha.iter_mut() {
        for av in ak.iter_mut() {
            *av = (rng.gen::<f64>() - 0.5) * 0.01;
        }
        for _ in 0..(v / k).max(1) {
            let j = (rng.gen::<f64>() * v as f64) as usize % v;
            ak[j] += 0.5;
        }
    }
    // Loadings start near zero (so the fit starts as LDA, then differentiates).
    let mut w = vec![vec![vec![0.0f64; v]; dd]; k];
    for wk in w.iter_mut() {
        for wkj in wk.iter_mut() {
            for wv in wkj.iter_mut() {
                *wv = (rng.gen::<f64>() - 0.5) * 0.01;
            }
        }
    }
    // Positions from the leading PCs of the author-word matrix (Wordfish-style), so
    // the axis carries signal from iteration 1 and escapes the W=0 fixed point.
    let mut x = if a_n <= 1500 {
        init_positions(docs, group, a_n, v, dd, rng)
    } else {
        let mut x = vec![vec![0.0f64; dd]; a_n];
        for xa in x.iter_mut() {
            for xj in xa.iter_mut() {
                *xj = rng.gen::<f64>() - 0.5;
            }
        }
        x
    };
    standardize_positions(&mut x, &mut alpha, &mut w);

    let mut mu_shared = vec![0.0f64; km1];
    let mut sigma = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];

    let mut bound_history: Vec<f64> = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;

    for em in 0..em_iters {
        em_iters_run = em + 1;

        // Per-author beta (K x V), reused across an author's documents.
        let author_beta: Vec<Vec<Vec<f64>>> = (0..a_n)
            .into_par_iter()
            .map(|a| {
                (0..k)
                    .map(|t| topic_beta(&alpha[t], &w[t], &x[a]))
                    .collect()
            })
            .collect();

        let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, km1);
            spd_inverse(&s, km1).unwrap()
        });
        let entropy = match cholesky(&sigma, km1) {
            Some(l) => half_logdet(&l, km1),
            None => 0.0,
        };

        // E-step: per-document logistic-normal variational inference.
        let doc_results: Vec<(usize, Vec<f64>, HpbResult)> = sparse
            .par_iter()
            .enumerate()
            .filter(|(_, (words, _))| !words.is_empty())
            .map(|(di, (words, counts))| {
                let a = group[di];
                let beta_a = &author_beta[a];
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| {
                        (
                            ctm_lhood(eta, beta_a, words, counts, &mu_shared, &siginv),
                            ctm_grad(eta, beta_a, words, counts, &mu_shared, &siginv),
                        )
                    },
                    40,
                    7,
                    1e-5,
                );
                let res = ctm_hpb(
                    &opt, beta_a, words, counts, &mu_shared, &siginv, entropy, false,
                );
                (di, opt, res)
            })
            .collect();

        let total_bound: f64 = doc_results.iter().map(|(_, _, r)| r.bound).sum();
        bound_history.push(total_bound);

        // Sufficient statistics: S[a][k][v] expected counts, and ntot[a][k].
        let mut s_stat = vec![vec![vec![0.0f64; v]; k]; a_n];
        let mut ntot = vec![vec![0.0f64; k]; a_n];
        let mut sigma_ss = vec![0.0f64; km1 * km1];
        let mut lambda_sum = vec![0.0f64; km1];

        for (di, opt, res) in &doc_results {
            let di = *di;
            let a = group[di];
            let words = &sparse[di].0;
            lambda[di] = opt.clone();
            for (wi, &wv) in words.iter().enumerate() {
                for t in 0..k {
                    let c = res.phi[t][wi];
                    ntot[a][t] += c;
                    s_stat[a][t][wv] += c;
                }
            }
            for i in 0..km1 {
                lambda_sum[i] += opt[i];
                for j in 0..km1 {
                    sigma_ss[i * km1 + j] += res.nu[i * km1 + j];
                }
            }
        }

        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }

        // M-step: shared prior mean mu and covariance Sigma (as in CTM).
        for i in 0..km1 {
            mu_shared[i] = lambda_sum[i] / d as f64;
        }
        for i in 0..km1 {
            for j in 0..km1 {
                let mut cross = 0.0;
                for li in lambda.iter() {
                    cross += (li[i] - mu_shared[i]) * (li[j] - mu_shared[j]);
                }
                sigma[i * km1 + j] = (sigma_ss[i * km1 + j] + cross) / d as f64;
            }
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

        // M-step: topic profiles alpha and loadings W, jointly by L-BFGS.
        update_alpha_w(
            &mut alpha,
            &mut w,
            &x,
            &s_stat,
            &ntot,
            prior_variance,
            w_prior_variance,
            max_inner,
        );

        // M-step: author positions x, per author by L-BFGS.
        x = (0..a_n)
            .into_par_iter()
            .map(|a| update_position(&alpha, &w, &s_stat[a], &ntot[a], &x[a], x_prior_variance))
            .collect();

        // Identification: standardize positions (lossless into alpha/W), orient sign.
        standardize_positions(&mut x, &mut alpha, &mut w);
        if dd >= 1 && !anchors.is_empty() {
            let mut dot = 0.0;
            for &(au, target) in anchors {
                if au < a_n {
                    dot += x[au][0] * target;
                }
            }
            if dot < 0.0 {
                for xa in x.iter_mut() {
                    xa[0] = -xa[0];
                }
                for wk in w.iter_mut() {
                    for wv in wk[0].iter_mut() {
                        *wv = -*wv;
                    }
                }
            }
        }
    }

    let beta0: Vec<Vec<f64>> = alpha
        .iter()
        .map(|ak| topic_beta(ak, &vec![vec![0.0; v]; dd], &vec![0.0; dd]))
        .collect();
    IdealPointLdaModel {
        num_topics: k,
        num_types: v,
        num_dims: dd,
        num_authors: a_n,
        alpha,
        w,
        x,
        group: group.to_vec(),
        beta0,
        mu: mu_shared,
        sigma,
        lambda,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
    }
}

/// Initialize author positions from the top `dd` principal components of the
/// centered author-word frequency matrix (Wordfish-style power iteration on the
/// A x A gram). Identical to IdealPointTM's initializer.
fn init_positions<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    a_n: usize,
    num_types: usize,
    dd: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };
    let mut m = vec![vec![0.0f64; num_types]; a_n];
    let mut tot = vec![0.0f64; a_n];
    for (d, doc) in docs.iter().enumerate() {
        let a = group[d];
        for &w in doc {
            m[a][w as usize] += 1.0;
            tot[a] += 1.0;
        }
    }
    for a in 0..a_n {
        if tot[a] > 0.0 {
            for v in 0..num_types {
                m[a][v] /= tot[a];
            }
        }
    }
    let mut colmean = vec![0.0f64; num_types];
    for row in m.iter() {
        for v in 0..num_types {
            colmean[v] += row[v];
        }
    }
    for v in 0..num_types {
        colmean[v] /= a_n as f64;
    }
    for row in m.iter_mut() {
        for v in 0..num_types {
            row[v] -= colmean[v];
        }
    }
    let mut g = vec![vec![0.0f64; a_n]; a_n];
    for a in 0..a_n {
        for b in a..a_n {
            let s = dot(&m[a], &m[b]);
            g[a][b] = s;
            g[b][a] = s;
        }
    }
    let mut comps: Vec<Vec<f64>> = Vec::new();
    for _ in 0..dd {
        let mut vv: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() - 0.5).collect();
        for _ in 0..100 {
            for u in &comps {
                let dp = dot(&vv, u);
                for i in 0..a_n {
                    vv[i] -= dp * u[i];
                }
            }
            let mut gv = vec![0.0f64; a_n];
            for i in 0..a_n {
                gv[i] = dot(&g[i], &vv);
            }
            for u in &comps {
                let dp = dot(&gv, u);
                for i in 0..a_n {
                    gv[i] -= dp * u[i];
                }
            }
            let norm = dot(&gv, &gv).sqrt();
            if norm < 1e-12 {
                break;
            }
            for i in 0..a_n {
                vv[i] = gv[i] / norm;
            }
        }
        comps.push(vv);
    }
    let mut x = vec![vec![0.0f64; dd]; a_n];
    for (c, comp) in comps.iter().enumerate() {
        for a in 0..a_n {
            x[a][c] = comp[a];
        }
    }
    x
}

/// Standardize positions to mean 0 / unit variance per dimension, absorbing the
/// transform losslessly into `alpha` (centering) and `W` (scaling).
fn standardize_positions(x: &mut [Vec<f64>], alpha: &mut [Vec<f64>], w: &mut [Vec<Vec<f64>>]) {
    let a_n = x.len();
    if a_n == 0 {
        return;
    }
    let dd = x[0].len();
    let v = if alpha.is_empty() { 0 } else { alpha[0].len() };
    for j in 0..dd {
        let mean: f64 = x.iter().map(|xa| xa[j]).sum::<f64>() / a_n as f64;
        for (ak, wk) in alpha.iter_mut().zip(w.iter()) {
            for vv in 0..v {
                ak[vv] += mean * wk[j][vv];
            }
        }
        for xa in x.iter_mut() {
            xa[j] -= mean;
        }
        let var: f64 = x.iter().map(|xa| xa[j] * xa[j]).sum::<f64>() / a_n as f64;
        let sd = var.sqrt();
        if sd > 1e-8 {
            for xa in x.iter_mut() {
                xa[j] /= sd;
            }
            for wk in w.iter_mut() {
                for wv in wk[j].iter_mut() {
                    *wv *= sd;
                }
            }
        }
    }
}

/// Joint L-BFGS update of topic profiles `alpha` (K x V) and loadings `W` (K x d x V)
/// holding positions fixed. Objective per author/topic is
/// `S_{a,k} . eta_{a,k} - n_{a,k} logZ_{a,k}` with Gaussian priors; the gradient uses
/// `r_{a,k,v} = S_{a,k,v} - n_{a,k} beta_{a,k,v}`. Parallelized over topics.
#[allow(clippy::too_many_arguments)]
fn update_alpha_w(
    alpha: &mut [Vec<f64>],
    w: &mut [Vec<Vec<f64>>],
    x: &[Vec<f64>],
    s_stat: &[Vec<Vec<f64>>],
    ntot: &[Vec<f64>],
    prior_variance: f64,
    w_prior_variance: f64,
    max_iter: usize,
) {
    let k = alpha.len();
    if k == 0 {
        return;
    }
    let v = alpha[0].len();
    let dd = if w.is_empty() { 0 } else { w[0].len() };
    let a_n = x.len();
    let inv_va = 1.0 / prior_variance;
    let inv_vw = 1.0 / w_prior_variance;
    let alpha_len = k * v;

    let mut x0 = vec![0.0f64; alpha_len + k * dd * v];
    for (kk, ak) in alpha.iter().enumerate() {
        x0[kk * v..(kk + 1) * v].copy_from_slice(ak);
    }
    for kk in 0..k {
        for j in 0..dd {
            let off = alpha_len + (kk * dd + j) * v;
            x0[off..off + v].copy_from_slice(&w[kk][j]);
        }
    }

    let obj = |flat: &[f64]| -> (f64, Vec<f64>) {
        let per_topic: Vec<(f64, Vec<f64>, Vec<Vec<f64>>)> = (0..k)
            .into_par_iter()
            .map(|kk| {
                let alpha_k = &flat[kk * v..(kk + 1) * v];
                let w_k: Vec<&[f64]> = (0..dd)
                    .map(|j| {
                        let off = alpha_len + (kk * dd + j) * v;
                        &flat[off..off + v]
                    })
                    .collect();
                let mut val = 0.0;
                let mut g_alpha = vec![0.0f64; v];
                let mut g_w = vec![vec![0.0f64; v]; dd];
                for a in 0..a_n {
                    let n = ntot[a][kk];
                    let s_ak = &s_stat[a][kk];
                    // eta_v = alpha_k[v] + sum_j x_aj w_k[j][v]; softmax for beta, logZ.
                    let mut eta = vec![0.0f64; v];
                    let mut max = f64::NEG_INFINITY;
                    for vv in 0..v {
                        let mut e = alpha_k[vv];
                        for j in 0..dd {
                            e += x[a][j] * w_k[j][vv];
                        }
                        eta[vv] = e;
                        if e > max {
                            max = e;
                        }
                    }
                    let mut z = 0.0;
                    for &e in &eta {
                        z += (e - max).exp();
                    }
                    let logz = max + z.ln();
                    let mut s_dot = 0.0;
                    for vv in 0..v {
                        s_dot += s_ak[vv] * eta[vv];
                    }
                    val += s_dot - n * logz;
                    for vv in 0..v {
                        let beta = (eta[vv] - logz).exp();
                        let r = s_ak[vv] - n * beta;
                        g_alpha[vv] += r;
                        for j in 0..dd {
                            g_w[j][vv] += x[a][j] * r;
                        }
                    }
                }
                (val, g_alpha, g_w)
            })
            .collect();

        let mut value = 0.0;
        let mut grad = vec![0.0f64; flat.len()];
        for (kk, (val, g_alpha, g_w)) in per_topic.iter().enumerate() {
            value += val;
            grad[kk * v..(kk + 1) * v].copy_from_slice(g_alpha);
            for j in 0..dd {
                let off = alpha_len + (kk * dd + j) * v;
                grad[off..off + v].copy_from_slice(&g_w[j]);
            }
        }
        // Gaussian priors.
        for kk in 0..k {
            for vv in 0..v {
                let ai = kk * v + vv;
                value -= 0.5 * inv_va * flat[ai] * flat[ai];
                grad[ai] -= inv_va * flat[ai];
            }
            for j in 0..dd {
                let off = alpha_len + (kk * dd + j) * v;
                for vv in 0..v {
                    value -= 0.5 * inv_vw * flat[off + vv] * flat[off + vv];
                    grad[off + vv] -= inv_vw * flat[off + vv];
                }
            }
        }
        (-value, grad.iter().map(|g| -g).collect())
    };

    let xres = lbfgs_minimize(x0, obj, max_iter, 7, 1e-4);
    for kk in 0..k {
        alpha[kk].copy_from_slice(&xres[kk * v..(kk + 1) * v]);
        for j in 0..dd {
            let off = alpha_len + (kk * dd + j) * v;
            w[kk][j].copy_from_slice(&xres[off..off + v]);
        }
    }
}

/// L-BFGS update of one author's position `x_a` holding alpha/W fixed.
/// `dQ/dx_j = sum_k [ sum_v (S_{a,k,v} - n_{a,k} beta_{a,k,v}) W_{k,j,v} ] - x_j/var`.
fn update_position(
    alpha: &[Vec<f64>],
    w: &[Vec<Vec<f64>>],
    s_a: &[Vec<f64>],
    ntot_a: &[f64],
    x_a: &[f64],
    x_prior_variance: f64,
) -> Vec<f64> {
    let k = alpha.len();
    let dd = x_a.len();
    let inv_v = 1.0 / x_prior_variance;
    let obj = |flat: &[f64]| -> (f64, Vec<f64>) {
        let mut value = 0.0;
        let mut grad = vec![0.0f64; dd];
        for kk in 0..k {
            let n = ntot_a[kk];
            let vsz = alpha[kk].len();
            let mut eta = vec![0.0f64; vsz];
            let mut max = f64::NEG_INFINITY;
            for v in 0..vsz {
                let mut e = alpha[kk][v];
                for (j, &xj) in flat.iter().enumerate() {
                    e += xj * w[kk][j][v];
                }
                eta[v] = e;
                if e > max {
                    max = e;
                }
            }
            let mut z = 0.0;
            for &e in &eta {
                z += (e - max).exp();
            }
            let logz = max + z.ln();
            let beta: Vec<f64> = eta.iter().map(|&e| (e - logz).exp()).collect();
            for j in 0..dd {
                let mut s_w = 0.0;
                let mut eb_w = 0.0;
                for v in 0..vsz {
                    s_w += s_a[kk][v] * w[kk][j][v];
                    eb_w += beta[v] * w[kk][j][v];
                }
                value += flat[j] * s_w;
                grad[j] += s_w - n * eb_w;
            }
            value -= n * logz;
        }
        for j in 0..dd {
            value -= 0.5 * inv_v * flat[j] * flat[j];
            grad[j] -= inv_v * flat[j];
        }
        (-value, grad.iter().map(|g| -g).collect())
    };
    lbfgs_minimize(x_a.to_vec(), obj, 30, 7, 1e-5)
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for IdealPointLdaModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.beta0.clone()
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
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn sample_cat<R: Rng>(probs: &[f64], rng: &mut R) -> usize {
        let r: f64 = rng.gen();
        let mut c = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            c += p;
            if r < c {
                return i;
            }
        }
        probs.len() - 1
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for (&ai, &bi) in a.iter().zip(b) {
            cov += (ai - ma) * (bi - mb);
            va += (ai - ma) * (ai - ma);
            vb += (bi - mb) * (bi - mb);
        }
        cov / (va.sqrt() * vb.sqrt() + 1e-12)
    }

    #[test]
    fn fit_idealpoint_lda_recovers_positions() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (k, dd) = (2usize, 1usize);
        let vsz = 40usize;
        let a_n = 30usize;
        // Topic profiles: topic 0 over first half of vocab, topic 1 over second half.
        let mut alpha_true = vec![vec![-3.0f64; vsz]; k];
        for vv in 0..vsz / 2 {
            alpha_true[0][vv] = 0.5;
        }
        for vv in vsz / 2..vsz {
            alpha_true[1][vv] = 0.5;
        }
        // Topic 0 discriminates: a within-topic split across position.
        let mut w_true = vec![vec![vec![0.0f64; vsz]; dd]; k];
        for vv in 0..vsz / 2 {
            w_true[0][0][vv] = if vv % 2 == 0 { 2.0 } else { -2.0 };
        }
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        for a in 0..a_n {
            let xa = vec![x_true[a]];
            let beta_a: Vec<Vec<f64>> = (0..k)
                .map(|t| topic_beta(&alpha_true[t], &w_true[t], &xa))
                .collect();
            for _ in 0..12 {
                let mut doc = Vec::new();
                for _ in 0..40 {
                    let t = rng.gen_range(0..k);
                    doc.push(sample_cat(&beta_a[t], &mut rng) as u32);
                }
                docs.push(doc);
                group.push(a);
            }
        }

        let anchors = vec![(0usize, x_true[0]), (1usize, x_true[1])];
        let m = fit_idealpoint_lda(
            &docs, &group, a_n, k, vsz, dd, &anchors, 40, 1e-6, 0.0, 1e6, 10.0, 1.0, 15, &mut rng,
        );
        let x_hat: Vec<f64> = (0..a_n).map(|a| m.x[a][0]).collect();
        let r = pearson(&x_hat, &x_true).abs();
        assert!(r > 0.8, "position correlation too low: {r}");

        // Discrimination is finite and non-trivial on at least one topic. (Unlike the
        // embedding model, the full-vocabulary loadings do not force discrimination to
        // concentrate on a single topic; the recovered axis is the validated output.)
        let disc = m.topic_discrimination();
        assert!(
            disc.iter().all(|d| d.is_finite()),
            "non-finite disc: {disc:?}"
        );
        assert!(
            disc.iter().cloned().fold(0.0, f64::max) > 0.5,
            "no discrimination: {disc:?}"
        );
    }

    #[test]
    fn idealpoint_lda_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 6usize);
        let vsz = k * block;
        let docs: Vec<Vec<u32>> = (0..60)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let group: Vec<usize> = (0..docs.len()).map(|d| d % 12).collect();
        let m = fit_idealpoint_lda(
            &docs,
            &group,
            12,
            k,
            vsz,
            1,
            &[],
            25,
            1e-5,
            0.0,
            1e6,
            10.0,
            1.0,
            10,
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
