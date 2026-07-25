//! IdealPointSentenceTM: a continuous ideal-point topic model over sentence (or
//! document) embeddings. The embedding-native analog of [`crate::idealpoint`]:
//! where IdealPointTM displaces a topic's word *softmax* by the author's position,
//! IdealPointSentenceTM displaces a topic's *centroid* in embedding space.
//!
//! Each topic `k` is a Gaussian cluster with centroid `mu_k in R^D`; each author has
//! a position `x_a in R^d`; each topic has a loading `V_{k,j} in R^D`. An embedding
//! `e_i` from author `a` is drawn from a mixture over topics whose `k`-th component
//! is `N(mu_k + sum_j x_{a,j} V_{k,j}, sigma^2 I)`. So an author's position shifts
//! where in embedding space their sentences land *within* a topic, and `||V_k||` is
//! the topic's discrimination. The position is latent and estimated.
//!
//! Inference is closed-form EM (a Gaussian mixture): the E-step is the soft topic
//! assignment; the M-step solves weighted least squares for `(mu_k, V_k)` jointly,
//! a small linear system for each author's `x_a`, and a residual update for
//! `sigma^2`. Identification matches the other ideal-point models: positions are
//! standardized each iteration (absorbed losslessly into `mu`/`V`) and the sign is
//! oriented to the anchors.

use crate::linalg::spd_inverse;
use rand::Rng;
use rayon::prelude::*;

/// A fitted IdealPointSentenceTM. `mu` (K x D) are the topic centroids at the neutral
/// position, `v` (K x d x D) the position loadings, `x` (A x d) the author
/// positions, `resp` (N x K) the soft topic assignments, `pi` the mixture weights.
pub struct SentenceIdealModel {
    pub num_topics: usize,
    pub dim: usize,
    pub num_dims: usize,
    pub num_authors: usize,
    pub mu: Vec<Vec<f64>>,
    pub v: Vec<Vec<Vec<f64>>>,
    pub x: Vec<Vec<f64>>,
    pub pi: Vec<f64>,
    pub sigma2: f64,
    pub resp: Vec<Vec<f64>>,
    pub group: Vec<usize>,
    pub log_likelihood: f64,
    pub ll_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
}

impl SentenceIdealModel {
    /// The displaced centroid `mu_k + sum_j x_j V_{k,j}` for a topic at position `x`.
    pub fn position_topic_centroid(&self, k: usize, x: &[f64]) -> Vec<f64> {
        let mut c = self.mu[k].clone();
        for (j, &xj) in x.iter().enumerate() {
            for (d, cd) in c.iter_mut().enumerate() {
                *cd += xj * self.v[k][j][d];
            }
        }
        c
    }

    /// The conditional incomplete-data (marginal) log-likelihood of the fitted
    /// mixture on `emb`, holding the positions fixed:
    /// `sum_i log sum_k pi_k N(e_i | mu_k + x_a V_k, sigma^2 I)`. A properly
    /// normalized spherical-Gaussian density, so it includes the per-observation
    /// normalizer `-(D/2) ln(2 pi sigma^2)`. This is the quantity `log_likelihood`
    /// reports for the *returned* parameters (no one-iteration lag), and the oracle
    /// the parity test recomputes independently.
    pub fn incomplete_data_ll(&self, emb: &[Vec<f64>]) -> f64 {
        if self.sigma2 <= 0.0 || !self.sigma2.is_finite() {
            return f64::NAN;
        }
        let inv2s = 1.0 / (2.0 * self.sigma2);
        let log_norm = -0.5 * self.dim as f64 * (2.0 * std::f64::consts::PI * self.sigma2).ln();
        let log_pi: Vec<f64> = self.pi.iter().map(|&p| p.max(1e-300).ln()).collect();
        emb.iter()
            .zip(self.group.iter())
            .map(|(e, &a)| {
                let mut logr = vec![f64::NEG_INFINITY; self.num_topics];
                let mut max = f64::NEG_INFINITY;
                for t in 0..self.num_topics {
                    let mean = self.position_topic_centroid(t, &self.x[a]);
                    let mut sq = 0.0;
                    for d in 0..self.dim {
                        let z = e[d] - mean[d];
                        sq += z * z;
                    }
                    logr[t] = log_pi[t] - inv2s * sq;
                    if logr[t] > max {
                        max = logr[t];
                    }
                }
                let z: f64 = logr.iter().map(|&l| (l - max).exp()).sum();
                max + z.ln() + log_norm
            })
            .sum()
    }

    /// Per-topic discrimination `||V_k||` (Frobenius over the d x D loading).
    pub fn topic_discrimination(&self) -> Vec<f64> {
        self.v
            .iter()
            .map(|vk| {
                vk.iter()
                    .flat_map(|vkj| vkj.iter())
                    .map(|z| z * z)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    }

    /// Asymptotic standard error of each author position `(A x d)`. The position is
    /// a linear-Gaussian weighted least squares (E-step responsibilities held
    /// fixed), so the negative-log-posterior Hessian is exact and constant in `x`:
    /// `H_a = (sum_k N_{a,k} G_k) / sigma^2 + I / x_prior_variance`, with
    /// `N_{a,k} = sum_{i in a} r_{i,k}` the expected number of the author's
    /// observations in topic `k` and `G_k[j,l] = V_{k,j}.V_{k,l}`. The SE is
    /// `sqrt(diag(H_a^{-1}))` -- the closed-form analog of Wordfish's `se.theta`,
    /// exact here because the model is Gaussian in `x` given the responsibilities.
    pub fn position_se(&self, x_prior_variance: f64) -> Vec<Vec<f64>> {
        let dd = self.num_dims;
        let inv_xvar = 1.0 / x_prior_variance;
        let inv_s2 = 1.0 / self.sigma2.max(1e-300);
        // Per-topic G_k[j,l] = V_{k,j} . V_{k,l} (d x d), constant across authors.
        let g_topic: Vec<Vec<f64>> = self
            .v
            .iter()
            .map(|vk| {
                let mut g = vec![0.0f64; dd * dd];
                for j in 0..dd {
                    for l in j..dd {
                        let dot: f64 = vk[j].iter().zip(&vk[l]).map(|(&a, &b)| a * b).sum();
                        g[j * dd + l] = dot;
                        g[l * dd + j] = dot;
                    }
                }
                g
            })
            .collect();
        // N_{a,k} = sum_{i in a} resp[i][k].
        let mut nak = vec![vec![0.0f64; self.num_topics]; self.num_authors];
        for (i, ri) in self.resp.iter().enumerate() {
            let a = self.group[i];
            for (kk, &r) in ri.iter().enumerate() {
                nak[a][kk] += r;
            }
        }
        (0..self.num_authors)
            .map(|a| {
                let mut h = vec![0.0f64; dd * dd];
                for kk in 0..self.num_topics {
                    let n = nak[a][kk];
                    if n <= 0.0 {
                        continue;
                    }
                    for idx in 0..dd * dd {
                        h[idx] += n * inv_s2 * g_topic[kk][idx];
                    }
                }
                for j in 0..dd {
                    h[j * dd + j] += inv_xvar;
                }
                if dd == 1 {
                    return vec![if h[0] > 0.0 {
                        h[0].sqrt().recip()
                    } else {
                        f64::NAN
                    }];
                }
                let cov = spd_inverse(&h, dd).unwrap_or_else(|| {
                    let mut id = vec![0.0f64; dd * dd];
                    for j in 0..dd {
                        id[j * dd + j] = f64::NAN;
                    }
                    id
                });
                (0..dd)
                    .map(|j| {
                        let var = cov[j * dd + j];
                        if var > 0.0 {
                            var.sqrt()
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Displaced mean `mu_k + sum_j x_{a,j} V_{k,j}` (length D).
fn topic_mean(mu_k: &[f64], v_k: &[Vec<f64>], x_a: &[f64]) -> Vec<f64> {
    let mut m = mu_k.to_vec();
    for (j, &xj) in x_a.iter().enumerate() {
        for (d, md) in m.iter_mut().enumerate() {
            *md += xj * v_k[j][d];
        }
    }
    m
}

/// Fit IdealPointSentenceTM by EM. `emb` are the per-observation embeddings (N x D),
/// `group[i]` the author of observation `i`. `anchors` orient the sign of the first
/// dimension. `x_prior_variance` is the Gaussian prior on the positions.
#[allow(clippy::too_many_arguments)]
pub fn fit_sentence_ideal<R: Rng>(
    emb: &[Vec<f64>],
    group: &[usize],
    num_authors: usize,
    num_topics: usize,
    num_dims: usize,
    anchors: &[(usize, f64)],
    em_iters: usize,
    em_tol: f64,
    x_prior_variance: f64,
    rng: &mut R,
) -> SentenceIdealModel {
    let n = emb.len();
    let dim = if n > 0 { emb[0].len() } else { 0 };
    let k = num_topics;
    let dd = num_dims;
    let a_n = num_authors;
    let inv_xvar = 1.0 / x_prior_variance;

    // Initialize centroids at K distinct observations; loadings near zero; positions
    // from the leading PCs of the author-mean embedding matrix.
    let mut mu: Vec<Vec<f64>> = Vec::with_capacity(k);
    {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..n.min(k) {
            let j = (i + (rng.gen::<f64>() * (n - i) as f64) as usize).min(n - 1);
            idx.swap(i, j);
        }
        for t in 0..k {
            mu.push(emb[idx[t % n.max(1)].min(n - 1)].clone());
        }
    }
    let mut v = vec![vec![vec![0.0f64; dim]; dd]; k];
    for vk in v.iter_mut() {
        for vkj in vk.iter_mut() {
            for z in vkj.iter_mut() {
                *z = (rng.gen::<f64>() - 0.5) * 1e-3;
            }
        }
    }
    let mut x = init_positions(emb, group, a_n, dd, rng);
    standardize_positions(&mut x, &mut mu, &mut v);
    let mut pi = vec![1.0 / k as f64; k];
    // Spherical variance initialized from the overall embedding spread.
    let mut sigma2 = {
        let mut mean = vec![0.0f64; dim];
        for e in emb {
            for d in 0..dim {
                mean[d] += e[d];
            }
        }
        for m in mean.iter_mut() {
            *m /= n.max(1) as f64;
        }
        let mut s = 0.0;
        for e in emb {
            for d in 0..dim {
                s += (e[d] - mean[d]) * (e[d] - mean[d]);
            }
        }
        (s / (n.max(1) * dim.max(1)) as f64).max(1e-6)
    };

    let mut resp = vec![vec![0.0f64; k]; n];
    let mut ll_history = Vec::with_capacity(em_iters);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iters_run = 0;

    for it in 0..em_iters {
        iters_run = it + 1;

        // E-step: responsibilities r_{i,k} ∝ pi_k N(e_i | mu_k + x_a V_k, sigma2 I).
        let inv2s = 1.0 / (2.0 * sigma2);
        // Per-observation Gaussian log-normalizer of the spherical density
        // N(.| ., sigma2 I) in D dimensions: -(D/2) ln(2 pi sigma2). It is
        // omitted from the responsibilities (it cancels in the per-row
        // normalization) but is included in the reported incomplete-data
        // log-likelihood so the value is a properly normalized log-density.
        // Because sigma2 is re-estimated each sweep the term is not a constant
        // offset, so dropping it distorts the per-iteration `ll_history` trace.
        // (Note: `ll_history` is the data log-likelihood, which EM does not
        // maximize here — the x M-step is MAP and the positions are re-standardized
        // each sweep — so it is not guaranteed monotone when `x_prior_variance`
        // differs from the unit identification scale; see the `fit_history` docs.)
        let log_norm = -0.5 * dim as f64 * (2.0 * std::f64::consts::PI * sigma2).ln();
        let log_pi: Vec<f64> = pi.iter().map(|&p| p.max(1e-300).ln()).collect();
        // Per-document log-likelihood, collected in document order then summed
        // sequentially so the total is independent of rayon's work-stealing order
        // (a parallel `.sum()` of f64 is not associative and would break the
        // fixed-seed determinism guarantee).
        let ll_parts: Vec<f64> = emb
            .par_iter()
            .zip(resp.par_iter_mut())
            .zip(group.par_iter())
            .map(|((e, ri), &a)| {
                let mut logr = vec![0.0f64; k];
                let mut max = f64::NEG_INFINITY;
                for t in 0..k {
                    let mean = topic_mean(&mu[t], &v[t], &x[a]);
                    let mut sq = 0.0;
                    for d in 0..dim {
                        let z = e[d] - mean[d];
                        sq += z * z;
                    }
                    logr[t] = log_pi[t] - inv2s * sq;
                    if logr[t] > max {
                        max = logr[t];
                    }
                }
                let mut z = 0.0;
                for t in 0..k {
                    let p = (logr[t] - max).exp();
                    ri[t] = p;
                    z += p;
                }
                for t in 0..k {
                    ri[t] /= z;
                }
                max + z.ln() + log_norm
            })
            .collect();
        let ll: f64 = ll_parts.iter().sum();
        ll_history.push(ll);

        // M-step: mixture weights.
        let mut nk = vec![0.0f64; k];
        for ri in &resp {
            for t in 0..k {
                nk[t] += ri[t];
            }
        }
        for t in 0..k {
            pi[t] = (nk[t] / n as f64).max(1e-12);
        }

        // M-step: joint (mu_k, V_k) by weighted least squares on design [1, x_a].
        // Solve A c = B with A ((1+d)x(1+d)) = sum_i r_ik g g^T, B ((1+d)xD) =
        // sum_i r_ik g e_i^T, g = [1, x_a]. Row 0 of c is mu_k, rows 1..d are V_k.
        let p = 1 + dd;
        let updated: Vec<(Vec<f64>, Vec<Vec<f64>>)> = (0..k)
            .into_par_iter()
            .map(|t| {
                let mut a_mat = vec![0.0f64; p * p];
                let mut b_mat = vec![vec![0.0f64; dim]; p];
                for i in 0..n {
                    let r = resp[i][t];
                    if r <= 0.0 {
                        continue;
                    }
                    let a = group[i];
                    let mut g = vec![1.0f64; p];
                    g[1..(dd + 1)].copy_from_slice(&x[a][..dd]);
                    for r1 in 0..p {
                        for c1 in 0..p {
                            a_mat[r1 * p + c1] += r * g[r1] * g[c1];
                        }
                        for d in 0..dim {
                            b_mat[r1][d] += r * g[r1] * emb[i][d];
                        }
                    }
                }
                // Ridge for stability, then solve via SPD inverse.
                for r1 in 0..p {
                    a_mat[r1 * p + r1] += 1e-6;
                }
                let ainv = spd_inverse(&a_mat, p).unwrap_or_else(|| {
                    let mut id = vec![0.0f64; p * p];
                    for r1 in 0..p {
                        id[r1 * p + r1] = 1.0;
                    }
                    id
                });
                // c = ainv * B  ((1+d) x D)
                let mut c = vec![vec![0.0f64; dim]; p];
                for r1 in 0..p {
                    for c1 in 0..p {
                        let w = ainv[r1 * p + c1];
                        for d in 0..dim {
                            c[r1][d] += w * b_mat[c1][d];
                        }
                    }
                }
                let mu_t = c[0].clone();
                let v_t: Vec<Vec<f64>> = (0..dd).map(|j| c[1 + j].clone()).collect();
                (mu_t, v_t)
            })
            .collect();
        for t in 0..k {
            mu[t] = updated[t].0.clone();
            v[t] = updated[t].1.clone();
        }

        // M-step: author positions x_a by a small weighted least squares.
        // minimize sum_{i in a} sum_k r_ik || e_i - mu_k - sum_j x_j V_kj ||^2 / sigma2
        //          + ||x||^2 / prior.  Linear in x -> solve (d x d) system per author.
        let mut amats = vec![vec![0.0f64; dd * dd]; a_n];
        let mut bvecs = vec![vec![0.0f64; dd]; a_n];
        for i in 0..n {
            let a = group[i];
            for t in 0..k {
                let r = resp[i][t];
                if r <= 0.0 {
                    continue;
                }
                // residual to the neutral centroid
                let mut res = vec![0.0f64; dim];
                for d in 0..dim {
                    res[d] = emb[i][d] - mu[t][d];
                }
                for j1 in 0..dd {
                    let vj1 = &v[t][j1];
                    let mut bv = 0.0;
                    for d in 0..dim {
                        bv += vj1[d] * res[d];
                    }
                    bvecs[a][j1] += r * bv;
                    for j2 in 0..dd {
                        let vj2 = &v[t][j2];
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += vj1[d] * vj2[d];
                        }
                        amats[a][j1 * dd + j2] += r * dot;
                    }
                }
            }
        }
        for a in 0..a_n {
            // add sigma2/prior ridge (the prior term, scaled by sigma2 since the
            // data term carries 1/sigma2)
            for j in 0..dd {
                amats[a][j * dd + j] += sigma2 * inv_xvar;
            }
            let ainv = spd_inverse(&amats[a], dd);
            if let Some(ainv) = ainv {
                let mut nx = vec![0.0f64; dd];
                for j1 in 0..dd {
                    for j2 in 0..dd {
                        nx[j1] += ainv[j1 * dd + j2] * bvecs[a][j2];
                    }
                }
                x[a] = nx;
            }
        }

        // M-step: spherical variance from the mean squared residual. Collected in
        // index order then summed sequentially (fixed-order, see the E-step note).
        let ss_parts: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = group[i];
                let mut s = 0.0;
                for t in 0..k {
                    let r = resp[i][t];
                    if r <= 0.0 {
                        continue;
                    }
                    let mean = topic_mean(&mu[t], &v[t], &x[a]);
                    let mut sq = 0.0;
                    for d in 0..dim {
                        let z = emb[i][d] - mean[d];
                        sq += z * z;
                    }
                    s += r * sq;
                }
                s
            })
            .collect();
        let ss: f64 = ss_parts.iter().sum();
        sigma2 = (ss / (n.max(1) * dim.max(1)) as f64).max(1e-8);

        // Identification: standardize positions (lossless into mu/V), orient sign.
        standardize_positions(&mut x, &mut mu, &mut v);
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
                for vk in v.iter_mut() {
                    for z in vk[0].iter_mut() {
                        *z = -*z;
                    }
                }
            }
        }

        if prev_ll.is_finite() {
            let denom = prev_ll.abs().max(1.0);
            if (ll - prev_ll).abs() / denom < em_tol {
                converged = true;
                prev_ll = ll;
                break;
            }
        }
        prev_ll = ll;
    }

    let mut model = SentenceIdealModel {
        num_topics: k,
        dim,
        num_dims: dd,
        num_authors: a_n,
        mu,
        v,
        x,
        pi,
        sigma2,
        resp,
        group: group.to_vec(),
        log_likelihood: prev_ll,
        ll_history,
        converged,
        iters_run,
    };
    // Report the incomplete-data log-likelihood of the *returned* parameters, not
    // the top-of-loop value from the previous sweep's parameters (which lags by one
    // M-step). `ll_history` keeps the per-iteration E-step trace for convergence
    // monitoring.
    model.log_likelihood = model.incomplete_data_ll(emb);
    model
}

/// Initialize positions from the top `dd` PCs of the author-mean embedding matrix
/// (power iteration on the A x A gram). The leading axis of averaged sentence
/// embeddings carries the dominant cleavage (the empirical motivation for the model).
fn init_positions<R: Rng>(
    emb: &[Vec<f64>],
    group: &[usize],
    a_n: usize,
    dd: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let dim = if emb.is_empty() { 0 } else { emb[0].len() };
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };
    // Author-mean embeddings, column-centered.
    let mut m = vec![vec![0.0f64; dim]; a_n];
    let mut cnt = vec![0.0f64; a_n];
    for (i, e) in emb.iter().enumerate() {
        let a = group[i];
        for d in 0..dim {
            m[a][d] += e[d];
        }
        cnt[a] += 1.0;
    }
    for a in 0..a_n {
        if cnt[a] > 0.0 {
            for d in 0..dim {
                m[a][d] /= cnt[a];
            }
        }
    }
    let mut colmean = vec![0.0f64; dim];
    for row in m.iter() {
        for d in 0..dim {
            colmean[d] += row[d];
        }
    }
    for d in 0..dim {
        colmean[d] /= a_n.max(1) as f64;
    }
    for row in m.iter_mut() {
        for d in 0..dim {
            row[d] -= colmean[d];
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

/// Standardize positions to mean 0 / unit variance per dimension, absorbed losslessly
/// into `mu` (centering) and `V` (scaling): `mu_k += xbar_j V_{k,j}`, `V_{k,j} *= sd_j`.
fn standardize_positions(x: &mut [Vec<f64>], mu: &mut [Vec<f64>], v: &mut [Vec<Vec<f64>>]) {
    let a_n = x.len();
    if a_n == 0 {
        return;
    }
    let dd = x[0].len();
    let dim = if mu.is_empty() { 0 } else { mu[0].len() };
    for j in 0..dd {
        let mean: f64 = x.iter().map(|xa| xa[j]).sum::<f64>() / a_n as f64;
        for (mk, vk) in mu.iter_mut().zip(v.iter()) {
            for d in 0..dim {
                mk[d] += mean * vk[j][d];
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
            for vk in v.iter_mut() {
                for z in vk[j].iter_mut() {
                    *z *= sd;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

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
    fn fit_sentence_ideal_recovers_positions() {
        // Two clusters in 6-D; topic 0 discriminates (centroid shifts along V with
        // author position). Sample embeddings and check position recovery.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (k, dim, dd) = (2usize, 6usize, 1usize);
        let a_n = 40usize;
        let mut mu_true = vec![vec![0.0f64; dim]; k];
        mu_true[0][0] = 3.0;
        mu_true[1][1] = 3.0;
        let mut v_true = vec![vec![vec![0.0f64; dim]; dd]; k];
        // topic 0 shifts along dims 2-3 with position
        v_true[0][0][2] = 2.0;
        v_true[0][0][3] = -2.0;
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        let mut emb: Vec<Vec<f64>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        let sigma = 0.5;
        for a in 0..a_n {
            for _ in 0..20 {
                let t = rng.gen_range(0..k);
                let mean = topic_mean(&mu_true[t], &v_true[t], &[x_true[a]]);
                let e: Vec<f64> = mean
                    .iter()
                    .map(|&m| m + (rng.gen::<f64>() - 0.5) * 2.0 * sigma * 1.732)
                    .collect();
                emb.push(e);
                group.push(a);
            }
        }
        let anchors = vec![(0usize, x_true[0]), (1usize, x_true[1])];
        let m = fit_sentence_ideal(&emb, &group, a_n, k, dd, &anchors, 60, 1e-7, 1.0, &mut rng);
        let x_hat: Vec<f64> = (0..a_n).map(|a| m.x[a][0]).collect();
        let r = pearson(&x_hat, &x_true).abs();
        assert!(r > 0.8, "position recovery r={r}");
        // discrimination concentrates on the planted discriminating topic
        let disc = m.topic_discrimination();
        let (hi, lo) = if disc[0] >= disc[1] { (0, 1) } else { (1, 0) };
        assert!(
            disc[hi] > disc[lo],
            "discrimination should differ: {disc:?}"
        );

        // Position SEs are finite, positive, capped by the prior SD (=1), and the
        // exact Laplace covariance, so they are reproducible across calls.
        let se = m.position_se(1.0);
        assert_eq!(se.len(), a_n);
        for row in &se {
            assert_eq!(row.len(), dd);
            assert!(row[0].is_finite() && row[0] > 0.0, "bad SE: {row:?}");
            assert!(row[0] <= 1.0 + 1e-9, "SE exceeds prior SD: {}", row[0]);
        }
    }

    // The reported `log_likelihood` equals the conditional incomplete-data
    // log-likelihood recomputed independently from the returned parameters, and it
    // includes the spherical-Gaussian normalizer (regression for #499: the bug
    // dropped -(D/2) ln(2 pi sigma^2)). Checked across prior scales that make the
    // data LL non-monotone, so it does not rely on the monotone regime.
    #[test]
    fn reported_ll_matches_recompute() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (k, dim, dd, a_n) = (2usize, 5usize, 1usize, 20usize);
        let mut emb: Vec<Vec<f64>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        // Imbalanced authors + moderate noise: the regime where the data LL is not
        // EM-monotone, so this is a genuine test of the reported value, not luck.
        let counts = [1usize, 2, 40, 3, 30, 2, 1, 25];
        for (a, &c) in counts.iter().enumerate() {
            let pos = rng.gen::<f64>() - 0.5;
            for _ in 0..c {
                let t = rng.gen_range(0..k);
                let mut e = vec![0.0f64; dim];
                for (d, ed) in e.iter_mut().enumerate() {
                    *ed = (t == 0) as usize as f64 * 3.0
                        + pos * 0.5
                        + (rng.gen::<f64>() - 0.5) * 1.4
                        + d as f64 * 1e-9;
                }
                emb.push(e);
                group.push(a.min(a_n - 1));
            }
        }
        for xpv in [0.05f64, 0.2, 1.0] {
            let m = fit_sentence_ideal(&emb, &group, a_n, k, dd, &[], 200, 0.0, xpv, &mut rng);
            // Independent oracle: recompute the incomplete-data mixture LL inline
            // from the returned parameters (NOT via `incomplete_data_ll`, which is
            // the code path that sets `log_likelihood`), including the normalizer.
            let inv2s = 1.0 / (2.0 * m.sigma2);
            let cst = -0.5 * dim as f64 * (2.0 * std::f64::consts::PI * m.sigma2).ln();
            let mut oracle = 0.0f64;
            let mut oracle_normless = 0.0f64; // same, but with the normalizer dropped
            for (e, &a) in emb.iter().zip(group.iter()) {
                let mut comp = vec![0.0f64; k];
                let mut mx = f64::NEG_INFINITY;
                for t in 0..k {
                    let mut mean = m.mu[t].clone();
                    for d in 0..dim {
                        mean[d] += m.x[a][0] * m.v[t][0][d];
                    }
                    let sq: f64 = (0..dim).map(|d| (e[d] - mean[d]).powi(2)).sum();
                    comp[t] = m.pi[t].max(1e-300).ln() - inv2s * sq;
                    mx = mx.max(comp[t]);
                }
                let z: f64 = comp.iter().map(|&c| (c - mx).exp()).sum();
                oracle += mx + z.ln() + cst;
                oracle_normless += mx + z.ln();
            }
            let tol = 1e-9 * m.log_likelihood.abs().max(1.0);
            assert!(
                (m.log_likelihood - oracle).abs() <= tol,
                "xpv={xpv}: reported ll {} != independent oracle {oracle}",
                m.log_likelihood
            );
            // Guard the specific #499 regression: the reported value is the
            // *normalized* density, i.e. it differs from the normalizer-dropped sum
            // by the full N * (D/2) ln(2 pi sigma^2) (a large, non-zero gap here).
            assert!(
                (m.log_likelihood - oracle_normless - cst * emb.len() as f64).abs() <= tol,
                "xpv={xpv}: normalizer term missing from the reported ll"
            );
            assert!(m.log_likelihood.is_finite());
        }
    }

    // An author with many observations is placed more precisely than one with few.
    #[test]
    fn sentence_position_se_shrinks_with_data() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let (k, dim, dd) = (2usize, 6usize, 1usize);
        let a_n = 12usize;
        let mut mu_true = vec![vec![0.0f64; dim]; k];
        mu_true[0][0] = 3.0;
        mu_true[1][1] = 3.0;
        let mut v_true = vec![vec![vec![0.0f64; dim]; dd]; k];
        v_true[0][0][2] = 2.0;
        v_true[0][0][3] = -2.0;
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let sigma = 0.5;
        let mut emb: Vec<Vec<f64>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        for a in 0..a_n {
            // author 0 data-rich, author 1 data-poor, the rest moderate
            let nobs = if a == 0 {
                120
            } else if a == 1 {
                6
            } else {
                25
            };
            for _ in 0..nobs {
                let t = rng.gen_range(0..k);
                let mean = topic_mean(&mu_true[t], &v_true[t], &[x_true[a]]);
                let e: Vec<f64> = mean
                    .iter()
                    .map(|&m| m + (rng.gen::<f64>() - 0.5) * 2.0 * sigma * 1.732)
                    .collect();
                emb.push(e);
                group.push(a);
            }
        }
        let m = fit_sentence_ideal(
            &emb,
            &group,
            a_n,
            k,
            dd,
            &[(0, x_true[0])],
            60,
            1e-7,
            1.0,
            &mut rng,
        );
        let se = m.position_se(1.0);
        assert!(
            se[0][0] < se[1][0],
            "data-rich author should have smaller SE: {} vs {}",
            se[0][0],
            se[1][0]
        );
    }
}
