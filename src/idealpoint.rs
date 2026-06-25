//! IdealPointTM: an embedded topic model with a latent ideal-point head.
//!
//! IdealPointTM is [`crate::etm`] (ETM) plus a low-dimensional latent trait per
//! author. As in ETM each topic `k` is a point `alpha_k` in the word-embedding
//! space and `beta_{k,v} = softmax_v(rho_v . alpha_k)`. IdealPointTM adds a
//! position `x_a in R^d` per author and a loading `W_{k,j} in R^E` per topic and
//! latent dimension, and displaces the topic embedding by the author's position:
//!
//! ```text
//!   eta_{a,k,v} = rho_v . alpha_k + sum_j x_{a,j} (rho_v . W_{k,j})
//!   beta_{a,k,v} = softmax_v( eta_{a,k} )
//! ```
//!
//! So two authors discussing the same topic produce systematically different word
//! distributions, displaced along `W_k^T x_a` in embedding space. `alpha_k` is the
//! topic at the neutral position `x = 0`; `||W_k||` is the topic's discrimination
//! (how sharply it separates positions). The position is latent and estimated, not
//! observed, so this is the unsupervised, latent-trait twin of the STM content
//! covariate, and the embedding-native generalization of Wordfish (Slapin & Proksch
//! 2008): with `K=1, d=1` the log word-rate is `base_v + x_a (rho_v . w)`, i.e.
//! Wordfish with discrimination `rho_v . w` shared across semantically similar words.
//!
//! Inference is variational EM on ETM's core. Position affects *content* (within-
//! topic word choice); document topic proportions theta keep ETM's logistic-normal
//! treatment. The single sufficient statistic the E-step must pass to the M-step is
//! the embedding-weighted expected count `S_{a,k} = sum_v n_{a,k,v} rho_v` (an
//! `E`-vector) and the token total `n_{a,k}`; the whole objective (value, and the
//! gradients in alpha, W, x) reduces to those, so the cost is `A.K.V.E` for the
//! per-author softmax normalizations and never the `A.K.V` count tensor.
//!
//! Identification is exact and likelihood-preserving: centering the positions is
//! absorbed by `alpha_k += sum_j xbar_j W_{k,j}` (the base shift is
//! `rho_v . (sum_j xbar_j W_{k,j})`), and scaling `x_{.,j} *= 1/sd_j` by
//! `W_{k,j} *= sd_j`. Each EM iteration standardizes `x` to mean 0 / unit variance
//! per dimension and pushes the inverse into `alpha`/`W` (the fit is unchanged),
//! then (for `d=1`) orients the sign to supplied anchors.

use crate::ctm::{ctm_grad, ctm_hpb, ctm_lhood, HpbResult};
use crate::etm::softmax_beta;
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse};
use crate::variational::{doc_sparse, lbfgs_minimize};
use rand::Rng;
use rayon::prelude::*;

/// Base logits `base[k][v] = rho_v . alpha_k` (K x V) and discrimination logits
/// `disc[k][j][v] = rho_v . W_{k,j}` (K x d x V), the per-EM-iteration cache.
fn build_cache(
    rho: &[Vec<f64>],
    alpha: &[Vec<f64>],
    w: &[Vec<Vec<f64>>],
) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
    let dot = |emb: &[f64], v: usize| -> f64 { rho[v].iter().zip(emb).map(|(r, a)| r * a).sum() };
    let vsz = rho.len();
    let base: Vec<Vec<f64>> = alpha
        .par_iter()
        .map(|ak| (0..vsz).map(|v| dot(ak, v)).collect())
        .collect();
    let disc: Vec<Vec<Vec<f64>>> = w
        .par_iter()
        .map(|wk| {
            wk.iter()
                .map(|wkj| (0..vsz).map(|v| dot(wkj, v)).collect())
                .collect()
        })
        .collect();
    (base, disc)
}

/// `eta_{a,k,v} = base_k[v] + sum_j x_{a,j} disc_{k,j}[v]`, returned as the softmax
/// distribution `beta_{a,k}` over the vocabulary (length V).
fn topic_beta(base_k: &[f64], disc_k: &[Vec<f64>], x_a: &[f64]) -> Vec<f64> {
    let vsz = base_k.len();
    let mut eta = vec![0.0f64; vsz];
    let mut max = f64::NEG_INFINITY;
    for v in 0..vsz {
        let mut e = base_k[v];
        for (j, &xj) in x_a.iter().enumerate() {
            e += xj * disc_k[j][v];
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

/// For one author and topic, the log-normalizer `logZ_{a,k}` and the expected
/// embedding `Ebeta_{a,k} = sum_v beta_{a,k,v} rho_v` (length E). These are the only
/// position-dependent quantities the M-step objective and its gradients need.
fn topic_logz_ebeta(
    base_k: &[f64],
    disc_k: &[Vec<f64>],
    x_a: &[f64],
    rho: &[Vec<f64>],
    e_dim: usize,
) -> (f64, Vec<f64>) {
    let vsz = base_k.len();
    let mut eta = vec![0.0f64; vsz];
    let mut max = f64::NEG_INFINITY;
    for v in 0..vsz {
        let mut et = base_k[v];
        for (j, &xj) in x_a.iter().enumerate() {
            et += xj * disc_k[j][v];
        }
        eta[v] = et;
        if et > max {
            max = et;
        }
    }
    let mut z = 0.0;
    for et in &eta {
        z += (et - max).exp();
    }
    let log_z = max + z.ln();
    let mut ebeta = vec![0.0f64; e_dim];
    for v in 0..vsz {
        let b = (eta[v] - log_z).exp();
        for (ee, ebe) in ebeta.iter_mut().enumerate() {
            *ebe += b * rho[v][ee];
        }
    }
    (log_z, ebeta)
}

/// A fitted IdealPointTM. `beta0` (K x V) is the topic-word matrix at the neutral
/// position `x = 0`; `alpha` (K x E) the topic embeddings, `w` (K x d x E) the
/// position loadings, `x` (A x d) the author positions. `lambda` are the per-
/// document logistic-normal means (theta = softmax([eta, 0])); `mu`/`sigma` the
/// document-topic prior.
pub struct IdealPointModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_dims: usize,
    pub num_authors: usize,
    pub alpha: Vec<Vec<f64>>,
    pub w: Vec<Vec<Vec<f64>>>,
    pub x: Vec<Vec<f64>>,
    pub group: Vec<usize>,
    pub beta0: Vec<Vec<f64>>,
    /// Base logits `base[k][v] = rho_v . alpha_k` and discrimination logits
    /// `disc[k][j][v] = rho_v . W_{k,j}` at the fitted parameters; together they
    /// give `beta` at any position without retaining the embeddings.
    pub base: Vec<Vec<f64>>,
    pub disc: Vec<Vec<Vec<f64>>>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub lambda: Vec<Vec<f64>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl IdealPointModel {
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

    /// Topic-word distribution for topic `k` evaluated at an arbitrary position
    /// `x` (length d): `softmax_v(base_k + sum_j x_j disc_{k,j})`. Used to read off
    /// how word choice within a topic shifts along the latent axis.
    pub fn position_topic_beta(&self, k: usize, x: &[f64]) -> Vec<f64> {
        topic_beta(&self.base[k], &self.disc[k], x)
    }

    /// Per-topic discrimination `||W_k||` (Frobenius over the d x E loading),
    /// large where the topic sharply separates positions, ~0 where it is neutral.
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
}

/// Fit IdealPointTM by variational EM. `group[d] in 0..num_authors` maps each
/// document to its author; `rho` (V x E) are the fixed pretrained word embeddings.
/// `anchors` are `(author, target)` pairs used to orient the sign of the first
/// latent dimension (so positions align with the supplied direction). `em_tol`
/// stops EM on the relative change in the corpus bound. The prior variances are
/// Gaussian priors on `alpha` (weak by default), the loadings `W`, and the
/// positions `x` (unit, matching the standardization).
#[allow(clippy::too_many_arguments)]
pub fn fit_idealpoint<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    num_authors: usize,
    num_topics: usize,
    num_types: usize,
    num_dims: usize,
    rho: &[Vec<f64>],
    anchors: &[(usize, f64)],
    em_iters: usize,
    em_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    w_prior_variance: f64,
    x_prior_variance: f64,
    max_inner: usize,
    rng: &mut R,
) -> IdealPointModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let a_n = num_authors;
    let dd = num_dims;
    let e = if num_types > 0 { rho[0].len() } else { 0 };
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Initialize topic embeddings at K distinct words' embeddings plus jitter, as
    // ETM does, so topics start differentiated.
    let mut idx: Vec<usize> = (0..num_types).collect();
    for i in 0..num_types.min(k) {
        let j = (i + (rng.gen::<f64>() * (num_types - i) as f64) as usize).min(num_types - 1);
        idx.swap(i, j);
    }
    let mut alpha = vec![vec![0.0f64; e]; k];
    for (t, ak) in alpha.iter_mut().enumerate() {
        let zero = vec![0.0; e];
        let src = if num_types > 0 {
            &rho[idx[t % num_types]]
        } else {
            &zero
        };
        for (ae, &r) in ak.iter_mut().zip(src) {
            *ae = r + (rng.gen::<f64>() - 0.5) * 0.01;
        }
    }
    // Loadings start near zero (so the fit starts as ETM, then differentiates).
    let mut w = vec![vec![vec![0.0f64; e]; dd]; k];
    for wk in w.iter_mut() {
        for wkj in wk.iter_mut() {
            for we in wkj.iter_mut() {
                *we = (rng.gen::<f64>() - 0.5) * 0.01;
            }
        }
    }
    // Initialize positions from the leading principal components of the author-
    // word matrix (as Wordfish does), so the latent axis carries real signal from
    // the first iteration and the fit escapes the trivial W=0 fixed point. For a
    // very large number of authors the A x A gram is impractical, so fall back to
    // small random draws there.
    let mut x = if a_n <= 1500 {
        init_positions(docs, group, a_n, num_types, dd, rng)
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
        let (base, disc) = build_cache(rho, &alpha, &w);

        // Per-author beta (K x V), built once and reused across an author's docs.
        let author_beta: Vec<Vec<Vec<f64>>> = (0..a_n)
            .into_par_iter()
            .map(|a| {
                (0..k)
                    .map(|t| topic_beta(&base[t], &disc[t], &x[a]))
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

        // E-step: per-document logistic-normal variational inference with the
        // author's beta, in parallel then accumulated in document order.
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

        // Sufficient statistics: S[a][k] = sum_v n_{a,k,v} rho_v, and ntot[a][k].
        let mut s_stat = vec![vec![vec![0.0f64; e]; k]; a_n];
        let mut ntot = vec![vec![0.0f64; k]; a_n];
        let mut sigma_ss = vec![0.0f64; km1 * km1];
        let mut lambda_sum = vec![0.0f64; km1];

        for (di, opt, res) in &doc_results {
            let di = *di;
            let a = group[di];
            let words = &sparse[di].0;
            lambda[di] = opt.clone();
            for (wi, &wv) in words.iter().enumerate() {
                let rv = &rho[wv];
                for t in 0..k {
                    let c = res.phi[t][wi];
                    ntot[a][t] += c;
                    let s_at = &mut s_stat[a][t];
                    for (ee, se) in s_at.iter_mut().enumerate() {
                        *se += c * rv[ee];
                    }
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

        // M-step: shared prior mean mu and covariance Sigma (as in CTM/ETM).
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

        // M-step: topic embeddings alpha and loadings W, jointly by L-BFGS.
        update_alpha_w(
            rho,
            &mut alpha,
            &mut w,
            &x,
            &s_stat,
            &ntot,
            prior_variance,
            w_prior_variance,
            max_inner,
        );

        // M-step: author positions x, per author by L-BFGS, against the refreshed
        // alpha/W cache.
        let (base2, disc2) = build_cache(rho, &alpha, &w);
        x = (0..a_n)
            .into_par_iter()
            .map(|a| {
                update_position(
                    &base2,
                    &disc2,
                    &w,
                    &s_stat[a],
                    &ntot[a],
                    &x[a],
                    x_prior_variance,
                    e,
                )
            })
            .collect();

        // Identification: standardize positions (absorbed losslessly into alpha/W),
        // then orient the first dimension's sign to the anchors.
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
                    for we in wk[0].iter_mut() {
                        *we = -*we;
                    }
                }
            }
        }
    }

    let beta0 = softmax_beta(rho, &alpha);
    let (base, disc) = build_cache(rho, &alpha, &w);
    IdealPointModel {
        num_topics: k,
        num_types,
        num_dims: dd,
        num_authors: a_n,
        alpha,
        w,
        x,
        group: group.to_vec(),
        beta0,
        base,
        disc,
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
/// centered author-word frequency matrix (Wordfish-style), by power iteration on
/// the A x A gram with Gram-Schmidt deflation. Deterministic given the RNG.
fn init_positions<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    a_n: usize,
    num_types: usize,
    dd: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let dot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };
    // Author-word frequencies, row-normalized then column-centered.
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
    // Gram matrix G = M M^T (A x A).
    let mut g = vec![vec![0.0f64; a_n]; a_n];
    for a in 0..a_n {
        for b in a..a_n {
            let s = dot(&m[a], &m[b]);
            g[a][b] = s;
            g[b][a] = s;
        }
    }
    // Power iteration for the top dd eigenvectors, orthogonalizing against the
    // previously found ones each step (deflation).
    let mut comps: Vec<Vec<f64>> = Vec::new();
    for _ in 0..dd {
        let mut v: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() - 0.5).collect();
        for _ in 0..100 {
            for u in &comps {
                let d = dot(&v, u);
                for i in 0..a_n {
                    v[i] -= d * u[i];
                }
            }
            let mut gv = vec![0.0f64; a_n];
            for i in 0..a_n {
                gv[i] = dot(&g[i], &v);
            }
            for u in &comps {
                let d = dot(&gv, u);
                for i in 0..a_n {
                    gv[i] -= d * u[i];
                }
            }
            let norm = dot(&gv, &gv).sqrt();
            if norm < 1e-12 {
                break;
            }
            for i in 0..a_n {
                v[i] = gv[i] / norm;
            }
        }
        comps.push(v);
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
/// transform into `alpha` (centering) and `W` (scaling) so the likelihood is
/// unchanged: `alpha_k += sum_j xbar_j W_{k,j}`, then `x_{.,j} -= xbar_j`,
/// `W_{k,j} *= sd_j`, `x_{.,j} /= sd_j`.
fn standardize_positions(x: &mut [Vec<f64>], alpha: &mut [Vec<f64>], w: &mut [Vec<Vec<f64>>]) {
    let a_n = x.len();
    if a_n == 0 {
        return;
    }
    let dd = x[0].len();
    let e = if alpha.is_empty() { 0 } else { alpha[0].len() };
    for j in 0..dd {
        let mean: f64 = x.iter().map(|xa| xa[j]).sum::<f64>() / a_n as f64;
        // Absorb centering into alpha (uses W before scaling).
        for (ak, wk) in alpha.iter_mut().zip(w.iter()) {
            for ee in 0..e {
                ak[ee] += mean * wk[j][ee];
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
                for we in wk[j].iter_mut() {
                    *we *= sd;
                }
            }
        }
    }
}

/// For one topic, the per-author `(logZ_{a,k}, Ebeta_{a,k})`. When the latent space
/// is one-dimensional and there are many authors, `logZ` and `Ebeta` are smooth
/// functions of the scalar position, so they are evaluated exactly on a grid of
/// `G` positions and linearly interpolated to each author. This replaces the
/// `A.V.E` per-topic cost with `G.V.E` (`G << A`). Linear interpolation commutes
/// with differentiation in `alpha`/`W`, so the interpolated value and gradient stay
/// consistent for L-BFGS. For `d > 1` (or few authors) it falls back to exact.
const POSGRID: usize = 64;

fn topic_author_logz_ebeta(
    base_k: &[f64],
    disc_k: &[Vec<f64>],
    x: &[Vec<f64>],
    rho: &[Vec<f64>],
    e: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let a_n = x.len();
    let dd = disc_k.len();
    if dd == 1 && a_n > POSGRID {
        let xs: Vec<f64> = x.iter().map(|xa| xa[0]).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if hi - lo < 1e-9 {
            let (lz, eb) = topic_logz_ebeta(base_k, disc_k, &[lo], rho, e);
            return (vec![lz; a_n], vec![eb; a_n]);
        }
        let g = POSGRID;
        let step = (hi - lo) / (g as f64 - 1.0);
        let mut gl = vec![0.0f64; g];
        let mut ge = vec![vec![0.0f64; e]; g];
        for gi in 0..g {
            let (lz, eb) = topic_logz_ebeta(base_k, disc_k, &[lo + step * gi as f64], rho, e);
            gl[gi] = lz;
            ge[gi] = eb;
        }
        let mut logz = vec![0.0f64; a_n];
        let mut ebeta = vec![vec![0.0f64; e]; a_n];
        for a in 0..a_n {
            let t = (xs[a] - lo) / step;
            let mut gi = t.floor() as usize;
            if gi >= g - 1 {
                gi = g - 2;
            }
            let frac = t - gi as f64;
            logz[a] = gl[gi] * (1.0 - frac) + gl[gi + 1] * frac;
            for ee in 0..e {
                ebeta[a][ee] = ge[gi][ee] * (1.0 - frac) + ge[gi + 1][ee] * frac;
            }
        }
        (logz, ebeta)
    } else {
        let mut logz = vec![0.0f64; a_n];
        let mut ebeta = vec![vec![0.0f64; e]; a_n];
        for a in 0..a_n {
            let (lz, eb) = topic_logz_ebeta(base_k, disc_k, &x[a], rho, e);
            logz[a] = lz;
            ebeta[a] = eb;
        }
        (logz, ebeta)
    }
}

/// Joint L-BFGS update of topic embeddings `alpha` (K x E) and loadings `W`
/// (K x d x E) holding positions fixed. The objective per author and topic is
/// `S_{a,k} . (alpha_k + sum_j x_{a,j} W_{k,j}) - n_{a,k} logZ_{a,k}` with Gaussian
/// priors; the gradient uses `r_{a,k} = S_{a,k} - n_{a,k} Ebeta_{a,k}`:
/// `dalpha_k = sum_a r_{a,k}`, `dW_{k,j} = sum_a x_{a,j} r_{a,k}`. Parallelized over
/// topics (each topic's alpha/W gradients are independent), so the per-author
/// reductions stay in a fixed order and the result is thread-count independent.
#[allow(clippy::too_many_arguments)]
fn update_alpha_w(
    rho: &[Vec<f64>],
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
    let e = alpha[0].len();
    let dd = if w.is_empty() { 0 } else { w[0].len() };
    let a_n = x.len();
    let inv_va = 1.0 / prior_variance;
    let inv_vw = 1.0 / w_prior_variance;
    let alpha_len = k * e;

    let mut x0 = vec![0.0f64; alpha_len + k * dd * e];
    for (kk, ak) in alpha.iter().enumerate() {
        x0[kk * e..(kk + 1) * e].copy_from_slice(ak);
    }
    for kk in 0..k {
        for j in 0..dd {
            let off = alpha_len + (kk * dd + j) * e;
            x0[off..off + e].copy_from_slice(&w[kk][j]);
        }
    }

    let obj = |flat: &[f64]| -> (f64, Vec<f64>) {
        // Cache base/disc for these alpha/W.
        let alpha_v: Vec<Vec<f64>> = (0..k)
            .map(|kk| flat[kk * e..(kk + 1) * e].to_vec())
            .collect();
        let w_v: Vec<Vec<Vec<f64>>> = (0..k)
            .map(|kk| {
                (0..dd)
                    .map(|j| {
                        let off = alpha_len + (kk * dd + j) * e;
                        flat[off..off + e].to_vec()
                    })
                    .collect()
            })
            .collect();
        let (base, disc) = build_cache(rho, &alpha_v, &w_v);

        // Per-topic contribution: value, dalpha_k (E), and dW_{k,j} (d x E). Each
        // topic is independent, so this parallelizes cleanly and the per-author
        // reduction within a topic stays in a fixed order.
        let per_topic: Vec<(f64, Vec<f64>, Vec<Vec<f64>>)> = (0..k)
            .into_par_iter()
            .map(|kk| {
                let (logz, ebeta) = topic_author_logz_ebeta(&base[kk], &disc[kk], x, rho, e);
                let mut val = 0.0;
                let mut g_alpha = vec![0.0f64; e];
                let mut g_w = vec![vec![0.0f64; e]; dd];
                for a in 0..a_n {
                    let n = ntot[a][kk];
                    let s_ak = &s_stat[a][kk];
                    let mut s_dot = 0.0;
                    for ee in 0..e {
                        let mut disp = alpha_v[kk][ee];
                        for j in 0..dd {
                            disp += x[a][j] * w_v[kk][j][ee];
                        }
                        s_dot += s_ak[ee] * disp;
                    }
                    val += s_dot - n * logz[a];
                    for ee in 0..e {
                        let r = s_ak[ee] - n * ebeta[a][ee];
                        g_alpha[ee] += r;
                        for j in 0..dd {
                            g_w[j][ee] += x[a][j] * r;
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
            grad[kk * e..(kk + 1) * e].copy_from_slice(g_alpha);
            for j in 0..dd {
                let off = alpha_len + (kk * dd + j) * e;
                grad[off..off + e].copy_from_slice(&g_w[j]);
            }
        }
        // Gaussian priors on alpha and W.
        for kk in 0..k {
            for ee in 0..e {
                let ai = kk * e + ee;
                value -= 0.5 * inv_va * flat[ai] * flat[ai];
                grad[ai] -= inv_va * flat[ai];
            }
            for j in 0..dd {
                let off = alpha_len + (kk * dd + j) * e;
                for ee in 0..e {
                    value -= 0.5 * inv_vw * flat[off + ee] * flat[off + ee];
                    grad[off + ee] -= inv_vw * flat[off + ee];
                }
            }
        }
        (-value, grad.iter().map(|g| -g).collect())
    };

    let xres = lbfgs_minimize(x0, obj, max_iter, 7, 1e-4);
    for kk in 0..k {
        alpha[kk].copy_from_slice(&xres[kk * e..(kk + 1) * e]);
        for j in 0..dd {
            let off = alpha_len + (kk * dd + j) * e;
            w[kk][j].copy_from_slice(&xres[off..off + e]);
        }
    }
}

/// L-BFGS update of one author's position `x_a` (length d) holding alpha/W fixed.
/// `dQ/dx_j = sum_k [ S_{a,k}.W_{k,j} - n_{a,k} Ebeta_{a,k}.W_{k,j} ] - x_j/var`.
#[allow(clippy::too_many_arguments)]
fn update_position(
    base: &[Vec<f64>],
    disc: &[Vec<Vec<f64>>],
    w: &[Vec<Vec<f64>>],
    s_a: &[Vec<f64>],
    ntot_a: &[f64],
    x_a: &[f64],
    x_prior_variance: f64,
    e: usize,
) -> Vec<f64> {
    let k = base.len();
    let dd = x_a.len();
    let inv_v = 1.0 / x_prior_variance;
    // S_{a,k}.W_{k,j} is constant in x; precompute it once. The x-gradient needs
    // only the scalar disc-projection sum_v beta_v disc_{k,j,v}, never the full
    // E-dimensional Ebeta, so this avoids the embedding factor entirely.
    let s_dot_w: Vec<Vec<f64>> = (0..k)
        .map(|kk| {
            (0..dd)
                .map(|j| (0..e).map(|ee| s_a[kk][ee] * w[kk][j][ee]).sum())
                .collect()
        })
        .collect();
    let obj = |flat: &[f64]| -> (f64, Vec<f64>) {
        let mut value = 0.0;
        let mut grad = vec![0.0f64; dd];
        for kk in 0..k {
            let n = ntot_a[kk];
            let vsz = base[kk].len();
            // eta_v = base_k[v] + sum_j x_j disc_{k,j,v}; softmax for logZ and beta.
            let mut eta = vec![0.0f64; vsz];
            let mut max = f64::NEG_INFINITY;
            for v in 0..vsz {
                let mut et = base[kk][v];
                for (j, &xj) in flat.iter().enumerate() {
                    et += xj * disc[kk][j][v];
                }
                eta[v] = et;
                if et > max {
                    max = et;
                }
            }
            let mut z = 0.0;
            for &ev in &eta {
                z += (ev - max).exp();
            }
            let logz = max + z.ln();
            let beta: Vec<f64> = eta.iter().map(|&ev| (ev - logz).exp()).collect();
            for j in 0..dd {
                // sum_v beta_v disc_{k,j,v} = Ebeta_{a,k} . W_{k,j} without forming Ebeta.
                let eb_disc: f64 = (0..vsz).map(|v| beta[v] * disc[kk][j][v]).sum();
                value += flat[j] * s_dot_w[kk][j];
                grad[j] += s_dot_w[kk][j] - n * eb_disc;
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

impl Estimator for IdealPointModel {
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

    // Well-specified recovery: sample from the model with planted positions and a
    // discriminating topic 0 (neutral topic 1), then check that fitted positions
    // correlate with the truth and that topic 0 carries the higher discrimination.
    #[test]
    fn fit_idealpoint_recovers_positions() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (k, e, dd) = (2usize, 5usize, 1usize);
        let vsz = 30usize;
        let a_n = 24usize;
        // Random word embeddings.
        let rho: Vec<Vec<f64>> = (0..vsz)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        // Topic embeddings: well separated.
        let alpha_true: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        // Topic 0 discriminates (loading scale 2.5); topic 1 is neutral (0).
        let mut w_true = vec![vec![vec![0.0f64; e]; dd]; k];
        for ee in 0..e {
            w_true[0][0][ee] = (rng.gen::<f64>() * 2.0 - 1.0) * 2.5;
        }
        // Planted author positions.
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        // Build each author's per-topic beta and sample documents.
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        for a in 0..a_n {
            let xa = vec![x_true[a]];
            let (base, disc) = build_cache(&rho, &alpha_true, &w_true);
            let beta_a: Vec<Vec<f64>> = (0..k)
                .map(|t| topic_beta(&base[t], &disc[t], &xa))
                .collect();
            for _ in 0..10 {
                let mut doc = Vec::new();
                for _ in 0..40 {
                    let t = rng.gen_range(0..k); // uniform prevalence
                    let wv = sample_cat(&beta_a[t], &mut rng);
                    doc.push(wv as u32);
                }
                docs.push(doc);
                group.push(a);
            }
        }

        let anchors = vec![(0usize, x_true[0]), (1usize, x_true[1])];
        let m = fit_idealpoint(
            &docs, &group, a_n, k, vsz, dd, &rho, &anchors, 40, 1e-6, 0.0, 1e6, 10.0, 1.0, 15,
            &mut rng,
        );

        let x_hat: Vec<f64> = (0..a_n).map(|a| m.x[a][0]).collect();
        let r = pearson(&x_hat, &x_true).abs();
        assert!(r > 0.8, "position correlation too low: {r}");

        // Discrimination concentrates on one topic (topics permute, so do not
        // assume the label): the larger ||W_k|| must dominate the smaller, and the
        // high-discrimination topic must align by word distribution to the planted
        // discriminating topic (planted topic 0), not the neutral one.
        let disc = m.topic_discrimination();
        let (hi, lo) = if disc[0] >= disc[1] { (0, 1) } else { (1, 0) };
        assert!(
            disc[hi] > 2.0 * disc[lo],
            "discrimination should concentrate on one topic: {disc:?}"
        );
        let beta0_true: Vec<Vec<f64>> = {
            let (base, _) = build_cache(&rho, &alpha_true, &vec![vec![vec![0.0; e]; dd]; k]);
            (0..k).map(|t| topic_beta(&base[t], &[], &[])).collect()
        };
        let cos = |p: &[f64], q: &[f64]| -> f64 {
            let dp: f64 = p.iter().zip(q).map(|(a, b)| a * b).sum();
            let np: f64 = p.iter().map(|a| a * a).sum::<f64>().sqrt();
            let nq: f64 = q.iter().map(|a| a * a).sum::<f64>().sqrt();
            dp / (np * nq + 1e-12)
        };
        // The high-discrimination fitted topic matches planted topic 0 best.
        assert!(
            cos(&m.beta0[hi], &beta0_true[0]) > cos(&m.beta0[hi], &beta0_true[1]),
            "high-discrimination topic should align to the planted discriminating topic"
        );
    }

    // Exercises the position-grid interpolation path (A > POSGRID): with many
    // authors the alpha/W M-step interpolates logZ/Ebeta over a grid rather than
    // evaluating every author exactly. Recovery must hold to the same bar.
    #[test]
    fn fit_idealpoint_grid_path_recovers() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (k, e, dd) = (2usize, 5usize, 1usize);
        let vsz = 30usize;
        let a_n = 120usize; // > POSGRID (64), so the grid path is used
        assert!(a_n > POSGRID);
        let rho: Vec<Vec<f64>> = (0..vsz)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let alpha_true: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let mut w_true = vec![vec![vec![0.0f64; e]; dd]; k];
        for ee in 0..e {
            w_true[0][0][ee] = (rng.gen::<f64>() * 2.0 - 1.0) * 2.5;
        }
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let (base, disc) = build_cache(&rho, &alpha_true, &w_true);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        for a in 0..a_n {
            let beta_a: Vec<Vec<f64>> = (0..k)
                .map(|t| topic_beta(&base[t], &disc[t], &[x_true[a]]))
                .collect();
            for _ in 0..6 {
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
        let m = fit_idealpoint(
            &docs, &group, a_n, k, vsz, dd, &rho, &anchors, 40, 1e-6, 0.0, 1e6, 10.0, 1.0, 15,
            &mut rng,
        );
        let x_hat: Vec<f64> = (0..a_n).map(|a| m.x[a][0]).collect();
        let r = pearson(&x_hat, &x_true).abs();
        assert!(r > 0.8, "grid-path position correlation too low: {r}");
    }

    #[test]
    fn idealpoint_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, e) = (3usize, 6usize, 3usize);
        let vsz = k * block;
        let rho: Vec<Vec<f64>> = (0..vsz)
            .map(|w| {
                let b = w / block;
                (0..e)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                    .collect()
            })
            .collect();
        let docs: Vec<Vec<u32>> = (0..60)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let group: Vec<usize> = (0..docs.len()).map(|d| d % 12).collect();
        let m = fit_idealpoint(
            &docs,
            &group,
            12,
            k,
            vsz,
            1,
            &rho,
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
