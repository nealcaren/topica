//! ETM: the Embedded Topic Model (Dieng, Ruiz & Blei 2020).
//!
//! ETM is LDA with two changes: the topic-word distribution is factored through
//! embeddings, and the document-topic prior is logistic-normal. Each topic `k` is
//! a point `alpha_k` in the embedding space, each word `v` is a point `rho_v`, and
//!
//! ```text
//!   beta_{k,v} = softmax_v( rho_v . alpha_k )
//! ```
//!
//! so semantically related words share probability mass even when a topic never
//! saw them. The reference fits ETM with a variational autoencoder; topica fits it
//! on the same variational-EM core as [`crate::ctm`]:
//!
//! - E-step: per document, the logistic-normal Laplace approximation
//!   [`crate::ctm::ctm_hpb`], with `beta` computed from the embeddings. This is
//!   CTM's E-step unchanged.
//! - M-step: update the topic embeddings `alpha` (the word embeddings `rho` are
//!   fixed to the pretrained vectors by default) by L-BFGS against the expected
//!   token-topic counts, then rebuild `beta`. The objective and gradient are the
//!   softmax-cross-entropy form already used for the SAGE content model, with the
//!   embedding inner product in place of the per-word deviations.
//!
//! Per-document optimization avoids the amortization gap of the VAE encoder, so
//! the fit is at least as accurate; the trade is the usual variational-EM one,
//! less scalability to very large corpora than minibatch SGD. A future option is
//! an amortized VAE inference path (an encoder plus reparameterized minibatch
//! SGD) for corpora too large for the per-document E-step; it would reproduce the
//! reference's scaling at some cost in per-document posterior accuracy.

use crate::ctm::{ctm_grad, ctm_hpb, ctm_lhood, HpbResult};
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse};
use crate::variational::{doc_sparse, lbfgs_minimize};
use rand::Rng;
use rayon::prelude::*;

/// The topic-word matrix `beta[k][v] = softmax_v(rho_v . alpha_k)`.
///
/// `rho` is `V x E` (word embeddings), `alpha` is `K x E` (topic embeddings);
/// returns `K x V` rows that each sum to one.
pub fn softmax_beta(rho: &[Vec<f64>], alpha: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let v = rho.len();
    alpha
        .iter()
        .map(|ak| {
            let mut eta = vec![0.0f64; v];
            let mut max = f64::NEG_INFINITY;
            for (w, rv) in rho.iter().enumerate() {
                let dot: f64 = rv.iter().zip(ak).map(|(r, a)| r * a).sum();
                eta[w] = dot;
                if dot > max {
                    max = dot;
                }
            }
            let mut z = 0.0;
            for e in &eta {
                z += (e - max).exp();
            }
            let log_z = max + z.ln();
            eta.iter().map(|e| (e - log_z).exp()).collect()
        })
        .collect()
}

/// M-step: update the topic embeddings `alpha` in place to maximize the expected
/// complete-data log-likelihood of the embedding-factored `beta`, then return the
/// rebuilt `beta`.
///
/// `counts[k][v]` are the expected token-topic counts from the E-step (summed over
/// documents). For each topic the objective is the softmax cross-entropy
/// `sum_v counts[k][v] * (rho_v . alpha_k - log Z_k)` with a Gaussian prior on
/// `alpha`, and the gradient is `sum_v (counts[k][v] - n_k * beta_{k,v}) * rho_v`.
/// `rho` is held fixed (the pretrained word embeddings).
pub fn optimize_topic_embeddings(
    rho: &[Vec<f64>],
    alpha: &mut [Vec<f64>],
    counts: &[Vec<f64>],
    prior_variance: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let k = alpha.len();
    let v = rho.len();
    let e = if k > 0 { alpha[0].len() } else { 0 };
    let nk: Vec<f64> = counts.iter().map(|row| row.iter().sum()).collect();
    let inv_var = 1.0 / prior_variance;

    let mut x0 = Vec::with_capacity(k * e);
    for ak in alpha.iter() {
        x0.extend_from_slice(ak);
    }

    let x = lbfgs_minimize(
        x0,
        |flat| {
            let mut value = 0.0f64;
            let mut grad = vec![0.0f64; flat.len()];
            for topic in 0..k {
                let a = &flat[topic * e..(topic + 1) * e];
                // eta_v = rho_v . alpha_k, with a log-sum-exp for numerical safety.
                let mut eta = vec![0.0f64; v];
                let mut max = f64::NEG_INFINITY;
                for (w, rv) in rho.iter().enumerate() {
                    let dot: f64 = rv.iter().zip(a).map(|(r, av)| r * av).sum();
                    eta[w] = dot;
                    if dot > max {
                        max = dot;
                    }
                }
                let mut z = 0.0;
                for ev in &eta {
                    z += (ev - max).exp();
                }
                let log_z = max + z.ln();
                let g = &mut grad[topic * e..(topic + 1) * e];
                for (w, rv) in rho.iter().enumerate() {
                    let n = counts[topic][w];
                    value += n * (eta[w] - log_z);
                    let beta = (eta[w] - log_z).exp();
                    let resid = n - nk[topic] * beta;
                    for (ge, &r) in g.iter_mut().zip(rv) {
                        *ge += resid * r;
                    }
                }
            }
            // Gaussian prior N(0, prior_variance) on every embedding coordinate.
            for (i, &xi) in flat.iter().enumerate() {
                value -= 0.5 * inv_var * xi * xi;
                grad[i] -= inv_var * xi;
            }
            // lbfgs minimizes, so negate the maximization objective and gradient.
            (-value, grad.iter().map(|gv| -gv).collect())
        },
        max_iter,
        7,
        1e-4,
    );

    for topic in 0..k {
        alpha[topic].copy_from_slice(&x[topic * e..(topic + 1) * e]);
    }
    softmax_beta(rho, alpha)
}

/// A fitted ETM. `beta` (K×V) is the topic-word matrix, `alpha` (K×E) the topic
/// embeddings; `lambda` are the per-document logistic-normal variational means
/// (θ = softmax([η, 0])), and `mu`/`sigma` the document-topic prior.
pub struct EtmModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub beta: Vec<Vec<f64>>,
    pub alpha: Vec<Vec<f64>>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub lambda: Vec<Vec<f64>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl EtmModel {
    /// Per-document topic proportions θ = softmax([η, 0]) (D×K).
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
}

/// Fit ETM by variational EM. The E-step is CTM's logistic-normal Laplace step
/// (`ctm_hpb`) with `beta = softmax(rho · alpha)`; the M-step updates the prior
/// `mu`/`sigma` exactly as CTM does and the topic embeddings `alpha` via
/// [`optimize_topic_embeddings`]. `rho` (the V×E word embeddings) is fixed.
///
/// `prior_variance` is the Gaussian prior on the topic embeddings (large = weak),
/// `max_inner` caps the per-iteration L-BFGS steps for the embedding M-step, and
/// `em_tol` stops EM on the relative change in the corpus bound.
#[allow(clippy::too_many_arguments)]
pub fn fit_etm<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    rho: &[Vec<f64>],
    em_iters: usize,
    em_tol: f64,
    sigma_shrink: f64,
    prior_variance: f64,
    max_inner: usize,
    rng: &mut R,
) -> EtmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let e = if num_types > 0 { rho[0].len() } else { 0 };
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Initialize the topic embeddings at K distinct words' embeddings (plus a
    // little jitter), so topics start differentiated.
    let mut idx: Vec<usize> = (0..num_types).collect();
    for i in 0..num_types.min(k) {
        let j = (i + (rng.gen::<f64>() * (num_types - i) as f64) as usize).min(num_types - 1);
        idx.swap(i, j);
    }
    let mut alpha = vec![vec![0.0f64; e]; k];
    for (t, ak) in alpha.iter_mut().enumerate() {
        let src = if num_types > 0 {
            &rho[idx[t % num_types]]
        } else {
            &vec![0.0; e]
        };
        for (ae, &r) in ak.iter_mut().zip(src) {
            *ae = r + (rng.gen::<f64>() - 0.5) * 0.01;
        }
    }
    let mut beta = softmax_beta(rho, &alpha);

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
        let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, km1);
            spd_inverse(&s, km1).unwrap()
        });
        let entropy = match cholesky(&sigma, km1) {
            Some(l) => half_logdet(&l, km1),
            None => 0.0,
        };

        let mut beta_ss = vec![vec![1e-8f64; num_types]; k];
        let mut sigma_ss = vec![0.0f64; km1 * km1];
        let mut lambda_sum = vec![0.0f64; km1];

        // E-step: per-document logistic-normal variational inference, in parallel
        // then accumulated in document order for determinism (as in fit_ctm).
        let doc_results: Vec<(usize, Vec<f64>, HpbResult)> = sparse
            .par_iter()
            .enumerate()
            .filter(|(_, (words, _))| !words.is_empty())
            .map(|(di, (words, counts))| {
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| {
                        (
                            ctm_lhood(eta, &beta, words, counts, &mu_shared, &siginv),
                            ctm_grad(eta, &beta, words, counts, &mu_shared, &siginv),
                        )
                    },
                    40,
                    7,
                    1e-5,
                );
                // ETM uses the full Laplace covariance (diagonal=false).
                let res = ctm_hpb(
                    &opt, &beta, words, counts, &mu_shared, &siginv, entropy, false,
                );
                (di, opt, res)
            })
            .collect();

        let total_bound: f64 = doc_results.iter().map(|(_, _, r)| r.bound).sum();
        bound_history.push(total_bound);

        for (di, opt, res) in &doc_results {
            let di = *di;
            let words = &sparse[di].0;
            lambda[di] = opt.clone();
            for (wi, &w) in words.iter().enumerate() {
                for t in 0..k {
                    beta_ss[t][w] += res.phi[t][wi];
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

        // M-step: shared prior mean μ and covariance Σ (as in CTM).
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

        // M-step: topic embeddings α, rebuilding β = softmax(ρ·α).
        beta = optimize_topic_embeddings(rho, &mut alpha, &beta_ss, prior_variance, max_inner);
    }

    EtmModel {
        num_topics: k,
        num_types,
        beta,
        alpha,
        mu: mu_shared,
        sigma,
        lambda,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for EtmModel {
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
        // ETM does not store the Laplace covariance, so it is not a Tier-2
        // logistic-normal model; mirrors conformance.py.
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q)
            .map(|(&pi, &qi)| {
                if pi > 0.0 {
                    pi * (pi / qi.max(1e-12)).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    #[test]
    fn softmax_beta_rows_are_distributions() {
        let rho = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let alpha = vec![vec![2.0, -1.0], vec![-1.0, 2.0]];
        let beta = softmax_beta(&rho, &alpha);
        assert_eq!(beta.len(), 2);
        for row in &beta {
            assert_eq!(row.len(), 3);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(row.iter().all(|&p| p > 0.0));
        }
    }

    // Given expected counts generated from a planted beta, the M-step should
    // recover topic embeddings whose beta matches the planted one.
    #[test]
    fn m_step_recovers_planted_beta() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (k, v, e) = (3, 30, 8);
        // Random word embeddings (fixed) and planted topic embeddings.
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let alpha_true: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let beta_true = softmax_beta(&rho, &alpha_true);
        // Expected counts: a large multiple of the planted beta, so the M-step
        // target is essentially beta_true.
        let counts: Vec<Vec<f64>> = beta_true
            .iter()
            .map(|row| row.iter().map(|&p| p * 5000.0).collect())
            .collect();

        let mut alpha = vec![vec![0.0f64; e]; k]; // start from zero
        let beta_hat = optimize_topic_embeddings(&rho, &mut alpha, &counts, 100.0, 200);

        for t in 0..k {
            assert!(
                kl(&beta_true[t], &beta_hat[t]) < 1e-3,
                "topic {t} KL too large"
            );
        }
    }

    // Full EM on a planted corpus: K blocks of words, each word's embedding points
    // along its block's axis, and each document draws from one block. ETM should
    // recover topics whose top words come from a single block, covering all blocks.
    #[test]
    fn fit_etm_recovers_planted_blocks() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, e) = (3usize, 8usize, 3usize);
        let v = k * block;
        // Word embeddings: word in block b points along axis b (plus small noise).
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..e)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                    .collect()
            })
            .collect();
        // Documents: doc d draws 10 words from block d % k.
        let docs: Vec<Vec<u32>> = (0..90)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();

        let m = fit_etm(&docs, k, v, &rho, 50, 1e-5, 0.0, 1e6, 25, &mut rng);
        assert_eq!(m.beta.len(), k);
        // Each fitted topic's top words come from one block, and all blocks appear.
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut order: Vec<usize> = (0..v).collect();
            order.sort_by(|&a, &b| m.beta[t][b].total_cmp(&m.beta[t][a]));
            let blocks: std::collections::HashSet<usize> =
                order[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic {t} top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k, "topics did not cover all {k} blocks");
        // doc_topic rows are valid distributions.
        for row in m.doc_topics() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn etm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, e) = (3usize, 8usize, 3usize);
        let v = k * block;
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..e)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                    .collect()
            })
            .collect();
        let docs: Vec<Vec<u32>> = (0..90)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let m = fit_etm(&docs, k, v, &rho, 50, 1e-5, 0.0, 1e6, 25, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
