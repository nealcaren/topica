//! Dirichlet-Multinomial Regression (DMR) topic model (Mimno & McCallum, 2008).
//!
//! DMR replaces LDA's fixed document-topic prior with a per-document prior that
//! is a log-linear function of document features:
//!
//! ```text
//!     α_{d,t} = exp(λ_t · x_d)
//! ```
//!
//! where `x_d` is document `d`'s feature vector and `λ_t` is a learned weight
//! vector for topic `t`. Sampling is ordinary SparseLDA with this per-document
//! prior (we reuse [`crate::sampler::sample_doc`] verbatim, recomputing the
//! smoothing mass per document); the weights `λ` are fit by maximizing the
//! penalized likelihood of the topic assignments via L-BFGS.
//!
//! This module provides the per-document-α sweep and the objective/gradient;
//! the optimizer and Python surface build on top of it.

use rand::Rng;

use crate::optimize::digamma;
use crate::sampler::sample_doc;

/// Stirling-series log Γ. Shifts the argument to z ≥ 10 before applying the
/// asymptotic series so the result (and, importantly for the optimizer, its
/// numerical derivative) is accurate to ~1e-10. This is a local copy used only
/// by the DMR objective; LDA's MALLET-matched log Γ lives in `output.rs`.
fn log_gamma(mut z: f64) -> f64 {
    const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
    let mut shift = 0i32;
    while z < 10.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

/// Per-document, per-topic prior `α_{d,t} = exp(λ_t · x_d + s_{d,t})`.
///
/// `lambda` is `[num_topics][num_features]`, `features` is
/// `[num_docs][num_features]`; returns `[num_docs][num_topics]`. The optional
/// `offset` is a fixed `[num_docs][num_topics]` term added inside the exponent
/// (the embedding anchor `s_{d,t}`); pass `None` for the plain DMR prior.
pub fn compute_doc_alpha(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    offset: Option<&[Vec<f64>]>,
) -> Vec<Vec<f64>> {
    features
        .iter()
        .enumerate()
        .map(|(d, x)| {
            lambda
                .iter()
                .enumerate()
                .map(|(t, lt)| {
                    let dot: f64 = lt.iter().zip(x).map(|(l, xi)| l * xi).sum();
                    let off = offset.map_or(0.0, |o| o[d][t]);
                    (dot + off).exp()
                })
                .collect()
        })
        .collect()
}

/// One Gibbs sweep with a per-document prior `doc_alpha[d]` (DMR).
///
/// Identical to [`crate::sampler::run_sweep`] except the smoothing mass and
/// per-topic coefficients are recomputed for each document's own α vector
/// before sampling that document.
#[allow(clippy::too_many_arguments)]
pub fn run_sweep_dmr<R: Rng>(
    type_topic_counts: &mut [Vec<u32>],
    tokens_per_topic: &mut [u32],
    doc_topics: &mut [Vec<u32>],
    docs: &[Vec<u32>],
    doc_alpha: &[Vec<f64>],
    beta: f64,
    beta_sum: f64,
    topic_mask: u32,
    topic_bits: u32,
    num_topics: usize,
    rng: &mut R,
) {
    let mut cached_coefficients = vec![0.0f64; num_topics];
    let mut local_topic_counts = vec![0u32; num_topics];
    let mut local_topic_index = vec![0u32; num_topics];
    let mut scored_positions = vec![0usize; num_topics];
    let mut scored_values = vec![0.0f64; num_topics];

    for doc_idx in 0..docs.len() {
        let alpha = &doc_alpha[doc_idx];

        // Recompute the smoothing-only mass and coefficients for this document.
        let mut smoothing_only_mass = 0.0f64;
        for t in 0..num_topics {
            let denom = tokens_per_topic[t] as f64 + beta_sum;
            smoothing_only_mass += alpha[t] * beta / denom;
            cached_coefficients[t] = alpha[t] / denom;
        }

        sample_doc(
            type_topic_counts,
            tokens_per_topic,
            doc_topics,
            alpha,
            beta,
            beta_sum,
            topic_mask,
            topic_bits,
            num_topics,
            &docs[doc_idx],
            doc_idx,
            rng,
            &mut smoothing_only_mass,
            &mut cached_coefficients,
            &mut local_topic_counts,
            &mut local_topic_index,
            &mut scored_positions,
            &mut scored_values,
        );
    }
}

/// Penalized DMR log-likelihood of the current topic counts and its gradient
/// w.r.t. `lambda`.
///
/// For each document `d` with topic counts `n_{d,t}` (total `N_d`):
/// ```text
///   L = Σ_d [ logΓ(α_{d,·}) − logΓ(N_d + α_{d,·})
///             + Σ_t ( logΓ(n_{d,t} + α_{d,t}) − logΓ(α_{d,t}) ) ]
///       − 1/(2σ²) Σ_{t,f} λ_{t,f}²
/// ```
/// with `α_{d,t} = exp(λ_t · x_d + s_{d,t})` and `α_{d,·} = Σ_t α_{d,t}`. The
/// optional `offset` `s` is a fixed `[num_docs][num_topics]` term added inside
/// the exponent; since it is constant in `λ`, the gradient still uses
/// `∂α_{d,t}/∂λ_{t,f} = α_{d,t} · x_{d,f}`.
///
/// `doc_topic_counts` is `[num_docs][num_topics]`. Returns `(value, gradient)`
/// where `gradient` matches the shape of `lambda` (`[num_topics][num_features]`).
pub fn dmr_objective_and_gradient(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    offset: Option<&[Vec<f64>]>,
) -> (f64, Vec<Vec<f64>>) {
    let mut value = 0.0f64;
    let mut grad = vec![vec![0.0f64; num_features]; num_topics];

    let mut alpha = vec![0.0f64; num_topics];

    for (d, x) in features.iter().enumerate() {
        let mut alpha_sum = 0.0f64;
        for t in 0..num_topics {
            let dot: f64 = lambda[t].iter().zip(x).map(|(l, xi)| l * xi).sum();
            let off = offset.map_or(0.0, |o| o[d][t]);
            let a = (dot + off).exp();
            alpha[t] = a;
            alpha_sum += a;
        }

        let counts = &doc_topic_counts[d];
        let n_d: f64 = counts.iter().sum();

        value += log_gamma(alpha_sum) - log_gamma(alpha_sum + n_d);
        let dg_alpha_sum = digamma(alpha_sum);
        let dg_alpha_sum_n = digamma(alpha_sum + n_d);

        for t in 0..num_topics {
            let a = alpha[t];
            let n = counts[t];
            value += log_gamma(a + n) - log_gamma(a);

            // ∂L/∂α_{d,t}, then chain through ∂α/∂λ = α · x.
            let dl_da = dg_alpha_sum - dg_alpha_sum_n + digamma(a + n) - digamma(a);
            let coef = dl_da * a;
            let gt = &mut grad[t];
            for f in 0..num_features {
                gt[f] += coef * x[f];
            }
        }
    }

    // Gaussian prior N(0, σ²) on every weight.
    let inv_var = 1.0 / prior_variance;
    for t in 0..num_topics {
        for f in 0..num_features {
            value -= 0.5 * inv_var * lambda[t][f] * lambda[t][f];
            grad[t][f] -= inv_var * lambda[t][f];
        }
    }

    (value, grad)
}

use crate::variational::lbfgs_minimize;

/// Optimize `lambda` in place to maximize the penalized DMR likelihood for the
/// current topic counts (one L-BFGS run, used periodically during sampling).
pub fn optimize_lambda(
    lambda: &mut [Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    max_iter: usize,
    offset: Option<&[Vec<f64>]>,
) {
    let mut x0 = Vec::with_capacity(num_topics * num_features);
    for lt in lambda.iter() {
        x0.extend_from_slice(lt);
    }

    let x = lbfgs_minimize(
        x0,
        |flat| {
            let mut lam = vec![vec![0.0f64; num_features]; num_topics];
            for t in 0..num_topics {
                lam[t].copy_from_slice(&flat[t * num_features..(t + 1) * num_features]);
            }
            // We minimize, so negate the (maximization) objective and gradient.
            let (val, grad) = dmr_objective_and_gradient(
                &lam,
                features,
                doc_topic_counts,
                num_topics,
                num_features,
                prior_variance,
                offset,
            );
            let mut g = Vec::with_capacity(num_topics * num_features);
            for gt in &grad {
                g.extend(gt.iter().map(|v| -v));
            }
            (-val, g)
        },
        max_iter,
        7,
        1e-5,
    );

    for t in 0..num_topics {
        lambda[t].copy_from_slice(&x[t * num_features..(t + 1) * num_features]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The analytic gradient must match a finite-difference estimate.
    #[test]
    fn gradient_matches_finite_difference() {
        let num_topics = 3;
        let num_features = 2;
        let lambda = vec![vec![0.1, -0.2], vec![-0.3, 0.4], vec![0.05, 0.15]];
        let features = vec![
            vec![1.0, 0.5],
            vec![1.0, -1.0],
            vec![1.0, 2.0],
            vec![1.0, 0.0],
        ];
        let counts = vec![
            vec![3.0f64, 1.0, 0.0],
            vec![0.0f64, 2.0, 2.0],
            vec![1.0f64, 1.0, 5.0],
            vec![2.0f64, 0.0, 1.0],
        ];
        let sigma2 = 10.0;

        let (_, grad) = dmr_objective_and_gradient(
            &lambda,
            &features,
            &counts,
            num_topics,
            num_features,
            sigma2,
            None,
        );

        let eps = 1e-6;
        for t in 0..num_topics {
            for f in 0..num_features {
                let mut lp = lambda.clone();
                let mut lm = lambda.clone();
                lp[t][f] += eps;
                lm[t][f] -= eps;
                let (vp, _) = dmr_objective_and_gradient(
                    &lp,
                    &features,
                    &counts,
                    num_topics,
                    num_features,
                    sigma2,
                    None,
                );
                let (vm, _) = dmr_objective_and_gradient(
                    &lm,
                    &features,
                    &counts,
                    num_topics,
                    num_features,
                    sigma2,
                    None,
                );
                let numeric = (vp - vm) / (2.0 * eps);
                assert!(
                    (numeric - grad[t][f]).abs() < 1e-4,
                    "grad[{}][{}]: analytic {} vs numeric {}",
                    t,
                    f,
                    grad[t][f],
                    numeric
                );
            }
        }
    }

    // L-BFGS should recover known feature effects from synthetic topic counts.
    #[test]
    fn lbfgs_recovers_synthetic_effects() {
        // Two features (intercept + one covariate), two topics. Construct counts
        // where topic 1 is strongly favored when the covariate is high.
        let num_topics = 2;
        let num_features = 2;
        let mut features = Vec::new();
        let mut counts = Vec::new();
        for i in 0..200 {
            let cov = if i % 2 == 0 { 1.0 } else { -1.0 };
            features.push(vec![1.0, cov]);
            // High covariate -> more topic 1; low -> more topic 0.
            if cov > 0.0 {
                counts.push(vec![2.0f64, 8.0]);
            } else {
                counts.push(vec![8.0f64, 2.0]);
            }
        }
        let mut lambda = vec![vec![0.0f64; num_features]; num_topics];
        optimize_lambda(
            &mut lambda,
            &features,
            &counts,
            num_topics,
            num_features,
            100.0,
            100,
            None,
        );

        // The covariate weight should push topic 1 up and topic 0 down.
        let effect_topic1 = lambda[1][1] - lambda[0][1];
        assert!(
            effect_topic1 > 0.5,
            "expected positive covariate effect on topic 1, got {}",
            effect_topic1
        );
    }
}
