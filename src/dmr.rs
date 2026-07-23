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

/// Clamp on the linear predictor `λ_t · x_d + s_{d,t}` before exponentiation.
/// Standalone DMR deliberately does not standardize covariates, so a large-scale
/// covariate or coefficient can drive the predictor to `+inf` (or `-inf`),
/// yielding non-finite α and NaNs through digamma/trigamma (#419). `exp(500)`
/// (~1.4e217) is astronomically larger than any meaningful α yet leaves ample
/// headroom below `f64::MAX` even summed over many topics, so this bites only
/// pathological inputs and never a realistic fit.
const PREDICTOR_CLAMP: f64 = 500.0;

/// `exp` of the DMR linear predictor with the [`PREDICTOR_CLAMP`] guard.
#[inline]
fn predictor_exp(dot: f64, off: f64) -> f64 {
    (dot + off).clamp(-PREDICTOR_CLAMP, PREDICTOR_CLAMP).exp()
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
                    predictor_exp(dot, off)
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
            let a = predictor_exp(dot, off);
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

/// Standard errors of the DMR feature weights `lambda`, from the observed
/// information of the penalized Dirichlet-multinomial log-likelihood at the fit.
///
/// `lambda` is fit by maximizing the penalized likelihood (see
/// [`dmr_objective_and_gradient`]), so its asymptotic covariance is the inverse of
/// the negative Hessian of that objective. The topics couple through the Dirichlet
/// normalizer `α_{d,·}`, so the Hessian is not block-diagonal across topics; per
/// document `d` it is
/// ```text
///   ∂²L_d/∂λ_{t,f}∂λ_{u,g} = x_{d,f} x_{d,g} ( c_d a_t a_u + [t=u] a_t b_t )
/// ```
/// with `a_t = α_{d,t}`, `c_d = ψ'(α_{d,·}) − ψ'(α_{d,·}+N_d)`,
/// `g_t = ψ(α_·) − ψ(α_·+N_d) + ψ(a_t+n_t) − ψ(a_t)`, and
/// `b_t = a_t(ψ'(a_t+n_t) − ψ'(a_t)) + g_t`. The observed information is
/// `J = (1/σ²)I − Σ_d H_d` (a `(T·F)×(T·F)` SPD matrix); the SE of `λ_{t,f}` is
/// `sqrt(diag(J^{-1}))`. Returns `[num_topics][num_features]`, aligned to `lambda`.
pub fn dmr_lambda_se(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    offset: Option<&[Vec<f64>]>,
) -> Vec<Vec<f64>> {
    let t = num_topics;
    let f = num_features;
    let p = t * f;
    let cov = dmr_lambda_cov(
        lambda,
        features,
        doc_topic_counts,
        num_topics,
        num_features,
        prior_variance,
        offset,
    );
    (0..t)
        .map(|tt| {
            (0..f)
                .map(|ff| {
                    let idx = tt * f + ff;
                    let v = cov[idx * p + idx];
                    if v > 0.0 {
                        v.sqrt()
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect()
}

/// Full asymptotic covariance of the DMR feature weights `λ`: the inverse of the
/// observed information described in [`dmr_lambda_se`], returned as a flattened
/// row-major `(T·F)×(T·F)` matrix (index `(t·F+f)`). [`dmr_lambda_se`] is the
/// square root of its diagonal. The full matrix is needed when `λ` is later
/// transformed by a non-diagonal map (e.g. keyATM's standardization Jacobian),
/// which mixes the within-topic covariance entries.
pub fn dmr_lambda_cov(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    offset: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    use crate::linalg::{make_diagonally_dominant, spd_inverse};
    use crate::optimize::trigamma;

    let t = num_topics;
    let f = num_features;
    let p = t * f;
    // Observed information J, assembled as (1/σ²)I − Σ_d H_d (row-major p x p).
    let mut info = vec![0.0f64; p * p];
    let mut a = vec![0.0f64; t];
    let mut b = vec![0.0f64; t];

    for (d, x) in features.iter().enumerate() {
        let mut alpha_sum = 0.0f64;
        for tt in 0..t {
            let dot: f64 = lambda[tt].iter().zip(x).map(|(l, xi)| l * xi).sum();
            let off = offset.map_or(0.0, |o| o[d][tt]);
            a[tt] = predictor_exp(dot, off);
            alpha_sum += a[tt];
        }
        let counts = &doc_topic_counts[d];
        let n_d: f64 = counts.iter().sum();
        let c_d = trigamma(alpha_sum) - trigamma(alpha_sum + n_d);
        let dg_asum = digamma(alpha_sum);
        let dg_asum_n = digamma(alpha_sum + n_d);
        for tt in 0..t {
            let at = a[tt];
            let n = counts[tt];
            let g_t = dg_asum - dg_asum_n + digamma(at + n) - digamma(at);
            b[tt] = at * (trigamma(at + n) - trigamma(at)) + g_t;
        }
        // Add −H_d = −x_f x_g ( c_d a_t a_u + [t=u] a_t b_t ) into J.
        for tt in 0..t {
            for uu in 0..t {
                let mut coef = -c_d * a[tt] * a[uu];
                if tt == uu {
                    coef -= a[tt] * b[tt];
                }
                if coef == 0.0 {
                    continue;
                }
                for ff in 0..f {
                    let xf = x[ff];
                    if xf == 0.0 {
                        continue;
                    }
                    let row = (tt * f + ff) * p + uu * f;
                    for gg in 0..f {
                        info[row + gg] += coef * xf * x[gg];
                    }
                }
            }
        }
    }
    // Gaussian prior contributes (1/σ²) to the diagonal.
    let inv_var = 1.0 / prior_variance;
    for i in 0..p {
        info[i * p + i] += inv_var;
    }

    spd_inverse(&info, p).unwrap_or_else(|| {
        let mut s = info.clone();
        make_diagonally_dominant(&mut s, p);
        spd_inverse(&s, p).unwrap_or_else(|| vec![f64::NAN; p * p])
    })
}

use crate::mathfun::log_gamma;
use crate::variational::lbfgs_minimize_status;

/// Optimize `lambda` in place to maximize the penalized DMR likelihood for the
/// current topic counts (one L-BFGS run, used periodically during sampling).
///
/// Returns whether L-BFGS reached a stationary point on its own criterion, rather
/// than exhausting its iteration budget. The caller uses this to decide whether the
/// observed-information standard errors are valid — an SE is only a covariance at an
/// optimum, so a run that hit `max_iter` (or was given `max_iter == 0`) reports
/// `false` and no SE is emitted (#419).
#[must_use]
pub fn optimize_lambda(
    lambda: &mut [Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    max_iter: usize,
    offset: Option<&[Vec<f64>]>,
) -> bool {
    let mut x0 = Vec::with_capacity(num_topics * num_features);
    for lt in lambda.iter() {
        x0.extend_from_slice(lt);
    }

    let (x, converged) = lbfgs_minimize_status(
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
    converged
}

/// Guarded standard errors for the DMR weights: [`dmr_lambda_se`] evaluated at the
/// exact counts `lambda` was last optimized against, returning `None` when the
/// observed information is not a valid covariance at the returned `lambda`. The
/// caller must pass the `doc_topic_counts` snapshot from the last *converged*
/// [`optimize_lambda`]; this adds finiteness guards on the objective, gradient, and
/// the resulting SE matrix (the last catching a non-SPD information that the
/// diagonal-dominance fallback could not repair). See #419: previously the SE was
/// computed from the post-sampling counts, which have drifted away from the counts
/// `lambda` was fit to, so it was not `J^{-1}` at the optimum for the returned
/// `lambda`.
pub fn dmr_lambda_se_checked(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    offset: Option<&[Vec<f64>]>,
) -> Option<Vec<Vec<f64>>> {
    let (value, grad) = dmr_objective_and_gradient(
        lambda,
        features,
        doc_topic_counts,
        num_topics,
        num_features,
        prior_variance,
        offset,
    );
    if !value.is_finite() || grad.iter().flatten().any(|g| !g.is_finite()) {
        return None;
    }
    let se = dmr_lambda_se(
        lambda,
        features,
        doc_topic_counts,
        num_topics,
        num_features,
        prior_variance,
        offset,
    );
    if se.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    Some(se)
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

    // The observed-information matrix used for SEs must equal the negative Hessian
    // of the objective: finite-difference the analytic gradient w.r.t. each lambda
    // entry and compare to -(J - prior) reconstructed from dmr_lambda_se's assembly.
    // We check the full Hessian via the gradient's central difference.
    #[test]
    fn lambda_se_hessian_matches_finite_difference() {
        use crate::linalg::spd_inverse;
        let (t, f) = (3usize, 2usize);
        let lambda = vec![vec![0.1, -0.2], vec![-0.3, 0.4], vec![0.05, 0.15]];
        let features = vec![
            vec![1.0, 0.5],
            vec![1.0, -1.0],
            vec![1.0, 2.0],
            vec![1.0, 0.0],
            vec![1.0, 1.3],
        ];
        let counts = vec![
            vec![3.0f64, 1.0, 0.0],
            vec![0.0f64, 2.0, 2.0],
            vec![1.0f64, 1.0, 5.0],
            vec![2.0f64, 0.0, 1.0],
            vec![1.0f64, 3.0, 2.0],
        ];
        let sigma2 = 10.0;
        let p = t * f;

        // Analytic observed information J from the same assembly the SE uses.
        // Rebuild J by inverting the SE-derived covariance is circular, so instead
        // FD the gradient: H[i,j] = d g_i / d lambda_j, and J = -H + (1/sigma2)I.
        let grad_flat = |lam: &[Vec<f64>]| -> Vec<f64> {
            let (_, g) = dmr_objective_and_gradient(lam, &features, &counts, t, f, sigma2, None);
            let mut out = vec![0.0; p];
            for tt in 0..t {
                for ff in 0..f {
                    out[tt * f + ff] = g[tt][ff];
                }
            }
            out
        };
        let eps = 1e-6;
        let mut j_fd = vec![0.0f64; p * p];
        for j in 0..p {
            let (tj, fj) = (j / f, j % f);
            let mut lp = lambda.clone();
            let mut lm = lambda.clone();
            lp[tj][fj] += eps;
            lm[tj][fj] -= eps;
            let gp = grad_flat(&lp);
            let gm = grad_flat(&lm);
            for i in 0..p {
                // J = -H = -(dg/dlambda)
                j_fd[i * p + j] = -(gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        let se_fd: Vec<f64> = {
            let cov = spd_inverse(&j_fd, p).expect("FD info not invertible");
            (0..p).map(|i| cov[i * p + i].sqrt()).collect()
        };
        let se = dmr_lambda_se(&lambda, &features, &counts, t, f, sigma2, None);
        for tt in 0..t {
            for ff in 0..f {
                let analytic = se[tt][ff];
                let numeric = se_fd[tt * f + ff];
                assert!(
                    (analytic - numeric).abs() < 1e-4 * (1.0 + numeric.abs()),
                    "se[{tt}][{ff}]: analytic {analytic} vs FD {numeric}"
                );
            }
        }
    }

    // More documents at the same design shrinks the standard errors.
    #[test]
    fn lambda_se_shrinks_with_more_data() {
        let (t, f) = (2usize, 2usize);
        let lambda = vec![vec![0.0, 0.6], vec![0.0, -0.6]];
        let mk = |reps: usize| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
            let mut features = Vec::new();
            let mut counts = Vec::new();
            for i in 0..reps {
                let cov = if i % 2 == 0 { 1.0 } else { -1.0 };
                features.push(vec![1.0, cov]);
                counts.push(if cov > 0.0 {
                    vec![2.0f64, 8.0]
                } else {
                    vec![8.0f64, 2.0]
                });
            }
            (features, counts)
        };
        let (fs, cs) = mk(40);
        let (fl, cl) = mk(400);
        let se_small = dmr_lambda_se(&lambda, &fs, &cs, t, f, 100.0, None);
        let se_large = dmr_lambda_se(&lambda, &fl, &cl, t, f, 100.0, None);
        for tt in 0..t {
            for ff in 0..f {
                assert!(
                    se_large[tt][ff] < se_small[tt][ff],
                    "se[{tt}][{ff}] should shrink: {} -> {}",
                    se_small[tt][ff],
                    se_large[tt][ff]
                );
                assert!(se_large[tt][ff].is_finite() && se_large[tt][ff] > 0.0);
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
        let converged = optimize_lambda(
            &mut lambda,
            &features,
            &counts,
            num_topics,
            num_features,
            100.0,
            100,
            None,
        );
        assert!(converged, "L-BFGS should reach a stationary point here");

        // The covariate weight should push topic 1 up and topic 0 down.
        let effect_topic1 = lambda[1][1] - lambda[0][1];
        assert!(
            effect_topic1 > 0.5,
            "expected positive covariate effect on topic 1, got {}",
            effect_topic1
        );
    }

    // #419: optimize_lambda reports non-convergence when it cannot take a real step
    // (max_iter = 0), which the SE guard uses to withhold an invalid covariance.
    #[test]
    fn optimize_lambda_reports_nonconvergence_when_it_cannot_step() {
        let features = vec![vec![1.0, 1.0], vec![1.0, -1.0], vec![1.0, 0.5]];
        let counts = vec![vec![5.0, 1.0], vec![1.0, 5.0], vec![3.0, 3.0]];
        let mut lambda = vec![vec![0.0f64; 2]; 2];
        let converged = optimize_lambda(&mut lambda, &features, &counts, 2, 2, 100.0, 0, None);
        assert!(
            !converged,
            "max_iter=0 leaves lambda un-optimized, so it must not report convergence"
        );
    }

    // #419: dmr_lambda_se_checked returns Some finite SEs at a real optimum and None
    // when the objective/gradient are non-finite (e.g. corrupted counts).
    #[test]
    fn dmr_lambda_se_checked_guards_the_covariance() {
        let features = vec![
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            vec![1.0, 0.5],
            vec![1.0, -0.5],
        ];
        let counts = vec![
            vec![8.0, 2.0],
            vec![2.0, 8.0],
            vec![6.0, 4.0],
            vec![4.0, 6.0],
        ];
        let mut lambda = vec![vec![0.0f64; 2]; 2];
        let converged = optimize_lambda(&mut lambda, &features, &counts, 2, 2, 100.0, 200, None);
        assert!(converged);
        let se = dmr_lambda_se_checked(&lambda, &features, &counts, 2, 2, 100.0, None);
        let se = se.expect("SE should be available at a converged optimum");
        assert!(se.iter().flatten().all(|v| v.is_finite() && *v > 0.0));

        // A non-finite count makes the objective non-finite -> no SE.
        let bad_counts = vec![
            vec![f64::NAN, 2.0],
            vec![2.0, 8.0],
            vec![6.0, 4.0],
            vec![4.0, 6.0],
        ];
        assert!(
            dmr_lambda_se_checked(&lambda, &features, &bad_counts, 2, 2, 100.0, None).is_none()
        );
    }

    #[test]
    fn log_gamma_matches_known_values() {
        // ln Γ at integers/half-integers and a couple of interior points.
        // Γ(1)=Γ(2)=1 -> 0; Γ(0.5)=√π -> ln√π; Γ(5)=24; Γ(10)=9! = 362880.
        let cases = [
            (1.0, 0.0),
            (2.0, 0.0),
            (0.5, (std::f64::consts::PI).sqrt().ln()),
            (3.0, 2.0f64.ln()),                    // Γ(3)=2
            (5.0, 24.0f64.ln()),                   // Γ(5)=24
            (10.0, 362_880.0f64.ln()),             // Γ(10)=9!
            (0.1, 9.513_507_698_668_732_f64.ln()), // Γ(0.1)≈9.5135 -> ln Γ(0.1)
        ];
        for (z, expected) in cases {
            let got = log_gamma(z);
            assert!(
                (got - expected).abs() < 1e-9,
                "log_gamma({z}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn log_gamma_finite_for_tiny_argument() {
        // Regression: α = exp(λ·x) can be far below 1 (e.g. exp(-40) ≈ 4.25e-18).
        // The old shift-then-decrement reconstruction rounded z + 1.0 to 1.0 and
        // then hit ln(0) = -inf on the reverse pass, returning +inf. For small z,
        // Γ(z) ≈ 1/z so ln Γ(z) ≈ -ln z.
        for &e in &[-20.0f64, -40.0, -80.0] {
            let z = e.exp();
            let got = log_gamma(z);
            assert!(got.is_finite(), "log_gamma({z}) not finite: {got}");
            let approx = -z.ln(); // leading term of ln Γ for small z
            assert!(
                (got - approx).abs() < 1e-3,
                "log_gamma({z}) = {got}, expected ≈ {approx}"
            );
        }
    }
}
