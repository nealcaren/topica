//! Covariate-effect estimation by the method of composition (R `stm`'s
//! `estimateEffect`). Regresses sampled topic proportions on a prevalence design
//! and pools across draws by Rubin's rules, so the reported standard errors
//! propagate the per-document topic-estimation uncertainty (the Laplace
//! posterior N(lambda_d, nu_d)).
//!
//! This lives in the engine so faSTM, topica, and the Stata plugin share one
//! implementation. Linear (Gaussian) effects on the topic proportions, matching
//! stm's default.

use rand::Rng;

use crate::linalg::{cholesky_jitter, spd_inverse};

/// Draw a standard normal via Box-Muller (avoids a rand_distr dependency).
fn next_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Topic proportions theta (length K = km1+1) from variational means eta (km1):
/// softmax of [eta, 0] (the last topic is the reference).
fn theta_from_eta(eta: &[f64]) -> Vec<f64> {
    let mut e: Vec<f64> = eta.iter().map(|x| x.exp()).collect();
    let s: f64 = e.iter().sum::<f64>() + 1.0;
    e.push(1.0);
    e.iter().map(|x| x / s).collect()
}

/// estimateEffect for one topic. Returns `(coef, vcov)` where `coef` has length P
/// (the design columns, e.g. intercept + covariates) and `vcov` is the P×P pooled
/// covariance (row-major). The standard errors are `sqrt(vcov[j*P+j])`.
///
/// - `lambda`: D × (K-1) variational means.
/// - `nu`: D × (K-1)² row-major Laplace covariances; pass empty to treat the
///   topic proportions as fixed (no uncertainty propagation → single OLS).
/// - `x`: D × P design (include an intercept column yourself).
/// - `topic`: 0..K.
pub fn estimate_effect_topic<R: Rng>(
    lambda: &[Vec<f64>],
    nu: &[Vec<f64>],
    x: &[Vec<f64>],
    topic: usize,
    n_sims: usize,
    rng: &mut R,
) -> (Vec<f64>, Vec<f64>) {
    let d = lambda.len();
    let km1 = if d > 0 { lambda[0].len() } else { 0 };
    let p = if d > 0 { x[0].len() } else { 0 };
    if d == 0 || p == 0 || d <= p {
        return (vec![0.0; p], vec![0.0; p * p]);
    }

    // (X'X)^-1, and M = (X'X)^-1 X'  (P×D) — fixed across draws.
    let mut xtx = vec![0.0f64; p * p];
    for row in x.iter() {
        for i in 0..p {
            for j in 0..p {
                xtx[i * p + j] += row[i] * row[j];
            }
        }
    }
    let xtxinv = spd_inverse(&xtx, p).unwrap_or_else(|| {
        // Fall back to a jittered inverse if X'X is near-singular.
        let l = cholesky_jitter(&xtx, p);
        crate::linalg::spd_inverse_from_chol(&l, p)
    });
    let mut m = vec![0.0f64; p * d]; // P×D
    for pp in 0..p {
        for dd in 0..d {
            let mut s = 0.0;
            for j in 0..p {
                s += xtxinv[pp * p + j] * x[dd][j];
            }
            m[pp * d + dd] = s;
        }
    }

    // Per-document Cholesky of nu (for the MVN draw); None when nu is absent.
    let have_nu = nu.len() == d && km1 > 0;
    let chols: Vec<Vec<f64>> = if have_nu {
        nu.iter().map(|nd| cholesky_jitter(nd, km1)).collect()
    } else {
        Vec::new()
    };

    let sims = if have_nu { n_sims.max(1) } else { 1 };
    let mut beta_draws: Vec<Vec<f64>> = Vec::with_capacity(sims);
    let mut sigma2_draws: Vec<f64> = Vec::with_capacity(sims);

    for _ in 0..sims {
        // Sampled outcome y_d = sampled proportion of `topic` in doc d.
        let mut y = vec![0.0f64; d];
        for dd in 0..d {
            let eta: Vec<f64> = if have_nu {
                let l = &chols[dd];
                let z: Vec<f64> = (0..km1).map(|_| next_normal(rng)).collect();
                (0..km1)
                    .map(|i| {
                        let mut v = lambda[dd][i];
                        for j in 0..=i {
                            v += l[i * km1 + j] * z[j];
                        }
                        v
                    })
                    .collect()
            } else {
                lambda[dd].clone()
            };
            y[dd] = theta_from_eta(&eta)[topic];
        }

        // beta = M y
        let mut beta = vec![0.0f64; p];
        for pp in 0..p {
            let mut s = 0.0;
            for dd in 0..d {
                s += m[pp * d + dd] * y[dd];
            }
            beta[pp] = s;
        }
        // RSS, sigma2
        let mut rss = 0.0;
        for dd in 0..d {
            let mut fit = 0.0;
            for j in 0..p {
                fit += x[dd][j] * beta[j];
            }
            rss += (y[dd] - fit).powi(2);
        }
        let sigma2 = rss / (d - p) as f64;
        beta_draws.push(beta);
        sigma2_draws.push(sigma2);
    }

    // Pool by Rubin's rules into the full P×P covariance.
    let b = sims as f64;
    let mut coef = vec![0.0f64; p];
    for draw in &beta_draws {
        for j in 0..p {
            coef[j] += draw[j] / b;
        }
    }
    let sigbar: f64 = sigma2_draws.iter().sum::<f64>() / b;
    let mut vcov = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            // within = mean_b sigma2_b * (X'X)^-1_ij  (sigma2 averaged over draws)
            let within = sigbar * xtxinv[i * p + j];
            // between = sample covariance of (beta_i, beta_j) across draws
            let between = if sims > 1 {
                beta_draws
                    .iter()
                    .map(|dr| (dr[i] - coef[i]) * (dr[j] - coef[j]))
                    .sum::<f64>()
                    / (b - 1.0)
            } else {
                0.0
            };
            vcov[i * p + j] = within + (1.0 + 1.0 / b) * between;
        }
    }
    (coef, vcov)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn recovers_a_planted_linear_effect() {
        // K=2 (km1=1): topic-0 proportion rises with covariate x.
        // Build lambda so that softmax([eta,0])[0] tracks a linear signal in x.
        let d = 200;
        let mut lambda = Vec::new();
        let mut x = Vec::new();
        for i in 0..d {
            let cov = (i as f64) / (d as f64); // 0..1
                                               // eta chosen so proportion increases with cov
            let eta = -2.0 + 4.0 * cov;
            lambda.push(vec![eta]);
            x.push(vec![1.0, cov]); // intercept + covariate
        }
        let nu: Vec<Vec<f64>> = Vec::new(); // no uncertainty -> single OLS
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (coef, vcov) = estimate_effect_topic(&lambda, &nu, &x, 0, 50, &mut rng);
        // Proportion clearly increases with the covariate: positive slope.
        assert!(coef[1] > 0.1, "slope should be positive, got {}", coef[1]);
        // Diagonal of vcov is the variance of each coef.
        let p = 2;
        assert!(vcov[1 * p + 1] >= 0.0 && vcov[1 * p + 1].is_finite());
    }

    #[test]
    fn uncertainty_widens_se() {
        let d = 100;
        let mut lambda = Vec::new();
        let mut nu = Vec::new();
        let mut x = Vec::new();
        for i in 0..d {
            let cov = (i % 2) as f64;
            lambda.push(vec![0.5 * cov]);
            nu.push(vec![0.5]); // 1x1 covariance per doc
            x.push(vec![1.0, cov]);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let p = 2;
        let (_c0, v0) = estimate_effect_topic(&lambda, &Vec::new(), &x, 0, 100, &mut rng);
        let (_c1, v1) = estimate_effect_topic(&lambda, &nu, &x, 0, 200, &mut rng);
        let se0 = v0[1 * p + 1].sqrt();
        let se1 = v1[1 * p + 1].sqrt();
        // Propagating per-doc uncertainty should not shrink the SE below the
        // no-uncertainty case.
        assert!(se1 + 1e-9 >= se0, "se with nu {} vs without {}", se1, se0);
    }
}
