use crate::linalg::{make_diagonally_dominant, spd_inverse};

/// Pooled ridge regression of per-document latent `λ` on covariates `x`.
///
/// Returns Γ as `f × n`, where `Γ[i][t]` is the coefficient of covariate `i`
/// for latent dimension `t`. Solves `(X'X + ridge·I) Γ = X'Λ` via Cholesky
/// with a diagonal-dominance fallback for near-singular designs.
pub fn fit_gamma_ridge(
    x: &[Vec<f64>],
    lambda: &[Vec<f64>],
    f: usize,
    n: usize,
    ridge: f64,
) -> Vec<Vec<f64>> {
    let (xtx, xtl) = gamma_ss(x, lambda, f, n);
    fit_gamma_ridge_from_ss(&xtx, &xtl, f, n, ridge)
}

/// Accumulate the ridge-regression sufficient statistics `(X'X, X'Λ)` over the
/// supplied documents. `X'X` is `f × f` (row-major) and `X'Λ` is `f × n`
/// (row-major). Split out so the stochastic (minibatch) path can scale and blend
/// these stats before solving; the batch path goes through `fit_gamma_ridge`.
pub fn gamma_ss(x: &[Vec<f64>], lambda: &[Vec<f64>], f: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xtx = vec![0.0f64; f * f];
    let mut xtl = vec![0.0f64; f * n];
    for (xd, ld) in x.iter().zip(lambda) {
        for i in 0..f {
            for j in 0..f {
                xtx[i * f + j] += xd[i] * xd[j];
            }
            for t in 0..n {
                xtl[i * n + t] += xd[i] * ld[t];
            }
        }
    }
    (xtx, xtl)
}

/// Solve `(X'X + ridge·I) Γ = X'Λ` from precomputed sufficient statistics, via
/// Cholesky with a diagonal-dominance fallback. `xtx` is `f × f`, `xtl` is `f × n`
/// (both row-major); `ridge` is added to the diagonal here (so callers blend the
/// raw, un-ridged stats).
pub fn fit_gamma_ridge_from_ss(
    xtx: &[f64],
    xtl: &[f64],
    f: usize,
    n: usize,
    ridge: f64,
) -> Vec<Vec<f64>> {
    let mut a = xtx.to_vec();
    for i in 0..f {
        a[i * f + i] += ridge;
    }
    let inv = spd_inverse(&a, f).unwrap_or_else(|| {
        let mut a2 = a.clone();
        make_diagonally_dominant(&mut a2, f);
        spd_inverse(&a2, f).unwrap()
    });
    let mut gamma = vec![vec![0.0f64; n]; f];
    for i in 0..f {
        for t in 0..n {
            let mut s = 0.0;
            for j in 0..f {
                s += inv[i * f + j] * xtl[j * n + t];
            }
            gamma[i][t] = s;
        }
    }
    gamma
}
