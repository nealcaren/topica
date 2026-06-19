use crate::linalg::{make_diagonally_dominant, spd_inverse};

/// Empirical-Bayes ("pooled") prevalence regression of per-document latent `λ`
/// on covariates `x` — a faithful port of R `stm`'s `vb.variational.reg`
/// (`gamma.prior = "Pooled"`).
///
/// Returns Γ as `f × n` (`Γ[i][t]` is covariate `i`'s coefficient for latent
/// dimension `t`). For each latent dimension it runs a variational-Bayes ridge
/// that *estimates* the coefficient precision (an adaptive shrinkage strength)
/// and the noise precision from the data, rather than using a fixed ridge. The
/// intercept (column 0, the all-ones column) is left unpenalised. Adaptive
/// shrinkage is what keeps μ = Xγ stable across EM iterations when the design is
/// wide (e.g. a day spline), so EM converges in far fewer iterations than the
/// fixed near-OLS ridge did — matching `stm` (see issue #247).
///
/// `f` is the number of covariates (including the intercept), `n` the number of
/// latent dimensions (`K-1`), `maxits` the inner VB iteration cap (`stm` uses
/// 1000). `X'X` is formed once and shared across the `n` per-dimension solves.
pub fn fit_gamma_vb(
    x: &[Vec<f64>],
    lambda: &[Vec<f64>],
    f: usize,
    n: usize,
    maxits: usize,
) -> Vec<Vec<f64>> {
    let ndoc = x.len();
    // Xcorr = X'X (f×f, row-major), shared across the per-dimension regressions.
    let mut xcorr = vec![0.0f64; f * f];
    for row in x {
        for i in 0..f {
            let ri = row[i];
            for j in 0..f {
                xcorr[i * f + j] += ri * row[j];
            }
        }
    }
    let mut gamma = vec![vec![0.0f64; n]; f];
    for t in 0..n {
        // XYcorr = X'y and y'y for this latent dimension.
        let mut xy = vec![0.0f64; f];
        let mut yty = 0.0f64;
        for (d, row) in x.iter().enumerate() {
            let y = lambda[d][t];
            yty += y * y;
            for i in 0..f {
                xy[i] += row[i] * y;
            }
        }
        let w = vb_reg_column(&xcorr, &xy, yty, f, ndoc, maxits);
        for (i, &wi) in w.iter().enumerate() {
            gamma[i][t] = wi;
        }
    }
    gamma
}

/// One column of `fit_gamma_vb`: the `vb.variational.reg` inner loop for a single
/// response. Works from the precomputed `X'X` (`xcorr`), `X'y` (`xy`) and `y'y`
/// (`yty`) so the full design need not be revisited each iteration. `f` = number
/// of covariates (incl. intercept), `ndoc` = number of documents.
fn vb_reg_column(
    xcorr: &[f64],
    xy: &[f64],
    yty: f64,
    f: usize,
    ndoc: usize,
    maxits: usize,
) -> Vec<f64> {
    let an = (1.0 + ndoc as f64) / 2.0; // noise-precision Gamma shape
    let cn = f as f64; // coefficient-precision Gamma shape (stm uses ncol(X))
    let b0 = 1.0;
    let d0 = 1.0;

    let mut w = vec![0.0f64; f];
    let mut error_prec = 1.0f64; // E[noise precision]
    let mut ba = 1.0f64;
    let mut ea = cn; // E[coefficient precision], init cn/dn with dn=1
    let mut converge = f64::INFINITY;
    let mut ct = 1usize;

    while converge > 1e-4 {
        let w_old = w.clone();

        // invV = error_prec·X'X + diag(0, ea, …, ea)   (intercept unpenalised).
        let mut inv_v = vec![0.0f64; f * f];
        for (dst, &src) in inv_v.iter_mut().zip(xcorr.iter()) {
            *dst = error_prec * src;
        }
        for i in 1..f {
            inv_v[i * f + i] += ea;
        }
        let v = spd_inverse(&inv_v, f).unwrap_or_else(|| {
            let mut a2 = inv_v.clone();
            make_diagonally_dominant(&mut a2, f);
            spd_inverse(&a2, f).expect("VB prevalence covariance not invertible after repair")
        });

        // w = error_prec · V · X'y
        for i in 0..f {
            let mut s = 0.0;
            for j in 0..f {
                s += v[i * f + j] * xy[j];
            }
            w[i] = error_prec * s;
        }

        // sse = w'X'Xw − 2 w'X'y + y'y ; tr(X'X·V) = Σ_ij Xcorr[i,j] V[i,j].
        let mut wxcw = 0.0;
        let mut tr_xcorr_v = 0.0;
        for i in 0..f {
            let mut s = 0.0;
            for j in 0..f {
                let xc = xcorr[i * f + j];
                s += xc * w[j];
                tr_xcorr_v += xc * v[i * f + j];
            }
            wxcw += w[i] * s;
        }
        let wxy: f64 = w.iter().zip(xy).map(|(&wi, &xi)| wi * xi).sum();
        let sse = wxcw - 2.0 * wxy + yty;

        let bn = 0.5 * (sse + tr_xcorr_v) + ba;
        error_prec = an / bn;
        ba = 1.0 / (error_prec + b0);

        // Update the coefficient precision from the penalised coefficients only.
        let da = 2.0 / (ea + d0);
        let mut wpen2 = 0.0;
        let mut trv_pen = 0.0;
        for i in 1..f {
            wpen2 += w[i] * w[i];
            trv_pen += v[i * f + i];
        }
        let dn = 2.0 * da + wpen2 + trv_pen;
        ea = cn / dn;

        converge = w.iter().zip(&w_old).map(|(a, b)| (a - b).abs()).sum();
        ct += 1;
        if ct > maxits {
            // stm raises here; we keep the best-effort coefficients instead of
            // panicking (the surrounding EM still has a valid Γ to proceed with).
            break;
        }
    }
    w
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// `fit_gamma_vb` must reproduce R `stm`'s `vb.variational.reg` coefficients.
    /// Golden values are from `stm:::vb.variational.reg(Y=y, X=X, maxits=1000)`
    /// (stm 1.3.8) on the 8×3 design below (column 0 = intercept, unpenalised).
    #[test]
    fn vb_matches_r_stm_pooled() {
        let x = vec![
            vec![1.0, 0.5, 0.4],
            vec![1.0, -1.0, -0.8],
            vec![1.0, 0.2, 0.1],
            vec![1.0, 1.5, 1.3],
            vec![1.0, -0.7, -0.9],
            vec![1.0, 0.3, 0.6],
            vec![1.0, 2.0, 1.7],
            vec![1.0, -0.4, -0.2],
        ];
        let y = [2.1, -0.5, 1.0, 3.2, -0.8, 1.4, 4.0, 0.3];
        let lambda: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
        let gamma = fit_gamma_vb(&x, &lambda, 3, 1, 1000);
        let r_ref = [0.844306962472, 0.858090356030, 0.857330657160];
        for (i, &expected) in r_ref.iter().enumerate() {
            assert!(
                (gamma[i][0] - expected).abs() < 1e-6,
                "coef {i}: got {}, R stm gives {expected}",
                gamma[i][0]
            );
        }
    }
}
