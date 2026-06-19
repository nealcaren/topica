//! Minimal dense linear algebra for the variational engine: Cholesky,
//! symmetric-positive-definite inverse, and log-determinant for small,
//! row-major K×K matrices. Hand-rolled to avoid a LAPACK/BLAS dependency —
//! the variational E-step works in (num_topics − 1) dimensions, which is small.

/// Lower-triangular Cholesky factor `L` (row-major) with `L Lᵀ = A`.
/// Returns `None` if `A` is not positive definite.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// `0.5·log|A|` from a Cholesky factor: `Σ log L_ii`.
pub fn half_logdet(l: &[f64], n: usize) -> f64 {
    (0..n).map(|i| l[i * n + i].ln()).sum()
}

/// Inverse of a lower-triangular matrix `l` (row-major).
fn invert_lower(l: &[f64], n: usize) -> Vec<f64> {
    let mut li = vec![0.0f64; n * n];
    for i in 0..n {
        li[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s += l[i * n + k] * li[k * n + j];
            }
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    li
}

/// Inverse of an SPD matrix from its Cholesky factor: `A⁻¹ = L⁻ᵀ L⁻¹`.
///
/// The result is symmetric, so only the lower triangle (`j ≤ i`) is summed and
/// mirrored. `L⁻¹` is lower-triangular, so `L⁻¹_{ki}` is nonzero only for
/// `k ≥ i`: the inner product `Σ_k L⁻¹_{ki} L⁻¹_{kj}` for `j ≤ i` starts at
/// `k = i`. Both shortcuts keep each entry's summation order identical to the
/// dense `k = 0..n` loop (the skipped terms are exact zeros), so the inverse is
/// bit-for-bit identical while doing roughly a sixth of the multiplies.
pub fn spd_inverse_from_chol(l: &[f64], n: usize) -> Vec<f64> {
    let li = invert_lower(l, n);
    let mut inv = vec![0.0f64; n * n];
    // (L⁻ᵀ L⁻¹)_{ij} = Σ_k L⁻¹_{ki} L⁻¹_{kj}, with L⁻¹ lower-triangular.
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += li[k * n + i] * li[k * n + j];
            }
            inv[i * n + j] = s;
            inv[j * n + i] = s;
        }
    }
    inv
}

/// Inverse of an SPD matrix (Cholesky then back-substitution). `None` if not PD.
pub fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    cholesky(a, n).map(|l| spd_inverse_from_chol(&l, n))
}

/// Cholesky that never fails. If `a` is not positive-definite even after the
/// caller's diagonal-dominance repair, escalate a ridge on the diagonal until it
/// factors (a degraded but finite fallback for the "should be impossible" path,
/// instead of panicking). In the happy path (already PD) the result is identical
/// to [`cholesky`]; only a genuinely non-PD input is perturbed.
pub fn cholesky_jitter(a: &[f64], n: usize) -> Vec<f64> {
    if let Some(l) = cholesky(a, n) {
        return l;
    }
    let mut m = a.to_vec();
    let scale = (0..n).map(|i| a[i * n + i].abs()).sum::<f64>() / (n.max(1) as f64);
    let mut jit = 1e-10 * (1.0 + scale);
    for _ in 0..16 {
        for i in 0..n {
            m[i * n + i] += jit;
        }
        if let Some(l) = cholesky(&m, n) {
            return l;
        }
        jit *= 10.0;
    }
    // Last resort: a diagonal (always-PD) surrogate from the clamped diagonal.
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        d[i * n + i] = a[i * n + i].abs().max(1e-8).sqrt();
    }
    d
}

/// SPD inverse that never fails (see [`cholesky_jitter`]). Identical to
/// [`spd_inverse`] when `a` is positive-definite; otherwise ridge-jittered rather
/// than `None`/panic.
pub fn spd_inverse_jitter(a: &[f64], n: usize) -> Vec<f64> {
    spd_inverse_from_chol(&cholesky_jitter(a, n), n)
}

/// Force a symmetric matrix to be positive definite by diagonal dominance
/// (STM's fallback when the Hessian is indefinite): if a diagonal entry is
/// smaller than the sum of the magnitudes of its off-diagonal entries, raise it.
pub fn make_diagonally_dominant(a: &mut [f64], n: usize) {
    let diag: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    for i in 0..n {
        let off: f64 = (0..n).filter(|&j| j != i).map(|j| a[i * n + j].abs()).sum();
        if diag[i] < off {
            a[i * n + i] = off;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip() {
        // SPD matrix.
        let a = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let inv = spd_inverse(&a, 3).unwrap();
        // A · A⁻¹ ≈ I
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[i * 3 + k] * inv[k * 3 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-9, "({},{})={}", i, j, s);
            }
        }
    }

    #[test]
    fn non_pd_returns_none() {
        let a = vec![1.0, 2.0, 2.0, 1.0]; // indefinite
        assert!(spd_inverse(&a, 2).is_none());
    }
}
