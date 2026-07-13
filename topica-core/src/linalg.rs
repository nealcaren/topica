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

/// Reduced QR decomposition of a matrix `A` (rows x cols) stored in row-major format.
/// Returns `(Q, R)` where `Q` is rows x cols (row-major) and `R` is cols x cols (row-major).
pub fn qr_reduced(a: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = a.to_vec();
    let mut q = vec![0.0; rows * cols];
    let mut r = vec![0.0; cols * cols];

    for j in 0..cols {
        // Compute R_jj = norm of column j of V
        let mut r_jj2 = 0.0;
        for r_idx in 0..rows {
            let val = v[r_idx * cols + j];
            r_jj2 += val * val;
        }
        let r_jj = r_jj2.sqrt();
        r[j * cols + j] = r_jj;

        if r_jj > 1e-12 {
            for r_idx in 0..rows {
                q[r_idx * cols + j] = v[r_idx * cols + j] / r_jj;
            }
        }

        for k in (j + 1)..cols {
            // Compute R_jk = Q_j^T * V_k
            let mut r_jk = 0.0;
            for r_idx in 0..rows {
                r_jk += q[r_idx * cols + j] * v[r_idx * cols + k];
            }
            r[j * cols + k] = r_jk;

            // V_k = V_k - R_jk * Q_j
            for r_idx in 0..rows {
                v[r_idx * cols + k] -= r_jk * q[r_idx * cols + j];
            }
        }
    }
    (q, r)
}

/// Compute the eigenvalues and eigenvectors of a symmetric matrix `A` of size `n x n`.
/// Returns `Some((eigenvalues, eigenvectors))` where eigenvectors are stored as columns
/// of the `n x n` row-major matrix, sorted in descending order of the eigenvalues.
pub fn jacobi_eigen(
    a: &[f64],
    n: usize,
    tol: f64,
    max_iter: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = m[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            // Converged! Sort eigenvalues and eigenvectors.
            let eigenvalues: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));

            let mut sorted_eigenvals = vec![0.0; n];
            let mut sorted_eigenvecs = vec![0.0; n * n];
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                sorted_eigenvals[new_idx] = eigenvalues[old_idx];
                for r in 0..n {
                    sorted_eigenvecs[r * n + new_idx] = v[r * n + old_idx];
                }
            }
            return Some((sorted_eigenvals, sorted_eigenvecs));
        }

        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau_rot = s / (1.0 + c);

        m[p * n + p] = app - t * apq;
        m[q * n + q] = aqq + t * apq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;

        for r in 0..n {
            if r != p && r != q {
                let arp = m[r * n + p];
                let arq = m[r * n + q];
                m[r * n + p] = arp - s * (arq + tau_rot * arp);
                m[r * n + q] = arq + s * (arp - tau_rot * arq);
                m[p * n + r] = m[r * n + p];
                m[q * n + r] = m[r * n + q];
            }
        }

        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = vrp - s * (vrq + tau_rot * vrp);
            v[r * n + q] = vrq + s * (vrp - tau_rot * vrq);
        }
    }
    None
}

/// Helper function to perform matrix multiplication: `C = A * B` where `A` is `ar x ac`
/// and `B` is `ac x bc`. All matrices are row-major.
fn matmul(a: &[f64], b: &[f64], ar: usize, ac: usize, bc: usize) -> Vec<f64> {
    let mut c = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik != 0.0 {
                for j in 0..bc {
                    c[i * bc + j] += aik * b[k * bc + j];
                }
            }
        }
    }
    c
}

/// Helper to transpose a matrix of size `rows x cols`.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Compute the randomized SVD of matrix `A` (rows x cols).
/// Returns `Some((singular_values, right_singular_vectors))` where the right singular
/// vectors are stored as columns of the `cols x k` row-major matrix.
pub fn randomized_svd(
    a: &[f64],
    rows: usize,
    cols: usize,
    k: usize,
    oversample: usize,
    power_iters: usize,
    seed: u64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let d = (k + oversample).min(rows).min(cols);
    if d == 0 {
        return None;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut omega = vec![0.0; cols * d];
    for i in (0..(cols * d)).step_by(2) {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let r = (-2.0 * u1.max(1e-15).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        omega[i] = r * theta.cos();
        if i + 1 < cols * d {
            omega[i + 1] = r * theta.sin();
        }
    }

    let at = transpose(a, rows, cols);
    let mut q = qr_reduced(&matmul(a, &omega, rows, cols, d), rows, d).0;

    for _ in 0..power_iters {
        let z = matmul(&at, &q, cols, rows, d);
        let q_z = qr_reduced(&z, cols, d).0;
        let y = matmul(a, &q_z, rows, cols, d);
        q = qr_reduced(&y, rows, d).0;
    }

    let qt = transpose(&q, rows, d);
    let b = matmul(&qt, a, d, rows, cols);
    let bt = transpose(&b, d, cols);
    let m = matmul(&b, &bt, d, cols, d);

    let (eig_vals, u_m) = jacobi_eigen(&m, d, 1e-15, 200)?;
    let mut s_vals = vec![0.0; d];
    for i in 0..d {
        s_vals[i] = eig_vals[i].max(0.0).sqrt();
    }

    let bt_um = matmul(&bt, &u_m, cols, d, d);
    let mut v_b = vec![0.0; cols * d];
    for i in 0..cols {
        for j in 0..d {
            if s_vals[j] > 1e-12 {
                v_b[i * d + j] = bt_um[i * d + j] / s_vals[j];
            } else {
                v_b[i * d + j] = 0.0;
            }
        }
    }

    let k_trunc = k.min(d);
    let mut truncated_s = vec![0.0; k_trunc];
    truncated_s.copy_from_slice(&s_vals[..k_trunc]);

    let mut truncated_v = vec![0.0; cols * k_trunc];
    for i in 0..cols {
        for j in 0..k_trunc {
            truncated_v[i * k_trunc + j] = v_b[i * d + j];
        }
    }

    Some((truncated_s, truncated_v))
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

    #[test]
    fn test_qr_reduced() {
        // 4 x 3 matrix
        let a = vec![
            12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0, -1.0, 1.0, 2.0,
        ];
        let (q, r) = qr_reduced(&a, 4, 3);

        // Check Q is orthogonal: Q^T * Q = I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for r_idx in 0..4 {
                    dot += q[r_idx * 3 + i] * q[r_idx * 3 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-9,
                    "Q orthogonality fails at ({}, {}): {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Check A = Q * R
        for i in 0..4 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[i * 3 + k] * r[k * 3 + j];
                }
                let idx = i * 3 + j;
                assert!(
                    (sum - a[idx]).abs() < 1e-9,
                    "QR reconstruction fails at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    a[idx],
                    sum
                );
            }
        }
    }

    #[test]
    fn test_jacobi_eigen() {
        // Symmetric 3x3 matrix
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 2.0];
        let (vals, vecs) = jacobi_eigen(&a, 3, 1e-15, 100).expect("eigenvalues");

        // Check V is orthogonal: V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for r_idx in 0..3 {
                    dot += vecs[r_idx * 3 + i] * vecs[r_idx * 3 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-9,
                    "V orthogonality fails at ({}, {}): {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Check A * V = V * diag(L)  => (A * V)_{ij} = V_{ij} * L_j
        let av = matmul(&a, &vecs, 3, 3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expect = vecs[i * 3 + j] * vals[j];
                assert!(
                    (av[i * 3 + j] - expect).abs() < 1e-9,
                    "A*V = V*L fails at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expect,
                    av[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn test_randomized_svd() {
        // Generate a low-rank matrix plus small noise
        // Rank 2 matrix of size 10 x 5
        let u_true = vec![
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 2.0, 0.0, -1.0, 2.0, 0.5, 0.5, -0.5, 1.5, 1.0,
            1.0, 0.0, 0.0,
        ]; // 10 x 2
        let s_true = vec![10.0, 5.0];
        let v_true = vec![1.0, 0.5, 0.0, 1.0, -1.0, 0.0, 2.0, -1.0, 0.5, 0.5]; // 5 x 2

        let mut a = vec![0.0; 10 * 5];
        for i in 0..10 {
            for j in 0..5 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u_true[i * 2 + k] * s_true[k] * v_true[j * 2 + k];
                }
                a[i * 5 + j] = sum;
            }
        }

        let (s, v) = randomized_svd(&a, 10, 5, 2, 2, 2, 42).expect("SVD");
        println!("s = {:?}", s);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 78.937).abs() < 1e-2);
        assert!((s[1] - 23.454).abs() < 1e-2);
        // Check right singular vectors are orthogonal
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for r in 0..5 {
                    dot += v[r * 2 + i] * v[r * 2 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9, "V orthogonality failed");
            }
        }
    }
}
