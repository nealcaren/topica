//! Deterministic, BLAS-free dense matrix multiply for the ProdLDA-family VAE
//! decoder (issue #378).
//!
//! The dense-decoder VAE backbone shared by `ProdLDA`, `CombinedTM`,
//! `ZeroShotTM`, `InfoCTM`, and `Scholar` spends most of a fit
//! iteration in three dense `O(N·K·V)` decoder terms — a forward `theta·beta`, a
//! backward `dlogit·betaᵀ`, and the cross-document `g.beta += theta_doᵀ·dlogit`
//! reduction. Written as scalar triple-loops they were 4.7–6.8× slower than a CPU
//! PyTorch reference at scale. These thin wrappers route them through
//! [`matrixmultiply`] (pure-Rust, cache-blocked + SIMD, BLAS-free), which is 2.7–6.8×
//! faster single-threaded on those shapes and scales across cores with its
//! `threading` feature. (`ETM(inference="vae")` is *not* in this set: it has its
//! own sparse per-document decoder over each document's actual words, not the dense
//! `theta·beta`, so it is untouched here — a candidate follow-up.)
//!
//! **Determinism.** topica's VAE family must fit bit-for-bit identically regardless
//! of thread count. `matrixmultiply` parallelizes over *output blocks*; every output
//! element's `k`-dimension reduction is still summed by one thread in a fixed order,
//! so the result is bit-identical no matter how many threads run (verified across
//! `MATMUL_NUM_THREADS = 1, 2, 4`). This is the property a rayon-over-batch reduction
//! of `g.beta` could not offer without per-thread accumulation (which would reorder
//! the sum). Switching from the scalar triple-loop *does* change the exact bits vs
//! the old code — a blocked GEMM sums in a different order — but only at the level of
//! floating-point rounding (~1e-15 relative), which is why the reference-parity and
//! FD-gradient tests still pass.
//!
//! All matrices are row-major and contiguous. The transposed operands are expressed
//! by swapping `matrixmultiply`'s row/column strides, so no operand is ever copied
//! or transposed in memory.

use matrixmultiply::dgemm;

/// `c (m×n) = a (m×k) · b (k×n)`, all row-major and contiguous. Overwrites `c`.
///
/// Panics (in debug) if the slice lengths do not match the given shape.
pub fn matmul_nn(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // Empty contraction: C is the sum over nothing = 0. `matrixmultiply` may
        // skip writing C when k == 0, so zero it here to honor the "overwrites c"
        // contract unconditionally (the decoder's tn reduction hits this on an
        // empty batch, where C's buffer happens to be pre-zeroed anyway).
        c.fill(0.0);
        return;
    }
    // A: row-major m×k -> rsa=k, csa=1. B: row-major k×n -> rsb=n, csb=1.
    // C: row-major m×n -> rsc=n, csc=1. beta=0 overwrites C.
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// `c (m×n) = a (m×k) · bᵀ`, where `b` is stored row-major as `n×k`. Overwrites `c`.
///
/// Used for the decoder backward `dtheta_do (N×K) = dlogit_raw (N×V) · betaᵀ`, with
/// `b = beta` stored `K×V` (so here `m=N`, `k=V`, `n=K`).
pub fn matmul_nt(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // Empty contraction: C is the sum over nothing = 0. `matrixmultiply` may
        // skip writing C when k == 0, so zero it here to honor the "overwrites c"
        // contract unconditionally (the decoder's tn reduction hits this on an
        // empty batch, where C's buffer happens to be pre-zeroed anyway).
        c.fill(0.0);
        return;
    }
    // A: row-major m×k -> rsa=k, csa=1. bᵀ is k×n from a row-major n×k `b`:
    // view it with rsb=1, csb=k (no copy). C: row-major m×n.
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// `c (m×n) = aᵀ · b`, where `a` is stored row-major as `k×m` and `b` as `k×n`.
/// Overwrites `c`.
///
/// Used for the decoder backward reduction `g.beta (K×V) = theta_doᵀ (K×N) ·
/// dlogit_raw (N×V)`, with `a = theta_do` stored `N×K` (so here `m=K`, `k=N`,
/// `n=V`).
pub fn matmul_tn(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    debug_assert_eq!(a.len(), k * m);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // Empty contraction: C is the sum over nothing = 0. `matrixmultiply` may
        // skip writing C when k == 0, so zero it here to honor the "overwrites c"
        // contract unconditionally (the decoder's tn reduction hits this on an
        // empty batch, where C's buffer happens to be pre-zeroed anyway).
        c.fill(0.0);
        return;
    }
    // aᵀ is m×k from a row-major k×m `a`: view it with rsa=1, csa=m (no copy).
    // B: row-major k×n -> rsb=n, csb=1. C: row-major m×n.
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Naive scalar references, matching the exact loop shape they replace.
    fn nn_ref(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for t in 0..k {
                let av = a[i * k + t];
                for j in 0..n {
                    c[i * n + j] += av * b[t * n + j];
                }
            }
        }
        c
    }
    fn nt_ref(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        // c[i][t] = sum_j a[i][j] * b[t][j], b is n×k
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for t in 0..n {
                let mut s = 0.0;
                for j in 0..k {
                    s += a[i * k + j] * b[t * k + j];
                }
                c[i * n + t] = s;
            }
        }
        c
    }
    fn tn_ref(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        // c[t][j] = sum_i a[i][t] * b[i][j], a is k×m, b is k×n
        let mut c = vec![0.0; m * n];
        for i in 0..k {
            for t in 0..m {
                let av = a[i * m + t];
                for j in 0..n {
                    c[t * n + j] += av * b[i * n + j];
                }
            }
        }
        c
    }

    fn seq(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic pseudo-random fill in [-1, 1).
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_rel(a: &[f64], b: &[f64]) -> f64 {
        let scale = a.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
            / scale
    }

    #[test]
    fn matmul_nn_matches_scalar() {
        let (m, k, n) = (17, 23, 13);
        let a = seq(m * k, 1);
        let b = seq(k * n, 2);
        let mut c = vec![0.0; m * n];
        matmul_nn(m, k, n, &a, &b, &mut c);
        assert!(max_rel(&c, &nn_ref(m, k, n, &a, &b)) < 1e-12);
    }

    #[test]
    fn matmul_nt_matches_scalar() {
        let (m, k, n) = (17, 23, 13);
        let a = seq(m * k, 3);
        let b = seq(n * k, 4); // n×k
        let mut c = vec![0.0; m * n];
        matmul_nt(m, k, n, &a, &b, &mut c);
        assert!(max_rel(&c, &nt_ref(m, k, n, &a, &b)) < 1e-12);
    }

    #[test]
    fn matmul_tn_matches_scalar() {
        let (m, k, n) = (17, 23, 13);
        let a = seq(k * m, 5); // k×m
        let b = seq(k * n, 6); // k×n
        let mut c = vec![0.0; m * n];
        matmul_tn(m, k, n, &a, &b, &mut c);
        assert!(max_rel(&c, &tn_ref(m, k, n, &a, &b)) < 1e-12);
    }

    #[test]
    fn zero_dims_are_noops() {
        let mut c: Vec<f64> = Vec::new();
        matmul_nn(0, 5, 0, &[], &[], &mut c);
        matmul_nt(0, 5, 0, &[], &[], &mut c);
        matmul_tn(0, 5, 0, &[], &[], &mut c);
        assert!(c.is_empty());
    }

    #[test]
    fn zero_contraction_zeroes_c() {
        // k == 0 with m,n > 0 (e.g. the tn reduction on an empty batch): the result
        // is the empty sum = 0, and the "overwrites c" contract must hold even
        // though C's buffer starts non-zero.
        let (m, n) = (3, 4);
        for f in [matmul_nn, matmul_nt, matmul_tn] {
            let mut c = vec![7.0; m * n];
            f(m, 0, n, &[], &[], &mut c);
            assert!(c.iter().all(|&x| x == 0.0), "k==0 must zero C");
        }
    }
}
