//! NMF, non-negative matrix factorization for topic modeling (Lee & Seung,
//! "Algorithms for Non-negative Matrix Factorization", NeurIPS 2001; Boutsidis &
//! Gallopoulos, "SVD based initialization", Pattern Recognition 2008).
//!
//! We factor the non-negative document-term matrix `X (D x V)` as `X ~ W H` with
//! `W (D x K) >= 0` and `H (K x V) >= 0`, by multiplicative updates that never
//! leave the non-negative orthant. Two divergences are available through
//! `beta_loss`: the squared Frobenius loss `0.5 ||X - WH||_F^2` (default) and the
//! generalized Kullback-Leibler divergence (the pLSA-equivalent on counts).
//!
//! The reference implementation we match is scikit-learn's
//! `sklearn.decomposition.NMF` (BSD-3-Clause). We read it to match the
//! initialization and update detail; we credit it here and reimplement the
//! numerics from the papers.
//!
//! Topics come from the factor matrices: `topic_word` is each row of `H`
//! normalized to sum 1, and `doc_topic` is each row of `W` normalized to sum 1.
//!
//! ```text
//!   X (D x V) ~ W (D x K) . H (K x V),   W, H >= 0
//!   Frobenius:  H *= (W^T X) / (W^T W H + eps);  W *= (X H^T) / (W H H^T + eps)
//!   KL:         H_kj *= [sum_i W_ik X_ij/(WH)_ij] / [sum_i W_ik]   (symmetric for W)
//! ```
//!
//! The initialization is NNDSVD (Boutsidis & Gallopoulos): a top-K truncated SVD
//! of `X`, with `W, H` built from the signed singular vectors. The "a" variant
//! fills exact zeros with `mean(X)` so the multiplicative updates are not pinned
//! at zero. The truncated SVD is a randomized SVD (Halko et al.) with a fixed
//! internal seed, so the init is deterministic and independent of the user seed.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Small additive constant guarding the multiplicative-update denominators.
const EPS: f64 = 1e-10;

/// Which divergence the updates minimize.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BetaLoss {
    Frobenius,
    KullbackLeibler,
}

/// Which initialization to use for `W` and `H`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Init {
    Nndsvd,
    Random,
}

// ---------------------------------------------------------------------------
// Dense row-major matrix helpers (local to this module). All matrices are
// stored row-major as `(rows, cols, data)`.
// ---------------------------------------------------------------------------

/// A dense row-major matrix.
///
/// `pub(crate)` so `src/lsa.rs` (LSA/LSI) can reuse the truncated-SVD output
/// without duplicating the linear-algebra primitives. The fields and methods are
/// crate-internal; the public Python surface never exposes `Mat` directly.
#[derive(Clone)]
pub(crate) struct Mat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

impl Mat {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    #[inline]
    pub(crate) fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub(crate) fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    #[inline]
    pub(crate) fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

// The three matmuls are rayon-parallelized over INDEPENDENT output rows: each
// output row is computed entirely by one task, into its own disjoint output
// slice, and each output cell keeps a FIXED inner-product summation order (the
// `p` loop accumulates left-to-right regardless of how rows are scheduled).
// Therefore results are bit-identical for any thread count (no parallel
// reduction whose combination order depends on thread completion).

/// `A . B` for row-major `A (m x k)` and `B (k x n)`.
///
/// ikj loop order: for each output row `i` (parallel), accumulate `B[k,:]` scaled
/// by `A[i,k]` over `k`. The inner `j`-loop is contiguous over both `orow` and
/// `brow`, which LLVM auto-vectorizes (FMA). Each `orow[j]` still accumulates over
/// `k` in increasing order, so the per-cell summation order is identical to a
/// naive ijk kernel and the result is bit-identical and thread-count-independent.
fn matmul(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(m, n);
    out.data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, orow)| {
            let arow = &a.data[i * k..(i + 1) * k];
            for p in 0..k {
                let aip = arow[p];
                if aip == 0.0 {
                    continue;
                }
                let brow = &b.data[p * n..(p + 1) * n];
                // Contiguous, dependence-free over j; LLVM vectorizes this FMA.
                for j in 0..n {
                    orow[j] += aip * brow[j];
                }
            }
        });
    out
}

/// `A^T . B` for row-major `A (m x k)` and `B (m x n)`, giving `(k x n)`. Output
/// row `p` is `sum_i A[i,p] * B[i,:]`, accumulated over `i` in fixed (increasing)
/// order.
///
/// ikj order with the output row fixed to a column of A: for each output row `p`
/// (= A-column `p`, parallel), accumulate `B[i,:]` scaled by `A[i,p]` over `i`.
/// `A[i,p]` is a strided scalar load (broadcast), but the inner `j`-loop stays
/// contiguous over `orow` and `brow` so it auto-vectorizes. `i`-order preserved,
/// so the result is bit-identical and thread-count-independent.
fn matmul_at(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.rows, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(k, n);
    out.data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(p, orow)| {
            for i in 0..m {
                let aip = a.data[i * k + p];
                if aip == 0.0 {
                    continue;
                }
                let brow = &b.data[i * n..(i + 1) * n];
                for j in 0..n {
                    orow[j] += aip * brow[j];
                }
            }
        });
    out
}

/// `A . B^T` for row-major `A (m x k)` and `B (n x k)`, giving `(m x n)`.
///
/// Each cell `(i,j)` is a dot product over contiguous rows `A[i,:]` and `B[j,:]`,
/// summed in increasing `k` into a local accumulator `s`, which lets LLVM emit
/// FMA / vectorize. `k`-order preserved (bit-identical, thread-count-independent).
fn matmul_bt(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.cols);
    let (m, k, n) = (a.rows, a.cols, b.rows);
    let mut out = Mat::zeros(m, n);
    out.data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, orow)| {
            let arow = &a.data[i * k..(i + 1) * k];
            for j in 0..n {
                let brow = &b.data[j * k..(j + 1) * k];
                let mut s = 0.0;
                for p in 0..k {
                    s += arow[p] * brow[p];
                }
                orow[j] = s;
            }
        });
    out
}

// ---------------------------------------------------------------------------
// Sparse X (CSR). X is the document-term matrix, which is very sparse, so every
// product against X iterates only its nonzeros. Per-row (col, val) lists in CSR
// layout; nonzeros within a row stay in ascending-column order so each output
// cell's summation order is fixed and thread-count-independent.
// ---------------------------------------------------------------------------

/// Compressed sparse row matrix `(rows x cols)`.
///
/// `pub(crate)` so `src/lsa.rs` can build the weighted document-term matrix
/// through the shared `count_matrix`/`tfidf_matrix` builders and feed it to the
/// shared `randomized_svd`. Crate-internal only.
pub(crate) struct SpMat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// Row `r`'s nonzeros are `cols[indptr[r]..indptr[r+1]]` / `vals[...]`.
    pub(crate) indptr: Vec<usize>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) vals: Vec<f64>,
}

impl SpMat {
    #[inline]
    pub(crate) fn row(&self, r: usize) -> (&[usize], &[f64]) {
        let s = self.indptr[r];
        let e = self.indptr[r + 1];
        (&self.col_idx[s..e], &self.vals[s..e])
    }
    /// `sum_{i,j} X_ij^2`.
    fn frob_sq(&self) -> f64 {
        self.vals.iter().map(|&v| v * v).sum()
    }
    /// `sum_{i,j} X_ij`.
    fn total(&self) -> f64 {
        self.vals.iter().sum()
    }
}

/// `X^T . B` for sparse `X (m x cols)` and dense `B (m x n)`, giving `(cols x n)`.
/// Output row `p` = `sum over rows i with X[i,p]!=0 of X[i,p] * B[i,:]`. We invert
/// the loop: scatter each nonzero into the corresponding output row. To keep
/// output rows independent (and the per-cell sum order fixed) we accumulate into
/// a per-output-row layout sequentially over rows `i` ascending.
fn sp_xt_b(x: &SpMat, b: &Mat) -> Mat {
    debug_assert_eq!(x.rows, b.rows);
    let n = b.cols;
    let mut out = Mat::zeros(x.cols, n);
    // Sequential over i ascending: each (i) contributes X[i,p]*B[i,:] to out row
    // p; summation order over i is fixed, so the result is deterministic.
    for i in 0..x.rows {
        let (cols, vals) = x.row(i);
        let brow = &b.data[i * n..(i + 1) * n];
        for (&p, &xv) in cols.iter().zip(vals.iter()) {
            let orow = &mut out.data[p * n..(p + 1) * n];
            for j in 0..n {
                orow[j] += xv * brow[j];
            }
        }
    }
    out
}

/// `X . B^T` for sparse `X (m x cols)` and dense `B (n x cols)`, giving `(m x n)`.
/// Parallel over independent output rows `i`; each cell `(i,j)` sums over the
/// nonzeros of `X` row `i` in ascending-column order (fixed).
fn sp_x_bt(x: &SpMat, b: &Mat) -> Mat {
    debug_assert_eq!(x.cols, b.cols);
    let n = b.rows;
    let k = b.cols;
    let mut out = Mat::zeros(x.rows, n);
    out.data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, orow)| {
            let (cols, vals) = x.row(i);
            for (j, ocell) in orow.iter_mut().enumerate() {
                let brow = &b.data[j * k..(j + 1) * k];
                let mut s = 0.0;
                for (&c, &xv) in cols.iter().zip(vals.iter()) {
                    s += xv * brow[c];
                }
                *ocell = s;
            }
        });
    out
}

/// `X . B` for sparse `X (m x cols)` and dense `B (cols x n)`, giving `(m x n)`.
/// Parallel over independent output rows `i`; cell `(i,j)` sums over the nonzeros
/// of `X` row `i` in ascending-column order (fixed).
fn sp_x_b(x: &SpMat, b: &Mat) -> Mat {
    debug_assert_eq!(x.cols, b.rows);
    let n = b.cols;
    let mut out = Mat::zeros(x.rows, n);
    out.data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, orow)| {
            let (cols, vals) = x.row(i);
            for (&c, &xv) in cols.iter().zip(vals.iter()) {
                let brow = &b.data[c * n..(c + 1) * n];
                for j in 0..n {
                    orow[j] += xv * brow[j];
                }
            }
        });
    out
}

/// `A^T . X` for dense `A (m x r)` and sparse `X (m x cols)`, giving `(r x cols)`.
/// Sequential over rows `i` ascending so each output cell's sum order is fixed.
fn sp_at_x(a: &Mat, x: &SpMat) -> Mat {
    debug_assert_eq!(a.rows, x.rows);
    let r = a.cols;
    let mut out = Mat::zeros(r, x.cols);
    let cols = x.cols;
    for i in 0..x.rows {
        let arow = &a.data[i * r..(i + 1) * r];
        let (xcols, xvals) = x.row(i);
        for p in 0..r {
            let aip = arow[p];
            if aip == 0.0 {
                continue;
            }
            let orow = &mut out.data[p * cols..(p + 1) * cols];
            for (&c, &xv) in xcols.iter().zip(xvals.iter()) {
                orow[c] += aip * xv;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Randomized truncated SVD (Halko et al.), used by NNDSVD. Fixed internal seed
// so the initialization is deterministic regardless of the user seed.
// ---------------------------------------------------------------------------

/// Seed for the randomized-SVD test matrix. Kept distinct from any user seed so
/// the NNDSVD initialization is identical on every run.
// The trailing `64` is the hex byte for 'd' in "nmf_svd", not a type suffix.
#[allow(clippy::mistyped_literal_suffixes)]
const SVD_SEED: u64 = 0x6e6d_665f_7376_64; // "nmf_svd"

/// A standard-normal draw via Box-Muller.
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Modified Gram-Schmidt QR; returns the orthonormal `Q (m x n)` (the `R` factor
/// is not needed). Columns that collapse to near-zero are replaced by zeros.
pub(crate) fn mgs_q(a: &Mat) -> Mat {
    let (m, n) = (a.rows, a.cols);
    // Work column-wise: columns[j] is column j of the running matrix.
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a.at(i, j)).collect::<Vec<f64>>())
        .collect();
    for j in 0..n {
        // Orthogonalize against the already-finalized columns.
        for p in 0..j {
            let dot: f64 = (0..m).map(|i| cols[p][i] * cols[j][i]).sum();
            for i in 0..m {
                cols[j][i] -= dot * cols[p][i];
            }
        }
        let norm: f64 = cols[j].iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..m {
                cols[j][i] /= norm;
            }
        } else {
            for i in 0..m {
                cols[j][i] = 0.0;
            }
        }
    }
    let mut q = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            q.set(i, j, cols[j][i]);
        }
    }
    q
}

/// Symmetric eigendecomposition of a small dense `(n x n)` matrix by the cyclic
/// Jacobi method. Returns `(eigenvalues, eigenvectors)` with eigenvectors as
/// columns of the returned matrix, sorted by descending eigenvalue.
pub(crate) fn jacobi_eigen(a_in: &Mat) -> (Vec<f64>, Mat) {
    let n = a_in.rows;
    let mut a = a_in.clone();
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v.set(i, i, 1.0);
    }
    for _sweep in 0..100 {
        // Off-diagonal magnitude.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a.at(p, q) * a.at(p, q);
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a.at(p, q);
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a.at(p, p);
                let aqq = a.at(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p and q.
                for k in 0..n {
                    let akp = a.at(k, p);
                    let akq = a.at(k, q);
                    a.set(k, p, c * akp - s * akq);
                    a.set(k, q, s * akp + c * akq);
                }
                for k in 0..n {
                    let apk = a.at(p, k);
                    let aqk = a.at(q, k);
                    a.set(p, k, c * apk - s * aqk);
                    a.set(q, k, s * apk + c * aqk);
                }
                // Accumulate the rotation into V.
                for k in 0..n {
                    let vkp = v.at(k, p);
                    let vkq = v.at(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }
    let mut eig: Vec<(f64, usize)> = (0..n).map(|i| (a.at(i, i), i)).collect();
    eig.sort_by(|x, y| y.0.total_cmp(&x.0));
    let vals: Vec<f64> = eig.iter().map(|&(e, _)| e).collect();
    let mut vecs = Mat::zeros(n, n);
    for (newc, &(_, oldc)) in eig.iter().enumerate() {
        for r in 0..n {
            vecs.set(r, newc, v.at(r, oldc));
        }
    }
    (vals, vecs)
}

/// Top-`k` truncated SVD of `X (d x v)` by randomized range finding. Returns
/// `(U (d x k), S (k), Vt (k x v))`. Deterministic (fixed internal seed).
///
/// NOTE: this builds `B B^T` and takes `sqrt(eig(B B^T))`, which SQUARES the
/// conditioning of the singular spectrum (small singular values lose precision).
/// That is acceptable here only because the SVD is scoped to seeding the NNDSVD
/// initialization, not to any reported factorization output.
fn randomized_svd(x: &SpMat, k: usize) -> (Mat, Vec<f64>, Mat) {
    randomized_svd_seeded(x, k, SVD_SEED)
}

/// Top-`k` truncated SVD of `X (d x v)` by randomized range finding, with the
/// sketch RNG seeded by `seed`. Same algorithm as [`randomized_svd`] (which fixes
/// the seed to `SVD_SEED` for NMF's NNDSVD init); LSA/LSI calls this with the
/// user seed so its truncated SVD is reproducible per seed. Returns
/// `(U (d x k), S (k), Vt (k x v))`.
pub(crate) fn randomized_svd_seeded(x: &SpMat, k: usize, seed: u64) -> (Mat, Vec<f64>, Mat) {
    let (d, v) = (x.rows, x.cols);
    let p = 10usize.min(v.saturating_sub(k));
    let r = (k + p).min(v).min(d);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Omega (v x r) Gaussian test matrix.
    let mut omega = Mat::zeros(v, r);
    for i in 0..v {
        for j in 0..r {
            omega.set(i, j, randn(&mut rng));
        }
    }

    // Y = X Omega, with power iterations Y <- X (X^T Y), re-orthonormalizing.
    let mut y = sp_x_b(x, &omega); // d x r
    let mut q = mgs_q(&y);
    for _ in 0..4 {
        let xtq = sp_xt_b(x, &q); // v x r
        y = sp_x_b(x, &xtq); // d x r
        q = mgs_q(&y);
    }

    // B = Q^T X  (r x v).
    let b = sp_at_x(&q, x);
    // Small SVD of B via the eigendecomposition of B B^T (r x r).
    let bbt = matmul_bt(&b, &b); // r x r
    let (eigvals, ub) = jacobi_eigen(&bbt); // ub columns are left singular vecs of B

    let kk = k.min(r);
    let mut s = vec![0.0; kk];
    let mut u = Mat::zeros(d, kk);
    let mut vt = Mat::zeros(kk, v);
    for c in 0..kk {
        let sigma = eigvals[c].max(0.0).sqrt();
        s[c] = sigma;
        // U[:, c] = Q . ub[:, c].
        for i in 0..d {
            let mut acc = 0.0;
            for t in 0..r {
                acc += q.at(i, t) * ub.at(t, c);
            }
            u.set(i, c, acc);
        }
        // Vt[c, :] = (1/sigma) ub[:, c]^T B.
        if sigma > 1e-12 {
            for j in 0..v {
                let mut acc = 0.0;
                for t in 0..r {
                    acc += ub.at(t, c) * b.at(t, j);
                }
                vt.set(c, j, acc / sigma);
            }
        }
    }
    (u, s, vt)
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// NNDSVD initialization with the "a" zero-fill (Boutsidis & Gallopoulos). Builds
/// `W (d x k)` and `H (k x v)` from the signed singular vectors, then fills exact
/// zeros with `mean(X)`.
fn nndsvd_init(x: &SpMat, k: usize) -> (Mat, Mat) {
    let (d, v) = (x.rows, x.cols);
    let (u, s, vt) = randomized_svd(x, k);

    let mut w = Mat::zeros(d, k);
    let mut h = Mat::zeros(k, v);

    // First component: nonneg by construction (leading singular triplet of a
    // nonnegative matrix has a nonnegative dominant sign).
    let sqrt_s0 = s[0].max(0.0).sqrt();
    for i in 0..d {
        w.set(i, 0, u.at(i, 0).abs() * sqrt_s0);
    }
    for j in 0..v {
        h.set(0, j, vt.at(0, j).abs() * sqrt_s0);
    }

    // Remaining components: split each singular vector into its positive and
    // negative parts and keep whichever pairing carries more energy.
    //
    // LOAD-BEARING sign coupling: `u` (cols of U) and `vt` (rows of Vt) are both
    // derived from the SAME left singular vectors of B (`ub`), so a global sign
    // flip of a singular triplet flips U[:,c] and Vt[c,:] TOGETHER. NNDSVD's
    // positive/negative split therefore picks a consistent (W,H) pairing
    // regardless of the arbitrary SVD sign. Do not derive U and Vt from
    // independent eigenproblems, or the signs would decouple and break this.
    for c in 1..k {
        let mut up = vec![0.0; d];
        let mut un = vec![0.0; d];
        for i in 0..d {
            let x = u.at(i, c);
            if x > 0.0 {
                up[i] = x;
            } else {
                un[i] = -x;
            }
        }
        let mut vp = vec![0.0; v];
        let mut vn = vec![0.0; v];
        for j in 0..v {
            let x = vt.at(c, j);
            if x > 0.0 {
                vp[j] = x;
            } else {
                vn[j] = -x;
            }
        }
        let up_n = norm(&up);
        let un_n = norm(&un);
        let vp_n = norm(&vp);
        let vn_n = norm(&vn);
        let term_p = up_n * vp_n;
        let term_n = un_n * vn_n;
        let (u_part, v_part, u_norm, v_norm, mu) = if term_p >= term_n {
            (up, vp, up_n, vp_n, term_p)
        } else {
            (un, vn, un_n, vn_n, term_n)
        };
        let lbd = (s[c].max(0.0) * mu).sqrt();
        if u_norm > 1e-12 && v_norm > 1e-12 {
            for i in 0..d {
                w.set(i, c, lbd * u_part[i] / u_norm);
            }
            for j in 0..v {
                h.set(c, j, lbd * v_part[j] / v_norm);
            }
        }
    }

    // "a" variant: fill exact zeros with mean(X).
    let mean = x.total() / (d * v).max(1) as f64;
    for x in w.data.iter_mut() {
        if *x == 0.0 {
            *x = mean;
        }
    }
    for x in h.data.iter_mut() {
        if *x == 0.0 {
            *x = mean;
        }
    }
    (w, h)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Random nonnegative initialization matching sklearn's `init="random"`
/// heuristic. Each factor entry is a half-normal draw `|randn| * sqrt(mean(X)/K)`,
/// seeded by the user `seed` via the caller's ChaCha8 RNG.
///
/// Magnitude target: with `E[|randn|] = sqrt(2/pi)` and `E[randn^2] = 1`, each
/// W,H entry has mean `sqrt(2/pi) * sqrt(mean/K)` and second moment `mean/K`, so
/// `E[(WH)_ij] = sum_k E[W_ik] E[H_kj] = K * (2/pi) * (mean/K) = (2/pi)*mean`.
/// That is the same `O(mean)` scaling sklearn uses (it does not aim for exactly
/// `mean`; the multiplicative updates rescale within the first few iterations).
fn random_init<R: Rng>(x: &SpMat, k: usize, rng: &mut R) -> (Mat, Mat) {
    let (d, v) = (x.rows, x.cols);
    let mean = x.total() / (d * v).max(1) as f64;
    let scale = (mean / k as f64).max(EPS).sqrt();
    let mut w = Mat::zeros(d, k);
    let mut h = Mat::zeros(k, v);
    for x in w.data.iter_mut() {
        *x = randn(rng).abs() * scale;
    }
    for x in h.data.iter_mut() {
        *x = randn(rng).abs() * scale;
    }
    (w, h)
}

// ---------------------------------------------------------------------------
// Reconstruction error
// ---------------------------------------------------------------------------

/// `(W H)_ij = sum_k W[i,k] H[k,j]` for column `j` only (k-length inner product),
/// summed in fixed `k` order. `h` is `K x V` row-major, so `H[k,j]` strides by V.
#[inline]
fn wh_cell(wrow: &[f64], h: &Mat, j: usize, k: usize) -> f64 {
    let mut s = 0.0;
    for p in 0..k {
        s += wrow[p] * h.data[p * h.cols + j];
    }
    s
}

/// Frobenius reconstruction error `0.5 ||X - WH||_F^2` WITHOUT forming dense WH:
/// `0.5(||X||_F^2 - 2 <X,WH>_nnz + tr((W^T W)(H H^T)))`. The cross term touches
/// only nnz(X); the `tr` term is `O(d k + k^2 + k v)` via the small Gram matrices.
fn frobenius_error(x: &SpMat, w: &Mat, h: &Mat) -> f64 {
    let k = w.cols;
    let v = h.cols;
    // Cross term <X, WH> over nonzeros, parallel over independent rows i. We
    // collect per-row partials into an indexed Vec and sum sequentially, so the
    // total is independent of thread completion order (deterministic).
    let mut partials = vec![0.0; x.rows];
    partials.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let wrow = &w.data[i * k..(i + 1) * k];
        let (cols, vals) = x.row(i);
        let mut s = 0.0;
        for (&j, &xv) in cols.iter().zip(vals.iter()) {
            s += xv * wh_cell(wrow, h, j, k);
        }
        *slot = s;
    });
    let cross: f64 = partials.iter().sum();
    // tr((W^T W)(H H^T)) = sum_{a,b} (W^T W)_ab (H H^T)_ab.
    let wtw = matmul_at(w, w); // k x k
    let hht = matmul_bt(h, h); // k x k
    let mut tr = 0.0;
    for a in 0..k {
        for b in 0..k {
            tr += wtw.data[a * k + b] * hht.data[a * k + b];
        }
    }
    let _ = v;
    0.5 * (x.frob_sq() - 2.0 * cross + tr)
}

/// Generalized KL `sum_ij [X ln(X/WH) - X + WH]` WITHOUT forming dense WH. The
/// `X ln(X/WH)` term is nonzero only at nnz(X) (compute WH there); `sum WH =
/// sum_k (sum_i W_ik)(sum_j H_kj)`; `-sum X` is a scalar.
fn kl_error(x: &SpMat, w: &Mat, h: &Mat) -> f64 {
    let k = w.cols;
    // sum over nnz of X ln(X/WH), parallel over independent rows. Per-row
    // partials are summed sequentially so the total is thread-count-independent.
    let mut partials = vec![0.0; x.rows];
    partials.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let wrow = &w.data[i * k..(i + 1) * k];
        let (cols, vals) = x.row(i);
        let mut s = 0.0;
        for (&j, &xv) in cols.iter().zip(vals.iter()) {
            if xv > 0.0 {
                let whij = wh_cell(wrow, h, j, k).max(EPS);
                s += xv * (xv / whij).ln();
            }
        }
        *slot = s;
    });
    let nnz_term: f64 = partials.iter().sum();
    // sum WH = sum_k (sum_i W_ik) (sum_j H_kj).
    let mut sum_wh = 0.0;
    for c in 0..k {
        let mut wsum = 0.0;
        for i in 0..w.rows {
            wsum += w.data[i * k + c];
        }
        let mut hsum = 0.0;
        for j in 0..h.cols {
            hsum += h.data[c * h.cols + j];
        }
        sum_wh += wsum * hsum;
    }
    nnz_term - x.total() + sum_wh
}

// ---------------------------------------------------------------------------
// Multiplicative updates
// ---------------------------------------------------------------------------

/// One Frobenius multiplicative update of `H` then `W`. X is sparse: `W^T X` and
/// `X H^T` iterate only nonzeros; the small `k`-dim Grams and their dense
/// products stay dense (and parallelized).
fn mu_frobenius(x: &SpMat, w: &mut Mat, h: &mut Mat) {
    // H *= (W^T X) / (W^T W H + eps).
    let wtx = sp_at_x(w, x); // k x v   (sparse-X product)
    let wtw = matmul_at(w, w); // k x k
    let wtwh = matmul(&wtw, h); // k x v
    for i in 0..h.data.len() {
        h.data[i] *= wtx.data[i] / (wtwh.data[i] + EPS);
    }
    // W *= (X H^T) / (W H H^T + eps).
    let xht = sp_x_bt(x, h); // d x k   (sparse-X product)
    let hht = matmul_bt(h, h); // k x k
    let whht = matmul(w, &hht); // d x k
    for i in 0..w.data.len() {
        w.data[i] *= xht.data[i] / (whht.data[i] + EPS);
    }
}

/// Sparse ratio `R = X (./) WH` with the SAME sparsity as `X`: `R_ij = X_ij /
/// ((WH)_ij + eps)` only at nnz(X) (zero elsewhere, since `X_ij = 0` there).
/// Returns a CSR matrix sharing X's structure. Parallel over independent rows;
/// each `(WH)_ij` keeps a fixed inner-product order over `k`.
fn sparse_ratio(x: &SpMat, w: &Mat, h: &Mat) -> SpMat {
    let k = w.cols;
    // Parallel over independent rows: each row owns a disjoint slice of `vals`.
    let mut out_vals = vec![0.0; x.vals.len()];
    let chunks: Vec<&mut [f64]> = {
        // Build per-row mutable slices following X's indptr layout.
        let mut rest = out_vals.as_mut_slice();
        let mut v = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let len = x.indptr[i + 1] - x.indptr[i];
            let (head, tail) = rest.split_at_mut(len);
            v.push(head);
            rest = tail;
        }
        v
    };
    chunks.into_par_iter().enumerate().for_each(|(i, slot)| {
        let wrow = &w.data[i * k..(i + 1) * k];
        let (cols, vals) = x.row(i);
        for (out, (&j, &xv)) in slot.iter_mut().zip(cols.iter().zip(vals.iter())) {
            let whij = wh_cell(wrow, h, j, k) + EPS;
            *out = xv / whij;
        }
    });
    SpMat {
        rows: x.rows,
        cols: x.cols,
        indptr: x.indptr.clone(),
        col_idx: x.col_idx.clone(),
        vals: out_vals,
    }
}

/// One KL multiplicative update of `H` then `W`. The ratio `X (./) WH` is zero
/// wherever `X` is zero, so it is formed sparsely (WH evaluated only at nnz(X));
/// numerators are sparse-X products, denominators are column-sums of W / row-sums
/// of H (`O(dk)` / `O(kv)`).
fn mu_kl(x: &SpMat, w: &mut Mat, h: &mut Mat) {
    let (d, v, k) = (x.rows, x.cols, w.cols);
    // H_kj *= [sum_i W_ik (X_ij / (WH)_ij)] / [sum_i W_ik].
    let ratio = sparse_ratio(x, w, h);
    let numer = sp_at_x(w, &ratio); // k x v
    let mut wsum = vec![0.0; k]; // sum_i W_ik
    for i in 0..d {
        for c in 0..k {
            wsum[c] += w.at(i, c);
        }
    }
    for c in 0..k {
        let denom = wsum[c] + EPS;
        for j in 0..v {
            let idx = c * v + j;
            h.data[idx] *= numer.data[idx] / denom;
        }
    }
    // W_ik *= [sum_j H_kj (X_ij / (WH)_ij)] / [sum_j H_kj].
    let ratio = sparse_ratio(x, w, h);
    let numer = sp_x_bt(&ratio, h); // d x k
    let mut hsum = vec![0.0; k]; // sum_j H_kj
    for c in 0..k {
        let mut s = 0.0;
        for j in 0..v {
            s += h.at(c, j);
        }
        hsum[c] = s;
    }
    for i in 0..d {
        for c in 0..k {
            let idx = i * k + c;
            w.data[idx] *= numer.data[idx] / (hsum[c] + EPS);
        }
    }
}

// ---------------------------------------------------------------------------
// Fitted model
// ---------------------------------------------------------------------------

/// A fitted NMF model. `topic_word` is each row of `H` normalized to sum 1;
/// `doc_topic` is each row of `W` normalized to sum 1.
pub struct NmfModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub topic_word: Vec<Vec<f64>>,
    pub doc_topic: Vec<Vec<f64>>,
    /// Raw factor `H (K x V)`, before row normalization.
    pub h: Vec<Vec<f64>>,
    /// Raw factor `W (D x K)`, before row normalization.
    pub w: Vec<Vec<f64>>,
    pub reconstruction_error: f64,
    pub error_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
}

impl NmfModel {
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word.clone()
    }
}

/// Normalize each row to sum 1; an all-zero row becomes uniform.
fn normalize_rows(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|r| {
            let row = m.row(r);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                row.iter().map(|&x| x / s).collect()
            } else {
                vec![1.0 / m.cols as f64; m.cols]
            }
        })
        .collect()
}

/// Build the sparse document-term count matrix `X (D x V)` (CSR) from token-id
/// documents. Each row's nonzeros are stored in ascending-column order.
pub(crate) fn count_matrix(docs: &[Vec<u32>], num_types: usize) -> SpMat {
    let d = docs.len();
    let mut indptr = Vec::with_capacity(d + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    indptr.push(0);
    // Per-row dense accumulator (reused), scanned in ascending column order so
    // CSR nonzeros come out sorted.
    let mut counts = vec![0.0f64; num_types];
    let mut touched: Vec<usize> = Vec::new();
    for doc in docs {
        for &w in doc {
            let w = w as usize;
            if w < num_types {
                if counts[w] == 0.0 {
                    touched.push(w);
                }
                counts[w] += 1.0;
            }
        }
        touched.sort_unstable();
        for &c in &touched {
            col_idx.push(c);
            vals.push(counts[c]);
            counts[c] = 0.0;
        }
        touched.clear();
        indptr.push(col_idx.len());
    }
    SpMat {
        rows: d,
        cols: num_types,
        indptr,
        col_idx,
        vals,
    }
}

/// Build the sparse TF-IDF document-term matrix (CSR): `tf * (ln((1+D)/(1+df))
/// + 1)`, then L2 normalize each document row. topica's own formula; no
/// dependence on any external transformer. The IDF reweighting and L2 norm only
/// rescale existing nonzeros, so the sparsity pattern is unchanged.
pub(crate) fn tfidf_matrix(docs: &[Vec<u32>], num_types: usize) -> SpMat {
    let d = docs.len();
    let mut x = count_matrix(docs, num_types);
    // Document frequency per term (count of rows with a nonzero in that column).
    let mut df = vec![0usize; num_types];
    for &c in &x.col_idx {
        df[c] += 1;
    }
    let idf: Vec<f64> = (0..num_types)
        .map(|c| ((1.0 + d as f64) / (1.0 + df[c] as f64)).ln() + 1.0)
        .collect();
    for r in 0..d {
        let s = x.indptr[r];
        let e = x.indptr[r + 1];
        // Apply IDF, then L2-normalize the row over its nonzeros.
        let mut sumsq = 0.0;
        for nz in s..e {
            let v = x.vals[nz] * idf[x.col_idx[nz]];
            x.vals[nz] = v;
            sumsq += v * v;
        }
        let n = sumsq.sqrt();
        if n > 0.0 {
            for nz in s..e {
                x.vals[nz] /= n;
            }
        }
    }
    x
}

/// Fit NMF by multiplicative updates. `weighting_tfidf` selects the input matrix;
/// `beta_loss` selects the divergence; `init` selects the factor initialization;
/// `iters` is the maximum iteration count; `convergence_tol` stops early on the
/// relative reconstruction-error decrease; `seed` seeds `init = Random` only.
///
/// NOTE on `convergence_tol`: this is a PER-ITERATION relative-decrease test
/// (`|prev - err| / |prev| < tol` stops). sklearn instead checks the CUMULATIVE
/// relative improvement since iteration 0, evaluated only every 10 iterations.
/// The two stopping rules differ, so `converged` and `iters_run` here are NOT
/// directly comparable to sklearn's `n_iter_`.
#[allow(clippy::too_many_arguments)]
pub fn fit_nmf(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    beta_loss: BetaLoss,
    init: Init,
    weighting_tfidf: bool,
    iters: usize,
    convergence_tol: f64,
    seed: u64,
) -> NmfModel {
    let k = num_topics;
    let x = if weighting_tfidf {
        tfidf_matrix(docs, num_types)
    } else {
        count_matrix(docs, num_types)
    };

    let (mut w, mut h) = match init {
        Init::Nndsvd => nndsvd_init(&x, k),
        Init::Random => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            random_init(&x, k, &mut rng)
        }
    };

    let err_fn = |w: &Mat, h: &Mat| match beta_loss {
        BetaLoss::Frobenius => frobenius_error(&x, w, h),
        BetaLoss::KullbackLeibler => kl_error(&x, w, h),
    };

    let mut error_history: Vec<f64> = Vec::with_capacity(iters + 1);
    let mut prev = err_fn(&w, &h);
    error_history.push(prev);
    let mut converged = false;
    let mut iters_run = 0usize;

    for it in 0..iters {
        iters_run = it + 1;
        match beta_loss {
            BetaLoss::Frobenius => mu_frobenius(&x, &mut w, &mut h),
            BetaLoss::KullbackLeibler => mu_kl(&x, &mut w, &mut h),
        }
        let err = err_fn(&w, &h);
        error_history.push(err);
        let rel = (prev - err).abs() / (prev.abs() + 1e-12);
        prev = err;
        if convergence_tol > 0.0 && rel < convergence_tol {
            converged = true;
            break;
        }
    }

    let topic_word = normalize_rows(&h);
    let doc_topic = normalize_rows(&w);
    let h_rows: Vec<Vec<f64>> = (0..h.rows).map(|r| h.row(r).to_vec()).collect();
    let w_rows: Vec<Vec<f64>> = (0..w.rows).map(|r| w.row(r).to_vec()).collect();

    NmfModel {
        num_topics: k,
        num_types,
        topic_word,
        doc_topic,
        h: h_rows,
        w: w_rows,
        reconstruction_error: prev,
        error_history,
        converged,
        iters_run,
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for NmfModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word.clone()
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.error_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// A planted-block corpus: K well-separated word blocks, each document drawn
    /// from a single block.
    fn planted(
        k: usize,
        block: usize,
        ndocs: usize,
        dlen: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, usize) {
        let v = k * block;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let docs: Vec<Vec<u32>> = (0..ndocs)
            .map(|d| {
                let b = d % k;
                (0..dlen)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        (docs, v)
    }

    #[test]
    fn fit_recovers_planted_blocks() {
        let (k, block) = (3usize, 8usize);
        let (docs, v) = planted(k, block, 180, 15, 1);
        let m = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Nndsvd,
            false,
            200,
            1e-4,
            42,
        );
        assert_eq!(m.num_topics, k);
        let tw = &m.topic_word;
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| tw[t][b].total_cmp(&tw[t][a]));
            let blocks: std::collections::HashSet<usize> =
                ord[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic {t} top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k, "topics did not cover all blocks");
    }

    #[test]
    fn determinism_same_seed() {
        let (k, block) = (3usize, 6usize);
        let (docs, v) = planted(k, block, 90, 12, 7);
        // NNDSVD path (seed-independent init).
        let a = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Nndsvd,
            false,
            60,
            0.0,
            42,
        );
        let b = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Nndsvd,
            false,
            60,
            0.0,
            42,
        );
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        // Random path (seeded by user seed).
        let c = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Random,
            false,
            60,
            0.0,
            99,
        );
        let d = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Random,
            false,
            60,
            0.0,
            99,
        );
        assert_eq!(c.topic_word, d.topic_word);
        assert_eq!(c.doc_topic, d.doc_topic);
    }

    #[test]
    fn row_sum_invariants() {
        let (k, block) = (4usize, 5usize);
        let (docs, v) = planted(k, block, 120, 10, 3);
        for &loss in &[BetaLoss::Frobenius, BetaLoss::KullbackLeibler] {
            let m = fit_nmf(&docs, k, v, loss, Init::Nndsvd, false, 80, 0.0, 42);
            for row in &m.topic_word {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
            for row in &m.doc_topic {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn svd_recovers_singular_values() {
        // A small matrix with a known dominant structure. Build X = a b^T + small,
        // whose leading singular value is ||a|| ||b||.
        let d = 6usize;
        let v = 5usize;
        let a = [3.0, 1.0, 2.0, 0.5, 1.5, 2.5];
        let b = [2.0, 1.0, 0.5, 1.5, 1.0];
        let mut indptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        for i in 0..d {
            for j in 0..v {
                col_idx.push(j);
                vals.push(a[i] * b[j]);
            }
            indptr.push(col_idx.len());
        }
        let x = SpMat {
            rows: d,
            cols: v,
            indptr,
            col_idx,
            vals,
        };
        let (_, s, _) = randomized_svd(&x, 2);
        let want = norm(&a) * norm(&b);
        assert!(
            (s[0] - want).abs() / want < 1e-6,
            "sigma0 {} vs {}",
            s[0],
            want
        );
        // The second singular value of a rank-1 matrix is ~0.
        assert!(s[1] < 1e-6, "sigma1 {} should be near zero", s[1]);
    }

    #[test]
    fn thread_count_independent() {
        // Same seed at 1 thread vs many threads must give bit-identical factors,
        // proving the parallel matmuls do not depend on thread completion order.
        let (k, block) = (4usize, 7usize);
        let (docs, v) = planted(k, block, 200, 18, 11);
        let fit = |loss, init| fit_nmf(&docs, k, v, loss, init, false, 120, 0.0, 77);
        for &loss in &[BetaLoss::Frobenius, BetaLoss::KullbackLeibler] {
            for &init in &[Init::Nndsvd, Init::Random] {
                let one = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(|| fit(loss, init));
                let many = rayon::ThreadPoolBuilder::new()
                    .num_threads(8)
                    .build()
                    .unwrap()
                    .install(|| fit(loss, init));
                assert_eq!(
                    one.topic_word, many.topic_word,
                    "topic_word differs by thread count (loss={loss:?}, init={init:?})"
                );
                assert_eq!(
                    one.doc_topic, many.doc_topic,
                    "doc_topic differs by thread count (loss={loss:?}, init={init:?})"
                );
            }
        }
    }

    #[test]
    fn nmf_conforms() {
        let (k, block) = (3usize, 8usize);
        let (docs, v) = planted(k, block, 180, 15, 1);
        let m = fit_nmf(
            &docs,
            k,
            v,
            BetaLoss::Frobenius,
            Init::Nndsvd,
            false,
            150,
            1e-4,
            42,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
