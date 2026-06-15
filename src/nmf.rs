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
#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    #[inline]
    fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    #[inline]
    fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// `A . B` for row-major `A (m x k)` and `B (k x n)`.
fn matmul(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(m, n);
    for i in 0..m {
        for p in 0..k {
            let aip = a.at(i, p);
            if aip == 0.0 {
                continue;
            }
            let brow = &b.data[p * n..(p + 1) * n];
            let orow = &mut out.data[i * n..(i + 1) * n];
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
    out
}

/// `A^T . B` for row-major `A (m x k)` and `B (m x n)`, giving `(k x n)`.
fn matmul_at(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.rows, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(k, n);
    for i in 0..m {
        let arow = &a.data[i * k..(i + 1) * k];
        let brow = &b.data[i * n..(i + 1) * n];
        for p in 0..k {
            let aip = arow[p];
            if aip == 0.0 {
                continue;
            }
            let orow = &mut out.data[p * n..(p + 1) * n];
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
    out
}

/// `A . B^T` for row-major `A (m x k)` and `B (n x k)`, giving `(m x n)`.
fn matmul_bt(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.cols);
    let (m, k, n) = (a.rows, a.cols, b.rows);
    let mut out = Mat::zeros(m, n);
    for i in 0..m {
        let arow = &a.data[i * k..(i + 1) * k];
        for j in 0..n {
            let brow = &b.data[j * k..(j + 1) * k];
            let mut s = 0.0;
            for p in 0..k {
                s += arow[p] * brow[p];
            }
            out.set(i, j, s);
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
const SVD_SEED: u64 = 0x6e6d_665f_7376_64; // "nmf_svd"

/// A standard-normal draw via Box-Muller.
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Modified Gram-Schmidt QR; returns the orthonormal `Q (m x n)` (the `R` factor
/// is not needed). Columns that collapse to near-zero are replaced by zeros.
fn mgs_q(a: &Mat) -> Mat {
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
fn jacobi_eigen(a_in: &Mat) -> (Vec<f64>, Mat) {
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
fn randomized_svd(x: &Mat, k: usize) -> (Mat, Vec<f64>, Mat) {
    let (d, v) = (x.rows, x.cols);
    let p = 10usize.min(v.saturating_sub(k));
    let r = (k + p).min(v).min(d);
    let mut rng = ChaCha8Rng::seed_from_u64(SVD_SEED);

    // Omega (v x r) Gaussian test matrix.
    let mut omega = Mat::zeros(v, r);
    for i in 0..v {
        for j in 0..r {
            omega.set(i, j, randn(&mut rng));
        }
    }

    // Y = X Omega, with power iterations Y <- X (X^T Y), re-orthonormalizing.
    let mut y = matmul(x, &omega); // d x r
    let mut q = mgs_q(&y);
    for _ in 0..4 {
        let xtq = matmul_at(x, &q); // v x r
        y = matmul(x, &xtq); // d x r
        q = mgs_q(&y);
    }

    // B = Q^T X  (r x v).
    let b = matmul_at(&q, x);
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
fn nndsvd_init(x: &Mat, k: usize) -> (Mat, Mat) {
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
    let mean = x.data.iter().sum::<f64>() / (d * v).max(1) as f64;
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

/// Random nonnegative initialization, scaled so `W H ~ X` in magnitude: entries
/// uniform on `[0, 1)` times `sqrt(mean(X)/K)`, seeded by the user `seed`.
fn random_init<R: Rng>(x: &Mat, k: usize, rng: &mut R) -> (Mat, Mat) {
    let (d, v) = (x.rows, x.cols);
    let mean = x.data.iter().sum::<f64>() / (d * v).max(1) as f64;
    let scale = (mean / k as f64).max(EPS).sqrt();
    let mut w = Mat::zeros(d, k);
    let mut h = Mat::zeros(k, v);
    for x in w.data.iter_mut() {
        *x = rng.gen::<f64>() * scale;
    }
    for x in h.data.iter_mut() {
        *x = rng.gen::<f64>() * scale;
    }
    (w, h)
}

// ---------------------------------------------------------------------------
// Reconstruction error
// ---------------------------------------------------------------------------

fn frobenius_error(x: &Mat, w: &Mat, h: &Mat) -> f64 {
    let wh = matmul(w, h);
    let mut s = 0.0;
    for i in 0..x.data.len() {
        let d = x.data[i] - wh.data[i];
        s += d * d;
    }
    0.5 * s
}

fn kl_error(x: &Mat, w: &Mat, h: &Mat) -> f64 {
    let wh = matmul(w, h);
    let mut s = 0.0;
    for i in 0..x.data.len() {
        let xi = x.data[i];
        let whi = wh.data[i].max(EPS);
        if xi > 0.0 {
            s += xi * (xi / whi).ln() - xi + whi;
        } else {
            s += whi;
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Multiplicative updates
// ---------------------------------------------------------------------------

/// One Frobenius multiplicative update of `H` then `W`.
fn mu_frobenius(x: &Mat, w: &mut Mat, h: &mut Mat) {
    // H *= (W^T X) / (W^T W H + eps).
    let wtx = matmul_at(w, x); // k x v
    let wtw = matmul_at(w, w); // k x k
    let wtwh = matmul(&wtw, h); // k x v
    for i in 0..h.data.len() {
        h.data[i] *= wtx.data[i] / (wtwh.data[i] + EPS);
    }
    // W *= (X H^T) / (W H H^T + eps).
    let xht = matmul_bt(x, h); // d x k
    let hht = matmul_bt(h, h); // k x k
    let whht = matmul(w, &hht); // d x k
    for i in 0..w.data.len() {
        w.data[i] *= xht.data[i] / (whht.data[i] + EPS);
    }
}

/// One KL multiplicative update of `H` then `W`.
fn mu_kl(x: &Mat, w: &mut Mat, h: &mut Mat) {
    let (d, v, k) = (x.rows, x.cols, w.cols);
    // H_kj *= [sum_i W_ik (X_ij / (WH)_ij)] / [sum_i W_ik].
    let wh = matmul(w, h); // d x v
    let mut ratio = Mat::zeros(d, v);
    for i in 0..d * v {
        ratio.data[i] = x.data[i] / (wh.data[i] + EPS);
    }
    let numer = matmul_at(w, &ratio); // k x v
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
    let wh = matmul(w, h);
    let mut ratio = Mat::zeros(d, v);
    for i in 0..d * v {
        ratio.data[i] = x.data[i] / (wh.data[i] + EPS);
    }
    let numer = matmul_bt(&ratio, h); // d x k
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

/// Build the document-term count matrix `X (D x V)` from token-id documents.
fn count_matrix(docs: &[Vec<u32>], num_types: usize) -> Mat {
    let d = docs.len();
    let mut x = Mat::zeros(d, num_types);
    for (i, doc) in docs.iter().enumerate() {
        for &w in doc {
            let w = w as usize;
            if w < num_types {
                x.data[i * num_types + w] += 1.0;
            }
        }
    }
    x
}

/// Build the TF-IDF document-term matrix: `tf * (ln((1+D)/(1+df)) + 1)`, then L2
/// normalize each document row. topica's own formula; no dependence on any
/// external transformer.
fn tfidf_matrix(docs: &[Vec<u32>], num_types: usize) -> Mat {
    let d = docs.len();
    let mut x = count_matrix(docs, num_types);
    // Document frequency per term.
    let mut df = vec![0usize; num_types];
    for r in 0..d {
        for c in 0..num_types {
            if x.at(r, c) > 0.0 {
                df[c] += 1;
            }
        }
    }
    let idf: Vec<f64> = (0..num_types)
        .map(|c| ((1.0 + d as f64) / (1.0 + df[c] as f64)).ln() + 1.0)
        .collect();
    for r in 0..d {
        for c in 0..num_types {
            let v = x.at(r, c) * idf[c];
            x.set(r, c, v);
        }
        // L2 normalize the row.
        let row = x.row(r);
        let n = norm(row);
        if n > 0.0 {
            for c in 0..num_types {
                let v = x.at(r, c) / n;
                x.set(r, c, v);
            }
        }
    }
    x
}

/// Fit NMF by multiplicative updates. `weighting_tfidf` selects the input matrix;
/// `beta_loss` selects the divergence; `init` selects the factor initialization;
/// `iters` is the maximum iteration count; `convergence_tol` stops early on the
/// relative reconstruction-error decrease; `seed` seeds `init = Random` only.
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
    fn planted(k: usize, block: usize, ndocs: usize, dlen: usize, seed: u64) -> (Vec<Vec<u32>>, usize) {
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
        let m = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Nndsvd, false, 200, 1e-4, 42);
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
        let a = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Nndsvd, false, 60, 0.0, 42);
        let b = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Nndsvd, false, 60, 0.0, 42);
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        // Random path (seeded by user seed).
        let c = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Random, false, 60, 0.0, 99);
        let d = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Random, false, 60, 0.0, 99);
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
        let mut x = Mat::zeros(d, v);
        for i in 0..d {
            for j in 0..v {
                x.set(i, j, a[i] * b[j]);
            }
        }
        let (_, s, _) = randomized_svd(&x, 2);
        let want = norm(&a) * norm(&b);
        assert!((s[0] - want).abs() / want < 1e-6, "sigma0 {} vs {}", s[0], want);
        // The second singular value of a rank-1 matrix is ~0.
        assert!(s[1] < 1e-6, "sigma1 {} should be near zero", s[1]);
    }

    #[test]
    fn nmf_conforms() {
        let (k, block) = (3usize, 8usize);
        let (docs, v) = planted(k, block, 180, 15, 1);
        let m = fit_nmf(&docs, k, v, BetaLoss::Frobenius, Init::Nndsvd, false, 150, 1e-4, 42);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
