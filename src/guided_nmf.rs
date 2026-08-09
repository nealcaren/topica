//! GuidedNMF: seed-word-guided semi-supervised NMF (Vendrow, Haddock, Rebrova &
//! Needell, "On a Guided Nonnegative Matrix Factorization," IEEE ICASSP 2021).
//!
//! Standard NMF factors the nonnegative document-term matrix `X (D x V)` as
//! `X ~ A S` with `A (D x K) >= 0` (document-topic) and `S (K x V) >= 0`
//! (topic-word). GuidedNMF adds a supervision term that ties the topic-word factor
//! to user-supplied seed-word groups, steering some learned topics toward those
//! words. With a seed matrix `Y (G x V)` (row `g` marks group `g`'s seed words) and
//! a nonnegative mixing matrix `B (G x K)`, the objective is
//!
//!   minimize over A, S, B >= 0:   ||X - A S||_F^2  +  lambda ||Y - B S||_F^2.
//!
//! The operational reference is the `ssnmf` package (MIT), pinned `ssnmf==0.0.2` by
//! the GuidedNMF repo, in its supervised Frobenius mode (`snmfmult`). We transcribe
//! its multiplicative updates (eps = 1e-10, W = L = 1, no early stop by default) and
//! reimplement them in Rust; we do not copy code. The per-iteration updates, in
//! ssnmf's order (A and B use the current S; S then uses the new A and B):
//!
//!   A <- A .* (X S^T)               ./ (A (S S^T)                 + eps)
//!   B <- B .* (Y S^T)               ./ (B (S S^T)                 + eps)
//!   S <- S .* (A^T X + lam B^T Y)   ./ ((A^T A) S + lam (B^T B) S + eps)
//!
//! Reported outputs are scale-invariant (NMF is invariant to S_k <- c_k S_k,
//! A_.k <- A_.k / c_k, B_.k <- B_.k / c_k): `topic_word` is each row of `S`
//! normalized to sum 1; `doc_topic` is `theta_dk ∝ A_dk ||S_k||_1` row-normalized;
//! `seed_topic_indices[g] = argmax_k B_gk ||S_k||_1`. Raw `A`, `S`, `B` are also
//! exposed.

use crate::estimator::{Estimator, ModelFamily};
use crate::nmf::{
    count_matrix, frobenius_error, matmul, matmul_at, matmul_bt, nndsvd_init, sp_at_x, sp_x_bt,
    tfidf_matrix, Mat, SpMat,
};
use rand::Rng;

/// Denominator guard, matching ssnmf's `eps`.
const EPS: f64 = 1e-10;

/// How to initialize the factors `A`, `S`, `B`.
pub enum GnmfInit {
    /// Independent `Uniform[0, 1]` draws (the reference default; seeded by the
    /// caller's RNG so a fixed seed reproduces bit-for-bit).
    Random,
    /// NNDSVDa on `X` for `A`, `S` (a deterministic topica extension); `B` is
    /// seeded `Uniform[0, 1]`. NNDSVD ignores `Y`, so it can settle a different
    /// basin than random init; documented, not the default.
    Nndsvd,
    /// Caller-supplied nonnegative factors (`A` D x K, `S` K x V, `B` G x K), as
    /// row vectors. The parity lever: feeding the reference's exact init isolates
    /// the update math.
    Explicit {
        a: Vec<Vec<f64>>,
        s: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
    },
}

/// Fitted GuidedNMF state; the PyO3 binding reads these back.
pub struct GuidedNMFModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_groups: usize,
    /// Topic-word matrix (K x V): each row of `S` normalized to sum 1.
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic matrix (D x K): `theta_dk ∝ A_dk ||S_k||_1`, row-normalized.
    pub doc_topic: Vec<Vec<f64>>,
    /// Raw factor `A` (D x K), before any normalization.
    pub a: Vec<Vec<f64>>,
    /// Raw factor `S` (K x V), before any normalization.
    pub s: Vec<Vec<f64>>,
    /// Raw factor `B` (G x K), before any normalization.
    pub b: Vec<Vec<f64>>,
    /// For each seed group `g`, the learned topic it most steers:
    /// `argmax_k B_gk ||S_k||_1`.
    pub seed_topic_indices: Vec<usize>,
    /// Final `||X - A S||_F` (the reconstruction part of the objective).
    pub reconstruction_error: f64,
    /// Objective `||X - A S||_F^2 + lambda ||Y - B S||_F^2` per iteration, with the
    /// initial value first (before any update).
    pub error_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
}

/// `sum_k` row-L1 norms of `S` (K x V), i.e. `||S_k||_1 = sum_j S_kj` (S >= 0).
fn row_l1(s: &Mat) -> Vec<f64> {
    (0..s.rows).map(|k| s.row(k).iter().sum()).collect()
}

/// Normalize each row to sum 1, leaving an all-zero row as zeros (no fabricated
/// uniform mass). Guards against NaN from a zero-sum divide.
fn normalize_rows_keep_zero(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|r| {
            let row = m.row(r);
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                row.iter().map(|&v| v / s).collect()
            } else {
                vec![0.0; m.cols]
            }
        })
        .collect()
}

/// `||Y - B S||_F^2` for dense `Y (G x V)` and factors `B (G x K)`, `S (K x V)`.
/// `G` is small (one row per seed group), so forming `B S` densely is cheap.
fn class_error_sq(y: &Mat, b: &Mat, s: &Mat) -> f64 {
    let bs = matmul(b, s); // G x V
    y.data
        .iter()
        .zip(bs.data.iter())
        .map(|(&yij, &bsij)| {
            let d = yij - bsij;
            d * d
        })
        .sum()
}

/// Fit GuidedNMF. `docs` are token-id sequences, `num_types` = V, `y` is the dense
/// seed matrix (G x V) built by the caller from the seed groups and vocabulary.
/// `lambda` weights the guidance term; `weighting_tfidf` selects the `X` weighting;
/// `convergence_tol > 0.0` enables an (opt-in, non-reference) early stop. Seed all
/// randomness from `rng` for bit-for-bit reproducibility.
#[allow(clippy::too_many_arguments)]
pub fn fit_guided_nmf<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    y: &[Vec<f64>],
    num_topics: usize,
    lambda: f64,
    weighting_tfidf: bool,
    init: GnmfInit,
    iters: usize,
    convergence_tol: f64,
    rng: &mut R,
) -> GuidedNMFModel {
    let k = num_topics;
    let y = Mat::from_rows(y);
    let g = y.rows;
    let x: SpMat = if weighting_tfidf {
        tfidf_matrix(docs, num_types)
    } else {
        count_matrix(docs, num_types)
    };
    let d = x.rows;

    let (mut a, mut s, mut b) = match init {
        GnmfInit::Explicit { a, s, b } => {
            (Mat::from_rows(&a), Mat::from_rows(&s), Mat::from_rows(&b))
        }
        GnmfInit::Nndsvd => {
            // NNDSVD is a deterministic (RNG-free) extension: A, S from the SVD,
            // and B from a deterministic uniform 1/K so a fixed config is
            // bit-exact across runs (no seed dependence). The multiplicative
            // updates then shape B toward the seed matrix.
            let (a0, s0) = nndsvd_init(&x, k);
            let mut b0 = Mat::zeros(g, k);
            b0.data.iter_mut().for_each(|v| *v = 1.0 / k as f64);
            (a0, s0, b0)
        }
        GnmfInit::Random => (
            rand_mat(d, k, rng),
            rand_mat(k, num_types, rng),
            rand_mat(g, k, rng),
        ),
    };

    let objective = |a: &Mat, s: &Mat, b: &Mat| -> f64 {
        // ||X - A S||_F^2 = 2 * (0.5 ||X - A S||_F^2); frobenius_error returns the
        // half-norm. The guidance term is dense but tiny (G x V).
        2.0 * frobenius_error(&x, a, s) + lambda * class_error_sq(&y, b, s)
    };

    let mut error_history = Vec::with_capacity(iters + 1);
    let mut prev = objective(&a, &s, &b);
    error_history.push(prev);
    let initial = prev;
    let mut converged = false;
    let mut iters_run = 0usize;

    for it in 0..iters {
        iters_run = it + 1;
        // S S^T (K x K) is shared by the A and B denominators (both use the
        // current S). Associate as A (S S^T) / B (S S^T) — never (A S) S^T, which
        // would materialize a dense D x V.
        let sst = matmul_bt(&s, &s); // K x K

        // A <- A .* (X S^T) ./ (A (S S^T) + eps).
        let x_st = sp_x_bt(&x, &s); // D x K
        let a_denom = matmul(&a, &sst); // D x K
        for i in 0..a.data.len() {
            a.data[i] *= x_st.data[i] / (a_denom.data[i] + EPS);
        }
        // B <- B .* (Y S^T) ./ (B (S S^T) + eps)  [still the current S].
        let y_st = matmul_bt(&y, &s); // G x K
        let b_denom = matmul(&b, &sst); // G x K
        for i in 0..b.data.len() {
            b.data[i] *= y_st.data[i] / (b_denom.data[i] + EPS);
        }
        // S <- S .* (A^T X + lam B^T Y) ./ ((A^T A) S + lam (B^T B) S + eps)
        // using the NEW A and B.
        let at_x = sp_at_x(&a, &x); // K x V
        let bt_y = matmul_at(&b, &y); // K x V
        let ata = matmul_at(&a, &a); // K x K
        let btb = matmul_at(&b, &b); // K x K
        let ata_s = matmul(&ata, &s); // K x V
        let btb_s = matmul(&btb, &s); // K x V
        for i in 0..s.data.len() {
            let numer = at_x.data[i] + lambda * bt_y.data[i];
            let denom = ata_s.data[i] + lambda * btb_s.data[i] + EPS;
            s.data[i] *= numer / denom;
        }

        let err = objective(&a, &s, &b);
        error_history.push(err);
        // Opt-in early stop (ssnmf 0.0.2 has none: default tol = 0.0 runs the full
        // budget). Relative decrease against the initial objective, as ssnmf's
        // newer builds and topica's NMF do.
        let rel = (prev - err).abs() / (initial.abs() + 1e-12);
        prev = err;
        if convergence_tol > 0.0 && rel < convergence_tol {
            converged = true;
            break;
        }
    }

    // Scale-invariant reported outputs. Unlike NMF's `normalize_rows` (which maps
    // an all-zero row to uniform), we keep all-zero rows zero: an extinct topic or
    // an empty document carries no evidence, so we do not fabricate uniform mass.
    let sl1 = row_l1(&s); // ||S_k||_1
    let topic_word = normalize_rows_keep_zero(&s);
    let mut theta = Mat::zeros(d, k);
    for i in 0..d {
        for kk in 0..k {
            theta.set(i, kk, a.at(i, kk) * sl1[kk]);
        }
    }
    let doc_topic = normalize_rows_keep_zero(&theta);
    let seed_topic_indices: Vec<usize> = (0..g)
        .map(|gg| {
            (0..k)
                .map(|kk| b.at(gg, kk) * sl1[kk])
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(kk, _)| kk)
                .unwrap_or(0)
        })
        .collect();

    let reconstruction_error = (2.0 * frobenius_error(&x, &a, &s)).max(0.0).sqrt();

    GuidedNMFModel {
        num_topics: k,
        num_types,
        num_groups: g,
        topic_word,
        doc_topic,
        a: a.rows_vec(),
        s: s.rows_vec(),
        b: b.rows_vec(),
        seed_topic_indices,
        reconstruction_error,
        error_history,
        converged,
        iters_run,
    }
}

/// `rows x cols` matrix of independent `Uniform[0, 1)` draws from `rng`, filled in
/// row-major order (deterministic for a fixed seed).
fn rand_mat<R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Mat {
    let mut m = Mat::zeros(rows, cols);
    for v in m.data.iter_mut() {
        *v = rng.gen::<f64>();
    }
    m
}

impl Estimator for GuidedNMFModel {
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
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // A planted 3-block corpus: docs 0..8 use words {0,1,2}, 8..16 {3,4,5},
    // 16..24 {6,7,8}. Two seed groups point at blocks 0 and 2.
    fn planted() -> (Vec<Vec<u32>>, usize) {
        let mut docs = Vec::new();
        for d in 0..24u32 {
            let base = (d / 8) * 3;
            docs.push(vec![base, base + 1, base + 2, base, base + 1]);
        }
        (docs, 9)
    }

    fn seed_y(num_types: usize) -> Vec<Vec<f64>> {
        // group 0 -> words {0,1}; group 1 -> words {6,7}.
        let mut y = vec![vec![0.0; num_types]; 2];
        y[0][0] = 1.0;
        y[0][1] = 1.0;
        y[1][6] = 1.0;
        y[1][7] = 1.0;
        y
    }

    #[test]
    fn guided_nmf_recovers_planted_topics() {
        let (docs, v) = planted();
        let y = seed_y(v);
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit_guided_nmf(
            &docs,
            v,
            &y,
            3,
            20.0,
            false,
            GnmfInit::Random,
            100,
            0.0,
            &mut rng,
        );
        // Each seed group should steer a topic whose top word is in its block.
        let t0 = m.seed_topic_indices[0];
        let t1 = m.seed_topic_indices[1];
        assert_ne!(t0, t1, "distinct seed groups should steer distinct topics");
        let top = |k: usize| {
            (0..v)
                .max_by(|&i, &j| m.topic_word[k][i].partial_cmp(&m.topic_word[k][j]).unwrap())
                .unwrap()
        };
        assert!(top(t0) <= 2, "group 0's topic should peak on block 0 words");
        assert!(
            (6..=8).contains(&top(t1)),
            "group 1's topic should peak on block 2 words"
        );
    }

    #[test]
    fn guided_nmf_is_deterministic() {
        let (docs, v) = planted();
        let y = seed_y(v);
        let fit = || {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            fit_guided_nmf(
                &docs,
                v,
                &y,
                3,
                20.0,
                false,
                GnmfInit::Random,
                50,
                0.0,
                &mut rng,
            )
        };
        let m1 = fit();
        let m2 = fit();
        assert_eq!(m1.s, m2.s);
        assert_eq!(m1.a, m2.a);
        assert_eq!(m1.b, m2.b);
    }

    #[test]
    fn guided_nmf_conforms() {
        let (docs, v) = planted();
        let y = seed_y(v);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_guided_nmf(
            &docs,
            v,
            &y,
            3,
            20.0,
            false,
            GnmfInit::Random,
            20,
            0.0,
            &mut rng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    #[test]
    fn objective_decreases_monotonically() {
        let (docs, v) = planted();
        let y = seed_y(v);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let m = fit_guided_nmf(
            &docs,
            v,
            &y,
            3,
            20.0,
            false,
            GnmfInit::Random,
            60,
            0.0,
            &mut rng,
        );
        for w in m.error_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "objective rose: {} -> {}", w[0], w[1]);
        }
    }
}
