//! Semantic Signal Separation (S³): topics as independent axes of semantic space.
//!
//! S³ (Kardos, Kostkan, Enevoldsen, Vermillet, Nielbo & Rocca, "S³ - Semantic
//! Signal Separation", ACL 2025) treats each topic as an *independent component*
//! of the contextual embedding space rather than a distribution over a
//! bag-of-words. It decomposes the document embeddings with FastICA; each
//! recovered independent component is a topic axis, and a term's importance to a
//! topic is read off by projecting the vocabulary embeddings onto that axis. It
//! operates directly in the embedding space, with no bag-of-words modelling.
//!
//! We follow the reference implementation, turftopic's `SemanticSignalSeparation`
//! (MIT, Márton Kardos), which wraps scikit-learn's `FastICA` with the default
//! `logcosh`/parallel/unit-variance configuration and then projects the vocabulary
//! embeddings through the same decomposition. The FastICA numerics
//! (unit-variance whitening, symmetric decorrelation, the parallel logcosh
//! fixed-point) mirror scikit-learn's `_fastica.py`. Nothing here uses torch; the
//! decomposition is pure linear algebra over embeddings the caller supplies (a
//! document-embedding matrix and an aligned vocabulary-embedding matrix in one
//! shared space), exactly as [`crate::top2vec`] takes its embeddings.
//!
//! ICA axes are *signed*: each topic has a positive and a negative pole. We expose
//! the signed per-word importance (`components`, and the raw `axial` projection)
//! and the signed document loadings (`source_scores`) as the native S³ outputs,
//! and additionally derive the nonnegative, row-normalized `topic_word` (φ) and
//! `doc_topic` (θ) that topica's shared analysis surface expects, by taking the
//! positive pole of each and normalizing (the same clamp-and-normalize the
//! embedding-cluster models use). Storage is `Vec<Vec<f64>>` throughout so the
//! fitted state does not depend on `ndarray` (which lives behind the `embeddings`
//! feature).

use crate::estimator::{Estimator, ModelFamily};
use crate::linalg::qr_reduced;
use crate::reduce::jacobi_eigen_symmetric;
use ndarray::Array2;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Which per-word importance the model reports as `components` / `topic_word`.
/// Matches turftopic's `feature_importance` knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureImportance {
    /// The raw projection of the vocabulary onto each axis (`axial_components_`).
    Axial,
    /// Cosine of each word's axial vector to the standard basis axis
    /// (`angular_components_`).
    Angular,
    /// `square(axial) * angular` (turftopic's default): sharp, sign from angular.
    Combined,
}

impl FeatureImportance {
    /// Parse the public string form; `None` for an unknown value.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "axial" => Some(Self::Axial),
            "angular" => Some(Self::Angular),
            "combined" => Some(Self::Combined),
            _ => None,
        }
    }
    /// The public string form (round-trips `parse`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Axial => "axial",
            Self::Angular => "angular",
            Self::Combined => "combined",
        }
    }
}

/// Fitted state for [`fit`]. `topic_word`/`doc_topic` are the nonnegative,
/// row-normalized surface every topica model shares; `components`, `axial`, and
/// `source_scores` are the signed S³-native outputs.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticSignalSeparationModel {
    pub num_topics: usize,
    pub feature_importance: FeatureImportance,
    /// Signed per-word importance under `feature_importance` (K x V). Drives the
    /// positive/negative poles of `top_words`.
    pub components: Vec<Vec<f64>>,
    /// The raw axial projection of the vocabulary onto each axis (K x V), i.e.
    /// `axial_components_`. Kept so the angular/combined views are reconstructible
    /// and for callers who want the unweighted projection.
    pub axial: Vec<Vec<f64>>,
    /// Signed document loadings on each axis, the raw ICA sources (D x K).
    pub source_scores: Vec<Vec<f64>>,
    /// Nonnegative topic-word distribution φ (K x V), the positive pole of
    /// `components` row-normalized to sum to 1.
    pub topic_word: Vec<Vec<f64>>,
    /// Nonnegative document-topic distribution θ (D x K), the positive pole of
    /// `source_scores` row-normalized to sum to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// `(iteration, convergence measure)` for the FastICA fixed-point.
    pub fit_history: Vec<(usize, f64)>,
    /// Whether FastICA reached `tol` before `iters`.
    pub converged: bool,
}

/// Fit S³ on document + vocabulary embeddings.
///
/// * `doc_emb` — (D x M) document embeddings, one row per document.
/// * `vocab_emb` — (V x M) vocabulary embeddings, aligned to the corpus vocabulary
///   and in the same space as `doc_emb`.
/// * `num_topics` — K, the number of independent components (topics).
/// * `feature_importance` — how signed per-word importance is scored.
/// * `iters` — FastICA `max_iter`.
/// * `tol` — FastICA convergence tolerance.
/// * `missing` — optional per-vocabulary-term mask (length V); a `true` entry marks
///   a corpus term the caller supplied no embedding for (its `vocab_emb` row is a
///   placeholder). Such terms get exactly zero importance on every axis, rather
///   than the spurious score a placeholder vector would project to. An empty slice
///   means every term is present.
/// * `rng` — seeds the FastICA `w_init`; a fixed seed reproduces bit-for-bit.
///
/// `num_topics` must be `<= min(D, M)` (FastICA cannot extract more components than
/// samples or features); the binding validates this before calling.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng>(
    doc_emb: &[Vec<f64>],
    vocab_emb: &[Vec<f64>],
    num_topics: usize,
    feature_importance: FeatureImportance,
    iters: usize,
    tol: f64,
    missing: &[bool],
    rng: &mut R,
) -> SemanticSignalSeparationModel {
    let d = doc_emb.len();
    let m = doc_emb.first().map(|r| r.len()).unwrap_or(0);
    let k = num_topics;
    let vv = vocab_emb.len();
    // FastICA cannot extract more components than samples or features; the binding
    // rejects this before calling. Guard the core too so a direct Rust caller gets a
    // clear failure instead of an out-of-bounds panic in `whiten_kw` (where L clamps
    // to `min(k+10, m, d)` and the top-K read would index past `l`).
    debug_assert!(
        k <= d.min(m),
        "num_topics ({k}) must be <= min(num_docs {d}, embedding_dim {m}) for FastICA"
    );
    let dp = d as f64;
    let sqrt_d = dp.sqrt();

    // 1. Center the document embeddings (per feature, over documents). `mean_`
    //    also centers the vocabulary in the projection step. `xc` (D x M) is the
    //    centered matrix, built row-major for the whitening range finder.
    let mut mean = vec![0.0f64; m];
    for row in doc_emb {
        for (f, &x) in row.iter().enumerate() {
            mean[f] += x;
        }
    }
    for mval in mean.iter_mut() {
        *mval /= dp;
    }
    let mut xc_flat = vec![0.0f64; d * m];
    for (i, row) in doc_emb.iter().enumerate() {
        for f in 0..m {
            xc_flat[i * m + f] = row[f] - mean[f];
        }
    }

    // 2. Unit-variance whitening. The whitening rows are Kw[j] = √D · V[:,j] / σ_j,
    //    where V (M x K) are the top-K right singular vectors of the centered data
    //    Xc (equivalently the top-K eigenvectors of Xcᵀ Xc) and σ_j its singular
    //    values. The √D folds in scikit-learn's unit-variance scale so `components_`
    //    (and hence `axial`) match the reference magnitude. ICA is invariant to the
    //    whitening rotation, so the SVD sign/order need not match sklearn. `whiten_kw`
    //    finds only the top-K via a seeded randomized range finder (fast and
    //    deterministic), so a fixed seed reproduces the fit bit-for-bit.
    let xc = Array2::from_shape_vec((d, m), xc_flat).expect("xc shape");
    let svd_seed = rng.gen::<u64>();
    let kw = whiten_kw(&xc, d, m, k, sqrt_d, svd_seed);
    // X1 (K x D) = Kw @ Xcᵀ. The √D is already folded into Kw.
    let x1 = kw.dot(&xc.t());

    // 3. Parallel FastICA with the logcosh nonlinearity (alpha = 1). The two hot
    //    matmuls per iteration (W @ X1 and gwtx @ X1ᵀ) go through ndarray's GEMM.
    let mut w_init = vec![vec![0.0f64; k]; k];
    for row in w_init.iter_mut() {
        for v in row.iter_mut() {
            *v = next_standard_normal(rng);
        }
    }
    let mut w = vecs_to_nd(&sym_decorrelation(&w_init, k));
    let mut history = Vec::new();
    let mut converged = false;
    for ii in 0..iters {
        // gwtx = tanh(W @ X1) (K x D). `mapv_into` applies tanh in place on the GEMM
        // output instead of allocating a second K x D array each iteration.
        let gwtx = w.dot(&x1).mapv_into(f64::tanh);
        // g_mean[j] = mean_i (1 - gwtx[j,i]²).
        let mut g_mean = vec![0.0f64; k];
        for j in 0..k {
            let ss: f64 = gwtx.row(j).iter().map(|&t| t * t).sum();
            g_mean[j] = 1.0 - ss / dp;
        }
        // W1_pre = (gwtx @ X1ᵀ)/D − diag(g_mean) · W.
        let mut w1_pre = gwtx.dot(&x1.t()) / dp;
        for j in 0..k {
            let gm = g_mean[j];
            for l in 0..k {
                w1_pre[[j, l]] -= gm * w[[j, l]];
            }
        }
        let w1 = vecs_to_nd(&sym_decorrelation(&nd_to_vecs(&w1_pre), k));
        // lim = max_j | |<W1[j], W[j]>| − 1 |. NaN must propagate (a non-finite
        // update is not "converged"): `f64::max` drops NaN, so track it explicitly.
        let mut lim = 0.0f64;
        for j in 0..k {
            let mut dot = 0.0;
            for l in 0..k {
                dot += w1[[j, l]] * w[[j, l]];
            }
            let v = (dot.abs() - 1.0).abs();
            lim = if v.is_nan() { f64::NAN } else { lim.max(v) };
        }
        w = w1;
        history.push((ii + 1, lim));
        if lim.is_finite() && lim < tol {
            converged = true;
            break;
        }
    }

    // 4. Sources S = (W @ X1)ᵀ (D x K), then unit-variance rescale: divide each
    //    column by its (population) std and fold the same factor into W.
    let s_nd = w.dot(&x1); // K x D
    let mut s = vec![vec![0.0f64; k]; d];
    for i in 0..d {
        for j in 0..k {
            s[i][j] = s_nd[[j, i]];
        }
    }
    for j in 0..k {
        let mut mu = 0.0;
        for srow in &s {
            mu += srow[j];
        }
        mu /= dp;
        let mut var = 0.0;
        for srow in &s {
            let e = srow[j] - mu;
            var += e * e;
        }
        let std = (var / dp).sqrt().max(f64::MIN_POSITIVE);
        for srow in s.iter_mut() {
            srow[j] /= std;
        }
        for l in 0..k {
            w[[j, l]] /= std;
        }
    }

    // 5. Unmixing components_ = W @ Kw (K x M): the map from a centered embedding
    //    to its source scores.
    let comps_emb = w.dot(&kw); // K x M

    // 6. Project the vocabulary: axial[k][v] = components_[k] · (vocab[v] − mean_).
    //    Build the centered vocabulary Vc (V x M), then axial = comps_emb @ Vcᵀ.
    let mut vc = Array2::<f64>::zeros((vv, m));
    for (v, wrow) in vocab_emb.iter().enumerate() {
        for f in 0..m {
            vc[[v, f]] = wrow[f] - mean[f];
        }
    }
    let axial_nd = comps_emb.dot(&vc.t()); // K x V
    let mut axial: Vec<Vec<f64>> = (0..k)
        .map(|j| (0..vv).map(|v| axial_nd[[j, v]]).collect())
        .collect();

    // 6b. Terms with no caller-supplied embedding carry a placeholder vector, which
    //     would otherwise project to a spurious (often dominant) score; force their
    //     importance to exactly zero on every axis so they never surface as a topic
    //     word.
    if !missing.is_empty() {
        for arow in axial.iter_mut() {
            for (v, a) in arow.iter_mut().enumerate() {
                if missing[v] {
                    *a = 0.0;
                }
            }
        }
    }

    // 6c. Canonicalize each axis's sign (ICA recovers components only up to sign):
    //     orient each topic so its positive pole carries the larger word mass. This
    //     makes the reported "positive pole" the dominant one and guarantees the
    //     nonnegative `topic_word` has real support instead of collapsing to uniform.
    //     Flip `axial` and the matching `source_scores` column together.
    for j in 0..k {
        let mut pos = 0.0;
        let mut neg = 0.0;
        for &a in &axial[j] {
            if a > 0.0 {
                pos += a;
            } else {
                neg -= a;
            }
        }
        if neg > pos {
            for a in axial[j].iter_mut() {
                *a = -*a;
            }
            for srow in s.iter_mut() {
                srow[j] = -srow[j];
            }
        }
    }

    // 7. angular[k][v] = axial[k][v] / ||axial[:,v]||₂ (cosine of the word's axial
    //    vector to the standard basis axis e_k); combined = axial² * angular.
    let mut col_norm = vec![0.0f64; vv];
    for v in 0..vv {
        let mut s2 = 0.0;
        for j in 0..k {
            s2 += axial[j][v] * axial[j][v];
        }
        col_norm[v] = s2.sqrt();
    }
    let components: Vec<Vec<f64>> = match feature_importance {
        FeatureImportance::Axial => axial.clone(),
        FeatureImportance::Angular => (0..k)
            .map(|j| {
                (0..vv)
                    .map(|v| {
                        if col_norm[v] > 0.0 {
                            axial[j][v] / col_norm[v]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect(),
        FeatureImportance::Combined => (0..k)
            .map(|j| {
                (0..vv)
                    .map(|v| {
                        let ang = if col_norm[v] > 0.0 {
                            axial[j][v] / col_norm[v]
                        } else {
                            0.0
                        };
                        axial[j][v] * axial[j][v] * ang
                    })
                    .collect()
            })
            .collect(),
    };

    // 8. Nonnegative, row-normalized φ / θ for the shared surface.
    let topic_word = positive_normalized_rows(&components);
    // `s` is already the D x K row layout the normalizer wants (no clone needed).
    let doc_topic = positive_normalized_rows(&s);

    SemanticSignalSeparationModel {
        num_topics: k,
        feature_importance,
        components,
        axial,
        source_scores: s,
        topic_word,
        doc_topic,
        fit_history: history,
        converged,
    }
}

/// The positive pole of each row, normalized to sum to 1. A row with no positive
/// entry falls back to uniform, so every row is a valid distribution.
fn positive_normalized_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let mut out: Vec<f64> = row.iter().map(|&x| x.max(0.0)).collect();
            let sum: f64 = out.iter().sum();
            if sum > 0.0 {
                for x in out.iter_mut() {
                    *x /= sum;
                }
            } else {
                let n = out.len().max(1);
                for x in out.iter_mut() {
                    *x = 1.0 / n as f64;
                }
            }
            out
        })
        .collect()
}

/// The reduced-QR orthonormal factor Q (rows x cols) of `a`, as an ndarray.
fn qr_q(a: &Array2<f64>, rows: usize, cols: usize) -> Array2<f64> {
    let flat: Vec<f64> = a.iter().copied().collect();
    let (q, _r) = qr_reduced(&flat, rows, cols);
    Array2::from_shape_vec((rows, cols), q).expect("qr Q shape")
}

/// Unit-variance whitening matrix Kw (K x M) via a seeded randomized range finder.
///
/// Finds the top-K right singular vectors V (and singular values σ) of the centered
/// data Xc (D x M) without a full M x M eigendecomposition: sketch Xc onto an
/// L = K + oversample subspace, orthonormalize (with power iterations for accuracy),
/// then eigendecompose the small L x L Gram of the projected data. Returns
/// `Kw[j] = √D · V[:,j] / σ_j`. Seeded, so the whitening is deterministic; the L x L
/// eigenproblem uses the robust cyclic-Jacobi solver so large K stays stable.
fn whiten_kw(
    xc: &Array2<f64>,
    d: usize,
    m: usize,
    k: usize,
    sqrt_d: f64,
    seed: u64,
) -> Array2<f64> {
    let l = (k + 10).min(m).min(d);
    let mut srng = ChaCha8Rng::seed_from_u64(seed);
    let mut omega = Array2::<f64>::zeros((m, l));
    for x in omega.iter_mut() {
        *x = next_standard_normal(&mut srng);
    }
    // Range finder Q (D x L) for Xc, refined by a few power iterations.
    let mut q = qr_q(&xc.dot(&omega), d, l);
    for _ in 0..7 {
        let z = xc.t().dot(&q); // M x L
        let qz = qr_q(&z, m, l);
        let y = xc.dot(&qz); // D x L
        q = qr_q(&y, d, l);
    }
    // B = Qᵀ Xc (L x M); the right singular vectors of Xc equal those of B.
    let b = q.t().dot(xc); // L x M
    let bbt = b.dot(&b.t()); // L x L = Ub diag(σ²) Ubᵀ
    let bbt_flat: Vec<f64> = bbt.iter().copied().collect();
    let (evals, evecs) = jacobi_eigen_symmetric(&bbt_flat, l);
    let lam0 = evals.first().copied().unwrap_or(0.0).max(1.0);
    let lam_floor = lam0 * 1e-24;
    let mut kw = Array2::<f64>::zeros((k, m));
    for j in 0..k {
        let sj = evals[j].max(lam_floor).sqrt();
        // V[:,j] = Bᵀ Ub[:,j] / σ_j, so Kw[j][f] = √D · V[f][j] / σ_j.
        for f in 0..m {
            let mut vfj = 0.0;
            for (l2, &u) in evecs[j].iter().enumerate() {
                vfj += b[[l2, f]] * u;
            }
            kw[[j, f]] = sqrt_d * vfj / (sj * sj);
        }
    }
    kw
}

/// Row-major `Vec<Vec<f64>>` (K x K) to a K x K ndarray.
fn vecs_to_nd(rows: &[Vec<f64>]) -> Array2<f64> {
    let k = rows.len();
    let mut a = Array2::<f64>::zeros((k, k));
    for (j, row) in rows.iter().enumerate() {
        for (l, &x) in row.iter().enumerate() {
            a[[j, l]] = x;
        }
    }
    a
}

/// A K x K ndarray back to row-major `Vec<Vec<f64>>`.
fn nd_to_vecs(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// Symmetric decorrelation: W <- (W Wᵀ)^{-1/2} W. Eigendecompose the K x K Gram
/// W Wᵀ, form U diag(1/√s) Uᵀ, and left-multiply W.
fn sym_decorrelation(w: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    // Gram G = W Wᵀ (K x K), row-major.
    let mut g = vec![0.0f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for l in 0..w[a].len() {
                s += w[a][l] * w[b][l];
            }
            g[a * k + b] = s;
        }
    }
    let (s, u) = jacobi_eigen_symmetric(&g, k);
    let tiny = f64::MIN_POSITIVE;
    // P = U diag(1/√s) Uᵀ (K x K, symmetric). u[j] is the eigenvector for s[j].
    let mut p = vec![vec![0.0f64; k]; k];
    for (j, &sj) in s.iter().enumerate() {
        let scale = 1.0 / sj.max(tiny).sqrt();
        for a in 0..k {
            let ua = u[j][a] * scale;
            for b in 0..k {
                p[a][b] += ua * u[j][b];
            }
        }
    }
    // W_new = P @ W.
    let cols = w[0].len();
    let mut out = vec![vec![0.0f64; cols]; k];
    for a in 0..k {
        for (l, wl) in w.iter().enumerate() {
            let pal = p[a][l];
            for c in 0..cols {
                out[a][c] += pal * wl[c];
            }
        }
    }
    out
}

/// One standard-normal draw via Box-Muller from a uniform RNG. Deterministic given
/// the seeded `rng`, so `w_init` (and the whole fit) reproduces bit-for-bit.
fn next_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>(); // in (0, 1], safe for ln
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl SemanticSignalSeparationModel {
    /// Top `n` words of `topic` at the given pole, as `(word_id, importance)`.
    /// The positive pole is the axis's highest `components` values (topica's usual
    /// "top φ" meaning); the negative pole is the lowest (most negative), returned
    /// with their signed importance.
    pub fn top_words(&self, n: usize, topic: usize, negative: bool) -> Vec<(usize, f64)> {
        let row = &self.components[topic];
        let mut idx: Vec<usize> = (0..row.len()).collect();
        // `total_cmp` gives a total order (NaN-safe), so a stray non-finite score
        // sorts deterministically instead of panicking on `partial_cmp().unwrap()`.
        if negative {
            idx.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        } else {
            idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        }
        idx.into_iter().take(n).map(|i| (i, row[i])).collect()
    }
}

impl Estimator for SemanticSignalSeparationModel {
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
        self.fit_history.clone()
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

    /// Build embeddings with a planted independent-axis structure: two latent
    /// signals drive disjoint embedding dimensions, and vocabulary words load on
    /// one signal or the other. S³ should recover the two axes and each axis's
    /// words.
    fn planted() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, usize) {
        // 4 embedding dims: dims 0-1 carry signal A, dims 2-3 carry signal B.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 120;
        let mut docs = Vec::new();
        for _ in 0..n {
            // Independent, non-Gaussian sources (uniform), the ICA-friendly case.
            let a = rng.gen::<f64>() * 2.0 - 1.0;
            let b = rng.gen::<f64>() * 2.0 - 1.0;
            docs.push(vec![a, 0.9 * a, b, 0.9 * b]);
        }
        // Vocab: 2 words on axis A (dims 0-1), 2 on axis B (dims 2-3).
        let vocab = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.8, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.8],
        ];
        (docs, vocab, 2)
    }

    #[test]
    fn recovers_planted_axes() {
        let (docs, vocab, k) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            200,
            1e-4,
            &[],
            &mut rng,
        );
        // Each topic's strongest pole should own one of the two word blocks:
        // words {0,1} (axis A) or {2,3} (axis B), not a mix.
        let block_a = [0usize, 1];
        let block_b = [2usize, 3];
        let mut owns = Vec::new();
        for t in 0..k {
            // Which pole is stronger for this axis?
            let pos = m.top_words(2, t, false);
            let neg = m.top_words(2, t, true);
            let top = if pos[0].1.abs() >= neg[0].1.abs() {
                pos
            } else {
                neg
            };
            let ids: Vec<usize> = top.iter().map(|&(i, _)| i).collect();
            let in_a = ids.iter().filter(|i| block_a.contains(i)).count();
            let in_b = ids.iter().filter(|i| block_b.contains(i)).count();
            owns.push(if in_a >= in_b { 0 } else { 1 });
        }
        owns.sort();
        assert_eq!(owns, vec![0, 1], "the two axes must recover the two blocks");
    }

    #[test]
    fn is_deterministic() {
        let (docs, vocab, k) = planted();
        let mut r1 = ChaCha8Rng::seed_from_u64(3);
        let mut r2 = ChaCha8Rng::seed_from_u64(3);
        let a = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            100,
            1e-4,
            &[],
            &mut r1,
        );
        let b = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            100,
            1e-4,
            &[],
            &mut r2,
        );
        assert_eq!(a.components, b.components);
        assert_eq!(a.source_scores, b.source_scores);
        // A different seed must give a different fit (so the test cannot pass trivially).
        let mut r3 = ChaCha8Rng::seed_from_u64(999);
        let c = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            100,
            1e-4,
            &[],
            &mut r3,
        );
        assert!(a.source_scores != c.source_scores);
    }

    #[test]
    fn sources_are_unit_variance() {
        let (docs, vocab, k) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let m = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            200,
            1e-4,
            &[],
            &mut rng,
        );
        let d = m.source_scores.len() as f64;
        for j in 0..k {
            let mu: f64 = m.source_scores.iter().map(|r| r[j]).sum::<f64>() / d;
            let var: f64 = m
                .source_scores
                .iter()
                .map(|r| (r[j] - mu).powi(2))
                .sum::<f64>()
                / d;
            assert!(
                (var.sqrt() - 1.0).abs() < 1e-6,
                "source column {j} std != 1"
            );
        }
    }

    #[test]
    fn conforms() {
        let (docs, vocab, k) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            100,
            1e-4,
            &[],
            &mut rng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }
}
