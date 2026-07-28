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
use crate::reduce::jacobi_eigen_symmetric;
use rand::Rng;

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
    rng: &mut R,
) -> SemanticSignalSeparationModel {
    let d = doc_emb.len();
    let m = doc_emb.first().map(|r| r.len()).unwrap_or(0);
    let k = num_topics;

    // 1. Center the document embeddings (per feature, over documents). `mean_`
    //    also centers the vocabulary in the projection step.
    let mut mean = vec![0.0f64; m];
    for row in doc_emb {
        for (f, &x) in row.iter().enumerate() {
            mean[f] += x;
        }
    }
    for mval in mean.iter_mut() {
        *mval /= d as f64;
    }
    let xc: Vec<Vec<f64>> = doc_emb
        .iter()
        .map(|row| row.iter().zip(&mean).map(|(&x, &mu)| x - mu).collect())
        .collect();

    // 2. Unit-variance whitening. From the covariance C = Xcᵀ Xc (M x M), whose
    //    eigenpairs are the squared singular values / left singular vectors of Xcᵀ,
    //    the whitening rows are Kw[j] = evec_j / sqrt(eval_j). ICA is invariant to
    //    the whitening rotation, so we need not match scikit-learn's SVD sign/order.
    let mut cov = vec![0.0f64; m * m];
    for row in &xc {
        for a in 0..m {
            let ra = row[a];
            let base = a * m;
            for b in a..m {
                cov[base + b] += ra * row[b];
            }
        }
    }
    for a in 0..m {
        for b in (a + 1)..m {
            cov[b * m + a] = cov[a * m + b];
        }
    }
    let (evals, evecs) = jacobi_eigen_symmetric(&cov, m);
    let tiny = f64::MIN_POSITIVE;
    // Whitening matrix Kw (K x M) and the whitened, unit-variance-scaled data
    // X1 (K x D) = Kw @ Xcᵀ * sqrt(D).
    let sqrt_d = (d as f64).sqrt();
    let mut kw = vec![vec![0.0f64; m]; k];
    for j in 0..k {
        let dj = evals[j].max(tiny).sqrt();
        for f in 0..m {
            kw[j][f] = evecs[j][f] / dj;
        }
    }
    // X1[j][i] = sum_f Kw[j][f] * Xc[i][f] * sqrt(D).
    let mut x1 = vec![vec![0.0f64; d]; k];
    for j in 0..k {
        for (i, row) in xc.iter().enumerate() {
            let mut s = 0.0;
            for f in 0..m {
                s += kw[j][f] * row[f];
            }
            x1[j][i] = s * sqrt_d;
        }
    }

    // 3. Parallel FastICA with the logcosh nonlinearity (alpha = 1).
    let mut w_init = vec![vec![0.0f64; k]; k];
    for row in w_init.iter_mut() {
        for v in row.iter_mut() {
            *v = next_standard_normal(rng);
        }
    }
    let mut w = sym_decorrelation(&w_init, k);
    let mut history = Vec::new();
    let mut converged = false;
    let dp = d as f64;
    for ii in 0..iters {
        // WX = W @ X1 (K x D); gwtx = tanh(WX); g_wtx[j] = mean_i (1 - gwtx²).
        let mut gwtx = vec![vec![0.0f64; d]; k];
        let mut g_mean = vec![0.0f64; k];
        for j in 0..k {
            let mut gm = 0.0;
            for i in 0..d {
                let mut wx = 0.0;
                for l in 0..k {
                    wx += w[j][l] * x1[l][i];
                }
                let t = wx.tanh();
                gwtx[j][i] = t;
                gm += 1.0 - t * t;
            }
            g_mean[j] = gm / dp;
        }
        // W1_pre[j][l] = (1/D) sum_i gwtx[j][i] X1[l][i]  -  g_mean[j] W[j][l].
        let mut w1_pre = vec![vec![0.0f64; k]; k];
        for j in 0..k {
            for l in 0..k {
                let mut s = 0.0;
                for i in 0..d {
                    s += gwtx[j][i] * x1[l][i];
                }
                w1_pre[j][l] = s / dp - g_mean[j] * w[j][l];
            }
        }
        let w1 = sym_decorrelation(&w1_pre, k);
        // lim = max_j | |<W1[j], W[j]>| - 1 |.
        let mut lim = 0.0f64;
        for j in 0..k {
            let mut dot = 0.0;
            for l in 0..k {
                dot += w1[j][l] * w[j][l];
            }
            lim = lim.max((dot.abs() - 1.0).abs());
        }
        w = w1;
        history.push((ii + 1, lim));
        if lim < tol {
            converged = true;
            break;
        }
    }

    // 4. Sources S = (W @ X1)ᵀ (D x K), then unit-variance rescale: divide each
    //    column by its (population) std and fold the same factor into W.
    let mut s = vec![vec![0.0f64; k]; d];
    for (i, srow) in s.iter_mut().enumerate() {
        for (j, sval) in srow.iter_mut().enumerate() {
            let mut acc = 0.0;
            for l in 0..k {
                acc += w[j][l] * x1[l][i];
            }
            *sval = acc;
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
        let std = (var / dp).sqrt().max(tiny);
        for srow in s.iter_mut() {
            srow[j] /= std;
        }
        for l in 0..k {
            w[j][l] /= std;
        }
    }

    // 5. Unmixing components_ = W @ Kw (K x M): the map from a centered embedding
    //    to its source scores.
    let mut comps_emb = vec![vec![0.0f64; m]; k];
    for j in 0..k {
        for f in 0..m {
            let mut acc = 0.0;
            for l in 0..k {
                acc += w[j][l] * kw[l][f];
            }
            comps_emb[j][f] = acc;
        }
    }

    // 6. Project the vocabulary: axial[k][v] = components_[k] · (vocab[v] - mean_).
    let vv = vocab_emb.len();
    let mut axial = vec![vec![0.0f64; vv]; k];
    for (v, wrow) in vocab_emb.iter().enumerate() {
        for j in 0..k {
            let mut acc = 0.0;
            for f in 0..m {
                acc += comps_emb[j][f] * (wrow[f] - mean[f]);
            }
            axial[j][v] = acc;
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
    let doc_topic = positive_normalized_rows(&source_scores_rows(&s));

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

/// Identity pass-through kept for readability at the call site (the sources are
/// already the D x K row layout the normalizer wants).
fn source_scores_rows(s: &[Vec<f64>]) -> Vec<Vec<f64>> {
    s.to_vec()
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
        if negative {
            idx.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap());
        } else {
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
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
            &mut r1,
        );
        let b = fit(
            &docs,
            &vocab,
            k,
            FeatureImportance::Combined,
            100,
            1e-4,
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
            &mut rng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }
}
