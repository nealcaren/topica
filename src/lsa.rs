//! LSA / LSI, latent semantic analysis (Deerwester et al., "Indexing by Latent
//! Semantic Analysis", JASIS 1990; the SVD machinery follows Halko et al.,
//! "Finding Structure with Randomness", SIAM Review 2011).
//!
//! We take a truncated SVD of the weighted document-term matrix `X (D x V)`:
//!
//! ```text
//!   X ~ U_k Sigma_k V_k^T,   U_k (D x K), Sigma_k (K), V_k^T (K x V)
//! ```
//!
//! Unlike the probabilistic topic models in topica, LSA is a linear-algebraic
//! decomposition, not a generative model. Its outputs are SIGNED latent
//! coordinates, not probabilities:
//!
//! - `topic_word (K x V)` is the right singular vectors `V_k` (each row a
//!   component). These are signed term loadings; `top_words` ranks by absolute
//!   value because a large-magnitude negative loading is as defining of a
//!   component as a large positive one.
//! - `doc_topic (D x K)` is `U_k Sigma_k`, the document coordinates in the
//!   reduced space. Rows do NOT sum to 1; LSA is not mixed-membership.
//! - `singular_values (K)` are `Sigma_k`, the energy of each component.
//!
//! The SVD is unique up to a per-component sign flip. We fix the sign with the
//! `svd_flip` convention used by scikit-learn's `TruncatedSVD`: for each
//! component we flip the `(u, v)` pair together so the largest-magnitude entry of
//! the right singular vector is positive. That makes the fit deterministic and
//! directly comparable to the reference.
//!
//! The truncated SVD itself is the same randomized range-finder NMF uses for its
//! NNDSVD initialization (`nmf::randomized_svd_seeded`), and the weighted-matrix
//! builders are NMF's (`nmf::count_matrix` / `nmf::tfidf_matrix`). We do not
//! duplicate that linear algebra here.

use crate::estimator::{Estimator, ModelFamily};
use crate::nmf::{count_matrix, randomized_svd_seeded, tfidf_matrix};

/// A fitted LSA/LSI model. `topic_word` are the signed right singular vectors,
/// `doc_topic` are the signed document coordinates `U Sigma`, and
/// `singular_values` are the truncated singular values.
pub struct LsaModel {
    pub num_topics: usize,
    pub num_types: usize,
    /// Right singular vectors `V_k` as rows (K x V). Signed term loadings.
    pub topic_word: Vec<Vec<f64>>,
    /// Document coordinates `U_k Sigma_k` (D x K). Signed; rows do not sum to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// Truncated singular values `Sigma_k` (length K).
    pub singular_values: Vec<f64>,
    /// Frobenius reconstruction error of the rank-K truncation,
    /// `sqrt(||X||_F^2 - sum_k Sigma_k^2)` on the same weighted matrix the SVD
    /// factors. Monotone-decreasing in K, so it is the LSA scree curve (the SVD
    /// analogue of NMF's `reconstruction_error`).
    pub reconstruction_error: f64,
}

/// Apply the scikit-learn `svd_flip` sign convention in place: for each component
/// `c`, find the entry of the right singular vector `vt[c, :]` with the largest
/// absolute value; if that entry is negative, negate the whole component (both
/// `vt` row `c` and `u` column `c`). Ties on `|value|` resolve to the lowest
/// index, matching numpy's `argmax`, so the convention is deterministic.
///
/// `u` is `D x K` (singular vectors in columns), `vt` is `K x V` (singular vectors
/// in rows). Flipping the pair together preserves `U Sigma V^T`.
fn svd_flip(u: &mut [Vec<f64>], vt: &mut [Vec<f64>], k: usize, v: usize) {
    for c in 0..k {
        // argmax over |vt[c, j]| (first occurrence on ties).
        let mut best_j = 0usize;
        let mut best_abs = -1.0f64;
        for j in 0..v {
            let a = vt[c][j].abs();
            if a > best_abs {
                best_abs = a;
                best_j = j;
            }
        }
        if vt[c][best_j] < 0.0 {
            for j in 0..v {
                vt[c][j] = -vt[c][j];
            }
            for row in u.iter_mut() {
                row[c] = -row[c];
            }
        }
    }
}

/// Fit LSA/LSI: build the weighted document-term matrix (TF-IDF when
/// `weighting_tfidf`, else raw counts), take a truncated SVD to `num_topics`
/// components, apply the `svd_flip` sign convention, and read off the signed
/// factors. `seed` seeds the randomized-SVD sketch (deterministic per seed).
pub fn fit_lsa(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    weighting_tfidf: bool,
    seed: u64,
) -> LsaModel {
    let k = num_topics;
    let x = if weighting_tfidf {
        tfidf_matrix(docs, num_types)
    } else {
        count_matrix(docs, num_types)
    };
    let d = x.rows;
    let v = x.cols;

    let (u, s, vt) = randomized_svd_seeded(&x, k, seed);
    let kk = s.len(); // randomized_svd caps k at min(d, v, k); usually == k.

    // doc_topic = U Sigma (D x K); topic_word = Vt rows (K x V).
    let mut u_rows: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..kk).map(|c| u.at(i, c)).collect())
        .collect();
    let mut vt_rows: Vec<Vec<f64>> = (0..kk)
        .map(|c| (0..v).map(|j| vt.at(c, j)).collect())
        .collect();

    svd_flip(&mut u_rows, &mut vt_rows, kk, v);

    let singular_values = s.clone();
    let doc_topic: Vec<Vec<f64>> = u_rows
        .iter()
        .map(|row| (0..kk).map(|c| row[c] * s[c]).collect())
        .collect();

    // Frobenius residual of the rank-K truncation: ||X||_F^2 = sum of all squared
    // singular values, and the top-K SVD captures the K largest, so the discarded
    // energy is ||X||_F^2 - sum_k Sigma_k^2. Computed from X directly (no need for
    // the full spectrum) so it is exact up to the randomized SVD's approximation.
    let kept_sq: f64 = s.iter().map(|v| v * v).sum();
    let reconstruction_error = (x.frob_sq() - kept_sq).max(0.0).sqrt();

    LsaModel {
        num_topics: kk,
        num_types,
        topic_word: vt_rows,
        doc_topic,
        singular_values,
        reconstruction_error,
    }
}

impl LsaModel {
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word.clone()
    }
}

impl Estimator for LsaModel {
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
        // SVD is a direct (non-iterative) solve, so there is no convergence
        // trajectory to report.
        Vec::new()
    }
    fn converged(&self) -> Option<bool> {
        // Direct solve: "converged" is not a meaningful state for a one-shot SVD.
        None
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// A planted-block corpus: K well-separated word blocks, each document drawn
    /// from a single block. Mirrors the NMF test fixture.
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
    fn fit_recovers_planted_structure() {
        // With K disjoint word blocks, the top-K right singular vectors span the
        // block structure: each component's largest-|loading| words should come
        // from a single block, and together the components should touch all K
        // blocks. (LSA components are signed and orthogonal, not one-per-topic
        // probability simplices, so we check block-purity of the dominant
        // loadings, not a Hungarian topic match.)
        let (k, block) = (3usize, 8usize);
        let (docs, v) = planted(k, block, 180, 15, 1);
        let m = fit_lsa(&docs, k, v, true, 42);
        assert_eq!(m.num_topics, k);
        assert_eq!(m.topic_word.len(), k);
        assert_eq!(m.topic_word[0].len(), v);
        assert_eq!(m.doc_topic.len(), docs.len());
        assert_eq!(m.singular_values.len(), k);
        // Singular values are non-increasing and non-negative.
        for c in 1..k {
            assert!(m.singular_values[c - 1] >= m.singular_values[c] - 1e-9);
            assert!(m.singular_values[c] >= -1e-12);
        }
        // Each component's top-|loading| words concentrate in one block.
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| {
                m.topic_word[t][b]
                    .abs()
                    .total_cmp(&m.topic_word[t][a].abs())
            });
            let blocks: std::collections::HashSet<usize> =
                ord[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "component {t} top loadings mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k, "components did not cover all blocks");
    }

    #[test]
    fn determinism_same_seed() {
        let (k, block) = (3usize, 6usize);
        let (docs, v) = planted(k, block, 90, 12, 7);
        for &tfidf in &[true, false] {
            let a = fit_lsa(&docs, k, v, tfidf, 42);
            let b = fit_lsa(&docs, k, v, tfidf, 42);
            assert_eq!(a.topic_word, b.topic_word);
            assert_eq!(a.doc_topic, b.doc_topic);
            assert_eq!(a.singular_values, b.singular_values);
        }
    }

    #[test]
    fn thread_count_independent() {
        // The randomized SVD reuses NMF's rayon+sparse path, which combines partial
        // sums in a fixed order; the fit must be bit-identical at 1 vs 8 threads.
        let (k, block) = (4usize, 7usize);
        let (docs, v) = planted(k, block, 200, 18, 11);
        for &tfidf in &[true, false] {
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| fit_lsa(&docs, k, v, tfidf, 77));
            let many = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap()
                .install(|| fit_lsa(&docs, k, v, tfidf, 77));
            assert_eq!(
                one.topic_word, many.topic_word,
                "topic_word differs by thread count"
            );
            assert_eq!(
                one.doc_topic, many.doc_topic,
                "doc_topic differs by thread count"
            );
            assert_eq!(one.singular_values, many.singular_values);
        }
    }

    #[test]
    fn svd_flip_sign_stability() {
        // After svd_flip, the largest-|value| entry of every right singular vector
        // is non-negative, by construction. This is the determinism anchor for the
        // sign convention (no dependence on the arbitrary SVD sign).
        let (k, block) = (3usize, 6usize);
        let (docs, v) = planted(k, block, 120, 12, 5);
        let m = fit_lsa(&docs, k, v, true, 42);
        for c in 0..k {
            let mut best = 0usize;
            let mut ba = -1.0f64;
            for j in 0..v {
                let a = m.topic_word[c][j].abs();
                if a > ba {
                    ba = a;
                    best = j;
                }
            }
            assert!(
                m.topic_word[c][best] >= 0.0,
                "component {c}: dominant loading should be non-negative after svd_flip"
            );
        }
    }

    #[test]
    fn recovers_singular_values_rank_one() {
        // X = a b^T (rank 1): sigma_0 = ||a|| ||b||, sigma_1 ~ 0. We build the
        // corpus directly as a count matrix would: make every document identical
        // in shape so the leading singular triplet is the rank-1 structure.
        //
        // Construct a corpus of `nd` copies of a single document over `v` words,
        // with word j appearing b[j] times. Then the count matrix has identical
        // rows, i.e. X = 1 (b^T), a rank-1 matrix with sigma_0 = sqrt(nd) ||b||.
        let v = 5usize;
        let b = [2u32, 1, 3, 1, 2];
        let nd = 9usize;
        let doc: Vec<u32> = (0..v)
            .flat_map(|j| std::iter::repeat_n(j as u32, b[j] as usize))
            .collect();
        let docs: Vec<Vec<u32>> = (0..nd).map(|_| doc.clone()).collect();
        let m = fit_lsa(&docs, 2, v, false, 42);
        let bnorm: f64 = b
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let want = (nd as f64).sqrt() * bnorm;
        assert!(
            (m.singular_values[0] - want).abs() / want < 1e-6,
            "sigma0 {} vs {}",
            m.singular_values[0],
            want
        );
        assert!(
            m.singular_values[1] < 1e-6,
            "sigma1 {} should be near zero for a rank-1 matrix",
            m.singular_values[1]
        );
    }

    #[test]
    fn lsa_conforms() {
        let (k, block) = (3usize, 8usize);
        let (docs, v) = planted(k, block, 180, 15, 1);
        let m = fit_lsa(&docs, k, v, true, 42);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
