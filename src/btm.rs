//! Biterm Topic Model (BTM) -- Yan, Guo, Lan & Cheng, "A Biterm Topic Model for
//! Short Text", WWW 2013.
//!
//! Short texts give LDA too few words per document to estimate a document-topic
//! mixture. BTM sidesteps that by modelling the corpus as a bag of **biterms** --
//! unordered word pairs co-occurring within a short window -- and learning one
//! *global* topic distribution plus per-topic word distributions from the biterm
//! co-occurrences. Both words of a biterm are drawn from the same topic, so the
//! topic-word distributions absorb the co-occurrence signal directly.
//!
//! Ported from the reference R `BTM` package (Jan Wijffels, Apache-2.0), which
//! wraps Xiaohui Yan's original C++. The collapsed-Gibbs conditional, the biterm
//! window construction, and the `sum_b` document inference match that source; the
//! RNG differs (topica uses `ChaCha8`), so a fit is reproducible for a fixed seed
//! but not bit-identical to R -- parity is measured by aligned topic-word cosine.

use crate::estimator::{Estimator, ModelFamily};
use rand::Rng;

/// A fitted Biterm Topic Model.
pub struct BtmModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: f64,
    pub beta: f64,
    pub window: usize,
    pub background: bool,
    /// Topic-word matrix φ (K × V); each row sums to 1.
    pub topic_word: Vec<Vec<f64>>,
    /// Global topic distribution θ (K); sums to 1.
    pub theta: Vec<f64>,
    /// Document-topic matrix (D × K) via `sum_b` inference; rows sum to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// Number of biterms extracted from the corpus.
    pub num_biterms: usize,
}

impl Estimator for BtmModel {
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
        Vec::new()
    }
    fn converged(&self) -> Option<bool> {
        None
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

/// Extract biterms from one document: every within-window unordered word pair,
/// stored `(min, max)`. Mirrors the reference `Doc::gen_biterms` (window `w`:
/// pairs `(i, j)` with `i < j < min(i + w, n)`). Documents shorter than two
/// tokens contribute nothing.
fn gen_biterms(doc: &[u32], window: usize, out: &mut Vec<(u32, u32)>) {
    let n = doc.len();
    if n < 2 {
        return;
    }
    for i in 0..n - 1 {
        let jmax = (i + window).min(n);
        for &wj in &doc[i + 1..jmax] {
            let wi = doc[i];
            out.push((wi.min(wj), wi.max(wj)));
        }
    }
}

/// Sample an index from an unnormalized weight vector by the inverse-CDF scan the
/// reference uses (`Sampler::mult_sample`): draw `u ~ U(0,1)`, return the first
/// `k` whose cumulative weight reaches `u * total`.
fn mult_sample<R: Rng>(p: &[f64], rng: &mut R) -> usize {
    let k = p.len();
    let mut cum = vec![0.0; k];
    let mut acc = 0.0;
    for i in 0..k {
        acc += p[i];
        cum[i] = acc;
    }
    let total = cum[k - 1];
    let u: f64 = rng.gen::<f64>() * total;
    for (i, &c) in cum.iter().enumerate() {
        if c >= u {
            return i;
        }
    }
    k - 1
}

/// `sum_b` document-topic inference: `p(z|d) = Σ_b p(z|b) p(b|d)` with `p(b|d)`
/// uniform over the document's biterms. Matches `Infer::doc_infer_sum_b`.
pub fn infer_doc(
    doc: &[u32],
    theta: &[f64],
    topic_word: &[Vec<f64>],
    num_topics: usize,
    window: usize,
) -> Vec<f64> {
    let k = num_topics;
    let mut pz_d = vec![0.0; k];
    if doc.is_empty() {
        return vec![1.0 / k as f64; k];
    }
    if doc.len() == 1 {
        let w = doc[0] as usize;
        for (z, p) in pz_d.iter_mut().enumerate() {
            *p = theta[z] * topic_word[z][w];
        }
    } else {
        let mut biterms = Vec::new();
        gen_biterms(doc, window, &mut biterms);
        for &(w1, w2) in &biterms {
            let (w1, w2) = (w1 as usize, w2 as usize);
            let mut pz_b = vec![0.0; k];
            let mut s = 0.0;
            for z in 0..k {
                pz_b[z] = theta[z] * topic_word[z][w1] * topic_word[z][w2];
                s += pz_b[z];
            }
            if s > 0.0 {
                for (z, pd) in pz_d.iter_mut().enumerate() {
                    *pd += pz_b[z] / s;
                }
            }
        }
    }
    let s: f64 = pz_d.iter().sum();
    if s > 0.0 {
        for p in pz_d.iter_mut() {
            *p /= s;
        }
    } else {
        pz_d.iter_mut().for_each(|p| *p = 1.0 / k as f64);
    }
    pz_d
}

/// Fit BTM on the given token-id documents with collapsed Gibbs sampling over
/// biterm topic assignments.
#[allow(clippy::too_many_arguments)]
pub fn fit_btm<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    alpha: f64,
    beta: f64,
    iters: usize,
    window: usize,
    background: bool,
    rng: &mut R,
) -> BtmModel {
    let k = num_topics;
    let v = num_types;

    // Empirical word distribution (for the optional background topic 0).
    let mut pw_b = vec![0.0f64; v];
    let mut biterms: Vec<(u32, u32)> = Vec::new();
    for doc in docs {
        gen_biterms(doc, window, &mut biterms);
        for &w in doc {
            if (w as usize) < v {
                pw_b[w as usize] += 1.0;
            }
        }
    }
    let pw_sum: f64 = pw_b.iter().sum();
    if pw_sum > 0.0 {
        for p in pw_b.iter_mut() {
            *p /= pw_sum;
        }
    }

    let b = biterms.len();
    // Count tables: nb_z[k] = #biterms in topic k, nwz[k][w] = word occurrences
    // (each biterm contributes both of its words).
    let mut nb_z = vec![0i64; k];
    let mut nwz = vec![vec![0i64; v]; k];
    let mut z_assign = vec![0usize; b];

    // Random initialization: each biterm to a uniform topic.
    for (bi, &(w1, w2)) in biterms.iter().enumerate() {
        let z = (rng.gen::<f64>() * k as f64) as usize;
        let z = z.min(k - 1);
        z_assign[bi] = z;
        nb_z[z] += 1;
        nwz[z][w1 as usize] += 1;
        nwz[z][w2 as usize] += 1;
    }

    let vbeta = v as f64 * beta;
    let denom_z = b as f64 + k as f64 * alpha;
    let mut pz = vec![0.0f64; k];

    for _ in 0..iters {
        for bi in 0..b {
            let (w1, w2) = (biterms[bi].0 as usize, biterms[bi].1 as usize);
            let z_old = z_assign[bi];
            // Remove this biterm's contribution.
            nb_z[z_old] -= 1;
            nwz[z_old][w1] -= 1;
            nwz[z_old][w2] -= 1;

            // Conditional p(z|b) ∝ p(z) p(w1|z) p(w2|z), matching compute_pz_b
            // (note the asymmetric +1 in the second word's denominator).
            for z in 0..k {
                let nbz = nb_z[z] as f64;
                let (pw1, pw2) = if background && z == 0 {
                    (pw_b[w1], pw_b[w2])
                } else {
                    (
                        (nwz[z][w1] as f64 + beta) / (2.0 * nbz + vbeta),
                        (nwz[z][w2] as f64 + beta) / (2.0 * nbz + 1.0 + vbeta),
                    )
                };
                let pk = (nbz + alpha) / denom_z;
                pz[z] = pk * pw1 * pw2;
            }
            let z_new = mult_sample(&pz, rng);
            z_assign[bi] = z_new;
            nb_z[z_new] += 1;
            nwz[z_new][w1] += 1;
            nwz[z_new][w2] += 1;
        }
    }

    // θ_k = (nb_z[k] + α) / (B + Kα).
    let theta: Vec<f64> = (0..k).map(|z| (nb_z[z] as f64 + alpha) / denom_z).collect();
    // φ_{k,w} = (nwz[k][w] + β) / (2 nb_z[k] + Vβ).
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|z| {
            let denom = 2.0 * nb_z[z] as f64 + vbeta;
            (0..v).map(|w| (nwz[z][w] as f64 + beta) / denom).collect()
        })
        .collect();

    let doc_topic: Vec<Vec<f64>> = docs
        .iter()
        .map(|doc| infer_doc(doc, &theta, &topic_word, k, window))
        .collect();

    BtmModel {
        num_topics: k,
        num_types: v,
        alpha,
        beta,
        window,
        background,
        topic_word,
        theta,
        doc_topic,
        num_biterms: b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Planted short-text corpus: `k` blocks of `block` words; each "document" is
    /// a handful of words drawn from one block.
    fn planted(
        k: usize,
        block: usize,
        ndocs: usize,
        dlen: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, usize) {
        let v = k * block;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let docs = (0..ndocs)
            .map(|d| {
                let bl = d % k;
                (0..dlen)
                    .map(|_| (bl * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        (docs, v)
    }

    #[test]
    fn test_btm_recovers_and_conforms() {
        let (k, block) = (3, 6);
        let (docs, v) = planted(k, block, 150, 5, 42);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_btm(&docs, k, v, 50.0 / k as f64, 0.01, 200, 15, false, &mut rng);

        assert_eq!(m.topic_word.len(), k);
        assert_eq!(m.topic_word[0].len(), v);
        assert_eq!(m.doc_topic.len(), docs.len());
        for row in &m.topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
        assert!((m.theta.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // Each recovered topic should concentrate on one planted block.
        let mut covered = std::collections::HashSet::new();
        for row in &m.topic_word {
            let top = (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            covered.insert(top / block);
        }
        assert_eq!(covered.len(), k, "topics should cover all planted blocks");
    }

    #[test]
    fn test_btm_determinism() {
        let (k, block) = (2, 5);
        let (docs, v) = planted(k, block, 60, 4, 123);
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            fit_btm(&docs, k, v, 1.0, 0.01, 80, 15, false, &mut rng)
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.theta, b.theta);
    }

    #[test]
    fn test_btm_biterms_window() {
        // window bounds which pairs form: "a b c" with window 2 gives (a,b),(b,c)
        // but not (a,c); window 3 (>= len) gives all three.
        let doc = vec![0u32, 1, 2];
        let mut w2 = Vec::new();
        gen_biterms(&doc, 2, &mut w2);
        assert_eq!(w2, vec![(0, 1), (1, 2)]);
        let mut w3 = Vec::new();
        gen_biterms(&doc, 3, &mut w3);
        assert_eq!(w3, vec![(0, 1), (0, 2), (1, 2)]);
    }
}
