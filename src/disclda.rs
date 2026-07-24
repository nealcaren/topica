//! DiscLDA -- Lacoste-Julien, Sha & Jordan, "DiscLDA: Discriminative Learning for
//! Dimensionality Reduction and Classification", NIPS 2008.
//!
//! DiscLDA augments LDA with a class-dependent transform on the topic space so the
//! latent representation is both descriptive and discriminative. With the paper's
//! block-structured transform (their eq. 1 & 3), the `L` actual topics partition
//! into `k_class` topics **specific to each class** (one disjoint block per class)
//! plus `k_shared` topics **shared by all classes**: a document of class `c` places
//! topic mass only on its own class block together with the shared block. That is
//! the same shared/class-specific separation a "contrastive topic model" wants, but
//! as a first-class, published feature.
//!
//! This module implements the **fixed-transform** variant (paper §4.1): with the
//! block transform frozen, DiscLDA is LDA with per-document topic restriction, a
//! restricted collapsed-Gibbs sampler (structurally like `LabeledLDA`, restricting
//! to `class-block ∪ shared-block` instead of a document's label set). It gives the
//! shared/class-specific topics and the discriminative features the paper reports on
//! 20 Newsgroups. (The learned-transform variant, §4.2, is a planned follow-up.)
//!
//! The RNG differs from any reference (topica uses ChaCha8), so a fit is reproducible
//! for a fixed seed. There is no canonical reference implementation; fidelity is
//! measured against the paper's 20 Newsgroups result (DiscLDA features beat plain-LDA
//! features through a linear classifier).

use crate::estimator::{Estimator, ModelFamily};
use rand::Rng;

/// A fitted DiscLDA model (fixed block transform).
pub struct DiscLdaModel {
    pub num_classes: usize,
    pub k_class: usize,
    pub k_shared: usize,
    pub num_types: usize,
    pub alpha: f64,
    pub beta: f64,
    /// Number of actual topics, `L = num_classes * k_class + k_shared`.
    pub num_topics: usize,
    /// Topic-word matrix φ (L × V); each row sums to 1.
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic matrix θ (D × L); rows sum to 1, mass only on the document's
    /// class block and the shared block.
    pub doc_topic: Vec<Vec<f64>>,
    /// Log class prior `ln p(c)` used by `predict`/`predict_proba`, length
    /// `num_classes`. The direct classifier is a topica-native plug-in: it scores
    /// a document under each class and combines the scores with this prior. The
    /// binding resolves it from the constructor's `class_prior` (empirical class
    /// frequencies by default, uniform, or a user-supplied vector); `fit_disclda`
    /// initialises it uniform. A shift-invariant softmax means only the prior's
    /// *relative* magnitudes matter.
    pub class_log_prior: Vec<f64>,
}

impl DiscLdaModel {
    /// Topic indices specific to class `c`: `[c*k_class, (c+1)*k_class)`.
    pub fn class_block(&self, c: usize) -> std::ops::Range<usize> {
        (c * self.k_class)..(c * self.k_class + self.k_class)
    }
    /// The shared topic indices: `[num_classes*k_class, L)`.
    pub fn shared_block(&self) -> std::ops::Range<usize> {
        let start = self.num_classes * self.k_class;
        start..(start + self.k_shared)
    }
    /// Topics a document of class `c` may use: its class block then the shared block.
    fn allowed(&self, c: usize) -> Vec<usize> {
        self.class_block(c).chain(self.shared_block()).collect()
    }
}

impl Estimator for DiscLdaModel {
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

fn allowed_for(c: usize, num_classes: usize, k_class: usize, k_shared: usize) -> Vec<usize> {
    let cls = (c * k_class)..(c * k_class + k_class);
    let sh_start = num_classes * k_class;
    cls.chain(sh_start..(sh_start + k_shared)).collect()
}

/// Fit fixed-transform DiscLDA by restricted collapsed Gibbs sampling. `docs[d]` is
/// the token ids of document `d`; `labels[d]` is its class in `0..num_classes`.
#[allow(clippy::too_many_arguments)]
pub fn fit_disclda<R: Rng>(
    docs: &[Vec<u32>],
    labels: &[usize],
    num_classes: usize,
    k_class: usize,
    k_shared: usize,
    num_types: usize,
    alpha: f64,
    beta: f64,
    iters: usize,
    rng: &mut R,
) -> DiscLdaModel {
    let l = num_classes * k_class + k_shared;
    let v = num_types;
    let d = docs.len();

    // Count tables: word-topic (L × V), topic totals (L), doc-topic (D × L).
    let mut nwt = vec![vec![0i64; v]; l];
    let mut nt = vec![0i64; l];
    let mut ndt = vec![vec![0i64; l]; d];
    let mut z: Vec<Vec<usize>> = docs.iter().map(|doc| vec![0usize; doc.len()]).collect();

    // Precompute each class's allowed-topic set.
    let allowed: Vec<Vec<usize>> = (0..num_classes)
        .map(|c| allowed_for(c, num_classes, k_class, k_shared))
        .collect();

    // Random init: each token to a random allowed topic for its document's class.
    for (di, doc) in docs.iter().enumerate() {
        let allow = &allowed[labels[di]];
        for (pos, &w) in doc.iter().enumerate() {
            let t = allow[(rng.gen::<f64>() * allow.len() as f64) as usize % allow.len()];
            z[di][pos] = t;
            ndt[di][t] += 1;
            nwt[t][w as usize] += 1;
            nt[t] += 1;
        }
    }

    let vbeta = v as f64 * beta;
    let mut scores: Vec<f64> = Vec::with_capacity(k_class + k_shared);
    for _ in 0..iters {
        for di in 0..d {
            let allow = &allowed[labels[di]];
            for pos in 0..docs[di].len() {
                let w = docs[di][pos] as usize;
                let old = z[di][pos];
                ndt[di][old] -= 1;
                nwt[old][w] -= 1;
                nt[old] -= 1;

                scores.clear();
                let mut total = 0.0f64;
                for &t in allow {
                    let s = (ndt[di][t] as f64 + alpha) * (nwt[t][w] as f64 + beta)
                        / (nt[t] as f64 + vbeta);
                    scores.push(s);
                    total += s;
                }
                let mut r = rng.gen::<f64>() * total;
                let mut chosen = allow[allow.len() - 1];
                for (i, &t) in allow.iter().enumerate() {
                    r -= scores[i];
                    if r <= 0.0 {
                        chosen = t;
                        break;
                    }
                }
                z[di][pos] = chosen;
                ndt[di][chosen] += 1;
                nwt[chosen][w] += 1;
                nt[chosen] += 1;
            }
        }
    }

    // φ_{l,w} = (nwt[l][w] + β) / (nt[l] + Vβ). Empty topics -> uniform.
    let topic_word: Vec<Vec<f64>> = (0..l)
        .map(|t| {
            let denom = nt[t] as f64 + vbeta;
            (0..v).map(|w| (nwt[t][w] as f64 + beta) / denom).collect()
        })
        .collect();

    // θ_{d,l} = (ndt[d][l] + α) / (len_d + (k_class+k_shared)·α) over allowed topics;
    // 0 elsewhere.
    let n_allowed = (k_class + k_shared) as f64;
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let len: i64 = ndt[di].iter().sum();
            let denom = len as f64 + n_allowed * alpha;
            let mut row = vec![0.0f64; l];
            for &t in &allowed[labels[di]] {
                row[t] = (ndt[di][t] as f64 + alpha) / denom;
            }
            row
        })
        .collect();

    DiscLdaModel {
        num_classes,
        k_class,
        k_shared,
        num_types,
        alpha,
        beta,
        num_topics: l,
        topic_word,
        doc_topic,
        // Uniform by default; the binding overrides from `class_prior` after fit.
        class_log_prior: vec![-(num_classes as f64).ln(); num_classes],
    }
}

/// Infer a document's topic distribution θ_c under a hypothesized class `c` (topics
/// restricted to class `c`'s block and the shared block) with φ fixed, plus the
/// document's log-likelihood under that class. Runs `sweeps` restricted Gibbs passes.
/// Returns `(theta_over_L, loglik)`.
///
/// The log-likelihood is a plug-in `Σ_w log Σ_l θ̂_l φ_{l,w}` at the posterior-mean
/// θ̂, not the fully marginalized evidence. This is unbiased *across classes* only
/// because every class's allowed set has the same size (`k_class + k_shared`), so no
/// class gets a capacity advantage in the `predict`/`predict_proba` comparison — do
/// not make `k_class` vary by class without revisiting this.
/// An empty document (no in-vocab tokens) yields loglik 0 for every class, so its
/// posterior in `predict_doc` reduces to the class prior.
pub fn infer_doc_class<R: Rng>(
    doc: &[u32],
    c: usize,
    model: &DiscLdaModel,
    sweeps: usize,
    rng: &mut R,
) -> (Vec<f64>, f64) {
    let l = model.num_topics;
    let allow = model.allowed(c);
    let alpha = model.alpha;
    let n_allowed = allow.len() as f64;

    let mut ndt = vec![0i64; l];
    let mut z = vec![0usize; doc.len()];
    for (pos, _) in doc.iter().enumerate() {
        let t = allow[(rng.gen::<f64>() * allow.len() as f64) as usize % allow.len()];
        z[pos] = t;
        ndt[t] += 1;
    }
    let mut scores = vec![0.0f64; allow.len()];
    for _ in 0..sweeps {
        for (pos, &w) in doc.iter().enumerate() {
            let w = w as usize;
            let old = z[pos];
            ndt[old] -= 1;
            let mut total = 0.0;
            for (i, &t) in allow.iter().enumerate() {
                let s = (ndt[t] as f64 + alpha) * model.topic_word[t][w];
                scores[i] = s;
                total += s;
            }
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = allow[allow.len() - 1];
            for (i, &t) in allow.iter().enumerate() {
                r -= scores[i];
                if r <= 0.0 {
                    chosen = t;
                    break;
                }
            }
            z[pos] = chosen;
            ndt[chosen] += 1;
        }
    }
    let len: i64 = ndt.iter().sum();
    let denom = len as f64 + n_allowed * alpha;
    let mut theta = vec![0.0f64; l];
    for &t in &allow {
        theta[t] = (ndt[t] as f64 + alpha) / denom;
    }
    // doc log-likelihood under class c: Σ_w log Σ_l θ_l φ_{l,w}
    let mut loglik = 0.0f64;
    for &w in doc {
        let w = w as usize;
        let mut p = 0.0f64;
        for &t in &allow {
            p += theta[t] * model.topic_word[t][w];
        }
        loglik += (p + 1e-300).ln();
    }
    (theta, loglik)
}

/// Predict class posteriors p(c|w) for a document and the class-marginalized
/// discriminative representation Σ_c p(c|w)·θ_c (length L). Runs one
/// `infer_doc_class` per class and softmaxes `ln p(c) + plug-in loglik(doc|c)`,
/// using `model.class_log_prior` (empirical class frequencies by default).
///
/// This is a plug-in *approximate* posterior: the per-class score is the
/// posterior-mean-θ likelihood, not the fully marginalized evidence. An empty /
/// all-OOV document has an identical (zero) likelihood under every class, so its
/// posterior is exactly the class prior — `predict`'s argmax is then the most
/// probable class under the prior (the majority class for an empirical prior),
/// not a sorted-order tie break.
pub fn predict_doc<R: Rng>(
    doc: &[u32],
    model: &DiscLdaModel,
    sweeps: usize,
    rng: &mut R,
) -> (Vec<f64>, Vec<f64>) {
    let c = model.num_classes;
    let mut logliks = vec![0.0f64; c];
    let mut thetas = vec![Vec::new(); c];
    for ci in 0..c {
        let (theta, ll) = infer_doc_class(doc, ci, model, sweeps, rng);
        logliks[ci] = ll + model.class_log_prior.get(ci).copied().unwrap_or(0.0);
        thetas[ci] = theta;
    }
    // softmax over class log-posteriors (log-prior + plug-in log-likelihood)
    let m = logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut post: Vec<f64> = logliks.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = post.iter().sum();
    for p in post.iter_mut() {
        *p /= s;
    }
    // marginalized representation
    let mut rep = vec![0.0f64; model.num_topics];
    for ci in 0..c {
        for t in 0..model.num_topics {
            rep[t] += post[ci] * thetas[ci][t];
        }
    }
    (post, rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Planted corpus with class-specific + shared structure. Class c documents draw
    /// from a class-specific word block (distinct per class) plus a shared block that
    /// all classes use. Returns (docs, labels, vocab_size, n_class_words, n_shared).
    fn planted(
        num_classes: usize,
        class_block_w: usize,
        shared_block_w: usize,
        ndocs: usize,
        dlen: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, Vec<usize>, usize) {
        // vocab layout: [class0 words][class1 words]...[shared words]
        let v = num_classes * class_block_w + shared_block_w;
        let shared_start = num_classes * class_block_w;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut docs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..ndocs {
            let c = i % num_classes;
            let mut doc = Vec::new();
            for _ in 0..dlen {
                // mostly class-specific words, some shared words
                if rng.gen::<f64>() < 0.65 {
                    let w = c * class_block_w + (rng.gen::<f64>() * class_block_w as f64) as usize;
                    doc.push(w as u32);
                } else {
                    let w = shared_start + (rng.gen::<f64>() * shared_block_w as f64) as usize;
                    doc.push(w as u32);
                }
            }
            docs.push(doc);
            labels.push(c);
        }
        (docs, labels, v)
    }

    #[test]
    fn test_disclda_recovers_class_and_shared_structure() {
        let (nc, cbw, sbw) = (3, 6, 6);
        let (docs, labels, v) = planted(nc, cbw, sbw, 240, 12, 42);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        // k_class=1 per class, k_shared=2
        let m = fit_disclda(&docs, &labels, nc, 1, 2, v, 0.1, 0.01, 300, &mut rng);

        assert_eq!(m.num_topics, nc + 2); // k_class=1 per class + k_shared=2
        for row in &m.topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
        // Each class's specific topic should peak on that class's word block.
        let shared_start = nc * cbw;
        for c in 0..nc {
            let t = m.class_block(c).start; // the single class-c topic
            let row = &m.topic_word[t];
            let top = (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            assert!(
                top < shared_start && top / cbw == c,
                "class {c} topic should peak on its own block, peaked on word {top}"
            );
        }
        // The shared topics should peak on the shared word block.
        for t in m.shared_block() {
            let row = &m.topic_word[t];
            let top = (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            assert!(
                top >= shared_start,
                "shared topic {t} should peak on shared block"
            );
        }
    }

    #[test]
    fn test_disclda_predicts_class() {
        let (nc, cbw, sbw) = (3, 6, 6);
        let (docs, labels, v) = planted(nc, cbw, sbw, 240, 12, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_disclda(&docs, &labels, nc, 2, 2, v, 0.1, 0.01, 300, &mut rng);
        // Held-out docs of a known class should be predicted correctly a large
        // majority of the time.
        let (test, tlabels, _) = planted(nc, cbw, sbw, 60, 12, 999);
        let mut correct = 0;
        let mut prng = ChaCha8Rng::seed_from_u64(3);
        for (doc, &c) in test.iter().zip(&tlabels) {
            let (post, _) = predict_doc(doc, &m, 60, &mut prng);
            let pred = (0..nc)
                .max_by(|&a, &b| post[a].total_cmp(&post[b]))
                .unwrap();
            if pred == c {
                correct += 1;
            }
        }
        assert!(
            correct as f64 / test.len() as f64 > 0.7,
            "predicted {correct}/{}",
            test.len()
        );
    }

    #[test]
    fn test_disclda_determinism() {
        let (nc, cbw, sbw) = (2, 5, 5);
        let (docs, labels, v) = planted(nc, cbw, sbw, 80, 10, 123);
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            fit_disclda(&docs, &labels, nc, 2, 2, v, 0.1, 0.01, 120, &mut rng)
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
    }

    #[test]
    fn empty_doc_posterior_equals_class_prior() {
        // An empty / all-OOV document has zero likelihood under every class, so its
        // posterior must be exactly the (softmaxed) class prior — not a uniform
        // tie or a sorted-order argmax.
        let (nc, cbw, sbw) = (2, 5, 5);
        let (docs, labels, v) = planted(nc, cbw, sbw, 80, 10, 5);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut m = fit_disclda(&docs, &labels, nc, 2, 2, v, 0.1, 0.01, 100, &mut rng);
        // A skewed empirical-style prior: p = [0.8, 0.2].
        m.class_log_prior = vec![0.8f64.ln(), 0.2f64.ln()];
        let mut prng = ChaCha8Rng::seed_from_u64(2);
        let (post, _) = predict_doc(&[], &m, 30, &mut prng);
        assert!((post[0] - 0.8).abs() < 1e-9, "post {post:?}");
        assert!((post[1] - 0.2).abs() < 1e-9, "post {post:?}");
    }
}
