//! Polylingual Topic Model (PLTM) -- Mimno, Wallach, Naradowsky, Smith &
//! McCallum, "Polylingual Topic Models", EMNLP 2009.
//!
//! PLTM extends LDA to **document tuples**: a tuple is a set of documents that
//! are loosely equivalent (parallel translations, or comparable articles such as
//! linked Wikipedia pages) written in `L` different languages. Every document in
//! a tuple shares ONE tuple-specific topic distribution θ; each topic is a *set*
//! of word distributions Φ¹…Φ^L, one per language, drawn from a language-specific
//! symmetric Dirichlet(βˡ). Because the topic index is shared across languages,
//! topic `k` denotes the same latent theme in every language -- the topics are
//! aligned by construction, with no post-hoc matching.
//!
//! Inference is collapsed Gibbs sampling. Integrating out θ and Φ, the training
//! conditional for a token in tuple `d`, language `l`, with word `w` and current
//! topic `t` is standard LDA collapsed Gibbs **except the document-topic count is
//! shared across all languages in the tuple**:
//!
//! ```text
//! P(z = t | ·) ∝ (N_dt\ + α m_t) · (M^l_{w,t}\ + βˡ) / (M^l_{t·}\ + V_l βˡ)
//! ```
//!
//! where `N_dt` counts topic `t` across *all languages* of tuple `d`, `M^l_{w,t}`
//! is the language-`l` word-topic count, `M^l_{t·}` is the language-`l` topic
//! total, and `\` excludes the token being resampled. (Paper eq. 4 is the *test*
//! conditional with fixed Φ; the training sampler integrates Φ out, hence the
//! word-count ratio here.)
//!
//! The asymmetric prior αm is re-estimated by a Minka fixed-point step every
//! `optimize_interval` iterations (the paper re-estimates every 10), reusing the
//! same sufficient-statistic recipe as topica's LDA (`optimize.rs`).
//!
//! Validated against MALLET's `cc.mallet.topics.PolylingualTopicModel` as a
//! black-box oracle (MALLET is weak-copyleft, so this port is paper-derived; the
//! reference is used only to compare outputs). The RNG differs (topica uses
//! ChaCha8), so a fit is reproducible for a fixed seed but not bit-identical to
//! MALLET -- parity is measured by aligned per-language topic-word cosine.

use crate::estimator::{Estimator, ModelFamily};
use crate::optimize::digamma;
use rand::Rng;

/// A fitted Polylingual Topic Model.
pub struct PltmModel {
    pub num_topics: usize,
    /// Per-language vocabulary sizes V_l (length L).
    pub vocab_sizes: Vec<usize>,
    /// Per-topic asymmetric prior αm (length K); `alpha_sum` is its total.
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    /// Per-language topic-word prior βˡ (length L).
    pub beta: Vec<f64>,
    /// Per-language topic-word matrices φˡ: `topic_word[l]` is (K × V_l), rows
    /// sum to 1. Topic index `k` denotes the same theme across languages.
    pub topic_word: Vec<Vec<Vec<f64>>>,
    /// Tuple-topic matrix θ (D × K); rows sum to 1.
    pub doc_topic: Vec<Vec<f64>>,
}

impl Estimator for PltmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    /// The Estimator surface returns a single topic-word matrix; PLTM exposes the
    /// **first language's** φ here (the full per-language set is on the model).
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word
            .first()
            .cloned()
            .unwrap_or_default()
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

/// Sample an index from an unnormalized weight vector by an inverse-CDF scan
/// (matching the family's `mult_sample`): draw `u ~ U(0,1)·total`, return the
/// first `k` whose cumulative weight reaches `u`.
fn mult_sample<R: Rng>(p: &[f64], rng: &mut R) -> usize {
    let k = p.len();
    let mut acc = 0.0;
    let total: f64 = p.iter().sum();
    let u: f64 = rng.gen::<f64>() * total;
    for (i, &pi) in p.iter().enumerate() {
        acc += pi;
        if acc >= u {
            return i;
        }
    }
    k - 1
}

/// One Minka fixed-point step on the asymmetric prior αm, using tuple-level topic
/// counts as the sufficient statistics (the tuple plays LDA's document role, its
/// length being the total tokens across all languages). Mirrors
/// `optimize::optimize_alpha` but reads the shared `n_dk` counts. Deterministic.
fn optimize_alpha_step(alpha: &mut [f64], alpha_sum: &mut f64, n_dk: &[Vec<i64>], num_topics: usize) {
    let max_len = n_dk
        .iter()
        .map(|row| row.iter().sum::<i64>() as usize)
        .max()
        .unwrap_or(0);
    if max_len == 0 {
        return;
    }
    let mut doc_length_hist = vec![0u32; max_len + 1];
    let mut topic_doc_hist = vec![vec![0u32; max_len + 1]; num_topics];
    for row in n_dk {
        let len: i64 = row.iter().sum();
        doc_length_hist[len as usize] += 1;
        for (t, &c) in row.iter().enumerate() {
            if c > 0 {
                topic_doc_hist[t][c as usize] += 1;
            }
        }
    }

    let dg_alpha_sum = digamma(*alpha_sum);
    let denominator: f64 = doc_length_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + *alpha_sum) - dg_alpha_sum))
        .sum();
    if denominator <= 0.0 {
        return;
    }

    let mut new_sum = 0.0;
    for t in 0..num_topics {
        let dg_alpha_t = digamma(alpha[t]);
        let numerator: f64 = topic_doc_hist[t]
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, &c)| c > 0)
            .map(|(c, &cnt)| cnt as f64 * (digamma(c as f64 + alpha[t]) - dg_alpha_t))
            .sum();
        alpha[t] = (alpha[t] * numerator / denominator).max(1e-10);
        new_sum += alpha[t];
    }
    *alpha_sum = new_sum;
}

/// Infer the tuple-topic distribution θ for one (possibly held-out) tuple, given
/// fixed language topic-word matrices. Runs `sweeps` Gibbs passes over the
/// tuple's tokens with θ integrated out at the tuple level. `tuple[l]` is the
/// language-`l` token ids (already mapped to that language's vocabulary).
pub fn infer_tuple<R: Rng>(
    tuple: &[Vec<u32>],
    topic_word: &[Vec<Vec<f64>>],
    alpha: &[f64],
    alpha_sum: f64,
    sweeps: usize,
    rng: &mut R,
) -> Vec<f64> {
    let k = alpha.len();
    let mut n_dk = vec![0i64; k];
    // z[l] holds the current topic of each token in language l.
    let mut z: Vec<Vec<usize>> = tuple
        .iter()
        .map(|toks| {
            toks.iter()
                .map(|_| {
                    let t = (rng.gen::<f64>() * k as f64) as usize;
                    let t = t.min(k - 1);
                    n_dk[t] += 1;
                    t
                })
                .collect()
        })
        .collect();

    let len: i64 = n_dk.iter().sum();
    let mut p = vec![0.0f64; k];
    for _ in 0..sweeps {
        for (l, toks) in tuple.iter().enumerate() {
            for (n, &w) in toks.iter().enumerate() {
                let w = w as usize;
                let z_old = z[l][n];
                n_dk[z_old] -= 1;
                for t in 0..k {
                    p[t] = (n_dk[t] as f64 + alpha[t]) * topic_word[l][t][w];
                }
                let z_new = mult_sample(&p, rng);
                z[l][n] = z_new;
                n_dk[z_new] += 1;
            }
        }
    }
    let denom = len as f64 + alpha_sum;
    if denom <= 0.0 {
        return vec![1.0 / k as f64; k];
    }
    (0..k).map(|t| (n_dk[t] as f64 + alpha[t]) / denom).collect()
}

/// Fit PLTM on aligned per-language token documents. `docs[l][d]` is the
/// language-`l` bag of word ids for tuple `d`; every language must have the same
/// number of tuples `D` (a tuple absent in a language is an empty inner vector).
/// `vocab_sizes[l]` is V_l, `beta[l]` is that language's topic-word prior.
#[allow(clippy::too_many_arguments)]
pub fn fit_pltm<R: Rng>(
    docs: &[Vec<Vec<u32>>],
    num_topics: usize,
    vocab_sizes: &[usize],
    alpha_init: f64,
    beta: &[f64],
    iters: usize,
    optimize_alpha: bool,
    optimize_interval: usize,
    optimize_burn_in: usize,
    rng: &mut R,
) -> PltmModel {
    let k = num_topics;
    let num_langs = docs.len();
    let num_docs = docs.first().map(|l| l.len()).unwrap_or(0);

    // Shared tuple-topic counts (D × K).
    let mut n_dk = vec![vec![0i64; k]; num_docs];
    // Per-language word-major counts wk[l][w] = [K]; topic totals mk[l] = [K].
    let mut wk: Vec<Vec<Vec<i64>>> = vocab_sizes
        .iter()
        .map(|&v| vec![vec![0i64; k]; v])
        .collect();
    let mut mk: Vec<Vec<i64>> = vec![vec![0i64; k]; num_langs];
    // Topic assignments z[l][d] parallel to docs[l][d].
    let mut z: Vec<Vec<Vec<usize>>> = docs
        .iter()
        .map(|lang| lang.iter().map(|doc| vec![0usize; doc.len()]).collect())
        .collect();

    // Random initialization.
    for l in 0..num_langs {
        for d in 0..num_docs {
            for (n, &w) in docs[l][d].iter().enumerate() {
                let w = w as usize;
                let t = ((rng.gen::<f64>() * k as f64) as usize).min(k - 1);
                z[l][d][n] = t;
                n_dk[d][t] += 1;
                wk[l][w][t] += 1;
                mk[l][t] += 1;
            }
        }
    }

    // Asymmetric prior αm, initialised symmetric (α_init per topic, uniform m).
    let mut alpha = vec![alpha_init; k];
    let mut alpha_sum = alpha_init * k as f64;
    let vbeta: Vec<f64> = (0..num_langs)
        .map(|l| vocab_sizes[l] as f64 * beta[l])
        .collect();

    let mut p = vec![0.0f64; k];
    for it in 0..iters {
        for l in 0..num_langs {
            let beta_l = beta[l];
            let vbeta_l = vbeta[l];
            for d in 0..num_docs {
                for n in 0..docs[l][d].len() {
                    let w = docs[l][d][n] as usize;
                    let z_old = z[l][d][n];
                    n_dk[d][z_old] -= 1;
                    wk[l][w][z_old] -= 1;
                    mk[l][z_old] -= 1;

                    let wk_lw = &wk[l][w];
                    for t in 0..k {
                        p[t] = (n_dk[d][t] as f64 + alpha[t]) * (wk_lw[t] as f64 + beta_l)
                            / (mk[l][t] as f64 + vbeta_l);
                    }
                    let z_new = mult_sample(&p, rng);
                    z[l][d][n] = z_new;
                    n_dk[d][z_new] += 1;
                    wk[l][w][z_new] += 1;
                    mk[l][z_new] += 1;
                }
            }
        }
        // Hold αm fixed through the burn-in, then re-estimate on the schedule
        // (MALLET's default optimize-burn-in is 200; optimizing from iteration 0
        // over-sparsifies and can starve a topic into a merge).
        if optimize_alpha
            && optimize_interval > 0
            && it + 1 > optimize_burn_in
            && (it + 1) % optimize_interval == 0
        {
            optimize_alpha_step(&mut alpha, &mut alpha_sum, &n_dk, k);
        }
    }

    // φˡ_{t,w} = (wk[l][w][t] + βˡ) / (mk[l][t] + V_l βˡ). Row t sums to 1.
    let topic_word: Vec<Vec<Vec<f64>>> = (0..num_langs)
        .map(|l| {
            let beta_l = beta[l];
            let v = vocab_sizes[l];
            (0..k)
                .map(|t| {
                    let denom = mk[l][t] as f64 + vbeta[l];
                    (0..v)
                        .map(|w| (wk[l][w][t] as f64 + beta_l) / denom)
                        .collect()
                })
                .collect()
        })
        .collect();

    // θ_{d,t} = (n_dk[d][t] + α_t) / (tuple_len_d + α_sum).
    let doc_topic: Vec<Vec<f64>> = (0..num_docs)
        .map(|d| {
            let len: i64 = n_dk[d].iter().sum();
            let denom = len as f64 + alpha_sum;
            (0..k).map(|t| (n_dk[d][t] as f64 + alpha[t]) / denom).collect()
        })
        .collect();

    PltmModel {
        num_topics: k,
        vocab_sizes: vocab_sizes.to_vec(),
        alpha,
        alpha_sum,
        beta: beta.to_vec(),
        topic_word,
        doc_topic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Planted aligned multilingual corpus: `k` topics, each a block of `block`
    /// words. Each language has its own independent vocabulary of `k * block`
    /// ids laid out in the same block structure, so aligned topic `t` peaks on
    /// block `t` in every language. Each tuple is assigned one topic; every
    /// language draws `dlen` tokens from that topic's block.
    fn planted(
        k: usize,
        block: usize,
        num_langs: usize,
        ndocs: usize,
        dlen: usize,
        seed: u64,
    ) -> (Vec<Vec<Vec<u32>>>, Vec<usize>) {
        let v_per_lang = k * block;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut docs: Vec<Vec<Vec<u32>>> = vec![Vec::with_capacity(ndocs); num_langs];
        for d in 0..ndocs {
            let topic = d % k;
            for lang_docs in docs.iter_mut() {
                let base = topic * block; // per-language id space starts at 0
                let doc: Vec<u32> = (0..dlen)
                    .map(|_| (base + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect();
                lang_docs.push(doc);
            }
        }
        let vocab_sizes = vec![v_per_lang; num_langs];
        (docs, vocab_sizes)
    }

    #[test]
    fn test_pltm_recovers_and_conforms() {
        let (k, block, langs) = (3, 5, 3);
        let (docs, vocab) = planted(k, block, langs, 180, 8, 42);
        let beta = vec![0.01; langs];
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_pltm(&docs, k, &vocab, 0.1, &beta, 300, true, 10, 50, &mut rng);

        assert_eq!(m.topic_word.len(), langs);
        for l in 0..langs {
            assert_eq!(m.topic_word[l].len(), k);
            assert_eq!(m.topic_word[l][0].len(), vocab[l]);
            for row in &m.topic_word[l] {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
        }
        assert_eq!(m.doc_topic.len(), docs[0].len());
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }

        // Each language should recover all planted blocks, and -- the PLTM
        // property -- the SAME topic index should peak on the SAME block across
        // languages. Build, per topic, the peaked block in each language; they
        // must agree.
        let peak_block = |l: usize, t: usize| -> usize {
            let row = &m.topic_word[l][t];
            let top = (0..vocab[l]).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            top / block
        };
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let b0 = peak_block(0, t);
            for l in 1..langs {
                assert_eq!(peak_block(l, t), b0, "topic {t} misaligned across languages");
            }
            covered.insert(b0);
        }
        assert_eq!(covered.len(), k, "topics should cover all planted blocks");
    }

    #[test]
    fn test_pltm_determinism() {
        let (k, block, langs) = (2, 4, 2);
        let (docs, vocab) = planted(k, block, langs, 60, 5, 123);
        let beta = vec![0.01; langs];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            fit_pltm(&docs, k, &vocab, 0.1, &beta, 120, true, 10, 30, &mut rng)
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.alpha, b.alpha);
    }

    #[test]
    fn test_pltm_shared_theta_single_language_reduces_to_lda() {
        // With one language, PLTM is exactly LDA; θ rows must be valid simplices
        // and topics must separate the planted blocks.
        let (k, block) = (3, 5);
        let (docs, vocab) = planted(k, block, 1, 120, 8, 1);
        let beta = vec![0.01];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let m = fit_pltm(&docs, k, &vocab, 0.1, &beta, 200, false, 10, 0, &mut rng);
        assert_eq!(m.topic_word.len(), 1);
        let mut covered = std::collections::HashSet::new();
        for row in &m.topic_word[0] {
            let top = (0..vocab[0]).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            covered.insert(top / block);
        }
        assert_eq!(covered.len(), k);
    }
}
