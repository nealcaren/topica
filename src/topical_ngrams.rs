//! Topical N-Grams (Wang, McCallum & Wei 2007): an LDA extension that jointly
//! discovers topics and topic-specific multiword expressions. Each token carries a
//! topic `z` and a binary bigram-status `x` (does it continue a phrase from the
//! previous token, conditional on the previous word and its topic). A phrase is a
//! head token followed by its run of `x=1` continuations, so the same word can join
//! different phrases in different topical contexts. Distinct from
//! `learn_phrases()`/`apply_phrases()`, which fix a phrase vocabulary *before*
//! fitting; TNG learns phrase structure *during* fitting.
//!
//! Faithful to MALLET `cc.mallet.topics.TopicalNGrams` (CPL, read for algorithmic
//! understanding only; implemented from the paper's equations, MALLET used only as
//! a runnable parity oracle). Collapsed Gibbs sampling jointly draws `(z_i, x_i)`.
//! `z_i` is drawn freely (NOT tied to `z_{i-1}`); the status `x_i` has a prior keyed
//! by the previous word and previous topic. The word is drawn from the unigram
//! topic-word distribution `phi` (x=0) or the topic-and-previous-word bigram
//! distribution `sigma` (x=1). Deterministic from a fixed seed (single-threaded,
//! fixed visitation order).
//!
//! Documented deviations (Gate A): the bigram-status prior defaults to a balanced
//! `delta1 = delta2 = 1.0` rather than MALLET's `0.2 / 1000` (empirically the tamer
//! prior recovers real collocations while MALLET's default forces ~all tokens into
//! whole-document "phrases"); and the phrase extraction is a clean head-inclusive
//! maximal-run definition, not MALLET's per-topic overlapping re-scan (topic-word
//! parity is the validation target, not the phrase list).

use rand::Rng;
use std::collections::HashMap;

use crate::estimator::{Estimator, ModelFamily};

/// One token: its vocabulary id, and whether a bigram with the immediately
/// preceding (surviving) token is eligible — false at a document start or when a
/// pruned/OOV token broke the adjacency in the raw text.
#[derive(Clone, Copy)]
pub struct Token {
    pub word: u32,
    pub eligible: bool,
}

/// A fitted Topical N-Grams model.
pub struct TopicalNGramsModel {
    pub num_topics: usize,
    pub num_types: usize,
    /// Unigram topic-word phi (K, V): `(unigram_count + beta)/(unigram_tokens + V·beta)`.
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic theta (D, K): `(n_dk + alpha)/(len_d + K·alpha)`.
    pub doc_topic: Vec<Vec<f64>>,
    /// Per-token topic assignment (D docs, in token order).
    pub token_topic: Vec<Vec<usize>>,
    /// Per-token bigram status 0/1 (D docs, in token order).
    pub token_gram: Vec<Vec<u8>>,
    pub doc_lengths: Vec<usize>,
    pub alpha: f64,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// Fit Topical N-Grams by collapsed Gibbs. `docs` are token sequences (order
/// matters) with per-token bigram eligibility. `num_types` is the unigram vocab
/// size. `alpha` is the per-topic doc-topic prior (`alpha_sum / K`); `beta` the
/// unigram topic-word prior; `gamma` the bigram topic-word prior; `delta1`/`delta2`
/// the Beta pseudocounts for unigram/bigram status.
#[allow(clippy::too_many_arguments)]
pub fn fit_tng<R: Rng>(
    docs: &[Vec<Token>],
    num_types: usize,
    num_topics: usize,
    iters: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta1: f64,
    delta2: f64,
    rng: &mut R,
) -> TopicalNGramsModel {
    let k = num_topics;
    let v = num_types;
    let d = docs.len();
    let v_beta = beta * v as f64;
    let v_gamma = gamma * v as f64;

    // Observed-bigram index over eligible (prev_word, word) pairs. The sigma
    // denominator normalizes over V (unigram vocab), NOT this map's size.
    let mut bimap: HashMap<(u32, u32), usize> = HashMap::new();
    for doc in docs {
        for i in 1..doc.len() {
            if doc[i].eligible {
                let key = (doc[i - 1].word, doc[i].word);
                let next = bimap.len();
                bimap.entry(key).or_insert(next);
            }
        }
    }
    let num_bi = bimap.len();

    // Count tables (i64 so removals can't underflow silently).
    let mut ndz = vec![vec![0i64; k]; d]; // doc-topic
    let mut unitype = vec![vec![0i64; k]; v]; // unigram word-topic
    let mut tokens_per_topic = vec![0i64; k];
    let mut bitype_topic = vec![vec![0i64; k]; num_bi]; // bigram (prev,word)-topic
    let mut bitokens = vec![vec![0i64; k]; v]; // bigram tokens by prev word, topic
                                               // status[prev_word][gram][prev_topic]: how often, after prev_word in prev_topic,
                                               // the next token had status gram. Drives the delta1/delta2 status prior.
    let mut status = vec![[vec![0i64; k], vec![0i64; k]]; v];

    let mut token_topic: Vec<Vec<usize>> = docs.iter().map(|doc| vec![0usize; doc.len()]).collect();
    let mut token_gram: Vec<Vec<u8>> = docs.iter().map(|doc| vec![0u8; doc.len()]).collect();

    // Random init: uniform topic; random status where eligible, else 0.
    for (di, doc) in docs.iter().enumerate() {
        for i in 0..doc.len() {
            let topic = rng.gen_range(0..k);
            let gram: u8 = if doc[i].eligible && rng.gen::<bool>() {
                1
            } else {
                0
            };
            token_topic[di][i] = topic;
            token_gram[di][i] = gram;
            ndz[di][topic] += 1;
            let w = doc[i].word as usize;
            if i > 0 {
                // outgoing status contribution of the previous token (its role as
                // the predecessor of this one), keyed by prev word + prev topic.
                let prev_w = doc[i - 1].word as usize;
                let prev_t = token_topic[di][i - 1];
                status[prev_w][gram as usize][prev_t] += 1;
            }
            if gram == 0 {
                unitype[w][topic] += 1;
                tokens_per_topic[topic] += 1;
            } else {
                let bi = bimap[&(doc[i - 1].word, doc[i].word)];
                bitype_topic[bi][topic] += 1;
                bitokens[doc[i - 1].word as usize][topic] += 1;
            }
        }
    }

    let mut weights = vec![0.0f64; 2 * k];
    let mut uni_weights = vec![0.0f64; k];

    for _it in 0..iters {
        for (di, doc) in docs.iter().enumerate() {
            let len = doc.len();
            for i in 0..len {
                let w = doc[i].word as usize;
                let old_topic = token_topic[di][i];
                let old_gram = token_gram[di][i];
                let next_gram = if i + 1 < len {
                    token_gram[di][i + 1] as usize
                } else {
                    0
                };

                if !doc[i].eligible {
                    // --- unigram-only position (doc start or broken adjacency) ---
                    ndz[di][old_topic] -= 1;
                    unitype[w][old_topic] -= 1;
                    tokens_per_topic[old_topic] -= 1;
                    if i + 1 < len {
                        status[w][next_gram][old_topic] -= 1;
                    }
                    let mut sum = 0.0;
                    for ti in 0..k {
                        let tw = (unitype[w][ti] as f64 + beta)
                            / (tokens_per_topic[ti] as f64 + v_beta)
                            * (ndz[di][ti] as f64 + alpha);
                        uni_weights[ti] = tw;
                        sum += tw;
                    }
                    let new_topic = sample(&uni_weights, k, sum, rng);
                    token_topic[di][i] = new_topic;
                    token_gram[di][i] = 0;
                    ndz[di][new_topic] += 1;
                    unitype[w][new_topic] += 1;
                    tokens_per_topic[new_topic] += 1;
                    if i + 1 < len {
                        status[w][next_gram][new_topic] += 1;
                    }
                } else {
                    // --- bigram-eligible position: joint (topic, status) sample ---
                    let prev_w = doc[i - 1].word as usize;
                    let prev_t = token_topic[di][i - 1];
                    let bi = bimap[&(doc[i - 1].word, doc[i].word)];
                    // remove this token from all counts
                    ndz[di][old_topic] -= 1;
                    status[prev_w][old_gram as usize][prev_t] -= 1; // incoming
                    if i + 1 < len {
                        status[w][next_gram][old_topic] -= 1; // outgoing
                    }
                    if old_gram == 0 {
                        unitype[w][old_topic] -= 1;
                        tokens_per_topic[old_topic] -= 1;
                    } else {
                        bitype_topic[bi][old_topic] -= 1;
                        bitokens[prev_w][old_topic] -= 1;
                    }
                    // The status prior terms are constant across candidate topic ti
                    // (they depend only on prev word + prev topic).
                    let s0 = status[prev_w][0][prev_t] as f64 + delta1;
                    let s1 = status[prev_w][1][prev_t] as f64 + delta2;
                    let mut sum = 0.0;
                    for ti in 0..k {
                        let dt = ndz[di][ti] as f64 + alpha;
                        let uni = (unitype[w][ti] as f64 + beta)
                            / (tokens_per_topic[ti] as f64 + v_beta)
                            * dt
                            * s0;
                        let big = (bitype_topic[bi][ti] as f64 + gamma)
                            / (bitokens[prev_w][ti] as f64 + v_gamma)
                            * dt
                            * s1;
                        weights[2 * ti] = uni;
                        weights[2 * ti + 1] = big;
                        sum += uni + big;
                    }
                    let idx = sample(&weights, 2 * k, sum, rng);
                    let new_topic = idx / 2;
                    let new_gram = (idx % 2) as u8;
                    token_topic[di][i] = new_topic;
                    token_gram[di][i] = new_gram;
                    ndz[di][new_topic] += 1;
                    status[prev_w][new_gram as usize][prev_t] += 1;
                    if i + 1 < len {
                        status[w][next_gram][new_topic] += 1;
                    }
                    if new_gram == 0 {
                        unitype[w][new_topic] += 1;
                        tokens_per_topic[new_topic] += 1;
                    } else {
                        bitype_topic[bi][new_topic] += 1;
                        bitokens[prev_w][new_topic] += 1;
                    }
                }
            }
        }
    }

    // Unigram topic-word phi (K, V): a proper simplex per topic.
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|ti| {
            let denom = tokens_per_topic[ti] as f64 + v_beta;
            (0..v)
                .map(|w| (unitype[w][ti] as f64 + beta) / denom)
                .collect()
        })
        .collect();

    // Document-topic theta (D, K): smoothed posterior mean, rows sum to 1.
    let k_alpha = alpha * k as f64;
    let doc_lengths: Vec<usize> = docs.iter().map(|doc| doc.len()).collect();
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let denom = doc_lengths[di] as f64 + k_alpha;
            (0..k)
                .map(|ti| (ndz[di][ti] as f64 + alpha) / denom)
                .collect()
        })
        .collect();

    TopicalNGramsModel {
        num_topics: k,
        num_types: v,
        topic_word,
        doc_topic,
        token_topic,
        token_gram,
        doc_lengths,
        alpha,
        fit_history: Vec::new(),
        converged: false,
    }
}

/// Sample an index in `0..n` proportional to `weights[0..n]` (already summing to
/// `sum`), using one uniform draw. Deterministic given the RNG stream.
fn sample<R: Rng>(weights: &[f64], n: usize, sum: f64, rng: &mut R) -> usize {
    let mut r = rng.gen::<f64>() * sum;
    for (i, &w) in weights.iter().enumerate().take(n) {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    n - 1
}

impl Estimator for TopicalNGramsModel {
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
        // Like TopicsOverTime / AuthorTopic / GaussianLDA (collapsed-Gibbs models
        // that do not expose the full transform / theta-draws surface): None_ so the
        // conformance suite does not require Tier-2 Dirichlet posterior machinery.
        ModelFamily::None_
    }
}

/// A phrase and its count, for `top_phrases`.
pub struct Phrase {
    pub words: Vec<u32>,
    pub topic: usize,
    pub count: usize,
}

/// Extract phrases: each is a head token followed by its maximal run of `x=1`
/// continuations, attributed to the terminal token's topic. Returns per-topic
/// phrase counts keyed by the word-id sequence. Deterministic.
pub fn extract_phrases(model: &TopicalNGramsModel, docs: &[Vec<Token>]) -> Vec<Phrase> {
    let mut counts: HashMap<(usize, Vec<u32>), usize> = HashMap::new();
    for (di, doc) in docs.iter().enumerate() {
        let grams = &model.token_gram[di];
        let topics = &model.token_topic[di];
        let mut i = 0;
        while i < doc.len() {
            // A phrase begins at head i and extends while the NEXT token is x=1.
            let mut j = i + 1;
            while j < doc.len() && grams[j] == 1 {
                j += 1;
            }
            if j > i + 1 {
                // tokens i..j form a phrase (head i + continuations); terminal j-1.
                let words: Vec<u32> = (i..j).map(|p| doc[p].word).collect();
                let topic = topics[j - 1];
                *counts.entry((topic, words)).or_insert(0) += 1;
                i = j;
            } else {
                i += 1;
            }
        }
    }
    let mut phrases: Vec<Phrase> = counts
        .into_iter()
        .map(|((topic, words), count)| Phrase {
            words,
            topic,
            count,
        })
        .collect();
    // Deterministic order: count desc, then word-id sequence asc.
    phrases.sort_by(|a, b| b.count.cmp(&a.count).then(a.words.cmp(&b.words)));
    phrases
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a corpus with planted collocations: each doc mixes filler unigrams and
    /// two-word collocations. TNG should recover the collocations as phrases.
    fn planted(seed: u64, n_docs: usize) -> (Vec<Vec<Token>>, usize, Vec<(u32, u32)>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // vocab: 0..8 collocation words (4 pairs), 8..18 filler
        let collocs = vec![(0u32, 1u32), (2, 3), (4, 5), (6, 7)];
        let filler: Vec<u32> = (8..18).collect();
        let v = 18;
        let mut docs = Vec::new();
        for _ in 0..n_docs {
            let mut toks: Vec<Token> = Vec::new();
            let mut prev_present = false;
            let len = 20 + (rng.gen::<u32>() % 20) as usize;
            for _ in 0..len {
                if rng.gen::<f64>() < 0.5 {
                    let c = collocs[(rng.gen::<u32>() as usize) % collocs.len()];
                    toks.push(Token {
                        word: c.0,
                        eligible: prev_present,
                    });
                    toks.push(Token {
                        word: c.1,
                        eligible: true,
                    });
                    prev_present = true;
                } else {
                    let f = filler[(rng.gen::<u32>() as usize) % filler.len()];
                    toks.push(Token {
                        word: f,
                        eligible: prev_present,
                    });
                    prev_present = true;
                }
            }
            docs.push(toks);
        }
        (docs, v, collocs)
    }

    #[test]
    fn recovers_planted_collocations() {
        let (docs, v, collocs) = planted(1, 200);
        let model = fit_tng(
            &docs,
            v,
            4,
            300,
            12.5,
            0.01,
            0.01,
            1.0,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(1),
        );
        let phrases = extract_phrases(&model, &docs);
        // The planted 2-word collocations should be among the top phrases.
        let top: std::collections::HashSet<Vec<u32>> = phrases
            .iter()
            .take(12)
            .filter(|p| p.words.len() == 2)
            .map(|p| p.words.clone())
            .collect();
        let mut found = 0;
        for c in &collocs {
            if top.contains(&vec![c.0, c.1]) {
                found += 1;
            }
        }
        assert!(found >= 3, "recovered only {found}/4 planted collocations");
    }

    #[test]
    fn deterministic() {
        let (docs, v, _) = planted(2, 60);
        let a = fit_tng(
            &docs,
            v,
            4,
            50,
            12.5,
            0.01,
            0.01,
            1.0,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(7),
        );
        let b = fit_tng(
            &docs,
            v,
            4,
            50,
            12.5,
            0.01,
            0.01,
            1.0,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(7),
        );
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.token_gram, b.token_gram);
        assert_eq!(a.token_topic, b.token_topic);
    }

    #[test]
    fn topic_word_is_simplex() {
        let (docs, v, _) = planted(3, 40);
        let m = fit_tng(
            &docs,
            v,
            3,
            40,
            16.6,
            0.01,
            0.01,
            1.0,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        for row in &m.topic_word {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "topic-word row sums to {s}");
        }
        for row in &m.doc_topic {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "doc-topic row sums to {s}");
        }
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    #[test]
    fn ineligible_first_token_forced_unigram() {
        // First token of every doc must be x=0 (no predecessor).
        let (docs, v, _) = planted(4, 20);
        let m = fit_tng(
            &docs,
            v,
            3,
            30,
            16.6,
            0.01,
            0.01,
            1.0,
            1.0,
            &mut ChaCha8Rng::seed_from_u64(2),
        );
        for grams in &m.token_gram {
            assert_eq!(grams[0], 0, "first token must be a unigram");
        }
    }
}
