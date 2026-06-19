//! Seeded LDA — guided topic modeling with asymmetric word-topic priors.
//!
//! Jagarlamudi, Daumé III & Udupa (2012), "Incorporating Lexical Priors into
//! Topic Models", EACL 2012; Watanabe & Sumita (2015).
//!
//! Standard collapsed-Gibbs LDA, except the topic-word Dirichlet prior is
//! **asymmetric**: a *seed word* for topic k receives an extra prior
//! pseudocount of `seed_weight` in that topic, encouraging the topic to form
//! around its seeds.  Topics with no seeds behave as ordinary LDA topics.
//!
//! ## Prior parameterisation
//!
//! For topic k and word w:
//! ```text
//! β_{k,w} = β  +  (seed_weight  if  w ∈ seeds[k]  else  0)
//! β_sum[k] = V·β  +  |seeds[k]|·seed_weight
//! ```
//!
//! ## Algorithm (per-token collapsed Gibbs)
//!
//! Each sweep, for every token (d, i) with word w and current topic z:
//! 1. Remove the token: decrement ndk[d][z], nkw[z][w], nk[z].
//! 2. For each topic t:
//!    `score(t) = (α + ndk[d][t]) × (β_{t,w} + nkw[t][w]) / (β_sum[t] + nk[t])`
//! 3. Sample a new topic proportionally; increment counts.

use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use rand::Rng;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Fitted Seeded-LDA model.
///
/// Stores the final Gibbs state together with the prior information needed to
/// compute normalised φ and θ matrices.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SeededModel {
    /// Number of topics K.
    pub num_topics: usize,
    /// Vocabulary size V.
    pub num_types: usize,
    /// Symmetric per-topic document prior α.
    pub alpha: f64,
    /// Base topic-word smoothing scalar β.
    pub beta: f64,
    /// Extra pseudocount added to seed words in their seed topic.
    pub seed_weight: f64,
    /// `seeds[k]` — set of seed word-ids for topic k (sorted, de-duped).
    pub seeds: Vec<Vec<usize>>,
    /// `nkw[k][w]` — count of word type w assigned to topic k.  Shape: K × V.
    pub nkw: Vec<Vec<u32>>,
    /// `nk[k]` — total token count in topic k.  Length: K.
    pub nk: Vec<u32>,
    /// `ndk[d][k]` — count of tokens in doc d assigned to topic k.  Shape: D × K.
    pub ndk: Vec<Vec<u32>>,
    /// Optional per-document, per-topic Dirichlet prior `α_{d,k}` (D × K). When
    /// `Some`, it replaces the symmetric `alpha` in both sampling and `θ` — the
    /// vehicle for a document-level prior (e.g. embedding-anchored topic
    /// prevalence). `None` for the ordinary symmetric-α model.
    #[serde(default)]
    pub doc_alpha: Option<Vec<Vec<f64>>>,
    /// Thinned MCMC θ snapshots (issue #31): the last `num_theta_draws` per-doc
    /// topic distributions taken every `thin` sweeps, f32. Real cross-sweep
    /// posterior draws that `composition_theta` prefers over the within-document
    /// Dirichlet approximation. A fit-time artifact, not persisted.
    #[serde(skip)]
    pub theta_draws: Vec<Vec<Vec<f32>>>,
}

impl SeededModel {
    // -----------------------------------------------------------------------
    // Prior helpers
    // -----------------------------------------------------------------------

    /// β_{k,w}: base prior plus the seed boost (if word w seeds topic k).
    #[inline]
    fn beta_kw(&self, k: usize, w: usize) -> f64 {
        // seeds[k] is small (typically 0–10 entries); linear scan is fine.
        if self.seeds[k].contains(&w) {
            self.beta + self.seed_weight
        } else {
            self.beta
        }
    }

    /// β_sum[k]: sum of β_{k,w} over the full vocabulary V.
    #[inline]
    fn beta_sum(&self, k: usize) -> f64 {
        self.num_types as f64 * self.beta + self.seeds[k].len() as f64 * self.seed_weight
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Collapsed Gibbs log-likelihood (same formula as LDA / MALLET) using the
    /// seeded asymmetric β_{k,w} prior.  Cheap to call; does not allocate.
    pub fn log_likelihood(&self, docs: &[Vec<u32>]) -> f64 {
        use crate::optimize::digamma;
        let k = self.num_topics;
        let v = self.num_types;
        let mut ll = 0.0f64;

        // Document-topic contribution.
        let k_alpha = k as f64 * self.alpha;
        for (d, doc) in docs.iter().enumerate() {
            let n_d = doc.len() as f64;
            for t in 0..k {
                let n_dt = self.ndk[d][t] as f64;
                if n_dt > 0.0 {
                    ll += digamma(self.alpha + n_dt) - digamma(self.alpha);
                }
            }
            ll -= digamma(k_alpha + n_d) - digamma(k_alpha);
        }

        // Topic-word contribution.
        for t in 0..k {
            let beta_sum_t = v as f64 * self.beta + self.seeds[t].len() as f64 * self.seed_weight;
            for w in 0..v {
                let n_tw = self.nkw[t][w] as f64;
                if n_tw > 0.0 {
                    let beta_tw = if self.seeds[t].contains(&w) {
                        self.beta + self.seed_weight
                    } else {
                        self.beta
                    };
                    ll += digamma(beta_tw + n_tw) - digamma(beta_tw);
                }
            }
            ll -= digamma(beta_sum_t + self.nk[t] as f64) - digamma(beta_sum_t);
        }

        ll
    }

    /// Row-normalised topic-word distribution φ for topic k.
    ///
    /// φ_{k,w} = (nkw[k][w] + β_{k,w}) / (nk[k] + β_sum[k])
    ///
    /// Length = `num_types`; row sums to 1.
    pub fn topic_word(&self, k: usize) -> Vec<f64> {
        let denom = self.nk[k] as f64 + self.beta_sum(k);
        (0..self.num_types)
            .map(|w| (self.nkw[k][w] as f64 + self.beta_kw(k, w)) / denom)
            .collect()
    }

    /// All K topic-word rows as a K × V matrix.
    pub fn topic_word_all(&self) -> Vec<Vec<f64>> {
        (0..self.num_topics).map(|k| self.topic_word(k)).collect()
    }

    /// Document-topic distribution θ for all documents.
    ///
    /// θ_{d,k} = (ndk[d][k] + α) / (N_d + K·α)
    ///
    /// Shape: D × K; each row sums to 1.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        if let Some(da) = &self.doc_alpha {
            return self
                .ndk
                .iter()
                .zip(da)
                .map(|(row, a)| {
                    let n_d: u32 = row.iter().sum();
                    let a_sum: f64 = a.iter().sum();
                    let denom = n_d as f64 + a_sum;
                    row.iter()
                        .zip(a)
                        .map(|(&c, &av)| (c as f64 + av) / denom)
                        .collect()
                })
                .collect();
        }
        let k_alpha = k as f64 * self.alpha;
        self.ndk
            .iter()
            .map(|row| {
                let n_d: u32 = row.iter().sum();
                let denom = n_d as f64 + k_alpha;
                row.iter()
                    .map(|&c| (c as f64 + self.alpha) / denom)
                    .collect()
            })
            .collect()
    }
}

impl Estimator for SeededModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word_all()
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic()
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }
    fn converged(&self) -> Option<bool> {
        None
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for SeededModel {
    fn alpha(&self) -> Vec<f64> {
        vec![self.alpha; self.num_topics]
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        self.theta_draws
            .iter()
            .map(|d| {
                d.iter()
                    .map(|r| r.iter().map(|&x| x as f64).collect())
                    .collect()
            })
            .collect()
    }
    fn doc_lengths(&self) -> Vec<usize> {
        self.ndk
            .iter()
            .map(|r| r.iter().map(|&c| c as usize).sum())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sampler internals
// ---------------------------------------------------------------------------

/// Weighted categorical sample; `scores` need not be normalised.
///
/// Identical implementation to `gsdmm::sample_index`.
#[inline]
fn sample_index<R: Rng>(scores: &[f64], rng: &mut R) -> usize {
    let total: f64 = scores.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, &s) in scores.iter().enumerate() {
        r -= s;
        if r <= 0.0 {
            return i;
        }
    }
    scores.len() - 1
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

/// One smoothed θ snapshot (D×K) as f32 from the current counts:
/// θ_{d,k} = (n_dk + α_dk) / (N_d + Σα_d), using the per-document prior when
/// present and the symmetric `alpha` otherwise.
fn seeded_theta_snapshot(
    ndk: &[Vec<u32>],
    doc_alpha: Option<&Vec<Vec<f64>>>,
    alpha: f64,
    k: usize,
) -> Vec<Vec<f32>> {
    ndk.iter()
        .enumerate()
        .map(|(d, row)| {
            let n_d: u32 = row.iter().sum();
            match doc_alpha {
                Some(da) => {
                    let a = &da[d];
                    let denom = n_d as f64 + a.iter().sum::<f64>();
                    row.iter()
                        .zip(a)
                        .map(|(&c, &av)| ((c as f64 + av) / denom) as f32)
                        .collect()
                }
                None => {
                    let denom = n_d as f64 + k as f64 * alpha;
                    row.iter()
                        .map(|&c| ((c as f64 + alpha) / denom) as f32)
                        .collect()
                }
            }
        })
        .collect()
}

/// Fit a Seeded-LDA model by collapsed Gibbs sampling.
///
/// # Arguments
/// * `docs`        — corpus; each document is a slice of word ids in `0..num_types`.
/// * `num_types`   — vocabulary size V.
/// * `num_topics`  — number of topics K.
/// * `seeds`       — `seeds[k]` is the list of seed word-ids for topic k.
///                   Must have length K; entries must be valid ids `< num_types`.
///                   Empty slices are allowed (unseeded / residual topics).
/// * `alpha`       — symmetric per-topic document-topic prior (α).
/// * `beta`        — base topic-word Dirichlet smoothing scalar (β).
/// * `seed_weight` — extra pseudocount added to a seed word in its topic.
///                   Set to 0.0 for ordinary LDA with symmetric priors.
/// * `iters`       — number of full Gibbs sweeps.
/// * `draws`            — thinned θ-draw retention schedule (issue #31).
/// * `convergence_tol`  — relative-change tolerance for early stopping (0 = off).
/// * `check_every`      — LL-trace cadence in sweeps (0 = no trace).
/// * `rng`              — random-number source; deterministic for a fixed seed.
///
/// Returns `(model, ll_history, converged)` where `ll_history` is a vector of
/// `(iteration, log_likelihood)` pairs recorded every `check_every` sweeps.
#[allow(clippy::too_many_arguments)]
pub fn fit_seeded_lda<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    seeds: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    seed_weight: f64,
    doc_alpha: Option<Vec<Vec<f64>>>,
    iters: usize,
    draws: crate::keyatm::ThetaDrawOpts,
    convergence_tol: f64,
    check_every: usize,
    rng: &mut R,
) -> (SeededModel, Vec<(usize, f64)>, bool) {
    let k = num_topics;
    let v = num_types;
    let d_count = docs.len();
    if let Some(da) = &doc_alpha {
        assert_eq!(
            da.len(),
            d_count,
            "doc_alpha must have one row per document"
        );
        assert!(
            da.iter().all(|r| r.len() == k),
            "each doc_alpha row must have num_topics entries"
        );
    }

    // Normalise seeds: sort and de-dup so `contains` scans are minimal.
    let seeds_clean: Vec<Vec<usize>> = seeds
        .iter()
        .map(|sv| {
            let mut s = sv.clone();
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect();

    // Precompute per-word, which topics seed it: word_seed_topics[w] is a
    // sorted list of topic indices k such that w ∈ seeds[k].
    // Used only during the initial β_kw / beta_sum lookups in the sweep; the
    // hot path uses `seeds_clean[k].contains(&w)` which is O(|seeds[k]|) — fast
    // because seed lists are tiny.  The HashSet approach below provides O(1)
    // membership tests when iterating over all K topics for each token.
    let seed_sets: Vec<HashSet<usize>> = seeds_clean
        .iter()
        .map(|sv| sv.iter().cloned().collect())
        .collect();

    // Precompute β_sum[k] once (it is constant after init).
    let beta_sum_k: Vec<f64> = (0..k)
        .map(|kk| v as f64 * beta + seeds_clean[kk].len() as f64 * seed_weight)
        .collect();

    // For each word, which topics seed it (used for seeded initialisation).
    let mut word_seed_topics: Vec<Vec<usize>> = vec![Vec::new(); v];
    for (kk, sv) in seeds_clean.iter().enumerate() {
        for &w in sv {
            word_seed_topics[w].push(kk);
        }
    }

    // --- Initialise. A token whose word seeds some topic starts in that topic
    // (random among ties); other tokens start in a uniformly random topic. This
    // seeded initialisation is what propagates the seeds: documents that contain
    // seed words begin with mass on the seeded topic, pulling their co-occurring
    // words along. A weak β prior alone does not do this. ---
    let mut nkw: Vec<Vec<u32>> = vec![vec![0u32; v]; k];
    let mut nk: Vec<u32> = vec![0u32; k];
    let mut ndk: Vec<Vec<u32>> = vec![vec![0u32; k]; d_count];
    // z[d][i] = current topic of token i in document d.
    let mut z: Vec<Vec<usize>> = docs
        .iter()
        .map(|doc| {
            doc.iter()
                .map(|&w| {
                    let cands = &word_seed_topics[w as usize];
                    if cands.is_empty() {
                        (rng.gen::<f64>() * k as f64) as usize % k
                    } else if cands.len() == 1 {
                        cands[0]
                    } else {
                        cands[(rng.gen::<f64>() * cands.len() as f64) as usize % cands.len()]
                    }
                })
                .collect()
        })
        .collect();

    for (d, doc) in docs.iter().enumerate() {
        for (i, &w) in doc.iter().enumerate() {
            let t = z[d][i];
            nkw[t][w as usize] += 1;
            nk[t] += 1;
            ndk[d][t] += 1;
        }
    }

    // --- Gibbs sweeps ---
    let mut scores: Vec<f64> = vec![0.0f64; k];
    let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut ll_history: Vec<(usize, f64)> = Vec::new();
    let mut converged = false;

    // Build a temporary SeededModel view for LL computation (borrows nkw/nk/ndk).
    // We compute LL inline using the same formula as SeededModel::log_likelihood.
    let compute_ll = |nkw: &[Vec<u32>], nk: &[u32], ndk: &[Vec<u32>]| -> f64 {
        use crate::optimize::digamma;
        let k_alpha = k as f64 * alpha;
        let mut ll = 0.0f64;
        for (d, doc) in docs.iter().enumerate() {
            let n_d = doc.len() as f64;
            for t in 0..k {
                let n_dt = ndk[d][t] as f64;
                if n_dt > 0.0 {
                    ll += digamma(alpha + n_dt) - digamma(alpha);
                }
            }
            ll -= digamma(k_alpha + n_d) - digamma(k_alpha);
        }
        for t in 0..k {
            let beta_sum_t = v as f64 * beta + seeds_clean[t].len() as f64 * seed_weight;
            for w in 0..v {
                let n_tw = nkw[t][w] as f64;
                if n_tw > 0.0 {
                    let beta_tw = if seed_sets[t].contains(&w) {
                        beta + seed_weight
                    } else {
                        beta
                    };
                    ll += digamma(beta_tw + n_tw) - digamma(beta_tw);
                }
            }
            ll -= digamma(beta_sum_t + nk[t] as f64) - digamma(beta_sum_t);
        }
        ll
    };

    for it in 0..iters {
        let iter = it + 1;
        for d in 0..d_count {
            let doc = &docs[d];
            // The document's α row: per-document when supplied, else symmetric.
            let a_row: Option<&Vec<f64>> = doc_alpha.as_ref().map(|da| &da[d]);
            for i in 0..doc.len() {
                let w = doc[i] as usize;
                let old = z[d][i];

                // Remove token from counts.
                nkw[old][w] -= 1;
                nk[old] -= 1;
                ndk[d][old] -= 1;

                // Compute unnormalised sampling probabilities.
                for t in 0..k {
                    let beta_tw = if seed_sets[t].contains(&w) {
                        beta + seed_weight
                    } else {
                        beta
                    };
                    let a_t = a_row.map_or(alpha, |r| r[t]);
                    scores[t] = (a_t + ndk[d][t] as f64) * (beta_tw + nkw[t][w] as f64)
                        / (beta_sum_k[t] + nk[t] as f64);
                }

                // Sample new topic and update counts.
                let new_t = sample_index(&scores, rng);
                nkw[new_t][w] += 1;
                nk[new_t] += 1;
                ndk[d][new_t] += 1;
                z[d][i] = new_t;
            }
        }
        if draws.thin > 0 && iter % draws.thin == 0 {
            theta_draw_buf.push(seeded_theta_snapshot(&ndk, doc_alpha.as_ref(), alpha, k));
            if theta_draw_buf.len() > draws.cap {
                theta_draw_buf.remove(0);
            }
        }
        // Trace recording and optional convergence check (never alters RNG).
        if check_every > 0 && iter % check_every == 0 {
            let ll = compute_ll(&nkw, &nk, &ndk);
            ll_history.push((iter, ll));
            if convergence_tol > 0.0 && ll_history.len() >= 2 {
                let prev = ll_history[ll_history.len() - 2].1;
                let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                if rel < convergence_tol {
                    converged = true;
                    break;
                }
            }
        }
    }

    let model = SeededModel {
        num_topics: k,
        num_types: v,
        alpha,
        beta,
        seed_weight,
        seeds: seeds_clean,
        nkw,
        nk,
        ndk,
        doc_alpha,
        theta_draws: theta_draw_buf,
    };
    (model, ll_history, converged)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // -----------------------------------------------------------------------
    // Helper: build a 3-block synthetic corpus.
    //
    // Vocabulary layout (block_size words each):
    //   block 0: words  0 ..  9
    //   block 1: words 10 .. 19
    //   block 2: words 20 .. 29
    //
    // Each document draws `signal` tokens from its home block and `noise`
    // tokens from a random other word, giving a clearly separated corpus.
    // -----------------------------------------------------------------------
    fn synthetic_corpus(
        num_blocks: usize,
        block_size: usize,
        docs_per_block: usize,
        tokens_per_doc: usize,
        noise_tokens: usize,
        rng: &mut impl Rng,
    ) -> (Vec<Vec<u32>>, usize) {
        let v = num_blocks * block_size;
        let mut docs = Vec::new();
        for b in 0..num_blocks {
            let offset = (b * block_size) as u32;
            for d in 0..docs_per_block {
                let mut doc: Vec<u32> = (0..tokens_per_doc)
                    .map(|i| offset + ((i + d) % block_size) as u32)
                    .collect();
                // A few noise tokens anywhere in the vocabulary.
                for _ in 0..noise_tokens {
                    doc.push(rng.gen_range(0..v as u32));
                }
                docs.push(doc);
            }
        }
        (docs, v)
    }

    /// Seeds steer topics toward the planted vocabulary blocks.
    ///
    /// K=3 topics; topic 0 seeded with words from block A, topic 1 seeded with
    /// words from block B, topic 2 left unseeded (residual).
    ///
    /// LDA has a label-switching symmetry: topic indices can permute across
    /// runs.  Instead of asserting that topic *index* 0 covers block 0, we
    /// assert that for each seeded topic k, its dominant block equals the block
    /// that supplied its seeds — i.e., the seed-word probability mass in the
    /// seeded topic exceeds the mass in any other topic.
    #[test]
    fn seeds_steer_topics() {
        let mut setup_rng = ChaCha8Rng::seed_from_u64(7);
        let num_blocks = 3;
        let block_size = 10;
        let (docs, v) = synthetic_corpus(num_blocks, block_size, 60, 8, 2, &mut setup_rng);

        // Seed topic 0 with the first two words of block 0, topic 1 with the
        // first two words of block 1, topic 2 unseeded.
        // Use more seed words per topic so the prior strongly identifies each
        // topic with its block; seed_weight=50 gives each seed word ~50x the
        // base beta prior, which is far stronger than the random-init pull.
        let seeds = vec![
            vec![0usize, 1usize, 2usize, 3usize], // first 4 words of block 0
            vec![10usize, 11usize, 12usize, 13usize], // first 4 words of block 1
            vec![],                               // unseeded / residual
        ];
        let seed_blocks = [0usize, 1usize]; // expected dominant block for topics 0 and 1

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (model, _, _) = fit_seeded_lda(
            &docs,
            v,
            3,
            &seeds,
            0.1,
            0.01,
            50.0,
            None,
            300,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            &mut rng,
        );

        // For each seeded topic (0 and 1), the seed words' total φ mass should
        // exceed the corresponding mass in the other seeded topic.  This checks
        // that the seeded topic "owns" its seed words more than any other topic
        // does, regardless of which block ends up dominating topic 0 globally.
        for (ki, &expected_block) in seed_blocks.iter().enumerate() {
            let phi_ki = model.topic_word(ki);
            // Mass on the seed words' block in topic ki.
            let mass_ki: f64 = phi_ki
                [expected_block * block_size..(expected_block + 1) * block_size]
                .iter()
                .sum();
            // Check that no other topic has MORE mass on this block than topic ki does.
            // (Equivalently: ki is the topic most concentrated on its seed block.)
            for other in 0..3usize {
                if other == ki {
                    continue;
                }
                let phi_other = model.topic_word(other);
                let mass_other: f64 = phi_other
                    [expected_block * block_size..(expected_block + 1) * block_size]
                    .iter()
                    .sum();
                assert!(
                    mass_ki > mass_other,
                    "seeded topic {ki} (seeds on block {expected_block}) has less mass \
                     on block {expected_block} ({mass_ki:.4}) than topic {other} ({mass_other:.4}); \
                     seeds did not steer topic {ki}"
                );
            }
        }
    }

    /// With all-empty seeds and seed_weight=0 the model is ordinary LDA:
    /// topic_word rows and doc_topic rows must each sum to 1.
    #[test]
    fn unseeded_matches_plain_lda_shape() {
        let v = 30usize;
        let k = 4usize;
        let docs: Vec<Vec<u32>> = (0..100usize)
            .map(|d| (0..6).map(|i| ((i + d * 3) % v) as u32).collect())
            .collect();
        let seeds: Vec<Vec<usize>> = vec![vec![]; k];

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let (model, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            0.0,
            None,
            50,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            &mut rng,
        );

        // topic_word rows sum to 1.
        for t in 0..k {
            let phi = model.topic_word(t);
            assert_eq!(
                phi.len(),
                v,
                "topic_word({t}) length should equal num_types"
            );
            let s: f64 = phi.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "topic_word({t}) sums to {s:.12}, expected 1.0"
            );
        }

        // doc_topic rows sum to 1.
        let theta = model.doc_topic();
        assert_eq!(
            theta.len(),
            docs.len(),
            "doc_topic() row count should equal D"
        );
        for (d, row) in theta.iter().enumerate() {
            assert_eq!(row.len(), k, "doc_topic row {d} length should equal K");
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "doc_topic row {d} sums to {s:.12}, expected 1.0"
            );
        }
    }

    /// Two fits with the same RNG seed must produce bit-for-bit identical results.
    #[test]
    fn deterministic_for_fixed_seed() {
        let v = 20usize;
        let k = 3usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..5).map(|i| ((i + d * 2) % v) as u32).collect())
            .collect();
        let seeds = vec![vec![0usize, 1usize], vec![10usize, 11usize], vec![]];

        let mut r1 = ChaCha8Rng::seed_from_u64(55);
        let mut r2 = ChaCha8Rng::seed_from_u64(55);
        let (m1, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            2.0,
            None,
            80,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            &mut r1,
        );
        let (m2, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            2.0,
            None,
            80,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            &mut r2,
        );

        assert_eq!(
            m1.topic_word_all(),
            m2.topic_word_all(),
            "topic_word_all() differs between two identical-seed runs"
        );
        assert_eq!(
            m1.doc_topic(),
            m2.doc_topic(),
            "doc_topic() differs between two identical-seed runs"
        );
    }

    #[test]
    fn seeded_conforms() {
        let v = 30usize;
        let k = 3usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..5).map(|i| ((i + d * 3) % v) as u32).collect())
            .collect();
        let seeds: Vec<Vec<usize>> = vec![vec![]; k];
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let (m, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            0.0,
            None,
            20,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
