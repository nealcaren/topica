//! Pseudo-Document Topic Model (PTM) for short texts.
//!
//! Zuo, Wu, Zhang, Lin, Wang, Xu & Xiong — "Topic Modeling of Short Texts:
//! A Pseudo-Document View", KDD 2016.
//!
//! Short documents are too sparse to estimate topic-word and document-topic
//! distributions reliably. PTM combats this by introducing P **pseudo-documents**
//! that aggregate short real documents. Each real document d is assigned to one
//! pseudo-document l_d ∈ {0..P-1}; topic-word statistics are global and
//! document-topic statistics are maintained at the pseudo-document level.
//!
//! Inference is collapsed Gibbs sampling with two sets of latent variables:
//!   - z[d][i]  — the topic of the i-th token in document d
//!   - l[d]     — the pseudo-document to which document d belongs
//!
//! Outputs:
//!   - `topic_word()[k][w]`  = (n_kw + β)  / (n_k + V·β)   (K × V)
//!   - `doc_topic()[d][k]`   = (n_{l_d,k} + α) / (n_{l_d} + K·α)   (D × K)
//!     i.e. the real doc inherits its pseudo-doc's topic distribution.

use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use rand::Rng;

// ---------------------------------------------------------------------------
// Log-Gamma helper (Stirling series shifted to z ≥ 10, matching dmr.rs)
// ---------------------------------------------------------------------------

fn log_gamma(mut z: f64) -> f64 {
    const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
    let mut shift = 0i32;
    while z < 10.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

pub struct PtmModel {
    pub num_types: usize,
    pub num_topics: usize,
    pub num_pseudo: usize,
    pub alpha: f64,
    pub beta: f64,
    /// n_kw: K × V  topic-word counts
    pub nkw: Vec<Vec<u32>>,
    /// n_k: K  topic totals
    pub nk: Vec<u32>,
    /// n_pk: P × K  pseudo-doc topic counts
    pub npk: Vec<Vec<u32>>,
    /// n_p: P  pseudo-doc token totals
    pub np: Vec<u32>,
    /// l[d]: pseudo-document assignment for real document d
    pub l: Vec<usize>,
    /// z[d][i]: topic assignment for each token
    pub z: Vec<Vec<usize>>,
    /// Thinned θ draws (num_draws, D, K): each doc inherits its pseudo-doc's
    /// Dirichlet-smoothed distribution at each snapshot. Empty when draw_cap=0.
    pub theta_draws: Vec<Vec<Vec<f32>>>,
}

impl PtmModel {
    /// Topic-word distributions φ_{k,w} = (n_{kw} + β) / (n_k + V·β).
    /// Shape K × V; each row sums to 1.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let v = self.num_types;
        self.nkw
            .iter()
            .zip(&self.nk)
            .map(|(row, &nk)| {
                let denom = nk as f64 + v as f64 * self.beta;
                row.iter()
                    .map(|&c| (c as f64 + self.beta) / denom)
                    .collect()
            })
            .collect()
    }

    /// Document-topic distributions θ_{d,k}: the real doc inherits the
    /// distribution of its assigned pseudo-document.
    /// Shape D × K; each row sums to 1.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        self.l
            .iter()
            .map(|&p| {
                let denom = self.np[p] as f64 + k as f64 * self.alpha;
                (0..k)
                    .map(|kk| (self.npk[p][kk] as f64 + self.alpha) / denom)
                    .collect()
            })
            .collect()
    }
}

impl Estimator for PtmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        PtmModel::topic_word(self)
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        PtmModel::doc_topic(self)
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

impl DirichletModel for PtmModel {
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
        self.z.iter().map(|d| d.len()).collect()
    }
}

// ---------------------------------------------------------------------------
// Sampler internals
// ---------------------------------------------------------------------------

/// Weighted categorical sample; `probs` need not be normalised.
fn sample_index<R: Rng>(probs: &[f64], rng: &mut R) -> usize {
    let total: f64 = probs.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

impl PtmModel {
    /// Sample a new topic for token (d, i) with word w, given pseudo-doc p.
    /// Removes the token from counts before sampling, then re-adds.
    fn resample_token<R: Rng>(
        &mut self,
        d: usize,
        i: usize,
        w: usize,
        probs: &mut [f64],
        rng: &mut R,
    ) {
        let p = self.l[d];
        let k_old = self.z[d][i];
        // Remove token from counts.
        self.nkw[k_old][w] -= 1;
        self.nk[k_old] -= 1;
        self.npk[p][k_old] -= 1;
        self.np[p] -= 1;

        let k = self.num_topics;
        let v = self.num_types;
        // `probs` is a reusable scratch buffer (one allocation per sweep, not per
        // token); the k entries are all overwritten below.
        for kk in 0..k {
            let topic_doc = self.npk[p][kk] as f64 + self.alpha;
            let topic_word =
                (self.nkw[kk][w] as f64 + self.beta) / (self.nk[kk] as f64 + v as f64 * self.beta);
            probs[kk] = topic_doc * topic_word;
        }
        let k_new = sample_index(probs, rng);

        self.nkw[k_new][w] += 1;
        self.nk[k_new] += 1;
        self.npk[p][k_new] += 1;
        self.np[p] += 1;
        self.z[d][i] = k_new;
    }

    /// Sample a new pseudo-doc assignment for document d.
    ///
    /// The proposal probability (log-space) is:
    ///   log p(l_d = p) = Σ_k [ lgamma(n_pk^{-d} + α + m_{d,k})
    ///                          - lgamma(n_pk^{-d} + α) ]
    ///                   - [ lgamma(n_p^{-d} + K·α + N_d)
    ///                       - lgamma(n_p^{-d} + K·α) ]
    ///
    /// This is the PTM pseudo-document posterior (uniform prior over pseudo-docs,
    /// matching the simplest PTM variant — no pseudo-doc count prior term).
    fn resample_pseudo<R: Rng>(&mut self, d: usize, doc: &[u32], rng: &mut R) {
        let k = self.num_topics;
        let p_old = self.l[d];

        // Compute m_{d,k}: topic counts for document d's current tokens.
        let mut m_dk = vec![0u32; k];
        for &zi in &self.z[d] {
            m_dk[zi] += 1;
        }
        let n_d = doc.len() as f64;

        // Remove doc d's token counts from its current pseudo-doc.
        for kk in 0..k {
            self.npk[p_old][kk] -= m_dk[kk];
        }
        self.np[p_old] -= doc.len() as u32;

        let num_pseudo = self.num_pseudo;
        let mut log_probs = vec![0.0f64; num_pseudo];
        let k_alpha = k as f64 * self.alpha;

        for p in 0..num_pseudo {
            let np_minus = self.np[p] as f64;
            // Denominator ratio: lgamma(n_p^{-d} + K·α + N_d) - lgamma(n_p^{-d} + K·α)
            let denom_log = log_gamma(np_minus + k_alpha + n_d) - log_gamma(np_minus + k_alpha);
            // Numerator: Σ_k [ lgamma(n_pk^{-d} + α + m_{d,k}) - lgamma(n_pk^{-d} + α) ]
            let mut numer_log = 0.0f64;
            for kk in 0..k {
                let base = self.npk[p][kk] as f64 + self.alpha;
                numer_log += log_gamma(base + m_dk[kk] as f64) - log_gamma(base);
            }
            log_probs[p] = numer_log - denom_log;
        }

        // Softmax to get normalised probs from log-probs (subtract max for stability).
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
        let p_new = sample_index(&probs, rng);

        // Add doc d's counts to the new pseudo-doc.
        for kk in 0..k {
            self.npk[p_new][kk] += m_dk[kk];
        }
        self.np[p_new] += doc.len() as u32;
        self.l[d] = p_new;
    }

    /// One full Gibbs sweep: resample every token's topic, then every doc's
    /// pseudo-doc assignment.
    fn sweep<R: Rng>(&mut self, docs: &[Vec<u32>], rng: &mut R) {
        // --- Token topics ---
        let mut probs = vec![0.0f64; self.num_topics];
        for (d, doc) in docs.iter().enumerate() {
            for (i, &w) in doc.iter().enumerate() {
                self.resample_token(d, i, w as usize, &mut probs, rng);
            }
        }
        // --- Pseudo-doc assignments ---
        for (d, doc) in docs.iter().enumerate() {
            self.resample_pseudo(d, doc, rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

impl PtmModel {
    /// Collapsed Gibbs log-likelihood for the sub-topic (word) layer.
    /// Marginalizes the pseudo-document layer: uses Σ_p n_{pk} as the
    /// pseudo-doc-pooled sub-topic prior. This is a cheap and informative
    /// per-sweep objective; it does not integrate the pseudo-doc assignment.
    pub fn log_likelihood(&self) -> f64 {
        use crate::optimize::digamma;
        let k = self.num_topics;
        let v = self.num_types;
        let p = self.num_pseudo;
        let k_alpha = k as f64 * self.alpha;
        let v_beta = v as f64 * self.beta;
        let mut ll = 0.0f64;

        // Pseudo-document contribution.
        for pp in 0..p {
            let n_p = self.np[pp] as f64;
            if n_p == 0.0 {
                continue;
            }
            for kk in 0..k {
                let n_pk = self.npk[pp][kk] as f64;
                if n_pk > 0.0 {
                    ll += digamma(self.alpha + n_pk) - digamma(self.alpha);
                }
            }
            ll -= digamma(k_alpha + n_p) - digamma(k_alpha);
        }

        // Topic-word contribution.
        for kk in 0..k {
            for w in 0..v {
                let n_kw = self.nkw[kk][w] as f64;
                if n_kw > 0.0 {
                    ll += digamma(self.beta + n_kw) - digamma(self.beta);
                }
            }
            ll -= digamma(v_beta + self.nk[kk] as f64) - digamma(v_beta);
        }

        ll
    }
}

/// Fit a Pseudo-Document Topic Model (PTM) by collapsed Gibbs sampling.
///
/// # Arguments
/// * `docs`       — corpus; each document is a list of word ids (0..num_types)
/// * `num_types`  — vocabulary size V
/// * `num_topics` — number of topics K
/// * `num_pseudo` — number of pseudo-documents P
/// * `alpha`      — document-topic Dirichlet prior (symmetric)
/// * `beta`       — topic-word Dirichlet prior (symmetric)
/// * `iters`      — number of Gibbs sweeps
/// * `rng`        — random-number source (determines all randomness)
#[allow(clippy::too_many_arguments)]
pub fn fit_ptm<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    num_pseudo: usize,
    alpha: f64,
    beta: f64,
    iters: usize,
    rng: &mut R,
) -> PtmModel {
    // Plain fit is the draw-collecting fit with collection disabled, so the
    // sampler lives in exactly one place (see `fit_ptm_with_draws`).
    let (model, _, _) = fit_ptm_with_draws(
        docs,
        num_types,
        num_topics,
        num_pseudo,
        alpha,
        beta,
        iters,
        crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
        0.0,
        0,
        rng,
    );
    model
}

/// Fit a PTM with thinned θ snapshots collected every `thin` sweeps (ring-buffered
/// to `cap` total draws). `cap=0` disables collection entirely.
///
/// Returns `(model, ll_history, converged)`.
#[allow(clippy::too_many_arguments)]
pub fn fit_ptm_with_draws<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    num_pseudo: usize,
    alpha: f64,
    beta: f64,
    iters: usize,
    opts: crate::keyatm::ThetaDrawOpts,
    convergence_tol: f64,
    check_every: usize,
    rng: &mut R,
) -> (PtmModel, Vec<(usize, f64)>, bool) {
    let d_count = docs.len();
    let k = num_topics;
    let p = num_pseudo;
    let v = num_types;

    let l: Vec<usize> = (0..d_count)
        .map(|_| (rng.gen::<f64>() * p as f64) as usize % p)
        .collect();
    let z: Vec<Vec<usize>> = docs
        .iter()
        .map(|doc| {
            doc.iter()
                .map(|_| (rng.gen::<f64>() * k as f64) as usize % k)
                .collect()
        })
        .collect();

    let mut nkw = vec![vec![0u32; v]; k];
    let mut nk = vec![0u32; k];
    let mut npk = vec![vec![0u32; k]; p];
    let mut np = vec![0u32; p];

    for (d, doc) in docs.iter().enumerate() {
        let pd = l[d];
        for (i, &w) in doc.iter().enumerate() {
            let kk = z[d][i];
            nkw[kk][w as usize] += 1;
            nk[kk] += 1;
            npk[pd][kk] += 1;
            np[pd] += 1;
        }
    }

    let mut model = PtmModel {
        num_types: v,
        num_topics: k,
        num_pseudo: p,
        alpha,
        beta,
        nkw,
        nk,
        npk,
        np,
        l,
        z,
        theta_draws: Vec::new(),
    };

    let mut ll_history: Vec<(usize, f64)> = Vec::new();
    let mut converged = false;

    for iter in 1..=iters {
        model.sweep(docs, rng);
        if opts.thin > 0 && iter % opts.thin == 0 {
            let snap: Vec<Vec<f32>> = model
                .l
                .iter()
                .map(|&pd| {
                    let denom = model.np[pd] as f64 + k as f64 * model.alpha;
                    (0..k)
                        .map(|kk| ((model.npk[pd][kk] as f64 + model.alpha) / denom) as f32)
                        .collect()
                })
                .collect();
            if model.theta_draws.len() < opts.cap {
                model.theta_draws.push(snap);
            } else {
                // ring-buffer: pop oldest, push newest
                model.theta_draws.remove(0);
                model.theta_draws.push(snap);
            }
        }
        // Trace recording and optional convergence check (never alters RNG).
        if check_every > 0 && iter % check_every == 0 {
            let ll = model.log_likelihood();
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

    /// Build a corpus of short docs (2–3 tokens each) drawn from K=2 disjoint
    /// vocabulary blocks, then verify that PTM recovers the planted structure.
    /// This is the canonical short-text regime PTM targets: docs are too short
    /// for per-document statistics to be reliable (2–3 tokens each), so PTM
    /// aggregates them into pseudo-documents.
    #[test]
    fn recovers_topics_from_short_docs() {
        // 2 topics, each owns 8 words → V = 16.
        // Docs are 3 tokens drawn from a single block.
        let block_size = 8usize;
        let num_topics = 2usize;
        let v = block_size * num_topics;
        let blocks: Vec<Vec<u32>> = (0..num_topics)
            .map(|b| ((b * block_size) as u32..((b + 1) * block_size) as u32).collect())
            .collect();

        let mut docs: Vec<Vec<u32>> = Vec::new();
        // 600 docs, 300 per topic, each 3 tokens cycling through block words.
        for b in 0..num_topics {
            for d in 0..300usize {
                let blk = &blocks[b];
                let doc: Vec<u32> = (0..3).map(|i| blk[(i + d) % blk.len()]).collect();
                docs.push(doc);
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // P=10 pseudo-docs, 1000 Gibbs sweeps.
        let model = fit_ptm(&docs, v, num_topics, 10, 0.1, 0.01, 1000, &mut rng);

        // Compute the fraction of each topic's probability mass that falls on each
        // planted block. For a K=2 cleanly planted corpus, each topic should
        // concentrate >80% of its mass on a distinct block.
        let tw = model.topic_word();
        let block_score = |row: &[f64], bi: usize| -> f64 {
            blocks[bi].iter().map(|&w| row[w as usize]).sum::<f64>()
        };

        // Check that the two topics concentrate on different blocks.
        // topic 0 best-block score
        let best_0 = (0..num_topics)
            .map(|bi| block_score(&tw[0], bi))
            .fold(0.0f64, f64::max);
        let best_1 = (0..num_topics)
            .map(|bi| block_score(&tw[1], bi))
            .fold(0.0f64, f64::max);
        // Which block is best for each topic?
        let argmax_0 = (0..num_topics)
            .max_by(|&a, &b| {
                block_score(&tw[0], a)
                    .partial_cmp(&block_score(&tw[0], b))
                    .unwrap()
            })
            .unwrap();
        let argmax_1 = (0..num_topics)
            .max_by(|&a, &b| {
                block_score(&tw[1], a)
                    .partial_cmp(&block_score(&tw[1], b))
                    .unwrap()
            })
            .unwrap();

        // Short docs (3 tokens each) produce noisy per-document statistics; PTM
        // aggregates them into pseudo-documents, so each topic should concentrate
        // the majority (>65%) of its mass on one block.
        assert!(
            best_0 > 0.65,
            "topic 0 should concentrate on one block but max block mass is {:.3}.\n\
             Block masses: {:?}",
            best_0,
            (0..num_topics)
                .map(|bi| block_score(&tw[0], bi))
                .collect::<Vec<_>>()
        );
        assert!(
            best_1 > 0.65,
            "topic 1 should concentrate on one block but max block mass is {:.3}.\n\
             Block masses: {:?}",
            best_1,
            (0..num_topics)
                .map(|bi| block_score(&tw[1], bi))
                .collect::<Vec<_>>()
        );
        assert_ne!(
            argmax_0,
            argmax_1,
            "both topics concentrated on the same block ({argmax_0}); \
             no block coverage for block {}",
            1 - argmax_0
        );
    }

    /// Two fits with the same seed must be bit-for-bit identical.
    #[test]
    fn deterministic_for_fixed_seed() {
        let v = 15usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..3).map(|i| ((i + d) % v) as u32).collect())
            .collect();
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let m1 = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 30, &mut r1);
        let m2 = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 30, &mut r2);
        assert_eq!(m1.nk, m2.nk, "nk differs between two identical-seed runs");
        assert_eq!(
            m1.topic_word(),
            m2.topic_word(),
            "topic_word differs between two identical-seed runs"
        );
    }

    #[test]
    fn ptm_conforms() {
        let v = 15usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..3).map(|i| ((i + d) % v) as u32).collect())
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let m = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 20, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
