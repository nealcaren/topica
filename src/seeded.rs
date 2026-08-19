//! Seeded LDA — guided topic modeling with asymmetric word-topic priors.
//!
//! Jagarlamudi, Daumé III & Udupa (2012), "Incorporating Lexical Priors into
//! Topic Models", EACL 2012; Watanabe & Sumita (2015).
//!
//! Standard collapsed-Gibbs LDA, except the topic-word Dirichlet prior is
//! **asymmetric**: a *seed word* for topic k receives an extra prior
//! pseudocount `m_{k,w}` in that topic, encouraging the topic to form around
//! its seeds.  Topics with no seeds behave as ordinary LDA topics.
//!
//! ## Prior parameterisation
//!
//! For topic k and word w (with `m_{k,w}` the seed pseudocount, 0 for non-seeds):
//! ```text
//! β_{k,w} = β  +  m_{k,w}
//! β_sum[k] = V·β  +  Σ_w m_{k,w}
//! ```
//!
//! The mass `m_{k,w}` is supplied per (topic, seed word). Two constructions are
//! used by the binding: the **seededlda-package** default, where a seed word's
//! mass scales with its corpus frequency (`count_w · weight · 100`, matching
//! `seededlda::tfm`), and the topica-native **uniform** scheme, where every seed
//! word gets the same `weight · 100`. This is exactly `seededlda`'s model: it
//! adds the same seed matrix to the word-topic count once and never removes it,
//! so the mass acts as a persistent asymmetric-β pseudocount in both the
//! sampling conditional and φ.
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
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::collections::HashMap;

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
    /// `seeds[k]` — set of seed word-ids for topic k (sorted, de-duped).
    pub seeds: Vec<Vec<usize>>,
    /// `seed_mass[k][i]` — the prior pseudocount for `seeds[k][i]` in topic k
    /// (aligned to `seeds`). The seededlda-compatible construction scales each by
    /// its corpus frequency; the uniform scheme gives every seed word the same
    /// mass. An older save without this field falls back to a per-seed `1.0`
    /// (any refit rebuilds it from the binding).
    #[serde(default)]
    pub seed_mass: Vec<Vec<f64>>,
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

    /// The seed pseudocount `m_{k,w}` for word w in topic k (0 if w does not seed
    /// k). `seeds[k]` is sorted, so a binary search indexes the aligned mass.
    #[inline]
    fn mass_kw(&self, k: usize, w: usize) -> f64 {
        match self.seeds[k].binary_search(&w) {
            // Old saves may carry `seeds` without `seed_mass`; fall back to 1.0.
            Ok(i) => self
                .seed_mass
                .get(k)
                .and_then(|m| m.get(i))
                .copied()
                .unwrap_or(1.0),
            Err(_) => 0.0,
        }
    }

    /// β_{k,w}: base prior plus the seed pseudocount for word w in topic k.
    #[inline]
    fn beta_kw(&self, k: usize, w: usize) -> f64 {
        self.beta + self.mass_kw(k, w)
    }

    /// β_sum[k]: sum of β_{k,w} over the full vocabulary V = V·β + Σ_w m_{k,w}.
    #[inline]
    fn beta_sum(&self, k: usize) -> f64 {
        let seed_total: f64 = match self.seed_mass.get(k) {
            Some(m) if !m.is_empty() => m.iter().sum(),
            // Old-save fallback: 1.0 per seed word.
            _ => self.seeds[k].len() as f64,
        };
        self.num_types as f64 * self.beta + seed_total
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Collapsed Gibbs marginal log-likelihood log P(w, z | α, β), the same
    /// MALLET formula as LDA (`output::model_log_likelihood`) but with the seeded
    /// asymmetric β_{k,w} prior. Negative; it rises toward 0 during burn-in, then
    /// fluctuates around a plateau (collapsed Gibbs is a sampler, not an optimizer).
    /// Uses the shared `output::log_gamma` (the MALLET-parity log Gamma the
    /// canonical LDA log-likelihood uses); an earlier version used `digamma` (the
    /// derivative of `log_gamma`), which is not a log-likelihood and moved the
    /// wrong way. Cheap to call; does not allocate.
    pub fn log_likelihood(&self, docs: &[Vec<u32>]) -> f64 {
        use crate::output::log_gamma;
        let k = self.num_topics;
        let v = self.num_types;
        let mut ll = 0.0f64;

        // Document-topic contribution. Uses the per-document asymmetric prior
        // α_{d,k} when set (e.g. an embedding-anchored doc prior), matching the
        // prior the sampler actually used; otherwise the symmetric α.
        for (d, doc) in docs.iter().enumerate() {
            let n_d = doc.len() as f64;
            let alpha_dt = |t: usize| match &self.doc_alpha {
                Some(da) => da[d][t],
                None => self.alpha,
            };
            let mut k_alpha = 0.0f64;
            for t in 0..k {
                let a = alpha_dt(t);
                k_alpha += a;
                let n_dt = self.ndk[d][t] as f64;
                if n_dt > 0.0 {
                    ll += log_gamma(a + n_dt) - log_gamma(a);
                }
            }
            ll -= log_gamma(k_alpha + n_d) - log_gamma(k_alpha);
        }

        // Topic-word contribution.
        for t in 0..k {
            let beta_sum_t = self.beta_sum(t);
            for w in 0..v {
                let n_tw = self.nkw[t][w] as f64;
                if n_tw > 0.0 {
                    let beta_tw = self.beta_kw(t, w);
                    ll += log_gamma(beta_tw + n_tw) - log_gamma(beta_tw);
                }
            }
            ll -= log_gamma(beta_sum_t + self.nk[t] as f64) - log_gamma(beta_sum_t);
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

/// One restricted collapsed-Gibbs sweep over a contiguous block of documents.
///
/// `ndk`, `z`, `docs`, and `doc_alpha` (when present) are all indexed by the same
/// local document index `0..docs.len()`, so a caller can pass the whole corpus
/// (serial) or a `[start..end]` partition (one AD-LDA worker). `nkw`/`nk` are the
/// (possibly per-worker private) count tables; `scores` is a reused length-K
/// scratch buffer. The per-token math is identical to the original inline loop.
#[allow(clippy::too_many_arguments)]
fn run_sweep_seeded_range<R: Rng>(
    nkw: &mut [Vec<u32>],
    nk: &mut [u32],
    ndk: &mut [Vec<u32>],
    z: &mut [Vec<usize>],
    docs: &[Vec<u32>],
    doc_alpha: Option<&[Vec<f64>]>,
    mass_map: &[HashMap<usize, f64>],
    beta_sum_k: &[f64],
    beta: f64,
    alpha: f64,
    k: usize,
    scores: &mut [f64],
    rng: &mut R,
) {
    for d in 0..docs.len() {
        let doc = &docs[d];
        // The document's α row: per-document when supplied, else symmetric.
        let a_row: Option<&Vec<f64>> = doc_alpha.map(|da| &da[d]);
        for i in 0..doc.len() {
            let w = doc[i] as usize;
            let old = z[d][i];

            // Remove token from counts.
            nkw[old][w] -= 1;
            nk[old] -= 1;
            ndk[d][old] -= 1;

            // Compute unnormalised sampling probabilities.
            for t in 0..k {
                let beta_tw = beta + mass_map[t].get(&w).copied().unwrap_or(0.0);
                let a_t = a_row.map_or(alpha, |r| r[t]);
                scores[t] = (a_t + ndk[d][t] as f64) * (beta_tw + nkw[t][w] as f64)
                    / (beta_sum_k[t] + nk[t] as f64);
            }

            // Sample new topic and update counts.
            let new_t = sample_index(scores, rng);
            nkw[new_t][w] += 1;
            nk[new_t] += 1;
            ndk[d][new_t] += 1;
            z[d][i] = new_t;
        }
    }
}

/// Contiguous, non-overlapping document ranges — one per worker (clamped so the
/// worker count never exceeds the document count). Mirrors the binding's
/// `partition_ranges`, kept local so the core stays self-contained.
fn partition_ranges(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(n.max(1));
    let base = n / parts;
    let rem = n % parts;
    let mut ranges = Vec::with_capacity(parts);
    let mut start = 0;
    for i in 0..parts {
        let len = base + if i < rem { 1 } else { 0 };
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

/// One approximate-parallel (AD-LDA) sweep over the dense SeededLDA count tables.
/// Documents are partitioned across workers; each worker samples its slice against
/// a private clone of the global `nkw`/`nk`, owning copies of its `z`/`ndk` slice,
/// then the results are reconciled: `nkw` is the exact additive merge
/// `Σ_workers − (W−1)·original` (accumulated in `i64`, clamped ≥ 0), `nk` is
/// **recomputed** from the merged `nkw` rows so `nk[k] == Σ_w nkw[k][w]` holds
/// exactly (no zero-clamp drift), and each worker's `z`/`ndk` slice is written
/// straight back (documents are disjoint). The seeded asymmetric prior `β_{k,w}`
/// is a fixed pseudocount that never enters `nkw`, so the merge is unaffected by
/// it. Deterministic for a fixed `sweep_seed`/`num_threads`.
#[allow(clippy::too_many_arguments)]
fn parallel_sweep_seeded(
    nkw: &mut [Vec<u32>],
    nk: &mut [u32],
    ndk: &mut [Vec<u32>],
    z: &mut [Vec<usize>],
    docs: &[Vec<u32>],
    doc_alpha: Option<&[Vec<f64>]>,
    mass_map: &[HashMap<usize, f64>],
    beta_sum_k: &[f64],
    beta: f64,
    alpha: f64,
    k: usize,
    num_threads: usize,
    sweep_seed: u64,
) {
    struct SeededWorkerOut {
        nkw: Vec<Vec<u32>>,
        z: Vec<Vec<usize>>,
        ndk: Vec<Vec<u32>>,
        start: usize,
    }

    let v = nkw.first().map_or(0, |row| row.len());
    let ranges = partition_ranges(docs.len(), num_threads);
    let orig_nkw = nkw.to_vec();
    let orig_nk = nk.to_vec();
    // Read-only views so workers can copy their slices inside the parallel map.
    let z_ro: &[Vec<usize>] = z;
    let ndk_ro: &[Vec<u32>] = ndk;

    let outs: Vec<SeededWorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut wnkw = orig_nkw.clone();
            let mut wnk = orig_nk.clone();
            let mut wz: Vec<Vec<usize>> = z_ro[start..end].to_vec();
            let mut wndk: Vec<Vec<u32>> = ndk_ro[start..end].to_vec();
            let da_slice = doc_alpha.map(|da| &da[start..end]);
            let mut rng = Pcg64Mcg::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut scores = vec![0.0f64; k];
            run_sweep_seeded_range(
                &mut wnkw,
                &mut wnk,
                &mut wndk,
                &mut wz,
                &docs[start..end],
                da_slice,
                mass_map,
                beta_sum_k,
                beta,
                alpha,
                k,
                &mut scores,
                &mut rng,
            );
            SeededWorkerOut {
                nkw: wnkw,
                z: wz,
                ndk: wndk,
                start,
            }
        })
        .collect();

    // --- Reconcile nkw: final = Σ_workers − (W−1)·original, clamped ≥ 0. ---
    let wm1 = (outs.len() as i64) - 1;
    for t in 0..k {
        for w in 0..v {
            let sum_w: i64 = outs.iter().map(|o| o.nkw[t][w] as i64).sum();
            let val = sum_w - wm1 * orig_nkw[t][w] as i64;
            nkw[t][w] = val.max(0) as u32;
        }
        // Recompute nk from the merged nkw row so the two stay exactly consistent.
        nk[t] = nkw[t].iter().map(|&c| c as u64).sum::<u64>() as u32;
    }

    // --- Write back each worker's per-document z / ndk slice (disjoint docs). ---
    for out in outs {
        for (i, (zrow, ndkrow)) in out.z.into_iter().zip(out.ndk).enumerate() {
            z[out.start + i] = zrow;
            ndk[out.start + i] = ndkrow;
        }
    }
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
/// * `seed_masses` — `seed_masses[k][i]` is the prior pseudocount for
///                   `seeds[k][i]`, aligned to `seeds`. Empty lists (no seeds)
///                   give an ordinary LDA topic.
/// * `random_init` — when true, every token starts in a uniformly random topic
///                   (the seededlda-package behaviour: seeds enter only through
///                   the β pseudocount). When false, a token whose word seeds a
///                   topic is anchored there at init (topica's uniform scheme).
/// * `iters`       — number of full Gibbs sweeps.
/// * `draws`            — thinned θ-draw retention schedule (issue #31).
/// * `convergence_tol`  — relative-change tolerance for early stopping (0 = off).
/// * `check_every`      — LL-trace cadence in sweeps (0 = no trace).
/// * `num_threads`      — `1` runs the exact serial sweep; `>1` runs MALLET-style
///                        approximate-parallel (AD-LDA) sampling, deterministic
///                        for a fixed `num_threads` + seed.
/// * `rng`              — random-number source; deterministic for a fixed seed.
///
/// Returns `(model, ll_history, converged)` where `ll_history` is a vector of
/// `(iteration, log_likelihood)` pairs recorded every `check_every` sweeps.
#[allow(clippy::too_many_arguments)]
pub fn fit_seeded_lda<R: Rng, F: FnMut(usize, usize, f64)>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    seeds: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    seed_masses: &[Vec<f64>],
    random_init: bool,
    doc_alpha: Option<Vec<Vec<f64>>>,
    iters: usize,
    draws: crate::keyatm::ThetaDrawOpts,
    convergence_tol: f64,
    check_every: usize,
    num_threads: usize,
    mut on_progress: F,
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

    // Normalise seeds: sort by word id (carrying the aligned mass) and drop
    // duplicate words within a topic (keeping the first mass), so `binary_search`
    // and the mass maps are well-defined.
    assert_eq!(
        seed_masses.len(),
        seeds.len(),
        "seed_masses must have one row per topic, aligned to seeds"
    );
    let mut seeds_clean: Vec<Vec<usize>> = Vec::with_capacity(seeds.len());
    let mut seed_mass_clean: Vec<Vec<f64>> = Vec::with_capacity(seeds.len());
    for (sv, mv) in seeds.iter().zip(seed_masses.iter()) {
        assert_eq!(
            sv.len(),
            mv.len(),
            "each seed row must align with its masses"
        );
        let mut pairs: Vec<(usize, f64)> = sv.iter().copied().zip(mv.iter().copied()).collect();
        pairs.sort_by_key(|p| p.0);
        pairs.dedup_by_key(|p| p.0);
        seeds_clean.push(pairs.iter().map(|p| p.0).collect());
        seed_mass_clean.push(pairs.iter().map(|p| p.1).collect());
    }

    // Per-topic mass map for O(1) lookups in the hot loop: mass_map[k][w] = m_{k,w}.
    let mass_map: Vec<std::collections::HashMap<usize, f64>> = seeds_clean
        .iter()
        .zip(seed_mass_clean.iter())
        .map(|(sv, mv)| sv.iter().copied().zip(mv.iter().copied()).collect())
        .collect();

    // Precompute β_sum[k] once (it is constant after init).
    let beta_sum_k: Vec<f64> = (0..k)
        .map(|kk| v as f64 * beta + seed_mass_clean[kk].iter().sum::<f64>())
        .collect();

    // For each word, which topics seed it (used only for anchored initialisation).
    let mut word_seed_topics: Vec<Vec<usize>> = vec![Vec::new(); v];
    for (kk, sv) in seeds_clean.iter().enumerate() {
        for &w in sv {
            word_seed_topics[w].push(kk);
        }
    }

    // --- Initialise. Under `random_init` (the seededlda-package default) every
    // token starts in a uniformly random topic and the seeds bias the fit purely
    // through the β pseudocount. Otherwise (topica's uniform scheme) a token
    // whose word seeds some topic is anchored there at init, so documents with
    // seed words begin with mass on the seeded topic. ---
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
                    if random_init || cands.is_empty() {
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
        use crate::output::log_gamma;
        let mut ll = 0.0f64;
        for (d, doc) in docs.iter().enumerate() {
            let n_d = doc.len() as f64;
            // Per-document asymmetric α_{d,k} when a doc prior is in effect,
            // matching the sampler; otherwise the symmetric α.
            let alpha_dt = |t: usize| match &doc_alpha {
                Some(da) => da[d][t],
                None => alpha,
            };
            let mut k_alpha = 0.0f64;
            for t in 0..k {
                let a = alpha_dt(t);
                k_alpha += a;
                let n_dt = ndk[d][t] as f64;
                if n_dt > 0.0 {
                    ll += log_gamma(a + n_dt) - log_gamma(a);
                }
            }
            ll -= log_gamma(k_alpha + n_d) - log_gamma(k_alpha);
        }
        for t in 0..k {
            let beta_sum_t = beta_sum_k[t];
            for w in 0..v {
                let n_tw = nkw[t][w] as f64;
                if n_tw > 0.0 {
                    let beta_tw = beta + mass_map[t].get(&w).copied().unwrap_or(0.0);
                    ll += log_gamma(beta_tw + n_tw) - log_gamma(beta_tw);
                }
            }
            ll -= log_gamma(beta_sum_t + nk[t] as f64) - log_gamma(beta_sum_t);
        }
        ll
    };

    for it in 0..iters {
        let iter = it + 1;
        if num_threads <= 1 {
            // Exact serial path — byte-identical to the pre-threading loop.
            run_sweep_seeded_range(
                &mut nkw,
                &mut nk,
                &mut ndk,
                &mut z,
                docs,
                doc_alpha.as_deref(),
                &mass_map,
                &beta_sum_k,
                beta,
                alpha,
                k,
                &mut scores,
                rng,
            );
        } else {
            // Approximate-parallel AD-LDA. One per-sweep seed drawn from the main
            // RNG keeps it deterministic for a fixed num_threads + seed.
            let sweep_seed = rng.gen::<u64>();
            parallel_sweep_seeded(
                &mut nkw,
                &mut nk,
                &mut ndk,
                &mut z,
                docs,
                doc_alpha.as_deref(),
                &mass_map,
                &beta_sum_k,
                beta,
                alpha,
                k,
                num_threads,
                sweep_seed,
            );
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
            on_progress(iter, iters, ll);
            if convergence_tol > 0.0 && ll_history.len() >= 2 {
                let prev = ll_history[ll_history.len() - 2].1;
                let rel = (ll - prev).abs() / (prev.abs() + 1e-12);
                if rel < convergence_tol {
                    on_progress(iter, iter, ll); // snap bar to 100% (#786)
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
        seeds: seeds_clean,
        seed_mass: seed_mass_clean,
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

    /// Uniform seed masses: every seed word in a topic gets `w`. Aligned to `seeds`.
    fn uniform_masses(seeds: &[Vec<usize>], w: f64) -> Vec<Vec<f64>> {
        seeds.iter().map(|s| vec![w; s.len()]).collect()
    }

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
        let masses = uniform_masses(&seeds, 50.0);
        let (model, _, _) = fit_seeded_lda(
            &docs,
            v,
            3,
            &seeds,
            0.1,
            0.01,
            &masses,
            false,
            None,
            300,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            1,
            |_, _, _| {},
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
        let masses = uniform_masses(&seeds, 0.0);
        let (model, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            &masses,
            true,
            None,
            50,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            1,
            |_, _, _| {},
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

    /// The recorded log-likelihood trace must be a genuine (negative) collapsed
    /// marginal that rises toward 0 as sampling improves, not the old digamma
    /// surrogate, which was positive and fell. Locks in the log_gamma fix (#663).
    #[test]
    fn log_likelihood_is_negative_and_rises() {
        let mut setup_rng = ChaCha8Rng::seed_from_u64(9);
        let (docs, v) = synthetic_corpus(3, 10, 40, 8, 2, &mut setup_rng);
        let seeds = vec![vec![0usize, 1usize], vec![10usize, 11usize], vec![]];
        let masses = uniform_masses(&seeds, 10.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (model, ll_history, _) = fit_seeded_lda(
            &docs,
            v,
            3,
            &seeds,
            0.1,
            0.01,
            &masses,
            false,
            None,
            200,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            20, // check_every: record the trace
            1,
            |_, _, _| {},
            &mut rng,
        );

        assert!(
            ll_history.len() >= 2,
            "trace should have several checkpoints"
        );
        for &(_, ll) in &ll_history {
            // The old digamma surrogate was large-POSITIVE; a real collapsed
            // marginal is negative. This alone distinguishes the fix.
            assert!(
                ll < 0.0,
                "collapsed log-likelihood must be negative, got {ll}"
            );
        }
        // It rises during burn-in then plateaus (collapsed Gibbs is stochastic, so
        // it is not strictly monotone once converged): the final checkpoint is no
        // worse than the first beyond a small plateau tolerance.
        let first = ll_history.first().unwrap().1;
        let last = ll_history.last().unwrap().1;
        assert!(
            last >= first - first.abs() * 0.05,
            "log-likelihood should not fall meaningfully: {first} -> {last}",
        );
        // Direct accessor agrees in sign.
        assert!(model.log_likelihood(&docs) < 0.0);
    }

    /// With a per-document asymmetric prior (the EmbeddingLDA doc-embedding mode),
    /// the log-likelihood must still be a finite, negative marginal, computed with
    /// that same α_{d,k}, not the symmetric α (#663 / #664 review).
    #[test]
    fn log_likelihood_uses_doc_alpha() {
        let mut setup_rng = ChaCha8Rng::seed_from_u64(11);
        let (docs, v) = synthetic_corpus(3, 10, 30, 8, 2, &mut setup_rng);
        let seeds = vec![vec![0usize, 1usize], vec![10usize, 11usize], vec![]];
        let masses = uniform_masses(&seeds, 10.0);
        // A non-uniform per-doc prior: bias each doc toward one topic.
        let doc_alpha: Vec<Vec<f64>> = (0..docs.len())
            .map(|d| {
                let mut a = vec![0.1; 3];
                a[d % 3] += 1.0;
                a
            })
            .collect();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (model, ll_history, _) = fit_seeded_lda(
            &docs,
            v,
            3,
            &seeds,
            0.1,
            0.01,
            &masses,
            false,
            Some(doc_alpha),
            150,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            25,
            1,
            |_, _, _| {},
            &mut rng,
        );

        assert!(!ll_history.is_empty());
        for &(_, ll) in &ll_history {
            assert!(
                ll.is_finite() && ll < 0.0,
                "LL must be finite & negative: {ll}"
            );
        }
        assert!(model.log_likelihood(&docs).is_finite());
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
        let masses = uniform_masses(&seeds, 2.0);

        let mut r1 = ChaCha8Rng::seed_from_u64(55);
        let mut r2 = ChaCha8Rng::seed_from_u64(55);
        let (m1, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            &masses,
            false,
            None,
            80,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            1,
            |_, _, _| {},
            &mut r1,
        );
        let (m2, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            &masses,
            false,
            None,
            80,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            1,
            |_, _, _| {},
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
        let masses = uniform_masses(&seeds, 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let (m, _, _) = fit_seeded_lda(
            &docs,
            v,
            k,
            &seeds,
            0.1,
            0.1,
            &masses,
            true,
            None,
            20,
            crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
            0.0,
            0,
            1,
            |_, _, _| {},
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
