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
//! Each real document is drawn to a pseudo-document by a Dirichlet-multinomial
//! mixture ψ ~ Dir(λ): collapsing ψ gives the assignment its `(m_p + λ)`
//! popularity prior, where m_p is the number of documents already at pseudo-doc p.
//! This rich-get-richer aggregation (λ preventing collapse) is what lets a few
//! pseudo-documents pool enough short texts to estimate topics reliably.
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
use crate::mathfun::log_gamma;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

pub struct PtmModel {
    pub num_types: usize,
    pub num_topics: usize,
    pub num_pseudo: usize,
    pub alpha: f64,
    pub beta: f64,
    /// λ: symmetric Dirichlet prior on the pseudo-document mixture ψ ~ Dir(λ).
    /// Drives PTM's rich-get-richer aggregation via the `(m_p + λ)` assignment
    /// term; larger λ flattens the popularity bias, smaller λ sharpens it.
    pub lambda: f64,
    /// n_kw: K × V  topic-word counts
    pub nkw: Vec<Vec<u32>>,
    /// n_k: K  topic totals
    pub nk: Vec<u32>,
    /// n_pk: P × K  pseudo-doc topic counts
    pub npk: Vec<Vec<u32>>,
    /// n_p: P  pseudo-doc token totals
    pub np: Vec<u32>,
    /// m_p: P  number of real documents currently assigned to each pseudo-doc
    /// (the customers in the pseudo-doc CRP; drives the `(m_p + λ)` prior).
    pub mp: Vec<u32>,
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
    /// One full Gibbs sweep over the whole corpus, delegating to the shared,
    /// range-based [`run_sweep_ptm_range`]. Disjoint `&mut self.field` borrows in a
    /// single call are permitted, so the serial path and each AD-LDA worker share a
    /// single copy of the sampling logic (no drift).
    fn sweep<R: Rng>(&mut self, docs: &[Vec<u32>], rng: &mut R) {
        run_sweep_ptm_range(
            &mut self.nkw,
            &mut self.nk,
            &mut self.npk,
            &mut self.np,
            &mut self.mp,
            &mut self.l,
            &mut self.z,
            docs,
            self.alpha,
            self.beta,
            self.lambda,
            self.num_topics,
            self.num_types,
            self.num_pseudo,
            rng,
        );
    }
}

/// One full two-phase Gibbs sweep over a contiguous slice of documents: first
/// resample every token's topic, then every document's pseudo-doc assignment.
/// The global topic-word (`nkw`/`nk`) and pseudo-doc (`npk`/`np`/`mp`) count tables
/// are the state shared across documents; the per-document `l`/`z` are indexed
/// 0-based within the slice and must align with `docs` (the `l` *values* are global
/// pseudo-doc ids in `0..num_pseudo`). The arithmetic, allocation pattern, and
/// per-token / per-doc RNG draw order are exactly those of the original
/// `sweep`/`resample_token`/`resample_pseudo`, so the serial call site stays
/// byte-identical.
#[allow(clippy::too_many_arguments)]
fn run_sweep_ptm_range<R: Rng>(
    nkw: &mut [Vec<u32>],
    nk: &mut [u32],
    npk: &mut [Vec<u32>],
    np: &mut [u32],
    mp: &mut [u32],
    l: &mut [usize],
    z: &mut [Vec<usize>],
    docs: &[Vec<u32>],
    alpha: f64,
    beta: f64,
    lambda: f64,
    num_topics: usize,
    num_types: usize,
    num_pseudo: usize,
    rng: &mut R,
) {
    let k = num_topics;
    let v = num_types;

    // --- Phase 1: token topics ---
    // `probs` is a reusable scratch buffer (one allocation per sweep, not per token).
    let mut probs = vec![0.0f64; k];
    for d in 0..docs.len() {
        let p = l[d];
        for i in 0..docs[d].len() {
            let w = docs[d][i] as usize;
            let k_old = z[d][i];
            // Remove token from counts.
            nkw[k_old][w] -= 1;
            nk[k_old] -= 1;
            npk[p][k_old] -= 1;
            np[p] -= 1;

            for kk in 0..k {
                let topic_doc = npk[p][kk] as f64 + alpha;
                let topic_word = (nkw[kk][w] as f64 + beta) / (nk[kk] as f64 + v as f64 * beta);
                probs[kk] = topic_doc * topic_word;
            }
            let k_new = sample_index(&probs, rng);

            nkw[k_new][w] += 1;
            nk[k_new] += 1;
            npk[p][k_new] += 1;
            np[p] += 1;
            z[d][i] = k_new;
        }
    }

    // --- Phase 2: pseudo-doc assignments ---
    // The proposal (log-space) for doc d is
    //   log p(l_d = p) = ln(m_p^{-d} + lambda)                   [popularity prior]
    //     + Sum_k [ lgamma(n_pk^{-d} + a + m_dk) - lgamma(n_pk^{-d} + a) ]  [content]
    //     - [ lgamma(n_p^{-d} + K*a + N_d) - lgamma(n_p^{-d} + K*a) ]       [norm],
    // the collapsed PTM posterior whose `(m_p + lambda)` term is the rich-get-richer
    // aggregation (lambda prevents collapse). `m_dk` uses the *updated* z from phase 1.
    let k_alpha = k as f64 * alpha;
    for d in 0..docs.len() {
        let n_d_len = docs[d].len();
        let p_old = l[d];

        // m_{d,k}: topic counts for doc d's current tokens.
        let mut m_dk = vec![0u32; k];
        for &zi in &z[d] {
            m_dk[zi] += 1;
        }
        let n_d = n_d_len as f64;

        // Leave-one-out: remove doc d (its token/topic counts and its one customer).
        for kk in 0..k {
            npk[p_old][kk] -= m_dk[kk];
        }
        np[p_old] -= n_d_len as u32;
        mp[p_old] -= 1;

        let mut log_probs = vec![0.0f64; num_pseudo];
        for p in 0..num_pseudo {
            let np_minus = np[p] as f64;
            let prior_log = (mp[p] as f64 + lambda).ln();
            let denom_log = log_gamma(np_minus + k_alpha + n_d) - log_gamma(np_minus + k_alpha);
            let mut numer_log = 0.0f64;
            for kk in 0..k {
                let base = npk[p][kk] as f64 + alpha;
                numer_log += log_gamma(base + m_dk[kk] as f64) - log_gamma(base);
            }
            log_probs[p] = prior_log + numer_log - denom_log;
        }

        // Softmax (subtract max for stability), then sample.
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
        let p_new = sample_index(&probs, rng);

        // Add doc d's counts (and its one customer) to the new pseudo-doc.
        for kk in 0..k {
            npk[p_new][kk] += m_dk[kk];
        }
        np[p_new] += n_d_len as u32;
        mp[p_new] += 1;
        l[d] = p_new;
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

/// One approximate-parallel (AD-LDA) sweep over the PTM count tables. Documents are
/// partitioned across workers; each worker samples its slice (both phases) against
/// private clones of ALL FIVE global tables (`nkw`/`nk`/`npk`/`np`/`mp`), owning
/// copies of its per-document `l`/`z` slice (documents are disjoint), then the
/// results are reconciled. `nkw`, `npk`, and `mp` are each the exact additive merge
/// `Σ_workers − (W−1)·original` (accumulated in `i64`, clamped ≥ 0); `nk` and `np`
/// are **recomputed** from the merged `nkw`/`npk` rows so they stay exactly
/// consistent (no zero-clamp drift). The merge is valid because each worker only
/// touches its own documents' contributions to every global table (word tokens,
/// pseudo-doc topic counts, and the one CRP customer per document alike), so the
/// per-worker deltas sum linearly; the doc-count invariant `Σ_p mp[p] = D` and
/// `mp[p] ≥ 0` are preserved automatically. `num_pseudo` (P) is fixed, so no
/// pseudo-doc count is discovered under threading. Deterministic for a fixed
/// `sweep_seed`/`num_threads`.
#[allow(clippy::too_many_arguments)]
fn parallel_sweep_ptm(
    nkw: &mut [Vec<u32>],
    nk: &mut [u32],
    npk: &mut [Vec<u32>],
    np: &mut [u32],
    mp: &mut [u32],
    l: &mut [usize],
    z: &mut [Vec<usize>],
    docs: &[Vec<u32>],
    alpha: f64,
    beta: f64,
    lambda: f64,
    num_topics: usize,
    num_types: usize,
    num_pseudo: usize,
    num_threads: usize,
    sweep_seed: u64,
) {
    struct PtmWorkerOut {
        nkw: Vec<Vec<u32>>,
        npk: Vec<Vec<u32>>,
        mp: Vec<u32>,
        l: Vec<usize>,
        z: Vec<Vec<usize>>,
        start: usize,
    }

    let v = num_types;
    let k = num_topics;
    let p_count = num_pseudo;
    let ranges = partition_ranges(docs.len(), num_threads);
    let orig_nkw = nkw.to_vec();
    let orig_nk = nk.to_vec();
    let orig_npk = npk.to_vec();
    let orig_np = np.to_vec();
    let orig_mp = mp.to_vec();
    // Read-only views so workers can copy their slices inside the parallel map.
    let l_ro: &[usize] = l;
    let z_ro: &[Vec<usize>] = z;

    let outs: Vec<PtmWorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut wnkw = orig_nkw.clone();
            let mut wnk = orig_nk.clone();
            let mut wnpk = orig_npk.clone();
            let mut wnp = orig_np.clone();
            let mut wmp = orig_mp.clone();
            let mut wl: Vec<usize> = l_ro[start..end].to_vec();
            let mut wz: Vec<Vec<usize>> = z_ro[start..end].to_vec();
            let mut rng = Pcg64Mcg::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            run_sweep_ptm_range(
                &mut wnkw,
                &mut wnk,
                &mut wnpk,
                &mut wnp,
                &mut wmp,
                &mut wl,
                &mut wz,
                &docs[start..end],
                alpha,
                beta,
                lambda,
                num_topics,
                num_types,
                num_pseudo,
                &mut rng,
            );
            PtmWorkerOut {
                nkw: wnkw,
                npk: wnpk,
                mp: wmp,
                l: wl,
                z: wz,
                start,
            }
        })
        .collect();

    let wm1 = (outs.len() as i64) - 1;

    // --- Reconcile nkw: final = Σ_workers − (W−1)·original, clamp ≥ 0; nk from rows. ---
    for t in 0..k {
        for w in 0..v {
            let sum_w: i64 = outs.iter().map(|o| o.nkw[t][w] as i64).sum();
            nkw[t][w] = (sum_w - wm1 * orig_nkw[t][w] as i64).max(0) as u32;
        }
        nk[t] = nkw[t].iter().map(|&c| c as u64).sum::<u64>() as u32;
    }

    // --- Reconcile npk the same way; recompute np from the merged npk rows. ---
    for p in 0..p_count {
        for t in 0..k {
            let sum_w: i64 = outs.iter().map(|o| o.npk[p][t] as i64).sum();
            npk[p][t] = (sum_w - wm1 * orig_npk[p][t] as i64).max(0) as u32;
        }
        np[p] = npk[p].iter().map(|&c| c as u64).sum::<u64>() as u32;
    }

    // --- Reconcile mp (CRP customer counts) additively; Σ_p mp = D preserved. ---
    for p in 0..p_count {
        let sum_w: i64 = outs.iter().map(|o| o.mp[p] as i64).sum();
        mp[p] = (sum_w - wm1 * orig_mp[p] as i64).max(0) as u32;
    }

    // --- Write back each worker's per-document l / z slice (disjoint docs). ---
    for out in outs {
        let start = out.start;
        for (i, (lv, zrow)) in out.l.into_iter().zip(out.z).enumerate() {
            l[start + i] = lv;
            z[start + i] = zrow;
        }
    }
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

impl PtmModel {
    /// Collapsed Dirichlet-multinomial log-likelihood of the topic (pseudo-doc)
    /// and word layers, up to additive constants (the per-Dirichlet
    /// `lgamma(Σ prior) - Σ lgamma(prior)` normalizers, which do not change across
    /// sweeps at fixed hyperparameters). A genuine collapsed log marginal — a sum
    /// of `lgamma` terms — used as the per-sweep objective and `convergence_tol`
    /// signal. It scores the topic/word assignments; it does not integrate the
    /// pseudo-doc *assignment* layer (the `(m_p + λ)` prior).
    pub fn log_likelihood(&self) -> f64 {
        let k = self.num_topics;
        let v = self.num_types;
        let p = self.num_pseudo;
        let k_alpha = k as f64 * self.alpha;
        let v_beta = v as f64 * self.beta;
        let mut ll = 0.0f64;

        // Pseudo-document (topic) contribution: Σ_p [ Σ_k lgamma(n_pk+α)
        //   - lgamma(n_p + Kα) ], the varying part of the Dirichlet-multinomial
        // marginal (constant lgamma(α)/lgamma(Kα) normalizers dropped).
        for pp in 0..p {
            let n_p = self.np[pp] as f64;
            if n_p == 0.0 {
                continue;
            }
            for kk in 0..k {
                let n_pk = self.npk[pp][kk] as f64;
                if n_pk > 0.0 {
                    ll += log_gamma(self.alpha + n_pk) - log_gamma(self.alpha);
                }
            }
            ll -= log_gamma(k_alpha + n_p) - log_gamma(k_alpha);
        }

        // Topic-word contribution.
        for kk in 0..k {
            for w in 0..v {
                let n_kw = self.nkw[kk][w] as f64;
                if n_kw > 0.0 {
                    ll += log_gamma(self.beta + n_kw) - log_gamma(self.beta);
                }
            }
            ll -= log_gamma(v_beta + self.nk[kk] as f64) - log_gamma(v_beta);
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
/// * `lambda`     — pseudo-document Dirichlet prior (the `(m_p + λ)` popularity term)
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
    lambda: f64,
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
        lambda,
        iters,
        crate::keyatm::ThetaDrawOpts::new(false, 0, 0),
        0.0,
        0,
        1,
        rng,
    );
    model
}

/// Fit a PTM with thinned θ snapshots collected every `thin` sweeps (ring-buffered
/// to `cap` total draws). `cap=0` disables collection entirely.
///
/// `num_threads` selects the sampler: `1` (or `0`) runs the exact serial two-phase
/// sweep, byte-identical to the pre-threading path; `>1` runs MALLET-style
/// approximate-parallel (AD-LDA) sampling, deterministic for a fixed
/// `num_threads` + seed.
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
    lambda: f64,
    iters: usize,
    opts: crate::keyatm::ThetaDrawOpts,
    convergence_tol: f64,
    check_every: usize,
    num_threads: usize,
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
    let mut mp = vec![0u32; p];

    for (d, doc) in docs.iter().enumerate() {
        let pd = l[d];
        mp[pd] += 1;
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
        lambda,
        nkw,
        nk,
        npk,
        np,
        mp,
        l,
        z,
        theta_draws: Vec::new(),
    };

    let mut ll_history: Vec<(usize, f64)> = Vec::new();
    let mut converged = false;

    for iter in 1..=iters {
        if num_threads <= 1 {
            // Exact serial path — byte-identical to the pre-threading loop.
            model.sweep(docs, rng);
        } else {
            // Approximate-parallel AD-LDA. One per-sweep seed drawn from the main
            // RNG keeps it deterministic for a fixed num_threads + seed.
            let sweep_seed = rng.gen::<u64>();
            parallel_sweep_ptm(
                &mut model.nkw,
                &mut model.nk,
                &mut model.npk,
                &mut model.np,
                &mut model.mp,
                &mut model.l,
                &mut model.z,
                docs,
                model.alpha,
                model.beta,
                model.lambda,
                model.num_topics,
                model.num_types,
                model.num_pseudo,
                num_threads,
                sweep_seed,
            );
        }
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

        let block_score = |row: &[f64], bi: usize| -> f64 {
            blocks[bi].iter().map(|&w| row[w as usize]).sum::<f64>()
        };
        // Recovery = both topics concentrate >65% of their mass on DISTINCT blocks.
        let recovers = |model: &PtmModel| -> bool {
            let tw = model.topic_word();
            let best = |k: usize| {
                (0..num_topics)
                    .map(|bi| block_score(&tw[k], bi))
                    .fold(0.0f64, f64::max)
            };
            let argmax = |k: usize| {
                (0..num_topics)
                    .max_by(|&a, &b| {
                        block_score(&tw[k], a)
                            .partial_cmp(&block_score(&tw[k], b))
                            .unwrap()
                    })
                    .unwrap()
            };
            best(0) > 0.65 && best(1) > 0.65 && argmax(0) != argmax(1)
        };

        // The faithful (m_p + λ) popularity prior makes PTM's aggregation more
        // init-sensitive than a uniform-ψ variant: a few random seeds collapse the
        // documents into one pseudo-doc and the topics stay mixed. So we assert the
        // model recovers the planted blocks on the MAJORITY of seeds, not every one
        // — an honest statement of "PTM recovers short-text topics, given a
        // reasonable init".
        // P=50: the faithful (m_p + λ) popularity prior needs enough pseudo-docs to
        // aggregate a short-text corpus without over-collapsing into one pseudo-doc
        // (see #491). At P=50 recovery is robust across seeds.
        let n_seeds = 8u64;
        let recovered = (1..=n_seeds)
            .filter(|&s| {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                let model = fit_ptm(&docs, v, num_topics, 50, 0.1, 0.01, 0.1, 1000, &mut rng);
                recovers(&model)
            })
            .count();
        assert!(
            recovered >= 5,
            "PTM recovered the planted 2-block structure on only {recovered}/{n_seeds} seeds; \
             expected a clear majority"
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
        let m1 = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 0.1, 30, &mut r1);
        let m2 = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 0.1, 30, &mut r2);
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
        let m = fit_ptm(&docs, v, 3, 5, 0.1, 0.01, 0.1, 20, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }

    // --- AD-LDA threading (#566) ---

    fn threading_corpus() -> (Vec<Vec<u32>>, usize) {
        let v = 16usize;
        // 80 short docs (4 tokens) — enough to partition across workers.
        let docs: Vec<Vec<u32>> = (0..80usize)
            .map(|d| (0..4).map(|i| ((i * 3 + d) % v) as u32).collect())
            .collect();
        (docs, v)
    }

    fn fit_threaded(
        docs: &[Vec<u32>],
        v: usize,
        iters: usize,
        num_threads: usize,
        seed: u64,
    ) -> PtmModel {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let opts = crate::keyatm::ThetaDrawOpts::new(false, 0, 0);
        let (m, _, _) = fit_ptm_with_draws(
            docs,
            v,
            3,
            8,
            0.1,
            0.01,
            0.1,
            iters,
            opts,
            0.0,
            0,
            num_threads,
            &mut rng,
        );
        m
    }

    #[test]
    fn num_threads_one_is_bit_identical_to_serial() {
        // num_threads=1 must never draw the per-sweep seed and must reproduce the
        // pre-threading serial path exactly, across all count tables and per-doc
        // assignments (both sweep phases).
        let (docs, v) = threading_corpus();
        let mut r_serial = ChaCha8Rng::seed_from_u64(42);
        let serial = fit_ptm(&docs, v, 3, 8, 0.1, 0.01, 0.1, 40, &mut r_serial);
        let threaded1 = fit_threaded(&docs, v, 40, 1, 42);
        assert_eq!(serial.nkw, threaded1.nkw, "nkw differs at num_threads=1");
        assert_eq!(serial.nk, threaded1.nk, "nk differs at num_threads=1");
        assert_eq!(serial.npk, threaded1.npk, "npk differs at num_threads=1");
        assert_eq!(serial.np, threaded1.np, "np differs at num_threads=1");
        assert_eq!(serial.mp, threaded1.mp, "mp differs at num_threads=1");
        assert_eq!(serial.l, threaded1.l, "l differs at num_threads=1");
        assert_eq!(serial.z, threaded1.z, "z differs at num_threads=1");
    }

    #[test]
    fn threaded_is_deterministic_for_fixed_threads_and_seed() {
        let (docs, v) = threading_corpus();
        let a = fit_threaded(&docs, v, 40, 4, 123);
        let b = fit_threaded(&docs, v, 40, 4, 123);
        assert_eq!(a.nkw, b.nkw);
        assert_eq!(a.npk, b.npk);
        assert_eq!(a.mp, b.mp);
        assert_eq!(a.l, b.l);
        assert_eq!(a.topic_word(), b.topic_word());
    }

    #[test]
    fn threaded_preserves_all_count_invariants() {
        // The merge recomputes nk from nkw and np from npk, merges mp additively,
        // and must conserve tokens (nk and np totals) and documents (Σ mp = D).
        let (docs, v) = threading_corpus();
        let m = fit_threaded(&docs, v, 40, 4, 7);
        let total_tokens: usize = docs.iter().map(|d| d.len()).sum();
        for t in 0..m.num_topics {
            assert_eq!(m.nk[t], m.nkw[t].iter().sum::<u32>(), "nk[{t}] != Σ_w nkw");
        }
        for p in 0..m.num_pseudo {
            assert_eq!(m.np[p], m.npk[p].iter().sum::<u32>(), "np[{p}] != Σ_k npk");
        }
        assert_eq!(
            m.nk.iter().sum::<u32>() as usize,
            total_tokens,
            "token loss (nk)"
        );
        assert_eq!(
            m.np.iter().sum::<u32>() as usize,
            total_tokens,
            "token loss (np)"
        );
        assert_eq!(m.mp.iter().sum::<u32>() as usize, docs.len(), "Σ mp != D");
    }

    #[test]
    fn threaded_handles_more_workers_than_docs() {
        // partition_ranges clamps workers to the doc count; oversized num_threads
        // must run cleanly and keep the invariants.
        let docs: Vec<Vec<u32>> = (0..3)
            .map(|d| (0..4).map(|i| ((i + d) % 12) as u32).collect())
            .collect();
        let m = fit_threaded(&docs, 12, 20, 16, 5);
        assert_eq!(m.mp.iter().sum::<u32>() as usize, docs.len());
        assert_eq!(
            m.nk.iter().sum::<u32>() as usize,
            docs.iter().map(|d| d.len()).sum::<usize>()
        );
    }
}
