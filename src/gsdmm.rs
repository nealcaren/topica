//! Gibbs Sampling Dirichlet Multinomial Mixture (GSDMM) model, a.k.a. the
//! Movie Group Process (MGP), for short-text clustering.
//!
//! Yin & Wang (2014), "A Dirichlet Multinomial Mixture Model-based Approach
//! for Short Text Clustering", KDD 2014.
//!
//! Unlike LDA, **each document belongs to exactly one cluster** — there is no
//! per-document topic mixture. This makes the model much better suited to
//! very short texts (tweets, survey responses, headlines) where a per-document
//! distribution cannot be estimated reliably from only a handful of tokens.
//!
//! ## Algorithm (collapsed Gibbs / Movie Group Process)
//!
//! Latent state:
//! - `z[d]`    — cluster assignment of document d (one integer per doc)
//! - `m[k]`    — number of documents assigned to cluster k
//! - `n[k]`    — total word tokens in cluster k
//! - `nw[k][w]`— count of word type w in cluster k
//!
//! Sampling probability (Yin-Wang Eq. 4, log-space):
//!
//! ```text
//! p(z_d = k) ∝ (m[k] + α)
//!              × Π_{w, j=1..c_{dw}} (nw[k][w] + β + j − 1)
//!              / Π_{i=1..N_d}       (n[k]  + V·β + i − 1)
//! ```
//!
//! where `c_{dw}` is the count of word `w` in document `d` and `N_d` is the
//! total token count of document `d`.  The denominator of the document-cluster
//! prior `(D − 1 + K·α)` is constant across clusters and is omitted.
//!
//! Note on the popular `rwalk/gsdmm` implementation: it assumes deduplicated
//! documents and omits the `+ j − 1` rising-factorial term in the numerator
//! (its `j` is always 1). The two therefore agree on unique-token documents and
//! diverge only when a document repeats a token, where this port follows the
//! paper — scoring a twice-seen word as `(nw+β)(nw+β+1)` rather than `(nw+β)²`.

use rand::Rng;

/// Distinct word ids of `doc` with their within-document counts, in **ascending
/// word-id order**.
///
/// Iterating this reproduces the exact accumulation order of a dense `0..V` scan
/// that skips zero counts, so the numerator log-sums stay bit-for-bit identical
/// while dropping the per-document `O(V)` allocation and full-vocabulary scan.
/// Short-text documents touch only a handful of the `V` types, so the paper's
/// `O(D·K·L̄)` complexity is restored without changing any result (#717).
fn word_counts_sorted(doc: &[u32]) -> Vec<(usize, u32)> {
    let mut ws: Vec<u32> = doc.to_vec();
    ws.sort_unstable();
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(ws.len());
    for &w in &ws {
        match out.last_mut() {
            Some(last) if last.0 == w as usize => last.1 += 1,
            _ => out.push((w as usize, 1)),
        }
    }
    out
}

/// Unnormalized log `p(z_d = k)` for every cluster `k` (Yin-Wang Eq. 4), given
/// the current counts and this document's sorted `(word, count)` list.
///
/// Two bit-exact speedups over the naive per-cluster/per-word double loop (#781):
/// - **Empty-cluster memoization.** Every cluster with `m[k]==0` has `n[k]==0` and
///   all `nw[k][*]==0`, so its log-prob is a single value that does not depend on
///   `k`. GSDMM keeps `k_max` clusters and empties most of them, so this collapses
///   the dominant cost from `O(k_max)` to `O(#non-empty)` per document.
/// - **`ln(beta)` caching.** For a word absent from cluster `k` (`nw[k][w]==0`) the
///   `j=0` numerator term is `ln(0+beta)=ln(beta)`; most (word, cluster) pairs are
///   absent, so this skips the majority of the numerator's `ln` calls.
///
/// Both reuse values that are bit-identical to recomputing them inline (`0.0+x==x`
/// exactly), so the sampler draws and the committed gold are unchanged.
fn cluster_log_probs(
    m: &[u32],
    n: &[u32],
    nw: &[Vec<u32>],
    wc: &[(usize, u32)],
    n_d: u32,
    alpha: f64,
    beta: f64,
    vbeta: f64,
) -> Vec<f64> {
    let k = m.len();
    let ln_beta = beta.ln();
    let mut empty_lp: Option<f64> = None;
    let mut out = vec![0.0f64; k];
    for kk in 0..k {
        if m[kk] == 0 {
            // Empty cluster: n==0, nw==0 for all words. Same value for every empty
            // cluster, so compute once and reuse. This memoization is load-bearing
            // on the count invariant (m/n/nw are only ever mutated together, one
            // doc at a time), so guard it: n[k]==0 with non-negative counts implies
            // every nw[k][w]==0, so the memoized template is exact.
            debug_assert_eq!(
                n[kk], 0,
                "empty cluster {kk} (m=0) has nonzero token count n={} — the \
                 empty-cluster memoization would be wrong",
                n[kk]
            );
            out[kk] = *empty_lp.get_or_insert_with(|| {
                let mut lp = alpha.ln();
                for &(_w, cw) in wc {
                    lp += ln_beta; // j=0: ln(0+beta)
                    for j in 1..cw {
                        lp += (beta + j as f64).ln();
                    }
                }
                for i in 0..n_d {
                    lp -= (vbeta + i as f64).ln();
                }
                lp
            });
            continue;
        }
        let mut lp = (m[kk] as f64 + alpha).ln();
        for &(w, cw) in wc {
            let nwk = nw[kk][w];
            if nwk == 0 {
                lp += ln_beta; // j=0: ln(0+beta), cached
                for j in 1..cw {
                    lp += (beta + j as f64).ln();
                }
            } else {
                let base = nwk as f64 + beta;
                for j in 0..cw {
                    lp += (base + j as f64).ln();
                }
            }
        }
        let base_d = n[kk] as f64 + vbeta;
        for i in 0..n_d {
            lp -= (base_d + i as f64).ln();
        }
        out[kk] = lp;
    }
    out
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Fitted GSDMM model.
///
/// Stores the final Gibbs state; query it with the provided methods.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GsdmmModel {
    /// Vocabulary size V (number of distinct word types).
    pub num_types: usize,
    /// Maximum number of clusters K (the "restaurant capacity").
    pub k_max: usize,
    /// Dirichlet prior on document-cluster assignments (α).
    pub alpha: f64,
    /// Dirichlet prior on cluster-word distributions (β).
    pub beta: f64,
    /// `m[k]` — number of documents assigned to cluster k.
    pub m: Vec<u32>,
    /// `n[k]` — total word tokens in cluster k.
    pub n: Vec<u32>,
    /// `nw[k][w]` — count of word type w in cluster k. Shape: K × V.
    pub nw: Vec<Vec<u32>>,
    /// `z[d]` — final cluster assignment for each document.
    pub z: Vec<usize>,
    /// Discovery/convergence trace, one entry per recorded sweep:
    /// `(iteration, num_non_empty_clusters, per-token log-likelihood)`. The
    /// cluster count collapsing to a stable value is the Movie Group Process's
    /// headline convergence check.
    pub trace: Vec<(usize, usize, f64)>,
}

impl GsdmmModel {
    /// Number of non-empty clusters after fitting (the "effective K").
    pub fn num_clusters(&self) -> usize {
        self.m.iter().filter(|&&c| c > 0).count()
    }

    /// Per-token log-likelihood of each document under its assigned cluster:
    /// `(1/N) Σ_d Σ_{w∈d} log φ_{z_d, w}`, with
    /// `φ_{k,w} = (nw[k][w]+β)/(n[k]+Vβ)`. Returns `NaN` for an empty corpus.
    pub fn cluster_log_likelihood(&self, docs: &[Vec<u32>]) -> f64 {
        let vbeta = self.num_types as f64 * self.beta;
        let mut ll = 0.0f64;
        let mut ntok = 0usize;
        for (d, doc) in docs.iter().enumerate() {
            let k = self.z[d];
            let denom = self.n[k] as f64 + vbeta;
            for &w in doc {
                let p = (self.nw[k][w as usize] as f64 + self.beta) / denom;
                ll += p.max(1e-300).ln();
                ntok += 1;
            }
        }
        if ntok == 0 {
            f64::NAN
        } else {
            ll / ntok as f64
        }
    }

    /// Indices of the non-empty clusters, in ascending order.
    pub fn used_clusters(&self) -> Vec<usize> {
        self.m
            .iter()
            .enumerate()
            .filter_map(|(k, &c)| if c > 0 { Some(k) } else { None })
            .collect()
    }

    /// Smoothed word distribution for cluster k:
    /// φ_{k,w} = (nw[k][w] + β) / (n[k] + V·β).
    ///
    /// Length = `num_types`; values sum to 1.
    pub fn cluster_word(&self, k: usize) -> Vec<f64> {
        let v = self.num_types;
        let denom = self.n[k] as f64 + v as f64 * self.beta;
        self.nw[k]
            .iter()
            .map(|&c| (c as f64 + self.beta) / denom)
            .collect()
    }

    /// Hard cluster assignment of every document.
    ///
    /// Length = D; values in `0..k_max`.
    pub fn doc_cluster(&self) -> Vec<usize> {
        self.z.clone()
    }

    /// Per-document posterior cluster probability vector (Yin & Wang Eq. 4).
    ///
    /// This is an **in-sample plug-in** estimate: the document is NOT held out
    /// before scoring, so its own tokens still inflate its current cluster. The
    /// result is therefore over-peaked and can disagree with the hard
    /// `doc_cluster` on the exact ordering — use `doc_cluster` for the label and
    /// treat this as a soft-confidence view, not a leave-one-out posterior.
    ///
    /// Shape: D × k_max; each inner vec sums to 1.
    pub fn doc_cluster_dist(&self, docs: &[Vec<u32>]) -> Vec<Vec<f64>> {
        let v = self.num_types;
        let vbeta = v as f64 * self.beta;

        docs.iter()
            .map(|doc| {
                // Distinct (word, count) in ascending word order — same log-sum
                // order as a dense 0..V scan, but O(distinct) not O(V) (#717).
                let wc = word_counts_sorted(doc);
                let n_d = doc.len() as u32;
                let log_probs = cluster_log_probs(
                    &self.m, &self.n, &self.nw, &wc, n_d, self.alpha, self.beta, vbeta,
                );

                // Stable softmax.
                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
                let total: f64 = probs.iter().sum();
                for p in &mut probs {
                    *p /= total;
                }
                probs
            })
            .collect()
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

/// Resample the cluster assignment of document `d` in-place (Movie Group
/// Process step).  The document is removed from its current cluster before
/// computing the sampling probabilities and is added to the new cluster
/// afterwards.
fn resample_doc<R: Rng>(model: &mut GsdmmModel, d: usize, doc: &[u32], rng: &mut R) {
    let v = model.num_types;
    let vbeta = v as f64 * model.beta;
    let alpha = model.alpha;

    let z_old = model.z[d];
    let n_d = doc.len() as u32;

    // --- Remove document d from its current cluster ---
    model.m[z_old] -= 1;
    model.n[z_old] -= n_d;
    for &w in doc {
        model.nw[z_old][w as usize] -= 1;
    }

    // --- Compute unnormalised log-probabilities for each cluster ---
    // Distinct (word, count) for this doc in ascending word order: the same
    // numerator accumulation order as a dense 0..V scan (so results stay
    // bit-identical), but O(distinct words) rather than O(V) per cluster (#717).
    // `cluster_log_probs` additionally memoizes the shared empty-cluster value and
    // caches ln(beta) for absent words (#781) — still bit-identical.
    let wc = word_counts_sorted(doc);
    let log_probs = cluster_log_probs(
        &model.m, &model.n, &model.nw, &wc, n_d, alpha, model.beta, vbeta,
    );

    // --- Stable softmax then categorical sample ---
    let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
    let z_new = sample_index(&probs, rng);

    // --- Add document d to the new cluster ---
    model.m[z_new] += 1;
    model.n[z_new] += n_d;
    for &w in doc {
        model.nw[z_new][w as usize] += 1;
    }
    model.z[d] = z_new;
}

/// One full Gibbs sweep: resample every document's cluster assignment.
fn sweep<R: Rng>(model: &mut GsdmmModel, docs: &[Vec<u32>], rng: &mut R) {
    for d in 0..docs.len() {
        resample_doc(model, d, &docs[d], rng);
    }
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

/// Fit a GSDMM (Movie Group Process) model by collapsed Gibbs sampling.
///
/// # Arguments
/// * `docs`      — corpus; each document is a list of word ids in `0..num_types`
///                 (tokens may repeat within a document).
/// * `num_types` — vocabulary size V
/// * `k_max`     — maximum number of clusters K (some will collapse to empty)
/// * `alpha`     — Dirichlet prior on document-cluster assignments (α ≈ 0.1)
/// * `beta`      — Dirichlet prior on cluster-word distributions   (β ≈ 0.1)
/// * `iters`     — number of full Gibbs sweeps
/// * `rng`       — random-number source; determines all randomness (deterministic
///                 for a fixed seed)
#[allow(clippy::too_many_arguments)]
pub fn fit_gsdmm<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    k_max: usize,
    alpha: f64,
    beta: f64,
    iters: usize,
    report_interval: usize,
    rng: &mut R,
) -> GsdmmModel {
    let d_count = docs.len();
    let k = k_max;
    let v = num_types;

    // --- Initialisation: assign each document to a uniformly random cluster ---
    let z: Vec<usize> = (0..d_count)
        .map(|_| (rng.gen::<f64>() * k as f64) as usize % k)
        .collect();

    let mut m = vec![0u32; k];
    let mut n = vec![0u32; k];
    let mut nw = vec![vec![0u32; v]; k];

    for (d, doc) in docs.iter().enumerate() {
        let kk = z[d];
        m[kk] += 1;
        n[kk] += doc.len() as u32;
        for &w in doc {
            nw[kk][w as usize] += 1;
        }
    }

    let mut model = GsdmmModel {
        num_types: v,
        k_max: k,
        alpha,
        beta,
        m,
        n,
        nw,
        z,
        trace: Vec::new(),
    };

    for it in 0..iters {
        sweep(&mut model, docs, rng);
        if report_interval > 0 && ((it + 1) % report_interval == 0 || it + 1 == iters) {
            let ll = model.cluster_log_likelihood(docs);
            model.trace.push((it + 1, model.num_clusters(), ll));
        }
    }

    model
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for GsdmmModel {
    fn num_topics(&self) -> usize {
        self.k_max
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        (0..self.k_max).map(|k| self.cluster_word(k)).collect()
    }

    // The generic Estimator surface returns the HARD assignment as a one-hot over
    // `k_max` (what composition/conformance machinery expects for a single-membership
    // model). This intentionally differs from the Python `doc_topic` getter, which
    // exposes the SOFT in-sample Eq. 4 conditional remapped to the used clusters —
    // the same distinction as `doc_cluster` (hard) vs the soft scores (#490).
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.z
            .iter()
            .map(|&k| {
                let mut v = vec![0.0f64; self.k_max];
                v[k] = 1.0;
                v
            })
            .collect()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.trace.iter().map(|&(it, _, ll)| (it, ll)).collect()
    }

    fn converged(&self) -> Option<bool> {
        None
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a corpus of short docs (2–4 tokens each) drawn from 3 disjoint
    /// vocabulary blocks, then verify that GSDMM collapses to roughly the
    /// planted number of clusters and recovers the block structure.
    #[test]
    fn recovers_clusters_from_short_docs() {
        // 3 blocks of 10 words each → V = 30.
        let num_blocks = 3usize;
        let block_size = 10usize;
        let v = num_blocks * block_size;

        // 300 docs, 100 per block, each 3 tokens cycling through the block words.
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for b in 0..num_blocks {
            let offset = (b * block_size) as u32;
            for d in 0..100usize {
                let doc: Vec<u32> = (0..3)
                    .map(|i| offset + ((i + d) % block_size) as u32)
                    .collect();
                docs.push(doc);
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // k_max=10, expect it to collapse toward 3 non-empty clusters.
        let model = fit_gsdmm(&docs, v, 10, 0.1, 0.1, 200, 0, &mut rng);

        let nc = model.num_clusters();
        // MGP may over- or under-cluster a bit; just assert it is in a sane range.
        assert!(
            nc >= num_blocks && nc <= model.k_max,
            "expected roughly {num_blocks} clusters but got {nc}"
        );

        // Verify the top words of the three largest clusters fall in distinct blocks.
        let used = model.used_clusters();
        // Sort by cluster size descending.
        let mut used_sorted = used.clone();
        used_sorted.sort_by(|&a, &b| model.m[b].cmp(&model.m[a]));
        let top3: Vec<usize> = used_sorted.into_iter().take(num_blocks).collect();

        // For each of those clusters find the dominant block.
        let dominant_block = |k: usize| -> usize {
            let phi = model.cluster_word(k);
            // Sum φ over each block and pick the best.
            (0..num_blocks)
                .max_by(|&ba, &bb| {
                    let sa: f64 = (0..block_size).map(|i| phi[ba * block_size + i]).sum();
                    let sb: f64 = (0..block_size).map(|i| phi[bb * block_size + i]).sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap()
        };

        let mut assigned_blocks: Vec<usize> = top3.iter().map(|&k| dominant_block(k)).collect();
        assigned_blocks.sort_unstable();
        assigned_blocks.dedup();
        assert_eq!(
            assigned_blocks.len(),
            num_blocks,
            "top clusters do not span all {num_blocks} planted blocks; \
             dominant blocks: {assigned_blocks:?}"
        );
    }

    /// Two fits with the same seed must be bit-for-bit identical.
    #[test]
    fn deterministic_for_fixed_seed() {
        let v = 20usize;
        let docs: Vec<Vec<u32>> = (0..80usize)
            .map(|d| (0..3).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let mut r1 = ChaCha8Rng::seed_from_u64(99);
        let mut r2 = ChaCha8Rng::seed_from_u64(99);
        let m1 = fit_gsdmm(&docs, v, 8, 0.1, 0.1, 50, 0, &mut r1);
        let m2 = fit_gsdmm(&docs, v, 8, 0.1, 0.1, 50, 0, &mut r2);

        assert_eq!(
            m1.doc_cluster(),
            m2.doc_cluster(),
            "doc_cluster() differs between two identical-seed runs"
        );
        // Compare cluster_word for every used cluster.
        for &k in m1.used_clusters().iter() {
            assert_eq!(
                m1.cluster_word(k),
                m2.cluster_word(k),
                "cluster_word({k}) differs between two identical-seed runs"
            );
        }
    }

    /// Shape and normalisation invariants.
    #[test]
    fn shape_and_normalisation() {
        let v = 12usize;
        let docs: Vec<Vec<u32>> = (0..40usize)
            .map(|d| (0..2).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let model = fit_gsdmm(&docs, v, 6, 0.1, 0.1, 30, 0, &mut rng);

        // doc_cluster() has length D.
        assert_eq!(
            model.doc_cluster().len(),
            docs.len(),
            "doc_cluster() length should equal number of documents"
        );

        // cluster_word(k) sums to 1 for every non-empty cluster.
        for &k in model.used_clusters().iter() {
            let phi = model.cluster_word(k);
            assert_eq!(
                phi.len(),
                v,
                "cluster_word({k}) length should equal num_types"
            );
            let s: f64 = phi.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "cluster_word({k}) sums to {s:.12}, expected 1.0"
            );
        }

        // doc_cluster_dist() has shape D × k_max with rows summing to 1.
        let dists = model.doc_cluster_dist(&docs);
        assert_eq!(dists.len(), docs.len());
        for (d, row) in dists.iter().enumerate() {
            assert_eq!(
                row.len(),
                model.k_max,
                "doc_cluster_dist row {d} wrong length"
            );
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "doc_cluster_dist row {d} sums to {s:.12}, expected 1.0"
            );
        }
    }

    /// The sparse `word_counts_sorted` numerator must reproduce a dense `0..V`
    /// scan bit-for-bit, including repeated tokens (rising factorial) (#717 F11).
    #[test]
    fn doc_cluster_dist_matches_dense_reference() {
        let v = 15usize;
        // Include a doc with a thrice-repeated token to exercise the rising factorial.
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 2],
            vec![3, 3, 3, 4],
            vec![5, 6, 5, 7, 6],
            vec![8, 9],
            vec![10, 11, 12, 13, 14, 0],
        ];
        // K=12 on 5 docs guarantees many empty clusters, exercising the
        // empty-cluster memoization against the naive dense reference below (#781).
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let model = fit_gsdmm(&docs, v, 12, 0.1, 0.1, 40, 0, &mut rng);

        // Independent dense reference: full 0..V scan, skipping zero counts.
        let dense = |doc: &[u32]| -> Vec<f64> {
            let k = model.k_max;
            let vbeta = v as f64 * model.beta;
            let mut wc = vec![0u32; v];
            for &w in doc {
                wc[w as usize] += 1;
            }
            let n_d = doc.len() as u32;
            let mut lps = vec![0.0f64; k];
            for kk in 0..k {
                let mut lp = (model.m[kk] as f64 + model.alpha).ln();
                for (w, &cw) in wc.iter().enumerate() {
                    if cw == 0 {
                        continue;
                    }
                    let base = model.nw[kk][w] as f64 + model.beta;
                    for j in 0..cw {
                        lp += (base + j as f64).ln();
                    }
                }
                let base_d = model.n[kk] as f64 + vbeta;
                for i in 0..n_d {
                    lp -= (base_d + i as f64).ln();
                }
                lps[kk] = lp;
            }
            let max_lp = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs: Vec<f64> = lps.iter().map(|&lp| (lp - max_lp).exp()).collect();
            let total: f64 = probs.iter().sum();
            for p in &mut probs {
                *p /= total;
            }
            probs
        };

        let got = model.doc_cluster_dist(&docs);
        for (d, doc) in docs.iter().enumerate() {
            let want = dense(doc);
            for (a, b) in got[d].iter().zip(want.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "doc {d}: sparse != dense");
            }
        }
    }

    #[test]
    fn gsdmm_conforms() {
        let num_blocks = 3usize;
        let block_size = 10usize;
        let v = num_blocks * block_size;
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for b in 0..num_blocks {
            let offset = (b * block_size) as u32;
            for d in 0..100usize {
                let doc: Vec<u32> = (0..3)
                    .map(|i| offset + ((i + d) % block_size) as u32)
                    .collect();
                docs.push(doc);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = fit_gsdmm(&docs, v, 10, 0.1, 0.1, 200, 0, &mut rng);
        let base = crate::conformance::check_conformance(&model);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
