//! CSATM: Conversational Structure Aware and Context Sensitive Topic Model for
//! online discussions (Sun, Loparo & Kolacinski, IEEE ICSC 2020,
//! arXiv:2002.02353).
//!
//! CSATM is a weighted collapsed-Gibbs LDA for threaded forum data (posts +
//! nested reply trees, one comment per document). It layers two thread-structure
//! constructs onto LDA:
//!
//! * **Popularity** `p_c` (paper §III-A, eq. 1) — a per-comment weight equal to
//!   the level-weighted node count of the comment's reply subtree, so comments
//!   that attract many replies carry more weight in inference. Each token of
//!   comment `c` contributes weight `λ·p_c` (not 1) to the count tables — the
//!   "weighted LDA" pattern also used by `keyatm.rs` (there the weight is
//!   per-word; here it is per-comment).
//! * **Transitivity** (paper §III-C) — a post-Gibbs smoothing of each comment's
//!   topic distribution toward its ancestors along the root path, with weight
//!   decaying by distance (nearer ancestor = stronger; paper's intuition 2).
//!
//! Faithfulness notes (Gate-A dual review, see /private/tmp/csatm-spec.md):
//! * Popularity is the literal **absolute per-level** sum `p_i = Σ_l w_l·n_l(i)`
//!   (`n_l(i)` = node count at relative level `l` in `i`'s subtree, node `i`
//!   itself at level 1). A scalar compounding recursion would only match eq. 1
//!   for the geometric sequence, not the paper's arithmetic default.
//! * The count tables hold the **λ·p_c-weighted** counts; the `λ·p_c` factor in
//!   the paper's conditional/estimators is that already-baked-in weight, applied
//!   once (no second multiplication).
//! * With `λ=1` and every `p_c=1` (a flat corpus / all roots) and self-only
//!   transitivity, CSATM reduces exactly to ordinary LDA(α, β).

use crate::corpus::Corpus;
use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use rand::Rng;

/// Decreasing level-weight sequence `w_l` (paper §III-A), shared by the
/// popularity score and the transitivity smoothing. `level` is 1-based.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightSeq {
    /// `w_l = max(c - (l-1)·d, 0)`. The paper's default for sparse corpora
    /// (sharpest fall-off); clamped at 0 so deep levels never go negative.
    Arithmetic { c: f64, d: f64 },
    /// `w_l = c · r^(l-1)`.
    Geometric { c: f64, r: f64 },
    /// `w_l = (c + (l-1)·b)^(-g)` — harmonic progression with "gravity" power `g`.
    Harmonic { c: f64, b: f64, g: f64 },
}

impl WeightSeq {
    /// Weight for 1-based `level` (level 1 = the node itself in its own subtree,
    /// or the deepest node on a transitivity path).
    pub fn weight(&self, level: usize) -> f64 {
        let l = level as f64;
        match *self {
            WeightSeq::Arithmetic { c, d } => (c - (l - 1.0) * d).max(0.0),
            WeightSeq::Geometric { c, r } => c * r.powf(l - 1.0),
            WeightSeq::Harmonic { c, b, g } => (c + (l - 1.0) * b).powf(-g),
        }
    }
}

impl Default for WeightSeq {
    fn default() -> Self {
        // Arithmetic with c=1 ties the leaf/own-contribution = w_1·1 = 1 invariant
        // (paper: a comment with no reply has popularity 1). `d` is a free choice
        // (the paper gives no numeric value); 0.5 keeps a few levels alive.
        WeightSeq::Arithmetic { c: 1.0, d: 0.5 }
    }
}

/// Hyperparameters for [`fit`].
#[derive(Clone, Debug)]
pub struct CsatmParams {
    pub num_topics: usize,
    /// Symmetric doc-topic Dirichlet prior (paper §IV-A default 0.1).
    pub alpha: f64,
    /// Symmetric topic-word Dirichlet prior (paper §IV-A default 0.01).
    pub beta: f64,
    /// Popularity count-scaling ratio `λ`. Not paper-specified for CSATM; 0.1 is
    /// a documented free choice.
    pub lambda: f64,
    /// Level-weight sequence for popularity and transitivity.
    pub weight: WeightSeq,
}

impl Default for CsatmParams {
    fn default() -> Self {
        CsatmParams {
            num_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            lambda: 0.1,
            weight: WeightSeq::default(),
        }
    }
}

/// Fitted state for [`fit`].
pub struct CSATMModel {
    pub num_topics: usize,
    pub alpha: f64,
    /// φ (K×V), each row sums to 1.
    pub topic_word: Vec<Vec<f64>>,
    /// θ′ (D×K) — the transitivity-smoothed doc-topic distribution, each row sums
    /// to 1. This is the model's `doc_topic`.
    pub doc_topic: Vec<Vec<f64>>,
    /// Raw Gibbs θ (D×K) before transitivity smoothing, each row sums to 1. Kept
    /// for inspection (`doc_topic_raw`).
    pub doc_topic_raw: Vec<Vec<f64>>,
    /// Per-comment popularity `p_c` (diagnostic).
    pub popularity: Vec<f64>,
    /// Per-document token counts.
    pub doc_lengths: Vec<usize>,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// Compute popularity `p_i` for every node from the reply forest, as the literal
/// absolute per-level sum of eq. 1: `p_i = Σ_l w_l · n_l(i)`, where `n_l(i)` is
/// the number of nodes at relative level `l` (1-based, node `i` itself at level
/// 1) within `i`'s subtree.
///
/// `parents[i]` is the document index of `i`'s parent, or a negative value if `i`
/// is a thread root (a post / a top-level comment with no modeled parent). Any
/// parent index out of range is treated as "no parent".
fn popularity_scores(parents: &[i64], weight: &WeightSeq) -> Vec<f64> {
    let n = parents.len();
    // Children adjacency.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, &p) in parents.iter().enumerate() {
        if p >= 0 && (p as usize) < n && (p as usize) != i {
            children[p as usize].push(i);
        }
    }
    // Per-node depth-count vector: counts[i][l-1] = number of nodes at relative
    // level l in i's subtree. Built bottom-up via an explicit post-order stack so
    // deep threads never overflow the call stack. Trees only (parents form a
    // forest); a stray cycle is broken by the visited guard.
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    for start in 0..n {
        // Roots (and any node whose parent link is not a real parent) seed a DFS.
        let p = parents[start];
        let is_root = !(p >= 0 && (p as usize) < n && (p as usize) != start);
        if !is_root {
            continue;
        }
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            order.push(node);
            for &c in &children[node] {
                if !visited[c] {
                    stack.push(c);
                }
            }
        }
    }
    // Any node not reached from a root (e.g. a cycle) still gets its own subtree.
    for node in 0..n {
        if !visited[node] {
            visited[node] = true;
            order.push(node);
        }
    }
    // Process in reverse pre-order so every child precedes its parent.
    let mut counts: Vec<Vec<u64>> = vec![Vec::new(); n];
    for &node in order.iter().rev() {
        let mut vec = vec![1u64]; // the node itself at level 1
        for &c in &children[node] {
            // Child's subtree sits one level deeper under `node`.
            let cv = &counts[c];
            if vec.len() < cv.len() + 1 {
                vec.resize(cv.len() + 1, 0);
            }
            for (l, &cnt) in cv.iter().enumerate() {
                vec[l + 1] += cnt;
            }
        }
        counts[node] = vec;
    }
    // p_i = Σ_l w_l · n_l(i).
    counts
        .iter()
        .map(|vec| {
            vec.iter()
                .enumerate()
                .map(|(l, &cnt)| weight.weight(l + 1) * cnt as f64)
                .sum()
        })
        .collect()
}

/// Depth (1-based level, root = 1) of each node in the reply forest. Currently
/// exercised only by unit tests (the transitivity smoother tracks distance
/// directly), kept as a documented helper.
#[cfg(test)]
fn node_levels(parents: &[i64]) -> Vec<usize> {
    let n = parents.len();
    let mut level = vec![0usize; n];
    // Memoized walk to the root; guard against cycles with a step cap.
    for i in 0..n {
        if level[i] != 0 {
            continue;
        }
        let mut path = Vec::new();
        let mut cur = i;
        let mut steps = 0;
        loop {
            if level[cur] != 0 {
                break;
            }
            let p = parents[cur];
            let has_parent = p >= 0 && (p as usize) < n && (p as usize) != cur;
            path.push(cur);
            if !has_parent {
                level[cur] = 1;
                break;
            }
            steps += 1;
            if steps > n {
                // Cycle fallback: treat as root.
                level[cur] = 1;
                break;
            }
            cur = p as usize;
        }
        // Fill descendants down the path we just walked.
        // `path` is deepest-first from `i` up to (but excluding) the first node
        // that already had a level; `cur` holds that anchor's node.
        let mut base = level[cur];
        for &node in path.iter().rev() {
            if level[node] == 0 {
                base += 1;
                level[node] = base;
            } else {
                base = level[node];
            }
        }
    }
    level
}

/// Fit CSATM. `parents[d]` is document `d`'s parent index (negative = thread
/// root). When `parents` is empty, every document is treated as a root
/// (`p_c` from a singleton subtree), so with `λ=1` the fit reduces to LDA.
pub fn fit<R: Rng>(
    corpus: &Corpus,
    parents: &[i64],
    params: &CsatmParams,
    iters: usize,
    rng: &mut R,
) -> CSATMModel {
    let k = params.num_topics.max(1);
    let d = corpus.num_docs();
    let v = corpus.num_types();
    let docs = &corpus.docs;
    let alpha = params.alpha;
    let beta = params.beta;
    let vbeta = v as f64 * beta;

    // Parent forest: fall back to "all roots" when none supplied.
    let all_roots: Vec<i64>;
    let parents: &[i64] = if parents.is_empty() {
        all_roots = vec![-1; d];
        &all_roots
    } else {
        parents
    };

    // Popularity p_c and the per-document token weight w_d = λ·p_c.
    let popularity = popularity_scores(parents, &params.weight);
    let doc_weight: Vec<f64> = popularity.iter().map(|&p| params.lambda * p).collect();

    // Weighted count tables (f64). Each token of doc `dd` contributes w_d.
    let mut ndk = vec![vec![0.0f64; k]; d]; // doc-topic
    let mut nkw = vec![vec![0.0f64; v]; k]; // topic-word
    let mut nk = vec![0.0f64; k]; // topic totals
    let mut z: Vec<Vec<usize>> = docs.iter().map(|doc| vec![0usize; doc.len()]).collect();

    // Random init, incrementing tables by the document weight.
    for dd in 0..d {
        let w_d = doc_weight[dd];
        for pos in 0..docs[dd].len() {
            let w = docs[dd][pos] as usize;
            let topic = (rng.gen::<f64>() * k as f64) as usize % k;
            z[dd][pos] = topic;
            ndk[dd][topic] += w_d;
            nkw[topic][w] += w_d;
            nk[topic] += w_d;
        }
    }

    // Collapsed Gibbs sweeps. Single-threaded and in fixed (document, position)
    // order, so a fixed seed reproduces bit-for-bit. Counts are decremented and
    // re-incremented by the identical f64 `w_d`; a `.max(0.0)` guards the rare
    // -1e-15 residue from f64 round-off so the CDF never sees a negative mass.
    let mut cond = vec![0.0f64; k];
    for _sweep in 0..iters {
        for dd in 0..d {
            let w_d = doc_weight[dd];
            for pos in 0..docs[dd].len() {
                let w = docs[dd][pos] as usize;
                let old = z[dd][pos];
                ndk[dd][old] = (ndk[dd][old] - w_d).max(0.0);
                nkw[old][w] = (nkw[old][w] - w_d).max(0.0);
                nk[old] = (nk[old] - w_d).max(0.0);

                let mut total = 0.0f64;
                for t in 0..k {
                    // (ñ_{k,c} + α)·(ñ_{k,w} + β)/(Σ_w ñ_{k,w} + Vβ), on the
                    // already-weighted counts (no further ×λp_c).
                    let val = (ndk[dd][t] + alpha) * (nkw[t][w] + beta) / (nk[t] + vbeta);
                    cond[t] = val;
                    total += val;
                }
                // Categorical draw (cumulative-sum + seeded RNG). Guard against a
                // non-finite or non-positive total: with α,β>0 and finite weights
                // `total` is always strictly positive, but if a pathological weight
                // magnitude ever pushed a term to inf/NaN (rejected at construction,
                // but belt-and-suspenders) fall back to a uniform draw rather than
                // silently biasing toward the last topic (which `r <= 0.0` would do
                // when `r` is NaN).
                let new = if total.is_finite() && total > 0.0 {
                    let mut r = rng.gen::<f64>() * total;
                    let mut pick = k - 1;
                    for t in 0..k {
                        r -= cond[t];
                        if r <= 0.0 {
                            pick = t;
                            break;
                        }
                    }
                    pick
                } else {
                    (rng.gen::<f64>() * k as f64) as usize % k
                };

                z[dd][pos] = new;
                ndk[dd][new] += w_d;
                nkw[new][w] += w_d;
                nk[new] += w_d;
            }
        }
    }

    // φ (topic-word), rows normalized by construction: Σ_w (nkw+β) = nk + Vβ.
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|t| {
            let denom = nk[t] + vbeta;
            (0..v).map(|w| (nkw[t][w] + beta) / denom).collect()
        })
        .collect();

    // Raw θ, rows normalized: (ñ_{k,c}+α)/(Σ_k ñ_{k,c}+Kα).
    let ka = k as f64 * alpha;
    let doc_topic_raw: Vec<Vec<f64>> = (0..d)
        .map(|dd| {
            let denom: f64 = ndk[dd].iter().sum::<f64>() + ka;
            ndk[dd].iter().map(|&x| (x + alpha) / denom).collect()
        })
        .collect();

    // Transitivity smoothing (paper §III-C): for each comment i, average the RAW
    // θ of the nodes on the path root→i, weighting node i (distance 0) by w_1,
    // its parent by w_2, …, the root by w_{l_i} — nearer ancestor = stronger.
    let doc_topic = smooth_transitivity(&doc_topic_raw, parents, &params.weight, k);

    let doc_lengths = docs.iter().map(|doc| doc.len()).collect();

    CSATMModel {
        num_topics: k,
        alpha,
        topic_word,
        doc_topic,
        doc_topic_raw,
        popularity,
        doc_lengths,
        fit_history: Vec::new(),
        converged: false,
    }
}

/// Path-weighted average of the raw θ along each comment's root path (eq. in
/// §III-C). Node `i` itself gets `w_1`; each step up toward the root drops one
/// level in the weight sequence.
fn smooth_transitivity(
    theta_raw: &[Vec<f64>],
    parents: &[i64],
    weight: &WeightSeq,
    k: usize,
) -> Vec<Vec<f64>> {
    let n = theta_raw.len();
    (0..n)
        .map(|i| {
            let mut acc = vec![0.0f64; k];
            let mut wsum = 0.0f64;
            let mut cur = i;
            let mut dist = 0usize; // distance from i (0 = i itself)
            let mut steps = 0usize;
            loop {
                let w = weight.weight(dist + 1);
                if w > 0.0 {
                    for t in 0..k {
                        acc[t] += w * theta_raw[cur][t];
                    }
                    wsum += w;
                }
                let p = parents[cur];
                let has_parent = p >= 0 && (p as usize) < n && (p as usize) != cur;
                if !has_parent {
                    break;
                }
                cur = p as usize;
                dist += 1;
                steps += 1;
                if steps > n {
                    break; // cycle guard
                }
            }
            if wsum > 0.0 {
                for t in 0..k {
                    acc[t] /= wsum;
                }
            } else {
                // All path weights were 0 (degenerate sequence): fall back to raw.
                acc.copy_from_slice(&theta_raw[i]);
            }
            acc
        })
        .collect()
}

impl Estimator for CSATMModel {
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
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for CSATMModel {
    fn alpha(&self) -> Vec<f64> {
        vec![self.alpha; self.num_topics]
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }
    fn doc_lengths(&self) -> Vec<usize> {
        self.doc_lengths.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn corpus_from_ids(docs: Vec<Vec<u32>>, vocab: usize) -> Corpus {
        Corpus {
            id_to_word: (0..vocab).map(|i| format!("w{i}")).collect(),
            docs,
            doc_names: Vec::new(),
            doc_labels: Vec::new(),
            doc_freqs: vec![0; vocab],
            total_freqs: vec![0; vocab],
        }
    }

    // Two disjoint-vocabulary blocks; each planted topic should own one block.
    fn planted_corpus() -> (Corpus, usize) {
        // vocab 0..3 = "block A", 3..6 = "block B".
        let mut docs = Vec::new();
        for _ in 0..15 {
            docs.push(vec![0, 1, 2, 0, 1, 2]);
        }
        for _ in 0..15 {
            docs.push(vec![3, 4, 5, 3, 4, 5]);
        }
        (corpus_from_ids(docs, 6), 6)
    }

    #[test]
    fn csatm_recovers_planted_topics() {
        let (corpus, _v) = planted_corpus();
        let params = CsatmParams {
            num_topics: 2,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit(&corpus, &[], &params, 300, &mut rng);
        // Each topic's top-3 words should be a single block.
        let block_a: std::collections::HashSet<usize> = [0, 1, 2].into_iter().collect();
        let block_b: std::collections::HashSet<usize> = [3, 4, 5].into_iter().collect();
        let owned: Vec<usize> = (0..2)
            .map(|t| {
                let mut idx: Vec<usize> = (0..6).collect();
                idx.sort_by(|&a, &b| m.topic_word[t][b].partial_cmp(&m.topic_word[t][a]).unwrap());
                let top: std::collections::HashSet<usize> = idx.into_iter().take(3).collect();
                if top.intersection(&block_a).count() >= top.intersection(&block_b).count() {
                    0
                } else {
                    1
                }
            })
            .collect();
        assert_eq!(
            owned.iter().collect::<std::collections::HashSet<_>>().len(),
            2,
            "the two topics should split the two blocks"
        );
    }

    #[test]
    fn csatm_is_deterministic() {
        let (corpus, _v) = planted_corpus();
        let params = CsatmParams {
            num_topics: 3,
            ..Default::default()
        };
        let a = fit(
            &corpus,
            &[],
            &params,
            120,
            &mut ChaCha8Rng::seed_from_u64(3),
        );
        let b = fit(
            &corpus,
            &[],
            &params,
            120,
            &mut ChaCha8Rng::seed_from_u64(3),
        );
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        // A different seed should differ, so the test can't pass trivially.
        let c = fit(
            &corpus,
            &[],
            &params,
            120,
            &mut ChaCha8Rng::seed_from_u64(99),
        );
        assert_ne!(a.topic_word, c.topic_word);
    }

    #[test]
    fn csatm_conforms() {
        let (corpus, _v) = planted_corpus();
        let params = CsatmParams {
            num_topics: 2,
            ..Default::default()
        };
        let m = fit(&corpus, &[], &params, 20, &mut ChaCha8Rng::seed_from_u64(0));
        assert!(crate::conformance::check_conformance(&m).is_empty());
        assert!(crate::conformance::check_dirichlet(&m).is_empty());
    }

    // Popularity on the paper's Fig-2 tree (nodes 1..9), arithmetic w with c=1,
    // d=0.5 => w1=1, w2=0.5, w3=0, w4=0. Absolute per-level eq. 1.
    // Tree (parent links): 1 is root; 2,3,4 -> 1; 5,6,7 -> 3 (per Fig 2a: 5,6
    // under 2? see below); we encode the figure's structure explicitly.
    #[test]
    fn popularity_matches_absolute_per_level() {
        // Fig 2a structure: 1←{2,3,4}; 2←{5}; 5←{8}; 3←{6,7}; 7←{9}. (0-indexed)
        // node: 0=1,1=2,2=3,3=4,4=5,5=6,6=7,7=8,8=9
        let parents: Vec<i64> = vec![-1, 0, 0, 0, 1, 2, 2, 4, 6];
        let w = WeightSeq::Arithmetic { c: 1.0, d: 0.5 };
        let p = popularity_scores(&parents, &w);
        // Leaves (3=node4, 5=node6, 7=node8, 8=node9) have p=1.
        assert_eq!(p[3], 1.0);
        assert_eq!(p[5], 1.0);
        assert_eq!(p[7], 1.0);
        assert_eq!(p[8], 1.0);
        // node2 (idx1): subtree {2,5,8} at levels 1,2,3 => w1+w2+w3 = 1+0.5+0 = 1.5
        assert!((p[1] - 1.5).abs() < 1e-12);
        // node3 (idx2): subtree {3,6,7,9} levels: 3@1, 6@2, 7@2, 9@3
        //   => w1 + 2·w2 + w3 = 1 + 1.0 + 0 = 2.0
        assert!((p[2] - 2.0).abs() < 1e-12);
        // node1 (idx0, root): subtree levels: 1@1; {2,3,4}@2; {5,6,7}@3; {8,9}@4
        //   => w1 + 3·w2 + 3·w3 + 2·w4 = 1 + 1.5 + 0 + 0 = 2.5
        assert!((p[0] - 2.5).abs() < 1e-12);
    }

    // With λ=1 and every p_c=1 (all roots) and self-only transitivity paths,
    // the smoothed θ equals the raw θ (transitivity is a no-op on singletons).
    #[test]
    fn flat_corpus_transitivity_is_identity() {
        let (corpus, _v) = planted_corpus();
        let params = CsatmParams {
            num_topics: 2,
            lambda: 1.0,
            ..Default::default()
        };
        let m = fit(&corpus, &[], &params, 50, &mut ChaCha8Rng::seed_from_u64(1));
        for dd in 0..corpus.num_docs() {
            assert_eq!(m.doc_topic[dd], m.doc_topic_raw[dd]);
        }
        // All popularities are 1 for a flat (all-root) corpus.
        assert!(m.popularity.iter().all(|&p| (p - 1.0).abs() < 1e-12));
    }

    #[test]
    fn node_levels_are_correct() {
        // chain 0<-1<-2<-3 plus a separate root 4
        let parents: Vec<i64> = vec![-1, 0, 1, 2, -1];
        let lv = node_levels(&parents);
        assert_eq!(lv, vec![1, 2, 3, 4, 1]);
    }
}
