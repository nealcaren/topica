//! Topic representation for embedding-based clustering pipelines (Top2Vec,
//! BERTopic).
//!
//! Once documents are embedded and clustered, a cluster is just a set of
//! document ids; it carries no words and no topic vector yet. This stage turns
//! each cluster into something we can read and rank against.
//!
//! Two complementary views of a cluster live here:
//!
//! 1. **Words.** BERTopic scores terms with class-based TF-IDF: treat every
//!    cluster as one long pseudo-document, count terms within it, and down-weight
//!    terms that are common across all clusters. The top-weighted terms become the
//!    topic's label. See [`ctfidf`].
//! 2. **Geometry.** Top2Vec represents a topic by the mean of its documents'
//!    embeddings, the cluster centroid, and finds words or documents for that
//!    topic by cosine proximity in the embedding space. See [`centroids`],
//!    [`nearest_by_cosine`].
//!
//! [`top_indices`] is the shared "pick the top-N" helper both views lean on.
//!
//! Clustering conventions follow HDBSCAN: a label of `-1` marks a noise document
//! that belongs to no cluster, and the cluster labels are the contiguous range
//! `0..=max_label`. We exclude noise everywhere and key our output rows on the
//! label value, so cluster `c` is always row `c`.

/// Class-based TF-IDF over token-id documents (BERTopic's c-TF-IDF).
///
/// Each cluster (label `>= 0`) is one class; documents labeled `-1` are noise and
/// take no part. For class `c` and term `t`,
///
/// ```text
/// c-TF-IDF_{t,c} = tf_{t,c} * ln(1 + A / f_t)
/// ```
///
/// where `tf_{t,c}` is the raw count of `t` in class `c`, `f_t` is the count of
/// `t` summed over every class, and `A` is the average class size (total tokens
/// across classes divided by the number of classes). A term seen everywhere has a
/// small `ln(1 + A / f_t)` and so contributes little; a term concentrated in one
/// class keeps its weight there. Terms with `f_t == 0` get weight `0`.
///
/// Returns a `num_classes x vocab_size` matrix with `num_classes = max_label + 1`
/// (so row `c` is class `c`), or an empty matrix when there are no non-noise docs.
///
/// This is the plain c-TF-IDF; [`ctfidf_weighted`] adds the BM25 and
/// frequent-word knobs. BERTopic ships with both knobs off by default, so this
/// default matches BERTopic's documented default (`bm25_weighting=False`,
/// `reduce_frequent_words=False`).
pub fn ctfidf(docs: &[Vec<u32>], labels: &[i64], vocab_size: usize) -> Vec<Vec<f64>> {
    ctfidf_weighted(docs, labels, vocab_size, false, false)
}

/// Number of topics implied by a label vector (max non-noise label + 1).
fn label_count(labels: &[i64]) -> usize {
    labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l as usize + 1)
        .max()
        .unwrap_or(0)
}

/// Relabel after merging topics: every id in a group collapses to one cluster,
/// then surviving topics are renumbered to a dense `0..k`. Groups may chain
/// (a topic appearing in two groups links them); noise (`-1`) is preserved. Used
/// by both Top2Vec and BERTopic for manual `merge_topics`.
pub fn merge_labels(labels: &[i64], groups: &[Vec<usize>]) -> Vec<i64> {
    let k = label_count(labels);
    if k == 0 {
        return labels.to_vec();
    }
    // Union each group to its smallest member, then resolve chains.
    let mut rep: Vec<usize> = (0..k).collect();
    for g in groups {
        if let Some(&r0) = g.iter().filter(|&&t| t < k).min() {
            for &t in g {
                if t < k {
                    rep[t] = r0;
                }
            }
        }
    }
    for i in 0..k {
        let mut r = i;
        while rep[r] != r {
            r = rep[r];
        }
        rep[i] = r;
    }
    // Dense ids for the surviving representatives, in sorted order.
    let mut distinct: Vec<usize> = rep.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let dense: std::collections::HashMap<usize, i64> = distinct
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as i64))
        .collect();
    labels
        .iter()
        .map(|&l| if l < 0 { -1 } else { dense[&rep[l as usize]] })
        .collect()
}

/// Reassign every noise document (label `-1`) to the topic whose `topic_word`
/// (K×V, rows sum to ~1) best explains its tokens, by the argmax of
/// `sum_w log topic_word[k][w]`. Non-noise labels are unchanged. This is the
/// embedding-free outlier-reduction strategy shared by Top2Vec and BERTopic.
pub fn assign_outliers(docs: &[Vec<u32>], labels: &[i64], topic_word: &[Vec<f64>]) -> Vec<i64> {
    let k = topic_word.len();
    labels
        .iter()
        .enumerate()
        .map(|(d, &l)| {
            if l >= 0 || k == 0 || docs[d].is_empty() {
                return l;
            }
            let mut best = (f64::NEG_INFINITY, -1i64);
            for (t, tw) in topic_word.iter().enumerate() {
                let s: f64 = docs[d].iter().map(|&w| (tw[w as usize] + 1e-12).ln()).sum();
                if s > best.0 {
                    best = (s, t as i64);
                }
            }
            best.1
        })
        .collect()
}

/// Class-based TF-IDF with BERTopic's two documented tuning knobs.
///
/// The base score is the same as [`ctfidf`]: for class `c` and term `t`,
///
/// ```text
/// c-TF-IDF_{t,c} = tf_{t,c} * ln(1 + A / f_t)
/// ```
///
/// with `tf_{t,c}` the raw count of `t` in class `c`, `f_t` the count of `t`
/// summed over every class, and `A` the average class size. With `bm25 == false`
/// and `reduce_frequent == false` we return exactly what [`ctfidf`] returns.
///
/// `reduce_frequent` swaps `tf_{t,c}` for `sqrt(tf_{t,c})`, BERTopic's
/// `reduce_frequent_words`. Taking the square root before the idf factor flattens
/// the gap between very frequent and merely common terms, which trims stop-word
/// leakage from the top of a topic.
///
/// `bm25` swaps the idf factor for BERTopic's class-based BM25 idf,
///
/// ```text
/// ln(1 + (A - f_t + 0.5) / (f_t + 0.5))
/// ```
///
/// where `A` plays the corpus-size role and `f_t` the document-frequency role.
/// This matches upstream BERTopic's `ClassTfidfTransformer(bm25_weighting=True)`
/// exactly, including the **unclamped** log: when `f_t > A + 1` the argument drops
/// below 1 and the idf goes negative, which is how BERTopic ranks a term that is
/// ubiquitous across classes *below* a term that is merely absent. We do not clamp
/// it (issue #488; earlier versions floored the argument at 1.0, tying ubiquitous
/// terms with absent ones instead of ranking them last). Negative c-TF-IDF is
/// carried through the raw matrix; the caller that builds a normalized topic-word
/// distribution from it floors negatives to zero for that probability surface
/// only, leaving the ranking BERTopic intends.
///
/// The two flags compose: `bm25 && reduce_frequent` applies both.
///
/// Returns a `num_classes x vocab_size` matrix with `num_classes = max_label + 1`
/// (so row `c` is class `c`), or an empty matrix when there are no non-noise docs.
pub fn ctfidf_weighted(
    docs: &[Vec<u32>],
    labels: &[i64],
    vocab_size: usize,
    bm25: bool,
    reduce_frequent: bool,
) -> Vec<Vec<f64>> {
    // Number of classes is one past the largest label; -1 contributes nothing.
    let max_label = labels.iter().copied().max().unwrap_or(-1);
    if max_label < 0 {
        return Vec::new();
    }
    let num_classes = (max_label + 1) as usize;

    // Per-class term counts and the document corpus partitioned by class.
    let mut tf = vec![vec![0.0f64; vocab_size]; num_classes];
    for (d, doc) in docs.iter().enumerate() {
        let label = labels[d];
        if label < 0 {
            continue;
        }
        let c = label as usize;
        for &w in doc {
            let w = w as usize;
            if w < vocab_size {
                tf[c][w] += 1.0;
            }
        }
    }

    // f_t: total occurrences of each term across all classes. A: mean class size.
    let mut f = vec![0.0f64; vocab_size];
    let mut total_tokens = 0.0f64;
    for class in &tf {
        for (t, &count) in class.iter().enumerate() {
            f[t] += count;
            total_tokens += count;
        }
    }
    let avg_class_size = total_tokens / num_classes as f64;

    // The idf-like factor depends only on the term, so compute it once. BM25 uses
    // the class-based BM25 form; the plain form is BERTopic's default. A term with
    // `f_t == 0` gets weight 0. The BM25 log is unclamped to match upstream, so it
    // is negative for terms with `f_t > A + 1` (issue #488).
    let idf: Vec<f64> = f
        .iter()
        .map(|&ft| {
            if ft == 0.0 {
                0.0
            } else if bm25 {
                (1.0 + (avg_class_size - ft + 0.5) / (ft + 0.5)).ln()
            } else {
                (1.0 + avg_class_size / ft).ln()
            }
        })
        .collect();

    for class in &mut tf {
        for (t, weight) in class.iter_mut().enumerate() {
            // reduce_frequent_words damps the term frequency with a square root
            // before the idf factor multiplies it in.
            let tf_t = if reduce_frequent {
                weight.sqrt()
            } else {
                *weight
            };
            *weight = tf_t * idf[t];
        }
    }
    tf
}

/// CETopic's TFIDF×IDF_i topic-word weighting (Zhang et al., NAACL 2022, Eq. 4).
///
/// c-TF-IDF ([`ctfidf`]) asks only "which terms are frequent in this cluster and
/// rare across clusters". CETopic's winning scheme instead multiplies a
/// **corpus-level TF-IDF** (a term's importance in the whole corpus, averaged over
/// the cluster's documents) by a **cross-cluster IDF** that penalizes terms spread
/// across many clusters. The first factor keeps globally salient words; the second
/// lifts topic *diversity* by demoting words several clusters would otherwise share.
/// The paper's ablation (their Table 3) finds this the decisive win over plain
/// per-cluster TF/TF-IDF.
///
/// Faithful to the reference (`hyintell/topicx`, MIT), which layers two
/// scikit-learn `TfidfTransformer`s at their defaults (smoothed idf, per-row L2,
/// then L1). For cluster `i` and term `t`:
///
/// 1. **Global TF-IDF, averaged per cluster.** For document `d`,
///    `g_{t,d} = n_{t,d} * idf(t)` with the smoothed corpus idf
///    `idf(t) = ln((1 + |D|) / (1 + df_t)) + 1` (`df_t` = #docs containing `t`,
///    `|D|` = #docs), then L2-normalized across `t` within each document. Row `i`
///    of the cluster matrix is the mean of `g_d` over the documents in cluster `i`.
/// 2. **Cross-cluster IDF.** Treating each cluster as one pseudo-document,
///    `idf_i(t) = ln((1 + |K|) / (1 + cf_t)) + 1` (`cf_t` = #clusters containing
///    `t`, `|K|` = #clusters).
/// 3. **Combine and L1-normalize.** `score_{i,t} = avg_global_{i,t} * idf_i(t)`,
///    then each cluster row is scaled to sum to 1.
///
/// Noise documents (`-1`) join no cluster and so enter no cluster mean; to keep the
/// corpus idf and the averaged population consistent, they are excluded from `|D|`
/// and `df_t` as well (CETopic's own clusterer is K-Means, which leaves no noise, so
/// the reference never faces the choice). Scores are non-negative throughout
/// (`avg_global >= 0`, smoothed `idf_i >= 1`). Returns a `num_classes x vocab_size`
/// matrix with `num_classes = max_label + 1` (row `c` is cluster `c`), or an empty
/// matrix when there are no non-noise documents. An empty cluster (a label with no
/// documents) yields a zero row, which the caller renders as a uniform topic, matching
/// [`ctfidf`]'s empty-cluster behavior.
pub fn tfidf_idf_cluster(docs: &[Vec<u32>], labels: &[i64], vocab_size: usize) -> Vec<Vec<f64>> {
    let max_label = labels.iter().copied().max().unwrap_or(-1);
    if max_label < 0 {
        return Vec::new();
    }
    let num_classes = (max_label + 1) as usize;

    // Per-cluster term counts (K x V) and per-cluster document counts. Only
    // non-noise documents (label >= 0) participate, as in `ctfidf`.
    let mut cluster_tf = vec![vec![0.0f64; vocab_size]; num_classes];
    let mut docs_in_cluster = vec![0usize; num_classes];
    // Global document frequency (df_t) and the non-noise corpus size (|D|).
    let mut df = vec![0.0f64; vocab_size];
    let mut num_docs = 0usize;
    for (doc, &label) in docs.iter().zip(labels) {
        if label < 0 {
            continue;
        }
        let c = label as usize;
        docs_in_cluster[c] += 1;
        num_docs += 1;
        // Count each term once per document for df_t; accumulate raw counts per cluster.
        let mut seen = std::collections::HashSet::new();
        for &w in doc {
            let w = w as usize;
            if w < vocab_size {
                cluster_tf[c][w] += 1.0;
                if seen.insert(w) {
                    df[w] += 1.0;
                }
            }
        }
    }
    if num_docs == 0 {
        // Every non-noise label pointed only at empty/out-of-vocab documents.
        return vec![vec![0.0; vocab_size]; num_classes];
    }

    // Smoothed corpus idf, sklearn's TfidfTransformer default: ln((1+|D|)/(1+df))+1.
    let d = num_docs as f64;
    let idf_global: Vec<f64> = df
        .iter()
        .map(|&dft| ((1.0 + d) / (1.0 + dft)).ln() + 1.0)
        .collect();

    // Cross-cluster idf_i: treat each cluster as one pseudo-document. cf_t is the
    // number of clusters whose count of t is nonzero. Smoothed the same way over
    // |K| = num_classes "documents": ln((1+|K|)/(1+cf))+1.
    let mut cf = vec![0.0f64; vocab_size];
    for class in &cluster_tf {
        for (t, &count) in class.iter().enumerate() {
            if count > 0.0 {
                cf[t] += 1.0;
            }
        }
    }
    let k = num_classes as f64;
    let idf_i: Vec<f64> = cf
        .iter()
        .map(|&cft| ((1.0 + k) / (1.0 + cft)).ln() + 1.0)
        .collect();

    // Sum of per-document, L2-normalized global TF-IDF vectors within each cluster.
    // Summing over documents in the fixed input order keeps the reduction
    // deterministic. Dividing by `docs_in_cluster` gives the per-cluster mean.
    let mut avg_global = vec![vec![0.0f64; vocab_size]; num_classes];
    // Reused scratch for one document's (term, weight) pairs.
    let mut pairs: Vec<(usize, f64)> = Vec::new();
    for (doc, &label) in docs.iter().zip(labels) {
        if label < 0 {
            continue;
        }
        let c = label as usize;
        // Fold repeats into per-term counts, then weight by the global idf.
        pairs.clear();
        let mut counts: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for &w in doc {
            let w = w as usize;
            if w < vocab_size {
                *counts.entry(w).or_insert(0.0) += 1.0;
            }
        }
        let mut norm_sq = 0.0f64;
        for (t, count) in counts {
            let v = count * idf_global[t];
            norm_sq += v * v;
            pairs.push((t, v));
        }
        if norm_sq == 0.0 {
            continue; // all-out-of-vocab document: contributes a zero vector.
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        // Sort by term id so the accumulation order into avg_global is fixed
        // regardless of the HashMap iteration order (determinism).
        pairs.sort_unstable_by_key(|&(t, _)| t);
        for &(t, v) in &pairs {
            avg_global[c][t] += v * inv_norm;
        }
    }

    // Mean, multiply by the cross-cluster idf, and L1-normalize each cluster row.
    for (c, row) in avg_global.iter_mut().enumerate() {
        let n = docs_in_cluster[c];
        if n == 0 {
            continue; // empty cluster stays a zero row (uniform topic downstream).
        }
        let inv_n = 1.0 / n as f64;
        let mut sum = 0.0f64;
        for (t, x) in row.iter_mut().enumerate() {
            *x = *x * inv_n * idf_i[t];
            sum += *x;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for x in row.iter_mut() {
                *x *= inv_sum;
            }
        }
    }
    avg_global
}

/// Mean embedding vector per cluster (Top2Vec's topic vector).
///
/// `vectors[d]` is document `d`'s embedding; rows labeled `-1` are noise and are
/// left out of every mean. Returns `num_clusters` rows, where row `c` is the
/// centroid of cluster `c`. An empty cluster yields a zero vector of the embedding
/// dimension; when `vectors` is empty we cannot infer a dimension and return
/// `num_clusters` empty vectors.
pub fn centroids(vectors: &[Vec<f64>], labels: &[i64], num_clusters: usize) -> Vec<Vec<f64>> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; dim]; num_clusters];
    let mut counts = vec![0usize; num_clusters];

    for (d, vec) in vectors.iter().enumerate() {
        let label = labels[d];
        if label < 0 {
            continue;
        }
        let c = label as usize;
        if c >= num_clusters {
            continue;
        }
        counts[c] += 1;
        for (s, &x) in sums[c].iter_mut().zip(vec.iter()) {
            *s += x;
        }
    }

    for (c, sum) in sums.iter_mut().enumerate() {
        if counts[c] > 0 {
            let n = counts[c] as f64;
            for s in sum.iter_mut() {
                *s /= n;
            }
        }
    }
    sums
}

/// L2 norm of a vector.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity, with zero-norm vectors treated as similarity `0.0` rather
/// than dividing by zero.
fn cosine(a: &[f64], b: &[f64], norm_a: f64) -> f64 {
    let norm_b = norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// The up-to-`n` candidates most cosine-similar to `query`, as `(index,
/// similarity)` pairs sorted by similarity descending.
///
/// Zero-norm vectors (the query or any candidate) contribute a similarity of
/// `0.0` and never trigger a division by zero. Ties break toward the lower index,
/// so the ordering is deterministic.
pub fn nearest_by_cosine(query: &[f64], candidates: &[Vec<f64>], n: usize) -> Vec<(usize, f64)> {
    let norm_q = norm(query);
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine(query, c, norm_q)))
        .collect();
    sort_desc_by_value(&mut scored);
    scored.truncate(n);
    scored
}

/// The top-`n` `(index, weight)` pairs by weight descending, lower index winning
/// ties. We use this to pull the highest c-TF-IDF words for a topic.
pub fn top_indices(weights: &[f64], n: usize) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    sort_desc_by_value(&mut scored);
    scored.truncate(n);
    scored
}

/// Sort `(index, value)` pairs by value descending, breaking ties on the lower
/// index. NaN values sort to the end; they are not expected here but should not
/// panic the comparator.
fn sort_desc_by_value(scored: &mut [(usize, f64)]) {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_labels_collapses_groups_and_densifies() {
        // 4 topics; merge {0,2} and {1,3} -> 2 dense topics.
        let labels = vec![0i64, 1, 2, 3, -1, 0];
        let merged = merge_labels(&labels, &[vec![0, 2], vec![1, 3]]);
        // 0 and 2 share a new id; 1 and 3 share the other; -1 preserved.
        assert_eq!(merged[0], merged[2]);
        assert_eq!(merged[1], merged[3]);
        assert_ne!(merged[0], merged[1]);
        assert_eq!(merged[4], -1);
        // dense range 0..2
        assert!(merged
            .iter()
            .filter(|&&l| l >= 0)
            .all(|&l| l == 0 || l == 1));
    }

    #[test]
    fn assign_outliers_sends_noise_to_best_topic() {
        // Topic 0 favors word 0, topic 1 favors word 1.
        let topic_word = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let docs = vec![vec![0u32, 0], vec![1, 1], vec![1, 1]];
        let labels = vec![0i64, 1, -1]; // doc 2 is noise, all word-1
        let out = assign_outliers(&docs, &labels, &topic_word);
        assert_eq!(out[0], 0); // unchanged
        assert_eq!(out[1], 1); // unchanged
        assert_eq!(out[2], 1); // noise doc of word-1 -> topic 1
    }

    #[test]
    fn ctfidf_picks_distinctive_words() {
        // Vocab: 0 = shared, 1 = distinctive-A, 2 = distinctive-B.
        // Class 0 is all word 1 plus some shared word 0; class 1 is all word 2
        // plus the same shared word 0.
        let docs = vec![vec![0, 1, 1], vec![0, 1, 1], vec![0, 2, 2], vec![0, 2, 2]];
        let labels = vec![0, 0, 1, 1];
        let m = ctfidf(&docs, &labels, 3);

        assert_eq!(m.len(), 2);
        // Word 1 is the top term in class 0; word 2 the top term in class 1.
        assert!(m[0][1] > m[0][0]);
        assert!(m[0][1] > m[0][2]);
        assert!(m[1][2] > m[1][0]);
        assert!(m[1][2] > m[1][1]);
        // The shared word appears in both classes, so its idf factor is smaller
        // than the distinctive words', which appear in only one class.
        assert!(m[0][1] > m[1][1]); // word 1 absent from class 1
        assert_eq!(m[1][1], 0.0);
    }

    #[test]
    fn ctfidf_weighted_default_matches_ctfidf() {
        // With both knobs off, ctfidf_weighted must reproduce ctfidf exactly.
        let docs = vec![vec![0, 1, 1], vec![0, 1, 1], vec![0, 2, 2], vec![0, 2, 2]];
        let labels = vec![0, 0, 1, 1];
        let base = ctfidf(&docs, &labels, 3);
        let weighted = ctfidf_weighted(&docs, &labels, 3, false, false);
        assert_eq!(base.len(), weighted.len());
        for (br, wr) in base.iter().zip(weighted.iter()) {
            assert_eq!(br.len(), wr.len());
            for (&b, &w) in br.iter().zip(wr.iter()) {
                assert_eq!(b, w);
            }
        }
    }

    #[test]
    fn ctfidf_reduce_frequent_damps_ubiquitous_term() {
        // Word 0 is deliberately ubiquitous: it appears many times in every class.
        // Words 1 and 2 are distinctive to classes 0 and 1 respectively.
        let docs = vec![
            vec![0, 0, 0, 0, 1, 1],
            vec![0, 0, 0, 0, 1, 1],
            vec![0, 0, 0, 0, 2, 2],
            vec![0, 0, 0, 0, 2, 2],
        ];
        let labels = vec![0, 0, 1, 1];
        let base = ctfidf(&docs, &labels, 3);
        let reduced = ctfidf_weighted(&docs, &labels, 3, false, true);

        // The square root damps the ubiquitous word 0 relative to its plain weight.
        // (Word 0 still has nonzero idf since A / f_t > 0 even when seen everywhere.)
        assert!(reduced[0][0] < base[0][0]);
        // The distinctive word still ranks above the ubiquitous one in its class,
        // so the ranking we care about is preserved under reduce_frequent.
        assert!(reduced[0][1] > reduced[0][0]);
        assert!(base[0][1] > base[0][0]);
    }

    #[test]
    fn ctfidf_bm25_is_finite_nonneg_and_downweights_ubiquitous() {
        // Word 0 appears in every class; words 1 and 2 are class-specific.
        let docs = vec![vec![0, 1, 1], vec![0, 1, 1], vec![0, 2, 2], vec![0, 2, 2]];
        let labels = vec![0, 0, 1, 1];
        let m = ctfidf_weighted(&docs, &labels, 3, true, false);

        // Every weight is finite and non-negative.
        for row in &m {
            for &w in row {
                assert!(w.is_finite());
                assert!(w >= 0.0);
            }
        }
        // The term seen in every class is downweighted below the distinctive term
        // in the same class.
        assert!(m[0][1] > m[0][0]);
        assert!(m[1][2] > m[1][0]);
    }

    #[test]
    fn ctfidf_bm25_goes_negative_when_df_exceeds_avg_class_size() {
        // Word 0 is ubiquitous (10x in each of 2 classes); words 1/2 are rare.
        // f_0 = 20, A = 22/2 = 11, so f_0 > A + 1 and upstream BM25 idf is
        // negative. topica now matches upstream (issue #488): no clamp to 0.
        let docs = vec![
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        ];
        let labels = vec![0, 1];
        let bm = ctfidf_weighted(&docs, &labels, 3, true, false);
        // The ubiquitous term carries a negative c-TF-IDF in its class...
        assert!(bm[0][0] < 0.0, "bm25 w0 in class 0 = {}", bm[0][0]);
        // ...ranked below the distinctive term, exactly as upstream intends.
        assert!(bm[0][1] > bm[0][0]);
        // The plain (default) path keeps the same term non-negative.
        let plain = ctfidf_weighted(&docs, &labels, 3, false, false);
        assert!(plain[0][0] > 0.0);
    }

    #[test]
    fn ctfidf_handles_only_noise() {
        let docs = vec![vec![0, 1], vec![1, 2]];
        let labels = vec![-1, -1];
        assert!(ctfidf(&docs, &labels, 3).is_empty());
    }

    #[test]
    fn tfidf_idf_penalizes_cross_cluster_words() {
        // Vocab: 0 = shared across both clusters, 1 = distinctive-A, 2 = distinctive-B.
        // Same layout as `ctfidf_picks_distinctive_words`, so the two schemes are
        // directly comparable: the cross-cluster IDF must demote the shared word.
        let docs = vec![vec![0, 1, 1], vec![0, 1, 1], vec![0, 2, 2], vec![0, 2, 2]];
        let labels = vec![0, 0, 1, 1];
        let m = tfidf_idf_cluster(&docs, &labels, 3);

        assert_eq!(m.len(), 2);
        // Each cluster row is a valid L1 distribution and non-negative.
        for row in &m {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "row sums to {s}");
            assert!(row.iter().all(|&w| w >= 0.0));
        }
        // The distinctive word tops its own cluster; the cross-cluster word does not.
        assert!(
            m[0][1] > m[0][0],
            "distinctive A beats shared word in cluster 0"
        );
        assert!(
            m[1][2] > m[1][0],
            "distinctive B beats shared word in cluster 1"
        );
        // The word absent from a cluster scores exactly zero there.
        assert_eq!(m[0][2], 0.0);
        assert_eq!(m[1][1], 0.0);

        // The cross-cluster IDF penalty is the whole point: the shared word (in 2/2
        // clusters, idf_i = ln(3/3)+1 = 1) is penalized relative to a distinctive
        // word (in 1/2 clusters, idf_i = ln(3/2)+1 > 1). So TFIDF×IDF_i suppresses
        // the shared word's share of cluster 0 more than plain c-TF-IDF does.
        let base = ctfidf(&docs, &labels, 3);
        let share = |row: &[f64]| -> f64 {
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                row[0] / s
            } else {
                0.0
            }
        };
        assert!(
            share(&m[0]) < share(&base[0]),
            "cross-cluster word share {} should drop below c-TF-IDF's {}",
            share(&m[0]),
            share(&base[0])
        );
    }

    #[test]
    fn tfidf_idf_matches_sklearn_reference_values() {
        // Gold values computed with the reference's exact scikit-learn pipeline
        // (TfidfTransformer defaults: smooth_idf, per-row L2, then l1-normalized
        // score = avg_global_tfidf * idf_i). Corpus below; the numbers are frozen
        // from that pipeline so this test pins bit-level fidelity to CETopic.
        //
        //   docs   = [[0,0,1], [0,1,2], [1,2,2], [2,3,3]]
        //   labels = [0, 0, 1, 1]   (vocab size 4)
        let docs = vec![vec![0, 0, 1], vec![0, 1, 2], vec![1, 2, 2], vec![2, 3, 3]];
        let labels = vec![0, 0, 1, 1];
        let m = tfidf_idf_cluster(&docs, &labels, 4);
        // Reference output (scikit-learn 1.x), row-wise l1-normalized:
        let want = [
            [
                0.6072850707222552,
                0.2475092236893676,
                0.1452057055883772,
                0.0,
            ],
            [
                0.0,
                0.14983999450857777,
                0.40154781498192094,
                0.44861219050950124,
            ],
        ];
        assert_eq!(m.len(), 2);
        for (i, want_row) in want.iter().enumerate() {
            for (t, &wv) in want_row.iter().enumerate() {
                assert!(
                    (m[i][t] - wv).abs() < 1e-9,
                    "row {i} term {t}: got {}, want {}",
                    m[i][t],
                    wv
                );
            }
        }
    }

    #[test]
    fn tfidf_idf_only_noise_is_empty() {
        let docs = vec![vec![0, 1], vec![1, 2]];
        let labels = vec![-1, -1];
        assert!(tfidf_idf_cluster(&docs, &labels, 3).is_empty());
    }

    #[test]
    fn tfidf_idf_empty_cluster_is_zero_row() {
        // Labels 0 and 2 are used, 1 is not: cluster 1 must be an all-zero row.
        let docs = vec![vec![0, 0, 1], vec![2, 3, 3]];
        let labels = vec![0, 2];
        let m = tfidf_idf_cluster(&docs, &labels, 4);
        assert_eq!(m.len(), 3);
        assert!(
            m[1].iter().all(|&w| w == 0.0),
            "empty cluster 1 is zero row"
        );
        assert!((m[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((m[2].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn centroids_average_each_cluster() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0], // cluster 0 mean -> (1, 0)
            vec![0.0, 4.0],
            vec![0.0, 6.0], // cluster 1 mean -> (0, 5)
            vec![9.0, 9.0], // noise, excluded
        ];
        let labels = vec![0, 0, 1, 1, -1];
        let c = centroids(&vectors, &labels, 2);

        assert_eq!(c.len(), 2);
        assert_eq!(c[0], vec![1.0, 0.0]);
        assert_eq!(c[1], vec![0.0, 5.0]);
    }

    #[test]
    fn centroids_empty_cluster_is_zero_vector() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let labels = vec![0];
        let c = centroids(&vectors, &labels, 3);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(c[1], vec![0.0, 0.0, 0.0]); // empty cluster
        assert_eq!(c[2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn nearest_by_cosine_ranks_aligned_first() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![0.0, 1.0], // orthogonal, sim 0
            vec![1.0, 0.0], // aligned, sim 1
            vec![0.5, 0.5], // 45 degrees
        ];
        let ranked = nearest_by_cosine(&query, &candidates, 3);
        assert_eq!(ranked[0].0, 1);
        assert!((ranked[0].1 - 1.0).abs() < 1e-9);
        // Decreasing similarity down the ranking.
        assert!(ranked[0].1 >= ranked[1].1);
        assert!(ranked[1].1 >= ranked[2].1);
    }

    #[test]
    fn nearest_by_cosine_tolerates_zero_norm() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![0.0, 0.0], // zero norm: sim 0, no panic
            vec![1.0, 0.0],
        ];
        let ranked = nearest_by_cosine(&query, &candidates, 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 1);
        // The zero-norm candidate scores exactly 0.0.
        let zero = ranked.iter().find(|&&(i, _)| i == 0).unwrap();
        assert_eq!(zero.1, 0.0);
    }

    #[test]
    fn nearest_by_cosine_zero_norm_query() {
        let query = vec![0.0, 0.0];
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ranked = nearest_by_cosine(&query, &candidates, 2);
        // All similarities are 0; ties break toward the lower index.
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[1].0, 1);
    }

    #[test]
    fn top_indices_returns_largest_in_order() {
        let weights = vec![0.1, 0.9, 0.5, 0.9, 0.0];
        let top = top_indices(&weights, 3);
        assert_eq!(top.len(), 3);
        // Two entries tie at 0.9; the lower index comes first.
        assert_eq!(top[0], (1, 0.9));
        assert_eq!(top[1], (3, 0.9));
        assert_eq!(top[2], (2, 0.5));
    }

    #[test]
    fn top_indices_caps_at_available() {
        let weights = vec![3.0, 1.0];
        let top = top_indices(&weights, 10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], (0, 3.0));
    }
}
