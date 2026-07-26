//! BERTopic: the second head on topica's embedding-clustering pipeline.
//!
//! BERTopic (Grootendorst 2022) shares Top2Vec's `reduce -> cluster -> represent`
//! shape but differs in the representation: a topic is defined by **class-based
//! TF-IDF** over its documents' words, not by a point in the embedding space, so
//! BERTopic needs no word embeddings. Two features are characteristic and we port
//! them here:
//!
//! - `nr_topics`: reduce the discovered topics down to a target count. topica's
//!   reduction is a **greedy** merge — it repeatedly folds the single most
//!   cosine-similar pair of topic c-TF-IDF rows and recomputes — which differs
//!   from upstream BERTopic. Upstream fits one `AgglomerativeClustering`
//!   (`linkage="ward"`, Euclidean) over the topic *embeddings* and cuts it at
//!   `nr_topics - _outliers` clusters. Different distance space (c-TF-IDF vs
//!   embeddings) and different merge tree (greedy vs ward), so the two need not
//!   pick the same merges. topica also reads `nr_topics` as the number of **real**
//!   topics (the `-1` noise topic is not counted), whereas upstream counts the
//!   `-1` topic toward the total. See `fit_bertopic` (issue #488).
//! - `approximate_distribution`: a soft per-document topic distribution. We slide
//!   a window over each document, build the window's c-TF-IDF vector *with the same
//!   `bm25`/`reduce_frequent` weighting the topics were built with* (issue #488),
//!   measure its cosine to every topic, drop any window↔topic similarity below
//!   `min_similarity`, and average across windows. This is the document-topic
//!   distribution BERTopic reports without re-running the clustering. A document
//!   with no surviving evidence (all out-of-vocabulary, or every similarity gated
//!   out) becomes a uniform row rather than a zero row, so `doc_topic` stays a
//!   valid per-document distribution (topica's cross-model surface contract);
//!   upstream leaves such a document as a zero row.
//!
//! As elsewhere in this branch, the caller brings the document embeddings; topica
//! does not embed the text.

use crate::{cluster, reduce, represent};
use rayon::prelude::*;

/// A fitted BERTopic model. Exposes the branch's shared surface: `topic_word`
/// (K x V, normalized c-TF-IDF), `doc_topic` (D x K, the approximate
/// distribution), plus the hard `labels` (`-1` is noise).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BertopicModel {
    pub num_topics: usize,
    pub labels: Vec<i64>,
    pub topic_word: Vec<Vec<f64>>,
    pub doc_topic: Vec<Vec<f64>>,
    /// Unnormalized c-TF-IDF rows, kept so `approximate_distribution` can be
    /// recomputed at other window sizes after fitting. May carry negative entries
    /// when `bm25` is on (issue #488).
    ctfidf_raw: Vec<Vec<f64>>,
    /// idf weights per vocabulary term, computed with the same `bm25` setting the
    /// model was fit with so `approximate_distribution` weights its windows the way
    /// the topics were weighted (issue #488). Plain idf is `ln(1 + A / f_t)`.
    idf: Vec<f64>,
    /// Whether the c-TF-IDF term frequency was damped by a square root
    /// (`reduce_frequent_words`). Kept so `approximate_distribution` can reproduce
    /// the same window weighting. `#[serde(default)]` for old-save compatibility.
    #[serde(default)]
    reduce_frequent: bool,
    /// Minimum window↔topic cosine that contributes to `approximate_distribution`;
    /// lower similarities are dropped (BERTopic's `min_similarity`). `0.0` keeps
    /// every non-negative similarity. `#[serde(default)]` for old-save
    /// compatibility.
    #[serde(default)]
    min_similarity: f64,
}

impl BertopicModel {
    /// Top `n` words of `topic` by c-TF-IDF weight, as `(word_id, weight)`.
    pub fn top_words(&self, n: usize, topic: usize) -> Vec<(usize, f64)> {
        represent::top_indices(&self.topic_word[topic], n)
    }

    /// Recompute the soft document-topic distribution at a chosen `window`/`stride`
    /// over each document's tokens. `min_similarity` overrides the fit-time gate
    /// when `Some`. Rows sum to one.
    pub fn approximate_distribution(
        &self,
        docs: &[Vec<u32>],
        window: usize,
        stride: usize,
        min_similarity: Option<f64>,
    ) -> Vec<Vec<f64>> {
        approximate_distribution(
            docs,
            &self.ctfidf_raw,
            &self.idf,
            self.num_topics,
            window,
            stride,
            self.reduce_frequent,
            min_similarity.unwrap_or(self.min_similarity),
        )
    }

    /// Merge each group of topics into one and rebuild the representation.
    #[allow(clippy::too_many_arguments)]
    pub fn merge_topics(
        &mut self,
        docs: &[Vec<u32>],
        groups: &[Vec<usize>],
        vocab_size: usize,
        bm25: bool,
        reduce_frequent: bool,
        window: usize,
        stride: usize,
    ) {
        self.labels = represent::merge_labels(&self.labels, groups);
        self.rebuild(docs, vocab_size, bm25, reduce_frequent, window, stride);
    }

    /// Reassign noise documents to their nearest topic and rebuild.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_outliers(
        &mut self,
        docs: &[Vec<u32>],
        vocab_size: usize,
        bm25: bool,
        reduce_frequent: bool,
        window: usize,
        stride: usize,
    ) {
        self.labels = represent::assign_outliers(docs, &self.labels, &self.topic_word);
        self.rebuild(docs, vocab_size, bm25, reduce_frequent, window, stride);
    }

    /// Recompute everything that depends on the labels (after a merge or an
    /// outlier reassignment): the c-TF-IDF, the topic-word distribution, and the
    /// document-topic distribution.
    fn rebuild(
        &mut self,
        docs: &[Vec<u32>],
        vocab_size: usize,
        bm25: bool,
        reduce_frequent: bool,
        window: usize,
        stride: usize,
    ) {
        self.num_topics = topic_count(&self.labels);
        self.ctfidf_raw =
            represent::ctfidf_weighted(docs, &self.labels, vocab_size, bm25, reduce_frequent);
        self.idf = idf_weights(docs, &self.labels, vocab_size, bm25);
        self.reduce_frequent = reduce_frequent;
        self.topic_word = topic_word_from_ctfidf(&self.ctfidf_raw);
        self.doc_topic = approximate_distribution(
            docs,
            &self.ctfidf_raw,
            &self.idf,
            self.num_topics,
            window,
            stride,
            reduce_frequent,
            self.min_similarity,
        );
    }
}

/// Fit BERTopic on token-id documents plus document embeddings. `nr_topics`, if
/// set, reduces the discovered **real** topics (the `-1` noise topic is never
/// counted) to that count by greedily folding the most c-TF-IDF-similar pair, which
/// differs from upstream BERTopic's ward agglomeration over topic embeddings (see
/// the module docs, issue #488). `window`/`stride`/`min_similarity` parameterize the
/// approximate distribution used for `doc_topic`.
#[allow(clippy::too_many_arguments)]
pub fn fit_bertopic(
    docs: &[Vec<u32>],
    doc_embeddings: &[Vec<f64>],
    vocab_size: usize,
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    nr_topics: Option<usize>,
    window: usize,
    stride: usize,
    bm25: bool,
    reduce_frequent: bool,
    min_similarity: f64,
    clusterer: &str,
    num_clusters: Option<usize>,
    resolution: f64,
    knn_neighbors: usize,
    umap_params: &reduce::UmapParams,
    seed: u64,
) -> BertopicModel {
    let emb_dim = doc_embeddings.first().map_or(0, |r| r.len());

    // (1) reduce, (2) cluster. HDBSCAN (default) leaves `-1` noise; KMeans /
    // agglomerative assign every document instead.
    let did_reduce = emb_dim > n_components && n_components > 0;
    let mut reduced: Vec<Vec<f64>> = if did_reduce {
        reduce::reduce(
            doc_embeddings,
            n_components,
            use_umap,
            n_neighbors,
            umap_params,
            seed,
        )
    } else {
        doc_embeddings.to_vec()
    };
    // L2-normalize the PCA scores onto the unit sphere before Euclidean
    // clustering so the metric tracks cosine, the geometry sentence embeddings
    // are trained for; otherwise a few high-variance PCA directions dominate and
    // HDBSCAN under-splits real embeddings. UMAP output is already
    // cosine-structured, so skip it there. `use_umap` falls back to PCA when the
    // `umap` feature is not compiled, so gate on what actually ran.
    let did_pca = did_reduce && !(use_umap && reduce::umap_available());
    if did_pca {
        reduce::l2_normalize_rows(&mut reduced);
    }
    // For GMM (without topic reduction) we capture the EM posterior
    // responsibilities as a soft doc-topic membership (issue #357). With
    // `nr_topics`, the c-TF-IDF merge is a hard-label operation that soft
    // responsibilities can't compose through while keeping `argmax == label`, so
    // we use hard GMM labels there and fall back to the c-TF-IDF `doc_topic`, as
    // every non-GMM clusterer does.
    let gmm_soft_path = clusterer == "gmm" && nr_topics.is_none();
    let (mut labels, gmm_soft) = if gmm_soft_path {
        let (labels, soft) =
            cluster::gmm_soft_labels(&reduced, num_clusters.unwrap_or(min_cluster_size), seed);
        (labels, Some(soft))
    } else {
        let labels = cluster::cluster_points(
            &reduced,
            clusterer,
            num_clusters,
            min_cluster_size,
            min_samples,
            resolution,
            knn_neighbors,
            seed,
        );
        (labels, None)
    };
    let mut num_topics = topic_count(&labels);

    // (3) optional topic reduction: greedily merge the most c-TF-IDF-similar pair
    // of topics until `target` real topics remain. This is topica's own greedy
    // scheme, not upstream's ward agglomeration over topic embeddings (#488).
    if let Some(target) = nr_topics {
        while num_topics > target.max(1) {
            let ctfidf =
                represent::ctfidf_weighted(docs, &labels, vocab_size, bm25, reduce_frequent);
            let (a, b) = most_similar_pair(&ctfidf);
            if a == b {
                break;
            }
            merge_topic(&mut labels, b, a); // fold b into a, relabel above b down
            num_topics -= 1;
        }
    }

    // Final c-TF-IDF and its idf, then the topic-word distribution and the soft
    // document-topic distribution. The idf is computed with the same `bm25` setting
    // so `approximate_distribution` weights its windows the way the topics were.
    let ctfidf_raw = represent::ctfidf_weighted(docs, &labels, vocab_size, bm25, reduce_frequent);
    let idf = idf_weights(docs, &labels, vocab_size, bm25);
    let topic_word = topic_word_from_ctfidf(&ctfidf_raw);
    // GMM contributes a calibrated soft membership (the EM responsibilities); every
    // other clusterer uses the c-TF-IDF approximate distribution. `argmax` of the
    // GMM `doc_topic` equals `labels`, so the hard and soft views agree.
    let doc_topic = match gmm_soft {
        Some(soft) => normalize_rows(soft),
        None => approximate_distribution(
            docs,
            &ctfidf_raw,
            &idf,
            num_topics,
            window,
            stride,
            reduce_frequent,
            min_similarity,
        ),
    };

    BertopicModel {
        num_topics,
        labels,
        topic_word,
        doc_topic,
        ctfidf_raw,
        idf,
        reduce_frequent,
        min_similarity,
    }
}

/// Row-normalize c-TF-IDF into the `topic_word` probability surface topica shares
/// across models (`top_words`, `topic_table`, `coherence` all read it). This is a
/// cross-model compatibility surface, not a probability claim: c-TF-IDF is not
/// P(w|topic), and the viz layer labels it "c-TF-IDF weight" accordingly.
///
/// With `bm25` on, a c-TF-IDF row may carry negative entries for ubiquitous terms
/// (issue #488). We floor those to zero before normalizing so the surface stays a
/// valid non-negative distribution and the row sum never collapses or flips sign;
/// the flooring only affects terms BERTopic already ranks at the bottom, so the
/// top-word ordering matches upstream. The raw c-TF-IDF (with its negatives) is
/// kept separately for the cosine similarities in `approximate_distribution` and
/// topic reduction, which is where upstream uses the signed values.
fn topic_word_from_ctfidf(ctfidf_raw: &[Vec<f64>]) -> Vec<Vec<f64>> {
    ctfidf_raw
        .iter()
        .map(|row| {
            let clamped: Vec<f64> = row.iter().map(|&w| w.max(0.0)).collect();
            let sum: f64 = clamped.iter().sum();
            if sum > 0.0 {
                clamped.iter().map(|w| w / sum).collect()
            } else if !clamped.is_empty() {
                // Every term in this topic had a non-positive (e.g. negative bm25)
                // c-TF-IDF, so flooring leaves an all-zero row. Fall back to uniform
                // to keep `topic_word` a valid distribution (rows sum to 1), matching
                // the `normalize_rows` / `approximate_distribution` zero-row contract.
                let u = 1.0 / clamped.len() as f64;
                vec![u; clamped.len()]
            } else {
                clamped
            }
        })
        .collect()
}

fn topic_count(labels: &[i64]) -> usize {
    labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l as usize + 1)
        .max()
        .unwrap_or(0)
}

/// idf factor per term, matching `represent::ctfidf_weighted` (A is the average
/// class size, f_t the total count of term t across classes). With `bm25` on this
/// is the unclamped class-based BM25 idf `ln(1 + (A - f_t + 0.5) / (f_t + 0.5))`
/// (issue #488), otherwise the plain `ln(1 + A / f_t)`. Kept in step with the
/// weighting the topics were built with so `approximate_distribution` points its
/// windows in the same per-dimension space as the topic rows.
fn idf_weights(docs: &[Vec<u32>], labels: &[i64], vocab_size: usize, bm25: bool) -> Vec<f64> {
    let k = topic_count(labels);
    let mut f = vec![0.0f64; vocab_size];
    let mut class_size = vec![0.0f64; k];
    for (doc, &lab) in docs.iter().zip(labels) {
        if lab < 0 {
            continue;
        }
        let c = lab as usize;
        for &w in doc {
            let w = w as usize;
            if w < vocab_size {
                f[w] += 1.0;
                class_size[c] += 1.0;
            }
        }
    }
    let a = if k > 0 {
        class_size.iter().sum::<f64>() / k as f64
    } else {
        0.0
    };
    f.iter()
        .map(|&ft| {
            if ft <= 0.0 {
                0.0
            } else if bm25 {
                (1.0 + (a - ft + 0.5) / (ft + 0.5)).ln()
            } else {
                (1.0 + a / ft).ln()
            }
        })
        .collect()
}

/// The most cosine-similar pair of topics by their c-TF-IDF rows, `(keep, drop)`
/// with `keep < drop`. Returns `(0, 0)` when there is nothing to merge.
fn most_similar_pair(ctfidf: &[Vec<f64>]) -> (usize, usize) {
    let k = ctfidf.len();
    let mut best = (0usize, 0usize);
    let mut best_sim = f64::NEG_INFINITY;
    for i in 0..k {
        for j in (i + 1)..k {
            let s = cosine(&ctfidf[i], &ctfidf[j]);
            if s > best_sim {
                best_sim = s;
                best = (i, j);
            }
        }
    }
    best
}

/// Fold topic `drop` into topic `keep` and shift every label above `drop` down by
/// one, so labels stay a dense `0..k-1`.
fn merge_topic(labels: &mut [i64], drop: usize, keep: usize) {
    let drop = drop as i64;
    let keep = keep as i64;
    for l in labels.iter_mut() {
        if *l == drop {
            *l = keep;
        } else if *l > drop {
            *l -= 1;
        }
    }
}

/// Row-normalize a matrix to sum to one per row (a zero row becomes uniform).
fn normalize_rows(mut m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    for row in m.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        } else if !row.is_empty() {
            let u = 1.0 / row.len() as f64;
            for x in row.iter_mut() {
                *x = u;
            }
        }
    }
    m
}

/// The soft document-topic distribution. For each document we slide a `window` of
/// tokens (step `stride`), build the window's c-TF-IDF vector with the **same**
/// weighting the topics were built with — `reduce_frequent` damps the window's term
/// counts by a square root before the (possibly BM25) `idf` multiplies them in, so
/// windows and topics live in the same per-dimension space (issue #488) — then take
/// the cosine to every topic. A window↔topic cosine below `min_similarity` (and any
/// negative cosine) is dropped before summing, matching BERTopic's `min_similarity`
/// gate. Rows are normalized to sum to one; a document with no surviving evidence
/// becomes a uniform row so `doc_topic` stays a valid distribution (topica's
/// cross-model contract; upstream leaves it a zero row).
#[allow(clippy::too_many_arguments)]
fn approximate_distribution(
    docs: &[Vec<u32>],
    ctfidf: &[Vec<f64>],
    idf: &[f64],
    num_topics: usize,
    window: usize,
    stride: usize,
    reduce_frequent: bool,
    min_similarity: f64,
) -> Vec<Vec<f64>> {
    if num_topics == 0 {
        return docs.iter().map(|_| Vec::new()).collect();
    }
    let w = window.max(1);
    let s = stride.max(1);

    // A window's weighted vector has at most `w` nonzero tokens, but each topic row is
    // dense over the vocabulary V. The original code built a dense V-length window vector
    // and ran a full O(V) cosine against every topic row, per window, per document —
    // O(docs·windows·topics·V), single-threaded, which dominated `fit` on medium corpora
    // (#557). Here we keep only the window's nonzero (token, weight) pairs and reduce the
    // cosine to O(topics·w) per window, precompute each topic row's L2 norm once (instead
    // of recomputing it inside `cosine` every call), and parallelize across documents.
    //
    // For a fitted model — the only way this runs — every `ctfidf` row has length
    // `idf.len() = V` and holds finite weights, and there are exactly `num_topics` rows.
    // Under that invariant this is BIT-IDENTICAL to the dense path: iterating the nonzero
    // tokens in ascending index order reproduces the exact floating-point accumulation of
    // the `t = 0..V` loop (a skipped entry contributed `vec[t] * row[t] = 0.0 * finite =
    // ±0.0`, which leaves the accumulator unchanged), the topic-row norm is summed over the
    // same `0..V` order, and topics with a zero window/row norm contribute `0.0` (skipped)
    // just as `cosine` returned `0.0` and failed the `sim > 0.0` gate before.
    debug_assert!(
        ctfidf.len() == num_topics && ctfidf.iter().all(|r| r.len() == idf.len()),
        "approximate_distribution expects num_topics rows each of length V = idf.len()"
    );
    let topic_norms: Vec<f64> = ctfidf
        .iter()
        .map(|row| {
            let mut nb = 0.0;
            for &y in row {
                nb += y * y;
            }
            nb.sqrt()
        })
        .collect();

    docs.par_iter()
        .map(|doc| {
            let mut acc = vec![0.0f64; num_topics];
            let mut windows = 0usize;
            let mut start = 0usize;
            // Cap the buffers by the document length: a window holds at most `doc.len()`
            // tokens, and `w` is a public, unbounded parameter (window=usize::MAX must not
            // trigger a capacity-overflow panic the dense path never had).
            let cap = w.min(doc.len());
            let mut toks: Vec<u32> = Vec::with_capacity(cap);
            let mut nonzeros: Vec<(usize, f64)> = Vec::with_capacity(cap);
            loop {
                let end = (start + w).min(doc.len());
                if end > start {
                    // Collect the window's in-vocabulary tokens, then fold repeats into a
                    // single count per token in ascending index order. `sort_unstable`
                    // then a dedup-with-count matches the dense `counts[t]` aggregation.
                    toks.clear();
                    for &tok in &doc[start..end] {
                        if (tok as usize) < idf.len() {
                            toks.push(tok);
                        }
                    }
                    toks.sort_unstable();
                    nonzeros.clear();
                    let mut na = 0.0f64;
                    let mut i = 0;
                    while i < toks.len() {
                        let t = toks[i] as usize;
                        let mut c = 1.0f64;
                        while i + 1 < toks.len() && toks[i + 1] as usize == t {
                            c += 1.0;
                            i += 1;
                        }
                        let tf = if reduce_frequent { c.sqrt() } else { c };
                        let v = tf * idf[t];
                        na += v * v;
                        nonzeros.push((t, v));
                        i += 1;
                    }
                    let na_sqrt = na.sqrt();
                    if na_sqrt > 0.0 {
                        for (k, row) in ctfidf.iter().enumerate() {
                            let nb = topic_norms[k];
                            if nb > 0.0 {
                                let mut dot = 0.0f64;
                                for &(t, v) in &nonzeros {
                                    dot += v * row[t];
                                }
                                let sim = dot / (na_sqrt * nb);
                                if sim >= min_similarity && sim > 0.0 {
                                    acc[k] += sim;
                                }
                            }
                        }
                    }
                    windows += 1;
                }
                if end >= doc.len() {
                    break;
                }
                start += s;
            }
            let sum: f64 = acc.iter().sum();
            if windows > 0 && sum > 0.0 {
                for v in acc.iter_mut() {
                    *v /= sum;
                }
            } else {
                acc.iter_mut().for_each(|v| *v = 1.0 / num_topics as f64);
            }
            acc
        })
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for BertopicModel {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Planted: docs in cluster c use vocabulary block c and embed near center c.
    fn planted(n_clusters: usize, per: usize, seed: u64) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = 12;
        let block = 5;
        let vocab_size = n_clusters * block;
        let centers: Vec<Vec<f64>> = (0..n_clusters)
            .map(|c| {
                let mut v = vec![0.0; dim];
                v[c % dim] = 8.0;
                v
            })
            .collect();
        let mut docs = Vec::new();
        let mut emb = Vec::new();
        for d in 0..(n_clusters * per) {
            let c = d % n_clusters;
            let toks: Vec<u32> = (0..8)
                .map(|_| (c * block + rng.gen_range(0..block)) as u32)
                .collect();
            docs.push(toks);
            emb.push(
                centers[c]
                    .iter()
                    .map(|&v| v + rng.gen::<f64>() * 0.5)
                    .collect(),
            );
        }
        (docs, emb, vocab_size)
    }

    // Clusters that differ by *direction* (each along its own axis) but carry a
    // wide random *radial* magnitude, the geometry L2-normalization is meant to
    // fix (issue #342). On the raw PCA scores the radial spread dominates the
    // Euclidean metric, so a density clusterer mis-splits the rays (here it
    // fragments them); normalizing onto the sphere collapses each ray to its
    // direction and recovers the true clusters. The real sentence-embedding
    // failure is the same cause with the opposite symptom (under-splitting a
    // dense core), but a small deterministic synthetic reproduces the fragmenting
    // form; either way raw clustering is wrong and normalization corrects it.
    fn radial_rays(
        n_clusters: usize,
        per: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = 16;
        let block = 5;
        let vocab_size = n_clusters * block;
        let mut docs = Vec::new();
        let mut emb = Vec::new();
        for d in 0..(n_clusters * per) {
            let c = d % n_clusters;
            let toks: Vec<u32> = (0..8)
                .map(|_| (c * block + rng.gen_range(0..block)) as u32)
                .collect();
            docs.push(toks);
            // Direction = axis c; magnitude spread wide so the radial variance
            // dominates the raw Euclidean metric.
            let mag = 1.0 + rng.gen::<f64>() * 11.0;
            let row: Vec<f64> = (0..dim)
                .map(|t| {
                    let dir = if t == c { mag } else { 0.0 };
                    dir + (rng.gen::<f64>() - 0.5) * 0.3
                })
                .collect();
            emb.push(row);
        }
        (docs, emb, vocab_size)
    }

    #[test]
    fn l2_normalization_recovers_rays_that_raw_pca_misclusters() {
        let (docs, emb, vocab) = radial_rays(4, 60, 7);
        // Raw PCA scores: radial spread dominates the Euclidean metric, so
        // HDBSCAN gets the cluster count wrong.
        let raw = crate::reduce::pca(&emb, 5, 7);
        let raw_k = topic_count(&crate::cluster::cluster_points(
            &raw, "hdbscan", None, 15, 5, 1.0, 15, 7,
        ));
        assert_ne!(raw_k, 4, "raw PCA clustering should mis-count the 4 rays");
        // L2-normalized: each point maps to its ray direction, cleanly separable
        // into exactly the 4 true clusters.
        let mut normed = crate::reduce::pca(&emb, 5, 7);
        crate::reduce::l2_normalize_rows(&mut normed);
        let norm_k = topic_count(&crate::cluster::cluster_points(
            &normed, "hdbscan", None, 15, 5, 1.0, 15, 7,
        ));
        assert_eq!(
            norm_k, 4,
            "normalized clustering should recover exactly 4 rays, got {norm_k}"
        );
        // The shipped pipeline normalizes on the PCA path, so fit_bertopic itself
        // recovers block-pure topics (each topic's top words from one planted block).
        let m = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            5,
            None,
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            7,
        );
        assert_eq!(
            m.num_topics, 4,
            "fit_bertopic recovered {} topics",
            m.num_topics
        );
        for t in 0..m.num_topics {
            let blocks: std::collections::HashSet<usize> =
                m.top_words(4, t).into_iter().map(|(w, _)| w / 5).collect();
            assert_eq!(
                blocks.len(),
                1,
                "topic {t} mixes planted blocks: {blocks:?}"
            );
        }
    }

    #[test]
    fn recovers_topics_via_ctfidf() {
        let (docs, emb, vocab) = planted(3, 40, 1);
        let m = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            2,
            None,
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            1,
        );
        assert!(
            m.num_topics >= 3,
            "expected >=3 topics, got {}",
            m.num_topics
        );
        // Each topic's top words come from a single planted block (block = ids 0..5,
        // 5..10, 10..15).
        for t in 0..m.num_topics {
            let blocks: std::collections::HashSet<usize> =
                m.top_words(4, t).into_iter().map(|(w, _)| w / 5).collect();
            assert_eq!(blocks.len(), 1, "topic {t} mixes blocks: {blocks:?}");
        }
        // doc_topic rows are distributions.
        for row in &m.doc_topic {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sums to {s}");
        }
    }

    #[test]
    fn nr_topics_reduces_to_target() {
        let (docs, emb, vocab) = planted(4, 40, 2);
        let full = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            2,
            None,
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            2,
        );
        assert!(full.num_topics >= 3);
        let reduced = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            2,
            Some(2),
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            2,
        );
        assert_eq!(reduced.num_topics, 2, "should reduce to 2 topics");
    }

    // The pre-#557 dense implementation, verbatim, as the bit-identity oracle.
    fn approximate_distribution_dense_reference(
        docs: &[Vec<u32>],
        ctfidf: &[Vec<f64>],
        idf: &[f64],
        num_topics: usize,
        window: usize,
        stride: usize,
        reduce_frequent: bool,
        min_similarity: f64,
    ) -> Vec<Vec<f64>> {
        let w = window.max(1);
        let s = stride.max(1);
        docs.iter()
            .map(|doc| {
                if num_topics == 0 {
                    return Vec::new();
                }
                let mut acc = vec![0.0f64; num_topics];
                let mut windows = 0usize;
                let mut start = 0usize;
                loop {
                    let end = (start + w).min(doc.len());
                    if end > start {
                        let mut counts = vec![0.0f64; idf.len()];
                        for &tok in &doc[start..end] {
                            let t = tok as usize;
                            if t < counts.len() {
                                counts[t] += 1.0;
                            }
                        }
                        let mut vec = vec![0.0f64; idf.len()];
                        for (t, &c) in counts.iter().enumerate() {
                            if c > 0.0 {
                                let tf = if reduce_frequent { c.sqrt() } else { c };
                                vec[t] = tf * idf[t];
                            }
                        }
                        for (k, row) in ctfidf.iter().enumerate() {
                            let sim = cosine(&vec, row);
                            if sim >= min_similarity && sim > 0.0 {
                                acc[k] += sim;
                            }
                        }
                        windows += 1;
                    }
                    if end >= doc.len() {
                        break;
                    }
                    start += s;
                }
                let sum: f64 = acc.iter().sum();
                if windows > 0 && sum > 0.0 {
                    for v in acc.iter_mut() {
                        *v /= sum;
                    }
                } else {
                    acc.iter_mut().for_each(|v| *v = 1.0 / num_topics as f64);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn approximate_distribution_bit_identical_to_dense() {
        // The sparse+parallel rewrite (#557) must reproduce the dense reference to the
        // last bit across every divergence-prone case (some flagged by the adversarial
        // review of #558): OOV tokens (>= V), repeated tokens in a window, reduce_frequent
        // on/off, the exact min_similarity boundary, a short final window, an EMPTY doc, an
        // ALL-OOV doc, a ZERO-NORM topic row, and an oversized `window` (larger than any
        // doc — must not panic, must match).
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let vocab = 30usize;
        let idf: Vec<f64> = (0..vocab).map(|_| 0.2 + rng.gen::<f64>() * 3.0).collect();
        let num_topics = 6usize;
        let mut ctfidf: Vec<Vec<f64>> = (0..num_topics)
            .map(|_| (0..vocab).map(|_| rng.gen::<f64>() * 2.0 - 0.5).collect())
            .collect();
        ctfidf[num_topics - 1] = vec![0.0; vocab]; // a zero-norm topic row
                                                   // Docs draw tokens in 0..vocab+4, so some land out of vocabulary (>= V).
        let mut docs: Vec<Vec<u32>> = (0..25)
            .map(|_| {
                let len = rng.gen_range(0..40);
                (0..len)
                    .map(|_| rng.gen_range(0..(vocab as u32 + 4)))
                    .collect()
            })
            .collect();
        docs.push(Vec::new()); // an empty document
        docs.push(vec![vocab as u32 + 1; 6]); // an all-out-of-vocabulary document
        for &reduce_frequent in &[false, true] {
            // 0.15 exercises a nonzero gate; 0.0 exercises the exact boundary (sim >= 0.0).
            for &min_sim in &[0.0, 0.15] {
                // The last pair is an oversized window (bigger than every doc).
                for &(win, stride) in &[(4usize, 1usize), (3, 2), (8, 4), (10_000, 1)] {
                    let want = approximate_distribution_dense_reference(
                        &docs,
                        &ctfidf,
                        &idf,
                        num_topics,
                        win,
                        stride,
                        reduce_frequent,
                        min_sim,
                    );
                    let got = approximate_distribution(
                        &docs,
                        &ctfidf,
                        &idf,
                        num_topics,
                        win,
                        stride,
                        reduce_frequent,
                        min_sim,
                    );
                    assert_eq!(
                        want, got,
                        "mismatch at reduce_frequent={reduce_frequent} min_sim={min_sim} win={win} stride={stride}"
                    );
                }
            }
        }
        // num_topics == 0 returns one empty row per document in both paths.
        let want0 = approximate_distribution_dense_reference(&docs, &[], &idf, 0, 4, 1, false, 0.0);
        let got0 = approximate_distribution(&docs, &[], &idf, 0, 4, 1, false, 0.0);
        assert_eq!(want0, got0);
        assert_eq!(got0.len(), docs.len());
        assert!(got0.iter().all(|r| r.is_empty()));
    }

    #[test]
    fn approximate_distribution_favors_own_topic() {
        let (docs, emb, vocab) = planted(3, 40, 3);
        let m = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            2,
            None,
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            3,
        );
        // A document made only of block-0 words should put its largest mass on the
        // topic whose top words are block 0.
        let block0_topic = (0..m.num_topics)
            .find(|&t| m.top_words(1, t)[0].0 / 5 == 0)
            .expect("a block-0 topic exists");
        let doc0: Vec<u32> = vec![0, 1, 2, 3, 4, 0, 1, 2];
        let dist = m.approximate_distribution(&[doc0], 4, 1, None);
        let argmax = (0..m.num_topics)
            .max_by(|&a, &b| dist[0][a].total_cmp(&dist[0][b]))
            .unwrap();
        assert_eq!(argmax, block0_topic, "dist: {:?}", dist[0]);
    }

    #[test]
    fn topic_word_floors_negative_bm25_weights_and_normalizes() {
        // A row with a negative (ubiquitous-term) weight and two positives. The
        // topic_word surface floors the negative to 0 and normalizes the rest, so
        // it stays a valid distribution and the positive ranking is preserved.
        let ctfidf = vec![vec![-2.0, 3.0, 1.0]];
        let tw = topic_word_from_ctfidf(&ctfidf);
        assert_eq!(tw[0][0], 0.0, "negative weight floored to zero");
        let s: f64 = tw[0].iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "row sums to {s}");
        assert!(tw[0][1] > tw[0][2], "positive ranking preserved");
    }

    #[test]
    fn topic_word_all_negative_row_falls_back_to_uniform() {
        // If every term in a topic has a non-positive (all-ubiquitous, negative
        // bm25) c-TF-IDF, flooring leaves an all-zero row; topic_word must fall back
        // to uniform so it stays a valid distribution (rows sum to 1), not all-zero.
        let ctfidf = vec![vec![-2.0, -0.5, -3.0]];
        let tw = topic_word_from_ctfidf(&ctfidf);
        let s: f64 = tw[0].iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "all-negative row must sum to 1, got {s}"
        );
        assert!(
            tw[0].iter().all(|&w| (w - 1.0 / 3.0).abs() < 1e-12),
            "uniform fallback"
        );
    }

    #[test]
    fn idf_weights_bm25_matches_ctfidf_weighted() {
        // The stored idf must equal the per-term BM25 idf embedded in
        // ctfidf_weighted, so windows and topics share one weighting (#488).
        let docs = vec![
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        ];
        let labels = vec![0, 1];
        let idf = idf_weights(&docs, &labels, 3, true);
        // Reconstruct the idf from ctfidf_weighted: divide class-0's word-0 weight
        // (tf=10) by its idf factor.
        let ct = represent::ctfidf_weighted(&docs, &labels, 3, true, false);
        let recovered = ct[0][0] / 10.0;
        assert!(
            (idf[0] - recovered).abs() < 1e-12,
            "{} vs {}",
            idf[0],
            recovered
        );
        assert!(idf[0] < 0.0, "ubiquitous term has negative bm25 idf");
    }

    #[test]
    fn approximate_distribution_min_similarity_gates_low_matches() {
        // Two orthogonal topics; a window matching topic 0 exactly. With no gate
        // both topics could receive mass; a high gate keeps only the strong match.
        let ctfidf = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]];
        let idf = vec![1.0, 1.0, 1.0, 1.0];
        let docs = vec![vec![0u32]]; // one token, aligns with topic 0 only
        let gated = approximate_distribution(&docs, &ctfidf, &idf, 2, 4, 1, false, 0.5);
        assert!((gated[0][0] - 1.0).abs() < 1e-12, "all mass on topic 0");
        assert_eq!(gated[0][1], 0.0);
        // A window with no surviving similarity (gate above every match) falls back
        // to a uniform row rather than a zero row (topica's distribution contract).
        let all_gated = approximate_distribution(&docs, &ctfidf, &idf, 2, 4, 1, false, 1.5);
        assert!((all_gated[0][0] - 0.5).abs() < 1e-12);
        assert!((all_gated[0][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bertopic_conforms() {
        let (docs, emb, vocab) = planted(3, 40, 1);
        let m = fit_bertopic(
            &docs,
            &emb,
            vocab,
            5,
            false,
            15,
            15,
            2,
            None,
            4,
            1,
            false,
            false,
            0.0,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            1,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
