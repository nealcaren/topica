//! BERTopic: the second head on topica's embedding-clustering pipeline.
//!
//! BERTopic (Grootendorst 2022) shares Top2Vec's `reduce -> cluster -> represent`
//! shape but differs in the representation: a topic is defined by **class-based
//! TF-IDF** over its documents' words, not by a point in the embedding space, so
//! BERTopic needs no word embeddings. Two features are characteristic and we port
//! them here:
//!
//! - `nr_topics`: agglomeratively merge the most c-TF-IDF-similar topics down to a
//!   target count, BERTopic's topic reduction.
//! - `approximate_distribution`: a soft per-document topic distribution. We slide
//!   a window over each document, build the window's c-TF-IDF vector, measure its
//!   cosine to every topic, and average across windows. This is the document-topic
//!   distribution BERTopic reports without re-running the clustering.
//!
//! As elsewhere in this branch, the caller brings the document embeddings; topica
//! does not embed the text.

use crate::{cluster, reduce, represent};

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
    /// recomputed at other window sizes after fitting.
    ctfidf_raw: Vec<Vec<f64>>,
    /// idf weights `ln(1 + A / f_t)` per vocabulary term.
    idf: Vec<f64>,
}

impl BertopicModel {
    /// Top `n` words of `topic` by c-TF-IDF weight, as `(word_id, weight)`.
    pub fn top_words(&self, n: usize, topic: usize) -> Vec<(usize, f64)> {
        represent::top_indices(&self.topic_word[topic], n)
    }

    /// Recompute the soft document-topic distribution at a chosen `window`/`stride`
    /// over each document's tokens. Rows sum to one.
    pub fn approximate_distribution(
        &self,
        docs: &[Vec<u32>],
        window: usize,
        stride: usize,
    ) -> Vec<Vec<f64>> {
        approximate_distribution(
            docs,
            &self.ctfidf_raw,
            &self.idf,
            self.num_topics,
            window,
            stride,
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
        self.idf = idf_weights(docs, &self.labels, vocab_size);
        self.topic_word = self
            .ctfidf_raw
            .iter()
            .map(|row| {
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    row.iter().map(|w| w / s).collect()
                } else {
                    row.clone()
                }
            })
            .collect();
        self.doc_topic = approximate_distribution(
            docs,
            &self.ctfidf_raw,
            &self.idf,
            self.num_topics,
            window,
            stride,
        );
    }
}

/// Fit BERTopic on token-id documents plus document embeddings. `nr_topics`, if
/// set, reduces the discovered topics to that count by merging the most similar.
/// `window`/`stride` parameterize the approximate distribution used for
/// `doc_topic`.
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
    clusterer: &str,
    num_clusters: Option<usize>,
    resolution: f64,
    knn_neighbors: usize,
    seed: u64,
) -> BertopicModel {
    let emb_dim = doc_embeddings.first().map_or(0, |r| r.len());

    // (1) reduce, (2) cluster. HDBSCAN (default) leaves `-1` noise; KMeans /
    // agglomerative assign every document instead.
    let did_reduce = emb_dim > n_components && n_components > 0;
    let mut reduced: Vec<Vec<f64>> = if did_reduce {
        reduce::reduce(doc_embeddings, n_components, use_umap, n_neighbors, seed)
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

    // (3) optional topic reduction: merge the most c-TF-IDF-similar topics.
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
    // document-topic distribution.
    let ctfidf_raw = represent::ctfidf_weighted(docs, &labels, vocab_size, bm25, reduce_frequent);
    let idf = idf_weights(docs, &labels, vocab_size);
    // Row-normalize c-TF-IDF to sum to one so topic_word has the same surface as
    // the probability models' (top_words, topic_table, coherence all read it).
    // This is for cross-model compatibility, not a probability claim: c-TF-IDF is
    // not P(w|topic), and the viz layer labels it "c-TF-IDF weight" accordingly.
    let mut topic_word = ctfidf_raw.clone();
    for row in topic_word.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            for w in row.iter_mut() {
                *w /= sum;
            }
        }
    }
    // GMM contributes a calibrated soft membership (the EM responsibilities); every
    // other clusterer uses the c-TF-IDF approximate distribution. `argmax` of the
    // GMM `doc_topic` equals `labels`, so the hard and soft views agree.
    let doc_topic = match gmm_soft {
        Some(soft) => normalize_rows(soft),
        None => approximate_distribution(docs, &ctfidf_raw, &idf, num_topics, window, stride),
    };

    BertopicModel {
        num_topics,
        labels,
        topic_word,
        doc_topic,
        ctfidf_raw,
        idf,
    }
}

fn topic_count(labels: &[i64]) -> usize {
    labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l as usize + 1)
        .max()
        .unwrap_or(0)
}

/// idf factor `ln(1 + A / f_t)` per term, matching `represent::ctfidf` (A is the
/// average class size, f_t the total count of term t across classes).
fn idf_weights(docs: &[Vec<u32>], labels: &[i64], vocab_size: usize) -> Vec<f64> {
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
        .map(|&ft| if ft > 0.0 { (1.0 + a / ft).ln() } else { 0.0 })
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
/// tokens (step `stride`), weight the window's term counts by idf to form its
/// c-TF-IDF vector, take the cosine to every topic, clamp negatives, and average
/// across windows. Rows are normalized to sum to one (uniform when empty).
fn approximate_distribution(
    docs: &[Vec<u32>],
    ctfidf: &[Vec<f64>],
    idf: &[f64],
    num_topics: usize,
    window: usize,
    stride: usize,
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
                    // Build the window's idf-weighted bag of words.
                    let mut vec = vec![0.0f64; idf.len()];
                    for &tok in &doc[start..end] {
                        let t = tok as usize;
                        if t < vec.len() {
                            vec[t] += idf[t];
                        }
                    }
                    for (k, row) in ctfidf.iter().enumerate() {
                        acc[k] += cosine(&vec, row).max(0.0);
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
            &docs, &emb, vocab, 5, false, 15, 15, 5, None, 4, 1, false, false, "hdbscan", None,
            1.0, 15, 7,
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
            &docs, &emb, vocab, 5, false, 15, 15, 2, None, 4, 1, false, false, "hdbscan", None,
            1.0, 15, 1,
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
            &docs, &emb, vocab, 5, false, 15, 15, 2, None, 4, 1, false, false, "hdbscan", None,
            1.0, 15, 2,
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
            "hdbscan",
            None,
            1.0,
            15,
            2,
        );
        assert_eq!(reduced.num_topics, 2, "should reduce to 2 topics");
    }

    #[test]
    fn approximate_distribution_favors_own_topic() {
        let (docs, emb, vocab) = planted(3, 40, 3);
        let m = fit_bertopic(
            &docs, &emb, vocab, 5, false, 15, 15, 2, None, 4, 1, false, false, "hdbscan", None,
            1.0, 15, 3,
        );
        // A document made only of block-0 words should put its largest mass on the
        // topic whose top words are block 0.
        let block0_topic = (0..m.num_topics)
            .find(|&t| m.top_words(1, t)[0].0 / 5 == 0)
            .expect("a block-0 topic exists");
        let doc0: Vec<u32> = vec![0, 1, 2, 3, 4, 0, 1, 2];
        let dist = m.approximate_distribution(&[doc0], 4, 1);
        let argmax = (0..m.num_topics)
            .max_by(|&a, &b| dist[0][a].total_cmp(&dist[0][b]))
            .unwrap();
        assert_eq!(argmax, block0_topic, "dist: {:?}", dist[0]);
    }

    #[test]
    fn bertopic_conforms() {
        let (docs, emb, vocab) = planted(3, 40, 1);
        let m = fit_bertopic(
            &docs, &emb, vocab, 5, false, 15, 15, 2, None, 4, 1, false, false, "hdbscan", None,
            1.0, 15, 1,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
