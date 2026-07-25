//! Top2Vec: the first port in topica's embedding-native model branch.
//!
//! Top2Vec (Angelov 2020) finds topics by clustering document embeddings: reduce
//! the embeddings to a few dimensions, run density clustering, and read each
//! topic off its cluster. A topic is a point in the embedding space (the mean of
//! its documents' embeddings), and its words are the vocabulary terms whose
//! embeddings sit nearest that point. We follow the same three stages topica
//! shares across this branch, `reduce -> cluster -> represent`:
//!
//! 1. `reduce::pca` takes the document embeddings down to `n_components` dims.
//! 2. `cluster::hdbscan_labels` groups the reduced points; sparse points become
//!    noise (label `-1`), the Top2Vec/HDBSCAN convention.
//! 3. `represent` builds the topic representation: the topic vector is the mean
//!    of the cluster's document embeddings, the topic words come two ways, by
//!    nearest word vectors (Top2Vec's own definition) and by class-based TF-IDF
//!    (so the model also exposes the `topic_word` distribution every other
//!    topica model has).
//!
//! Unlike the original, topica does not embed the text itself. The caller brings
//! `doc_embeddings` and `word_embeddings` in a shared space (e.g. from a
//! sentence-transformer run over the documents and the vocabulary), exactly as
//! [`crate::seeded`]-backed `EmbeddingLDA` does.

use crate::{cluster, reduce, represent};

/// A fitted Top2Vec model. The fields are the surface this whole branch shares:
/// `topic_word` (K x V) and `doc_topic` (D x K) match every other topica model,
/// and `topic_vectors` (K x E) is the embedding-native addition.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Top2VecModel {
    /// Number of topics discovered (clusters found by HDBSCAN).
    pub num_topics: usize,
    /// Hard cluster assignment per document; `-1` marks a noise document that
    /// joined no topic.
    pub labels: Vec<i64>,
    /// Each topic's location in the embedding space: the mean of its documents'
    /// embeddings (K x E).
    pub topic_vectors: Vec<Vec<f64>>,
    /// Topic-word distribution from class-based TF-IDF, row-normalized to sum to
    /// one (K x V), so coherence and the rest of topica's surface work unchanged.
    pub topic_word: Vec<Vec<f64>>,
    /// Soft document-topic membership (D x K): cosine of each document embedding
    /// to each topic vector, clamped at zero and normalized. Rows of an all-noise
    /// or zero-similarity document fall back to uniform.
    pub doc_topic: Vec<Vec<f64>>,
    /// The word embeddings, kept so `topic_neighbors` can rank vocabulary terms
    /// against a topic vector (V x E).
    word_vectors: Vec<Vec<f64>>,
    /// The document embeddings, kept so the search API can rank documents by
    /// cosine to a topic vector or a keyword query (D x E). `#[serde(default)]`
    /// leaves this empty whenever the field is absent, and `has_doc_vectors()`
    /// then makes the document searches raise cleanly instead of indexing a
    /// missing matrix. Note this default does NOT rescue a genuinely pre-search
    /// save: topica's save format is bincode (not self-describing), so a payload
    /// written before this field existed fails to `load()` outright ("unexpected
    /// end of file") rather than deserializing with an empty set — such a model
    /// must be refit. The default/guard are defensive, not an old-save migration.
    #[serde(default)]
    doc_vectors: Vec<Vec<f64>>,
}

impl Top2VecModel {
    /// The top `n` words of `topic` by class-based TF-IDF weight, as
    /// `(word_id, weight)`. This is the BERTopic-style representation.
    pub fn top_words(&self, n: usize, topic: usize) -> Vec<(usize, f64)> {
        represent::top_indices(&self.topic_word[topic], n)
    }

    /// The top `n` vocabulary words nearest `topic`'s embedding by cosine, as
    /// `(word_id, cosine)`. This is Top2Vec's own topic-word definition and the
    /// branch-wide `topic_neighbors` surface.
    pub fn topic_neighbors(&self, n: usize, topic: usize) -> Vec<(usize, f64)> {
        represent::nearest_by_cosine(&self.topic_vectors[topic], &self.word_vectors, n)
    }

    /// Soft topic membership for new document embeddings (D×K): cosine to each
    /// topic vector, clamped at zero and normalized. This is held-out `transform`,
    /// the same assignment the fit uses for in-sample documents.
    pub fn assign(&self, doc_embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        soft_doc_topic(doc_embeddings, &self.topic_vectors)
    }

    /// Whether the document embeddings were retained (they are for any model fit
    /// after the search API landed; models saved earlier deserialize without
    /// them). The document-ranking searches gate on this.
    pub fn has_doc_vectors(&self) -> bool {
        !self.doc_vectors.is_empty()
    }

    /// The embedding dimension `E` (word, document, and topic vectors share it),
    /// or 0 for an empty model. Used to validate a query vector's length.
    pub fn embedding_dim(&self) -> usize {
        self.word_vectors
            .first()
            .or_else(|| self.topic_vectors.first())
            .or_else(|| self.doc_vectors.first())
            .map_or(0, |v| v.len())
    }

    /// Per-topic document count (non-noise), indexed by topic id. Noise (`-1`)
    /// documents are excluded, so the counts sum to the number of clustered docs.
    pub fn topic_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.num_topics];
        for &l in &self.labels {
            if l >= 0 && (l as usize) < self.num_topics {
                sizes[l as usize] += 1;
            }
        }
        sizes
    }

    /// The centroid (mean) of the word vectors for `word_ids`, the query a
    /// keyword search embeds. `None` when the list is empty, so a search with no
    /// in-vocabulary keyword raises cleanly rather than dividing by zero.
    pub fn keyword_centroid(&self, word_ids: &[usize]) -> Option<Vec<f64>> {
        if word_ids.is_empty() {
            return None;
        }
        let e = self.word_vectors.first().map_or(0, |v| v.len());
        let mut c = vec![0.0f64; e];
        for &id in word_ids {
            for (d, &x) in self.word_vectors[id].iter().enumerate() {
                c[d] += x;
            }
        }
        let n = word_ids.len() as f64;
        for x in c.iter_mut() {
            *x /= n;
        }
        Some(c)
    }

    /// The `n` vocabulary words nearest `query` by cosine, as `(word_id, cosine)`.
    pub fn search_words_by_vector(&self, n: usize, query: &[f64]) -> Vec<(usize, f64)> {
        represent::nearest_by_cosine(query, &self.word_vectors, n)
    }

    /// The `n` topics whose vectors are nearest `query` by cosine, as
    /// `(topic_id, cosine)`.
    pub fn search_topics_by_vector(&self, n: usize, query: &[f64]) -> Vec<(usize, f64)> {
        represent::nearest_by_cosine(query, &self.topic_vectors, n)
    }

    /// The `n` documents whose embeddings are nearest `query` by cosine, as
    /// `(doc_index, cosine)`. Ranks over every document (noise included, since a
    /// free keyword query is not tied to a topic).
    pub fn search_documents_by_vector(&self, n: usize, query: &[f64]) -> Vec<(usize, f64)> {
        represent::nearest_by_cosine(query, &self.doc_vectors, n)
    }

    /// The documents hard-assigned to `topic`, ranked by cosine of their
    /// embedding to the topic vector, as `(doc_index, cosine)` (up to `n`). This
    /// mirrors the reference `search_documents_by_topic`: only a topic's own
    /// members are ranked, so noise (`-1`) documents never appear.
    pub fn documents_in_topic(&self, n: usize, topic: usize) -> Vec<(usize, f64)> {
        let tv = &self.topic_vectors[topic];
        let mut scored: Vec<(usize, f64)> = self
            .labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == topic as i64)
            .map(|(i, _)| (i, cosine(&self.doc_vectors[i], tv)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(n);
        scored
    }

    /// Merge each group of topics into one. The merged topic vector is the
    /// size-weighted mean of its members' vectors (which is exactly the centroid
    /// of the merged documents), the merged document-topic column is the sum of
    /// its members' columns, and the topic-word matrix is recomputed by c-TF-IDF.
    pub fn merge_topics(&mut self, docs: &[Vec<u32>], groups: &[Vec<usize>], vocab_size: usize) {
        let old_k = self.num_topics;
        if old_k == 0 {
            return;
        }
        let map = represent::merge_labels(&(0..old_k as i64).collect::<Vec<_>>(), groups);
        let new_k = map.iter().map(|&m| m + 1).max().unwrap_or(0) as usize;

        let mut sizes = vec![0.0f64; old_k];
        for &l in &self.labels {
            if l >= 0 {
                sizes[l as usize] += 1.0;
            }
        }
        let e = self.topic_vectors.first().map_or(0, |v| v.len());
        let mut vectors = vec![vec![0.0f64; e]; new_k];
        let mut wsum = vec![0.0f64; new_k];
        for ot in 0..old_k {
            let nt = map[ot] as usize;
            let w = sizes[ot].max(1e-9);
            for d in 0..e {
                vectors[nt][d] += self.topic_vectors[ot][d] * w;
            }
            wsum[nt] += w;
        }
        for nt in 0..new_k {
            if wsum[nt] > 0.0 {
                for d in 0..e {
                    vectors[nt][d] /= wsum[nt];
                }
            }
        }

        let doc_topic = self
            .doc_topic
            .iter()
            .map(|row| {
                let mut nr = vec![0.0f64; new_k];
                for (ot, &p) in row.iter().enumerate().take(old_k) {
                    nr[map[ot] as usize] += p;
                }
                let s: f64 = nr.iter().sum();
                if s > 0.0 {
                    nr.iter_mut().for_each(|x| *x /= s);
                } else {
                    nr.iter_mut().for_each(|x| *x = 1.0 / new_k as f64);
                }
                nr
            })
            .collect();

        self.labels = represent::merge_labels(&self.labels, groups);
        self.num_topics = new_k;
        self.topic_vectors = vectors;
        self.doc_topic = doc_topic;
        self.topic_word = normalize_rows(represent::ctfidf(docs, &self.labels, vocab_size));
    }

    /// Reassign noise documents to their nearest topic (by topic-word fit) and
    /// recompute the topic-word matrix. The topic vectors and soft memberships are
    /// left unchanged.
    pub fn reduce_outliers(&mut self, docs: &[Vec<u32>], vocab_size: usize) {
        self.labels = represent::assign_outliers(docs, &self.labels, &self.topic_word);
        self.topic_word = normalize_rows(represent::ctfidf(docs, &self.labels, vocab_size));
    }

    /// Reduce to `target` topics the reference-top2vec way: repeatedly merge the
    /// smallest topic (fewest documents) into its nearest topic by topic-vector
    /// cosine, recomputing the merged vector as the size-weighted mean, until
    /// `target` topics remain. This is the driver behind the reference
    /// `hierarchical_topic_reduction(num_topics)`; the merge geometry itself is
    /// [`Self::merge_topics`]. Topics are reordered by size afterwards (topic 0
    /// largest), as the reference does. `docs`/`vocab_size` rebuild the c-TF-IDF
    /// `topic_word` after each merge. No-op when `target` is already >= the current
    /// topic count.
    pub fn hierarchical_topic_reduction(
        &mut self,
        docs: &[Vec<u32>],
        target: usize,
        vocab_size: usize,
    ) {
        while self.num_topics > target && self.num_topics > 1 {
            let sizes = self.topic_sizes();
            // Smallest topic (fewest documents), ties broken by smallest id.
            let smallest = (0..self.num_topics)
                .min_by(|&a, &b| sizes[a].cmp(&sizes[b]).then(a.cmp(&b)))
                .unwrap();
            // Nearest other topic by topic-vector cosine, ties -> smallest id.
            let sv = &self.topic_vectors[smallest];
            let nearest = (0..self.num_topics)
                .filter(|&t| t != smallest)
                .max_by(|&x, &y| {
                    cosine(sv, &self.topic_vectors[x])
                        .partial_cmp(&cosine(sv, &self.topic_vectors[y]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(y.cmp(&x))
                })
                .unwrap();
            self.merge_topics(docs, &[vec![smallest, nearest]], vocab_size);
        }
        self.reorder_by_size();
    }

    /// Permute topic ids so topic 0 is the largest (most documents), matching the
    /// reference top2vec, which reorders topics by document count descending right
    /// after clustering (`_calculate_topic_sizes` + `_reorder_topics`). Ties break
    /// by original id. Noise (`-1`) labels are untouched. Every topic-indexed
    /// surface — `topic_vectors`, `topic_word`, the `doc_topic` columns, and the
    /// `labels` — is permuted together so the model stays internally consistent.
    fn reorder_by_size(&mut self) {
        let k = self.num_topics;
        if k <= 1 {
            return;
        }
        let sizes = self.topic_sizes();
        // order[new_id] = old_id, sorted by size descending then id ascending.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
        if order.iter().enumerate().all(|(new, &old)| new == old) {
            return; // already size-ordered
        }
        let mut old_to_new = vec![0usize; k];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old] = new;
        }
        self.topic_vectors = order
            .iter()
            .map(|&o| self.topic_vectors[o].clone())
            .collect();
        self.topic_word = order.iter().map(|&o| self.topic_word[o].clone()).collect();
        for row in self.doc_topic.iter_mut() {
            *row = order.iter().map(|&o| row[o]).collect();
        }
        for l in self.labels.iter_mut() {
            if *l >= 0 {
                *l = old_to_new[*l as usize] as i64;
            }
        }
    }
}

/// Row-normalize a matrix to sum to one per row (zero rows left as is).
fn normalize_rows(mut m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    for row in m.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }
    m
}

/// Fit Top2Vec on token-id documents plus document and word embeddings.
///
/// `docs[d]` lists the word ids in document `d` (ids in `0..vocab_size`), used
/// only for the class-TF-IDF representation. `doc_embeddings` (D x E) drives the
/// clustering; `word_embeddings` (V x E) places the vocabulary in the same space
/// for `topic_neighbors`. `n_components` is the reduced dimensionality before
/// clustering; `min_cluster_size`/`min_samples` are HDBSCAN's. `clusterer` is
/// `"hdbscan"` (default), `"kmeans"`, or `"agglomerative"`; the latter two assign
/// every document to `num_clusters` clusters (no `-1` noise bucket).
#[allow(clippy::too_many_arguments)]
pub fn fit_top2vec(
    docs: &[Vec<u32>],
    doc_embeddings: &[Vec<f64>],
    word_embeddings: &[Vec<f64>],
    vocab_size: usize,
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    clusterer: &str,
    num_clusters: Option<usize>,
    resolution: f64,
    knn_neighbors: usize,
    umap_params: &reduce::UmapParams,
    seed: u64,
) -> Top2VecModel {
    let n_docs = doc_embeddings.len();
    let emb_dim = if n_docs > 0 {
        doc_embeddings[0].len()
    } else {
        0
    };

    // (1) Reduce, unless the embeddings are already at or below the target dim.
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
    // clustering so the metric tracks cosine (see `fit_bertopic`); UMAP output is
    // already cosine-structured, and `use_umap` silently falls back to PCA when
    // the `umap` feature is absent, so gate on what actually ran.
    let did_pca = did_reduce && !(use_umap && reduce::umap_available());
    if did_pca {
        reduce::l2_normalize_rows(&mut reduced);
    }

    // (2) Cluster the reduced points. HDBSCAN (the default) leaves sparse points
    // as `-1` noise; KMeans / agglomerative assign every document instead.
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
    let num_topics = labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l as usize + 1)
        .max()
        .unwrap_or(0);

    // (3) Represent: topic vectors (centroids in the original space), topic-word
    // distribution (normalized class-TF-IDF), and soft doc-topic memberships.
    let topic_vectors = represent::centroids(doc_embeddings, &labels, num_topics);
    let mut topic_word = represent::ctfidf(docs, &labels, vocab_size);
    for row in topic_word.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            for w in row.iter_mut() {
                *w /= sum;
            }
        }
    }

    let doc_topic = soft_doc_topic(doc_embeddings, &topic_vectors);

    let mut model = Top2VecModel {
        num_topics,
        labels,
        topic_vectors,
        topic_word,
        doc_topic,
        word_vectors: word_embeddings.to_vec(),
        doc_vectors: doc_embeddings.to_vec(),
    };
    // Reorder topics by size (topic 0 largest), as the reference top2vec does
    // right after clustering, so per-index comparisons line up.
    model.reorder_by_size();
    model
}

/// Soft membership: cosine of each document embedding to each topic vector,
/// negatives clamped to zero, then normalized to a distribution. A document with
/// no positive similarity (or when there are no topics) gets a uniform row.
fn soft_doc_topic(doc_embeddings: &[Vec<f64>], topic_vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = topic_vectors.len();
    doc_embeddings
        .iter()
        .map(|d| {
            if k == 0 {
                return Vec::new();
            }
            let mut row: Vec<f64> = topic_vectors
                .iter()
                .map(|t| cosine(d, t).max(0.0))
                .collect();
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            } else {
                row.iter_mut().for_each(|v| *v = 1.0 / k as f64);
            }
            row
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

impl Estimator for Top2VecModel {
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

    // Two well-separated topics: documents in cluster 0 use words 0..5 and embed
    // near center A; cluster 1 uses words 5..10 and embeds near center B. Word
    // embeddings sit near the center of the block they belong to. Top2Vec should
    // recover two topics whose top words come from the matching block.
    #[test]
    fn recovers_two_planted_topics() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let dim = 8;
        let mut center_a = vec![0.0; dim];
        let mut center_b = vec![0.0; dim];
        center_a[0] = 5.0;
        center_b[4] = 5.0;

        let jitter = |rng: &mut ChaCha8Rng, c: &[f64]| -> Vec<f64> {
            c.iter().map(|&v| v + rng.gen::<f64>() * 0.4).collect()
        };

        let mut docs = Vec::new();
        let mut doc_emb = Vec::new();
        for d in 0..40 {
            if d % 2 == 0 {
                docs.push(vec![0u32, 1, 2, 3, 4].into_iter().collect::<Vec<_>>());
                doc_emb.push(jitter(&mut rng, &center_a));
            } else {
                docs.push((5u32..10).collect::<Vec<_>>());
                doc_emb.push(jitter(&mut rng, &center_b));
            }
        }
        // Word embeddings: words 0..5 near center A, 5..10 near center B.
        let mut word_emb = Vec::new();
        for w in 0..10 {
            let c = if w < 5 { &center_a } else { &center_b };
            word_emb.push(jitter(&mut rng, c));
        }

        let m = fit_top2vec(
            &docs,
            &doc_emb,
            &word_emb,
            10,
            5,
            false,
            15,
            5,
            2,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            1,
        );
        assert!(
            m.num_topics >= 2,
            "expected >=2 topics, got {}",
            m.num_topics
        );

        // Each topic's nearest words should come from a single block.
        for t in 0..m.num_topics {
            let words: Vec<usize> = m
                .topic_neighbors(4, t)
                .into_iter()
                .map(|(w, _)| w)
                .collect();
            let low = words.iter().filter(|&&w| w < 5).count();
            assert!(
                low == 0 || low == words.len(),
                "topic {t} mixes word blocks: {words:?}"
            );
        }
        // doc_topic rows are valid distributions.
        for row in &m.doc_topic {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "doc_topic row sums to {s}");
        }
    }

    #[test]
    fn all_noise_yields_no_topics() {
        // A handful of scattered points with a large min_cluster_size finds no
        // cluster; the model should be empty, not panic.
        let doc_emb: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 10.0, 0.0]).collect();
        let docs: Vec<Vec<u32>> = (0..5).map(|_| vec![0u32]).collect();
        let word_emb = vec![vec![1.0, 0.0]];
        let m = fit_top2vec(
            &docs,
            &doc_emb,
            &word_emb,
            1,
            2,
            false,
            15,
            5,
            2,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            1,
        );
        assert_eq!(m.num_topics, 0);
        assert!(m.topic_vectors.is_empty());
    }

    #[test]
    fn top2vec_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let dim = 8;
        let mut center_a = vec![0.0; dim];
        let mut center_b = vec![0.0; dim];
        center_a[0] = 5.0;
        center_b[4] = 5.0;

        let jitter = |rng: &mut ChaCha8Rng, c: &[f64]| -> Vec<f64> {
            c.iter().map(|&v| v + rng.gen::<f64>() * 0.4).collect()
        };

        let mut docs = Vec::new();
        let mut doc_emb = Vec::new();
        for d in 0..40 {
            if d % 2 == 0 {
                docs.push(vec![0u32, 1, 2, 3, 4].into_iter().collect::<Vec<_>>());
                doc_emb.push(jitter(&mut rng, &center_a));
            } else {
                docs.push((5u32..10).collect::<Vec<_>>());
                doc_emb.push(jitter(&mut rng, &center_b));
            }
        }
        let mut word_emb = Vec::new();
        for w in 0..10 {
            let c = if w < 5 { &center_a } else { &center_b };
            word_emb.push(jitter(&mut rng, c));
        }
        let m = fit_top2vec(
            &docs,
            &doc_emb,
            &word_emb,
            10,
            5,
            false,
            15,
            5,
            2,
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

    // Three planted blocks of unequal size (24 / 12 / 6 documents). After fit the
    // topics must be size-ordered (topic 0 largest), and reducing to two topics
    // must merge the two nearest blocks and leave two size-ordered topics.
    fn fit_three_blocks() -> (Vec<Vec<u32>>, Top2VecModel) {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let dim = 9;
        let mut centers = vec![vec![0.0; dim]; 3];
        centers[0][0] = 6.0;
        centers[1][3] = 6.0;
        // Block 2 sits close to block 1 so the reduction merges 1 and 2 first.
        centers[2][3] = 5.0;
        centers[2][6] = 2.0;

        let jitter = |rng: &mut ChaCha8Rng, c: &[f64]| -> Vec<f64> {
            c.iter().map(|&v| v + rng.gen::<f64>() * 0.3).collect()
        };

        let counts = [24usize, 12, 6];
        let words: [Vec<u32>; 3] = [vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
        let mut docs = Vec::new();
        let mut doc_emb = Vec::new();
        for (b, &n) in counts.iter().enumerate() {
            for _ in 0..n {
                docs.push(words[b].clone());
                doc_emb.push(jitter(&mut rng, &centers[b]));
            }
        }
        let mut word_emb = Vec::new();
        for w in 0..12 {
            word_emb.push(jitter(&mut rng, &centers[w / 4]));
        }
        let m = fit_top2vec(
            &docs,
            &doc_emb,
            &word_emb,
            12,
            5,
            false,
            15,
            4,
            2,
            "hdbscan",
            None,
            1.0,
            15,
            &crate::reduce::UmapParams::default(),
            7,
        );
        (docs, m)
    }

    #[test]
    fn fit_is_size_ordered() {
        let (_docs, m) = fit_three_blocks();
        assert!(
            m.num_topics >= 2,
            "expected planted topics, got {}",
            m.num_topics
        );
        let sizes = m.topic_sizes();
        assert!(
            sizes.windows(2).all(|w| w[0] >= w[1]),
            "topics not size-ordered: {sizes:?}"
        );
    }

    #[test]
    fn hierarchical_reduction_reaches_target() {
        let (docs, mut m) = fit_three_blocks();
        let k0 = m.num_topics;
        assert!(k0 >= 3, "need >=3 topics to reduce, got {k0}");
        m.hierarchical_topic_reduction(&docs, 2, 12);
        assert_eq!(m.num_topics, 2, "reduction did not reach target");
        // Still size-ordered and a valid doc_topic after reduction.
        let sizes = m.topic_sizes();
        assert!(
            sizes.windows(2).all(|w| w[0] >= w[1]),
            "not size-ordered: {sizes:?}"
        );
        for row in &m.doc_topic {
            assert_eq!(row.len(), 2);
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "doc_topic row sums to {s}");
        }
        // No-op when the target is not smaller than the current count.
        let before = m.num_topics;
        m.hierarchical_topic_reduction(&docs, 5, 12);
        assert_eq!(
            m.num_topics, before,
            "reduction changed count when target >= current"
        );
    }
}
