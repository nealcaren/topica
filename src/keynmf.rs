//! KeyNMF (Kristensen-McLachlan et al. 2024): an NMF topic model whose factored
//! matrix is an **embedding-derived keyword-importance** matrix rather than raw
//! counts. For each document, every candidate word (a word present in the document
//! and in the vocabulary) is scored by the similarity between the document embedding
//! and the word embedding; the top-N positive words become a sparse doc x word
//! importance matrix, which is then factored by NMF. Semantics come from the
//! embeddings, giving sparse, readable topics robust to short/noisy text — the
//! bridge between topica's `NMF`/`GuidedNMF` (count matrix) and the embedding backend.
//!
//! Faithful to the KeyNMF *method*, validated against the MIT-licensed turftopic
//! reference at the method level. topica implements the **correct** keyword
//! extraction (paired top-N positive) and does NOT replicate turftopic's stage-1
//! implementation bugs (a `zip` that scrambles word->importance pairs when a selected
//! similarity is <= 0, and an off-by-one `kth = min(top_n, n_present-1)` that drops a
//! keyword). The NMF backend is topica's own (NNDSVD init + Frobenius multiplicative
//! updates), so it matches turftopic's sklearn NMF on the reconstruction objective and
//! aligned topics, not on the specific decomposition (NMF is non-convex). The fit is
//! deterministic (NNDSVD init, fixed-order reductions).

use crate::nmf::{fit_nmf_on_matrix, BetaLoss, Init, NmfModel, SpMat};

/// The similarity metric between a document embedding and a word embedding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Dot,
}

/// A fitted KeyNMF model: the NMF factorization plus the extracted keywords.
pub struct KeyNmfModel {
    /// The NMF over the keyword-importance matrix (topic_word / doc_topic / raw H,W).
    pub nmf: NmfModel,
    /// Per-document extracted keywords as `(word_id, importance)`, importance
    /// descending — the sparse rows of the factored matrix.
    pub keywords: Vec<Vec<(u32, f64)>>,
}

/// Fit KeyNMF. `doc_words[d]` is the set of vocabulary word-ids present in document
/// `d` (deduplicated). `doc_emb` is (D, E) row-major; `word_emb` is (V, E) row-major,
/// aligned to the vocabulary. `top_n` keeps that many highest-similarity positive
/// words per document. K must be `<= min(D, V)` (an NNDSVD requirement, checked by
/// the binding).
#[allow(clippy::too_many_arguments)]
pub fn fit_keynmf(
    doc_words: &[Vec<u32>],
    doc_emb: &[f64],
    word_emb: &[f64],
    num_docs: usize,
    num_types: usize,
    emb_dim: usize,
    num_topics: usize,
    top_n: usize,
    metric: Metric,
    iters: usize,
    convergence_tol: f64,
    seed: u64,
) -> KeyNmfModel {
    // Precompute L2 norms for cosine (word norms once; doc norm per doc).
    let word_norm: Vec<f64> = (0..num_types)
        .map(|w| {
            let s = &word_emb[w * emb_dim..(w + 1) * emb_dim];
            s.iter().map(|x| x * x).sum::<f64>().sqrt()
        })
        .collect();

    let mut keywords: Vec<Vec<(u32, f64)>> = Vec::with_capacity(num_docs);
    for (d, present) in doc_words.iter().enumerate() {
        let de = &doc_emb[d * emb_dim..(d + 1) * emb_dim];
        let dnorm = de.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Similarity of every present candidate word.
        let mut sims: Vec<(u32, f64)> = Vec::with_capacity(present.len());
        for &w in present {
            let we = &word_emb[w as usize * emb_dim..(w as usize + 1) * emb_dim];
            let dot: f64 = de.iter().zip(we).map(|(a, b)| a * b).sum();
            let sim = match metric {
                Metric::Dot => dot,
                Metric::Cosine => {
                    let denom = dnorm * word_norm[w as usize];
                    if denom > 0.0 {
                        dot / denom
                    } else {
                        0.0
                    }
                }
            };
            sims.push((w, sim));
        }
        // Top-N by similarity (descending), deterministic tie-break by word id,
        // then keep only strictly-positive similarities — paired correctly.
        sims.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        sims.truncate(top_n);
        sims.retain(|&(_, s)| s > 0.0);
        // Store sorted by word id for CSR construction and stable output.
        let mut row = sims;
        row.sort_by_key(|&(w, _)| w);
        keywords.push(row);
    }

    // Build the sparse D x V keyword-importance matrix (CSR).
    let mut indptr = Vec::with_capacity(num_docs + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    indptr.push(0);
    for row in &keywords {
        for &(w, s) in row {
            col_idx.push(w as usize);
            vals.push(s);
        }
        indptr.push(col_idx.len());
    }
    let x = SpMat {
        rows: num_docs,
        cols: num_types,
        indptr,
        col_idx,
        vals,
    };

    let nmf = fit_nmf_on_matrix(
        x,
        num_topics,
        num_types,
        BetaLoss::Frobenius,
        Init::Nndsvd,
        iters,
        convergence_tol,
        seed,
    );

    // Re-sort each document's keywords by descending importance for the public view.
    for row in &mut keywords {
        row.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    }

    KeyNmfModel { nmf, keywords }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a planted corpus: K topics, each with its own word cluster in embedding
    /// space; documents draw words from one topic and their embedding sits at that
    /// topic's center. KeyNMF should recover the topic word-clusters.
    #[allow(clippy::type_complexity)]
    fn planted() -> (
        Vec<Vec<u32>>,
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
        usize,
        Vec<usize>,
    ) {
        let k = 3usize;
        let words_per = 6usize;
        let v = k * words_per;
        let e = 4usize;
        // word embeddings: topic t's words point along axis t (with small offset).
        let mut word_emb = vec![0.0f64; v * e];
        let mut word_topic = vec![0usize; v];
        for t in 0..k {
            for j in 0..words_per {
                let w = t * words_per + j;
                word_topic[w] = t;
                word_emb[w * e + t] = 1.0 + 0.1 * j as f64;
                word_emb[w * e + (j % e)] += 0.05; // slight spread
            }
        }
        // 12 docs per topic; each doc has all its topic's words; doc embedding = axis t.
        let n = k * 12;
        let mut doc_words = Vec::new();
        let mut doc_emb = vec![0.0f64; n * e];
        for i in 0..n {
            let t = i % k;
            doc_emb[i * e + t] = 1.0;
            let words: Vec<u32> = (0..v as u32)
                .filter(|&w| word_topic[w as usize] == t)
                .collect();
            doc_words.push(words);
        }
        (doc_words, doc_emb, word_emb, n, v, e, word_topic)
    }

    #[test]
    fn recovers_planted_word_clusters() {
        let (dw, de, we, n, v, e, word_topic) = planted();
        let m = fit_keynmf(&dw, &de, &we, n, v, e, 3, 10, Metric::Cosine, 200, 1e-5, 13);
        // Each recovered topic's top words should come from a single planted cluster.
        let mut pure = 0;
        for k in 0..3 {
            let row = &m.nmf.topic_word[k];
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let top: Vec<usize> = idx.into_iter().take(4).map(|w| word_topic[w]).collect();
            if top.windows(2).all(|p| p[0] == p[1]) {
                pure += 1;
            }
        }
        assert!(pure >= 2, "only {pure}/3 topics were word-cluster-pure");
        // topic_word rows are a simplex.
        for row in &m.nmf.topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let (dw, de, we, n, v, e, _) = planted();
        let a = fit_keynmf(&dw, &de, &we, n, v, e, 3, 10, Metric::Cosine, 100, 1e-5, 13);
        let b = fit_keynmf(&dw, &de, &we, n, v, e, 3, 10, Metric::Cosine, 100, 1e-5, 13);
        assert_eq!(a.nmf.topic_word, b.nmf.topic_word);
        assert_eq!(a.keywords, b.keywords);
    }

    #[test]
    fn top_n_and_positive_filter() {
        // A doc whose words have mixed-sign similarity: only positive, at most top_n.
        let e = 2usize;
        let v = 4usize;
        // words: 0,1 point +x ; 2,3 point -x
        let word_emb = vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.8, 0.2];
        let doc_emb = vec![1.0, 0.0]; // +x doc
        let dw = vec![vec![0u32, 1, 2, 3]];
        let m = fit_keynmf(
            &dw,
            &doc_emb,
            &word_emb,
            1,
            v,
            e,
            1,
            3,
            Metric::Cosine,
            20,
            1e-5,
            0,
        );
        // top_n=3 but only words 0,1 have positive cosine -> exactly those two.
        let kw: Vec<u32> = m.keywords[0].iter().map(|&(w, _)| w).collect();
        assert_eq!(kw.len(), 2);
        assert!(kw.contains(&0) && kw.contains(&1));
    }

    #[test]
    fn zero_similarity_is_dropped() {
        // A word orthogonal to the doc has sim == 0 exactly; the strict `> 0` filter
        // must drop it (not keep it as a zero-importance keyword).
        let e = 2usize;
        let v = 2usize;
        // word 0 aligns with the doc (+x); word 1 is orthogonal (+y) -> cosine 0.
        let word_emb = vec![1.0, 0.0, 0.0, 1.0];
        let doc_emb = vec![1.0, 0.0];
        let dw = vec![vec![0u32, 1]];
        let m = fit_keynmf(
            &dw,
            &doc_emb,
            &word_emb,
            1,
            v,
            e,
            1,
            5,
            Metric::Cosine,
            10,
            1e-5,
            0,
        );
        let kw: Vec<u32> = m.keywords[0].iter().map(|&(w, _)| w).collect();
        assert_eq!(kw, vec![0], "the orthogonal (sim==0) word must be dropped");
    }
}
