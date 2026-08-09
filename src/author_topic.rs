//! AuthorTopic: the Author-Topic Model (Rosen-Zvi, Griffiths, Steyvers & Smyth,
//! "The Author-Topic Model for Authors and Documents," UAI 2004).
//!
//! ATM extends LDA so that topics are conditioned on *authors* rather than
//! documents. Each author `a` has a topic distribution `θ_a ~ Dir(α)`; each topic
//! `t` has a word distribution `φ_t ~ Dir(β)`. For every token in a document with
//! author set `A_d`, an author `x` is drawn uniformly from `A_d`, a topic `z` from
//! `θ_x`, and a word from `φ_z`. LDA is the special case where every document has
//! one unique author.
//!
//! Inference is collapsed Gibbs from the paper (§3): integrate out θ and φ, and
//! resample the per-token PAIR `(author, topic)` jointly from
//!
//!   P(x=a, z=t | ·) ∝ (C^{WT}_{w,t} + β)/(n_{·t} + Vβ)
//!                   · (C^{AT}_{a,t} + α_t)/(n_{a·} + Σ_j α_j)
//!
//! over `a ∈ A_d`, `t ∈ 0..K`, with the current token decremented from every count
//! table first. The uniform `1/|A_d|` author-prior factor is identical for every
//! candidate `(a,t)` and cancels in the normalization, so it is omitted.
//!
//! The word-topic side is byte-for-byte the LDA machinery — this module reuses
//! `crate::model::TopicModel` for `C^{WT}` (packed word→topic counts, per-topic
//! totals, and the per-token topic assignments) and adds an author×topic count
//! table, per-author totals, and a per-token author assignment.
//!
//! Determinism: single-threaded; every draw comes from the seeded `rng`, documents
//! are processed in corpus order, so a fixed seed reproduces bit-for-bit. Outputs
//! are `Vec<Vec<f64>>` (no `ndarray`) so a default build carries no `embeddings`
//! dependency; the binding converts with `vecs_to_arr2`.

use crate::estimator::{Estimator, ModelFamily};
use crate::model::TopicModel;
use rand::Rng;

/// Fitted Author-Topic state read back by the PyO3 binding.
pub struct AuthorTopicModel {
    pub num_topics: usize,
    pub num_authors: usize,
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    /// Topic-word point estimate φ (K×V).
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic point estimate θ_d (D×K): the *empirical posterior*, i.e. the
    /// proportions of each document's sampled token→topic assignments (content
    /// based, exactly as LDA reports it). Each row sums to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// Author-topic point estimate θ_a (A×K): (C^{AT}_{a,t} + α_t)/(n_{a·} + Σα).
    /// The model-defining output. Each row sums to 1.
    pub author_topic: Vec<Vec<f64>>,
    /// Held-in log-likelihood trace `(iter, ll)` recorded every stride — a
    /// convergence diagnostic, not an early-stop criterion.
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// Fit ATM by collapsed Gibbs.
///
/// - `docs`: word-id documents (`Vec<u32>` per doc).
/// - `num_types`: vocabulary size V.
/// - `doc_authors`: per-document author-id sets, parallel to `docs`, each deduped
///   and non-empty (the binding validates this and builds the id map).
/// - `num_authors`: A (author-id space; ids in `doc_authors` must be `< num_authors`).
/// - `alpha`: length-K author-topic Dirichlet (per-topic; may be asymmetric).
/// - `beta`: symmetric topic-word Dirichlet scalar.
/// - `iters`: Gibbs sweeps.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    doc_authors: &[Vec<u32>],
    num_authors: usize,
    num_topics: usize,
    alpha: Vec<f64>,
    beta: f64,
    iters: usize,
    rng: &mut R,
) -> AuthorTopicModel {
    let k = num_topics;
    let v = num_types;
    let alpha_sum: f64 = alpha.iter().sum();
    let beta_sum = beta * v as f64;

    // Word-topic side: reuse TopicModel's packed count tables + per-token topics.
    let mut wt = TopicModel::new(k, alpha_sum, beta, v);
    // Size each word's packed count row (mirrors labeled.rs init).
    let mut type_totals = vec![0usize; v];
    for doc in docs {
        for &w in doc {
            type_totals[w as usize] += 1;
        }
    }
    wt.type_topic_counts = type_totals.iter().map(|&n| vec![0u32; k.min(n)]).collect();
    wt.tokens_per_topic = vec![0u32; k];
    wt.doc_topics = docs.iter().map(|d| vec![0u32; d.len()]).collect();

    // Author-topic side (author-major, indexed [a][t]).
    let mut author_topic_counts = vec![vec![0u32; k]; num_authors];
    let mut author_totals = vec![0u32; num_authors];
    // Per-token author assignment, parallel to wt.doc_topics.
    let mut token_authors: Vec<Vec<u32>> = docs.iter().map(|d| vec![0u32; d.len()]).collect();

    // --- initialize: uniform author from A_d, uniform topic ---
    for (d, doc) in docs.iter().enumerate() {
        let authors = &doc_authors[d];
        for (pos, &w) in doc.iter().enumerate() {
            let a = authors[rng.gen_range(0..authors.len())];
            let t = rng.gen_range(0..k);
            wt.doc_topics[d][pos] = t as u32;
            token_authors[d][pos] = a;
            wt.tokens_per_topic[t] += 1;
            wt.increment_type_topic(w as usize, t);
            author_topic_counts[a as usize][t] += 1;
            author_totals[a as usize] += 1;
        }
    }

    // --- Gibbs sweeps ---
    let mut wt_term = vec![0.0f64; k]; // (C^{WT}_{w,t}+β)/(n_{·t}+Vβ), per token
    let mut scores = Vec::with_capacity(num_authors.max(1) * k);
    let eval_stride = (iters / 50).max(1);
    let mut fit_history = Vec::new();

    for it in 0..iters {
        for (d, doc) in docs.iter().enumerate() {
            let authors = &doc_authors[d];
            for pos in 0..doc.len() {
                let w = doc[pos] as usize;
                let t_old = wt.doc_topics[d][pos] as usize;
                let a_old = token_authors[d][pos] as usize;

                // decrement current token from every table
                wt.decrement_type_topic(w, t_old);
                wt.tokens_per_topic[t_old] -= 1;
                author_topic_counts[a_old][t_old] -= 1;
                author_totals[a_old] -= 1;

                // word-topic term for this word, all topics
                for t in 0..k {
                    let n_wt = wt.get_type_topic_count(w, t) as f64;
                    wt_term[t] = (n_wt + beta) / (wt.tokens_per_topic[t] as f64 + beta_sum);
                }

                // score every (author, topic) candidate; author-outer, topic-inner
                scores.clear();
                let mut total = 0.0f64;
                for &a in authors {
                    let a = a as usize;
                    let a_denom = author_totals[a] as f64 + alpha_sum;
                    let row = &author_topic_counts[a];
                    for t in 0..k {
                        let s = wt_term[t] * (row[t] as f64 + alpha[t]) / a_denom;
                        scores.push(s);
                        total += s;
                    }
                }

                // sample a flat index into the |A_d|×K grid
                let mut r = rng.gen::<f64>() * total;
                let mut choice = scores.len() - 1;
                for (i, &s) in scores.iter().enumerate() {
                    r -= s;
                    if r <= 0.0 {
                        choice = i;
                        break;
                    }
                }
                let a_new = authors[choice / k] as usize;
                let t_new = choice % k;

                // re-increment
                wt.increment_type_topic(w, t_new);
                wt.tokens_per_topic[t_new] += 1;
                author_topic_counts[a_new][t_new] += 1;
                author_totals[a_new] += 1;
                wt.doc_topics[d][pos] = t_new as u32;
                token_authors[d][pos] = a_new as u32;
            }
        }

        if it % eval_stride == 0 || it == iters - 1 {
            let ll = held_in_loglik(
                docs,
                doc_authors,
                &wt,
                &author_topic_counts,
                &author_totals,
                &alpha,
                alpha_sum,
                beta,
                beta_sum,
                k,
            );
            fit_history.push((it + 1, ll));
        }
    }

    // --- materialize point estimates ---
    let topic_word = wt.topic_word();

    let doc_topic: Vec<Vec<f64>> = wt
        .doc_topics
        .iter()
        .map(|topics| {
            if topics.is_empty() {
                return vec![1.0 / k as f64; k];
            }
            let mut cnt = vec![0.0f64; k];
            for &t in topics {
                cnt[t as usize] += 1.0;
            }
            let n = topics.len() as f64;
            cnt.iter().map(|&c| c / n).collect()
        })
        .collect();

    let author_topic: Vec<Vec<f64>> = (0..num_authors)
        .map(|a| {
            let denom = author_totals[a] as f64 + alpha_sum;
            (0..k)
                .map(|t| (author_topic_counts[a][t] as f64 + alpha[t]) / denom)
                .collect()
        })
        .collect();

    AuthorTopicModel {
        num_topics: k,
        num_authors,
        alpha,
        alpha_sum,
        beta,
        topic_word,
        doc_topic,
        author_topic,
        fit_history,
        converged: false,
    }
}

/// Held-in log-likelihood of the corpus under current point estimates:
/// Σ_i log[ (1/|A_d|) Σ_{a∈A_d} Σ_t θ_a[t] φ_t[w_i] ]. A convergence diagnostic.
#[allow(clippy::too_many_arguments)]
fn held_in_loglik(
    docs: &[Vec<u32>],
    doc_authors: &[Vec<u32>],
    wt: &TopicModel,
    atc: &[Vec<u32>],
    atot: &[u32],
    alpha: &[f64],
    alpha_sum: f64,
    beta: f64,
    beta_sum: f64,
    k: usize,
) -> f64 {
    // phi[t][w] on demand is expensive; precompute per-topic denom, use counts.
    let phi_denom: Vec<f64> = (0..k)
        .map(|t| wt.tokens_per_topic[t] as f64 + beta_sum)
        .collect();
    let mut ll = 0.0f64;
    for (d, doc) in docs.iter().enumerate() {
        let authors = &doc_authors[d];
        let inv_a = 1.0 / authors.len() as f64;
        // theta[a][t] for the doc's authors
        let thetas: Vec<Vec<f64>> = authors
            .iter()
            .map(|&a| {
                let a = a as usize;
                let denom = atot[a] as f64 + alpha_sum;
                (0..k)
                    .map(|t| (atc[a][t] as f64 + alpha[t]) / denom)
                    .collect()
            })
            .collect();
        for &w in doc {
            let w = w as usize;
            let mut p = 0.0f64;
            for t in 0..k {
                let phi = (wt.get_type_topic_count(w, t) as f64 + beta) / phi_denom[t];
                let mut theta_sum = 0.0f64;
                for th in &thetas {
                    theta_sum += th[t];
                }
                p += phi * theta_sum;
            }
            p *= inv_a;
            if p > 0.0 {
                ll += p.ln();
            }
        }
    }
    ll
}

impl Estimator for AuthorTopicModel {
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
        // doc_topic is a per-document empirical topic simplex, like LDA.
        ModelFamily::Dirichlet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Three disjoint word blocks, three authors, each author writes only its
    /// block. Single author per document (so each doc has a clear planted topic).
    fn planted() -> (Vec<Vec<u32>>, usize, Vec<Vec<u32>>, usize) {
        let block = 5usize;
        let k = 3usize;
        let v = k * block;
        let mut docs = Vec::new();
        let mut authors = Vec::new();
        for a in 0..k {
            for _ in 0..20 {
                let doc: Vec<u32> = (0..12).map(|i| (a * block + (i % block)) as u32).collect();
                docs.push(doc);
                authors.push(vec![a as u32]);
            }
        }
        (docs, v, authors, k)
    }

    #[test]
    fn author_topic_recovers_planted_topics() {
        let (docs, v, authors, k) = planted();
        let alpha = vec![50.0 / k as f64; k];
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit(&docs, v, &authors, k, k, alpha, 0.01, 300, &mut rng);
        // Each author's dominant topic is distinct (authors map 1:1 to topics).
        let dom: Vec<usize> = m
            .author_topic
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();
        let distinct: std::collections::HashSet<usize> = dom.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            k,
            "authors should occupy distinct topics: {dom:?}"
        );
        // Each recovered topic is concentrated on one block (peak >> uniform).
        for row in &m.topic_word {
            let peak = row.iter().cloned().fold(0.0f64, f64::max);
            assert!(peak > 3.0 / v as f64, "topic not concentrated: peak={peak}");
        }
    }

    #[test]
    fn author_topic_is_deterministic() {
        let (docs, v, authors, k) = planted();
        let alpha = vec![50.0 / k as f64; k];
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let a = fit(&docs, v, &authors, k, k, alpha.clone(), 0.01, 80, &mut r1);
        let b = fit(&docs, v, &authors, k, k, alpha, 0.01, 80, &mut r2);
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.author_topic, b.author_topic);
        assert_eq!(a.doc_topic, b.doc_topic);
    }

    #[test]
    fn author_topic_conforms() {
        let (docs, v, authors, k) = planted();
        let alpha = vec![50.0 / k as f64; k];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit(&docs, v, &authors, k, k, alpha, 0.01, 20, &mut rng);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    /// Gemini Gate-A nit #3: A far larger than K, and K far larger than A, must
    /// not panic (the author×topic table is [a][t]).
    #[test]
    fn author_topic_handles_lopsided_a_and_k() {
        // A=2 authors, K=20 topics.
        let docs: Vec<Vec<u32>> = (0..10).map(|_| (0..8).map(|i| i % 6).collect()).collect();
        let authors: Vec<Vec<u32>> = (0..10).map(|d| vec![(d % 2) as u32]).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let m = fit(&docs, 6, &authors, 2, 20, vec![2.5; 20], 0.01, 15, &mut rng);
        assert_eq!(m.author_topic.len(), 2);
        assert_eq!(m.author_topic[0].len(), 20);
        // K=2 topics, A=8 authors (co-authored docs).
        let docs2: Vec<Vec<u32>> = (0..12).map(|_| (0..8).map(|i| i % 6).collect()).collect();
        let authors2: Vec<Vec<u32>> = (0..12)
            .map(|d| vec![(d % 8) as u32, ((d + 3) % 8) as u32])
            .collect();
        let mut rng2 = ChaCha8Rng::seed_from_u64(2);
        let m2 = fit(
            &docs2,
            6,
            &authors2,
            8,
            2,
            vec![25.0; 2],
            0.01,
            15,
            &mut rng2,
        );
        assert_eq!(m2.author_topic.len(), 8);
        assert_eq!(m2.author_topic[0].len(), 2);
    }

    /// LDA degeneracy (Gemini Gate-A blocker #2): with one unique author per
    /// document, ATM's author term equals LDA's document term, so ATM recovers the
    /// same planted structure. Here `doc_topic` (empirical) and `author_topic`
    /// coincide because each author owns exactly one document.
    #[test]
    fn lda_degeneracy_unique_author_per_doc() {
        let (docs, v, _sa, k) = planted();
        // one unique author per document
        let authors: Vec<Vec<u32>> = (0..docs.len()).map(|d| vec![d as u32]).collect();
        let alpha = vec![50.0 / k as f64; k];
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let m = fit(
            &docs,
            v,
            &authors,
            docs.len(),
            k,
            alpha,
            0.01,
            200,
            &mut rng,
        );
        // Each document's empirical dominant topic and its (sole) author's dominant
        // topic agree. (Cosine equality does NOT hold under heavy α smoothing:
        // author_topic is α-smoothed while doc_topic is raw empirical, and with
        // α_sum=50 ≫ ~12 tokens/author the author rows are pulled toward uniform.
        // The argmax is smoothing-invariant and is the real degeneracy signal.)
        let argmax = |row: &[f64]| {
            row.iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0
        };
        for d in 0..docs.len() {
            assert_eq!(
                argmax(&m.doc_topic[d]),
                argmax(&m.author_topic[d]),
                "doc {d} empirical vs author dominant topic disagree"
            );
        }
    }
}
