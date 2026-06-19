//! Collapsed Variational Bayes, zeroth-order (CVB0) inference for LDA
//! (Asuncion, Welling, Smyth & Teh, *UAI* 2009, "On Smoothing and Inference for
//! Topic Models").
//!
//! A deterministic alternative to collapsed Gibbs. Instead of a hard topic per
//! token, each *(document, word-type)* cell keeps a soft responsibility vector
//! `γ` over topics, updated from **expected** counts:
//!
//! ```text
//! γ_{d,w,k} ∝ (E[n_dk]^{-dw} + α_k) · (E[n_wk]^{-dw} + β) / (E[n_k]^{-dw} + β̄)
//! ```
//!
//! the same factored conditional as CGS, mean-field instead of sampled. So it is
//! reproducible (no seed dependence beyond the initial γ), has no burn-in, and
//! tends to match or beat CGS on held-out perplexity. The cost is `O(K)` per
//! cell (γ is dense), so CVB0 does **not** have the sparse samplers' large-K
//! speed; it competes on quality and determinism.
//!
//! Memory is kept to `O(#unique (doc,word) cells · K)` by sharing one γ across
//! all tokens of the same word-type in a document (the standard efficient CVB0;
//! those tokens have an identical conditional), rather than one γ per token.

use rand::Rng;

use crate::corpus::Corpus;

/// A CVB0 inference engine over per-(document, word-type) responsibilities.
pub struct Cvb0 {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,

    /// `cells[d]` = the unique `(word_id, count)` pairs in document `d`.
    cells: Vec<Vec<(u32, u32)>>,
    /// `gamma[d][i]` = the length-`K` responsibility vector for cell `i` of doc `d`.
    gamma: Vec<Vec<Vec<f64>>>,

    /// Expected document-topic counts `E[n_dk]`.
    n_dk: Vec<Vec<f64>>,
    /// Expected word-topic counts `E[n_wk]`.
    n_wk: Vec<Vec<f64>>,
    /// Expected topic counts `E[n_k]`.
    n_k: Vec<f64>,
    /// Document lengths (token counts), for θ.
    doc_len: Vec<f64>,

    /// Per-document prior `α_{d,k}` (DMR). When `Some`, replaces the uniform α.
    doc_alpha: Option<Vec<Vec<f64>>>,
    /// Asymmetric β for SeededLDA (`β_{k,w}` and per-topic `β_sum_k`).
    seed: Option<CvbSeedBeta>,
    /// Per-document allowed-topic sets (LabeledLDA). When `Some`, a document's
    /// responsibilities are confined to `allowed[d]` (an empty set = all topics).
    /// Masking is free in CVB0: γ is simply zero off the allowed set.
    allowed: Option<Vec<Vec<usize>>>,
}

/// SeededLDA's asymmetric-β bookkeeping for CVB0 (mirrors the WarpLDA one).
struct CvbSeedBeta {
    seed_weight: f64,
    beta_sum_k: Vec<f64>,
    inv_seeds: Vec<Vec<u32>>,
}

impl Cvb0 {
    /// Initialize from `corpus` with a random γ per cell (the only stochastic
    /// step; everything after is deterministic).
    pub fn new<R: Rng>(
        corpus: &Corpus,
        num_topics: usize,
        alpha: &[f64],
        beta: f64,
        rng: &mut R,
    ) -> Cvb0 {
        let num_types = corpus.num_types();
        let beta_sum = beta * num_types as f64;
        let alpha_sum: f64 = alpha.iter().sum();
        let k = num_topics;

        let mut n_dk = vec![vec![0.0f64; k]; corpus.docs.len()];
        let mut n_wk = vec![vec![0.0f64; k]; num_types];
        let mut n_k = vec![0.0f64; k];
        let mut cells: Vec<Vec<(u32, u32)>> = Vec::with_capacity(corpus.docs.len());
        let mut gamma: Vec<Vec<Vec<f64>>> = Vec::with_capacity(corpus.docs.len());
        let mut doc_len = vec![0.0f64; corpus.docs.len()];

        for (d, doc) in corpus.docs.iter().enumerate() {
            // Collapse the document to unique (word, count) cells.
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &w in doc {
                *counts.entry(w).or_insert(0) += 1;
            }
            doc_len[d] = doc.len() as f64;
            let mut dcells: Vec<(u32, u32)> = counts.into_iter().collect();
            dcells.sort_unstable(); // deterministic cell order
            let mut dgamma: Vec<Vec<f64>> = Vec::with_capacity(dcells.len());

            for &(w, c) in &dcells {
                // Random initial responsibilities, normalized.
                let mut g = vec![0.0f64; k];
                let mut s = 0.0;
                for x in g.iter_mut() {
                    *x = rng.gen::<f64>() + 1e-6;
                    s += *x;
                }
                let cf = c as f64;
                for (t, x) in g.iter_mut().enumerate() {
                    *x /= s;
                    n_dk[d][t] += cf * *x;
                    n_wk[w as usize][t] += cf * *x;
                    n_k[t] += cf * *x;
                }
                dgamma.push(g);
            }
            cells.push(dcells);
            gamma.push(dgamma);
        }

        Cvb0 {
            num_topics: k,
            num_types,
            alpha: alpha.to_vec(),
            alpha_sum,
            beta,
            beta_sum,
            cells,
            gamma,
            n_dk,
            n_wk,
            n_k,
            doc_len,
            doc_alpha: None,
            seed: None,
            allowed: None,
        }
    }

    /// Recompute all expected counts from the current γ (used after a structural
    /// change such as applying a topic mask).
    fn rebuild_counts(&mut self) {
        let k = self.num_topics;
        for row in self.n_dk.iter_mut() {
            for x in row.iter_mut() {
                *x = 0.0;
            }
        }
        for row in self.n_wk.iter_mut() {
            for x in row.iter_mut() {
                *x = 0.0;
            }
        }
        for x in self.n_k.iter_mut() {
            *x = 0.0;
        }
        for d in 0..self.cells.len() {
            for (i, &(w, c)) in self.cells[d].iter().enumerate() {
                let (w, cf) = (w as usize, c as f64);
                for t in 0..k {
                    let g = self.gamma[d][i][t];
                    self.n_dk[d][t] += cf * g;
                    self.n_wk[w][t] += cf * g;
                    self.n_k[t] += cf * g;
                }
            }
        }
    }

    /// Confine each document to its allowed topics (LabeledLDA). An empty
    /// `allowed[d]` leaves that document unconstrained. Zeroes γ off the allowed
    /// set, renormalizes, and rebuilds the expected counts.
    pub fn set_allowed(&mut self, allowed: Vec<Vec<usize>>) {
        let k = self.num_topics;
        for d in 0..self.cells.len() {
            if allowed[d].is_empty() {
                continue;
            }
            for i in 0..self.cells[d].len() {
                let g = &mut self.gamma[d][i];
                for t in 0..k {
                    if allowed[d].binary_search(&t).is_err() {
                        g[t] = 0.0;
                    }
                }
                let sum: f64 = g.iter().sum();
                if sum > 0.0 {
                    for x in g.iter_mut() {
                        *x /= sum;
                    }
                } else {
                    let u = 1.0 / allowed[d].len() as f64;
                    for &t in &allowed[d] {
                        g[t] = u;
                    }
                }
            }
        }
        self.allowed = Some(allowed);
        self.rebuild_counts();
    }

    /// Use a per-document prior `α_{d,k}` (DMR); called each outer iteration
    /// after recomputing α from the current λ.
    pub fn set_doc_alpha(&mut self, doc_alpha: Vec<Vec<f64>>) {
        self.doc_alpha = Some(doc_alpha);
    }

    /// Enable SeededLDA's asymmetric β. `seeds[k]` are the seed word-ids for
    /// topic `k`, each gaining `seed_weight` in topic `k`.
    pub fn set_seeds(&mut self, seeds: &[Vec<usize>], seed_weight: f64) {
        let k = self.num_topics;
        let v = self.num_types;
        let mut beta_sum_k = vec![self.beta * v as f64; k];
        let mut inv_seeds: Vec<Vec<u32>> = vec![Vec::new(); v];
        for (t, ws) in seeds.iter().enumerate().take(k) {
            beta_sum_k[t] += ws.len() as f64 * seed_weight;
            for &w in ws {
                if w < v {
                    inv_seeds[w].push(t as u32);
                }
            }
        }
        for row in inv_seeds.iter_mut() {
            row.sort_unstable();
        }
        self.seed = Some(CvbSeedBeta { seed_weight, beta_sum_k, inv_seeds });
    }

    /// Expected document-topic counts `E[n_dk]` — the soft input the DMR λ
    /// optimizer consumes directly (it already takes `&[Vec<f64>]`).
    pub fn doc_topic_expected(&self) -> &[Vec<f64>] {
        &self.n_dk
    }

    /// One deterministic CVB0 sweep over every cell. Returns the mean absolute
    /// change in γ across all cells, a convergence signal (→ 0 as the
    /// responsibilities settle), so the caller can early-stop.
    pub fn sweep(&mut self) -> f64 {
        let k = self.num_topics;
        let beta = self.beta;
        let beta_sum = self.beta_sum;
        let mut total_change = 0.0f64;
        let mut n_cells = 0usize;
        let mut old = vec![0.0f64; k];
        let all_topics: Vec<usize> = (0..k).collect();

        for d in 0..self.cells.len() {
            // The topic support for this document: its allowed set (LabeledLDA)
            // or all topics. γ stays 0 off the support, so the loops touch only
            // the supported topics.
            let topics: &[usize] = match &self.allowed {
                Some(al) if !al[d].is_empty() => al[d].as_slice(),
                _ => all_topics.as_slice(),
            };
            let uniform = 1.0 / topics.len() as f64;

            for i in 0..self.cells[d].len() {
                let (w, c) = self.cells[d][i];
                let (w, cf) = (w as usize, c as f64);

                // Snapshot the old γ and remove this cell's mass (`-dw` counts).
                for &t in topics {
                    let g = self.gamma[d][i][t];
                    old[t] = g;
                    self.n_dk[d][t] -= cf * g;
                    self.n_wk[w][t] -= cf * g;
                    self.n_k[t] -= cf * g;
                }

                // Recompute γ ∝ (n_dk+α_{d,k})(n_wk+β_{k,w})/(n_k+β_sum_k), with
                // the per-document α (DMR) and asymmetric β (SeededLDA) folded in.
                let mut sum = 0.0f64;
                for &t in topics {
                    let a = match &self.doc_alpha {
                        Some(da) => da[d][t],
                        None => self.alpha[t],
                    };
                    let (bt, bsum) = match &self.seed {
                        Some(s) => (
                            if s.inv_seeds[w].binary_search(&(t as u32)).is_ok() {
                                beta + s.seed_weight
                            } else {
                                beta
                            },
                            s.beta_sum_k[t],
                        ),
                        None => (beta, beta_sum),
                    };
                    let val = (self.n_dk[d][t] + a) * (self.n_wk[w][t] + bt) / (self.n_k[t] + bsum);
                    let g = if val > 0.0 { val } else { 0.0 };
                    self.gamma[d][i][t] = g;
                    sum += g;
                }

                // Normalize, accumulate |Δγ|, and add the new mass back.
                for &t in topics {
                    let new = if sum > 0.0 { self.gamma[d][i][t] / sum } else { uniform };
                    self.gamma[d][i][t] = new;
                    total_change += (new - old[t]).abs();
                    self.n_dk[d][t] += cf * new;
                    self.n_wk[w][t] += cf * new;
                    self.n_k[t] += cf * new;
                }
                n_cells += 1;
            }
        }
        if n_cells == 0 {
            0.0
        } else {
            total_change / n_cells as f64
        }
    }

    /// Smoothed topic-word φ `(E[n_wk]+β)/(E[n_k]+β̄)`, accumulated into
    /// `acc[word][topic]` (matches the MH samplers' orientation).
    pub fn phi_into(&self, acc: &mut [Vec<f64>]) {
        let beta = self.beta;
        for w in 0..self.num_types {
            for t in 0..self.num_topics {
                let (bt, bsum) = match &self.seed {
                    Some(s) => (
                        if s.inv_seeds[w].binary_search(&(t as u32)).is_ok() {
                            beta + s.seed_weight
                        } else {
                            beta
                        },
                        s.beta_sum_k[t],
                    ),
                    None => (beta, self.beta_sum),
                };
                acc[w][t] += (self.n_wk[w][t] + bt) / (self.n_k[t] + bsum);
            }
        }
    }

    /// Smoothed topic-word matrix, shape `[topic][word]` (seeded-β aware) — the
    /// topic-major orientation the SeededLDA output expects.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let beta = self.beta;
        let mut phi = vec![vec![0.0f64; self.num_types]; self.num_topics];
        for w in 0..self.num_types {
            for t in 0..self.num_topics {
                let (bt, bsum) = match &self.seed {
                    Some(s) => (
                        if s.inv_seeds[w].binary_search(&(t as u32)).is_ok() {
                            beta + s.seed_weight
                        } else {
                            beta
                        },
                        s.beta_sum_k[t],
                    ),
                    None => (beta, self.beta_sum),
                };
                phi[t][w] = (self.n_wk[w][t] + bt) / (self.n_k[t] + bsum);
            }
        }
        phi
    }

    /// Smoothed doc-topic θ matrix, shape `[doc][topic]` (per-document α).
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let mut th = vec![vec![0.0f64; self.num_topics]; self.cells.len()];
        self.theta_into(&mut th);
        th
    }

    /// Smoothed doc-topic θ `(E[n_dk]+α_{d,k})/(n_d+α_sum_d)`, into
    /// `acc[doc][topic]` (per-document α for DMR).
    pub fn theta_into(&self, acc: &mut [Vec<f64>]) {
        let all_topics: Vec<usize> = (0..self.num_topics).collect();
        for d in 0..self.cells.len() {
            let a_row: &[f64] = match &self.doc_alpha {
                Some(da) => &da[d],
                None => &self.alpha,
            };
            // Confine θ to the allowed topics (LabeledLDA): off the set it is 0,
            // and the normalizer sums α over the allowed topics only.
            let topics: &[usize] = match &self.allowed {
                Some(al) if !al[d].is_empty() => al[d].as_slice(),
                _ => all_topics.as_slice(),
            };
            let a_sum: f64 = topics.iter().map(|&t| a_row[t]).sum();
            let denom = self.doc_len[d] + a_sum;
            for &t in topics {
                acc[d][t] += (self.n_dk[d][t] + a_row[t]) / denom;
            }
        }
    }

    /// The MAP hard assignment (argmax γ per cell): `doc_topics[d][i]` is the
    /// most-probable topic for token `i` of document `d`. This is the only piece
    /// of the old `to_topic_model` that needs the private `cells`/`gamma`; the
    /// `topica` crate's `Cvb0ToModel` extension trait packs these into a
    /// `TopicModel` (which lives in `topica`, not the core), so the rest of the
    /// codebase — coherence, save/load, held-out inference — is reused unchanged.
    pub fn map_topic_assignments(&self, corpus: &Corpus) -> Vec<Vec<u32>> {
        corpus
            .docs
            .iter()
            .enumerate()
            .map(|(d, doc)| {
                // Map word -> argmax topic for this doc's cells.
                let mut best: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for (i, &(w, _)) in self.cells[d].iter().enumerate() {
                    let g = &self.gamma[d][i];
                    let mut bt = 0usize;
                    for t in 1..self.num_topics {
                        if g[t] > g[bt] {
                            bt = t;
                        }
                    }
                    best.insert(w, bt as u32);
                }
                doc.iter().map(|&w| best[&w]).collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn planted(n_blocks: usize, wpb: usize, n_docs: usize) -> (Corpus, usize) {
        let v = n_blocks * wpb;
        let docs: Vec<Vec<u32>> = (0..n_docs)
            .map(|d| {
                let b = d % n_blocks;
                let block: Vec<u32> = (b * wpb..(b + 1) * wpb).map(|w| w as u32).collect();
                let mut doc = block.clone();
                doc.extend(block);
                doc
            })
            .collect();
        let corpus = Corpus {
            id_to_word: (0..v).map(|i| format!("w{i}")).collect(),
            doc_names: (0..n_docs).map(|i| format!("d{i}")).collect(),
            doc_labels: vec![String::new(); n_docs],
            doc_freqs: vec![0u32; v],
            total_freqs: vec![0u32; v],
            docs,
        };
        (corpus, n_blocks)
    }

    #[test]
    fn recovers_planted_blocks() {
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 200);
        let k = n_blocks;
        let v = corpus.num_types();
        let alpha = vec![0.1f64; k];
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut m = Cvb0::new(&corpus, k, &alpha, 0.01, &mut rng);
        for _ in 0..100 {
            m.sweep();
        }
        let mut phi = vec![vec![0.0f64; k]; v]; // [word][topic]
        m.phi_into(&mut phi);
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[b][t].partial_cmp(&phi[a][t]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..n_blocks {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), n_blocks, "only recovered {covered:?}");
    }

    #[test]
    fn masked_path_respects_labels() {
        // LabeledLDA-style: each doc may only use its own block's topic. After
        // fitting, θ must put zero mass on disallowed topics, and each block's
        // words must top its own topic.
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 200);
        let k = n_blocks;
        let alpha = vec![0.1f64; k];
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut m = Cvb0::new(&corpus, k, &alpha, 0.01, &mut rng);
        // doc d (block d % n_blocks) is allowed only topic (d % n_blocks).
        let allowed: Vec<Vec<usize>> =
            (0..corpus.docs.len()).map(|d| vec![d % n_blocks]).collect();
        m.set_allowed(allowed.clone());
        for _ in 0..60 {
            m.sweep();
        }
        let mut th = vec![vec![0.0f64; k]; corpus.docs.len()];
        m.theta_into(&mut th);
        for d in 0..corpus.docs.len() {
            let b = d % n_blocks;
            for t in 0..k {
                if t != b {
                    assert!(th[d][t].abs() < 1e-12, "doc {d} leaked mass to topic {t}");
                }
            }
            assert!(th[d][b] > 0.5, "doc {d} not concentrated on its label");
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let (corpus, _) = planted(3, 4, 90);
        let alpha = vec![0.1f64; 3];
        let run = || {
            let mut rng = Pcg64Mcg::seed_from_u64(7);
            let mut m = Cvb0::new(&corpus, 3, &alpha, 0.01, &mut rng);
            for _ in 0..40 {
                m.sweep();
            }
            let mut th = vec![vec![0.0f64; 3]; corpus.docs.len()];
            m.theta_into(&mut th);
            th
        };
        let a = run();
        let b = run();
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }
}
