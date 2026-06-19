//! LightLDA: an O(1)-per-token Metropolis-Hastings sampler for LDA
//! (Yuan, Gao, Ho, Dai, Wei, Zheng, Xing, Liu & Ma, WWW 2015).
//!
//! Where SparseLDA (see [`crate::sampler`]) costs O(K_d + K_w) per token, the
//! LightLDA sampler draws from cheap **proposal** distributions via Walker alias
//! tables and corrects the bias with a Metropolis-Hastings accept/reject step,
//! giving O(1) amortized work per token. This pays off when the topic count K is
//! large, where the SparseLDA buckets are no longer sparse.
//!
//! We implement the paper's *cycle proposal*, alternating two proposals:
//!
//! * **Word-proposal** `p_w(k) ∝ (n_kw + β) / (n_k + β̄)` — a per-word alias table,
//!   built lazily once per sweep and reused (stale) across all of the word's
//!   tokens, so its O(K) build cost amortizes away.
//! * **Doc-proposal** `p_d(k) ∝ n_kd + α_k` — sampled in O(1) without a table by
//!   treating the document's own topic assignments as an implicit alias (with
//!   probability ∝ doc length pick a random token's topic; otherwise draw from
//!   the α alias).
//!
//! The MH acceptance ratios (Eq. 8 and Eq. 11 of the paper) are computed exactly
//! against the live token-excluded (`-di`) counts, so the chain's stationary
//! distribution is the true LDA collapsed posterior regardless of how stale the
//! proposal tables are — staleness only affects mixing speed, not correctness.
//!
//! The sampler runs on dense count tables for O(1) random access; once training
//! finishes the state is packed back into the shared [`TopicModel`] encoding so
//! all the downstream machinery (φ/θ, `save`/`load`, log-likelihood, held-out
//! `transform`) is reused unchanged.

use rand::Rng;

use crate::corpus::Corpus;
use crate::model::TopicModel;

/// Walker's alias method: turns sampling from a fixed K-way categorical into two
/// O(1) draws (a uniform index plus a coin). Built in O(K). Shared with the
/// WarpLDA sampler ([`crate::warplda`]) for its α-proposal table.
pub(crate) struct Alias {
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl Alias {
    /// Build an alias table proportional to non-negative `weights`. A degenerate
    /// (all-zero) input falls back to a uniform table.
    pub(crate) fn build(weights: &[f64]) -> Alias {
        let n = weights.len();
        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0u32; n];

        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            for i in 0..n {
                prob[i] = 1.0;
                alias[i] = i as u32;
            }
            return Alias { prob, alias };
        }

        // Scale so the average bucket mass is 1.
        let scale = n as f64 / sum;
        let mut scaled: Vec<f64> = weights.iter().map(|&w| w * scale).collect();

        let mut small: Vec<usize> = Vec::new();
        let mut large: Vec<usize> = Vec::new();
        for i in 0..n {
            if scaled[i] < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while let (Some(s), Some(&l)) = (small.pop(), large.last()) {
            prob[s] = scaled[s];
            alias[s] = l as u32;
            // Move the borrowed mass off the large bucket.
            scaled[l] -= 1.0 - scaled[s];
            if scaled[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Anything left over is (numerically) a full bucket.
        for &l in &large {
            prob[l] = 1.0;
        }
        for &s in &small {
            prob[s] = 1.0;
        }
        Alias { prob, alias }
    }

    #[inline]
    pub(crate) fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        let i = rng.gen_range(0..self.prob.len());
        if rng.gen::<f64>() < self.prob[i] {
            i
        } else {
            self.alias[i] as usize
        }
    }
}

/// A LightLDA sampler over dense count tables.
pub struct LightLda {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,
    /// Number of MH proposals per token (alternating word/doc proposals).
    pub mh_steps: usize,

    /// Word-topic counts, `n_wk[word][topic]` (dense).
    n_wk: Vec<Vec<i32>>,
    /// Total tokens per topic, `n_k[topic]`.
    n_k: Vec<i64>,
    /// Per-document topic assignments, parallel to `corpus.docs`.
    z: Vec<Vec<u32>>,

    /// Alias table over the (fixed-until-optimized) α prior, for the doc-proposal.
    alpha_alias: Alias,
}

impl LightLda {
    /// Random-initialise the sampler over `corpus` (same scheme as
    /// [`TopicModel::initialize`]).
    pub fn new<R: Rng>(
        corpus: &Corpus,
        num_topics: usize,
        alpha: &[f64],
        beta: f64,
        rng: &mut R,
    ) -> LightLda {
        let num_types = corpus.num_types();
        let beta_sum = beta * num_types as f64;
        let alpha_sum: f64 = alpha.iter().sum();

        let mut n_wk = vec![vec![0i32; num_topics]; num_types];
        let mut n_k = vec![0i64; num_topics];
        let mut z: Vec<Vec<u32>> = Vec::with_capacity(corpus.docs.len());

        for doc in &corpus.docs {
            let mut zd = Vec::with_capacity(doc.len());
            for &w in doc {
                let t = rng.gen_range(0..num_topics);
                n_wk[w as usize][t] += 1;
                n_k[t] += 1;
                zd.push(t as u32);
            }
            z.push(zd);
        }

        let alpha_alias = Alias::build(alpha);
        LightLda {
            num_topics,
            num_types,
            alpha: alpha.to_vec(),
            alpha_sum,
            beta,
            beta_sum,
            mh_steps: 2,
            n_wk,
            n_k,
            z,
            alpha_alias,
        }
    }

    /// Replace the hyperparameters (after an optimisation step) and rebuild the
    /// α alias.
    pub fn set_hyper(&mut self, alpha: &[f64], beta: f64) {
        self.alpha.copy_from_slice(alpha);
        self.alpha_sum = alpha.iter().sum();
        self.beta = beta;
        self.beta_sum = beta * self.num_types as f64;
        self.alpha_alias = Alias::build(alpha);
    }

    /// One full MH sweep over every token of the corpus.
    pub fn sweep<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        let k = self.num_topics;
        let beta = self.beta;
        let beta_sum = self.beta_sum;

        // Per-word proposal tables, built lazily on first use this sweep and
        // reused (stale) for every later token of the same word. `qw` stores the
        // unnormalised proposal weights so the acceptance ratio can read
        // p_w(s)/p_w(t) directly (the normaliser cancels).
        let mut word_tables: Vec<Option<(Alias, Vec<f64>)>> =
            (0..self.num_types).map(|_| None).collect();

        // Full document-topic counts for the current document, reused per doc.
        let mut n_dk = vec![0i64; k];

        for d in 0..corpus.docs.len() {
            let doc = &corpus.docs[d];
            let doc_len = doc.len();
            if doc_len == 0 {
                continue;
            }
            let doc_len_f = doc_len as f64;

            // Build full doc-topic counts for this document.
            for v in n_dk.iter_mut() {
                *v = 0;
            }
            for &t in &self.z[d] {
                n_dk[t as usize] += 1;
            }

            for pos in 0..doc_len {
                let w = doc[pos] as usize;
                let old = self.z[d][pos] as usize;

                // --- remove the token from the global counts (now `-di`) ---
                self.n_wk[w][old] -= 1;
                self.n_k[old] -= 1;

                // Lazily build this word's proposal alias table.
                if word_tables[w].is_none() {
                    let mut qw = vec![0.0f64; k];
                    for t in 0..k {
                        qw[t] = (self.n_wk[w][t] as f64 + beta) / (self.n_k[t] as f64 + beta_sum);
                    }
                    let table = Alias::build(&qw);
                    word_tables[w] = Some((table, qw));
                }
                let (w_alias, w_qw) = word_tables[w].as_ref().unwrap();

                // Current candidate topic; `z[pos]` is kept equal to it so the
                // doc-proposal's random-token branch stays consistent.
                let mut s = old;

                for step in 0..self.mh_steps {
                    let word_prop = step % 2 == 0;

                    let t_prop = if word_prop {
                        w_alias.sample(rng)
                    } else {
                        // Doc-proposal in O(1): with prob ∝ doc length, copy a
                        // random token's topic (∝ n_kd, full counts); otherwise
                        // draw from the α alias.
                        let r = rng.gen::<f64>() * (doc_len_f + self.alpha_sum);
                        if r < doc_len_f {
                            self.z[d][rng.gen_range(0..doc_len)] as usize
                        } else {
                            self.alpha_alias.sample(rng)
                        }
                    };

                    if t_prop == s {
                        continue;
                    }

                    // Token-excluded (`-di`) sufficient statistics.
                    let n_td_excl = (n_dk[t_prop] - if t_prop == old { 1 } else { 0 }) as f64;
                    let n_sd_excl = (n_dk[s] - if s == old { 1 } else { 0 }) as f64;
                    let n_tw_di = self.n_wk[w][t_prop] as f64;
                    let n_sw_di = self.n_wk[w][s] as f64;
                    let n_t_di = self.n_k[t_prop] as f64;
                    let n_s_di = self.n_k[s] as f64;

                    // True conditional masses p(t), p(s) (shared numerator forms).
                    let p_t =
                        (n_td_excl + self.alpha[t_prop]) * (n_tw_di + beta) / (n_t_di + beta_sum);
                    let p_s = (n_sd_excl + self.alpha[s]) * (n_sw_di + beta) / (n_s_di + beta_sum);

                    let pi = if word_prop {
                        // π_w = (p(t)/p(s)) · (p_w(s)/p_w(t)), Eq. 8.
                        (p_t / p_s) * (w_qw[s] / w_qw[t_prop])
                    } else {
                        // π_d = (p(t)/p(s)) · (n_sd+α_s)/(n_td+α_t), Eq. 11, with
                        // FULL doc counts (the token sits at candidate s).
                        let n_sd_full = n_sd_excl + 1.0; // token counted at s
                        let n_td_full = n_td_excl; // t_prop != s here
                        (p_t / p_s)
                            * ((n_sd_full + self.alpha[s]) / (n_td_full + self.alpha[t_prop]))
                    };

                    if pi >= 1.0 || rng.gen::<f64>() < pi {
                        s = t_prop;
                        self.z[d][pos] = s as u32; // keep doc-proposal consistent
                    }
                }

                // --- re-add the token at its final topic ---
                self.n_wk[w][s] += 1;
                self.n_k[s] += 1;
                n_dk[old] -= 1;
                n_dk[s] += 1;
                self.z[d][pos] = s as u32;
            }
        }
    }

    /// Accumulate a smoothed φ snapshot `(n_wk+β)/(n_k+β̄)` into `acc[word][topic]`.
    pub fn phi_into(&self, acc: &mut [Vec<f64>]) {
        for w in 0..self.num_types {
            for t in 0..self.num_topics {
                let denom = self.n_k[t] as f64 + self.beta_sum;
                acc[w][t] += (self.n_wk[w][t] as f64 + self.beta) / denom;
            }
        }
    }

    /// Accumulate a smoothed θ snapshot `(n_dk+α)/(len+α_sum)` into `acc[doc][topic]`.
    pub fn theta_into(&self, corpus: &Corpus, acc: &mut [Vec<f64>]) {
        let mut counts = vec![0u32; self.num_topics];
        for d in 0..corpus.docs.len() {
            for c in counts.iter_mut() {
                *c = 0;
            }
            for &t in &self.z[d] {
                counts[t as usize] += 1;
            }
            let denom = corpus.docs[d].len() as f64 + self.alpha_sum;
            for t in 0..self.num_topics {
                acc[d][t] += (counts[t] as f64 + self.alpha[t]) / denom;
            }
        }
    }

    /// Pack the dense state into a [`TopicModel`] so the rest of the codebase
    /// (optimisation, save/load, log-likelihood, held-out inference) can be
    /// reused without knowing the sampler ran on dense tables.
    pub fn to_topic_model(&self) -> TopicModel {
        let mut model = TopicModel::new(self.num_topics, self.alpha_sum, self.beta, self.num_types);
        model.alpha.copy_from_slice(&self.alpha);
        model.alpha_sum = self.alpha_sum;
        model.beta = self.beta;
        model.beta_sum = self.beta_sum;

        model.tokens_per_topic = self.n_k.iter().map(|&c| c as u32).collect();
        model.doc_topics = self.z.clone();

        let bits = model.topic_bits;
        let mut ttc: Vec<Vec<u32>> = Vec::with_capacity(self.num_types);
        for w in 0..self.num_types {
            let mut entries: Vec<u32> = (0..self.num_topics)
                .filter_map(|t| {
                    let c = self.n_wk[w][t];
                    if c > 0 {
                        Some(((c as u32) << bits) | t as u32)
                    } else {
                        None
                    }
                })
                .collect();
            // Packed entries are sorted descending by count (high bits), matching
            // the SparseLDA invariant the readers rely on.
            entries.sort_unstable_by(|a, b| b.cmp(a));
            ttc.push(entries);
        }
        model.type_topic_counts = ttc;
        model
    }
}

impl crate::mh::MhSampler for LightLda {
    fn sweep(&mut self, corpus: &Corpus, rng: &mut rand_pcg::Pcg64Mcg) {
        LightLda::sweep(self, corpus, rng)
    }
    fn set_hyper(&mut self, alpha: &[f64], beta: f64) {
        LightLda::set_hyper(self, alpha, beta)
    }
    fn phi_into(&self, acc: &mut [Vec<f64>]) {
        LightLda::phi_into(self, acc)
    }
    fn theta_into(&self, corpus: &Corpus, acc: &mut [Vec<f64>]) {
        LightLda::theta_into(self, corpus, acc)
    }
    fn to_topic_model(&self) -> TopicModel {
        LightLda::to_topic_model(self)
    }
}
