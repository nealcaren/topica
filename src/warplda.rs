//! WarpLDA: a cache-efficient O(1) MH sampler for LDA
//! (Chen, Li, Zhu & Chen, *PVLDB* 9(10), 2016).
//!
//! Like LightLDA ([`crate::lightlda`]) this is a Metropolis-Hastings sampler that
//! draws from cheap proposal distributions and corrects the bias with an accept/
//! reject step, so each token costs O(1) amortized work. What WarpLDA adds is a
//! **reordering** that makes the random memory access cache-resident, which is
//! where LightLDA bottlenecks (its per-token access to the K×V word-topic matrix
//! misses the cache). The trick is a Monte-Carlo EM (MCEM) formulation in which
//! the count tables are held **fixed while every token is sampled** (a *delayed
//! update*), and the observation that the two MH acceptance ratios decouple:
//!
//! * **Doc-proposal** `q_d(k) ∝ C_dk + α_k`, accepted with (Eq. 7, word phase)
//!   `π_d = min(1, (C_wk' + β)/(C_wk + β) · (C_k + β̄)/(C_k' + β̄))` — needs only
//!   the word counts `C_w` and the global `C_k`.
//! * **Word-proposal** `q_w(k) ∝ C_wk + β`, accepted with (Eq. 7, doc phase)
//!   `π_w = min(1, (C_dk' + α_k')/(C_dk + α_k) · (C_k + β̄)/(C_k' + β̄))` — needs
//!   only the doc counts `C_d` and `C_k`.
//!
//! The `C_d` terms cancel in `π_d` and the `C_w` terms cancel in `π_w`. So one
//! iteration runs as two passes, each touching a single count matrix:
//!
//! * **Doc phase** — visit tokens document-by-document. Build `C_d` for the
//!   document on the fly (one row, cache-resident), *accept the word-proposals*
//!   pending from the last word phase (uses `C_d`), then *draw fresh
//!   doc-proposals* (uses `C_d`).
//! * **Word phase** — visit tokens word-by-word. Build `C_w` for the word on the
//!   fly, *accept the doc-proposals* (uses `C_w`), then *draw fresh
//!   word-proposals* (uses `C_w`).
//!
//! `C_k` is snapshotted at the start of each phase and recomputed between phases.
//! Neither `C_d` nor `C_w` is ever stored as a matrix — both are rebuilt per
//! document / per word from the contiguous run of token topics, so the only
//! random-accessed state is the small `C_k` vector. WarpLDA targets the MCEM/MAP
//! solution, which Asuncion et al. (2009) show is almost identical to the
//! collapsed-Gibbs posterior at matched hyperparameters; the proposal forms are
//! exactly LightLDA's, so the same `to_topic_model` packing is reused unchanged.
//!
//! Stage 1 (this file) implements the algorithm for correctness on the dense
//! per-document / per-word histograms; the word phase still gathers each word's
//! tokens through an index (one `O(N)` scatter), which the physical token
//! reordering of stage 2 removes.

use rand::Rng;

use crate::corpus::Corpus;
use crate::lightlda::Alias;
use crate::model::TopicModel;

/// A WarpLDA sampler. The only persistent per-token state is the current topic
/// `z` and one pending proposal `prop`, both stored document-major; a word→token
/// index lets the word phase visit the same tokens word-major.
pub struct WarpLda {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,

    /// Global topic counts `C_k`, held fixed within a phase.
    n_k: Vec<i64>,
    /// Current topic assignment per token, document-major (parallel to docs).
    z: Vec<Vec<u32>>,
    /// Pending MH proposal per token, document-major. The doc phase fills these
    /// with doc-proposals (consumed by the word phase) and the word phase fills
    /// them with word-proposals (consumed by the next doc phase).
    prop: Vec<Vec<u32>>,
    /// Word → list of (doc, position) for every occurrence of the word.
    word_index: Vec<Vec<(u32, u32)>>,

    /// Alias table over α, for the smoothing branch of the doc-proposal.
    alpha_alias: Alias,

    /// Per-document prior `α_{d,k}` (DMR). When `Some`, the doc phase uses
    /// `doc_alpha[d]` in place of the uniform `alpha`, rebuilding the
    /// doc-proposal smoothing alias per document. `None` is the plain-LDA path.
    doc_alpha: Option<Vec<Vec<f64>>>,

    /// Asymmetric β for SeededLDA. When `Some`, a seed word gets an extra
    /// `seed_weight` pseudocount in each topic it seeds, so β and its per-topic
    /// sum become topic/word dependent (see [`Self::beta_at`] /
    /// [`Self::beta_sum_at`]). `None` is the plain symmetric-β path.
    seed: Option<SeedBeta>,
}

/// SeededLDA's asymmetric-β bookkeeping: `β_{k,w} = β + seed_weight·[w ∈ seeds[k]]`
/// and `β_sum_k = V·β + |seeds[k]|·seed_weight`.
struct SeedBeta {
    seed_weight: f64,
    /// Per-topic β normalizer `β_sum_k`.
    beta_sum_k: Vec<f64>,
    /// `inv_seeds[w]` = the (sorted) topics that word `w` seeds.
    inv_seeds: Vec<Vec<u32>>,
}

impl WarpLda {
    /// Random-initialise the sampler over `corpus` (same scheme as
    /// [`TopicModel::initialize`]). Each token's pending proposal starts equal to
    /// its topic, so the first doc-phase accept step is a no-op.
    pub fn new<R: Rng>(
        corpus: &Corpus,
        num_topics: usize,
        alpha: &[f64],
        beta: f64,
        rng: &mut R,
    ) -> WarpLda {
        let num_types = corpus.num_types();
        let beta_sum = beta * num_types as f64;
        let alpha_sum: f64 = alpha.iter().sum();

        let mut n_k = vec![0i64; num_topics];
        let mut z: Vec<Vec<u32>> = Vec::with_capacity(corpus.docs.len());
        let mut word_index: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_types];

        for (d, doc) in corpus.docs.iter().enumerate() {
            let mut zd = Vec::with_capacity(doc.len());
            for (pos, &w) in doc.iter().enumerate() {
                let t = rng.gen_range(0..num_topics);
                n_k[t] += 1;
                zd.push(t as u32);
                word_index[w as usize].push((d as u32, pos as u32));
            }
            z.push(zd);
        }
        let prop = z.clone();
        let alpha_alias = Alias::build(alpha);

        WarpLda {
            num_topics,
            num_types,
            alpha: alpha.to_vec(),
            alpha_sum,
            beta,
            beta_sum,
            n_k,
            z,
            prop,
            word_index,
            alpha_alias,
            doc_alpha: None,
            seed: None,
        }
    }

    /// Switch the doc-proposal to a per-document prior `α_{d,k}` (DMR). Called
    /// each sweep by the DMR fit loop after recomputing α from the current λ.
    pub fn set_doc_alpha(&mut self, doc_alpha: Vec<Vec<f64>>) {
        self.doc_alpha = Some(doc_alpha);
    }

    /// Enable SeededLDA's asymmetric β. `seeds[k]` is the list of seed word-ids
    /// for topic `k`; each gets an extra `seed_weight` pseudocount in topic `k`.
    pub fn set_seeds(&mut self, seeds: &[Vec<usize>], seed_weight: f64) {
        let k = self.num_topics;
        let v = self.num_types;
        let mut beta_sum_k = vec![self.beta * v as f64; k];
        let mut inv_seeds: Vec<Vec<u32>> = vec![Vec::new(); v];
        for (t, ws) in seeds.iter().enumerate().take(k) {
            // Count each DISTINCT, in-vocabulary seed word once. `beta_at` adds the
            // seed weight per (topic, word) via a membership test, so the per-topic
            // normalizer `beta_sum_k[t]` must equal Σ_w beta_at(t, w) = V·β +
            // (#distinct valid seeds)·seed_weight. Adding `ws.len()` verbatim
            // over-counted out-of-vocabulary and duplicate seed words, inflating the
            // denominator and under-normalizing the seeded topic's φ.
            let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &w in ws {
                if w < v && seen.insert(w) {
                    beta_sum_k[t] += seed_weight;
                    inv_seeds[w].push(t as u32);
                }
            }
        }
        for row in inv_seeds.iter_mut() {
            row.sort_unstable();
        }
        self.seed = Some(SeedBeta {
            seed_weight,
            beta_sum_k,
            inv_seeds,
        });
    }

    /// β_{k,w}: the base β plus the seed boost when word `w` seeds topic `k`.
    #[inline]
    fn beta_at(&self, k: usize, w: usize) -> f64 {
        match &self.seed {
            Some(s) if s.inv_seeds[w].binary_search(&(k as u32)).is_ok() => {
                self.beta + s.seed_weight
            }
            _ => self.beta,
        }
    }

    /// β_sum_k: the per-topic β normalizer (uniform `beta_sum` when unseeded).
    #[inline]
    fn beta_sum_at(&self, k: usize) -> f64 {
        match &self.seed {
            Some(s) => s.beta_sum_k[k],
            None => self.beta_sum,
        }
    }

    /// The current per-token topic assignments, document-major — the input the
    /// DMR λ optimizer needs (its document-topic counts).
    pub fn doc_topics(&self) -> &[Vec<u32>] {
        &self.z
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

    /// Recompute the global topic counts from the current assignments.
    fn recount_n_k(&mut self) {
        for v in self.n_k.iter_mut() {
            *v = 0;
        }
        for zd in &self.z {
            for &t in zd {
                self.n_k[t as usize] += 1;
            }
        }
    }

    /// One WarpLDA iteration: a document phase then a word phase. `C_k` is fixed
    /// during each phase (delayed update) and recomputed in between.
    pub fn sweep<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        if self.doc_alpha.is_some() {
            self.doc_phase_dmr(corpus, rng);
        } else {
            self.doc_phase(corpus, rng);
        }
        self.recount_n_k();
        if self.seed.is_some() {
            self.word_phase_seeded(rng);
        } else {
            self.word_phase(rng);
        }
        self.recount_n_k();
    }

    /// Document phase: accept the pending **word**-proposals (ratio uses `C_d`),
    /// then draw fresh **doc**-proposals. Visits tokens document-by-document, so
    /// only the current document's `C_d` row is randomly accessed.
    fn doc_phase<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        let k = self.num_topics;
        let mut c_d = vec![0i64; k];

        for (d, doc) in corpus.docs.iter().enumerate() {
            let doc_len = doc.len();
            if doc_len == 0 {
                continue;
            }
            // C_d snapshot for this document, held fixed while its tokens sample.
            for v in c_d.iter_mut() {
                *v = 0;
            }
            for &t in &self.z[d] {
                c_d[t as usize] += 1;
            }
            let doc_len_f = doc_len as f64;

            for pos in 0..doc_len {
                let cur = self.z[d][pos] as usize;
                let prop = self.prop[d][pos] as usize;

                // (a) Accept the pending word-proposal against `cur` using C_d/C_k.
                // The counts are the phase-start (delayed) snapshot, so `cur` is
                // this token's `originalk`: exclude its own contribution with a
                // -1 on the `cur` side (the standard collapsed -di term).
                let mut new = cur;
                if prop != cur {
                    let num = (c_d[prop] as f64 + self.alpha[prop])
                        * (self.n_k[cur] as f64 - 1.0 + self.beta_sum_at(cur));
                    let den = (c_d[cur] as f64 - 1.0 + self.alpha[cur])
                        * (self.n_k[prop] as f64 + self.beta_sum_at(prop));
                    if num >= den || rng.gen::<f64>() * den < num {
                        new = prop;
                    }
                }
                self.z[d][pos] = new as u32;

                // (b) Draw a fresh doc-proposal q_d(k) ∝ C_dk + α_k in O(1): with
                // prob ∝ doc length copy a random token's topic (∝ C_dk), else
                // draw from the α alias.
                let r = rng.gen::<f64>() * (doc_len_f + self.alpha_sum);
                let dp = if r < doc_len_f {
                    self.z[d][rng.gen_range(0..doc_len)] as usize
                } else {
                    self.alpha_alias.sample(rng)
                };
                self.prop[d][pos] = dp as u32;
            }
        }
    }

    /// Document phase with a per-document prior `α_{d,k}` (DMR). Same structure
    /// as [`Self::doc_phase`] but the acceptance and the doc-proposal smoothing
    /// draw use this document's own α (from `self.doc_alpha`, cloned into a local
    /// so the token loop can mutate `z`/`prop`; the smoothing alias is rebuilt
    /// per document). Only called when `doc_alpha` is set.
    fn doc_phase_dmr<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        let k = self.num_topics;
        let mut c_d = vec![0i64; k];

        for (d, doc) in corpus.docs.iter().enumerate() {
            let doc_len = doc.len();
            if doc_len == 0 {
                continue;
            }
            for v in c_d.iter_mut() {
                *v = 0;
            }
            for &t in &self.z[d] {
                c_d[t as usize] += 1;
            }
            let doc_len_f = doc_len as f64;

            // Clone this document's α into a local so the token loop below can
            // mutate `self.z`/`self.prop` without holding a borrow of `self`.
            let alpha_d: Vec<f64> = self.doc_alpha.as_ref().unwrap()[d].clone();
            let alpha_sum_d: f64 = alpha_d.iter().sum();
            let doc_alias = Alias::build(&alpha_d);

            for pos in 0..doc_len {
                let cur = self.z[d][pos] as usize;
                let prop = self.prop[d][pos] as usize;

                let mut new = cur;
                if prop != cur {
                    let num = (c_d[prop] as f64 + alpha_d[prop])
                        * (self.n_k[cur] as f64 - 1.0 + self.beta_sum_at(cur));
                    let den = (c_d[cur] as f64 - 1.0 + alpha_d[cur])
                        * (self.n_k[prop] as f64 + self.beta_sum_at(prop));
                    if num >= den || rng.gen::<f64>() * den < num {
                        new = prop;
                    }
                }
                self.z[d][pos] = new as u32;

                let r = rng.gen::<f64>() * (doc_len_f + alpha_sum_d);
                let dp = if r < doc_len_f {
                    self.z[d][rng.gen_range(0..doc_len)] as usize
                } else {
                    doc_alias.sample(rng)
                };
                self.prop[d][pos] = dp as u32;
            }
        }
    }

    /// Word phase: accept the pending **doc**-proposals (ratio uses `C_w`), then
    /// draw fresh **word**-proposals. Visits tokens word-by-word, so only the
    /// current word's `C_w` row is randomly accessed.
    fn word_phase<R: Rng>(&mut self, rng: &mut R) {
        let k = self.num_topics;
        let beta = self.beta;
        let beta_sum = self.beta_sum;
        let mut c_w = vec![0i64; k];

        for w in 0..self.num_types {
            let toks = &self.word_index[w];
            let n_w = toks.len();
            if n_w == 0 {
                continue;
            }
            // C_w snapshot for this word, held fixed while its tokens sample.
            for v in c_w.iter_mut() {
                *v = 0;
            }
            for &(d, pos) in toks {
                c_w[self.z[d as usize][pos as usize] as usize] += 1;
            }
            let n_w_f = n_w as f64;
            // Normaliser of q_w over topics is Σ_k (C_wk + β) = n_w + Kβ.
            let kbeta = k as f64 * beta;

            for &(d, pos) in toks {
                let (d, pos) = (d as usize, pos as usize);
                let cur = self.z[d][pos] as usize;
                let prop = self.prop[d][pos] as usize;

                // (a) Accept the pending doc-proposal against `cur` using C_w/C_k.
                // As in the doc phase, `cur` is this token's `originalk` under the
                // delayed snapshot, so exclude it with a -1 on the `cur` side.
                let mut new = cur;
                if prop != cur {
                    let num = (c_w[prop] as f64 + beta) * (self.n_k[cur] as f64 - 1.0 + beta_sum);
                    let den = (c_w[cur] as f64 - 1.0 + beta) * (self.n_k[prop] as f64 + beta_sum);
                    if num >= den || rng.gen::<f64>() * den < num {
                        new = prop;
                    }
                }
                self.z[d][pos] = new as u32;

                // (b) Draw a fresh word-proposal q_w(k) ∝ C_wk + β in O(1): with
                // prob ∝ n_w copy a random token-of-w's topic (∝ C_wk), else draw
                // a uniform topic (the symmetric β smoothing term).
                let r = rng.gen::<f64>() * (n_w_f + kbeta);
                let wp = if r < n_w_f {
                    let (rd, rp) = toks[rng.gen_range(0..n_w)];
                    self.z[rd as usize][rp as usize] as usize
                } else {
                    rng.gen_range(0..k)
                };
                self.prop[d][pos] = wp as u32;
            }
        }
    }

    /// Word phase with SeededLDA's asymmetric β. Same structure as
    /// [`Self::word_phase`] but `β_{k,w}` carries the seed boost and the
    /// normalizer is the per-topic `β_sum_k`. Only called when `seed` is set.
    fn word_phase_seeded<R: Rng>(&mut self, rng: &mut R) {
        let k = self.num_topics;
        let beta = self.beta;
        let kbeta = k as f64 * beta;
        // Borrow the seed bookkeeping for the whole pass: disjoint from the
        // z/prop/word_index fields the token loop touches.
        let seed = self.seed.as_ref().unwrap();
        let seed_weight = seed.seed_weight;
        let mut c_w = vec![0i64; k];

        for w in 0..self.num_types {
            let toks = &self.word_index[w];
            let n_w = toks.len();
            if n_w == 0 {
                continue;
            }
            for v in c_w.iter_mut() {
                *v = 0;
            }
            for &(d, pos) in toks {
                c_w[self.z[d as usize][pos as usize] as usize] += 1;
            }
            let n_w_f = n_w as f64;
            let inv_w = &seed.inv_seeds[w];
            // β_{k,w} = β + seed_weight·[k seeds w]; Σ_k β_{k,w} = Kβ + |inv_w|·sw.
            let bt = |t: usize| -> f64 {
                if inv_w.binary_search(&(t as u32)).is_ok() {
                    beta + seed_weight
                } else {
                    beta
                }
            };
            let sum_beta_w = kbeta + inv_w.len() as f64 * seed_weight;

            for &(d, pos) in toks {
                let (d, pos) = (d as usize, pos as usize);
                let cur = self.z[d][pos] as usize;
                let prop = self.prop[d][pos] as usize;

                let mut new = cur;
                if prop != cur {
                    let num = (c_w[prop] as f64 + bt(prop))
                        * (self.n_k[cur] as f64 - 1.0 + seed.beta_sum_k[cur]);
                    let den = (c_w[cur] as f64 - 1.0 + bt(cur))
                        * (self.n_k[prop] as f64 + seed.beta_sum_k[prop]);
                    if num >= den || rng.gen::<f64>() * den < num {
                        new = prop;
                    }
                }
                self.z[d][pos] = new as u32;

                // Word-proposal q_w(k) ∝ C_wk + β_{k,w}: with prob ∝ n_w copy a
                // random token-of-w's topic; else draw from β_{k,w} (uniform with
                // prob ∝ Kβ, else a random topic that w seeds).
                let r = rng.gen::<f64>() * (n_w_f + sum_beta_w);
                let wp = if r < n_w_f {
                    let (rd, rp) = toks[rng.gen_range(0..n_w)];
                    self.z[rd as usize][rp as usize] as usize
                } else if (r - n_w_f) < kbeta || inv_w.is_empty() {
                    rng.gen_range(0..k)
                } else {
                    inv_w[rng.gen_range(0..inv_w.len())] as usize
                };
                self.prop[d][pos] = wp as u32;
            }
        }
    }

    /// Smoothed topic-word matrix `(C_wk+β_{k,w})/(C_k+β_sum_k)`, shape
    /// `[topic][word]` (seeded-β aware). Used by the SeededLDA warp path, whose
    /// output orientation is topic-major.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        let v = self.num_types;
        let mut c_w = vec![0i64; k];
        let mut phi = vec![vec![0.0f64; v]; k];
        for w in 0..v {
            for x in c_w.iter_mut() {
                *x = 0;
            }
            for &(d, pos) in &self.word_index[w] {
                c_w[self.z[d as usize][pos as usize] as usize] += 1;
            }
            for t in 0..k {
                phi[t][w] = (c_w[t] as f64 + self.beta_at(t, w))
                    / (self.n_k[t] as f64 + self.beta_sum_at(t));
            }
        }
        phi
    }

    /// Accumulate a smoothed φ snapshot `(C_wk+β_{k,w})/(C_k+β_sum_k)` into
    /// `acc[word][topic]`.
    pub fn phi_into(&self, acc: &mut [Vec<f64>]) {
        // Rebuild the word-topic counts from the assignments (not stored densely).
        let k = self.num_topics;
        let mut c_w = vec![0i64; k];
        for w in 0..self.num_types {
            for v in c_w.iter_mut() {
                *v = 0;
            }
            for &(d, pos) in &self.word_index[w] {
                c_w[self.z[d as usize][pos as usize] as usize] += 1;
            }
            for t in 0..k {
                let denom = self.n_k[t] as f64 + self.beta_sum_at(t);
                acc[w][t] += (c_w[t] as f64 + self.beta_at(t, w)) / denom;
            }
        }
    }

    /// Accumulate a smoothed θ snapshot `(C_dk+α)/(len+α_sum)` into `acc[doc][topic]`.
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

    /// Pack the state into a [`TopicModel`] so the rest of the codebase
    /// (optimisation, save/load, log-likelihood, held-out inference) is reused
    /// unchanged. Mirrors [`crate::lightlda::LightLda::to_topic_model`].
    pub fn to_topic_model(&self) -> TopicModel {
        let mut model = TopicModel::new(self.num_topics, self.alpha_sum, self.beta, self.num_types);
        model.alpha.copy_from_slice(&self.alpha);
        model.alpha_sum = self.alpha_sum;
        model.beta = self.beta;
        model.beta_sum = self.beta_sum;
        model.tokens_per_topic = self.n_k.iter().map(|&c| c as u32).collect();
        model.doc_topics = self.z.clone();

        let bits = model.topic_bits;
        let k = self.num_topics;
        let mut c_w = vec![0i64; k];
        let mut ttc: Vec<Vec<u32>> = Vec::with_capacity(self.num_types);
        for w in 0..self.num_types {
            for v in c_w.iter_mut() {
                *v = 0;
            }
            for &(d, pos) in &self.word_index[w] {
                c_w[self.z[d as usize][pos as usize] as usize] += 1;
            }
            let mut entries: Vec<u32> = (0..k)
                .filter_map(|t| {
                    let c = c_w[t];
                    if c > 0 {
                        Some(((c as u32) << bits) | t as u32)
                    } else {
                        None
                    }
                })
                .collect();
            entries.sort_unstable_by(|a, b| b.cmp(a));
            ttc.push(entries);
        }
        model.type_topic_counts = ttc;
        model
    }
}

impl crate::mh::MhSampler for WarpLda {
    fn sweep(&mut self, corpus: &Corpus, rng: &mut rand_pcg::Pcg64Mcg) {
        WarpLda::sweep(self, corpus, rng)
    }
    fn set_hyper(&mut self, alpha: &[f64], beta: f64) {
        WarpLda::set_hyper(self, alpha, beta)
    }
    fn phi_into(&self, acc: &mut [Vec<f64>]) {
        WarpLda::phi_into(self, acc)
    }
    fn theta_into(&self, corpus: &Corpus, acc: &mut [Vec<f64>]) {
        WarpLda::theta_into(self, corpus, acc)
    }
    fn to_topic_model(&self) -> TopicModel {
        WarpLda::to_topic_model(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    /// Disjoint-vocabulary planted topics over integer word ids: block `b` owns
    /// ids `[b*wpb, b*wpb+wpb)`, and each document draws (twice) from one block.
    /// WarpLDA should place each block's words on a single topic.
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
        let mut s = WarpLda::new(&corpus, k, &alpha, 0.01, &mut rng);
        for _ in 0..200 {
            s.sweep(&corpus, &mut rng);
        }
        let phi = s.to_topic_model().topic_word(); // (k, v) smoothed

        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[t][b].partial_cmp(&phi[t][a]).unwrap());
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
    fn doc_alpha_path_recovers_planted_blocks() {
        // The per-document-α (DMR) doc phase must recover planted topics too.
        // Set a per-doc prior that mildly favours each document's own block, the
        // regime a fitted DMR λ would produce.
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 200);
        let k = n_blocks;
        let v = corpus.num_types();
        let alpha = vec![0.1f64; k];
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut s = WarpLda::new(&corpus, k, &alpha, 0.01, &mut rng);
        // doc d belongs to block d % n_blocks; nudge its prior toward that topic.
        let doc_alpha: Vec<Vec<f64>> = (0..corpus.docs.len())
            .map(|d| {
                let b = d % n_blocks;
                (0..k).map(|t| if t == b { 0.5 } else { 0.1 }).collect()
            })
            .collect();
        s.set_doc_alpha(doc_alpha);
        for _ in 0..200 {
            s.sweep(&corpus, &mut rng);
        }
        let phi = s.to_topic_model().topic_word();
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[t][b].partial_cmp(&phi[t][a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..n_blocks {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), n_blocks, "only recovered {covered:?}");
        // doc_topics() exposes assignments for the λ optimizer.
        assert_eq!(s.doc_topics().len(), corpus.docs.len());
    }

    #[test]
    fn seeded_path_recovers_planted_blocks() {
        // Seed each block's first word into its topic; the seeded word phase must
        // still recover every block (and respect the asymmetric β).
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 200);
        let k = n_blocks;
        let v = corpus.num_types();
        let alpha = vec![0.1f64; k];
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut s = WarpLda::new(&corpus, k, &alpha, 0.01, &mut rng);
        // seeds[t] = the first word id of block t.
        let seeds: Vec<Vec<usize>> = (0..k).map(|t| vec![t * wpb]).collect();
        s.set_seeds(&seeds, 50.0);
        for _ in 0..200 {
            s.sweep(&corpus, &mut rng);
        }
        let phi = s.to_topic_model().topic_word();
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[t][b].partial_cmp(&phi[t][a]).unwrap());
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
    fn deterministic_for_seed() {
        let (corpus, _) = planted(3, 4, 90);
        let alpha = vec![0.1f64; 3];
        let run = || {
            let mut rng = Pcg64Mcg::seed_from_u64(7);
            let mut s = WarpLda::new(&corpus, 3, &alpha, 0.01, &mut rng);
            for _ in 0..40 {
                s.sweep(&corpus, &mut rng);
            }
            s.to_topic_model().doc_topics
        };
        assert_eq!(run(), run());
    }

    /// Rough per-sweep cost of WarpLDA (stage 1) vs SparseLDA at large K on a
    /// poliblog-sized synthetic corpus. Run with:
    ///   cargo test --release --lib warplda::tests::bench_vs_sparse -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_vs_sparse() {
        use crate::model::TopicModel;
        use std::time::Instant;

        // 2,000 docs x ~120 tokens over V=3,000, mildly clustered so SparseLDA
        // sees realistic sparsity (each doc favours one of 40 vocab bands).
        let v = 3000usize;
        let bands = 40usize;
        let band = v / bands;
        let n_docs = 2000usize;
        let docs: Vec<Vec<u32>> = (0..n_docs)
            .map(|d| {
                let b = d % bands;
                (0..120)
                    .map(|i| ((b * band) + (i * 7 + d * 13) % band) as u32)
                    .collect()
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

        for &k in &[100usize, 500usize] {
            let alpha_each = 0.1f64;
            let alpha = vec![alpha_each; k];
            let beta = 0.01f64;
            let sweeps = 30;

            // WarpLDA
            let mut rng = Pcg64Mcg::seed_from_u64(1);
            let mut w = WarpLda::new(&corpus, k, &alpha, beta, &mut rng);
            let t0 = Instant::now();
            for _ in 0..sweeps {
                w.sweep(&corpus, &mut rng);
            }
            let warp = t0.elapsed().as_secs_f64() / sweeps as f64;

            // SparseLDA
            let mut rng = Pcg64Mcg::seed_from_u64(1);
            let mut m = TopicModel::new(k, alpha_each * k as f64, beta, v);
            m.initialize(&corpus, &mut rng);
            let t0 = Instant::now();
            for _ in 0..sweeps {
                crate::sampler::run_iteration(&mut m, &corpus, &mut rng);
            }
            let sparse = t0.elapsed().as_secs_f64() / sweeps as f64;

            println!(
                "K={k:>4}  warp {:>7.1}ms/sweep   sparse {:>7.1}ms/sweep   warp/sparse {:.2}x",
                warp * 1e3,
                sparse * 1e3,
                warp / sparse
            );
        }
    }

    /// #430: `set_seeds` must keep the per-topic normalizer consistent with the
    /// per-word betas: Σ_w beta_at(t, w) == beta_sum_at(t). Duplicate and
    /// out-of-vocabulary seed words previously inflated `beta_sum_k` (it added
    /// `ws.len()` verbatim) while `beta_at` counts each distinct in-vocab seed once.
    #[test]
    fn set_seeds_normalizer_matches_per_word_betas() {
        let (corpus, _) = planted(2, 3, 20); // V = 6, ids 0..5
        let k = 2;
        let mut rng = Pcg64Mcg::seed_from_u64(0);
        let mut w = WarpLda::new(&corpus, k, &vec![0.1; k], 0.05, &mut rng);
        // Topic 0: a duplicate (0 twice) and an OOV id (99); topic 1: a duplicate.
        let seeds = vec![vec![0usize, 1, 0, 99], vec![2usize, 2, 3]];
        w.set_seeds(&seeds, 7.0);
        let v = corpus.num_types();
        for t in 0..k {
            let summed: f64 = (0..v).map(|word| w.beta_at(t, word)).sum();
            let norm = w.beta_sum_at(t);
            assert!(
                (summed - norm).abs() < 1e-9,
                "topic {t}: Σ_w beta_at = {summed} != beta_sum_at = {norm}"
            );
        }
    }
}
