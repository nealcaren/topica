use crate::corpus::Corpus;
use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use rand::Rng;

/// Largest per-`(word, topic)` count representable in the packed `u32` entry
/// when `topic_bits` low bits are reserved for the topic index. The count lives
/// in the high `32 - topic_bits` bits, so a count of `2^(32 - topic_bits)` (or
/// more) shifts into or past bit 32 and silently corrupts the entry. `initialize`
/// refuses any corpus whose most frequent word could reach that many tokens in a
/// single topic.
#[inline]
pub fn max_packable_count(topic_bits: u32) -> u32 {
    match topic_bits {
        // K == 1: the topic index needs no bits, so the entire 32-bit width holds
        // the count. `1u32 << 32` is an out-of-range shift (it would wrap to 1 and
        // wrongly yield a ceiling of 0, rejecting every K=1 fit), so special-case it.
        0 => u32::MAX,
        // Degenerate: no bits left for the count. Cannot happen for real K, but
        // keep the arithmetic well-defined rather than shifting by 32.
        b if b >= 32 => 0,
        b => (1u32 << (32 - b)) - 1,
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TopicModel {
    pub num_topics: usize,
    pub num_types: usize,
    // Bitmask and shift for packing (count, topic) into a u32.
    // Entry = (count << topic_bits) | topic
    // Sorted descending by value (so descending by count), zeroes terminate.
    pub topic_mask: u32,
    pub topic_bits: u32,

    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,

    // For each word type: sorted Vec<u32> of packed (count, topic) pairs.
    pub type_topic_counts: Vec<Vec<u32>>,
    // Total tokens assigned to each topic across all documents.
    pub tokens_per_topic: Vec<u32>,
    // Per-document topic assignments, parallel to corpus.docs.
    pub doc_topics: Vec<Vec<u32>>,
}

impl TopicModel {
    pub fn new(num_topics: usize, alpha_sum: f64, beta: f64, num_types: usize) -> Self {
        // Compute the topic_mask and topic_bits.
        // We need enough bits to represent any topic index.
        // MALLET uses: if power-of-2, mask = num_topics-1; else next higher power-of-2 mask.
        let (topic_mask, topic_bits) = if num_topics.is_power_of_two() {
            let mask = (num_topics - 1) as u32;
            let bits = mask.count_ones();
            (mask, bits)
        } else {
            let mask = (num_topics as u32).next_power_of_two() - 1;
            let bits = mask.count_ones();
            (mask, bits)
        };

        let alpha = vec![alpha_sum / num_topics as f64; num_topics];
        let beta_sum = beta * num_types as f64;

        TopicModel {
            num_topics,
            num_types,
            topic_mask,
            topic_bits,
            alpha,
            alpha_sum,
            beta,
            beta_sum,
            type_topic_counts: Vec::new(),
            tokens_per_topic: vec![0; num_topics],
            doc_topics: Vec::new(),
        }
    }

    pub fn initialize<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        // Count total occurrences of each word type so we can size the inner vecs.
        let mut type_totals = vec![0usize; self.num_types];
        for doc in &corpus.docs {
            for &word_id in doc {
                type_totals[word_id as usize] += 1;
            }
        }

        // Fail fast before any packing if the most frequent word could exceed the
        // packed-count ceiling. A word with `n` total tokens can, in the worst
        // case, land all `n` in one topic, so `n` is a safe upper bound on any
        // single (word, topic) count. Overflowing the ceiling would silently
        // corrupt entries mid-sampling; refuse the run here with a clear message.
        let ceiling = max_packable_count(self.topic_bits) as usize;
        if let Some(&worst) = type_totals.iter().max() {
            assert!(
                worst <= ceiling,
                "topica: a word occurs {worst} times, but the packed (count, topic) \
                 table for num_topics={} can represent at most {ceiling} occurrences \
                 of a single word in one topic (32-bit packing). Reduce num_topics or \
                 split the corpus.",
                self.num_topics
            );
        }

        // Allocate type_topic_counts: size = min(num_topics, type_total) per word.
        self.type_topic_counts = type_totals
            .iter()
            .map(|&total| vec![0u32; self.num_topics.min(total)])
            .collect();

        // Random initial topic assignments.
        self.doc_topics = corpus
            .docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .map(|_| rng.gen_range(0..self.num_topics) as u32)
                    .collect()
            })
            .collect();

        // Build type_topic_counts and tokens_per_topic from initial assignments.
        // Collect (word_id, topic) pairs first to avoid borrow conflict.
        let assignments: Vec<(usize, usize)> = corpus
            .docs
            .iter()
            .zip(self.doc_topics.iter())
            .flat_map(|(doc, topics)| {
                doc.iter()
                    .map(|&w| w as usize)
                    .zip(topics.iter().map(|&t| t as usize))
            })
            .collect();

        for (word_id, topic) in assignments {
            self.tokens_per_topic[topic] += 1;
            self.increment_type_topic(word_id, topic);
        }
    }

    /// Rebuild the count tables from explicit per-token topic assignments
    /// instead of random initialization — used to restore a model from a saved
    /// Gibbs state (e.g. a MALLET ``--output-state`` file). `doc_topics` must be
    /// parallel to `corpus.docs` (one topic per token, same order).
    pub fn initialize_from_assignments(&mut self, corpus: &Corpus, doc_topics: Vec<Vec<u32>>) {
        let mut type_totals = vec![0usize; self.num_types];
        for doc in &corpus.docs {
            for &word_id in doc {
                type_totals[word_id as usize] += 1;
            }
        }
        self.type_topic_counts = type_totals
            .iter()
            .map(|&total| vec![0u32; self.num_topics.min(total)])
            .collect();
        self.doc_topics = doc_topics;

        let assignments: Vec<(usize, usize)> = corpus
            .docs
            .iter()
            .zip(self.doc_topics.iter())
            .flat_map(|(doc, topics)| {
                doc.iter()
                    .map(|&w| w as usize)
                    .zip(topics.iter().map(|&t| t as usize))
            })
            .collect();
        for (word_id, topic) in assignments {
            self.tokens_per_topic[topic] += 1;
            self.increment_type_topic(word_id, topic);
        }
    }

    /// Anchor-word / spectral initialization: seed each token's topic by
    /// sampling from the categorical p(t | w) ∝ β[t][w], where `beta` is the
    /// K×V topic-word matrix from [`crate::spectral::spectral_init`]. This
    /// biases the starting state toward the recovered anchor structure while
    /// keeping the assignment stochastic, so within-document mixing is
    /// preserved and the sampler is not pinned to a degenerate start. Any word
    /// whose β column carries no mass falls back to a uniform draw.
    ///
    /// Deterministic given `rng`. Equivalent in every other respect to
    /// [`Self::initialize`] (same count-table sizing and accumulation); only
    /// the initial topic draw differs.
    pub fn initialize_spectral<R: Rng>(&mut self, corpus: &Corpus, beta: &[Vec<f64>], rng: &mut R) {
        let k = self.num_topics;

        let mut type_totals = vec![0usize; self.num_types];
        for doc in &corpus.docs {
            for &word_id in doc {
                type_totals[word_id as usize] += 1;
            }
        }
        self.type_topic_counts = type_totals
            .iter()
            .map(|&total| vec![0u32; self.num_topics.min(total)])
            .collect();

        // Per-word cumulative topic weights from the β columns (one length-k
        // prefix sum per word type). The last entry is the column's total mass;
        // a (near-)zero total marks a word we sample uniformly instead.
        let mut col_cum: Vec<Vec<f64>> = Vec::with_capacity(self.num_types);
        for w in 0..self.num_types {
            let mut cum = Vec::with_capacity(k);
            let mut s = 0.0f64;
            for t in 0..k {
                s += beta[t][w].max(0.0);
                cum.push(s);
            }
            col_cum.push(cum);
        }

        self.doc_topics = corpus
            .docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .map(|&w| {
                        let cum = &col_cum[w as usize];
                        let total = cum[k - 1];
                        if total <= 0.0 {
                            return rng.gen_range(0..k) as u32;
                        }
                        let u = rng.gen::<f64>() * total;
                        // First topic whose prefix sum exceeds u.
                        let mut t = 0usize;
                        while t + 1 < k && cum[t] <= u {
                            t += 1;
                        }
                        t as u32
                    })
                    .collect()
            })
            .collect();

        let assignments: Vec<(usize, usize)> = corpus
            .docs
            .iter()
            .zip(self.doc_topics.iter())
            .flat_map(|(doc, topics)| {
                doc.iter()
                    .map(|&w| w as usize)
                    .zip(topics.iter().map(|&t| t as usize))
            })
            .collect();
        for (word_id, topic) in assignments {
            self.tokens_per_topic[topic] += 1;
            self.increment_type_topic(word_id, topic);
        }
    }

    /// Increment the count for (word_id, topic) in type_topic_counts,
    /// maintaining descending sort.
    pub fn increment_type_topic(&mut self, word_id: usize, topic: usize) {
        let counts = &mut self.type_topic_counts[word_id];
        let topic = topic as u32;

        // Find the existing entry for this topic, or the first empty slot.
        let mut index = 0;
        while index < counts.len() && counts[index] > 0 {
            if counts[index] & self.topic_mask == topic {
                break;
            }
            index += 1;
        }

        if index == counts.len() {
            // This shouldn't happen if we sized correctly, but grow if needed.
            counts.push(0);
        }

        let current_count = counts[index] >> self.topic_bits;
        counts[index] = ((current_count + 1) << self.topic_bits) | topic;

        // Bubble up to maintain descending order.
        while index > 0 && counts[index] > counts[index - 1] {
            counts.swap(index, index - 1);
            index -= 1;
        }
    }

    /// Decrement the count for (word_id, topic) in type_topic_counts,
    /// maintaining descending sort (mirror of `increment_type_topic`, used by
    /// the Labeled-LDA restricted sampler). No-op if the entry is absent.
    pub fn decrement_type_topic(&mut self, word_id: usize, topic: usize) {
        let counts = &mut self.type_topic_counts[word_id];
        let topic = topic as u32;

        let mut index = 0;
        while index < counts.len() && counts[index] > 0 {
            if counts[index] & self.topic_mask == topic {
                break;
            }
            index += 1;
        }
        if index >= counts.len() || counts[index] == 0 {
            return; // not present
        }

        let new_count = (counts[index] >> self.topic_bits) - 1;
        counts[index] = if new_count == 0 {
            0
        } else {
            (new_count << self.topic_bits) | topic
        };

        // Bubble down: a smaller value (or a zero) sinks past larger entries.
        while index + 1 < counts.len() && counts[index] < counts[index + 1] {
            counts.swap(index, index + 1);
            index += 1;
        }
    }

    /// Look up the count for (word_id, topic). Returns 0 if not present.
    pub fn get_type_topic_count(&self, word_id: usize, topic: usize) -> u32 {
        let counts = &self.type_topic_counts[word_id];
        let topic = topic as u32;
        for &entry in counts {
            if entry == 0 {
                break;
            }
            if entry & self.topic_mask == topic {
                return entry >> self.topic_bits;
            }
        }
        0
    }

    /// Smoothed topic-word point estimate φ (K×V):
    /// φ[t][w] = (β + count(w,t)) / (β·V + tokens_per_topic[t]), unpacking the
    /// packed type_topic_counts via topic_mask/topic_bits. Matches build_phi_cache.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        let v = self.num_types;
        let denom: Vec<f64> = (0..k)
            .map(|t| self.beta_sum + self.tokens_per_topic[t] as f64)
            .collect();
        // start every cell at the no-count value β/denom
        let mut phi: Vec<Vec<f64>> = (0..k).map(|t| vec![self.beta / denom[t]; v]).collect();
        for w in 0..v {
            for &entry in &self.type_topic_counts[w] {
                if entry == 0 {
                    break;
                }
                let t = (entry & self.topic_mask) as usize;
                let count = (entry >> self.topic_bits) as f64;
                phi[t][w] = (self.beta + count) / denom[t];
            }
        }
        phi
    }

    /// Smoothed document-topic point estimate θ (D×K):
    /// θ[d][t] = (count_t(d) + α[t]) / (N_d + α_sum). Matches the load_state build.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        self.doc_topics
            .iter()
            .map(|topics| {
                let mut cnt = vec![0.0f64; k];
                for &t in topics {
                    cnt[t as usize] += 1.0;
                }
                let denom = topics.len() as f64 + self.alpha_sum;
                (0..k).map(|t| (cnt[t] + self.alpha[t]) / denom).collect()
            })
            .collect()
    }
}

impl Estimator for TopicModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        TopicModel::topic_word(self)
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        TopicModel::doc_topic(self)
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }
    fn converged(&self) -> Option<bool> {
        None
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for TopicModel {
    fn alpha(&self) -> Vec<f64> {
        self.alpha.clone()
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }
    fn doc_lengths(&self) -> Vec<usize> {
        self.doc_topics.iter().map(|d| d.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use crate::sampler::run_iteration;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn small_corpus() -> Corpus {
        // 20 docs, vocabulary size 10, 5 tokens each
        let v = 10usize;
        let docs: Vec<Vec<u32>> = (0..20usize)
            .map(|d| (0..5).map(|i| ((i + d * 3) % v) as u32).collect())
            .collect();
        Corpus {
            id_to_word: (0..v).map(|i| format!("w{i}")).collect(),
            docs,
            doc_names: (0..20).map(|i| format!("d{i}")).collect(),
            doc_labels: vec![String::new(); 20],
            doc_freqs: vec![0u32; v],
            total_freqs: vec![0u32; v],
        }
    }

    #[test]
    fn topicmodel_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let corpus = small_corpus();
        let v = corpus.num_types();
        let mut m = TopicModel::new(3, 3.0, 0.1, v);
        m.initialize(&corpus, &mut rng);
        for _ in 0..10 {
            run_iteration(&mut m, &corpus, &mut rng);
        }

        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }

    /// Rebuild the count tables from the token assignments and assert they exactly
    /// equal the incrementally-maintained ones. The packed `type_topic_counts` and
    /// `tokens_per_topic` are updated in place every token; `doc_topics` is the
    /// consistency reference (this checks the tables stay consistent with the
    /// assignments, not that the assignments are a correct posterior draw). A
    /// packed-count or increment/decrement bug corrupts the tables while `doc_topics`
    /// stays intact, so it moves φ/θ only slightly and slips past parity's
    /// central-tendency thresholds -- this catches it exactly.
    fn assert_counts_reconstruct(m: &TopicModel, corpus: &Corpus, sweep: usize) {
        let (k, v) = (m.num_topics, m.num_types);
        let mut dense = vec![vec![0u32; k]; v];
        let mut tpt = vec![0u32; k];
        for (doc, topics) in corpus.docs.iter().zip(m.doc_topics.iter()) {
            assert_eq!(
                doc.len(),
                topics.len(),
                "sweep {sweep}: doc/assignment length mismatch"
            );
            for (&w, &t) in doc.iter().zip(topics.iter()) {
                dense[w as usize][t as usize] += 1;
                tpt[t as usize] += 1;
            }
        }
        // Decode the packed table directly (entry = (count << topic_bits) | topic):
        // every real entry must have an in-range topic, the rebuilt count, and no
        // duplicate slot; every nonzero rebuilt count must have a packed entry.
        for w in 0..v {
            let entries = &m.type_topic_counts[w];
            // Structural invariant the sparse readers depend on (they break at the
            // first zero): nonzero entries are descending and all zeros are trailing.
            // A stranded nonzero after a zero would make production silently drop that
            // topic's mass, so guard it explicitly rather than skipping past zeros.
            let mut zero_seen = false;
            for i in 0..entries.len() {
                if entries[i] == 0 {
                    zero_seen = true;
                } else {
                    assert!(
                        !zero_seen,
                        "sweep {sweep}: word {w} has a nonzero packed entry after a \
                         zero slot (broken zeros-at-tail)"
                    );
                    assert!(
                        i == 0 || entries[i - 1] >= entries[i],
                        "sweep {sweep}: word {w} packed entries are not descending"
                    );
                }
            }
            let mut seen = vec![false; k];
            for &entry in entries {
                if entry == 0 {
                    break; // matches the sparse readers: zeros are trailing (asserted)
                }
                let t = (entry & m.topic_mask) as usize;
                let c = entry >> m.topic_bits;
                assert!(
                    t < k,
                    "sweep {sweep}: word {w} packed entry topic {t} >= K={k}"
                );
                assert!(
                    !seen[t],
                    "sweep {sweep}: word {w} has two packed entries for topic {t}"
                );
                seen[t] = true;
                assert_eq!(
                    c, dense[w][t],
                    "sweep {sweep}: type_topic_counts[{w}][{t}] = {c} != rebuilt {}",
                    dense[w][t]
                );
            }
            for t in 0..k {
                assert!(
                    seen[t] || dense[w][t] == 0,
                    "sweep {sweep}: word {w} topic {t} has rebuilt count {} but no packed entry",
                    dense[w][t]
                );
            }
        }
        assert_eq!(
            m.tokens_per_topic, tpt,
            "sweep {sweep}: tokens_per_topic drifted"
        );
    }

    #[test]
    fn counts_reconstruct_from_assignments_every_sweep() {
        // Both packing regimes: K=4 (power of two -> exact-width mask) and K=5
        // (not -> the ceil-log2 mask), across init and 15 sweeps.
        for &k in &[4usize, 5usize] {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let corpus = small_corpus();
            let v = corpus.num_types();
            let mut m = TopicModel::new(k, k as f64 * 0.1, 0.1, v);
            m.initialize(&corpus, &mut rng);
            assert_counts_reconstruct(&m, &corpus, 0);
            for sweep in 1..=15 {
                run_iteration(&mut m, &corpus, &mut rng);
                assert_counts_reconstruct(&m, &corpus, sweep);
            }
        }
    }

    #[test]
    fn max_packable_count_matches_the_high_bit_width() {
        // Count occupies the high (32 - topic_bits) bits.
        assert_eq!(max_packable_count(0), u32::MAX); // K=1: full 32-bit count width
        assert_eq!(max_packable_count(1), (1u32 << 31) - 1);
        assert_eq!(max_packable_count(10), (1u32 << 22) - 1); // K up to 1024
        assert_eq!(max_packable_count(16), 65_535); // K up to 65536
        assert_eq!(max_packable_count(30), 3);
        assert_eq!(max_packable_count(32), 0); // degenerate, well-defined
    }

    // A word appearing more times than the packed count can hold must be rejected
    // at initialize, before any packing can silently corrupt an entry. We force a
    // tiny ceiling with topic_bits=30 (=> max 3) while keeping num_topics small so
    // the allocations stay trivial; this is the same overflow path a real
    // high-K + very-frequent-word corpus would hit.
    #[test]
    #[should_panic(expected = "can represent at most")]
    fn initialize_rejects_a_word_that_would_overflow_the_packed_count() {
        let mut m = TopicModel {
            num_topics: 4,
            num_types: 1,
            topic_mask: (1u32 << 30) - 1,
            topic_bits: 30, // ceiling = 3
            alpha: vec![0.25; 4],
            alpha_sum: 1.0,
            beta: 0.01,
            beta_sum: 0.01,
            type_topic_counts: Vec::new(),
            tokens_per_topic: vec![0; 4],
            doc_topics: Vec::new(),
        };
        // One word type occurring 4 times (> ceiling of 3), all in one document.
        let corpus = Corpus {
            id_to_word: vec!["w0".to_string()],
            docs: vec![vec![0u32, 0, 0, 0]],
            doc_names: vec!["d0".to_string()],
            doc_labels: vec![String::new()],
            doc_freqs: vec![0u32; 1],
            total_freqs: vec![0u32; 1],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        m.initialize(&corpus, &mut rng);
    }

    // The exact per-sweep bound (a word occurring exactly `ceiling` times) is
    // accepted: incrementing a count up to `ceiling` never overflows.
    #[test]
    fn initialize_accepts_a_word_at_exactly_the_ceiling() {
        let mut m = TopicModel {
            num_topics: 4,
            num_types: 1,
            topic_mask: (1u32 << 30) - 1,
            topic_bits: 30, // ceiling = 3
            alpha: vec![0.25; 4],
            alpha_sum: 1.0,
            beta: 0.01,
            beta_sum: 0.01,
            type_topic_counts: Vec::new(),
            tokens_per_topic: vec![0; 4],
            doc_topics: Vec::new(),
        };
        let corpus = Corpus {
            id_to_word: vec!["w0".to_string()],
            docs: vec![vec![0u32, 0, 0]], // 3 == ceiling
            doc_names: vec!["d0".to_string()],
            doc_labels: vec![String::new()],
            doc_freqs: vec![0u32; 1],
            total_freqs: vec![0u32; 1],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        m.initialize(&corpus, &mut rng); // must not panic
    }

    // Regression for the K=1 panic (#447): num_topics=1 => topic_bits=0 => the count
    // uses the full 32-bit width, so the ceiling is u32::MAX and a frequent word must
    // be accepted rather than rejected by the overflow guard.
    #[test]
    fn initialize_accepts_a_frequent_word_when_k_is_one() {
        let mut m = TopicModel::new(1, 1.0, 0.01, 1);
        assert_eq!(m.topic_bits, 0);
        let corpus = Corpus {
            id_to_word: vec!["w0".to_string()],
            docs: vec![vec![0u32; 49]], // 49 occurrences, as in the CI-failing fits
            doc_names: vec!["d0".to_string()],
            doc_labels: vec![String::new()],
            doc_freqs: vec![0u32; 1],
            total_freqs: vec![0u32; 1],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        m.initialize(&corpus, &mut rng); // must not panic
    }
}
