use crate::corpus::Corpus;
use crate::model::TopicModel;
use rand::Rng;

/// Run one full Gibbs sweep over all documents.
pub fn run_iteration<R: Rng>(model: &mut TopicModel, corpus: &Corpus, rng: &mut R) {
    run_sweep(
        &mut model.type_topic_counts,
        &mut model.tokens_per_topic,
        &mut model.doc_topics,
        &corpus.docs,
        &model.alpha,
        model.beta,
        model.beta_sum,
        model.topic_mask,
        model.topic_bits,
        model.num_topics,
        rng,
    );
}

/// Run one Gibbs sweep over `docs`, mutating `type_topic_counts`,
/// `tokens_per_topic`, and `doc_topics` in place. `doc_topics` must be aligned
/// with `docs` (same length, same order).
///
/// This is the reusable core behind `run_iteration`; the Python bindings call
/// it directly to sample document partitions on worker threads against
/// per-worker copies of the count tables (MALLET-style parallel sampling).
#[allow(clippy::too_many_arguments)]
pub fn run_sweep<R: Rng>(
    type_topic_counts: &mut [Vec<u32>],
    tokens_per_topic: &mut [u32],
    doc_topics: &mut [Vec<u32>],
    docs: &[Vec<u32>],
    alpha: &[f64],
    beta: f64,
    beta_sum: f64,
    topic_mask: u32,
    topic_bits: u32,
    num_topics: usize,
    rng: &mut R,
) {
    // Pre-compute the smoothing-only mass and per-topic coefficients.
    let mut smoothing_only_mass = 0.0f64;
    let mut cached_coefficients = vec![0.0f64; num_topics];
    for t in 0..num_topics {
        let denom = tokens_per_topic[t] as f64 + beta_sum;
        smoothing_only_mass += alpha[t] * beta / denom;
        cached_coefficients[t] = alpha[t] / denom;
    }

    // Scratch space: allocated once per sweep, reused across every doc and token.
    let mut local_topic_counts = vec![0u32; num_topics];
    // u32 instead of usize halves the size of this hot array.
    let mut local_topic_index = vec![0u32; num_topics];
    let mut scored_positions = vec![0usize; num_topics];
    let mut scored_values = vec![0.0f64; num_topics];

    for doc_idx in 0..docs.len() {
        sample_doc(
            type_topic_counts,
            tokens_per_topic,
            doc_topics,
            alpha,
            beta,
            beta_sum,
            topic_mask,
            topic_bits,
            num_topics,
            &docs[doc_idx],
            doc_idx,
            rng,
            &mut smoothing_only_mass,
            &mut cached_coefficients,
            &mut local_topic_counts,
            &mut local_topic_index,
            &mut scored_positions,
            &mut scored_values,
        );
    }
}

/// Sample new topics for a single document with the SparseLDA three-bucket
/// decomposition. Exposed within the crate so the DMR sampler can reuse the
/// exact same per-token math with a per-document `alpha` vector (recomputing
/// the smoothing mass/coefficients per document, since DMR's prior varies by
/// document).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_doc<R: Rng>(
    type_topic_counts: &mut [Vec<u32>],
    tokens_per_topic: &mut [u32],
    doc_topics: &mut [Vec<u32>],
    alpha: &[f64],
    beta: f64,
    beta_sum: f64,
    topic_mask: u32,
    topic_bits: u32,
    num_topics: usize,
    corpus_doc: &[u32],
    doc_idx: usize,
    rng: &mut R,
    smoothing_only_mass: &mut f64,
    cached_coefficients: &mut [f64],
    local_topic_counts: &mut [u32],
    local_topic_index: &mut [u32],
    scored_positions: &mut [usize],
    scored_values: &mut [f64],
) {
    let doc_len = corpus_doc.len();
    if doc_len == 0 {
        return;
    }

    // --- Populate local topic counts for this document ---
    //
    // Both `local_topic_counts` (all zero) and `cached_coefficients` (the
    // smoothing-only value α[t]/denom for every t) hold their inter-document
    // invariant on entry, restored by the reset loop at the end of the previous
    // document. We therefore touch only the topics this document actually uses
    // (O(doc_len + nonzero·log nonzero)) instead of scanning all K topics — a
    // large win at high K with short documents, and bit-identical: the set of
    // nonzero entries and their ascending order in `local_topic_index` are
    // exactly what the old O(K) scan produced.
    let mut non_zero_topics: usize = 0;
    for &topic in &doc_topics[doc_idx] {
        let t = topic as usize;
        if local_topic_counts[t] == 0 {
            local_topic_index[non_zero_topics] = topic;
            non_zero_topics += 1;
        }
        local_topic_counts[t] += 1;
    }
    // Restore the ascending-by-topic order the linear scan guaranteed; the
    // sampler then maintains this order incrementally for the rest of the doc.
    local_topic_index[..non_zero_topics].sort_unstable();

    let mut topic_beta_mass: f64 = 0.0;
    for di in 0..non_zero_topics {
        let t = local_topic_index[di] as usize;
        let n = local_topic_counts[t] as f64;
        topic_beta_mass += beta * n / (tokens_per_topic[t] as f64 + beta_sum);
        cached_coefficients[t] = (alpha[t] + n) / (tokens_per_topic[t] as f64 + beta_sum);
    }

    // --- Sample each token ---
    for pos in 0..doc_len {
        let word_id = corpus_doc[pos] as usize;
        let old_topic = doc_topics[doc_idx][pos] as usize;

        // --- Phase A: remove old token from all counts ---
        *smoothing_only_mass -=
            alpha[old_topic] * beta / (tokens_per_topic[old_topic] as f64 + beta_sum);
        topic_beta_mass -= beta * local_topic_counts[old_topic] as f64
            / (tokens_per_topic[old_topic] as f64 + beta_sum);

        local_topic_counts[old_topic] -= 1;

        if local_topic_counts[old_topic] == 0 {
            let mut di = 0;
            while local_topic_index[di] as usize != old_topic {
                di += 1;
            }
            while di + 1 < non_zero_topics {
                local_topic_index[di] = local_topic_index[di + 1];
                di += 1;
            }
            non_zero_topics -= 1;
        }

        tokens_per_topic[old_topic] -= 1;

        *smoothing_only_mass +=
            alpha[old_topic] * beta / (tokens_per_topic[old_topic] as f64 + beta_sum);
        topic_beta_mass += beta * local_topic_counts[old_topic] as f64
            / (tokens_per_topic[old_topic] as f64 + beta_sum);
        cached_coefficients[old_topic] = (alpha[old_topic] + local_topic_counts[old_topic] as f64)
            / (tokens_per_topic[old_topic] as f64 + beta_sum);

        // --- Phase B: score and sample using type_topic_counts[word_id] ---
        //
        // We obtain a direct reference to the inner Vec once per token.
        // All subsequent accesses go through `type_counts[index]` — a single
        // pointer dereference — rather than re-resolving
        // type_topic_counts[word_id][index] through two levels of indirection
        // on every loop iteration.
        //
        // This is safe because:
        //  (a) type_topic_counts and tokens_per_topic are separate parameters,
        //      so Rust's field-splitting rules let both be accessed concurrently.
        //  (b) We only ever touch index word_id in this scope; no aliasing.
        //  (c) The inner Vec never reallocates: it is pre-sized to
        //      min(num_topics, word_total_freq), which is the strict upper bound
        //      on distinct topic assignments for any word.
        let type_counts = &mut type_topic_counts[word_id];
        let entries_len = type_counts.len();

        let mut topic_term_mass: f64 = 0.0;
        let mut num_scored: usize = 0;
        let mut already_decremented = false;
        let mut index: usize = 0;

        while index < entries_len && type_counts[index] > 0 {
            let entry = type_counts[index];
            let current_topic = (entry & topic_mask) as usize;
            let current_count = entry >> topic_bits;

            if !already_decremented && current_topic == old_topic {
                let new_count = current_count - 1;
                type_counts[index] = if new_count == 0 {
                    0
                } else {
                    (new_count << topic_bits) | (old_topic as u32)
                };
                let mut si = index;
                while si + 1 < entries_len && type_counts[si] < type_counts[si + 1] {
                    type_counts.swap(si, si + 1);
                    si += 1;
                }
                already_decremented = true;
                // Don't advance: re-examine slot (may now hold next entry).
            } else {
                let score = cached_coefficients[current_topic] * current_count as f64;
                topic_term_mass += score;
                scored_positions[num_scored] = index;
                scored_values[num_scored] = score;
                num_scored += 1;
                index += 1;
            }
        }

        let total_mass = *smoothing_only_mass + topic_beta_mass + topic_term_mass;
        let mut sample = rng.gen::<f64>() * total_mass;

        let new_topic: usize;

        if sample < topic_term_mass {
            // Topic-term bucket: use scored_positions to jump directly.
            let mut i = 0;
            while sample > 0.0 && i < num_scored {
                sample -= scored_values[i];
                if sample > 0.0 {
                    i += 1;
                }
            }
            let array_idx = scored_positions[i.min(num_scored - 1)];
            let entry = type_counts[array_idx];
            let ct = (entry & topic_mask) as usize;
            let cv = entry >> topic_bits;
            new_topic = ct;

            type_counts[array_idx] = ((cv + 1) << topic_bits) | (ct as u32);
            let mut bi = array_idx;
            while bi > 0 && type_counts[bi] > type_counts[bi - 1] {
                type_counts.swap(bi, bi - 1);
                bi -= 1;
            }
        } else {
            sample -= topic_term_mass;

            if sample < topic_beta_mass {
                // Topic-beta bucket.
                sample /= beta;
                let mut chosen = local_topic_index[non_zero_topics - 1] as usize;
                for di in 0..non_zero_topics {
                    let t = local_topic_index[di] as usize;
                    sample -=
                        local_topic_counts[t] as f64 / (tokens_per_topic[t] as f64 + beta_sum);
                    if sample <= 0.0 {
                        chosen = t;
                        break;
                    }
                }
                new_topic = chosen;
            } else {
                // Smoothing-only bucket.
                sample -= topic_beta_mass;
                sample /= beta;
                let mut chosen = num_topics - 1;
                for t in 0..num_topics {
                    sample -= alpha[t] / (tokens_per_topic[t] as f64 + beta_sum);
                    if sample <= 0.0 {
                        chosen = t;
                        break;
                    }
                }
                new_topic = chosen;
            }

            // Find or create the slot for new_topic and increment.
            let mut idx = 0;
            while idx < entries_len && type_counts[idx] > 0 {
                if (type_counts[idx] & topic_mask) as usize == new_topic {
                    break;
                }
                idx += 1;
            }
            if idx == entries_len {
                // Overflow guard: pre-allocation in model::initialize should prevent this.
                type_counts.push(0);
            }
            let cv = type_counts[idx] >> topic_bits;
            type_counts[idx] = ((cv + 1) << topic_bits) | (new_topic as u32);
            let mut bi = idx;
            while bi > 0 && type_counts[bi] > type_counts[bi - 1] {
                type_counts.swap(bi, bi - 1);
                bi -= 1;
            }
        }
        // type_counts borrow ends here.

        // --- Phase C: assign new topic and update all counts ---
        doc_topics[doc_idx][pos] = new_topic as u32;

        *smoothing_only_mass -=
            alpha[new_topic] * beta / (tokens_per_topic[new_topic] as f64 + beta_sum);
        topic_beta_mass -= beta * local_topic_counts[new_topic] as f64
            / (tokens_per_topic[new_topic] as f64 + beta_sum);

        local_topic_counts[new_topic] += 1;

        if local_topic_counts[new_topic] == 1 {
            let mut di = non_zero_topics;
            while di > 0 && local_topic_index[di - 1] as usize > new_topic {
                local_topic_index[di] = local_topic_index[di - 1];
                di -= 1;
            }
            local_topic_index[di] = new_topic as u32;
            non_zero_topics += 1;
        }

        tokens_per_topic[new_topic] += 1;

        cached_coefficients[new_topic] = (alpha[new_topic] + local_topic_counts[new_topic] as f64)
            / (tokens_per_topic[new_topic] as f64 + beta_sum);

        *smoothing_only_mass +=
            alpha[new_topic] * beta / (tokens_per_topic[new_topic] as f64 + beta_sum);
        topic_beta_mass += beta * local_topic_counts[new_topic] as f64
            / (tokens_per_topic[new_topic] as f64 + beta_sum);
    }

    // Restore both inter-document invariants over only the touched topics:
    // cached_coefficients back to its smoothing-only value, and
    // local_topic_counts back to zero. At this point the document's tokens are
    // still distributed across exactly local_topic_index[0..non_zero_topics],
    // so zeroing those entries clears the whole array without an O(K) wipe.
    for di in 0..non_zero_topics {
        let t = local_topic_index[di] as usize;
        cached_coefficients[t] = alpha[t] / (tokens_per_topic[t] as f64 + beta_sum);
        local_topic_counts[t] = 0;
    }
}
