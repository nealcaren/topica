use crate::corpus::Corpus;
use crate::model::TopicModel;

/// Digamma function via recurrence + asymptotic expansion.
/// Matches the accuracy of MALLET's Dirichlet.logGammaStirling derivative.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence: ψ(x) = ψ(x+1) - 1/x  →  shift x into the asymptotic region
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic: ln x - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// One Minka fixed-point step for a symmetric Dirichlet concentration parameter.
///
/// Both alpha (document-topic) and beta (topic-word) optimisation use this
/// same update.  The caller passes histograms rather than raw counts so the
/// inner loop is over distinct count values, not over every observation.
///
/// * `count_hist[c]`  – number of (component, observation) pairs with count c
/// * `length_hist[n]` – number of observations with total n
/// * `num_dims`       – number of Dirichlet components (K or W)
/// * `concentration`  – current value of α_sum or β_sum
///
/// Returns the updated concentration, or the original value if the update
/// would be degenerate.
fn update_symmetric_concentration(
    count_hist: &[u32],
    length_hist: &[u32],
    num_dims: usize,
    concentration: f64,
) -> f64 {
    let per_dim = concentration / num_dims as f64;
    let dg_per_dim = digamma(per_dim);
    let dg_conc = digamma(concentration);

    let numerator: f64 = count_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + per_dim) - dg_per_dim))
        .sum();

    let denominator: f64 = length_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + concentration) - dg_conc))
        .sum::<f64>()
        * num_dims as f64;

    if numerator > 0.0 && denominator > 0.0 {
        (concentration * numerator / denominator).max(1e-10)
    } else {
        concentration
    }
}

/// Optimise per-topic alpha values (asymmetric Dirichlet) using one
/// Minka fixed-point step.
///
/// Sufficient statistics:
///   doc_length_hist[n]        – number of documents with n tokens
///   topic_doc_hist[t][c]      – number of documents where topic t appears c times
pub fn optimize_alpha(model: &mut TopicModel, corpus: &Corpus) {
    let max_len = corpus.docs.iter().map(|d| d.len()).max().unwrap_or(0);

    let mut doc_length_hist = vec![0u32; max_len + 1];
    let mut topic_doc_hist = vec![vec![0u32; max_len + 1]; model.num_topics];

    for doc_idx in 0..corpus.num_docs() {
        let doc_len = corpus.docs[doc_idx].len();
        doc_length_hist[doc_len] += 1;

        let mut counts = vec![0u32; model.num_topics];
        for &t in &model.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        for t in 0..model.num_topics {
            if counts[t] > 0 {
                topic_doc_hist[t][counts[t] as usize] += 1;
            }
            counts[t] = 0;
        }
    }

    // Shared denominator: depends only on document lengths and current alpha_sum.
    let dg_alpha_sum = digamma(model.alpha_sum);
    let denominator: f64 = doc_length_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + model.alpha_sum) - dg_alpha_sum))
        .sum();

    if denominator <= 0.0 {
        return;
    }

    let mut new_alpha_sum = 0.0;
    for t in 0..model.num_topics {
        let dg_alpha_t = digamma(model.alpha[t]);
        let numerator: f64 = topic_doc_hist[t]
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, &c)| c > 0)
            .map(|(c, &cnt)| cnt as f64 * (digamma(c as f64 + model.alpha[t]) - dg_alpha_t))
            .sum();

        model.alpha[t] = (model.alpha[t] * numerator / denominator).max(1e-10);
        new_alpha_sum += model.alpha[t];
    }
    model.alpha_sum = new_alpha_sum;
}

/// Optimise a *symmetric* document-topic prior: one Minka step on the shared
/// alpha concentration, keeping every `alpha[t]` equal. This is MALLET's
/// `--use-symmetric-alpha true` path — only the total alpha_sum is learned, not
/// the per-topic shape (which `optimize_alpha` learns).
pub fn optimize_alpha_symmetric(model: &mut TopicModel, corpus: &Corpus) {
    let max_len = corpus.docs.iter().map(|d| d.len()).max().unwrap_or(0);
    let mut doc_length_hist = vec![0u32; max_len + 1];
    // Aggregated over all topics: number of (topic, doc) pairs with count c.
    let mut count_hist = vec![0u32; max_len + 1];
    let mut counts = vec![0u32; model.num_topics];

    for doc_idx in 0..corpus.num_docs() {
        doc_length_hist[corpus.docs[doc_idx].len()] += 1;
        for &t in &model.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        for c in counts.iter_mut() {
            if *c > 0 {
                count_hist[*c as usize] += 1;
                *c = 0;
            }
        }
    }

    let new_sum = update_symmetric_concentration(
        &count_hist,
        &doc_length_hist,
        model.num_topics,
        model.alpha_sum,
    );
    model.alpha_sum = new_sum;
    let per_topic = new_sum / model.num_topics as f64;
    for a in model.alpha.iter_mut() {
        *a = per_topic;
    }
}

/// Optimise the symmetric beta (topic-word prior) using one Minka step.
///
/// Sufficient statistics:
///   count_hist[c]       – number of (word, topic) pairs with c tokens
///   topic_size_hist[s]  – number of topics with s total tokens
pub fn optimize_beta(model: &mut TopicModel) {
    // Build count histogram over non-zero (word, topic) entries.
    let max_count = model
        .type_topic_counts
        .iter()
        .flat_map(|v| v.iter().take_while(|&&e| e > 0))
        .map(|&e| e >> model.topic_bits)
        .max()
        .unwrap_or(0) as usize;

    let mut count_hist = vec![0u32; max_count + 1];
    for word_id in 0..model.num_types {
        for &entry in model.type_topic_counts[word_id]
            .iter()
            .take_while(|&&e| e > 0)
        {
            count_hist[(entry >> model.topic_bits) as usize] += 1;
        }
    }

    // Build topic-size histogram.
    let max_size = model.tokens_per_topic.iter().copied().max().unwrap_or(0) as usize;
    let mut topic_size_hist = vec![0u32; max_size + 1];
    for t in 0..model.num_topics {
        topic_size_hist[model.tokens_per_topic[t] as usize] += 1;
    }

    let new_beta_sum = update_symmetric_concentration(
        &count_hist,
        &topic_size_hist,
        model.num_types,
        model.beta_sum,
    );

    model.beta = new_beta_sum / model.num_types as f64;
    model.beta_sum = new_beta_sum;
}
