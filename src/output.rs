use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::corpus::Corpus;
use crate::model::TopicModel;

const HALF_LOG_TWO_PI: f64 = 0.9189385332046727;

/// Stirling-series approximation of log Gamma, matching MALLET's Dirichlet.logGammaStirling.
fn log_gamma(mut z: f64) -> f64 {
    let mut shift = 0i32;
    while z < 2.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

/// Compute model log-likelihood (same formula as ParallelTopicModel.modelLogLikelihood).
pub fn model_log_likelihood(model: &TopicModel, corpus: &Corpus) -> f64 {
    let mut ll: f64 = 0.0;

    // Pre-compute logGamma(alpha[t]) for each topic.
    let topic_log_gammas: Vec<f64> = (0..model.num_topics)
        .map(|t| log_gamma(model.alpha[t]))
        .collect();

    // Document contribution.
    let mut topic_counts = vec![0u32; model.num_topics];
    for doc_idx in 0..corpus.num_docs() {
        for t in 0..model.num_topics {
            topic_counts[t] = 0;
        }
        for &t in &model.doc_topics[doc_idx] {
            topic_counts[t as usize] += 1;
        }
        let doc_len = corpus.docs[doc_idx].len();
        for t in 0..model.num_topics {
            if topic_counts[t] > 0 {
                ll += log_gamma(model.alpha[t] + topic_counts[t] as f64) - topic_log_gammas[t];
            }
        }
        ll -= log_gamma(model.alpha_sum + doc_len as f64);
    }
    ll += corpus.num_docs() as f64 * log_gamma(model.alpha_sum);

    // Topic-word contribution.
    let mut non_zero_type_topics: u64 = 0;
    for word_id in 0..model.num_types {
        let counts = &model.type_topic_counts[word_id];
        let mut idx = 0;
        while idx < counts.len() && counts[idx] > 0 {
            let count = counts[idx] >> model.topic_bits;
            non_zero_type_topics += 1;
            ll += log_gamma(model.beta + count as f64);
            idx += 1;
        }
    }
    for t in 0..model.num_topics {
        ll -= log_gamma(model.beta_sum + model.tokens_per_topic[t] as f64);
    }
    ll += model.num_topics as f64 * log_gamma(model.beta_sum);
    ll -= log_gamma(model.beta) * non_zero_type_topics as f64;

    ll
}

/// Display top words per topic in MALLET's inline format:
/// `topic_idx\talpha\tword1 word2 ...`
pub fn display_top_words(model: &TopicModel, corpus: &Corpus, n: usize) -> String {
    let mut out = String::new();
    for topic in 0..model.num_topics {
        let mut word_scores: Vec<(u32, usize)> = (0..model.num_types)
            .filter_map(|word_id| {
                let count = model.get_type_topic_count(word_id, topic);
                if count > 0 {
                    Some((count, word_id))
                } else {
                    None
                }
            })
            .collect();
        word_scores.sort_by(|a, b| b.0.cmp(&a.0));

        out.push_str(&format!("{}\t{:.5}\t", topic, model.alpha[topic]));
        for (i, (_, id)) in word_scores.iter().take(n).enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&corpus.id_to_word[*id]);
        }
        out.push('\n');
    }
    out
}

/// Write topic-word probabilities: one row per (topic, word) pair.
/// Format: topic \t word \t probability
/// φ_{w,t} = (n_{w,t} + β) / (n_t + β·W)
pub fn write_topic_word(model: &TopicModel, corpus: &Corpus, path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "topic\tword\tprobability")?;

    for topic in 0..model.num_topics {
        let denominator = model.tokens_per_topic[topic] as f64 + model.beta_sum;

        for word_id in 0..model.num_types {
            let count = model.get_type_topic_count(word_id, topic);
            let prob = (count as f64 + model.beta) / denominator;
            writeln!(
                writer,
                "{}\t{}\t{:.8}",
                topic, corpus.id_to_word[word_id], prob
            )?;
        }
    }

    Ok(())
}

/// Write document-topic probabilities: one row per document.
/// Format: doc_name [\t label] \t p(topic_0) \t p(topic_1) \t ...
/// θ_{t,d} = (n_{t,d} + α_t) / (N_d + α_sum)
pub fn write_doc_topic(model: &TopicModel, corpus: &Corpus, path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    let has_labels = corpus.has_labels();

    // Header
    write!(writer, "doc")?;
    if has_labels {
        write!(writer, "\tlabel")?;
    }
    for t in 0..model.num_topics {
        write!(writer, "\ttopic_{}", t)?;
    }
    writeln!(writer)?;

    for doc_idx in 0..corpus.num_docs() {
        let doc_len = corpus.docs[doc_idx].len() as f64;
        let denominator = doc_len + model.alpha_sum;

        // Count topic assignments in this document.
        let mut topic_counts = vec![0u32; model.num_topics];
        for &t in &model.doc_topics[doc_idx] {
            topic_counts[t as usize] += 1;
        }

        write!(writer, "{}", corpus.doc_names[doc_idx])?;
        if has_labels {
            write!(writer, "\t{}", corpus.doc_labels[doc_idx])?;
        }
        for t in 0..model.num_topics {
            let prob = (topic_counts[t] as f64 + model.alpha[t]) / denominator;
            write!(writer, "\t{:.8}", prob)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Write topic-word probabilities from a pre-averaged matrix.
/// `phi[word_id][topic]` = averaged probability estimate.
pub fn write_topic_word_matrix(phi: &[Vec<f64>], corpus: &Corpus, path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_topics = phi.first().map(|r| r.len()).unwrap_or(0);

    writeln!(writer, "topic\tword\tprobability")?;
    for topic in 0..num_topics {
        for word_id in 0..corpus.num_types() {
            writeln!(
                writer,
                "{}\t{}\t{:.8}",
                topic, corpus.id_to_word[word_id], phi[word_id][topic]
            )?;
        }
    }
    Ok(())
}

/// Write document-topic probabilities from a pre-averaged matrix.
/// `theta[doc_idx][topic]` = averaged probability estimate.
pub fn write_doc_topic_matrix(theta: &[Vec<f64>], corpus: &Corpus, path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_topics = theta.first().map(|r| r.len()).unwrap_or(0);
    let has_labels = corpus.has_labels();

    write!(writer, "doc")?;
    if has_labels {
        write!(writer, "\tlabel")?;
    }
    for t in 0..num_topics {
        write!(writer, "\ttopic_{}", t)?;
    }
    writeln!(writer)?;

    for doc_idx in 0..corpus.num_docs() {
        write!(writer, "{}", corpus.doc_names[doc_idx])?;
        if has_labels {
            write!(writer, "\t{}", corpus.doc_labels[doc_idx])?;
        }
        for t in 0..num_topics {
            write!(writer, "\t{:.8}", theta[doc_idx][t])?;
        }
        writeln!(writer)?;
    }
    Ok(())
}
