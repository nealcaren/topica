#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::ptr_arg
)]

use std::path::Path;
use std::time::Instant;
use topica::cli::{next_val, parse_val};
use topica::{corpus, model, optimize, output, sampler};

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn print_usage() {
    eprintln!(
        "Usage: train --corpus <file> [options]

Sampling:
  --corpus <file>             Preprocessed binary corpus file (from preprocess)
  --num-topics <k>            Number of topics (default: 10)
  --iterations <n>            Number of Gibbs iterations (default: 1000)
  --burn-in <n>               Iterations before hyperparameter optimization begins (default: 200)
  --seed <n>                  Random seed (default: 13)

Hyperparameter optimization:
  --optimize-interval <n>     Optimize alpha and beta every N iterations after burn-in
                              (default: 50; set 0 to disable)

Output estimation:
  --num-samples <n>           Samples to average for final estimates (default: 5)
  --sample-interval <n>       Gibbs iterations between samples (default: 25)

Priors:
  --alpha-sum <f>             Initial symmetric Dirichlet alpha sum (default: num_topics)
  --beta <f>                  Initial Dirichlet beta per word (default: 0.01)

Output:
  --topic-word <file>         Topic-word probabilities (default: topic_word.tsv)
  --doc-topic <file>          Document-topic probabilities (default: doc_topic.tsv)
  --show-topics-interval <n>  Print top words every N iterations (default: 50)
  --words-per-topic <n>       Words shown per topic in progress output (default: 7)
"
    );
}

struct Args {
    corpus: Option<String>,
    num_topics: usize,
    iterations: usize,
    burn_in: usize,
    optimize_interval: usize,
    num_samples: usize,
    sample_interval: usize,
    topic_word_output: String,
    doc_topic_output: String,
    alpha_sum: Option<f64>,
    beta: f64,
    seed: u64,
    show_topics_interval: usize,
    words_per_topic: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            corpus: None,
            num_topics: 10,
            iterations: 1000,
            burn_in: 200,
            optimize_interval: 50,
            num_samples: 5,
            sample_interval: 25,
            topic_word_output: "topic_word.tsv".to_string(),
            doc_topic_output: "doc_topic.tsv".to_string(),
            alpha_sum: None,
            beta: 0.01,
            seed: 13,
            show_topics_interval: 50,
            words_per_topic: 7,
        }
    }
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        return None;
    }
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--corpus" => args.corpus = Some(next_val(&raw, &mut i)?),
            "--num-topics" => args.num_topics = parse_val(&raw, &mut i)?,
            "--iterations" => args.iterations = parse_val(&raw, &mut i)?,
            "--burn-in" => args.burn_in = parse_val(&raw, &mut i)?,
            "--optimize-interval" => args.optimize_interval = parse_val(&raw, &mut i)?,
            "--num-samples" => args.num_samples = parse_val(&raw, &mut i)?,
            "--sample-interval" => args.sample_interval = parse_val(&raw, &mut i)?,
            "--topic-word" => args.topic_word_output = next_val(&raw, &mut i)?,
            "--doc-topic" => args.doc_topic_output = next_val(&raw, &mut i)?,
            "--alpha-sum" => args.alpha_sum = Some(parse_val(&raw, &mut i)?),
            "--beta" => args.beta = parse_val(&raw, &mut i)?,
            "--seed" => args.seed = parse_val(&raw, &mut i)?,
            "--show-topics-interval" => args.show_topics_interval = parse_val(&raw, &mut i)?,
            "--words-per-topic" => args.words_per_topic = parse_val(&raw, &mut i)?,
            "--help" | "-h" => return None,
            other => {
                eprintln!("Unknown argument: {}", other);
                return None;
            }
        }
        i += 1;
    }
    Some(args)
}

fn format_elapsed(total_secs: u64) -> String {
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let mut s = String::new();
    if days > 0 {
        s.push_str(&format!("{} days ", days));
    }
    if hours > 0 {
        s.push_str(&format!("{} hours ", hours));
    }
    if minutes > 0 {
        s.push_str(&format!("{} minutes ", minutes));
    }
    s.push_str(&format!("{} seconds", seconds));
    s
}

/// Snapshot the current smoothed topic-word distribution into acc_phi.
fn accumulate_phi(m: &model::TopicModel, acc: &mut Vec<Vec<f64>>) {
    for word_id in 0..m.num_types {
        for topic in 0..m.num_topics {
            let count = m.get_type_topic_count(word_id, topic);
            let denom = m.tokens_per_topic[topic] as f64 + m.beta_sum;
            acc[word_id][topic] += (count as f64 + m.beta) / denom;
        }
    }
}

/// Snapshot the current smoothed document-topic distribution into acc_theta.
fn accumulate_theta(m: &model::TopicModel, c: &corpus::Corpus, acc: &mut Vec<Vec<f64>>) {
    let mut counts = vec![0u32; m.num_topics];
    for doc_idx in 0..c.num_docs() {
        for t in 0..m.num_topics {
            counts[t] = 0;
        }
        for &t in &m.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }

        let doc_len = c.docs[doc_idx].len() as f64;
        let denom = doc_len + m.alpha_sum;
        for t in 0..m.num_topics {
            acc[doc_idx][t] += (counts[t] as f64 + m.alpha[t]) / denom;
        }
    }
}

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            print_usage();
            std::process::exit(1);
        }
    };

    let corpus_path = match &args.corpus {
        Some(p) => p.clone(),
        None => {
            eprintln!("Error: --corpus is required");
            print_usage();
            std::process::exit(1);
        }
    };

    let c = match corpus::load_corpus(Path::new(&corpus_path)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading corpus: {}", e);
            std::process::exit(1);
        }
    };

    if c.num_docs() == 0 {
        eprintln!("Error: corpus contains no documents");
        std::process::exit(1);
    }

    let alpha_sum = args.alpha_sum.unwrap_or(args.num_topics as f64);
    let mut m = model::TopicModel::new(args.num_topics, alpha_sum, args.beta, c.num_types());

    eprintln!(
        "Mallet LDA: {} topics, {} topic bits, {:b} topic mask",
        m.num_topics, m.topic_bits, m.topic_mask
    );
    eprintln!(
        "max tokens: {}",
        c.docs.iter().map(|d| d.len()).max().unwrap_or(0)
    );
    eprintln!("total tokens: {}", c.total_tokens());
    if args.optimize_interval > 0 {
        eprintln!(
            "Hyperparameter optimization every {} iterations after burn-in ({})",
            args.optimize_interval, args.burn_in
        );
    }

    let mut rng = Pcg64Mcg::seed_from_u64(args.seed);
    m.initialize(&c, &mut rng);

    let total_tokens = c.total_tokens();
    let train_start = Instant::now();

    // -----------------------------------------------------------------------
    // Main training loop
    // -----------------------------------------------------------------------
    for iter in 1..=args.iterations {
        sampler::run_iteration(&mut m, &c, &mut rng);

        // Hyperparameter optimization after burn-in
        if args.optimize_interval > 0 && iter > args.burn_in && iter % args.optimize_interval == 0 {
            optimize::optimize_alpha(&mut m, &c);
            optimize::optimize_beta(&mut m);
            eprintln!("[O] alpha_sum={:.5}  beta={:.5}", m.alpha_sum, m.beta);
        }

        if args.show_topics_interval > 0 && iter % args.show_topics_interval == 0 {
            eprint!(
                "\n{}",
                output::display_top_words(&m, &c, args.words_per_topic)
            );
        }

        if iter % 10 == 0 {
            let ll = output::model_log_likelihood(&m, &c);
            eprintln!("<{}> LL/token: {:.5}", iter, ll / total_tokens as f64);
        }
    }

    eprintln!(
        "\nTotal time: {}",
        format_elapsed(train_start.elapsed().as_secs())
    );

    // -----------------------------------------------------------------------
    // Sampling phase: collect num_samples samples separated by sample_interval
    // iterations each, then average for final distribution estimates.
    // -----------------------------------------------------------------------
    let num_samples = args.num_samples;
    let sample_interval = args.sample_interval;

    eprintln!(
        "\nCollecting {} samples ({} iterations apart)...",
        num_samples, sample_interval
    );

    let mut acc_phi = vec![vec![0.0f64; m.num_topics]; m.num_types];
    let mut acc_theta = vec![vec![0.0f64; m.num_topics]; c.num_docs()];

    for s in 0..num_samples {
        for _ in 0..sample_interval {
            sampler::run_iteration(&mut m, &c, &mut rng);
        }
        accumulate_phi(&m, &mut acc_phi);
        accumulate_theta(&m, &c, &mut acc_theta);
        eprintln!("  sample {}/{}", s + 1, num_samples);
    }

    // Normalise by sample count
    let n = num_samples as f64;
    for row in acc_phi.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    for row in acc_theta.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }

    // -----------------------------------------------------------------------
    // Write output
    // -----------------------------------------------------------------------
    eprintln!(
        "Writing topic-word probabilities to: {}",
        args.topic_word_output
    );
    if let Err(e) =
        output::write_topic_word_matrix(&acc_phi, &c, Path::new(&args.topic_word_output))
    {
        eprintln!("Error writing topic-word file: {}", e);
    }

    eprintln!(
        "Writing document-topic probabilities to: {}",
        args.doc_topic_output
    );
    if let Err(e) =
        output::write_doc_topic_matrix(&acc_theta, &c, Path::new(&args.doc_topic_output))
    {
        eprintln!("Error writing doc-topic file: {}", e);
    }
}
