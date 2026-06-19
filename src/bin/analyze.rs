use std::collections::HashSet;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use topica::cli::{next_val, parse_val};
use topica::corpus;

fn print_usage() {
    eprintln!(
        "Usage: analyze --corpus <file> [options]

Options:
  --corpus <file>          Preprocessed binary corpus file
  --max-doc-fraction <f>   Flag words appearing in > this fraction of docs (default: 0.1)
  --max-word-length <n>    Flag words shorter than this length (default: 4)
  --min-doc-freq <n>       Flag words appearing in fewer than N docs (default: 2)
  --num-candidates <n>     Max candidates shown per heuristic (default: 50)
  --output-stoplist <file> Also write suggested stopwords to file

The analysis report goes to stderr; the suggested word list goes to stdout
(one word per line). This means:
  analyze --corpus corpus.corp > stop.txt   writes a ready-to-use stoplist
  analyze --corpus corpus.corp 2>/dev/null  prints only the word list
"
    );
}

struct Args {
    corpus: Option<String>,
    max_doc_fraction: f64,
    max_word_length: usize,
    min_doc_freq: u32,
    num_candidates: usize,
    output_stoplist: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            corpus: None,
            max_doc_fraction: 0.10,
            max_word_length: 4,
            min_doc_freq: 2,
            num_candidates: 50,
            output_stoplist: None,
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
            "--max-doc-fraction" => args.max_doc_fraction = parse_val(&raw, &mut i)?,
            "--max-word-length" => args.max_word_length = parse_val(&raw, &mut i)?,
            "--min-doc-freq" => args.min_doc_freq = parse_val(&raw, &mut i)?,
            "--num-candidates" => args.num_candidates = parse_val(&raw, &mut i)?,
            "--output-stoplist" => args.output_stoplist = Some(next_val(&raw, &mut i)?),
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

fn idf(doc_freq: u32, num_docs: usize) -> f64 {
    (num_docs as f64 / doc_freq as f64).ln()
}

fn is_numeric(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_digit())
}

fn is_mostly_non_alpha(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let alpha = word.chars().filter(|c| c.is_alphabetic()).count();
    alpha == 0 || (alpha as f64 / word.len() as f64) < 0.5
}

// All report helpers write to stderr so stdout stays clean for the word list.

fn report_table_header() {
    eprintln!(
        "{:<6}  {:<24}  {:>8}  {:>8}  {:>7}  {:>10}",
        "rank", "word", "doc_freq", "% docs", "idf", "total_freq"
    );
    eprintln!("{}", "-".repeat(70));
}

fn report_word_row(rank: usize, word: &str, df: u32, num_docs: usize, tf: u32) {
    let pct = df as f64 / num_docs as f64 * 100.0;
    let idf_val = idf(df, num_docs);
    eprintln!(
        "{:>6}  {:<24}  {:>8}  {:>7.2}%  {:>7.4}  {:>10}",
        rank, word, df, pct, idf_val, tf
    );
}

fn run_heuristic(
    label: &str,
    mut candidates: Vec<usize>,
    sort_key: impl Fn(usize) -> u32,
    id_to_word: &[String],
    doc_freqs: &[u32],
    total_freqs: &[u32],
    num_docs: usize,
    num_candidates: usize,
    all_suggested: &mut HashSet<String>,
) {
    eprintln!("=== {} ===", label);
    candidates.sort_by_key(|&id| std::cmp::Reverse(sort_key(id)));

    if candidates.is_empty() {
        eprintln!("  (none found)");
    } else {
        report_table_header();
        for (rank, &id) in candidates.iter().take(num_candidates).enumerate() {
            report_word_row(
                rank + 1,
                &id_to_word[id],
                doc_freqs[id],
                num_docs,
                total_freqs[id],
            );
            all_suggested.insert(id_to_word[id].clone());
        }
        // Words beyond num_candidates are still added to the suggested set.
        for &id in candidates.iter().skip(num_candidates) {
            all_suggested.insert(id_to_word[id].clone());
        }
        if candidates.len() > num_candidates {
            eprintln!(
                "  ... and {} more (all added to suggested list)",
                candidates.len() - num_candidates
            );
        }
    }
    eprintln!();
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

    let num_docs = c.num_docs();
    let num_types = c.num_types();
    let max_doc_thresh = (num_docs as f64 * args.max_doc_fraction).ceil() as u32;

    eprintln!("=== Corpus Statistics ===");
    eprintln!("Documents:    {}", num_docs);
    eprintln!("Vocabulary:   {} types", num_types);
    eprintln!("Total tokens: {}", c.total_tokens());
    eprintln!();

    let mut all_suggested: HashSet<String> = HashSet::new();

    // --- Heuristic 1: High document frequency ---
    eprintln!(
        "=== High Document Frequency (IDF < {:.3}) ===",
        idf(max_doc_thresh.max(1), num_docs)
    );
    eprintln!(
        "Words appearing in more than {:.1}% of documents ({} docs):",
        args.max_doc_fraction * 100.0,
        max_doc_thresh
    );
    eprintln!();
    run_heuristic(
        "High Document Frequency (continued)",
        (0..num_types)
            .filter(|&id| c.doc_freqs[id] > max_doc_thresh)
            .collect(),
        |id| c.doc_freqs[id],
        &c.id_to_word,
        &c.doc_freqs,
        &c.total_freqs,
        num_docs,
        args.num_candidates,
        &mut all_suggested,
    );

    // --- Heuristic 2: Short tokens ---
    run_heuristic(
        &format!("Short Tokens (length < {})", args.max_word_length),
        (0..num_types)
            .filter(|&id| {
                let w = &c.id_to_word[id];
                w.chars().count() < args.max_word_length && !is_numeric(w)
            })
            .collect(),
        |id| c.doc_freqs[id],
        &c.id_to_word,
        &c.doc_freqs,
        &c.total_freqs,
        num_docs,
        args.num_candidates,
        &mut all_suggested,
    );

    // --- Heuristic 3: Numeric tokens ---
    run_heuristic(
        "Numeric Tokens",
        (0..num_types)
            .filter(|&id| is_numeric(&c.id_to_word[id]))
            .collect(),
        |id| c.total_freqs[id],
        &c.id_to_word,
        &c.doc_freqs,
        &c.total_freqs,
        num_docs,
        args.num_candidates,
        &mut all_suggested,
    );

    // --- Heuristic 4: Non-alphabetic tokens ---
    run_heuristic(
        "Non-Alphabetic Tokens",
        (0..num_types)
            .filter(|&id| is_mostly_non_alpha(&c.id_to_word[id]))
            .collect(),
        |id| c.total_freqs[id],
        &c.id_to_word,
        &c.doc_freqs,
        &c.total_freqs,
        num_docs,
        args.num_candidates,
        &mut all_suggested,
    );

    // --- Heuristic 5: Rare words (report only, not added to suggested list) ---
    let rare_count = (0..num_types)
        .filter(|&id| c.doc_freqs[id] < args.min_doc_freq)
        .count();
    eprintln!(
        "=== Rare Words (fewer than {} documents) ===",
        args.min_doc_freq
    );
    eprintln!(
        "  {} words ({:.1}% of vocabulary) appear in fewer than {} documents.",
        rare_count,
        rare_count as f64 / num_types as f64 * 100.0,
        args.min_doc_freq
    );
    eprintln!(
        "  Consider removing with --min-doc-freq {} in preprocess.",
        args.min_doc_freq
    );
    eprintln!();

    // --- Emit the word list ---
    let mut words: Vec<&String> = all_suggested.iter().collect();
    words.sort();

    eprintln!("=== Summary: {} suggested stopwords ===", words.len());

    // stdout: one word per line, suitable for use directly with --stoplist
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for word in &words {
        writeln!(out, "{}", word).unwrap();
    }
    drop(out);

    // Optionally also write to a named file
    if let Some(ref out_path) = args.output_stoplist {
        let file = match fs::File::create(out_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error creating stoplist file: {}", e);
                std::process::exit(1);
            }
        };
        let mut writer = BufWriter::new(file);
        for word in &words {
            writeln!(writer, "{}", word).unwrap();
        }
        eprintln!("Wrote {} stopwords to {}", words.len(), out_path);
    }
}
