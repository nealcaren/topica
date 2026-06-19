/// Display trained topic model results in human-readable form.
/// Reads the TSV files written by `train` and prints formatted output.
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

fn print_usage() {
    eprintln!(
        "Usage: show [options]

Options:
  --topic-word <file>   Topic-word probability file (default: topic_word.tsv)
  --doc-topic <file>    Document-topic probability file (default: doc_topic.tsv)
  --words <n>           Words to show per topic (default: 10)
  --doc-topics <n>      Top topics to show per document (default: 0 = off)
  --threshold <f>       Only show document topics above this probability (default: 0.1)
"
    );
}

struct Args {
    topic_word: String,
    doc_topic: String,
    words: usize,
    doc_topics: usize,
    threshold: f64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            topic_word: "topic_word.tsv".to_string(),
            doc_topic: "doc_topic.tsv".to_string(),
            words: 10,
            doc_topics: 0,
            threshold: 0.1,
        }
    }
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--topic-word" => {
                i += 1;
                args.topic_word = raw[i].clone();
            }
            "--doc-topic" => {
                i += 1;
                args.doc_topic = raw[i].clone();
            }
            "--words" => {
                i += 1;
                args.words = raw[i].parse().ok()?;
            }
            "--doc-topics" => {
                i += 1;
                args.doc_topics = raw[i].parse().ok()?;
            }
            "--threshold" => {
                i += 1;
                args.threshold = raw[i].parse().ok()?;
            }
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

fn load_topic_word(path: &Path) -> io::Result<BTreeMap<usize, Vec<(String, f64)>>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut topics: BTreeMap<usize, Vec<(String, f64)>> = BTreeMap::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line_idx == 0 {
            continue;
        } // skip header
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let topic: usize = match cols[0].parse() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let word = cols[1].to_string();
        let prob: f64 = cols[2].parse().unwrap_or(0.0);
        topics.entry(topic).or_default().push((word, prob));
    }

    // Sort each topic's words by probability descending.
    for words in topics.values_mut() {
        words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    Ok(topics)
}

fn show_topics(topics: &BTreeMap<usize, Vec<(String, f64)>>, n_words: usize) {
    let num_topics = topics.len();
    let width = num_topics.to_string().len();

    for (&topic_idx, words) in topics {
        print!("Topic {:>width$}: ", topic_idx, width = width);
        let top: Vec<&str> = words
            .iter()
            .take(n_words)
            .map(|(w, _)| w.as_str())
            .collect();
        println!("{}", top.join("  "));
    }
}

fn show_doc_topics(path: &Path, n_topics: usize, threshold: f64) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header to find topic columns and whether a label column is present.
    let header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Ok(()),
    };
    let cols: Vec<&str> = header.split('\t').collect();

    // Find where topic columns start: after "doc" and optional "label".
    let first_topic_col = if cols.get(1).map(|&c| c == "label").unwrap_or(false) {
        2
    } else {
        1
    };
    let has_label = first_topic_col == 2;
    let n_topic_cols = cols.len() - first_topic_col;

    println!();
    println!("Document topics (threshold {:.0}%):", threshold * 100.0);
    println!();

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < first_topic_col + n_topic_cols {
            continue;
        }

        let doc_name = fields[0];
        let label = if has_label { Some(fields[1]) } else { None };

        // Collect (topic_idx, probability) pairs.
        let mut probs: Vec<(usize, f64)> = fields[first_topic_col..]
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                let p: f64 = v.parse().ok()?;
                Some((i, p))
            })
            .collect();

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = probs
            .iter()
            .take(n_topics)
            .filter(|(_, p)| *p >= threshold)
            .map(|(t, p)| format!("{} ({:.0}%)", t, p * 100.0))
            .collect();

        if top.is_empty() {
            continue;
        }

        match label {
            Some(lbl) => println!("{} [{}]:  {}", doc_name, lbl, top.join(", ")),
            None => println!("{}:  {}", doc_name, top.join(", ")),
        }
    }

    Ok(())
}

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            print_usage();
            std::process::exit(1);
        }
    };

    let tw_path = Path::new(&args.topic_word);
    if !tw_path.exists() {
        eprintln!(
            "Error: '{}' not found. Run `train` first, or specify a file with --topic-word.",
            args.topic_word
        );
        std::process::exit(1);
    }

    let topics = match load_topic_word(tw_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", args.topic_word, e);
            std::process::exit(1);
        }
    };

    show_topics(&topics, args.words);

    if args.doc_topics > 0 {
        let dt_path = Path::new(&args.doc_topic);
        if dt_path.exists() {
            if let Err(e) = show_doc_topics(dt_path, args.doc_topics, args.threshold) {
                eprintln!("Error reading {}: {}", args.doc_topic, e);
            }
        } else {
            eprintln!(
                "Note: '{}' not found; skipping document display. \
                 Use --doc-topic to specify a different path.",
                args.doc_topic
            );
        }
    }
}
