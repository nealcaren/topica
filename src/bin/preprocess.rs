use std::path::Path;
use topica::corpus::{self, InputFormat, LoadOptions, DEFAULT_TOKEN_REGEX};

fn print_usage() {
    eprintln!(
        r"Usage: preprocess --input <file> --output <file> [options]

Input format (default: plain text, one document per line):
  --format tsv             Tab-delimited columns (default columns: 0=id, 1=label, 2=text)
  --id-column <n>          Column index for document name in TSV mode (default: 0)
  --label-column <n>       Column index for document label in TSV mode (default: 1)
  --no-label               Disable label column in TSV mode
  --text-column <n>        Column index for document text in TSV mode (default: 2)
  --id-field               Plain mode: first whitespace token is the document name
  --token-regex <pattern>  Regex for token extraction
                           (default: \p{{L}}[-'\u{{2019}}.\u{{00B7}}\p{{L}}]*\p{{L}})

Filtering:
  --stoplist <file>        File with one stopword per line (lines starting with # ignored)
  --min-doc-freq <n>       Drop words appearing in fewer than N documents (default: 1)
  --max-doc-fraction <f>   Drop words in more than this fraction of docs (default: 1.0)

Output:
  --input <file>           Raw text input file
  --output <file>          Binary corpus output file
"
    );
}

struct Args {
    input: Option<String>,
    output: Option<String>,
    stoplist: Option<String>,
    // Format selection
    tsv_mode: bool,
    id_field: bool, // plain-mode: first whitespace token is name
    // TSV column indices
    id_column: usize,
    label_column: Option<usize>, // None = no label column
    text_column: usize,
    // Tokenisation
    token_regex: String,
    // Frequency filters
    min_doc_freq: u32,
    max_doc_fraction: f64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            input: None,
            output: None,
            stoplist: None,
            tsv_mode: false,
            id_field: false,
            id_column: 0,
            label_column: Some(1),
            text_column: 2,
            token_regex: DEFAULT_TOKEN_REGEX.to_string(),
            min_doc_freq: 1,
            max_doc_fraction: 1.0,
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
            "--input" => {
                i += 1;
                args.input = Some(raw[i].clone());
            }
            "--output" => {
                i += 1;
                args.output = Some(raw[i].clone());
            }
            "--stoplist" => {
                i += 1;
                args.stoplist = Some(raw[i].clone());
            }
            "--format" => {
                i += 1;
                match raw[i].as_str() {
                    "tsv" => args.tsv_mode = true,
                    "plain" => args.tsv_mode = false,
                    other => {
                        eprintln!("Unknown format: {} (use 'plain' or 'tsv')", other);
                        return None;
                    }
                }
            }
            "--id-field" => {
                args.id_field = true;
            }
            "--id-column" => {
                i += 1;
                args.id_column = raw[i].parse().ok()?;
            }
            "--label-column" => {
                i += 1;
                args.label_column = Some(raw[i].parse().ok()?);
            }
            "--no-label" => {
                args.label_column = None;
            }
            "--text-column" => {
                i += 1;
                args.text_column = raw[i].parse().ok()?;
            }
            "--token-regex" => {
                i += 1;
                args.token_regex = raw[i].clone();
            }
            "--min-doc-freq" => {
                i += 1;
                args.min_doc_freq = raw[i].parse().ok()?;
            }
            "--max-doc-fraction" => {
                i += 1;
                args.max_doc_fraction = raw[i].parse().ok()?;
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

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            print_usage();
            std::process::exit(1);
        }
    };

    let input_path = match &args.input {
        Some(p) => p.clone(),
        None => {
            eprintln!("Error: --input is required");
            print_usage();
            std::process::exit(1);
        }
    };
    let output_path = match &args.output {
        Some(p) => p.clone(),
        None => {
            eprintln!("Error: --output is required");
            print_usage();
            std::process::exit(1);
        }
    };

    let format = if args.tsv_mode {
        eprintln!(
            "Format: TSV  id_col={}  {}  text_col={}",
            args.id_column,
            args.label_column
                .map(|c| format!("label_col={}", c))
                .unwrap_or_else(|| "(no label)".to_string()),
            args.text_column
        );
        InputFormat::Tsv {
            id_column: args.id_column,
            label_column: args.label_column,
            text_column: args.text_column,
        }
    } else {
        InputFormat::Plain {
            id_field: args.id_field,
        }
    };

    eprintln!("Token regex: {}", args.token_regex);

    let mut opts = LoadOptions {
        format,
        token_regex: args.token_regex.clone(),
        min_doc_freq: args.min_doc_freq,
        max_doc_fraction: args.max_doc_fraction,
        ..Default::default()
    };

    if let Some(ref sl_path) = args.stoplist {
        match corpus::load_stoplist(Path::new(sl_path)) {
            Ok(words) => {
                eprintln!("Loaded {} stopwords from {}", words.len(), sl_path);
                opts.stopwords = words;
            }
            Err(e) => {
                eprintln!("Error reading stoplist: {}", e);
                std::process::exit(1);
            }
        }
    }

    eprintln!("Reading: {}", input_path);
    let c = match corpus::load_text_file(Path::new(&input_path), &opts) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Corpus: {} documents, {} word types, {} tokens{}",
        c.num_docs(),
        c.num_types(),
        c.total_tokens(),
        if c.has_labels() {
            ", labels present"
        } else {
            ""
        }
    );

    if c.num_docs() == 0 {
        eprintln!("Error: no documents after preprocessing");
        std::process::exit(1);
    }

    eprintln!("Writing: {}", output_path);
    if let Err(e) = corpus::save_corpus(&c, Path::new(&output_path)) {
        eprintln!("Error writing corpus: {}", e);
        std::process::exit(1);
    }
}
