use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, BufWriter, Read, Write};
use std::path::Path;

use regex::Regex;

/// Default token pattern: starts and ends with a Unicode letter, minimum length 2.
/// Interior characters may be letters or a small set of non-breaking punctuation:
///   -  U+002D  hyphen-minus   (compound words across many languages)
///   '  U+0027  apostrophe     (English contractions, French elision, etc.)
///   '  U+2019  right single quote / typographic apostrophe (same role as U+0027)
///   .  U+002E  full stop      (abbreviations: U.S.A, e.g.)
///   ·  U+00B7  middle dot     (Catalan col·legi, Welsh, other scripts)
///
/// Em-dash (U+2014), en-dash (U+2013), and all other punctuation break tokens.
pub const DEFAULT_TOKEN_REGEX: &str = r"\p{L}[-'\u{2019}.\u{00B7}\p{L}]*\p{L}";

// Version 2 adds per-document labels.
const MAGIC: &[u8; 4] = b"CRP2";

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Corpus {
    pub id_to_word: Vec<String>,
    pub docs: Vec<Vec<u32>>,
    pub doc_names: Vec<String>,
    /// Per-document label strings; empty string when no label was provided.
    pub doc_labels: Vec<String>,
    /// Number of documents each word type appears in (document frequency).
    pub doc_freqs: Vec<u32>,
    /// Total occurrences of each word type across the whole corpus.
    pub total_freqs: Vec<u32>,
}

impl Corpus {
    pub fn num_types(&self) -> usize {
        self.id_to_word.len()
    }
    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }
    pub fn total_tokens(&self) -> usize {
        self.docs.iter().map(|d| d.len()).sum()
    }

    /// True when at least one document has a non-empty label.
    pub fn has_labels(&self) -> bool {
        self.doc_labels.iter().any(|l| !l.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Input format
// ---------------------------------------------------------------------------

pub enum InputFormat {
    /// One document per line, whitespace-tokenised.
    /// If `id_field` is true the first whitespace token is the document name.
    Plain { id_field: bool },

    /// Tab-delimited columns (e.g. MALLET's id TAB label TAB text layout).
    Tsv {
        id_column: usize,
        label_column: Option<usize>,
        text_column: usize,
    },
}

impl Default for InputFormat {
    fn default() -> Self {
        InputFormat::Plain { id_field: false }
    }
}

pub struct LoadOptions {
    pub format: InputFormat,
    /// Regex pattern used to extract tokens from text.
    pub token_regex: String,
    /// Words in this set are dropped during tokenisation.
    pub stopwords: HashSet<String>,
    /// Drop words appearing in fewer than this many documents.
    pub min_doc_freq: u32,
    /// Drop words appearing in more than this fraction of documents (0.0–1.0).
    pub max_doc_fraction: f64,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            format: InputFormat::default(),
            token_regex: DEFAULT_TOKEN_REGEX.to_string(),
            stopwords: HashSet::new(),
            min_doc_freq: 1,
            max_doc_fraction: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Text loading
// ---------------------------------------------------------------------------

pub fn load_text_file(path: &Path, opts: &LoadOptions) -> io::Result<Corpus> {
    let re = Regex::new(&opts.token_regex)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: Vec<String> = Vec::new();
    let mut docs: Vec<Vec<u32>> = Vec::new();
    let mut doc_names: Vec<String> = Vec::new();
    let mut doc_labels: Vec<String> = Vec::new();
    let mut total_freqs: Vec<u32> = Vec::new();
    let mut per_doc_type_sets: Vec<HashSet<usize>> = Vec::new();

    let mut skipped = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let (doc_name, doc_label, text_slice): (String, String, &str) = match &opts.format {
            InputFormat::Plain { id_field } => {
                if *id_field {
                    // First whitespace token is the name; rest is text.
                    match line.find(|c: char| c.is_whitespace()) {
                        Some(pos) => (line[..pos].to_string(), String::new(), line[pos..].trim()),
                        None => (format!("doc_{}", line_idx), String::new(), line),
                    }
                } else {
                    (format!("doc_{}", line_idx), String::new(), line)
                }
            }

            InputFormat::Tsv {
                id_column,
                label_column,
                text_column,
            } => {
                let cols: Vec<&str> = line
                    .splitn(
                        // Only need to split up to max column + 1; splitn remainder holds the rest.
                        // For safety just collect all tabs and index into the vec.
                        usize::MAX,
                        '\t',
                    )
                    .collect();

                let max_needed = [*id_column, *text_column]
                    .iter()
                    .chain(label_column.iter())
                    .copied()
                    .max()
                    .unwrap_or(0);

                if cols.len() <= max_needed {
                    skipped += 1;
                    continue;
                }

                let name = cols[*id_column].trim().to_string();
                let label = label_column
                    .map(|c| cols[c].trim().to_string())
                    .unwrap_or_default();
                let text = cols[*text_column];
                (name, label, text)
            }
        };

        let mut token_ids: Vec<u32> = Vec::new();
        let mut seen_in_doc: HashSet<usize> = HashSet::new();

        for m in re.find_iter(text_slice) {
            let token_lower = m.as_str().to_lowercase();
            if opts.stopwords.contains(&token_lower) {
                continue;
            }

            let id = if let Some(&eid) = vocab.get(&token_lower) {
                eid
            } else {
                let new_id = id_to_word.len();
                vocab.insert(token_lower.clone(), new_id);
                id_to_word.push(token_lower);
                total_freqs.push(0);
                new_id
            };

            total_freqs[id] += 1;
            token_ids.push(id as u32);
            seen_in_doc.insert(id);
        }

        if !token_ids.is_empty() {
            doc_names.push(doc_name);
            doc_labels.push(doc_label);
            docs.push(token_ids);
            per_doc_type_sets.push(seen_in_doc);
        }
    }

    if skipped > 0 {
        eprintln!("Warning: skipped {} lines with too few columns", skipped);
    }

    let num_types = id_to_word.len();
    let num_docs = docs.len();

    // Accumulate document frequencies.
    let mut doc_freqs = vec![0u32; num_types];
    for set in &per_doc_type_sets {
        for &id in set {
            doc_freqs[id] += 1;
        }
    }

    // Apply frequency filters.
    let max_df = (num_docs as f64 * opts.max_doc_fraction).ceil() as u32;
    let keep: Vec<bool> = (0..num_types)
        .map(|id| doc_freqs[id] >= opts.min_doc_freq && doc_freqs[id] <= max_df)
        .collect();

    if keep.iter().all(|&k| k) {
        return Ok(Corpus {
            id_to_word,
            docs,
            doc_names,
            doc_labels,
            doc_freqs,
            total_freqs,
        });
    }

    // Remap vocabulary.
    let mut remap: Vec<Option<usize>> = vec![None; num_types];
    let mut new_id_to_word: Vec<String> = Vec::new();
    let mut new_doc_freqs: Vec<u32> = Vec::new();
    let mut new_total_freqs: Vec<u32> = Vec::new();

    for id in 0..num_types {
        if keep[id] {
            remap[id] = Some(new_id_to_word.len());
            new_id_to_word.push(id_to_word[id].clone());
            new_doc_freqs.push(doc_freqs[id]);
            new_total_freqs.push(total_freqs[id]);
        }
    }

    let new_docs: Vec<Vec<u32>> = docs
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .filter_map(|id| remap[id as usize].map(|r| r as u32))
                .collect()
        })
        .collect();

    // Drop documents emptied by pruning, keeping labels aligned.
    let mut final_docs: Vec<Vec<u32>> = Vec::new();
    let mut final_names: Vec<String> = Vec::new();
    let mut final_labels: Vec<String> = Vec::new();

    for ((doc, name), label) in new_docs.into_iter().zip(doc_names).zip(doc_labels) {
        if !doc.is_empty() {
            final_docs.push(doc);
            final_names.push(name);
            final_labels.push(label);
        }
    }

    Ok(Corpus {
        id_to_word: new_id_to_word,
        docs: final_docs,
        doc_names: final_names,
        doc_labels: final_labels,
        doc_freqs: new_doc_freqs,
        total_freqs: new_total_freqs,
    })
}

// ---------------------------------------------------------------------------
// In-memory text loading
// ---------------------------------------------------------------------------

/// Build a [`Corpus`] from in-memory documents (one `String` per document),
/// applying the same tokenisation, stopword removal, and document-frequency
/// filtering as [`load_text_file`]. This is the in-process counterpart used by
/// embedding hosts (e.g. the Stata plugin) that already hold the text in memory.
///
/// `names` and `labels`, when `Some`, are indexed positionally against `texts`;
/// entries past their end fall back to `doc_<i>` / empty. `opts.format` is
/// ignored (each element of `texts` is already one document's raw text).
///
/// Documents left empty after tokenisation and pruning are dropped, with
/// `doc_names` / `doc_labels` kept aligned. A caller can therefore pass row
/// indices as `names` and read them back from `doc_names` to learn which input
/// rows survived (the rest produced no usable tokens).
pub fn from_texts(
    texts: &[String],
    names: Option<&[String]>,
    labels: Option<&[String]>,
    opts: &LoadOptions,
) -> io::Result<Corpus> {
    let re = Regex::new(&opts.token_regex)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: Vec<String> = Vec::new();
    let mut docs: Vec<Vec<u32>> = Vec::new();
    let mut doc_names: Vec<String> = Vec::new();
    let mut doc_labels: Vec<String> = Vec::new();
    let mut total_freqs: Vec<u32> = Vec::new();
    let mut per_doc_type_sets: Vec<HashSet<usize>> = Vec::new();

    for (i, text) in texts.iter().enumerate() {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut seen_in_doc: HashSet<usize> = HashSet::new();

        for m in re.find_iter(text) {
            let token_lower = m.as_str().to_lowercase();
            if opts.stopwords.contains(&token_lower) {
                continue;
            }
            let id = if let Some(&eid) = vocab.get(&token_lower) {
                eid
            } else {
                let new_id = id_to_word.len();
                vocab.insert(token_lower.clone(), new_id);
                id_to_word.push(token_lower);
                total_freqs.push(0);
                new_id
            };
            total_freqs[id] += 1;
            token_ids.push(id as u32);
            seen_in_doc.insert(id);
        }

        if !token_ids.is_empty() {
            let name = names
                .and_then(|n| n.get(i))
                .cloned()
                .unwrap_or_else(|| format!("doc_{}", i));
            let label = labels.and_then(|l| l.get(i)).cloned().unwrap_or_default();
            doc_names.push(name);
            doc_labels.push(label);
            docs.push(token_ids);
            per_doc_type_sets.push(seen_in_doc);
        }
    }

    Ok(finalize_corpus(
        id_to_word,
        docs,
        doc_names,
        doc_labels,
        total_freqs,
        per_doc_type_sets,
        opts,
    ))
}

/// Apply document-frequency filtering and drop emptied documents, producing the
/// final [`Corpus`]. Shared tail used by [`from_texts`].
fn finalize_corpus(
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
    doc_names: Vec<String>,
    doc_labels: Vec<String>,
    total_freqs: Vec<u32>,
    per_doc_type_sets: Vec<HashSet<usize>>,
    opts: &LoadOptions,
) -> Corpus {
    let num_types = id_to_word.len();
    let num_docs = docs.len();

    let mut doc_freqs = vec![0u32; num_types];
    for set in &per_doc_type_sets {
        for &id in set {
            doc_freqs[id] += 1;
        }
    }

    let max_df = (num_docs as f64 * opts.max_doc_fraction).ceil() as u32;
    let keep: Vec<bool> = (0..num_types)
        .map(|id| doc_freqs[id] >= opts.min_doc_freq && doc_freqs[id] <= max_df)
        .collect();

    if keep.iter().all(|&k| k) {
        return Corpus {
            id_to_word,
            docs,
            doc_names,
            doc_labels,
            doc_freqs,
            total_freqs,
        };
    }

    let mut remap: Vec<Option<usize>> = vec![None; num_types];
    let mut new_id_to_word: Vec<String> = Vec::new();
    let mut new_doc_freqs: Vec<u32> = Vec::new();
    let mut new_total_freqs: Vec<u32> = Vec::new();
    for id in 0..num_types {
        if keep[id] {
            remap[id] = Some(new_id_to_word.len());
            new_id_to_word.push(id_to_word[id].clone());
            new_doc_freqs.push(doc_freqs[id]);
            new_total_freqs.push(total_freqs[id]);
        }
    }

    let new_docs: Vec<Vec<u32>> = docs
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .filter_map(|id| remap[id as usize].map(|r| r as u32))
                .collect()
        })
        .collect();

    let mut final_docs: Vec<Vec<u32>> = Vec::new();
    let mut final_names: Vec<String> = Vec::new();
    let mut final_labels: Vec<String> = Vec::new();
    for ((doc, name), label) in new_docs.into_iter().zip(doc_names).zip(doc_labels) {
        if !doc.is_empty() {
            final_docs.push(doc);
            final_names.push(name);
            final_labels.push(label);
        }
    }

    Corpus {
        id_to_word: new_id_to_word,
        docs: final_docs,
        doc_names: final_names,
        doc_labels: final_labels,
        doc_freqs: new_doc_freqs,
        total_freqs: new_total_freqs,
    }
}

// ---------------------------------------------------------------------------
// Binary serialisation  (magic "CRP2")
// Header:   4 magic | u32 num_types | u32 num_docs
// Vocab:    per type: str word | u32 doc_freq | u32 total_freq
// Docs:     per doc:  str name | str label | u32 num_tokens | u32×n tokens
// ---------------------------------------------------------------------------

pub fn save_corpus(corpus: &Corpus, path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(MAGIC)?;
    write_u32(&mut w, corpus.num_types() as u32)?;
    write_u32(&mut w, corpus.num_docs() as u32)?;

    for id in 0..corpus.num_types() {
        write_str(&mut w, &corpus.id_to_word[id])?;
        write_u32(&mut w, corpus.doc_freqs[id])?;
        write_u32(&mut w, corpus.total_freqs[id])?;
    }

    for doc_idx in 0..corpus.num_docs() {
        write_str(&mut w, &corpus.doc_names[doc_idx])?;
        write_str(&mut w, &corpus.doc_labels[doc_idx])?;
        let tokens = &corpus.docs[doc_idx];
        write_u32(&mut w, tokens.len() as u32)?;
        for &id in tokens {
            write_u32(&mut w, id)?;
        }
    }

    Ok(())
}

pub fn load_corpus(path: &Path) -> io::Result<Corpus> {
    let mut f = io::BufReader::new(fs::File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unrecognised corpus format (got {:?}, expected {:?}). \
                 Re-run preprocess to regenerate the corpus file.",
                std::str::from_utf8(&magic).unwrap_or("?"),
                std::str::from_utf8(MAGIC).unwrap_or("?")
            ),
        ));
    }

    let num_types = read_u32(&mut f)? as usize;
    let num_docs = read_u32(&mut f)? as usize;

    let mut id_to_word = Vec::with_capacity(num_types);
    let mut doc_freqs = Vec::with_capacity(num_types);
    let mut total_freqs = Vec::with_capacity(num_types);

    for _ in 0..num_types {
        id_to_word.push(read_str(&mut f)?);
        doc_freqs.push(read_u32(&mut f)?);
        total_freqs.push(read_u32(&mut f)?);
    }

    let mut docs = Vec::with_capacity(num_docs);
    let mut doc_names = Vec::with_capacity(num_docs);
    let mut doc_labels = Vec::with_capacity(num_docs);

    for _ in 0..num_docs {
        doc_names.push(read_str(&mut f)?);
        doc_labels.push(read_str(&mut f)?);
        let n = read_u32(&mut f)? as usize;
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            tokens.push(read_u32(&mut f)?);
        }
        docs.push(tokens);
    }

    Ok(Corpus {
        id_to_word,
        docs,
        doc_names,
        doc_labels,
        doc_freqs,
        total_freqs,
    })
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u16).to_le_bytes())?;
    w.write_all(bytes)
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_str(r: &mut impl Read) -> io::Result<String> {
    let mut lbuf = [0u8; 2];
    r.read_exact(&mut lbuf)?;
    let len = u16::from_le_bytes(lbuf) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// Stoplist loading
// ---------------------------------------------------------------------------

pub fn load_stoplist(path: &Path) -> io::Result<HashSet<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut words = HashSet::new();
    for line in reader.lines() {
        let w = line?.trim().to_lowercase();
        if !w.is_empty() && !w.starts_with('#') {
            words.insert(w);
        }
    }
    Ok(words)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod from_texts_tests {
    use super::*;

    #[test]
    fn tokenises_and_filters_in_memory() {
        let texts = vec![
            "the cat sat on the mat".to_string(),
            "a dog and a cat".to_string(),
            "".to_string(),              // empty -> dropped
            "!!! 1 2 3 ???".to_string(), // no word tokens -> dropped
        ];
        // Pass row indices as names so we can recover which rows survived.
        let names: Vec<String> = (0..texts.len()).map(|i| i.to_string()).collect();
        let opts = LoadOptions::default();

        let c = from_texts(&texts, Some(&names), None, &opts).unwrap();

        assert_eq!(c.num_docs(), 2, "two non-empty docs survive");
        assert_eq!(c.doc_names, vec!["0".to_string(), "1".to_string()]);
        assert!(c.id_to_word.contains(&"cat".to_string()));
        // "cat" appears in both docs -> doc_freq 2.
        let cat = c.id_to_word.iter().position(|w| w == "cat").unwrap();
        assert_eq!(c.doc_freqs[cat], 2);
        // tokens are valid ids and round-trip to words.
        for doc in &c.docs {
            for &t in doc {
                assert!((t as usize) < c.num_types());
            }
        }
    }

    #[test]
    fn min_doc_freq_prunes_singletons() {
        let texts = vec![
            "alpha beta gamma".to_string(),
            "alpha beta".to_string(),
            "alpha".to_string(),
        ];
        let mut opts = LoadOptions::default();
        opts.min_doc_freq = 2; // drop words in <2 docs (gamma appears once)
        let c = from_texts(&texts, None, None, &opts).unwrap();
        assert!(c.id_to_word.contains(&"alpha".to_string()));
        assert!(c.id_to_word.contains(&"beta".to_string()));
        assert!(!c.id_to_word.contains(&"gamma".to_string()), "gamma pruned");
    }
}
