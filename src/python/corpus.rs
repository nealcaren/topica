//! The `Corpus` pyclass: a preprocessed, integer-encoded document collection.
//!
//! Bound into the module as `py_corpus` (the file is `corpus.rs`; the module is
//! renamed to avoid clashing with the `crate::corpus` import). Model `fit`
//! methods accept a `Corpus` and read its `pub(crate) inner`.

use std::collections::HashSet;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use super::build_corpus_from_docs_ext;
use super::error::io_err;
use crate::corpus::{self, InputFormat, LoadOptions};

/// Collect an optional stopword argument into a set, accepting any iterable of
/// strings (list, tuple, set, frozenset) so the bundled `ENGLISH_STOPWORDS`
/// frozenset composes directly with the corpus builders (issue #742), the same
/// way `tokenize` already accepts it.
fn stopwords_set(stopwords: Option<&Bound<'_, PyAny>>) -> PyResult<HashSet<String>> {
    match stopwords {
        Some(obj) => {
            let mut s = HashSet::new();
            for item in obj.iter()? {
                s.insert(item?.extract::<String>()?);
            }
            Ok(s)
        }
        None => Ok(HashSet::new()),
    }
}

/// The vocabulary-filtering parameters Topica applied when building a corpus.
/// Recorded so a fitted corpus can report how it was preprocessed (issue #399).
/// `None` on a corpus loaded from disk, where the parameters were not persisted.
#[derive(Clone)]
pub(crate) struct PrepInfo {
    pub min_doc_freq: u32,
    pub max_doc_fraction: f64,
    pub min_cf: u32,
    pub rm_top: usize,
    /// Cap on the vocabulary size (top-N by frequency); `None` when unlimited.
    pub max_features: Option<usize>,
    /// True when the corpus was built against a fixed, user-supplied vocabulary
    /// (or produced by :meth:`Corpus.transform`), in which case the frequency
    /// filters above were not applied.
    pub vocabulary: bool,
}

/// A preprocessed, integer-encoded document collection.
///
/// Build one from already-tokenised documents with
/// :meth:`Corpus.from_documents`, from a raw text file with
/// :meth:`Corpus.from_text_file`, or load a binary corpus written by the
/// ``preprocess`` CLI with :meth:`Corpus.load`. Starting from a pandas
/// ``DataFrame``? Use the module-level :func:`topica.from_dataframe` (it builds
/// the ``Corpus`` and keeps your metadata row-aligned through pruning) — there is
/// no ``Corpus.from_dataframe``; the DataFrame on-ramp is a module function.
///
/// Accessor convention: scalar/array *facts about the corpus* are attribute
/// **properties** — access them with no parentheses (``corpus.num_docs``,
/// ``corpus.num_words``, ``corpus.vocabulary``, ``corpus.word_counts``,
/// ``corpus.doc_lengths``, ``corpus.kept_indices``, ``corpus.metadata``).
/// Operations that *do work or produce a new object* are **methods** — call
/// them with parentheses (``corpus.documents()``, ``corpus.transform(...)``,
/// ``corpus.save(path)``). So ``corpus.num_docs`` is an int, while
/// ``corpus.num_docs()`` raises ``TypeError: 'int' object is not callable``.
#[pyclass(module = "topica")]
pub struct Corpus {
    pub(crate) inner: corpus::Corpus,
    // Original document indices that survived pruning (parallel to the rows of
    // the corpus). Lets callers realign external covariate/metadata arrays.
    kept_indices: Vec<usize>,
    // Optional per-document metadata (e.g. a pandas DataFrame), already filtered
    // to the surviving rows. Round-tripped as a plain Python object.
    metadata: Option<PyObject>,
    // The preprocessing parameters Topica applied, when known (None after load).
    preprocessing: Option<PrepInfo>,
}

// Manual Clone: PyObject needs the GIL to bump its refcount, so it can't derive.
impl Clone for Corpus {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Corpus {
            inner: self.inner.clone(),
            kept_indices: self.kept_indices.clone(),
            metadata: self.metadata.as_ref().map(|m| m.clone_ref(py)),
            preprocessing: self.preprocessing.clone(),
        })
    }
}

// Metadata is embedded in the SAME corpus file as a trailer appended after the
// corpus body, so there is nothing to orphan when the file is moved or copied.
// The `CRP2` corpus reader (topica-core::load_corpus) stops exactly at the end of
// the corpus body and ignores trailing bytes, so a plain CLI-`preprocess` corpus
// (no trailer) and every model `fit` still read the file unchanged; only
// `Corpus.load` looks for the trailer and reattaches the covariates.
//
// Trailer layout, read from EOF backwards:
//   [.. corpus body ..][pickle bytes][u64-LE pickle_len][8-byte FOOTER magic]
// A file whose last 8 bytes are not the magic simply has no metadata (None).
const META_FOOTER: &[u8; 8] = b"TPCAMET1";

/// Emit a Python ``UserWarning`` from Rust, swallowing any failure to warn.
fn warn(py: Python<'_>, message: &str) {
    if let Ok(warnings) = py.import_bound("warnings") {
        let _ = warnings.call_method1("warn", (message,));
    }
}

/// Append `metadata`, pickled, as a trailer on the corpus file at `path`.
/// With no metadata, leaves the file as a plain corpus (no trailer). Never fails
/// the save: metadata that cannot be pickled warns and is dropped, and the file
/// stays a valid plain corpus.
fn append_metadata_trailer(
    py: Python<'_>,
    path: &str,
    metadata: Option<&PyObject>,
) -> PyResult<()> {
    let Some(meta) = metadata else {
        return Ok(()); // save_corpus already wrote a fresh, trailer-free file
    };
    let pickled = || -> PyResult<Vec<u8>> {
        let pickle = py.import_bound("pickle")?;
        pickle.call_method1("dumps", (meta,))?.extract::<Vec<u8>>()
    };
    let bytes = match pickled() {
        Ok(b) => b,
        Err(err) => {
            warn(
                py,
                &format!(
                    "corpus.metadata could not be pickled, so it was not saved \
                     with the corpus and will be missing after load: {err}"
                ),
            );
            return Ok(());
        }
    };
    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(io_err)?;
    f.write_all(&bytes).map_err(io_err)?;
    f.write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(io_err)?;
    f.write_all(META_FOOTER).map_err(io_err)?;
    Ok(())
}

/// Read a metadata trailer back off the corpus file at `path`, if one is present.
/// A file with no trailer returns ``None`` silently (a plain / CLI corpus); a
/// present-but-corrupt trailer warns and returns ``None``.
fn read_metadata_trailer(py: Python<'_>, path: &str) -> Option<PyObject> {
    let read_tail = || -> std::io::Result<Option<Vec<u8>>> {
        let mut f = fs::File::open(path)?;
        let len = f.seek(SeekFrom::End(0))?;
        if len < 16 {
            return Ok(None);
        }
        let mut footer = [0u8; 8];
        f.seek(SeekFrom::End(-8))?;
        f.read_exact(&mut footer)?;
        if &footer != META_FOOTER {
            return Ok(None); // plain corpus, no metadata
        }
        let mut len_buf = [0u8; 8];
        f.seek(SeekFrom::End(-16))?;
        f.read_exact(&mut len_buf)?;
        let plen = u64::from_le_bytes(len_buf);
        if plen + 16 > len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "metadata trailer length exceeds file size",
            ));
        }
        f.seek(SeekFrom::End(-16 - plen as i64))?;
        let mut buf = vec![0u8; plen as usize];
        f.read_exact(&mut buf)?;
        Ok(Some(buf))
    };
    let bytes = match read_tail() {
        Ok(Some(b)) => b,
        Ok(None) => return None,
        Err(err) => {
            warn(
                py,
                &format!(
                    "corpus {path:?} has a metadata trailer that could not be read; \
                     corpus.metadata is None: {err}"
                ),
            );
            return None;
        }
    };
    let unpickled = || -> PyResult<PyObject> {
        let pickle = py.import_bound("pickle")?;
        let obj = pickle.call_method1("loads", (PyBytes::new_bound(py, &bytes),))?;
        Ok(obj.unbind())
    };
    match unpickled() {
        Ok(obj) => Some(obj),
        Err(err) => {
            warn(
                py,
                &format!(
                    "corpus {path:?} metadata trailer could not be unpickled and was \
                     skipped; corpus.metadata is None: {err}"
                ),
            );
            None
        }
    }
}

#[pymethods]
impl Corpus {
    /// Build a corpus from pre-tokenised documents.
    ///
    /// `documents` is a sequence of token lists. Optional `doc_names` /
    /// `doc_labels` (each the same length as `documents`) attach an id and a
    /// label to every document. `stopwords` are dropped. Vocabulary is pruned by
    /// `min_doc_freq` (minimum document frequency) and `max_doc_fraction`
    /// (maximum fraction of documents), by `min_cf` (minimum collection/total
    /// frequency), and by `rm_top` (drop the N most frequent words) — matching
    /// tomotopy's `min_df` / `min_cf` / `rm_top`. `max_features` then caps the
    /// vocabulary to the N most frequent surviving word types (scikit-learn's
    /// `CountVectorizer(max_features=)`); `None` leaves it unbounded.
    ///
    /// `vocabulary` pins the vocabulary to a fixed, ordered term list
    /// (scikit-learn's `vocabulary=`): tokens are mapped to those columns,
    /// out-of-vocabulary tokens are dropped, and the frequency filters above are
    /// not applied, so passing `vocabulary` together with any of them is an error.
    /// To vectorize new documents against an *existing* corpus's vocabulary
    /// (held-out data), prefer :meth:`Corpus.transform`.
    ///
    /// A document left with no tokens by pruning is dropped, so `num_docs` can be
    /// smaller than `len(documents)`. The surviving original indices are in
    /// `kept_indices`; realign any external covariate matrix with
    /// `X[corpus.kept_indices]`. (Without a fixed `vocabulary`, an input document
    /// that is empty before any pruning is retained.)
    #[staticmethod]
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None,
                        stopwords=None, min_doc_freq=1, max_doc_fraction=1.0,
                        min_cf=0, rm_top=0, max_features=None, vocabulary=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_documents(
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
        stopwords: Option<&Bound<'_, PyAny>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        min_cf: u32,
        rm_top: usize,
        max_features: Option<usize>,
        vocabulary: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if let Some(mf) = max_features {
            if mf == 0 {
                return Err(PyValueError::new_err(
                    "max_features must be a positive integer",
                ));
            }
        }
        if vocabulary.is_some()
            && (min_doc_freq > 1
                || max_doc_fraction != 1.0
                || min_cf > 0
                || rm_top > 0
                || max_features.is_some())
        {
            return Err(PyValueError::new_err(
                "vocabulary= pins a fixed vocabulary and cannot be combined with \
                 frequency pruning (a non-default min_doc_freq, max_doc_fraction, \
                 min_cf, rm_top, or max_features); drop those arguments or prune first \
                 and pass the resulting corpus.vocabulary",
            ));
        }
        let used_fixed_vocab = vocabulary.is_some();
        let stop: HashSet<String> = stopwords_set(stopwords)?;
        let (inner, kept_indices) = build_corpus_from_docs_ext(
            documents,
            doc_names,
            doc_labels,
            stop,
            min_doc_freq,
            max_doc_fraction,
            min_cf,
            rm_top,
            max_features,
            vocabulary,
        )?;
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
            preprocessing: Some(PrepInfo {
                min_doc_freq,
                max_doc_fraction,
                min_cf,
                rm_top,
                max_features,
                vocabulary: used_fixed_vocab,
            }),
        })
    }

    /// Vectorize new documents against this corpus's vocabulary.
    ///
    /// The returned corpus shares this one's vocabulary exactly (same terms, same
    /// order, same ids, at full width; terms absent from `documents` remain as
    /// zero-frequency columns), so a model fitted on this corpus keeps its
    /// `topic_word` columns aligned to the result. Out-of-vocabulary tokens are
    /// dropped; `doc_freqs` / `word_counts` are recomputed over `documents`. This
    /// is scikit-learn's `vectorizer.transform` / gensim's `doc2bow` on held-out
    /// text.
    ///
    /// A document with no in-vocabulary tokens is dropped (as elsewhere in
    /// topica); `kept_indices` on the result gives the surviving `documents`
    /// indices so external labels/covariates can be realigned. `metadata` is not
    /// carried over. Raises if no document retains any in-vocabulary token.
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None))]
    fn transform(
        &self,
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let (inner, kept_indices) = build_corpus_from_docs_ext(
            documents,
            doc_names,
            doc_labels,
            HashSet::new(),
            1,
            1.0,
            0,
            0,
            None,
            Some(self.inner.id_to_word.clone()),
        )?;
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
            preprocessing: Some(PrepInfo {
                min_doc_freq: 1,
                max_doc_fraction: 1.0,
                min_cf: 0,
                rm_top: 0,
                max_features: None,
                vocabulary: true,
            }),
        })
    }

    /// Load and tokenise a raw text file (MALLET-style), matching the
    /// ``preprocess`` CLI.
    ///
    /// `format` is ``"plain"`` (one document per line) or ``"tsv"``. In plain
    /// mode, `id_field=True` treats the first whitespace token as the doc id.
    /// In tsv mode, `id_column`/`label_column`/`text_column` select columns
    /// (`label_column=None` disables labels).
    #[staticmethod]
    #[pyo3(signature = (path, *, format="plain", id_field=false,
                        id_column=0, label_column=1, text_column=2,
                        token_regex=None, stopwords=None,
                        min_doc_freq=1, max_doc_fraction=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn from_text_file(
        path: &str,
        format: &str,
        id_field: bool,
        id_column: usize,
        label_column: Option<usize>,
        text_column: usize,
        token_regex: Option<String>,
        stopwords: Option<&Bound<'_, PyAny>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
    ) -> PyResult<Self> {
        let fmt = match format {
            "plain" => InputFormat::Plain { id_field },
            "tsv" => InputFormat::Tsv {
                id_column,
                label_column,
                text_column,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown format {:?} (use 'plain' or 'tsv')",
                    other
                )))
            }
        };
        let stop: HashSet<String> = stopwords_set(stopwords)?;
        let opts = LoadOptions {
            format: fmt,
            token_regex: token_regex.unwrap_or_else(|| corpus::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
            lowercase: true,
        };
        let inner = corpus::load_text_file(Path::new(path), &opts).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
            preprocessing: Some(PrepInfo {
                min_doc_freq,
                max_doc_fraction,
                min_cf: 0,
                rm_top: 0,
                max_features: None,
                vocabulary: false,
            }),
        })
    }

    /// Load a binary corpus file written by the ``preprocess`` CLI or
    /// :meth:`save`.
    ///
    /// If :meth:`save` embedded :attr:`metadata` in the file (because the corpus
    /// carried it), it is read back and reattached — it travels inside the one
    /// file, so moving/copying the corpus keeps its covariates. A corrupt
    /// metadata trailer warns and is skipped rather than failing the load.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = corpus::load_corpus(Path::new(path)).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: read_metadata_trailer(py, path),
            // Not persisted in the corpus save format; unknown after load.
            preprocessing: None,
        })
    }

    /// Write this corpus to a binary file (the ``preprocess`` format), so it
    /// can be reused by the CLI tools or reloaded with :meth:`load`.
    ///
    /// When the corpus carries :attr:`metadata` (e.g. from
    /// :func:`topica.from_dataframe`), it is pickled into a trailer on the *same*
    /// file, so :meth:`load` reattaches it and there is nothing to lose when the
    /// file is moved. The trailer is invisible to the CLI tools and model ``fit``
    /// (they read the corpus body and ignore it). Metadata that cannot be pickled
    /// warns and is dropped rather than failing the save.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        corpus::save_corpus(&self.inner, Path::new(path)).map_err(io_err)?;
        append_metadata_trailer(py, path, self.metadata.as_ref())
    }

    #[getter]
    fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    #[getter]
    fn num_words(&self) -> usize {
        self.inner.num_types()
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    /// Tokens per document in the pruned vocabulary, one entry per kept document
    /// (parallel to the rows of a fitted model's ``doc_topic``). This is the
    /// document length ``N_d`` that :func:`topica.dirichlet_theta_samples` needs to
    /// recover each document's Dirichlet posterior for method-of-composition
    /// standard errors.
    #[getter]
    fn doc_lengths(&self) -> Vec<usize> {
        self.inner.docs.iter().map(|d| d.len()).collect()
    }

    #[getter]
    fn vocabulary(&self) -> Vec<String> {
        self.inner.id_to_word.clone()
    }

    /// Corpus word frequencies: total occurrences of each vocabulary term across
    /// all documents, parallel to :attr:`vocabulary` (length ``num_words``). This
    /// is the empirical ``P(w)`` (up to normalization) that stm's lift and FREX
    /// James-Stein shrinkage use; pass it (or the corpus) to
    /// :func:`topica.label_topics` / :func:`topica.frex` for stm-faithful labels.
    #[getter]
    fn word_counts(&self) -> Vec<u32> {
        self.inner.total_freqs.clone()
    }

    /// The corpus as token lists — one list of word strings per document, in the
    /// pruned vocabulary and the kept-document order. The inverse of
    /// ``from_documents``: use it to recover tokens for ``prepare_pyldavis``,
    /// ``coherence``, or any function that wants ``list[list[str]]`` after you have
    /// committed to a ``Corpus``.
    fn documents(&self) -> Vec<Vec<String>> {
        self.inner
            .docs
            .iter()
            .map(|d| {
                d.iter()
                    .map(|&w| self.inner.id_to_word[w as usize].clone())
                    .collect()
            })
            .collect()
    }

    /// Original document indices that survived pruning, parallel to the rows of
    /// this corpus. Use it to realign an external covariate array or DataFrame
    /// to the documents the corpus actually kept: ``X = X[corpus.kept_indices]``.
    #[getter]
    fn kept_indices(&self) -> Vec<usize> {
        self.kept_indices.clone()
    }

    /// Optional per-document metadata, already aligned to the surviving rows
    /// (set by :func:`topica.from_dataframe`, or assign your own). ``None`` if
    /// unset. Persisted across :meth:`save`/:meth:`load` inside the corpus file
    /// itself, so a prune-once, save, reuse-across-models workflow keeps its
    /// covariates with nothing to lose when the file is moved.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Option<PyObject> {
        self.metadata.as_ref().map(|m| m.clone_ref(py))
    }

    #[setter]
    fn set_metadata(&mut self, value: Option<PyObject>) {
        self.metadata = value;
    }

    /// The vocabulary-filtering parameters Topica applied when this corpus was
    /// built (``min_doc_freq``, ``max_doc_fraction``, ``min_cf``, ``rm_top``), as
    /// a dict. ``None`` for a corpus loaded from disk, where they are not stored.
    #[getter]
    fn preprocessing<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.preprocessing {
            None => Ok(None),
            Some(p) => {
                let d = PyDict::new_bound(py);
                d.set_item("min_doc_freq", p.min_doc_freq)?;
                d.set_item("max_doc_fraction", p.max_doc_fraction)?;
                d.set_item("min_cf", p.min_cf)?;
                d.set_item("rm_top", p.rm_top)?;
                d.set_item("max_features", p.max_features)?;
                d.set_item("vocabulary", p.vocabulary)?;
                Ok(Some(d))
            }
        }
    }

    #[getter]
    fn doc_names(&self) -> Vec<String> {
        self.inner.doc_names.clone()
    }

    #[getter]
    fn doc_labels(&self) -> Vec<String> {
        self.inner.doc_labels.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Corpus(num_docs={}, num_words={}, total_tokens={})",
            self.inner.num_docs(),
            self.inner.num_types(),
            self.inner.total_tokens()
        )
    }
}
