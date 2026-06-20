//! The `Corpus` pyclass: a preprocessed, integer-encoded document collection.
//!
//! Bound into the module as `py_corpus` (the file is `corpus.rs`; the module is
//! renamed to avoid clashing with the `crate::corpus` import). Model `fit`
//! methods accept a `Corpus` and read its `pub(crate) inner`.

use std::collections::HashSet;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::build_corpus_from_docs;
use super::error::io_err;
use crate::corpus::{self, InputFormat, LoadOptions};

/// A preprocessed, integer-encoded document collection.
///
/// Build one from already-tokenised documents with
/// :meth:`Corpus.from_documents`, from a raw text file with
/// :meth:`Corpus.from_text_file`, or load a binary corpus written by the
/// ``preprocess`` CLI with :meth:`Corpus.load`.
#[pyclass(module = "topica")]
pub struct Corpus {
    pub(crate) inner: corpus::Corpus,
    // Original document indices that survived pruning (parallel to the rows of
    // the corpus). Lets callers realign external covariate/metadata arrays.
    kept_indices: Vec<usize>,
    // Optional per-document metadata (e.g. a pandas DataFrame), already filtered
    // to the surviving rows. Round-tripped as a plain Python object.
    metadata: Option<PyObject>,
}

// Manual Clone: PyObject needs the GIL to bump its refcount, so it can't derive.
impl Clone for Corpus {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Corpus {
            inner: self.inner.clone(),
            kept_indices: self.kept_indices.clone(),
            metadata: self.metadata.as_ref().map(|m| m.clone_ref(py)),
        })
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
    /// tomotopy's `min_df` / `min_cf` / `rm_top`.
    ///
    /// A document left with no tokens by pruning is dropped, so `num_docs` can be
    /// smaller than `len(documents)`. The surviving original indices are in
    /// `kept_indices`; realign any external covariate matrix with
    /// `X[corpus.kept_indices]`. (An input document that is empty before any
    /// pruning is retained.)
    #[staticmethod]
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None,
                        stopwords=None, min_doc_freq=1, max_doc_fraction=1.0,
                        min_cf=0, rm_top=0))]
    #[allow(clippy::too_many_arguments)]
    fn from_documents(
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        min_cf: u32,
        rm_top: usize,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let (inner, kept_indices) = build_corpus_from_docs(
            documents,
            doc_names,
            doc_labels,
            stop,
            min_doc_freq,
            max_doc_fraction,
            min_cf,
            rm_top,
        )?;
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
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
        stopwords: Option<Vec<String>>,
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
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
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
        })
    }

    /// Load a binary corpus file written by the ``preprocess`` CLI or
    /// :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = corpus::load_corpus(Path::new(path)).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
        })
    }

    /// Write this corpus to a binary file (the ``preprocess`` format), so it
    /// can be reused by the CLI tools or reloaded with :meth:`load`.
    fn save(&self, path: &str) -> PyResult<()> {
        corpus::save_corpus(&self.inner, Path::new(path)).map_err(io_err)
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
    /// unset.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Option<PyObject> {
        self.metadata.as_ref().map(|m| m.clone_ref(py))
    }

    #[setter]
    fn set_metadata(&mut self, value: Option<PyObject>) {
        self.metadata = value;
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
