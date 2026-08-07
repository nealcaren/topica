//! The `Corpus` pyclass: a preprocessed, integer-encoded document collection.
//!
//! Bound into the module as `py_corpus` (the file is `corpus.rs`; the module is
//! renamed to avoid clashing with the `crate::corpus` import). Model `fit`
//! methods accept a `Corpus` and read its `pub(crate) inner`.

use std::collections::HashSet;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::build_corpus_from_docs_ext;
use super::error::io_err;
use crate::corpus::{self, InputFormat, LoadOptions};

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
        stopwords: Option<Vec<String>>,
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
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
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

    /// Build a corpus directly from a document x feature count matrix in
    /// compressed sparse-row (CSR) form.
    ///
    /// This is the low-level constructor behind :func:`topica.from_feature_matrix`;
    /// most callers should use that helper, which accepts a dense NumPy array or a
    /// SciPy sparse matrix and converts it to the CSR arrays this method needs.
    ///
    /// `feature_names` is the ordered vocabulary, one name per matrix column.
    /// `indptr` (length ``num_docs + 1``), `indices`, and `data` are the standard
    /// CSR arrays: for document ``d`` the stored columns are
    /// ``indices[indptr[d]..indptr[d+1]]`` with counts ``data[...]``. Counts are
    /// non-negative integers and are expanded once, in Rust, into the token-stream
    /// representation the samplers consume — so a dense feature dimension never
    /// inflates memory on the Python side (topica issue #575). No vocabulary
    /// pruning is applied; filter columns before building.
    #[staticmethod]
    #[pyo3(signature = (feature_names, indptr, indices, data, *,
                        doc_names=None, doc_labels=None))]
    fn from_feature_matrix(
        feature_names: Vec<String>,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        data: Vec<u32>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if indptr.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have at least one element (num_docs + 1)",
            ));
        }
        if indices.len() != data.len() {
            return Err(PyValueError::new_err(format!(
                "indices ({}) and data ({}) must have the same length",
                indices.len(),
                data.len()
            )));
        }
        let nnz = data.len();
        let num_docs = indptr.len() - 1;
        let mut rows: Vec<Vec<(u32, u32)>> = Vec::with_capacity(num_docs);
        for d in 0..num_docs {
            let (start, end) = (indptr[d], indptr[d + 1]);
            if start > end || end > nnz {
                return Err(PyValueError::new_err(format!(
                    "indptr is not a valid, monotonically non-decreasing CSR pointer at document {d}"
                )));
            }
            let mut row: Vec<(u32, u32)> = Vec::with_capacity(end - start);
            for k in start..end {
                row.push((indices[k], data[k]));
            }
            rows.push(row);
        }

        let inner = corpus::Corpus::from_counts(feature_names, rows, doc_names, doc_labels)
            .map_err(PyValueError::new_err)?;
        let num_docs = inner.num_docs();
        Ok(Corpus {
            inner,
            kept_indices: (0..num_docs).collect(),
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
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = corpus::load_corpus(Path::new(path)).map_err(io_err)?;
        let kept_indices = (0..inner.num_docs()).collect();
        Ok(Corpus {
            inner,
            kept_indices,
            metadata: None,
            // Not persisted in the corpus save format; unknown after load.
            preprocessing: None,
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
    /// unset.
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
