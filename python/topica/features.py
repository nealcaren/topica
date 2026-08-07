"""Sparse feature-count matrices as first-class topica corpora.

Most topica models start from text, which :func:`topica.from_dataframe` and
:meth:`topica.Corpus.from_documents` turn into a bag-of-words count matrix. Some
inputs are *already* a document x feature count matrix — most notably the
sparse-autoencoder (SAE) feature activations of a Mechanistic Topic Model
(Zheng et al. 2025; topica issue #575), but also concept counts, dictionary hits,
or any other bag-of-features representation. :func:`from_feature_matrix` builds a
:class:`~topica.Corpus` directly from such a matrix, so any count-based topic
model can consume it exactly as it would a bag-of-words corpus.

The heavy, model-specific step of *producing* the feature matrix (running an LLM
and a pretrained SAE over the documents) stays outside the core, the same way
``topica.llm_embed`` sits outside the embedding-cluster models: you bring the
features, topica models them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._topica import Corpus

__all__ = ["from_feature_matrix"]

# CSR counts are stored as u32 in the Rust core.
_MAX_COUNT = 2**32 - 1


def _to_csr(counts):
    """Return (num_docs, num_features, indptr, indices, data) as plain Python
    lists of non-negative ``int`` counts, from a dense array, a nested sequence,
    or any SciPy sparse matrix — without importing SciPy when it is not needed."""
    # SciPy sparse: detected structurally so SciPy stays an optional dependency.
    if hasattr(counts, "tocsr") and hasattr(counts, "shape"):
        m = counts.tocsr()
        m.sum_duplicates()
        m.sort_indices()
        num_docs, num_features = m.shape
        data = m.data
        if not np.all(np.isfinite(data)):
            raise ValueError("feature counts must be finite")
        if np.any(data < 0):
            raise ValueError("feature counts must be non-negative")
        if not np.all(np.equal(np.mod(data, 1.0), 0.0)):
            raise ValueError(
                "feature counts must be whole numbers (activation counts); "
                "threshold or round your activations before building a corpus"
            )
        data_i = data.astype(np.int64)
        if data_i.size and int(data_i.max()) > _MAX_COUNT:
            raise ValueError(f"feature counts must be <= {_MAX_COUNT}")
        return (
            int(num_docs),
            int(num_features),
            [int(x) for x in m.indptr],
            [int(x) for x in m.indices],
            [int(x) for x in data_i],
        )

    arr = np.asarray(counts)
    if arr.ndim != 2:
        raise ValueError(
            f"counts must be a 2-D (num_docs, num_features) matrix, got {arr.ndim}-D"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("feature counts must be finite")
    if np.any(arr < 0):
        raise ValueError("feature counts must be non-negative")
    if not np.all(np.equal(np.mod(arr, 1.0), 0.0)):
        raise ValueError(
            "feature counts must be whole numbers (activation counts); "
            "threshold or round your activations before building a corpus"
        )
    arr = arr.astype(np.int64)
    if arr.size and int(arr.max()) > _MAX_COUNT:
        raise ValueError(f"feature counts must be <= {_MAX_COUNT}")

    num_docs, num_features = arr.shape
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for row in arr:
        nz = np.nonzero(row)[0]
        indices.extend(int(c) for c in nz)
        data.extend(int(row[c]) for c in nz)
        indptr.append(len(indices))
    return int(num_docs), int(num_features), indptr, indices, data


def from_feature_matrix(
    counts,
    feature_names: Sequence[str] | None = None,
    *,
    doc_ids: Sequence[str] | None = None,
    doc_labels: Sequence[str] | None = None,
    metadata=None,
) -> Corpus:
    """Build a :class:`~topica.Corpus` from a document x feature count matrix.

    This is the count-matrix analogue of :meth:`topica.Corpus.from_documents`:
    each column is a fixed-vocabulary "feature" (named by ``feature_names``) and
    each cell is that feature's non-negative integer activation count in the
    document. The resulting corpus feeds any count-based topica model — most
    directly :class:`topica.MechanisticLDA`, the Mechanistic Topic Model over
    sparse-autoencoder features (Zheng et al. 2025; issue #575).

    Parameters
    ----------
    counts
        The document x feature matrix. A dense NumPy array / nested sequence, or
        any SciPy sparse matrix (recommended for the wide, sparse feature spaces
        SAEs produce — the expansion into the sampler's token stream then happens
        once inside Rust, never materializing the dense dimension in Python).
        Values must be finite, non-negative whole numbers.
    feature_names
        One name per column (an SAE feature's description, a concept label, …).
        Defaults to ``["feature_0", "feature_1", …]``. These become the corpus
        ``vocabulary``, so a model's ``top_words`` / ``top_features`` reports them.
    doc_ids, doc_labels
        Optional per-document identifiers and labels, each ``num_docs`` long.
    metadata
        Optional per-document metadata (e.g. a pandas DataFrame), attached to the
        corpus for downstream covariate-effect analysis. Must have one row per
        document, aligned to ``counts``' rows.

    Returns
    -------
    Corpus
        A corpus whose ``vocabulary`` is ``feature_names`` and whose token counts
        are the supplied activations. No vocabulary pruning is applied — filter
        columns beforehand.
    """
    num_docs, num_features, indptr, indices, data = _to_csr(counts)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]
    else:
        feature_names = [str(f) for f in feature_names]
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but counts has "
                f"{num_features} columns"
            )

    names = None if doc_ids is None else [str(x) for x in doc_ids]
    if names is not None and len(names) != num_docs:
        raise ValueError(
            f"doc_ids has {len(names)} entries but counts has {num_docs} rows"
        )
    labels = None if doc_labels is None else [str(x) for x in doc_labels]
    if labels is not None and len(labels) != num_docs:
        raise ValueError(
            f"doc_labels has {len(labels)} entries but counts has {num_docs} rows"
        )

    corpus = Corpus.from_feature_matrix(
        feature_names,
        indptr,
        indices,
        data,
        doc_names=names,
        doc_labels=labels,
    )
    if metadata is not None:
        n = len(metadata) if hasattr(metadata, "__len__") else None
        if n is not None and n != num_docs:
            raise ValueError(
                f"metadata has {n} rows but counts has {num_docs} rows"
            )
        corpus.metadata = metadata
    return corpus
