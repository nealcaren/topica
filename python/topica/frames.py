"""DataFrame-aware corpus construction and metadata alignment.

Topic models drop documents that become empty after tokenization and vocabulary
pruning. When document-level covariates live in a separate array, that silently
misaligns them with the surviving documents, which quietly corrupts any STM
prevalence regression. These helpers keep text and metadata bound together.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from . import Corpus, tokenize


def _is_polars(obj) -> bool:
    """True for a Polars DataFrame/Series, without importing Polars (it is an
    optional dependency)."""
    return type(obj).__module__.split(".", 1)[0] == "polars"


def from_dataframe(
    df,
    *,
    text_col,
    metadata_cols=None,
    tokenizer=None,
    stopwords=None,
    min_length=1,
    min_doc_freq=1,
    max_doc_fraction=1.0,
    min_cf=0,
    rm_top=0,
    max_features=None,
    vocabulary=None,
):
    """Build a :class:`Corpus` from a pandas or Polars DataFrame, keeping
    per-document metadata aligned to the documents that survive pruning.

    ``df[text_col]`` is tokenized (with ``tokenizer`` if given, otherwise
    :func:`topica.tokenize`), a :class:`Corpus` is built with the usual pruning
    options, and the surviving rows of ``metadata_cols`` (default: every column
    except ``text_col``) are attached as ``corpus.metadata`` — a DataFrame of the
    same kind you passed in (pandas in, pandas out; Polars in, Polars out),
    aligned one-to-one with the corpus documents, in the same row order. Feed
    that metadata straight to an STM prevalence design with no manual alignment.

    To turn that metadata into a design matrix with an R-style formula, pass
    ``corpus.metadata`` to :func:`topica.design_matrix`, which needs the optional
    ``formulaic`` package (``pip install "topica[formula]"``); or build the design
    by hand with :func:`topica.one_hot` / :func:`topica.spline`, which need no
    extra dependency.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        One row per document.
    text_col : str
        Column holding the document text.
    metadata_cols : sequence[str], optional
        Columns to carry as aligned metadata. Defaults to all columns except
        ``text_col``.
    tokenizer : callable, optional
        ``str -> list[str]``. Defaults to :func:`topica.tokenize` with the
        ``stopwords`` and ``min_length`` arguments below. This is also where you
        plug in lemmatization: the default tokenizer does **not** stem (stemming
        truncates words to roots like ``militari``/``economi``, which read as
        broken in a topic table), so pass a lemmatizing tokenizer here if you want
        to merge inflections while keeping readable surface forms. See the
        preprocessing guide ("Readable topic words: lemmatize, don't stem").
    max_features : int, optional
        Cap the vocabulary to the ``max_features`` most frequent surviving terms
        (scikit-learn's ``CountVectorizer(max_features=)``). ``None`` leaves it
        unbounded. Passed through to :meth:`Corpus.from_documents`.
    vocabulary : sequence[str], optional
        Pin the vocabulary to this fixed, ordered term list (scikit-learn's
        ``vocabulary=``). Mutually exclusive with the frequency-pruning arguments
        and ``max_features``; see :meth:`Corpus.from_documents`.
    """
    texts = list(df[text_col])  # pandas Series and Polars Series both iterate to values
    if tokenizer is None:
        sw = list(stopwords) if stopwords is not None else None
        docs = [
            tokenize(t if isinstance(t, str) else "", stopwords=sw, min_length=min_length)
            for t in texts
        ]
    else:
        docs = [tokenizer(t if isinstance(t, str) else "") for t in texts]

    # max_doc_fraction removes the highest document-frequency terms — which on a
    # focused corpus can be the very words it is about (e.g. "immigration" in an
    # immigration corpus). The pruning happens in the Rust core with no feedback, so
    # surface the dropped high-frequency terms here before they vanish silently.
    if max_doc_fraction < 1.0 and docs and vocabulary is None:
        from collections import Counter

        n_docs = len(docs)
        doc_freq: Counter = Counter()
        for d in docs:
            doc_freq.update(set(d))
        thresh = max_doc_fraction * n_docs
        dropped = sorted(
            (w for w, c in doc_freq.items() if c > thresh),
            key=lambda w: doc_freq[w],
            reverse=True,
        )
        if dropped:
            shown = ", ".join(f"{w!r} ({doc_freq[w] / n_docs:.0%})" for w in dropped[:8])
            more = "" if len(dropped) <= 8 else f", +{len(dropped) - 8} more"
            warnings.warn(
                f"max_doc_fraction={max_doc_fraction} drops {len(dropped)} very "
                f"common term(s) from the vocabulary: {shown}{more}. On a focused "
                "corpus these can be the words it is about; raise max_doc_fraction "
                "(or leave it at 1.0) to keep them.",
                UserWarning,
                stacklevel=2,
            )

    corpus = Corpus.from_documents(
        docs,
        min_doc_freq=min_doc_freq,
        max_doc_fraction=max_doc_fraction,
        min_cf=min_cf,
        rm_top=rm_top,
        max_features=max_features,
        vocabulary=vocabulary,
    )

    cols = (
        list(metadata_cols)
        if metadata_cols is not None
        else [c for c in df.columns if c != text_col]
    )
    idx = corpus.kept_indices
    if _is_polars(df):
        corpus.metadata = df[list(idx)].select(cols)  # row-select then column-select
    else:
        corpus.metadata = df.iloc[idx][cols].reset_index(drop=True)
    return corpus


def prep_documents(
    corpus,
    meta=None,
    *,
    lower_thresh=1,
    upper_thresh=None,
    rm_top=0,
):
    """Filter rare (and optionally common) vocabulary from a corpus while keeping
    metadata row-aligned with the documents that survive.

    This is topica's analogue of R ``stm``'s ``prepDocuments``. Terms that appear
    in fewer than ``lower_thresh`` documents are dropped from the vocabulary; after
    dropping, documents that become empty are removed. The ``meta`` frame is
    subsetted to exactly the rows of the surviving documents, so the returned
    corpus and metadata stay one-to-one and in the same order. Feeding the returned
    meta straight into an STM prevalence design requires no further alignment.

    Parameters
    ----------
    corpus : Corpus
        A :class:`~topica.Corpus` built by :func:`~topica.Corpus.from_documents`
        or :func:`~topica.from_dataframe`.  The corpus may already carry a
        ``corpus.metadata`` attribute; if ``meta`` is also supplied, ``meta``
        takes precedence and ``corpus.metadata`` is ignored.
    meta : pandas.DataFrame, polars.DataFrame, sequence, or numpy.ndarray, optional
        Per-document covariates, one entry per document in ``corpus`` (before this
        call's filtering).  Accepts a pandas or Polars DataFrame, a numpy array,
        or a plain list/sequence. When ``None``, ``corpus.metadata`` is used if
        present; the returned metadata may then be ``None`` if neither is set.
    lower_thresh : int, default 1
        Minimum document frequency for a term to be kept.  Terms appearing in
        fewer than ``lower_thresh`` documents are removed.  ``lower_thresh=1``
        keeps all terms (no filtering); ``lower_thresh=2`` drops hapax legomena.
    upper_thresh : int or None, default None
        Maximum document frequency for a term to be kept.  Terms appearing in more
        than ``upper_thresh`` documents are removed.  ``None`` disables the upper
        bound.  Passed as ``rm_top`` is handled separately; ``upper_thresh`` is a
        raw count ceiling.
    rm_top : int, default 0
        Number of the most-frequent terms to remove (regardless of count).
        Mirrors :func:`~topica.Corpus.from_documents`'s ``rm_top`` parameter.

    Returns
    -------
    filtered_corpus : Corpus
        A new corpus with the rare-term vocabulary and empty documents removed.
        ``filtered_corpus.kept_indices`` reports which of the *input corpus's*
        document positions survived; ``filtered_corpus.doc_lengths`` is parallel to
        the returned ``filtered_meta`` rows.
    filtered_meta : same type as ``meta``, or None
        The subset of ``meta`` (or ``corpus.metadata``) rows corresponding to the
        surviving documents, in the same order.  Guaranteed
        ``len(filtered_meta) == len(filtered_corpus.doc_lengths)`` when meta is not
        None.
    """
    # Resolve which metadata to use
    if meta is None:
        meta = getattr(corpus, "metadata", None)

    # Get the token-list representation of the current corpus
    docs = corpus.documents()

    # Compute max_doc_fraction from upper_thresh
    n_docs = len(docs)
    if upper_thresh is not None and n_docs > 0:
        max_doc_fraction = upper_thresh / n_docs
    else:
        max_doc_fraction = 1.0

    # Build a new corpus applying the frequency thresholds
    filtered = Corpus.from_documents(
        docs,
        min_doc_freq=lower_thresh,
        max_doc_fraction=max_doc_fraction,
        rm_top=rm_top,
    )

    # filtered.kept_indices are positions into `docs` (= the input corpus docs).
    # Subset meta to those positions.
    idx = filtered.kept_indices
    if meta is not None:
        if hasattr(meta, "iloc"):  # pandas DataFrame / Series
            filtered_meta = meta.iloc[idx].reset_index(drop=True)
        elif _is_polars(meta):  # polars DataFrame / Series
            filtered_meta = meta[list(idx)]
        else:
            try:
                import numpy as np
                if isinstance(meta, np.ndarray):
                    filtered_meta = meta[idx]
                else:
                    filtered_meta = [meta[i] for i in idx]
            except ImportError:
                filtered_meta = [meta[i] for i in idx]
        filtered.metadata = filtered_meta
    else:
        filtered_meta = None

    return filtered, filtered_meta


def plot_removed(corpus, thresholds, *, ax=None):
    """Sweep document-frequency thresholds and plot how many documents and words
    are removed at each level (R ``stm``'s ``plotRemoved``).

    For each threshold value in ``thresholds``, :func:`prep_documents` is called
    and the number of removed documents and removed vocabulary terms is recorded.
    The result is a two-line chart that helps you choose a threshold: a very low
    threshold removes few items; a high threshold may eliminate many documents whose
    only terms are rare, which would corrupt a downstream covariate analysis.

    Parameters
    ----------
    corpus : Corpus
        The corpus to sweep. Passed unchanged to :func:`prep_documents` at each
        threshold.
    thresholds : sequence of int
        Document-frequency thresholds to evaluate (x-axis). Typically a range
        such as ``range(1, 10)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. When ``None`` a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The primary axes (left y-axis = documents removed; right y-axis = words
        removed).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_removed needs matplotlib (pip install matplotlib)."
        ) from e

    thresholds = list(thresholds)
    n_docs_orig = corpus.num_docs
    n_words_orig = corpus.num_words

    docs_removed = []
    words_removed = []
    for t in thresholds:
        filtered, _ = prep_documents(corpus, lower_thresh=t)
        docs_removed.append(n_docs_orig - filtered.num_docs)
        words_removed.append(n_words_orig - filtered.num_words)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    (line_docs,) = ax.plot(
        thresholds, docs_removed, color="C0", marker="o", label="documents removed"
    )
    ax.set_xlabel("lower threshold (minimum document frequency)")
    ax.set_ylabel("documents removed", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2 = ax.twinx()
    (line_words,) = ax2.plot(
        thresholds, words_removed, color="C1", marker="s", label="words removed"
    )
    ax2.set_ylabel("words removed", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    lines = [line_docs, line_words]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax.figure.tight_layout()
    return ax


def align(x, corpus):
    """Realign an external covariate array, DataFrame, Series, or list to the
    documents a :class:`Corpus` kept after pruning. Accepts pandas and Polars
    DataFrames/Series, numpy arrays, and plain lists.

    Use it when your covariates were built against the original documents and
    the corpus dropped some during pruning::

        corpus = topica.Corpus.from_documents(docs, min_doc_freq=5)
        X = topica.align(X, corpus)          # now aligned to corpus rows
        model.fit(corpus, X, prevalence_names=names)
    """
    idx = corpus.kept_indices
    if hasattr(x, "iloc"):  # pandas DataFrame / Series
        return x.iloc[idx].reset_index(drop=True)
    if _is_polars(x):  # polars DataFrame / Series: positional row selection
        return x[list(idx)]
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x[idx]
    except ImportError:
        pass
    return [x[i] for i in idx]
