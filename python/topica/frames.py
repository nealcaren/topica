"""DataFrame-aware corpus construction and metadata alignment.

Topic models drop documents that become empty after tokenization and vocabulary
pruning. When document-level covariates live in a separate array, that silently
misaligns them with the surviving documents, which quietly corrupts any STM
prevalence regression. These helpers keep text and metadata bound together.
"""

from __future__ import annotations

import re
import warnings
from typing import Sequence

from . import Corpus

# HTML tags/entities and http/www URLs, removed by from_dataframe(strip_html=True).
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
# URL: stop at whitespace or the quote/bracket chars that delimit HTML attributes,
# so a URL inside <a href="..."> does not swallow the closing tag.
_URL = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+", re.IGNORECASE)
# Tokens that betray un-stripped web boilerplate if they survive into the vocab.
_WEB_BOILERPLATE = frozenset({"http", "https", "www", "href", "aspx", "html", "nbsp"})


def _strip_web(text: str) -> str:
    """Remove HTML tags/entities and http/www URLs from a raw string, leaving a
    space so adjacent words do not fuse."""
    text = _HTML_TAG.sub(" ", text)     # tags first, before URLs touch the '>'
    text = _URL.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    return text


def _is_polars(obj) -> bool:
    """True for a Polars DataFrame/Series, without importing Polars (it is an
    optional dependency)."""
    return type(obj).__module__.split(".", 1)[0] == "polars"


def _build_reporter(verbose, total, *, label):
    """Return a `topica.progress()` reporter for a corpus-build pass, or None.

    ``verbose=None`` (the default) shows the bar only when stderr is an interactive
    terminal — the same rule ``fit(progress=)`` uses; ``True`` / ``False`` force it.
    The reporter is called as ``reporter(done, total, {})`` between chunks."""
    import sys

    if verbose is False or total == 0:
        return None
    if verbose is None:
        stream = sys.stderr
        if not (hasattr(stream, "isatty") and stream.isatty()):
            return None
    from .progress import progress as _progress

    return _progress(label=label)


def _resolve_df_aliases(min_df, max_df, min_doc_freq, max_doc_fraction, n_docs):
    """Map scikit-learn ``min_df`` / ``max_df`` onto topica's ``min_doc_freq`` /
    ``max_doc_fraction``, using sklearn's convention (int = absolute document count,
    float in ``[0, 1]`` = proportion). Raises if an alias collides with its native
    argument or is out of range."""
    import math

    def _is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    def _is_float(v):
        return isinstance(v, float) and not isinstance(v, bool)

    if min_df is not None:
        if min_doc_freq != 1:
            raise ValueError("pass either min_df or min_doc_freq, not both")
        if _is_float(min_df):
            if not 0.0 <= min_df <= 1.0:
                raise ValueError(f"float min_df must be in [0, 1], got {min_df!r}")
            min_doc_freq = max(1, math.ceil(min_df * n_docs))
        elif _is_int(min_df):
            if min_df < 1:
                raise ValueError(f"int min_df must be >= 1, got {min_df!r}")
            min_doc_freq = min_df
        else:
            raise ValueError(f"min_df must be an int or a float, got {min_df!r}")

    if max_df is not None:
        if max_doc_fraction != 1.0:
            raise ValueError("pass either max_df or max_doc_fraction, not both")
        if _is_float(max_df):
            if not 0.0 <= max_df <= 1.0:
                raise ValueError(f"float max_df must be in [0, 1], got {max_df!r}")
            max_doc_fraction = max_df
        elif _is_int(max_df):
            if max_df < 1:
                raise ValueError(f"int max_df must be >= 1, got {max_df!r}")
            # Absolute doc-count cap -> the fraction the core prunes on.
            max_doc_fraction = min(1.0, max_df / n_docs)
        else:
            raise ValueError(f"max_df must be an int or a float, got {max_df!r}")

    return min_doc_freq, max_doc_fraction


def from_dataframe(
    df,
    *,
    text_col,
    metadata_cols=None,
    tokenizer=None,
    stopwords=None,
    strip_html=False,
    min_length=1,
    min_doc_freq=1,
    max_doc_fraction=1.0,
    min_cf=0,
    rm_top=0,
    max_features=None,
    vocabulary=None,
    min_df=None,
    max_df=None,
    verbose=None,
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
    ``corpus.metadata`` to :func:`topica.design.design_matrix`, which needs the optional
    ``formulaic`` package (``pip install "topica[formula]"``); or build the design
    by hand with :func:`topica.design.one_hot` / :func:`topica.design.spline`, which need no
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
    strip_html : bool, default False
        When True, remove HTML tags, entities, and ``http``/``www`` URLs from each
        document before tokenizing. Web-scraped corpora (news blurbs, press
        releases, forum posts) often carry markup such as ``<a href=...>`` or
        ``www.example.com/page.aspx``; left in, tokens like ``href``, ``http``,
        ``aspx`` survive pruning and form a spurious "boilerplate" topic. This is a
        conservative clean (tags + URLs only); for heavier normalization pass your
        own ``tokenizer``. When left False, a vocabulary that still contains obvious
        web-boilerplate terms triggers a warning pointing here.
    stopwords : iterable of str or str, optional
        Words to drop during tokenizing. Pass an iterable of words (a set, or the
        bundled :data:`topica.data.ENGLISH_STOPWORDS`), or a language name/code string
        like ``"english"`` / ``"en"``, which is resolved through
        :func:`topica.data.stopwords` (the larger stopwords-iso list). Ignored when a
        custom ``tokenizer`` is given.
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
    min_df, max_df : scikit-learn ``CountVectorizer`` aliases for the two document-
        frequency pruning cutoffs, so an sklearn/gensim habit works unchanged.
        Following sklearn's convention, an ``int`` is an absolute document count and
        a ``float`` in ``[0, 1]`` is a proportion of documents: ``min_df=5`` keeps
        terms in at least 5 documents (topica's ``min_doc_freq``); ``max_df=0.5``
        drops terms in more than half the documents (topica's ``max_doc_fraction``).
        Pass at most one of each pair — ``min_df`` or ``min_doc_freq``, ``max_df`` or
        ``max_doc_fraction`` — not both.
    verbose : bool, optional
        Show a progress bar for the tokenization pass, which on a large corpus is
        the longest silent wait (it can exceed the model fit). ``None`` (default)
        shows the bar only when stderr is an interactive terminal, matching
        ``fit(progress=)``; pass ``True`` / ``False`` to force it on or off.
    """
    texts = list(df[text_col])  # pandas Series and Polars Series both iterate to values
    if strip_html:
        texts = [_strip_web(t) if isinstance(t, str) else t for t in texts]

    reporter = _build_reporter(verbose, len(texts), label="corpus")
    if tokenizer is None:
        if isinstance(stopwords, str):
            # A bare string is a language name/code (the scikit-learn
            # stop_words="english" habit), not an iterable of words. Resolve it
            # once here rather than letting list("english") shatter it into
            # single characters that remove nothing useful (#766).
            from .stopwords import stopwords as _resolve_stopwords

            stopwords = _resolve_stopwords(stopwords)
        sw = list(stopwords) if stopwords is not None else None
        # Batch tokenizer: compiles the regex and builds the stopword set once,
        # then tokenizes multi-core with the GIL released — far faster than a
        # Python loop over single-doc tokenize() on a large corpus (#786).
        from ._topica import tokenize_many

        safe = [t if isinstance(t, str) else "" for t in texts]
        docs = tokenize_many(safe, stopwords=sw, min_length=min_length, progress=reporter)
    else:
        # A custom tokenizer is an arbitrary Python callable, so it cannot be
        # parallelized under the GIL; loop, but still report progress if asked.
        # Throttle to ~200 ticks (matching tokenize_many's chunk cadence) so a
        # redirected stderr does not collect one bar frame per document.
        n = len(texts)
        step = max(1, n // 200)
        docs = []
        for i, t in enumerate(texts, 1):
            docs.append(tokenizer(t if isinstance(t, str) else ""))
            if reporter is not None and (i % step == 0 or i == n):
                reporter(i, n, {})

    # scikit-learn min_df/max_df aliases -> topica's min_doc_freq/max_doc_fraction.
    min_doc_freq, max_doc_fraction = _resolve_df_aliases(
        min_df, max_df, min_doc_freq, max_doc_fraction, len(docs)
    )

    # max_doc_fraction removes the highest document-frequency terms — which on a
    # focused corpus can be the very words it is about (e.g. "immigration" in an
    # immigration corpus). The pruning happens in the Rust core with no feedback, so
    # surface the dropped high-frequency terms here before they vanish silently.
    if max_doc_fraction < 1.0 and docs and vocabulary is None:
        import math
        from collections import Counter

        n_docs = len(docs)
        doc_freq: Counter = Counter()
        for d in docs:
            doc_freq.update(w for w in set(d) if w)  # ignore empty tokens, as the core does
        # Match the Rust core's cutoff exactly: a term is dropped iff its document
        # frequency exceeds ceil(n_docs * max_doc_fraction) (mod.rs `max_df`). Using
        # the un-rounded product would over-report on a non-integral threshold.
        max_df = math.ceil(n_docs * max_doc_fraction)
        # Frequency-descending, then alphabetical, so the sampled list is
        # deterministic under any hash seed.
        dropped = sorted(
            (w for w, c in doc_freq.items() if c > max_df),
            key=lambda w: (-doc_freq[w], w),
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

    try:
        corpus = Corpus.from_documents(
            docs,
            min_doc_freq=min_doc_freq,
            max_doc_fraction=max_doc_fraction,
            min_cf=min_cf,
            rm_top=rm_top,
            max_features=max_features,
            vocabulary=vocabulary,
        )
    except ValueError as e:
        # A frequent first-timer mistake is pointing text_col= at a covariate
        # (e.g. a blog-name column), which tokenizes to almost nothing and empties
        # the vocabulary. The core's "no words after frequency filtering" is
        # column-blind; re-raise naming the column, how little it produced, and the
        # other columns to pick from (issue #732).
        if "no words after frequency filtering" in str(e):
            n_tok = sum(len(d) for d in docs)
            n_nonempty = sum(1 for d in docs if d)
            others = [str(c) for c in df.columns if c != text_col]
            raise ValueError(
                f"text_col={text_col!r} produced an empty corpus: it tokenized to "
                f"{n_tok} token(s) across {n_nonempty} non-empty document(s), then "
                f"pruning removed them all. If that count is small, {text_col!r} may "
                f"be a covariate rather than the text column — other columns are "
                f"{others}. Otherwise relax the pruning (min_doc_freq / "
                f"max_doc_fraction / rm_top)."
            ) from e
        raise

    # Web-scraped text often leaves markup tokens (href/http/aspx) in the vocab,
    # where they form a spurious boilerplate topic. If the user did not strip and
    # such tokens survived, point them at strip_html=.
    if not strip_html and vocabulary is None:
        leaked = sorted(_WEB_BOILERPLATE.intersection(corpus.vocabulary))
        if leaked:
            warnings.warn(
                f"vocabulary contains web-boilerplate token(s) {leaked}, which "
                "suggests HTML/URLs in the raw text. Pass strip_html=True to "
                "from_dataframe to remove tags and http/www URLs before "
                "tokenizing, or they may form a spurious topic.",
                UserWarning,
                stacklevel=2,
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
        X = topica.data.align(X, corpus)          # now aligned to corpus rows
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
