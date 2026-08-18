"""Reading and describing fitted topics: labels, FREX, tables, representative documents.

Split out of the former monolithic ``topica.validation`` (issue #757). The names
here are also re-exported from :mod:`topica.validation` (a compatibility shim) and
from the workflow namespace :mod:`topica.inspect`.
"""

from __future__ import annotations

import html as _html
import inspect
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .coherence import (
    _as_topic_word, _as_doc_topic, _vocabulary_of, _rbo, _ref_corpus,
    coherence as _coherence, exclusivity as _exclusivity,
)

__all__ = [
    'PyLDAvisInputs',
    'TopicCorrelation',
    'TopicLabels',
    'TopicTable',
    'find_thoughts',
    'find_thoughts_html',
    'frex',
    'label_topics',
    'mmr',
    'prepare_pyldavis',
    'relevance',
    'topic_correlation',
    'topic_table',
    'topics_for_term',
]



# ---------------------------------------------------------------------------
# labelTopics: prob / FREX / lift / score
# ---------------------------------------------------------------------------

def _counts_from(word_counts, corpus):
    """Resolve ``word_counts`` / ``corpus`` to a counts array, or ``None``.

    Pass at most one. A ``corpus`` contributes its ``word_counts`` (empirical
    corpus frequencies, aligned to the vocabulary the model was fit on).
    """
    if word_counts is not None and corpus is not None:
        raise ValueError("pass either word_counts= or corpus=, not both")
    if corpus is not None:
        wc = getattr(corpus, "word_counts", None)
        if wc is None:
            raise ValueError(
                "corpus= must be a topica.Corpus (it exposes word_counts)"
            )
        return np.asarray(wc, dtype=np.float64)
    return None if word_counts is None else np.asarray(word_counts, dtype=np.float64)



def _resolve_word_counts(word_counts, V):
    """Validate ``word_counts`` to a length-``V`` list of ints, or ``[]`` if None."""
    if word_counts is None:
        return []
    wc = np.asarray(word_counts, dtype=np.float64)
    if wc.shape != (V,):
        raise ValueError(f"word_counts must have length {V} (the vocabulary size)")
    if np.any(wc < 0) or not np.all(np.isfinite(wc)):
        raise ValueError("word_counts must be finite and non-negative")
    return [int(round(c)) for c in wc]



def frex(topic_word, vocabulary=None, *, w=0.5, n=10, word_counts=None, corpus=None):
    """FREX (FRequency–EXclusivity) top words per topic.

    For each topic, words are scored by the weighted harmonic mean of the rank of
    their probability (frequency) and the rank of their exclusivity
    ``φ_{t,v} / Σ_k φ_{k,v}`` — stm's ``calcfrex``. ``w`` weights frequency vs
    exclusivity. Returns a list (per topic) of ``(word, frex)``.

    The scores come from the single, stm-faithful implementation in topica's Rust
    core (``topica-core``'s ``inspect`` module — the same one faSTM and the Stata
    plugin use), so the FREX definition can never drift between languages.

    Pass ``word_counts`` (a length-``V`` array of corpus word frequencies) or
    ``corpus`` (a :class:`topica.Corpus`, whose word counts are read for you) to
    apply stm's James-Stein exclusivity shrinkage, which is stm's default; it damps
    the exclusivity of rare words that appear in only one topic by chance. Without
    either (the default here) no shrinkage is applied.

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"w (frequency weight) must be in [0, 1], got {w!r}")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    from ._topica import inspect_frex_scores

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    K, V = phi.shape
    wc = _resolve_word_counts(_counts_from(word_counts, corpus), V)
    scores = np.asarray(inspect_frex_scores(phi.tolist(), wc, float(w)))

    results = []
    for t in range(K):
        idx = np.argsort(scores[t])[::-1][:n]
        results.append([(vocabulary[i], float(scores[t][i])) for i in idx])
    return results



def mmr(topic_word, word_embeddings, vocabulary=None, *, n=10, diversity=0.3, n_candidates=None):
    """Maximal-marginal-relevance top words, to cut redundant near-synonyms.

    For each topic, take the top ``n_candidates`` words by ``topic_word`` weight
    and greedily reselect ``n`` of them, each pick maximizing

        ``(1 - diversity) * relevance(word) - diversity * max_cos(word, picked)``

    where relevance is the (per-topic, max-normalized) ``topic_word`` weight and the
    redundancy term is the cosine between word embeddings. ``diversity=0`` returns
    the plain top words; higher trades relevance for variety, like BERTopic's
    ``MaximalMarginalRelevance(diversity=...)``.

    Parameters
    ----------
    topic_word : a fitted model (uses its ``topic_word`` and ``vocabulary``) or a
        ``(K, V)`` array, in which case pass ``vocabulary``.
    word_embeddings : a ``(V, E)`` matrix aligned to the vocabulary — the word
        vectors (for Top2Vec, the ones you fit with; otherwise embed the vocabulary
        with your embedding model, as BERTopic's MMR does internally).
    n : words returned per topic.
    diversity : in ``[0, 1]``; 0 is the plain top words, higher is more diverse.
    n_candidates : how many top words to rerank (default ``max(5 * n, n)``).

    Returns
    -------
    A list per topic of ``(word, topic_word_weight)`` pairs, like ``top_words``.
    """
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be in [0, 1]")
    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    k, v = phi.shape
    emb = np.asarray(word_embeddings, dtype=np.float64)
    if emb.shape[0] != v:
        raise ValueError(
            f"word_embeddings has {emb.shape[0]} rows but the vocabulary has {v}"
        )
    embn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    n_cand = n_candidates or max(5 * n, n)

    out = []
    for t in range(k):
        cand = np.argsort(phi[t])[::-1][: min(n_cand, v)]
        rel = phi[t, cand].astype(np.float64)
        rel = rel / (rel.max() if rel.max() > 0 else 1.0)
        sims = embn[cand] @ embn[cand].T  # candidate-candidate cosine
        picked = [0]                       # seed with the most relevant word
        rest = list(range(1, len(cand)))
        while rest and len(picked) < n:
            scores = [(1.0 - diversity) * rel[r] - diversity * sims[r, picked].max()
                      for r in rest]
            best = rest[int(np.argmax(scores))]
            picked.append(best)
            rest.remove(best)
        out.append([(vocabulary[cand[i]], float(phi[t, cand[i]])) for i in picked])
    return out



class TopicLabels(dict):
    """One topic's stm-style labels: a dict with keys ``prob``, ``frex``,
    ``lift``, ``score``, each a list of ``(word, value)`` pairs.

    A ``dict`` subclass, so ``labels["frex"]`` works as always. It only adds two
    guard rails, because the nested shape is easy to misread: an ``int`` index
    (``labels[0]``, or unpacking it as if it were a word list) raises a directive
    error naming the real access, and the repr shows the shape instead of dumping
    every word."""

    __slots__ = ()

    def __repr__(self):
        keys = ", ".join(self.keys())
        return f"TopicLabels({keys}); e.g. this['frex'] -> [(word, score), ...]"

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            raise TypeError(
                "a topic's labels are a dict keyed by 'prob'/'frex'/'lift'/'score', "
                "not a word list, so an integer index is undefined. Use "
                "labels['frex'] for the (word, score) pairs, e.g. "
                "[w for w, _ in labels['frex']]. label_topics returns one such "
                "dict per topic."
            )
        return super().__getitem__(key)



def label_topics(topic_word, vocabulary=None, *, n=10, word_counts=None, corpus=None):
    """stm-style topic labels: prob, FREX, lift, and score word lists per topic.

    Returns a list with one :class:`TopicLabels` per topic. Each is a dict with
    keys ``prob``, ``frex``, ``lift``, ``score``, and each value is a list of
    ``(word, value)`` pairs — so select a labeling before reading words::

        labels = topica.inspect.label_topics(model)     # one TopicLabels per topic
        frex_words = [w for w, _ in labels[0]["frex"]]   # top FREX words of topic 0
        prob_score = labels[0]["prob"]                   # [(word, prob), ...]

    Iterating a topic directly (``for w in labels[0]``) yields the dict *keys*
    (``'prob'``, ``'frex'``, ...), not words — a common first-timer trap, so an
    integer index on a topic raises a directive error. (For a table of bare word
    strings instead of pairs, use :func:`topic_table`.) FREX, lift, and score all
    come from the single stm-faithful implementation in topica's Rust core
    (``topica-core``'s ``inspect``), so they cannot drift from faSTM / the Stata
    plugin.

    ``lift`` is stm's lift, ``log P(w|topic) − log P(w)``, where ``P(w)`` is the
    empirical word frequency. Pass ``word_counts`` (a length-``V`` array) or
    ``corpus`` (a :class:`topica.Corpus`, whose word counts are read for you) for
    the exact value; without either, ``P(w)`` is estimated from the topic-word
    matrix's column marginal (lift depends only on relative word frequency, so the
    ranking matches). ``word_counts`` / ``corpus`` also enable stm's James-Stein
    FREX shrinkage (see :func:`frex`).

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    from ._topica import inspect_lift_scores, inspect_score_scores

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    if phi.ndim != 2 or phi.shape[0] == 0:
        raise ValueError(
            "the model has no topics (empty topic_word). For BERTopic/Top2Vec this "
            "means clustering found no clusters — lower min_cluster_size, add data, "
            "or check the scale of your embeddings."
        )
    K, V = phi.shape
    word_counts = _counts_from(word_counts, corpus)

    if word_counts is None:
        # Estimate P(w) from the column marginal. Lift depends only on count
        # ratios, so any common scale works; this yields log(beta) - log(marginal).
        marginal = phi.mean(axis=0)
        lift_counts = [max(int(round(m * 1e6)), 1) for m in marginal]
    else:
        lift_counts = _resolve_word_counts(word_counts, V)
    lift_mat = np.asarray(inspect_lift_scores(phi.tolist(), lift_counts))
    score_mat = np.asarray(inspect_score_scores(phi.tolist()))

    frex_words = frex(topic_word, vocabulary, n=n, word_counts=word_counts)
    out = []
    for t in range(K):
        prob_idx = np.argsort(phi[t])[::-1][:n]
        lift_idx = np.argsort(lift_mat[t])[::-1][:n]
        score_idx = np.argsort(score_mat[t])[::-1][:n]
        out.append(TopicLabels({
            "prob": [(vocabulary[i], float(phi[t, i])) for i in prob_idx],
            "frex": frex_words[t],
            "lift": [(vocabulary[i], float(lift_mat[t, i])) for i in lift_idx],
            "score": [(vocabulary[i], float(score_mat[t, i])) for i in score_idx],
        }))
    return out



class TopicTable(list):
    """The :func:`topic_table` result: a ``list`` of per-topic dict rows, with a
    ``.to_frame()`` for the one-line jump to a pandas DataFrame.

    It is a ``list`` subclass, so it iterates, indexes, and hands to
    ``pandas.DataFrame(...)`` exactly like the plain list it always returned.
    ``.to_frame()`` is the same tidy shape :func:`search_k` and the effect /
    robustness results expose, so you do not have to remember which helpers wrap
    their output and which do not.
    """

    def to_frame(self):
        """The per-topic rows as a pandas DataFrame, one row per topic (raises if
        pandas is absent)."""
        import pandas as pd

        return pd.DataFrame(list(self))



def topic_table(model, vocabulary=None, *, doc_topic=None, n=7, weights=False):
    """A publication-ready topic table: one row per topic with its prevalence and
    its top probability and FREX words.

    Returns a :class:`TopicTable` (a ``list`` subclass) of dicts with ``topic``,
    ``prevalence`` (mean θ), ``prob`` (the top-`n` highest-probability words), and
    ``frex`` (the top-`n` FREX words — usually the better label). Call
    ``.to_frame()`` for the pandas DataFrame that goes in a results section, or
    hand the object straight to ``pandas.DataFrame`` (either works).

    Note the word shape differs from :func:`label_topics`: here ``prob`` and
    ``frex`` are **bare word strings** (ready to drop straight into a DataFrame
    cell), whereas ``label_topics`` returns ``(word, score)`` tuples. So
    ``", ".join(row["frex"])`` is correct on a ``topic_table`` row, while the
    ``", ".join(w for w, _ in ...)`` idiom you would use on ``label_topics`` output
    raises here.

    Pass ``weights=True`` to also carry the numbers behind the words — two extra
    columns ``prob_weights`` and ``frex_weights`` (``list[float]``, aligned with
    ``prob`` / ``frex``) so you can print ``"war (0.03)"``-style cells without a
    separate call. ``prob_weights`` are the topic-word probabilities φ (the same
    numbers :meth:`top_words(n, weights=True) <>` returns). ``frex_weights`` are
    the FREX **scores** from :func:`frex`, not probabilities: a bounded quantile
    score in ``[0, 1]`` (stm's ECDF FREX), so the top word sits near ``1.0`` and
    the values fall off by rank — they rank a topic's words, they are not a
    distribution and do not sum to one.

    Accepts either a **fitted model** (uses its ``topic_word``, ``doc_topic``, and
    ``vocabulary``) or a bare ``(K, V)`` **topic-word array**, matching the sibling
    helpers :func:`frex` / :func:`relevance`::

        topic_table(model)                      # from a fitted model
        topic_table(model.topic_word, vocab)    # from matrices

    With a bare array, pass ``vocabulary``; prevalence needs the document-topic
    matrix, so pass ``doc_topic=`` to get it, otherwise the ``prevalence`` column
    is ``None``.
    """
    phi = _as_topic_word(model)
    vocab = _vocabulary_of(model, vocabulary)
    if doc_topic is not None:
        prevalence = np.asarray(doc_topic, dtype=np.float64).mean(axis=0)
    elif hasattr(model, "doc_topic") and not isinstance(model, np.ndarray):
        prevalence = _as_doc_topic(model).mean(axis=0)
    else:
        prevalence = None  # a bare topic-word array carries no prevalence
    labels = label_topics(phi, vocab, n=n)
    rows = []
    for t in range(len(labels)):
        row = {
            "topic": t,
            "prevalence": float(prevalence[t]) if prevalence is not None else None,
            "prob": [w for w, _ in labels[t]["prob"]],
            "frex": [w for w, _ in labels[t]["frex"]],
        }
        if weights:
            row["prob_weights"] = [float(s) for _, s in labels[t]["prob"]]
            row["frex_weights"] = [float(s) for _, s in labels[t]["frex"]]
        rows.append(row)
    return TopicTable(rows)


def _model_topic_table(self, vocabulary=None, *, doc_topic=None, n=7, weights=False):
    """A publication-ready topic table for this fitted model.

    Method form of :func:`topica.inspect.topic_table`, so ``m.topic_table()`` works by
    analogy with ``m.top_words()`` / ``m.coherence()``. Equivalent to
    ``topica.inspect.topic_table(m, ...)``; see that function for the full argument and
    return description.
    """
    return topic_table(self, vocabulary, doc_topic=doc_topic, n=n, weights=weights)


def _bind_topic_table_method(classes):
    """Attach a ``.topic_table(...)`` method to each model class that exposes
    ``top_words`` (issue #758).

    The method/function split is a coin-flip a newcomer loses: they reach for
    ``m.topic_table()`` by analogy with ``m.top_words()``. Bind the method form
    onto every topic-word model so both spellings work. Classes without a
    ``top_words`` accessor (scaling / embedding models with no topic-word matrix)
    are skipped, matching what the top-level :func:`topic_table` already accepts.
    """
    _model_topic_table.__name__ = "topic_table"
    _model_topic_table.__qualname__ = "topic_table"
    for cls in classes:
        if cls is None or not hasattr(cls, "top_words"):
            continue
        # Time-sliced models (e.g. DTM) expose ``topic_word`` as a method that
        # takes a time index rather than a plain ``(K, V)`` property, so a single
        # flat topic table is ill-defined; skip them rather than binding a method
        # that could only ever raise. ``topica.inspect.topic_table(m.topic_word(t), vocab)``
        # remains the per-slice route.
        if callable(getattr(cls, "topic_word", None)):
            continue
        if "topic_table" in vars(cls):  # already a real (Rust) method; leave it
            continue
        try:
            cls.topic_table = _model_topic_table
        except (TypeError, AttributeError):
            pass  # a class that refuses attribute assignment keeps the function form



def topics_for_term(topic_word, terms, vocabulary=None, *, top_n=5, per_term=False,
                    normalize=False, with_labels=False, label_n=5):
    """The inverse of "top words for a topic": the top topics for a term.

    Given the topic-word matrix φ, rank topics by the weight they place on a
    queried term (or terms) — "which topics is ``'immigration'`` important in, and
    how strongly?". Where :func:`label_topics` / :func:`frex` go *topic → words*,
    this goes *word → topics*, which is handy for corpus exploration and for
    checking where a seed/anchor word actually landed after fitting.

    Parameters
    ----------
    topic_word : a fitted model (uses its ``topic_word`` and ``vocabulary``) or a
        ``(K, V)`` array, in which case pass ``vocabulary``.
    terms : str or sequence of str
        A single term, or several. Terms are matched against ``vocabulary``
        exactly — a topic model's vocabulary is its *processed* tokens (usually
        lowercased, often stemmed), so query the stored form (``"immigr"``), not
        the surface word (``"Immigration"``). A term absent from the vocabulary is
        dropped with a warning; if *every* term is absent a ``ValueError`` is
        raised. Duplicate terms are collapsed to their first occurrence.
    vocabulary : sequence of str, optional
        Required only when ``topic_word`` is a bare array with no vocabulary.
    top_n : int or None, default 5
        How many topics to return, highest-weighted first. ``None`` returns all
        topics ranked. Ties are broken by ascending topic id.
    per_term : bool, default False
        Ignored for a single term. For several terms, ``False`` (the default)
        pools the terms — topics are ranked by the **summed** weight across the
        queried terms — and returns one ranked list. ``True`` instead returns a
        ``dict`` mapping each (found) term to its own ranked list, in the order
        the terms were given.
    normalize : bool, default False
        How each term's per-topic weight is defined. ``False`` uses the raw φ
        entry, ``P(word | topic)`` — directly "what share of this topic's mass is
        this word". Because φ columns are not comparable across words (a frequent
        word carries more mass everywhere), raw weights and the raw pooled sum are
        dominated by the most frequent term. ``True`` instead column-normalizes
        each term to ``P(topic | word) = φ[:, w] / Σ_t φ[t, w]`` — "what share of
        *this word* lives in each topic" — which is comparable across terms and
        frequency-robust, so pooling weights terms equally. This is a distribution
        only for a nonnegative φ; on a signed-axis model (S³, an ideal-point model)
        ``normalize=True`` raises rather than return meaningless ratios, so leave it
        off there.
    with_labels : bool, default False
        When ``True``, each ranked entry gains a third element: the topic's
        ``label_n`` highest-probability words (from φ), so the result reads on its
        own without a manual join back to :func:`label_topics`. Entries become
        ``(topic_id, weight, top_words)`` where ``top_words`` is a ``list[str]``.
    label_n : int, default 5
        How many words to attach per topic when ``with_labels=True``.

    Returns
    -------
    For a single term, or several terms with ``per_term=False``: a list of
    ``(topic_id, weight)`` pairs sorted by descending weight. With several terms
    and ``per_term=True``: a ``dict`` ``{term: [(topic_id, weight), ...]}``. When
    ``with_labels=True`` every pair is instead a ``(topic_id, weight, top_words)``
    triple.

    Examples
    --------
    >>> topics_for_term(model, "immigr", top_n=5)
    [(12, 0.031), (4, 0.018), ...]
    >>> topics_for_term(model, ["immigr", "border"], per_term=True)
    {"immigr": [...], "border": [...]}
    >>> topics_for_term(model, "immigr", top_n=2, with_labels=True)
    [(12, 0.031, ["immigr", "border", "illeg", ...]), (4, 0.018, [...])]
    """
    single = isinstance(terms, str)
    term_list = [terms] if single else list(terms)
    if not term_list:
        raise ValueError("terms is empty; pass a term or a non-empty list of terms")
    if not all(isinstance(t, str) for t in term_list):
        raise ValueError("terms must be a string or a sequence of strings")
    # Collapse to plain str, first occurrence wins. This keeps the pooled sum from
    # double-counting a repeated term and the per_term dict from silently dropping
    # one (dict keys are unique), so both modes see the same term set.
    term_list = list(dict.fromkeys(str(t) for t in term_list))
    # bool is an int subclass; reject it so top_n=True isn't silently read as 1.
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1
    ):
        raise ValueError(f"top_n must be a positive integer or None, got {top_n!r}")
    if with_labels and (
        isinstance(label_n, bool) or not isinstance(label_n, (int, np.integer))
        or label_n < 1
    ):
        raise ValueError(f"label_n must be a positive integer, got {label_n!r}")

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    if phi.ndim != 2 or phi.shape[0] == 0:
        raise ValueError(
            "the model has no topics (empty topic_word). For BERTopic/Top2Vec this "
            "means clustering found no clusters — lower min_cluster_size, add data, "
            "or check the scale of your embeddings."
        )
    K, V = phi.shape
    if len(vocabulary) != V:
        raise ValueError(
            f"vocabulary length ({len(vocabulary)}) does not match the number of "
            f"topic_word columns ({V}); pass the vocabulary aligned to this φ."
        )
    # normalize forms P(topic | word) = φ[:, w] / Σ_t φ[t, w], which is a
    # distribution only for a nonnegative φ. On a signed-axis surface (S³, an
    # ideal-point model) the column sum can be near zero or negative, so the ratio
    # silently returns negative / >1 / enormous "weights". Reject it up front rather
    # than hand back nonsense. Tolerate float-noise negatives (e.g. an LSA/NMF
    # surface reconstructing to ~-1e-16); genuine signed axes carry values of order
    # 0.1–1, far past this floor.
    signed_tol = -1e-8 * max(1.0, float(np.abs(phi).max())) if phi.size else 0.0
    if normalize and phi.size and phi.min() < signed_tol:
        raise ValueError(
            "normalize=True forms P(topic | word) = φ[:, w] / Σ_t φ[t, w], which "
            "assumes a nonnegative topic-word matrix; this φ has negative entries "
            "(a signed-axis model like S³ or an ideal-point model). Leave normalize "
            "off for signed models, or pass a nonnegative φ."
        )
    index = {w: i for i, w in enumerate(vocabulary)}

    found = [(t, index[t]) for t in term_list if t in index]
    missing = [t for t in term_list if t not in index]
    if missing:
        if not found:
            raise ValueError(
                f"none of the requested terms are in the vocabulary: {missing!r}. "
                "The vocabulary holds the model's processed tokens (usually "
                "lowercased, often stemmed) — query that form, not the surface word."
            )
        warnings.warn(
            f"terms not in the vocabulary, dropped: {missing!r}",
            stacklevel=2,
        )

    limit = K if top_n is None else min(int(top_n), K)

    # Each topic's top-probability words, only when labels are asked for. Built
    # once (K is small) and shared across every ranked list. Match label_topics's
    # "prob" ordering exactly (np.argsort ascending, then reversed) so the attached
    # words agree with label_topics(...)[t]["prob"] word-for-word, ties included.
    labels = None
    if with_labels:
        labels = [
            [vocabulary[i] for i in np.argsort(phi[t])[::-1][:label_n]]
            for t in range(K)
        ]

    def _col(j):
        col = phi[:, j]
        if normalize:
            total = col.sum()
            return col / total if total != 0 else np.zeros_like(col)
        return col

    def _rank(weights):
        # Descending by weight; np.argsort is stable, so equal weights keep
        # ascending topic-id order — a deterministic tie-break.
        order = np.argsort(-weights, kind="stable")[:limit]
        if with_labels:
            return [(int(t), float(weights[t]), labels[t]) for t in order]
        return [(int(t), float(weights[t])) for t in order]

    if single:
        return _rank(_col(found[0][1]))
    if per_term:
        return {t: _rank(_col(j)) for t, j in found}
    # Pool the terms: rank topics by their summed (per-term normalized, if asked)
    # weight over the queried terms.
    pooled = np.sum([_col(j) for _, j in found], axis=0)
    return _rank(pooled)



# ---------------------------------------------------------------------------
# topicCorr: topic-correlation network
# ---------------------------------------------------------------------------

@dataclass
class TopicCorrelation:
    cor: np.ndarray
    adjacency: np.ndarray
    edges: list[tuple[int, int, float]] = field(default_factory=list)



def topic_correlation(doc_topic, *, threshold=0.05):
    """Topic-correlation network (≈ stm's ``topicCorr`` "simple" method).

    Correlates topic proportions across documents; topic pairs whose correlation
    exceeds ``threshold`` become network edges. Returns a
    :class:`TopicCorrelation` with the correlation matrix, a 0/1 adjacency
    matrix (zero diagonal), and the edge list.

    This is the raw across-document theta correlation, matching ``stm``'s
    ``topicCorr`` default ("simple") method. Raw theta correlation is
    compositionally biased (the simplex constraint induces spurious negative
    correlation); for the closure-corrected alternatives use
    ``viz.topic_correlation(model, method="clr")`` (the viz layer's default) or
    ``method="partial"``/``"eta"``.

    `doc_topic` is a fitted model (uses its ``doc_topic``) or a ``(D, K)`` array.
    """
    theta = _as_doc_topic(doc_topic)
    cor = np.corrcoef(theta.T)
    cor = np.nan_to_num(cor)
    K = cor.shape[0]
    adj = (cor > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edges = [
        (i, j, float(cor[i, j]))
        for i in range(K)
        for j in range(i + 1, K)
        if cor[i, j] > threshold
    ]
    return TopicCorrelation(cor=cor, adjacency=adj, edges=edges)



# ---------------------------------------------------------------------------
# findThoughts: representative documents per topic
# ---------------------------------------------------------------------------

def find_thoughts(doc_topic, texts=None, *, topic, n=3):
    """The `n` documents most associated with `topic` (≈ stm's ``findThoughts``).

    Returns a list of ``(doc_index, proportion, text)`` sorted by descending
    topic proportion; ``text`` is ``None`` when ``texts`` is not supplied.

    `doc_topic` is a fitted model (uses its ``doc_topic``) or a ``(D, K)`` array.
    `texts` is a sequence of the documents' texts, or a :class:`~topica.Corpus`.
    A Corpus only retains tokens, so its ``text`` field comes back as the
    space-joined *processed* tokens (lowercased, stopword-stripped), not the
    original prose; to read the raw documents, pass your original text sequence
    indexed by ``corpus.kept_indices`` (pruning may have dropped some rows).
    """
    theta = _as_doc_topic(doc_topic)
    # Accept a Corpus for `texts` (recover its token lists), so the natural
    # find_thoughts(model, corpus, topic=...) call works instead of raising an
    # opaque "not subscriptable" error. Join the tokens back into a display string
    # so the returned `text` is a string for every input type, matching
    # find_thoughts_html and the list-of-strings path (#717).
    if texts is not None and hasattr(texts, "documents") and callable(getattr(texts, "documents")):
        texts = [" ".join(d) for d in texts.documents()]
    if topic < 0 or topic >= theta.shape[1]:
        raise ValueError(f"topic {topic} out of range (num_topics={theta.shape[1]})")
    if texts is not None and len(texts) != theta.shape[0]:
        raise ValueError(
            f"texts has {len(texts)} entries but doc_topic has {theta.shape[0]} "
            "rows; pass texts aligned to the kept documents (corpus.kept_indices), "
            "not the original documents — pruning may have dropped some."
        )
    col = theta[:, topic]
    # argpartition for the top-n (O(D)) then sort just those n, rather than a full
    # O(D log D) argsort of every document.
    n_eff = min(n, col.shape[0])
    part = np.argpartition(col, -n_eff)[-n_eff:]
    idx = part[np.argsort(col[part])[::-1]]
    out = []
    for i in idx:
        text = texts[i] if texts is not None else None
        out.append((int(i), float(theta[i, topic]), text))
    return out



# ---------------------------------------------------------------------------
# LDAvis relevance + pyLDAvis export
# ---------------------------------------------------------------------------

def relevance(topic_word, vocabulary=None, *, topic=None, lam=0.6, n=10, term_frequency=None):
    """LDAvis *relevance* of words to topics (Sievert & Shirley 2014):

    ``relevance(w | t) = λ·log p(w|t) + (1-λ)·log[p(w|t) / p(w)]``

    λ=1 ranks by probability; λ=0 by lift (exclusivity); the LDAvis default 0.6
    balances them. ``p(w)`` is the corpus word marginal — pass ``term_frequency``
    (word counts in `vocabulary` order) for the empirical marginal, else the
    topic-averaged φ is used. Returns ``(word, relevance)`` lists per topic, or
    for one ``topic``.

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    k, _ = phi.shape
    if term_frequency is not None:
        tf = np.asarray(term_frequency, dtype=np.float64)
        pw = tf / tf.sum()
    else:
        pw = phi.mean(axis=0)
    pw = np.clip(pw, 1e-12, None)
    log_phi = np.log(np.clip(phi, 1e-12, None))
    rel = lam * log_phi + (1.0 - lam) * (log_phi - np.log(pw))  # (K, V)

    def top(t):
        idx = np.argsort(rel[t])[::-1][:n]
        return [(vocabulary[i], float(rel[t, i])) for i in idx]

    if topic is not None:
        if topic < 0 or topic >= k:
            raise ValueError(f"topic {topic} out of range (num_topics={k})")
        return top(topic)
    return [top(t) for t in range(k)]



@dataclass
class PyLDAvisInputs:
    """The five arrays ``pyLDAvis.prepare`` needs, for when pyLDAvis is not
    installed. ``pyLDAvis.prepare(*inputs.unpack())`` reconstructs the view."""

    topic_term_dists: np.ndarray
    doc_topic_dists: np.ndarray
    doc_lengths: np.ndarray
    vocab: list
    term_frequency: np.ndarray

    def unpack(self):
        return (self.topic_term_dists, self.doc_topic_dists, self.doc_lengths,
                self.vocab, self.term_frequency)



def prepare_pyldavis(model, docs, **kwargs):
    """Build the LDAvis intertopic-distance visualization for a fitted model.

    `docs` are the tokenized training documents (``list[list[str]]``), used for
    document lengths and term frequencies. If ``pyLDAvis`` is installed this
    returns its ``PreparedData`` (pass to ``pyLDAvis.display`` / ``save_html``);
    otherwise it returns a :class:`PyLDAvisInputs` you can feed to
    ``pyLDAvis.prepare`` later. Extra ``kwargs`` go to ``pyLDAvis.prepare``
    (e.g. ``sort_topics=False``).
    """
    # Accept a Corpus directly (recover its token lists), so a corpus built via
    # from_dataframe does not need re-tokenizing from the original text.
    if hasattr(docs, "documents") and callable(getattr(docs, "documents")):
        docs = docs.documents()
    phi = np.asarray(model.topic_word, dtype=np.float64)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    if len(docs) != theta.shape[0]:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {theta.shape[0]} rows; "
            "pass the same documents used to fit the model"
        )
    vindex = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros(len(vocab))
    doc_lengths = np.zeros(len(docs), dtype=np.int64)
    for d, doc in enumerate(docs):
        for w in doc:
            i = vindex.get(w)
            if i is not None:
                tf[i] += 1.0
                doc_lengths[d] += 1
    inputs = PyLDAvisInputs(phi, theta, doc_lengths, vocab, tf)
    try:
        import pyLDAvis
    except ImportError:
        return inputs
    return pyLDAvis.prepare(phi, theta, doc_lengths, vocab, tf, **kwargs)



# ---------------------------------------------------------------------------
# Qualitative validation: highlighted close-reading export
# ---------------------------------------------------------------------------

def find_thoughts_html(
    model,
    texts,
    *,
    topics=None,
    n_docs=3,
    n_words=8,
    max_chars=400,
    markdown=False,
):
    """Render each topic's most representative documents for close reading, with
    the topic's top words **highlighted** in the document text.

    Distant reading (top words) is only half of topic validation; the other half
    is reading the actual documents a topic loads on. This builds a self-contained
    HTML snippet (or Markdown) you can ``display`` in a notebook: per topic, its
    top words followed by its `n_docs` highest-θ documents, each truncated to
    ``max_chars`` with the topic's words marked.

    `model` is any fitted model exposing ``topic_word``, ``doc_topic`` and
    ``vocabulary``; `texts` are the original document strings, aligned to the
    rows of ``doc_topic``. A :class:`~topica.Corpus` is also accepted, in which
    case its tokenized documents are joined back into text for display. Returns a
    string (HTML unless ``markdown=True``).
    """
    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    vocab = list(model.vocabulary)
    # Accept a Corpus for `texts`: use its tokenized documents (joined for the
    # highlighter) rather than crashing on the natural (model, corpus) call (#717).
    if hasattr(texts, "documents") and callable(getattr(texts, "documents")):
        texts = [" ".join(d) for d in texts.documents()]
    if len(texts) != theta.shape[0]:
        raise ValueError("texts must be aligned with the model's documents")
    K = phi.shape[0]
    topics = range(K) if topics is None else topics

    blocks = []
    for t in topics:
        top_ids = np.argsort(phi[t])[::-1][:n_words]
        words = [vocab[i] for i in top_ids]
        docs = np.argsort(theta[:, t])[::-1][:n_docs]
        if markdown:
            blocks.append(_thoughts_md(t, words, docs, theta, texts, max_chars))
        else:
            blocks.append(_thoughts_html(t, words, docs, theta, texts, max_chars))
    if markdown:
        return "\n\n".join(blocks)
    return "<div class=\"tt-thoughts\">\n" + "\n".join(blocks) + "\n</div>"



def _keyword_pattern(words):
    # Match the readable surface form of each top word (phrase tokens use "_").
    surfaces = sorted({w.replace("_", " ") for w in words}, key=len, reverse=True)
    surfaces = [re.escape(s) for s in surfaces if s]
    if not surfaces:
        return None
    return re.compile(r"\b(" + "|".join(surfaces) + r")\b", re.IGNORECASE)



def _truncate(text, max_chars):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"



def _thoughts_html(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    head = (f"<h4>Topic {t}</h4>\n<p><em>"
            + ", ".join(_html.escape(w.replace('_', ' ')) for w in words)
            + "</em></p>\n<ul>")
    items = []
    for d in docs:
        body = _html.escape(_truncate(str(texts[d]), max_chars))
        if pat is not None:
            body = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", body)
        items.append(f"<li><small>doc {int(d)} (θ={theta[d, t]:.2f})</small><br>{body}</li>")
    return head + "\n" + "\n".join(items) + "\n</ul>"



def _thoughts_md(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    lines = [f"### Topic {t}",
             "*" + ", ".join(w.replace("_", " ") for w in words) + "*", ""]
    for d in docs:
        body = _truncate(str(texts[d]), max_chars)
        if pat is not None:
            body = pat.sub(lambda m: f"**{m.group(0)}**", body)
        lines.append(f"- **doc {int(d)}** (θ={theta[d, t]:.2f}): {body}")
    return "\n".join(lines)


def __dir__():
    """Show only the public workflow surface in tab-completion (#757), hiding the
    module's own imports (np, re, dataclass, ...)."""
    return sorted(__all__)
