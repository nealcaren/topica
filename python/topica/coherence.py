"""Topic coherence and diversity diagnostics.

Windowed PMI-based coherence measures (Röder, Both & Hinneburg, *Exploring the
Space of Topic Coherence Measures*, WSDM 2015) alongside UMass (Mimno et al.
2011) and topic diversity (Dieng, Ruiz & Blei 2020), exposed through a single
gensim-style ``coherence_type=`` switch:

- ``"u_mass"``  — document co-occurrence, intrinsic; range roughly ``(-inf, 0]``.
- ``"c_uci"``   — pairwise PMI over a sliding window (Newman et al. 2010).
- ``"c_npmi"``  — pairwise normalized PMI; range ``[-1, 1]``.
- ``"c_v"``     — the indirect-cosine/NPMI measure that correlates best with human
  judgements in Röder et al.; range roughly ``[0, 1]``.

Every measure scores each topic's top words against a *reference corpus* of
tokenized documents. By default that is your training corpus, but — as with
gensim's :class:`CoherenceModel` — you can pass any external reference (e.g. a
Wikipedia dump) via ``texts`` for a more human-aligned signal. ``topic_diversity``
reports the fraction of unique words across all topics' top-N, the standard
companion to coherence in modern topic-model papers.

These are pure-Python/numpy and work with any model here: pass a fitted model
(its top words are read automatically) or an explicit list of word lists.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# Default sliding-window widths, following gensim's conventions.
_DEFAULT_WINDOW = {"c_v": 110, "c_uci": 10, "c_npmi": 10, "u_mass": None}
_VALID = ("u_mass", "c_uci", "c_npmi", "c_v")


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def _extract_topics(topics, topn):
    """Normalize `topics` to a list of word lists, truncated to `topn`.

    Accepts a fitted model (its ``top_words(topn)`` is read; if absent, the top
    words are derived from ``topic_word`` + ``vocabulary``), a list of word
    lists, or a list of ``(word, prob)`` lists.
    """
    if hasattr(topics, "top_words") and not isinstance(topics, (list, tuple)):
        try:
            rows = topics.top_words(topn)
            return [[w for w, _ in row] for row in rows]
        except (TypeError, ValueError):
            # Some models' top_words takes extra positionals (SAGE's `topic`,
            # DTM's `time`), so top_words(topn) misfires. Fall through to the
            # topic_word + vocabulary contract below.
            pass
    # A model exposing the analysis contract (topic_word + vocabulary) but no
    # usable top_words(n): derive the top words from the matrix, so any conforming
    # model works with coherence/topic_diversity.
    if (hasattr(topics, "topic_word") and hasattr(topics, "vocabulary")
            and not isinstance(topics, (list, tuple, np.ndarray))):
        phi = _as_topic_word(topics)  # marginalizes SAGE's (K, G, V); rejects DTM
        vocab = list(topics.vocabulary)
        n = topn or phi.shape[1]
        return [[vocab[i] for i in np.argsort(row)[::-1][:n]] for row in phi]
    if isinstance(topics, np.ndarray):
        raise ValueError(
            "topics must be a fitted model or a list of word lists, not a raw "
            "topic_word matrix; for a (K, V) matrix pass it through "
            "label_topics(topic_word, vocabulary) or frex(topic_word, vocabulary) first."
        )
    out = []
    for row in topics:
        words = [item[0] if isinstance(item, (tuple, list)) else item for item in row]
        out.append(words[:topn] if topn else list(words))
    return out


# ---------------------------------------------------------------------------
# Co-occurrence accumulation (restricted to the relevant words)
# ---------------------------------------------------------------------------

def _doc_occurrence(texts, vocab):
    """Document frequencies and pairwise document co-occurrence over the
    relevant words (for UMass). Returns (occ[R], co[R,R])."""
    r = len(vocab)
    occ = np.zeros(r)
    co = np.zeros((r, r))
    for doc in texts:
        present = {vocab[w] for w in doc if w in vocab}
        pl = list(present)
        for a in pl:
            occ[a] += 1.0
        for x in range(len(pl)):
            for y in range(x + 1, len(pl)):
                a, b = pl[x], pl[y]
                co[a, b] += 1.0
                co[b, a] += 1.0
    return occ, co


def _window_occurrence(texts, vocab, window):
    """Boolean sliding-window word and pairwise co-occurrence counts over the
    relevant words. Returns (occ[R], co[R,R], n_windows).

    A window of width `window` slides one token at a time; a document shorter
    than the window contributes a single window spanning the whole document.
    Counting is incremental (O(1) per step) and restricted to relevant words,
    which are sparse, so the per-window work is tiny.
    """
    r = len(vocab)
    occ = np.zeros(r)
    co = np.zeros((r, r))
    n_windows = 0

    def emit(present):
        for a in present:
            occ[a] += 1.0
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                a, b = present[x], present[y]
                co[a, b] += 1.0
                co[b, a] += 1.0

    for doc in texts:
        ids = [vocab.get(w, -1) for w in doc]
        length = len(ids)
        if length == 0:
            continue
        w = window if (window and window > 0) else length
        if length <= w:
            present = list({i for i in ids if i >= 0})
            emit(present)
            n_windows += 1
            continue
        cnt = defaultdict(int)
        for p in range(w):
            if ids[p] >= 0:
                cnt[ids[p]] += 1
        emit([k for k, v in cnt.items() if v > 0])
        n_windows += 1
        for s in range(1, length - w + 1):
            out_i = ids[s - 1]
            in_i = ids[s + w - 1]
            if out_i >= 0:
                cnt[out_i] -= 1
                if cnt[out_i] == 0:
                    del cnt[out_i]
            if in_i >= 0:
                cnt[in_i] += 1
            emit([k for k, v in cnt.items() if v > 0])
            n_windows += 1
    return occ, co, n_windows


# ---------------------------------------------------------------------------
# Per-topic scoring
# ---------------------------------------------------------------------------

def _idx(topic, vocab):
    return [vocab[w] for w in topic if w in vocab]


def _score_umass(topic, vocab, occ, co, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(1, len(idx)):
        for j in range(i):
            a, b = idx[i], idx[j]  # a follows b in the ranked list
            # If the conditioning word b never appears in the reference corpus,
            # skip this pair rather than using eps as denominator, which would
            # produce a spuriously large positive score.
            if occ[b] == 0:
                continue
            total += math.log((co[a, b] + 1.0) / occ[b])
            n += 1
    return total / n if n else float("nan")


def _pair_npmi(pi, pj, pij, eps):
    pi = max(pi, eps)
    pj = max(pj, eps)
    if pij <= 0.0:
        pij = eps
    if pij >= 1.0:
        return 1.0
    return math.log(pij / (pi * pj)) / (-math.log(pij))


def _score_uci(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            pij = co[a, b] / nw
            total += math.log((pij + eps) / (max(p[a], eps) * max(p[b], eps)))
            n += 1
    return total / n if n else float("nan")


def _score_npmi(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            total += _pair_npmi(p[a], p[b], co[a, b] / nw, eps)
            n += 1
    return total / n if n else float("nan")


def _score_cv(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    n = len(idx)
    if n < 2:
        return float("nan")
    # NPMI matrix over the topic's words (diagonal = 1, since P(w,w) = P(w)).
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                m[i, j] = 1.0
            else:
                a, b = idx[i], idx[j]
                m[i, j] = _pair_npmi(p[a], p[b], co[a, b] / nw, eps)
    # Indirect cosine: each word's context vector (its row) vs. the set vector
    # (column sums), averaged (Röder et al. 2015, "C_v").
    set_vec = m.sum(axis=0)
    sn = np.linalg.norm(set_vec)
    sims = []
    for i in range(n):
        rn = np.linalg.norm(m[i])
        sims.append(float(m[i] @ set_vec / (rn * sn)) if rn > 0 and sn > 0 else 0.0)
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Fast co-occurrence (Rust core) with a pure-Python fallback
# ---------------------------------------------------------------------------

_SENTINEL = (1 << 32) - 1  # marks a non-relevant token for the Rust core


class _CoLookup:
    """Pairwise co-occurrence backed by the Rust core's flat, pair-indexed
    counts. Supports ``co[a, b]`` for any (a, b), returning 0 for pairs that
    were never requested (the scorers only ask for within-topic pairs)."""

    __slots__ = ("_d",)

    def __init__(self, pairs, counts):
        self._d = {pair: counts[i] for i, pair in enumerate(pairs)}

    def __getitem__(self, key):
        a, b = key
        if a > b:
            a, b = b, a
        return self._d.get((a, b), 0.0)


def _needed_pairs(tops, vocab):
    """The set of within-topic word-id pairs (a < b) that any scorer will read."""
    pairs = set()
    for t in tops:
        ids = [vocab[w] for w in t if w in vocab]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def _occurrences(texts, vocab, tops, window):
    """Return (occ, co, n_windows) for the relevant words, using the Rust core
    when available and falling back to the pure-Python scan otherwise. A
    ``window`` of 0 requests document-level co-occurrence (UMass)."""
    try:
        from ._topica import window_cooccurrence
    except ImportError:
        window_cooccurrence = None

    if window_cooccurrence is not None:
        pairs = _needed_pairs(tops, vocab)
        docs_ids = [[vocab.get(w, _SENTINEL) for w in d] for d in texts]
        occ, counts, nw = window_cooccurrence(docs_ids, len(vocab), pairs, int(window))
        return np.asarray(occ), _CoLookup(pairs, counts), nw

    # Fallback: dense R×R matrices.
    if window == 0:
        occ, co = _doc_occurrence(texts, vocab)
        return occ, co, float(len(texts))
    occ, co, nw = _window_occurrence(texts, vocab, window)
    return occ, co, nw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _as_reference(texts):
    """Normalize a coherence reference corpus to ``list[list[str]]``.

    Accepts a Corpus, raw strings (split on whitespace), or already-tokenized
    documents. Without this, a list of raw strings would be iterated character by
    character, so every top word misses the vocabulary and the score silently
    degenerates — e.g. ``c_v == 1.0`` for every topic (issue #648). Mirrors
    ``validation._ref_corpus`` so ``coherence`` and ``diagnostics`` agree on input.

    A fitted model passed as the reference (the usual ``topics``/``texts`` argument
    slip) is rejected with a directive error rather than a cryptic failure.
    """
    if hasattr(texts, "documents"):
        return texts.documents()
    if hasattr(texts, "topic_word") or hasattr(texts, "doc_topic"):
        raise TypeError(
            "texts looks like a fitted model, not a reference corpus. Pass the model "
            "as the first argument (topics/model) and the corpus — a topica.Corpus, "
            "raw-string documents, or list[list[str]] — as texts."
        )
    try:
        texts = list(texts)
    except TypeError as e:
        raise TypeError(
            "texts must be a topica.Corpus, raw-string documents, or a list of "
            f"tokenized documents (list[list[str]]); got {type(texts).__name__}"
        ) from e
    if texts and isinstance(texts[0], str):
        return [t.split() for t in texts]
    return [list(t) for t in texts]


def coherence(topics, texts, *, coherence_type="c_v", topn=10, window_size=None, epsilon=1e-12):
    """Per-topic coherence against a reference corpus.

    Parameters
    ----------
    topics : a fitted model, or a list of topics (each a list of words, or of
        ``(word, prob)`` pairs).
    texts : the reference corpus, as a :class:`Corpus`, a list of raw-string
        documents (split on whitespace), or already-tokenized documents
        (``list[list[str]]``). Pass your training documents, or an external corpus.
        (Raw strings are tokenized for you — passing them is not silently scored
        character-by-character; see issue #648.)
    coherence_type : one of ``"u_mass"``, ``"c_uci"``, ``"c_npmi"``, ``"c_v"``
        (default ``"c_v"``).
    topn : number of top words per topic to score (default 10).
    window_size : sliding-window width for the windowed measures; ``None`` uses
        the per-measure default (110 for ``c_v``, 10 for ``c_uci``/``c_npmi``).
        Ignored by ``u_mass``.

    Returns
    -------
    numpy.ndarray of shape ``(num_topics,)`` — the coherence of each topic.
    Take ``.mean()`` for the overall model score.
    """
    ct = coherence_type.lower()
    if ct not in _VALID:
        raise ValueError(f"coherence_type must be one of {_VALID}, got {coherence_type!r}")
    if not isinstance(topn, (int, np.integer)) or topn < 1:
        raise ValueError(f"topn must be a positive integer, got {topn!r}")
    texts = _as_reference(texts)
    if len(texts) == 0:
        raise ValueError("texts is empty; pass the reference corpus as list[list[str]]")
    tops = _extract_topics(topics, topn)
    relevant = sorted({w for t in tops for w in t})
    vocab = {w: i for i, w in enumerate(relevant)}

    if ct == "u_mass":
        # window=0 → document-level co-occurrence.
        occ, co, _ = _occurrences(texts, vocab, tops, 0)
        return np.array([_score_umass(t, vocab, occ, co, epsilon) for t in tops])

    win = window_size if window_size is not None else _DEFAULT_WINDOW[ct]
    occ, co, nw = _occurrences(texts, vocab, tops, int(win) if win else 0)
    if nw == 0:
        return np.full(len(tops), float("nan"))
    p = occ / nw
    scorer = {"c_uci": _score_uci, "c_npmi": _score_npmi, "c_v": _score_cv}[ct]
    return np.array([scorer(t, vocab, p, co, nw, epsilon) for t in tops])


@dataclass
class CoherenceCI:
    """Per-topic coherence with a bootstrap standard error and interval.

    ``estimate``/``se``/``ci_low``/``ci_high`` are each ``(num_topics,)`` arrays:
    the coherence on the full reference corpus, the bootstrap standard error, and
    the lower/upper percentile bounds.
    """

    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def coherence_ci(
    topics,
    texts,
    *,
    coherence_type="c_v",
    topn=10,
    window_size=None,
    n_boot=200,
    ci=0.9,
    seed=0,
    epsilon=1e-12,
):
    """Bootstrap standard errors and a credible interval for topic coherence.

    Coherence is a corpus statistic with no model likelihood or posterior behind
    it, so its uncertainty is obtained by bootstrap: hold each topic's top words
    fixed, resample the reference documents with replacement ``n_boot`` times,
    recompute coherence on each resample, and report the per-topic standard error
    and percentile interval. The topics never change, so there is no refit and no
    topic-alignment step — the interval reflects how much a topic's coherence score
    would wobble under a different sample of the reference corpus, the right answer
    to "is topic A's coherence reliably higher than topic B's?".

    ``estimate`` is the coherence on the full corpus (the conventional point
    summary); because resampling documents estimates the sampling distribution of
    that same statistic, the percentile interval is centered on it (unlike the
    posterior-draw intervals elsewhere).

    Parameters
    ----------
    topics : a fitted model, or a list of topics (each a list of words / ``(word,
        prob)`` pairs). The top words are extracted once and held fixed.
    texts : the reference corpus to resample — a :class:`Corpus`, raw-string
        documents, or tokenized documents (as in :func:`coherence`). A fitted model
        passed here (the ``topics``/``texts`` arguments swapped) is rejected with a
        clear error.
    coherence_type, topn, window_size, epsilon : as in :func:`coherence`.
    n_boot : number of bootstrap resamples (each recomputes co-occurrence, so this
        is O(n_boot x corpus size); the windowed measures (``c_v`` etc.) are the
        costliest).
    ci : central interval mass (default 0.9 for a 90% interval).
    seed : seed for the document resampling.

    Returns
    -------
    CoherenceCI
        ``(estimate, se, ci_low, ci_high)``, each ``(num_topics,)``.
    """
    texts = _as_reference(texts)
    if len(texts) == 0:
        raise ValueError("texts is empty; pass the reference corpus as list[list[str]]")
    # Fix the top words once; pass them as the `topics` argument on every resample
    # so the coherence is recomputed for the same words against new corpora.
    tops = _extract_topics(topics, topn)
    common = dict(coherence_type=coherence_type, topn=topn, window_size=window_size, epsilon=epsilon)
    estimate = coherence(tops, texts, **common)

    d = len(texts)
    k = len(tops)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, k), dtype=float)
    for b in range(n_boot):
        picks = rng.integers(0, d, size=d)
        boot = [texts[i] for i in picks]
        draws[b] = coherence(tops, boot, **common)

    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    se = np.full(k, np.nan)
    ci_low = np.full(k, np.nan)
    ci_high = np.full(k, np.nan)
    for t in range(k):
        col = draws[:, t]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            se[t] = col.std(ddof=1)
            ci_low[t], ci_high[t] = np.quantile(col, [lo_q, hi_q])
    return CoherenceCI(np.asarray(estimate, dtype=float), se, ci_low, ci_high)


def topic_diversity(topics, topn=25):
    """Fraction of unique words across all topics' top-`topn` words (Dieng,
    Ruiz & Blei 2020). 1.0 means every top word is unique to its topic; low
    values indicate topics that recycle the same words.

    `topics` is a fitted model or a list of word lists.
    """
    if not isinstance(topn, (int, np.integer)) or topn < 1:
        raise ValueError(f"topn must be a positive integer, got {topn!r}")
    tops = _extract_topics(topics, topn)
    seen = set()
    total = 0
    for t in tops:
        for w in t[:topn]:
            seen.add(w)
            total += 1
    return len(seen) / total if total else float("nan")


def topic_semantic_diversity(topics, topn=25):
    """Fraction of unique top-word *pairs* across all topics (Wu, Nguyen & Luu
    2024, "A Survey on Neural Topic Models", Eq. 18). Where `topic_diversity`
    counts unique single words, this counts unique *pairs* drawn from each
    topic's top-`topn` words: a pair occurrence is "unique" when that unordered
    pair appears in exactly one topic's top words. 1.0 means every top-word pair
    is unique to its topic; higher = more diverse. A pair disambiguates word
    sense, so this is "semantic-aware" — no embeddings are needed.

    `topics` is a fitted model or a list of word lists. `topn` must be an
    integer >= 2 (pairs require at least two words).
    """
    if not isinstance(topn, (int, np.integer)) or topn < 2:
        raise ValueError(f"topn must be an integer >= 2, got {topn!r}")
    tops = _extract_topics(topics, topn)
    # Per-topic unordered pairs (top words within a topic are distinct).
    topic_pairs = [
        {frozenset(p) for p in combinations(t[:topn], 2)} for t in tops
    ]
    # How many topics contain each unordered pair.
    global_count = Counter()
    for pairs in topic_pairs:
        for p in pairs:
            global_count[p] += 1
    total = 0
    unique = 0
    for pairs in topic_pairs:
        for p in pairs:
            total += 1
            if global_count[p] == 1:
                unique += 1
    return unique / total if total else float("nan")


# ---------------------------------------------------------------------------
# Exclusivity + human-validation intrusion tests
#
# These are general topic-model diagnostics — they operate on any fitted
# model's ``topic_word`` (φ) / ``doc_topic`` (θ), not just STM — so they live
# beside coherence/topic_diversity rather than in the stm toolkit.
# ---------------------------------------------------------------------------

def _as_topic_word(obj):
    """A fitted model (use its ``topic_word``) or a ``(K, V)`` array.

    Two models do not present a plain ``(K, V)`` matrix and are normalized here so
    the whole static topic-word surface (coherence, frex, exclusivity, ...) works
    uniformly: SAGE's ``topic_word`` is ``(K, G, V)`` (per content group), reduced
    to the group-averaged marginal; DTM's ``topic_word`` is time-sliced (a
    ``topic_word(time)`` callable) and has no single static matrix, so it is
    rejected with a clear message rather than coerced into an object array.
    """
    if hasattr(obj, "topic_word") and not isinstance(obj, np.ndarray):
        tw = obj.topic_word
        if callable(tw):
            raise ValueError(
                f"{type(obj).__name__}.topic_word is time-sliced — call "
                "topic_word(time) for a slice. The static topic-word diagnostics "
                "do not apply to it; pass a single (K, V) time slice instead."
            )
        phi = np.asarray(tw, dtype=np.float64)
        if phi.ndim == 3:
            # SAGE: (K, G, V). Use the model's group-averaged marginal when it
            # exposes one, else average over the group axis.
            marg = getattr(obj, "topic_word_marginal", None)
            phi = np.asarray(marg, dtype=np.float64) if marg is not None else phi.mean(axis=1)
        return phi
    return np.asarray(obj, dtype=np.float64)


def _as_doc_topic(obj):
    """A fitted model (use its ``doc_topic``) or a ``(D, K)`` array."""
    if hasattr(obj, "doc_topic") and not isinstance(obj, np.ndarray):
        return np.asarray(obj.doc_topic, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def _vocabulary_of(obj, vocabulary):
    if vocabulary is not None:
        return list(vocabulary)
    if hasattr(obj, "vocabulary"):
        return list(obj.vocabulary)
    raise ValueError("vocabulary is required when the model/array carries none")


def exclusivity(model_or_phi, *, n=10, w=0.7):
    """Per-topic exclusivity, shape ``(num_topics,)`` — stm's ``exclusivity``.

    For each topic, the **FREX summary** over its top-``n`` words (by probability):
    the sum of each word's frequency–exclusivity score (the rank harmonic mean of
    probability and exclusivity ``φ_{t,v} / Σ_k φ_{k,v}``, weighted by ``w``, stm's
    default 0.7). Higher means the topic's top words are more distinctive. Pair with
    per-topic coherence to make stm's coherence-vs-exclusivity quality plot: good
    topics sit toward the upper-right (coherent *and* distinctive).

    The scores come from the single stm-faithful implementation in topica's Rust
    core (``topica-core``'s ``inspect``), shared with faSTM and the Stata plugin.

    .. note::
       This is stm's exclusivity (a sum of FREX scores over the top ``n`` words,
       roughly in ``[0, n]``), not a mean exclusivity in ``[0, 1]``. The scale
       changed in the move to the shared stm-faithful core.

    `model_or_phi` is a fitted model (uses its ``topic_word``) or a ``(K, V)``
    array.
    """
    from ._topica import inspect_exclusivity

    phi = _as_topic_word(model_or_phi)
    return np.asarray(
        inspect_exclusivity(phi.tolist(), int(n), float(w)), dtype=np.float64
    )


def semantic_coherence(model_or_phi, texts, vocabulary=None, *, n=10):
    """Per-topic semantic coherence, shape ``(num_topics,)`` — stm's ``semCoh1beta``.

    The UMass document-co-occurrence coherence over each topic's top-``n`` words,
    with stm's 0.01 smoothing (higher = better). This is stm's exact semantic
    coherence, from topica's Rust core (``topica-core``'s ``inspect``), shared with
    faSTM and the Stata plugin. For the broader, gensim-aligned coherence measures
    (``c_v``, ``c_npmi``, ``u_mass``) use :func:`coherence` instead.

    `model_or_phi` is a fitted model (uses its ``topic_word`` / ``vocabulary``) or a
    ``(K, V)`` array (then pass ``vocabulary``). ``texts`` is the reference corpus:
    a :class:`topica.Corpus`, raw-string documents, or a list of token lists (raw
    strings are tokenized, not scored character-by-character; see issue #648).
    """
    from ._topica import inspect_semantic_coherence

    phi = _as_topic_word(model_or_phi)
    # Fall back to the corpus's own vocabulary when the model/array carries none.
    if vocabulary is None and not hasattr(model_or_phi, "vocabulary") \
            and hasattr(texts, "vocabulary"):
        vocabulary = list(texts.vocabulary)
    vocab_list = _vocabulary_of(model_or_phi, vocabulary)
    vocab = {w: i for i, w in enumerate(vocab_list)}
    docs = _as_reference(texts)  # Corpus / raw strings / token lists -> token lists (#648)
    docs_ids = [[vocab[w] for w in d if w in vocab] for d in docs]
    return np.asarray(
        inspect_semantic_coherence(phi.tolist(), docs_ids, int(n)), dtype=np.float64
    )


def word_intrusion(model_or_phi, vocabulary=None, *, n_words=5, seed=0):
    """Build a *word intrusion* test for human topic validation.

    For each topic, take its top ``n_words`` words and splice in one **intruder**
    — a word that ranks highly in some *other* topic but has low probability in
    this one. A coherent topic is one where a human can reliably spot the
    intruder (Chang et al. 2009, "Reading Tea Leaves"). Returns a list (per
    topic) of dicts with:

    - ``topic`` — the topic index,
    - ``words`` — the ``n_words + 1`` words in shuffled, presentation order,
    - ``intruder`` — the intruder word,
    - ``intruder_index`` — its position in ``words`` (the answer key).

    `model_or_phi` is a fitted model (uses its ``topic_word`` / ``vocabulary``)
    or a ``(K, V)`` array (then pass ``vocabulary``). Deterministic for a fixed
    ``seed``.
    """
    phi = _as_topic_word(model_or_phi)
    vocab = _vocabulary_of(model_or_phi, vocabulary)
    K, V = phi.shape
    if K < 2:
        raise ValueError("word intrusion needs at least 2 topics")
    order = np.argsort(phi, axis=1)[:, ::-1]      # words per topic, best first
    top_sets = [set(order[t, :n_words]) for t in range(K)]
    salient = set().union(*top_sets)              # any topic's top words

    out = []
    for t in range(K):
        rng = np.random.RandomState(seed + t)
        top = list(order[t, :n_words])
        top_set = top_sets[t]
        # Intruder candidates: salient in another topic, not a top word here, and
        # low probability in this topic (below this topic's median word prob).
        median = float(np.median(phi[t]))
        cands = [w for w in salient if w not in top_set and phi[t, w] <= median]
        if not cands:  # fall back to any low-prob word in this topic
            low = order[t, ::-1]
            cands = [int(w) for w in low[: max(1, V // 2)]]
        intruder = int(cands[rng.randint(len(cands))])
        words_idx = top + [intruder]
        perm = rng.permutation(len(words_idx))
        shuffled = [int(words_idx[i]) for i in perm]
        out.append({
            "topic": t,
            "words": [vocab[i] for i in shuffled],
            "intruder": vocab[intruder],
            "intruder_index": int(np.where(perm == n_words)[0][0]),
        })
    return out


def document_intrusion(model_or_theta, texts=None, *, n_docs=3, seed=0):
    """Build a *document intrusion* test for human topic validation.

    For each topic, take the ``n_docs`` documents with the highest proportion of
    that topic and splice in one **intruder** — a document where the topic is
    nearly absent (and another topic dominates). A topic that captures real
    document similarity is one where a human can spot the intruder. Returns a
    list (per topic) of dicts with:

    - ``topic`` — the topic index,
    - ``doc_indices`` — the ``n_docs + 1`` document indices in shuffled order,
    - ``intruder_index`` — the intruder's position in ``doc_indices``,
    - ``texts`` — the corresponding text previews (only if ``texts`` is given).

    `model_or_theta` is a ``(D, K)`` θ array (or a fitted model, whose
    ``doc_topic`` is used). Deterministic for a fixed ``seed``.
    """
    theta = _as_doc_topic(model_or_theta)
    D, K = theta.shape
    if K < 2:
        raise ValueError("document intrusion needs at least 2 topics")
    if D < n_docs + 1:
        raise ValueError(f"need at least {n_docs + 1} documents, got {D}")
    if texts is not None and len(texts) != D:
        raise ValueError(
            f"texts has {len(texts)} entries but doc_topic has {D} rows; pass "
            "texts aligned to the kept documents (corpus.kept_indices), not the "
            "original documents — pruning may have dropped some."
        )
    dominant = theta.argmax(axis=1)

    out = []
    for t in range(K):
        rng = np.random.RandomState(seed + t)
        ranked = np.argsort(theta[:, t])[::-1]
        top = [int(d) for d in ranked[:n_docs]]
        # Intruder: a doc dominated by another topic, drawn from the bottom
        # quartile of this topic's proportion.
        tail = ranked[-max(1, D // 4):]
        cands = [int(d) for d in tail if dominant[d] != t]
        if not cands:
            cands = [int(d) for d in tail]
        intruder = int(cands[rng.randint(len(cands))])
        docs_idx = top + [intruder]
        perm = rng.permutation(len(docs_idx))
        shuffled = [int(docs_idx[i]) for i in perm]
        entry = {
            "topic": t,
            "doc_indices": shuffled,
            "intruder_index": int(np.where(perm == n_docs)[0][0]),
        }
        if texts is not None:
            entry["texts"] = [str(texts[i])[:120] for i in shuffled]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# LLM-based topic evaluation (Stammbach, Zouhar, Hoyle, Sachan & Ash 2023,
# "Revisiting Automated Topic Model Evaluation with Large Language Models",
# EMNLP; arXiv:2305.12152). An LLM, prompted with the same instructions the
# crowd-workers received, correlates with human judgment better than NPMI / c_v
# -- especially the *rating* task (`llm_coherence`). These are `llm-bounded`: they
# call an external model, so they are NOT bit-deterministic like the rest of the
# coherence suite. Use `temperature=0` (or `n_samples>1` and aggregate) for
# stability, and read the result as a measurement with model-dependent noise.
# ---------------------------------------------------------------------------

# The prompts ARE the method: kept verbatim from the paper for comparability, as a
# module-level overridable dict (pass `prompts=` to override). `{dataset}` is an
# optional dataset-description clause (small reported gains); `{words}` is the
# comma-separated, shuffled word list.
RATING_PROMPT = (
    "You are a helpful assistant evaluating the top words of a topic model output "
    "for a given topic. {dataset}Please rate how related the following words are to "
    'each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately '
    'related, "3" = very related). Reply with a single number, indicating the '
    "overall appropriateness of the topic.\n\n{words}"
)
INTRUSION_PROMPT = (
    "You are a helpful assistant evaluating the top words of a topic model output "
    "for a given topic. {dataset}Select which word is the least related to all other "
    "words. If multiple words do not fit, choose the word that is most out of place. "
    "Reply with a single word.\n\n{words}"
)
LABEL_PROMPT = (
    "You are a helpful assistant labeling documents by their main theme. {dataset}"
    "{research}Read the document below and annotate it with a {granularity} label "
    "naming its single main theme.{examples} Reply with only the label, a single "
    "word or short phrase.\n\nDocument:\n{document}"
)
# Tan & D'Souza (2025, IRCDL; arXiv:2502.07352; repo MIT) extend the suite beyond
# coherence rating: unsupervised outlier detection, repetitiveness, cross-topic
# diversity, topic-document alignment, and gold-free adversarial self-checks.
OUTLIER_PROMPT = (
    "You are a helpful assistant evaluating the top words of a topic model output for "
    "a given topic. {dataset}Identify the words that do not semantically belong to the "
    "same conceptual theme as the others. Reply with a comma-separated list of only "
    'those words, or "none".\n\n{words}'
)
REPETITIVE_RATE_PROMPT = (
    "You are a helpful assistant evaluating the top words of a topic model output for "
    "a given topic. {dataset}Evaluate whether there are semantically equivalent "
    "(redundant) words. Rate the repetitiveness from 1 to 3, where 1 = highly "
    "repetitive with significant semantic overlap and 3 = minimal repetition with "
    "diverse, distinctive words. Reply with a single number.\n\n{words}"
)
DUPLICATE_PROMPT = (
    "You are a helpful assistant evaluating the top words of a topic model output for "
    "a given topic. {dataset}Identify pairs of words that refer to the exact same "
    "concept or idea (not merely related or similar). Reply with a comma-separated "
    'list of pairs like (word1, word2), or "none".\n\n{words}'
)
DIVERSITY_PROMPT = (
    "You are a helpful assistant comparing two topics from a topic model. {dataset}"
    "Rate the thematic distinctiveness between the two groups of words from 1 to 3, "
    "where 1 = partially overlapping themes and 3 = highly distinctive themes. Reply "
    "with a single number.\n\nGroup 1: {words_a}\nGroup 2: {words_b}"
)
ALIGN_IRRELEVANT_PROMPT = (
    "You are a helpful assistant evaluating how well a topic's words describe a "
    "document. {dataset}Identify which of the topic words are NOT relevant to the "
    'document. Reply with a comma-separated list of the irrelevant words, or "none".'
    "\n\nDocument:\n{document}\n\nTopic words: {words}"
)
ALIGN_MISSING_PROMPT = (
    "You are a helpful assistant evaluating how well a topic's words cover a "
    "document's themes. {dataset}Identify significant themes present in the document "
    "that are NOT captured by the topic words. Reply with a comma-separated list of "
    'the missing themes, or "none".\n\nDocument:\n{document}\n\nTopic words: {words}'
)
# Zheng et al. (2025, "Model Directions, Not Words", arXiv:2507.23220) wrap their
# topic models in model-agnostic LLM tools. The topic *judge* (App. G) is the
# flagship: a Chatbot-Arena-style pairwise comparison that scores topic-document fit
# across models with *different* vocabularies (words vs summaries vs features). The
# summary rendering (F.2) is a uniform one-sentence topic description so that
# comparison is apples-to-apples regardless of a model's native representation.
JUDGE_PROMPT = (
    "You are a helpful assistant comparing two topic models by how well their topics "
    "describe a document. {dataset}Below is a document, then two sets of topics (A and "
    "B) that two models assigned to it. Decide which set better captures the "
    "document's main themes. Reply with \"A\", \"B\", or \"tie\" on the first line, then "
    "one sentence of reasoning.\n\nDocument:\n{document}\n\nTopic set A:\n{set_a}\n\n"
    "Topic set B:\n{set_b}"
)
SUMMARY_PROMPT = (
    "You are a helpful assistant summarizing a topic from a topic model. {dataset}"
    "Given the topic's top words, describe its central theme in a single short "
    "sentence. Reply with only that sentence.\n\n{words}"
)
LLM_EVAL_PROMPTS: dict[str, str] = {
    "rating": RATING_PROMPT, "intrusion": INTRUSION_PROMPT, "label": LABEL_PROMPT,
    "outlier": OUTLIER_PROMPT, "repetitive_rate": REPETITIVE_RATE_PROMPT,
    "duplicate": DUPLICATE_PROMPT, "diversity": DIVERSITY_PROMPT,
    "align_irrelevant": ALIGN_IRRELEVANT_PROMPT, "align_missing": ALIGN_MISSING_PROMPT,
    "judge": JUDGE_PROMPT, "summary": SUMMARY_PROMPT,
}


def _resolve_llm_call(backend):
    """Turn `backend` into a callable ``str -> str``. Accepts a callable (used as-is)
    or a model-name string (routed through :func:`topica.llm_backend`)."""
    if callable(backend):
        return backend
    if isinstance(backend, str):
        from .labeling import llm_backend

        return llm_backend(backend)
    raise TypeError(
        "backend must be a callable (str -> str) or a model-name string; pass e.g. "
        'backend=topica.llm.backend("openrouter/meta-llama/llama-3.3-70b-instruct") '
        'or backend="<your model>".'
    )


def _dataset_clause(dataset_description):
    if not dataset_description:
        return ""
    return f"The topics are from the following corpus: {dataset_description.strip()} "


def _parse_rating(reply, lo, hi):
    """The first integer in `reply` within ``[lo, hi]``, else None."""
    for tok in re.findall(r"-?\d+", str(reply)):
        v = int(tok)
        if lo <= v <= hi:
            return v
    return None


def _match_intruder(reply, words):
    """Match the model's reply to one of `words` (case-insensitive, whole-word
    first, then substring). Returns the matched word or None."""
    r = str(reply).strip().lower()
    lowered = {w.lower(): w for w in words}
    if r in lowered:
        return lowered[r]
    # whole-word hit anywhere in the reply
    toks = set(re.findall(r"[a-z0-9']+", r))
    for w in words:
        if w.lower() in toks:
            return w
    # last resort: substring
    for w in words:
        if w.lower() in r:
            return w
    return None


def llm_coherence(model, *, backend, n_words=10, scale=(1, 3), dataset_description=None,
                  seed=0, n_samples=1, shuffle=True, prompts=None):
    """LLM-rated topic coherence (Stammbach et al. 2023): the headline LLM metric.

    For each topic, the top ``n_words`` words are shuffled and an LLM rates how
    related they are on a ``scale`` (default 1-3). Returns a per-topic numpy array
    of mean ratings (higher = more coherent). This is the metric that **beats
    NPMI / c_v at tracking human judgment** in the paper; it sits beside
    :func:`coherence`, :func:`topic_diversity`, and :func:`topic_semantic_diversity`,
    but is ``llm-bounded`` -- it calls an external model and is not bit-deterministic.

    Parameters
    ----------
    model : fitted model or list of word lists
        Anything :func:`_extract_topics` accepts.
    backend : callable ``str -> str`` or model-name str
        The LLM. Pass ``topica.llm.backend(name, temperature=0)`` or a model name.
    n_words, scale : the number of top words shown and the rating range.
    dataset_description : optional str
        A one-line corpus description added to the prompt (small reported gains).
    seed : int
        Seeds the per-topic word shuffles (reproducible task; the *LLM* is not).
    n_samples : int
        Calls the LLM this many times per topic and averages (tames non-determinism;
        the paper uses temperature=1 to mimic annotator variation).
    prompts : optional dict
        Override the editable templates (key ``"rating"``); defaults to
        :data:`LLM_EVAL_PROMPTS`.
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["rating"]
    lo, hi = int(scale[0]), int(scale[1])
    ds = _dataset_clause(dataset_description)
    topics = _extract_topics(model, n_words)
    out = []
    for t, words in enumerate(topics):
        scores = []
        for s in range(max(1, n_samples)):
            w = list(words)
            if shuffle:
                np.random.RandomState(seed + t * 1000 + s).shuffle(w)
            prompt = tmpl.format(dataset=ds, words=", ".join(w))
            val = _parse_rating(backend(prompt), lo, hi)
            if val is not None:
                scores.append(val)
        out.append(float(np.mean(scores)) if scores else float("nan"))
    return np.array(out, dtype=float)


def llm_intrusion(model, vocabulary=None, *, backend, n_words=5, dataset_description=None,
                  seed=0, n_samples=1, prompts=None):
    """LLM word-intrusion accuracy (Stammbach et al. 2023).

    Builds the intrusion task with :func:`word_intrusion` (top ``n_words`` words plus
    one intruder, shuffled), asks the LLM to pick the intruder, and scores it against
    the answer key. Returns ``{"accuracy": float, "per_topic": [...]}`` where each
    per-topic dict has ``topic``, ``intruder``, ``picked``, and ``correct``.

    The paper finds an LLM matches human accuracy on this *task* (~72%), but rating
    (:func:`llm_coherence`) tracks human topic *rankings* better -- lead with
    ``llm_coherence`` and report this alongside. ``llm-bounded``; see
    :func:`llm_coherence` for the shared ``backend`` / ``n_samples`` semantics.
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["intrusion"]
    ds = _dataset_clause(dataset_description)
    items = word_intrusion(model, vocabulary, n_words=n_words, seed=seed)
    per_topic, correct = [], []
    for item in items:
        votes = []
        for _ in range(max(1, n_samples)):
            prompt = tmpl.format(dataset=ds, words=", ".join(item["words"]))
            picked = _match_intruder(backend(prompt), item["words"])
            if picked is not None:
                votes.append(picked)
        chosen = Counter(votes).most_common(1)
        picked = chosen[0][0] if chosen else None
        hit = picked == item["intruder"]
        per_topic.append({
            "topic": item["topic"], "intruder": item["intruder"],
            "picked": picked, "correct": bool(hit),
        })
        correct.append(hit)
    return {"accuracy": float(np.mean(correct)) if correct else float("nan"),
            "per_topic": per_topic}


def _normalize_label(text):
    """Collapse an LLM label reply to a comparison key (lowercase, trimmed, no
    surrounding quotes/punctuation)."""
    s = str(text).strip().strip("\"'.").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _doc_texts(docs):
    """Normalize `docs` to a list of display strings (Corpus, raw strings, or token
    lists), matching the doc order the model was fit on."""
    if hasattr(docs, "documents"):
        return [" ".join(d) for d in docs.documents()]
    return [d if isinstance(d, str) else " ".join(str(t) for t in d) for d in docs]


def llm_select_k(models, docs, *, backend, n_docs=10, granularity="broad",
                 example_labels=None, research_question=None, criterion="knee",
                 tol=0.03, seed=0, n_samples=1, max_chars=1500, prompts=None):
    """Choose the number of topics by LLM document-label purity (Stammbach et al.
    2023). For each candidate fitted `model`, take each topic's top ``n_docs``
    documents, have an LLM assign each a theme label, and score the topic by **label
    purity** — the fraction of its documents sharing the majority label. The model's
    score is the mean per-topic purity.

    This is the paper's *working* number-of-topics signal: doc-label purity tracks
    ground-truth cluster quality (ARI), whereas rating the top *words* across K does
    not (their negative result). Complements :func:`search_k` (coherence /
    exclusivity / perplexity) with a human-aligned, ``llm-bounded`` criterion.

    .. note::
       Purity **rises then plateaus** as ``K`` grows — over-splitting one theme into
       two topics yields two same-labelled, still-pure topics — so the raw maximum
       tends to over-split (the mirror of coherence's bias toward small ``K``; cf.
       :func:`search_k`'s frontier). The default ``criterion="knee"`` therefore
       returns the **smallest** ``K`` whose purity is within ``tol`` of the best
       (the plateau onset), not the bare ``argmax``. Always read the full ``scores``
       curve; ``criterion="max"`` restores the literal highest-purity pick.

    Parameters
    ----------
    models : sequence of fitted models
        Candidates, typically the same corpus fit at different ``num_topics``.
    docs : Corpus | list of str | list of token lists
        The documents, in the order the models were fit on (their ``doc_topic`` rows).
    backend : callable ``str -> str`` or model-name str
        The LLM (see :func:`llm_coherence`).
    n_docs : int
        Top documents per topic to label.
    granularity : {"broad", "narrow"}
        Whether to ask for a broad or a narrow theme label.
    example_labels : optional sequence of str
        Example label vocabulary shown to the model (steers granularity/format).
    research_question : optional str
        A one-line framing ("label by the policy area discussed", ...).
    criterion : {"knee", "max"}
        How ``best`` is chosen from the purity curve. ``"knee"`` (default) returns
        the smallest ``K`` within ``tol`` of the best purity (the plateau onset);
        ``"max"`` returns the highest-purity model (which tends to over-split).
    tol : float
        Purity tolerance for the knee (default 0.03).
    n_samples : int
        Majority-vote the label over this many calls per document.
    max_chars : int
        Truncate each document to this many characters in the prompt.

    Returns
    -------
    dict with ``best`` (the chosen model's ``num_topics``), ``best_index``, and
    ``scores`` (a list of ``{"num_topics", "purity", "per_topic_purity"}`` per model).
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["label"]
    texts = _doc_texts(docs)
    gran = "broad" if granularity not in ("broad", "narrow") else granularity
    ex = ""
    if example_labels:
        ex = " Example labels: " + ", ".join(str(x) for x in example_labels) + "."
    rq = f"{research_question.strip()} " if research_question else ""

    def label_doc(text, salt):
        votes = []
        for s in range(max(1, n_samples)):
            prompt = tmpl.format(dataset="", research=rq, granularity=gran,
                                 examples=ex, document=str(text)[:max_chars])
            reply = _normalize_label(backend(prompt))
            if reply:
                votes.append(reply)
        c = Counter(votes).most_common(1)
        return c[0][0] if c else None

    scores = []
    for mi, model in enumerate(models):
        theta = np.asarray(model.doc_topic)  # (D, K)
        K = theta.shape[1]
        per_topic = []
        salt = 0
        for t in range(K):
            top = np.argsort(theta[:, t])[::-1][:n_docs]
            labels = [label_doc(texts[d], salt + i) for i, d in enumerate(top)]
            labels = [x for x in labels if x is not None]
            salt += len(top)
            if not labels:
                per_topic.append(float("nan"))
                continue
            majority = Counter(labels).most_common(1)[0][1]
            per_topic.append(majority / len(labels))
        finite = [p for p in per_topic if not math.isnan(p)]
        purity = float(np.mean(finite)) if finite else float("nan")
        scores.append({
            "num_topics": int(K), "purity": purity, "per_topic_purity": per_topic,
        })

    valid = [(i, s["purity"]) for i, s in enumerate(scores) if not math.isnan(s["purity"])]
    if not valid:
        best_index = None
    elif criterion == "max":
        best_index = max(valid, key=lambda x: x[1])[0]
    else:
        # Knee: the smallest-K model within `tol` of the best purity (the plateau
        # onset), preferring parsimony over the over-splitting raw maximum.
        best_purity = max(p for _, p in valid)
        within = [i for i, p in valid if p >= best_purity - tol]
        best_index = min(within, key=lambda i: scores[i]["num_topics"])
    return {
        "best": scores[best_index]["num_topics"] if best_index is not None else None,
        "best_index": best_index,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Tan & D'Souza (2025) metrics: outlier, repetitiveness, diversity, alignment,
# and gold-free adversarial self-checks. All llm-bounded; exposed under topica.llm.
# ---------------------------------------------------------------------------

def _parse_word_list(reply, allowed=None):
    """Parse a comma-separated word list from an LLM reply, dropping ``none``/empty
    and bracket noise. If `allowed` (a set of lowercased words) is given, keep only
    those. Returns a list of lowercased words."""
    s = str(reply).strip().strip("[](){}").strip()
    if not s or s.lower() in ("none", "[]", "n/a", "no outliers", "none."):
        return []
    parts = re.split(r"[,\n;]+", s)
    out = []
    for p in parts:
        w = re.sub(r"[^a-z0-9'\- ]+", "", p.strip().lower()).strip()
        if not w or w == "none":
            continue
        if allowed is None or w in allowed:
            out.append(w)
        elif allowed is not None:
            # a multi-word reply phrase: keep any allowed token it contains
            for tok in w.split():
                if tok in allowed:
                    out.append(tok)
    return out


def _parse_pairs(reply):
    """Parse ``(a, b)`` style pairs from an LLM reply. Returns a list of
    lowercased ``(a, b)`` tuples (order-normalized)."""
    out = []
    for a, b in re.findall(r"\(\s*([\w'\-]+)\s*,\s*([\w'\-]+)\s*\)", str(reply).lower()):
        pair = tuple(sorted((a.strip(), b.strip())))
        if pair[0] and pair[1] and pair[0] != pair[1]:
            out.append(pair)
    return out


def llm_outlier(model, *, backend, n_words=10, n_samples=5, threshold=3,
                dataset_description=None, seed=0, prompts=None):
    """Unsupervised semantic-outlier detection (Tan & D'Souza 2025, ``C_outlier``).

    For each topic, asks the LLM to list the words that do not fit the topic, over
    ``n_samples`` runs, and keeps a word flagged in at least ``threshold`` runs (the
    paper's 3-of-5 vote). Returns a per-topic list of dicts with ``topic``,
    ``outliers`` (the flagged words), and ``count``. Unlike :func:`llm_intrusion`
    there is no planted answer — this surfaces *which* words make a topic incoherent.
    ``llm-bounded``; see :func:`llm_coherence` for ``backend``/``n_samples`` semantics.
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["outlier"]
    ds = _dataset_clause(dataset_description)
    out = []
    for t, words in enumerate(_extract_topics(model, n_words)):
        allowed = {w.lower() for w in words}
        votes = Counter()
        for _ in range(max(1, n_samples)):
            flagged = _parse_word_list(backend(tmpl.format(dataset=ds, words=", ".join(words))), allowed)
            votes.update(set(flagged))
        outliers = [w for w, c in votes.items() if c >= threshold]
        out.append({"topic": t, "outliers": outliers, "count": len(outliers)})
    return out


def llm_repetitiveness(model, *, backend, n_words=10, n_samples=1,
                       dataset_description=None, seed=0, prompts=None):
    """LLM repetitiveness (Tan & D'Souza 2025): is apparent coherence just redundancy?

    Returns a per-topic list of dicts with ``rate`` (``R_rate``: 1 = highly
    repetitive, 3 = diverse/distinctive; averaged over ``n_samples``),
    ``duplicate_pairs`` (``R_duplicate``: word pairs the LLM judges the *same*
    concept), and ``duplicate_count``. A robust coherent topic has a *high* rate and
    a *low* duplicate count. Complements :func:`topic_semantic_diversity` on the LLM
    side. ``llm-bounded``.
    """
    backend = _resolve_llm_call(backend)
    P = prompts or LLM_EVAL_PROMPTS
    ds = _dataset_clause(dataset_description)
    out = []
    for t, words in enumerate(_extract_topics(model, n_words)):
        ws = ", ".join(words)
        rates = []
        for _ in range(max(1, n_samples)):
            v = _parse_rating(backend(P["repetitive_rate"].format(dataset=ds, words=ws)), 1, 3)
            if v is not None:
                rates.append(v)
        allowed = {w.lower() for w in words}
        pair_votes = Counter()
        for _ in range(max(1, n_samples)):
            for pr in _parse_pairs(backend(P["duplicate"].format(dataset=ds, words=ws))):
                if pr[0] in allowed and pr[1] in allowed:
                    pair_votes[pr] += 1
        pairs = [p for p, c in pair_votes.items() if c >= (max(1, n_samples) + 1) // 2]
        out.append({"topic": t, "rate": float(np.mean(rates)) if rates else float("nan"),
                    "duplicate_pairs": pairs, "duplicate_count": len(pairs)})
    return out


def llm_diversity(model, *, backend, n_words=10, n_samples=1, max_pairs=None,
                  dataset_description=None, seed=0, prompts=None):
    """Cross-topic LLM diversity (Tan & D'Souza 2025, ``D_rate``).

    Rates the thematic distinctiveness of every pair of topics 1-3 (1 = overlapping,
    3 = distinctive) and averages. Returns ``{"mean": float, "pairwise": [...]}`` with
    one ``{"topics": (i, j), "rate": r}`` per scored pair. O(K²) calls; pass
    ``max_pairs`` to score a deterministic random subset. The LLM analog of
    :func:`topic_diversity` / :func:`topic_semantic_diversity`. ``llm-bounded``.
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["diversity"]
    ds = _dataset_clause(dataset_description)
    topics = _extract_topics(model, n_words)
    pairs = list(combinations(range(len(topics)), 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.RandomState(seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
    rows = []
    for i, j in pairs:
        scores = []
        for _ in range(max(1, n_samples)):
            v = _parse_rating(backend(tmpl.format(
                dataset=ds, words_a=", ".join(topics[i]), words_b=", ".join(topics[j]))), 1, 3)
            if v is not None:
                scores.append(v)
        if scores:
            rows.append({"topics": (i, j), "rate": float(np.mean(scores))})
    mean = float(np.mean([r["rate"] for r in rows])) if rows else float("nan")
    return {"mean": mean, "pairwise": rows}


def llm_adversarial(model, *, backend, intruder="shakespeare", n_words=10,
                    n_samples=5, threshold=3, dataset_description=None, seed=0, prompts=None):
    """Gold-free adversarial self-check (Tan & D'Souza 2025, ``AdvT_outlier``).

    Plants a known-unrelated word (default ``"shakespeare"``) into each topic's top
    words and measures how often the LLM's :func:`llm_outlier` detection flags it.
    This validates the metric *and* the model's capability **without human-gold data**,
    on any corpus — a low detection rate means the model is too weak for these tasks.
    Returns ``{"detection_rate": float, "intruder": str, "per_topic": [...]}``.
    """
    backend = _resolve_llm_call(backend)
    tmpl = (prompts or LLM_EVAL_PROMPTS)["outlier"]
    ds = _dataset_clause(dataset_description)
    intr = intruder.lower()
    per_topic, hits = [], []
    for t, words in enumerate(_extract_topics(model, n_words)):
        rng = np.random.RandomState(seed + t)
        planted = list(words) + [intruder]
        rng.shuffle(planted)
        allowed = {w.lower() for w in planted}
        votes = Counter()
        for _ in range(max(1, n_samples)):
            votes.update(set(_parse_word_list(backend(tmpl.format(dataset=ds, words=", ".join(planted))), allowed)))
        caught = votes[intr] >= threshold
        per_topic.append({"topic": t, "caught": bool(caught), "flagged": votes[intr]})
        hits.append(caught)
    return {"detection_rate": float(np.mean(hits)) if hits else float("nan"),
            "intruder": intruder, "per_topic": per_topic}


def llm_alignment(model, docs, *, backend, n_words=10, n_docs=5,
                  dataset_description=None, seed=0, prompts=None, max_chars=1500):
    """Topic-document alignment (Tan & D'Souza 2025, ``A_ir-topic`` / ``A_missing-theme``).

    For each topic, takes its top ``n_docs`` documents and asks the LLM, per document,
    (1) how many topic words are *irrelevant* to it (overrepresentation) and (2) how
    many document themes are *missing* from the topic words (underrepresentation),
    averaging over the documents. Returns a per-topic list of dicts with ``topic``,
    ``irrelevant`` (mean count) and ``missing`` (mean count); lower is better on both.
    Needs the documents and O(K·n_docs) calls. ``llm-bounded``.
    """
    backend = _resolve_llm_call(backend)
    P = prompts or LLM_EVAL_PROMPTS
    ds = _dataset_clause(dataset_description)
    texts = _doc_texts(docs)
    theta = np.asarray(model.doc_topic)
    topics = _extract_topics(model, n_words)
    out = []
    for t, words in enumerate(topics):
        ws = ", ".join(words)
        allowed = {w.lower() for w in words}
        top = np.argsort(theta[:, t])[::-1][:n_docs]
        irr, mis = [], []
        for d in top:
            doc = str(texts[d])[:max_chars]
            irr.append(len(_parse_word_list(backend(P["align_irrelevant"].format(dataset=ds, document=doc, words=ws)), allowed)))
            mis.append(len(_parse_word_list(backend(P["align_missing"].format(dataset=ds, document=doc, words=ws)))))
        out.append({"topic": t,
                    "irrelevant": float(np.mean(irr)) if irr else float("nan"),
                    "missing": float(np.mean(mis)) if mis else float("nan")})
    return out


# ---------------------------------------------------------------------------
# Topic judge: pairwise document-topic evaluation -> Bradley-Terry -> Elo
# (Zheng et al. 2025, "Model Directions, Not Words", App. G)
# ---------------------------------------------------------------------------


def _top_topics_for_doc(theta_row, q, p):
    """A document's top topics: the top ``q`` by weight, or the fewest topics whose
    cumulative weight reaches fraction ``p`` of the row, whichever is *fewer*
    (Zheng et al. 2025, §5.1). Always at least one topic."""
    order = np.argsort(theta_row)[::-1]
    if order.size == 0:
        return []
    csum = np.cumsum(theta_row[order])
    total = csum[-1]
    n_p = int(np.searchsorted(csum, p * total) + 1) if total > 0 else order.size
    n = min(int(q), n_p) if q else n_p
    n = max(1, min(n, order.size))
    return [int(i) for i in order[:n]]


def _render_topic_set(model, topic_ids, representation, backend, ds, n_words,
                      summary_tmpl, cache):
    """Render a model's topics for one document as text: a word list per topic, or a
    one-sentence LLM summary per topic (cached per ``(model, topic)`` so a topic is
    summarized once no matter how many comparisons reuse it)."""
    from .analysis import _top_words

    lines = []
    for t in topic_ids:
        words = _top_words(model, int(t), n_words)
        if representation == "summary":
            key = (id(model), int(t))
            if key not in cache:
                reply = backend(summary_tmpl.format(dataset=ds, words=", ".join(words)))
                cache[key] = " ".join(str(reply).strip().split()) or ", ".join(words)
            lines.append(f"- {cache[key]}")
        else:
            lines.append(f"- {', '.join(words)}")
    return "\n".join(lines)


def _parse_judge_choice(reply):
    """Parse an ``A`` / ``B`` / ``tie`` verdict from the judge's reply. Returns
    ``"a"``, ``"b"``, ``"tie"``, or ``None`` when nothing is recognisable (treated as
    a tie by the caller). The prompt asks for the verdict on the first line, so that
    is read first; a mention of "tie" then beats a bare stray ``a``/``b`` (e.g. the
    English article "a" in "it is a tie") so a verbose tie is not misread as an
    A-win."""
    r = str(reply).strip().lower()
    first = re.sub(r"[^a-z]", " ", r.split("\n", 1)[0]).split()
    if first and first[0] in ("a", "b", "tie"):
        return first[0]
    if re.search(r"\btie\b", r):
        return "tie"
    m = re.search(r"\b([ab])\b", r)
    return m.group(1) if m else None


def _bradley_terry(win, *, smoothing=1.0, iters=1000, tol=1e-10):
    """Bradley-Terry strengths from a win-credit matrix by the MM algorithm (Hunter
    2004). ``win[i, j]`` is the credit model ``i`` earned against ``j`` (a win = 1, a
    tie = 0.5 each). ``smoothing`` adds a phantom tie to every ordered pair so a model
    that always or never wins still gets a finite strength. Strengths are normalised
    to geometric mean 1."""
    W = np.asarray(win, dtype=float).copy()
    M = W.shape[0]
    if smoothing:
        off = (np.ones((M, M)) - np.eye(M)) * (0.5 * smoothing)
        W = W + off
    n = W + W.T  # games played between i and j
    w = W.sum(axis=1)  # total win-credit for i
    p = np.ones(M)
    for _ in range(iters):
        p_new = np.empty(M)
        for i in range(M):
            denom = 0.0
            for j in range(M):
                if i != j:
                    s = p[i] + p[j]
                    if s > 0:
                        denom += n[i, j] / s
            p_new[i] = (w[i] / denom) if denom > 0 else p[i]
        p_new = np.maximum(p_new, 1e-300)
        p_new /= np.exp(np.mean(np.log(p_new)))
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    return p


def _bt_to_elo(strengths, base=1500.0, scale=400.0):
    """Rescale Bradley-Terry strengths to Elo (default mean 1500): a 400-point gap is
    a 10:1 predicted win ratio, matching ``P(i beats j) = p_i / (p_i + p_j)``."""
    logp = np.log10(np.maximum(np.asarray(strengths, dtype=float), 1e-300))
    return scale * (logp - logp.mean()) + base


@dataclass
class JudgeResult:
    """Result of :func:`llm_judge`: an Elo ranking of the compared models.

    Attributes
    ----------
    elo : dict[str, float]
        Bradley-Terry strengths rescaled to Elo (mean 1500); higher is better.
    win_matrix : numpy.ndarray
        ``(M, M)`` win-credit matrix (``[i, j]`` = credit model ``i`` earned against
        ``j``; a tie adds 0.5 each way), in ``names`` order.
    bootstrap_ci : dict[str, tuple[float, float]]
        Per-model percentile CI on Elo from resampling the comparisons.
    comparisons : list[dict]
        The raw audit records (``doc``, ``model_a``, ``model_b`` as presented,
        ``choice``, ``winner``, ``reasoning``), so a run is re-aggregatable without
        re-calling the LLM.
    names : list[str]
        Model names, the order of ``win_matrix`` rows/columns.
    representation : str
        ``"summary"`` or ``"words"`` -- how each topic set was rendered to the judge.
    """

    elo: dict
    win_matrix: np.ndarray
    bootstrap_ci: dict
    comparisons: list
    names: list
    representation: str

    def ranking(self) -> list:
        """Model names best-to-worst by Elo."""
        return sorted(self.names, key=lambda k: -self.elo[k])

    def to_frame(self):
        """A pandas DataFrame: one row per model with ``elo``, ``ci_low``,
        ``ci_high``, sorted best-first (raises if pandas is absent)."""
        import pandas as pd

        rows = [{"model": nm, "elo": self.elo[nm],
                 "ci_low": self.bootstrap_ci.get(nm, (float("nan"),) * 2)[0],
                 "ci_high": self.bootstrap_ci.get(nm, (float("nan"),) * 2)[1]}
                for nm in self.ranking()]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """A short text leaderboard (model, Elo, CI), best-first. Flags when the top
        two models' bootstrap CIs overlap -- with few comparisons they usually do,
        and an overlap means the ranking does not separate them (treat as no
        decision, and raise ``n_comparisons``)."""
        rank = self.ranking()
        lines = [f"LLM topic judge ({self.representation}, {len(self.comparisons)} "
                 f"comparisons, Elo mean 1500)"]
        for nm in rank:
            lo, hi = self.bootstrap_ci.get(nm, (float("nan"), float("nan")))
            lines.append(f"  {nm:<20s} {self.elo[nm]:7.1f}  [{lo:7.1f}, {hi:7.1f}]")
        if len(rank) >= 2:
            (lo0, hi0), (lo1, hi1) = (self.bootstrap_ci.get(rank[0], (float("nan"),) * 2),
                                      self.bootstrap_ci.get(rank[1], (float("nan"),) * 2))
            if not math.isnan(lo0) and not math.isnan(hi1) and lo0 <= hi1:
                lines.append("  note: the top two CIs overlap -- not a decision at "
                             "this n_comparisons; raise it to separate the models.")
        return "\n".join(lines)


def llm_judge(models, docs, *, backend, n_comparisons=100, q=2, p=0.75,
              representation="summary", n_words=10, dataset_description=None,
              bootstrap=100, confidence_level=0.95, bt_smoothing=1.0, max_chars=1500,
              seed=0, prompts=None):
    """Rank topic models by an LLM's pairwise topic-document judgments (Zheng et al.
    2025, App. G) -- the paper's flagship, vocabulary-agnostic evaluation.

    For each unordered model pair and each of ``n_comparisons`` rounds: sample a
    document, take each model's top topics for it (top ``q``, or the topics under
    cumulative mass ``p``, whichever fewer), render each set as text, and ask the LLM
    which set better captures the document. Outcomes (wins, ties as half-wins) are
    aggregated with a Bradley-Terry model and rescaled to Elo (mean 1500), with a
    bootstrap CI over the comparisons.

    Unlike :func:`llm_coherence` (intra-topic word relatedness), this scores
    topic-*document* fit and compares models with *different* vocabularies fairly
    (words vs. one-sentence summaries), so it is the natural engine for ranking a set
    of fitted models on the same corpus. ``llm-bounded``: the LLM is not
    bit-reproducible, but ``seed`` fixes the document sampling and presentation order,
    so the comparison *design* is reproducible.

    Cost: judge makes ``n_comparisons * M*(M-1)/2`` LLM calls for ``M`` models, plus
    (for ``representation="summary"``) up to one cached summary call per surfaced
    ``(model, topic)``. The default ``n_comparisons=100`` follows the paper and can be
    hundreds of calls; lower it (and read the CIs) while exploring. With few
    comparisons the bootstrap CIs overlap and the ranking does not separate the
    models -- :meth:`JudgeResult.summary` flags that -- so scale ``n_comparisons`` up
    before reporting an Elo table.

    Parameters
    ----------
    models : dict[str, fitted model]
        Two or more fitted models, all fit on the same ``docs`` in the same order.
        judge aligns each model's ``doc_topic`` row ``d`` to ``docs[d]`` and cannot
        verify the correspondence beyond the row count, so models fit on different
        documents produce a silently invalid ranking (it warns when their
        vocabularies disagree). Same corpus, same order.
    docs : Corpus | list of str | list of token lists
        The documents, in the order the models were fit on (their ``doc_topic`` rows).
    backend : callable ``str -> str`` or model-name str
        The LLM judge (see :func:`llm_coherence`); use ``temperature=0``.
    n_comparisons : int
        Comparisons per model pair (paper: 100).
    q : int
        Cap on the number of top topics shown per document per model (``0`` = no
        cap, use ``p`` alone).
    p : float
        Cumulative-probability cap on top topics (the smaller of ``q`` / ``p`` wins).
    representation : {"summary", "words"}
        Render each topic as a one-sentence LLM summary (paper default; use it to
        compare *different model families* fairly) or as its top-``n_words`` word list
        ("words" is cheaper -- no summary calls -- and fine for a same-family sweep,
        e.g. LDA at several K).
    n_words : int
        Top words shown per topic in ``"words"`` mode, and summarized per topic in
        ``"summary"`` mode.
    dataset_description : optional str
        A one-line corpus description added to every prompt (e.g. "Usenet posts about
        computing and politics"), which can sharpen the judge and the summaries.
    bootstrap : int
        Resamples for the per-model Elo CI; 0 skips (CIs come back NaN).
    confidence_level : float
        Central mass of the bootstrap percentile interval (default 0.95).
    bt_smoothing : float
        Phantom-tie pseudocount per pair for a finite Elo when a model always/never
        wins.
    max_chars : int
        Truncate each document to this many characters in the judge prompt.
    seed : int
        Seeds document sampling and A/B presentation order (the LLM is still bounded).
    prompts : optional dict
        Override the editable templates (keys ``"judge"``, ``"summary"``); defaults to
        :data:`LLM_EVAL_PROMPTS`.

    Returns
    -------
    JudgeResult
        The Elo ranking, bootstrap CIs, win matrix, and the raw comparison records
        (see :class:`JudgeResult`).
    """
    if not isinstance(models, dict):
        raise TypeError("models must be a dict {name: fitted_model}")
    names = list(models)
    if len(names) < 2:
        raise ValueError("judge needs at least two models to compare")
    if representation not in ("summary", "words"):
        raise ValueError("representation must be 'summary' or 'words'")
    backend = _resolve_llm_call(backend)
    P = prompts or LLM_EVAL_PROMPTS
    judge_tmpl, summary_tmpl = P["judge"], P["summary"]
    ds = _dataset_clause(dataset_description)
    texts = _doc_texts(docs)
    D = len(texts)
    thetas = {}
    for nm, m in models.items():
        th = np.asarray(m.doc_topic, dtype=float)
        if th.shape[0] != D:
            raise ValueError(
                f"model {nm!r} has {th.shape[0]} doc-topic rows but there are {D} "
                "documents; every model must be fit on the same docs, in order")
        thetas[nm] = th

    # judge aligns doc_topic row d to docs[d] and cannot otherwise verify that every
    # model saw the same documents in the same order. The row-count check above only
    # catches a length mismatch, not two models fit on *different* same-length slices
    # (a silently invalid ranking). Differing vocabularies are a strong signal of
    # that mistake, so warn on it -- the cheapest guard short of a stored corpus hash.
    vocabs = {}
    for nm, m in models.items():
        try:
            vocabs[nm] = tuple(m.vocabulary)
        except Exception:
            pass
    if len(set(vocabs.values())) > 1:
        warnings.warn(
            "the models have different vocabularies, which usually means they were "
            "fit on different corpora; judge assumes every model was fit on the same "
            "documents in the same order (it aligns doc_topic row d to docs[d]) and "
            "the ranking is invalid otherwise. Re-fit all models on one corpus.",
            stacklevel=2)

    idx = {nm: i for i, nm in enumerate(names)}
    M = len(names)
    win = np.zeros((M, M))
    cache: dict = {}
    comparisons: list = []
    rng = np.random.RandomState(seed)

    for a, b in combinations(names, 2):
        for _ in range(max(1, n_comparisons)):
            d = int(rng.randint(D))
            flip = bool(rng.randint(2))  # randomize A/B slot to blunt position bias
            first, second = (b, a) if flip else (a, b)
            set_a = _render_topic_set(models[first],
                                      _top_topics_for_doc(thetas[first][d], q, p),
                                      representation, backend, ds, n_words,
                                      summary_tmpl, cache)
            set_b = _render_topic_set(models[second],
                                      _top_topics_for_doc(thetas[second][d], q, p),
                                      representation, backend, ds, n_words,
                                      summary_tmpl, cache)
            reply = backend(judge_tmpl.format(
                dataset=ds, document=str(texts[d])[:max_chars],
                set_a=set_a, set_b=set_b))
            choice = _parse_judge_choice(reply)
            if choice == "a":
                winner = first
            elif choice == "b":
                winner = second
            else:
                winner = None  # tie / unparseable -> split credit
            comparisons.append({"doc": d, "model_a": first, "model_b": second,
                                "choice": choice, "winner": winner,
                                "reasoning": str(reply).strip()})
            if winner == a:
                win[idx[a], idx[b]] += 1.0
            elif winner == b:
                win[idx[b], idx[a]] += 1.0
            else:
                win[idx[a], idx[b]] += 0.5
                win[idx[b], idx[a]] += 0.5

    elo_arr = _bt_to_elo(_bradley_terry(win, smoothing=bt_smoothing))
    elo = {nm: float(elo_arr[i]) for i, nm in enumerate(names)}

    ci = {nm: (float("nan"), float("nan")) for nm in names}
    if bootstrap and bootstrap > 0 and comparisons:
        C = len(comparisons)
        boot = np.empty((bootstrap, M))
        for bi in range(bootstrap):
            bw = np.zeros((M, M))
            for k in rng.randint(0, C, size=C):
                rec = comparisons[k]
                w, ma, mb = rec["winner"], rec["model_a"], rec["model_b"]
                if w is None:
                    bw[idx[ma], idx[mb]] += 0.5
                    bw[idx[mb], idx[ma]] += 0.5
                else:
                    loser = mb if w == ma else ma
                    bw[idx[w], idx[loser]] += 1.0
            boot[bi] = _bt_to_elo(_bradley_terry(bw, smoothing=bt_smoothing))
        alpha = (1.0 - confidence_level) / 2.0
        lo, hi = np.quantile(boot, alpha, axis=0), np.quantile(boot, 1 - alpha, axis=0)
        ci = {nm: (float(lo[i]), float(hi[i])) for i, nm in enumerate(names)}

    return JudgeResult(elo=elo, win_matrix=win, bootstrap_ci=ci,
                       comparisons=comparisons, names=names,
                       representation=representation)
