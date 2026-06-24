"""Corpus-comparison keyword metrics — which words distinguish two groups.

Unlike :class:`~topica.SAGE` and the model-based ``word_contrast``, these
operate directly on two raw token corpora with no fitted model. The headline is
the **Fighting Words** estimator (Monroe, Colaresi & Quinn 2008, *Fightin'
Words*, *Political Analysis*): a log-odds-ratio with an informative Dirichlet
prior that accounts for how a word's sampling variance shrinks with frequency,
so rare words don't dominate the way they do with raw log-ratios.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _counts(corpus, vocab_index):
    c = np.zeros(len(vocab_index), dtype=np.float64)
    for doc in corpus:
        for tok in doc:
            j = vocab_index.get(tok)
            if j is not None:
                c[j] += 1.0
    return c


def _zeta(y_a, y_b, *, prior=0.01, informative=False):
    """The Monroe-Colaresi-Quinn weighted log-odds z-score for two word-count
    vectors over a shared vocabulary.

    ``y_a`` and ``y_b`` may be fractional (e.g. responsibility-weighted counts),
    which is what makes the same estimator reusable for the topic-conditional
    contrast in :func:`topica.contrastive_topics`. Returns the per-word ``ζ``;
    positive marks an A-distinctive word, negative a B-distinctive one. The math
    is Monroe et al. eq. 16-22 and must stay identical to the body of
    :func:`fighting_words`.
    """
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    combined = y_a + y_b
    alpha = prior * combined if informative else np.full(len(combined), float(prior))
    a0 = alpha.sum()
    n_a, n_b = y_a.sum(), y_b.sum()
    odds_a = (y_a + alpha) / (n_a + a0 - y_a - alpha)
    odds_b = (y_b + alpha) / (n_b + a0 - y_b - alpha)
    delta = np.log(odds_a) - np.log(odds_b)
    var = 1.0 / (y_a + alpha) + 1.0 / (y_b + alpha)
    return delta / np.sqrt(var)


def fighting_words(corpus_a, corpus_b, *, prior=0.01, informative=False, min_count=1):
    """Monroe-Colaresi-Quinn *Fighting Words* — words that distinguish corpus A
    from corpus B, with their statistical significance.

    For each word the weighted log-odds-ratio with an informative Dirichlet prior
    is computed and standardized to a z-score ``ζ``:

    ``δ_w = log[(y_Aw+α_w)/(n_A+α₀-y_Aw-α_w)] - log[(y_Bw+α_w)/(n_B+α₀-y_Bw-α_w)]``,
    ``Var(δ_w) ≈ 1/(y_Aw+α_w) + 1/(y_Bw+α_w)``, ``ζ_w = δ_w / √Var(δ_w)``,

    where ``y_·w`` are word counts, ``n_·`` are corpus token totals, and ``α₀ =
    Σ_w α_w``. A large **positive** ``ζ`` marks a word distinctive of corpus A; a
    large negative ``ζ`` marks one distinctive of corpus B. Because the variance
    term grows for rare words, ``|ζ| > 1.96`` is a defensible ~95% cutoff.

    Parameters
    ----------
    corpus_a, corpus_b : sequence of token lists (``list[list[str]]``).
    prior : float, default 0.01
        The Dirichlet pseudocount. With ``informative=False`` it is a symmetric
        prior ``α_w = prior`` for every word. With ``informative=True`` the prior
        is scaled by each word's overall frequency, ``α_w = prior · c_w`` where
        ``c_w`` is the word's combined count — Monroe et al.'s informative
        Dirichlet prior (IDP), which pulls extreme estimates toward the corpus
        background.
    min_count : int, default 1
        Drop words whose combined count across both corpora is below this.

    Returns
    -------
    list[(word, zeta)] sorted by descending ``zeta`` — corpus-A markers at the
    top, corpus-B markers at the bottom.
    """
    vocab = sorted({t for doc in corpus_a for t in doc}
                   | {t for doc in corpus_b for t in doc})
    index = {w: i for i, w in enumerate(vocab)}
    y_a = _counts(corpus_a, index)
    y_b = _counts(corpus_b, index)
    combined = y_a + y_b

    # Compute the statistic over the FULL vocabulary so the corpus totals n_·
    # and prior mass α₀ stay correct; `min_count` only filters what is returned.
    zeta = _zeta(y_a, y_b, prior=prior, informative=informative)

    keep = combined >= float(min_count)
    order = np.argsort(zeta)[::-1]
    return [(vocab[i], float(zeta[i])) for i in order if keep[i]]


def top_fighting_words(corpus_a, corpus_b, *, n=20, **kwargs):
    """Convenience wrapper around :func:`fighting_words` returning the ``n`` most
    distinctive words for each corpus: a dict ``{"a": [...], "b": [...]}`` where
    each value is a list of ``(word, zeta)`` (corpus-B list has the most negative
    z-scores first). Keyword args are passed through to ``fighting_words``.
    """
    scored = fighting_words(corpus_a, corpus_b, **kwargs)
    return {"a": scored[:n], "b": scored[: -n - 1: -1]}


__all__ = ["fighting_words", "top_fighting_words"]
