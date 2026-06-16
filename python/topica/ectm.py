"""Interpretation helpers for the Evolving Content Topic Model (:class:`ECTM`).

ECTM fits one topic-word distribution per (group, period) cell. These helpers
read that content surface (``model.content_word_dist(group, period)``) and report
it on the normalized word-probability scale: the words a group uses for a topic
at a point in time, the contrast between two groups, and how that contrast moves
across periods.

``group`` and ``period`` arguments accept either a label (str) or an index (int),
matching :meth:`ECTM.content_word_dist`. Period indices run ``0..num_periods`` in
the sorted period order (``model.periods``).
"""
from __future__ import annotations

import numpy as np


def _cell(model, topic: int, group, period) -> np.ndarray:
    """The word-probability row (length V) for a topic in one (group, period)."""
    if topic < 0 or topic >= model.num_topics:
        raise ValueError(f"topic {topic} out of range [0, {model.num_topics})")
    return np.asarray(model.content_word_dist(group, period))[topic]


def content_words(model, topic: int, group, period, n: int = 10):
    """Top-``n`` ``(word, probability)`` pairs for ``topic`` in one
    (group, period) cell — how this group worded the topic at this time.
    """
    row = _cell(model, topic, group, period)
    vocab = model.vocabulary
    idx = np.argsort(row)[::-1][:n]
    return [(vocab[i], float(row[i])) for i in idx]


def content_contrast(model, topic: int, group_a, group_b, period, n: int = 10):
    """Words most distinctive of ``group_a`` vs ``group_b`` for ``topic`` at
    ``period``, scored by the difference in word probability (prob scale).

    Returns a dict with two ranked lists of ``(word, prob_difference)``:
    ``"toward_<a>"`` (words ``group_a`` uses more) and ``"toward_<b>"``
    (words ``group_b`` uses more). A positive difference favours ``group_a``.
    """
    a = _cell(model, topic, group_a, period)
    b = _cell(model, topic, group_b, period)
    diff = a - b
    vocab = model.vocabulary
    order = np.argsort(diff)
    toward_b = [(vocab[i], float(diff[i])) for i in order[:n]]
    toward_a = [(vocab[i], float(diff[i])) for i in order[::-1][:n]]
    return {f"toward_{group_a}": toward_a, f"toward_{group_b}": toward_b}


def content_trajectory(model, topic: int, word, contrast=None):
    """The trajectory of ``word``'s probability in ``topic`` across all periods.

    ``contrast`` selects what is traced:

    - ``None`` — the period-by-period probability averaged over groups;
    - a single group (label or index) — that group's probability each period;
    - a ``(group_a, group_b)`` pair — the ``a - b`` probability contrast each
      period (the changing lexical gap, the headline ECTM quantity).

    Returns a list of ``(period_label, value)`` in period order.
    """
    vocab = model.vocabulary
    wi = vocab.index(word) if isinstance(word, str) else int(word)
    periods = model.periods
    out = []
    for t in range(len(periods)):
        if contrast is None:
            vals = [
                float(np.asarray(model.content_word_dist(g, t))[topic, wi])
                for g in range(model.num_groups)
            ]
            value = float(np.mean(vals))
        elif isinstance(contrast, (tuple, list)) and len(contrast) == 2:
            a, b = contrast
            value = float(_cell(model, topic, a, t)[wi] - _cell(model, topic, b, t)[wi])
        else:
            value = float(_cell(model, topic, contrast, t)[wi])
        out.append((periods[t], value))
    return out


def content_divergence(model, topic: int, group_a, group_b):
    """Total-variation distance between ``group_a`` and ``group_b``'s word
    distributions for ``topic`` in each period — a single-number summary of how
    far apart the two groups' vocabulary is, traced over time.

    Total variation is ``0.5 * sum_v |p_a(v) - p_b(v)|`` ∈ [0, 1]: 0 when the two
    groups word the topic identically, 1 when they share no vocabulary. Returns a
    list of ``(period_label, tv_distance)`` in period order.
    """
    periods = model.periods
    out = []
    for t in range(len(periods)):
        a = _cell(model, topic, group_a, t)
        b = _cell(model, topic, group_b, t)
        out.append((periods[t], float(0.5 * np.abs(a - b).sum())))
    return out
