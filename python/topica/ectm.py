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


def _effective_counts(model, topic, group, period, groups, periods, doc_lengths):
    """Effective number of topic-`topic` tokens in the (group, period) cell:
    sum over the cell's documents of theta_{d,topic} * document length."""
    theta = np.asarray(model.doc_topic)[:, topic]
    dl = np.asarray(doc_lengths, dtype=float)
    glab, plab = str(group), _period_label(period)
    mask = np.array([str(g) == glab for g in groups]) & \
           np.array([_period_label(p) == plab for p in periods])
    return float((theta[mask] * dl[mask]).sum())


def content_contrast_se(model, topic, group_a, group_b, period, groups, periods, doc_lengths, n=10):
    """Analytic standard errors for the per-word content contrast between two
    groups in one period.

    Each cell's topic-word estimate is treated as a proportion measured from its
    effective token count ``N`` (derived from the document-topic weights, theta *
    document length), giving the multinomial sampling variance
    ``beta_v (1 - beta_v) / N`` per word; the contrast SE is the root of the sum
    across the two independent cells. This is the standard word-level delta-method
    SE (as in weighted log-odds / fightin'-words) and is **instant**.

    Two caveats, pulling opposite ways. It **treats each token as independent**, so
    when documents cluster (e.g. paragraphs of one platform, where the real number
    of independent units is the platform count, not the token count) it can badly
    *overstate* precision -- there :func:`content_trajectory_ci` with ``clusters=``
    is the honest measure. It also **ignores the random-walk pooling and prior
    shrinkage**, which pull the other way. Use it as a fast within-cell screen for
    which words separate the groups, not as a final p-value. Returns a list of
    ``(word, contrast, se)`` for the ``n`` words with the largest ``|contrast|``.
    """
    # Resolve the period to its label once (int in range = index, else a value),
    # so content_word_dist and the effective-count match use the same key.
    plab = (model.periods[period] if isinstance(period, int) and 0 <= period < model.num_periods
            else _period_label(period))
    ba = np.asarray(model.content_word_dist(group_a, plab))[topic]
    bb = np.asarray(model.content_word_dist(group_b, plab))[topic]
    na = _effective_counts(model, topic, group_a, plab, groups, periods, doc_lengths)
    nb = _effective_counts(model, topic, group_b, plab, groups, periods, doc_lengths)
    contrast = ba - bb
    var = ba * (1 - ba) / max(na, 1e-9) + bb * (1 - bb) / max(nb, 1e-9)
    se = np.sqrt(var)
    vocab = model.vocabulary
    order = np.argsort(np.abs(contrast))[::-1][:n]
    return [(vocab[i], float(contrast[i]), float(se[i])) for i in order]


def content_trajectory_ci(refit, docs, groups, periods, *, anchor_words, word, contrast,
                          clusters=None, n_boot=40, ci=0.95, seed=0):
    """Cluster-bootstrap confidence band for a content trajectory.

    The content-side estimates are MAP point values; this resamples the data and
    refits to put uncertainty on them. ``refit(docs, groups, periods)`` must return
    a freshly fitted :class:`ECTM` with your settings (a closure capturing
    ``num_topics``, ``iters``, the priors, etc.).

    ``clusters`` is a per-document id (e.g. the source document a paragraph came
    from, or a ``(party, year)`` platform key) that is resampled *with
    replacement*; pass it whenever documents within a cluster are not independent,
    so the band reflects the number of independent units rather than the number of
    paragraphs. ``None`` resamples documents individually.

    Topics are not aligned across refits, so each bootstrap's topic is matched to
    the reference by top-word overlap with ``anchor_words`` (e.g.
    ``[w for w, _ in model.top_words(20, topic=k)]``). Returns a list of
    ``(period_label, mean, ci_low, ci_high)`` for the ``contrast`` trajectory of
    ``word`` (same ``contrast`` semantics as :func:`content_trajectory`).
    """
    rng = np.random.default_rng(seed)
    docs = list(docs)
    groups = list(groups)
    periods = list(periods)
    n = len(docs)
    clusters = list(range(n)) if clusters is None else list(clusters)
    uniq = list(dict.fromkeys(clusters))
    by_cluster = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    anchor = set(anchor_words)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    acc = {}  # period_label -> [values]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = [i for j in pick for i in by_cluster[uniq[j]]]
        m = refit([docs[i] for i in idx], [groups[i] for i in idx], [periods[i] for i in idx])
        # match the topic by top-word overlap with the anchor signature
        best, best_ov = 0, -1
        for k in range(m.num_topics):
            ov = len(anchor & {w for w, _ in m.top_words(len(anchor), topic=k)})
            if ov > best_ov:
                best, best_ov = k, ov
        for p, v in content_trajectory(m, best, word, contrast=contrast):
            acc.setdefault(p, []).append(v)
    out = []
    for p in sorted(acc, key=lambda s: _period_sort_key(s)):
        a = np.array(acc[p])
        out.append((p, float(a.mean()), float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q))))
    return out


def _period_sort_key(label):
    try:
        return (0, float(label))
    except (TypeError, ValueError):
        return (1, label)


def _period_label(v):
    """Format a raw period value the way the fit did (numeric -> int label)."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(v)


def prevalence_by_group(model, groups, periods, topic=None):
    """Descriptive mean topic *prevalence* (share of ``doc_topic``) by
    (group, period) -- the "how often does each group discuss a topic" half of
    the ECTM picture, complementing the content helpers' "how worded".

    ``groups`` and ``periods`` are the per-document arrays passed to
    :meth:`ECTM.fit`, aligned to ``model.doc_topic`` rows. Returns an array of
    mean prevalence indexed in ``model.groups`` x ``model.periods`` order:
    shape ``(num_groups, num_periods, num_topics)``, or ``(num_groups,
    num_periods)`` when ``topic`` is given. Cells with no documents are ``nan``.

    This is the quick descriptive view; for prevalence with standard errors and
    smooth trajectories, fit with a ``prevalence=`` design and use
    :func:`topica.stm.predicted_prevalence` / :func:`topica.estimate_effect`,
    which work on ECTM through its logistic-normal posterior.
    """
    theta = np.asarray(model.doc_topic)
    gi = {g: i for i, g in enumerate(model.groups)}
    pi = {p: i for i, p in enumerate(model.periods)}
    g_idx = np.array([gi.get(str(g), -1) for g in groups])
    p_idx = np.array([pi.get(_period_label(p), -1) for p in periods])
    G, P, K = model.num_groups, model.num_periods, model.num_topics
    out = np.full((G, P, K), np.nan)
    for a in range(G):
        for b in range(P):
            mask = (g_idx == a) & (p_idx == b)
            if mask.any():
                out[a, b] = theta[mask].mean(axis=0)
    return out if topic is None else out[:, :, topic]


def prevalence_contrast(model, topic: int, group_a, group_b, groups, periods):
    """Descriptive prevalence gap ``group_a - group_b`` for ``topic`` in each
    period -- how much more (or less) often one group discusses the topic, traced
    over time. Returns a list of ``(period_label, gap)`` in period order.
    (Sign uses mean ``doc_topic`` shares; for an inferential gap with a CI use
    :func:`topica.stm.predicted_prevalence` with ``contrast=``.)
    """
    pv = prevalence_by_group(model, groups, periods, topic=topic)
    ia = model.groups.index(str(group_a))
    ib = model.groups.index(str(group_b))
    return [(p, float(pv[ia, t] - pv[ib, t])) for t, p in enumerate(model.periods)]


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
