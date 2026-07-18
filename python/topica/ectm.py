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

from dataclasses import dataclass

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
    is the correct measure. It also **ignores the random-walk pooling and prior
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


def _match_anchor_topic(m, anchor):
    """Topic index in ``m`` whose top words overlap ``anchor`` most."""
    best, best_ov = 0, -1
    for k in range(m.num_topics):
        ov = len(anchor & {w for w, _ in m.top_words(len(anchor), topic=k)})
        if ov > best_ov:
            best, best_ov = k, ov
    return best


def _theta_sampler(model, k):
    """Return a callable ``() -> θ`` that draws a fresh topic-proportion vector
    from the fitted logistic-normal, or ``None`` when the parameters to do so are
    unavailable (then the caller falls back to the fitted per-document ``θ̂``).

    A parametric bootstrap must regenerate the topic proportions, not freeze them:
    holding ``θ̂`` fixed and resampling only tokens omits the document-level
    topic-mixture variance, so the refits barely differ and the resulting band is
    **too narrow** (empirically under-covers). We reconstruct the fitted marginal
    ``η ~ N(μ̂, Σ̂)`` from the per-document logistic-normal parameters — ``μ̂`` the
    mean of ``eta_mean`` and, by the law of total variance, ``Σ̂`` the between-doc
    covariance of ``eta_mean`` plus the mean within-doc ``eta_cov`` — then draw
    ``θ = softmax([η, 0])``."""
    em = getattr(model, "eta_mean", None)
    if em is None:
        return None
    em = np.asarray(em, dtype=np.float64)
    if em.ndim != 2 or em.shape[0] < 2 or em.shape[1] != k - 1:
        return None
    mu = em.mean(axis=0)
    between = np.cov(em, rowvar=False).reshape(k - 1, k - 1)
    ec = getattr(model, "eta_cov", None)
    within = np.asarray(ec, dtype=np.float64).mean(axis=0) if ec is not None else 0.0
    cov = between + within + 1e-9 * np.eye(k - 1)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(cov), 0.0, None)))

    def draw(rng):
        eta = np.append(mu + chol @ rng.standard_normal(k - 1), 0.0)
        z = np.exp(eta - eta.max())
        return z / z.sum()

    return draw


def _simulate_from_ectm(model, docs, groups, periods, rng, resample_theta=True):
    """Draw one synthetic corpus from the fitted ECTM's generative model, holding
    each document's length, group and period fixed. Each of a document's tokens
    draws a topic ``k ~ θ_d`` then a word ``v ~ β_{k, g_d, t_d}``.

    With ``resample_theta`` (default), each document's ``θ_d`` is redrawn from the
    fitted logistic-normal (:func:`_theta_sampler`) so the bootstrap propagates
    the document-level topic-mixture variance — without this the parametric band
    under-covers. Falls back to the fitted ``θ̂`` when the logistic-normal
    parameters are unavailable (e.g. ``keep_eta_cov=False``). Returns token lists
    aligned to ``docs`` (``groups``/``periods`` unchanged)."""
    vocab = list(model.vocabulary)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    k = theta.shape[1]
    period_index = {str(p): t for t, p in enumerate(model.periods)}
    sampler = _theta_sampler(model, k) if resample_theta else None
    # β cube per (group, period) label -> (K, V), fetched once.
    beta_cache = {}

    def _beta(g, plabel):
        key = (str(g), plabel)
        if key not in beta_cache:
            beta_cache[key] = np.asarray(model.content_word_dist(g, period_index[plabel]))
        return beta_cache[key]

    sim = []
    for d, doc in enumerate(docs):
        length = len(doc)
        if length == 0:
            sim.append([])
            continue
        if sampler is not None:
            th = sampler(rng)
        else:
            th = theta[d]
            th = th / th.sum() if th.sum() > 0 else np.full(k, 1.0 / k)
        beta = _beta(groups[d], str(periods[d]))
        # tokens per topic, then words per topic — bag of words, order irrelevant.
        z_counts = rng.multinomial(length, th)
        wcount = np.zeros(len(vocab))
        for kk in np.nonzero(z_counts)[0]:
            row = beta[kk]
            row = row / row.sum() if row.sum() > 0 else np.full(len(vocab), 1.0 / len(vocab))
            wcount += rng.multinomial(int(z_counts[kk]), row)
        toks = []
        for j in np.nonzero(wcount)[0]:
            toks.extend([vocab[j]] * int(wcount[j]))
        sim.append(toks)
    return sim


def content_trajectory_ci(refit, docs, groups, periods, *, anchor_words, word, contrast,
                          method="cluster", model=None,
                          clusters=None, n_boot=40, ci=0.95, seed=0):
    """Confidence band for a content trajectory, by cluster or parametric bootstrap.

    The content-side estimates are MAP point values; this resamples and refits to
    put uncertainty on them. ``refit(docs, groups, periods)`` must return a freshly
    fitted :class:`ECTM` with your settings (a closure capturing ``num_topics``,
    ``iters``, the priors, etc.). Topics are not aligned across refits, so each
    bootstrap's topic is matched to the reference by top-word overlap with
    ``anchor_words`` (e.g. ``[w for w, _ in model.top_words(20, topic=k)]``).
    Returns ``(period_label, mean, ci_low, ci_high)`` for the ``contrast``
    trajectory of ``word`` (same ``contrast`` semantics as
    :func:`content_trajectory`).

    ``method`` selects the resampling design:

    - ``"cluster"`` (default): nonparametric cluster bootstrap. ``clusters`` is a
      per-document id (e.g. the source document a paragraph came from, or a
      ``(party, year)`` platform key) resampled *with replacement*; pass it whenever
      documents within a cluster are not independent, so the band reflects the
      number of independent units rather than the number of paragraphs. ``None``
      resamples documents individually. **Prefer this when there are many
      independent clusters per cell.**
    - ``"parametric"``: parametric bootstrap from the fitted generative model. Draw
      ``n_boot`` synthetic corpora from the fit (each document keeps its length,
      group and period; its topic mixture is redrawn from the fitted
      logistic-normal and tokens are generated from that and the per-cell ``β̂`` —
      resampling ``θ`` rather than freezing it is what gives the band nominal
      width), refit each, and take quantiles. The target is well defined — the
      fitted DGP — so coverage is checkable on synthetic ground truth. **Prefer
      this for thin,
      one-observation-per-cell designs** (e.g. one manifesto per ``(party,
      election)``), where the cluster bootstrap cannot preserve the design because
      no resample can redraw whole clusters and keep every cell present; its
      interval then conditions on retaining coverage and has no established nominal
      coverage. Pass the fitted reference in ``model=`` to simulate from it, or
      leave ``model=None`` to fit it once via ``refit``.

    Parameters
    ----------
    model : fitted :class:`~topica.ECTM`, optional
        The reference fit the parametric bootstrap simulates from. Ignored by the
        cluster method. If ``None`` under ``method="parametric"``, it is obtained
        once as ``refit(docs, groups, periods)``.
    """
    if method not in ("cluster", "parametric"):
        raise ValueError(f"method must be 'cluster' or 'parametric', got {method!r}")

    rng = np.random.default_rng(seed)
    docs = list(docs)
    groups = list(groups)
    periods = list(periods)
    n = len(docs)
    anchor = set(anchor_words)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    acc = {}  # period_label -> [values]

    if method == "cluster":
        clusters = list(range(n)) if clusters is None else list(clusters)
        uniq = list(dict.fromkeys(clusters))
        by_cluster = {}
        for i, c in enumerate(clusters):
            by_cluster.setdefault(c, []).append(i)
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = [i for j in pick for i in by_cluster[uniq[j]]]
            m = refit([docs[i] for i in idx], [groups[i] for i in idx], [periods[i] for i in idx])
            best = _match_anchor_topic(m, anchor)
            for p, v in content_trajectory(m, best, word, contrast=contrast):
                acc.setdefault(p, []).append(v)
    else:  # parametric
        ref = model if model is not None else refit(docs, groups, periods)
        for _ in range(n_boot):
            sim_docs = _simulate_from_ectm(ref, docs, groups, periods, rng)
            m = refit(sim_docs, groups, periods)
            best = _match_anchor_topic(m, anchor)
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


# ---------------------------------------------------------------------------
# Debiased content divergence (issue #337)
# ---------------------------------------------------------------------------

def _tv(p, q) -> float:
    """Total-variation distance ``0.5 * sum_v |p_v - q_v|`` between two rows."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def _doc_word_counts(docs, vocab):
    """Dense ``(num_docs, V)`` word-count matrix over ``vocab`` (out-of-vocab
    tokens dropped)."""
    index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for d, toks in enumerate(docs):
        for w in toks:
            j = index.get(w)
            if j is not None:
                counts[d, j] += 1.0
    return counts


def _theta_weighted_dist(counts, theta, topic, doc_idx, alpha=1e-6):
    """The topic-``topic`` word distribution implied by a set of documents:
    the θ-weighted word frequency ``sum_d θ_{d,topic} c_{d,·}`` normalized. This
    is the *data-side* estimate of a (group, period) cell's topic vocabulary —
    consistent for the true cell distribution and, unlike the model's smoothed
    β surface, linear in the counts, so it debiases cleanly."""
    if len(doc_idx) == 0:
        return None
    w = theta[doc_idx, topic][:, None] * counts[doc_idx]
    total = w.sum(axis=0) + alpha
    return total / total.sum()


def content_divergence_debiased(
    model,
    topic: int,
    group_a,
    group_b,
    *,
    docs,
    groups,
    periods,
    n_splits: int = 8,
    se_blocks: int = 0,
    seed: int = 0,
):
    """Finite-sample-debiased content divergence between two groups for ``topic``.

    The raw plug-in :func:`content_divergence` reports ``TV(β̂_a, β̂_b)`` off the
    fitted content surface, which carries a **finite-sample floor**: comparing two
    *estimated* high-dimensional word distributions inflates the total-variation
    distance above the true ``TV(β_a, β_b)`` even when the two groups are identical
    (Gentzkow–Shapiro–Taddy 2019; Lupu 2017). That plug-in — and the
    null-subtraction ``observed − mean(shuffled)`` — is a good screening score but
    estimates no defined population quantity. This estimator differences the noise
    out so the reported contrast targets one.

    **Estimand and consistency.** For each ``(group, period)`` cell this reads the
    topic's word distribution as the θ-weighted empirical word frequency (the fitted
    ``model``'s ``doc_topic`` weights times the raw counts), which is *consistent*
    for the cell's true topic-word distribution. Splitting the cell's documents into
    disjoint halves gives two independent estimates that share the **single** fit's
    topic orientation, so the divergence decomposes as

        E[TV_cross-group] = signal + floor ,   E[TV_within-group] = floor ,

    and the difference isolates the signal. As documents per cell grow the floor
    (and the split's own variance) vanish and the estimate → ``TV(β_a, β_b)``. It
    uses only ``model``, so unlike a cross-fit split-sample it never compares β
    across independently-fit models (whose per-fit non-identifiability leaves a
    *non-vanishing* floor that over-subtracts the signal).

    Parameters
    ----------
    model : fitted :class:`~topica.ECTM`
        The reference fit; supplies ``doc_topic`` (θ), ``vocabulary`` and ``periods``.
    topic : int
        Topic index.
    group_a, group_b : str or int
        The two content groups to contrast.
    docs, groups, periods : array-like (num_docs,)
        The documents and their per-document content/period labels — the same
        corpus, ``content=`` and ``times=`` passed to :meth:`ECTM.fit`.
    n_splits : int
        Random half-splits to average over. More splits lower the estimator's own
        Monte-Carlo variance. Averaging keeps the point estimate stable on thin cells.
    se_blocks : int
        If > 1, also return a **delete-block jackknife** standard error over
        documents (partition docs into ``se_blocks`` blocks, recompute the whole
        per-period estimate leaving each block out, take the jackknife SD). 0
        (default) skips it and returns ``nan`` SEs; sampling confidence *bands* are
        better obtained from :func:`content_trajectory_ci` (issue #340).
    seed : int
        Master RNG seed.

    Returns
    -------
    list of (period_label, estimate, se)
        Per-period debiased divergence, in period order. Note that raw and
        null-subtracted TV do **not** converge to the population divergence as the
        corpus grows; this estimator does.
    """
    docs = list(docs)
    groups = [str(g) for g in groups]
    periods = [_period_label(p) for p in periods]
    n = len(docs)
    if not (len(groups) == len(periods) == n):
        raise ValueError(
            f"docs, groups, periods must have equal length; got "
            f"{n}, {len(groups)}, {len(periods)}"
        )

    vocab = list(model.vocabulary)
    counts = _doc_word_counts(docs, vocab)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    if theta.shape[0] != n:
        raise ValueError(
            f"corpus has {n} documents but model.doc_topic has {theta.shape[0]} "
            "rows; pass the same documents the model was fit on"
        )
    if topic < 0 or topic >= theta.shape[1]:
        raise ValueError(f"topic {topic} out of range [0, {theta.shape[1]})")

    plabels = [str(p) for p in model.periods]
    ga, gb = str(group_a), str(group_b)

    # cell (group, period) -> document indices
    cells = {}
    for i, (g, p) in enumerate(zip(groups, periods)):
        cells.setdefault((g, p), []).append(i)

    def _estimate(active_mask, rng):
        """Debiased per-period vector over the documents flagged in ``active_mask``,
        averaged over ``n_splits`` random cell-stratified half-splits."""
        acc = np.zeros(len(plabels))
        used = 0
        for _ in range(n_splits):
            row = np.full(len(plabels), np.nan)
            ok = True
            for t, pl in enumerate(plabels):
                ia = [i for i in cells.get((ga, pl), []) if active_mask[i]]
                ib = [i for i in cells.get((gb, pl), []) if active_mask[i]]
                if len(ia) < 2 or len(ib) < 2:
                    ok = False
                    break
                rng.shuffle(ia); rng.shuffle(ib)
                a1, a2 = ia[: len(ia) // 2], ia[len(ia) // 2:]
                b1, b2 = ib[: len(ib) // 2], ib[len(ib) // 2:]
                pa1 = _theta_weighted_dist(counts, theta, topic, a1)
                pa2 = _theta_weighted_dist(counts, theta, topic, a2)
                pb1 = _theta_weighted_dist(counts, theta, topic, b1)
                pb2 = _theta_weighted_dist(counts, theta, topic, b2)
                between = 0.5 * (_tv(pa1, pb2) + _tv(pa2, pb1))
                within = 0.5 * (_tv(pa1, pa2) + _tv(pb1, pb2))
                row[t] = between - within
            if ok:
                acc += row
                used += 1
        if used == 0:
            raise RuntimeError(
                "every cell has fewer than 4 documents for one of the groups; "
                "debiasing needs at least 2 documents per half per cell"
            )
        return acc / used

    rng = np.random.default_rng(seed)
    all_active = np.ones(n, dtype=bool)
    est = _estimate(all_active, rng)

    # Optional delete-block jackknife SE over documents.
    se = np.full(len(plabels), np.nan)
    if se_blocks and se_blocks > 1:
        order = list(range(n))
        np.random.default_rng(seed + 1).shuffle(order)
        loo = []
        for b in range(se_blocks):
            drop = set(order[b::se_blocks])
            mask = np.array([i not in drop for i in range(n)])
            try:
                loo.append(_estimate(mask, np.random.default_rng(seed + 100 + b)))
            except RuntimeError:
                continue
        if len(loo) > 1:
            loo = np.vstack(loo)
            g = loo.shape[0]
            # jackknife SD of the delete-block replicates
            se = np.sqrt((g - 1) / g * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))

    return [(pl, float(est[i]), float(se[i])) for i, pl in enumerate(plabels)]


# ---------------------------------------------------------------------------
# Content-divergence placebo (issue #230)
# ---------------------------------------------------------------------------

@dataclass
class ContentPlaceboResult:
    """Result of :func:`content_placebo`: the content-divergence permutation
    placebo for every topic of one fitted ECTM.

    The headline ECTM quantity is the between-group **content** divergence (the
    total-variation distance between two groups' word distributions for a topic,
    via :func:`content_divergence`), averaged over periods. Because each
    (group, period) cell carries its own parameters, the estimated divergence has
    a finite-sample floor above zero even when the two groups are identical. This
    object holds the observed divergence against the permutation null that floor
    is read from.

    Attributes
    ----------
    topics : list[int]
        Reference topic indices, aligned to ``model``.
    topic_names : list[str]
        Topic labels (or ``"topic_t"`` when none are set).
    observed : numpy.ndarray, shape (K,)
        Per-topic observed mean divergence (mean over periods of the reference
        fit's TV distance between ``group_a`` and ``group_b``).
    null : numpy.ndarray, shape (n_perm, K)
        Per-permutation mean divergence. Entries for a topic that a refit could
        not be matched to are ``nan``.
    floor : numpy.ndarray, shape (K,)
        The finite-sample floor: the mean of the (non-``nan``) null for each
        topic.
    pval : numpy.ndarray, shape (K,)
        One-sided p-value per topic (proportion of the null at or above the
        observed divergence), using the ``(1 + count) / (1 + n)`` convention so
        it is never exactly zero.
    group_a, group_b : str
        The two content groups the divergence is measured between.
    """

    topics: list
    topic_names: list
    observed: np.ndarray
    null: np.ndarray
    floor: np.ndarray
    pval: np.ndarray
    group_a: str
    group_b: str

    def as_dict(self) -> list:
        """One plain-dict row per topic (omits the full null array), ready to
        hand to ``pandas.DataFrame``."""
        rows = []
        for i, t in enumerate(self.topics):
            col = self.null[:, i]
            valid = col[~np.isnan(col)]
            rows.append({
                "topic": int(t),
                "topic_name": self.topic_names[i],
                "observed": float(self.observed[i]),
                "floor": float(self.floor[i]),
                "pvalue": float(self.pval[i]),
                "null_std": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
            })
        return rows


def _mean_divergence(model, topic, group_a, group_b):
    """Mean over periods of the per-topic TV divergence between two groups."""
    vals = [d for _, d in content_divergence(model, topic, group_a, group_b)]
    return float(np.mean(vals)) if vals else float("nan")


def content_placebo(
    model,
    corpus,
    groups,
    periods,
    *,
    group_a=None,
    group_b=None,
    n_perm=100,
    within_period=True,
    seed=0,
    iters=None,
    topn=10,
    model_factory=None,
):
    """Permutation placebo for ECTM's between-group content divergence.

    The complement of :func:`topica.permutation_test`. ``permutation_test``
    shuffles a covariate to test topic *prevalence* (does a group talk about a
    topic more); this shuffles the **content** group labels to test topic
    *wording* (do two groups word a topic differently), the quantity ECTM is
    built to estimate. Each (group, period) cell gets its own parameters, so the
    estimated divergence sits above zero even for identical groups; this test
    measures that finite-sample floor and how far the observed divergence clears
    it.

    Each permutation reassigns the group labels (by default within each period,
    so every period keeps its group composition), refits the model from a fresh
    start on the same documents with the shuffled labels, aligns the refit's
    topics to the reference with the Hungarian top-word matcher
    (:func:`~topica.validation._hungarian`), and recomputes each topic's mean
    :func:`content_divergence`. The observed divergence is then compared to that
    null.

    Parameters
    ----------
    model : a fitted :class:`~topica.ECTM`.
        The reference fit. Its type rebuilds each refit
        (``type(model)(num_topics=K, seed=s)``) unless ``model_factory`` is given.
    corpus : list of token lists or a ``Corpus``.
        The documents ``model`` was fit on. Each permutation refits on these same
        documents with shuffled group labels.
    groups : array-like (num_docs,).
        Per-document content (group) labels, the ``content=`` argument passed to
        :meth:`ECTM.fit`. These are what get shuffled.
    periods : array-like (num_docs,).
        Per-document period labels, the ``times=`` argument passed to
        :meth:`ECTM.fit`. Held fixed; with ``within_period=True`` the shuffle
        stays inside each period.
    group_a, group_b : str or int, optional
        The two groups the divergence is measured between. Default to the model's
        two groups; required when the model has more than two.
    n_perm : int
        Number of permutation refits. 100 screens, 500 for publication.
    within_period : bool
        Shuffle labels within each period (default), preserving every period's
        group composition. ``False`` shuffles globally.
    seed : int
        Master RNG seed; permutation seeds derive as ``seed + perm_index + 1``.
    iters : int, optional
        Iterations for each refit. ``None`` uses the model's fit default; pass a
        smaller value to speed up screening.
    topn : int
        Top-word count used to align refit topics to the reference.
    model_factory : callable(seed) -> unfitted model, optional
        Override the default ``type(model)(num_topics=K, seed=s)`` builder. Use it
        to preserve non-default constructor settings (``variational=``,
        ``sigma_shrink=``, ``init=``). With the default ``init="spectral"`` the
        fit is deterministic, so the null's spread comes from the shuffles, not
        the seed.

    Returns
    -------
    :class:`ContentPlaceboResult`
        ``observed`` (K,), ``null`` (n_perm, K), ``floor`` (K,), ``pval`` (K,).

    Notes
    -----
    The p-value is one-sided (the alternative is "more divergent than chance")
    and uses the ``(1 + count) / (1 + n)`` convention, so it is never exactly
    zero. A topic that a refit cannot be matched to contributes ``nan`` to that
    row of the null and is dropped before the floor and p-value are computed.
    """
    # --- resolve corpus to token lists ----------------------------------------
    if hasattr(corpus, "documents"):
        docs = corpus.documents()
    else:
        docs = [list(d) for d in corpus]
    n_docs = len(docs)

    theta_ref = np.asarray(model.doc_topic, dtype=np.float64)
    if theta_ref.shape[0] != n_docs:
        raise ValueError(
            f"corpus has {n_docs} documents but model.doc_topic has "
            f"{theta_ref.shape[0]} rows; pass the same documents the model "
            "was fit on"
        )
    groups = list(groups)
    periods = list(periods)
    if len(groups) != n_docs or len(periods) != n_docs:
        raise ValueError(
            f"groups and periods must each have length {n_docs} (one per "
            f"document); got {len(groups)} and {len(periods)}"
        )

    # --- resolve the two groups to contrast -----------------------------------
    if group_a is None or group_b is None:
        if model.num_groups != 2:
            raise ValueError(
                f"model has {model.num_groups} groups; pass group_a= and group_b= "
                "to choose the two to contrast"
            )
        group_a, group_b = model.groups[0], model.groups[1]

    k = int(model.num_topics)
    from .effects import _match_to_reference, _top_word_strings, _topic_names

    names = _topic_names(model, k)
    _, ref_sets = _top_word_strings(model, topn)

    # --- observed mean divergence per topic -----------------------------------
    observed = np.array(
        [_mean_divergence(model, t, group_a, group_b) for t in range(k)],
        dtype=np.float64,
    )

    # --- default refit factory -------------------------------------------------
    if model_factory is None:
        cls = type(model)

        def model_factory(s, _cls=cls, _k=k):
            try:
                return _cls(num_topics=_k, seed=s)
            except TypeError as exc:
                raise TypeError(
                    f"could not rebuild {_cls.__name__} for the content placebo; "
                    "pass model_factory=callable(seed)->unfitted model"
                ) from exc

    fit_kwargs = {}
    if iters is not None:
        fit_kwargs["iters"] = iters

    # --- period blocks for the within-period shuffle --------------------------
    plabels = np.array([_period_label(p) for p in periods])
    if within_period:
        blocks = [np.where(plabels == pl)[0] for pl in dict.fromkeys(plabels)]
    else:
        blocks = [np.arange(n_docs)]
    groups_arr = np.array([str(g) for g in groups], dtype=object)

    # --- permutation loop ------------------------------------------------------
    null = np.full((n_perm, k), np.nan)
    for perm in range(n_perm):
        rng = np.random.default_rng(seed + perm + 1)
        g_perm = groups_arr.copy()
        for idx in blocks:
            g_perm[idx] = groups_arr[rng.permutation(idx)]

        m_perm = model_factory(seed + perm + 1)
        m_perm.fit(docs, list(plabels), list(g_perm), **fit_kwargs)

        _, perm_sets = _top_word_strings(m_perm, topn)
        match, _, _ = _match_to_reference(ref_sets, perm_sets)
        for ref_t in range(k):
            perm_t = match.get(ref_t)
            if perm_t is not None and perm_t < m_perm.num_topics:
                null[perm, ref_t] = _mean_divergence(m_perm, perm_t, group_a, group_b)

    # --- floor + one-sided p-value --------------------------------------------
    floor = np.full(k, np.nan)
    pval = np.full(k, np.nan)
    for t in range(k):
        col = null[:, t]
        valid = col[~np.isnan(col)]
        if valid.size:
            floor[t] = float(valid.mean())
            pval[t] = float((1 + np.sum(valid >= observed[t])) / (1 + valid.size))

    return ContentPlaceboResult(
        topics=list(range(k)),
        topic_names=[names[t] if t < len(names) else f"topic_{t}" for t in range(k)],
        observed=observed,
        null=null,
        floor=floor,
        pval=pval,
        group_a=str(group_a),
        group_b=str(group_b),
    )


# ---------------------------------------------------------------------------
# Content-prior hyperparameter selection (issue #339)
# ---------------------------------------------------------------------------

@dataclass
class ContentPriorSelection:
    """Result of :func:`tune_content_prior`: the held-out-selected content-prior
    hyperparameters plus the full scan.

    Attributes
    ----------
    best : dict
        The selected ``{"content_prior_var", "interaction_shrink",
        "period_smooth"}``, ready to splat into :meth:`ECTM.fit`.
    best_score : float
        Held-out mean per-document log-likelihood at ``best`` (higher is better).
    table : list[dict]
        One row per grid point — the three hyperparameters plus ``heldout_loglik``
        — sorted best first, ready for ``pandas.DataFrame``.
    n_docs, n_tokens : int
        Scored held-out documents and tokens (shared across grid points).
    """

    best: dict
    best_score: float
    table: list
    n_docs: int
    n_tokens: int


def _ectm_heldout_loglik(model, heldout, times, content):
    """Held-out word log-likelihood for a fitted ECTM, scored the ECTM-native way.

    ``eval_heldout`` needs ``model.transform`` to infer a held-out document's
    topic mixture, which ECTM does not expose. Instead we score each withheld
    token under the document's *own* fitted mixture and its cell's content
    surface: ``p(w) = sum_k θ_{d,k} · β_{k, g_d, t_d, w}``, where ``θ`` is the
    mixture the fit inferred from the retained tokens. Returns
    ``(mean_per_doc_loglik, n_tokens, n_docs)``."""
    vocab = list(model.vocabulary)
    vindex = {w: i for i, w in enumerate(vocab)}
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    period_index = {str(p): t for t, p in enumerate(model.periods)}
    beta_cache = {}

    def _beta(g, plabel):
        key = (str(g), plabel)
        if key not in beta_cache:
            beta_cache[key] = np.asarray(model.content_word_dist(g, period_index[plabel]))
        return beta_cache[key]

    total, n_tokens, n_docs = 0.0, 0, 0
    for di, held in heldout.missing:
        pw = theta[di] @ _beta(content[di], str(times[di]))
        pw = np.clip(pw, 1e-12, None)
        ll, c = 0.0, 0
        for w in held:
            j = vindex.get(w)
            if j is not None:
                ll += float(np.log(pw[j]))
                c += 1
        if c > 0:
            total += ll
            n_tokens += c
            n_docs += 1
    mean = total / n_docs if n_docs else float("nan")
    return mean, n_tokens, n_docs


def tune_content_prior(
    fit,
    docs,
    times,
    content,
    *,
    content_prior_var=(0.5, 1.0, 2.0),
    interaction_shrink=(1.2, 1.5, 2.0),
    period_smooth=(5.0,),
    prop_docs: float = 0.5,
    prop_words: float = 0.5,
    seed: int = 0,
):
    """Select ECTM's content-prior hyperparameters by held-out predictive fit.

    The content-prior knobs — the coefficient variance ``content_prior_var`` (σ²),
    ``interaction_shrink`` and ``period_smooth`` — are otherwise set by hand, and
    the reported divergence *magnitude* moves with them, inviting a
    researcher-degrees-of-freedom criticism. This grid-searches them by a
    within-corpus word-heldout log-likelihood (R stm's ``make.heldout`` /
    ``eval.heldout`` design), so the fit is reproducible without hand-tuning.

    Parameters
    ----------
    fit : callable(docs, times, content, *, content_prior_var, interaction_shrink,
        period_smooth) -> fitted ECTM
        Fits a model with your other settings fixed (a closure capturing
        ``num_topics``, ``iters``, ``seed``, ``init``, ...). Called once per grid
        point on the held-out *training* corpus.
    docs, times, content : array-like (num_docs,)
        The corpus and its per-document period/content labels.
    content_prior_var, interaction_shrink, period_smooth : sequence of float
        Candidate values for each hyperparameter; their Cartesian product is the
        grid. Pass a single-element sequence to hold one fixed.
    prop_docs, prop_words : float
        Fraction of documents sampled and fraction of each sampled document's
        tokens withheld (passed to :func:`~topica.diagnostics.make_heldout`).
    seed : int
        Seed for the held-out split.

    Returns
    -------
    :class:`ContentPriorSelection`
        ``best`` (the selected hyperparameters), ``best_score`` and the full
        ``table``. Report the selected values alongside the fit; refit on the full
        corpus with ``**selection.best``.

    Notes
    -----
    ECTM exposes no ``transform``, so scoring uses an ECTM-native predictive
    likelihood (:func:`_ectm_heldout_loglik`) rather than the generic
    ``eval_heldout``. The held-out split is fixed across grid points, so scores are
    comparable. This is the held-out arm of #339; an empirical-Bayes (type-II ML)
    variant that maximizes the variational bound for the prior variances is a
    natural follow-up.
    """
    from itertools import product

    from .validation import make_heldout

    docs = list(docs)
    times = [str(t) for t in times]
    content = [str(c) for c in content]
    if not (len(times) == len(content) == len(docs)):
        raise ValueError(
            f"docs, times, content must have equal length; got "
            f"{len(docs)}, {len(times)}, {len(content)}"
        )

    heldout = make_heldout(docs, prop_docs=prop_docs, prop_words=prop_words, seed=seed)

    table = []
    n_docs = n_tokens = 0
    for cpv, ishrink, psmooth in product(content_prior_var, interaction_shrink, period_smooth):
        m = fit(heldout.documents, times, content,
                content_prior_var=cpv, interaction_shrink=ishrink, period_smooth=psmooth)
        ll, ntok, ndoc = _ectm_heldout_loglik(m, heldout, times, content)
        n_docs, n_tokens = ndoc, ntok
        table.append({
            "content_prior_var": float(cpv),
            "interaction_shrink": float(ishrink),
            "period_smooth": float(psmooth),
            "heldout_loglik": float(ll),
        })

    table.sort(key=lambda r: r["heldout_loglik"], reverse=True)
    if not table:
        raise ValueError("the hyperparameter grid is empty")
    top = table[0]
    best = {
        "content_prior_var": top["content_prior_var"],
        "interaction_shrink": top["interaction_shrink"],
        "period_smooth": top["period_smooth"],
    }
    return ContentPriorSelection(
        best=best,
        best_score=top["heldout_loglik"],
        table=table,
        n_docs=n_docs,
        n_tokens=n_tokens,
    )
