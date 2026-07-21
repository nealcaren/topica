"""Diagnostics for content-covariate models (STM, STS, SAGE).

These read a fitted model's *group-specific* topic-word tensor beta_{k,g,v} -- how
each topic is worded by each group -- which the global topic-word average hides.

- :func:`topic_polarization` -- per-topic Jensen-Shannon divergence across groups.
  High = the topic is worded very differently by different groups (a group
  difference captured *within* the topic); low = all groups word it alike.
- :func:`group_exclusivity` -- group-adjusted exclusivity (a FREX tensor summary):
  does a topic stay distinctive across every group's sub-vocabulary?
- :func:`split_topics` -- near-duplicate topics pulled apart by group prevalence,
  the signature of a single discourse *fragmenting* into parallel group-topics
  instead of living within one topic.

The models expose the tensor through different doors -- STM via
``topic_word_by_group``, SAGE via its 3-D ``topic_word`` -- and
:func:`group_topic_word` normalizes them to one ``(K, G, V)`` array plus group
labels. For an STM fit with an ordered ``content_time`` covariate the group axis
is the base-by-period cross (labels ``"base@period"``); :func:`content_trajectory`
and :func:`content_divergence` read that surface over ordered time.

STS is the odd one out: its content axis is a *continuous* sentiment rather than
discrete groups, so :func:`group_topic_word` discretizes it, stacking the
topic-word distribution ``topic_word_at(level)`` at a few sentiment levels
(default the poles ``-1``/``0``/``+1`` = negative/neutral/positive). Pass
``levels=`` to choose your own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "group_topic_word",
    "topic_polarization",
    "group_exclusivity",
    "split_topics",
    "stratified_coherence",
    "diagnostics",
    "content_trajectory",
    "content_divergence",
    "ContentTrajectory",
    "ContentDivergence",
]

# STS has no discrete groups -- its content axis is a continuous sentiment. We
# discretize it by evaluating topic_word_at() at these sentiment levels.
_STS_DEFAULT_LEVELS = ((-1.0, "negative"), (0.0, "neutral"), (1.0, "positive"))


def _sts_levels(levels):
    """Normalize an STS ``levels`` argument to an ordered list of ``(value, label)``.

    Accepts ``None`` (the default poles), a mapping ``{value: label}``, or a
    sequence of numeric levels (auto-labeled ``s=<value>``).
    """
    if levels is None:
        return list(_STS_DEFAULT_LEVELS)
    if isinstance(levels, dict):
        return [(float(v), str(lab)) for v, lab in levels.items()]
    return [(float(v), f"s={float(v):+g}") for v in levels]


def _ref_corpus(texts):
    """Normalize a coherence reference to ``list[list[str]]`` (Corpus, raw strings,
    or token lists)."""
    if hasattr(texts, "documents"):
        return texts.documents()
    texts = list(texts)
    if texts and isinstance(texts[0], str):
        return [t.split() for t in texts]
    return [list(t) for t in texts]


def group_topic_word(model, *, levels=None):
    """Normalize any content-covariate model to ``(beta_KGV, group_labels)``.

    ``beta_KGV`` is ``(num_topics, num_groups, num_words)`` with each row a
    probability distribution over words. Raises ``ValueError`` for a model with no
    group-specific topic-word structure (not a content-covariate model, or fit
    without a content covariate).

    Parameters
    ----------
    levels : mapping or sequence, optional
        STS only: the sentiment levels to discretize its continuous content axis
        into groups. A ``{value: label}`` mapping or a sequence of numeric levels;
        defaults to the poles ``-1``/``0``/``+1`` (negative/neutral/positive).
        Ignored by the discrete-group models.
    """
    # STS: continuous sentiment axis -- discretize via topic_word_at(level).
    if hasattr(model, "topic_word_at") and not hasattr(model, "topic_word_by_group"):
        items = _sts_levels(levels)
        beta = np.stack(
            [np.asarray(model.topic_word_at(val)) for val, _ in items], axis=1
        )  # (K, G, V)
        groups = [label for _, label in items]
    else:
        # STM / STS expose topic_word_by_group; SAGE exposes a 3-D topic_word.
        twbg = None
        try:
            twbg = getattr(model, "topic_word_by_group")
        except Exception:
            twbg = None
        if twbg is not None:
            beta = np.asarray(twbg)
        else:
            tw = np.asarray(getattr(model, "topic_word", None))
            if tw.ndim != 3:
                raise ValueError(
                    f"{type(model).__name__} has no group-specific topic-word "
                    "structure; content diagnostics need a content-covariate model "
                    "(STM/STS/SAGE) fit with a content covariate."
                )
            beta = tw
        groups = list(getattr(model, "groups", range(beta.shape[1])))
    beta = np.clip(beta.astype(float), 1e-12, None)
    beta = beta / beta.sum(axis=2, keepdims=True)
    return beta, groups


def _entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis=axis)


def topic_polarization(model, *, weights=None, levels=None) -> np.ndarray:
    """Per-topic Jensen-Shannon divergence across groups, normalized to [0, 1].

    For each topic ``k``, ``JSD(beta_{k,1}, ..., beta_{k,G})`` -- the spread of its
    group-specific wordings. ``0`` = every group words the topic identically;
    ``1`` = groups use disjoint vocabularies for it (maximal framing divergence).

    Parameters
    ----------
    weights : array (num_groups,), optional
        Group weights ``p(g)`` (e.g. group prevalence). Defaults to uniform.
    levels : mapping or sequence, optional
        STS only: sentiment levels to discretize into groups (see
        :func:`group_topic_word`).

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    beta, groups = group_topic_word(model, levels=levels)  # (K, G, V)
    G = beta.shape[1]
    if G < 2:
        raise ValueError("topic_polarization needs at least 2 content groups")
    w = np.full(G, 1.0 / G) if weights is None else np.asarray(weights, float) / np.sum(weights)
    mixture = np.einsum("g,kgv->kv", w, beta)          # (K, V)
    h_mix = _entropy(mixture)                           # (K,)
    h_avg = np.einsum("g,kg->k", w, _entropy(beta))     # (K,)
    return (h_mix - h_avg) / np.log(G)


def group_exclusivity(model, *, n=10, summary="min", levels=None) -> np.ndarray:
    """Group-adjusted exclusivity per topic (a FREX-tensor summary), in [0, 1].

    Extends the usual exclusivity ``beta_{k,v} / sum_j beta_{j,v}`` to the group
    tensor: ``excl_{k,g,v} = beta_{k,g,v} / sum_j sum_g' beta_{j,g',v}``. For each
    ``(k, g)`` it averages the exclusivity of that group's top-``n`` words, then
    reduces across groups by ``summary`` (``"min"`` = worst-case group, the default;
    ``"mean"`` = average). High = the topic stays distinctive in every group's
    sub-vocabulary; low = at least one group's wording overlaps other topics.

    ``levels`` (STS only) chooses the sentiment discretization (see
    :func:`group_topic_word`).

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    beta, groups = group_topic_word(model, levels=levels)  # (K, G, V)
    K, G, V = beta.shape
    denom = beta.sum(axis=(0, 1))                        # (V,)  sum over topics & groups
    excl = beta / np.clip(denom, 1e-12, None)           # (K, G, V) exclusivity per cell
    per_kg = np.empty((K, G))
    for k in range(K):
        for g in range(G):
            top = np.argsort(beta[k, g])[::-1][:n]
            per_kg[k, g] = excl[k, g, top].mean()
    if summary == "min":
        return per_kg.min(axis=1)
    if summary == "mean":
        return per_kg.mean(axis=1)
    raise ValueError(f"summary must be 'min' or 'mean', got {summary!r}")


def split_topics(model, content, *, cos_thresh=0.6, skew_thresh=0.65):
    """Near-duplicate topics pulled apart by group prevalence (fragmentation).

    When a single discourse fragments into parallel group-topics instead of living
    within one topic, you get two topics that (a) share vocabulary -- high cosine
    between their group-averaged topic-word rows -- yet (b) are each dominated by a
    *different* group in the document-topic loadings. This flags those pairs.

    Parameters
    ----------
    content : sequence
        The per-document content-group labels passed to ``fit`` (one per row of
        ``doc_topic``), used to compute each topic's group prevalence.
    cos_thresh : float
        Minimum cosine similarity of the two topics' word rows.
    skew_thresh : float
        Minimum single-group share of a topic's mass for it to count as
        group-skewed.

    Returns
    -------
    list of dict, each ``{"pair": (k, j), "cosine": float, "skew": (float, float),
    "groups": (label_k, label_j)}``.
    """
    # Baseline (group-averaged) topic-word for the vocabulary similarity.
    marginal = getattr(model, "topic_word_marginal", None)
    if marginal is not None:
        phi = np.asarray(marginal)
    else:
        beta, _ = group_topic_word(model)
        phi = beta.mean(axis=1)                          # (K, V)
    phi = phi / np.clip(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12, None)
    cos = phi @ phi.T

    theta = np.asarray(model.doc_topic)                  # (D, K)
    labels = list(getattr(model, "groups", sorted(set(map(str, content)))))
    idx = {str(g): i for i, g in enumerate(labels)}
    gvec = np.array([idx[str(g)] for g in content])
    mass = np.zeros((theta.shape[1], len(labels)))
    for gi in range(len(labels)):
        rows = theta[gvec == gi]
        if len(rows):
            mass[:, gi] = rows.sum(axis=0)
    prev = mass / np.clip(mass.sum(axis=1, keepdims=True), 1e-12, None)
    skew_val, skew_grp = prev.max(axis=1), prev.argmax(axis=1)

    out = []
    K = phi.shape[0]
    for k in range(K):
        for j in range(k + 1, K):
            if (cos[k, j] > cos_thresh and skew_val[k] > skew_thresh
                    and skew_val[j] > skew_thresh and skew_grp[k] != skew_grp[j]):
                out.append({
                    "pair": (k, j),
                    "cosine": float(cos[k, j]),
                    "skew": (float(skew_val[k]), float(skew_val[j])),
                    "groups": (labels[skew_grp[k]], labels[skew_grp[j]]),
                })
    return out


def stratified_coherence(model, texts, content, *, coherence_type="c_v", n=10,
                         weights=None) -> np.ndarray:
    """Group-stratified topic coherence for a content-covariate model.

    Global coherence scores a topic's group-averaged words against the whole
    corpus, which hides that a topic can be coherent for one group and gibberish
    for another. This instead scores each group's *own* top words
    (``TopWords(beta_{k,g})``) against that group's *own* subcorpus
    (:math:`\\mathcal{D}_g`, the documents with ``content == g``), and averages
    the per-group coherences weighted by group prevalence:

    .. math:: \\text{Coherence}(k) = \\sum_g p(g)\\,
              \\text{Coherence}(\\text{TopWords}(\\beta_{k,g}), \\mathcal{D}_g)

    Parameters
    ----------
    texts : Corpus, list of strings, or list of token lists
        The training corpus, aligned row-for-row with ``content``.
    content : sequence
        The per-document content-group labels passed to ``fit``.
    coherence_type : one of ``"u_mass"``, ``"c_uci"``, ``"c_npmi"``, ``"c_v"``.
    weights : array (num_groups,), optional
        Group weights ``p(g)``. Defaults to each group's document share.

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    from .coherence import coherence as _coherence

    beta, groups = group_topic_word(model)              # (K, G, V)
    vocab = list(model.vocabulary)
    toks = _ref_corpus(texts)
    content = [str(c) for c in content]
    if len(toks) != len(content):
        raise ValueError(
            f"texts has {len(toks)} documents but content has {len(content)} labels"
        )
    K, G, _ = beta.shape
    result = np.zeros(K)
    wsum = 0.0
    for gi, g in enumerate(groups):
        sub = [t for t, c in zip(toks, content) if c == str(g)]
        if not sub:
            continue
        w = float(weights[gi]) if weights is not None else float(len(sub))
        topics = [[vocab[i] for i in np.argsort(beta[k, gi])[::-1][:n]] for k in range(K)]
        cg = np.asarray(_coherence(topics, sub, coherence_type=coherence_type, topn=n),
                        dtype=float)
        result += w * cg
        wsum += w
    if wsum == 0.0:
        raise ValueError("no documents matched any content group label")
    return result / wsum


def diagnostics(model, texts=None, content=None, *, n=10, coherence_type="c_v"):
    """One per-topic content-diagnostics table for a content-covariate model.

    Consolidates the content metrics into a single row-per-topic table:
    ``polarization`` (JSD across groups), ``group_exclusivity`` (worst-case group),
    and -- when ``texts`` and ``content`` are supplied -- ``stratified_coherence``.
    Returns a pandas ``DataFrame`` if pandas is available, else a list of row
    dicts.
    """
    pol = topic_polarization(model)
    gex = group_exclusivity(model, n=n, summary="min")
    rows = []
    strat = None
    if texts is not None and content is not None:
        strat = stratified_coherence(model, texts, content, n=n,
                                     coherence_type=coherence_type)
    for k in range(len(pol)):
        row = {"topic": k, "polarization": float(pol[k]),
               "group_exclusivity": float(gex[k])}
        if strat is not None:
            row["stratified_coherence"] = float(strat[k])
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("topic")
    except ImportError:
        return rows


# ===================================================================
# STM content_time reading layer (issue #365): read "how a group words a
# topic over ordered time" off the STM content surface fit with
# `content_time=`. Ports faSTM's
# content_trajectory() / content_divergence() (R/content-trajectory.R).
#
# The STM content_time fit crosses base groups with ordered periods into
# a saturated content axis whose group labels are "base@period" (index
# base*num_time_periods + period). `group_topic_word` returns that whole
# (K, G, V) surface; these readers decode the cross, pick the target
# topic, and read the per-period between-group signal off it.
# ===================================================================


def _parse_content_time_groups(groups):
    """Split ``"base@period"`` content-group labels into the base levels (in first
    appearance order), the periods (ordered numerically when possible), and a
    ``{(base, period): group_index}`` lookup. Raises if the model was not fit with a
    ``content_time`` covariate (no ``@`` in the labels)."""
    groups = [str(g) for g in groups]
    if not any("@" in g for g in groups):
        raise ValueError(
            "model was not fit with a content_time covariate (no 'base@period' "
            "content groups found). Fit STM with content_time= to read a trajectory."
        )
    parsed = [g.split("@", 1) for g in groups]
    bases = list(dict.fromkeys(b for b, _ in parsed))
    period_labels = list(dict.fromkeys(p for _, p in parsed))
    try:
        periods = sorted(period_labels, key=lambda p: float(p))
    except ValueError:
        periods = sorted(period_labels)
    lookup = {(b, p): i for i, (b, p) in enumerate(parsed)}
    return bases, periods, lookup


def _resolve_content_topic(beta_KGV, vocab, topic, anchor_words):
    """Target topic index: the one whose group-averaged top-20 words overlap
    ``anchor_words`` most (so it can be tracked across bootstrap refits that number
    topics differently), or an explicit ``topic``."""
    if anchor_words is not None:
        anchor = {str(w) for w in anchor_words}
        avg = beta_KGV.mean(axis=1)  # (K, V)
        overlaps = [
            sum(vocab[i] in anchor for i in np.argsort(avg[k])[::-1][:20])
            for k in range(avg.shape[0])
        ]
        return int(np.argmax(overlaps))
    if topic is not None:
        return int(topic)
    raise ValueError("supply either `topic` or `anchor_words`")


def _content_groups_default(bases, groups):
    if groups is None:
        if len(bases) < 2:
            raise ValueError("need at least two base content groups to contrast")
        return bases[0], bases[1]
    if len(groups) != 2:
        raise ValueError("`groups` must be a pair of base content-group labels")
    return str(groups[0]), str(groups[1])


def _traj_point(beta_KGV, vocab, words, g1, g2, k, periods, lookup):
    """Per-word, per-period contrast p(w|g1,p) - p(w|g2,p) for topic k. Returns
    (kept_words, estimate array of shape (n_words, n_periods))."""
    widx = {w: i for i, w in enumerate(vocab)}
    kept = [w for w in words if w in widx]
    est = np.full((len(kept), len(periods)), np.nan)
    for r, w in enumerate(kept):
        wi = widx[w]
        for c, p in enumerate(periods):
            i1, i2 = lookup.get((g1, p)), lookup.get((g2, p))
            if i1 is not None and i2 is not None:
                est[r, c] = beta_KGV[k, i1, wi] - beta_KGV[k, i2, wi]
    return kept, est


def _div_point(beta_KGV, g1, g2, k, periods, lookup, measure):
    """Per-period distributional distance between the two groups' topic-word rows."""
    out = np.full(len(periods), np.nan)
    for c, p in enumerate(periods):
        i1, i2 = lookup.get((g1, p)), lookup.get((g2, p))
        if i1 is None or i2 is None:
            continue
        pd_ = beta_KGV[k, i1]
        pr = beta_KGV[k, i2]
        pd_ = pd_ / pd_.sum()
        pr = pr / pr.sum()
        if measure == "hellinger":
            out[c] = np.sqrt(0.5 * np.sum((np.sqrt(pd_) - np.sqrt(pr)) ** 2))
        else:  # total variation
            out[c] = 0.5 * np.sum(np.abs(pd_ - pr))
    return out


def _boot_indices(n, clusters, rng):
    """One bootstrap resample of document indices. With ``clusters`` (a length-n
    label per document) it is a cluster bootstrap: whole clusters are drawn with
    replacement, matching faSTM's ``.boot_index``."""
    if clusters is None:
        return rng.integers(0, n, n)
    clusters = np.asarray(clusters)
    labels = list(dict.fromkeys(clusters.tolist()))
    members = {lab: np.where(clusters == lab)[0] for lab in labels}
    drawn = rng.integers(0, len(labels), len(labels))
    return np.concatenate([members[labels[d]] for d in drawn])


def _subset(seq, idx):
    if seq is None:
        return None
    arr = np.asarray(seq)
    return arr[idx] if arr.ndim == 1 else arr[idx, :]


def _refit_stm(docs, idx, fit_kwargs):
    """Refit an STM content_time model on the resampled documents. ``fit_kwargs``
    carries the constructor + fit arguments (num_topics/seed/content/content_time/
    prevalence/...); the per-document covariate arrays are subset to ``idx``."""
    import topica

    fk = dict(fit_kwargs)
    docs_b = [docs[i] for i in idx]
    model = topica.STM(
        num_topics=int(fk["num_topics"]),
        seed=int(fk.get("seed", 0)),
        init=fk.get("init", "spectral"),
    )
    fit_args = dict(
        content=_subset(fk.get("content"), idx),
        content_names=fk.get("content_names"),
        content_time=_subset(fk.get("content_time"), idx),
        content_smooth=fk.get("content_smooth", 1.0),
        content_prior=fk.get("content_prior", "l2"),
        content_prior_var=fk.get("content_prior_var", 0.5),
        iters=int(fk.get("iters", 500)),
    )
    prev = _subset(fk.get("prevalence"), idx)
    if prev is not None:
        fit_args["prevalence"] = prev
        if fk.get("prevalence_names") is not None:
            fit_args["prevalence_names"] = fk["prevalence_names"]
    model.fit(docs_b, **{k: v for k, v in fit_args.items() if v is not None})
    return model


def _percentile_ci(stack, level):
    """Percentile CIs down axis 0 of a stack of bootstrap replicates (which may
    contain NaN where a period/word was absent in a replicate)."""
    lo = (1.0 - level) / 2.0 * 100.0
    hi = (1.0 + level) / 2.0 * 100.0
    with np.errstate(invalid="ignore"):
        low = np.nanpercentile(stack, lo, axis=0)
        high = np.nanpercentile(stack, hi, axis=0)
    return low, high


@dataclass
class ContentTrajectory:
    """Per-word between-group wording contrast across ordered periods.

    ``estimate`` is ``(n_words, n_periods)`` with ``estimate[i, j] =
    p(word_i | groups[0], periods[j]) - p(word_i | groups[1], periods[j])`` for the
    target ``topic``. ``ci_low`` / ``ci_high`` are the bootstrap percentile bands
    (``None`` unless ``ci=True``).
    """

    words: list
    periods: list
    groups: tuple
    topic: int
    estimate: np.ndarray
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None

    def to_frame(self):
        """Long tidy table: one row per (word, period)."""
        import pandas as pd

        rows = []
        for i, w in enumerate(self.words):
            for j, p in enumerate(self.periods):
                row = {"word": w, "period": p, "estimate": float(self.estimate[i, j])}
                if self.ci_low is not None:
                    row["ci_low"] = float(self.ci_low[i, j])
                    row["ci_high"] = float(self.ci_high[i, j])
                rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class ContentDivergence:
    """Per-period whole-vocabulary distance between two groups' wording of a topic.

    ``divergence[j]`` is the Hellinger (or TV) distance between the two groups'
    topic-word rows at ``periods[j]``. ``ci_low`` / ``ci_high`` are bootstrap bands.
    """

    periods: list
    groups: tuple
    topic: int
    measure: str
    divergence: np.ndarray
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None

    def to_frame(self):
        import pandas as pd

        rows = []
        for j, p in enumerate(self.periods):
            row = {"period": p, "divergence": float(self.divergence[j])}
            if self.ci_low is not None:
                row["ci_low"] = float(self.ci_low[j])
                row["ci_high"] = float(self.ci_high[j])
            rows.append(row)
        return pd.DataFrame(rows)


def content_trajectory(
    model,
    words,
    *,
    groups=None,
    topic=None,
    anchor_words=None,
    ci=False,
    corpus=None,
    fit_kwargs=None,
    cluster=None,
    B=50,
    level=0.95,
    seed=0,
):
    """How two content groups word a topic differently, across ordered time.

    Reads the STM ``content_time`` surface (fit with ``STM.fit(..., content_time=)``)
    for the per-period, per-word contrast ``p(w | group1, period) - p(w | group2,
    period)`` on the target topic -- how the between-group wording contrast moves
    over ordered time, read off the smoothed STM surface rather than independent
    ``(group, period)`` cells.

    Parameters
    ----------
    model : fitted STM (content_time)
        Fit with an ordered ``content_time=`` covariate.
    words : sequence[str]
        Words to trace (silently dropped if out of vocabulary).
    groups : (str, str), optional
        The two base content groups to contrast. Defaults to the first two.
    topic : int, optional
        Target topic. Give this or ``anchor_words``.
    anchor_words : sequence[str], optional
        Identify the target topic by top-word overlap. Required when ``ci=True`` so
        the topic can be realigned across bootstrap refits.
    ci : bool
        Attach a design-preserving bootstrap CI (needs ``corpus`` + ``fit_kwargs``).
    corpus, fit_kwargs, cluster, B, level, seed
        Bootstrap controls. ``corpus`` is the documents (a Corpus or list of token
        lists); ``fit_kwargs`` the STM refit arguments (``num_topics``, ``content``,
        ``content_time``, ``content_prior``, ...); ``cluster`` a length-``num_docs``
        label to resample whole clusters (e.g. speaker) instead of documents.

    Returns
    -------
    ContentTrajectory
    """
    beta, glabels = group_topic_word(model)
    vocab = list(model.vocabulary)
    bases, periods, lookup = _parse_content_time_groups(glabels)
    g1, g2 = _content_groups_default(bases, groups)
    k = _resolve_content_topic(beta, vocab, topic, anchor_words)
    kept, est = _traj_point(beta, vocab, list(words), g1, g2, k, periods, lookup)

    low = high = None
    if ci:
        reps = _bootstrap(
            corpus, fit_kwargs, cluster, B, seed, anchor_words,
            lambda m: _traj_from_model(m, kept, (g1, g2), anchor_words, periods),
            shape=est.shape,
        )
        low, high = _percentile_ci(reps, level)
    return ContentTrajectory(kept, periods, (g1, g2), k, est, low, high)


def content_divergence(
    model,
    *,
    groups=None,
    topic=None,
    anchor_words=None,
    measure="hellinger",
    ci=False,
    corpus=None,
    fit_kwargs=None,
    cluster=None,
    B=50,
    level=0.95,
    seed=0,
):
    """Whole-vocabulary distance between two groups' wording of a topic, per period.

    Reads the STM ``content_time`` surface for the per-period Hellinger (default) or
    total-variation (``measure="tv"``) distance between the two groups' topic-word
    distributions -- the whole-distribution group distance per period, pooled over
    the vocabulary so the aggregate has a usable bootstrap CI where single-word
    contrasts do not. Parameters mirror :func:`content_trajectory`.

    Returns
    -------
    ContentDivergence
    """
    if measure not in ("hellinger", "tv"):
        raise ValueError("measure must be 'hellinger' or 'tv'")
    beta, glabels = group_topic_word(model)
    vocab = list(model.vocabulary)
    bases, periods, lookup = _parse_content_time_groups(glabels)
    g1, g2 = _content_groups_default(bases, groups)
    k = _resolve_content_topic(beta, vocab, topic, anchor_words)
    div = _div_point(beta, g1, g2, k, periods, lookup, measure)

    low = high = None
    if ci:
        reps = _bootstrap(
            corpus, fit_kwargs, cluster, B, seed, anchor_words,
            lambda m: _div_from_model(m, (g1, g2), anchor_words, measure, periods),
            shape=div.shape,
        )
        low, high = _percentile_ci(reps, level)
    return ContentDivergence(periods, (g1, g2), k, measure, div, low, high)


def _traj_from_model(model, words, groups, anchor_words, periods):
    # Evaluate on the ORIGINAL periods (passed in), not the refit's own — a
    # resample can drop a period, and every replicate must line up column-for-column
    # (missing cells become NaN via the lookup).
    beta, glabels = group_topic_word(model)
    vocab = list(model.vocabulary)
    _, _, lookup = _parse_content_time_groups(glabels)
    k = _resolve_content_topic(beta, vocab, None, anchor_words)
    _, est = _traj_point(beta, vocab, words, groups[0], groups[1], k, periods, lookup)
    return est


def _div_from_model(model, groups, anchor_words, measure, periods):
    beta, glabels = group_topic_word(model)
    vocab = list(model.vocabulary)
    _, _, lookup = _parse_content_time_groups(glabels)
    k = _resolve_content_topic(beta, vocab, None, anchor_words)
    return _div_point(beta, groups[0], groups[1], k, periods, lookup, measure)


def _bootstrap(corpus, fit_kwargs, cluster, B, seed, anchor_words, read_fn, shape):
    """Design-preserving bootstrap: resample documents (or clusters), refit the STM
    content_time model, realign the topic by ``anchor_words``, and collect the
    reader statistic. Returns a ``(B_ok, *shape)`` stack of replicates."""
    if anchor_words is None:
        raise ValueError(
            "anchor_words is required when ci=True (to realign the topic across "
            "bootstrap refits)."
        )
    if corpus is None or fit_kwargs is None:
        raise ValueError("ci=True needs corpus= (the documents) and fit_kwargs= "
                         "(the STM refit arguments).")
    docs = corpus.documents() if hasattr(corpus, "documents") else [list(d) for d in corpus]
    n = len(docs)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(int(B)):
        idx = _boot_indices(n, cluster, rng)
        try:
            m = _refit_stm(docs, idx, fit_kwargs)
            reps.append(read_fn(m))
        except Exception:
            continue  # a degenerate resample (e.g. a period drops out) is skipped
    if not reps:
        return np.full((1,) + tuple(shape), np.nan)
    return np.stack(reps, axis=0)
