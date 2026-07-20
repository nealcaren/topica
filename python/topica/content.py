"""Diagnostics for content-covariate models (STM, STS, SAGE, ECTM).

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
``topic_word_by_group``, SAGE via its 3-D ``topic_word``, ECTM via
``content_word_dist(group, period)`` -- and :func:`group_topic_word` normalizes
them to one ``(K, G, V)`` array plus group labels. For ECTM the cells are
averaged over periods by default; pass ``period=`` for a single period, which
turns :func:`topic_polarization` into a per-period *trajectory* input.

STS is the odd one out: its content axis is a *continuous* sentiment rather than
discrete groups, so :func:`group_topic_word` discretizes it, stacking the
topic-word distribution ``topic_word_at(level)`` at a few sentiment levels
(default the poles ``-1``/``0``/``+1`` = negative/neutral/positive). Pass
``levels=`` to choose your own.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_topic_word",
    "topic_polarization",
    "group_exclusivity",
    "split_topics",
    "stratified_coherence",
    "diagnostics",
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


def group_topic_word(model, *, period=None, levels=None):
    """Normalize any content-covariate model to ``(beta_KGV, group_labels)``.

    ``beta_KGV`` is ``(num_topics, num_groups, num_words)`` with each row a
    probability distribution over words. Raises ``ValueError`` for a model with no
    group-specific topic-word structure (not a content-covariate model, or fit
    without a content covariate).

    Parameters
    ----------
    period : int or str, optional
        ECTM only: evaluate one period instead of the period average.
    levels : mapping or sequence, optional
        STS only: the sentiment levels to discretize its continuous content axis
        into groups. A ``{value: label}`` mapping or a sequence of numeric levels;
        defaults to the poles ``-1``/``0``/``+1`` (negative/neutral/positive).
        Ignored by the discrete-group models.
    """
    # ECTM: (group, period) content cells.
    if hasattr(model, "content_word_dist") and hasattr(model, "num_periods"):
        groups = list(model.groups)
        periods = range(model.num_periods) if period is None else [period]
        cells = []
        for g in range(len(groups)):
            beta = sum(np.asarray(model.content_word_dist(g, t)) for t in periods) / len(list(periods))
            cells.append(beta)
        beta = np.stack(cells, axis=1)  # (K, G, V)
    # STS: continuous sentiment axis -- discretize via topic_word_at(level).
    elif hasattr(model, "topic_word_at") and not hasattr(model, "topic_word_by_group"):
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
                    "(STM/STS/SAGE/ECTM) fit with a content covariate."
                )
            beta = tw
        groups = list(getattr(model, "groups", range(beta.shape[1])))
    beta = np.clip(beta.astype(float), 1e-12, None)
    beta = beta / beta.sum(axis=2, keepdims=True)
    return beta, groups


def _entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis=axis)


def topic_polarization(model, *, weights=None, period=None, levels=None) -> np.ndarray:
    """Per-topic Jensen-Shannon divergence across groups, normalized to [0, 1].

    For each topic ``k``, ``JSD(beta_{k,1}, ..., beta_{k,G})`` -- the spread of its
    group-specific wordings. ``0`` = every group words the topic identically;
    ``1`` = groups use disjoint vocabularies for it (maximal framing divergence).

    Parameters
    ----------
    weights : array (num_groups,), optional
        Group weights ``p(g)`` (e.g. group prevalence). Defaults to uniform.
    period : int or str, optional
        ECTM only: evaluate at one period instead of the period average, so a
        loop over periods yields a polarization *trajectory* per topic.
    levels : mapping or sequence, optional
        STS only: sentiment levels to discretize into groups (see
        :func:`group_topic_word`).

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    beta, groups = group_topic_word(model, period=period, levels=levels)  # (K, G, V)
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
