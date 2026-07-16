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

The models expose the tensor through different doors -- STM/STS via
``topic_word_by_group``, SAGE via its 3-D ``topic_word``, ECTM via
``content_word_dist(group, period)`` -- and :func:`group_topic_word` normalizes
them to one ``(K, G, V)`` array plus group labels. For ECTM the cells are
averaged over periods by default; pass ``period=`` for a single period, which
turns :func:`topic_polarization` into a per-period *trajectory* input.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_topic_word",
    "topic_polarization",
    "group_exclusivity",
    "split_topics",
]


def group_topic_word(model, *, period=None):
    """Normalize any content-covariate model to ``(beta_KGV, group_labels)``.

    ``beta_KGV`` is ``(num_topics, num_groups, num_words)`` with each row a
    probability distribution over words. Raises ``ValueError`` for a model with no
    group-specific topic-word structure (not a content-covariate model, or fit
    without a content covariate).
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


def topic_polarization(model, *, weights=None, period=None) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    beta, groups = group_topic_word(model, period=period)  # (K, G, V)
    G = beta.shape[1]
    if G < 2:
        raise ValueError("topic_polarization needs at least 2 content groups")
    w = np.full(G, 1.0 / G) if weights is None else np.asarray(weights, float) / np.sum(weights)
    mixture = np.einsum("g,kgv->kv", w, beta)          # (K, V)
    h_mix = _entropy(mixture)                           # (K,)
    h_avg = np.einsum("g,kg->k", w, _entropy(beta))     # (K,)
    return (h_mix - h_avg) / np.log(G)


def group_exclusivity(model, *, n=10, summary="min") -> np.ndarray:
    """Group-adjusted exclusivity per topic (a FREX-tensor summary), in [0, 1].

    Extends the usual exclusivity ``beta_{k,v} / sum_j beta_{j,v}`` to the group
    tensor: ``excl_{k,g,v} = beta_{k,g,v} / sum_j sum_g' beta_{j,g',v}``. For each
    ``(k, g)`` it averages the exclusivity of that group's top-``n`` words, then
    reduces across groups by ``summary`` (``"min"`` = worst-case group, the default;
    ``"mean"`` = average). High = the topic stays distinctive in every group's
    sub-vocabulary; low = at least one group's wording overlaps other topics.

    Returns
    -------
    numpy.ndarray, shape (num_topics,)
    """
    beta, groups = group_topic_word(model)              # (K, G, V)
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
