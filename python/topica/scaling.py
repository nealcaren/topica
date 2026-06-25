"""Intrinsic diagnostics for latent author scales (the ideal-point family: Wordfish,
IdealPointLDA, IdealPointTM, SentenceIdealTM).

These answer "how partisan / how real is the discovered axis?" from the model and the
text alone, with no external scale (no DW-NOMINATE, no party labels):

- :func:`bimodality` — is the position distribution two-camped (polarized) vs one blob?
- :func:`split_half_reliability` — refit the scale on two disjoint halves of each
  author's documents and correlate; this is how reproducible (hence real) the axis is.

Validated on U.S. House press releases: split-half reliability tracks the external
DW-NOMINATE recovery across congresses (it ranks them correctly and approximates the
magnitude), so it is a usable stand-in when no external scale exists. By measurement
theory, reliability also caps how well the axis can correlate with any external score.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bimodality", "split_half_reliability"]


def bimodality(positions) -> float:
    """Sample bimodality coefficient (BC) of a set of latent positions.

    ``BC = (skew^2 + 1) / (kurtosis + 3 (n-1)^2 / ((n-2)(n-3)))`` with excess kurtosis.
    BC ranges in (0, 1]; values above ~0.555 (the uniform-distribution reference)
    indicate a bimodal / two-camp distribution, i.e. the authors split into two poles.
    A unimodal Gaussian sits well below it.

    `positions` is array-like, shape ``(n,)`` or ``(n, 1)`` (e.g. ``model.author_positions``).
    """
    x = np.asarray(positions, dtype=float).reshape(-1)
    n = x.size
    if n < 4:
        raise ValueError("bimodality needs at least 4 positions")
    c = x - x.mean()
    m2 = np.mean(c ** 2)
    if m2 <= 0:
        return 0.0
    skew = np.mean(c ** 3) / m2 ** 1.5
    kurt = np.mean(c ** 4) / m2 ** 2 - 3.0  # excess kurtosis
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew ** 2 + 1.0) / denom)


def _positions_dict(labels, positions) -> dict:
    pos = np.asarray(positions, dtype=float).reshape(len(labels), -1)[:, 0]
    return {lab: float(p) for lab, p in zip(labels, pos)}


def split_half_reliability(fit, group, *, seed: int = 0, repeats: int = 1) -> float:
    """Split-half reliability of a latent author scale, with no external data.

    Each author's units (documents, or per-observation embeddings) are split into two
    disjoint halves; the model is refit on each half independently; the two recovered
    position vectors are correlated over the authors present in both. A high value means
    the axis is a stable, reproducible trait of the author's text (signal, not an
    artifact of one fit). It is model-agnostic — you supply how to fit.

    Parameters
    ----------
    fit : callable(unit_indices) -> (author_labels, positions)
        Fit a *fresh* model on the given subset of unit indices (positions into
        ``group``) and return the author labels and their 1-D positions. For example::

            def fit(idx):
                m = topica.IdealPointLDA(20, seed=1)
                m.fit([docs[i] for i in idx], group=[group[i] for i in idx])
                return m.author_names, m.author_positions[:, 0]

    group : sequence, length n_units
        The author label of each unit (the same `group` passed to ``fit``).
    seed : int
        Seed for the random per-author half split.
    repeats : int
        Number of independent random splits to average over (more = lower variance).

    Returns
    -------
    float
        The mean absolute Pearson correlation between the two half-fits (sign-aligned),
        over authors that appear in both halves. ``nan`` if fewer than 3 such authors.
    """
    group = list(group)
    by_author: dict = {}
    for i, a in enumerate(group):
        by_author.setdefault(a, []).append(i)
    # only authors with >= 2 units can appear in both halves
    splittable = {a: idx for a, idx in by_author.items() if len(idx) >= 2}
    if len(splittable) < 3:
        return float("nan")

    rs = []
    for rep in range(max(1, repeats)):
        rng = np.random.default_rng(seed + rep)
        idx_a, idx_b = [], []
        for idx in splittable.values():
            perm = list(rng.permutation(idx))
            half = len(perm) // 2
            idx_a.extend(perm[:half])
            idx_b.extend(perm[half:])
        pa = _positions_dict(*fit(sorted(idx_a)))
        pb = _positions_dict(*fit(sorted(idx_b)))
        common = [a for a in splittable if a in pa and a in pb]
        if len(common) < 3:
            continue
        va = np.array([pa[a] for a in common])
        vb = np.array([pb[a] for a in common])
        if va.std() == 0 or vb.std() == 0:
            continue
        rs.append(abs(float(np.corrcoef(va, vb)[0, 1])))
    return float(np.mean(rs)) if rs else float("nan")
