"""External validation: score a topic assignment against gold labels.

:func:`agreement` answers the most basic validation question a topic model can be
asked — *given documents I have hand-labeled, how well do the discovered topics
recover those labels?* It reports the standard partition-comparison metrics (ARI,
NMI, homogeneity, completeness, V-measure, and cluster purity), computed from the
two label vectors alone.

This complements :func:`topica.coherence`. Coherence rates the *interpretability*
of a topic's top words; it does not tell you whether documents were assigned to the
right topic, and for embedding-based cluster models it can be actively misleading
(a model can keep tight, coherent top-words while the document partition drifts).
When you have labels to check against, ``agreement`` is the number that tracks
recovery.

The metrics are label-agnostic (invariant to how the cluster/class ids are named),
so they work whether ``pred`` is ``0..k`` cluster ids and ``gold`` is category
codes, or any other integer labeling. The formulas match ``scikit-learn``'s
``adjusted_rand_score``, ``normalized_mutual_info_score`` (arithmetic averaging),
and ``homogeneity_completeness_v_measure``; ``agreement`` needs only numpy.
"""

from __future__ import annotations

import numpy as np


def _contingency(pred, gold):
    """Contingency table with gold classes on rows and predicted clusters on
    columns, plus the two label vectors factorized to dense ``0..`` codes and the
    total count. Raises on length mismatch or empty input."""
    pred = np.asarray(pred).ravel()
    gold = np.asarray(gold).ravel()
    if pred.shape[0] != gold.shape[0]:
        raise ValueError(
            f"pred and gold must be the same length; got {pred.shape[0]} and {gold.shape[0]}"
        )
    n = pred.shape[0]
    if n == 0:
        raise ValueError("pred and gold are empty; nothing to score")
    _, gold_idx = np.unique(gold, return_inverse=True)
    _, pred_idx = np.unique(pred, return_inverse=True)
    table = np.zeros((gold_idx.max() + 1, pred_idx.max() + 1), dtype=np.float64)
    np.add.at(table, (gold_idx, pred_idx), 1.0)
    return table, n


def _comb2(x):
    """Vectorized ``C(x, 2) = x*(x-1)/2`` for a float array (or scalar)."""
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _adjusted_rand(table, n):
    """Adjusted Rand index from a contingency table (matches sklearn)."""
    a = table.sum(axis=1)  # gold class sizes
    b = table.sum(axis=0)  # predicted cluster sizes
    sum_comb = _comb2(table).sum()
    a_comb = _comb2(a).sum()
    b_comb = _comb2(b).sum()
    total_comb = _comb2(np.array([n])).sum()
    expected = a_comb * b_comb / total_comb if total_comb > 0 else 0.0
    max_index = (a_comb + b_comb) / 2.0
    denom = max_index - expected
    if denom == 0.0:
        # Both labelings trivial (each all-in-one or all-singletons) and identical.
        return 1.0
    return float((sum_comb - expected) / denom)


def _entropy(counts, n):
    """Shannon entropy (nats) of a label distribution given its class counts."""
    counts = counts[counts > 0]
    p = counts / n
    return float(-(p * np.log(p)).sum())


def _mutual_info(table, n):
    """Mutual information (nats) between the two labelings."""
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    nz = table > 0
    ii, jj = np.nonzero(nz)
    nij = table[ii, jj]
    # I = sum nij/n * log( (nij * n) / (a_i * b_j) )
    contrib = (nij / n) * np.log((nij * n) / (a[ii] * b[jj]))
    return float(contrib.sum())


def _normalized_mutual_info(table, n):
    """NMI with arithmetic averaging of the two entropies (sklearn default)."""
    h_gold = _entropy(table.sum(axis=1), n)
    h_pred = _entropy(table.sum(axis=0), n)
    mi = _mutual_info(table, n)
    denom = (h_gold + h_pred) / 2.0
    if denom == 0.0:
        # Both labelings have a single class: perfectly (trivially) agree.
        return 1.0
    # Clip tiny negative/over-one values from floating error.
    return float(min(1.0, max(0.0, mi / denom)))


def _homogeneity_completeness_v(table, n):
    """Homogeneity, completeness, and their harmonic mean (V-measure)."""
    h_gold = _entropy(table.sum(axis=1), n)  # H(classes)
    h_pred = _entropy(table.sum(axis=0), n)  # H(clusters)
    mi = _mutual_info(table, n)
    # H(C|K) = H(C) - I(C;K); H(K|C) = H(K) - I(C;K).
    homogeneity = 1.0 if h_gold == 0.0 else mi / h_gold
    completeness = 1.0 if h_pred == 0.0 else mi / h_pred
    if homogeneity + completeness == 0.0:
        v = 0.0
    else:
        v = 2.0 * homogeneity * completeness / (homogeneity + completeness)
    return float(homogeneity), float(completeness), float(v)


def _purity(table, n):
    """Cluster purity: each predicted cluster contributes its majority gold class."""
    # Columns are predicted clusters; take the max gold count within each.
    return float(table.max(axis=0).sum() / n)


def agreement(pred, gold, *, noise="keep"):
    """Score a topic/cluster assignment against gold labels.

    Parameters
    ----------
    pred : array-like of int
        Predicted topic/cluster per document — e.g. ``model.labels`` for a cluster
        model, or ``model.doc_topic.argmax(1)`` for a mixture model.
    gold : array-like of int
        Reference (hand-coded) label per document, aligned one-to-one with
        ``pred``. To score a partially-labeled corpus, pass only the labeled subset:
        ``agreement(pred[mask], gold[mask])``.
    noise : {"keep", "drop"}, default "keep"
        How to treat documents that ``pred`` left unassigned (label ``-1``, the
        HDBSCAN/BERTopic noise bucket). ``"keep"`` scores ``-1`` as its own topic
        (so a large noise bucket is penalized honestly). ``"drop"`` excludes those
        documents before scoring (scores only the assigned ones). Documents with a
        gold label of ``-1`` are dropped under ``"drop"`` as well.

    Returns
    -------
    dict
        ``{"ari", "nmi", "homogeneity", "completeness", "v_measure", "purity"}``.
        ARI is adjusted for chance (0 = random, 1 = identical partitions, and it can
        go slightly negative); the others lie in ``[0, 1]``. ``purity`` is asymmetric
        — it measures how class-pure the predicted clusters are — and, unlike the
        others, is not penalized for splitting one class across many clusters.

    Notes
    -----
    All metrics are invariant to how the labels are named. Values match
    ``scikit-learn`` (``normalized_mutual_info_score`` with arithmetic averaging).
    Pair with :func:`topica.coherence`: coherence for whether the top words read as a
    theme, ``agreement`` for whether the document partition is right.
    """
    if noise not in ("keep", "drop"):
        raise ValueError(f"noise must be 'keep' or 'drop', got {noise!r}")
    pred = np.asarray(pred).ravel()
    gold = np.asarray(gold).ravel()
    if noise == "drop":
        mask = (pred != -1) & (gold != -1)
        pred, gold = pred[mask], gold[mask]
        if pred.shape[0] == 0:
            raise ValueError("no documents left to score after dropping -1 noise")

    table, n = _contingency(pred, gold)
    homogeneity, completeness, v = _homogeneity_completeness_v(table, n)
    return {
        "ari": _adjusted_rand(table, n),
        "nmi": _normalized_mutual_info(table, n),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v,
        "purity": _purity(table, n),
    }
