"""Ensemble topic modeling: combine several independent fits into one consensus.

A single topic-model fit is a draw from a noisy procedure — change the seed or a
hyperparameter and the topics shift, sometimes a lot (Hoyle et al. 2022, "Are
Neural Topic Models Broken?"). Combining several independent runs is more reliable
than any one run: across Hoyle et al.'s experiments the ensemble improves on the
median run in 97% of contexts and never loses to the worst. This module builds
that consensus.

It is the natural follow-on to :func:`~topica.select_model`, which fits N runs at a
fixed K. Instead of *picking* the best run with ``plot_models``, ``ensemble``
*combines* all of them.

Three methods are available:

``method="cluster"`` (default) reproduces Hoyle et al. §6. Pool the topics from
every run (m runs of K topics each give m·K topics), measure the pairwise distance
between them — a blend ``lambda_·D(topic-word) + (1-lambda_)·D(doc-topic)`` using a
top-weighted rank distance (Rank-Biased Overlap, or average Jaccard) — cluster the
pooled topics into K groups, and take the element-wise mean within each cluster.
Clustering does not force a one-to-one match, so a topic that splits or merges
across runs is handled naturally, and a cluster only a few runs contributed to is
flagged as low-support.

``method="align"`` is a lighter, fully deterministic alternative (the Miller &
McCoy 2017 / Mäntylä et al. 2018 lineage): align every run's topics one-to-one to
a single reference run (Hungarian matching on the topic-word distributions) and
average the aligned topics. No clustering, no Θ, no λ.

``method="stable"`` reimplements gensim's ``EnsembleLda`` (Brigl 2019). It does not
fix K: it pools the topics, measures an asymmetric masked-cosine distance between
them, runs Checkback DBSCAN (CBDBSCAN) to find dense, reproducible "cores", and
keeps only the clusters with enough cores as *stable topics* (averaging their
members). Unstable topics — those that do not recur densely across runs — are
discarded as noise rather than averaged in, so the number of consensus topics is
discovered from the data. Validated against gensim in ``parity/``.

The result duck-types as a fitted model for the model-neutral analysis surface (it
exposes ``topic_word``, ``doc_topic``, and ``vocabulary``), so the consensus flows
straight into :func:`~topica.coherence`, the diagnostics, and the rest. Each
ensemble topic carries a ``stability`` score and a ``reliable`` flag, so a
consensus topic the individual runs do not actually agree on is marked, not
silently trusted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# A topic is "reliable" when it is both internally consistent (its members agree)
# and well-supported (enough runs found it). Same alignment threshold as the
# bootstrap path in effects.py, so a reliable topic means the same thing across the
# library.
_MIN_STABILITY = 0.5
_MIN_SUPPORT = 0.5     # cluster: fraction of runs that must back a topic
_MIN_MARGIN = 0.1      # align: how much the best match must beat the next-best
_RBO_P = 0.9  # Rank-Biased Overlap persistence; ~0.9 weights roughly the top ten.


@dataclass
class EnsembleResult:
    """Consensus of several topic-model fits, returned by :func:`ensemble`.

    Exposes ``topic_word``, ``doc_topic``, and ``vocabulary`` so it can be passed
    wherever a fitted model is accepted by the model-neutral analysis functions
    (:func:`~topica.coherence`, the diagnostics surface, :func:`~topica.align_topics`).

    Attributes
    ----------
    topic_word : ``(K, V)`` averaged, row-normalized topic-word matrix.
    doc_topic : ``(D, K)`` averaged document-topic matrix, or ``None`` when the
        runs were not fit on the same documents in the same order.
    vocabulary : the shared vocabulary, or ``None`` when raw arrays were passed.
    stability : ``(K,)`` per-topic consistency in ``[0, 1]``. For ``"cluster"`` and
        ``"stable"`` it is one minus the mean pairwise distance among the run
        topics that formed the cluster; for ``"align"`` it is the mean top-word
        Jaccard with the matched run topics. 1.0 means the contributing topics are
        identical -- how many *distinct* runs contributed is recorded separately by
        ``support``. For ``"cluster"`` and ``"stable"``, a topic only one run
        contributed, with nothing to corroborate it, scores 0.0 (a lone run is not
        agreement), so it never looks like a trustworthy consensus.
    support : ``(K,)`` how well-backed each topic is. For ``"cluster"`` and
        ``"stable"`` it is the fraction of runs that contributed a topic to the
        cluster (1.0 = all runs found it); for ``"align"`` it is the match margin
        over the next-best run topic. A small value means few runs really support
        the topic.
    reliable : ``(K,)`` bool — ``stability >= 0.5`` *and* well-supported. An
        unreliable topic is a consensus the individual runs do not agree on; treat
        it with suspicion. (``"stable"`` topics are reproducible by construction,
        so this is usually all ``True`` -- except a single-run core, reachable when
        ``min_cores == 1``, which scores 0.0 and is not reliable.)
    agreement : scalar mean of ``stability`` — an overall "how reproducible is this
        K?" number (``nan`` if ``"stable"`` found no topics).
    method : ``"cluster"``, ``"align"``, or ``"stable"``.
    cluster_sizes : ``(K,)`` number of run topics in each cluster (``"cluster"``
        and ``"stable"``; ``None`` for ``"align"``).
    reference : index of the reference run (``"align"`` only; ``None`` for
        ``"cluster"``).
    n_runs : number of fits combined.
    runs : the input fits, in the order given.
    """

    topic_word: np.ndarray
    doc_topic: np.ndarray | None
    vocabulary: list | None
    stability: np.ndarray
    support: np.ndarray
    reliable: np.ndarray
    agreement: float
    method: str
    cluster_sizes: np.ndarray | None
    reference: int | None
    n_runs: int
    runs: list = field(repr=False)

    def __repr__(self):  # noqa: D105
        K = self.topic_word.shape[0]
        return (
            f"EnsembleResult(method={self.method!r}, n_runs={self.n_runs}, K={K}, "
            f"agreement={self.agreement:.3f}, reliable={int(np.sum(self.reliable))}/{K})"
        )

    def top_words(self, n=10):
        """Top-`n` ``(term, probability)`` pairs per ensemble topic, matching the
        fitted-model contract so the result drops into the analysis surface. The
        term is a word when a vocabulary is known, else the integer term index."""
        phi = self.topic_word
        vocab = list(self.vocabulary) if self.vocabulary is not None else None
        out = []
        for t in range(phi.shape[0]):
            idx = np.argsort(phi[t])[::-1][:n]
            out.append([((vocab[i] if vocab is not None else int(i)), float(phi[t, i])) for i in idx])
        return out


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def _coerce_runs(runs):
    """Accept a list of fits, or a ``SelectModelResult`` (use its ``.models``)."""
    models = getattr(runs, "models", None)
    if models is not None:
        runs = models
    runs = list(runs)
    if len(runs) < 2:
        raise ValueError("need at least two runs to form an ensemble")
    return runs


def _gather(runs):
    """Topic-word arrays, optional document-topic arrays, and the vocabulary.

    Returns ``(betas, thetas, vocab, K, V)``. ``thetas`` is ``None`` unless every
    run exposes a document-topic matrix of the same shape — the only case in which
    averaging per-document proportions across runs is meaningful."""
    from .coherence import _as_doc_topic, _as_topic_word

    betas = [_as_topic_word(r) for r in runs]
    K, V = betas[0].shape
    for b in betas:
        if b.shape != (K, V):
            raise ValueError(
                f"all runs must share the same shape (K, V); got {betas[0].shape} and {b.shape}"
            )

    vocab = getattr(runs[0], "vocabulary", None)
    if vocab is not None:
        vocab = list(vocab)

    # Only fitted models carry a distinct document-topic matrix. A raw (K, V)
    # array has none — _as_doc_topic would pass the array straight through and we
    # would mistake the topic-word matrix for Θ — so require an actual attribute.
    thetas = None
    if all(hasattr(r, "doc_topic") and not isinstance(r, np.ndarray) for r in runs):
        try:
            cand = [_as_doc_topic(r) for r in runs]
            if all(t.shape == cand[0].shape and t.shape[1] == K for t in cand):
                thetas = cand
        except (AttributeError, TypeError, ValueError):
            thetas = None

    return betas, thetas, vocab, K, V


def _normalize_weights(weights, n):
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have length {n}, got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must not be all zero")
    return w / total


def _ranked(vectors, topn):
    """Per-row ranked top-`topn` index lists (descending by value)."""
    return [list(np.argsort(v)[::-1][:topn]) for v in vectors]


def _rbo(a, b, p=_RBO_P):
    """Rank-Biased Overlap (Webber et al. 2010), extrapolated, on two equal-depth
    ranked lists. Top-weighted: early-rank agreement counts most. Returns a
    similarity in ``(0, 1]`` (1 = identical ordering)."""
    k = max(len(a), len(b))
    if k == 0:
        return 1.0
    s = 0.0
    x_d = 0
    for d in range(1, k + 1):
        x_d = len(set(a[:d]) & set(b[:d]))
        s += (x_d / d) * (p ** d)
    return (x_d / k) * (p ** k) + ((1.0 - p) / p) * s


def _avg_jaccard(a, b):
    """Average Jaccard overlap across prefix depths (Greene et al. 2014). Returns a
    similarity in ``[0, 1]``."""
    k = max(len(a), len(b))
    if k == 0:
        return 1.0
    s = 0.0
    for d in range(1, k + 1):
        A, B = set(a[:d]), set(b[:d])
        u = A | B
        s += (len(A & B) / len(u)) if u else 0.0
    return s / k


def _rank_distance(ranked, distance):
    """Pairwise distance matrix among ranked lists, using a top-weighted rank
    similarity (1 - similarity)."""
    sim = {"rbo": _rbo, "jaccard": _avg_jaccard}[distance]
    n = len(ranked)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - sim(ranked[i], ranked[j])
    return D


# ---------------------------------------------------------------------------
# method="cluster" — Hoyle et al. 2022, §6
# ---------------------------------------------------------------------------

def _agglomerative(D, k):
    """Average-linkage agglomerative clustering on a precomputed distance matrix.
    Returns a length-T label array with ``k`` clusters. Deterministic. Hand-rolled
    (numpy only) to keep the library free of a scipy/scikit dependency; the inner
    loop is vectorized, so it is fine for the m·K topics a topic-model ensemble
    produces (cost grows with that total, so very large run counts are slow)."""
    T = D.shape[0]
    if k >= T:
        return np.arange(T)
    D = D.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    sizes = np.ones(T)
    alive = np.ones(T, dtype=bool)
    members = [[i] for i in range(T)]
    n_active = T
    while n_active > k:
        i, j = divmod(int(np.argmin(D)), T)
        ni, nj = sizes[i], sizes[j]
        merged = (ni * D[i] + nj * D[j]) / (ni + nj)  # Lance-Williams average linkage
        merged[~alive] = np.inf
        merged[i] = np.inf
        D[i, :] = merged
        D[:, i] = merged
        D[j, :] = np.inf
        D[:, j] = np.inf
        alive[j] = False
        sizes[i] = ni + nj
        members[i] += members[j]
        members[j] = []
        n_active -= 1
    labels = np.empty(T, dtype=int)
    for ci, i in enumerate(np.flatnonzero(alive)):
        for m in members[i]:
            labels[m] = ci
    return labels


def _ensemble_cluster(runs, betas, thetas, vocab, K, V, *,
                      num_topics, lambda_, distance, topn, weights):
    m = len(runs)
    nc = K if num_topics is None else int(num_topics)
    if nc < 1:
        raise ValueError(f"num_topics must be a positive integer, got {num_topics!r}")

    # Pool every run's topics: one row of the concatenated B̄, one column of Θ̄ each.
    all_beta = np.vstack(betas)                       # (m*K, V)
    run_of = np.repeat(np.arange(m), K)               # which run each pooled topic came from
    w = _normalize_weights(weights, m)

    # Each pooled topic also carries its document-topic column, used both for the
    # blended distance and for the averaged Θ output.
    all_theta = np.hstack(thetas).T if thetas is not None else None  # (m*K, D)

    D = _rank_distance(_ranked(all_beta, topn), distance)
    lam = float(lambda_)
    if all_theta is not None and lam < 1.0:
        D_theta = _rank_distance(_ranked(all_theta, topn), distance)
        D = lam * D + (1.0 - lam) * D_theta
    elif lam < 1.0:
        warnings.warn(
            "runs do not share a document set, so the document-topic distance is "
            "unavailable; using topic-word distance only (lambda_=1).",
            stacklevel=3,
        )
        lam = 1.0

    labels = _agglomerative(D, nc)

    rows_beta, rows_theta = [], [] if thetas is not None else None
    stability, support, sizes = [], [], []
    for c in range(labels.max() + 1):
        members = np.flatnonzero(labels == c)
        wm = w[run_of[members]]
        wm = wm / wm.sum()
        rows_beta.append(np.average(all_beta[members], axis=0, weights=wm))
        if thetas is not None:
            rows_theta.append(np.average(all_theta[members], axis=0, weights=wm))
        # Internal consistency: 1 - mean pairwise distance among members. A
        # cluster only one run contributed to has no *other* run corroborating
        # it, so its consistency is 0.0, not 1.0 -- a lone topic is the *least*
        # trustworthy consensus, and scoring it 1.0 both misreads it and inflates
        # `agreement`. We key on the number of distinct contributing runs, not on
        # member count: two near-identical topics from the *same* run still form
        # an uncorroborated cluster and must not score as agreement.
        n_runs_c = len(np.unique(run_of[members]))
        if n_runs_c > 1:
            sub = D[np.ix_(members, members)]
            stab = 1.0 - sub[np.triu_indices(len(members), 1)].mean()
        else:
            stab = 0.0
        stability.append(float(stab))
        support.append(n_runs_c / m)
        sizes.append(len(members))

    beta_bar = np.vstack(rows_beta)
    stability = np.array(stability)
    support = np.array(support)
    sizes = np.array(sizes, dtype=int)

    n_singletons = int((support <= 1.0 / m).sum())
    if n_singletons:
        warnings.warn(
            f"{n_singletons} of {len(sizes)} consensus topics come from a single "
            "run, with nothing to corroborate them; they score 0.0 stability and "
            "are not marked reliable. Inspect them before trusting them, or filter "
            "the result arrays with res.reliable.",
            stacklevel=3,
        )

    # Order topics by how well-backed they are (support, then consistency), so the
    # most trustworthy consensus topics come first.
    order = np.lexsort((-stability, -support))
    beta_bar = beta_bar[order]
    beta_bar = beta_bar / np.clip(beta_bar.sum(axis=1, keepdims=True), 1e-300, None)
    theta_bar = None
    if thetas is not None:
        theta_bar = np.vstack(rows_theta)[order].T  # back to (D, K)
        theta_bar = theta_bar / np.clip(theta_bar.sum(axis=1, keepdims=True), 1e-300, None)

    stability, support, sizes = stability[order], support[order], sizes[order]
    reliable = (stability >= _MIN_STABILITY) & (support >= _MIN_SUPPORT)
    return EnsembleResult(
        topic_word=beta_bar, doc_topic=theta_bar, vocabulary=vocab,
        stability=stability, support=support, reliable=reliable,
        agreement=float(np.mean(stability)), method="cluster",
        cluster_sizes=sizes, reference=None, n_runs=m, runs=runs,
    )


# ---------------------------------------------------------------------------
# method="align" — reference matching (Miller & McCoy 2017; Mäntylä et al. 2018)
# ---------------------------------------------------------------------------

def _top_sets(beta, topn):
    """Per-topic set of top-`topn` term indices (vocabulary-independent)."""
    return [set(np.argsort(beta[t])[::-1][:topn].tolist()) for t in range(beta.shape[0])]


def _align_cost(ref, other, metric):
    from .validation import align_topics

    pairs = align_topics(ref, other, metric=metric)
    return float(np.mean([d for _, _, d in pairs])) if pairs else float("inf")


def _choose_reference(betas, reference, metric):
    n = len(betas)
    if reference == "first":
        return 0
    if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
        if not 0 <= reference < n:
            raise ValueError(f"reference index {reference} out of range for {n} runs")
        return int(reference)
    if reference != "medoid":
        raise ValueError("reference must be 'medoid', 'first', or a run index")
    cost = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                cost[a, b] = _align_cost(betas[a], betas[b], metric)
    return int(np.argmin(cost.sum(axis=1)))


def _perm_to_ref(ref, beta, metric):
    from .validation import align_topics

    pairs = align_topics(ref, beta, metric=metric)
    perm = [0] * len(pairs)
    for i, j, _ in pairs:
        perm[i] = j
    return perm


def _reliability_from_perm(ref_sets, run_sets, perm):
    """Per-reference-topic Jaccard quality and ambiguity margin under a fixed
    permutation (mirrors effects._match_to_reference, but honors the externally
    chosen alignment so reported quality matches the averaged topics)."""
    k, kb = len(ref_sets), len(run_sets)
    jac = np.zeros((k, kb))
    for i, rs in enumerate(ref_sets):
        for j, bs in enumerate(run_sets):
            union = rs | bs
            jac[i, j] = (len(rs & bs) / len(union)) if union else 0.0
    quality, margin = np.zeros(k), np.zeros(k)
    for i in range(k):
        j = perm[i]
        quality[i] = jac[i, j]
        margin[i] = jac[i, j] - float(np.max(np.delete(jac[i], j))) if kb > 1 else jac[i, j]
    return quality, margin


def _ensemble_align(runs, betas, thetas, vocab, K, V, *,
                    reference, metric, topn, weights):
    n = len(runs)
    ref_idx = _choose_reference(betas, reference, metric)
    ref = betas[ref_idx]
    ref_sets = _top_sets(ref, topn)
    w = _normalize_weights(weights, n)

    aligned_betas = np.empty((n, K, V))
    aligned_thetas = np.empty((n, *thetas[0].shape)) if thetas is not None else None
    qualities = np.zeros((n, K))
    margins = np.zeros((n, K))
    for r in range(n):
        perm = _perm_to_ref(ref, betas[r], metric)
        aligned_betas[r] = betas[r][perm]
        if thetas is not None:
            aligned_thetas[r] = thetas[r][:, perm]
        qualities[r], margins[r] = _reliability_from_perm(ref_sets, _top_sets(betas[r], topn), perm)

    beta_bar = np.tensordot(w, aligned_betas, axes=1)
    beta_bar = beta_bar / np.clip(beta_bar.sum(axis=1, keepdims=True), 1e-300, None)
    theta_bar = None
    if aligned_thetas is not None:
        theta_bar = np.tensordot(w, aligned_thetas, axes=1)
        theta_bar = theta_bar / np.clip(theta_bar.sum(axis=1, keepdims=True), 1e-300, None)

    stability = np.tensordot(w, qualities, axes=1)
    support = np.tensordot(w, margins, axes=1)
    reliable = (stability >= _MIN_STABILITY) & (support >= _MIN_MARGIN)
    return EnsembleResult(
        topic_word=beta_bar, doc_topic=theta_bar, vocabulary=vocab,
        stability=stability, support=support, reliable=reliable,
        agreement=float(np.mean(stability)), method="align",
        cluster_sizes=None, reference=ref_idx, n_runs=n, runs=runs,
    )


# ---------------------------------------------------------------------------
# method="stable" — gensim's EnsembleLda (Brigl 2019): CBDBSCAN stable cores
# ---------------------------------------------------------------------------
# Reimplemented numpy-native and validated against gensim in parity/. Constants
# and defaults mirror gensim.models.ensemblelda so the two agree on identical
# input.

_COSINE_SKIP = 0.05  # gensim's _COSINE_DISTANCE_CALCULATION_THRESHOLD


def _mass_mask(a, threshold):
    """Smallest set of top terms whose cumulative mass reaches ``threshold``
    (gensim ``mass_masking``)."""
    if threshold is None:
        threshold = 0.95
    sorted_a = np.sort(a)[::-1]
    keep = sorted_a.cumsum() < threshold
    if not keep.any():  # a single term already exceeds the threshold
        return a >= sorted_a[0]
    return a >= sorted_a[keep][-1]


def _rank_mask(a, threshold):
    """Top ``threshold`` fraction of terms by rank (gensim ``rank_masking``)."""
    if threshold is None:
        threshold = 0.11
    cut = int(len(a) * threshold)
    return a > np.sort(a)[::-1][cut]


def _cosine_distance(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 1.0
    return 1.0 - float(np.dot(u, v) / (nu * nv))


def _asymmetric_distance(ttda, masking, threshold):
    """Asymmetric masked-cosine distance between every pair of pooled topics, per
    gensim. Topic ``i``'s mask selects its own high-mass terms; the distance to
    ``j`` is the cosine distance restricted to those terms (so it measures whether
    ``i`` is contained in ``j``)."""
    mask_fn = {"mass": _mass_mask, "rank": _rank_mask}[masking]
    T = len(ttda)
    D = np.ones((T, T))
    for i in range(T):
        mask = mask_fn(ttda[i], threshold)
        a = ttda[i][mask]
        for j in range(T):
            if i == j:
                D[i, j] = 0.0
                continue
            b = ttda[j][mask]
            D[i, j] = 1.0 if b.sum() <= _COSINE_SKIP else _cosine_distance(a, b)
    return D


class _CBDBSCANTopic:
    __slots__ = ("is_core", "label", "neighboring_labels", "valid_neighboring_labels")

    def __init__(self):
        self.is_core = False
        self.label = None
        self.neighboring_labels = set()
        self.valid_neighboring_labels = set()


def _cbdbscan(D, eps, min_samples):
    """Checkback DBSCAN (Brigl 2019). Faithful port of gensim's ``CBDBSCAN.fit``:
    grows clusters from the densest topics outward, and the checkback step starts a
    new cluster when a candidate is too far from its parent's neighborhood (<25%
    close), keeping clusters compact. Returns a list of per-topic results."""
    import sys

    T = D.shape[0]
    results = [_CBDBSCANTopic() for _ in range(T)]
    A = D.copy()
    np.fill_diagonal(A, 1.0)

    ordered = [i for _, i in sorted((A[i].min(), i) for i in range(T))]
    state = {"next_label": 0}

    def scan_topic(topic_index, current_label=None, parent_neighbors=None):
        neighbors = [j for _, j in sorted((A[topic_index][j], j) for j in range(T)) if A[topic_index][j] < eps]
        if len(neighbors) >= min_samples:
            results[topic_index].is_core = True
            if current_label is None:
                current_label = state["next_label"]
                state["next_label"] += 1
            else:
                close = A[topic_index][parent_neighbors] < eps
                if close.mean() < 0.25:
                    current_label = state["next_label"]
                    state["next_label"] += 1
            results[topic_index].label = current_label
            for nb in neighbors:
                if results[nb].label is None:
                    ordered.remove(nb)
                    scan_topic(nb, current_label, neighbors + [topic_index])
                results[nb].neighboring_labels.add(current_label)
        else:
            results[topic_index].label = current_label if current_label is not None else -1

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, T * 2 + 100))
    try:
        while ordered:
            scan_topic(ordered.pop(0))
    finally:
        sys.setrecursionlimit(old_limit)
    return results


def _stable_labels(results, num_models, min_cores):
    """Apply gensim's cluster validation to CBDBSCAN results and return the set of
    labels that form stable topics. Mirrors ``_aggregate_topics`` /
    ``_validate_clusters`` / ``_is_valid_core``."""
    # Group cores by label; collect each core's neighboring-label set per cluster.
    by_label = {}
    for t in results:
        if t.is_core:
            by_label.setdefault(t.label, []).append(t)

    clusters = []
    for label, topics in by_label.items():
        nbls = [set(t.neighboring_labels) for t in topics if t.neighboring_labels]
        clusters.append({
            "label": label,
            "num_cores": len(topics),
            "max_nbl": max((len(t.neighboring_labels) for t in topics), default=0),
            "neighboring_labels": nbls,
        })

    def remove_label(label):
        for c in clusters:
            for s in c["neighboring_labels"]:
                s.discard(label)

    clusters.sort(key=lambda c: (c["max_nbl"], c["num_cores"], c["label"]))
    for c in clusters:
        c["valid"] = None
        if c["num_cores"] < min_cores:
            c["valid"] = False
            remove_label(c["label"])
    for c in clusters:
        if c["valid"] is None:
            isolated = sum(s == {c["label"]} for s in c["neighboring_labels"])
            if isolated >= min_cores:
                c["valid"] = True
            else:
                c["valid"] = False
                remove_label(c["label"])

    return {c["label"] for c in clusters if c["valid"]}


def _ensemble_stable(runs, betas, thetas, vocab, K, V, *,
                     eps, min_samples, min_cores, masking, masking_threshold):
    m = len(runs)
    if min_samples is None:
        min_samples = int(m / 2)
    if min_cores is None:
        min_cores = min(3, max(1, int(m / 4 + 1)))
    elif min_cores == 0:
        min_cores = 1
    if masking not in ("mass", "rank"):
        raise ValueError("masking must be 'mass' or 'rank'")

    ttda = np.vstack(betas)                     # (m*K, V), rows already normalized
    run_of = np.repeat(np.arange(m), K)
    all_theta = np.hstack(thetas).T if thetas is not None else None

    D = _asymmetric_distance(ttda, masking, masking_threshold)
    results = _cbdbscan(D, eps, min_samples)
    stable = _stable_labels(results, m, min_cores)

    for t in results:
        t.valid_neighboring_labels = {lbl for lbl in t.neighboring_labels if lbl in stable}
    valid = np.array([t.is_core and t.valid_neighboring_labels == {t.label} for t in results])
    labels = np.array([t.label for t in results])

    rows_beta, rows_theta = [], [] if all_theta is not None else None
    stability, support, sizes = [], [], []
    for lbl in sorted(set(labels[valid].tolist())):
        members = np.flatnonzero(valid & (labels == lbl))
        rows_beta.append(ttda[members].mean(axis=0))
        if all_theta is not None:
            rows_theta.append(all_theta[members].mean(axis=0))
        # A core only one run contributed to (reachable when min_cores == 1) has
        # no other run corroborating it, so its consistency is 0.0, not the
        # vacuous 1.0 that "1 - mean pairwise distance" gives with no pair. This
        # matches the "cluster"/cross_ensemble paths; the gensim-mirroring part is
        # the clustering above, not this post-hoc score.
        n_runs_c = len(np.unique(run_of[members]))
        if n_runs_c > 1:
            sub = D[np.ix_(members, members)]
            stab = 1.0 - sub[~np.eye(len(members), dtype=bool)].mean()
        else:
            stab = 0.0
        stability.append(float(stab))
        support.append(n_runs_c / m)
        sizes.append(len(members))

    if not rows_beta:
        warnings.warn(
            "no stable topic was detected; try a larger eps, more runs, or a lower "
            "min_cores. Returning an empty ensemble.",
            stacklevel=3,
        )
        empty = np.empty((0, V))
        return EnsembleResult(
            topic_word=empty, doc_topic=None, vocabulary=vocab,
            stability=np.empty(0), support=np.empty(0), reliable=np.empty(0, bool),
            agreement=float("nan"), method="stable",
            cluster_sizes=np.empty(0, int), reference=None, n_runs=m, runs=runs,
        )

    beta_bar = np.vstack(rows_beta)
    beta_bar = beta_bar / np.clip(beta_bar.sum(axis=1, keepdims=True), 1e-300, None)
    theta_bar = None
    if rows_theta is not None:
        theta_bar = np.vstack(rows_theta).T
        theta_bar = theta_bar / np.clip(theta_bar.sum(axis=1, keepdims=True), 1e-300, None)

    stability = np.array(stability)
    support = np.array(support)
    sizes = np.array(sizes, dtype=int)
    reliable = (stability >= _MIN_STABILITY) & (support >= _MIN_SUPPORT)
    return EnsembleResult(
        topic_word=beta_bar, doc_topic=theta_bar, vocabulary=vocab,
        stability=stability, support=support, reliable=reliable,
        agreement=float(np.mean(stability)), method="stable",
        cluster_sizes=sizes, reference=None, n_runs=m, runs=runs,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensemble(runs, *, method="cluster", num_topics=None, lambda_=0.5,
             distance="rbo", topn=10, reference="medoid", metric="cosine",
             weights=None, eps=0.1, min_samples=None, min_cores=None,
             masking="mass", masking_threshold=None):
    """Combine several topic-model fits into one consensus model.

    The consensus is more reliable than any single run — it beats the median run
    and rarely loses to the best (Hoyle et al. 2022). This is the natural
    follow-on to :func:`~topica.select_model`: fit N runs, then combine them here
    instead of picking one.

    Parameters
    ----------
    runs : a list of fitted models (or ``(K, V)`` topic-word arrays sharing a
        vocabulary), or a :class:`~topica.validation.SelectModelResult`. All runs
        must share the same K and vocabulary.
    method : ``"cluster"`` (default) reproduces Hoyle et al. §6 — pool the topics
        from all runs, cluster them, and average within each cluster. ``"align"``
        matches every run's topics to one reference run and averages the aligned
        topics (simpler, deterministic, no document-topic distance).
    num_topics : number of consensus topics for ``"cluster"`` (default: the runs'
        K). Ignored by ``"align"`` (which always returns K).
    lambda_ : ``"cluster"`` only — weight on the topic-word distance when pooling
        topics; ``1 - lambda_`` weights the document-topic distance. Falls back to
        ``1.0`` when the runs were not fit on the same documents.
    distance : ``"cluster"`` only — the top-weighted rank distance between topics:
        ``"rbo"`` (Rank-Biased Overlap, default) or ``"jaccard"`` (average Jaccard).
    topn : top-word (and top-document) count for the distances and diagnostics.
    reference : ``"align"`` only — which run anchors the matching. ``"medoid"``
        (default) picks the run that aligns most cheaply to all others; ``"first"``
        uses run 0; an int uses that run.
    metric : ``"align"`` only — topic-word distance for the matching, ``"cosine"``
        (default) or ``"js"``.
    weights : optional per-run weights (length ``n_runs``) for a weighted average —
        e.g. down-weight low-coherence runs. ``None`` (default) weights equally.
        Used by ``"cluster"`` and ``"align"``.
    eps, min_samples, min_cores, masking, masking_threshold : ``"stable"`` only —
        the gensim ``EnsembleLda`` knobs. ``eps`` (default 0.1) is the CBDBSCAN
        neighbor radius; ``min_samples`` (default ``int(n_runs/2)``) the neighbors
        needed to be a core; ``min_cores`` (default ``min(3, n_runs//4 + 1)``) the
        cores a cluster needs to count as a stable topic; ``masking`` is ``"mass"``
        (default) or ``"rank"`` and ``masking_threshold`` its cutoff (gensim
        defaults 0.95 / 0.11). A larger ``eps`` or smaller ``min_cores`` yields more
        (looser) stable topics.

    Returns
    -------
    An :class:`EnsembleResult`. It exposes ``topic_word``, ``doc_topic``, and
    ``vocabulary``, so it passes straight into :func:`~topica.coherence`, the
    diagnostics, and other model-neutral analyses. Per-topic ``stability`` and
    ``reliable`` flags mark consensus topics the individual runs do not agree on.
    """
    runs = _coerce_runs(runs)
    betas, thetas, vocab, K, V = _gather(runs)

    if method == "cluster":
        if distance not in ("rbo", "jaccard"):
            raise ValueError("distance must be 'rbo' or 'jaccard'")
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_!r}")
        return _ensemble_cluster(
            runs, betas, thetas, vocab, K, V,
            num_topics=num_topics, lambda_=lambda_, distance=distance,
            topn=topn, weights=weights,
        )
    if method == "align":
        return _ensemble_align(
            runs, betas, thetas, vocab, K, V,
            reference=reference, metric=metric, topn=topn, weights=weights,
        )
    if method == "stable":
        return _ensemble_stable(
            runs, betas, thetas, vocab, K, V,
            eps=eps, min_samples=min_samples, min_cores=min_cores,
            masking=masking, masking_threshold=masking_threshold,
        )
    raise ValueError("method must be 'cluster', 'align', or 'stable'")


def cross_ensemble(
    models,
    texts=None,
    *,
    method="cluster",
    num_topics=None,
    lambda_=0.5,
    distance="rbo",
    topn=10,
    weights=None,
) -> EnsembleResult:
    """Combine several topic-model fits from different architectures into one consensus.

    Unlike ``ensemble``, ``cross_ensemble`` allows combining models with different
    architectures (e.g. classical parametric models like LDA/STM and neural embedding
    models like BERTopic) and automatically intersects/aligns their vocabularies if they
    expose a ``.vocabulary`` attribute.

    Parameters
    ----------
    models : list of fitted model instances.
    texts : optional text corpus / tokenized documents, used for validating document counts.
    method : ``"cluster"`` (default) - pools and clusters the topics.
    num_topics : number of consensus topics (default: median K of the input models).
    lambda_ : weight on topic-word distance vs document-topic distance.
    distance : distance metric for clustering (``"rbo"`` or ``"jaccard"``).
    topn : number of top words to use for distance calculation.
    weights : optional per-model weights.

    Returns
    -------
    An ``EnsembleResult``.
    """
    if len(models) < 2:
        raise ValueError("need at least two models to build an ensemble")
    if method != "cluster":
        raise ValueError("Currently only method='cluster' is supported for cross-model ensembling.")

    from .coherence import _as_topic_word, _as_doc_topic

    # 1. Align vocabularies across all models
    vocabularies = [getattr(m, "vocabulary", None) for m in models]
    if all(v is not None for v in vocabularies):
        vocabularies = [list(v) for v in vocabularies]
        common_vocab = set(vocabularies[0])
        for v in vocabularies[1:]:
            common_vocab = common_vocab.intersection(set(v))
        common_vocab = sorted(list(common_vocab))
        if len(common_vocab) == 0:
            raise ValueError("The models share no common vocabulary terms.")
        
        # Project topic_word distributions to common_vocab
        betas = []
        for i, m in enumerate(models):
            raw_beta = _as_topic_word(m)
            idx_map = {w: idx for idx, w in enumerate(vocabularies[i])}
            idx_proj = [idx_map[w] for w in common_vocab]
            proj_beta = raw_beta[:, idx_proj]
            # Re-normalize rows to sum to 1.0
            proj_beta = proj_beta / np.clip(proj_beta.sum(axis=1, keepdims=True), 1e-12, None)
            betas.append(proj_beta)
    else:
        # Fallback if vocabulary attributes are not present
        betas = [_as_topic_word(m) for m in models]
        K_ref, V_ref = betas[0].shape
        for i, b in enumerate(betas[1:]):
            if b.shape[1] != V_ref:
                raise ValueError(
                    f"Models have incompatible vocabulary sizes: {V_ref} and {b.shape[1]}."
                )
        common_vocab = getattr(models[0], "vocabulary", None)
        if common_vocab is not None:
            common_vocab = list(common_vocab)
        else:
            common_vocab = [str(i) for i in range(V_ref)]

    # 2. Extract document-topic distributions
    thetas = []
    has_thetas = True
    for i, m in enumerate(models):
        if hasattr(m, "doc_topic") and m.doc_topic is not None:
            thetas.append(_as_doc_topic(m).T)
        else:
            has_thetas = False
            break

    # Verify document counts and texts length
    if has_thetas:
        num_docs = thetas[0].shape[1]
        for i, t in enumerate(thetas[1:]):
            if t.shape[1] != num_docs:
                raise ValueError(
                    f"Models must share the same document count; model 0 has {num_docs} "
                    f"but model {i+1} has {t.shape[1]}."
                )
        if texts is not None and len(texts) != num_docs:
            raise ValueError(
                f"Model document count ({num_docs}) does not match length of texts ({len(texts)})."
            )
    else:
        thetas = None

    # 3. Pool topics
    m_runs = len(models)
    K_list = [b.shape[0] for b in betas]
    all_beta = np.vstack(betas)                       # (N_total, V_common)
    run_of = np.repeat(np.arange(m_runs), K_list)      # which model each pooled topic came from
    w = _normalize_weights(weights, m_runs)

    all_theta = np.vstack(thetas) if thetas is not None else None  # (N_total, D)

    # 4. Calculate distances
    D = _rank_distance(_ranked(all_beta, topn), distance)
    lam = float(lambda_)
    if all_theta is not None and lam < 1.0:
        D_theta = _rank_distance(_ranked(all_theta, topn), distance)
        D = lam * D + (1.0 - lam) * D_theta
    elif lam < 1.0:
        warnings.warn(
            "Models do not all share document-topic distributions, so document-topic "
            "distance is unavailable; using topic-word distance only (lambda_=1).",
            stacklevel=2,
        )
        lam = 1.0

    # 5. Cluster
    nc = int(num_topics) if num_topics is not None else int(np.median(K_list))
    if nc < 1:
        raise ValueError(f"num_topics must be a positive integer, got {num_topics!r}")

    labels = _agglomerative(D, nc)

    # 6. Construct consensus
    rows_beta, rows_theta = [], [] if thetas is not None else None
    stability, support, sizes = [], [], []
    for c in range(labels.max() + 1):
        members = np.flatnonzero(labels == c)
        wm = w[run_of[members]]
        wm = wm / wm.sum()
        rows_beta.append(np.average(all_beta[members], axis=0, weights=wm))
        if thetas is not None:
            rows_theta.append(np.average(all_theta[members], axis=0, weights=wm))
        
        # Internal consistency: 1 - mean pairwise distance among members. A
        # cluster only one model contributed to has no other model corroborating
        # it, so it scores 0.0, not 1.0: it is the least trustworthy consensus,
        # and scoring it 1.0 both misreads it and inflates `agreement`. We key on
        # the number of distinct contributing models, not on member count: two
        # near-identical topics from the *same* model are not agreement.
        n_runs_c = len(np.unique(run_of[members]))
        if n_runs_c > 1:
            sub = D[np.ix_(members, members)]
            stab = 1.0 - sub[np.triu_indices(len(members), 1)].mean()
        else:
            stab = 0.0
        stability.append(float(stab))
        support.append(n_runs_c / m_runs)
        sizes.append(len(members))

    beta_bar = np.vstack(rows_beta)
    stability = np.array(stability)
    support = np.array(support)
    sizes = np.array(sizes, dtype=int)

    n_singletons = int((support <= 1.0 / m_runs).sum())
    if n_singletons:
        warnings.warn(
            f"{n_singletons} of {len(sizes)} consensus topics come from a single "
            "model, with nothing to corroborate them; they score 0.0 stability and "
            "are not marked reliable. Inspect them before trusting them, or filter "
            "the result arrays with res.reliable.",
            stacklevel=2,
        )

    # Order topics by trustworthiness (support first, then consistency)
    order = np.lexsort((-stability, -support))
    beta_bar = beta_bar[order]
    beta_bar = beta_bar / np.clip(beta_bar.sum(axis=1, keepdims=True), 1e-300, None)
    theta_bar = None
    if thetas is not None:
        theta_bar = np.vstack(rows_theta)[order].T  # back to (D, K)
        theta_bar = theta_bar / np.clip(theta_bar.sum(axis=1, keepdims=True), 1e-300, None)

    reliable = (stability >= _MIN_STABILITY) & (support >= _MIN_SUPPORT)
    return EnsembleResult(
        topic_word=beta_bar, doc_topic=theta_bar, vocabulary=common_vocab,
        stability=stability[order], support=support[order], reliable=reliable[order],
        agreement=float(np.mean(stability)), method="cluster",
        cluster_sizes=sizes[order], reference=None, n_runs=m_runs, runs=models,
    )
