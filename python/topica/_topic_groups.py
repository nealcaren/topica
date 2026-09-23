"""Group the topics of N runs into a runs-by-groups table (issue #803).

TopicCheck (Chuang et al. 2015) aligns many runs of a topic model at once and shows
the result as a grid: one row per *topic group*, one column per run, and in each cell
the run's topic that joined the group (or nothing). Empty cells are the finding: a
group every run reproduces is solid, and a group only a few runs produce is fragile.
:func:`topica.evaluate.align_topics` matches two runs; :func:`topica.ensemble`
averages many runs into one consensus model and discards which run contributed what.
:func:`topic_groups` keeps that assignment map.

The grouping is average-linkage agglomerative clustering of all runs' topics, with
TopicCheck's up-to-one constraint (section 4.1): a group holds at most one topic from
each run, so two topics of the same run are never merged. Merge distances are
recorded once, so :meth:`TopicGroups.cut` re-cuts the same tree at any threshold (the
threshold sweep of TopicCheck's figure 2) without re-clustering.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ["topic_groups", "TopicGroups"]


def _topic_word_and_prevalence(run):
    """A run's (K, V) topic-word matrix, its per-topic mean prevalence (or None), and
    its vocabulary (or None). Accepts a fitted model or a raw (K, V) array."""
    if isinstance(run, np.ndarray) or not hasattr(run, "topic_word"):
        beta = np.asarray(run, dtype=np.float64)
        return beta, None, None
    beta = np.asarray(run.topic_word, dtype=np.float64)
    prevalence = None
    try:
        theta = np.asarray(run.doc_topic, dtype=np.float64)
        if theta.ndim == 2 and theta.shape[1] == beta.shape[0] and theta.shape[0] > 0:
            prevalence = theta.mean(axis=0)
    except Exception:
        prevalence = None
    vocab = getattr(run, "vocabulary", None)
    return beta, prevalence, (list(vocab) if vocab is not None else None)


def _similarity(pooled, metric):
    """Pairwise similarity in [0, 1] among the pooled topic-word rows."""
    if metric == "cosine":
        norm = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return np.clip(norm @ norm.T, 0.0, 1.0)
    if metric == "js":
        p = pooled / np.clip(pooled.sum(axis=1, keepdims=True), 1e-300, None)
        logp = np.log(np.clip(p, 1e-300, None))
        n = p.shape[0]
        sim = np.ones((n, n))
        for i in range(n):
            m = 0.5 * (p[i] + p[i + 1:])
            logm = np.log(np.clip(m, 1e-300, None))
            kl_i = (p[i] * (logp[i] - logm)).sum(axis=1)
            kl_j = (p[i + 1:] * (logp[i + 1:] - logm)).sum(axis=1)
            js = np.clip(0.5 * (kl_i + kl_j) / np.log(2.0), 0.0, 1.0)  # in bits, [0, 1]
            sim[i, i + 1:] = sim[i + 1:, i] = 1.0 - js
        return sim
    raise ValueError(f"unknown metric {metric!r}; use 'cosine' or 'js'")


def _constrained_merges(dist, run_of):
    """Average-linkage agglomeration with a cannot-link between topics of the same run.

    Returns the merge sequence as ``(a, b, distance)`` over pooled-topic cluster ids
    (the surviving id is ``a``). Merging stops when no two clusters can join. The
    distances are non-decreasing: a merged row is a size-weighted average of two rows
    whose finite entries are all at least the current minimum, and a forbidden pair
    stays forbidden because a cluster's set of runs only grows.

    Each row caches its minimum and the column where it occurs, so a merge rescans
    only the rows whose cached minimum it invalidated, rather than the whole matrix.
    Ties resolve exactly as a row-major ``argmin`` over the full matrix would (lowest
    row, then lowest column)."""
    n = dist.shape[0]
    n_runs = int(run_of.max()) + 1
    D = dist.astype(np.float64).copy()
    D[run_of[:, None] == run_of[None, :]] = np.inf
    np.fill_diagonal(D, np.inf)
    has_run = np.zeros((n, n_runs), dtype=bool)  # which runs each cluster contains
    has_run[np.arange(n), run_of] = True
    sizes = np.ones(n)
    alive = np.ones(n, dtype=bool)
    row_arg = np.argmin(D, axis=1)
    row_min = D[np.arange(n), row_arg]
    merges = []
    while True:
        a = int(np.argmin(row_min))
        b = int(row_arg[a])
        d = row_min[a]
        if not np.isfinite(d):
            break
        na, nb = sizes[a], sizes[b]
        row = (na * D[a] + nb * D[b]) / (na + nb)
        has_run[a] |= has_run[b]
        # the merged cluster can no longer join any cluster sharing one of its runs
        row[(has_run[:, has_run[a]]).any(axis=1)] = np.inf
        row[~alive] = np.inf
        row[a] = row[b] = np.inf
        D[a, :] = row
        D[:, a] = row
        D[b, :] = np.inf
        D[:, b] = np.inf
        alive[b] = False
        sizes[a] = na + nb
        row_min[b] = np.inf
        merges.append((a, b, float(d)))

        # refresh the cached minima: row a is new; a row whose minimum sat in column a
        # or b is rescanned; any other row can only improve through its new entry in
        # column a (column b is now inf, and every other entry is unchanged)
        stale = alive & ((row_arg == a) | (row_arg == b))
        stale[a] = alive[a]
        for c in np.flatnonzero(stale):
            row_arg[c] = int(np.argmin(D[c]))
            row_min[c] = D[c, row_arg[c]]
        fresh = alive & ~stale
        fresh[a] = False
        new = D[fresh, a]
        cur = row_min[fresh]
        better = (new < cur) | ((new == cur) & (a < row_arg[fresh]))
        idx = np.flatnonzero(fresh)[better]
        row_min[idx] = new[better]
        row_arg[idx] = a
    return merges


@dataclass
class TopicGroups:
    """Topic groups across N runs, TopicCheck style.

    Attributes
    ----------
    threshold : float
        The similarity cut: two groups merged only while their average-linkage
        similarity was at least this value.
    groups : list of dict
        One entry per group, ordered by solidity then weight (both descending). Each
        has ``members`` (``{run: topic}``), ``solidity`` (share of runs represented),
        ``weight`` (mean prevalence across runs, a run that lacks the topic counting
        zero; ``None`` when the runs carry no document-topic matrix), and
        ``similarity`` (mean pairwise similarity among the members; 1.0 for a
        single-topic group).
    num_runs : int
    """

    threshold: float
    groups: list
    num_runs: int
    _betas: list = field(repr=False)
    _prevalences: list = field(repr=False)
    _vocab: list | None = field(repr=False)
    _sim: np.ndarray = field(repr=False)
    _offsets: np.ndarray = field(repr=False)
    _run_of: np.ndarray = field(repr=False)
    _merges: list = field(repr=False)

    # --- views -----------------------------------------------------------------

    @property
    def solidity(self) -> np.ndarray:
        """Per-group share of runs that contributed a topic, in group order."""
        return np.array([g["solidity"] for g in self.groups])

    @property
    def merge_similarities(self) -> np.ndarray:
        """Similarity at each merge of the full tree, in merge order (non-increasing);
        useful for choosing where to cut."""
        return np.array([1.0 - d for _, _, d in self._merges])

    def assignments(self):
        """A groups-by-runs DataFrame: the topic index each run contributed to each
        group, ``<NA>`` where the run has no topic in the group."""
        import pandas as pd

        rows = [
            [g["members"].get(r, pd.NA) for r in range(self.num_runs)] for g in self.groups
        ]
        return pd.DataFrame(
            rows, columns=[f"run {r}" for r in range(self.num_runs)], dtype="Int64"
        ).rename_axis("group")

    def top_words(self, n=10):
        """Per-group top words from the mean of the members' topic-word rows. Needs
        the runs' vocabulary (fitted models); raw arrays give word indices."""
        out = []
        for g in self.groups:
            mean = np.mean([self._betas[r][t] for r, t in g["members"].items()], axis=0)
            idx = np.argsort(mean)[::-1][:n]
            out.append([self._vocab[i] for i in idx] if self._vocab is not None else list(idx))
        return out

    def to_frame(self, n=8):
        """One row per group: solidity, runs represented, weight, member similarity,
        top words, and the run-to-topic map."""
        import pandas as pd

        words = self.top_words(n)
        return pd.DataFrame(
            {
                "solidity": [g["solidity"] for g in self.groups],
                "runs": [len(g["members"]) for g in self.groups],
                "weight": [g["weight"] for g in self.groups],
                "similarity": [g["similarity"] for g in self.groups],
                "top_words": [" ".join(map(str, w)) for w in words],
                "members": [dict(g["members"]) for g in self.groups],
            }
        ).rename_axis("group")

    def cut(self, threshold):
        """The same tree cut at a different similarity ``threshold`` (no re-clustering),
        for sweeping from rock-solid groups down to fringe ones."""
        return _build(
            self._betas, self._prevalences, self._vocab, self._sim, self._offsets,
            self._run_of, self._merges, threshold,
        )

    # --- rendering -------------------------------------------------------------

    def __str__(self):
        n_solid = int(np.sum(self.solidity == 1.0)) if self.groups else 0
        head = (
            f"TopicGroups: {len(self.groups)} groups across {self.num_runs} runs "
            f"(similarity >= {self.threshold:g}); {n_solid} found in every run"
        )
        words = self.top_words(6)
        lines = [head]
        for i, (g, w) in enumerate(zip(self.groups, words)):
            got = len(g["members"])
            lines.append(
                f"  {i:>3}  {got:>3}/{self.num_runs}  {' '.join(map(str, w))}"
            )
        return "\n".join(lines)

    __repr__ = __str__

    def to_markdown(self, n=8):
        words = self.top_words(n)
        head = "| group | runs | solidity | top words |\n| --- | --- | --- | --- |"
        body = "\n".join(
            f"| {i} | {len(g['members'])}/{self.num_runs} | {g['solidity']:.2f} | "
            f"{' '.join(map(str, w))} |"
            for i, (g, w) in enumerate(zip(self.groups, words))
        )
        return f"{head}\n{body}\n"


def _build(betas, prevalences, vocab, sim, offsets, run_of, merges, threshold):
    """Replay the merges whose similarity is at least ``threshold`` into groups."""
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    m = len(betas)
    parent = np.arange(sim.shape[0])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, d in merges:
        # merge similarities are non-increasing, so the rest are lower still; the
        # tolerance keeps a threshold sitting exactly on a tied value from stopping
        # one merge early through float round-off in the averaged distances
        if 1.0 - d < threshold - 1e-12:
            break
        parent[find(b)] = find(a)

    clusters = {}
    for i in range(sim.shape[0]):
        clusters.setdefault(find(i), []).append(i)

    groups = []
    for ids in clusters.values():
        members = {int(run_of[i]): int(i - offsets[run_of[i]]) for i in ids}
        if len(ids) > 1:
            sub = sim[np.ix_(ids, ids)]
            cohesion = float(sub[~np.eye(len(ids), dtype=bool)].mean())
        else:
            cohesion = 1.0
        if all(p is not None for p in prevalences):
            weight = float(sum(prevalences[r][t] for r, t in members.items()) / m)
        else:
            weight = None
        groups.append(
            {
                "members": dict(sorted(members.items())),
                "solidity": len(members) / m,
                "weight": weight,
                "similarity": cohesion,
            }
        )
    groups.sort(key=lambda g: (-g["solidity"], -(g["weight"] or 0.0), -g["similarity"]))
    return TopicGroups(
        threshold=threshold, groups=groups, num_runs=m, _betas=betas,
        _prevalences=prevalences, _vocab=vocab, _sim=sim, _offsets=offsets,
        _run_of=run_of, _merges=merges,
    )


def topic_groups(runs, *, threshold=0.5, metric="cosine"):
    """Group the topics of several runs into a TopicCheck-style groups-by-runs table.

    Parameters
    ----------
    runs : sequence of fitted models or (K, V) topic-word arrays, or a
        :func:`topica.select.select_model` result (its ``.models``)
        At least two runs over the same vocabulary. The runs may use different ``K``
        (for example a K sweep).
    threshold : float, default 0.5
        Similarity cut in ``[0, 1]``: groups keep merging while the average-linkage
        similarity between them is at least this value. Higher values give tighter,
        more numerous groups. :meth:`TopicGroups.cut` re-cuts at another value without
        re-clustering, and ``merge_similarities`` lists the similarity at every merge.
    metric : {"cosine", "js"}, default "cosine"
        Similarity between two topics' word distributions: cosine, or one minus the
        Jensen-Shannon divergence in bits.

    Returns
    -------
    TopicGroups
        ``assignments()`` is the groups-by-runs grid (empty cells mark a run with no
        topic in the group); ``solidity`` is each group's share of runs represented;
        ``to_frame()`` adds the aggregated top words and prevalence weight.

    Notes
    -----
    Group ``weight`` (mean topic prevalence across runs) needs every run's
    document-topic matrix; with raw topic-word arrays it is ``None``.

    A group never holds two topics from the same run (TopicCheck's up-to-one
    constraint), so a run that splits one theme into two topics contributes one of
    them and leaves the other in a separate, less solid group. Solidity counts runs,
    so with ``m`` runs a group reproduced by every run has solidity 1 and a topic no
    other run reproduces has solidity ``1/m``.
    """
    models = getattr(runs, "models", None)
    runs = list(models if models is not None else runs)
    if len(runs) < 2:
        raise ValueError("topic_groups needs at least two runs")

    betas, prevalences, vocabs = [], [], []
    for r in runs:
        beta, prev, vocab = _topic_word_and_prevalence(r)
        if beta.ndim != 2 or beta.shape[0] < 1:
            raise ValueError(f"each run needs a 2-D (K, V) topic-word matrix, got shape {beta.shape}")
        if not np.all(np.isfinite(beta)) or np.any(beta < 0):
            raise ValueError("topic-word matrices must be finite and non-negative")
        betas.append(beta)
        prevalences.append(prev)
        vocabs.append(vocab)
    V = betas[0].shape[1]
    if any(b.shape[1] != V for b in betas):
        raise ValueError("all runs must share one vocabulary (the same number of columns V)")
    known = [v for v in vocabs if v is not None]
    if known and any(v != known[0] for v in known[1:]):
        raise ValueError(
            "the runs' vocabularies differ; fit every run on the same Corpus so topic-word "
            "columns line up"
        )
    vocab = known[0] if known else None
    have_prev = [p is not None for p in prevalences]
    if any(have_prev) and not all(have_prev):
        warnings.warn(
            "some runs have no document-topic matrix (for example raw topic-word arrays), "
            "so group weights are not reported; pass fitted models for every run to get them",
            UserWarning,
            stacklevel=2,
        )

    pooled = np.vstack(betas)
    run_of = np.concatenate([np.full(b.shape[0], r) for r, b in enumerate(betas)])
    offsets = np.concatenate([[0], np.cumsum([b.shape[0] for b in betas])[:-1]])
    sim = _similarity(pooled, metric)
    merges = _constrained_merges(1.0 - sim, run_of)
    return _build(betas, prevalences, vocab, sim, offsets, run_of, merges, threshold)
