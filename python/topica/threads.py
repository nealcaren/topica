"""Thread-context shrinkage for topic models of reply threads (``topica.threads``).

A reply in a forum thread is written to something, and a short reply carries too few words
to place on its own. :class:`ThreadSmoother` keeps an ordinary topic model (LDA, STM, CTM, or
any fitted model with ``doc_topic`` and ``topic_word``) as the base and borrows topic mass from
each document's reply context:

.. math::

    \\tilde\\theta_d = \\frac{n_d\\,\\theta_d + a_p\\,\\theta_{\\mathrm{par}(d)}
                         + a_t\\,\\theta_{\\mathrm{thr}(d)}}{n_d + a_p + a_t}

where :math:`n_d` is the document's in-vocabulary token count, :math:`\\theta_{\\mathrm{par}(d)}`
its parent's base topic mix, and :math:`\\theta_{\\mathrm{thr}(d)}` the token-weighted mean mix
of every other document in its thread except the parent, so the two contexts do not overlap. This is the posterior mean of :math:`\\theta_d` under a
Dirichlet prior whose base measure mixes the parent's and the thread's topics, with pseudo-counts
:math:`a_p, a_t`. It is length-adaptive by construction: a three-token reply is dominated by its
context and a three-hundred-token reply barely moves.

The pseudo-counts are estimated, not assumed. :meth:`ThreadSmoother.fit` holds out half the
tokens of eligible leaf replies, fits the base on the rest, chooses :math:`(a_p, a_t)` by held-out
log likelihood on validation threads, and reports on separate test threads. A parent-permutation
placebo (parents shuffled within thread and depth, which preserves every parent's number of
children) nets out what any same-thread, same-depth comment would supply, so
:attr:`ThreadSmoother.edge_effect` measures what the *specific* parent adds. The share
:math:`a_p / (a_p + a_t)` summarizes how dyadic a community's conversation is.

Quoted text inflates parent-specific signal. Strip it before fitting with :func:`strip_quotes`
(markup) and :func:`strip_copied_runs` (verbatim runs copied from the parent).

One limit to keep in mind: a single pseudo-count per context borrows for every reply alike. When
only some replies take up their parent's topics and the rest turn to sharply different ones,
borrowing costs the second group about as much as it helps the first, and the fit can return
zero even though some replies do inherit. A zero pseudo-count therefore means "borrowing does
not help on average", not "no reply follows its parent". ``ThreadSmoother(switch=True)`` handles
that case with a per-reply inherit-or-innovate switch: a two-component mixture prior, the
shrinkage above or an innovate component centered on the corpus mean mix, whose posterior
weights come from each reply's own tokens.

``contexts=("parent", "op", "thread")`` adds the original post (the thread root) as a third,
non-overlapping context with its own pseudo-count. :attr:`ThreadSmoother.op_effect` contrasts
it with a random same-thread comment in the original post's slot, both arms sharing one thread
pool.

Experimental: an original construction validated by planted recovery. Enable with
:func:`topica.enable_experimental`.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from itertools import product
from typing import Callable, Sequence

import numpy as np

from . import experimental_enabled

__all__ = ["ThreadTM", "ThreadSmoother", "strip_quotes", "strip_copied_runs", "shuffle_parents",
           "thread_structure"]

CONTEXTS = ("parent", "thread")
ALL_CONTEXTS = ("parent", "op", "thread")
OP_SHARE_STEPS = 13
SWITCH_SHARE_STEPS = (25, 7)     # inherit-share lattice: two contexts, three (per axis)
DEFAULT_STRENGTH_GRID = np.geomspace(0.1, 1000.0, 40)
MAX_PSEUDOCOUNT = 1e4

_QUOTE_LINE = re.compile(r"^\s*(>|&gt;)")
_ITALIC = re.compile(r"<i>.*?</i>", re.S)


# --------------------------------------------------------------------------- helpers


def strip_quotes(text: str) -> str:
    """Remove quoted text from one comment.

    Drops lines or paragraphs that start with a quote marker (``>`` in Markdown, ``&gt;`` in
    HTML-escaped text such as Hacker News) and ``<i>...</i>`` spans (the Hacker News quoting
    convention), then unescapes HTML entities.
    """
    text = _ITALIC.sub(" ", text)
    parts = re.split(r"<p>|\n", text)
    return html.unescape(" ".join(p for p in parts if not _QUOTE_LINE.match(p)))


def strip_copied_runs(docs: Sequence[Sequence[str]], parents: Sequence[int],
                      n: int = 5) -> list[list[str]]:
    """Remove every token of a reply that sits in a run of ``n`` or more tokens copied
    verbatim from its parent (unmarked quoting). Roots are returned unchanged."""
    out = []
    for i, d in enumerate(docs):
        p = parents[i]
        d = list(d)
        if p < 0 or len(d) < n or len(docs[p]) < n:
            out.append(d)
            continue
        par = list(docs[p])
        grams = {tuple(par[j:j + n]) for j in range(len(par) - n + 1)}
        drop = np.zeros(len(d), bool)
        for j in range(len(d) - n + 1):
            if tuple(d[j:j + n]) in grams:
                drop[j:j + n] = True
        out.append([w for w, x in zip(d, drop) if not x])
    return out


def thread_structure(parents: Sequence[int]):
    """Depth, thread root and has-child flag for every document of a reply forest.

    ``parents[d]`` is the index of ``d``'s parent, or a negative number for a root.
    Returns ``(depth, root, has_child)`` as NumPy arrays. Raises on out-of-range parents and
    cycles.
    """
    parents = [int(p) for p in parents]
    n = len(parents)
    for i, p in enumerate(parents):
        if p >= n:
            raise ValueError(f"parents[{i}] = {p} is out of range for {n} documents")
    depth = [-1] * n
    root = [-1] * n
    for i in range(n):
        path, j, seen = [], i, set()
        while depth[j] < 0 and parents[j] >= 0:
            if j in seen:
                raise ValueError("parents contains a cycle")
            seen.add(j)
            path.append(j)
            j = parents[j]
        if depth[j] < 0:
            depth[j], root[j] = 0, j
        for node in reversed(path):
            depth[node], root[node] = depth[parents[node]] + 1, root[parents[node]]
    has_child = np.zeros(n, bool)
    for p in parents:
        if p >= 0:
            has_child[p] = True
    return np.array(depth), np.array(root), has_child


def shuffle_parents(parents: Sequence[int], *, seed: int | None = None,
                    rng: np.random.Generator | None = None) -> list[int]:
    """Placebo reply tree: permute parent assignments within (thread, depth) strata.

    A permutation of the parent list within each stratum keeps every document's depth and
    thread and every parent's number of children, and so changes only which comment each reply
    answers. Strata with a single reply are unchanged.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    depth, root, _ = thread_structure(parents)
    strata = defaultdict(list)
    for i, p in enumerate(parents):
        if p >= 0:
            strata[(root[i], depth[i])].append(i)
    out = [int(p) for p in parents]
    for nodes in strata.values():
        ps = [out[i] for i in nodes]
        rng.shuffle(ps)
        for i, p in zip(nodes, ps):
            out[i] = int(p)
    return out


def _doc_topic(model, how: str) -> np.ndarray:
    if how == "posterior" or (how == "auto" and hasattr(model, "posterior_doc_topic")):
        if not hasattr(model, "posterior_doc_topic"):
            raise ValueError("theta='posterior' needs a model with posterior_doc_topic()")
        return np.asarray(model.posterior_doc_topic(), float)
    return np.asarray(model.doc_topic, float)


def _context_mixes(theta, lengths, kept, parents, root, contexts, exclude_parent=False,
                   op_source=None, thread_drop=None):
    """Per-original-document context topic mixes (NaN rows where unavailable).

    The thread mix always excludes the document itself; with ``exclude_parent`` it also
    excludes the document's parent, so the parent and thread contexts do not overlap. The
    ``"op"`` context is the thread root's mix, available only to replies at depth 2 or deeper
    (for a top-level reply the root is the parent); when it is requested the thread mix also
    excludes the root, so the three contexts do not overlap. ``op_source`` (the OP placebo)
    names another document to stand in for each reply's root; the thread mix then excludes
    the stand-in as well as the root. ``thread_drop`` names one more document per reply to
    leave out of the thread mix (-1 for none)."""
    op_source = root if op_source is None else op_source
    n_docs = len(parents)
    k = theta.shape[1]
    row = np.full(n_docs, -1)
    row[np.asarray(kept, int)] = np.arange(len(kept))
    full_theta = np.full((n_docs, k), np.nan)
    full_theta[kept] = theta
    n = np.zeros(n_docs)
    n[kept] = lengths
    out = {}
    if "parent" in contexts:
        par = np.full((n_docs, k), np.nan)
        for d, p in enumerate(parents):
            if p >= 0 and row[p] >= 0:
                par[d] = full_theta[p]
        out["parent"] = par
    if "op" in contexts:
        op = np.full((n_docs, k), np.nan)
        for d, p in enumerate(parents):
            if p >= 0 and parents[p] >= 0 and row[op_source[d]] >= 0:
                op[d] = full_theta[op_source[d]]
        out["op"] = op
    if "thread" in contexts:
        wsum = defaultdict(lambda: np.zeros(k))
        nsum = defaultdict(float)
        for d in np.asarray(kept, int):
            wsum[root[d]] += n[d] * full_theta[d]
            nsum[root[d]] += n[d]
        thr = np.full((n_docs, k), np.nan)
        for d in range(n_docs):
            drop = {d}
            if exclude_parent and parents[d] >= 0:
                drop.add(parents[d])
            if "op" in contexts:
                drop.update((int(root[d]), int(op_source[d])))
            if thread_drop is not None and thread_drop[d] >= 0:
                drop.add(int(thread_drop[d]))
            own = sum((n[j] * full_theta[j] for j in drop if row[j] >= 0), np.zeros(k))
            rest = nsum[root[d]] - sum(n[j] for j in drop)
            if rest > 0:
                thr[d] = (wsum[root[d]] - own) / rest
        out["thread"] = thr
    return full_theta, n, out


def _placebo_contexts(theta, lengths, kept, parents, root, contexts, exclude_parent,
                      n_shuffle, seed):
    """Context mixes under ``n_shuffle`` shuffled trees, one dict per draw.

    Every context is built from the shuffled tree, so the placebo differs from the true tree
    only in which comment is each reply's parent. With non-overlapping contexts the true parent
    therefore moves into the placebo's thread mix and the stand-in parent leaves it."""
    return [_context_mixes(theta, lengths, kept, sp, root, contexts,
                           exclude_parent=exclude_parent)[2]
            for sp in _shuffled_trees(parents, n_shuffle, seed)]


def _op_placebo_contexts(theta, lengths, kept, parents, root, contexts, exclude_parent,
                         n_shuffle, seed):
    """Matched (true, placebo) context mixes for the original-post contrast, one pair per draw.

    For each reply at depth 2 or deeper whose root the corpus kept, a random other comment of
    the same thread (not the reply, its parent or the root) is the donor. Both arms leave the
    root *and* the donor out of the thread mix, so they share one thread pool; the true arm uses
    the root as the original post and the placebo uses the donor. The contrast therefore asks
    only whether the original post predicts a reply better than a random same-thread comment in
    the same slot. A reply with no donor, or whose root the corpus dropped, is identical in both
    arms (and has no original post in either)."""
    parents = [int(p) for p in parents]
    kept_set = set(np.asarray(kept, int).tolist())
    members = defaultdict(list)
    for d in range(len(parents)):
        if d in kept_set and parents[d] >= 0:
            members[int(root[d])].append(d)
    out = []
    for s in range(n_shuffle):
        rng = np.random.default_rng(seed * 1000 + 500 + s)
        donor = np.full(len(parents), -1)
        for d, p in enumerate(parents):
            if p < 0 or parents[p] < 0 or int(root[d]) not in kept_set:
                continue
            cand = [j for j in members[int(root[d])] if j != d and j != p]
            if cand:
                donor[d] = cand[int(rng.integers(len(cand)))]
        src = np.where(donor >= 0, donor, np.asarray(root, int))
        true = _context_mixes(theta, lengths, kept, parents, root, contexts,
                              exclude_parent=exclude_parent, thread_drop=donor)[2]
        placebo = _context_mixes(theta, lengths, kept, parents, root, contexts,
                                 exclude_parent=exclude_parent, op_source=src)[2]
        out.append((true, placebo))
    return out


def _shuffled_trees(parents, n_shuffle, seed):
    return [shuffle_parents(parents, rng=np.random.default_rng(seed * 1000 + s))
            for s in range(n_shuffle)]


CONC_GRID = np.geomspace(0.01, 1000.0, 21)


def _sequential_log_ml(ids, prior_mean, conc_grid, beta):
    """Log marginal likelihood of each document's tokens under a Dirichlet prior on its topic
    mix, with the topics ``beta`` held fixed: ``log p(w_d | a * m_d)`` for every ``a`` in
    ``conc_grid``. Returns ``(D, len(conc_grid))``; NaN rows where ``prior_mean`` is NaN.

    The Dirichlet-multinomial marginal sums over topic assignments; we use the sequential
    Polya-urn form with soft counts, ``p(w_i | w_<i) = sum_k (a m_k + c_k) beta_kw / (a + i)``,
    where ``c`` accumulates each earlier token's topic responsibilities. The document's own
    fitted mix never enters, so a reply's words cannot vouch for a component by having been
    fit to it."""
    n_docs, k = prior_mean.shape
    conc = np.asarray(conc_grid, float)[:, None, None]
    out = np.full((n_docs, conc.shape[0]), np.nan)
    rows_all = np.flatnonzero(~np.isnan(prior_mean[:, 0]))
    for start in range(0, rows_all.size, 1000):          # chunks bound the (G, R, K) state
        rows = rows_all[start:start + 1000]
        lens = np.array([len(ids[r]) for r in rows], int)
        width = int(lens.max(initial=0))
        idm = np.zeros((rows.size, width), int)
        for j, r in enumerate(rows):
            idm[j, :lens[j]] = ids[r]
        am = conc * prior_mean[rows][None]                # (G, R, K)
        counts = np.zeros_like(am)
        ll = np.zeros((conc.shape[0], rows.size))
        for i in range(width):
            live = np.flatnonzero(lens > i)
            bw = beta[:, idm[live, i]].T[None]            # (1, R_live, K)
            p = (am[:, live] + counts[:, live]) * bw
            tot = p.sum(2)
            ll[:, live] += np.log(tot / (conc[:, :, 0] + i))
            counts[:, live] += p / tot[:, :, None]
        out[rows] = ll.T
    return out


def _interp_cols(table, x, grid):
    """Linear interpolation of each row of ``table`` (columns on ``grid``) at ``x`` (a scalar,
    or one value per row), on the log scale of the grid; ``x`` is clipped to the grid's
    range."""
    lg = np.log(grid)
    x = np.maximum(np.broadcast_to(np.asarray(x, float), (table.shape[0],)), grid[0])
    lx = np.clip(np.log(x), lg[0], lg[-1])
    j = np.clip(np.searchsorted(lg, lx) - 1, 0, len(grid) - 2)
    f = (lx - lg[j]) / (lg[j + 1] - lg[j])
    r = np.arange(table.shape[0])
    return table[r, j] * (1 - f) + table[r, j + 1] * f


def _coord_search(f, x0, lo, hi, step=1.0, min_step=0.02, max_rounds=60):
    """Maximize ``f`` within the box ``[lo, hi]`` by coordinate steps of ``+-step`` that halve
    when nothing improves."""
    x = np.clip(np.array(x0, float), lo, hi)
    best = f(x)
    for _ in range(max_rounds):
        improved = False
        for j in range(x.size):
            for d in (step, -step):
                cand = x.copy()
                cand[j] = np.clip(cand[j] + d, lo[j], hi[j])
                if cand[j] == x[j]:
                    continue
                v = f(cand)
                if v > best + 1e-12:
                    x, best, improved = cand, v, True
        if not improved:
            step /= 2
            if step < min_step:
                break
    return x, best


def _call_base(base, corpus, seed):
    """Call a base factory as ``base(corpus, seed=seed)`` when it has a parameter named
    ``seed`` (or ``**kwargs``), else ``base(corpus)``."""
    import inspect
    try:
        params = inspect.signature(base).parameters
    except (TypeError, ValueError):
        return base(corpus)
    takes_seed = "seed" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return base(corpus, seed=seed) if takes_seed else base(corpus)


def _gate():
    if not experimental_enabled():
        raise RuntimeError(
            "ThreadSmoother is experimental: it is an original construction validated by "
            "planted recovery, with no published reference yet. Enable experimental models "
            "with `topica.enable_experimental()` or set TOPICA_EXPERIMENTAL=1. Experimental "
            "features may change or be removed without a deprecation cycle.")


# ------------------------------------------------------------------------ the smoother


class ThreadSmoother:
    """Thread-context shrinkage over any fitted topic model (experimental).

    Parameters
    ----------
    contexts : which reply contexts to borrow from: any of ``"parent"``, ``"op"`` and
        ``"thread"`` (default ``("parent", "thread")``). ``"op"`` is the thread root's mix,
        available to replies at depth 2 or deeper (a top-level reply's parent is the root, so
        it does not borrow from ``"op"``); with it, the thread mix also excludes the root.
        Three contexts are searched on a coarser share lattice (13 logit steps per axis).
    theta : the base topic mix. ``"auto"`` uses ``posterior_doc_topic()`` when the base model
        has it (the logistic-normal models) and ``doc_topic`` otherwise; ``"doc_topic"`` and
        ``"posterior"`` force one.
    strength_grid : candidate total pseudo-counts ``s = sum(a)``. Defaults to 0 plus 40
        log-spaced values from 0.1 to 1000.
    thread_excludes_parent : when both contexts are used, leave the parent out of the thread
        mix (the default), so ``a_p`` and ``a_t`` weight separate sources: the parent, and
        every other document in the thread. With ``False`` the thread mix contains the parent
        and ``a_p`` is the parent's weight beyond its share of the thread.
    share_steps : number of parent-share values ``pi = a_p / s`` searched (both contexts
        only). They are evenly spaced on the logit scale from ``logit(pi) = -6`` to ``+6``
        (about 0.0025 to 0.9975), so the share is never estimated as exactly 0 or 1 and its
        interval stays inside (0, 1); this acts as a very weak prior against the boundary.
        Default 97 (steps of 0.125 on the logit scale, about 0.03 in share near 0.5).
    switch : per-reply inherit-or-innovate switch (#897). Each document's topic mix gets a
        two-component mixture prior: *inherit*, a Dirichlet centered on the share-weighted
        context mix with the context pseudo-counts ``a_c`` (the pooled shrinkage), or *new*,
        centered on the corpus mean mix. The inherit probability ``w_d`` comes from an
        approximate marginal likelihood of the document's own tokens under each component,
        with the topics held fixed (a sequential Polya-urn with soft topic counts: exact for
        two tokens, an approximation beyond that, and dependent on token order), so the
        document's fitted mix never scores its own words, and

            theta_tilde_d = w_d (n_d theta_d + sum_c a_c theta_c) / (n_d + sum_c a_c)
                            + (1 - w_d) theta_d.

        The context shares are profiled over a coarse logit lattice (25 values for two
        contexts, 7 per axis for three); the total pseudo-count, the innovate concentration
        and the prior inherit weight ``rho`` are optimized on validation threads, and every
        reported number comes from test threads, as in the pooled fit. A context a document
        lacks (a top-level reply has no original post) drops its pseudo-count rather than
        passing it to the others. Bootstrap draws re-choose the share and re-optimize the other
        parameters on resampled validation threads; the held-out gains are evaluated at the
        point estimate and resampled over test threads. Use it when only some
        replies take up their context: the pooled pseudo-count then borrows little or nothing
        even though some replies inherit. ``groups`` is not supported yet.

    Attributes (after :meth:`fit`)
    ------------------------------
    alpha : dict ``{context: pseudo-count}``, the validation-optimal point estimate (grid
        search, then continuous refinement).
    alpha_ci : dict ``{context: (lo, hi)}`` 95% bootstrap interval.
    parent_share, parent_share_ci : ``a_p / sum(a)`` and its 95% percentile interval
        (two or more contexts including the parent), estimated on the logit-spaced share grid. ``parent_share_at_bound``
        is ``True`` when the estimate is the outermost grid value, meaning the data cannot
        distinguish the smaller pseudo-count from zero. ``p_no_borrowing`` is the share of
        bootstrap draws that chose no borrowing at all (share undefined); those draws are
        excluded from the interval and counted here.
    alpha_by_group : per-group pseudo-counts when ``groups`` is given to :meth:`fit`.
    rho, rho_ci : (``switch=True``) prior probability that a reply inherits, and its 95%
        bootstrap interval. ``alpha`` and ``parent_share`` then describe the inherit
        component: among replies that inherit, how much comes from each context.
        ``p_no_borrowing`` is the share of bootstrap draws with ``rho < 0.01``.
    inherit_weights : (``switch=True``) ``(D,)`` posterior inherit probability per input
        document from the latest :meth:`transform`. Its mean over replies estimates the share
        of replies that inherit.
    op_effect : (``"op"`` in ``contexts``) held-out gain of the true original post over a
        random other comment of the same thread in its slot, nats per token:
        ``{"estimate", "lo", "hi"}``. Both arms leave the root and that comment out of the
        thread mix, so they share one thread pool and differ only in the original post.
    completion : held-out gain over the base on test threads, nats per token:
        ``{"estimate", "lo", "hi", "by_length"}``.
    edge_effect : held-out gain of the true tree over the shuffled-parent placebo on test
        threads, nats per token: ``{"estimate", "lo", "hi"}``.
    uncertainty : ``"threads"`` (thread bootstrap on one calibration) or ``"refit"`` (pooled
        over ``n_refit`` further calibrations with new masks and base seeds).
    replicates : per-calibration point estimates when ``n_refit > 0``.
    draws : the pooled bootstrap draws (``alpha``, ``parent_share``, ``completion``,
        ``edge_effect``), for intervals on derived quantities such as a difference in parent
        share between two corpora.
    theta_tilde : ``(D, K)`` smoothed topic mixes of a full-data base fit, one row per input
        document (a document the vocabulary empties gets its pure context mix).
    base_model, corpus : that full-data fit and its Corpus.
    settings : the calibration settings.
    """

    def __init__(self, contexts: Sequence[str] = CONTEXTS, *, theta: str = "auto",
                 strength_grid: Sequence[float] | None = None, share_steps: int = 97,
                 thread_excludes_parent: bool = True, switch: bool = False) -> None:
        _gate()
        contexts = tuple(contexts)
        bad = set(contexts) - set(ALL_CONTEXTS)
        if not contexts or bad or len(set(contexts)) != len(contexts):
            raise ValueError(f"contexts must be a non-empty subset of {ALL_CONTEXTS}")
        if theta not in ("auto", "doc_topic", "posterior"):
            raise ValueError("theta must be 'auto', 'doc_topic' or 'posterior'")
        grid = np.asarray(DEFAULT_STRENGTH_GRID if strength_grid is None else strength_grid,
                          float)
        if grid.ndim != 1 or grid.size == 0 or (grid < 0).any():
            raise ValueError("strength_grid must be a non-empty 1-D array of non-negative values")
        if share_steps < 3:
            raise ValueError("share_steps must be >= 3")
        self.contexts = contexts
        self.theta_source = theta
        self.strength_grid = np.unique(np.concatenate([[0.0], grid]))
        self.share_steps = int(share_steps)
        self.thread_excludes_parent = bool(thread_excludes_parent) and "parent" in contexts
        self.switch = bool(switch)
        self.alpha = None
        self.alpha_by_group = None

    # -- parameterization ---------------------------------------------------------------

    def _grid(self):
        """Candidate pseudo-count vectors, ordered as ``self.contexts``.

        Several contexts are searched on (strength s, shares) so the shares are resolved
        directly; one context on strength alone. s = 0 appears once (no borrowing)."""
        pts = [np.zeros(len(self.contexts))]
        if len(self.contexts) == 1:
            return pts + [np.array([s]) for s in self.strength_grid[1:]]
        return pts + [s * sh for s in self.strength_grid[1:] for sh in self._share_vectors()]

    def _share_vectors(self, steps=None):
        """Share vectors over the contexts, a lattice on the logit scale from -6 to +6 against
        the last context: ``share_steps`` points for two contexts, ``OP_SHARE_STEPS`` per axis
        for three. No share is ever exactly 0 or 1."""
        c = len(self.contexts)
        if steps is None:
            steps = self.share_steps if c == 2 else OP_SHARE_STEPS
        axes = [np.linspace(-6.0, 6.0, steps)] * (c - 1)
        lat = np.array(list(product(*axes)))
        logits = np.column_stack([lat, np.zeros(len(lat))])
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def _share_ends(self):
        """Smallest and largest parent share on the grid."""
        v = self._share_vectors()[:, self.contexts.index("parent")]
        return np.array([v.min(), v.max()])

    def _share(self, a):
        """``a_parent / sum(a)`` (or of prior weights under the switch); None without a
        parent context or with a single context."""
        if "parent" not in self.contexts or len(self.contexts) == 1:
            return None
        s = a.sum()
        return float(a[self.contexts.index("parent")] / s) if s > 0 else np.nan

    # -- one calibration -------------------------------------------------------------------

    def _calibrate(self, docs, parents, base, groups, *, heldout_frac, val_frac,
                   min_eval_tokens, n_shuffle, n_boot, seed, corpus_kwargs, refine):
        import topica

        rng = np.random.default_rng(seed)
        depth, root, has_child = thread_structure(parents)

        # Mask half the tokens of eligible leaves; nothing held out enters any context.
        train_docs = [list(d) for d in docs]
        held = {}
        for i, d in enumerate(docs):
            if parents[i] < 0 or has_child[i] or len(d) < min_eval_tokens:
                continue
            m = int(round(heldout_frac * len(d)))
            if m == 0 or m == len(d):
                continue
            hset = set(rng.permutation(len(d))[:m].tolist())
            held[i] = [d[j] for j in sorted(hset)]
            train_docs[i] = [d[j] for j in range(len(d)) if j not in hset]
        if not held:
            raise ValueError("no leaf reply is long enough to mask; lower min_eval_tokens")

        corpus = topica.Corpus.from_documents(train_docs, **corpus_kwargs)
        model = _call_base(base, corpus, seed)
        theta = _doc_topic(model, self.theta_source)
        beta = np.asarray(model.topic_word, float)
        kept = np.asarray(corpus.kept_indices, int)
        lengths = np.asarray(corpus.doc_lengths, float)
        if theta.shape[0] != len(kept) or beta.shape[0] != theta.shape[1]:
            raise ValueError("base model's doc_topic / topic_word do not match its corpus")
        vidx = {w: j for j, w in enumerate(corpus.vocabulary)}

        _, n_doc, ctx = _context_mixes(theta, lengths, kept, parents, root, self.contexts,
                                       exclude_parent=self.thread_excludes_parent)
        shuf_ctx = (_placebo_contexts(theta, lengths, kept, parents, root, self.contexts,
                                      self.thread_excludes_parent, n_shuffle, seed)
                    if "parent" in self.contexts else [])
        op_ctx = (_op_placebo_contexts(theta, lengths, kept, parents, root, self.contexts,
                                       self.thread_excludes_parent, n_shuffle, seed)
                  if "op" in self.contexts else [])
        row = np.full(len(docs), -1)
        row[kept] = np.arange(len(kept))
        # Held-out tokens get the corpus's own filtering: in the vocabulary and not a stopword
        # (a fixed vocabulary can list a stopword the corpus then removed).
        stop = set(corpus_kwargs.get("stopwords") or [])

        # Common scored support: the leaf, every requested context and every placebo context
        # exist; at least one in-vocabulary held token. The original post is the exception: a
        # top-level reply has none (its parent is the root) and simply does not borrow from it.
        need = [c for c in self.contexts if c != "op"]
        leaves = []
        for i, toks in held.items():
            ids = np.array([vidx[w] for w in toks if w in vidx and w not in stop], int)
            if ids.size == 0 or row[i] < 0:
                continue
            if any(np.isnan(ctx[c][i, 0]) for c in need):
                continue
            if any(np.isnan(sc[c][i, 0]) for sc in shuf_ctx for c in need):
                continue
            leaves.append((i, ids))
        if not leaves:
            raise ValueError("no evaluable leaf replies after vocabulary filtering")
        roots = np.array(sorted({root[i] for i, _ in leaves}))
        if len(roots) < 4:
            raise ValueError("need at least four threads with evaluable leaves")
        n_val = min(max(1, int(round(val_frac * len(roots)))), len(roots) - 1)
        val_roots = set(rng.choice(roots, size=n_val, replace=False).tolist())

        # Flat per-token arrays.
        tpos = {r: j for j, r in enumerate(roots)}
        pL, n_tok, tj, grp = [], [], [], []
        # Per-token context predictives (0 where the context is missing) and availability.
        # Placebo keys are prefixed; each averages the predictive over its draws.
        sources = [("", c, [ctx]) for c in self.contexts]
        sources += [("_shuffled_", c, shuf_ctx) for c in self.contexts if shuf_ctx]
        if op_ctx:   # the matched arms of the original-post contrast
            sources += [("_optrue_", c, [t for t, _ in op_ctx]) for c in self.contexts]
            sources += [("_opshuf_", c, [p for _, p in op_ctx]) for c in self.contexts]
        P = {pre + c: [] for pre, c, _ in sources}
        avail = {key: [] for key in P}
        for i, ids in leaves:
            r = row[i]
            pL.append(theta[r] @ beta[:, ids])
            for pre, c, mixes in sources:
                # Draws can differ in availability (a donor can empty the thread mix): average
                # the predictive over the draws that have the context, and weight the context
                # by the share of draws that do.
                have = [m[c][i] for m in mixes if not np.isnan(m[c][i, 0])]
                P[pre + c].append(np.mean([h @ beta[:, ids] for h in have], 0) if have
                                  else np.zeros(ids.size))
                avail[pre + c].append(np.full(ids.size, len(have) / len(mixes)))
            n_tok.append(np.full(ids.size, n_doc[i]))
            tj.append(np.full(ids.size, tpos[root[i]]))
            grp.extend([None if groups is None else groups[i]] * ids.size)
        pL = np.concatenate(pL)
        P = {c: np.concatenate(v) for c, v in P.items()}
        avail = {c: np.concatenate(v) for c, v in avail.items()}
        n_tok = np.concatenate(n_tok)
        tj = np.concatenate(tj)
        grp = np.array(grp, dtype=object)
        n_threads = len(roots)
        is_val = np.array([r in val_roots for r in roots])
        val_idx, test_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
        settings = {"n_eval_leaves": len(leaves), "n_eval_threads": n_threads,
                    "n_val_threads": int(is_val.sum()),
                    "n_test_tokens": int(np.isin(tj, test_idx).sum())}

        if self.switch:
            # A leaf's observed tokens exactly as the corpus kept them.
            kept_docs = corpus.documents()
            obs = [np.array([vidx[w] for w in kept_docs[row[i]]], int) for i, _ in leaves]
            out = self._calibrate_switch(leaves, obs, theta, beta, ctx, shuf_ctx, op_ctx, pL,
                                         n_tok, tj, val_idx, test_idx, n_boot, seed)
            out["settings"], out["model"] = settings, model
            return out

        grid = self._grid()

        def token_ll(a, keys, mask):
            sl = (lambda x: x) if mask.all() else (lambda x: x[mask])
            nn = sl(n_tok)
            num, denom = nn * sl(pL), nn.copy()
            for a_c, c in zip(a, keys):
                if a_c > 0:          # a missing context (avail 0) contributes nothing
                    w = a_c * sl(avail[c])
                    num, denom = num + w * sl(P[c]), denom + w
            return np.log(num / denom)

        def tables(keys, mask):
            """Per-thread log-likelihood sums for every grid point, on tokens in ``mask``."""
            tt = tj[mask]
            out = np.empty((len(grid), n_threads))
            for g_i, a in enumerate(grid):
                out[g_i] = np.bincount(tt, token_ll(a, keys, mask), n_threads)
            return out, np.bincount(tt, np.log(pL[mask]), n_threads), \
                np.bincount(tt, None, n_threads)

        def val_obj(a, keys, mask):
            m = mask & np.isin(tj, val_idx)
            return token_ll(a, keys, m).mean()

        def refine_point(a0, keys, mask):
            """Coordinate search in log pseudo-count space around the grid optimum."""
            a = a0.copy()
            if a.sum() == 0 or not refine:
                return a
            best = val_obj(a, keys, mask)
            step = 0.5
            for _ in range(40):
                improved = False
                for j in range(len(a)):
                    for f in (np.exp(step), np.exp(-step)):
                        cand = a.copy()
                        cand[j] = cand[j] * f if cand[j] > 0 else (0.1 if f > 1 else 0.0)
                        if cand[j] > MAX_PSEUDOCOUNT:    # the objective is flat out there
                            continue
                        v = val_obj(cand, keys, mask)
                        if v > best + 1e-12:
                            a, best, improved = cand, v, True
                if not improved:
                    step /= 2
                    if step < 0.02:
                        break
            return a

        everything = np.ones(pL.size, bool)
        tab, base_t, tok_t = tables(self.contexts, everything)
        g_star = int(np.argmax(tab[:, val_idx].sum(1)))
        a_star = refine_point(grid[g_star], self.contexts, everything)

        brng = np.random.default_rng(seed + 1)
        boot_val = val_idx[brng.integers(0, len(val_idx), (n_boot, len(val_idx)))]
        boot_test = test_idx[brng.integers(0, len(test_idx), (n_boot, len(test_idx)))]
        boot_g = np.array([int(np.argmax(tab[:, b].sum(1))) for b in boot_val])

        # Test-thread quantities at the point estimate (per-thread sums for the bootstrap).
        ll_star = np.bincount(tj, token_ll(a_star, self.contexts, everything), n_threads)
        gain_t = ll_star - base_t

        def rate(num, idx):
            return num[idx].sum() / tok_t[idx].sum()

        out = {"alpha": a_star, "completion": rate(gain_t, test_idx),
               "draws_alpha": np.array([grid[g] for g in boot_g]),
               "draws_completion": gain_t[boot_test].sum(1) / tok_t[boot_test].sum(1)}

        # Gain by observed-length tercile (cuts from validation leaves).
        val_tok = np.isin(tj, val_idx)
        cuts = np.quantile(n_tok[val_tok], [1 / 3, 2 / 3])
        ter = np.digitize(n_tok, cuts, right=True)
        tok_gain = token_ll(a_star, self.contexts, everything) - np.log(pL)
        out["by_length"] = {"cuts": [float(c_) for c_ in cuts],
                            "gain": [float(tok_gain[~val_tok & (ter == t_)].mean())
                                     if (~val_tok & (ter == t_)).any() else float("nan")
                                     for t_ in range(3)]}

        if "parent" in self.contexts:
            keys_s = tuple("_shuffled_" + c for c in self.contexts)
            tab_s, _, _ = tables(keys_s, everything)
            g_s = int(np.argmax(tab_s[:, val_idx].sum(1)))
            a_s = refine_point(grid[g_s], keys_s, everything)
            ll_s = np.bincount(tj, token_ll(a_s, keys_s, everything), n_threads)
            edge_t = ll_star - ll_s
            out["edge_effect"] = rate(edge_t, test_idx)
            out["draws_edge"] = edge_t[boot_test].sum(1) / tok_t[boot_test].sum(1)
            out["placebo_alpha"] = a_s

        if "op" in self.contexts and op_ctx:
            def arm_ll(pre):
                keys = tuple(pre + c for c in self.contexts)
                tab_a, _, _ = tables(keys, everything)
                a = refine_point(grid[int(np.argmax(tab_a[:, val_idx].sum(1)))], keys,
                                 everything)
                return np.bincount(tj, token_ll(a, keys, everything), n_threads), a
            ll_true_o, _ = arm_ll("_optrue_")
            ll_plac_o, a_o = arm_ll("_opshuf_")
            op_t = ll_true_o - ll_plac_o
            out["op_effect"] = rate(op_t, test_idx)
            out["draws_op"] = op_t[boot_test].sum(1) / tok_t[boot_test].sum(1)
            out["op_placebo_alpha"] = a_o

        if groups is not None:
            out["alpha_by_group"] = {}
            for g_lab in sorted(set(grp.tolist()), key=str):
                m = grp == g_lab
                tab_g, _, tok_g = tables(self.contexts, m)
                if tok_g[val_idx].sum() == 0:
                    continue
                a_g = refine_point(grid[int(np.argmax(tab_g[:, val_idx].sum(1)))],
                                   self.contexts, m)
                out["alpha_by_group"][g_lab] = a_g

        out["settings"] = settings
        out["model"] = model
        return out

    # -- the inherit-or-innovate switch (#897) ---------------------------------------------

    def _switch_shares(self):
        """Candidate context shares for the inherit component (a coarse logit lattice)."""
        steps = {1: 1, 2: SWITCH_SHARE_STEPS[0], 3: SWITCH_SHARE_STEPS[1]}[len(self.contexts)]
        if len(self.contexts) == 1:
            return np.ones((1, 1))
        return self._share_vectors(steps)

    @staticmethod
    def _unpack(x):
        """Parameter vector ``(log A, log a_new, logit rho)`` -> bounded values: the inherit
        concentration ``A``, the innovate concentration ``a_new`` (both kept on the
        marginal-likelihood grid, where the objective is defined) and the prior inherit weight
        ``rho`` (kept within about (3e-4, 1 - 3e-4), beyond which the objective is flat)."""
        lo, hi = np.log(CONC_GRID[0]), np.log(CONC_GRID[-1])
        return (float(np.exp(np.clip(x[0], lo, hi))), float(np.exp(np.clip(x[1], lo, hi))),
                float(1.0 / (1.0 + np.exp(-np.clip(x[2], -8.0, 8.0)))))

    @staticmethod
    def _inherit_prob(lml_inh, lml_new, A, a_new, rho):
        """Posterior probability that each document inherits, from its log marginal
        likelihood tables (NaN = no context to inherit from)."""
        li = _interp_cols(lml_inh, A, CONC_GRID) + np.log(rho)
        ln = _interp_cols(lml_new, a_new, CONC_GRID) + np.log1p(-rho)
        with np.errstate(invalid="ignore"):
            w = 1.0 / (1.0 + np.exp(np.clip(ln - li, -700, 700)))
        return np.where(np.isnan(li), 0.0, w)

    def _calibrate_switch(self, leaves, obs, theta, beta, ctx, shuf_ctx, op_ctx, pL, n_tok,
                          tj, val_idx, test_idx, n_boot, seed):
        """Fit the switch: a two-component mixture prior per reply, *inherit* (the pooled
        multi-context shrinkage, centered on the share-weighted context mix with concentration
        A) or *new* (centered on the corpus mean mix). The inherit probability of each reply
        comes from the marginal likelihood of its observed tokens. The context shares are
        profiled over a coarse lattice (per-thread tables at each share's optimum, so the
        thread bootstrap re-chooses the share cheaply); A, a_new and rho are optimized per
        share on validation threads."""
        C = len(self.contexts)
        shares = self._switch_shares()
        leaf_rows = [i for i, _ in leaves]
        tl = np.concatenate([np.full(ids.size, j) for j, (_, ids) in enumerate(leaves)])
        n_threads = int(tj.max()) + 1
        tok_t = np.bincount(tj, None, n_threads)
        on_val = np.isin(tj, val_idx)
        lml_new = _sequential_log_ml(obs, np.tile(theta.mean(0), (len(leaves), 1)),
                                     CONC_GRID, beta)

        def share_tables(cx):
            """For one tree: per share vector, the per-token inherit predictive and the
            per-leaf log-ML table of the inherit component."""
            mix = np.stack([cx[c][leaf_rows] for c in self.contexts], 1)       # (L, C, K)
            av = ~np.isnan(mix[:, :, 0])                                       # (L, C)
            mix = np.nan_to_num(mix)
            Ptok = np.stack([np.concatenate([mix[j, c] @ beta[:, ids]
                                             for j, (_, ids) in enumerate(leaves)])
                             for c in range(C)], 1)                            # (T, C)
            out = []
            for s in shares:
                wts = av * s                                                   # (L, C)
                tot = wts.sum(1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    wts = wts / tot                                            # NaN if none
                center = np.einsum("lc,lck->lk", wts, mix)
                p_inh = np.nan_to_num((Ptok * wts[tl]).sum(1))
                # A missing context's pseudo-count is dropped, not handed to the others: the
                # leaf's concentration is A times the share of its available contexts.
                out.append((p_inh, _sequential_log_ml(obs, center, CONC_GRID, beta),
                            tot[:, 0]))
            return out

        def token_mix(x, p_inh, lml, frac, m):
            A, a_new, rho = self._unpack(x)
            a_leaf = A * frac
            w = self._inherit_prob(lml, lml_new, a_leaf, a_new, rho)[tl[m]]
            a_tok = a_leaf[tl[m]]
            nn, pl = n_tok[m], pL[m]
            den = nn + a_tok
            with np.errstate(invalid="ignore", divide="ignore"):
                shrunk = np.where(den > 0, (nn * pl + a_tok * p_inh[m]) / den, pl)
            return w * shrunk + (1 - w) * pl

        def token_ll(x, p_inh, lml, frac, m):
            return np.log(token_mix(x, p_inh, lml, frac, m))

        def thread_ll(x, tabs_j, m):
            """Per-thread log likelihood; placebo draws are averaged inside the log (the
            predictive averaged over draws), as in the pooled fit."""
            mix = np.mean([token_mix(x, *t, m) for t in tabs_j], 0)
            return np.bincount(tj[m], np.log(mix), n_threads)

        lo = np.array([np.log(CONC_GRID[0])] * 2 + [-8.0])
        hi = np.array([np.log(CONC_GRID[-1])] * 2 + [8.0])
        starts = [np.array([la, 0.0, 0.0]) for la in (0.0, np.log(10.0), np.log(100.0))]

        def profile(tab_sets, warm=None):
            """Optimize (A, a_new, rho) on validation threads for every share; return the
            parameters and the per-thread log likelihood table (all threads)."""
            xs, tab = [], np.empty((len(shares), n_threads))
            for j in range(len(shares)):
                tabs_j = [t[j] for t in tab_sets]

                def f(x):
                    return thread_ll(x, tabs_j, on_val)[val_idx].sum() / tok_t[val_idx].sum()
                s0 = starts if warm is None else [warm[j]]
                x = max((_coord_search(f, x0, lo, hi) for x0 in s0), key=lambda r: r[1])[0]
                xs.append(x)
                tab[j] = thread_ll(x, tabs_j, np.ones(tj.size, bool))
            return xs, tab

        true_tabs = share_tables(ctx)
        xs, tab = profile([true_tabs])
        j_star = int(np.argmax(tab[:, val_idx].sum(1)))
        x_star = xs[j_star]
        base_t = np.bincount(tj, np.log(pL), n_threads)
        ll_star = tab[j_star]
        gain_t = ll_star - base_t

        brng = np.random.default_rng(seed + 1)
        boot_val = val_idx[brng.integers(0, len(val_idx), (n_boot, len(val_idx)))]
        boot_test = test_idx[brng.integers(0, len(test_idx), (n_boot, len(test_idx)))]
        boot_j = np.array([int(np.argmax(tab[:, b].sum(1))) for b in boot_val])

        # Bootstrap re-selection: each draw re-chooses the share on its resampled validation
        # threads and re-optimizes (A, a_new, rho) there, warm-started at that share's
        # optimum. (The held-out gains below are evaluated at the point estimate and
        # resampled over test threads, as in the pooled fit.)
        boot_x = []
        for b, j in zip(boot_val, boot_j):
            wb = np.bincount(b, None, n_threads)
            tabs_j = [true_tabs[j]]

            def f(x, wb=wb, tabs_j=tabs_j):
                return thread_ll(x, tabs_j, on_val) @ wb / (tok_t @ wb)
            boot_x.append(_coord_search(f, xs[j], lo, hi, step=0.25, min_step=0.05)[0])

        def rate(num, idx):
            return num[idx].sum() / tok_t[idx].sum()

        def alpha_of(j, x):
            return self._unpack(x)[0] * shares[j]

        out = {"alpha": alpha_of(j_star, x_star), "switch_x": x_star,
               "switch_share": shares[j_star],
               "rho": self._unpack(x_star)[2],
               "new_concentration": self._unpack(x_star)[1],
               "completion": rate(gain_t, test_idx),
               "draws_alpha": np.array([alpha_of(j, x) for j, x in zip(boot_j, boot_x)]),
               "draws_rho": np.array([self._unpack(x)[2] for x in boot_x]),
               "draws_completion": gain_t[boot_test].sum(1) / tok_t[boot_test].sum(1)}

        cuts = np.quantile(n_tok[on_val], [1 / 3, 2 / 3])
        ter = np.digitize(n_tok, cuts, right=True)
        everything = np.ones(tj.size, bool)
        tok_gain = token_ll(x_star, *true_tabs[j_star], everything) - np.log(pL)
        out["by_length"] = {"cuts": [float(c_) for c_ in cuts],
                            "gain": [float(tok_gain[~on_val & (ter == t_)].mean())
                                     if (~on_val & (ter == t_)).any() else float("nan")
                                     for t_ in range(3)]}

        def arm(mixes):
            """Profile fit on a set of context draws: (per-thread ll, alpha)."""
            p_xs, p_tab = profile([share_tables(m) for m in mixes], warm=xs)
            p_j = int(np.argmax(p_tab[:, val_idx].sum(1)))
            return p_tab[p_j], alpha_of(p_j, p_xs[p_j])

        # Placebos: the same profile fit on shuffled trees; for the original post, matched
        # true and placebo arms that share one thread pool.
        if shuf_ctx:
            ll_s, a_s = arm(shuf_ctx)
            edge_t = ll_star - ll_s
            out["edge_effect"] = rate(edge_t, test_idx)
            out["draws_edge"] = edge_t[boot_test].sum(1) / tok_t[boot_test].sum(1)
            out["placebo_alpha"] = a_s
        if op_ctx:
            ll_t, _ = arm([t for t, _ in op_ctx])
            ll_p, a_p = arm([p for _, p in op_ctx])
            op_t = ll_t - ll_p
            out["op_effect"] = rate(op_t, test_idx)
            out["draws_op"] = op_t[boot_test].sum(1) / tok_t[boot_test].sum(1)
            out["op_placebo_alpha"] = a_p
        return out

    # -- calibration --------------------------------------------------------------------

    def fit(self, docs: Sequence[Sequence[str]], parents: Sequence[int], *,
            base: Callable, groups: Sequence | None = None, heldout_frac: float = 0.5,
            val_frac: float = 0.5, min_eval_tokens: int = 5, n_shuffle: int = 3,
            n_boot: int = 500, n_refit: int = 0, seed: int = 13,
            corpus_kwargs: dict | None = None, refine: bool = True,
            final: bool = True) -> "ThreadSmoother":
        """Estimate the pseudo-counts on held-out replies, then smooth a full-data fit.

        Parameters
        ----------
        docs : one token list per document.
        parents : parent index per document (negative for a thread root).
        base : factory ``base(corpus)`` or ``base(corpus, seed)`` returning a fitted model.
            It is called once on a masked corpus per calibration and once on the full corpus
            when ``final``. The model must expose ``doc_topic`` (rows aligned with
            ``corpus.kept_indices``) and a ``(K, V)`` ``topic_word``. Accepting ``seed`` lets
            ``n_refit`` vary the base fit as well as the mask.
        groups : optional community label per document; pseudo-counts are then also fit per
            group (``alpha_by_group``) and used by :meth:`transform`.
        heldout_frac : share of each eligible leaf's tokens held out for scoring.
        val_frac : share of evaluation threads used to choose the pseudo-counts; the rest are
            the test threads every reported number comes from.
        min_eval_tokens : leaves with fewer tokens are not masked.
        n_shuffle : placebo draws; the placebo parent mix is their average.
        n_boot : thread-bootstrap replicates per calibration.
        n_refit : further calibrations with a new mask, split and (if ``base`` accepts it)
            base seed. Intervals then pool the bootstrap draws of every calibration, so they
            reflect masking and base-fit variation as well as which threads were sampled.
            The point estimates stay those of the first calibration.
        corpus_kwargs : passed to :meth:`topica.Corpus.from_documents` for every fit.
        refine : refine the grid optimum by a local search in log pseudo-count space.
        """
        import topica

        docs = [list(d) for d in docs]
        parents = [int(p) for p in parents]
        groups = None if groups is None else list(groups)   # positional, not by label
        if len(docs) != len(parents):
            raise ValueError("docs and parents must have the same length")
        if groups is not None and len(groups) != len(docs):
            raise ValueError("groups must have one entry per document")
        if not 0 < heldout_frac < 1 or not 0 < val_frac < 1:
            raise ValueError("heldout_frac and val_frac must be in (0, 1)")
        if n_refit < 0:
            raise ValueError("n_refit must be >= 0")
        if self.switch and groups is not None:
            raise ValueError("groups= is not supported with switch=True yet")
        corpus_kwargs = dict(corpus_kwargs or {})
        kw = dict(heldout_frac=heldout_frac, val_frac=val_frac, min_eval_tokens=min_eval_tokens,
                  n_shuffle=n_shuffle, n_boot=n_boot, corpus_kwargs=corpus_kwargs,
                  refine=refine)

        runs = [self._calibrate(docs, parents, base, groups, seed=seed, **kw)]
        for r in range(1, n_refit + 1):
            runs.append(self._calibrate(docs, parents, base, groups,
                                        seed=seed + 7919 * r, **kw))
        main = runs[0]
        self.uncertainty = "refit" if n_refit else "threads"

        def pct(x, lo=2.5, hi=97.5):
            return float(np.percentile(x, lo)), float(np.percentile(x, hi))

        a_draws = np.vstack([r_["draws_alpha"] for r_ in runs])
        self.alpha = {c: float(a) for c, a in zip(self.contexts, main["alpha"])}
        self.alpha_ci = {c: pct(a_draws[:, j]) for j, c in enumerate(self.contexts)}
        self.draws = {"alpha": a_draws,
                      "completion": np.concatenate([r_["draws_completion"] for r_ in runs])}

        self.rho = self.rho_ci = self.inherit_weights = None
        if self.switch:
            self.draws["rho"] = np.concatenate([r_["draws_rho"] for r_ in runs])
            self.rho = float(main["rho"])
            self.rho_ci = pct(self.draws["rho"])
            self.new_concentration = float(main["new_concentration"])
            self._switch_x = main["switch_x"]
            self._switch_share = main["switch_share"]

        # Under the switch the pseudo-counts are those of the inherit component, so the parent
        # share keeps its meaning: among replies that inherit, how much comes from the parent.
        share = self._share(main["alpha"])
        no_borrow = (np.mean(self.draws["rho"] < 0.01) if self.switch else None)
        if share is None:
            self.parent_share = self.parent_share_ci = self.parent_share_at_bound = None
            self.p_no_borrowing = float(np.mean(a_draws.sum(1) == 0) if no_borrow is None
                                        else no_borrow)
        else:
            sd = np.array([self._share(a) for a in a_draws])
            self.p_no_borrowing = float(np.mean(np.isnan(sd)) if no_borrow is None
                                        else no_borrow)
            self.draws["parent_share"] = sd
            sd = sd[~np.isnan(sd)]
            self.parent_share = share
            self.parent_share_ci = pct(sd) if sd.size else (np.nan, np.nan)
            ends = self._share_ends()
            # Refinement can move a pseudo-count past the grid; clip the share back to the
            # grid's range so the logit-scale bound holds for the point estimate too.
            self.parent_share = float(np.clip(share, ends[0], ends[1]))
            self.parent_share_at_bound = bool(np.isclose(self.parent_share, ends).any())

        lo, hi = pct(self.draws["completion"])
        self.completion = {"estimate": float(main["completion"]), "lo": lo, "hi": hi,
                           "by_length": main["by_length"]}
        if "edge_effect" in main:
            self.draws["edge_effect"] = np.concatenate([r_["draws_edge"] for r_ in runs])
            lo, hi = pct(self.draws["edge_effect"])
            self.edge_effect = {"estimate": float(main["edge_effect"]), "lo": lo, "hi": hi,
                                "placebo_alpha": {c: float(a) for c, a in
                                                  zip(self.contexts, main["placebo_alpha"])}}
        else:
            self.edge_effect = None
        self.op_effect = None
        if "op_effect" in main:
            self.draws["op_effect"] = np.concatenate([r_["draws_op"] for r_ in runs])
            lo, hi = pct(self.draws["op_effect"])
            self.op_effect = {"estimate": float(main["op_effect"]), "lo": lo, "hi": hi,
                              "placebo_alpha": {c: float(a) for c, a in
                                                zip(self.contexts, main["op_placebo_alpha"])}}
        self.alpha_by_group = ({g: {c: float(a) for c, a in zip(self.contexts, v)}
                                for g, v in main["alpha_by_group"].items()}
                               if groups is not None else None)
        self.replicates = None
        if n_refit:
            self.replicates = [{"alpha": {c: float(a) for c, a in zip(self.contexts, r_["alpha"])},
                                "parent_share": self._share(r_["alpha"]),
                                "completion": float(r_["completion"]),
                                "edge_effect": (float(r_["edge_effect"]) if "edge_effect" in r_
                                                else None),
                                "op_effect": (float(r_["op_effect"]) if "op_effect" in r_
                                              else None)} for r_ in runs]
            if self.switch:
                for rep_, r_ in zip(self.replicates, runs):
                    rep_["rho"] = float(r_["rho"])

        self.settings = {
            "contexts": self.contexts, "theta": self.theta_source, "seed": seed,
            "thread_excludes_parent": self.thread_excludes_parent, "switch": self.switch,
            "n_refit": n_refit, "strength_grid_size": int(self.strength_grid.size),
            "share_steps": self.share_steps, **kw, **main["settings"]}
        self.calibration_model = main["model"]

        self.theta_tilde = self.base_model = self.corpus = None
        if final:
            self.corpus = topica.Corpus.from_documents(docs, **corpus_kwargs)
            self.base_model = _call_base(base, self.corpus, seed)
            self.theta_tilde = self.transform(self.base_model, self.corpus, parents,
                                              groups=groups)
        return self

    # -- application -------------------------------------------------------------------

    def transform(self, model, corpus, parents: Sequence[int], *,
                  groups: Sequence | None = None) -> np.ndarray:
        """Smoothed topic mixes for a fitted ``model`` using the fitted pseudo-counts.

        ``corpus`` is the Corpus ``model`` was fit on; ``parents`` indexes its *original*
        documents (before any the vocabulary emptied were dropped). Returns ``(D, K)`` with
        one row per original document. A document with no in-vocabulary tokens gets its pure
        context mix; one with neither tokens nor context is left as NaN.
        """
        if self.alpha is None:
            raise RuntimeError("call fit() before transform()")
        groups = None if groups is None else list(groups)
        parents = [int(p) for p in parents]
        _, root, _ = thread_structure(parents)
        theta = _doc_topic(model, self.theta_source)
        kept = np.asarray(corpus.kept_indices, int)
        lengths = np.asarray(corpus.doc_lengths, float)
        full_theta, n, ctx = _context_mixes(theta, lengths, kept, parents, root, self.contexts,
                                            exclude_parent=self.thread_excludes_parent)
        if self.switch:
            return self._transform_switch(model, corpus, theta, full_theta, n, ctx)
        num = np.where(np.isnan(full_theta), 0.0, full_theta) * n[:, None]
        den = n.copy()
        for d in range(len(parents)):
            alpha = self.alpha
            if groups is not None and self.alpha_by_group:
                alpha = self.alpha_by_group.get(groups[d], self.alpha)
            for c in self.contexts:
                if not np.isnan(ctx[c][d, 0]) and alpha[c] > 0:
                    num[d] += alpha[c] * ctx[c][d]
                    den[d] += alpha[c]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den[:, None]
        out[den == 0] = np.nan
        return out

    def _transform_switch(self, model, corpus, theta, full_theta, n, ctx):
        """Switch smoothing: each document's inherit probability comes from the marginal
        likelihood of its tokens; the smoothed mix is that probability times the pooled
        shrinkage plus the rest times the document's own mix. Sets ``inherit_weights``."""
        beta = np.asarray(model.topic_word, float)
        vidx = {w: j for j, w in enumerate(corpus.vocabulary)}
        ids = [np.zeros(0, int)] * len(n)
        for r, toks in zip(np.asarray(corpus.kept_indices, int), corpus.documents()):
            ids[r] = np.array([vidx[w] for w in toks if w in vidx], int)
        A, a_new, rho = self._unpack(self._switch_x)
        mix = np.stack([ctx[c] for c in self.contexts], 1)                     # (D, C, K)
        wts = ~np.isnan(mix[:, :, 0]) * self._switch_share
        frac = wts.sum(1)                   # share of the pseudo-count whose context exists
        with np.errstate(invalid="ignore", divide="ignore"):
            wts = wts / frac[:, None]
        center = np.einsum("dc,dck->dk", wts, np.nan_to_num(mix))              # NaN if none
        a_doc = A * frac
        lml_inh = _sequential_log_ml(ids, center, CONC_GRID, beta)
        lml_new = _sequential_log_ml(ids, np.tile(theta.mean(0), (len(n), 1)), CONC_GRID,
                                     beta)
        w = self._inherit_prob(lml_inh, lml_new, a_doc, a_new, rho)[:, None]
        # A document with no tokens carries no evidence of its own, whether or not the corpus
        # kept its row: it gets its pure context mix (NaN below when it has none).
        own = np.where(np.isnan(full_theta) | (n[:, None] == 0), 0.0, full_theta)
        a_col = a_doc[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            shrunk = (n[:, None] * own + a_col * np.nan_to_num(center)) / (n[:, None] + a_col)
        shrunk = np.where(np.isnan(shrunk), 0.0, shrunk)
        w = np.where(n[:, None] == 0, (frac > 0)[:, None].astype(float), w)
        out = w * shrunk + (1 - w) * own
        # A document with neither tokens nor context has nothing to report.
        with np.errstate(invalid="ignore"):
            out = out / out.sum(1, keepdims=True)
        self.inherit_weights = w[:, 0]
        return out

    def summary(self) -> dict:
        """The headline estimates as one dict."""
        if self.alpha is None:
            raise RuntimeError("call fit() first")
        return {"alpha": self.alpha, "alpha_ci": self.alpha_ci,
                "parent_share": self.parent_share, "parent_share_ci": self.parent_share_ci,
                "parent_share_at_bound": self.parent_share_at_bound,
                "p_no_borrowing": self.p_no_borrowing, "uncertainty": self.uncertainty,
                "replicates": self.replicates,
                "completion": self.completion, "edge_effect": self.edge_effect,
                "op_effect": self.op_effect,
                "alpha_by_group": self.alpha_by_group, "rho": self.rho, "rho_ci": self.rho_ci,
                "settings": self.settings}

    def __repr__(self) -> str:
        if self.alpha is None:
            return f"ThreadSmoother(contexts={self.contexts}, unfitted)"
        a = ", ".join(f"{c}={v:.3g}" for c, v in self.alpha.items())
        e = self.edge_effect
        edge = (f", edge_effect={e['estimate']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]"
                if e else "")
        return f"ThreadSmoother({a}{edge})"


# --------------------------------------------------------------------------- the model

_BASES = ("lda", "stm", "ctm")


class ThreadTM:
    """Threaded topic model: a standard base topic model plus thread-context shrinkage
    (experimental).

    ``ThreadTM`` fits an ordinary topic model (``base="lda"``, ``"stm"``, ``"ctm"``, or a
    factory) and then smooths each document's topic mix toward its reply context with
    pseudo-counts estimated on held-out replies (see :class:`ThreadSmoother`, which does the
    work). The topics are the base model's; ``doc_topic`` is the smoothed mix.

    Parameters
    ----------
    num_topics : number of topics.
    base : ``"lda"`` (default), ``"stm"`` (needs ``prevalence`` in :meth:`fit`; five
        best-bound restarts unless ``fit_kwargs`` says otherwise), ``"ctm"``, or a factory
        ``base(corpus)`` / ``base(corpus, seed)`` returning a fitted model.
    seed : seed for the base fits and the calibration.
    contexts, theta, thread_excludes_parent, strength_grid, share_steps, switch : passed to
        :class:`ThreadSmoother`.
    base_kwargs : extra constructor arguments for a named base.

    Attributes (after :meth:`fit`)
    ------------------------------
    topic_word, vocabulary : the base model's topics and vocabulary.
    doc_topic : smoothed topic mixes, one row per document the corpus kept (rows follow
        ``corpus.kept_indices``).
    doc_topic_all : ``(D, K)`` smoothed topic mixes, one row per input document.
    base_doc_topic : the unsmoothed base mixes, one row per document the corpus kept.
    alpha, alpha_ci, parent_share, parent_share_ci, parent_share_at_bound, p_no_borrowing,
    edge_effect, completion, alpha_by_group, draws, replicates, rho, rho_ci,
    inherit_weights : see :class:`ThreadSmoother`.
    base_model, corpus, smoother : the full-data base fit, its Corpus, and the calibrated
        :class:`ThreadSmoother`.
    """

    def __init__(self, num_topics: int, *, base="lda", seed: int = 13,
                 contexts: Sequence[str] = CONTEXTS, theta: str = "auto",
                 thread_excludes_parent: bool = True,
                 strength_grid: Sequence[float] | None = None, share_steps: int = 97,
                 switch: bool = False, base_kwargs: dict | None = None) -> None:
        _gate()
        if int(num_topics) < 1:
            raise ValueError("num_topics must be >= 1")
        if not callable(base) and base not in _BASES:
            raise ValueError(f"base must be one of {_BASES} or a callable")
        self.num_topics = int(num_topics)
        self.base = base
        self.seed = int(seed)
        self.base_kwargs = dict(base_kwargs or {})
        self.smoother = ThreadSmoother(contexts, theta=theta,
                                       thread_excludes_parent=thread_excludes_parent,
                                       strength_grid=strength_grid, share_steps=share_steps,
                                       switch=switch)
        self.settings = {"num_topics": self.num_topics,
                         "base": base if isinstance(base, str) else "callable",
                         "seed": self.seed, "contexts": list(contexts), "theta": theta,
                         "thread_excludes_parent": bool(thread_excludes_parent),
                         "strength_grid": (None if strength_grid is None
                                           else [float(x) for x in strength_grid]),
                         "share_steps": int(share_steps), "switch": bool(switch),
                         "base_kwargs": self.base_kwargs}
        self._fitted = False

    def _factory(self, prevalence, iters, fit_kwargs):
        import topica

        if callable(self.base):
            return self.base
        fit_kw = dict(fit_kwargs or {})
        if iters is not None:
            fit_kw["iters"] = iters
        k, kw = self.num_topics, self.base_kwargs
        if self.base == "lda":
            return lambda c, seed: topica.LDA(k, seed=seed, **kw).fit(c, **fit_kw)
        if self.base == "ctm":
            return lambda c, seed: topica.CTM(k, seed=seed, **kw).fit(c, **fit_kw)
        if prevalence is None:
            raise ValueError("base='stm' needs prevalence=; use base='ctm' for no covariates")
        X = np.asarray(prevalence, float)
        fit_kw.setdefault("restarts", 5)
        # STM's design rows must follow the documents the corpus kept.
        return lambda c, seed: topica.STM(k, seed=seed, **kw).fit(
            c, prevalence=X[np.asarray(c.kept_indices)], **fit_kw)

    def fit(self, docs: Sequence[Sequence[str]], parents: Sequence[int], *,
            prevalence=None, groups: Sequence | None = None, iters: int | None = None,
            fit_kwargs: dict | None = None, corpus_kwargs: dict | None = None,
            heldout_frac: float = 0.5, val_frac: float = 0.5, min_eval_tokens: int = 5,
            n_shuffle: int = 3, n_boot: int = 500, n_refit: int = 0) -> "ThreadTM":
        """Fit the base model, calibrate the thread pseudo-counts, and smooth.

        ``docs`` is one token list per document and ``parents`` the parent index per document
        (negative for a thread root). ``prevalence`` is the STM design, one row per input
        document. ``iters`` and ``fit_kwargs`` go to the base model's ``fit``. The remaining
        arguments are :meth:`ThreadSmoother.fit`'s.
        """
        if prevalence is not None and len(prevalence) != len(docs):
            raise ValueError("prevalence must have one row per document")
        factory = self._factory(prevalence, iters, fit_kwargs)
        self.smoother.fit(docs, parents, base=factory, groups=groups, seed=self.seed,
                          heldout_frac=heldout_frac, val_frac=val_frac,
                          min_eval_tokens=min_eval_tokens, n_shuffle=n_shuffle,
                          n_boot=n_boot, n_refit=n_refit, corpus_kwargs=corpus_kwargs,
                          final=True)
        self._fitted = True
        return self

    def _need(self):
        if not self._fitted:
            raise RuntimeError("ThreadTM is not fitted; call fit() first")

    # -- the model surface ------------------------------------------------------------

    @property
    def base_model(self):
        self._need()
        return self.smoother.base_model

    @property
    def corpus(self):
        self._need()
        return self.smoother.corpus

    @property
    def topic_word(self) -> np.ndarray:
        return np.asarray(self.base_model.topic_word, float)

    @property
    def vocabulary(self) -> list:
        return list(self.corpus.vocabulary)

    @property
    def doc_topic(self) -> np.ndarray:
        """Smoothed topic mixes, one row per document the corpus kept (rows follow
        ``corpus.kept_indices``, like every topica model)."""
        self._need()
        return self.smoother.theta_tilde[np.asarray(self.corpus.kept_indices)]

    @property
    def doc_topic_all(self) -> np.ndarray:
        """Smoothed topic mixes, one row per *input* document, including documents the
        vocabulary emptied (they get their pure context mix; NaN if they have none)."""
        self._need()
        return self.smoother.theta_tilde

    @property
    def base_doc_topic(self) -> np.ndarray:
        return _doc_topic(self.base_model, self.smoother.theta_source)

    def top_words(self, n: int = 10, topic: int | None = None, weights: bool = False):
        """The base model's top words (topics are the base model's)."""
        return self.base_model.top_words(n, topic=topic, weights=weights)

    def transform(self, model, corpus, parents: Sequence[int], *, groups=None) -> np.ndarray:
        """Smooth another fitted model over the same documents with these pseudo-counts."""
        self._need()
        return self.smoother.transform(model, corpus, parents, groups=groups)

    def summary(self) -> dict:
        self._need()
        return self.smoother.summary()

    def __getattr__(self, name):
        # Calibration results live on the smoother (alpha, parent_share, edge_effect, ...).
        if name in ("alpha", "alpha_ci", "parent_share", "parent_share_ci",
                    "parent_share_at_bound", "p_no_borrowing", "edge_effect", "completion",
                    "alpha_by_group", "draws", "replicates", "uncertainty", "rho", "rho_ci",
                    "inherit_weights", "op_effect"):
            if not self.__dict__.get("_fitted"):
                raise AttributeError(f"{name} is available after fit()")
            return getattr(self.__dict__["smoother"], name)
        raise AttributeError(name)

    def __repr__(self) -> str:
        base = self.base if isinstance(self.base, str) else "callable"
        if not self._fitted:
            return f"ThreadTM(num_topics={self.num_topics}, base={base!r}, unfitted)"
        return (f"ThreadTM(num_topics={self.num_topics}, base={base!r}, "
                + repr(self.smoother)[len("ThreadSmoother("):])
