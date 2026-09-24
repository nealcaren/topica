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
not help on average", not "no reply follows its parent".

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
DEFAULT_STRENGTH_GRID = np.geomspace(0.1, 1000.0, 40)

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


def _context_mixes(theta, lengths, kept, parents, root, contexts, exclude_parent=False):
    """Per-original-document context topic mixes (NaN rows where unavailable).

    The thread mix always excludes the document itself; with ``exclude_parent`` it also
    excludes the document's parent, so the parent and thread contexts do not overlap."""
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
    if "thread" in contexts:
        wsum = defaultdict(lambda: np.zeros(k))
        nsum = defaultdict(float)
        for d in np.asarray(kept, int):
            wsum[root[d]] += n[d] * full_theta[d]
            nsum[root[d]] += n[d]
        thr = np.full((n_docs, k), np.nan)
        for d in range(n_docs):
            drop = [d]
            if exclude_parent and parents[d] >= 0:
                drop.append(parents[d])
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
    out = []
    for s in range(n_shuffle):
        sp = shuffle_parents(parents, rng=np.random.default_rng(seed * 1000 + s))
        out.append(_context_mixes(theta, lengths, kept, sp, root, contexts,
                                  exclude_parent=exclude_parent)[2])
    return out


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
    contexts : which reply contexts to borrow from: any of ``"parent"`` and ``"thread"``.
    theta : the base topic mix. ``"auto"`` uses ``posterior_doc_topic()`` when the base model
        has it (the logistic-normal models) and ``doc_topic`` otherwise; ``"doc_topic"`` and
        ``"posterior"`` force one.
    strength_grid : candidate total pseudo-counts ``s = a_p + a_t``. Defaults to 0 plus 40
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

    Attributes (after :meth:`fit`)
    ------------------------------
    alpha : dict ``{context: pseudo-count}``, the validation-optimal point estimate (grid
        search, then continuous refinement).
    alpha_ci : dict ``{context: (lo, hi)}`` 95% bootstrap interval.
    parent_share, parent_share_ci : ``a_p / (a_p + a_t)`` and its 95% percentile interval
        (both contexts), estimated on the logit-spaced share grid. ``parent_share_at_bound``
        is ``True`` when the estimate is the outermost grid value, meaning the data cannot
        distinguish the smaller pseudo-count from zero. ``p_no_borrowing`` is the share of
        bootstrap draws that chose no borrowing at all (share undefined); those draws are
        excluded from the interval and counted here.
    alpha_by_group : per-group pseudo-counts when ``groups`` is given to :meth:`fit`.
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
                 thread_excludes_parent: bool = True) -> None:
        _gate()
        contexts = tuple(contexts)
        bad = set(contexts) - set(CONTEXTS)
        if not contexts or bad or len(set(contexts)) != len(contexts):
            raise ValueError(f"contexts must be a non-empty subset of {CONTEXTS}")
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
        self.alpha = None
        self.alpha_by_group = None

    # -- parameterization ---------------------------------------------------------------

    def _grid(self):
        """Candidate pseudo-count vectors, ordered as ``self.contexts``.

        Two contexts are searched on (strength s, parent share pi) so the share is resolved
        directly; one context on strength alone. s = 0 appears once (no borrowing)."""
        pts = [np.zeros(len(self.contexts))]
        if len(self.contexts) == 1:
            pts += [np.array([s]) for s in self.strength_grid[1:]]
        else:
            jp = self.contexts.index("parent")
            for s in self.strength_grid[1:]:
                for pi in self._share_values():
                    a = np.empty(2)
                    a[jp], a[1 - jp] = s * pi, s * (1 - pi)
                    pts.append(a)
        return pts

    def _share_values(self):
        return 1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, self.share_steps)))

    def _share(self, a):
        if set(self.contexts) != set(CONTEXTS):
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
        row = np.full(len(docs), -1)
        row[kept] = np.arange(len(kept))

        # Common scored support: the leaf, every requested context and every placebo context
        # exist; at least one in-vocabulary held token.
        leaves = []
        for i, toks in held.items():
            ids = np.array([vidx[w] for w in toks if w in vidx], int)
            if ids.size == 0 or row[i] < 0:
                continue
            if any(np.isnan(ctx[c][i, 0]) for c in self.contexts):
                continue
            if any(np.isnan(sc[c][i, 0]) for sc in shuf_ctx for c in self.contexts):
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
        P = {c: [] for c in self.contexts}
        P_shuf = {c: [] for c in self.contexts} if shuf_ctx else {}
        for i, ids in leaves:
            r = row[i]
            pL.append(theta[r] @ beta[:, ids])
            for c in self.contexts:
                P[c].append(ctx[c][i] @ beta[:, ids])
            for c in P_shuf:
                P_shuf[c].append(np.mean([sc[c][i] @ beta[:, ids] for sc in shuf_ctx], 0))
            n_tok.append(np.full(ids.size, n_doc[i]))
            tj.append(np.full(ids.size, tpos[root[i]]))
            grp.extend([None if groups is None else groups[i]] * ids.size)
        pL = np.concatenate(pL)
        P = {c: np.concatenate(v) for c, v in P.items()}
        for c, v in P_shuf.items():
            P["_shuffled_" + c] = np.concatenate(v)
        n_tok = np.concatenate(n_tok)
        tj = np.concatenate(tj)
        grp = np.array(grp, dtype=object)
        n_threads = len(roots)
        is_val = np.array([r in val_roots for r in roots])
        val_idx, test_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)

        grid = self._grid()

        def token_ll(a, keys, mask):
            nn, pl = n_tok[mask], pL[mask]
            denom = nn + a.sum()
            mix = nn / denom * pl
            for a_c, c in zip(a, keys):
                mix = mix + a_c / denom * P[c][mask]
            return np.log(mix)

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

        out["settings"] = {"n_eval_leaves": len(leaves), "n_eval_threads": n_threads,
                           "n_val_threads": int(is_val.sum()),
                           "n_test_tokens": int(tok_t[test_idx].sum())}
        out["model"] = model
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

        share = self._share(main["alpha"])
        if share is None:
            self.parent_share = self.parent_share_ci = self.parent_share_at_bound = None
            self.p_no_borrowing = float(np.mean(a_draws.sum(1) == 0))
        else:
            sd = np.array([self._share(a) for a in a_draws])
            self.p_no_borrowing = float(np.mean(np.isnan(sd)))
            self.draws["parent_share"] = sd
            sd = sd[~np.isnan(sd)]
            self.parent_share = share
            self.parent_share_ci = pct(sd) if sd.size else (np.nan, np.nan)
            ends = self._share_values()[[0, -1]]
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
        self.alpha_by_group = ({g: {c: float(a) for c, a in zip(self.contexts, v)}
                                for g, v in main["alpha_by_group"].items()}
                               if groups is not None else None)
        self.replicates = None
        if n_refit:
            self.replicates = [{"alpha": {c: float(a) for c, a in zip(self.contexts, r_["alpha"])},
                                "parent_share": self._share(r_["alpha"]),
                                "completion": float(r_["completion"]),
                                "edge_effect": (float(r_["edge_effect"]) if "edge_effect" in r_
                                                else None)} for r_ in runs]

        self.settings = {
            "contexts": self.contexts, "theta": self.theta_source, "seed": seed,
            "thread_excludes_parent": self.thread_excludes_parent,
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
                "alpha_by_group": self.alpha_by_group, "settings": self.settings}

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
    contexts, theta, thread_excludes_parent, strength_grid, share_steps : passed to
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
    edge_effect, completion, alpha_by_group, draws, replicates : see :class:`ThreadSmoother`.
    base_model, corpus, smoother : the full-data base fit, its Corpus, and the calibrated
        :class:`ThreadSmoother`.
    """

    def __init__(self, num_topics: int, *, base="lda", seed: int = 13,
                 contexts: Sequence[str] = CONTEXTS, theta: str = "auto",
                 thread_excludes_parent: bool = True,
                 strength_grid: Sequence[float] | None = None, share_steps: int = 97,
                 base_kwargs: dict | None = None) -> None:
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
                                       strength_grid=strength_grid, share_steps=share_steps)
        self.settings = {"num_topics": self.num_topics,
                         "base": base if isinstance(base, str) else "callable",
                         "seed": self.seed, "contexts": list(contexts), "theta": theta,
                         "thread_excludes_parent": bool(thread_excludes_parent),
                         "strength_grid": (None if strength_grid is None
                                           else [float(x) for x in strength_grid]),
                         "share_steps": int(share_steps), "base_kwargs": self.base_kwargs}
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
                    "alpha_by_group", "draws", "replicates", "uncertainty"):
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
