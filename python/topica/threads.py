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
of every other document in its thread. This is the posterior mean of :math:`\\theta_d` under a
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

__all__ = ["ThreadSmoother", "strip_quotes", "strip_copied_runs", "shuffle_parents",
           "thread_structure"]

CONTEXTS = ("parent", "thread")
DEFAULT_ALPHA_GRID = np.concatenate([[0.0], np.geomspace(0.25, 500.0, 20)])

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


def _context_mixes(theta, lengths, kept, parents, root, contexts):
    """Per-original-document context topic mixes (NaN rows where unavailable)."""
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
            own = n[d] * full_theta[d] if row[d] >= 0 else np.zeros(k)
            rest = nsum[root[d]] - n[d]
            if rest > 0:
                thr[d] = (wsum[root[d]] - own) / rest
        out["thread"] = thr
    return full_theta, n, out


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
    alpha_grid : candidate pseudo-counts, searched jointly for every context. Defaults to 0
        plus 20 log-spaced values from 0.25 to 500.

    Attributes (after :meth:`fit`)
    ------------------------------
    alpha : dict ``{context: pseudo-count}`` chosen on validation threads.
    alpha_ci : dict ``{context: (lo, hi)}`` 95% thread-bootstrap interval.
    parent_share, parent_share_ci : ``a_p / (a_p + a_t)`` and its interval (both contexts).
    alpha_by_group : per-group pseudo-counts when ``groups`` is given to :meth:`fit`.
    completion : held-out gain over the base on test threads, nats per token:
        ``{"estimate", "lo", "hi", "by_length"}``.
    edge_effect : held-out gain of the true tree over the shuffled-parent placebo on test
        threads, nats per token: ``{"estimate", "lo", "hi"}``.
    theta_tilde : ``(D, K)`` smoothed topic mixes of a full-data base fit, one row per input
        document (a document the vocabulary empties gets its pure context mix).
    base_model, corpus : that full-data fit and its Corpus.
    settings : the calibration settings.
    """

    def __init__(self, contexts: Sequence[str] = CONTEXTS, *, theta: str = "auto",
                 alpha_grid: Sequence[float] | None = None) -> None:
        _gate()
        contexts = tuple(contexts)
        bad = set(contexts) - set(CONTEXTS)
        if not contexts or bad:
            raise ValueError(f"contexts must be a non-empty subset of {CONTEXTS}")
        if theta not in ("auto", "doc_topic", "posterior"):
            raise ValueError("theta must be 'auto', 'doc_topic' or 'posterior'")
        grid = np.asarray(DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid, float)
        if grid.ndim != 1 or grid.size == 0 or (grid < 0).any():
            raise ValueError("alpha_grid must be a non-empty 1-D array of non-negative values")
        self.contexts = contexts
        self.theta_source = theta
        self.alpha_grid = np.unique(grid)
        self.alpha = None
        self.alpha_by_group = None

    # -- calibration ------------------------------------------------------------------

    def fit(self, docs: Sequence[Sequence[str]], parents: Sequence[int], *,
            base: Callable, groups: Sequence | None = None, heldout_frac: float = 0.5,
            val_frac: float = 0.5, min_eval_tokens: int = 5, n_shuffle: int = 3,
            n_boot: int = 500, seed: int = 13, corpus_kwargs: dict | None = None,
            final: bool = True) -> "ThreadSmoother":
        """Estimate the pseudo-counts on held-out replies, then smooth a full-data fit.

        Parameters
        ----------
        docs : one token list per document.
        parents : parent index per document (negative for a thread root).
        base : factory ``base(corpus) -> fitted model``; called once on the masked corpus for
            calibration and once on the full corpus when ``final``. The model must expose
            ``doc_topic`` (rows aligned with ``corpus.kept_indices``) and a ``(K, V)``
            ``topic_word``.
        groups : optional community label per document; pseudo-counts are then also fit per
            group (``alpha_by_group``) and used by :meth:`transform`.
        heldout_frac : share of each eligible leaf's tokens held out for scoring.
        val_frac : share of evaluation threads used to choose the pseudo-counts; the rest are
            the test threads every reported number comes from.
        min_eval_tokens : leaves with fewer tokens are not masked.
        n_shuffle : placebo draws; the placebo parent mix is their average.
        n_boot : thread-bootstrap replicates for every interval.
        corpus_kwargs : passed to :meth:`topica.Corpus.from_documents` for both fits.
        """
        import topica

        docs = [list(d) for d in docs]
        parents = [int(p) for p in parents]
        if len(docs) != len(parents):
            raise ValueError("docs and parents must have the same length")
        if groups is not None and len(groups) != len(docs):
            raise ValueError("groups must have one entry per document")
        if not 0 < heldout_frac < 1 or not 0 < val_frac < 1:
            raise ValueError("heldout_frac and val_frac must be in (0, 1)")
        corpus_kwargs = dict(corpus_kwargs or {})
        rng = np.random.default_rng(seed)
        depth, root, has_child = thread_structure(parents)

        # 1. Mask half the tokens of eligible leaves; nothing held out enters any context.
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
        model = base(corpus)
        theta = _doc_topic(model, self.theta_source)
        beta = np.asarray(model.topic_word, float)
        kept = np.asarray(corpus.kept_indices, int)
        lengths = np.asarray(corpus.doc_lengths, float)
        if theta.shape[0] != len(kept) or beta.shape[0] != theta.shape[1]:
            raise ValueError("base model's doc_topic / topic_word do not match its corpus")
        vidx = {w: j for j, w in enumerate(corpus.vocabulary)}

        _, n_doc, ctx = _context_mixes(theta, lengths, kept, parents, root, self.contexts)
        shuf_mix = []
        for s in range(n_shuffle):
            sp = shuffle_parents(parents, rng=np.random.default_rng(seed * 1000 + s))
            shuf_mix.append(_context_mixes(theta, lengths, kept, sp, root, ("parent",))[2]
                            ["parent"])
        row = np.full(len(docs), -1)
        row[kept] = np.arange(len(kept))

        # 2. Common scored support: the leaf, every requested context and every placebo
        #    context exist; at least one in-vocabulary held token.
        leaves = []
        for i, toks in held.items():
            ids = np.array([vidx[w] for w in toks if w in vidx], int)
            if ids.size == 0 or row[i] < 0:
                continue
            if any(np.isnan(ctx[c][i, 0]) for c in self.contexts):
                continue
            if "parent" in self.contexts and any(np.isnan(m_[i, 0]) for m_ in shuf_mix):
                continue
            leaves.append((i, ids))
        if not leaves:
            raise ValueError("no evaluable leaf replies after vocabulary filtering")
        roots = np.array(sorted({root[i] for i, _ in leaves}))
        if len(roots) < 4:
            raise ValueError("need at least four threads with evaluable leaves")
        val_roots = set(rng.choice(roots, size=max(1, int(round(val_frac * len(roots)))),
                                   replace=False).tolist())

        # 3. Flat per-token arrays.
        tpos = {r: j for j, r in enumerate(roots)}
        pL, n_tok, tj, grp = [], [], [], []
        P = {c: [] for c in self.contexts}
        P_shuf = []
        for i, ids in leaves:
            r = row[i]
            pL.append(theta[r] @ beta[:, ids])
            for c in self.contexts:
                P[c].append(ctx[c][i] @ beta[:, ids])
            if "parent" in self.contexts:
                P_shuf.append(np.mean([m_[i] @ beta[:, ids] for m_ in shuf_mix], 0))
            n_tok.append(np.full(ids.size, n_doc[i]))
            tj.append(np.full(ids.size, tpos[root[i]]))
            grp.extend([None if groups is None else groups[i]] * ids.size)
        pL = np.concatenate(pL)
        P = {c: np.concatenate(v) for c, v in P.items()}
        if P_shuf:
            P["_shuffled"] = np.concatenate(P_shuf)
        n_tok = np.concatenate(n_tok)
        tj = np.concatenate(tj)
        grp = np.array(grp, dtype=object)
        is_val_thread = np.array([r in val_roots for r in roots])
        n_threads = len(roots)

        grid = [np.array(g) for g in product(self.alpha_grid, repeat=len(self.contexts))]

        def tables(ctx_keys, mask):
            """Per-thread log-likelihood sums for every grid point, on tokens in ``mask``."""
            cs = [P[c][mask] for c in ctx_keys]
            nn, pl, tt = n_tok[mask], pL[mask], tj[mask]
            out = np.empty((len(grid), n_threads))
            for g_i, g in enumerate(grid):
                denom = nn + g.sum()
                mix = nn / denom * pl
                for a_c, c in zip(g, cs):
                    mix = mix + a_c / denom * c
                out[g_i] = np.bincount(tt, np.log(mix), n_threads)
            base_t = np.bincount(tt, np.log(pl), n_threads)
            tok_t = np.bincount(tt, None, n_threads)
            return out, base_t, tok_t

        brng = np.random.default_rng(seed + 1)
        val_idx = np.flatnonzero(is_val_thread)
        test_idx = np.flatnonzero(~is_val_thread)
        boot_val = val_idx[brng.integers(0, len(val_idx), (n_boot, len(val_idx)))]
        boot_test = test_idx[brng.integers(0, len(test_idx), (n_boot, len(test_idx)))]

        def pick(tab, tok, idx):
            return int(np.argmax(tab[:, idx].sum(1) / max(tok[idx].sum(), 1)))

        def ci_of(per_thread, tok):
            est = per_thread[test_idx].sum() / tok[test_idx].sum()
            bs = per_thread[boot_test].sum(1) / tok[boot_test].sum(1)
            return {"estimate": float(est), "lo": float(np.percentile(bs, 2.5)),
                    "hi": float(np.percentile(bs, 97.5))}

        everything = np.ones(pL.size, bool)
        tab, base_t, tok_t = tables(self.contexts, everything)
        g_star = pick(tab, tok_t, val_idx)
        self.alpha = {c: float(a) for c, a in zip(self.contexts, grid[g_star])}
        boot_g = [int(np.argmax(tab[:, b].sum(1) / tok_t[b].sum())) for b in boot_val]
        boot_alpha = np.array([grid[g] for g in boot_g])
        self.alpha_ci = {c: (float(np.percentile(boot_alpha[:, j], 2.5)),
                             float(np.percentile(boot_alpha[:, j], 97.5)))
                         for j, c in enumerate(self.contexts)}
        if set(self.contexts) == set(CONTEXTS):
            jp, jt = self.contexts.index("parent"), self.contexts.index("thread")

            def share(a):
                s_ = a[jp] + a[jt]
                return a[jp] / s_ if s_ > 0 else np.nan
            self.parent_share = float(share(grid[g_star]))
            shares = np.array([share(a) for a in boot_alpha])
            shares = shares[~np.isnan(shares)]
            self.parent_share_ci = ((float(np.percentile(shares, 2.5)),
                                     float(np.percentile(shares, 97.5)))
                                    if shares.size else (np.nan, np.nan))
        else:
            self.parent_share = self.parent_share_ci = None

        gain_t = tab[g_star] - base_t
        self.completion = ci_of(gain_t, tok_t)
        # Gain by observed-length tercile (cuts from validation leaves).
        val_tok = np.isin(tj, val_idx)
        cuts = np.quantile(n_tok[val_tok], [1 / 3, 2 / 3])
        ter = np.digitize(n_tok, cuts, right=True)
        g_vec = grid[g_star]
        denom = n_tok + g_vec.sum()
        mix = n_tok / denom * pL
        for a_c, c in zip(g_vec, self.contexts):
            mix = mix + a_c / denom * P[c]
        tok_gain = np.log(mix) - np.log(pL)
        test_tok = ~val_tok
        self.completion["by_length"] = {
            "cuts": [float(c_) for c_ in cuts],
            "gain": [float(tok_gain[test_tok & (ter == t_)].mean()) for t_ in range(3)]}

        if "parent" in self.contexts:
            placebo_keys = tuple("_shuffled" if c == "parent" else c for c in self.contexts)
            tab_s, _, _ = tables(placebo_keys, everything)
            g_s = pick(tab_s, tok_t, val_idx)
            self.edge_effect = ci_of(tab[g_star] - tab_s[g_s], tok_t)
            self.edge_effect["placebo_alpha"] = {
                c: float(a) for c, a in zip(self.contexts, grid[g_s])}
        else:
            self.edge_effect = None

        if groups is not None:
            self.alpha_by_group = {}
            for g_lab in sorted(set(grp.tolist()), key=str):
                m = grp == g_lab
                tab_g, _, tok_g = tables(self.contexts, m)
                if tok_g[val_idx].sum() == 0:
                    continue
                self.alpha_by_group[g_lab] = {
                    c: float(a) for c, a in zip(self.contexts, grid[pick(tab_g, tok_g, val_idx)])}

        self.settings = {
            "contexts": self.contexts, "theta": self.theta_source,
            "heldout_frac": heldout_frac, "val_frac": val_frac,
            "min_eval_tokens": min_eval_tokens, "n_shuffle": n_shuffle, "n_boot": n_boot,
            "seed": seed, "corpus_kwargs": corpus_kwargs,
            "n_eval_leaves": len(leaves), "n_eval_threads": n_threads,
            "n_val_threads": int(is_val_thread.sum()), "n_test_tokens": int(tok_t[test_idx].sum()),
        }
        self.calibration_model = model

        self.theta_tilde = self.base_model = self.corpus = None
        if final:
            self.corpus = topica.Corpus.from_documents(docs, **corpus_kwargs)
            self.base_model = base(self.corpus)
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
        parents = [int(p) for p in parents]
        _, root, _ = thread_structure(parents)
        theta = _doc_topic(model, self.theta_source)
        kept = np.asarray(corpus.kept_indices, int)
        lengths = np.asarray(corpus.doc_lengths, float)
        full_theta, n, ctx = _context_mixes(theta, lengths, kept, parents, root, self.contexts)
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
