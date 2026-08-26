"""Choosing the number of topics or among fits: search_k, select_model, quality frontier.

Split out of the former monolithic ``topica.validation`` (issue #757). The names
here are also re-exported from :mod:`topica.validation` (a compatibility shim) and
from the workflow namespace :mod:`topica.select`.
"""

from __future__ import annotations

import html as _html
import inspect
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .coherence import (
    _as_topic_word, _as_doc_topic, _vocabulary_of, _rbo, _ref_corpus,
    coherence as _coherence, exclusivity as _exclusivity,
)

from .evaluate import (
    Heldout, _kl, check_residuals, eval_heldout, make_heldout, perplexity,
)
# Cross-validation is a model-selection tool; co-locate it with search_k /
# select_model (crossval imports from evaluate/coherence, not select — no cycle).
from .crossval import cross_validate, make_folds, Folds, CrossValResult  # noqa: F401

__all__ = [
    'BestKExplanation',
    'CrossValResult',
    'Folds',
    'Heldout',
    'SEARCH_K_DIRECTIONS',
    'SearchKResult',
    'SelectModelResult',
    'check_residuals',
    'cross_validate',
    'eval_heldout',
    'make_folds',
    'make_heldout',
    'perplexity',
    'plot_models',
    'plot_search_k',
    'plot_topic_discovery',
    'quality_frontier',
    'search_k',
    'select_model',
]



# ---------------------------------------------------------------------------
# searchK: fit across topic counts, report quality
# ---------------------------------------------------------------------------

# Whether a higher or lower value of each metric is better. Coherence here is
# mean UMass (negative; less-negative is better), so "maximize".
# Default number of seeds per K in search_k. Flip this one value to make
# confidence intervals (num_seeds>1) the default; every downstream path already
# handles both cases.
_SEARCH_K_DEFAULT_SEEDS = 1


SEARCH_K_DIRECTIONS = {
    "coherence": "maximize",
    "exclusivity": "maximize",
    "heldout_loglik": "maximize",
    "perplexity": "minimize",
    "polarization": "maximize",
    # NMF/LSA reconstruction error: a scree curve, monotone-decreasing in K, so
    # rule='best' returns the grid edge (guarded below) and rule='elbow' is the
    # useful pick. Kept out of the frontier and the best_k default; selectable
    # only when named explicitly.
    "reconstruction_error": "minimize",
    # Opt-in ldatuning-style criteria (search_k(criteria=...)).
    "deveaud": "maximize",   # mean pairwise JS divergence between topics
    "cao_juan": "minimize",  # mean pairwise cosine similarity between topics
}


# Extra K-selection criteria computable from the topic-word matrix, requested via
# search_k(criteria=...). Off by default and out of the frontier.
_SEARCH_K_CRITERIA = ("deveaud", "cao_juan")



def _argbest_k(rows, score):
    """The ``k`` at the maximum score, breaking ties toward the *smallest* ``k``.

    Parsimony tie-break: with two K values equally good, prefer the simpler model.
    This also makes the pick independent of the order ``ks`` was passed in (a plain
    ``argmax`` returns the first index, so it would depend on grid ordering)."""
    score = np.asarray(score, dtype=np.float64)
    best = np.nanmax(score)
    tied = [i for i, s in enumerate(score) if s == best]
    return int(min(rows[i]["k"] for i in tied))



def _resolve_workers(n_jobs, n_tasks):
    """Thread-worker count for a grid of ``n_tasks`` fits. ``n_jobs=1`` stays
    serial; ``n_jobs<=0`` or ``None`` uses all cores; otherwise ``min(n_jobs,
    n_tasks)``. A single-task grid never parallelizes."""
    if n_tasks <= 1:
        return 1
    # None / non-positive / non-finite (inf, nan) -> all cores.
    if n_jobs is None or not np.isfinite(n_jobs) or n_jobs <= 0:
        import os
        n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), n_tasks))



def _frontier_score(coherence, exclusivity):
    """``z(coherence) + z(exclusivity)`` across a K-grid, both maximized.

    Each metric is z-scored across the scanned K values (comparable scales) and
    summed. A zero-variance metric contributes nothing; a NaN in the *other*
    metric is neutralized before scoring, but a K with a non-finite value in
    *either* metric is marked degenerate (``-inf``) so it is never recommended --
    unless every K is degenerate, in which case the all-zero score falls back to
    the smallest K. Shared by the aggregate frontier and the per-seed frontier so
    the two use identical logic."""
    score = np.zeros(len(coherence))
    finite = np.ones(len(coherence), dtype=bool)
    for v in (np.asarray(coherence, np.float64), np.asarray(exclusivity, np.float64)):
        finite &= np.isfinite(v)
        sd = np.nanstd(v)
        if sd > 0:
            score += np.nan_to_num((v - np.nanmean(v)) / sd, nan=0.0)
    if finite.any():
        score[~finite] = -np.inf
    return score



class BestKExplanation(int):
    """The return of ``SearchKResult.best_k(explain=True)``: the chosen ``k`` as a
    plain ``int`` (so it drops into ``LDA(num_topics=best_k)`` unchanged and
    compares equal to the bare pick), carrying the reasoning behind it.

    Attributes
    ----------
    metric : str
        The metric actually optimized (``"frontier"`` or a column name), after
        the ``metric=None`` default was resolved.
    rule : str
        The ``rule`` used (``"best"`` / ``"1se"`` / ``"elbow"``).
    rule_desc : str
        A one-line plain description of the composite/curve the pick came from,
        e.g. ``"frontier knee (max of z(coherence) + z(exclusivity))"``.
    scores : list[dict]
        The per-K score table the pick was read off — one row per scanned K with
        the value(s) compared. Hand it to ``pandas.DataFrame`` to see the curve.
    notes : list[str]
        Any boundary / monotonicity warnings ``best_k`` raised (e.g. the pick sat
        at the grid edge), captured here instead of only reaching the warning
        stream.
    """

    def __new__(cls, k, *, metric, rule, rule_desc, scores, notes):
        obj = super().__new__(cls, int(k))
        obj.metric = metric
        obj.rule = rule
        obj.rule_desc = rule_desc
        obj.scores = scores
        obj.notes = notes
        return obj

    def __repr__(self) -> str:
        head = f"best_k = {int(self)}  (metric={self.metric!r}, rule={self.rule!r})"
        why = f"  chosen by: {self.rule_desc}"
        rows = "\n".join(
            "    " + ", ".join(f"{k}={v}" for k, v in row.items())
            for row in self.scores
        )
        note_txt = ("\n  notes:\n" + "\n".join("    - " + n for n in self.notes)) \
            if self.notes else ""
        return f"{head}\n{why}\n  scores:\n{rows}{note_txt}"

    def to_frame(self):
        """The per-K score table as a pandas DataFrame (raises if pandas absent)."""
        import pandas as pd

        return pd.DataFrame(self.scores)



class SearchKResult(list):
    """The :func:`search_k` result: a list of per-K dict rows, with the
    optimization direction stamped in and a safe ``best_k`` selector.

    It is a ``list`` subclass, so it iterates and indexes exactly like the rows
    it always returned. The additions remove two traps. The first is sorting the
    wrong way: ``coherence`` is mean UMass (negative; less-negative is better),
    so naively taking the minimum picks the worst K. The second is subtler:
    UMass coherence is roughly *monotone-decreasing* in K, so selecting K by
    coherence alone returns the smallest K in the grid regardless of the data.
    ``best_k`` defaults to a coherence/exclusivity *frontier* (a knee, not a
    maximum) to avoid that, and to the held-out metric when one is supplied.
    """

    def __repr__(self) -> str:
        """A compact adjudicating summary: the chosen K and a per-K
        coherence/exclusivity table, so printing the result answers "which K?"
        instead of dumping the raw per-K dicts. ``.to_frame()`` gives every
        column; ``.best_k(explain=True)`` gives the selection rule."""
        if not self:
            return "SearchKResult([])"
        import warnings

        ks = [r.get("k") for r in self]
        try:
            with warnings.catch_warnings():  # a boundary pick warns; don't emit on display
                warnings.simplefilter("ignore")
                chosen = self.best_k()
        except Exception:
            chosen = None
        metric = self[0].get("coherence_metric", "coherence")
        head = f"SearchKResult: scanned K={ks}"
        if chosen is not None:
            head += f"; best_k()={chosen}"
        lines = [head, f"  {'K':>4}  {('coherence(' + metric + ')'):>20}  {'exclusivity':>11}"]
        for r in self:
            mark = " *" if r.get("k") == chosen else "  "
            coh = r.get("coherence")
            exc = r.get("exclusivity")
            coh_s = f"{coh:.2f}" if isinstance(coh, (int, float)) else "-"
            exc_s = f"{exc:.2f}" if isinstance(exc, (int, float)) else "-"
            lines.append(f"{mark}{r.get('k'):>4}  {coh_s:>20}  {exc_s:>11}")
        lines.append("  (* = best_k(); .to_frame() for all columns, .best_k(explain=True) for the rule)")
        return "\n".join(lines)

    @property
    def directions(self) -> dict:
        """``{metric: "maximize"|"minimize"}`` for the metrics actually present."""
        present = set().union(*[r.keys() for r in self]) if self else set()
        return {m: d for m, d in SEARCH_K_DIRECTIONS.items() if m in present}

    def to_frame(self):
        """The per-K rows as a pandas DataFrame, one row per K (raises if pandas
        is absent). The same tidy shape the effect/robustness results expose."""
        import pandas as pd

        return pd.DataFrame(list(self))

    def _frontier_k(self) -> int:
        """K that maximizes ``z(coherence) + z(exclusivity)`` across the grid.

        Each metric is z-scored across the scanned K values (so the two scales
        are comparable) and added in its own optimization direction. The pick
        is the K that is jointly high on both — the knee, not either extreme.
        A metric with zero variance across the grid contributes nothing.
        """
        for m in ("coherence", "exclusivity"):
            if m not in self[0]:
                raise ValueError(
                    f"frontier selection needs {m!r} in the results "
                    f"(present: {sorted(self[0])})"
                )
        if len(self) < 2:
            raise ValueError(
                "frontier selection needs at least two K values to z-score; "
                "scan a wider grid or pass a single metric"
            )
        # With multiple seeds, score the *same* per-seed frontier curve the 1-SE
        # rule bands around (mean of each seed's z(coherence)+z(exclusivity)), so
        # 'best' and '1se' are consistent. Single seed: frontier of the mean row.
        mean = getattr(self, "_frontier_mean", None)
        if mean is not None:
            return _argbest_k(self, mean)
        score = _frontier_score([r["coherence"] for r in self],
                                 [r["exclusivity"] for r in self])
        return _argbest_k(self, score)

    def _warn_grid_boundary(self, pick: int) -> None:
        """Warn when the frontier optimum sits at the low or high end of the
        scanned grid. There the knee is unidentified — the real best K may lie
        outside ``ks=`` — and the answer can flip with the grid resolution
        (``[3,5,7,10]`` vs ``[3,5,8,12,15,20]``). Symmetric to the coherence and
        held-out boundary guards. Silent on a <=2-point grid, where every pick is
        trivially a boundary."""
        ks = [r["k"] for r in self]
        if len(ks) < 3:
            return
        k_min, k_max = min(ks), max(ks)
        if pick in (k_min, k_max):
            end = "smallest" if pick == k_min else "largest"
            warnings.warn(
                f"best_k(metric='frontier') selected K={pick}, the {end} K in the "
                f"grid (ks spans {k_min}..{k_max}): the coherence/exclusivity knee "
                "is at the grid boundary, so the real best K may lie outside the "
                "scanned range and the pick can change if you widen or refine ks=. "
                "Widen ks= and refit before trusting a boundary K.",
                UserWarning,
                stacklevel=3,
            )

    def _resolve_metric(self, metric):
        """The metric ``best_k`` will actually optimize, applying the
        ``metric=None`` default (held-out → frontier → coherence) and the
        ``coherence_metric`` label alias. Shared by :meth:`best_k` and
        :meth:`_explain_best_k` so the two never drift."""
        if metric is None:
            if "heldout_loglik" in self[0]:
                metric = "heldout_loglik"
            elif "perplexity" in self[0]:
                metric = "perplexity"
            elif len(self) >= 2 and "coherence" in self[0] and "exclusivity" in self[0]:
                metric = "frontier"
            else:
                metric = "coherence"
        # The results advertise which coherence flavor the "coherence" column holds
        # via the coherence_metric label ("semcoh"/"u_mass"). Accept that label as
        # an alias so best_k(res[0]["coherence_metric"]) works instead of raising
        # "unknown metric" for a name the result itself displays (#733).
        if metric is not None and metric == self[0].get("coherence_metric"):
            metric = "coherence"
        return metric

    def best_k(self, metric: str | None = None, *, rule: str = "best",
               frontier_metrics=None, weights=None, explain: bool = False):
        """Return the ``k`` chosen by ``metric``.

        With ``explain=True`` the return is not the bare ``int`` but a
        :class:`BestKExplanation` — the chosen ``k`` plus the per-K score table
        that produced it (the frontier composite ``z(coherence)+z(exclusivity)``,
        or the scalar metric column) and any boundary/monotonicity notes — so the
        pick is auditable rather than opaque. It still prints as, and compares
        equal to, the integer ``k``.

        With ``metric=None`` (the default), selection is:

        - the held-out metric when a held-out set was supplied
          (``"heldout_loglik"`` for a :class:`Heldout`, ``"perplexity"`` for a
          legacy corpus) — the held-out criterion, which unlike bare coherence
          reflects generalization; note that on many real corpora it improves
          only slowly and near-monotonically in K, so ``rule="best"`` can land on
          the largest K scanned (a boundary warning fires and cites the
          ``elbow``/``frontier`` picks when it does);
        - otherwise the ``"frontier"`` (see below), since bare ``"coherence"``
          is roughly monotone in K and would just return the grid floor.

        ``metric`` may also be given explicitly:

        - ``"frontier"`` — the K maximizing ``z(coherence) + z(exclusivity)``,
          the knee the ``plot_search_k`` curve shows (needs at least two K).
        - any column metric (``"coherence"``, ``"exclusivity"``,
          ``"heldout_loglik"``, ``"perplexity"``), optimized in its correct
          direction. Asking for bare ``"coherence"`` on a multi-K grid warns,
          because UMass coherence is roughly monotone in K.
        - ``"reconstruction_error"`` (an NMF/LSA column) — a scree curve,
          monotone-decreasing in K, so ``rule="best"`` returns the grid edge and
          warns; pair it with ``rule="elbow"`` for the diminishing-returns knee.

        ``rule`` chooses how the optimum is turned into a pick:

        - ``"best"`` (default) — the K that optimizes the metric, ties broken
          toward the smaller K.
        - ``"1se"`` — the one-standard-error rule: the smallest (simplest) K whose
          metric is within one standard error of the optimum. Needs the per-K
          standard errors from ``search_k(num_seeds>1)``; raises otherwise.
        - ``"elbow"`` — the diminishing-returns knee of a *scalar* metric's
          K-curve (Kneedle: the K of maximum distance from the endpoints chord).
          For monotone-improving metrics like ``"heldout_loglik"`` whose optimum
          is the grid edge, the elbow is the more useful pick. Needs at least
          three K values; not defined for the ``"frontier"``.

        ``frontier_metrics`` / ``weights`` (only with ``metric="frontier"``)
        customize the ``"frontier"`` composite: by default it is an equal-weight
        ``z(coherence) + z(exclusivity)``. Pass a different metric list (e.g.
        ``["coherence", "deveaud"]``) and/or non-negative per-metric ``weights`` to
        reshape the knee. A custom frontier supports ``rule="best"`` only, and it
        z-scores the across-seed *mean* rows; under ``num_seeds>1`` that can differ
        marginally from the implicit default frontier, which z-scores each seed
        first (to keep ``rule="1se"`` consistent).
        """
        if rule not in ("best", "1se", "elbow"):
            raise ValueError(f"rule must be 'best', '1se', or 'elbow', got {rule!r}")
        if not self:
            raise ValueError("search_k returned no rows")
        if explain:
            return self._explain_best_k(metric, rule, frontier_metrics, weights)
        metric = self._resolve_metric(metric)
        if metric != "frontier" and (frontier_metrics is not None or weights is not None):
            raise ValueError(
                "frontier_metrics and weights only apply to metric='frontier'; "
                f"got metric={metric!r}")
        if metric == "frontier":
            if rule == "elbow":
                raise ValueError("rule='elbow' is not defined for the frontier; "
                                 "use it on a scalar metric like 'heldout_loglik'")
            if frontier_metrics is not None or weights is not None:
                pick = self._custom_frontier_k(frontier_metrics, weights, rule)
            else:
                pick = self._frontier_k_1se() if rule == "1se" else self._frontier_k()
            self._warn_grid_boundary(pick)
            return pick
        if metric not in SEARCH_K_DIRECTIONS:
            selectable = sorted(m for m in SEARCH_K_DIRECTIONS if m in self[0])
            raise ValueError(
                f"unknown metric {metric!r}; choose 'frontier' or one of the "
                f"selectable metrics in these results: {selectable} "
                f"(all recognized metrics: {sorted(SEARCH_K_DIRECTIONS)})"
            )
        if metric not in self[0]:
            raise ValueError(
                f"metric {metric!r} not in results (present: {sorted(self[0])}); "
                f"pass held_out= to get a held-out metric"
            )
        if metric == "coherence" and len(self) >= 2:
            coh_metric = self[0].get("coherence_metric", "u_mass")
            if coh_metric == "u_mass":
                warnings.warn(
                    "best_k(metric='coherence'): mean UMass coherence is roughly "
                    "monotone-decreasing in K, so this tends to return the smallest "
                    "K in the grid. Prefer metric='frontier' (coherence/exclusivity "
                    "knee) or pass held_out= for held-out log-likelihood.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"best_k(metric='coherence'): selecting on {coh_metric!r} "
                    "coherence alone ignores exclusivity and parsimony. Prefer "
                    "metric='frontier' (coherence/exclusivity knee) or pass "
                    "held_out= for held-out log-likelihood.",
                    UserWarning,
                    stacklevel=2,
                )
        present = [r for r in self if np.isfinite(r[metric])]
        if not present:
            raise ValueError(f"metric {metric!r} has no finite value")
        maximize = SEARCH_K_DIRECTIONS[metric] == "maximize"
        if rule == "1se":
            return self._one_se_k(present, metric, maximize)
        if rule == "elbow":
            return self._elbow_k(present, metric, maximize)
        best = (max if maximize else min)(r[metric] for r in present)
        # Parsimony tie-break: smallest k achieving the best value.
        pick = int(min(r["k"] for r in present if r[metric] == best))
        # Boundary guard: held-out log-likelihood and perplexity usually improve
        # with K on real corpora, so rule='best' tends to return the largest K
        # scanned — a grid artifact, not a real optimum. Warn (symmetric to the
        # coherence path) so a user doesn't publish the grid endpoint unexamined.
        if metric in ("heldout_loglik", "perplexity", "reconstruction_error") \
                and len(present) >= 2:
            k_max = max(r["k"] for r in present)
            if pick == k_max:
                # Cite the alternative picks we can already compute, so the user
                # sees the actual less-fragmented K rather than just a rule name
                # (issue #732). Guard each: elbow needs a bend, frontier needs the
                # coherence/exclusivity columns — skip whichever isn't available.
                hints = []
                with warnings.catch_warnings():
                    # The hint computation is exploratory; silence any warning the
                    # elbow/frontier helpers raise (e.g. "elbow not well defined")
                    # so only this boundary warning reaches the user.
                    warnings.simplefilter("ignore")
                    try:
                        e = self._elbow_k(present, metric, maximize)
                        if e != pick:
                            hints.append(f"rule='elbow' gives K={e}")
                    except Exception:
                        pass
                    try:
                        f = self._frontier_k()
                        if f != pick:
                            hints.append(f"metric='frontier' gives K={f}")
                    except Exception:
                        pass
                hint_txt = (" " + "; ".join(hints) + ".") if hints else ""
                warnings.warn(
                    f"best_k(metric={metric!r}) selected K={pick}, the largest K "
                    "scanned: this metric tends to keep improving with K, so the "
                    "optimum is at the grid boundary and the real best K may lie "
                    f"beyond it.{hint_txt} Widen ks=, or use rule='elbow' "
                    "(diminishing-returns knee) or metric='frontier' "
                    "(coherence/exclusivity knee).",
                    UserWarning,
                    stacklevel=2,
                )
        return pick

    def _explain_best_k(self, metric, rule, frontier_metrics, weights):
        """Build the :class:`BestKExplanation` for ``best_k(..., explain=True)``.

        Runs the ordinary selection (capturing any warning it raises as a note),
        then reconstructs the per-K score the pick was read off: the frontier
        composite ``z(coherence)+z(exclusivity)`` for the frontier, otherwise the
        scalar metric column in its optimization direction."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pick = self.best_k(metric, rule=rule, frontier_metrics=frontier_metrics,
                               weights=weights)
        notes = [str(w.message) for w in caught]
        resolved = self._resolve_metric(metric)
        ks = [r["k"] for r in self]
        if resolved == "frontier" and frontier_metrics is None and weights is None:
            comp = _frontier_score([r["coherence"] for r in self],
                                    [r["exclusivity"] for r in self])
            scores = [{"k": k, "score": float(s)} for k, s in zip(ks, comp)]
            basis = "z(coherence) + z(exclusivity)"
            rule_desc = f"frontier knee (max of {basis})"
        elif resolved == "frontier":
            mets = list(frontier_metrics) if frontier_metrics is not None else \
                ["coherence", "exclusivity"]
            scores = [{"k": r["k"], **{m: r.get(m) for m in mets}} for r in self]
            basis = " + ".join(f"z({m})" for m in mets)
            rule_desc = f"custom frontier knee (max of {basis})"
        else:
            direction = SEARCH_K_DIRECTIONS.get(resolved, "maximize")
            scores = [{"k": r["k"], resolved: r.get(resolved)} for r in self]
            basis = f"{resolved} ({direction})"
            rule_desc = f"rule={rule!r} on {basis}"
        return BestKExplanation(pick, metric=resolved, rule=rule,
                                rule_desc=rule_desc, scores=scores, notes=notes)

    def _one_se_k(self, present, metric, maximize):
        """One-standard-error rule on a scalar ``metric``: the smallest K whose
        mean is within one SE of the optimum (in the metric's direction)."""
        se_key = metric + "_se"
        if se_key not in present[0]:
            raise ValueError(
                f"rule='1se' needs per-K standard errors for {metric!r}; "
                "refit with search_k(num_seeds>1)"
            )
        best_row = (max if maximize else min)(present, key=lambda r: r[metric])
        se = best_row[se_key]
        if maximize:
            thresh = best_row[metric] - se
            within = [r for r in present if r[metric] >= thresh]
        else:
            thresh = best_row[metric] + se
            within = [r for r in present if r[metric] <= thresh]
        return int(min(r["k"] for r in within))  # simplest K within 1 SE

    def _elbow_k(self, present, metric, maximize):
        """Diminishing-returns knee of ``metric`` vs K (Kneedle): normalize the
        curve, orient it so higher = better, and take the K of maximum vertical
        distance above the chord joining the first and last scanned K."""
        rows = sorted(present, key=lambda r: r["k"])
        if len(rows) < 3:
            raise ValueError("rule='elbow' needs at least three K values")
        x = np.array([r["k"] for r in rows], dtype=np.float64)
        y = np.array([r[metric] for r in rows], dtype=np.float64)
        if not maximize:
            y = -y  # orient so a better metric is a higher y
        xr, yr = np.ptp(x), np.ptp(y)
        if xr == 0 or yr == 0:
            return int(x[0])  # flat curve -> simplest K
        xn = (x - x.min()) / xr
        yn = (y - y.min()) / yr
        # A diminishing-returns curve bulges above the endpoints chord; its peak
        # is the elbow.
        chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / (xn[-1] - xn[0])
        gap = yn - chord
        if np.max(gap) <= 1e-12:  # convex or straight: no diminishing-returns knee
            warnings.warn(
                f"rule='elbow': the {metric!r} curve does not bend toward "
                "diminishing returns (it is convex or straight), so the elbow is "
                "not well defined; returning the smallest K.",
                UserWarning, stacklevel=3)
            return int(x.min())
        return int(x[int(np.argmax(gap))])

    def _custom_frontier_k(self, frontier_metrics, weights, rule):
        """Frontier over a caller-chosen metric set / weights. Supports
        ``rule='best'`` only (the per-seed 1-SE band is the default frontier's)."""
        if rule != "best":
            raise ValueError(
                f"a custom frontier supports rule='best' only, got {rule!r}")
        metrics = list(frontier_metrics) if frontier_metrics is not None \
            else ["coherence", "exclusivity"]
        if len(metrics) < 1:
            raise ValueError("frontier_metrics must name at least one metric")
        for m in metrics:
            if m not in SEARCH_K_DIRECTIONS:
                raise ValueError(f"unknown frontier metric {m!r}")
            if m not in self[0]:
                raise ValueError(f"frontier metric {m!r} not in results "
                                 f"(present: {sorted(self[0])})")
        if weights is None:
            weights = [1.0] * len(metrics)
        if len(weights) != len(metrics):
            raise ValueError("weights must match frontier_metrics in length")
        if any((not np.isfinite(w)) or w < 0 for w in weights):
            raise ValueError("weights must be finite and non-negative")
        if len(self) < 2:
            raise ValueError("frontier selection needs at least two K values")
        score = np.zeros(len(self))
        finite = np.ones(len(self), dtype=bool)
        for w, m in zip(weights, metrics):
            v = np.array([r[m] for r in self], dtype=np.float64)
            if SEARCH_K_DIRECTIONS[m] == "minimize":
                v = -v
            finite &= np.isfinite(v)
            sd = np.nanstd(v)
            if sd > 0:
                score += float(w) * np.nan_to_num((v - np.nanmean(v)) / sd, nan=0.0)
        if finite.any():
            score[~finite] = -np.inf
        return _argbest_k(self, score)

    def _frontier_k_1se(self) -> int:
        """One-standard-error rule on the frontier composite: needs the per-seed
        frontier scores stored by ``search_k(num_seeds>1)``."""
        mean = getattr(self, "_frontier_mean", None)
        sem = getattr(self, "_frontier_se", None)
        if mean is None or sem is None:
            raise ValueError(
                "rule='1se' on the frontier needs per-seed scores; "
                "refit with search_k(num_seeds>1)"
            )
        finite = np.isfinite(mean)
        if not finite.any():  # every K degenerate -> smallest K, as _frontier_k does
            return int(min(r["k"] for r in self))
        best_i = int(np.nanargmax(np.where(finite, mean, np.nan)))
        thresh = mean[best_i] - sem[best_i]
        within = [i for i in range(len(self)) if finite[i] and mean[i] >= thresh]
        return int(min(self[i]["k"] for i in within))



def _aggregate_over_seeds(ks, seeds, tasks, fitted):
    """Collapse per-``(K, seed)`` fits into one row per K carrying each metric's
    mean and a ``<metric>_se`` (standard error), and stash the per-seed frontier
    scores on the result so ``best_k(rule='1se')`` works on the composite too."""
    by_k = {k: [] for k in ks}
    for (k, _fs), row in zip(tasks, fitted):
        by_k[k].append(row)
    n = len(seeds)
    keep_scalar = {"k", "coherence_metric"}
    no_se = {"dispersion_pvalue"}  # an SE on a p-value is not meaningful
    rows, coh_by_k, exc_by_k = [], [], []
    max_spread = 0.0  # largest across-seed range of any metric, over all K
    for k in ks:
        seed_rows = by_k[k]
        agg = {"k": k}
        if "coherence_metric" in seed_rows[0]:
            agg["coherence_metric"] = seed_rows[0]["coherence_metric"]
        for key in seed_rows[0]:
            if key in keep_scalar:
                continue
            vals = np.array([r[key] for r in seed_rows], dtype=np.float64)
            agg[key] = float(np.mean(vals))
            if key not in no_se:
                agg[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(n))
            if np.all(np.isfinite(vals)):
                max_spread = max(max_spread, float(np.ptp(vals)))
        rows.append(agg)
        coh_by_k.append([r["coherence"] for r in seed_rows])
        exc_by_k.append([r["exclusivity"] for r in seed_rows])

    # If every seed produced identical metrics, the model's initialization ignores
    # the seed (STM's spectral init, NMF/LSA init='nndsvd'), so the *_se columns
    # are all 0 and best_k(rule='1se') is meaningless — a user could read SE=0 as
    # "robust across seeds" when the seeds never varied. Warn, mirroring
    # topic_stability's identical-runs guard.
    if n > 1 and max_spread <= 1e-12:
        warnings.warn(
            f"search_k(num_seeds={n}): every seed produced an identical fit at each "
            "K, so all *_se columns are 0 and best_k(rule='1se') says nothing about "
            "robustness. Deterministic initializations ignore the seed (STM's "
            "spectral init; NMF/LSA with init='nndsvd'). To assess robustness, refit "
            "with init='random' (which responds to seed) or use bootstrap resampling "
            "(bootstrap_stability); num_seeds>1 adds no information for this model.",
            UserWarning,
            stacklevel=3,
        )

    result = SearchKResult(rows)
    # Per-seed frontier: z-score within each seed across K (via the shared helper),
    # then mean / SE over seeds. Lets the 1-SE rule apply to the frontier composite.
    coh = np.asarray(coh_by_k, np.float64)  # (nK, n_seeds)
    exc = np.asarray(exc_by_k, np.float64)
    fs = np.vstack([_frontier_score(coh[:, s], exc[:, s]) for s in range(n)])  # (n, nK)
    with np.errstate(invalid="ignore"):
        result._frontier_mean = fs.mean(axis=0)
        result._frontier_se = fs.std(axis=0, ddof=1) / np.sqrt(n)
    return result



def search_k(
    docs,
    ks,
    *,
    model="lda",
    fit=None,
    prevalence=None,
    prevalence_names=None,
    content=None,
    held_out=None,
    iters=500,
    num_samples=3,
    sample_interval=10,
    seed=13,
    coherence_n=10,
    coherence_type="u_mass",
    n_jobs=1,
    num_seeds=_SEARCH_K_DEFAULT_SEEDS,
    criteria=(),
):
    """Fit a model for each K and report quality metrics (stm's ``searchK``).

    ``model=`` selects a built-in: ``"lda"`` (default), ``"stm"``, ``"nmf"``, or
    ``"lsa"``. For ``"stm"`` pass ``prevalence`` (a covariate design matrix) and
    optional ``content`` (group labels) to scan K for the model you'll actually
    report. ``"nmf"`` additionally reports a ``reconstruction_error`` column.

    For **any other model**, pass ``fit=`` — a callable ``(k, seed) -> fitted
    model`` that builds and fits the model, closing over ``docs`` and any
    covariates or embeddings it needs. ``search_k`` then scores whatever it
    returns with the same generic metrics, so it works for every model without
    knowing its fit signature (the same escape hatch ``bootstrap_stability`` /
    ``diagnostics`` / ``standard_errors`` offer via ``model_factory=``)::

        search_k(docs, [10, 20, 30], fit=lambda k, s: topica.NMF(k, seed=s).fit(docs))

    A fitted model only needs ``topic_word`` and ``top_words`` for the coherence
    and exclusivity columns. The ``dispersion`` column and the ``held_out`` columns
    are generative-count diagnostics, so they are reported only for models that
    expose a generative ``transform`` (LDA/STM/DMR/CTM/HDP); they are omitted for
    matrix-factorization models (NMF factors a tf-idf matrix; LSA has signed SVD
    factors), which are not generative count models. The opt-in ``criteria``
    (``deveaud``/``cao_juan``) treat each topic as a word distribution, so they are
    omitted for a signed ``topic_word`` (LSA). The stm semantic-coherence metric is
    used only for the built-in ``model="stm"``; a ``fit=`` closure returning an STM
    is scored with plain UMass coherence.

    Returns a :class:`SearchKResult` (a list of per-K dicts) with ``k``,
    ``coherence`` (mean of the selected coherence type; for ``model="stm"`` with
    the default ``u_mass`` this is stm's semantic coherence, labelled
    ``coherence_metric="semcoh"``), ``exclusivity`` (mean top-word exclusivity),
    ``dispersion`` (residual dispersion, Taddy 2012 — ``>> 1`` means K is too
    small) with its ``dispersion_pvalue``, and — when ``held_out`` is supplied —
    a held-out quality metric. The result also carries ``.directions`` (whether
    higher or lower is better per metric) and a ``.best_k(metric=...)`` selector.
    ``best_k`` defaults to the held-out metric when one is supplied, otherwise to
    a coherence/exclusivity frontier (a knee), because bare UMass coherence is
    roughly monotone in K and would just return the smallest K scanned. Duplicate
    ``ks`` are dropped; ties in ``best_k`` break toward the smaller (simpler) K.

    Two held-out paths are supported, determined by the type of ``held_out``:

    - **Heldout object** (from :func:`make_heldout`): scored with
      :func:`eval_heldout`; results stored under ``"heldout_loglik"``
      (``mean_per_doc_loglik``, higher / less negative is better). Use this
      path for the standard within-corpus word-heldout diagnostic.
    - **Corpus or token lists** (legacy): scored with :func:`perplexity`;
      results stored under ``"perplexity"`` (lower is better). This is the
      document-completion perplexity on a separate held-out set.

    Parameters
    ----------
    docs : training documents (``list[list[str]]`` or a ``Corpus``).
    ks : sequence of topic counts to scan.
    model : ``"lda"`` (default), ``"stm"``, ``"nmf"``, or ``"lsa"``. Ignored when
        ``fit=`` is given.
    fit : optional ``callable(k, seed) -> fitted model``. When given it takes
        precedence over ``model`` and lets ``search_k`` scan any model type; the
        callable owns the fit (it closes over ``docs`` and any covariates).
    prevalence : covariate design matrix for ``model="stm"``; ignored otherwise.
    prevalence_names : accepted for signature-parity with ``STM.fit`` (so the same
        kwargs drop straight in) and ignored — ``search_k`` scans K on the design
        matrix and never labels the covariates, so their names do not affect any
        metric it reports.
    content : optional content group labels (sequence of str/int) for ``model="stm"``.
    held_out : optional held-out set. Pass a :class:`Heldout` (from
        :func:`make_heldout`) or a separate corpus / token lists.
    iters : training iterations per fit.
    num_samples : Gibbs samples per fit (LDA only).
    sample_interval : iterations between Gibbs samples (LDA only).
    seed : RNG seed for every fit and transform call.
    coherence_n : top-word count used for coherence and exclusivity.
    coherence_type : one of ``"u_mass"``, ``"c_uci"``, ``"c_npmi"``, ``"c_v"`` (default ``"u_mass"``).
    n_jobs : number of worker threads for the per-fit work (default ``1``, serial).
        The fits are independent and each keeps its own fixed seed, so the results
        are identical to the serial run; only the wall-clock changes (the Rust fits
        release the GIL). ``n_jobs<=0`` (or ``None``) uses all cores, capped at the
        number of fits. Note it multiplies with any intra-fit threading
        (``num_threads=`` on the model), so ``n_jobs`` above the core count can
        oversubscribe.
    num_seeds : number of seeds fit per K (default ``1``). With ``num_seeds>1``,
        each K is refit over seeds ``seed, seed+1, ...``; every metric column then
        holds the across-seed mean and gains a ``<metric>_se`` standard-error
        column, and ``best_k(rule="1se")`` becomes available (the simplest K within
        one SE of the optimum). The per-K work parallelizes over ``(K, seed)`` via
        ``n_jobs``. A single seed carries no standard errors (backward-compatible).
    criteria : optional extra K-selection criteria to report as columns (default
        none). ``"deveaud"`` (Deveaud et al. 2014; mean pairwise Jensen-Shannon
        divergence between topics, higher = more distinct) and ``"cao_juan"``
        (Cao Juan et al. 2009; mean pairwise topic cosine, lower = less redundant).
        Opt-in and out of the frontier, but selectable via ``best_k("deveaud")`` /
        ``best_k("cao_juan")`` and carry standard errors under ``num_seeds>1`` like
        any other metric.
    """
    from . import LDA, LSA, NMF, STM  # local import to avoid a cycle at module load

    use_fit = fit is not None
    if use_fit and not callable(fit):
        raise ValueError(f"fit= must be a callable (k, seed) -> fitted model, got {fit!r}")
    if not use_fit and model not in ("lda", "stm", "nmf", "lsa"):
        raise ValueError(
            f"search_k built-ins are 'lda', 'stm', 'nmf', 'lsa' (got {model!r}). "
            f"For any other model pass fit=(k, seed) -> fitted model, which closes "
            f"over docs and any covariates/embeddings the model needs — e.g. "
            f"fit=lambda k, s: topica.DMR(k, seed=s).fit(docs, prevalence). "
            f"Embedding+cluster models (BERTopic, Top2Vec) set K by the clusterer, "
            f"not by refitting, so they do not fit this paradigm; sweep their "
            f"clusterer settings instead (see docs/publishing/choosing-k.md).")
    if int(num_seeds) < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds!r}")
    if use_fit and (prevalence is not None or content is not None):
        raise ValueError(
            "prevalence=/content= are for the built-in model='stm'; with fit= the "
            "callable owns the fit, so pass any covariates inside it.")
    if content is not None and model != "stm":
        raise ValueError("content covariates are only supported when model='stm'")

    criteria = tuple(criteria)
    bad = [c for c in criteria if c not in _SEARCH_K_CRITERIA]
    if bad:
        raise ValueError(
            f"unknown criteria {bad}; choose from {list(_SEARCH_K_CRITERIA)}")

    valid_coh = ("u_mass", "c_uci", "c_npmi", "c_v")
    coherence_type = coherence_type.lower()
    # A "stratified_<type>" request (content models only) scores each group's own
    # top words against its own subcorpus (topica.content.stratified_coherence).
    stratified = coherence_type.startswith("stratified_")
    base_ct = coherence_type[len("stratified_"):] if stratified else coherence_type
    if base_ct not in valid_coh:
        raise ValueError(
            f"coherence_type must be one of {valid_coh} "
            f"(optionally 'stratified_'-prefixed for content models), got {coherence_type!r}")
    if stratified and (model != "stm" or content is None):
        raise ValueError(
            "stratified coherence needs model='stm' with a content covariate")

    if model == "stm" and content is not None and not stratified:
        warnings.warn(
            "Exclusivity and coherence calculations on a model with content covariates "
            "are computed on the baseline/group-average topic-word distributions. "
            "These metrics do not capture group-specific wording variations. Pass "
            "coherence_type='stratified_c_v' (etc.) for group-stratified metrics.",
            UserWarning,
            stacklevel=2,
        )

    ks_in = [int(k) for k in ks]
    ks = list(dict.fromkeys(ks_in))  # de-duplicate, preserving order
    if len(ks) != len(ks_in):
        warnings.warn(
            "search_k: duplicate values in `ks` were dropped so each K is fit "
            "once (duplicates would also overweight that K in the frontier).",
            UserWarning,
            stacklevel=2,
        )

    ref_docs = _ref_corpus(docs)  # token lists, reused across every K

    def _make_fitted(k, fit_seed):
        """Build and fit the model for this (K, seed). The fit= hook owns the fit;
        otherwise use the built-in per-model fit signature."""
        if use_fit:
            m = fit(k, fit_seed)
            if not hasattr(m, "topic_word"):
                raise TypeError(
                    "fit=(k, seed) must return a fitted model exposing topic_word "
                    f"(and top_words); got {type(m).__name__}")
            return m
        if model == "stm":
            m = STM(num_topics=k, seed=fit_seed)
            m.fit(docs, prevalence, content=content, iters=iters)
        elif model == "nmf":
            m = NMF(num_topics=k, seed=fit_seed).fit(docs, iters=iters)
        elif model == "lsa":
            m = LSA(num_topics=k, seed=fit_seed).fit(docs)  # SVD: no iters/seed effect
        else:
            m = LDA(num_topics=k, seed=fit_seed)
            m.fit(docs, iters=iters, num_samples=num_samples,
                  sample_interval=sample_interval)
        return m

    def _fit_row(k, fit_seed):
        m = _make_fitted(k, fit_seed)

        coh_label = coherence_type
        if stratified:
            from .content import (stratified_coherence as _strat,
                                  topic_polarization as _pol,
                                  group_exclusivity as _gex)
            coh_val = float(np.mean(_strat(m, docs, content, coherence_type=base_ct,
                                          n=coherence_n)))
        elif coherence_type == "u_mass" and model == "stm":
            # stm's searchK reports semantic coherence (semCoh1beta, 0.01
            # smoothing), not gensim UMass -- use the stm-faithful metric for STM.
            from .coherence import semantic_coherence
            coh_val = float(np.mean(semantic_coherence(m, ref_docs, n=coherence_n)))
            coh_label = "semcoh"
        elif coherence_type == "u_mass" and hasattr(m, "coherence"):
            coh_val = float(np.mean(m.coherence(coherence_n)))
        else:
            from .coherence import coherence as external_coherence
            # m.top_words(coherence_n) returns a list of lists of word strings.
            topics = [list(top_list) for top_list in m.top_words(coherence_n)]
            scores = external_coherence(topics, ref_docs, coherence_type=base_ct, topn=coherence_n)
            coh_val = float(np.mean(scores))

        row = {
            "k": k,
            "coherence": coh_val,
            "coherence_metric": coh_label,
            "exclusivity": (float(np.mean(_gex(m, n=coherence_n))) if stratified
                            else _mean_exclusivity(m.topic_word, coherence_n)),
        }
        # Residual dispersion (Taddy 2012): dispersion >> 1 is direct evidence K
        # is too small -- the non-monotone signal stm's searchK reports. Diagnostic
        # column, not a frontier metric (it keeps falling as K grows). It is a
        # generative *multinomial-count* residual test, so it only applies to
        # models that define p(counts) -- i.e. expose a generative `transform`
        # (LDA/STM/DMR/CTM/HDP). Matrix-factorization models are not generative
        # count models (NMF factors a tf-idf matrix; LSA's SVD factors are signed),
        # so their dispersion is meaningless and non-monotone; omit the column for
        # them, the same capability gate `held_out` uses.
        if hasattr(m, "transform"):
            rc = check_residuals(m, ref_docs)
            row["dispersion"] = float(rc.dispersion)
            row["dispersion_pvalue"] = float(rc.pvalue)
        # NMF (and any factorization model that exposes it) reports its residual
        # fit as an extra diagnostic column, like dispersion. Monotone in K, so it
        # stays out of the frontier / best_k, same as dispersion.
        if hasattr(m, "reconstruction_error"):
            row["reconstruction_error"] = float(m.reconstruction_error)
        # Opt-in ldatuning-style criteria from the topic-word matrix. These treat
        # each topic as a word *distribution* (deveaud = pairwise Jensen-Shannon
        # divergence, cao_juan = pairwise cosine), so they are only defined for a
        # non-negative topic_word. LSA's signed SVD loadings make deveaud NaN (log
        # of a negative) and cao_juan an orthogonality artifact, so omit these
        # columns for signed factorizations rather than emit noise.
        if criteria:
            phi = np.asarray(m.topic_word)
            if float(phi.min()) >= 0.0:
                for c in criteria:
                    row[c] = _extra_criterion(c, phi)
        if stratified:
            row["polarization"] = float(np.mean(_pol(m)))
        if held_out is not None:
            if not hasattr(m, "transform"):
                raise ValueError(
                    f"held_out= scoring needs a generative transform, but "
                    f"{type(m).__name__} has none (matrix-factorization and "
                    "embedding-cluster models do not). Drop held_out= and compare "
                    "these models on coherence / exclusivity instead.")
            if isinstance(held_out, Heldout):
                result = eval_heldout(m, held_out, seed=fit_seed)
                row["heldout_loglik"] = float(result.mean_per_doc_loglik)
            else:
                row["perplexity"] = float(perplexity(m, held_out, seed=fit_seed))
        return row

    # One task per (K, seed). Each fit is independent with its own fixed seed, so
    # results are identical to the serial path (verified). topica's Rust fits
    # release the GIL, so a thread pool parallelizes wall-clock with no pickling.
    seeds = [seed + s for s in range(int(num_seeds))]
    tasks = [(k, fs) for k in ks for fs in seeds]
    workers = _resolve_workers(n_jobs, len(tasks))
    if workers == 1:
        fitted = [_fit_row(k, fs) for (k, fs) in tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fitted = list(pool.map(lambda t: _fit_row(*t), tasks))  # preserves order

    if len(seeds) == 1:
        return SearchKResult(fitted)  # one row per K, no standard errors
    return _aggregate_over_seeds(ks, seeds, tasks, fitted)



# ---------------------------------------------------------------------------
# selectModel: best-of-N runs at fixed K  (stm §3.4)
# ---------------------------------------------------------------------------

@dataclass
class SelectModelResult:
    """Result of :func:`select_model`.

    Attributes
    ----------
    models : list of N fitted models, one per run.
    coherence : array of shape ``(N,)`` — per-run mean UMass coherence.
    exclusivity : array of shape ``(N,)`` — per-run mean top-word exclusivity.
    run_seeds : array of shape ``(N,)`` — seed used for each run.
    """

    models: list
    coherence: np.ndarray
    exclusivity: np.ndarray
    run_seeds: np.ndarray



def select_model(
    docs,
    K,
    *,
    runs=20,
    model="lda",
    prevalence=None,
    word_embeddings=None,
    vocabulary=None,
    doc_embeddings=None,
    iters=500,
    num_samples=3,
    sample_interval=10,
    seed=13,
    coherence_n=10,
    fraction=None,
    burn_in_iters=None,
):
    """Run N initializations at a fixed K and return the fitted candidates (stm's ``selectModel``).

    All ``runs`` models are fit from different random seeds. With
    ``fraction`` set, the procedure uses two stages: a short burn-in
    (``burn_in_iters``, defaulting to 20% of ``iters``) followed by
    full training of the top ``ceil(fraction * runs)`` models by their
    objective (ELBO where the model has one, else log-likelihood, else
    mean coherence). This mirrors stm's "run briefly, keep the best ~20%"
    heuristic.

    This is for models whose fit depends on the random seed — the ones that
    scatter across local optima. ``ETM``, ``ProdLDA``, ``FASTopic``,
    ``CombinedTM``, and ``ZeroShotTM`` all benefit. ``STM``/``CTM`` use a
    deterministic spectral init, so every run is identical and multi-start buys
    nothing — pick one of the stochastic models instead. (``DTM`` is not selected
    here: its topics are time-varying, so coherence/exclusivity are not a single
    number; use ``DTM(init="spectral")`` for a deterministic fit.)

    Parameters
    ----------
    docs : training documents (``list[list[str]]`` or a ``Corpus``).
    K : number of topics for every run.
    runs : number of random initializations.
    model : which model to fit. One of ``"lda"`` (default), ``"stm"``,
        ``"prodlda"``, ``"etm"``, ``"fastopic"``, ``"combinedtm"``,
        ``"zeroshottm"``.
    prevalence : covariate design matrix; required when ``model="stm"``.
    word_embeddings : ``(vocab, dim)`` word-embedding matrix; required when
        ``model="etm"`` (paired with ``vocabulary``).
    vocabulary : the word list aligning ``word_embeddings`` rows; required when
        ``model="etm"``.
    doc_embeddings : ``(num_docs, dim)`` document-embedding matrix; required when
        ``model`` is ``"fastopic"``, ``"combinedtm"``, or ``"zeroshottm"``.
    iters : full-training iterations per run (or per survivor when
        ``fraction`` is used).
    num_samples : Gibbs samples per run (LDA only).
    sample_interval : iterations between Gibbs samples (LDA only).
    seed : base RNG seed; run ``r`` uses seed ``seed + r``.
    coherence_n : top-word count for coherence and exclusivity.
    fraction : if given (a float in ``(0, 1]``), keep only the top
        ``ceil(fraction * runs)`` models (by their objective) after
        ``burn_in_iters`` and run those survivors to full ``iters``.
        ``None`` (default) runs all initializations to full ``iters``.
    burn_in_iters : burn-in length used for early discard; defaults to
        ``max(1, round(0.2 * iters))`` when ``fraction`` is set.

    Returns
    -------
    A :class:`SelectModelResult` with ``models``, ``coherence``,
    ``exclusivity``, and ``run_seeds`` arrays of length equal to the
    number of survivors (all ``runs`` when ``fraction`` is ``None``).
    """
    from . import (  # local import to avoid a cycle
        LDA, STM, CombinedTM, ETM, FASTopic, ProdLDA, ZeroShotTM,
    )

    valid = ("lda", "stm", "prodlda", "etm", "fastopic", "combinedtm",
             "zeroshottm")
    if model not in valid:
        raise ValueError(f"model must be one of {valid}, got {model!r}")
    if not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be a positive integer, got {runs!r}")
    if fraction is not None and not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    # Per-model required data.
    if model == "stm" and prevalence is None:
        raise ValueError("model='stm' requires prevalence=")
    if model == "etm" and (word_embeddings is None or vocabulary is None):
        raise ValueError("model='etm' requires word_embeddings= and vocabulary=")
    if model in ("fastopic", "combinedtm", "zeroshottm") and doc_embeddings is None:
        raise ValueError(f"model={model!r} requires doc_embeddings=")

    def _make(s):
        if model == "stm":
            return STM(num_topics=K, seed=s)
        if model == "prodlda":
            return ProdLDA(num_topics=K, seed=s)
        if model == "etm":
            return ETM(num_topics=K, seed=s)
        if model == "fastopic":
            return FASTopic(num_topics=K, seed=s)
        if model == "combinedtm":
            return CombinedTM(num_topics=K, seed=s)
        if model == "zeroshottm":
            return ZeroShotTM(num_topics=K, seed=s)
        return LDA(num_topics=K, seed=s)

    def _fit(m, n_iters):
        if model == "stm":
            m.fit(docs, prevalence, iters=n_iters)
        elif model == "etm":
            m.fit(docs, word_embeddings, vocabulary, iters=n_iters)
        elif model in ("fastopic", "combinedtm", "zeroshottm"):
            m.fit(docs, doc_embeddings, iters=n_iters)
        elif model == "lda":
            m.fit(docs, iters=n_iters, num_samples=num_samples,
                  sample_interval=sample_interval)
        else:  # prodlda
            m.fit(docs, iters=n_iters)

    def _objective(m):
        """Scalar objective for early discard: higher is better. Falls back to
        mean coherence for models with no scalar bound (e.g. FASTopic)."""
        if hasattr(m, "bound"):
            b = float(m.bound)
            if not np.isnan(b):
                return b
        if hasattr(m, "log_likelihood") and callable(m.log_likelihood):
            return float(m.log_likelihood())
        return float(np.mean(m.coherence(coherence_n)))

    run_seeds = [seed + r for r in range(runs)]

    if fraction is None:
        # Simple path: run every initialization to full iters.
        fitted = []
        for s in run_seeds:
            m = _make(s)
            _fit(m, iters)
            fitted.append(m)
        survivor_seeds = run_seeds
    else:
        # Two-stage: burn-in, then re-run survivors.
        n_burn = burn_in_iters if burn_in_iters is not None else max(1, round(0.2 * iters))
        import math
        n_keep = max(1, math.ceil(fraction * runs))

        # Stage 1: burn-in for all runs.
        burn_models = []
        for s in run_seeds:
            m = _make(s)
            _fit(m, n_burn)
            burn_models.append(m)

        # Rank by objective (higher is better); keep top n_keep.
        scored = sorted(
            zip(run_seeds, burn_models),
            key=lambda pair: _objective(pair[1]),
            reverse=True,
        )
        survivors = scored[:n_keep]

        # Stage 2: run survivors to full iters.
        fitted = []
        survivor_seeds = []
        for s, _ in survivors:
            m = _make(s)
            _fit(m, iters)
            fitted.append(m)
            survivor_seeds.append(s)

    coh = np.array([float(np.mean(m.coherence(coherence_n))) for m in fitted])
    excl = np.array([_mean_exclusivity(m.topic_word, coherence_n) for m in fitted])

    return SelectModelResult(
        models=fitted,
        coherence=coh,
        exclusivity=excl,
        run_seeds=np.array(survivor_seeds, dtype=np.intp),
    )



def plot_models(result, *, ax=None, label_runs=True):
    """Coherence-vs-exclusivity scatter for :func:`select_model` candidates (stm's ``plotModels``).

    Each point is one run. The upper-right corner is the best region:
    both coherent (interpretable) and exclusive (distinctive). Use
    this plot to pick a run from :func:`select_model` before fitting
    your full analysis.

    Parameters
    ----------
    result : a :class:`SelectModelResult` returned by :func:`select_model`.
    ax : matplotlib ``Axes`` to draw on; a new figure is created if
        ``None``.
    label_runs : annotate each point with its run index; default
        ``True``.

    Returns
    -------
    The matplotlib ``Axes``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_models needs matplotlib (pip install matplotlib)."
        ) from e

    coh = np.asarray(result.coherence)
    excl = np.asarray(result.exclusivity)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(coh, excl, color="C0", zorder=3)
    if label_runs:
        for i, (x, y) in enumerate(zip(coh, excl)):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Mean semantic coherence (UMass)")
    ax.set_ylabel("Mean exclusivity")
    ax.set_title("Model candidates: coherence vs. exclusivity")
    ax.figure.tight_layout()
    return ax



def plot_search_k(rows, *, metrics=("coherence", "exclusivity"), ax=None):
    """Plot :func:`search_k` results: each metric against the number of topics.

    Researchers read this curve to choose `K`: coherence and exclusivity usually
    trade off, so the goal is a knee, not a maximum. ``rows.best_k()`` returns
    that knee directly (the ``"frontier"`` selector). Each metric gets its own
    y-axis (they live on different scales). ``rows`` is the list returned by
    :func:`search_k`; ``metrics`` selects which of its keys to draw (any of
    ``"coherence"``, ``"exclusivity"``, ``"perplexity"``, ``"heldout_loglik"``).
    Only metrics present in the rows are drawn; absent keys are silently skipped.

    Returns the primary matplotlib ``Axes`` (consistent with topica's other
    ``plot_*`` helpers). To save the figure, go through the axes' figure::

        ax = topica.select.plot_search_k(rows)
        ax.figure.savefig("search_k.png", dpi=150, bbox_inches="tight")

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "plot_search_k needs matplotlib (pip install matplotlib)."
        ) from e

    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    metrics = [m for m in metrics if any(m in r for r in rows)]
    if not metrics:
        raise ValueError("none of the requested metrics are present in rows")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lines = []
    for i, metric in enumerate(metrics):
        a = ax if i == 0 else ax.twinx()
        if i >= 2:  # offset a third axis so it doesn't overlap the second
            a.spines["right"].set_position(("axes", 1.0 + 0.18 * (i - 1)))
        color = f"C{i}"
        vals = [r.get(metric, float("nan")) for r in rows]
        (line,) = a.plot(ks, vals, marker="o", color=color, label=metric)
        a.set_ylabel(metric, color=color)
        a.tick_params(axis="y", labelcolor=color)
        lines.append(line)

    ax.set_xlabel("number of topics (K)")
    ax.set_xticks(ks)
    ax.legend(lines, [li.get_label() for li in lines], loc="best")
    ax.figure.tight_layout()
    return ax



def plot_topic_discovery(model, *, ax=None):
    """Plot an HDP fit's topic-discovery trajectory: the inferred number of
    topics K against the Gibbs iteration, with the per-token log-likelihood on a
    twin axis. Watching K rise, fall, and settle (while the log-likelihood
    plateaus) is the nonparametric model's headline convergence check — the
    analog of reading a `search_k` curve, but learned in a single fit.

    ``model`` is a fitted :class:`~topica.HDP` (its ``topic_count_history`` and
    ``log_likelihood_history`` are read). Returns the primary matplotlib
    ``Axes``. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "plot_topic_discovery needs matplotlib (pip install matplotlib)."
        ) from e

    tch = list(model.topic_count_history)
    llh = list(model.log_likelihood_history)
    if not tch:
        raise ValueError(
            "no discovery trace recorded; fit with report_interval > 0 "
            "(or the default auto cadence)"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    iters = [it for it, _ in tch]
    ks = [k for _, k in tch]
    (line_k,) = ax.plot(iters, ks, color="C0", marker="o", ms=3, label="topics (K)")
    ax.set_xlabel("Gibbs iteration")
    ax.set_ylabel("number of topics (K)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    lines = [line_k]
    if llh:
        a2 = ax.twinx()
        (line_ll,) = a2.plot(
            [it for it, _ in llh], [ll for _, ll in llh],
            color="C1", marker="s", ms=2, label="log-likelihood",
        )
        a2.set_ylabel("per-token log-likelihood", color="C1")
        a2.tick_params(axis="y", labelcolor="C1")
        lines.append(line_ll)

    ax.legend(lines, [li.get_label() for li in lines], loc="best")
    ax.figure.tight_layout()
    return ax



def _mean_exclusivity(topic_word, n: int) -> float:
    from .coherence import exclusivity
    return float(np.mean(exclusivity(topic_word, n=n)))



def _cao_juan(topic_word) -> float:
    """Mean pairwise cosine similarity between topic-word distributions
    (Cao Juan et al. 2009). Lower is better -- redundant topics are similar, so
    the least-redundant K minimizes it. ``nan`` for a single topic."""
    phi = np.asarray(topic_word, dtype=np.float64)
    k = phi.shape[0]
    if k < 2:
        return float("nan")
    norm = np.linalg.norm(phi, axis=1)
    unit = phi / np.where(norm > 0, norm, 1.0)[:, None]
    sim = unit @ unit.T
    return float(sim[np.triu_indices(k, 1)].mean())



def _deveaud(topic_word) -> float:
    """Mean pairwise Jensen-Shannon divergence between topic-word distributions
    (Deveaud et al. 2014). Higher is better -- distinct topics diverge, so the
    most-distinct K maximizes it. ``nan`` for a single topic."""
    phi = np.asarray(topic_word, dtype=np.float64)
    phi = phi / np.clip(phi.sum(axis=1, keepdims=True), 1e-300, None)
    k = phi.shape[0]
    if k < 2:
        return float("nan")

    def _kl(p, q):
        # The mixture q = (p+q)/2 is strictly positive wherever p > 0, so no
        # smoothing is needed and the result is an exact Jensen-Shannon term.
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            mix = 0.5 * (phi[i] + phi[j])
            total += 0.5 * _kl(phi[i], mix) + 0.5 * _kl(phi[j], mix)
            pairs += 1
    return total / pairs



def _extra_criterion(name, topic_word) -> float:
    return {"deveaud": _deveaud, "cao_juan": _cao_juan}[name](topic_word)



# ---------------------------------------------------------------------------
# Model-quality frontier + bootstrap stability
# ---------------------------------------------------------------------------

def quality_frontier(model, *, n=10, texts=None, coherence_type="u_mass", plot=False):
    """Per-topic coherence, exclusivity, and prevalence — the data behind stm's
    classic coherence-vs-exclusivity quality plot.

    Returns a :class:`~topica._results.QualityFrontier` (a ``dict`` of
    equal-length arrays — ``topic``, ``coherence``, ``exclusivity``,
    ``prevalence`` (mean θ) — that also offers ``.to_frame()`` for a tidy
    one-row-per-topic DataFrame). By default coherence is the fast per-topic
    UMass score; pass ``texts`` and a windowed ``coherence_type`` (e.g. ``"c_v"``)
    for the human-aligned measure. With ``plot=True`` (and matplotlib installed) a
    labeled scatter ``Figure`` is returned alongside the data as ``(data, fig)``.
    """
    from .coherence import coherence as _coherence, exclusivity as _exclusivity

    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    K = phi.shape[0]
    if texts is not None and coherence_type != "u_mass":
        coh = np.asarray(_coherence(model, texts, coherence_type=coherence_type, topn=n))
    else:
        # The windowed coherence types need a reference corpus; without `texts`
        # the only score available is UMass. Warn rather than silently returning
        # UMass under the requested name — the scales differ (UMass ~ (-inf, 0],
        # c_v ~ [0, 1]), so a mislabeled axis invites wrong comparisons.
        if texts is None and coherence_type != "u_mass":
            warnings.warn(
                f"quality_frontier: coherence_type={coherence_type!r} needs texts "
                "(a reference corpus); without them coherence is UMass, which is on "
                "a different scale. Pass texts= or set coherence_type='u_mass'.",
                stacklevel=2,
            )
        coh = np.asarray(model.coherence(n))
    from ._results import QualityFrontier

    data = QualityFrontier({
        "topic": np.arange(K),
        "coherence": coh,
        "exclusivity": _exclusivity(phi, n=n),
        "prevalence": theta.mean(axis=0),
    })
    if not plot:
        return data
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot=True requires matplotlib") from exc
    fig, ax = plt.subplots()
    ax.scatter(data["coherence"], data["exclusivity"],
               s=300 * data["prevalence"] + 20)
    for t in range(K):
        ax.annotate(str(t), (data["coherence"][t], data["exclusivity"][t]))
    ax.set_xlabel("Semantic coherence")
    ax.set_ylabel("Exclusivity")
    ax.set_title("Topic quality (size ∝ prevalence)")
    return data, fig


def __dir__():
    """Show only the public workflow surface in tab-completion (#757), hiding the
    module's own imports (np, re, dataclass, ...)."""
    return sorted(__all__)
