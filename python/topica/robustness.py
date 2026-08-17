"""Effect robustness across K and across seeds (issue #644).

The table a reviewer asks for first is not "what is the effect?" but *"does the
effect survive a different K, or a different seed?"* Answering it by hand means
refitting, matching topics across fits, pulling one coefficient out of each, and
assembling the result — enough bookkeeping that it usually does not get done, or
gets done by eyeballing top words.

:func:`effects_across_k` and :func:`effects_across_seeds` do that bookkeeping:
refit, align each fit's topics back to a **reference** fit, re-estimate the
covariate effect, and report one tidy row per (reference topic, fit) with the
coefficient, its interval, and whether the sign and significance held.

Design commitments:

- **Topics are matched, not assumed.** Fits are aligned with
  :func:`topica.evaluate.align_topics`' one-to-one Hungarian assignment, so topic 3 in the
  K=20 fit is compared against whichever topic it actually corresponds to at
  K=25 — never against "topic 3" by index.
- **Unmatched is reported, never dropped.** When K differs, some reference topics
  have no counterpart; those rows are emitted with ``matched=False`` and a null
  coefficient rather than silently omitted, so the table cannot overstate coverage.
- **The verdict is descriptive.** ``stable`` / ``flipped`` summarize whether the
  sign (and, if asked, the significance) held across fits. That is a robustness
  *description*, not a test: it does not correct for multiple comparisons and
  should be reported as "the sign held across K ∈ {…}", not as a p-value.
- **Nothing is refit twice.** Pass ``fits=`` to reuse models you already have.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Topic-stability helpers live in `evaluate`, but "does a topic survive a
# different seed?" is a robustness question, so users reach for
# `topica.robustness.topic_stability` first and hit AttributeError (#775 T4.1).
# Re-export them here for discoverability; `topica.evaluate` remains their home.
from .evaluate import bootstrap_stability, topic_stability

__all__ = [
    "effects_across_k",
    "effects_across_seeds",
    "RobustnessResult",
    "bootstrap_stability",
    "topic_stability",
]


class RobustnessResult(list):
    """The result of :func:`effects_across_k` / :func:`effects_across_seeds`.

    A ``list`` of per-(reference topic, fit) row dicts — so it iterates, indexes,
    and converts to a DataFrame directly — with summary views layered on top.

    Each row carries ``reference_topic``, the varied setting (``k`` or ``seed``),
    the ``matched_topic`` it aligned to (``None`` when unmatched), ``similarity``,
    and the effect of the tracked ``feature``: ``coef``, ``se``, ``ci_low``,
    ``ci_high``, ``pvalue``, plus ``sign`` and ``significant``.
    """

    def __init__(self, rows, *, feature, varied, reference, ci):
        super().__init__(rows)
        self.feature = feature
        #: ``"k"`` or ``"seed"`` — which setting was varied.
        self.varied = varied
        #: The setting value of the reference fit every other fit is aligned to.
        self.reference = reference
        self.ci = ci

    # -- per-topic views ---------------------------------------------------

    def _by_topic(self) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for r in self:
            out.setdefault(r["reference_topic"], []).append(r)
        return out

    @property
    def topics(self) -> list[int]:
        """The reference topics, sorted."""
        return sorted(self._by_topic())

    @property
    def stable(self) -> list[int]:
        """Reference topics whose effect sign is the same in **every** fit where
        the topic matched (and which matched everywhere). The honest reading is
        "the direction held across the settings scanned"."""
        return [t for t in self.topics if self._verdict(t) == "stable"]

    @property
    def flipped(self) -> list[int]:
        """Reference topics whose effect changes sign across fits — the ones to
        report as *not* robust, or not to interpret at all."""
        return [t for t in self.topics if self._verdict(t) == "flipped"]

    @property
    def unmatched(self) -> list[int]:
        """Reference topics that failed to match in at least one fit, so their
        robustness is undetermined rather than confirmed."""
        return [t for t in self.topics if self._verdict(t) == "unmatched"]

    def _verdict(self, topic: int) -> str:
        rows = self._by_topic().get(topic, [])
        if not rows:
            return "unmatched"
        if any(not r["matched"] for r in rows):
            return "unmatched"
        signs = {r["sign"] for r in rows if r["sign"] is not None}
        if len(signs) > 1:
            return "flipped"
        return "stable"

    def verdicts(self) -> dict[int, str]:
        """``{reference_topic: "stable" | "flipped" | "unmatched"}``."""
        return {t: self._verdict(t) for t in self.topics}

    # -- output ------------------------------------------------------------

    def to_frame(self):
        """The rows as a pandas DataFrame (raises if pandas is absent)."""
        import pandas as pd

        return pd.DataFrame(list(self))

    def summary(self) -> str:
        """A short, honest text summary naming the settings actually scanned."""
        vals = sorted({r[self.varied] for r in self})
        lines = [
            f"Effect robustness of {self.feature!r} across {self.varied} "
            f"∈ {vals} (reference {self.varied}={self.reference})",
            f"  stable (sign held):   {self.stable}",
            f"  flipped (sign moved): {self.flipped}",
            f"  unmatched somewhere:  {self.unmatched}",
            "Descriptive, not a test: the sign held across the settings scanned; "
            "no multiple-comparison correction is applied.",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobustnessResult(feature={self.feature!r}, varied={self.varied!r}, "
            f"topics={len(self.topics)}, stable={len(self.stable)}, "
            f"flipped={len(self.flipped)}, unmatched={len(self.unmatched)})"
        )


def _fit_one(docs, *, model, num_topics, seed, prevalence, content, iters):
    """Fit one model of the requested family. Mirrors ``search_k``'s constructor
    pattern so the two agree on what "an STM/LDA fit at K" means."""
    import topica

    name = model.lower() if isinstance(model, str) else model
    if callable(name):  # a user-supplied factory: (num_topics, seed) -> fitted model
        return name(num_topics, seed)
    if name == "stm":
        m = topica.STM(num_topics=num_topics, seed=seed)
        m.fit(docs, prevalence, content=content, iters=iters)
        return m
    if name == "lda":
        m = topica.LDA(num_topics=num_topics, seed=seed)
        m.fit(docs, iters=iters)
        return m
    raise ValueError(
        f"model must be 'stm', 'lda', or a callable (num_topics, seed) -> fitted "
        f"model; got {model!r}"
    )


def _match_map(reference, fit, metric):
    """``{reference_topic: (fit_topic, similarity)}`` from the one-to-one Hungarian
    assignment :func:`align_topics` exposes as its list interface.

    Deliberately the Hungarian pairing rather than the threshold-based
    ``matches``/``splits``/``merges`` view: a robustness table needs *the*
    counterpart of each reference topic, and the assignment always provides one for
    ``min(K_ref, K_fit)`` topics.
    """
    from .validation import align_topics

    pairs = align_topics(reference, fit, metric=metric)
    return {int(i): (int(j), float(1.0 - d)) for (i, j, d) in pairs}


def _effect_rows(fit, match, *, setting_name, setting_value, X, feature_names,
                 feature, ref_topics, ci, nsims, corpus, seed, min_similarity):
    """One row per reference topic for a single fit."""
    from .stm import estimate_effect

    kwargs = dict(X=X, feature_names=feature_names, ci=ci)
    if nsims:
        # method-of-composition needs a concrete RNG seed; the seed helper varies
        # the *fit* seed (and passes None here), so fall back to a fixed 0.
        effects = estimate_effect(fit, nsims=nsims, corpus=corpus,
                                  seed=(0 if seed is None else seed), **kwargs)
    else:
        effects = estimate_effect(fit, **kwargs)
    by_topic = {e.topic: e for e in effects}

    rows = []
    for t in ref_topics:
        base = {"reference_topic": int(t), setting_name: setting_value}
        hit = match.get(int(t))
        # A counterpart counts only if the Hungarian assignment produced one AND it
        # clears ``min_similarity``. Without the similarity gate a larger fit
        # force-matches every reference topic to some leftover topic regardless of
        # quality, which would let the table report a junk pairing as robust. The
        # near-miss similarity is still recorded so a borderline case is visible.
        if hit is None or hit[0] not in by_topic or hit[1] < min_similarity:
            rows.append({**base, "matched_topic": None, "matched": False,
                         "similarity": (None if hit is None else hit[1]),
                         "coef": None, "se": None,
                         "ci_low": None, "ci_high": None, "pvalue": None,
                         "sign": None, "significant": None})
            continue
        j, sim = hit
        eff = by_topic[j].effect_of(feature)
        coef = float(eff["coef"])
        lo, hi = float(eff["ci_low"]), float(eff["ci_high"])
        rows.append({
            **base,
            "matched_topic": int(j),
            "matched": True,
            "similarity": sim,
            "coef": coef,
            "se": float(eff["se"]),
            "ci_low": lo,
            "ci_high": hi,
            "pvalue": float(eff["pvalue"]) if eff.get("pvalue") is not None else None,
            # Sign of the point estimate, and whether the interval excludes zero.
            "sign": int(np.sign(coef)),
            "significant": bool(lo > 0.0 or hi < 0.0),
        })
    return rows


def _robustness(
    docs, settings, *, varied, num_topics, model, prevalence, content, feature,
    feature_names, X, iters, ci, metric, reference, nsims, corpus, seed, fits,
    min_similarity,
):
    """Shared driver for the across-K and across-seed helpers."""
    if not len(settings):
        raise ValueError(f"{varied}s is empty; pass at least two values to compare")
    design = prevalence if X is None else X
    if design is None:
        raise ValueError(
            "pass the covariate design (prevalence= for an STM fit, or X= to "
            "regress on a design the model was not fit with)"
        )
    # STM must be fit *with* its prevalence design, so an X-only call cannot refit
    # it. (fits= sidesteps this by reusing models you fit yourself; lda/callable
    # models are fit without a design and take X= for the effects step.)
    if (fits is None and isinstance(model, str) and model.lower() == "stm"
            and prevalence is None):
        raise ValueError(
            "model='stm' is fit with its prevalence design, so pass prevalence= "
            "(optionally with X= for a different effects design). X= alone applies "
            "to model='lda' or a callable, or pass fits= to reuse fitted models."
        )

    # Fit (or accept) one model per setting.
    if fits is not None:
        if len(fits) != len(settings):
            raise ValueError(
                f"fits has {len(fits)} models but {len(settings)} {varied}s were "
                "requested; pass one fitted model per setting, in the same order"
            )
        models = list(fits)
    else:
        models = [
            _fit_one(
                docs, model=model,
                num_topics=(s if varied == "k" else num_topics),
                seed=(s if varied == "seed" else seed),
                prevalence=prevalence, content=content, iters=iters,
            )
            for s in settings
        ]

    # The reference every other fit is aligned back to.
    ref_value = settings[0] if reference is None else reference
    if ref_value not in list(settings):
        raise ValueError(
            f"reference={reference!r} is not among the {varied}s scanned {list(settings)}"
        )
    ref_idx = list(settings).index(ref_value)
    ref_model = models[ref_idx]
    ref_topics = list(range(int(getattr(ref_model, "num_topics", 0))))

    rows: list[dict] = []
    for s, fit in zip(settings, models):
        match = ({t: (t, 1.0) for t in ref_topics} if fit is ref_model
                 else _match_map(ref_model, fit, metric))
        rows.extend(_effect_rows(
            fit, match, setting_name=varied, setting_value=s, X=design,
            feature_names=feature_names, feature=feature, ref_topics=ref_topics,
            ci=ci, nsims=nsims, corpus=corpus, seed=seed,
            min_similarity=min_similarity,
        ))
    return RobustnessResult(rows, feature=feature, varied=varied,
                            reference=ref_value, ci=ci)


def effects_across_k(
    docs,
    ks,
    *,
    feature,
    prevalence=None,
    X=None,
    feature_names=None,
    model="stm",
    content=None,
    iters=500,
    ci=0.95,
    metric="cosine",
    min_similarity=0.3,
    reference=None,
    nsims=None,
    corpus=None,
    seed=13,
    fits=None,
):
    """Is a covariate effect robust to the number of topics?

    Refits at each K, aligns every fit's topics back to a reference fit, re-runs
    :func:`~topica.effects.estimate_effect`, and reports one row per (reference topic, K)
    for the tracked ``feature`` — the robustness table an STM reviewer asks for.

    ``docs`` are the tokenized documents (or a :class:`~topica.Corpus`), ``ks`` the
    topic counts to scan. ``feature`` is the covariate whose coefficient is tracked,
    by name (as in ``feature_names``, e.g. ``"rating[T.Liberal]"``).

    The default ``model="stm"`` is fit with its prevalence design, so pass
    ``prevalence=`` (it is used both to fit each STM and as the effects design). Use
    ``X=`` to regress the effect on a design the model was *not* fit with — alongside
    ``prevalence=`` for STM, or on its own for ``model="lda"`` or a callable
    ``(num_topics, seed) -> fitted model`` (which are fit without a design). ``fits=``
    reuses models you already fit (one per K, in order) instead of refitting.

    Topics are matched by :func:`~topica.evaluate.align_topics`' one-to-one assignment, so a
    topic is compared with its actual counterpart, not its index. A counterpart
    counts only if its similarity clears ``min_similarity`` (default ``0.3``,
    :func:`~topica.evaluate.align_topics`' own match threshold); a reference topic with no
    counterpart — because K differs, or because the best pairing is below that
    threshold — is reported with ``matched=False`` and verdict ``unmatched`` rather
    than dropped or counted robust. Read the ``similarity`` column to judge
    borderline matches; lower ``min_similarity`` to ``0.0`` to keep every Hungarian
    pairing. With ``nsims`` the per-fit effects use method-of-composition intervals
    (pass ``corpus=`` for a Gibbs model); without it, point-θ OLS.

    Returns a :class:`RobustnessResult`: iterate it as rows, or read ``.stable`` /
    ``.flipped`` / ``.unmatched`` and ``.summary()``. Those verdicts are
    descriptive — "the sign held across the K scanned" — not a significance test.
    """
    return _robustness(
        docs, list(ks), varied="k", num_topics=None, model=model,
        prevalence=prevalence, content=content, feature=feature,
        feature_names=feature_names, X=X, iters=iters, ci=ci, metric=metric,
        reference=reference, nsims=nsims, corpus=corpus, seed=seed, fits=fits,
        min_similarity=min_similarity,
    )


def effects_across_seeds(
    docs,
    seeds,
    *,
    num_topics,
    feature,
    prevalence=None,
    X=None,
    feature_names=None,
    model="stm",
    content=None,
    iters=500,
    ci=0.95,
    metric="cosine",
    min_similarity=0.3,
    reference=None,
    nsims=None,
    corpus=None,
    fits=None,
):
    """Is a covariate effect robust to the seed?

    The seed-wander counterpart of :func:`effects_across_k`: refits at a fixed
    ``num_topics`` under each seed in ``seeds``, aligns every fit back to a
    reference fit, and reports the tracked ``feature``'s coefficient per
    (reference topic, seed).

    A model whose fit is bit-reproducible from its seed can still land in a
    different local optimum under a *different* seed; this shows whether the
    substantive conclusion survives that. Arguments and the returned
    :class:`RobustnessResult` are as in :func:`effects_across_k`, with ``seed``
    replacing ``k`` as the varied column.
    """
    return _robustness(
        docs, list(seeds), varied="seed", num_topics=num_topics, model=model,
        prevalence=prevalence, content=content, feature=feature,
        feature_names=feature_names, X=X, iters=iters, ci=ci, metric=metric,
        reference=reference, nsims=nsims, corpus=corpus, seed=None, fits=fits,
        min_similarity=min_similarity,
    )
