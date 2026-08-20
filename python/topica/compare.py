"""``topica.compare(fit_a, fit_b)`` — a *statistical* comparison of two topic-model
fits (issue #415).

Comparing two fits is usually a manual affair: eyeball the top words, guess whether
the topics "changed". This makes it a first-class operation that answers the
question social scientists actually ask — *"did the topics really change, or did the
sampler just wander?"* — and doubles as a library regression test.

Three uses, one tool:

1. **Over time / two corpora** (same model, different slice): which topics are
   stable, which drift, which are new or gone.
2. **Two seeds / two versions of the same corpus**: how much of the apparent change
   is just Monte-Carlo wander — the *baseline* real drift is judged against.
3. **Library regression test**: refit a canonical corpus across topica versions and
   flag when a "harmless" refactor silently moves the topics.

Design commitments:

- **Alignment is reused, not reinvented.** Matching runs through
  :func:`topica.evaluate.align_topics`, which pairs two topics by the Hungarian 1-to-1
  assignment and keeps a pair when its similarity clears ``threshold``, then reports
  splits, merges, and unaligned topics. Splits/merges are a background-relative overlay
  (an extra partner close to a topic's own best match relative to the fit's cross-topic
  floor), so correlated-topic models are not mislabelled as unstable (issue #642). (The
  reseed null below reads the same Hungarian self-assignment, so a topic always has a
  self-match to measure wander against.)
- **The "unmatched" bucket is honest.** A topic with no good counterpart is reported
  as *appeared* / *vanished*, never paired to its least-bad neighbor; a split (one
  topic in A → two in B) is a named outcome. This matters most across different K.
- **Drift needs a null.** A raw distance between two topics is uninterpretable on its
  own. Supply a reseed baseline (``refit=`` a callable that refits A under a new
  seed, or ``baseline=`` a similarity ceiling) and each matched pair is flagged when
  it moves *beyond the range of self-agreement A shows across the reseeds*. This is a
  heuristic band, **not a calibrated test**: the floor is the worst self-match over
  ``n_reseed`` refits (so a genuinely-null pair trips it at a rough
  ``~1/(n_reseed+1)`` rate), and only A is reseeded. Without a null, distances are
  still reported but ``drifted`` is ``None`` (honestly "unknown").
- **Not an ensemble.** This describes and tests *difference*; it does not build a
  consensus model (that is :func:`topica.ensemble`).

Manifests, not just live models
-------------------------------
``compare`` also accepts two :class:`~topica.provenance.AnalysisManifest` records recorded with
``record_fit(..., topic_words_n>0)``, so two fits can be compared *without refitting*
— useful when the models themselves are gone but their provenance records remain.
A manifest stores only each topic's top-N words and mean prevalence, so the manifest
path aligns by **Jaccard overlap of the top-word sets** (``metric="jaccard"``) with a
threshold tuned to that scale, carries no prevalence-shift uncertainty (no posterior
is stored), and admits only ``baseline=`` as a drift null (a manifest cannot be
refit). Mixing a live model with a manifest is refused.
"""

from __future__ import annotations

import html as _html
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .evaluate import _classify_alignment, _hungarian, align_topics

__all__ = ["compare", "CompareResult", "MatchedPair", "UnmatchedTopic"]

# Cosine on distributions and Jaccard on top-word sets live on different scales, so
# the manifest path needs its own default. Cosine's 0.3 is permissive (shared common
# words inflate cross-topic cosine); Jaccard of two top-10 sets is 0.33 at 5 shared
# words, 0.18 at 3 — so a Jaccard default must be *lower* to stay comparably
# permissive. ~0.12 ≈ "share roughly 2 of 10 top words" as the floor for a match.
_LIVE_DEFAULT_THRESHOLD = 0.3
_MANIFEST_DEFAULT_THRESHOLD = 0.12

# An unmatched topic whose best cross-similarity reaches this fraction of the match
# threshold is flagged a "near-miss": likely the same topic seen through churned top
# words (esp. on the coarse-grained manifest/Jaccard path), not a real vanish/appear.
_NEAR_MISS_FRACTION = 0.8


def _esc(x: Any) -> str:
    return _html.escape(str(x))


def _top_words(model_or_phi, n: int) -> list[list[str]]:
    """Top-``n`` words per topic as string lists, for display and set overlap.

    Uses a model's ``top_words`` when present (the reference-faithful ranking),
    else argsorts a topic-word array against its vocabulary."""
    tw = getattr(model_or_phi, "top_words", None)
    if callable(tw):
        try:
            allrows = tw(n)  # list[list[str]] (or [(word, weight)] with weights=True)
            return [[w[0] if isinstance(w, tuple) else w for w in row] for row in allrows]
        except TypeError:
            pass
    from .coherence import _as_topic_word

    phi = _as_topic_word(model_or_phi)
    vocab = getattr(model_or_phi, "vocabulary", None)
    vocab = list(vocab) if vocab is not None else None
    order = np.argsort(-phi, axis=1)[:, :n]
    return [
        [vocab[j] if vocab is not None else str(int(j)) for j in row] for row in order
    ]


def _prevalence(model, corpus, nsims: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean topic prevalence over documents, ``(K,)``, and its per-topic standard
    error ``(K,)`` when a posterior over theta is available (``None`` otherwise).

    The point estimate is ``doc_topic.mean(axis=0)``. The uncertainty draws
    ``nsims`` theta matrices via :func:`topica.composition_theta` (no ``corpus``
    needed for logistic-normal models or Gibbs models with retained draws) and takes
    the between-draw spread of the per-draw document means."""
    from .coherence import _as_doc_topic, _as_topic_word

    # A raw K×V array (or a model with no doc-topic surface) has no prevalence;
    # report it as NaN rather than failing the whole comparison.
    dt = getattr(model, "doc_topic", None)
    if dt is None and isinstance(model, np.ndarray):
        return np.full(model.shape[0], np.nan), None
    try:
        point = _as_doc_topic(model).mean(axis=0)
    except Exception:
        k = _as_topic_word(model).shape[0]
        return np.full(k, np.nan), None
    se = None
    try:
        from .effects import composition_theta

        draws = composition_theta(model, corpus, nsims=nsims, seed=seed)  # (S, D, K)
        per_draw = draws.mean(axis=1)  # (S, K)
        if per_draw.shape[0] >= 2:
            se = per_draw.std(axis=0, ddof=1)
    except Exception:
        se = None  # no posterior / needs a corpus we were not given — point only.
    return np.asarray(point, dtype=np.float64), se


@dataclass
class MatchedPair:
    """One aligned topic pair ``(topic_a → topic_b)`` and how much it moved."""

    topic_a: int
    topic_b: int
    similarity: float
    distance: float
    #: ``True`` if the pair moved more than the reseed null allows, ``False`` if
    #: within reseed noise, ``None`` if no null baseline was supplied.
    drifted: bool | None
    #: The per-topic null threshold (a similarity floor) this pair was judged
    #: against, or ``None`` when no null was supplied.
    null_similarity: float | None
    top_words_a: list[str]
    top_words_b: list[str]
    prevalence_a: float
    prevalence_b: float
    prevalence_shift: float
    #: Uncertainty of the prevalence shift: the two fits' posterior-spread SDs of
    #: mean prevalence combined in quadrature (assumes the fits are independent —
    #: see :func:`compare`). ``None`` when no posterior over theta was available.
    prevalence_shift_se: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic_a": self.topic_a,
            "topic_b": self.topic_b,
            "similarity": self.similarity,
            "distance": self.distance,
            "drifted": self.drifted,
            "null_similarity": self.null_similarity,
            "prevalence_a": self.prevalence_a,
            "prevalence_b": self.prevalence_b,
            "prevalence_shift": self.prevalence_shift,
            "prevalence_shift_se": self.prevalence_shift_se,
            "top_words_a": self.top_words_a,
            "top_words_b": self.top_words_b,
        }


@dataclass
class UnmatchedTopic:
    """A topic with no honest counterpart: *appeared* (only in B) or *vanished*
    (only in A)."""

    topic: int
    side: str  # "a" or "b"
    status: str  # "vanished" (a-only) or "appeared" (b-only)
    top_words: list[str]
    prevalence: float
    #: This topic's best similarity to *any* topic on the other side. Usually below
    #: ``threshold`` (which is why the topic is unmatched), but not always: the
    #: Hungarian-anchored classifier can leave a topic unmatched with a best
    #: similarity at or above ``threshold`` — a split/merge child that lost the
    #: one-to-one assignment (named in ``splits``/``merges``). When it sits *just
    #: under* the threshold the "disappearance" may be top-word churn rather than a
    #: real vanish/appear (see :attr:`near_miss`); ``None`` if the other side is empty.
    best_similarity: float | None = None

    @property
    def near_miss(self) -> bool:
        """``True`` when this topic's best cross-similarity is close to (but under)
        the match threshold — a hint that it is the same topic seen through churned
        top words, not a genuine appearance/disappearance."""
        # bool(): the flag may be set from a numpy comparison (np.bool_), and
        # ``np.True_ is True`` is False, which would silently drop the flag.
        return bool(self._near_miss)

    #: Set by :func:`compare` once the threshold is known — a post-construction
    #: flag, not a constructor input (``init=False``), and kept out of ``repr`` /
    #: equality so two topics compare on their data, not on this derived hint.
    _near_miss: bool | None = field(default=None, init=False, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "side": self.side,
            "status": self.status,
            "prevalence": self.prevalence,
            "top_words": self.top_words,
            "best_similarity": self.best_similarity,
            "near_miss": self.near_miss,
        }


@dataclass
class CompareResult:
    """The outcome of :func:`compare`. See that function for the fields' meaning."""

    aligned: list[MatchedPair]
    unmatched_a: list[UnmatchedTopic]
    unmatched_b: list[UnmatchedTopic]
    splits: dict[int, list[int]]
    merges: dict[int, list[int]]
    metric: str
    threshold: float
    num_topics_a: int
    num_topics_b: int
    #: Provenance of the drift null: ``{"kind": "reseed"|"baseline"|"none", ...}``.
    baseline: dict[str, Any] = field(default_factory=dict)

    # -- convenience views ------------------------------------------------

    @property
    def drift(self) -> list[dict[str, Any]]:
        """Per-matched-pair drift: distance and whether it beats the reseed null."""
        return [
            {
                "topic_a": p.topic_a,
                "topic_b": p.topic_b,
                "distance": p.distance,
                "similarity": p.similarity,
                "drifted": p.drifted,
                "null_similarity": p.null_similarity,
            }
            for p in self.aligned
        ]

    @property
    def prevalence_shift(self) -> list[dict[str, Any]]:
        """Per-matched-pair change in topic prevalence, with a posterior-spread
        uncertainty (the two fits' between-draw SDs combined in quadrature; see
        :func:`compare`) when a posterior over theta was available for both fits."""
        return [
            {
                "topic_a": p.topic_a,
                "topic_b": p.topic_b,
                "prevalence_a": p.prevalence_a,
                "prevalence_b": p.prevalence_b,
                "shift": p.prevalence_shift,
                "se": p.prevalence_shift_se,
            }
            for p in self.aligned
        ]

    @property
    def n_drifted(self) -> int | None:
        """How many matched pairs drifted beyond the null (``None`` if no null)."""
        if any(p.drifted is None for p in self.aligned):
            return None
        return sum(1 for p in self.aligned if p.drifted)

    def __repr__(self) -> str:  # noqa: D105
        drifted = self.n_drifted
        drift_txt = "no null" if drifted is None else f"{drifted} drifted"
        return (
            f"CompareResult(K {self.num_topics_a}→{self.num_topics_b}, "
            f"matched={len(self.aligned)}, vanished={len(self.unmatched_a)}, "
            f"appeared={len(self.unmatched_b)}, splits={len(self.splits)}, "
            f"merges={len(self.merges)}, {drift_txt})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "num_topics_a": self.num_topics_a,
            "num_topics_b": self.num_topics_b,
            "baseline": self.baseline,
            "aligned": [p.as_dict() for p in self.aligned],
            "unmatched_a": [u.as_dict() for u in self.unmatched_a],
            "unmatched_b": [u.as_dict() for u in self.unmatched_b],
            "splits": {int(k): list(v) for k, v in self.splits.items()},
            "merges": {int(k): list(v) for k, v in self.merges.items()},
        }

    # -- rendering (matches the manifest analysis-card house style) -------

    def render(self, path: str | None = None, *, title: str | None = None) -> str:
        """Render an HTML *comparison card* (same house as
        :meth:`AnalysisManifest.render`). Returns the HTML string and writes it to
        ``path`` if given."""
        doc = _render_html(self, title=title)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(doc, encoding="utf-8")
        return doc

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown card (Quarto/notebook)."""
        return _render_markdown(self)


def _reseed_null(
    a,
    refit: Callable[[int], Any] | None,
    reseed_fits: Sequence[Any] | None,
    n_reseed: int,
    metric: str,
    threshold: float,
    seed: int,
    by: str = "words",
) -> dict[int, float] | None:
    """Per-A-topic similarity floor from reseeding: refit A under fresh seeds, align
    each refit back to A, and take the *worst* self-match similarity each topic
    achieves. An A↔B match below that floor moved more than reseeding A alone did.

    Returns ``{topic_a: floor}`` or ``None`` when no reseed source was supplied.
    """
    fits: list[Any] = list(reseed_fits) if reseed_fits is not None else []
    if refit is not None:
        for i in range(n_reseed):
            fits.append(refit(seed + 1 + i))
    if not fits:
        return None
    # For each reseed, the similarity each A-topic keeps under its best match.
    per_topic: dict[int, list[float]] = {}
    for f in fits:
        al = align_topics(a, f, by=by, metric=metric, threshold=threshold)
        best: dict[int, float] = {}
        for (ta, _tb, dist) in al:
            best[ta] = max(best.get(ta, 0.0), 1.0 - float(dist))
        for ta, s in best.items():
            per_topic.setdefault(ta, []).append(s)
    # The floor is the worst (min) self-agreement observed for the topic.
    return {ta: float(min(vals)) for ta, vals in per_topic.items() if vals}


def _is_manifest(x) -> bool:
    from .manifest import AnalysisManifest

    return isinstance(x, AnalysisManifest)


def _unmatched(
    idx, side, status, words, prev, sim, threshold, *, axis
) -> UnmatchedTopic:
    """Build an :class:`UnmatchedTopic`, recording its best similarity to the other
    side and flagging a near-miss (best sim just under ``threshold``). ``axis=0``
    reads a row of ``sim`` (an A-topic vs all B); ``axis=1`` reads a column."""
    row = sim[idx, :] if axis == 0 else sim[:, idx]
    best = float(row.max()) if row.size else None
    u = UnmatchedTopic(int(idx), side, status, words, float(prev), best_similarity=best)
    # A near-miss is a topic that *almost* matched: its best cross-similarity sits
    # just under the match threshold. The upper bound matters since the (#642)
    # Hungarian-anchored classifier can leave a topic unmatched with a best
    # similarity at or ABOVE threshold — a split/merge child that lost the 1-to-1
    # assignment. That is a strong partner, not a churn near-miss, and the
    # splits/merges overlay already names it, so it must not be flagged here.
    u._near_miss = (
        best is not None and _NEAR_MISS_FRACTION * threshold <= best < threshold
    )
    return u


def _manifest_top_words(m, side: str) -> list[list[str]]:
    """Retained top words from a manifest, or a clear error if it was recorded
    without them (``topic_words_n=0``, the default, or an older manifest)."""
    words = m.model.get("top_words")
    if not words:
        raise ValueError(
            f"manifest {side} carries no retained top words, so its topics cannot "
            "be compared. Re-record it with record_fit(..., topic_words_n=25) "
            "(top words are opt-in because they are corpus-derived content; ~25 is a "
            "safer floor than 10 for seed-varying fits), or compare the live models."
        )
    # The retained rows are the topics we align on; if the record's own num_topics
    # disagrees, the manifest is internally inconsistent — flag it rather than
    # silently comparing on a truncated/padded topic set.
    recorded_k = m.model.get("num_topics")
    if recorded_k is not None and recorded_k != len(words):
        raise ValueError(
            f"manifest {side} records num_topics={recorded_k} but retained "
            f"{len(words)} top-word rows; the record is inconsistent and cannot be "
            "compared."
        )
    return [list(row) for row in words]


def _manifest_prevalence(m, k: int) -> np.ndarray:
    """Retained per-topic prevalence, or NaN when the recorder stored none (a model
    with no doc-topic surface) — mirroring the live path's NaN guard."""
    prev = m.model.get("topic_prevalence")
    if prev is None:
        return np.full(k, np.nan)
    arr = np.asarray(prev, dtype=np.float64)
    if arr.shape[0] != k:
        raise ValueError(
            f"manifest stores {arr.shape[0]} prevalence values but {k} topics "
            "(top-word rows); the record is inconsistent and cannot be compared."
        )
    return arr


def _jaccard_matrix(words_a: list[list[str]], words_b: list[list[str]]) -> np.ndarray:
    """``|A∩B| / |A∪B|`` over each pair of top-word sets. Two empty sets score 0.0
    (undefined 0/0 → "no overlap", never a spurious match)."""
    sets_a = [set(w) for w in words_a]
    sets_b = [set(w) for w in words_b]
    m = np.zeros((len(sets_a), len(sets_b)))
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            union = len(sa | sb)
            m[i, j] = (len(sa & sb) / union) if union else 0.0
    return m


class _Alignment:
    """The subset of :class:`~topica.validation.AlignmentResult` that :func:`compare`
    consumes, produced from a precomputed similarity matrix. Kept local (rather than
    refactoring ``align_topics``) so the shipped Hungarian/``js`` paths and their
    consumers are untouched."""

    def __init__(self, matches, splits, merges, unaligned_a, unaligned_b):
        self.matches = matches
        self.splits = splits
        self.merges = merges
        self.unaligned_a = unaligned_a
        self.unaligned_b = unaligned_b


def _classify_similarity(sim: np.ndarray, threshold: float) -> _Alignment:
    """Hungarian-anchored classification from a precomputed similarity matrix, using the
    same shared classifier as :func:`align_topics` (``validation._classify_alignment``,
    issue #642): matches are the Hungarian 1-to-1 pairs above ``threshold``, and
    splits/merges are a background-relative overlay that stays empty on a correlated
    self-alignment. Used by the manifest path, which has only a Jaccard matrix (no live
    ``align_topics`` call) to classify."""
    pairs = _hungarian(1.0 - sim)
    matches, splits, merges, unaligned_a, unaligned_b = _classify_alignment(
        sim, pairs, threshold
    )
    return _Alignment(matches, splits, merges, unaligned_a, unaligned_b)


def _compare_manifests(
    a,
    b,
    *,
    metric: str | None,
    threshold: float | None,
    refit,
    reseed_fits,
    baseline: float | None,
) -> CompareResult:
    """Topic-for-topic comparison of two manifests via Jaccard overlap of their
    retained top-word sets — no refitting. See :func:`compare`."""
    if refit is not None or reseed_fits is not None:
        raise ValueError(
            "a manifest cannot be refit, so refit=/reseed_fits= are unavailable on "
            "the manifest path; pass baseline= for a drift null, or compare the "
            "live models."
        )
    if metric not in (None, "jaccard"):
        raise ValueError(
            f"manifest comparison aligns by top-word overlap only; metric must be "
            f"'jaccard' (got {metric!r}). Compare the live models for cosine/js/rbo/emd."
        )
    metric = "jaccard"
    thr = _MANIFEST_DEFAULT_THRESHOLD if threshold is None else float(threshold)

    words_a = _manifest_top_words(a, "a")
    words_b = _manifest_top_words(b, "b")
    ka, kb = len(words_a), len(words_b)
    prev_a = _manifest_prevalence(a, ka)
    prev_b = _manifest_prevalence(b, kb)

    # Jaccard of two top-word sets of different sizes is systematically depressed
    # (a 10-word set nested in a 25-word one scores at most 10/25 = 0.4), which can
    # push a genuine match under the threshold and manufacture spurious
    # vanished/appeared/split/merge. Two manifests recorded with different
    # topic_words_n are only comparable on their common window, so truncate both to
    # the smaller retained size (rows are ranked, so this keeps the top words) and
    # warn — silently comparing mismatched windows is the real trap.
    n_a = max((len(r) for r in words_a), default=0)
    n_b = max((len(r) for r in words_b), default=0)
    if n_a != n_b and n_a and n_b:
        common = min(n_a, n_b)
        warnings.warn(
            f"the two manifests retained different numbers of top words per topic "
            f"({n_a} vs {n_b}); comparing on the common top {common} so the Jaccard "
            "overlap is not systematically depressed. Re-record both with the same "
            "topic_words_n to use the full window.",
            stacklevel=3,
        )
        words_a = [r[:common] for r in words_a]
        words_b = [r[:common] for r in words_b]

    sim = _jaccard_matrix(words_a, words_b)
    al = _classify_similarity(sim, thr)

    baseline_info: dict[str, Any] = (
        {"kind": "baseline", "similarity_floor": float(baseline)}
        if baseline is not None
        else {"kind": "none"}
    )

    def _floor_for(_ta: int) -> float | None:
        return float(baseline) if baseline is not None else None

    aligned: list[MatchedPair] = []
    for (ta, tb, s) in al.matches:
        floor = _floor_for(ta)
        aligned.append(
            MatchedPair(
                topic_a=int(ta),
                topic_b=int(tb),
                similarity=float(s),
                distance=float(1.0 - s),
                drifted=None if floor is None else bool(s < floor),
                null_similarity=floor,
                top_words_a=words_a[ta],
                top_words_b=words_b[tb],
                prevalence_a=float(prev_a[ta]),
                prevalence_b=float(prev_b[tb]),
                prevalence_shift=float(prev_b[tb] - prev_a[ta]),
                prevalence_shift_se=None,  # no posterior retained in a manifest
            )
        )
    aligned.sort(key=lambda p: p.topic_a)

    unmatched_a = [
        _unmatched(t, "a", "vanished", words_a[t], prev_a[t], sim, thr, axis=0)
        for t in sorted(al.unaligned_a)
    ]
    unmatched_b = [
        _unmatched(t, "b", "appeared", words_b[t], prev_b[t], sim, thr, axis=1)
        for t in sorted(al.unaligned_b)
    ]
    splits = {int(k): [int(j) for (j, _s) in v] for k, v in al.splits.items()}
    merges = {int(k): [int(i) for (i, _s) in v] for k, v in al.merges.items()}

    return CompareResult(
        aligned=aligned,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        splits=splits,
        merges=merges,
        metric=metric,
        threshold=thr,
        num_topics_a=ka,
        num_topics_b=kb,
        baseline=baseline_info,
    )


def compare(
    a,
    b,
    *,
    by: str = "words",
    metric: str | None = None,
    threshold: float | None = None,
    refit: Callable[[int], Any] | None = None,
    reseed_fits: Sequence[Any] | None = None,
    n_reseed: int = 4,
    baseline: float | None = None,
    corpus_a=None,
    corpus_b=None,
    nsims: int = 25,
    seed: int = 0,
    top_n: int = 10,
) -> CompareResult:
    """Compare two fitted topic models statistically.

    ``a``, ``b`` are two fitted models (or ``K×V`` topic-word arrays), **or** two
    :class:`~topica.provenance.AnalysisManifest` records recorded with ``topic_words_n>0`` — in
    which case topics are aligned by Jaccard overlap of the retained top-word sets
    without refitting (see the module docstring; mixing a model with a manifest is
    refused). Topics are matched one-to-one by the Hungarian assignment
    (:func:`align_topics`), keeping a pair when its similarity clears ``threshold``;
    topics with no honest counterpart are reported as *vanished* (only in ``a``) or
    *appeared* (only in ``b``). One-to-many relationships (*splits* / *merges*) are an
    overlay on top of that matching, flagged only when an extra partner is close to a
    topic's own best match *relative to this fit's cross-topic similarity floor* — so
    correlated-topic families (STM/CTM), whose off-diagonal cosines are high, are no
    longer reported as near-total splits/merges, and comparing a fit with itself yields
    K matched / 0 split / 0 merged regardless of correlation (issue #642).

    ``by`` chooses the space topics are matched in. The default ``"words"`` matches
    two topics when they use the same vocabulary. ``by="documents"`` instead matches
    them when the same documents load on them (cosine of the two document-topic
    columns), so a topic that persists across two fits is recognized even when its
    top words churn; it needs two live fits on the *same documents in the same order*
    (not manifests). It complements :func:`topica.agreement`, which scores the two
    fits' whole document partitions rather than pairing topics.

    ``metric``/``threshold`` default per path: live fits use ``metric="cosine"`` with
    ``threshold=0.3`` (the minimum similarity for a one-to-one match; splits/merges
    self-calibrate to the fit and do not depend on it); manifests use
    ``metric="jaccard"`` with a lower default (~0.12) matched to the Jaccard scale of
    top-word sets. Pass ``threshold=`` to override either. ``metric`` applies to the
    word space only; ``by="documents"`` always uses cosine on the document loadings.

    **Drift needs a null.** Pass exactly one reseed source to judge whether a matched
    pair moved beyond the self-agreement A shows across reseeds (a heuristic band —
    see the module docstring — not a calibrated significance test):

    - ``refit`` — a callable ``seed -> fitted_model`` that refits ``a`` on the same
      corpus with a new seed (``compare`` calls it ``n_reseed`` times); or
    - ``reseed_fits`` — an iterable of already-refit models of ``a``; or
    - ``baseline`` — a single similarity floor (e.g. a family self-agreement ceiling
      from :func:`ensemble`) applied to every pair.

    With a null, each :class:`MatchedPair` gets ``drifted=True/False``; without one,
    ``drifted`` is ``None`` (reported honestly as "unknown", distances still given).

    **Prevalence shift** is ``b``'s minus ``a``'s topic prevalence
    (``doc_topic.mean(0)``) per matched pair. When a posterior over theta is
    available (logistic-normal models, or Gibbs models with retained draws — pass
    ``corpus_a``/``corpus_b`` for the Dirichlet approximation otherwise) the shift
    carries an uncertainty: each fit's posterior spread of the mean prevalence
    (the between-draw SD over ``nsims`` ``composition_theta`` draws) combined in
    quadrature, ``sqrt(se_a**2 + se_b**2)``. This treats the two fits' posteriors as
    independent — correct for two genuinely distinct fits, and conservative (it
    cannot go to zero) as the two fits approach identical.

    On the live path ``metric`` is passed to :func:`align_topics` (``"cosine"``,
    ``"js"``, ``"rbo"``, ``"emd"``); ``threshold`` is the minimum similarity for an
    honest match. On the manifest path only ``metric="jaccard"`` and ``baseline=``
    (not ``refit=``/``reseed_fits=``) are available.

    Returns a :class:`CompareResult` (see ``.aligned``, ``.unmatched_a/b``,
    ``.splits``, ``.merges``, ``.drift``, ``.prevalence_shift``, ``.render()``).
    """
    n_sources = sum(x is not None for x in (refit, reseed_fits, baseline))
    if n_sources > 1:
        raise ValueError(
            "pass at most one drift null: refit=, reseed_fits=, or baseline="
        )

    # Manifest-native path (issue #415): compare two provenance records directly.
    a_man, b_man = _is_manifest(a), _is_manifest(b)
    if a_man != b_man:
        raise TypeError(
            "cannot compare a live model/array with an AnalysisManifest; pass two "
            "manifests (recorded with topic_words_n>0) or two live models."
        )
    if a_man and b_man:
        if by == "documents":
            raise ValueError(
                "document-based comparison (by='documents') needs live fitted models "
                "with a document-topic matrix; manifests store only top words."
            )
        return _compare_manifests(
            a, b, metric=metric, threshold=threshold,
            refit=refit, reseed_fits=reseed_fits, baseline=baseline,
        )

    # Live path: resolve the documented cosine defaults from the None sentinels.
    metric = "cosine" if metric is None else metric
    threshold = _LIVE_DEFAULT_THRESHOLD if threshold is None else threshold

    al = align_topics(a, b, by=by, metric=metric, threshold=threshold)
    words_a = _top_words(a, top_n)
    words_b = _top_words(b, top_n)
    prev_a, se_a = _prevalence(a, corpus_a, nsims, seed)
    prev_b, se_b = _prevalence(b, corpus_b, nsims, seed)
    ka, kb = len(words_a), len(words_b)

    # Drift null: reseed floors per A-topic, or a flat baseline, or nothing.
    null_floor: dict[int, float] | None = None
    baseline_info: dict[str, Any]
    if baseline is not None:
        baseline_info = {"kind": "baseline", "similarity_floor": float(baseline)}
    elif refit is not None or reseed_fits is not None:
        null_floor = _reseed_null(a, refit, reseed_fits, n_reseed, metric, threshold, seed, by=by)
        n_used = (len(reseed_fits) if reseed_fits is not None else 0) + (
            n_reseed if refit is not None else 0
        )
        # An empty reseed source (e.g. reseed_fits=[]) yields no floor; report that
        # honestly as "no null" rather than a reseed run of size 0.
        baseline_info = (
            {"kind": "reseed", "n_reseed": n_used}
            if null_floor is not None
            else {"kind": "none"}
        )
    else:
        baseline_info = {"kind": "none"}

    def _floor_for(ta: int) -> float | None:
        if baseline is not None:
            return float(baseline)
        if null_floor is not None:
            # A topic the reseeds never matched has no floor → cannot judge.
            return null_floor.get(ta)
        return None

    aligned: list[MatchedPair] = []
    for (ta, tb, sim) in al.matches:
        floor = _floor_for(ta)
        drifted = None if floor is None else bool(sim < floor)
        shift = float(prev_b[tb] - prev_a[ta])
        shift_se = None
        if se_a is not None and se_b is not None:
            shift_se = float(np.sqrt(se_a[ta] ** 2 + se_b[tb] ** 2))
        aligned.append(
            MatchedPair(
                topic_a=int(ta),
                topic_b=int(tb),
                similarity=float(sim),
                distance=float(1.0 - sim),
                drifted=drifted,
                null_similarity=floor,
                top_words_a=words_a[ta],
                top_words_b=words_b[tb],
                prevalence_a=float(prev_a[ta]),
                prevalence_b=float(prev_b[tb]),
                prevalence_shift=shift,
                prevalence_shift_se=shift_se,
            )
        )
    aligned.sort(key=lambda p: p.topic_a)

    sim_mat = np.asarray(al.similarity_matrix)
    unmatched_a = [
        _unmatched(t, "a", "vanished", words_a[t], prev_a[t], sim_mat, threshold, axis=0)
        for t in sorted(al.unaligned_a)
    ]
    unmatched_b = [
        _unmatched(t, "b", "appeared", words_b[t], prev_b[t], sim_mat, threshold, axis=1)
        for t in sorted(al.unaligned_b)
    ]
    splits = {int(k): [int(j) for (j, _s) in v] for k, v in al.splits.items()}
    merges = {int(k): [int(i) for (i, _s) in v] for k, v in al.merges.items()}

    return CompareResult(
        aligned=aligned,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        splits=splits,
        merges=merges,
        metric=metric,
        threshold=threshold,
        num_topics_a=ka,
        num_topics_b=kb,
        baseline=baseline_info,
    )


# --- rendering -----------------------------------------------------------------

_CARD_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 color:#202124;max-width:900px;margin:1.5rem auto;padding:0 1rem;line-height:1.5}
h1{font-size:1.4rem;margin:0 0 .2rem} h2{font-size:1.05rem;margin:1.4rem 0 .4rem}
.sub{color:#5f6368;margin:0 0 1rem} .note{color:#5f6368;font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.3rem 0}
th,td{border:1px solid #e0e0e0;padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:#f8f9fa;font-weight:600} td.num{text-align:right;font-variant-numeric:tabular-nums}
.words{color:#3c4043;font-size:.85rem} .tag{font-size:.8rem;padding:.05rem .4rem;border-radius:.6rem}
.drift{background:#fce8e6;color:#c5221f} .stable{background:#e6f4ea;color:#137333}
.unk{background:#f1f3f4;color:#5f6368}
""".strip()


def _drift_tag(p: MatchedPair) -> str:
    if p.drifted is None:
        return "<span class='tag unk'>—</span>"
    if p.drifted:
        return "<span class='tag drift'>drifted</span>"
    return "<span class='tag stable'>stable</span>"


def _render_html(r: CompareResult, *, title: str | None) -> str:
    title = title or "Topic comparison"
    parts: list[str] = [f"<style>{_CARD_CSS}</style>"]
    drifted = r.n_drifted
    drift_txt = "no drift null supplied" if drifted is None else f"{drifted} drifted beyond the reseed range"
    sub = (
        f"K {r.num_topics_a} → {r.num_topics_b} · {len(r.aligned)} matched · "
        f"{len(r.unmatched_a)} vanished · {len(r.unmatched_b)} appeared · metric={r.metric}"
    )
    parts.append(f"<h1>{_esc(title)}</h1><p class='sub'>{_esc(sub)}</p>")
    kind = r.baseline.get("kind", "none")
    if kind == "none":
        # A manifest cannot be refit, so only baseline= is available there; advertise
        # what the path actually accepts rather than options that would raise.
        how = (
            "Pass <code>baseline=</code> to test drift against a similarity floor."
            if r.metric == "jaccard"
            else "Pass <code>refit=</code>, <code>reseed_fits=</code>, or "
            "<code>baseline=</code> to test drift against reseed noise."
        )
        parts.append(
            "<p class='note'>No drift baseline supplied, so drift is reported as a "
            f"raw distance only (<em>drifted = unknown</em>). {how}</p>"
        )
    else:
        parts.append(f"<p class='note'>Drift null: {_esc(kind)} ({_esc(drift_txt)}).</p>")

    # Matched topics.
    rows = [
        "<tr><th>A</th><th>B</th><th>similarity</th><th>drift</th>"
        "<th>prevalence Δ</th><th>top words (B)</th></tr>"
    ]
    for p in r.aligned:
        shift = f"{p.prevalence_shift:+.3f}"
        if p.prevalence_shift_se is not None:
            shift += f" ± {p.prevalence_shift_se:.3f}"
        rows.append(
            f"<tr><td class='num'>{p.topic_a}</td><td class='num'>{p.topic_b}</td>"
            f"<td class='num'>{p.similarity:.3f}</td><td>{_drift_tag(p)}</td>"
            f"<td class='num'>{_esc(shift)}</td>"
            f"<td class='words'>{_esc(', '.join(p.top_words_b[:8]))}</td></tr>"
        )
    parts.append(f"<h2>Matched topics</h2><table>{''.join(rows)}</table>")

    # Unmatched / split / merge.
    if r.unmatched_a or r.unmatched_b:
        urows = ["<tr><th>side</th><th>topic</th><th>status</th><th>prevalence</th>"
                 "<th>best sim</th><th>top words</th></tr>"]
        for u in r.unmatched_a + r.unmatched_b:
            best = "—" if u.best_similarity is None else f"{u.best_similarity:.3f}"
            if u.near_miss:
                best += " <span class='tag unk'>near-miss</span>"
            urows.append(
                f"<tr><td>{_esc(u.side.upper())}</td><td class='num'>{u.topic}</td>"
                f"<td>{_esc(u.status)}</td><td class='num'>{u.prevalence:.3f}</td>"
                f"<td class='num'>{best}</td>"
                f"<td class='words'>{_esc(', '.join(u.top_words[:8]))}</td></tr>"
            )
        note = ""
        if any(u.near_miss for u in r.unmatched_a + r.unmatched_b):
            # The remedy differs by path: the manifest/Jaccard path is coarse and
            # improves with a wider top-word window (or the live models); the live
            # path is already the fine-grained comparison, so point at the threshold.
            fix = (
                "Retaining more top words (higher <code>topic_words_n</code>) or "
                "comparing the live models resolves it."
                if r.metric == "jaccard"
                else "Lowering <code>threshold</code> would match it, if the overlap "
                "is real."
            )
            note = (
                f"<p class='note'>A <em>near-miss</em> sits just under the match "
                f"threshold ({r.threshold:.2f}) — likely the same topic seen through "
                f"churned top words rather than a real appearance/disappearance. "
                f"{fix}</p>"
            )
        parts.append(f"<h2>Unmatched</h2><table>{''.join(urows)}</table>{note}")
    if r.splits:
        parts.append(
            "<h2>Splits (A → many B)</h2><p class='words'>"
            + _esc("; ".join(f"{k} → {v}" for k, v in r.splits.items()))
            + "</p>"
        )
    if r.merges:
        parts.append(
            "<h2>Merges (many A → B)</h2><p class='words'>"
            + _esc("; ".join(f"{k} ← {v}" for k, v in r.merges.items()))
            + "</p>"
        )
    return "<div class='topica-compare'>" + "".join(parts) + "</div>"


def _render_markdown(r: CompareResult) -> str:
    lines: list[str] = ["# Topic comparison", ""]
    drifted = r.n_drifted
    drift_txt = "no drift null" if drifted is None else f"{drifted} drifted"
    lines.append(
        f"K {r.num_topics_a} → {r.num_topics_b} · {len(r.aligned)} matched · "
        f"{len(r.unmatched_a)} vanished · {len(r.unmatched_b)} appeared · "
        f"metric={r.metric} · {drift_txt}"
    )
    lines += ["", "## Matched topics", "", "| A | B | similarity | drift | prevalence Δ | top words (B) |",
              "|---|---|---|---|---|---|"]
    for p in r.aligned:
        d = "—" if p.drifted is None else ("drifted" if p.drifted else "stable")
        shift = f"{p.prevalence_shift:+.3f}"
        if p.prevalence_shift_se is not None:
            shift += f" ± {p.prevalence_shift_se:.3f}"
        lines.append(
            f"| {p.topic_a} | {p.topic_b} | {p.similarity:.3f} | {d} | {shift} | "
            f"{', '.join(p.top_words_b[:8])} |"
        )
    if r.unmatched_a or r.unmatched_b:
        lines += ["", "## Unmatched", ""]
        for u in r.unmatched_a + r.unmatched_b:
            flag = ""
            if u.near_miss:
                flag = f" _(near-miss: best sim {u.best_similarity:.3f}, just under {r.threshold:.2f})_"
            lines.append(f"- **{u.side.upper()} topic {u.topic}** ({u.status}): "
                         f"{', '.join(u.top_words[:8])}{flag}")
    if r.splits:
        lines += ["", "## Splits (A → many B)", ""]
        lines += [f"- {k} → {v}" for k, v in r.splits.items()]
    if r.merges:
        lines += ["", "## Merges (many A → B)", ""]
        lines += [f"- {k} ← {v}" for k, v in r.merges.items()]
    return "\n".join(lines) + "\n"


# Make the module itself callable so that `topica.compare(fit_a, fit_b)` (the
# function call the whole docstring teaches) and `topica.compare.CompareResult`
# (the workflow namespace) both resolve. Well-trodden pattern (cf. `sh`); keeps
# `callable(topica.compare)` true after `compare` becomes a namespace (issue #757).
import sys as _sys
from types import ModuleType as _ModuleType


class _CompareModule(_ModuleType):
    def __call__(self, *args, **kwargs):
        return compare(*args, **kwargs)


_sys.modules[__name__].__class__ = _CompareModule
