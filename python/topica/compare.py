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
  :func:`topica.align_topics`, which pairs two topics only when each is the other's
  *unique* above-``threshold`` partner (a mutual-best rule that never force-pairs),
  and reports splits, merges, and unaligned topics. (The reseed null below instead
  reads the Hungarian self-assignment `align_topics` also exposes, so a topic always
  has a self-match to measure wander against.)
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
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .validation import align_topics

__all__ = ["compare", "CompareResult", "MatchedPair", "UnmatchedTopic"]


def _esc(x: Any) -> str:
    return _html.escape(str(x))


def _top_words(model_or_phi, n: int) -> list[list[str]]:
    """Top-``n`` words per topic as string lists, for display and set overlap.

    Uses a model's ``top_words`` when present (the reference-faithful ranking),
    else argsorts a topic-word array against its vocabulary."""
    tw = getattr(model_or_phi, "top_words", None)
    if callable(tw):
        try:
            allrows = tw(n)  # list[list[(word, weight)]]
            return [[w for w, _ in row] for row in allrows]
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "side": self.side,
            "status": self.status,
            "prevalence": self.prevalence,
            "top_words": self.top_words,
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
        al = align_topics(a, f, metric=metric, threshold=threshold)
        best: dict[int, float] = {}
        for (ta, _tb, dist) in al:
            best[ta] = max(best.get(ta, 0.0), 1.0 - float(dist))
        for ta, s in best.items():
            per_topic.setdefault(ta, []).append(s)
    # The floor is the worst (min) self-agreement observed for the topic.
    return {ta: float(min(vals)) for ta, vals in per_topic.items() if vals}


def compare(
    a,
    b,
    *,
    metric: str = "cosine",
    threshold: float = 0.3,
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

    ``a``, ``b`` are two fitted models (or ``K×V`` topic-word arrays). Topics are
    matched one-to-one when each is the other's *unique* above-``threshold`` partner
    (:func:`align_topics`, a mutual-best rule — never force-paired); topics
    with no honest counterpart are reported as *vanished* (only in ``a``) or
    *appeared* (only in ``b``), and one-to-many relationships as *splits* / *merges*.
    ``threshold=0.3`` is permissive for ``metric="cosine"`` on real corpora (shared
    common words push cross-topic cosine up), so raise it if you see spurious
    splits/merges.

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

    ``metric`` is passed to :func:`align_topics` (``"cosine"``, ``"js"``, ``"rbo"``,
    ``"emd"``); ``threshold`` is the minimum similarity for an honest match.

    Returns a :class:`CompareResult` (see ``.aligned``, ``.unmatched_a/b``,
    ``.splits``, ``.merges``, ``.drift``, ``.prevalence_shift``, ``.render()``).
    """
    n_sources = sum(x is not None for x in (refit, reseed_fits, baseline))
    if n_sources > 1:
        raise ValueError(
            "pass at most one drift null: refit=, reseed_fits=, or baseline="
        )

    al = align_topics(a, b, metric=metric, threshold=threshold)
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
        null_floor = _reseed_null(a, refit, reseed_fits, n_reseed, metric, threshold, seed)
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

    unmatched_a = [
        UnmatchedTopic(int(t), "a", "vanished", words_a[t], float(prev_a[t]))
        for t in sorted(al.unaligned_a)
    ]
    unmatched_b = [
        UnmatchedTopic(int(t), "b", "appeared", words_b[t], float(prev_b[t]))
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
        parts.append(
            "<p class='note'>No reseed baseline supplied, so drift is reported as a "
            "raw distance only (<em>drifted = unknown</em>). Pass <code>refit=</code>, "
            "<code>reseed_fits=</code>, or <code>baseline=</code> to test drift "
            "against reseed noise.</p>"
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
        urows = ["<tr><th>side</th><th>topic</th><th>status</th><th>prevalence</th><th>top words</th></tr>"]
        for u in r.unmatched_a + r.unmatched_b:
            urows.append(
                f"<tr><td>{_esc(u.side.upper())}</td><td class='num'>{u.topic}</td>"
                f"<td>{_esc(u.status)}</td><td class='num'>{u.prevalence:.3f}</td>"
                f"<td class='words'>{_esc(', '.join(u.top_words[:8]))}</td></tr>"
            )
        parts.append(f"<h2>Unmatched</h2><table>{''.join(urows)}</table>")
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
            lines.append(f"- **{u.side.upper()} topic {u.topic}** ({u.status}): "
                         f"{', '.join(u.top_words[:8])}")
    if r.splits:
        lines += ["", "## Splits (A → many B)", ""]
        lines += [f"- {k} → {v}" for k, v in r.splits.items()]
    if r.merges:
        lines += ["", "## Merges (many A → B)", ""]
        lines += [f"- {k} ← {v}" for k, v in r.merges.items()]
    return "\n".join(lines) + "\n"
