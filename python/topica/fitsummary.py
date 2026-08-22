"""Per-model fit-stats summary (issue #806): the topic-model analogue of an OLS
table that always prints R2/N/F. A fitted model's ``repr`` shows the cheap stored
scalars (converged, iterations, the family objective, K/D/V), so ``print(m)`` or a
bare notebook eval gives the block by default. ``model.summary(texts=...)`` adds the
model-agnostic quality tier (coherence, exclusivity), which needs a reference corpus
and so is never computed by ``repr``. ``model.summary(heldout=...)`` adds the
held-out fit numbers (perplexity, or held-out log-likelihood for a ``make_heldout``
split), which need documents the model was not trained on.

The family objective in the fit tier is the *in-sample* training objective (the last
``fit_history`` entry), a convergence witness, not a quality score — labeled as such
so no one reads it as held-out fit. Held-out perplexity is deliberately kept out of
the free ``repr`` and the ``texts=`` path: computed on the training corpus it is
misleading, and even held-out it correlates negatively with topic interpretability
(Chang et al. 2009), so it is a diagnostic, never the number to optimize.

Spike status: wired for ``LDA`` first to prove the shape before rolling the binding
out to the full roster. The objective map and the builder are written generically so
the rollout is adding entries, not rewriting logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which native objective each model family reports. Exactly one of
# log_likelihood / elbo / reconstruction_error is filled for an iterative model;
# the rest render as n/a. Seeded with LDA for the spike; the rollout extends this.
_OBJECTIVE_KIND = {
    "LDA": "log_likelihood",
}


def _objective_kind(model) -> str:
    return _OBJECTIVE_KIND.get(type(model).__name__, "none")


def _is_fitted(model) -> bool:
    """A fitted model exposes its convergence trace; probing an unfitted one raises
    (there is no public ``fitted`` flag on the native classes)."""
    try:
        model.fit_history
        return True
    except Exception:
        return False


def _fmt(value) -> str:
    """Render a summary cell; ``None`` is the family-does-not-apply marker."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        a = abs(value)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


@dataclass(frozen=True)
class FitSummary:
    """One per-model header of fit statistics. Two tiers: family-internal *fit
    diagnostics* (comparable only within a family) and model-agnostic *quality*
    metrics (the cross-family analogue of R2), the latter populated only when
    ``summary(texts=...)`` is given a reference corpus."""

    model: str
    fitted: bool
    num_topics: int | None = None
    num_docs: int | None = None
    vocab_size: int | None = None
    # fit diagnostics (family-internal)
    converged: bool | None = None
    iterations: int | None = None
    objective_kind: str = "none"
    log_likelihood: float | None = None
    elbo: float | None = None
    reconstruction_error: float | None = None
    # held-out fit (needs documents the model was not trained on, so only with heldout=)
    perplexity: float | None = None
    heldout_loglik: float | None = None
    # topic health (cheap, corpus-free: from phi/theta already in memory)
    effective_topics: float | None = None
    diversity: float | None = None
    topic_redundancy: float | None = None
    topic_significance: float | None = None
    weak_topics: int | None = None
    # quality metrics (model-agnostic; need a reference corpus, so only with texts=)
    coherence: float | None = None
    exclusivity: float | None = None

    def _shape_rows(self):
        topics = self.num_topics
        if topics is not None and self.effective_topics is not None:
            topics = f"{topics} (effective {self.effective_topics:.2g})"
        return [
            ("topics", topics),
            ("documents", self.num_docs),
            ("vocabulary", self.vocab_size),
        ]

    def _fit_rows(self):
        return [
            ("converged", self.converged),
            ("iterations", self.iterations),
            ("log_likelihood (in-sample)", self.log_likelihood),
            ("elbo (in-sample)", self.elbo),
            ("reconstruction_error", self.reconstruction_error),
            ("perplexity (held-out)", self.perplexity),
            ("heldout_loglik", self.heldout_loglik),
        ]

    def _health_rows(self):
        weak = self.weak_topics
        if weak is not None and self.num_topics is not None:
            weak = f"{weak} of {self.num_topics}"
        return [
            ("diversity", self.diversity),
            ("topic_redundancy", self.topic_redundancy),
            ("topic_significance", self.topic_significance),
            ("weak_topics", weak),
        ]

    def _quality_rows(self):
        return [
            ("coherence", self.coherence),
            ("exclusivity", self.exclusivity),
        ]

    def _diagnostic_rows(self):
        return self._shape_rows() + self._fit_rows() + self._health_rows()

    def __str__(self) -> str:
        head = f"{self.model}"
        if not self.fitted:
            return f"{head} (unfitted)"
        lines = [head]
        width = max(len(k) for k, _ in self._diagnostic_rows())
        for label, value in self._diagnostic_rows():
            lines.append(f"  {label:<{width}}  {_fmt(value)}")
        if any(v is not None for _, v in self._quality_rows()):
            lines.append("  --- quality (vs. reference corpus) ---")
            for label, value in self._quality_rows():
                lines.append(f"  {label:<{width}}  {_fmt(value)}")
        else:
            lines.append("  (call .summary(texts=corpus) for coherence and exclusivity)")
        return "\n".join(lines)

    __repr__ = __str__

    def to_markdown(self) -> str:
        rows = self._diagnostic_rows()
        if any(v is not None for _, v in self._quality_rows()):
            rows = rows + self._quality_rows()
        body = "\n".join(f"| {label} | {_fmt(value)} |" for label, value in rows)
        return f"### {self.model} fit summary\n\n| statistic | value |\n| --- | --- |\n{body}\n"

    def _repr_html_(self) -> str:
        def cells(pairs):
            return "".join(
                f"<tr><td style='padding:2px 10px 2px 0;color:#666'>{label}</td>"
                f"<td style='padding:2px 0;font-variant-numeric:tabular-nums'>{_fmt(value)}</td></tr>"
                for label, value in pairs
            )

        if not self.fitted:
            return f"<b>{self.model}</b> <span style='color:#999'>(unfitted)</span>"
        html = [
            f"<div style='font-family:system-ui,sans-serif'><b>{self.model}</b> fit summary",
            "<table style='border-collapse:collapse;margin-top:4px'>",
            cells(self._diagnostic_rows()),
        ]
        if any(v is not None for _, v in self._quality_rows()):
            html.append(cells(self._quality_rows()))
        html.append("</table></div>")
        return "".join(html)


def _topic_word(model):
    """The ``(K, V)`` topic-word matrix as an array, or ``None`` if the model does
    not expose one as a plain attribute (e.g. time-sliced models where it is a
    callable) — those skip the corpus-free health tier rather than guessing a slice."""
    tw = getattr(model, "topic_word", None)
    if tw is None or callable(tw):
        return None
    arr = np.asarray(tw, dtype=np.float64)
    return arr if arr.ndim == 2 else None


def _topic_health(phi, doc_topic, *, n=10):
    """Cheap, corpus-free topic diagnostics from the fitted matrices:
    effective_topics (exp-entropy of prevalence), diversity (distinct top-n words),
    topic_redundancy (mean off-diagonal topic-word cosine). Any stat that cannot be
    computed for a given model is returned as ``None`` (renders n/a). The weak-topic
    count is computed separately in :func:`_significance_health`, off the maintained
    OCTIS significance score rather than an inline KL-to-uniform."""
    effective = diversity = redundancy = None

    theta = np.asarray(doc_topic, dtype=np.float64) if getattr(doc_topic, "shape", None) else None
    if theta is not None and theta.ndim == 2 and theta.shape[0] > 0:
        prev = theta.mean(axis=0)
        total = prev.sum()
        if total > 0:
            prev = prev / total
            nz = prev[prev > 0]
            effective = float(np.exp(-(nz * np.log(nz)).sum()))

    if phi is not None and phi.shape[0] > 0:
        k, v = phi.shape
        topn = min(n, v)
        # diversity: fraction of distinct words among the pooled top-n lists
        tops = np.argsort(phi, axis=1)[:, ::-1][:, :topn]
        pooled = tops.reshape(-1)
        diversity = float(len(np.unique(pooled)) / pooled.shape[0]) if pooled.size else None
        # redundancy: mean off-diagonal cosine between topic-word rows
        if k > 1:
            norm = phi / np.clip(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12, None)
            sim = norm @ norm.T
            off = sim[~np.eye(k, dtype=bool)]
            redundancy = float(off.mean())

    return effective, diversity, redundancy


def _significance_health(model):
    """Topic significance from the maintained OCTIS scorer
    (``evaluate.topic_significance``, Aletras and Stevenson 2013): the mean topic
    distinctiveness in nats, plus a count of near-background topics flagged by a
    relative threshold on the per-topic scores (the ranking use the score's own
    docstring recommends, since the nats are not comparable across corpora). Uses the
    ``vacuous`` null (distance from the corpus-average word distribution) when a
    document-topic matrix is present, else the ``uniform`` null. Corpus-free. Returns
    ``(None, None)`` for models the scorer cannot handle."""
    from . import evaluate

    kind = "vacuous" if getattr(model, "doc_topic", None) is not None else "uniform"
    try:
        scores = np.asarray(
            evaluate.topic_significance(model, kind=kind, per_topic=True), dtype=np.float64
        )
    except Exception:
        return None, None
    if scores.size == 0 or not np.isfinite(scores).all():
        return None, None
    mean_sig = float(scores.mean())
    weak = int((scores < 0.1 * scores.max()).sum()) if scores.max() > 0 else None
    return mean_sig, weak


def _build_summary(model, *, texts=None, heldout=None, assume_unseen=False, n=10) -> FitSummary:
    name = type(model).__name__
    if not _is_fitted(model):
        return FitSummary(model=name, fitted=False,
                          num_topics=getattr(model, "num_topics", None))

    vocab = getattr(model, "vocabulary", None)
    vocab_size = len(vocab) if vocab is not None else None
    doc_topic = getattr(model, "doc_topic", None)
    num_docs = int(doc_topic.shape[0]) if getattr(doc_topic, "shape", None) else None

    history = list(model.fit_history)
    iterations = int(history[-1][0]) if history else None
    objective_value = float(history[-1][1]) if history else None
    kind = _objective_kind(model)

    log_likelihood = objective_value if kind == "log_likelihood" else None
    elbo = objective_value if kind == "elbo" else None
    reconstruction_error = objective_value if kind == "reconstruction_error" else None

    # Topic-health tier: cheap and corpus-free, from the phi/theta already in memory.
    effective_topics, diversity, topic_redundancy = _topic_health(
        _topic_word(model), doc_topic, n=n
    )
    topic_significance, weak_topics = _significance_health(model)

    coherence = exclusivity = None
    if texts is not None:
        # Quality tier proper: coherence and exclusivity need the reference corpus.
        from . import evaluate

        table = evaluate.diagnostics(model, texts=texts, n=n)
        coherence = float(np.nanmean(table["coherence"].to_numpy()))
        exclusivity = float(np.nanmean(table["exclusivity"].to_numpy()))

    # Held-out fit tier: perplexity / held-out log-likelihood need documents the model
    # was not trained on, never the training texts. A make_heldout() split (word
    # holdout in place) is scored by eval_heldout; plain held-out documents by the
    # document-completion perplexity estimator. We cannot verify unseen-ness from here
    # (the native models keep no __dict__, so no training fingerprint can be stashed on
    # them), so a "held-out" number is only produced from a make_heldout split (safe by
    # construction) or from documents the caller explicitly affirms are unseen with
    # assume_unseen=True. A bare corpus therefore raises instead of silently reporting
    # training-data perplexity under a held-out label (sample-user audit, issue #806).
    perplexity = heldout_loglik = None
    if heldout is not None:
        from . import evaluate

        if isinstance(heldout, evaluate.Heldout):
            heldout_loglik = float(evaluate.eval_heldout(model, heldout).mean_per_doc_loglik)
        elif assume_unseen:
            perplexity = float(evaluate.perplexity(model, heldout))
        else:
            raise ValueError(
                "summary() will not label a number 'held-out' unless the documents are "
                "verifiably unseen, and it cannot check that for you. Two safe options:\n"
                "  - within-corpus split (recommended):\n"
                "        ho = topica.make_heldout(corpus)\n"
                "        model.fit(ho.documents)\n"
                "        model.summary(heldout=ho)\n"
                "  - a genuinely external test set the model was NOT trained on:\n"
                "        model.summary(heldout=test_docs, assume_unseen=True)\n"
                "Passing the training corpus here would report training-data perplexity "
                "as if it were held-out."
            )

    return FitSummary(
        model=name,
        fitted=True,
        num_topics=getattr(model, "num_topics", None),
        num_docs=num_docs,
        vocab_size=vocab_size,
        converged=bool(model.converged),
        iterations=iterations,
        objective_kind=kind,
        log_likelihood=log_likelihood,
        elbo=elbo,
        reconstruction_error=reconstruction_error,
        perplexity=perplexity,
        heldout_loglik=heldout_loglik,
        effective_topics=effective_topics,
        diversity=diversity,
        topic_redundancy=topic_redundancy,
        topic_significance=topic_significance,
        weak_topics=weak_topics,
        coherence=coherence,
        exclusivity=exclusivity,
    )


def summary(model, texts=None, *, heldout=None, assume_unseen=False, n=10) -> FitSummary:
    """The fit-stats header for a model. Three optional tiers over the free repr:

    - no argument: the cheap stored scalars and corpus-free topic health.
    - ``texts=`` a reference corpus: the quality tier (coherence, exclusivity).
    - ``heldout=`` for the held-out fit tier. A :func:`topica.make_heldout` split is
      safe by construction and yields held-out log-likelihood. Raw documents cannot be
      verified as unseen from here, so they raise unless you affirm them with
      ``assume_unseen=True`` (then you get document-completion perplexity) — this is the
      guard that stops a training corpus being reported as held-out. Never pass the
      training corpus.
    """
    return _build_summary(model, texts=texts, heldout=heldout, assume_unseen=assume_unseen, n=n)


def _model_repr(model) -> str:
    return str(_build_summary(model))


def _model_repr_html(model) -> str:
    return _build_summary(model)._repr_html_()


def _bind_fit_summary(classes) -> None:
    """Attach ``summary`` / ``__repr__`` / ``_repr_html_`` onto each model class.
    Mirrors ``inspect._bind_topic_table_method``: the native classes are heap types,
    so Python-side assignment updates the repr slot without a Rust change."""
    for cls in classes:
        if cls is None:
            continue
        try:
            cls.summary = summary
            cls.__repr__ = _model_repr
            cls._repr_html_ = _model_repr_html
        except (TypeError, AttributeError):
            pass
