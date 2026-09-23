"""Per-model fit-stats summary (issue #806): the topic-model analogue of an OLS
table that always prints R2/N/F. Printing a fitted model (``print(m)``, or a bare
notebook eval via ``_repr_html_``) shows the cheap stored scalars (converged,
iterations, the family objective, K/D/V); ``repr(m)`` stays the compact one-line
constructor form so lists of models and debuggers stay readable. ``model.summary(texts=...)`` adds the
model-agnostic quality tier (coherence, exclusivity), which needs a reference corpus
and so is never computed by ``repr``. ``model.summary(heldout=...)`` adds the
held-out fit numbers (perplexity, or held-out log-likelihood for a ``make_heldout``
split), which need documents the model was not trained on.

The family objective in the fit tier is the *in-sample* training objective (the last
``fit_history`` entry), a convergence witness, not a quality score, and labeled as such
so no one reads it as held-out fit. Held-out perplexity is deliberately kept out of
the free ``repr`` and the ``texts=`` path: computed on the training corpus it is
misleading, and even held-out it correlates negatively with topic interpretability
(Chang et al. 2009), so it is a diagnostic, never the number to optimize.

Every model in :data:`topica.REGISTRY` gets the binding. Families differ in what
they optimize, so the objective is one field whose label names what the model's own
``fit_history`` records (log-likelihood, ELBO, reconstruction error, ...); a model not
in the label table falls back to the neutral label "objective", and a model with no
iterative trace (LSA, BERTopic, ...) shows ``n/a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# What each model's ``fit_history`` second column records, in the words of that
# model's own documentation. These are in-sample training objectives: convergence
# witnesses, comparable only within one model, never across families. A model absent
# from this table gets the neutral "objective" label rather than a guessed one.
_LL = "log-likelihood"
_ELBO_DOC = "ELBO per document"
_OBJECTIVE_LABEL = {
    # collapsed Gibbs: joint log-likelihood of the training tokens (DMR, GDMR,
    # LabeledLDA, and SAGE return their log_likelihood_history)
    "LDA": _LL, "PT": _LL, "KeyATM": _LL, "SeededLDA": _LL, "FactorialLDA": _LL,
    "AuthorTopic": _LL, "MGLDA": _LL, "TopicsOverTime": _LL, "DMR": _LL, "GDMR": _LL,
    "LabeledLDA": _LL, "SAGE": _LL,
    # sLDA records only the Gaussian likelihood of the response y, not of the tokens
    "SupervisedLDA": "response log-likelihood",
    # Wordfish's Poisson log-likelihood drops the -log(y!) constant
    "Wordfish": "log-likelihood (up to a constant)",
    # per-token averages
    "HDP": "log-likelihood per token", "GSDMM": "log-likelihood per token",
    "GaussianLDA": "log-likelihood per token",
    # variational EM: corpus-level bound
    "OnlineLDA": "ELBO", "CTM": "variational bound", "STM": "variational bound",
    # amortized (VAE) inference: the batch-mean ELBO, per document
    "ProdLDA": _ELBO_DOC, "CombinedTM": _ELBO_DOC, "ZeroShotTM": _ELBO_DOC,
    "Scholar": _ELBO_DOC, "InfoCTM": _ELBO_DOC,
    # DETM rescales each minibatch to the corpus (D/batch), so its ELBO is corpus-scale
    "DETM": "ELBO",
    # ETM reports either, depending on its inference route
    "ETM": lambda m: _ELBO_DOC if _get(m, "inference") == "vae" else "variational bound",
    # factorization and other objectives
    "NMF": "reconstruction error", "CorEx": "total correlation",
    "FASTopic": "Sinkhorn loss (negated)",
    # traces that are not objectives
    "TopicalNGrams": "bigram tokens",
    "SemanticSignalSeparation": "convergence measure",
    "TensorLDA": "convergence measure",
}


def _objective_label(model) -> str:
    label = _OBJECTIVE_LABEL.get(type(model).__name__, "objective")
    return label(model) if callable(label) else label


def _sampler_kind(model) -> str:
    """The registry's inference route ("gibbs", "variational", ...). Collapsed Gibbs
    samplers run a fixed number of sweeps and do not early-stop by default, so their
    ``converged`` flag is off unless a tolerance was set and tripped."""
    from .registry import REGISTRY

    info = REGISTRY.get(type(model).__name__)
    return info.inference if info is not None else "none"


def _get(model, name):
    """An attribute, or ``None`` when the model does not have it or cannot compute it
    (many are properties that raise on an unfitted or structurally different model)."""
    try:
        value = getattr(model, name)
    except Exception:
        return None
    return None if callable(value) else value


def _is_fitted(model) -> bool:
    """Native classes carry no public ``fitted`` flag, but their result matrices raise
    until ``fit`` has run, so a model is fitted once one of them can be read. (An empty
    ``fit_history`` is not evidence either way: some classes return ``[]`` before
    fitting, and non-iterative models return ``[]`` after.)"""
    if any(_get(model, a) is not None for a in ("topic_word", "doc_topic")):
        return True
    return bool(_get(model, "fit_history"))


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
    ``summary(texts=...)`` is given a reference corpus.

    Field legend (each renders ``n/a`` when it does not apply to the family):

    - ``objective`` (labeled by ``objective_label``, for example ``log-likelihood``,
      ``ELBO``, or ``reconstruction error``): the *in-sample* training objective at the
      last recorded iteration, a convergence witness rather than a quality score, and
      not comparable across families.
    - ``perplexity``: held-out document-completion perplexity; lower is better. Only
      from ``summary(heldout=...)``.
    - ``heldout_loglik``: mean per-document held-out log-likelihood from a
      :func:`topica.make_heldout` split (R stm's ``eval.heldout``); higher (less
      negative) is better.
    - ``effective_topics``: exp-entropy of topic prevalence, in ``[1, K]``: how many
      topics actually carry mass (near ``K`` = balanced, near ``1`` = one topic
      dominates).
    - ``diversity``: fraction of distinct words across the pooled top-n lists, in
      ``(0, 1]``; higher = less word overlap between topics.
    - ``topic_redundancy``: mean off-diagonal cosine between topic-word rows, in
      ``[0, 1]``; lower is better (0 = orthogonal topics).
    - ``topic_significance``: mean OCTIS significance (``evaluate.topic_significance``,
      KL to the corpus-average word distribution in nats); higher = more distinctive.
      Read as a within-model ranking, not an absolute threshold.
    - ``weak_topics``: count of near-background topics flagged by a relative threshold
      on the per-topic significance scores; lower is better.
    - ``coherence`` / ``exclusivity``: the quality tier (mean over topics), from
      ``summary(texts=corpus)``; higher is better.
    """

    model: str
    fitted: bool
    num_topics: int | None = None
    num_docs: int | None = None
    vocab_size: int | None = None
    # fit diagnostics (family-internal)
    converged: bool | None = None
    sampler: str = "none"
    iterations: int | None = None
    objective: float | None = None
    objective_label: str = "objective"
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
    # whether the model has a flat topic-word matrix (scaling models do not), which
    # decides whether the texts= quality tier applies at all
    has_topics: bool = True

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
        # A Gibbs sampler runs fixed sweeps and does not early-stop by default, so a
        # bare "no" misreads as failure; show why instead. A True flag means a set
        # tolerance actually tripped, which is worth reporting as-is.
        converged = self.converged
        if self.sampler == "gibbs" and converged is False:
            converged = "n/a (fixed sweeps)"
        return [
            ("converged", converged),
            ("iterations", self.iterations),
            (f"{self.objective_label} (in-sample)", self.objective),
            ("perplexity (held-out, document-completion)", self.perplexity),
            ("heldout_loglik (mean per-doc)", self.heldout_loglik),
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
        elif self.has_topics:
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
    callable); those skip the corpus-free health tier rather than guessing a slice."""
    tw = _get(model, "topic_word")
    if tw is None:
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
        return FitSummary(model=name, fitted=False, num_topics=_get(model, "num_topics"))

    vocab = _get(model, "vocabulary")
    vocab_size = len(vocab) if vocab is not None else None
    doc_topic = _get(model, "doc_topic")
    if not (getattr(doc_topic, "ndim", 0) == 2):
        doc_topic = None
    num_docs = int(doc_topic.shape[0]) if doc_topic is not None else None
    num_topics = _get(model, "num_topics")
    if not isinstance(num_topics, (int, np.integer)):
        num_topics = None

    history = _get(model, "fit_history")
    history = list(history) if history else []
    iterations = objective = None
    if history:
        last = history[-1]
        if isinstance(last, (tuple, list)) and len(last) == 2:
            iterations, objective = int(last[0]), float(last[1])
        else:  # a bare per-iteration trace (InfoCTM)
            iterations, objective = len(history), float(last)
    converged = _get(model, "converged")

    # Topic-health tier: cheap and corpus-free, from the phi/theta already in memory.
    effective_topics, diversity, topic_redundancy = _topic_health(
        _topic_word(model), doc_topic, n=n
    )
    topic_significance, weak_topics = _significance_health(model)

    coherence = exclusivity = None
    if texts is not None and _topic_word(model) is None:
        raise ValueError(
            f"{name} has no flat topic-word matrix, so coherence and exclusivity do not "
            "apply; call summary() without texts="
        )
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
        num_topics=num_topics,
        num_docs=num_docs,
        vocab_size=vocab_size,
        converged=bool(converged) if converged is not None else None,
        sampler=_sampler_kind(model),
        iterations=iterations,
        objective=objective,
        objective_label=_objective_label(model),
        perplexity=perplexity,
        heldout_loglik=heldout_loglik,
        effective_topics=effective_topics,
        diversity=diversity,
        topic_redundancy=topic_redundancy,
        topic_significance=topic_significance,
        weak_topics=weak_topics,
        coherence=coherence,
        exclusivity=exclusivity,
        has_topics=_topic_word(model) is not None,
    )


def summary(model, texts=None, *, heldout=None, assume_unseen=False, n=10) -> FitSummary:
    """The fit-stats header for a model. Three optional tiers over the free repr:

    - no argument: the cheap stored scalars and corpus-free topic health.
    - ``texts=`` a reference corpus: the quality tier (coherence, exclusivity).
    - ``heldout=`` for the held-out fit tier. A :func:`topica.make_heldout` split is
      safe by construction and yields held-out log-likelihood. Raw documents cannot be
      verified as unseen from here, so they raise unless you affirm them with
      ``assume_unseen=True`` (then you get document-completion perplexity). This is the
      guard that stops a training corpus being reported as held-out. Never pass the
      training corpus.
    """
    return _build_summary(model, texts=texts, heldout=heldout, assume_unseen=assume_unseen, n=n)


def _model_str(model) -> str:
    """``print(model)``: the fit block once fitted; the compact repr before that."""
    if not _is_fitted(model):
        return repr(model)
    return str(_build_summary(model))


def _model_repr_html(model) -> str:
    return _build_summary(model)._repr_html_()


def _bind_fit_summary(classes) -> None:
    """Attach ``summary`` / ``__str__`` / ``_repr_html_`` onto each model class.
    ``__repr__`` is left as the native one-line constructor form. Mirrors
    ``inspect._bind_topic_table_method``: the native classes are heap types, so
    Python-side assignment works without a Rust change. A class that already defines
    its own ``summary`` keeps it."""
    for cls in classes:
        if cls is None:
            continue
        try:
            if "summary" not in vars(cls):
                cls.summary = summary
            cls.__str__ = _model_str
            cls._repr_html_ = _model_repr_html
        except (TypeError, AttributeError):
            pass
