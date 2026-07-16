"""Prevalence analysis over a model's document-topic proportions ``theta``.

These tools work on the ``theta`` of any topica model:

- :func:`estimate_effect` regresses each topic's prevalence on document
  covariates (OLS, or the method of composition when given posterior draws).
- :func:`by_strata` reports mean prevalence within each level of a covariate.
- :func:`top_topics` lists each document's most prevalent topics.

Uncertainty propagation needs *draws* of ``theta``. STM and CTM have a
logistic-normal posterior, so :func:`posterior_theta_samples` draws from it. A
Gibbs model (LDA, keyATM, SeededLDA, ...) has no such posterior, so
:func:`dirichlet_theta_samples` draws ``theta`` from each document's Dirichlet
conditional given its proportions and length.
"""

from __future__ import annotations

import inspect

import numpy as np

from .stm import (
    estimate_effect,
    posterior_theta_samples,
    predicted_prevalence,
    PredictedPrevalence,
    average_marginal_effects,
    ame,
    MarginalEffect,
    AverageMarginalEffects,
)
from .keyatm import by_strata, top_topics

__all__ = [
    "estimate_effect",
    "posterior_theta_samples",
    "predicted_prevalence",
    "PredictedPrevalence",
    "average_marginal_effects",
    "ame",
    "MarginalEffect",
    "AverageMarginalEffects",
    "dirichlet_theta_samples",
    "by_strata",
    "top_topics",
    "standard_errors",
    "permutation_test",
    "PermutationResult",
]


def dirichlet_theta_samples(doc_topic, doc_lengths, *, nsims=25, seed=0, prior=0.0):
    """Draw `nsims` samples of the document-topic matrix θ for a Gibbs model.

    A collapsed-Gibbs model's `doc_topic` is the posterior mean of each
    document's θ given its token-topic assignments, where
    ``θ_d ~ Dirichlet(α + n_d)`` and ``(α + n_d) = doc_topic_d · (N_d + Σα)``.
    With the document length `N_d` we recover that Dirichlet and sample it, so the
    draws carry each document's within-document estimation uncertainty. Feed the
    result to :func:`estimate_effect` for method-of-composition standard errors on
    a model that has no logistic-normal posterior of its own.

    Parameters
    ----------
    doc_topic : array (num_docs, num_topics)
        The fitted θ (rows sum to one), e.g. ``model.doc_topic``.
    doc_lengths : array (num_docs,)
        Tokens per document (``[len(d) for d in docs]``). Longer documents give
        tighter draws, exactly as they pin θ more firmly in the model.
    nsims : int
        Number of θ draws.
    seed : int
        RNG seed.
    prior : float
        Extra concentration added to every document (a flat pseudo-count `Σα`
        spread over the topics). 0 uses the token counts alone.

    Returns
    -------
    array (nsims, num_docs, num_topics)
        Matches :func:`posterior_theta_samples`, ready for
        :func:`estimate_effect`.
    """
    theta = np.asarray(doc_topic, dtype=np.float64)
    lengths = np.asarray(doc_lengths, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("doc_topic must be a 2-D (num_docs, num_topics) array")
    if lengths.shape != (theta.shape[0],):
        raise ValueError("doc_lengths must have one entry per document")
    if prior < 0:
        raise ValueError("prior must be >= 0")

    # Concentration α + n_d for each document, where doc_topic is the posterior
    # mean (n_dk + α_k)/(N_d + Σα) and `prior` is Σα, so α_k + n_dk is exactly
    # theta · (N_d + prior). Clip tiny values so the gamma draws are well defined
    # for topics a document never uses.
    conc = theta * (lengths[:, None] + prior)
    conc = np.clip(conc, 1e-6, None)

    rng = np.random.default_rng(seed)
    # Dirichlet via independent gammas, normalized — vectorized over draws/docs.
    g = rng.standard_gamma(conc[None, :, :], size=(nsims,) + conc.shape)
    return g / g.sum(axis=2, keepdims=True)


# ---------------------------------------------------------------------------
# Standard errors: one entry point that propagates topic-estimation uncertainty
# ---------------------------------------------------------------------------
#
# Two routes (see issue #15). Method-of-composition (the cheap, principled default)
# draws theta from a model's own posterior and pools by Rubin's rules — it covers
# effects and prevalence on the models that *have* a posterior (logistic-normal
# for STM/CTM, Dirichlet for the Gibbs models). The bootstrap refits on resampled
# documents and is the only route for top-word/quality uncertainty and for the
# embedding models, at the cost of having to align topics across refits.

from dataclasses import dataclass  # noqa: E402


def _is_model(obj):
    """A fitted topica model (as opposed to a raw array)."""
    return hasattr(obj, "doc_topic") and not isinstance(obj, np.ndarray)


def model_family(model):
    """Which method-of-composition theta sampler suits ``model``.

    ``"logistic_normal"`` for STM/CTM (a variational ``eta`` posterior),
    ``"dirichlet"`` for the collapsed-Gibbs models (LDA, keyATM, SeededLDA, ...),
    or ``"none"`` for models with no posterior over theta (the embedding models),
    which need ``method="bootstrap"``.
    """
    # Check the class, not the instance: a PyO3 getter on an unfitted model raises
    # "not fitted" rather than being absent, so ``hasattr(model, ...)`` would lie.
    cls = type(model)
    if hasattr(cls, "eta_mean") and hasattr(cls, "eta_cov"):
        return "logistic_normal"
    if hasattr(cls, "alpha") and hasattr(cls, "doc_topic"):
        return "dirichlet"
    return "none"


def _doc_lengths_for(model, corpus):
    if corpus is None:
        # A fitted Gibbs model retains its own per-document lengths (issue #32),
        # so the Dirichlet fallback is self-sufficient without re-threading the
        # corpus. Older / other models that did not retain them still need one.
        own = getattr(model, "doc_lengths", None)
        if own is None or len(own) == 0:
            raise ValueError(
                "a Gibbs model needs corpus= (the Corpus it was fit on, or the "
                "token lists) to recover per-document lengths for Dirichlet theta "
                "draws; this model did not retain its own doc_lengths"
            )
        lengths = np.asarray(own, dtype=np.float64)
    elif hasattr(corpus, "doc_lengths"):
        lengths = np.asarray(corpus.doc_lengths, dtype=np.float64)
    else:
        lengths = np.asarray([len(d) for d in corpus], dtype=np.float64)
    d = np.asarray(model.doc_topic).shape[0]
    if lengths.shape[0] != d:
        raise ValueError(
            f"corpus has {lengths.shape[0]} documents but the model's doc_topic has "
            f"{d} rows; pass the same Corpus the model was fit on (pruning can drop "
            "documents — use that Corpus so lengths line up)"
        )
    return lengths


def _resample_draws(draws, nsims, seed):
    """Match a stored draw stack ``(S, D, K)`` to exactly ``nsims`` draws: return
    as-is when ``S == nsims``, take an evenly spaced subset when ``S > nsims``, or
    resample with replacement when ``S < nsims`` (Rubin's-rules pooling still sees
    only the ``S`` distinct draws, but the downstream shape is honored)."""
    draws = np.asarray(draws, dtype=np.float64)
    s = draws.shape[0]
    if s == nsims:
        return draws
    if s > nsims:
        idx = np.linspace(0, s - 1, nsims).round().astype(int)
        return draws[idx]
    rng = np.random.default_rng(seed)
    return draws[rng.integers(0, s, size=nsims)]


def composition_theta(model, corpus=None, *, nsims=25, seed=0):
    """Draw ``nsims`` theta matrices for method-of-composition, auto-selecting the
    sampler from the model family. Returns ``(nsims, num_docs, num_topics)``.

    For the Gibbs family this prefers a model's retained MCMC ``theta_draws`` (real
    cross-sweep posterior samples; issue #31) when present, which also means no
    ``corpus`` is needed. It falls back to :func:`dirichlet_theta_samples` (the
    within-document Dirichlet approximation, which does need ``corpus`` for the
    document lengths) when the model was fit with ``keep_theta_draws=False``."""
    fam = model_family(model)
    if fam == "logistic_normal":
        return posterior_theta_samples(model, nsims=nsims, seed=seed)
    if fam == "dirichlet":
        draws = getattr(model, "theta_draws", None)
        if draws is not None and len(draws):
            return _resample_draws(draws, nsims, seed)
        lengths = _doc_lengths_for(model, corpus)
        return dirichlet_theta_samples(
            np.asarray(model.doc_topic, dtype=np.float64), lengths, nsims=nsims, seed=seed
        )
    raise ValueError(
        f"{type(model).__name__} has no posterior over theta for "
        "method='composition' (no logistic-normal or Dirichlet structure). "
        "Use method='bootstrap' for standard errors on this model."
    )


def prevalence_ci(model, groups, *, ci=0.95, normalize=True, corpus=None,
                  nsims=None, seed=0, labels=None):
    """Per-group topic prevalence with posterior credible bands.

    Splits the documents by ``groups`` (one label per document) and, within each
    group, reports the mean topic prevalence with an empirical credible interval
    drawn from the model's posterior over theta. For each posterior draw and each
    group we average theta over the documents in that group, giving a
    ``(S, num_groups, num_topics)`` stack of per-draw group prevalences; the point
    estimate is the posterior mean over draws and the band is the empirical
    ``(1-ci)/2`` and ``(1+ci)/2`` quantiles.

    This is the draws-based companion to :func:`by_strata`. ``by_strata`` widens a
    descriptive interval by Rubin's rules (a normal approximation); ``prevalence_ci``
    reads the credible band straight off the posterior draws, which is what
    keyATM's ``plot_timetrend`` does. It is model-neutral: the draws come from
    :func:`composition_theta`, so it prefers a Gibbs model's retained MCMC
    ``theta_draws`` (pass ``keep_theta_draws=True`` at fit) and otherwise falls
    back to the Dirichlet approximation (Gibbs, needs ``corpus=``) or the
    logistic-normal posterior (STM/CTM). :func:`topica.time_prevalence_ci` is the
    dynamic-keyATM wrapper, with ``groups`` the timestamps and ``labels`` fixed to
    the model's ``time_labels``.

    Parameters
    ----------
    model
        A fitted model with a posterior over theta (any Dirichlet or
        logistic-normal model).
    groups
        One label per document; documents are pooled within each distinct value.
    ci
        Credible-interval coverage (default 0.95 gives a 95 percent band).
    normalize
        When ``True`` (default), each per-draw per-group prevalence row is rescaled
        to sum to 1 before the summary statistics, so it reads as a topic share.
    corpus
        The ``Corpus`` (or token lists) the model was fit on. Needed only when the
        model has no retained ``theta_draws`` and the Dirichlet fallback must
        recover document lengths.
    nsims
        Number of posterior draws. ``None`` (default) uses all retained MCMC draws
        as they are; an integer resamples to that many via :func:`composition_theta`.
    seed
        Seed for the draw sampler (used only on the resample / fallback paths).
    labels
        Optional explicit ordering of the group labels (matched to ``groups`` by
        string form). When omitted, groups are sorted by their string form.

    Returns
    -------
    dict
        ``labels`` (the group labels in row order), and ``mean``, ``ci_low``,
        ``ci_high``, ``sd`` each a ``(num_groups, num_topics)`` array.
    """
    draws = getattr(model, "theta_draws", None)
    if draws is not None and len(draws) and nsims is None:
        draws = np.asarray(draws, dtype=np.float64)          # use all retained draws
    else:
        draws = composition_theta(model, corpus, nsims=nsims or 25, seed=seed)
    if draws.ndim != 3:
        raise ValueError(
            f"theta draws must be 3-D (num_draws, num_docs, num_topics); got {draws.shape}"
        )
    s, d, k = draws.shape

    groups = np.asarray(groups)
    if groups.shape[0] != d:
        raise ValueError(
            f"groups has length {groups.shape[0]} but the model has {d} documents; "
            "groups must be one label per document"
        )

    group_str = np.asarray([str(v) for v in groups])
    if labels is None:
        uniq = sorted(np.unique(groups), key=lambda v: str(v))
        labels = [u.item() if hasattr(u, "item") else u for u in uniq]
    label_to_idx = {str(lbl): i for i, lbl in enumerate(labels)}
    missing = set(group_str) - set(label_to_idx)
    if missing:
        raise ValueError(f"groups contains values not in labels: {sorted(missing)}")
    idx = np.array([label_to_idx[g] for g in group_str], dtype=np.intp)

    per_draw = np.zeros((s, len(labels), k), dtype=np.float64)
    for t in range(len(labels)):
        mask = idx == t
        if mask.any():
            per_draw[:, t, :] = draws[:, mask, :].mean(axis=1)

    if normalize:
        row_sums = per_draw.sum(axis=2, keepdims=True)
        nz = row_sums > 0
        per_draw = np.where(nz, per_draw / np.where(nz, row_sums, 1.0), 0.0)

    lo = (1.0 - ci) / 2.0
    return {
        "labels": list(labels),
        "mean": per_draw.mean(axis=0),
        "ci_low": np.quantile(per_draw, lo, axis=0),
        "ci_high": np.quantile(per_draw, 1.0 - lo, axis=0),
        "sd": per_draw.std(axis=0, ddof=1) if s > 1 else np.zeros((len(labels), k)),
    }


@dataclass
class TopicPrevalence:
    """Mean prevalence of one topic with an uncertainty-propagated interval."""

    topic: int
    name: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    alignment_quality: float | None = None  # bootstrap only: mean Jaccard
    alignment_margin: float | None = None    # bootstrap only: match unambiguity
    reliable: bool = True

    def as_dict(self) -> dict:
        d = {
            "topic": self.topic,
            "name": self.name,
            "estimate": self.estimate,
            "se": self.se,
            "ci": (self.ci_low, self.ci_high),
            "reliable": self.reliable,
        }
        if self.alignment_quality is not None:
            d["alignment_quality"] = self.alignment_quality
            d["alignment_margin"] = self.alignment_margin
        return d


@dataclass
class TopWordUncertainty:
    """Per-topic top words with bootstrap inclusion probabilities."""

    topic: int
    name: str
    words: list  # (word, inclusion_prob, ci_low, ci_high)
    alignment_quality: float
    alignment_margin: float
    reliable: bool

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "name": self.name,
            "words": [
                {"word": w, "inclusion_prob": p, "ci": (lo, hi)}
                for (w, p, lo, hi) in self.words
            ],
            "alignment_quality": self.alignment_quality,
            "alignment_margin": self.alignment_margin,
            "reliable": self.reliable,
        }


def _resolve_design(X, formula, data):
    """Return ``(X, feature_names)`` from either a design matrix or a formula."""
    if formula is not None:
        if data is None:
            raise ValueError("formula= requires data= (a pandas DataFrame).")
        from .formulas import design_matrix

        return design_matrix(formula, data)
    if X is None:
        raise ValueError("of='effect' needs X (a design matrix) or formula= with data=.")
    return np.asarray(X, dtype=np.float64), None


def _top_word_strings(model, topn):
    """Per-topic ordered top-`topn` word strings, and the same as sets."""
    phi = np.asarray(model.topic_word, dtype=np.float64)
    vocab = list(model.vocabulary)
    lists = [[vocab[i] for i in np.argsort(phi[t])[::-1][:topn]] for t in range(phi.shape[0])]
    return lists, [set(w) for w in lists]


def _match_to_reference(ref_sets, boot_sets):
    """Hungarian-match a refit's topics to the reference by top-word Jaccard
    (vocabulary-independent). Returns ``(match[i]->j, quality, margin)``, where
    ``quality[i]`` is the Jaccard with the matched topic and ``margin[i]`` is how
    much better that match is than the next-best boot topic. A small margin means
    the match is ambiguous (e.g. topics that split/merge, or a reference whose
    topics are not distinct) even when the Jaccard itself looks high, so it is the
    reliable flag for unstable alignment."""
    from .validation import _hungarian

    k, kb = len(ref_sets), len(boot_sets)
    jac = np.zeros((k, kb))
    for i, rs in enumerate(ref_sets):
        for j, bs in enumerate(boot_sets):
            union = rs | bs
            jac[i, j] = (len(rs & bs) / len(union)) if union else 0.0
    cost = 1.0 - jac
    match, quality, margin = {}, [0.0] * k, [0.0] * k
    for i, j in _hungarian(cost):
        match[i] = j
        quality[i] = float(jac[i, j])
        if kb > 1:
            others = np.delete(jac[i], j)
            margin[i] = float(jac[i, j] - np.max(others))
        else:
            margin[i] = float(jac[i, j])
    return match, quality, margin


def _bootstrap_refits(model, docs, *, n_boot, topn, seed, model_factory, refit, **fit_kwargs):
    """Refit on `n_boot` document resamples, matching each refit's topics back to
    the reference model. Yields ``(picks, boot_model, match, jaccard)`` per
    resample. `refit(picks)->fitted model` overrides the default
    factory+fit path (use it for embedding models, where embeddings must be
    resampled alongside the documents)."""
    from ._topica import LDA

    k = np.asarray(model.topic_word).shape[0]
    _, ref_sets = _top_word_strings(model, topn)
    d = len(docs)
    if d < 2:
        raise ValueError("need at least two documents to resample")

    if refit is None:
        if model_factory is None:
            cls = type(model)

            def model_factory(s, _cls=cls, _k=k):  # noqa: ANN001
                try:
                    return _cls(num_topics=_k, seed=s)
                except TypeError as exc:
                    raise TypeError(
                        f"could not rebuild {_cls.__name__} for the bootstrap; pass "
                        "model_factory=callable(seed)->unfitted model, or "
                        "refit=callable(doc_indices)->fitted model (needed for "
                        "models whose fit takes embeddings)."
                    ) from exc

        def refit(picks, _b_seed=None):  # noqa: ANN001
            m = model_factory(_b_seed)
            m.fit([docs[i] for i in picks], **fit_kwargs)
            return m

    # Decide refit's arity once, by inspecting its signature, rather than calling
    # the 2-arg form and treating any TypeError as "this is a 1-arg hook". A
    # TypeError raised *inside* refit (a bad kwarg, a type error in the hook body)
    # would otherwise be misread as an arity mismatch and silently retried as
    # refit(picks) — running every resample at the default seed and returning SEs
    # computed from mis-seeded refits with no error or warning.
    try:
        takes_seed = True
        inspect.signature(refit).bind(np.empty(0), seed + 1)
    except TypeError:
        takes_seed = False

    rng = np.random.RandomState(seed)
    for b in range(n_boot):
        picks = rng.randint(0, d, size=d)
        boot = refit(picks, seed + b + 1) if takes_seed else refit(picks)
        _, boot_sets = _top_word_strings(boot, topn)
        match, quality, margin = _match_to_reference(ref_sets, boot_sets)
        yield picks, boot, match, quality, margin


def standard_errors(
    model,
    corpus=None,
    *,
    of="effect",
    method="composition",
    formula=None,
    data=None,
    X=None,
    feature_names=None,
    nsims=25,
    n_boot=200,
    topn=10,
    ci=0.95,
    seed=0,
    min_alignment=0.5,
    min_margin=0.1,
    model_factory=None,
    refit=None,
    **fit_kwargs,
):
    """Standard errors for the quantities people publish, with topic-estimation
    uncertainty propagated — one entry point across the model families (issue #15).

    Parameters
    ----------
    model : a fitted topica model.
    corpus : the ``Corpus`` (or token lists) the model was fit on. Required for
        ``method="composition"`` on a Gibbs model (for document lengths) and for
        ``method="bootstrap"`` (to resample documents).
    of : ``"effect"`` (covariate effects, needs ``formula``/``data`` or ``X``),
        ``"prevalence"`` (each topic's mean proportion), or ``"top_words"``
        (per-topic top-word stability; ``method="bootstrap"`` only).
    method : ``"composition"`` (default) draws theta from the model's posterior and
        pools by Rubin's rules — cheap, no refit, sound for effects/prevalence on
        STM/CTM/LDA/keyATM. ``"bootstrap"`` refits on resampled documents and
        aligns topics across refits — the only route for ``of="top_words"`` and for
        the embedding models, but it flags topics whose alignment is unstable.
    nsims : composition theta draws. n_boot : bootstrap resamples.
    min_alignment : a bootstrap topic whose mean top-word Jaccard with the
        reference falls below this is flagged ``reliable=False`` and its SE is
        suppressed (set to NaN), since a split/merge corrupts the estimate.
    min_margin : a topic is also flagged unreliable when its match is *ambiguous*
        — the best-matching refit topic is less than ``min_margin`` better (in
        Jaccard) than the next-best. This catches the case a high Jaccard misses:
        topics whose top words are not distinct, so the alignment is arbitrary.

    Returns
    -------
    ``of="effect"`` -> ``list[TopicEffect]`` (as :func:`estimate_effect`);
    ``of="prevalence"`` -> ``list[TopicPrevalence]``;
    ``of="top_words"`` -> ``list[TopWordUncertainty]``.
    """
    if of not in ("effect", "prevalence", "top_words"):
        raise ValueError("of must be 'effect', 'prevalence', or 'top_words'")
    if method not in ("composition", "bootstrap"):
        raise ValueError("method must be 'composition' or 'bootstrap'")
    if method == "composition" and of == "top_words":
        raise ValueError(
            "of='top_words' needs method='bootstrap'; the composition method only "
            "propagates theta (document-topic) uncertainty, not topic-word uncertainty."
        )

    z = _z_for(ci)

    if method == "composition":
        if of == "effect":
            Xm, fnames = _resolve_design(X, formula, data)
            draws = composition_theta(model, corpus, nsims=nsims, seed=seed)
            return estimate_effect(
                draws, X=Xm, feature_names=feature_names or fnames, ci=ci
            )
        # of == "prevalence"
        draws = composition_theta(model, corpus, nsims=nsims, seed=seed)
        return _prevalence_composition(draws, _topic_names(model, draws.shape[2]), z)

    names = _topic_names(model, np.asarray(model.topic_word).shape[0])

    # method == "bootstrap"
    docs = corpus.documents() if hasattr(corpus, "documents") else corpus
    if docs is None:
        raise ValueError("method='bootstrap' needs corpus= (a Corpus or token lists)")
    docs = [list(d) for d in docs]
    if of == "effect":
        Xm, fnames = _resolve_design(X, formula, data)
        return _effect_bootstrap(
            model, docs, Xm, feature_names or fnames, names, z, n_boot, topn, seed,
            min_alignment, min_margin, model_factory, refit, **fit_kwargs,
        )
    if of == "prevalence":
        return _prevalence_bootstrap(
            model, docs, names, z, n_boot, topn, seed, min_alignment, min_margin,
            model_factory, refit, **fit_kwargs,
        )
    return _top_words_bootstrap(
        model, docs, names, z, n_boot, topn, seed, min_alignment, min_margin,
        model_factory, refit, **fit_kwargs,
    )


def _z_for(ci):
    from .stm import _normal_ppf

    return _normal_ppf(0.5 + ci / 2.0)


def _topic_names(model, k):
    return list(getattr(model, "topic_names", [])) or [f"topic_{t}" for t in range(k)]


def _prevalence_composition(draws, names, z):
    draws = np.asarray(draws, dtype=np.float64)  # (M, D, K)
    m, d, _ = draws.shape
    per_draw = draws.mean(axis=1)  # (M, K) prevalence per draw
    estimate = per_draw.mean(axis=0)
    between = per_draw.var(axis=0, ddof=1) if m > 1 else np.zeros_like(estimate)
    within = draws.var(axis=1, ddof=1).mean(axis=0) / d  # mean sampling var of the mean
    total = within + (1.0 + 1.0 / m) * between
    se = np.sqrt(np.clip(total, 0.0, None))
    return [
        TopicPrevalence(t, names[t], float(estimate[t]), float(se[t]),
                        float(estimate[t] - z * se[t]), float(estimate[t] + z * se[t]))
        for t in range(len(names))
    ]


def _aligned_theta(boot, match, k):
    """Reorder a refit's doc_topic columns into the reference topic order."""
    theta = np.asarray(boot.doc_topic, dtype=np.float64)
    out = np.full((theta.shape[0], k), np.nan)
    for i in range(k):
        j = match.get(i)
        if j is not None and j < theta.shape[1]:
            out[:, i] = theta[:, j]
    return out


def _prevalence_bootstrap(model, docs, names, z, n_boot, topn, seed, min_alignment,
                          min_margin, model_factory, refit, **fit_kwargs):
    k = len(names)
    ref_prev = np.asarray(model.doc_topic, dtype=np.float64).mean(axis=0)
    samples = [[] for _ in range(k)]
    quals, margins = [[] for _ in range(k)], [[] for _ in range(k)]
    for picks, boot, match, quality, margin in _bootstrap_refits(
        model, docs, n_boot=n_boot, topn=topn, seed=seed,
        model_factory=model_factory, refit=refit, **fit_kwargs,
    ):
        th = _aligned_theta(boot, match, k)
        col = th.mean(axis=0)
        for i in range(k):
            if not np.isnan(col[i]):
                samples[i].append(col[i])
            quals[i].append(quality[i])
            margins[i].append(margin[i])
    out = []
    for i in range(k):
        q, mg, reliable = _reliability(quals[i], margins[i], min_alignment, min_margin, len(samples[i]))
        se = float(np.std(samples[i], ddof=1)) if reliable else float("nan")
        est = float(ref_prev[i])
        lo = est - z * se if reliable else float("nan")
        hi = est + z * se if reliable else float("nan")
        out.append(TopicPrevalence(i, names[i], est, se, lo, hi, q, mg, reliable))
    return out


def _reliability(quals, margins, min_alignment, min_margin, n_samples):
    """Mean Jaccard, mean match margin, and whether a bootstrap topic's SE can be
    trusted: the match must be both close (Jaccard) and unambiguous (margin), with
    at least two usable resamples."""
    q = float(np.mean(quals)) if quals else 0.0
    mg = float(np.mean(margins)) if margins else 0.0
    reliable = q >= min_alignment and mg >= min_margin and n_samples >= 2
    return q, mg, reliable


def _top_words_bootstrap(model, docs, names, z, n_boot, topn, seed, min_alignment,
                         min_margin, model_factory, refit, **fit_kwargs):
    k = len(names)
    ref_lists, _ = _top_word_strings(model, topn)
    # inclusion[i][w] = count of resamples whose matched topic keeps word w in its top-topn
    counts = [{w: 0 for w in ref_lists[i]} for i in range(k)]
    quals, margins = [[] for _ in range(k)], [[] for _ in range(k)]
    n_used = 0
    for picks, boot, match, quality, margin in _bootstrap_refits(
        model, docs, n_boot=n_boot, topn=topn, seed=seed,
        model_factory=model_factory, refit=refit, **fit_kwargs,
    ):
        boot_lists, _ = _top_word_strings(boot, topn)
        n_used += 1
        for i in range(k):
            quals[i].append(quality[i])
            margins[i].append(margin[i])
            j = match.get(i)
            bset = set(boot_lists[j]) if j is not None and j < len(boot_lists) else set()
            for w in ref_lists[i]:
                if w in bset:
                    counts[i][w] += 1
    out = []
    b = max(n_used, 1)
    for i in range(k):
        q, mg, reliable = _reliability(quals[i], margins[i], min_alignment, min_margin, b)
        words = []
        for w in ref_lists[i]:
            p = counts[i][w] / b
            se = (p * (1.0 - p) / b) ** 0.5
            lo, hi = max(0.0, p - z * se), min(1.0, p + z * se)
            words.append((w, float(p), float(lo), float(hi)))
        out.append(TopWordUncertainty(i, names[i], words, q, mg, reliable))
    return out


def _effect_bootstrap(model, docs, X, feature_names, names, z, n_boot, topn, seed,
                      min_alignment, min_margin, model_factory, refit, **fit_kwargs):
    from .stm import TopicEffect

    k = len(names)
    X = np.asarray(X, dtype=np.float64)
    # Reference coefficients from the full fit (point theta, OLS). Only the point
    # estimates are used here; the SE comes from the bootstrap spread, so the CI
    # level passed to estimate_effect is irrelevant.
    ref = {e.topic: e for e in estimate_effect(
        np.asarray(model.doc_topic, dtype=np.float64), X=X, feature_names=feature_names,
    )}
    fnames = ref[0].feature_names
    p = len(fnames)
    coefs = [[] for _ in range(k)]  # per topic, list of coef vectors
    quals, margins = [[] for _ in range(k)], [[] for _ in range(k)]
    for picks, boot, match, quality, margin in _bootstrap_refits(
        model, docs, n_boot=n_boot, topn=topn, seed=seed,
        model_factory=model_factory, refit=refit, **fit_kwargs,
    ):
        for i in range(k):
            quals[i].append(quality[i])
            margins[i].append(margin[i])
        th = _aligned_theta(boot, match, k)
        if th.shape[0] != X.shape[0]:
            continue  # a refit that dropped documents would misalign X; skip it
        Xb = X[picks]
        eff = {e.topic: e for e in estimate_effect(th, X=Xb, feature_names=feature_names)}
        for i in range(k):
            e = eff.get(i)
            if e is not None and not np.any(np.isnan(e.coef)):
                coefs[i].append(e.coef)
    out = []
    for i in range(k):
        _, _, reliable = _reliability(quals[i], margins[i], min_alignment, min_margin, len(coefs[i]))
        est = ref[i].coef
        se = np.std(np.vstack(coefs[i]), axis=0, ddof=1) if reliable else np.full(p, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = est / se
        out.append(TopicEffect(
            topic=i, feature_names=fnames, coef=est, se=se, z=zz,
            ci_low=est - z * se, ci_high=est + z * se, r_squared=float("nan"),
        ))
    return out


# ---------------------------------------------------------------------------
# Permutation test for a single binary prevalence covariate (issue #36)
# ---------------------------------------------------------------------------

@dataclass
class PermutationResult:
    """Result of :func:`permutation_test` for one topic.

    Attributes
    ----------
    topic : int
        Topic index (aligned to the reference model).
    topic_name : str
        Topic label (or ``"topic_t"`` when no labels are set).
    observed : float
        Observed difference in mean prevalence between the two covariate groups.
    null : numpy.ndarray, shape (n_perm,)
        Per-permutation covariate effects (the null distribution).
    pvalue : float
        Two-sided p-value: proportion of permutations whose absolute effect
        equals or exceeds the absolute observed effect.
    """

    topic: int
    topic_name: str
    observed: float
    null: np.ndarray
    pvalue: float

    def as_dict(self) -> dict:
        """Return a plain-dict summary (omits the full null array)."""
        return {
            "topic": self.topic,
            "topic_name": self.topic_name,
            "observed": self.observed,
            "pvalue": self.pvalue,
            "null_mean": float(self.null.mean()),
            "null_std": float(self.null.std(ddof=1)) if len(self.null) > 1 else float("nan"),
        }


def _binary_covariate_effect(theta, covariate):
    """Mean prevalence difference (group1 - group0) for each topic column.

    Parameters
    ----------
    theta : array (num_docs, num_topics)
    covariate : array (num_docs,), binary 0/1

    Returns
    -------
    array (num_topics,) — effect per topic
    """
    c = np.asarray(covariate, dtype=np.float64)
    mask1 = c == 1.0
    mask0 = c == 0.0
    if not mask1.any() or not mask0.any():
        return np.zeros(theta.shape[1])
    return theta[mask1].mean(axis=0) - theta[mask0].mean(axis=0)


def _covariate_kwarg_for(model):
    """Return the keyword name used to pass a prevalence covariate to model.fit().

    Returns a string (the kwarg name) for model families that accept a
    covariate at fit time, or None for covariate-free models (plain LDA, HDP,
    etc.) that should be fitted without one.

    The mapping mirrors each model family's public API:
      - STM / STS: second positional arg named ``prevalence``
      - DMR: second positional arg named ``features``
      - KeyATM: keyword arg ``covariates``
    """
    cls_name = type(model).__name__
    if cls_name in ("STM", "STS"):
        return "prevalence"
    if cls_name == "DMR":
        return "features"
    if cls_name == "KeyATM":
        return "covariates"
    # For any other model, inspect fit() for known covariate parameter names.
    fit_fn = getattr(type(model), "fit", None)
    if fit_fn is not None:
        try:
            params = inspect.signature(fit_fn).parameters
            for name in ("prevalence", "features", "covariates"):
                if name in params:
                    return name
        except (ValueError, TypeError):
            pass
    return None


def permutation_test(
    model,
    corpus,
    covariate,
    *,
    n_perm=100,
    topics=None,
    topn=10,
    seed=0,
    model_factory=None,
    iters=None,
):
    """Permutation test for a binary prevalence covariate (R ``stm``'s ``permutationTest``).

    Assesses whether a binary document-level covariate genuinely shifts topic
    prevalence, or whether an apparent association could arise by chance. Each
    permutation randomly reassigns the covariate at its empirical rate, refits
    the model from fresh starting values (passing the permuted covariate to the
    model for covariate-aware families), aligns the refit topics to the
    reference (using the Hungarian top-word matcher from
    :func:`~topica.validation._hungarian`), and records the covariate's effect
    on every topic. The observed effect for each topic is then compared to the
    permutation null to compute a two-sided p-value.

    Parameters
    ----------
    model : a fitted topica model.
        The reference fit. Its type is used to build each permutation refit
        (``type(model)(num_topics=K, seed=s)``), unless ``model_factory`` is
        given. The model must expose ``doc_topic``, ``topic_word``, and
        ``vocabulary``.
    corpus : list of token lists or a ``Corpus``.
        The documents the model was fit on. Each permutation refits on the
        same documents with a shuffled covariate.
    covariate : array-like (num_docs,), binary (0/1 or True/False).
        The binary prevalence covariate to test. Must have exactly two unique
        values; they are mapped to 0 and 1 in sorted order.
    n_perm : int
        Number of permutation refits. Higher values give more stable p-values;
        100 is enough for a screening test, 500 for publication.
    topics : sequence of int, optional
        Restrict the output to these topic indices. Defaults to all topics.
    topn : int
        Top-word count used for topic alignment across refits.
    seed : int
        Master RNG seed. Permutation seeds are derived as ``seed + perm_index``.
    model_factory : callable(seed) -> unfitted model, optional
        Override the default ``type(model)(num_topics=K, seed=s)`` builder. Use
        this when the model's constructor needs extra arguments (e.g. keyword
        lists for KeyATM, or content covariates for STM).
    iters : int, optional
        Iterations for each permutation refit. When ``None`` the refit's default
        iteration count is used. Pass a smaller value (e.g. ``iters=100``) to
        speed up screening tests.

    Returns
    -------
    list of :class:`PermutationResult`
        One entry per topic (restricted to ``topics`` when given), each with
        the observed effect, the permutation null distribution, and a two-sided
        p-value.

    Notes
    -----
    For covariate-aware models (STM, DMR, KeyATM) the permuted covariate is
    passed directly to each refit so the null model is correctly specified
    (matching R ``stm``'s ``permutationTest`` behaviour). For covariate-free
    models (LDA, HDP, etc.) permuting the labels used only in the effect
    statistic remains a valid null.

    The p-value uses the ``(1 + count) / (1 + n_perm)`` convention, so it is
    never exactly zero. Permutation statistics that are NaN (from unmatched
    topics in variable-K refits such as HDP) are dropped from the null before
    computing the p-value; the effective n is reduced accordingly.
    """
    # --- resolve corpus to token lists ----------------------------------------
    if hasattr(corpus, "documents"):
        docs = corpus.documents()
    else:
        docs = [list(d) for d in corpus]

    n_docs = len(docs)
    theta_ref = np.asarray(model.doc_topic, dtype=np.float64)
    if theta_ref.shape[0] != n_docs:
        raise ValueError(
            f"corpus has {n_docs} documents but model.doc_topic has "
            f"{theta_ref.shape[0]} rows; pass the same documents the model "
            "was fit on"
        )

    # --- validate and normalise the binary covariate --------------------------
    cov_raw = np.asarray(covariate)
    if cov_raw.shape != (n_docs,):
        raise ValueError(
            f"covariate must have shape ({n_docs},); got {cov_raw.shape}"
        )
    unique_vals = np.unique(cov_raw)
    if len(unique_vals) != 2:
        raise ValueError(
            "covariate must be binary (exactly two unique values); "
            f"got {len(unique_vals)}: {unique_vals.tolist()}"
        )
    # Map to 0/1 regardless of the input encoding (bool, {0,1}, {1,2}, …).
    cov = np.where(cov_raw == unique_vals[0], 0.0, 1.0)

    k = theta_ref.shape[1]
    topic_list = list(range(k)) if topics is None else list(topics)
    names = _topic_names(model, k)

    # --- top-word sets for the reference model (for alignment) ----------------
    _, ref_sets = _top_word_strings(model, topn)

    # --- observed effect -------------------------------------------------------
    obs_all = _binary_covariate_effect(theta_ref, cov)  # (k,)

    # --- detect covariate kwarg for this model family -------------------------
    # For covariate-aware models (STM, DMR, KeyATM) the permuted covariate is
    # passed to each refit so the null is correctly specified.  For plain LDA
    # and other covariate-free models this is None and we skip that argument.
    _cov_kwarg = _covariate_kwarg_for(model)

    # --- build the model factory and fit kwargs --------------------------------
    if model_factory is None:
        cls = type(model)

        def model_factory(s, _cls=cls, _k=k):
            try:
                return _cls(num_topics=_k, seed=s)
            except TypeError as exc:
                raise TypeError(
                    f"could not rebuild {_cls.__name__} for the permutation test; "
                    "pass model_factory=callable(seed)->unfitted model"
                ) from exc

    fit_kwargs = {}
    if iters is not None:
        fit_kwargs["iters"] = iters

    # --- permutation loop -----------------------------------------------------
    rng = np.random.RandomState(seed)
    rate = float(cov.mean())  # empirical rate of the positive class

    # null_effects[i] holds one float per permutation for topic i
    null_effects = {t: [] for t in topic_list}

    for perm in range(n_perm):
        perm_seed = seed + perm + 1
        cov_perm = (rng.random(n_docs) < rate).astype(np.float64)

        m_perm = model_factory(perm_seed)
        if _cov_kwarg is not None:
            # Covariate-aware model: pass the permuted covariate so the null
            # is correctly specified (mirrors R stm::permutationTest).
            m_perm.fit(docs, **{_cov_kwarg: cov_perm[:, None], **fit_kwargs})
        else:
            m_perm.fit(docs, **fit_kwargs)

        theta_perm = np.asarray(m_perm.doc_topic, dtype=np.float64)

        # Align permutation topics to the reference.
        _, perm_sets = _top_word_strings(m_perm, topn)
        match, _, _ = _match_to_reference(ref_sets, perm_sets)

        # Reorder perm theta columns into reference topic order.
        theta_aligned = np.full((n_docs, k), np.nan)
        for ref_t in range(k):
            perm_t = match.get(ref_t)
            if perm_t is not None and perm_t < theta_perm.shape[1]:
                theta_aligned[:, ref_t] = theta_perm[:, perm_t]

        # Compute effect for this permutation.
        eff_perm = _binary_covariate_effect(theta_aligned, cov_perm)

        for t in topic_list:
            null_effects[t].append(float(eff_perm[t]))

    # --- assemble results ------------------------------------------------------
    out = []
    for t in topic_list:
        obs = float(obs_all[t])
        null = np.array(null_effects[t], dtype=np.float64)
        # Drop NaN entries (unmatched topics from variable-K models like HDP).
        null_valid = null[~np.isnan(null)]
        # Two-sided p-value using the (1 + count) / (1 + n) convention so
        # p is never exactly zero regardless of how extreme the observed stat is.
        n_valid = null_valid.size
        if n_valid > 0:
            pval = float((1 + np.sum(np.abs(null_valid) >= abs(obs))) / (1 + n_valid))
        else:
            pval = float("nan")
        out.append(PermutationResult(
            topic=t,
            topic_name=names[t] if t < len(names) else f"topic_{t}",
            observed=obs,
            null=null,
            pvalue=pval,
        ))
    return out
