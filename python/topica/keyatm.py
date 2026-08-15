"""keyATM-specific workflow helpers, mirroring the R ``keyATM`` package.

The model-agnostic analyses already live in :mod:`topica.diagnostics` and work
on a fitted :class:`~topica.KeyATM`'s numpy outputs, so they cover most of the R
workflow directly:

- ``keyATM::top_words``       -> :meth:`topica.KeyATM.top_words`
- ``keyATM::top_docs``        -> :func:`topica.find_thoughts`
- ``keyATM::semantic_coherence`` -> :meth:`topica.KeyATM.coherence`
- ``keyATM::plot_modelfit``   -> :attr:`topica.KeyATM.log_likelihood_history`
- ``keyATM::covariates_info`` -> :attr:`topica.KeyATM.feature_effects` / ``feature_names``
- ``estimateEffect``-style    -> :func:`topica.stm.estimate_effect`

This module adds the keyATM-flavored pieces that operate on the keywords and the
document-topic matrix:

- :func:`top_topics`        ~ ``keyATM::top_topics``       (top topics per document)
- :func:`by_strata`         ~ ``keyATM::by_strata_DocTopic`` (covariate-stratified prevalence)
- :func:`visualize_keywords` ~ ``keyATM::visualize_keywords`` (keyword corpus frequencies)
- :func:`refine_keywords`   ~ ``keyATM::refine_keywords``   (drop too-rare keywords)
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np


def _theta_and_names(model_or_theta, topic_names=None):
    """Accept either a fitted model (with ``doc_topic``/``topic_names``) or a raw
    theta array, returning ``(theta, names)``."""
    if hasattr(model_or_theta, "doc_topic"):
        theta = np.asarray(model_or_theta.doc_topic, dtype=np.float64)
        if topic_names is None:
            topic_names = list(getattr(model_or_theta, "topic_names", []))
    else:
        theta = np.asarray(model_or_theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("doc_topic must be 2-D (num_docs, num_topics)")
    k = theta.shape[1]
    if not topic_names:
        topic_names = [f"topic_{t}" for t in range(k)]
    if len(topic_names) != k:
        raise ValueError(f"topic_names has {len(topic_names)} entries but theta has {k} topics")
    return theta, list(topic_names)


def top_topics(model_or_theta, *, n=2, topic_names=None):
    """The ``n`` most prevalent topics in each document (≈ ``keyATM::top_topics``).

    Returns a list (one per document) of ``(topic_name, proportion)`` pairs,
    sorted by descending document-topic proportion. Pass a fitted
    :class:`~topica.KeyATM` (topic names are taken from it) or a raw ``theta``
    array.
    """
    theta, names = _theta_and_names(model_or_theta, topic_names)
    if n < 1:
        raise ValueError("n must be >= 1")
    n = min(n, theta.shape[1])
    out = []
    for row in theta:
        idx = np.argsort(row)[::-1][:n]
        out.append([(names[t], float(row[t])) for t in idx])
    return out


@dataclass
class StrataPrevalence:
    """Mean topic prevalence within one covariate stratum, with intervals."""

    stratum: object
    n: int
    topic_names: list
    mean: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray

    def as_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "n": self.n,
            **{
                name: {
                    "mean": float(self.mean[t]),
                    "ci": (float(self.ci_low[t]), float(self.ci_high[t])),
                }
                for t, name in enumerate(self.topic_names)
            },
        }


def by_strata(model_or_theta, strata, *, ci=0.95, topic_names=None, corpus=None, nsims=None, seed=0):
    """Mean topic prevalence within each level of a document covariate
    (≈ ``keyATM::by_strata_DocTopic``).

    Splits documents by their value in ``strata`` (one label per document) and,
    for each level, reports the mean of each topic's proportion with a
    normal-approximation confidence interval on that mean. This is keyATM's
    descriptive answer to "how does topic prevalence differ across groups".

    With ``nsims`` (and a fitted **model** as the first argument), the interval is
    widened by the **method of composition**: the model's θ posterior is drawn for
    you (logistic-normal for STM/CTM, Dirichlet for the Gibbs models — pass
    ``corpus=`` so document lengths are available) and the per-stratum means are
    pooled by Rubin's rules, so the topic-estimation uncertainty is propagated, not
    just the across-document spread. For a regression with the same propagation use
    :func:`topica.stm.estimate_effect`.

    Returns a list of :class:`StrataPrevalence`, one per unique stratum (sorted).
    ``[s.as_dict() for s in result]`` builds a table.
    """
    from .stm import _normal_ppf

    z = _normal_ppf(0.5 + ci / 2.0)
    strata = np.asarray(strata)

    if nsims:
        from .effects import composition_theta

        if not hasattr(model_or_theta, "doc_topic"):
            raise ValueError("by_strata(..., nsims=) needs a fitted model to draw theta")
        names = list(getattr(model_or_theta, "topic_names", [])) or None
        draws = composition_theta(model_or_theta, corpus, nsims=nsims, seed=seed)
        m, d, k = draws.shape
        names = names or [f"topic_{t}" for t in range(k)]
        if strata.shape[0] != d:
            raise ValueError("strata must have one label per document")
        out = []
        for level in sorted(np.unique(strata), key=lambda v: str(v)):
            mask = strata == level
            n = int(mask.sum())
            sub = draws[:, mask, :]                       # (M, n, K)
            per_draw = sub.mean(axis=1)                   # (M, K)
            estimate = per_draw.mean(axis=0)
            between = per_draw.var(axis=0, ddof=1) if m > 1 else np.zeros(k)
            within = sub.var(axis=1, ddof=1).mean(axis=0) / n if n > 1 else np.zeros(k)
            se = np.sqrt(np.clip(within + (1.0 + 1.0 / m) * between, 0.0, None))
            out.append(
                StrataPrevalence(
                    stratum=level.item() if hasattr(level, "item") else level,
                    n=n,
                    topic_names=names,
                    mean=estimate,
                    ci_low=np.clip(estimate - z * se, 0.0, 1.0),
                    ci_high=np.clip(estimate + z * se, 0.0, 1.0),
                )
            )
        return out

    theta, names = _theta_and_names(model_or_theta, topic_names)
    if strata.shape[0] != theta.shape[0]:
        raise ValueError("strata must have one label per document")

    out = []
    for level in sorted(np.unique(strata), key=lambda v: str(v)):
        rows = theta[strata == level]
        n = rows.shape[0]
        mean = rows.mean(axis=0)
        # Standard error of the mean per topic (0 when a single document).
        se = rows.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        out.append(
            StrataPrevalence(
                stratum=level.item() if hasattr(level, "item") else level,
                n=int(n),
                topic_names=names,
                mean=mean,
                ci_low=np.clip(mean - z * se, 0.0, 1.0),
                ci_high=np.clip(mean + z * se, 0.0, 1.0),
            )
        )
    return out


def _docs_and_vocab(docs):
    """Token lists plus the known vocabulary set (or ``None``).

    Accepts either raw token lists or a built :class:`~topica.Corpus`. For a
    Corpus we read its *pruned* token lists (``documents()``) and vocabulary, so
    keyword diagnostics reflect exactly what ``fit`` will see — a substantive seed
    word dropped by ``rm_top`` / ``min_doc_freq`` / ``max_doc_fraction`` then
    shows up as absent, instead of being reported from the raw pre-pruning counts
    (issue #743).
    """
    if hasattr(docs, "documents") and hasattr(docs, "vocabulary"):
        return docs.documents(), set(docs.vocabulary)
    return docs, None


def _corpus_counts(docs):
    """(per-word corpus count, per-word document frequency, total tokens)."""
    token_lists, _ = _docs_and_vocab(docs)
    counts = Counter()
    doc_freq = Counter()
    total = 0
    for d in token_lists:
        counts.update(d)
        doc_freq.update(set(d))
        total += len(d)
    return counts, doc_freq, total


def visualize_keywords(docs, keywords):
    """Corpus frequency of each keyword (≈ ``keyATM::visualize_keywords``).

    For every keyword in every set, reports how common it is in ``docs`` so you
    can catch keywords that are too rare to anchor a topic or so frequent they
    dominate it — the diagnostic keyATM asks you to run *before* fitting.

    Pass the **built** :class:`~topica.Corpus` you will fit, not the raw token
    lists: frequency pruning (``rm_top``, ``min_doc_freq``, ``max_doc_fraction``)
    can drop a substantive seed word from the vocabulary, and ``fit`` then
    silently ignores it. Scoring against the corpus reflects what ``fit`` sees, so
    a pruned seed shows ``count == 0`` and is flagged; scoring against raw docs
    reports the pre-pruning counts and can disagree with the fit (issue #743).

    Returns a :class:`~topica._results.KeywordDiagnostics` (a ``dict``) mapping
    each keyword-set name to a list of dicts
    ``{"keyword", "count", "proportion", "doc_freq", "in_vocab"}`` sorted by
    descending proportion, where ``proportion`` is the keyword's share of all
    corpus tokens, ``doc_freq`` is the number of documents containing it, and
    ``in_vocab`` is ``False`` for a keyword ``fit`` will not see (count 0). Call
    ``.to_frame()`` for one long DataFrame across all sets (a leading ``set``
    column names each row's keyword set). When a Corpus is passed, keywords absent
    from its vocabulary raise a warning.
    """
    from ._results import KeywordDiagnostics

    counts, doc_freq, total = _corpus_counts(docs)
    _, vocab = _docs_and_vocab(docs)
    total = max(total, 1)
    out = KeywordDiagnostics()
    missing = {}
    for name, words in keywords.items():
        rows = [
            {
                "keyword": w,
                "count": int(counts.get(w, 0)),
                "proportion": counts.get(w, 0) / total,
                "doc_freq": int(doc_freq.get(w, 0)),
                "in_vocab": counts.get(w, 0) > 0,
            }
            for w in words
        ]
        rows.sort(key=lambda r: r["proportion"], reverse=True)
        out[name] = rows
        gone = [w for w in words if counts.get(w, 0) == 0]
        if gone:
            missing[name] = gone
    if missing and vocab is not None:
        # A Corpus was passed, so absent keywords are ones the built vocabulary
        # does not contain — pruned or out-of-vocabulary — and fit will drop them.
        detail = "; ".join(f"{name}: {kw}" for name, kw in missing.items())
        warnings.warn(
            "visualize_keywords: these keywords are not in the corpus vocabulary "
            f"and fit will ignore them (pruned by rm_top/min_doc_freq or "
            f"out-of-vocabulary): {detail}. Adjust the pruning or the seed list.",
            UserWarning,
            stacklevel=2,
        )
    return out


def time_prevalence_ci(model, timestamps, *, ci=0.95, normalize=True):
    """Per-period topic prevalence with credible intervals from the dynamic keyATM posterior.

    .. note::
       This is **dynamic-keyATM only** — it reads that model's retained MCMC draws.
       For prevalence over a time covariate on any other model (LDA, STM, …), use
       :func:`topica.predicted_prevalence` with a continuous ``year`` (optionally a
       :func:`topica.spline`), which gives the same over-time curve with intervals
       via the method of composition.

    For a dynamic :class:`~topica.KeyATM` (fit with ``timestamps=`` and
    ``keep_theta_draws=True``), this computes per-period prevalence uncertainty
    directly from the retained MCMC ``theta_draws``. For each posterior draw and
    each time period, the per-draw average of theta over the documents in that
    period is computed, giving a (S, T, K) array of per-draw period-level
    prevalences. The point estimate is the posterior mean over draws; ``ci_low``
    and ``ci_high`` are the empirical (1-ci)/2 and (1+ci)/2 quantiles; ``sd`` is
    the posterior standard deviation.

    The periods are ordered to match ``model.time_labels`` exactly, so the result
    aligns with ``model.time_prevalence``.

    Parameters
    ----------
    model
        A fitted dynamic :class:`~topica.KeyATM` with non-empty ``time_labels``
        and non-``None`` ``theta_draws``. Refit with ``keep_theta_draws=True``
        (the default) if draws are absent.
    timestamps
        One value per document — the same array passed to ``fit``.
    ci
        Credible interval coverage (default 0.95 gives a 95 percent interval).
    normalize
        When ``True`` (default), each per-draw per-period prevalence row is
        normalized to sum to 1 before computing the summary statistics.

    Returns
    -------
    :class:`~topica._results.TimePrevalenceCI` (a ``dict``) with keys:
        - ``labels``: list of period labels (equals ``model.time_labels``)
        - ``mean``: ndarray shape (T, K), posterior mean prevalence per period
        - ``ci_low``: ndarray shape (T, K), lower credible bound
        - ``ci_high``: ndarray shape (T, K), upper credible bound
        - ``sd``: ndarray shape (T, K), posterior standard deviation

    Call ``.to_frame()`` for a long tidy DataFrame, one row per (period, topic).
    """
    time_labels = list(getattr(model, "time_labels", []))
    if not time_labels:
        raise ValueError(
            "model does not have time_labels: this helper requires a dynamic KeyATM "
            "(fit with timestamps= and num_states=)"
        )
    theta_draws = getattr(model, "theta_draws", None)
    if theta_draws is None:
        raise ValueError(
            "model.theta_draws is None: refit with keep_theta_draws=True "
            "(the default) to enable per-period posterior credible intervals"
        )
    d = np.asarray(theta_draws).shape[1]
    if len(np.asarray(timestamps)) != d:
        raise ValueError(
            f"timestamps has length {len(np.asarray(timestamps))} but theta_draws has "
            f"{d} documents; timestamps must be one value per document"
        )

    # The aggregation is model-neutral: group the retained theta draws by period
    # and read posterior quantiles off them. The only keyATM-specific parts are
    # requiring the HMM's own draws (above) and pinning the period order to
    # time_labels, so this is a thin wrapper over the general primitive.
    from .effects import prevalence_ci
    from ._results import TimePrevalenceCI

    return TimePrevalenceCI(
        prevalence_ci(
            model, timestamps, ci=ci, normalize=normalize, labels=time_labels
        )
    )


def refine_keywords(docs, keywords, *, min_count=2, min_doc_freq=1, verbose=False):
    """Drop keywords too rare to anchor a topic (≈ ``keyATM::refine_keywords``).

    Removes any keyword whose corpus count is below ``min_count`` or whose
    document frequency is below ``min_doc_freq`` (so out-of-vocabulary keywords,
    with count 0, always go). Keyword sets that end up empty are dropped, since
    a keyword topic needs at least one surviving keyword.

    As with :func:`visualize_keywords`, pass the **built** :class:`~topica.Corpus`
    you will fit rather than the raw token lists, so a seed word removed by the
    corpus's frequency pruning (``rm_top`` / ``min_doc_freq`` / ``max_doc_fraction``)
    is refined out here too, instead of surviving on its raw pre-pruning count and
    then being silently dropped by ``fit`` (issue #743).

    Returns ``(refined, dropped)`` where ``refined`` is the cleaned keyword dict
    and ``dropped`` maps each set name to the list of removed keywords. Set
    ``verbose=True`` to print a short report.
    """
    counts, doc_freq, _ = _corpus_counts(docs)
    refined, dropped = {}, {}
    for name, words in keywords.items():
        keep, drop = [], []
        for w in words:
            if counts.get(w, 0) >= min_count and doc_freq.get(w, 0) >= min_doc_freq:
                keep.append(w)
            else:
                drop.append(w)
        if drop:
            dropped[name] = drop
        if keep:
            refined[name] = keep
        if verbose and drop:
            print(f"  {name}: dropped {drop} (below threshold)")
    if verbose:
        gone = [n for n in keywords if n not in refined]
        if gone:
            print(f"  removed empty keyword sets: {gone}")
    return refined, dropped
