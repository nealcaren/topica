"""A model-neutral analysis surface for any fitted topica model.

These helpers read only a fitted model's public attributes — ``topic_word``
(K x V), ``doc_topic`` (D x K), ``topic_names``, ``vocabulary``, ``num_topics``,
and the optional ``labels`` (hard document assignments on the embedding-cluster
models, where ``-1`` marks a noise/outlier document) — so they work uniformly
across LDA, STM, CTM, keyATM, Top2Vec, BERTopic, and the rest. SAGE's
``(K, G, V)`` topic-word is reduced to its group marginal; DTM (time-sliced
``topic_word(time)``) and HLDA (a topic tree, no ``doc_topic``) do not present
this static surface and are not supported here. The goal is the
overview a researcher reaches for first: how big each topic is, what it is about,
who its representative documents are, and how its prevalence moves across time or
across groups.

- :func:`topic_info` — one summary row per topic (the headline table).
- :func:`topic_sizes` — hard size and expected mass per topic.
- :func:`topic_labels` / :func:`set_topic_labels` — effective topic labels,
  with a custom override.
- :func:`representative_docs` — each topic's highest-loading documents.
- :func:`topics_over_time` — mean prevalence per distinct timestamp.
- :func:`topics_per_class` — mean prevalence within each group.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import validation as _diagnostics
from . import effects as _effects
from .keywords import _zeta as _fw_zeta


# A custom-label registry keyed by ``id(model)``. PyO3 extension classes may not
# support weakref or attribute assignment, so we do not stash labels on the model
# itself; the caller's process holds the mapping for as long as the model lives.
_LABELS: dict[int, dict[int, str]] = {}


def _doc_topic(model) -> np.ndarray:
    return np.asarray(model.doc_topic, dtype=np.float64)


def _has_labels(model) -> bool:
    """Whether the model carries hard ``labels`` (the clustering models do)."""
    labels = getattr(model, "labels", None)
    return labels is not None and len(labels) > 0


def stop_reason(model) -> str:
    """Why a fitted model's training loop stopped: early convergence vs the
    iteration cap (issue #267).

    Reads the model-neutral ``converged`` flag and ``fit_history`` (the
    ``(iteration, objective)`` trace — the collapsed log-likelihood for the Gibbs
    samplers, the variational bound for STM/CTM/STS). It answers the question
    ``converged`` alone leaves implicit: did ``iters`` act as a floor (the run
    stopped early on ``convergence_tol``) or a ceiling (it ran the full budget)?

    Returns a human-readable line, e.g.
    ``"converged at iteration 312: ... (last relative change 8.0e-05)"`` or
    ``"ran to the iteration cap (800 iterations) without early stopping; ..."``.
    """
    converged = bool(getattr(model, "converged", False))
    hist = list(getattr(model, "fit_history", None) or [])
    last_iter = hist[-1][0] if hist else None
    rel = None
    if len(hist) >= 2:
        prev, cur = hist[-2][1], hist[-1][1]
        rel = abs(cur - prev) / (abs(prev) + 1e-12)
    rel_str = f" (last relative change {rel:.1e})" if rel is not None else ""
    if converged:
        where = f" at iteration {last_iter}" if last_iter is not None else ""
        return (
            f"converged{where}: the monitored objective's relative change fell "
            f"below convergence_tol{rel_str}"
        )
    ran = f" ({last_iter} iterations)" if last_iter is not None else ""
    return (
        f"ran to the iteration cap{ran} without early stopping{rel_str}; "
        f"convergence_tol was disabled (0) or never reached"
    )


def topic_sizes(model) -> dict:
    """Per-topic size and expected mass for any fitted model.

    The ``size`` is each topic's count of hard document assignments. On a
    clustering model that exposes ``labels`` (Top2Vec / BERTopic) we count those
    assignments directly and report the number of ``-1`` (noise/outlier)
    documents separately under ``"outliers"``; on every other model we take the
    argmax of ``doc_topic`` per document. The ``mass`` is the expected number of
    documents in each topic, ``doc_topic.sum(axis=0)`` — the soft analog of the
    hard count.

    Returns ``{"size": (K,) int array, "mass": (K,) float array,
    "outliers": int}``.
    """
    theta = _doc_topic(model)
    k = theta.shape[1]
    mass = theta.sum(axis=0)
    outliers = 0
    if _has_labels(model):
        labels = np.asarray(list(model.labels), dtype=np.int64)
        outliers = int(np.sum(labels == -1))
        size = np.bincount(labels[labels >= 0], minlength=k)[:k]
    else:
        size = np.bincount(theta.argmax(axis=1), minlength=k)[:k]
    return {"size": size.astype(np.int64), "mass": mass, "outliers": outliers}


def set_topic_labels(model, mapping: dict) -> None:
    """Store custom labels for some or all of a model's topics.

    ``mapping`` is ``{topic_id: label}``; labels merge over (and override)
    ``model.topic_names`` everywhere this module reports a topic. The store is
    keyed by ``id(model)`` rather than set on the model, since the compiled model
    classes may not allow attribute assignment.
    """
    store = _LABELS.setdefault(id(model), {})
    for topic, label in mapping.items():
        store[int(topic)] = str(label)


def topic_labels(model) -> list:
    """The effective per-topic labels: any custom labels set via
    :func:`set_topic_labels` override the model's own ``topic_names``."""
    names = list(getattr(model, "topic_names", []))
    k = int(getattr(model, "num_topics", len(names)))
    if len(names) < k:
        names = names + [f"topic_{t}" for t in range(len(names), k)]
    custom = _LABELS.get(id(model), {})
    for topic, label in custom.items():
        if 0 <= topic < len(names):
            names[topic] = label
    return names


def representative_docs(model, texts, *, topic=None, n=5):
    """The documents that load most heavily on a topic, with their text.

    Wraps :func:`topica.find_thoughts`, returning ``texts`` for the ``n``
    highest-``doc_topic`` documents. With ``topic`` given, returns that topic's
    list; with ``topic=None`` returns ``{topic_id: [texts]}`` for every topic.
    Each list is ordered by descending topic proportion.
    """
    def docs_for(t):
        thoughts = _diagnostics.find_thoughts(model.doc_topic, texts, topic=t, n=n)
        return [text for _, _, text in thoughts]

    if topic is not None:
        return docs_for(topic)
    k = _doc_topic(model).shape[1]
    return {t: docs_for(t) for t in range(k)}


def _top_words(model, t, n):
    """Top-``n`` words for topic ``t`` as a plain list of strings, using the
    model's ``top_words`` method when present and falling back to the raw φ row."""
    method = getattr(model, "top_words", None)
    if callable(method):
        try:
            pairs = method(n, topic=t)
            return [w for w, _ in pairs]
        except Exception as exc:
            warnings.warn(
                f"{type(model).__name__}.top_words failed ({type(exc).__name__}: "
                f"{exc}); falling back to raw topic-word rows, which drops any "
                "custom weighting (e.g. FREX) that top_words applies.",
                stacklevel=2,
            )
    phi = np.asarray(model.topic_word, dtype=np.float64)
    vocab = list(model.vocabulary)
    idx = np.argsort(phi[t])[::-1][:n]
    return [vocab[i] for i in idx]


def topic_info(model, texts=None, *, n=8, labels=None) -> list:
    """One summary row per topic — the headline table for a fitted model.

    Each row is a dict with ``topic`` (id), ``label``, ``size`` (hard
    assignments), ``prevalence`` (mean of the topic's ``doc_topic`` column), and
    ``top_words`` (the top-``n`` words, via ``model.top_words`` when available
    else the raw topic-word row). When ``texts`` is given each row also carries
    ``representative_docs``, its ``n`` highest-loading documents. On a clustering
    model with outliers a final ``topic=-1`` row reports the outlier count and
    carries no words. Rows are sorted by topic id.

    ``labels`` overrides the labels for this table only; otherwise
    :func:`topic_labels` (custom labels over ``topic_names``) is used.
    """
    theta = _doc_topic(model)
    k = theta.shape[1]
    sizes = topic_sizes(model)
    effective = labels if labels is not None else topic_labels(model)
    prevalence = theta.mean(axis=0)

    rows = []
    for t in range(k):
        row = {
            "topic": t,
            "label": effective[t] if t < len(effective) else f"topic_{t}",
            "size": int(sizes["size"][t]),
            "prevalence": float(prevalence[t]),
            "top_words": _top_words(model, t, n),
        }
        if texts is not None:
            row["representative_docs"] = representative_docs(model, texts, topic=t, n=n)
        rows.append(row)

    if sizes["outliers"] > 0:
        outlier_row = {
            "topic": -1,
            "label": "outliers",
            "size": int(sizes["outliers"]),
            "prevalence": 0.0,
            "top_words": [],
        }
        if texts is not None:
            outlier_row["representative_docs"] = []
        rows.append(outlier_row)
    return rows


def topics_over_time(model, timestamps, *, normalize=True) -> dict:
    """Mean topic prevalence at each distinct timestamp value.

    ``timestamps`` is one value per document. For each distinct timestamp we
    average ``doc_topic`` over the documents stamped with it, giving a topic
    prevalence trajectory you can plot directly. With ``normalize=True`` each
    row is rescaled to sum to one (so it reads as a topic share at that time).

    Returns ``{"labels": [sorted distinct timestamps], "prevalence": (T, K)
    array}``.
    """
    theta = _doc_topic(model)
    stamps = np.asarray(list(timestamps))
    if stamps.shape[0] != theta.shape[0]:
        raise ValueError("timestamps must have one value per document")
    levels = sorted(np.unique(stamps), key=lambda v: str(v))
    prevalence = np.zeros((len(levels), theta.shape[1]), dtype=np.float64)
    for i, level in enumerate(levels):
        prevalence[i] = theta[stamps == level].mean(axis=0)
    if normalize:
        totals = prevalence.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        prevalence = prevalence / totals
    labels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
    return {"labels": labels, "prevalence": prevalence}


def topics_per_class(model, groups, *, ci=0.95):
    """Mean topic prevalence within each level of a grouping variable.

    A thin wrapper over :func:`topica.by_strata` on ``model.doc_topic``:
    ``groups`` is one label per document, and the result is a list of
    per-stratum prevalence records (mean and confidence interval per topic).
    """
    return _effects.by_strata(model.doc_topic, groups, ci=ci)


def contrastive_topics(model, texts, groups, *, prior=0.01, informative=False,
                       min_count=5, n_words=10, group_order=None):
    """Which topics most separate two groups, and the words that shift inside each.

    This is the topic-conditional extension of :func:`topica.fighting_words`. A
    plain Fighting Words contrast pools the whole corpus into two bags of words;
    here we hold the topic fixed and ask, *within* topic ``t``, how the two groups
    word it differently. Each document's word counts are weighted by its
    responsibility for the topic (``model.doc_topic[d, t]``), the weighted counts
    are split by group, and the Monroe-Colaresi-Quinn z-score is computed per
    topic. We report two complementary signals, since a topic both groups use
    equally can still split sharply on *how* they word it:

    - ``usage_diff`` — mean ``doc_topic`` for group A minus group B: which topics
      one side simply talks about more.
    - ``vocab_shift`` — the root-mean-square topic-conditional z over the words it
      keeps: how much the two groups diverge in their wording of the topic.

    Works on any fitted model that exposes ``doc_topic`` and ``vocabulary`` (LDA,
    STM, DMR, CTM, keyATM, ...). ``texts`` must be the same documents, in the same
    order, that produced ``model.doc_topic``.

    Parameters
    ----------
    model : a fitted topica model with ``doc_topic`` (D x K) and ``vocabulary``.
    texts : sequence of token lists (``list[list[str]]``), one per document,
        aligned row-for-row with ``model.doc_topic``.
    groups : sequence of one label per document. Must take exactly two distinct
        values (a binary contrast).
    prior : float, default 0.01
        Dirichlet pseudocount passed to the z-score; see :func:`fighting_words`.
    informative : bool, default False
        Use Monroe et al.'s informative (frequency-scaled) prior.
    min_count : int, default 5
        Within a topic, ignore words whose responsibility-weighted count across
        both groups is below this. Keeps the per-topic word lists and
        ``vocab_shift`` from being dominated by near-zero-mass words.
    n_words : int, default 10
        How many distinctive words to return per side, per topic.
    group_order : (a, b), optional
        Fix which group is A (positive z, positive ``usage_diff``). Defaults to
        the two labels sorted, so the result is deterministic.

    Returns
    -------
    list[dict], one row per topic sorted by descending ``abs(usage_diff)``. Each
    row has ``topic`` (id), ``name`` (effective label), ``a_label``/``b_label``
    (the two groups), ``usage_diff``, ``leans`` (the label that uses the topic
    more), ``vocab_shift``, and ``a_words``/``b_words`` (lists of ``(word, z)``,
    each side's most distinctive within-topic words).
    """
    theta = _doc_topic(model)
    n_docs, k = theta.shape
    if len(texts) != n_docs:
        raise ValueError(
            f"texts has {len(texts)} documents but the model's doc_topic has "
            f"{n_docs} rows; pass the same corpus, in the same order, that was fit."
        )
    groups = list(groups)
    if len(groups) != n_docs:
        raise ValueError(
            f"groups has {len(groups)} labels but the model's doc_topic has "
            f"{n_docs} rows."
        )

    levels = sorted(set(groups)) if group_order is None else list(group_order)
    if len(levels) != 2 or len(set(levels)) != 2:
        raise ValueError(
            "contrastive_topics needs exactly two distinct groups; got "
            f"{sorted(set(groups))}. For more levels, contrast a pair at a time."
        )
    a_label, b_label = levels
    in_a = np.array([g == a_label for g in groups])
    in_b = np.array([g == b_label for g in groups])
    if not in_a.any() or not in_b.any():
        raise ValueError(
            f"both groups must have documents; '{a_label}' has {int(in_a.sum())}, "
            f"'{b_label}' has {int(in_b.sum())}."
        )

    vocab = list(model.vocabulary)
    vindex = {w: i for i, w in enumerate(vocab)}
    n_vocab = len(vocab)

    # Responsibility-weighted word counts per group, accumulated into K x V
    # without ever materializing a dense D x V matrix: each document adds the
    # outer product of its topic responsibilities and its (sparse) token counts.
    y_a = np.zeros((k, n_vocab), dtype=np.float64)
    y_b = np.zeros((k, n_vocab), dtype=np.float64)
    for d, doc in enumerate(texts):
        local = {}
        for tok in doc:
            j = vindex.get(tok)
            if j is not None:
                local[j] = local.get(j, 0.0) + 1.0
        if not local:
            continue
        idx = np.fromiter(local.keys(), dtype=np.intp, count=len(local))
        cnt = np.fromiter(local.values(), dtype=np.float64, count=len(local))
        target = y_a if in_a[d] else (y_b if in_b[d] else None)
        if target is not None:
            target[:, idx] += np.outer(theta[d], cnt)

    usage_diff = theta[in_a].mean(axis=0) - theta[in_b].mean(axis=0)
    names = topic_labels(model)

    rows = []
    for t in range(k):
        zeta = _fw_zeta(y_a[t], y_b[t], prior=prior, informative=informative)
        keep = (y_a[t] + y_b[t]) >= float(min_count)
        z = np.where(keep, zeta, 0.0)
        order = np.argsort(z)
        a_words = [(vocab[i], float(z[i])) for i in order[::-1][:n_words] if z[i] > 0]
        b_words = [(vocab[i], float(z[i])) for i in order[:n_words] if z[i] < 0]
        shift = float(np.sqrt(np.mean(zeta[keep] ** 2))) if keep.any() else 0.0
        rows.append({
            "topic": t,
            "name": names[t] if t < len(names) else f"topic_{t}",
            "a_label": a_label,
            "b_label": b_label,
            "usage_diff": float(usage_diff[t]),
            "leans": a_label if usage_diff[t] > 0 else b_label,
            "vocab_shift": shift,
            "a_words": a_words,
            "b_words": b_words,
        })

    rows.sort(key=lambda r: abs(r["usage_diff"]), reverse=True)
    return rows


def _short(label, words, width=42):
    """A compact 'label: word word word' caption, truncated to `width`."""
    ws = " ".join(w for w, _ in words) if words and isinstance(words[0], tuple) else " ".join(map(str, words))
    text = f"{label}: {ws}" if ws else str(label)
    return text if len(text) <= width else text[: width - 1] + "…"


def plot_report(model, *, texts=None, timestamps=None, groups=None, n=8,
                coherence_type="c_v", title=None, figsize=None):
    """A one-figure overview of a fitted model, composed from topica's diagnostics.

    Panels are adaptive: each is drawn only when its inputs and the model support
    it, so the report works across every model. Always included is the topic
    prevalence bar (mean ``doc_topic`` per topic, labelled with each topic's top
    words). Added when available:

    - **topic quality** — coherence vs exclusivity (the stm quality frontier); a
      windowed ``coherence_type`` is used when ``texts`` is given (raw strings or
      token lists are both accepted), else UMass;
    - **topic correlation** — the ``doc_topic`` correlation heatmap (K in 2..40);
    - **topics over time** — mean prevalence per distinct ``timestamps`` value;
    - **topics per class** — mean prevalence within each level of ``groups``.

    Returns a matplotlib ``Figure``; save it with ``fig.savefig("report.png")`` or
    ``.pdf``. Requires matplotlib (the only added dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_report requires matplotlib") from exc

    rows = [r for r in topic_info(model, n=n) if r["topic"] >= 0]
    K = len(rows)
    captions = [_short(r["label"], r["top_words"]) for r in rows]
    prevalence = np.array([r["prevalence"] for r in rows])

    # Decide which optional panels we can draw. A panel that fails for a real
    # reason warns (naming itself) rather than vanishing silently — a missing
    # panel should never read as "this model has nothing to show here".
    def _skip(panel, exc):
        warnings.warn(
            f"plot_report: '{panel}' panel skipped — {type(exc).__name__}: {exc}",
            stacklevel=2,
        )

    panels = ["prevalence"]
    quality = None
    try:
        # Coherence wants a tokenized reference corpus; accept raw strings too by
        # splitting them, so callers can reuse the same `texts` they pass elsewhere.
        ref = texts
        if ref is not None and len(ref) and isinstance(ref[0], str):
            ref = [t.split() for t in ref]
        quality = _diagnostics.quality_frontier(
            model, n=n, texts=ref,
            coherence_type=coherence_type if ref is not None else "u_mass",
        )
        panels.append("quality")
    except Exception as exc:
        _skip("quality", exc)
        ref = None
    corr = None
    if 2 <= K <= 40:
        try:
            corr = np.asarray(_diagnostics.topic_correlation(model.doc_topic).cor)
            panels.append("correlation")
        except Exception as exc:
            _skip("correlation", exc)
    over_time = None
    if timestamps is not None:
        try:
            over_time = topics_over_time(model, timestamps)
            panels.append("time")
        except Exception as exc:
            _skip("time", exc)
    per_class = None
    if groups is not None:
        try:
            theta = _doc_topic(model)
            g = np.asarray(list(groups))
            if g.shape[0] != theta.shape[0]:
                raise ValueError(
                    f"groups has {g.shape[0]} entries but doc_topic has "
                    f"{theta.shape[0]} rows; pass groups aligned to the kept "
                    "documents (corpus.kept_indices), not the original documents."
                )
            levels = sorted(np.unique(g), key=lambda v: str(v))
            per_class = (levels, np.array([theta[g == lv].mean(axis=0) for lv in levels]))
            panels.append("class")
        except Exception as exc:
            _skip("class", exc)

    ncols = 1 if len(panels) == 1 else 2
    nrows = (len(panels) + ncols - 1) // ncols
    if figsize is None:
        figsize = (7.0 * ncols, max(3.2, 0.30 * K + 1.4) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = [ax for r in axes for ax in r]
    for ax in flat[len(panels):]:
        ax.axis("off")

    for ax, panel in zip(flat, panels):
        if panel == "prevalence":
            order = np.argsort(prevalence)
            ax.barh(np.arange(K), prevalence[order], color="#4C72B0")
            ax.set_yticks(np.arange(K))
            ax.set_yticklabels([captions[i] for i in order], fontsize=8)
            ax.set_xlabel("Mean prevalence (θ)")
            ax.set_title("Topics by prevalence")
        elif panel == "quality":
            ax.scatter(quality["coherence"], quality["exclusivity"],
                       s=300 * quality["prevalence"] + 20, color="#55A868", alpha=0.8)
            for t in range(len(quality["topic"])):
                ax.annotate(str(int(quality["topic"][t])),
                            (quality["coherence"][t], quality["exclusivity"][t]), fontsize=7)
            # Name the coherence metric so its scale is not mistaken for another
            # (c_v ~ 0..1 when texts are given; u_mass is negative otherwise).
            coh_metric = coherence_type if ref is not None else "u_mass"
            ax.set_xlabel(f"Semantic coherence ({coh_metric})")
            ax.set_ylabel("Exclusivity")
            ax.set_title("Topic quality (size ∝ prevalence)")
            # Exclusivity saturates near 1; stop matplotlib's offset notation
            # (e.g. "1e-10+9.99e-1") from rendering on the axis.
            ax.ticklabel_format(useOffset=False, style="plain")
        elif panel == "correlation":
            # Mask the always-1.0 diagonal (self-correlation carries no
            # information and, left in, saturates the diverging scale and
            # dominates the panel) and set the color range from the off-diagonal
            # magnitudes so the real structure reads. Matches
            # topica.viz.TopicCorrelation.
            import matplotlib as mpl
            cmax = max(float(np.abs(corr - np.eye(K)).max()), 1e-3)
            disp = corr.astype(float).copy()
            np.fill_diagonal(disp, np.nan)
            cmap = mpl.colormaps["RdBu_r"].copy()
            cmap.set_bad("#f0f0f0")
            im = ax.imshow(disp, cmap=cmap, vmin=-cmax, vmax=cmax)
            ax.set_title("Topic correlation")
            ax.set_xlabel("topic")
            ax.set_ylabel("topic")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif panel == "time":
            labels, prev = over_time["labels"], over_time["prevalence"]
            top = np.argsort(prev.mean(axis=0))[::-1][:6]
            for t in top:
                ax.plot(range(len(labels)), prev[:, t], marker="o", ms=3,
                        label=captions[t].split(":")[0])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([str(lv) for lv in labels], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Topic share")
            ax.set_title("Topics over time (top 6)")
            ax.legend(fontsize=6, ncol=2)
        elif panel == "class":
            import matplotlib as mpl
            levels, means = per_class  # means: (num_levels, K)
            n_lv = len(levels)
            if n_lv <= 5:
                # Connected-dot ("dumbbell") plot: each topic is a row with one
                # dot per class joined by a line, so the between-class gap -- the
                # quantity of interest -- reads directly, which a few-row heatmap
                # hides. Topics are ordered by that gap, putting the most
                # class-differentiated topics on top.
                spread = means.max(axis=0) - means.min(axis=0)
                order = np.argsort(spread)
                ypos = np.arange(K)
                cmap = mpl.colormaps["tab10"]
                for yi, t in enumerate(order):
                    col = means[:, t]
                    ax.plot([col.min(), col.max()], [yi, yi],
                            color="#cccccc", lw=1.5, zorder=1)
                for j, lv in enumerate(levels):
                    ax.scatter(means[j, order], ypos, s=28, color=cmap(j),
                               label=str(lv), zorder=2)
                ax.set_yticks(ypos)
                ax.set_yticklabels([captions[i] for i in order], fontsize=7)
                ax.set_xlabel("Mean prevalence (θ)")
                ax.set_title("Prevalence by class")
                ax.legend(fontsize=6)
            else:
                im = ax.imshow(means, aspect="auto", cmap="viridis")
                ax.set_yticks(range(n_lv))
                ax.set_yticklabels([str(lv) for lv in levels], fontsize=7)
                ax.set_xlabel("topic")
                ax.set_title("Prevalence by class")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or f"{type(model).__name__} — {K} topics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


__all__ = [
    "topic_info",
    "topic_sizes",
    "topic_labels",
    "set_topic_labels",
    "representative_docs",
    "topics_over_time",
    "topics_per_class",
    "contrastive_topics",
    "plot_report",
]
