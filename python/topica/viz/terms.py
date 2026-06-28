"""Panel 3: the disciplined pyLDAvis replacement.

A **term barchart** (top words with named weighting modes, gated by the model's
capability descriptor) and a **seriated topic-similarity heatmap** (K x K, ordered
by hierarchical clustering) -- all pairwise relationships at full fidelity, with no
spurious 2-D metric plane. The interactive build pairs the two: the seriated heatmap
for the overview and a dropdown to pick a topic and read its top words.
"""

from __future__ import annotations

import numpy as np

from .base import SEQ_CMAP, Panel, _require
from .capability import capabilities

_DARK = "#4C72B0"
_LIGHT = "#C7D3E8"


def _topic_word(model):
    from ..coherence import _as_topic_word
    phi = _as_topic_word(model)
    if phi.ndim == 3:
        # A content-covariate model (SAGE) exposes a (K, G, V) per-group array; the
        # group-agnostic panels use the group-averaged marginal. (The per-group
        # wording lives in the dedicated content_covariate panel.)
        marg = getattr(model, "topic_word_marginal", None)
        phi = np.asarray(marg, dtype=np.float64) if marg is not None else phi.mean(axis=1)
    return phi


def _scored_words(model, topic, mode, n, cap):
    """Return ``[(word, weight), ...]`` for one topic under a weighting `mode`."""
    from ..validation import frex, relevance, label_topics

    if mode not in cap.word_modes:
        raise ValueError(
            f"mode={mode!r} is not valid for {cap.name} (its topic_word is "
            f"{cap.word_weight_label}); available modes: {cap.word_modes}"
        )
    # Use the 2-D (marginal) topic-word everywhere, so a content model's 3-D
    # topic_word (SAGE) does not reach frex/relevance/label_topics, which assume
    # (K, V). _topic_word collapses to the group-averaged marginal.
    phi = _topic_word(model)
    vocab = list(model.vocabulary)
    if mode == "prob":
        idx = np.argsort(phi[topic])[::-1][:n]
        return [(vocab[i], float(phi[topic, i])) for i in idx]
    if mode == "frex":
        return frex(phi, vocab, n=n)[topic]
    if mode == "relevance":
        return relevance(phi, vocab, topic=topic, n=n)
    if mode == "lift":
        marg = np.clip(phi.mean(axis=0), 1e-12, None)
        lift = phi[topic] / marg
        idx = np.argsort(lift)[::-1][:n]
        return [(vocab[i], float(lift[i])) for i in idx]
    if mode == "score":
        return label_topics(phi, vocab, n=n)[topic]["score"]
    raise ValueError(f"unknown mode {mode!r}")


class TermBarchart(Panel):
    """Top words of one topic, weighted by a named mode, with the corpus-frequency
    overlay. Optional inclusion-probability error bars (a bootstrap, so off by
    default)."""

    title = "Top words"

    def __init__(self, model, *, topic, mode="prob", n=10, texts=None,
                 error_bars=False, n_boot=100, seed=0):
        self.cap = capabilities(model)
        self.topic = int(topic)
        self.mode = mode
        self.n = n
        self._words = _scored_words(model, self.topic, mode, n, self.cap)
        # Corpus-frequency overlay: the prevalence-weighted marginal
        # p(w) = sum_t p(t) p(w|t). Only meaningful in "prob" mode -- frex / lift /
        # relevance live on a different (non-probability) scale, so an overlay of
        # p(w) ~ 0.02 against frex in [0, 1] would be a meaningless sliver.
        self._overlay = None
        if mode == "prob":
            phi = _topic_word(model)
            vocab = {w: i for i, w in enumerate(model.vocabulary)}
            theta = np.asarray(model.doc_topic, dtype=np.float64)
            pw = theta.mean(axis=0) @ phi
            self._overlay = [float(pw[vocab[w]]) if w in vocab else 0.0 for w, _ in self._words]
        self._inclusion = None
        if error_bars:
            from ..effects import standard_errors

            tw = standard_errors(model, texts, of="top_words", method="bootstrap",
                                 n_boot=n_boot, topn=n, seed=seed)
            inc = {w: (p, lo, hi) for (w, p, lo, hi) in tw[self.topic].words}
            self._inclusion = [inc.get(w) for w, _ in self._words]

    def to_frame(self):
        import pandas as pd

        rows = []
        for i, (w, val) in enumerate(self._words):
            row = {"topic": self.topic, "rank": i, "word": w, "weight": val}
            if self._overlay is not None:
                row["corpus_weight"] = self._overlay[i]
            if self._inclusion is not None and self._inclusion[i] is not None:
                p, lo, hi = self._inclusion[i]
                row.update(inclusion_prob=p, inclusion_low=lo, inclusion_high=hi)
            rows.append(row)
        return pd.DataFrame(rows)

    def _figsize(self):
        return (5.5, max(2.2, 0.34 * len(self._words) + 1.0))

    def _draw(self, fig):
        words = [w for w, _ in self._words][::-1]
        vals = [v for _, v in self._words][::-1]
        y = np.arange(len(words))
        ax = fig.subplots()
        ax.barh(y, vals, color=_DARK, height=0.7, label="in topic", zorder=2)
        if self._overlay is not None:
            ax.barh(y, self._overlay[::-1], color=_LIGHT, height=0.7,
                    label="corpus overall", zorder=1)
            ax.legend(fontsize=7, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        label = self.cap.word_weight_label if self.mode == "prob" else self.mode
        ax.set_xlabel(label)
        ax.set_title(f"Topic {self.topic} — {self.mode}", fontsize=10)


def _topic_distance(model, cap):
    """K x K topic distance: sqrt-Jensen-Shannon for probability topic_word,
    cosine distance for c-TF-IDF (which is not a distribution)."""
    phi = _topic_word(model).astype(np.float64)
    k = phi.shape[0]
    if cap.prob_simplex_words:
        p = phi / np.clip(phi.sum(axis=1, keepdims=True), 1e-12, None)
        # Vectorized over the inner pair: JS(p,q) = H(m) - (H(p)+H(q))/2, one numpy
        # pass per row (O(K) Python iterations, not O(K^2) with a per-element KL).
        ent = lambda a: -(a * np.log(np.clip(a, 1e-12, None))).sum(axis=-1)
        h = ent(p)  # (K,) per-topic entropy
        d = np.zeros((k, k))
        for i in range(k):
            m = 0.5 * (p[i] + p)            # (K, V)
            js = ent(m) - 0.5 * (h[i] + h)  # (K,)
            d[i] = np.sqrt(np.clip(js, 0.0, None))
        d = 0.5 * (d + d.T)                 # symmetrize away float drift
        np.fill_diagonal(d, 0.0)
        return d, "sqrt Jensen-Shannon"
    norm = phi / np.clip(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12, None)
    d = 1.0 - norm @ norm.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None), "cosine distance"


def _seriate(dist):
    """Hierarchical-clustering leaf order for a distance matrix (identity if scipy
    is unavailable or K is tiny)."""
    k = dist.shape[0]
    if k < 3:
        return list(range(k)), None
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
    except ImportError:
        return list(range(k)), None
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    z = optimal_leaf_ordering(z, condensed)
    return list(leaves_list(z)), z


class TopicSimilarity(Panel):
    """A seriated K x K topic-similarity heatmap -- every pairwise relationship at
    full fidelity, ordered by hierarchical clustering, no spurious metric plane."""

    title = "Topic similarity"

    def __init__(self, model):
        from ..analysis import topic_labels

        self.cap = capabilities(model)
        self._dist, self.metric = _topic_distance(model, self.cap)
        self._order, self._linkage = _seriate(self._dist)
        self._labels = topic_labels(model)

    def to_frame(self):
        import pandas as pd

        order = self._order
        sim = 1.0 - self._dist
        labels = [f'{t}: {self._labels[t] if t < len(self._labels) else t}' for t in order]
        return pd.DataFrame(sim[np.ix_(order, order)], index=labels, columns=labels)

    def _figsize(self):
        return (max(4.0, 0.4 * len(self._order) + 1.5),) * 2

    def _draw(self, fig):
        order = self._order
        k = len(order)
        sim = (1.0 - self._dist)[np.ix_(order, order)]
        ax = fig.subplots()
        # Anchor the scale at 0 so color encodes the absolute similarity level: an
        # all-0.9 block must read as uniformly high, not get contrast-stretched.
        im = ax.imshow(sim, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([str(t) for t in order], fontsize=7, rotation=90)
        ax.set_yticklabels([str(t) for t in order], fontsize=7)
        ax.set_title(f"{self.title} (1 − {self.metric}, seriated)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"1 − {self.metric}")


class _InteractiveFigure:
    """Thin wrapper around a Plotly ``Figure`` that gives it the panel-style
    ``.to_html(path)`` writer. A raw Plotly ``Figure.to_html`` takes ``config`` as
    its first positional argument and returns a string, so the documented
    ``.to_html("page.html")`` call would silently write nothing; here it writes the
    file. Every other attribute (``.show()``, ``.write_html()``, ``.write_image()``,
    ``.data``, ``.layout``, ...) delegates to the underlying figure, available as
    ``.figure``."""

    def __init__(self, figure):
        self.figure = figure

    def to_html(self, path=None, **kwargs):
        """Write a self-contained interactive HTML page to ``path`` (matching the
        panel renderers); with ``path=None`` return the HTML as a string."""
        if path is None:
            return self.figure.to_html(**kwargs)
        self.figure.write_html(path, **kwargs)
        return path

    def __getattr__(self, name):
        return getattr(self.figure, name)


def term_topic_browser(model, *, n=10, mode="prob"):
    """An interactive (Plotly) term browser + topic-similarity heatmap: the seriated
    K x K heatmap for the overview, and a dropdown to pick a topic and read its top
    words. Returns an interactive view whose ``.to_html("page.html")`` writes a
    self-contained page (the wrapped Plotly figure is at ``.figure``). Needs
    ``topica[viz]``."""
    go = _require("plotly.graph_objects", "viz")
    from plotly.subplots import make_subplots

    cap = capabilities(model)
    sim_panel = TopicSimilarity(model)
    order = sim_panel._order
    sim = (1.0 - sim_panel._dist)[np.ix_(order, order)]
    tick = [str(t) for t in order]

    topics = list(range(_topic_word(model).shape[0]))
    weight_label = cap.word_weight_label if mode == "prob" else mode

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
        subplot_titles=("Topic similarity (seriated)", f"Top words ({mode})"),
    )
    fig.add_trace(
        go.Heatmap(z=sim, x=tick, y=tick, colorscale="Viridis", colorbar=dict(title="sim", x=0.46)),
        row=1, col=1,
    )
    # One horizontal bar trace per topic; the dropdown toggles which is visible.
    for t in topics:
        words = _scored_words(model, t, mode, n, cap)
        ws = [w for w, _ in words][::-1]
        vs = [v for _, v in words][::-1]
        fig.add_trace(
            go.Bar(x=vs, y=ws, orientation="h", marker_color=_DARK,
                   visible=(t == topics[0]), name=f"topic {t}", showlegend=False),
            row=1, col=2,
        )
    buttons = []
    for i, t in enumerate(topics):
        vis = [True] + [j == i for j in range(len(topics))]  # heatmap + the chosen bar
        buttons.append(dict(label=f"topic {t}", method="update", args=[{"visible": vis}]))
    fig.update_layout(
        title=f"{cap.name}: topics and their words",
        updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.16, yanchor="top",
                          showactive=True)],
        height=420, width=820, bargap=0.25,
    )
    fig.update_xaxes(title_text="topic", row=1, col=1)
    fig.update_yaxes(title_text="topic", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text=weight_label, row=1, col=2)
    return _InteractiveFigure(fig)
