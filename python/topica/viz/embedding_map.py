"""The document map -- a 2-D projection of the document cloud (supplement figure).

A *document* projection, not a topic one. It answers "do the documents separate the
way the topics claim?" and is explicit about what a 2-D layout can and cannot say:

- The projection runs in topica's own Rust core (``topica.project``): PCA by default
  (deterministic, distance-faithful), or UMAP / t-SNE (neighbor-preserving but
  non-metric and not reproducible). For PCA the title reports the fraction of
  variance the two axes carry; for the others it carries the non-metric,
  not-reproducible caveat.
- Coordinates come from the document **embeddings** you pass (the array you fit an
  embedding model on -- the models do not retain it), or, for a count/soft-θ model,
  from the **clr-transformed θ** simplex. A hard/degenerate-θ cluster model with no
  embeddings is refused.
- Density is shown with alpha clouds / hexbin, never convex hulls (UMAP and t-SNE
  preserve neither density nor size). Past a handful of topics the default is to gray
  everything and color one ``highlight_topic``; small K gets a colorblind-safe
  (Okabe-Ito) categorical palette. A ``-1`` outlier layer is drawn separately.
- Large corpora are stratified-subsampled to ``max_points`` (by dominant topic,
  including ``-1``) with a fixed seed and a "showing N of D" badge.
"""

from __future__ import annotations

import numpy as np

from .base import SEQ_CMAP, Panel, _require
from .capability import capabilities
from .correlation import _clr

# Okabe-Ito: a colorblind-safe qualitative palette (8 hues incl. black).
_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
              "#0072B2", "#D55E00", "#CC79A7", "#000000"]
_GRAY = "#CCCCCC"
_HIGHLIGHT = "#0072B2"
_MAX_CATEGORICAL = 8  # beyond this no palette stays distinguishable / CVD-safe


class DocumentMap(Panel):
    """A 2-D projection of the document cloud, colored by dominant topic."""

    title = "Document map"
    interactive = True

    def __init__(self, model, doc_embeddings=None, *, method="pca",
                 highlight_topic=None, max_points=20000, n_neighbors=15,
                 perplexity=30.0, seed=0):
        import topica

        self.cap = capabilities(model)
        self.method = method
        self.highlight_topic = None if highlight_topic is None else int(highlight_topic)
        self.seed = int(seed)

        # Dominant topic and the outlier layer: cluster models carry hard labels
        # (with a -1 noise bucket); everything else takes argmax of theta.
        labels = getattr(model, "labels", None)
        if labels is not None:
            self._dominant = np.asarray(list(labels), dtype=np.int64)
        else:
            self._dominant = np.asarray(model.doc_topic, dtype=np.float64).argmax(axis=1)

        # Coordinate source, capability-routed.
        if doc_embeddings is not None:
            coords = np.asarray(doc_embeddings, dtype=np.float64)
            self.source = "document embeddings"
        elif self.cap.soft_theta:
            coords = _clr(np.asarray(model.doc_topic, dtype=np.float64))
            self.source = "θ simplex (clr)"
        else:
            raise ValueError(
                f"{self.cap.name} has degenerate (hard cluster) θ and does not retain "
                "its document embeddings; pass doc_embeddings= (the array you fit on) "
                "to map it in embedding space."
            )
        if coords.shape[0] != self._dominant.shape[0]:
            raise ValueError("doc_embeddings must have one row per document")

        # Stratified subsample (by dominant topic, incl. -1) for large corpora.
        self.n_total = coords.shape[0]
        self._sample = self._subsample(self.n_total, max_points)
        self.sampled = len(self._sample) < self.n_total
        coords = coords[self._sample]
        self._dom = self._dominant[self._sample]

        # Project in the Rust core. PCA gets a variance-explained diagnostic.
        self._xy = np.asarray(topica.project(
            coords, 2, method=method, n_neighbors=n_neighbors,
            perplexity=perplexity, seed=seed))
        self.var_explained = None
        if method == "pca":
            centered = coords - coords.mean(axis=0, keepdims=True)
            total = float(np.var(centered, axis=0).sum())
            shown = float(np.var(self._xy, axis=0).sum())
            self.var_explained = shown / total if total > 0 else None

    def _subsample(self, n, max_points):
        if n <= max_points:
            return np.arange(n)
        rng = np.random.default_rng(self.seed)
        idx = []
        for g in np.unique(self._dominant):
            gi = np.where(self._dominant == g)[0]
            take = min(len(gi), max(1, round(len(gi) * max_points / n)))
            idx.append(rng.choice(gi, size=take, replace=False))
        out = np.concatenate(idx)
        rng.shuffle(out)
        return out

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({
            "doc": self._sample,
            "x": self._xy[:, 0],
            "y": self._xy[:, 1],
            "dominant_topic": self._dom,
            "outlier": self._dom < 0,
            "sampled": self.sampled,
        })

    def _palette_topics(self):
        """The non-outlier topics present, and whether they fit a categorical
        palette (small K) or should fall back to a density view (large K)."""
        topics = sorted(int(t) for t in np.unique(self._dom) if t >= 0)
        return topics, len(topics) <= _MAX_CATEGORICAL

    def _figsize(self):
        return (6.5, 5.5)

    def _caption(self):
        if self.method == "pca" and self.var_explained is not None:
            tail = f"PCA, ≈{self.var_explained:.0%} of variance in 2 axes"
        else:
            # UMAP/t-SNE: distances are not meaningful and neither fit is reproducible
            # across runs (bhtsne is unseeded; umap-rs's negative sampling is too).
            tail = f"{self.method.upper()} (non-metric: distances not meaningful; not reproducible)"
        if self.sampled:
            tail += f"; showing {len(self._sample):,} of {self.n_total:,} (sampled)"
        return f"{self.title} — {self.source}; {tail}"

    def _draw(self, fig):
        ax = fig.subplots()
        xy, dom = self._xy, self._dom
        out = dom < 0
        # Outlier layer first, always a quiet gray.
        if out.any():
            ax.scatter(xy[out, 0], xy[out, 1], s=8, c=_GRAY, alpha=0.4,
                       linewidths=0, label="-1 (outlier)")
        inl = ~out
        topics, categorical = self._palette_topics()

        if self.highlight_topic is not None:
            # Gray everyone, color the one topic -- the readable default at any K.
            other = inl & (dom != self.highlight_topic)
            hit = inl & (dom == self.highlight_topic)
            ax.scatter(xy[other, 0], xy[other, 1], s=8, c=_GRAY, alpha=0.35, linewidths=0)
            ax.scatter(xy[hit, 0], xy[hit, 1], s=14, c=_HIGHLIGHT, alpha=0.85,
                       linewidths=0, label=f"topic {self.highlight_topic}")
            ax.legend(fontsize=7, loc="best", markerscale=1.5)
        elif categorical:
            for i, t in enumerate(topics):
                m = inl & (dom == t)
                ax.scatter(xy[m, 0], xy[m, 1], s=12, c=_OKABE_ITO[i % len(_OKABE_ITO)],
                           alpha=0.75, linewidths=0, label=f"topic {t}")
            ax.legend(fontsize=7, loc="best", ncol=2, markerscale=1.5)
        else:
            # Too many topics for a categorical palette: a plain density hexbin.
            hb = ax.hexbin(xy[inl, 0], xy[inl, 1], gridsize=40, cmap=SEQ_CMAP, mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="documents")
            ax.text(0.99, 0.01, f"{len(topics)} topics — pass highlight_topic= to color one",
                    transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color="0.4")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("dimension 1")
        ax.set_ylabel("dimension 2")
        ax.set_title(self._caption(), fontsize=9)

    def to_html(self, path=None, **kwargs):
        """An interactive Plotly scatter (WebGL), colored by dominant topic, with
        the outlier layer and per-point hover. Needs ``topica[viz]``."""
        px = _require("plotly.express", "viz")
        import pandas as pd

        df = self.to_frame()
        df["topic"] = np.where(df["outlier"], "-1 (outlier)", df["dominant_topic"].astype(str))
        topics, categorical = self._palette_topics()
        if self.highlight_topic is not None:
            df["topic"] = np.where(
                df["dominant_topic"] == self.highlight_topic,
                f"topic {self.highlight_topic}",
                np.where(df["outlier"], "-1 (outlier)", "other"))
        fig = px.scatter(
            df, x="x", y="y", color="topic", render_mode="webgl",
            hover_data={"doc": True, "x": False, "y": False},
            color_discrete_sequence=_OKABE_ITO if categorical else None,
            title=self._caption(),
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(xaxis_title="dimension 1", yaxis_title="dimension 2",
                          legend_title="dominant topic")
        if path is not None:
            fig.write_html(path)
        return fig
