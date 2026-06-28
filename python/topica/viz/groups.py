"""Prevalence-by-group heatmap -- topic share across the levels of a covariate.

A groups x topics heatmap of mean topic prevalence, reading the same
``by_strata`` surface ``topica.topics_per_class`` exposes. Pass a fitted model and
a per-document grouping label; with ``corpus=`` and ``nsims=`` the per-cell means
are widened by the method of composition (so the ``.to_frame()`` carries well-calibrated
intervals), but the heatmap itself encodes the mean -- a grid of CIs is unreadable.
"""

from __future__ import annotations

import numpy as np

from .base import SEQ_CMAP, Panel
from .capability import capabilities


class PrevalenceHeatmap(Panel):
    """Mean topic prevalence within each level of a grouping variable."""

    title = "Topic prevalence by group"

    def __init__(self, model, groups, *, corpus=None, nsims=None, ci=0.95, seed=0):
        from ..analysis import topic_labels
        from ..keyatm import by_strata

        self.cap = capabilities(model)
        strata = by_strata(model, groups, ci=ci, corpus=corpus, nsims=nsims, seed=seed) \
            if nsims else by_strata(model.doc_topic, groups, ci=ci)
        self._strata = strata
        self._groups = [s.stratum for s in strata]
        self._mean = np.vstack([np.asarray(s.mean) for s in strata])  # (G, K)
        self._low = np.vstack([np.asarray(s.ci_low) for s in strata])
        self._high = np.vstack([np.asarray(s.ci_high) for s in strata])
        self._n = [int(s.n) for s in strata]
        k = self._mean.shape[1]
        labels = topic_labels(model)
        self._labels = [labels[t] if t < len(labels) else f"topic_{t}" for t in range(k)]

    def to_frame(self):
        import pandas as pd

        rows = []
        for gi, g in enumerate(self._groups):
            for t in range(self._mean.shape[1]):
                rows.append({
                    "group": g,
                    "n": self._n[gi],
                    "topic": t,
                    "label": self._labels[t],
                    "prevalence": float(self._mean[gi, t]),
                    "ci_low": float(self._low[gi, t]),
                    "ci_high": float(self._high[gi, t]),
                })
        return pd.DataFrame(rows)

    def matrix(self):
        """The group x topic mean-prevalence matrix as a wide DataFrame."""
        import pandas as pd

        return pd.DataFrame(self._mean, index=[str(g) for g in self._groups],
                            columns=[f"{t}: {self._labels[t]}" for t in range(self._mean.shape[1])])

    def _figsize(self):
        g, k = self._mean.shape
        return (max(5.0, 0.5 * k + 2.0), max(2.5, 0.5 * g + 1.5))

    def _draw(self, fig, *, annot=None):
        g, k = self._mean.shape
        if annot is None:
            annot = g * k <= 120  # don't paper a large grid with numbers
        ax = fig.subplots()
        im = ax.imshow(self._mean, cmap=SEQ_CMAP, aspect="auto", vmin=0.0)
        ax.set_xticks(range(k))
        ax.set_yticks(range(g))
        ax.set_xticklabels([str(t) for t in range(k)], fontsize=7)
        ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(self._groups, self._n)], fontsize=8)
        ax.set_xlabel("topic")
        if annot:
            thresh = 0.5 * float(self._mean.max())
            for gi in range(g):
                for t in range(k):
                    v = self._mean[gi, t]
                    ax.text(t, gi, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < thresh else "black")
        ax.set_title(self.title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean prevalence")
