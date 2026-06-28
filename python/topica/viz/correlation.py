"""Topic correlation -- the compositionally-aware layer.

Raw across-document correlation of theta is *compositionally biased*: the
sum-to-one constraint forces a spurious negative correlation (about -1/(K-1) even
under independence), so ``topica.topic_correlation`` (and pyLDAvis-style topic
graphs built on it) over-reads anti-correlation. This panel offers the compositionally-aware
alternatives:

- ``method="clr"`` (default): correlate the centered-log-ratio transform of theta,
  which removes the closure constraint.
- ``method="partial"``: partial correlation from the clr precision matrix -- direct
  association with every other topic held fixed.
- ``method="eta"``: for CTM/STM, the model's fitted logistic-normal prior
  covariance Sigma over eta (``topic_covariance``, in transformed eta-space relative
  to the reference category), converted to a correlation. This is the model's own
  topic covariance, not an empirical re-estimate of theta or of the posterior means.
- ``method="raw"``: the biased theta correlation, available but labeled as such.

The map is a diverging heatmap centered at zero (the design's requirement), seriated
by hierarchical clustering. Refused for hard/degenerate-theta cluster models.
"""

from __future__ import annotations

import numpy as np

from .base import DIV_CMAP, Panel
from .capability import capabilities
from .terms import _seriate

_METHOD_LABEL = {
    "clr": "clr-transformed correlation (closure-corrected)",
    "partial": "clr partial correlation (others held fixed)",
    "eta": "logistic-normal η-space correlation (model)",
    "raw": "raw θ correlation (compositionally biased)",
}


def _clr(theta):
    """Centered-log-ratio of a (D, K) composition; clips zeros first."""
    t = np.clip(np.asarray(theta, dtype=np.float64), 1e-12, None)
    logt = np.log(t)
    return logt - logt.mean(axis=1, keepdims=True)


def _partial_from_corr(cor):
    """Partial-correlation matrix from a correlation matrix via its precision."""
    prec = np.linalg.pinv(cor)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    p = -prec / np.outer(d, d)
    np.fill_diagonal(p, 1.0)
    return p


class TopicCorrelation(Panel):
    """A compositionally-aware topic-correlation heatmap (clr / partial / η-space / raw)."""

    title = "Topic correlation"

    def __init__(self, model, *, method="clr", seriate=True):
        self.cap = capabilities(model)
        if not self.cap.soft_theta:
            raise ValueError(
                f"{self.cap.name} has degenerate (hard cluster) theta; topic "
                "correlation is not meaningful. Use the topic-similarity heatmap."
            )
        if method not in _METHOD_LABEL:
            raise ValueError(f"method must be one of {sorted(_METHOD_LABEL)}")
        self.method = method
        theta = np.asarray(model.doc_topic, dtype=np.float64)
        k = theta.shape[1]
        self._topics = list(range(k))

        if method == "eta":
            try:
                cov = np.asarray(model.topic_covariance, dtype=np.float64)  # (K-1, K-1) Σ
            except (AttributeError, RuntimeError):
                raise ValueError(
                    f"{self.cap.name} exposes no topic_covariance; method='eta' is for "
                    "CTM/STM. Use method='clr' for a closure-corrected estimate."
                ) from None
            # Σ -> correlation: the model's own logistic-normal topic covariance.
            d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            self._cor = np.nan_to_num(cov / np.outer(d, d))
            np.fill_diagonal(self._cor, 1.0)
            self._topics = list(range(cov.shape[0]))         # reference category dropped
            self.reference = k - 1
        elif method == "raw":
            self._cor = np.nan_to_num(np.corrcoef(theta.T))
            self.reference = None
        else:
            cor = np.nan_to_num(np.corrcoef(_clr(theta).T))
            self._cor = _partial_from_corr(cor) if method == "partial" else cor
            self.reference = None

        from ..analysis import topic_labels
        labels = topic_labels(model)
        self._labels = [labels[t] if t < len(labels) else f"topic_{t}" for t in self._topics]
        # Seriate on distance = 1 - correlation (clipped to a valid metric range).
        if seriate and len(self._topics) >= 3:
            dist = np.clip(1.0 - self._cor, 0.0, 2.0)
            np.fill_diagonal(dist, 0.0)
            dist = 0.5 * (dist + dist.T)
            self._order, _ = _seriate(dist)
        else:
            self._order = list(range(len(self._topics)))

    def to_frame(self):
        import pandas as pd

        order = self._order
        names = [f"{self._topics[i]}: {self._labels[i]}" for i in order]
        return pd.DataFrame(self._cor[np.ix_(order, order)], index=names, columns=names)

    def _figsize(self):
        return (max(4.0, 0.4 * len(self._order) + 1.5),) * 2

    def _draw(self, fig):
        import matplotlib as mpl

        order = self._order
        m = len(order)
        cor = self._cor[np.ix_(order, order)]
        vmax = max(float(np.abs(cor - np.eye(m)).max()), 1e-3)
        # The diagonal is self-correlation (always 1); it carries no information and,
        # left in, saturates the diverging scale and visually dominates the panel.
        # Mask it to a neutral background so the off-diagonal structure reads.
        disp = cor.copy()
        np.fill_diagonal(disp, np.nan)
        cmap = mpl.colormaps[DIV_CMAP].copy()
        cmap.set_bad("#f0f0f0")
        ax = fig.subplots()
        im = ax.imshow(disp, cmap=cmap, vmin=-vmax, vmax=vmax)  # diverging, 0-centered
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        tick = [str(self._topics[i]) for i in order]
        ax.set_xticklabels(tick, fontsize=7, rotation=90)
        ax.set_yticklabels(tick, fontsize=7)
        note = _METHOD_LABEL[self.method]
        if self.reference is not None:
            note += f"; topic {self.reference} is the reference"
        ax.set_title(f"{self.title}\n{note}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
