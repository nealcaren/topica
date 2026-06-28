"""Panel 2: the covariate effect plot -- *the* results figure for STM papers.

A per-topic forest plot of one covariate's effect on topic prevalence, with the
method-of-composition confidence intervals, and the guardrails the design
calls for:

- CIs are drawn only where a theta posterior exists. For an embedding / cluster
  model (``theta_posterior == "none"``) the panel draws point estimates and says
  so; pass ``method="bootstrap"`` to get intervals there.
- A topic the bootstrap flagged ``reliable == False`` is rendered as a ghosted
  point estimate, not a band (its SE is ``NaN``).
- The uncertainty is labeled for what it is: a Gibbs model's Dirichlet-conditional
  (within-document) uncertainty is not a logistic-normal posterior.
"""

from __future__ import annotations

from .base import Panel
from .capability import capabilities

_POSTERIOR_LABEL = {
    "logistic_normal": "logistic-normal posterior (method of composition)",
    "dirichlet": "Dirichlet-conditional, within-document (method of composition)",
    "none": "bootstrap (refit) intervals",
}


class EffectPlot(Panel):
    """One covariate's effect on each topic's prevalence, with CIs."""

    title = "Covariate effect on topic prevalence"

    def __init__(self, model, corpus=None, *, formula=None, data=None, X=None,
                 feature=None, feature_names=None, method="composition",
                 nsims=50, n_boot=200, ci=0.95, seed=0):
        from ..stm import estimate_effect
        from ..effects import standard_errors
        from ..analysis import topic_labels

        self.cap = capabilities(model)
        self.ci_level = ci
        self.note = ""

        no_posterior = method == "composition" and self.cap.theta_posterior == "none"
        if no_posterior:
            # No theta posterior to compose: point estimates only, say so.
            effects = estimate_effect(
                model.doc_topic, X=X, formula=formula, data=data,
                feature_names=feature_names, ci=ci,
            )
            self.has_ci = False
            self.uncertainty = "none"
            degenerate = "" if self.cap.soft_theta else (
                " — and on near-degenerate (hard cluster) theta, so even the points "
                "may mislead"
            )
            self.note = (
                f"{self.cap.name} has no theta posterior; showing point estimates"
                f"{degenerate}. Pass method='bootstrap' for intervals."
            )
        else:
            effects = standard_errors(
                model, corpus, of="effect", method=method, formula=formula,
                data=data, X=X, feature_names=feature_names, nsims=nsims,
                n_boot=n_boot, ci=ci, seed=seed,
            )
            self.has_ci = True
            self.uncertainty = "bootstrap" if method == "bootstrap" else self.cap.theta_posterior

        self._effects = {e.topic: e for e in effects}
        self._fnames = effects[0].feature_names
        self._labels = topic_labels(model)
        # default to the first non-intercept feature
        self.feature = feature if feature is not None else self._default_feature()

    def _default_feature(self):
        for f in self._fnames:
            if f.lower() not in ("intercept", "(intercept)", "const"):
                return f
        return self._fnames[0]

    def _feature_index(self):
        try:
            return self._fnames.index(self.feature)
        except ValueError:
            raise ValueError(
                f"feature {self.feature!r} not in {self._fnames}"
            ) from None

    def to_frame(self):
        import numpy as np
        import pandas as pd

        j = self._feature_index()
        rows = []
        for t in sorted(self._effects):
            e = self._effects[t]
            se = float(e.se[j])
            reliable = not np.isnan(se)
            rows.append({
                "topic": t,
                "label": self._labels[t] if t < len(self._labels) else f"topic_{t}",
                "feature": self.feature,
                "coef": float(e.coef[j]),
                "se": se,
                "ci_low": float(e.ci_low[j]),
                "ci_high": float(e.ci_high[j]),
                "reliable": reliable,
            })
        return pd.DataFrame(rows)

    def _figsize(self):
        return (6.5, max(2.5, 0.42 * len(self._effects) + 1.2))

    def _draw(self, fig, *, sort=True):
        import numpy as np

        df = self.to_frame()
        if sort:
            df = df.sort_values("coef")
        k = len(df)
        ax = fig.subplots()
        y = np.arange(k)
        # A single neutral color: position relative to the zero line already encodes
        # sign, so recoloring by sign would be redundant, colorblind-fragile, and
        # (for political-text results) value-laden. Reliability is what we encode:
        # a solid marker + band for reliable estimates, an open marker for ghosted.
        color = "#4C72B0"
        for i, (_, r) in enumerate(df.iterrows()):
            draw_band = self.has_ci and r["reliable"]
            if draw_band:
                ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=2.2, alpha=0.85,
                        solid_capstyle="round", zorder=2)
                ax.plot(r["coef"], i, "o", color=color, ms=6, zorder=3)
            else:
                # ghosted point estimate: open marker, no band
                ax.plot(r["coef"], i, "o", mfc="white", mec=color, ms=6, zorder=3)
        ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{int(r["topic"])}: {r["label"]}' for _, r in df.iterrows()],
                           fontsize=8)
        ax.set_xlabel(f"effect of {self.feature} on topic proportion")
        pct = int(round(self.ci_level * 100))
        unc = _POSTERIOR_LABEL.get(self.uncertainty, self.uncertainty)
        title = f"{self.title}\n{pct}% CI — {unc}" if self.has_ci else \
            f"{self.title}\npoint estimates ({self.note})"
        ax.set_title(title, fontsize=9)
