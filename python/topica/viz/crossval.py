"""Panel: cross-validation results for :func:`topica.cross_validate`.

One panel, two shapes, switched on ``result.kind``:

- **topic path** — the per-fold distribution of each held-out quality metric
  (perplexity / coherence / exclusivity), each fold a point with the macro mean
  and +/- std overlaid, so a reader sees both the level and the fold-to-fold
  spread the aggregate hides.
- **supervised path** — the out-of-fold reliability plot: binned observed vs
  predicted with the 45-degree line (built from ``result.calibration_table``),
  beside the per-fold RMSE / R-squared spread.

Like every :class:`~topica.viz.base.Panel`, it also exposes ``.to_frame()`` (the
numbers) and ``.to_png()`` (the figure).
"""

from __future__ import annotations

from .base import Panel

# Shared palette, consistent with the other panels.
_C_POINT = "#4C72B0"
_C_MEAN = "#C44E52"
_C_REF = "0.5"

# The metrics the topic path reports, in display order, with a "higher is better?"
# hint used only for the axis label (never to reorder or rescore).
_TOPIC_METRICS = [
    ("perplexity", "held-out perplexity", "lower better"),
    ("coherence", "coherence", "higher better"),
    ("exclusivity", "exclusivity", "higher better"),
]


class CrossValidationPlot(Panel):
    """Per-fold cross-validation results, produced by :func:`topica.viz.plot_cv`.

    Renders the topic-path metric distributions or the supervised-path OOF
    calibration + error spread, depending on ``result.kind``.
    """

    title = "Cross-validation"

    def __init__(self, result):
        """
        Parameters
        ----------
        result : CrossValResult
            The output of :func:`topica.cross_validate`.
        """
        self._result = result
        self._kind = getattr(result, "kind", "topic")

    # --- data ---------------------------------------------------------------
    def to_frame(self):
        """The per-fold table behind the figure (``result.to_frame()``)."""
        return self._result.to_frame()

    def _topic_metric_values(self):
        """Available metrics -> list of per-fold values (skipping folds that lack it)."""
        out = []
        for key, label, hint in _TOPIC_METRICS:
            vals = [
                r[key] for r in self._result.per_fold
                if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
            ]
            if vals:
                out.append((key, label, hint, vals))
        return out

    # --- layout -------------------------------------------------------------
    def _has_covariate(self):
        return getattr(self._result, "covariate_stability", None) is not None

    def _figsize(self):
        if self._kind == "supervised":
            return (9.0, 4.0)
        n = max(1, len(self._topic_metric_values()))
        h = 4.0 + (3.0 if self._has_covariate() else 0.0)
        return (3.6 * n, h)

    def _figure(self, *, figsize=None, **kwargs):
        # The covariate band uses subfigures with differing column counts, where
        # figure.tight_layout() cannot solve and warns. Use the constrained-layout
        # engine for that case (subfigure-aware); the plain paths keep tight_layout.
        if self._kind != "supervised" and self._has_covariate():
            from .base import _require

            plt = _require("matplotlib.pyplot", "viz")
            fig = plt.figure(figsize=figsize or self._figsize(), layout="constrained")
            self._draw(fig, **kwargs)
            return fig
        return super()._figure(figsize=figsize, **kwargs)

    def _draw(self, fig, **kwargs):
        if self._kind == "supervised":
            self._draw_supervised(fig)
        elif self._has_covariate():
            # Two stacked bands: the held-out metric distributions on top, the
            # covariate-effect stability below (the reason a covariate model was run).
            top, bottom = fig.subfigures(2, 1, height_ratios=[4.0, 3.0])
            self._draw_topic(top)
            self._draw_covariate(bottom)
        else:
            self._draw_topic(fig)
        fig.suptitle(
            f"{self.title}: {len(self._result.per_fold)} folds, "
            f"strategy={self._result.folds.strategy}",
            fontsize=9, y=1.02,
        )

    # --- topic path ---------------------------------------------------------
    def _draw_topic(self, fig):
        import numpy as np

        metrics = self._topic_metric_values()
        if not metrics:
            ax = fig.subplots(1, 1)
            ax.text(0.5, 0.5, "no per-fold metrics to plot", ha="center", va="center")
            ax.set_axis_off()
            return
        axes = fig.subplots(1, len(metrics), squeeze=False)[0]
        agg = self._result.aggregate
        for ax, (key, label, hint, vals) in zip(axes, metrics):
            vals = np.asarray(vals, dtype=float)
            # Jitter the folds horizontally so overlapping values stay legible.
            x = np.linspace(-0.18, 0.18, vals.size) if vals.size > 1 else np.zeros(1)
            ax.scatter(x, vals, s=42, color=_C_POINT, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.6, label="fold")
            a = agg.get(key)
            mean = float(a["mean"]) if a else float(vals.mean())
            std = float(a["std"]) if a else float(vals.std())
            ax.axhline(mean, color=_C_MEAN, lw=1.8, zorder=2,
                       label=f"mean {mean:.4g}")
            ax.axhspan(mean - std, mean + std, color=_C_MEAN, alpha=0.12, zorder=1)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(f"{label}\n({hint})", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="best")

    # --- covariate-effect stability (keyATM / DMR / GDMR) -------------------
    def _draw_covariate(self, fig):
        import numpy as np

        cs = self._result.covariate_stability
        per = cs["per_feature"]
        names = list(per.keys())
        sign = [per[n]["sign_agreement"] for n in names]
        corr = [per[n]["magnitude_correlation"] for n in names]

        axes = fig.subplots(1, 2, squeeze=False)[0]
        y = np.arange(len(names))

        # (1) sign-agreement — a probability in [0, 1]; 0.5 is chance (dashed line).
        ax = axes[0]
        ax.barh(y, np.nan_to_num(sign, nan=0.0), color=_C_POINT, alpha=0.85,
                edgecolor="white")
        ax.axvline(0.5, color=_C_REF, ls="--", lw=1.0, label="chance (0.5)")
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("sign-agreement", fontsize=8)
        ax.set_title("Covariate-effect sign-agreement", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower right")

        # (2) magnitude correlation — Pearson r in [-1, 1]; 0 (no relation) at center.
        ax = axes[1]
        vals = np.nan_to_num(corr, nan=0.0)
        colors = [_C_MEAN if v < 0 else "#55A868" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color=_C_REF, lw=1.0)
        # A NaN correlation (a covariate whose lambda has no variance) has no bar; say so.
        for yi, c in zip(y, corr):
            if not np.isfinite(c):
                ax.text(0.02, yi, "n/a (no variance)", fontsize=6, va="center",
                        color=_C_REF)
        ax.set_xlim(-1, 1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("magnitude correlation (Pearson r)", fontsize=8)
        ax.set_title("Covariate-effect magnitude correlation", fontsize=8)
        ax.tick_params(labelsize=7)

        sub = (f"covariate-effect stability (NOT predictive coverage) - "
               f"{cs['n_pairs']} fold pairs, feature-macro headline "
               f"{cs['sign_agreement']:.2f} / {cs['magnitude_correlation']:.2f}")
        if cs.get("partial_alignment"):
            sub += f"  (partial: min {cs['topics_compared']['min']}/{cs['n_topics']} topics)"
        # Caption below the band (a suptitle here would collide with the metric row).
        fig.supxlabel(sub, fontsize=7.5)

    # --- supervised path ----------------------------------------------------
    def _draw_supervised(self, fig):
        import numpy as np

        axes = fig.subplots(1, 2, squeeze=False)[0]

        # (1) OOF reliability: binned observed vs predicted, with the 45-degree line.
        ax = axes[0]
        tbl = self._result.calibration_table
        if tbl is not None and len(tbl):
            pred = np.asarray(tbl["mean_pred"], dtype=float)
            obs = np.asarray(tbl["mean_obs"], dtype=float)
            n = np.asarray(tbl["n"], dtype=float)
            sizes = 30 + 220 * (n / n.max()) if n.max() > 0 else 60
            lo = float(min(pred.min(), obs.min()))
            hi = float(max(pred.max(), obs.max()))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--",
                    color=_C_REF, lw=1.2, label="perfect (y = x)", zorder=1)
            ax.scatter(pred, obs, s=sizes, color=_C_POINT, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.6, label="bin (size = n)")
            a = self._result.aggregate
            slope = a.get("calibration_slope")
            intc = a.get("calibration_intercept")
            if slope is not None and np.isfinite(slope):
                xs = np.array([lo - pad, hi + pad])
                ax.plot(xs, intc + slope * xs, color=_C_MEAN, lw=1.6, zorder=2,
                        label=f"fit (slope {slope:.2f})")
            ax.set_xlabel("predicted (out-of-fold)", fontsize=8)
            ax.set_ylabel("observed", fontsize=8)
            ax.set_title("OOF calibration", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "no calibration table\n(constant or too few predictions)",
                    ha="center", va="center", fontsize=8)
            ax.set_axis_off()

        # (2) Per-fold error spread: RMSE and R-squared, each fold a point + mean line.
        ax = axes[1]
        self._draw_fold_spread(ax)

    def _draw_fold_spread(self, ax):
        import numpy as np

        specs = [("rmse", "RMSE", _C_POINT), ("r2", "R2", "#55A868")]
        available = []
        for key, label, color in specs:
            vals = [r[key] for r in self._result.per_fold
                    if isinstance(r.get(key), float) and np.isfinite(r[key])]
            if vals:
                available.append((label, color, np.asarray(vals, dtype=float)))
        if not available:
            ax.text(0.5, 0.5, "no per-fold error metrics", ha="center", va="center",
                    fontsize=8)
            ax.set_axis_off()
            return
        for i, (label, color, vals) in enumerate(available):
            x = i + (np.linspace(-0.14, 0.14, vals.size) if vals.size > 1 else np.zeros(1))
            ax.scatter(x, vals, s=42, color=color, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.6)
            m = float(vals.mean())
            ax.hlines(m, i - 0.25, i + 0.25, color=_C_MEAN, lw=1.8, zorder=2)
            ax.annotate(f"{m:.3g}", (i, m), textcoords="offset points", xytext=(18, 0),
                        fontsize=7, va="center", color=_C_MEAN)
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels([label for label, _, _ in available], fontsize=8)
        ax.set_xlim(-0.5, len(available) - 0.5)
        ax.set_ylabel("per-fold value", fontsize=8)
        ax.set_title("Per-fold error spread", fontsize=8)
        ax.tick_params(labelsize=7)
