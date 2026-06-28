"""``dashboard()`` -- the convenience composite.

It introspects the capability descriptor and the arguments you pass to assemble
the applicable panels: the topic-similarity heatmap and a term barchart always,
the coherence frontier when ``texts`` is given, and the covariate effect plot when
a design (``formula``/``data`` or ``X``) is given. ``.to_png()`` stacks the static
panels; ``.to_html()`` writes a self-contained report (the interactive linked
term/heatmap browser plus the static panels embedded as images); ``.to_frame()``
returns the numbers behind every panel.
"""

from __future__ import annotations

import base64
import io

from .base import _require
from .capability import capabilities
from .content import ContentCovariate
from .correlation import TopicCorrelation
from .effect import EffectPlot
from .embedding_map import DocumentMap
from .groups import PrevalenceHeatmap
from .health import TopicHealth
from .inspector import DocumentInspector
from .quality import CoherenceFrontier
from .temporal import TopicsOverTime
from .terms import TermBarchart, TopicSimilarity, term_topic_browser


class Dashboard:
    def __init__(self, model, texts=None, *, corpus=None, formula=None, data=None,
                 X=None, groups=None, timestamps=None, doc_embeddings=None,
                 inspect_doc=None, topic=0, mode="prob"):
        self.model = model
        self.cap = capabilities(model)
        self.panels = {}
        #: ``{panel_name: reason}`` for panels that could not be built (so a skip is
        #: visible, not silently indistinguishable from "not applicable").
        self.skipped = {}
        # Best-effort throughout: a model that cannot support a panel is skipped, not
        # fatal (e.g. SAGE's coherence frontier, whose top_words signature differs).
        self._try("similarity", lambda: TopicSimilarity(model))
        self._try("terms", lambda: TermBarchart(model, topic=topic, mode=mode, texts=texts))
        self._try("health", lambda: TopicHealth(model))
        if texts is not None:
            self._try("frontier", lambda: CoherenceFrontier(model, texts))
        if (formula is not None and data is not None) or X is not None:
            self._try("effect", lambda: EffectPlot(
                model, corpus, formula=formula, data=data, X=X))
        if groups is not None:
            self._try("groups", lambda: PrevalenceHeatmap(model, groups))
        if timestamps is not None:
            self._try("temporal", lambda: TopicsOverTime(model, timestamps))
        if self.cap.soft_theta:  # the compositionally-aware correlation layer, where theta is real
            self._try("correlation", lambda: TopicCorrelation(model))
        if self.cap.content_covariate:  # STM/SAGE: how the lead topic is worded by group
            self._try("content", lambda: ContentCovariate(model, topic=topic))
        if doc_embeddings is not None:  # supplement: the document cloud in 2-D
            self._try("document_map", lambda: DocumentMap(model, doc_embeddings))
        if inspect_doc is not None and texts is not None:  # one document, read closely
            self._try("inspector", lambda: DocumentInspector(model, texts, doc=inspect_doc))

    def _try(self, name, build):
        try:  # a model/design that can't support the panel: skip it, but say so
            self.panels[name] = build()
        except Exception as exc:
            import warnings

            self.skipped[name] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"dashboard: skipped {name!r} panel ({self.skipped[name]})")

    def to_frame(self):
        """A dict of ``{panel_name: DataFrame}``."""
        return {name: p.to_frame() for name, p in self.panels.items()}

    def _order(self):
        seq = ("effect", "frontier", "similarity", "correlation", "content", "groups",
               "temporal", "document_map", "inspector", "health", "terms")
        return [k for k in seq if k in self.panels]

    def _png_bytes(self, panel, *, fmt="png"):
        plt = _require("matplotlib.pyplot", "viz")
        fig = panel._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def to_png(self, path: str | None = None, *, dpi: int = 150):
        """Compose the static panels into one figure with real subfigures, so the
        result stays vector (selectable text) for ``.pdf`` / ``.svg`` output."""
        plt = _require("matplotlib.pyplot", "viz")

        panels = [self.panels[k] for k in self._order()]
        heights = [p._figsize()[1] for p in panels]
        width = max(p._figsize()[0] for p in panels)
        fig = plt.figure(figsize=(width, sum(heights)), constrained_layout=True)
        subfigs = fig.subfigures(len(panels), 1, height_ratios=heights)
        if len(panels) == 1:
            subfigs = [subfigs]
        for sf, panel in zip(subfigs, panels):
            panel._draw(sf)
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return fig

    def to_html(self, path: str | None = None, *, title=None, mode="prob", n=10):
        """A self-contained HTML report: the interactive term/heatmap browser plus
        the static panels embedded as images."""
        chart = term_topic_browser(self.model, n=n, mode=mode)
        # an embed fragment with the Plotly runtime inlined once (self-contained)
        chart_html = chart.to_html(full_html=False, include_plotlyjs="inline")
        blocks = []
        for name in ("effect", "frontier", "correlation", "content", "groups",
                     "temporal", "document_map", "inspector", "health"):
            if name in self.panels:
                b64 = base64.b64encode(self._png_bytes(self.panels[name])).decode("ascii")
                blocks.append(f'<h2>{name}</h2><img src="data:image/png;base64,{b64}">')
        title = title or f"topica report — {self.cap.name}"
        html = (
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;}"
            "img{max-width:100%;}</style></head><body>"
            f"<h1>{title}</h1>{chart_html}{''.join(blocks)}</body></html>"
        )
        if path is not None:
            with open(path, "w") as fh:
                fh.write(html)
        return html

    def __repr__(self):
        return f"Dashboard({self.cap.name}, panels={list(self.panels)})"


def dashboard(model, texts=None, **kwargs) -> Dashboard:
    """Assemble the applicable panels for a model into one report."""
    return Dashboard(model, texts, **kwargs)
