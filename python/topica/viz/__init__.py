"""topica.viz -- a disciplined, manuscript-first visualization toolkit.

Each view is a :class:`~topica.viz.base.Panel`: it reads a model's analysis
surface plus a per-model capability descriptor and exposes three renderers --
``.to_frame()`` (the numbers, always), ``.to_png()`` (matplotlib, for papers), and
``.to_html()`` (Plotly, for the few views interaction genuinely helps). The
statistics and labels switch on the descriptor, so a c-TF-IDF ``topic_word`` is
never mislabeled a probability and a confidence interval is never drawn where
there is no posterior to draw it from.

    import topica.viz as viz
    viz.coherence_frontier(model, texts).to_png("quality.png")
    viz.effect_plot(model, corpus, formula="~ year", data=meta).to_png("effects.pdf")
    viz.term_barchart(model, topic=3, mode="frex").to_frame()
"""

from __future__ import annotations

from .base import Panel
from .capability import Capabilities, capabilities
from .quality import CoherenceFrontier, SearchK
from .effect import EffectPlot
from .prevalence import PrevalencePlot
from .terms import TermBarchart, TopicSimilarity, term_topic_browser
from .health import TopicHealth
from .groups import PrevalenceHeatmap
from .temporal import TopicsOverTime
from .correlation import TopicCorrelation
from .embedding_map import DocumentMap
from .inspector import DocumentInspector
from .content import ContentCovariate
from .dashboard import Dashboard, dashboard
from .permtest import PermutationTestPlot


def coherence_frontier(model, texts=None, *, n=10, coherence_type=None) -> CoherenceFrontier:
    """Per-topic coherence vs exclusivity (size = prevalence)."""
    return CoherenceFrontier(model, texts, n=n, coherence_type=coherence_type)


def search_k(rows) -> SearchK:
    """Coherence / exclusivity / perplexity across K. Pass ``topica.search_k(...)`` rows."""
    return SearchK(rows)


def effect_plot(model, corpus=None, **kwargs) -> EffectPlot:
    """One covariate's effect on each topic's prevalence, with well-calibrated CIs."""
    return EffectPlot(model, corpus, **kwargs)


def predicted_prevalence_plot(model, *, results, ci_level=0.95) -> PrevalencePlot:
    """Predicted topic prevalence at covariate values, with simulation-based CIs.

    Pass the output of :func:`topica.predicted_prevalence` as ``results``.
    Renders a forest plot for ``at`` / ``contrast`` mode and a curve-and-band
    plot for ``continuous`` mode.
    """
    return PrevalencePlot(model, results=results, ci_level=ci_level)


def term_barchart(model, *, topic, mode="prob", n=10, texts=None, error_bars=False, **kwargs) -> TermBarchart:
    """Top words of one topic as a weighted bar chart (mode = prob/frex/lift/...)."""
    return TermBarchart(model, topic=topic, mode=mode, n=n, texts=texts,
                        error_bars=error_bars, **kwargs)


def topic_similarity(model, **kwargs) -> TopicSimilarity:
    """Seriated K x K topic-similarity heatmap (the disciplined pyLDAvis overview)."""
    return TopicSimilarity(model, **kwargs)


def topic_health(model, *, min_mass_frac=0.01, dup_threshold=0.9) -> TopicHealth:
    """Dead (near-zero-mass) and duplicate (near-identical φ) topics, flagged."""
    return TopicHealth(model, min_mass_frac=min_mass_frac, dup_threshold=dup_threshold)


def prevalence_heatmap(model, groups, **kwargs) -> PrevalenceHeatmap:
    """Mean topic prevalence across the levels of a grouping variable."""
    return PrevalenceHeatmap(model, groups, **kwargs)


def topics_over_time(model, timestamps, **kwargs) -> TopicsOverTime:
    """Per-topic prevalence trajectories as small multiples (with optional CIs)."""
    return TopicsOverTime(model, timestamps, **kwargs)


def topic_correlation(model, *, method="clr", **kwargs) -> TopicCorrelation:
    """Compositionally-aware topic correlation (clr / partial / η-space / raw), 0-centered map."""
    return TopicCorrelation(model, method=method, **kwargs)


def document_map(model, doc_embeddings=None, *, method="pca", **kwargs) -> DocumentMap:
    """A 2-D projection of the document cloud (PCA / UMAP / t-SNE), via the Rust core."""
    return DocumentMap(model, doc_embeddings, method=method, **kwargs)


def document_inspector(model, texts, *, doc, **kwargs) -> DocumentInspector:
    """One document: its θ mixture, words shaded by topic attribution, neighbors."""
    return DocumentInspector(model, texts, doc=doc, **kwargs)


def content_covariate(model, *, topic, n=10) -> ContentCovariate:
    """One topic's wording across STM/SAGE content-covariate groups, p(w|topic,group)."""
    return ContentCovariate(model, topic=topic, n=n)


def permutation_test_plot(results, *, covariate_name=None) -> PermutationTestPlot:
    """Observed effect vs permutation null, one subplot per topic.

    Pass the output of :func:`topica.permutation_test` as ``results``.
    """
    return PermutationTestPlot(results, covariate_name=covariate_name)


__all__ = [
    "Panel",
    "Capabilities",
    "capabilities",
    "CoherenceFrontier",
    "SearchK",
    "EffectPlot",
    "PrevalencePlot",
    "TermBarchart",
    "TopicSimilarity",
    "TopicHealth",
    "PrevalenceHeatmap",
    "TopicsOverTime",
    "TopicCorrelation",
    "DocumentMap",
    "DocumentInspector",
    "ContentCovariate",
    "coherence_frontier",
    "search_k",
    "effect_plot",
    "predicted_prevalence_plot",
    "term_barchart",
    "topic_similarity",
    "topic_health",
    "prevalence_heatmap",
    "topics_over_time",
    "topic_correlation",
    "document_map",
    "document_inspector",
    "content_covariate",
    "term_topic_browser",
    "Dashboard",
    "dashboard",
    "PermutationTestPlot",
    "permutation_test_plot",
]
