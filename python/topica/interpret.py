"""Reading, labeling, and summarizing the topics of a fitted model.

Top-word rankings (prob / FREX / lift / relevance / MMR), per-topic labels
(hand-set or LLM-generated), representative documents, distinguishing words,
topic-correlation structure, and one-call model overviews.
"""

from __future__ import annotations

from .validation import (
    frex,
    mmr,
    label_topics,
    relevance,
    find_thoughts,
    find_thoughts_html,
    topic_correlation,
    topic_dendrogram,
    TopicDendrogram,
)
from .analysis import (
    topic_info,
    topic_sizes,
    topic_labels,
    set_topic_labels,
    representative_docs,
    contrastive_topics,
)
from .keywords import fighting_words, top_fighting_words
from .labeling import llm_topic_labels, llm_backend, topic_label_prompts

__all__ = [
    "frex",
    "mmr",
    "label_topics",
    "relevance",
    "find_thoughts",
    "find_thoughts_html",
    "topic_correlation",
    "topic_dendrogram",
    "TopicDendrogram",
    "topic_info",
    "topic_sizes",
    "topic_labels",
    "set_topic_labels",
    "representative_docs",
    "contrastive_topics",
    "fighting_words",
    "top_fighting_words",
    "llm_topic_labels",
    "llm_backend",
    "topic_label_prompts",
    "summary",
    "report",
]


def summary(model, topn=8):
    """A human-readable overview of a fitted model (à la tomotopy's ``summary``).

    Returns a multi-line string: the model's repr, its key scalar attributes
    (num_topics, concentrations, etc.), the vocabulary size, and the top words of
    each topic. Pass to ``print``. For models whose ``top_words`` needs extra
    arguments (``DTM`` by time, ``SAGE`` by group) the per-topic word lists are
    omitted.
    """
    lines = [repr(model)]
    for attr in ("num_topics", "num_times", "num_groups", "alpha", "gamma",
                 "sigma2", "bound"):
        try:
            value = getattr(model, attr)
        except Exception:
            continue
        if not callable(value):
            lines.append(f"  {attr}: {value}")
    try:
        lines.append(f"  vocab_size: {len(model.vocabulary)}")
    except Exception:
        pass
    try:
        tops = model.top_words(topn)
        if isinstance(tops, list) and tops and isinstance(tops[0], list):
            for i, words in enumerate(tops):
                lines.append(f"  topic {i}: " + " ".join(w for w, _ in words))
    except Exception:
        pass
    return "\n".join(lines)


def report(model, topn=8):
    """One-call overview of a fitted model. Alias for :func:`summary`.

    ``report`` reads like a verb, so ``report(model)`` is a natural thing to
    try; it returns the same multi-line overview as ``summary(model)``. The
    richer analysis surface (:func:`topic_info`, :func:`topic_sizes`,
    :func:`representative_docs`, :func:`~topica.effects.topics_over_time`,
    :func:`~topica.viz.plot_report`, …) lives across ``topica.interpret``,
    ``topica.effects``, and ``topica.viz``.
    """
    return summary(model, topn=topn)
