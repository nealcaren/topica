"""Per-model capability descriptor for the visualization toolkit.

Views read a uniform analysis surface (``topic_word``, ``doc_topic``,
``vocabulary``, ``num_topics``, optional ``labels`` / ``topic_names``), but the
*statistics and encodings* they apply switch on this small descriptor, mirroring
``topica.effects.model_family``. This is what keeps the toolkit disciplined: a c-TF-IDF
``topic_word`` is not a probability, a cluster model's ``doc_topic`` is degenerate,
and an effect-plot CI must be refused where there is no posterior to draw it from.
"""

from __future__ import annotations

from dataclasses import dataclass

# The embedding-cluster models: their ``topic_word`` is class-based TF-IDF (not a
# word distribution) and their ``doc_topic`` is not a mixed-membership simplex.
_CLUSTER = {"BERTopic", "Top2Vec"}
# Models whose ``doc_topic`` is hard/degenerate (one topic per document) or
# absent, so theta-bars and theta-correlation are meaningless: the cluster
# models and GSDMM (degenerate), plus HLDA (a topic *tree*, no D x K theta) and
# DTM (time-sliced, no static theta). Without this they would report
# ``soft_theta == True`` and the theta panels would hit AttributeError.
_DEGENERATE_THETA = {"BERTopic", "Top2Vec", "GSDMM", "HLDA", "DTM"}
# Content-covariate models: ``topic_word`` is covariate-conditional.
_CONTENT_COVARIATE = {"STM", "SAGE"}


@dataclass(frozen=True)
class Capabilities:
    """What a fitted model supports, so panels disable or relabel off it."""

    name: str
    prob_simplex_words: bool   # topic_word rows are P(w|t)? False for c-TF-IDF
    soft_theta: bool           # real D x K simplex vs hard/degenerate cluster theta
    has_outliers: bool         # a -1 noise layer is present (HDBSCAN cluster models)
    content_covariate: bool    # STM/SAGE: topic_word is covariate-conditional
    k_fixed: bool              # False for HDP (K random, many near-zero-mass topics)
    theta_posterior: str       # 'logistic_normal' | 'dirichlet' | 'none'

    @property
    def word_weight_label(self) -> str:
        """How to label the topic-word bars."""
        return "P(w | topic)" if self.prob_simplex_words else "c-TF-IDF weight"

    @property
    def word_modes(self) -> list:
        """Term-weighting modes that are valid for this model. The lift / FREX /
        relevance / score modes assume ``topic_word`` is a probability, so they are
        dropped for the c-TF-IDF models (where they would double-count exclusivity).
        """
        if self.prob_simplex_words:
            return ["prob", "frex", "lift", "relevance", "score"]
        return ["prob"]


def capabilities(model) -> Capabilities:
    """Infer the capability descriptor for a fitted model."""
    import numpy as np

    from ..effects import model_family

    name = type(model).__name__
    has_outliers = False
    labels = getattr(model, "labels", None)
    if labels is not None:
        try:
            has_outliers = int(np.min(np.asarray(list(labels)))) < 0
        except Exception:
            has_outliers = False

    return Capabilities(
        name=name,
        prob_simplex_words=name not in _CLUSTER,
        soft_theta=name not in _DEGENERATE_THETA,
        has_outliers=has_outliers,
        content_covariate=name in _CONTENT_COVARIATE,
        k_fixed=name != "HDP",
        # The same router the standard-error facility uses, so the effect panel's
        # CI gating agrees with what method-of-composition can actually draw.
        theta_posterior=model_family(model),
    )
