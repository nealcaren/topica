"""LLM-based topic evaluation, grouped under ``topica.llm``.

Everything here is ``llm-bounded``: it calls an external language model, so unlike the
rest of topica's diagnostics (``coherence``, ``topic_diversity``,
``topic_semantic_diversity``, ...) the results are *not* bit-reproducible. Use
``temperature=0`` (or ``n_samples>1`` with aggregation) for stability, and a capable
model — small models are fine for rating but fail the harder intrusion/labeling tasks.

The namespace collects the suite so it reads as a family and signals the shared
``llm-bounded`` contract at the call site::

    backend = topica.llm.backend("openrouter/meta-llama/llama-3.3-70b-instruct", temperature=0)
    topica.llm.coherence(model, backend=backend)        # per-topic 1-3 rating (Stammbach et al. 2023)
    topica.llm.intrusion(model, backend=backend)        # LLM picks the planted intruder -> accuracy
    topica.llm.select_k(models, docs, backend=backend)  # number of topics by doc-label purity
    # Tan & D'Souza (2025) multi-dimensional suite:
    topica.llm.outlier(model, backend=backend)          # unsupervised semantic-outlier words
    topica.llm.repetitiveness(model, backend=backend)   # is coherence just redundancy?
    topica.llm.diversity(model, backend=backend)        # pairwise cross-topic distinctiveness
    topica.llm.alignment(model, docs, backend=backend)  # irrelevant words / missing themes vs docs
    topica.llm.adversarial(model, backend=backend)      # gold-free capability self-check

The implementations live in :mod:`topica.coherence` (the metrics) and
:mod:`topica.labeling` (the backend); this module is the curated public surface.
``topica.llm.backend`` is the same callable constructor as the top-level
``topica.llm_backend`` (kept because it is also the bring-your-own-model adapter for
:class:`topica.TopicGPT` and :func:`topica.label_topics`).
"""

from __future__ import annotations

from .coherence import (
    llm_coherence as coherence,
    llm_intrusion as intrusion,
    llm_select_k as select_k,
    llm_outlier as outlier,
    llm_repetitiveness as repetitiveness,
    llm_diversity as diversity,
    llm_adversarial as adversarial,
    llm_alignment as alignment,
    LLM_EVAL_PROMPTS as PROMPTS,
)
from .labeling import llm_backend as backend

__all__ = [
    "coherence", "intrusion", "select_k", "outlier", "repetitiveness",
    "diversity", "adversarial", "alignment", "backend", "PROMPTS",
]
