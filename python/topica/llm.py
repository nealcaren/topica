"""LLM-based topic evaluation, grouped under ``topica.llm``.

Everything here is ``llm-bounded``: it calls an external language model, so unlike the
rest of topica's diagnostics (``coherence``, ``topic_diversity``,
``topic_semantic_diversity``, ...) the results are *not* bit-reproducible. Use
``temperature=0`` (or ``n_samples>1`` with aggregation) for stability, and a capable
model — small models are fine for rating but fail the harder intrusion/labeling tasks.

The namespace collects the suite so it reads as a family and signals the shared
``llm-bounded`` contract at the call site::

    call = topica.llm.backend("openrouter/meta-llama/llama-3.3-70b-instruct", temperature=0)
    topica.llm.coherence(model, call=call)        # per-topic 1-3 rating (Stammbach et al. 2023)
    topica.llm.intrusion(model, call=call)        # LLM picks the planted intruder -> accuracy
    topica.llm.select_k(models, docs, call=call)  # number of topics by doc-label purity

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
    LLM_EVAL_PROMPTS as PROMPTS,
)
from .labeling import llm_backend as backend

__all__ = ["coherence", "intrusion", "select_k", "backend", "PROMPTS"]
