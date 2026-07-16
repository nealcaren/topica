"""Evaluating and validating a fitted model.

Everything that answers "is this a good fit?": topic-quality metrics (coherence,
exclusivity, diversity), the human intrusion tests, the one-call per-topic
:func:`diagnostics` table, residual and stability checks, held-out likelihood,
MCMC convergence diagnostics for the Gibbs samplers, and the estimator-contract
conformance check.
"""

from __future__ import annotations

from .coherence import (
    coherence,
    coherence_ci,
    CoherenceCI,
    semantic_coherence,
    topic_diversity,
    topic_semantic_diversity,
    exclusivity,
    word_intrusion,
    document_intrusion,
)
from .validation import (
    diagnostics,
    topic_table,
    quality_frontier,
    check_residuals,
    document_residuals,
    flag_topics,
    bootstrap_stability,
    align_topics,
    topic_stability,
    perplexity,
    make_heldout,
    eval_heldout,
    Heldout,
    HeldoutResult,
)
from .mcmc import (
    mcmc_diagnostics,
    effective_sample_size,
    autocorrelation,
    integrated_autocorr_time,
    McmcDiagnostics,
)
from .conformance import check_conformance
from .analysis import stop_reason

__all__ = [
    # topic quality
    "coherence",
    "coherence_ci",
    "CoherenceCI",
    "semantic_coherence",
    "topic_diversity",
    "topic_semantic_diversity",
    "exclusivity",
    "word_intrusion",
    "document_intrusion",
    # tables / residuals / stability
    "diagnostics",
    "topic_table",
    "quality_frontier",
    "check_residuals",
    "document_residuals",
    "flag_topics",
    "bootstrap_stability",
    "align_topics",
    "topic_stability",
    "stop_reason",
    "check_conformance",
    # held-out likelihood
    "perplexity",
    "make_heldout",
    "eval_heldout",
    "Heldout",
    "HeldoutResult",
    # MCMC convergence
    "mcmc_diagnostics",
    "effective_sample_size",
    "autocorrelation",
    "integrated_autocorr_time",
    "McmcDiagnostics",
]
