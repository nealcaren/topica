"""Estimator conformance: registry, contract definition, and check helper.

Every topica estimator must expose a uniform interface organized into three tiers.
This module encodes that contract, the principled exemptions from it, and the
known gaps (fixable drift tracked as a burn-down worklist). It also provides
``check_conformance(model_or_class) -> list[str]``, which callers and the
conformance test use to verify a model against its required tier.

The three tiers
---------------
Tier 0  (floor, every estimator):
    ``fit(data, ...)`` — with the canonical iteration kwarg ``iters`` where the
    model is iterative.
    Properties/methods: ``topic_word``, ``doc_topic``, ``vocabulary``,
    ``num_topics``, ``topic_names``, ``doc_names``, ``top_words``,
    ``coherence``, ``save``, ``load``, ``fit_history``, ``converged``.

Tier 1  (generative models, i.e. ``model_family != "none"``):
    ``transform(docs) -> (n, num_topics)``.

Tier 2  (family-specific):
    ``model_family == "dirichlet"``  ->  ``alpha``, ``theta_draws``,
    ``doc_lengths``.
    ``model_family == "logistic_normal"``  ->  ``eta_mean``, ``eta_cov``.

Exemptions
----------
Some requirements are principled exemptions that will never be closed, because
the model's statistics differ structurally from the requirement. These are
recorded in :data:`EXEMPT` as ``(model_name, requirement) -> reason``.

Known gaps
----------
:data:`KNOWN_GAPS` records temporary non-conformance:
``(model_name, requirement) -> phase_note``. The conformance test treats these
as expected failures so the suite is green today while this map is the
burn-down worklist. The program ends when ``KNOWN_GAPS == {}``.
"""

from __future__ import annotations

import inspect

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry: (name, factory_callable, expected_model_family)
# Factories are zero/low-arg lambdas; construction follows the same patterns
# used in tests/test_standard_errors.py::_FAMILY_REGISTRY.

import topica as _topica

REGISTRY: list[tuple[str, object, str]] = [
    # collapsed-Gibbs / Dirichlet doc-topic posterior
    ("LDA",          lambda: _topica.models.LDA(2),                                     "dirichlet"),
    ("DMR",          lambda: _topica.models.DMR(2),                                     "dirichlet"),
    ("GDMR",         lambda: _topica.models.GDMR(2, degrees=[2]),                        "dirichlet"),
    ("SAGE",         lambda: _topica.models.SAGE(2),                                    "dirichlet"),
    ("PA",           lambda: _topica.models.PA(num_super=2, num_sub=4),                 "dirichlet"),
    ("PT",           lambda: _topica.models.PT(num_topics=2, num_pseudo=10),            "dirichlet"),
    ("HDP",          lambda: _topica.models.HDP(),                                      "dirichlet"),
    ("LabeledLDA",   lambda: _topica.models.LabeledLDA(),                               "dirichlet"),
    ("SupervisedLDA",lambda: _topica.models.SupervisedLDA(num_topics=2),                "dirichlet"),
    ("HLDA",         lambda: _topica.models.HLDA(),                                     "none"),
    ("DTM",          lambda: _topica.models.DTM(2),                                     "none"),
    ("KeyATM",       lambda: _topica.models.KeyATM({"a": ["x"]}, num_topics=2),         "dirichlet"),
    ("SeededLDA",    lambda: _topica.models.SeededLDA({"a": ["x"], "b": ["y"]}),        "dirichlet"),
    ("GSDMM",        lambda: _topica.models.GSDMM(num_topics=5),                        "none"),
    # logistic-normal (variational eta posterior)
    ("STM",          lambda: _topica.models.STM(2),                                     "logistic_normal"),
    ("CTM",          lambda: _topica.models.CTM(2),                                     "logistic_normal"),
    ("STS",          lambda: _topica.models.STS(2),                                     "logistic_normal"),
    # neural / embedding-based — no theta posterior
    ("ETM",          lambda: _topica.models.ETM(2),                                     "none"),
    ("DETM",         lambda: _topica.models.DETM(2),                                    "none"),
    ("ProdLDA",      lambda: _topica.models.ProdLDA(2),                                 "none"),
    ("CombinedTM",   lambda: _topica.models.CombinedTM(2),                              "none"),
    ("ZeroShotTM",   lambda: _topica.models.ZeroShotTM(2),                              "none"),
    ("FASTopic",     lambda: _topica.models.FASTopic(2),                                "none"),
    # matrix-factorization — no theta posterior
    ("NMF",          lambda: _topica.models.NMF(2),                                     "none"),
    ("LSA",          lambda: _topica.models.LSA(2),                                     "none"),
    # embedding-cluster — no generative word distribution
    ("BERTopic",     lambda: _topica.models.BERTopic(min_cluster_size=5),               "none"),
    ("Top2Vec",      lambda: _topica.models.Top2Vec(),                                  "none"),
    # llm-based — pure-Python prompting pipeline, no generative word distribution
    ("TopicGPT",     lambda: _topica.models.TopicGPT(backend=lambda p: ""),             "none"),
    # InfoCTM is EXCLUDED: it is a cross-lingual model that fits TWO aligned
    # ProdLDA sub-models (fit(data_a, data_b, dictionary=...)) and exposes a
    # topic-word distribution PER LANGUAGE, not a single flat (K, V) surface. It
    # therefore has no generic (D, K) doc_topic / single topic_word to check
    # against the flat contract here; its per-language outputs are validated in
    # parity/infoctm_compare.py, and the underlying ProdLDA (in this registry)
    # covers the single-language contract. Treat InfoCTM like DTM/HLDA: a
    # structurally different model outside the flat-K conformance surface.
    #
    # EmbeddingLDA is EXCLUDED: it is a Python wrapper around SeededLDA (see
    # module-level note in python/topica/embedding.py). It delegates every
    # fitted-model getter to self._model via __getattr__, has no class-level
    # PyO3 attributes, and its __init__ requires embeddings and vocabulary, so
    # it cannot be constructed generically. The underlying SeededLDA (which is
    # in this registry) covers the contract. Treat EmbeddingLDA as a user-facing
    # convenience wrapper, not a first-class estimator, until it graduates to a
    # proper PyO3 binding with its own class attributes.
]

# ---------------------------------------------------------------------------
# Contract tiers
# ---------------------------------------------------------------------------

# Tier 0: every estimator
TIER0_ATTRS = [
    "topic_word",
    "doc_topic",
    "vocabulary",
    "num_topics",
    "topic_names",
    "doc_names",
    "top_words",
    "coherence",
    "save",
    "load",
    "fit_history",   # list[(iter, objective)] — [] when no trace is available
    "converged",     # bool | None — None for non-iterative (cluster) models
]
TIER0_ITERS = "iters"   # canonical iteration kwarg name

# Tier 1: generative models (model_family != "none")
TIER1_ATTRS = ["transform"]

# Tier 2: family-specific
TIER2_DIRICHLET = ["alpha", "theta_draws", "doc_lengths"]
TIER2_LOGISTIC_NORMAL = ["eta_mean", "eta_cov"]

# ---------------------------------------------------------------------------
# Principled exemptions — PERMANENT, will not be closed
# ---------------------------------------------------------------------------
# Key: (model_name, requirement)   Value: human-readable reason

EXEMPT: dict[tuple[str, str], str] = {

    # --- HLDA: topic-tree model, not a flat K-topic model ---
    # doc_topic: HLDA assigns each document to a root-to-leaf *path*, not a
    # K-vector simplex; there is no static (D, K) theta matrix.
    ("HLDA", "doc_topic"):  "HLDA is a topic tree (nested CRP); documents have paths, not a (D,K) theta simplex",
    # num_topics: the number of active nodes is discovered, not fixed to a
    # scalar K.  The model exposes num_nodes instead.
    ("HLDA", "num_topics"): "HLDA has num_nodes (tree), not a fixed scalar K",
    # doc_names: no static doc rows -> no doc_names index.
    ("HLDA", "doc_names"):  "HLDA has no static per-document row index (tree paths, not theta)",
    # coherence: requires a (K, V) topic_word matrix; HLDA's topic_word is
    # node-indexed, not a flat K-topic array.
    ("HLDA", "coherence"):  "HLDA coherence requires a flat (K,V) topic_word; node-indexed tree structure is incompatible",
    # transform: no flat K-topic generative model -> cannot infer a (n, K) theta.
    ("HLDA", "transform"):  "HLDA has no flat K-topic generative distribution; held-out transform is undefined",
    # iters: HLDA does accept iters, so this is NOT exempted.
    # alpha/theta_draws/doc_lengths: family is 'none' so Tier 2 dirichlet
    # does not apply.

    # --- DTM: time-sliced model, no static single theta ---
    # doc_topic: DTM's theta is time-conditioned; there is no single static
    # (D, K) matrix.  Use topic_word(time) and word_evolution for trajectories.
    ("DTM", "doc_topic"):   "DTM is time-sliced; doc_topic is undefined without a time slice argument",
    # doc_names: follows from no static doc_topic rows.
    ("DTM", "doc_names"):   "DTM has no static per-document row index (time-sliced model)",
    # coherence: coherence needs a static (K, V) phi matrix; DTM's topic_word
    # varies by time slice (callable, not a property).
    ("DTM", "coherence"):   "DTM coherence requires a static (K,V) topic_word; time-varying phi is incompatible",
    # transform: DTM's EM is time-conditioned; there is no time-slice-free
    # held-out inference.
    ("DTM", "transform"):   "DTM has no time-slice-free held-out inference",

    # --- BERTopic / Top2Vec: embedding-cluster models ---
    # topic_names and doc_names: topic_names is present; doc_names is not (these
    # models expose cluster `labels`, and a names index is cluster-specific).
    ("BERTopic", "doc_names"): "BERTopic exposes labels (cluster ids) not a doc_names property; mapping to names is cluster-specific",
    ("Top2Vec",  "doc_names"): "Top2Vec exposes labels (cluster ids) not a doc_names property",
    # iters: these models are not iterative samplers in the Gibbs/EM sense (the
    # fit is UMAP + HDBSCAN clustering), so a standardized iteration count does
    # not apply.
    ("BERTopic", "iters"): "BERTopic is not an iterative sampler (UMAP + HDBSCAN); no iteration count applies",
    ("Top2Vec",  "iters"): "Top2Vec is not an iterative sampler (UMAP + HDBSCAN); no iteration count applies",

    # --- TopicGPT: LLM prompting pipeline, no generative word distribution ---
    # iters: TopicGPT runs a fixed three-stage prompt flow (generate / refine /
    # assign), not an iterative sampler, so no iteration count applies.
    ("TopicGPT", "iters"): "TopicGPT is a prompting pipeline (generate/refine/assign), not an iterative sampler; no iteration count applies",
    # doc_names: TopicGPT exposes a positional doc_names index (str ids), so this
    # is NOT exempted; topic_word/doc_topic/coherence are synthesized descriptors
    # (class-based TF-IDF), present and valid, so none of those are exempted.
    # Tier 2: family is 'none'; no Tier 2 applies. transform IS present (Tier 1
    # only required for family != 'none', which TopicGPT is not).
    # transform: both DO expose transform; NOT exempted.
    # Tier 2: family is 'none'; no Tier 2 applies.
    # coherence: a count-based UMass/NPMI is computable from any top-word list,
    # so a model-level coherence() is a fixable gap (see KNOWN_GAPS), not an
    # exemption -- even though their topic_word is c-TF-IDF rather than P(w|t).
    #
    # fit_history/converged: BERTopic and Top2Vec DO expose these (returning []
    # and None respectively), so they are NOT exempted here.

    # --- LSA/LSI: matrix factorization, signed latent coordinates ---
    # doc_topic: LSA's doc_topic is U Sigma, the document coordinates in the
    # reduced SVD space. These are SIGNED and rows do NOT sum to 1, so the (D, K)
    # array is not a topic simplex. The attribute is present and shaped (D, K);
    # this exemption documents that its simplex semantics intentionally do not
    # hold, mirroring the HLDA/DTM doc_topic entries.
    ("LSA", "doc_topic"): "LSA yields signed latent coordinates, not a (D,K) topic simplex",
    # iters: the truncated SVD is a direct (non-iterative) solve, so there is no
    # iteration count to expose. fit() takes no iters kwarg by design.
    ("LSA", "iters"): "LSA is a direct SVD solve, not an iterative sampler; no iteration count applies",
}

# ---------------------------------------------------------------------------
# Known gaps — TEMPORARY, tracked as a burn-down worklist
# ---------------------------------------------------------------------------
# Key: (model_name, requirement)   Value: phase note
#
# Issue #46 (both parts) and #95 have shipped: every iterative model records a
# real per-iteration fit_history and exposes a converged flag, and the whole
# collapsed-Gibbs family (LDA, DMR, SeededLDA, LabeledLDA, SAGE, SupervisedLDA,
# PA, PT, keyATM) supports opt-in convergence_tol early-stopping.  Per-iteration
# trace quality and early-stop semantics are verified in
# tests/test_convergence_interface.py.  No known convergence gaps remain; HDP and
# GSDMM intentionally never early-stop (they discover topic/cluster counts, so a
# log-likelihood plateau is not a convergence signal — see that test's
# _CONVERGED_FALSE_ALWAYS set).

KNOWN_GAPS: dict[tuple[str, str], str] = {}

# ---------------------------------------------------------------------------
# check_conformance helper
# ---------------------------------------------------------------------------

def _accepts_kwarg(fn, name: str) -> bool:
    """Whether ``fn`` accepts the keyword argument ``name``."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def check_conformance(model_or_class) -> list[str]:
    """Check ``model_or_class`` against the topica estimator contract.

    Returns a list of violation strings. An empty list means the model
    satisfies every applicable tier requirement (or has a valid exemption).
    Does NOT look up KNOWN_GAPS or EXEMPT -- it reports all raw violations so
    the conformance test can categorize them. Call this from the test or from
    your own CI after adding a new estimator.

    Parameters
    ----------
    model_or_class : an estimator instance or class.

    Returns
    -------
    list of str
        Each entry is a human-readable description of the violation, e.g.
        ``"missing class attribute: topic_names"``.
    """
    if isinstance(model_or_class, type):
        cls = model_or_class
        instance = None
    else:
        cls = type(model_or_class)
        instance = model_or_class

    name = cls.__name__
    violations: list[str] = []

    # Tier 0: class-level attribute checks
    for attr in TIER0_ATTRS:
        if not hasattr(cls, attr):
            violations.append(f"missing class attribute: {attr}")

    # Tier 0: iters kwarg
    fit_fn = getattr(cls, "fit", None)
    if fit_fn is None:
        violations.append("missing fit method")
    else:
        if not _accepts_kwarg(fit_fn, TIER0_ITERS):
            violations.append(f"fit() does not accept kwarg: {TIER0_ITERS}")

    # Determine family from an instance (works on unfitted instances per model_family docstring)
    if instance is None:
        # Construct a minimal instance from the registry if possible
        _instance = None
        for reg_name, factory, _ in REGISTRY:
            if reg_name == name:
                try:
                    _instance = factory()
                except Exception:
                    pass
                break
        family_instance = _instance if _instance is not None else cls.__new__(cls)
    else:
        family_instance = instance

    from .effects import model_family as _model_family
    try:
        family = _model_family(family_instance)
    except Exception:
        family = "none"

    # Tier 1: transform required for generative models
    if family != "none":
        if not hasattr(cls, "transform"):
            violations.append("missing class attribute: transform (Tier 1 generative model)")

    # Tier 2: family-specific
    if family == "dirichlet":
        for attr in TIER2_DIRICHLET:
            if not hasattr(cls, attr):
                violations.append(f"missing class attribute: {attr} (Tier 2 dirichlet)")
    elif family == "logistic_normal":
        for attr in TIER2_LOGISTIC_NORMAL:
            if not hasattr(cls, attr):
                violations.append(f"missing class attribute: {attr} (Tier 2 logistic_normal)")

    return violations
