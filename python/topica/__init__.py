"""topica: fast, all-purpose topic modeling for Python, with a Rust core.

More than a dozen models — LDA, STM, CTM, DMR, keyATM, SAGE, HDP, BERTopic,
ProdLDA, and more — behind one numpy-native API, each validated against its
reference implementation. Built for computational social scientists who need a
defensible, reviewer-ready analysis, not just topics.

Start here (a full LDA workflow is only a few lines)::

    import topica

    corpus = topica.from_dataframe(df, text_col="text")   # build + prune vocab
    res = topica.search_k(corpus, ks=[10, 20, 30])         # choose K defensibly
    model = topica.LDA(num_topics=res.best_k()).fit(corpus)
    topica.topic_table(model)                              # publication-ready labels
    topica.estimate_effect(model, X=X, corpus=corpus)      # covariate effects + CIs

The main entry points, by task:

- **Corpus**: :func:`from_dataframe`, :class:`Corpus`, :func:`tokenize`.
- **Models**: :class:`LDA` and the wider family (:class:`STM`, :class:`CTM`,
  :class:`DMR`, :class:`KeyATM`, :class:`SAGE`, :class:`HDP`, :class:`BERTopic`,
  …); ``model.fit(corpus)`` then read ``model.topic_word`` / ``model.doc_topic``.
- **Choosing K**: :func:`search_k` (+ ``.best_k()``), :func:`plot_search_k`.
- **Interpretation**: :func:`label_topics`, :func:`topic_table`, :func:`frex`,
  :func:`find_thoughts`, :func:`topics_for_term`.
- **Validation**: :func:`coherence`, :func:`coherence_ci`, :func:`exclusivity`,
  :func:`topic_diversity`, :func:`ensemble`, :func:`topic_stability`.
- **Covariate effects**: :func:`estimate_effect`, :func:`predicted_prevalence`,
  :func:`one_hot`, :func:`design_matrix`, :func:`spline`.
- **Provenance**: :func:`record_fit` → :class:`AnalysisManifest`.

The heavy lifting lives in the compiled extension ``topica._topica``; this module
re-exports its public surface so ``import topica`` works and editors/type-checkers
see a stable namespace.
"""

from ._topica import (
    LDA,
    OnlineLDA,
    DMR,
    LabeledLDA,
    SAGE,
    CTM,
    STM,
    STS,
    HDP,
    DTM,
    DETM,
    SupervisedLDA,
    PT,
    GSDMM,
    BTM,
    FactorialLDA,
    FactorialLDA as FLDA,
    PolylingualLDA,
    DiscLDA,
    SeededLDA,
    KeyATM,
    Top2Vec,
    BERTopic,
    SemanticSignalSeparation,
    ETM,
    IdealPointTM,
    IdealPointSentenceTM,
    TBIP,
    TensorLDA,
    KeyNMF,
    Wordfish,
    Wordshoal,
    PartyEmbeddings,
    ProdLDA,
    RTM,
    Scholar,
    InfoCTM,
    FASTopic,
    PA,
    HLDA,
    NMF,
    GuidedNMF,
    CorEx,
    AuthorTopic,
    MGLDA,
    TopicalNGrams,
    TopicsOverTime,
    GaussianLDA,
    LSA,
    CombinedTM,
    ZeroShotTM,
    Corpus,
    tokenize,
    project,
    set_experimental as _set_experimental,
    experimental_is_enabled as _experimental_is_enabled,
    DEFAULT_TOKEN_REGEX,
    __version__,
)


def enable_experimental(enabled: bool = True) -> None:
    """Opt into experimental, unvalidated models for this process.

    Some models ship before they have a published paper and a
    reference-implementation parity check (topica's bar for a *validated*
    model). They are flagged **experimental**: kept out of the validated roster
    and the README model table, documented separately, and refused at
    construction or load until you opt in here. Call this once, early, before
    constructing an experimental model; pass ``False`` to turn the gate back
    on. Use :func:`list_models` with ``experimental=True`` to see the current
    set. Equivalent to setting the ``TOPICA_EXPERIMENTAL=1``
    environment variable. Experimental models may change or be removed without a
    deprecation cycle.
    """
    _set_experimental(bool(enabled))


def experimental_enabled() -> bool:
    """Whether experimental models are currently enabled (see
    :func:`enable_experimental`)."""
    return bool(_experimental_is_enabled())

__citation__ = (
    "Caren, N. (2026). topica: fast, all-purpose topic modeling for Python. "
    "https://github.com/nealcaren/topica\n\n"
    "@software{caren_topica,\n"
    "  author = {Caren, Neal},\n"
    "  title  = {topica: fast, all-purpose topic modeling for Python},\n"
    "  year   = {2026},\n"
    "  url    = {https://github.com/nealcaren/topica}\n"
    "}\n\n"
    "Please also cite the model(s) you use; see "
    "https://nealcaren.github.io/topica/citing/."
)


# one_hot lives in topica.design (re-exported below); see design.py.


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
                lines.append(f"  topic {i}: " + " ".join(words))
    except Exception:
        pass
    return "\n".join(lines)


from .gdmr import GDMR  # noqa: E402  (pure-Python Legendre-basis DMR wrapper)
from .narrative import NarrativeTM  # noqa: E402  (pure-Python NarrativeTM wrapper)

# PLTM is the paper's acronym for the Polylingual Topic Model.
PLTM = PolylingualLDA  # noqa: E402
from . import stm  # noqa: E402  (stm imports names defined above)
from .stm import align_corpus, spline, interaction, topic_correlation_ci, TopicCorrelationCI  # noqa: E402  (general covariate-design helpers)
from .embedding_regression import (  # noqa: E402  (conText embedding regression: covariate effects on meaning)
    embedding_regression,
    EmbeddingRegression,
    alc_embeddings,
    compute_transform,
)
from . import keyatm  # noqa: E402  (keyATM-specific workflow helpers)
from . import effects  # noqa: E402  (model-neutral prevalence analysis)
from . import validation  # noqa: E402  (post-hoc topic diagnostics surface)
from . import content  # noqa: E402  (content-covariate diagnostics: STM/STS/SAGE)
from . import conformance  # noqa: E402  (estimator contract and registry)
from .conformance import check_conformance  # noqa: E402
from . import manifest  # noqa: E402  (analysis manifest / provenance record)
from .manifest import AnalysisManifest, record_fit  # noqa: E402
from .effects import (  # noqa: E402  general, work on any model's theta
    estimate_effect,
    by_strata,
    prevalence_ci,
    top_topics,
    posterior_theta_samples,
    dirichlet_theta_samples,
    composition_theta,
    standard_errors,
    model_family,
    predicted_prevalence,
    PredictedPrevalence,
    average_marginal_effects,
    ame,
    MarginalEffect,
    AverageMarginalEffects,
    permutation_test,
    PermutationResult,
)
from .keyatm import time_prevalence_ci  # noqa: E402  (dynamic keyATM credible bands)
from . import phrases  # noqa: E402
from .coherence import (  # noqa: E402
    coherence,
    coherence_ci,
    CoherenceCI,
    semantic_coherence,
    embedding_coherence,
    topic_diversity,
    topic_semantic_diversity,
    inverted_rbo,
    exclusivity,
    word_intrusion,
    document_intrusion,
)
from .agreement import agreement  # noqa: E402  (external validation vs gold labels)
# LLM-based evaluation is exposed as a namespace, topica.llm.* (coherence,
# intrusion, select_k, backend, PROMPTS) -- it is an llm-bounded family, kept
# distinct from the bit-exact diagnostics above. See topica/llm.py.
from . import llm  # noqa: E402
from . import mcmc  # noqa: E402  (single-chain MCMC diagnostics for the Gibbs models)
from .mcmc import (  # noqa: E402
    mcmc_diagnostics,
    effective_sample_size,
    autocorrelation,
    integrated_autocorr_time,
    McmcDiagnostics,
    rhat,
    multichain_diagnostics,
    MultiChainDiagnostics,
)
from .registry import list_models, ModelInfo, REGISTRY, effective_determinism  # noqa: E402  model taxonomy / discovery

from .validation import (  # noqa: E402  general, model-agnostic post-hoc analyses
    diagnostics,
    perplexity,
    make_heldout,
    eval_heldout,
    Heldout,
    HeldoutResult,
    frex,
    mmr,
    label_topics,
    topics_for_term,
    topic_table,
    topic_correlation,
    find_thoughts,
    find_thoughts_html,
    quality_frontier,
    bootstrap_stability,
    search_k,
    SearchKResult,
    select_model,
    SelectModelResult,
    plot_models,
    plot_search_k,
    plot_topic_discovery,
    relevance,
    prepare_pyldavis,
    check_residuals,
    document_residuals,
    flag_topics,
    topic_dendrogram,
    TopicDendrogram,
    align_topics,
    topic_stability,
)
from .crossval import (  # noqa: E402  (#701 cross-validation evaluation framework)
    cross_validate,
    make_folds,
    Folds,
    CrossValResult,
)
from .ensemble import ensemble, EnsembleResult, cross_ensemble  # noqa: E402  (consensus across runs)
from .compare import (  # noqa: E402  (statistical two-fit topic-drift comparison, #415)
    CompareResult,
    MatchedPair,
    UnmatchedTopic,
)
# `compare` is exposed as a callable module (issue #757): `topica.compare(a, b)`
# still calls the function, and `topica.compare.CompareResult` reaches the namespace.
from . import compare  # noqa: E402, F811
from .robustness import (  # noqa: E402  (effect robustness across K / seeds, #644)
    effects_across_k,
    effects_across_seeds,
    RobustnessResult,
)
from ._results import (  # noqa: E402  (dict results with .to_frame(); #742, #752)
    FrameDict,
    QualityFrontier,
    BootstrapStability,
    KeywordDiagnostics,
    TimePrevalenceCI,
)
from .analysis import (  # noqa: E402  (model-neutral fitted-model analysis surface)
    topic_info,
    topic_sizes,
    topic_labels,
    set_topic_labels,
    representative_docs,
    topics_over_time,
    topics_per_class,
    contrastive_topics,
    stop_reason,
    plot_report,
)


def report(model, topn=8):
    """One-call overview of a fitted model. Alias for :func:`summary`.

    ``report`` reads like a verb, so ``report(model)`` is a natural thing to
    try; it returns the same multi-line overview as ``summary(model)``. The
    richer analysis surface (``topic_info``, ``topic_sizes``,
    ``representative_docs``, ``topics_over_time``, ``plot_report``, …) lives in
    ``topica.analysis`` and is also exported as top-level functions.
    """
    return summary(model, topn=topn)
from .keywords import fighting_words, top_fighting_words  # noqa: E402
from .labeling import (  # noqa: E402  LLM topic labeling as plumbing
    llm_topic_labels,
    llm_backend,
    topic_label_prompts,
)
from .topicgpt import TopicGPT  # noqa: E402  (LLM-driven topic discovery)
from .anchor import AnchorLDA  # noqa: E402
from .embedding import (  # noqa: E402
    EmbeddingLDA,
    embedding_seeds,
    llm_embed,
    save_embeddings,
    load_embeddings,
)
from .preprocess import split_documents  # noqa: E402
from .stopwords import (  # noqa: E402
    ENGLISH_STOPWORDS,
    SENTIMENT_STOPWORDS,
    stopwords,
    stopword_languages,
)
from .phrases import learn_phrases, apply_phrases, add_ngrams, Phrases  # noqa: E402
from .frames import from_dataframe, align, prep_documents, plot_removed  # noqa: E402
from .formulas import design_matrix  # noqa: E402
from .scaling import bimodality, polarization, polarization_ci, split_half_reliability, position_intervals  # noqa: E402  (intrinsic ideal-point diagnostics)
from . import datasets  # noqa: E402  (bundled + fetch-on-demand example datasets)

# Workflow namespaces (issue #757): task-oriented facades that group the flat API
# by analysis stage. Every name they expose is also available at the package root,
# so this is purely additive; prefer the namespace path in new code
# (``topica.select.search_k``, ``topica.inspect.topic_table``, ``topica.data``).
from . import (  # noqa: E402, F401
    data,
    design,
    select,
    inspect,
    evaluate,
    embeddings,
    provenance,
)
from .design import one_hot  # noqa: E402  (one_hot's home is topica.design)

# The curated public surface (issue #757): workflow namespaces, corpus ingress,
# the flagship models (list_models(common_start=True)), the quick-path callables
# the module docstring teaches, and discovery/version metadata. Every other legacy
# name stays importable (`topica.X` and `from topica import X` both work) and is
# resolved lazily through the owning namespace by __getattr__ below; it is simply
# no longer part of `import *` / the advertised surface. Prefer the namespace path
# (topica.select.search_k, topica.inspect.label_topics, ...) in new code.
__all__ = [
    # workflow namespaces
    "data", "design", "select", "inspect", "evaluate",
    "effects", "compare", "embeddings", "provenance",
    # corpus ingress
    "Corpus", "tokenize", "from_dataframe",
    # flagship models (the newcomer starting set; the rest stay importable)
    "LDA", "STM", "NMF", "KeyATM", "GSDMM", "BERTopic",
    # permanent quick-path callables
    "search_k", "topic_table", "estimate_effect",
    # discovery / experimental gate / identity
    "list_models", "enable_experimental",
    "__version__", "__citation__",
]


from . import _compat as _compat  # noqa: E402  (legacy-name -> namespace map, #757)


def __getattr__(name):
    """Resolve a legacy top-level name that is no longer in the curated ``__all__``
    by importing it from the workflow namespace that owns it (PEP 562). Keeps every
    historical ``topica.X`` working after the surface was curated (issue #757)."""
    ns = _compat.LAZY.get(name)
    if ns is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{ns}"), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__():
    """Everything discoverable at the REPL: the curated surface, every eager name
    (models, result containers, ...), and the lazily-resolvable legacy names."""
    return sorted(set(globals()) | set(_compat.LAZY) | set(__all__))
