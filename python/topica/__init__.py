"""topica: fast SparseLDA topic modeling (MALLET's algorithm) in Rust.

The heavy lifting lives in the compiled extension ``topica._topica``;
this module just re-exports its public surface so ``import topica`` works
and editors/type-checkers see a stable namespace.
"""

from ._topica import (
    LDA,
    DMR,
    LabeledLDA,
    SAGE,
    CTM,
    STM,
    STS,
    HDP,
    DTM,
    SupervisedLDA,
    PT,
    GSDMM,
    SeededLDA,
    KeyATM,
    Top2Vec,
    BERTopic,
    ETM,
    ProdLDA,
    FASTopic,
    PA,
    HLDA,
    NMF,
    LSA,
    Corpus,
    tokenize,
    project,
    DEFAULT_TOKEN_REGEX,
    __version__,
)

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


def one_hot(values, *, drop_first=True, prefix=""):
    """One-hot encode a categorical covariate for use as DMR features.

    Given a sequence of category labels (one per document), returns
    ``(matrix, names)`` where ``matrix`` is a ``(num_docs, num_categories)``
    float array of 0/1 indicators and ``names`` are the corresponding column
    names. With ``drop_first=True`` (default) the first category (sorted) is
    omitted as the reference level, which avoids collinearity with the DMR
    intercept. Pass the result straight to ``DMR.fit(docs, matrix,
    feature_names=names)``; combine multiple covariates with
    ``numpy.hstack``.
    """
    import numpy as np

    values = list(values)
    categories = sorted(set(values))
    if drop_first and categories:
        categories = categories[1:]
    index = {c: j for j, c in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for i, v in enumerate(values):
        j = index.get(v)
        if j is not None:
            matrix[i, j] = 1.0
    names = [f"{prefix}{c}" for c in categories]
    return matrix, names


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


from .gdmr import GDMR  # noqa: E402  (pure-Python Legendre-basis DMR wrapper)
from . import stm  # noqa: E402  (stm imports names defined above)
from .stm import align_corpus, spline, interaction  # noqa: E402  (general covariate-design helpers)
from . import keyatm  # noqa: E402  (keyATM-specific workflow helpers)
from . import effects  # noqa: E402  (model-neutral prevalence analysis)
from . import validation  # noqa: E402  (post-hoc topic diagnostics surface)
from . import conformance  # noqa: E402  (estimator contract and registry)
from .conformance import check_conformance  # noqa: E402
from .effects import (  # noqa: E402  general, work on any model's theta
    estimate_effect,
    by_strata,
    prevalence_ci,
    top_topics,
    posterior_theta_samples,
    dirichlet_theta_samples,
    standard_errors,
    model_family,
    predicted_prevalence,
    PredictedPrevalence,
    permutation_test,
    PermutationResult,
)
from .keyatm import time_prevalence_ci  # noqa: E402  (dynamic keyATM credible bands)
from . import phrases  # noqa: E402
from .coherence import (  # noqa: E402
    coherence,
    topic_diversity,
    exclusivity,
    word_intrusion,
    document_intrusion,
)
from .registry import list_models, ModelInfo, REGISTRY  # noqa: E402  model taxonomy / discovery

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
    align_topics,
    topic_stability,
)
from .ensemble import ensemble, EnsembleResult  # noqa: E402  (consensus across runs)
from .analysis import (  # noqa: E402  (model-neutral fitted-model analysis surface)
    topic_info,
    topic_sizes,
    topic_labels,
    set_topic_labels,
    representative_docs,
    topics_over_time,
    topics_per_class,
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
from .embedding import (  # noqa: E402
    EmbeddingLDA,
    embedding_seeds,
    llm_embed,
    save_embeddings,
    load_embeddings,
)
from .preprocess import split_documents  # noqa: E402
from .stopwords import ENGLISH_STOPWORDS  # noqa: E402
from .phrases import learn_phrases, apply_phrases, add_ngrams, Phrases  # noqa: E402
from .frames import from_dataframe, align, prep_documents, plot_removed  # noqa: E402
from .formulas import design_matrix  # noqa: E402

__all__ = [
    "list_models",
    "ModelInfo",
    "REGISTRY",
    "LDA",
    "DMR",
    "GDMR",
    "LabeledLDA",
    "SAGE",
    "CTM",
    "STM",
    "STS",
    "HDP",
    "DTM",
    "SupervisedLDA",
    "PT",
    "GSDMM",
    "SeededLDA",
    "KeyATM",
    "Top2Vec",
    "BERTopic",
    "ETM",
    "ProdLDA",
    "FASTopic",
    "PA",
    "HLDA",
    "NMF",
    "LSA",
    "Corpus",
    "tokenize",
    "project",
    "one_hot",
    "stm",
    "keyatm",
    "spline",
    "interaction",
    "phrases",
    "coherence",
    "topic_diversity",
    "exclusivity",
    "word_intrusion",
    "document_intrusion",
    "frex",
    "label_topics",
    "topic_table",
    "topic_correlation",
    "find_thoughts",
    "search_k",
    "SearchKResult",
    "select_model",
    "SelectModelResult",
    "plot_models",
    "plot_search_k",
    "plot_topic_discovery",
    "relevance",
    "prepare_pyldavis",
    "check_residuals",
    "align_topics",
    "topic_stability",
    "ensemble",
    "EnsembleResult",
    "find_thoughts_html",
    "quality_frontier",
    "bootstrap_stability",
    "report",
    "topic_info",
    "topic_sizes",
    "topic_labels",
    "set_topic_labels",
    "representative_docs",
    "topics_over_time",
    "topics_per_class",
    "plot_report",
    "fighting_words",
    "top_fighting_words",
    "llm_topic_labels",
    "llm_backend",
    "topic_label_prompts",
    "estimate_effect",
    "by_strata",
    "prevalence_ci",
    "top_topics",
    "posterior_theta_samples",
    "dirichlet_theta_samples",
    "standard_errors",
    "model_family",
    "predicted_prevalence",
    "PredictedPrevalence",
    "permutation_test",
    "PermutationResult",
    "time_prevalence_ci",
    "EmbeddingLDA",
    "embedding_seeds",
    "llm_embed",
    "save_embeddings",
    "load_embeddings",
    "split_documents",
    "ENGLISH_STOPWORDS",
    "from_dataframe",
    "align",
    "prep_documents",
    "plot_removed",
    "design_matrix",
    "summary",
    "diagnostics",
    "perplexity",
    "make_heldout",
    "eval_heldout",
    "Heldout",
    "HeldoutResult",
    "mmr",
    "keywords",
    "conformance",
    "check_conformance",
    "preprocess",
    "learn_phrases",
    "apply_phrases",
    "add_ngrams",
    "Phrases",
    "align_corpus",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
    "__citation__",
]
