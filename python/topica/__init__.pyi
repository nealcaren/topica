from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy
import numpy.typing

from ._topica import (
    LDA as LDA,
    DMR as DMR,
    LabeledLDA as LabeledLDA,
    SAGE as SAGE,
    CTM as CTM,
    STM as STM,
    HDP as HDP,
    DTM as DTM,
    SupervisedLDA as SupervisedLDA,
    PT as PT,
    GSDMM as GSDMM,
    PA as PA,
    HLDA as HLDA,
    SeededLDA as SeededLDA,
    KeyATM as KeyATM,
    Top2Vec as Top2Vec,
    BERTopic as BERTopic,
    ETM as ETM,
    ProdLDA as ProdLDA,
    FASTopic as FASTopic,
    Corpus as Corpus,
    tokenize as tokenize,
    DEFAULT_TOKEN_REGEX as DEFAULT_TOKEN_REGEX,
    __version__ as __version__,
)
from . import stm as stm
from . import keyatm as keyatm
from . import effects as effects
from .effects import (
    estimate_effect as estimate_effect,
    average_marginal_effects as average_marginal_effects,
    ame as ame,
    by_strata as by_strata,
    top_topics as top_topics,
    posterior_theta_samples as posterior_theta_samples,
    dirichlet_theta_samples as dirichlet_theta_samples,
    standard_errors as standard_errors,
    model_family as model_family,
)
from .keyatm import time_prevalence_ci as time_prevalence_ci
from .embedding import (
    EmbeddingLDA as EmbeddingLDA,
    embedding_seeds as embedding_seeds,
    llm_embed as llm_embed,
    save_embeddings as save_embeddings,
    load_embeddings as load_embeddings,
)
from .topicgpt import TopicGPT as TopicGPT

__citation__: str
ENGLISH_STOPWORDS: frozenset[str]

def one_hot(
    values: Sequence[object],
    *,
    drop_first: bool = True,
    prefix: str = "",
) -> tuple[numpy.typing.NDArray[numpy.float64], list[str]]:
    """One-hot encode a categorical covariate into (matrix, names) for DMR.fit."""
    ...


def align_corpus(
    new_docs: Sequence[Sequence[str]],
    model: Any,
) -> list[list[str]]:
    """Restrict each token list to the tokens present in model.vocabulary.

    Out-of-vocabulary tokens are silently dropped. Documents that become empty
    after filtering are represented as empty lists. Returns aligned token lists
    ready to pass to model.transform or topica.stm.transform."""
    ...

def spline(
    x: Any,
    df: int = 4,
    knots: Any | None = None,
) -> tuple[numpy.typing.NDArray[numpy.float64], list[str]]:
    """Restricted (natural) cubic-spline basis for a covariate, returned as
    (basis (n, df), names). A general covariate-design helper: column_stack the
    basis into any model's design matrix (DMR, STM, STS, KeyATM) and extend
    feature_names with the returned names. Also exposed inside formula strings as
    spline(...)."""
    ...

def interaction(
    a: Any,
    b: Any,
    name: str = "interaction",
) -> tuple[numpy.typing.NDArray[numpy.float64], list[str]]:
    """Interaction columns (all pairwise products) between two covariate blocks,
    returned as (products (n, ncols), names). column_stack into any model's
    design matrix."""
    ...

def coherence(
    topics: Any,
    texts: Sequence[Sequence[str]],
    *,
    coherence_type: str = "c_v",
    topn: int = 10,
    window_size: int | None = None,
    epsilon: float = 1e-12,
) -> numpy.typing.NDArray[numpy.float64]:
    """Per-topic coherence (u_mass / c_uci / c_npmi / c_v) of a model or list of
    word lists against a reference corpus `texts`. Returns shape (num_topics,)."""
    ...

def topic_diversity(topics: Any, topn: int = 25) -> float:
    """Fraction of unique words across all topics' top-`topn` words."""
    ...

def topic_semantic_diversity(topics: Any, topn: int = 25) -> float:
    """Fraction of unique top-word *pairs* across all topics (Wu, Nguyen & Luu 2024)."""
    ...


def exclusivity(model_or_phi: Any, *, n: int = 10, w: float = 0.7) -> numpy.typing.NDArray[numpy.float64]:
    """Per-topic exclusivity (stm's FREX summary over the top-n words), shape
    (num_topics,). Pair with per-topic coherence for the coherence-vs-exclusivity
    quality plot. From topica's stm-faithful Rust core."""
    ...


def semantic_coherence(
    model_or_phi: Any,
    texts: Any,
    vocabulary: list[str] | None = None,
    *,
    n: int = 10,
) -> numpy.typing.NDArray[numpy.float64]:
    """Per-topic semantic coherence (stm's semCoh1beta, UMass with 0.01 smoothing)
    over the top-n words, shape (num_topics,). ``texts`` is a Corpus or list of
    token lists. From topica's stm-faithful Rust core."""
    ...


def word_intrusion(
    model_or_phi: Any,
    vocabulary: Sequence[str] | None = None,
    *,
    n_words: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Word-intrusion test (Chang et al. 2009): per topic, top words + one
    intruder. Dict keys: topic, words (shuffled), intruder, intruder_index."""
    ...


def document_intrusion(
    model_or_theta: Any,
    texts: Sequence[str] | None = None,
    *,
    n_docs: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Document-intrusion test: per topic, top docs + one low-share intruder.
    Dict keys: topic, doc_indices (shuffled), intruder_index, texts (if given)."""
    ...


# General, model-agnostic post-hoc analyses (also in topica.validation).
def diagnostics(
    model: Any,
    texts: Any = None,
    *,
    n: int = 10,
    coherence_type: str | None = None,
    stability: bool = False,
    n_boot: int = 20,
    model_factory: Any = None,
    seed: int = 0,
) -> Any:
    """One per-topic diagnostics table (coherence/exclusivity/FREX/size/stability)
    as a pandas DataFrame, or a list of row dicts when pandas is absent."""
    ...


def perplexity(model: Any, held_out: Any, *, seed: int = 0) -> float:
    """Document-completion held-out perplexity for a generative model (lower is
    better), a K-comparable signal for justifying a topic count."""
    ...


def frex(topic_word: Any, vocabulary: Sequence[str] | None = None, *, w: float = 0.5, n: int = 10) -> list:
    """FREX (frequency-exclusivity) top words per topic. Accepts a model or a (K, V) array."""
    ...


def mmr(
    topic_word: Any,
    word_embeddings: Any,
    vocabulary: Sequence[str] | None = None,
    *,
    n: int = 10,
    diversity: float = 0.3,
    n_candidates: int | None = None,
) -> list:
    """Maximal-marginal-relevance top words: rerank to cut near-synonyms using word
    embeddings (BERTopic's MaximalMarginalRelevance). Accepts a model or a (K, V) array."""
    ...


def label_topics(topic_word: Any, vocabulary: Sequence[str] | None = None, *, n: int = 10) -> list[dict]:
    """Per-topic word lists with keys prob / frex / lift / score."""
    ...


def topic_correlation(doc_topic: Any, *, threshold: float = 0.05) -> Any:
    """Topic-correlation network (.cor, .adjacency, .edges)."""
    ...


def find_thoughts(doc_topic: Any, texts: Sequence[str] | None = None, *, topic: int, n: int = 3) -> list:
    """The n documents most associated with a topic."""
    ...


def search_k(docs: Any, ks: Sequence[int], *, held_out: Any = None, **kwargs: Any) -> list[dict]:
    """Fit an LDA per K; report coherence, exclusivity, and (optional) perplexity."""
    ...


def relevance(
    topic_word: Any,
    vocabulary: Sequence[str] | None = None,
    *,
    topic: int | None = None,
    lam: float = 0.6,
    n: int = 10,
    term_frequency: Any = None,
) -> list:
    """LDAvis word relevance (Sievert & Shirley 2014)."""
    ...


def prepare_pyldavis(model: Any, docs: Any, **kwargs: Any) -> Any:
    """Build the LDAvis intertopic-distance view (pyLDAvis PreparedData or inputs)."""
    ...


def check_residuals(model: Any, docs: Any, *, tol: float = 0.01) -> Any:
    """Taddy (2012) residual-dispersion test for whether K is too small."""
    ...


def align_topics(a: Any, b: Any, *, metric: str = "cosine") -> list:
    """One-to-one topic matching across two fits (Hungarian)."""
    ...


def topic_stability(runs: Any, *, topn: int = 10, metric: str = "cosine") -> float:
    """Term-centric topic stability across fits (Greene et al. 2014)."""
    ...


def report(model: Any, topn: int = 8) -> str:
    """One-call overview of a fitted model. Alias for ``summary``."""
    ...


def time_prevalence_ci(
    model: Any,
    timestamps: Sequence[object],
    *,
    ci: float = 0.95,
    normalize: bool = True,
) -> dict:
    """Per-period topic prevalence with credible intervals from the dynamic keyATM posterior.

    A thin wrapper over prevalence_ci with the period order pinned to the model's
    time_labels. Requires a dynamic KeyATM fit with keep_theta_draws=True (the
    default). Returns a dict with keys: labels, mean, ci_low, ci_high, sd (all
    arrays shape (T, K) except labels which is a list).
    """
    ...


def prevalence_ci(
    model: Any,
    groups: Sequence[object],
    *,
    ci: float = 0.95,
    normalize: bool = True,
    corpus: Any | None = None,
    nsims: int | None = None,
    seed: int = 0,
    labels: Sequence[object] | None = None,
) -> dict:
    """Per-group topic prevalence with posterior credible bands, for any model.

    The draws-based companion to by_strata: groups documents by group label and
    reads the empirical credible band off the posterior theta draws (via
    composition_theta, so it works for Dirichlet and logistic-normal models).
    Returns a dict with keys: labels, mean, ci_low, ci_high, sd (arrays shape
    (num_groups, K) except labels which is a list).
    """
    ...


# Model-neutral fitted-model analysis surface (also in topica.analysis).
def topic_info(
    model: Any,
    texts: Sequence[str] | None = None,
    *,
    n: int = 8,
    labels: Sequence[str] | None = None,
) -> list[dict]:
    """One summary row per topic: topic, label, size, prevalence, top_words
    (and representative_docs when texts is given). Adds a topic=-1 outlier row
    for clustering models with outliers."""
    ...


def topic_sizes(model: Any) -> dict:
    """Per-topic hard size and expected mass: keys size, mass, outliers."""
    ...


def topic_labels(model: Any) -> list[str]:
    """Effective per-topic labels (custom labels over topic_names)."""
    ...


def set_topic_labels(model: Any, mapping: dict[int, str]) -> None:
    """Store custom per-topic labels, keyed by id(model)."""
    ...


def representative_docs(
    model: Any, texts: Sequence[str], *, topic: int | None = None, n: int = 5
) -> Any:
    """Each topic's highest-loading documents. A list for one topic, else
    {topic_id: [docs]} for every topic."""
    ...


def topics_over_time(model: Any, timestamps: Sequence[object], *, normalize: bool = True) -> dict:
    """Mean topic prevalence per distinct timestamp: keys labels, prevalence."""
    ...


def topics_per_class(model: Any, groups: Sequence[object], *, ci: float = 0.95) -> list:
    """Mean topic prevalence within each group (wraps by_strata)."""
    ...


def contrastive_topics(
    model: Any,
    texts: Sequence[Sequence[str]],
    groups: Sequence[object],
    *,
    prior: float = 0.01,
    informative: bool = False,
    min_count: int = 5,
    n_words: int = 10,
    group_order: tuple[object, object] | None = None,
) -> list:
    """Topic-conditional Fighting Words: per topic, which words separate two
    groups and how much each topic divides them (usage_diff / vocab_shift)."""
    ...


def plot_report(
    model: Any,
    *,
    texts: Sequence[str] | None = None,
    timestamps: Sequence[object] | None = None,
    groups: Sequence[object] | None = None,
    n: int = 8,
    coherence_type: str = "c_v",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """A one-figure overview of a fitted model (prevalence, quality, correlation,
    and, when the inputs are given, topics over time and per class). Returns a
    matplotlib Figure. Requires matplotlib."""
    ...


def topic_label_prompts(
    model: Any,
    texts: Sequence[str] | None = None,
    *,
    n_words: int = 12,
    n_docs: int = 3,
    max_chars: int = 300,
    instructions: str | None = None,
) -> list[str]:
    """One labeling prompt per topic (top words + representative docs) — the
    plumbing behind llm_topic_labels."""
    ...


def llm_backend(
    model: str = "gpt-4o-mini",
    *,
    key: str | None = None,
    system: str | None = None,
    **options: Any,
) -> Callable[[str], str]:
    """A str -> str callable backed by the `llm` library, for llm_topic_labels'
    call= argument. The key defaults to llm's resolution (stored key or
    OPENAI_API_KEY etc.); pass `key` to override. Requires the optional `llm`
    package."""
    ...


def llm_topic_labels(
    model: Any,
    texts: Sequence[str] | None = None,
    *,
    call: Callable[[str], str] | None = None,
    llm_model: str = "gpt-4o-mini",
    n_words: int = 12,
    n_docs: int = 3,
    max_chars: int = 300,
    instructions: str | None = None,
    set_labels: bool = False,
) -> list[str]:
    """A short LLM-generated label per topic. Pass `call` (any str->str callable,
    zero deps) or name an `llm_model` (the topica[llm] extra). With
    set_labels=True the labels flow into topic_info / plot_report."""
    ...


__all__ = [
    "LDA",
    "DMR",
    "LabeledLDA",
    "SAGE",
    "CTM",
    "STM",
    "HDP",
    "DTM",
    "SupervisedLDA",
    "Corpus",
    "tokenize",
    "one_hot",
    "stm",
    "coherence",
    "topic_diversity",
    "topic_semantic_diversity",
    "exclusivity",
    "word_intrusion",
    "document_intrusion",
    "frex",
    "label_topics",
    "topic_correlation",
    "find_thoughts",
    "search_k",
    "relevance",
    "prepare_pyldavis",
    "check_residuals",
    "align_topics",
    "topic_stability",
    "topic_info",
    "topic_sizes",
    "topic_labels",
    "set_topic_labels",
    "representative_docs",
    "topics_over_time",
    "topics_per_class",
    "contrastive_topics",
    "plot_report",
    "llm_topic_labels",
    "llm_backend",
    "topic_label_prompts",
    "TopicGPT",
    "align_corpus",
    "spline",
    "interaction",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
    "__citation__",
]
