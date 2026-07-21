"""The model registry: one entry per exported model, the single source of truth
for every model list (the README table, the docs roster, and the programmatic
``topica.list_models`` discovery helper).

The flat ``from topica import X`` namespace is frozen and stays flat; this module
is a *presentation and discovery* layer over it, not a second import path. A
conformance test (``tests/test_registry.py``) asserts the registry and the
exported model classes stay in one-to-one correspondence, so neither drifts as
models are added.

Adding a model: export its class from ``__init__`` and add one ``ModelInfo`` here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One model's place in the taxonomy and its cross-cutting properties.

    Attributes
    ----------
    name : the exported class name (``"LDA"`` resolves as ``topica.LDA``).
    group : the purpose group (one of :data:`GROUPS`).
    brings : what the user supplies beyond raw text — any of ``"text"``,
        ``"embeddings"``, ``"metadata"``, ``"seeds"``, ``"labels"``, ``"times"``.
    inference : the inference engine — ``"gibbs"``, ``"variational"``, ``"vae"``,
        ``"optimal-transport"``, ``"clustering"``, ``"neural-embedding"``
        (word2vec/doc2vec-style SGD) (more as models are added).
    determinism : ``"bit-exact"`` (identical regardless of thread count),
        ``"seed-reproducible"`` (identical from a fixed seed and thread count), or
        ``"llm-bounded"`` (subject to an external model's nondeterminism).
    tags : cross-cutting labels for filtered views (e.g. ``"short-text"``,
        ``"nonparametric"``, ``"temporal"``, ``"hierarchical"``, ``"cross-lingual"``).
    summary : a one-line description for tables.
    doc : the docs anchor, relative to the docs root.
    experimental : ``True`` for a model that ships before it has a published
        paper and a reference-implementation parity check (topica's bar for a
        *validated* model). Experimental models are gated at construction (see
        :func:`topica.enable_experimental`) and listed apart from the validated
        roster; they may change or be removed without a deprecation cycle.
    """

    name: str
    group: str
    brings: tuple[str, ...]
    inference: str
    determinism: str
    tags: tuple[str, ...]
    summary: str
    doc: str
    experimental: bool = False


# Purpose groups, in display order. Organized by what the user brings and wants,
# not by inference family.
GROUPS: dict[str, str] = {
    "general-purpose": "General-purpose",
    "covariates": "Covariates & structure",
    "guided": "Guided & supervised",
    "short-text": "Short text",
    "dynamic-hierarchical": "Dynamic & hierarchical",
    "embedding": "Embedding-based",
    "ideal-point": "Ideal point",
    "llm": "LLM-based",
}


def _m(*args, **kwargs) -> ModelInfo:
    return ModelInfo(*args, **kwargs)


REGISTRY: dict[str, ModelInfo] = {
    m.name: m for m in [
        # ---- General-purpose ------------------------------------------------
        _m("LDA", "general-purpose", ("text",), "gibbs", "seed-reproducible", (),
           "Classic latent Dirichlet allocation via a fast SparseLDA collapsed-Gibbs sampler.",
           "guides/models.md#lda"),
        _m("CTM", "general-purpose", ("text",), "variational", "bit-exact", (),
           "Correlated topic model: a logistic-normal prior that lets topics co-occur.",
           "guides/models.md#ctm"),
        _m("ProdLDA", "general-purpose", ("text",), "vae", "seed-reproducible", (),
           "Product-of-experts LDA (AVITM) for sharper, more coherent topics; hand-coded VAE.",
           "guides/models.md#prodlda"),
        _m("HDP", "general-purpose", ("text",), "gibbs", "seed-reproducible", ("nonparametric",),
           "Hierarchical Dirichlet process: infers the number of topics from the data.",
           "guides/models.md#hdp"),
        _m("NMF", "general-purpose", ("text",), "matrix-factorization", "bit-exact", (),
           "Non-negative matrix factorization of the document-term matrix via multiplicative updates.",
           "guides/models.md#nmf"),
        _m("LSA", "general-purpose", ("text",), "svd", "seed-reproducible", (),
           "Latent semantic analysis: a truncated SVD of the weighted document-term matrix.",
           "guides/models.md#lsa"),
        _m("AnchorLDA", "general-purpose", ("text",), "matrix-factorization", "bit-exact", (),
           "Anchor-words spectral recovery (Arora et al. 2013): deterministic, Gibbs-free topics from the word co-occurrence matrix.",
           "guides/models.md#anchorlda", experimental=True),
        _m("TensorLDA", "general-purpose", ("text",), "svd", "seed-reproducible", (),
           "Online Tensor LDA (Kangaslahti et al. 2026): deterministic method-of-moments topic modeling via second and third-order cumulants.",
           "guides/models.md#tensorlda", experimental=True),
        _m("PolylingualLDA", "general-purpose", ("text",), "gibbs", "seed-reproducible", ("cross-lingual",),
           "Polylingual topic model (Mimno et al. 2009): aligned topics across languages from document tuples that share one topic distribution.",
           "guides/models.md#polylinguallda"),
        # ---- Covariates & structure ----------------------------------------
        _m("STM", "covariates", ("text", "metadata"), "variational", "bit-exact", (),
           "Structural topic model: relate topic prevalence and content to covariates.",
           "guides/models.md#stm"),
        _m("STS", "covariates", ("text", "metadata"), "variational", "bit-exact", (),
           "Structural topic-and-sentiment model over document metadata.",
           "guides/models.md#sts"),
        _m("SAGE", "covariates", ("text", "metadata"), "gibbs", "seed-reproducible", (),
           "Sparse additive generative model: the same topic worded differently across groups.",
           "guides/models.md#sage"),
        _m("DMR", "covariates", ("text", "metadata"), "gibbs", "seed-reproducible", (),
           "Dirichlet-multinomial regression: a document-metadata prior on topic proportions.",
           "guides/models.md#dmr"),
        _m("GDMR", "covariates", ("text", "metadata"), "gibbs", "seed-reproducible", (),
           "Generalized DMR with a smooth (Legendre-basis) prior over continuous covariates.",
           "guides/models.md#gdmr"),
        _m("NarrativeTM", "covariates", ("text",), "gibbs", "seed-reproducible", ("temporal",),
           "Intra-document narrative trajectory model: captures how topic prevalence shifts across the progress of a text.",
           "guides/models.md#narrativetm", experimental=True),
        # ---- Guided & supervised -------------------------------------------
        _m("KeyATM", "guided", ("text", "seeds"), "gibbs", "seed-reproducible", (),
           "Keyword-assisted topics: anchor named topics with a few seed words each.",
           "guides/guided.md"),
        _m("SeededLDA", "guided", ("text", "seeds"), "gibbs", "seed-reproducible", (),
           "Seeded LDA: steer named topics toward supplied seed words.",
           "guides/guided.md"),
        _m("LabeledLDA", "guided", ("text", "labels"), "gibbs", "seed-reproducible", (),
           "Labeled LDA: each document label is a topic; tokens are restricted to its labels.",
           "guides/models.md#labeledlda"),
        _m("SupervisedLDA", "guided", ("text", "labels"), "gibbs", "seed-reproducible", (),
           "Supervised LDA: topics shaped to predict a per-document real-valued response.",
           "guides/models.md#supervisedlda"),
        # ---- Short text -----------------------------------------------------
        _m("GSDMM", "short-text", ("text",), "gibbs", "seed-reproducible", ("short-text",),
           "Gibbs-sampling Dirichlet mixture: one topic per short document.",
           "guides/short-text.md"),
        _m("PT", "short-text", ("text",), "gibbs", "seed-reproducible", ("short-text",),
           "Pseudo-document topic model: pool short texts into pseudo-documents.",
           "guides/short-text.md"),
        _m("BTM", "short-text", ("text",), "gibbs", "seed-reproducible", ("short-text",),
           "Biterm topic model: learns topics from corpus-level word co-occurrence (biterms).",
           "guides/short-text.md"),
        # ---- Dynamic & hierarchical ----------------------------------------
        _m("DTM", "dynamic-hierarchical", ("text", "times"), "variational", "seed-reproducible", ("temporal",),  # random seed default (gensim-style); init="spectral" is deterministic
           "Dynamic topic model: a fixed topic set whose word distributions drift across time slices.",
           "guides/models.md#dtm"),
        _m("DETM", "dynamic-hierarchical", ("text", "embeddings", "times"), "vae", "seed-reproducible",
           ("temporal",),
           "Dynamic embedded topic model: embedding-factored topics that drift across time slices, fit as an amortized VAE.",
           "guides/embedding.md"),
        _m("HLDA", "dynamic-hierarchical", ("text",), "gibbs", "seed-reproducible", ("hierarchical",),
           "Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics.",
           "guides/models.md#hierarchy-models"),
        _m("PA", "dynamic-hierarchical", ("text",), "gibbs", "seed-reproducible", ("hierarchical",),
           "Pachinko allocation: a DAG of super- and sub-topics.",
           "guides/models.md#hierarchy-models"),
        # ---- Embedding-based ------------------------------------------------
        _m("BERTopic", "embedding", ("text", "embeddings"), "clustering", "seed-reproducible", (),
           "Cluster document embeddings; label topics by class-based TF-IDF.",
           "guides/embedding.md"),
        _m("Top2Vec", "embedding", ("text", "embeddings"), "clustering", "seed-reproducible", (),
           "Topics as dense regions in a joint document-word embedding space.",
           "guides/embedding.md"),
        _m("ETM", "embedding", ("text", "embeddings"), "variational", "seed-reproducible", (),
           "Embedded topic model: topic-word distributions factored through word embeddings.",
           "guides/embedding.md"),
        _m("IdealPointTM", "embedding", ("text", "embeddings"), "variational", "seed-reproducible", (),
           "Topic model with a latent ideal-point head: each author gets a low-dimensional position that shifts within-topic word choice, with a per-topic discrimination. Consumes word tokens as counts (Wordfish with topics) or, when word embeddings are supplied to fit, factored through them as in ETM. The unsupervised, latent-trait twin of the STM content covariate.",
           "guides/models.md#idealpointtm", experimental=True),
        _m("Wordfish", "ideal-point", ("text",), "em", "bit-exact", (),
           "Poisson scaling (Slapin & Proksch 2008): an unsupervised one-dimensional ideal-point estimate from word frequencies alone, no topics. The word-frequency baseline companion to IdealPointTM.",
           "guides/models.md#wordfish"),
        _m("IdealPointSentenceTM", "embedding", ("text", "embeddings"), "em", "seed-reproducible", (),
           "Continuous ideal-point topic model over sentence/document embeddings: topics are Gaussian clusters whose centroids are displaced by a latent author position. The sentence-embedding sibling of IdealPointTM, fit by EM.",
           "guides/models.md#idealpointsentencetm", experimental=True),
        _m("TBIP", "ideal-point", ("text",), "variational", "seed-reproducible", (),
           "Text-Based Ideal Points (Vafa, Naidu & Blei 2020): a Poisson factorization whose neutral topic-word intensities are rescaled by a per-word ideological factor exp(x_s * eta_kv), with the author position x_s latent. Fit by the paper's mean-field variational inference (reparameterized SVI). Recovers ideological scales from unlabeled text.",
           "guides/models.md#tbip"),
        _m("PartyEmbeddings", "ideal-point", ("text", "metadata"), "neural-embedding", "seed-reproducible", (),
           "Party embeddings (Rheault & Cochrane 2020): a PV-DM paragraph-vector model trained by negative sampling with party-period metadata tags; the leading principal components of the learned party vectors give the ideological scale, and words share the space so a party's language can be read off by proximity. The corpus-trained word-embedding member of the ideal-point family.",
           "guides/models.md#partyembeddings"),
        _m("FASTopic", "embedding", ("text", "embeddings"), "optimal-transport", "seed-reproducible", (),
           "Topics from optimal-transport plans between document, topic, and word embeddings.",
           "guides/embedding.md"),
        _m("EmbeddingLDA", "embedding", ("text", "embeddings", "seeds"), "gibbs", "seed-reproducible", (),
           "Seeded LDA whose seed sets are expanded with nearest neighbors in an embedding space.",
           "guides/embedding.md"),
        _m("CombinedTM", "embedding", ("text", "embeddings"), "vae", "seed-reproducible", (),
           "Contextualized ProdLDA: encoder reads the bag of words plus a document embedding.",
           "guides/embedding.md#combinedtm"),
        _m("ZeroShotTM", "embedding", ("text", "embeddings"), "vae", "seed-reproducible", ("cross-lingual",),
           "Contextualized ProdLDA: encoder reads the document embedding alone, enabling cross-lingual transfer.",
           "guides/embedding.md#zeroshottm"),
        _m("InfoCTM", "embedding", ("text", "dictionary"), "vae", "seed-reproducible", ("cross-lingual",),
           "Cross-lingual: two ProdLDA models aligned by a bilingual dictionary through a mutual-information term.",
           "guides/models.md#infoctm"),
        # ---- LLM-based ------------------------------------------------------
        _m("TopicGPT", "llm", ("text", "llm"), "prompting", "llm-bounded", ("hierarchical",),
           "LLM-driven topic discovery: prompt a model to propose, refine, and assign a topic taxonomy with descriptions.",
           "guides/llm.md#topicgpt"),
    ]
}


def list_models(
    *,
    group: str | None = None,
    brings: str | None = None,
    inference: str | None = None,
    determinism: str | None = None,
    tag: str | None = None,
    experimental: bool | None = None,
) -> list[ModelInfo]:
    """Return the registered models matching every supplied filter.

    With no filters, returns all models in registry (insertion) order. Each
    filter narrows the result:

    - ``group`` — one of :data:`GROUPS` (e.g. ``"short-text"``).
    - ``brings`` — a single requirement the model accepts (e.g. ``"embeddings"``,
      ``"metadata"``, ``"seeds"``); matches models whose ``brings`` contains it.
    - ``inference`` — the engine (e.g. ``"gibbs"``, ``"variational"``, ``"vae"``).
    - ``determinism`` — ``"bit-exact"``, ``"seed-reproducible"``, ``"llm-bounded"``.
    - ``tag`` — a cross-cutting tag (e.g. ``"short-text"``, ``"nonparametric"``).
    - ``experimental`` — ``True`` for only the experimental (unvalidated) models,
      ``False`` for only the validated roster; the default ``None`` returns both.

    Examples
    --------
    >>> import topica
    >>> [m.name for m in topica.list_models(brings="embeddings")]
    ['BERTopic', 'Top2Vec', 'ETM', 'FASTopic', 'EmbeddingLDA']
    >>> [m.name for m in topica.list_models(group="short-text")]
    ['GSDMM', 'PT']
    """
    if group is not None and group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choose from {sorted(GROUPS)}")
    out = []
    for m in REGISTRY.values():
        if group is not None and m.group != group:
            continue
        if brings is not None and brings not in m.brings:
            continue
        if inference is not None and m.inference != inference:
            continue
        if determinism is not None and m.determinism != determinism:
            continue
        if tag is not None and tag not in m.tags:
            continue
        if experimental is not None and m.experimental != experimental:
            continue
        out.append(m)
    return out


def markdown_table(by_group: bool = True) -> str:
    """Render the registry as a Markdown table, grouped by purpose by default.

    Used to (re)generate the README model section and the docs roster so the
    hand-maintained lists cannot drift from the registry.
    """
    def _rows(models: list[ModelInfo]) -> list[str]:
        rows = ["| Model | Brings | Inference | Reproducibility | Summary |",
                "|---|---|---|---|---|"]
        for m in models:
            brings = ", ".join(m.brings)
            rows.append(
                f"| `{m.name}` | {brings} | {m.inference} | {m.determinism} | {m.summary} |"
            )
        return rows

    lines: list[str] = []
    if by_group:
        # Validated models, grouped by purpose. Experimental models are held out
        # and listed in their own section below so the roster above is exactly
        # the paper-backed, parity-checked set.
        for key, label in GROUPS.items():
            models = [m for m in REGISTRY.values() if m.group == key and not m.experimental]
            if not models:
                continue
            lines.append(f"### {label}\n")
            lines += _rows(models)
            lines.append("")
        experimental = [m for m in REGISTRY.values() if m.experimental]
        if experimental:
            lines.append("### Experimental\n")
            lines.append(
                "Shipped before a published paper and reference-implementation parity "
                "(topica's bar for a validated model). Gated: call "
                "`topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) before "
                "use. These may change or be removed without a deprecation cycle.\n"
            )
            lines += _rows(experimental)
            lines.append("")
    else:
        lines.append("| Model | Group | Brings | Inference | Reproducibility | Summary |")
        lines.append("|---|---|---|---|---|---|")
        for m in REGISTRY.values():
            brings = ", ".join(m.brings)
            lines.append(
                f"| `{m.name}` | {GROUPS[m.group]} | {brings} | {m.inference} | "
                f"{m.determinism} | {m.summary} |"
            )
    return "\n".join(lines)
