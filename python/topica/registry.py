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
        _m("Scholar", "covariates", ("text", "metadata", "labels"), "vae", "seed-reproducible", (),
           "SCHOLAR (Card et al. 2018): a ProdLDA VAE with a covariate-shifted prevalence prior, an optional supervised label head, and optional content (topic-covariate) word deviations — neural STM prevalence + sLDA + SAGE.",
           "guides/models.md#scholar"),
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
        _m("DiscLDA", "guided", ("text", "labels"), "gibbs", "seed-reproducible", (),
           "Discriminative LDA (Lacoste-Julien et al. 2008): topics split into per-class and shared blocks; reads how classes talk differently.",
           "guides/models.md#disclda"),
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


@dataclass(frozen=True)
class ImplInfo:
    """Where a model lives in the source tree and how it is validated (#381).

    The companion to :class:`ModelInfo`: that one is the user-facing taxonomy,
    this one is the contributor's map from a model name to the files a change
    touches. Keyed by the same model name, so a model cannot appear in one and
    not the other (``tests/test_registry.py`` asserts ``IMPL`` and ``REGISTRY``
    cover exactly the same set). Every path is checked to exist and every Cargo
    feature checked to be real by the same test, so this map cannot silently rot
    into a stale inventory.

    Attributes
    ----------
    source : the core algorithm file — a Rust ``src/*.rs`` / ``topica-core/``
        file, or a ``python/topica/*.py`` module for the pure-Python models.
    binding : the PyO3 binding location (``src/python/mod.rs`` or an extracted
        ``src/python/<model>.rs`` module); ``""`` for pure-Python models with no
        binding of their own.
    core : the shared machinery / family the model builds on (free text).
    feature : the Cargo feature required beyond the default build (``""`` builds
        with a plain ``cargo build``; ``"embeddings"`` for the clustering models).
    validation : reference-parity / gold / test artifacts, as comma-separated
        repo-relative paths. Every registered model is additionally covered by
        the conformance suite (``tests/test_conformance.py``).
    """

    source: str
    binding: str
    core: str
    feature: str
    validation: str


def _i(source: str, binding: str, core: str, feature: str, validation: str) -> ImplInfo:
    return ImplInfo(source, binding, core, feature, validation)


# name -> where it lives + how it is validated. Curated, but path- and
# feature-validated in CI (tests/test_registry.py), and required to cover exactly
# the same models as REGISTRY. Add a model here when you add it to REGISTRY.
IMPL: dict[str, ImplInfo] = {
    "LDA": _i("src/model.rs", "src/python/mod.rs", "SparseLDA collapsed Gibbs (model.rs, sampler.rs)", "", "parity/lda_gold.py, parity/mallet_parity.py"),
    "CTM": _i("topica-core/src/ctm.rs", "src/python/mod.rs", "CTM/STM variational core (topica-core)", "", "parity/ctm_gold.py, parity/ctm_r_compare.py"),
    "ProdLDA": _i("src/prodlda.rs", "src/python/mod.rs", "hand-coded batched VAE (prodlda.rs)", "", "parity/prodlda_gold.py, parity/prodlda_compare.py"),
    "HDP": _i("src/hdp.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/hdp_gold.py"),
    "NMF": _i("src/nmf.rs", "src/python/nmf_lsa.rs", "multiplicative-update matrix factorization", "", "parity/nmf_vs_sklearn.py"),
    "LSA": _i("src/lsa.rs", "src/python/nmf_lsa.rs", "truncated SVD (linalg)", "", "parity/lsa_vs_sklearn.py"),
    "AnchorLDA": _i("python/topica/anchor.py", "", "spectral anchor-word recovery (Python over Rust primitives)", "", "tests/test_anchor.py"),
    "TensorLDA": _i("src/tlda.rs", "src/python/tlda.rs", "method-of-moments cumulants (linalg, spectral)", "", "parity/tlda_gold.py, parity/tlda_compare.py"),
    "PolylingualLDA": _i("src/pltm.rs", "src/python/pltm.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/pltm_compare.py"),
    "STM": _i("src/sts.rs", "src/python/mod.rs", "CTM/STM variational core (topica-core)", "", "parity/stm_gold.py, parity/stm_r_compare.py"),
    "STS": _i("src/sts.rs", "src/python/mod.rs", "CTM/STM variational core (topica-core)", "", "parity/sts_gold.py, parity/sts_r_compare.py"),
    "SAGE": _i("src/sage.rs", "src/python/mod.rs", "collapsed Gibbs + SAGE deviations", "", "parity/sage_gold.py"),
    "DMR": _i("src/dmr.rs", "src/python/mod.rs", "collapsed Gibbs + DMR prior (optimize.rs)", "", "parity/dmr_gold.py"),
    "GDMR": _i("src/dmr.rs", "src/python/mod.rs", "collapsed Gibbs + DMR prior (optimize.rs)", "", "parity/gdmr_gold.py, parity/test_gdmr_tomotopy.py"),
    "Scholar": _i("src/scholar.rs", "src/python/scholar.rs", "ProdLDA VAE + covariate prior (prodlda.rs)", "", "tests/test_scholar.py"),
    "NarrativeTM": _i("python/topica/narrative.py", "", "intra-document trajectory over Gibbs core (Python)", "", "tests/test_content_trajectory.py"),
    "KeyATM": _i("src/keyatm.rs", "src/python/mod.rs", "collapsed Gibbs + keyword index", "", "parity/keyatm_gold.py, parity/keyatm_r_compare.py"),
    "SeededLDA": _i("src/seeded.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/seededlda_gold.py"),
    "LabeledLDA": _i("src/labeled.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/labeledlda_gold.py"),
    "SupervisedLDA": _i("src/slda.rs", "src/python/mod.rs", "collapsed Gibbs + response head", "", "parity/supervisedlda_gold.py"),
    "DiscLDA": _i("src/disclda.rs", "src/python/disclda.rs", "collapsed Gibbs + class transform", "", "parity/disclda_20ng.py, tests/test_disclda.py"),
    "GSDMM": _i("src/gsdmm.rs", "src/python/mod.rs", "collapsed Gibbs mixture (one topic/doc)", "", "parity/gsdmm_gold.py"),
    "PT": _i("src/pt.rs", "src/python/mod.rs", "collapsed Gibbs over pseudo-documents", "", "parity/pt_gold.py"),
    "BTM": _i("src/btm.rs", "src/python/btm.rs", "collapsed Gibbs over biterms", "", "parity/btm_compare.py, tests/test_btm.py"),
    "DTM": _i("src/dtm.rs", "src/python/mod.rs", "variational Kalman over time slices", "", "parity/dtm_gold.py"),
    "DETM": _i("src/detm.rs", "src/python/mod.rs", "embedding VAE + LSTM q(eta) (etm_vae.rs)", "", "parity/detm_gold.py"),
    "HLDA": _i("src/hlda.rs", "src/python/hierarchical.rs", "nested-CRP collapsed Gibbs", "", "parity/hlda_gold.py"),
    "PA": _i("src/pa.rs", "src/python/hierarchical.rs", "collapsed Gibbs over a topic DAG", "", "parity/pa_gold.py"),
    "BERTopic": _i("src/bertopic.rs", "src/python/embedding_cluster.rs", "embedding clustering (cluster.rs, reduce.rs, represent.rs)", "embeddings", "parity/bertopic_gold.py"),
    "Top2Vec": _i("src/top2vec.rs", "src/python/embedding_cluster.rs", "embedding clustering (cluster.rs, reduce.rs)", "embeddings", "parity/top2vec_gold.py, parity/top2vec_compare.py"),
    "ETM": _i("src/etm.rs", "src/python/mod.rs", "variational EM over word embeddings (ctm.rs)", "", "parity/etm_gold.py"),
    "IdealPointTM": _i("src/idealpoint.rs", "src/python/idealpoint.rs", "variational EM + ideal-point head", "", "tests/test_idealpoint.py, tests/test_idealpoint_counts.py"),
    "Wordfish": _i("src/wordfish.rs", "src/python/wordfish.rs", "Poisson-scaling EM", "", "parity/wordfish_r_compare.py, tests/test_wordfish.py"),
    "IdealPointSentenceTM": _i("src/sentence_ideal.rs", "src/python/sentence_ideal.rs", "Gaussian-cluster EM over embeddings", "", "tests/test_sentence_ideal.py"),
    "TBIP": _i("src/tbip.rs", "src/python/tbip.rs", "Poisson-factorization mean-field SVI", "", "parity/tbip_parity.py, tests/test_tbip.py"),
    "PartyEmbeddings": _i("src/party_embeddings.rs", "src/python/party_embeddings.rs", "PV-DM paragraph vectors (negative sampling)", "", "parity/party_embeddings_compare.py, tests/test_party_embeddings.py"),
    "FASTopic": _i("src/fastopic.rs", "src/python/mod.rs", "reverse-mode Sinkhorn optimal transport", "", "parity/fastopic_gold.py, parity/fastopic_compare.py"),
    "EmbeddingLDA": _i("python/topica/embedding.py", "", "seeded Gibbs + embedding NN expansion (Python)", "", "parity/embeddinglda_gold.py, tests/test_embedding_lda.py"),
    "CombinedTM": _i("src/prodlda.rs", "src/python/mod.rs", "contextualized ProdLDA VAE (prodlda.rs)", "", "parity/combinedtm_gold.py, parity/combinedtm_compare.py"),
    "ZeroShotTM": _i("src/prodlda.rs", "src/python/mod.rs", "contextualized ProdLDA VAE (prodlda.rs)", "", "parity/zeroshot_gold.py, parity/zeroshot_compare.py"),
    "InfoCTM": _i("src/infoctm.rs", "src/python/mod.rs", "two ProdLDA VAEs + TAMI alignment (prodlda.rs)", "", "parity/infoctm_gold.py, parity/infoctm_compare.py"),
    "TopicGPT": _i("python/topica/llm.py", "", "LLM prompting pipeline (Python)", "", "tests/test_topicgpt.py"),
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


def impl_markdown_table(by_group: bool = True) -> str:
    """Render the implementation map (:data:`IMPL`) as a Markdown table.

    One row per model: where its algorithm lives, where its PyO3 binding lives,
    the shared machinery it builds on, the Cargo feature it needs, and where its
    reference-parity / gold tests live. Grouped by purpose to match the roster.
    Used to (re)generate ``docs/contributing/model-map.md`` so the map cannot
    drift from the source tree.
    """
    header = ("| Model | Source | Binding | Core / family | Feature | Validation |",
              "|---|---|---|---|---|---|")

    def _cell_paths(csv: str) -> str:
        return ", ".join(f"`{p.strip()}`" for p in csv.split(","))

    def _rows(models: list[ModelInfo]) -> list[str]:
        rows = list(header)
        for m in models:
            im = IMPL[m.name]
            binding = f"`{im.binding}`" if im.binding else "— _(Python)_"
            feature = f"`{im.feature}`" if im.feature else "default"
            rows.append(
                f"| `{m.name}` | `{im.source}` | {binding} | {im.core} | "
                f"{feature} | {_cell_paths(im.validation)} |"
            )
        return rows

    lines: list[str] = []
    if by_group:
        for key, label in GROUPS.items():
            models = [m for m in REGISTRY.values() if m.group == key]
            if not models:
                continue
            lines.append(f"### {label}\n")
            lines += _rows(models)
            lines.append("")
    else:
        lines += _rows(list(REGISTRY.values()))
    return "\n".join(lines).rstrip()


def validate_impl() -> list[str]:
    """Return a list of problems with :data:`IMPL` (empty when it is sound).

    Checks that (1) IMPL covers exactly the registry, (2) every ``source`` /
    ``binding`` / ``validation`` path exists, and (3) every ``feature`` is a real
    Cargo feature. The anti-staleness guard the contributor map depends on;
    ``tests/test_registry.py`` fails on any returned problem.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    problems: list[str] = []

    missing = set(REGISTRY) - set(IMPL)
    extra = set(IMPL) - set(REGISTRY)
    if missing:
        problems.append(f"IMPL is missing models: {sorted(missing)}")
    if extra:
        problems.append(f"IMPL has non-registry models: {sorted(extra)}")

    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    features = set(re.findall(r"^([\w-]+)\s*=\s*\[", cargo, re.M))

    for name, im in IMPL.items():
        paths = [im.source] + ([im.binding] if im.binding else [])
        paths += [p.strip() for p in im.validation.split(",")]
        for p in paths:
            if not (root / p).exists():
                problems.append(f"{name}: path does not exist: {p}")
        if im.feature and im.feature not in features:
            problems.append(f"{name}: unknown Cargo feature {im.feature!r}")
    return problems
