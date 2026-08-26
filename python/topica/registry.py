"""The model registry: one entry per exported model, the single source of truth
for every model list (the README table, the docs roster, and the programmatic
``topica.list_models`` discovery helper).

Model classes stay at the top level (``from topica import LDA``) — the analysis
*helpers* moved into workflow namespaces (issue #757), but the model constructors
are nouns and remain flat. This module is a *presentation and discovery* layer
over them, not a second import path. A conformance test
(``tests/test_registry.py``) asserts the registry and the exported model classes
stay in one-to-one correspondence, so neither drifts as models are added.

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
        ``"embeddings"``, ``"metadata"``, ``"seeds"``, ``"labels"``, ``"times"``,
        ``"links"`` (a document graph).
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
    experimental : ``True`` for a model that has not earned a place on the
        validated roster, on either of two grounds: it is **unpublished** (a
        topica original with no paper, so it can only ever be validated by
        planted recovery, not by an independent yardstick), or it is a published
        method topica ships whose **benefit has not held up** against a simpler
        baseline. A published method with faithful inference stays validated even
        when its accuracy basis is planted-recovery (no maintained reference
        implementation exists); planted-recovery is a validation *basis*, not an
        experimental marker. Experimental models are gated at construction (see
        :func:`topica.enable_experimental`) and listed apart from the validated
        roster; they may change or be removed without a deprecation cycle.
        Graduation is the triple gate (accuracy, adversarial, user); see
        ``docs/contributing/validation.md``.
    common_start : ``True`` for a *common starting point*: one of the handful of
        models most social scientists reach for first, chosen by contemporary
        popularity and applied usefulness. This is an editorial suggestion about
        where a newcomer commonly begins, NOT a quality ranking: every model here
        is equally reference-validated, and a specialized model (short text, over
        time, networks, multiple languages, ideological scaling) is the correct
        first choice for its own design. Orthogonal to ``experimental``, which is
        a validation status.
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
    common_start: bool = False

    def __lt__(self, other: "ModelInfo") -> bool:
        # Order by name so ``sorted(list_models())`` just works (issue #742). A
        # frozen dataclass without ``order=True`` is unorderable by default, and
        # comparing every field would be meaningless here — name is the identity.
        if not isinstance(other, ModelInfo):
            return NotImplemented
        return self.name < other.name


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
           "guides/models.md#lda", common_start=True),
        _m("OnlineLDA", "general-purpose", ("text",), "variational", "seed-reproducible", ("streaming",),
           "Online (streaming) variational-Bayes LDA (Hoffman et al. 2010): minibatch stochastic VB with a decaying learning rate and a streaming partial_fit; the gensim LdaModel analogue for very large or streaming corpora.",
           "guides/models.md#onlinelda"),
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
           "guides/models.md#nmf", common_start=True),
        _m("LSA", "general-purpose", ("text",), "svd", "seed-reproducible", (),
           "Latent semantic analysis: a truncated SVD of the weighted document-term matrix.",
           "guides/models.md#lsa"),
        _m("AnchorLDA", "general-purpose", ("text",), "matrix-factorization", "bit-exact", (),
           "Anchor-words spectral recovery (Arora et al. 2013): deterministic, Gibbs-free topics from the word co-occurrence matrix.",
           "guides/models.md#anchorlda"),
        _m("TensorLDA", "general-purpose", ("text",), "svd", "seed-reproducible", (),
           "Online Tensor LDA (Kangaslahti et al. 2026): deterministic method-of-moments topic modeling via second and third-order cumulants.",
           "guides/models.md#tensorlda", experimental=True),
        _m("PolylingualLDA", "general-purpose", ("text",), "gibbs", "seed-reproducible", ("cross-lingual",),
           "Polylingual topic model (Mimno et al. 2009): aligned topics across languages from document tuples that share one topic distribution.",
           "guides/models.md#polylinguallda"),
        # ---- Covariates & structure ----------------------------------------
        _m("STM", "covariates", ("text", "metadata"), "variational", "bit-exact", (),
           "Structural topic model: relate topic prevalence and content to covariates.",
           "guides/models.md#stm", common_start=True),
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
        _m("RTM", "covariates", ("text", "links"), "variational", "seed-reproducible", ("network",),
           "Relational topic model (Chang & Blei 2010): jointly models document text and a link graph (citations, hyperlinks, adjacency); predicts links from words and words from links.",
           "guides/models.md#rtm"),
        _m("FactorialLDA", "covariates", ("text",), "gibbs", "seed-reproducible", (),
           "Factorial LDA (Paul & Dredze 2012): each token is a K-tuple of latent factors (e.g. topic x sentiment); structured word priors tie tuples sharing a component and a sparsity prior deactivates unsupported tuples.",
           "guides/models.md#factorial-lda"),
        # ---- Guided & supervised -------------------------------------------
        _m("KeyATM", "guided", ("text", "seeds"), "gibbs", "seed-reproducible", (),
           "Keyword-assisted topics: anchor named topics with a few seed words each.",
           "guides/guided.md", common_start=True),
        _m("SeededLDA", "guided", ("text", "seeds"), "gibbs", "seed-reproducible", (),
           "Seeded LDA: steer named topics toward supplied seed words.",
           "guides/guided.md"),
        _m("GuidedNMF", "guided", ("text", "seeds"), "matrix-factorization", "seed-reproducible", (),
           "Guided NMF: seed-word-guided semi-supervised NMF; the matrix-factorization analogue of SeededLDA.",
           "guides/models.md#guidednmf"),
        _m("KeyNMF", "embedding", ("text", "embeddings"), "matrix-factorization", "bit-exact", (),
           "KeyNMF (Kristensen-McLachlan et al. 2024): NMF over an embedding-derived keyword-importance matrix. For each document it scores its words by the similarity between the document embedding and the word embedding, keeps the top-N positive, and factors that sparse doc-word matrix. The bridge between the count-based NMF family and the embedding backend; sparse, readable topics robust to short/noisy text.",
           "guides/models.md#keynmf"),
        _m("CorEx", "general-purpose", ("text",), "information-theoretic", "seed-reproducible", (),
           "Correlation Explanation: information-theoretic topic model that maximizes total correlation; supports anchor words.",
           "guides/models.md#corex"),
        _m("AuthorTopic", "covariates", ("text", "metadata"), "gibbs", "seed-reproducible", (),
           "Author-Topic Model: each author has a topic distribution; documents mix their authors. Answers what an author writes about.",
           "guides/models.md#authortopic"),
        _m("AuthorRecipientTopic", "covariates", ("text", "metadata"), "gibbs", "seed-reproducible", ("network",),
           "Author-Recipient-Topic (McCallum et al. 2007): topics conditioned on the (sender, recipient) pair, for the language of a directed social network (who talks to whom about what). Realized over the AuthorTopic engine.",
           "guides/models.md#authorrecipienttopic"),
        _m("MGLDA", "general-purpose", ("text",), "gibbs", "seed-reproducible", (),
           "Multi-Grain LDA: global (document-level) + local (sliding-window aspect) topics with a per-token grain switch. For reviews / aspect extraction.",
           "guides/models.md#mglda"),
        _m("LabeledLDA", "guided", ("text", "labels"), "gibbs", "seed-reproducible", (),
           "Labeled LDA: each document label is a topic; tokens are restricted to its labels.",
           "guides/models.md#labeledlda"),
        _m("SupervisedLDA", "guided", ("text", "labels"), "variational", "seed-reproducible", (),
           "Supervised LDA: topics shaped to predict a per-document real-valued response.",
           "guides/models.md#supervisedlda"),
        _m("DiscLDA", "guided", ("text", "labels"), "gibbs", "seed-reproducible", (),
           "Discriminative LDA (Lacoste-Julien et al. 2008): topics split into per-class and shared blocks; reads how classes talk differently.",
           "guides/models.md#disclda"),
        # ---- Short text -----------------------------------------------------
        _m("GSDMM", "short-text", ("text",), "gibbs", "seed-reproducible", ("short-text",),
           "Gibbs-sampling Dirichlet mixture: one topic per short document.",
           "guides/short-text.md", common_start=True),
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
        _m("TopicsOverTime", "dynamic-hierarchical", ("text", "times"), "gibbs", "seed-reproducible",
           ("temporal",),
           "Topics over Time: LDA with a per-topic Beta density over continuous timestamps; each topic has a temporal peak. Descriptive continuous-time prevalence (not vocabulary drift).",
           "guides/models.md#topicsovertime"),
        _m("HLDA", "dynamic-hierarchical", ("text",), "gibbs", "seed-reproducible", ("hierarchical",),
           "Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics.",
           "guides/models.md#hierarchy-models"),
        _m("PA", "dynamic-hierarchical", ("text",), "gibbs", "seed-reproducible", ("hierarchical",),
           "Pachinko allocation: a DAG of super- and sub-topics.",
           "guides/models.md#hierarchy-models"),
        # ---- Embedding-based ------------------------------------------------
        _m("BERTopic", "embedding", ("text", "embeddings"), "clustering", "seed-reproducible", (),
           "Cluster document embeddings; label topics by class-based TF-IDF.",
           "guides/embedding.md", common_start=True),
        _m("Top2Vec", "embedding", ("text", "embeddings"), "clustering", "seed-reproducible", (),
           "Topics as dense regions in a joint document-word embedding space.",
           "guides/embedding.md"),
        _m("SemanticSignalSeparation", "embedding", ("text", "embeddings"), "ica", "seed-reproducible", (),
           "Topics as independent axes of semantic space (S3, Kardos et al. 2025): FastICA over the document embeddings, with each word's importance read off by projecting the vocabulary embeddings onto each axis. Signed poles.",
           "guides/models.md#semanticsignalseparation"),
        _m("ETM", "embedding", ("text", "embeddings"), "variational", "seed-reproducible", (),
           "Embedded topic model: topic-word distributions factored through word embeddings.",
           "guides/embedding.md"),
        _m("GaussianLDA", "embedding", ("text", "embeddings"), "gibbs", "seed-reproducible", (),
           "Gaussian LDA (Das, Zaheer & Dyer 2015): each topic is a Gaussian over the word-embedding space (Normal-Inverse-Wishart prior), so topics generalize over semantically similar words. Collapsed Gibbs with a Student-t posterior predictive and rank-1 Cholesky up/downdates.",
           "guides/models.md#gaussianlda"),
        _m("IdealPointTM", "embedding", ("text", "embeddings"), "variational", "seed-reproducible", (),
           "Topic model with a latent ideal-point head: each author gets a low-dimensional position that shifts within-topic word choice, with a per-topic discrimination. Consumes word tokens as counts (Wordfish with topics) or, when word embeddings are supplied to fit, factored through them as in ETM. The unsupervised, latent-trait twin of the STM content covariate.",
           "guides/models.md#idealpointtm", experimental=True),
        _m("Wordfish", "ideal-point", ("text",), "em", "bit-exact", (),
           "Poisson scaling (Slapin & Proksch 2008): an unsupervised one-dimensional ideal-point estimate from word frequencies alone, no topics. The word-frequency baseline companion to IdealPointTM.",
           "guides/models.md#wordfish"),
        _m("TopicalNGrams", "general-purpose", ("text",), "gibbs", "seed-reproducible",
           ("phrases",),
           "Topical N-Grams (Wang, McCallum & Wei 2007): an LDA extension that jointly discovers topics and topic-specific multiword phrases. A per-token bigram-status indicator, sampled with the topic, decides whether a token continues a phrase from the previous word given its topic, so phrase structure is learned during fitting rather than fixed beforehand. Exposes top_phrases alongside top_words.",
           "guides/models.md#topicalngrams"),
        _m("Wordshoal", "ideal-point", ("text", "metadata"), "em", "bit-exact", (),
           "Multi-domain scaling (Lauderdale & Herzog 2016): scales each debate/domain with Wordfish, then combines the within-domain positions into one cross-domain actor scale via a linear factor model. The multi-domain extension of Wordfish, for speeches carrying trusted debate labels.",
           "guides/models.md#wordshoal"),
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
        _m("EmbeddingLDA", "embedding", ("text", "embeddings"), "gibbs", "seed-reproducible", (),
           "LDA anchored by pre-trained embeddings: k-means clusters the vocabulary embeddings, seeds each topic with the words nearest a cluster centroid, and (optionally) biases each document's mixture toward its own embedding. A topica original; validated by planted-recovery only.",
           "guides/embedding.md", experimental=True),
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
    "OnlineLDA": _i("src/online_lda.rs", "src/python/online_lda.rs", "online-VB SVI schedule (variational/svi.rs), Dirichlet mean-field E-step (optimize.rs digamma)", "", "parity/online_lda_gensim_compare.py"),
    "CTM": _i("topica-core/src/ctm.rs", "src/python/mod.rs", "CTM/STM variational core (topica-core)", "", "parity/ctm_gold.py, parity/ctm_r_compare.py"),
    "ProdLDA": _i("src/prodlda.rs", "src/python/neural.rs", "hand-coded batched VAE (prodlda.rs)", "", "parity/prodlda_gold.py, parity/prodlda_compare.py"),
    "HDP": _i("src/hdp.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/hdp_gold.py"),
    "NMF": _i("src/nmf.rs", "src/python/nmf_lsa.rs", "multiplicative-update matrix factorization", "", "parity/nmf_vs_sklearn.py"),
    "GuidedNMF": _i("src/guided_nmf.rs", "src/python/guided_nmf.rs", "supervised multiplicative-update matrix factorization", "", "parity/guidednmf_gold.py"),
    "KeyNMF": _i("src/keynmf.rs", "src/python/keynmf.rs", "embedding-keyword extraction + multiplicative-update NMF (nmf.rs)", "", "parity/keynmf_compare.py, tests/test_keynmf.py"),
    "CorEx": _i("src/cor_ex.rs", "src/python/cor_ex.rs", "total-correlation info-theoretic optimizer", "", "parity/corex_gold.py"),
    "AuthorTopic": _i("src/author_topic.rs", "src/python/author_topic.rs", "collapsed Gibbs (author×topic + word×topic counts)", "", "parity/author_topic_gold.py"),
    "AuthorRecipientTopic": _i("python/topica/art.py", "", "ART as (sender,recipient)-pair isomorphism over the AuthorTopic Gibbs core (Python)", "", "tests/test_art.py"),
    "MGLDA": _i("src/mg_lda.rs", "src/python/mg_lda.rs", "two-grain collapsed Gibbs over sliding sentence windows", "", "parity/mglda_gold.py"),
    "TopicsOverTime": _i("src/topics_over_time.rs", "src/python/topics_over_time.rs", "collapsed Gibbs (LDA + per-topic Beta time factor, method-of-moments psi)", "", "parity/tot_gold.py"),
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
    "RTM": _i("src/rtm.rs", "src/python/rtm.rs", "variational EM + link head (optimize.rs digamma)", "", "parity/rtm_compare.py, parity/rtm_reference.py, tests/test_rtm.py"),
    "NarrativeTM": _i("python/topica/narrative.py", "", "intra-document trajectory over Gibbs core (Python)", "", "tests/test_content_trajectory.py"),
    "KeyATM": _i("src/keyatm.rs", "src/python/mod.rs", "collapsed Gibbs + keyword index", "", "parity/keyatm_gold.py, parity/keyatm_r_compare.py"),
    "SeededLDA": _i("src/seeded.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/seededlda_gold.py"),
    "LabeledLDA": _i("src/labeled.rs", "src/python/mod.rs", "collapsed Gibbs (model.rs, sampler.rs)", "", "parity/labeledlda_gold.py"),
    "SupervisedLDA": _i("src/slda.rs", "src/python/mod.rs", "variational EM + Gaussian response head", "", "parity/supervisedlda_gold.py"),
    "DiscLDA": _i("src/disclda.rs", "src/python/disclda.rs", "collapsed Gibbs + class transform", "", "parity/disclda_20ng.py, tests/test_disclda.py"),
    "GSDMM": _i("src/gsdmm.rs", "src/python/mod.rs", "collapsed Gibbs mixture (one topic/doc)", "", "parity/gsdmm_gold.py"),
    "PT": _i("src/pt.rs", "src/python/mod.rs", "collapsed Gibbs over pseudo-documents", "", "parity/pt_gold.py"),
    "BTM": _i("src/btm.rs", "src/python/btm.rs", "collapsed Gibbs over biterms", "", "parity/btm_compare.py, tests/test_btm.py"),
    "FactorialLDA": _i("src/factorial_lda.rs", "src/python/factorial_lda.rs", "collapsed Gibbs over tuples + MCEM gradient ascent on log-linear priors", "", "parity/factorial_lda_compare.py, tests/test_factorial_lda.py"),
    "DTM": _i("src/dtm.rs", "src/python/mod.rs", "variational Kalman over time slices", "", "parity/dtm_gold.py"),
    "DETM": _i("src/detm.rs", "src/python/neural.rs", "embedding VAE + LSTM q(eta) (etm_vae.rs)", "", "parity/detm_gold.py"),
    "HLDA": _i("src/hlda.rs", "src/python/hierarchical.rs", "nested-CRP collapsed Gibbs", "", "parity/hlda_gold.py"),
    "PA": _i("src/pa.rs", "src/python/hierarchical.rs", "collapsed Gibbs over a topic DAG", "", "parity/pa_gold.py"),
    "BERTopic": _i("src/bertopic.rs", "src/python/embedding_cluster.rs", "embedding clustering (cluster.rs, reduce.rs, represent.rs)", "embeddings", "parity/bertopic_gold.py"),
    "Top2Vec": _i("src/top2vec.rs", "src/python/embedding_cluster.rs", "embedding clustering (cluster.rs, reduce.rs)", "embeddings", "parity/top2vec_gold.py, parity/top2vec_compare.py"),
    "SemanticSignalSeparation": _i("src/semantic_signal_separation.rs", "src/python/semantic_signal_separation.rs", "FastICA over document embeddings + vocabulary projection (reduce.rs)", "embeddings", "parity/s3_compare.py, tests/test_semantic_signal_separation.py"),
    "ETM": _i("src/etm.rs", "src/python/neural.rs", "variational EM over word embeddings (ctm.rs)", "", "parity/etm_gold.py"),
    "GaussianLDA": _i("src/gaussian_lda.rs", "src/python/gaussian_lda.rs", "collapsed Gibbs (NIW-Gaussian topics, Student-t predictive, rank-1 Cholesky up/downdates)", "", "parity/gaussian_lda_gold.py"),
    "IdealPointTM": _i("src/idealpoint.rs", "src/python/idealpoint.rs", "variational EM + ideal-point head", "", "tests/test_idealpoint.py, tests/test_idealpoint_counts.py"),
    "TopicalNGrams": _i("src/topical_ngrams.rs", "src/python/topical_ngrams.rs", "collapsed Gibbs over token sequences (joint topic + bigram-status)", "", "parity/tng_mallet_compare.py, tests/test_topical_ngrams.py"),
    "Wordfish": _i("src/wordfish.rs", "src/python/wordfish.rs", "Poisson-scaling EM", "", "parity/wordfish_r_compare.py, tests/test_wordfish.py"),
    "Wordshoal": _i("src/wordshoal.rs", "src/python/wordshoal.rs", "per-domain Wordfish + cross-domain linear factor EM", "", "parity/wordshoal_r_compare.py, tests/test_wordshoal.py"),
    "IdealPointSentenceTM": _i("src/sentence_ideal.rs", "src/python/sentence_ideal.rs", "Gaussian-cluster EM over embeddings", "", "tests/test_sentence_ideal.py"),
    "TBIP": _i("src/tbip.rs", "src/python/tbip.rs", "Poisson-factorization mean-field SVI", "", "parity/tbip_parity.py, tests/test_tbip.py"),
    "PartyEmbeddings": _i("src/party_embeddings.rs", "src/python/party_embeddings.rs", "PV-DM paragraph vectors (negative sampling)", "", "parity/party_embeddings_compare.py, tests/test_party_embeddings.py"),
    "FASTopic": _i("src/fastopic.rs", "src/python/mod.rs", "reverse-mode Sinkhorn optimal transport", "", "parity/fastopic_gold.py, parity/fastopic_compare.py"),
    "EmbeddingLDA": _i("python/topica/embedding.py", "", "k-means embedding seeding over a SeededLDA Gibbs core (Python); planted-recovery gold only, no external reference", "", "parity/embeddinglda_gold.py, tests/test_embedding_lda.py"),
    "CombinedTM": _i("src/prodlda.rs", "src/python/neural.rs", "contextualized ProdLDA VAE (prodlda.rs)", "", "parity/combinedtm_gold.py, parity/combinedtm_compare.py"),
    "ZeroShotTM": _i("src/prodlda.rs", "src/python/neural.rs", "contextualized ProdLDA VAE (prodlda.rs)", "", "parity/zeroshot_gold.py, parity/zeroshot_compare.py"),
    "InfoCTM": _i("src/infoctm.rs", "src/python/neural.rs", "two ProdLDA VAEs + TAMI alignment (prodlda.rs)", "", "parity/infoctm_gold.py, parity/infoctm_compare.py"),
    "TopicGPT": _i("python/topica/llm.py", "", "LLM prompting pipeline (Python)", "", "tests/test_topicgpt.py"),
}


# The collapsed-Gibbs family (AD-LDA approximate parallel sampler): its
# seed-reproducibility is conditional on the thread count when num_threads>1. The
# cvb0 path within these models is exempt (deterministic sweeps, thread-independent).
_GIBBS_FAMILY = frozenset(
    name for name, info in REGISTRY.items() if info.inference == "gibbs"
)


def effective_determinism(model, *, fit_settings: dict | None = None) -> dict:
    """The determinism class this *instance* actually has, given its configuration
    (issue #401), refining the per-class :attr:`ModelInfo.determinism` tag.

    Determinism is often per-configuration, not per-class: the same model can be
    ``bit-exact`` under one setting and ``seed-reproducible`` under another. This
    reads the model's :attr:`~topica` ``settings`` (constructor config, issue #400)
    and the ``fit_settings`` you record, and returns::

        {"effective": "seed-reproducible",     # the config-aware class
         "registry_class": "bit-exact",        # the coarse per-class tag
         "replay_requires": {"seed": 13},      # machine-readable replay conditions
         "notes": ["..."]}                     # human-readable caveats

    ``replay_requires`` carries the conditions an exact replay needs — always the
    ``seed`` for a seed-reproducible fit, plus ``num_threads`` for the collapsed-Gibbs
    approximate parallel sampler. ``bit-exact`` fits carry no replay requirements.

    Scope (minimal, honest): claims the configuration determines are made exactly.
    Where the deciding factor is a *runtime* outcome the config cannot know — did a
    spectral initialization succeed or silently fall back to a seeded random one? did
    anchor selection hit its degenerate-basis fallback? — the report reads the route
    the fit actually took, recorded on the fitted model (``model.initialization`` for
    STM/CTM/STS/DTM, ``model.anchor_fallback_used`` for AnchorLDA, issue #410), and
    makes the exact call. For an *unfitted* model, or one saved before that route was
    recorded, it falls back to the common-case class plus a caveat in ``notes`` rather
    than over- or under-claiming.

    Parameters
    ----------
    model : a topica model (fitted or not; reads its construction config).
    fit_settings : the keyword arguments passed to ``fit`` (e.g. ``num_threads``,
        ``inference``), which can override the constructor. Optional.
    """
    cls = type(model).__name__
    info = REGISTRY.get(cls)
    base = info.determinism if info is not None else None
    settings = getattr(model, "settings", None) or {}
    fit_settings = fit_settings or {}
    notes: list[str] = []

    if base == "llm-bounded":
        return {
            "effective": "llm-bounded",
            "registry_class": base,
            "replay_requires": {},
            "notes": [
                "output depends on an external model; stable at temperature 0, "
                "not bit-reproducible"
            ],
        }

    effective = base
    sampler = settings.get("sampler")
    init = settings.get("init")
    inference = fit_settings.get("inference")
    is_cvb0 = sampler == "cvb0"
    # The initialization route the fit actually took, recorded on the fitted model
    # (issue #410). `None` before fit or for a model saved before it was recorded —
    # then we fall back to the config-only caveat below.
    route = getattr(model, "initialization", None)

    if is_cvb0:
        # cvb0 seeds only the initial responsibilities, then is deterministic;
        # it never uses the thread count. Downgrade from any Gibbs base.
        effective = "seed-reproducible"
        notes.append(
            "cvb0 seeds only the initial responsibilities, then runs a deterministic, "
            "thread-independent sweep"
        )
    elif cls == "NMF":
        if init == "random":
            effective = "seed-reproducible"
            notes.append(
                "init='random' draws both factors from the seeded RNG; bit-exact "
                "only with init='nndsvd'"
            )
    elif cls == "GuidedNMF":
        # Default init='random' is seed-reproducible; the deterministic inits
        # ('nndsvd', caller-supplied 'none') are bit-exact.
        if init in ("nndsvd", "none"):
            effective = "bit-exact"
            notes.append(
                f"init='{init}' is deterministic (no RNG draws), so the fit is "
                "bit-exact across runs"
            )
    elif cls in ("CTM", "STM", "STS"):
        if inference == "svi":
            effective = "seed-reproducible"
            notes.append(
                "inference='svi' shuffles documents with the seeded RNG each epoch"
            )
        elif route is not None:
            # The fit recorded which init actually ran (#410) — an exact claim.
            if route in ("random", "random-fallback"):
                effective = "seed-reproducible"
                notes.append(
                    "init ran as 'random-fallback' (spectral recovery returned None)"
                    if route == "random-fallback"
                    else "init='random' seeds the initialization"
                )
            # "spectral"/"provided" -> keep the bit-exact base, no caveat needed.
        elif init == "random":
            effective = "seed-reproducible"
            notes.append(
                "init='random' seeds the initialization; bit-exact only with "
                "init='spectral'"
            )
        else:  # init == "spectral", batch, unfitted or old save (route unknown)
            notes.append(
                "bit-exact assumes spectral recovery succeeded; fit the model (or "
                "re-save it) so the actual init route is recorded"
            )
    elif cls == "DTM":
        # DTM stays seed-reproducible either way (its post-init variational fit is
        # not established as bit-exact); the recorded route just sharpens the note.
        if route == "random-fallback":
            notes.append("init requested spectral but fell back to a seeded static-LDA init")
        elif route == "spectral":
            notes.append("spectral init succeeded (deterministic); the fit remains seed-reproducible")
        elif init == "spectral":
            notes.append(
                "init='spectral' is deterministic when spectral recovery succeeds; a "
                "degenerate corpus falls back to a seeded static-LDA init"
            )
    elif cls == "AnchorLDA":
        fallback = getattr(model, "anchor_fallback_used", None)
        if fallback:
            effective = "seed-reproducible"
            notes.append("anchor selection hit its seeded degenerate-basis fallback")
        else:
            notes.append(
                "not bit-identical across BLAS/LAPACK backends"
                + ("" if fallback is False else "; assumes anchor selection did not hit its seeded fallback")
            )

    replay: dict = {}
    if effective == "seed-reproducible":
        replay["seed"] = settings.get("seed")
        # Only the collapsed-Gibbs approximate parallel sampler is thread-conditional;
        # the cvb0 path and the variational E-steps preserve serial reduction order.
        if cls in _GIBBS_FAMILY and not is_cvb0:
            threads = fit_settings.get("num_threads", settings.get("num_threads", 1))
            threads = 1 if threads in (None, 0) else threads
            replay["num_threads"] = threads
            if threads > 1:
                notes.append(
                    f"num_threads={threads} uses the approximate parallel sampler; "
                    f"replay requires the same thread count"
                )

    return {
        "effective": effective,
        "registry_class": base,
        "replay_requires": replay,
        "notes": notes,
    }


def list_models(
    *,
    group: str | None = None,
    brings: str | None = None,
    inference: str | None = None,
    determinism: str | None = None,
    tag: str | None = None,
    experimental: bool | None = None,
    common_start: bool | None = None,
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
    - ``common_start`` — ``True`` for only the common starting points (the models
      most social scientists reach for first). This is an editorial suggestion,
      not a quality ranking: every validated model is equally reference-checked,
      and a specialized model is the right first choice for its own design.

    Examples
    --------
    >>> import topica
    >>> [m.name for m in topica.list_models(brings="embeddings")]
    ['BERTopic', 'Top2Vec', 'ETM', 'FASTopic', 'EmbeddingLDA']
    >>> [m.name for m in topica.list_models(group="short-text")]
    ['GSDMM', 'PT']
    >>> [m.name for m in topica.list_models(common_start=True)]
    ['LDA', 'NMF', 'STM', 'KeyATM', 'GSDMM', 'BERTopic']
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
        if common_start is not None and m.common_start != common_start:
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
        # Every validated model is checked before it ships, but the bar is tiered:
        # a maintained reference implementation where one exists, otherwise planted
        # recovery on a synthetic corpus with a known answer. State that honestly,
        # up front, so the grouping below cannot be read as a validity gradient and
        # the header does not overclaim reference parity for the planted-only
        # models. The "common starting points" band is an editorial convenience
        # (where newcomers most often begin), not a ranking.
        lines.append(
            "*Every model below is validated before it enters the roster: against a "
            "maintained reference implementation where one exists (MALLET, gensim, R "
            "`stm`, tomotopy, and the like), otherwise by planted recovery on a "
            "synthetic corpus with a known answer.* See "
            "[validation](https://nealcaren.github.io/topica/contributing/validation/) "
            "for where each model stands. "
            "The groupings are about **fit to your research design**, not quality: "
            "a specialized model is the right first choice when your data calls "
            "for it.\n"
        )
        # Common starting points: the handful of models most social scientists
        # reach for first, surfaced at the top so a newcomer is not met with 40-odd
        # equal-weight rows.
        common = [m for m in REGISTRY.values() if m.common_start and not m.experimental]
        if common:
            lines.append("### Common starting points\n")
            lines.append(
                "One per common goal, where most social scientists begin. "
                "`BERTopic` works differently from the others: it clusters "
                "document embeddings rather than fitting a posterior, so "
                "topic-proportion uncertainty and covariate-effect estimation do "
                "not carry over directly.\n"
            )
            lines += _rows(common)
            lines.append("")
        # Specialized approaches, grouped by purpose. Common-start models appear
        # only in the band above (not repeated here). Experimental models are held
        # out and listed in their own section below.
        lines.append("### Specialized approaches\n")
        lines.append(
            "The right first choice when your design calls for one: short text, "
            "change over time, document networks, multiple languages, ideological "
            "scaling, and more.\n"
        )
        for key, label in GROUPS.items():
            models = [m for m in REGISTRY.values()
                      if m.group == key and not m.experimental and not m.common_start]
            if not models:
                continue
            lines.append(f"#### {label}\n")
            lines += _rows(models)
            lines.append("")
        experimental = [m for m in REGISTRY.values() if m.experimental]
        if experimental:
            lines.append("### Experimental\n")
            lines.append(
                "Not (yet) on the validated roster, on one of two grounds: the model "
                "is unpublished (a topica original with no paper), or it is a "
                "published method whose benefit has not held up against a simpler "
                "baseline. A published method with faithful inference stays validated "
                "even when its basis is planted-recovery, so planted validation alone "
                "does not land a model here. Gated: call "
                "`topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) before "
                "use. These may change or be removed without a deprecation cycle. For "
                "the triple gate a model clears to graduate to the validated roster, "
                "and where every model stands on validation evidence, see "
                "[Validation & graduation]"
                "(https://nealcaren.github.io/topica/contributing/validation/).\n"
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


# The front-door chooser: research goal -> the model to reach for first. Two
# sections, so a newcomer sees both the common openings and an explicit route into
# each specialized family (rather than reading "specialized" as second-rate). Each
# row's ``primary`` is where to start for that goal; ``also`` names a close
# alternative. Model names are checked against REGISTRY at render time (see
# :func:`chooser_markdown_table`), so a renamed or removed model fails loudly in
# ``gen_model_tables.py`` rather than drifting silently in the docs.
@dataclass(frozen=True)
class ChooserRow:
    """One row of the front-door decision matrix."""

    goal: str  # the researcher's question, in their terms
    primary: str  # the model to start with (a REGISTRY key)
    also: str  # a close alternative (a REGISTRY key), or "" for none
    calls: str  # the first functions to call, as inline code
    note: str  # one why/watch-out line
    section: str = "common"  # "common" or "specialized"


CHOOSER: tuple[ChooserRow, ...] = (
    # --- Common openings -------------------------------------------------------
    ChooserRow(
        "Explore themes with no prior structure",
        "LDA", "NMF", "`search_k()`, `topic_table()`",
        "The default first pass. `NMF` is a fast, deterministic alternative.",
    ),
    ChooserRow(
        "Relate topics to metadata (author, date, party)",
        "STM", "DMR", "`estimate_effect()`, `one_hot()`, `spline()`",
        "`STM` gives covariate effects with uncertainty; `DMR` is a lighter Gibbs prior.",
    ),
    ChooserRow(
        "Measure concepts you can name in advance",
        "KeyATM", "SeededLDA", "`KeyATM(keywords=…)`, `.keyword_rate`",
        "Anchor named topics with a few seed words each.",
    ),
    ChooserRow(
        "Very short documents: tweets, headlines, survey answers",
        "GSDMM", "PT", "`fit()`",
        "One topic per document; standard LDA over-fragments short text.",
    ),
    ChooserRow(
        "Cluster by meaning using embeddings",
        "BERTopic", "ETM", "`fit(docs, doc_embeddings=…)`",
        "Clustering, not a posterior: topic-proportion uncertainty and effect "
        "estimation behave differently than the models above.",
    ),
    # --- Specialized: the right first choice when your design calls for it ------
    ChooserRow(
        "Topics shift over time slices",
        "DTM", "DETM", "`fit(docs, times=…)`",
        "Prevalence and content evolve across periods; `DETM` adds embeddings.",
        section="specialized",
    ),
    ChooserRow(
        "Documents linked in a network (citations, replies)",
        "RTM", "", "`fit(docs, links=…)`",
        "Models the text and the link graph jointly.",
        section="specialized",
    ),
    ChooserRow(
        "Documents in more than one language",
        "PolylingualLDA", "", "`fit(doc_tuples)`",
        "Aligned topics across languages from translation-linked tuples.",
        section="specialized",
    ),
    ChooserRow(
        "Place authors or actors on an ideological scale",
        "Wordfish", "TBIP", "`fit(docs)`",
        "Scaling from word usage; `TBIP` adds a text-based ideal-point prior.",
        section="specialized",
    ),
    ChooserRow(
        "How tone or sentiment varies with metadata",
        "STS", "", "`estimate_effect()`",
        "Sentiment-discourse decomposition; reach for it when tone is the question.",
        section="specialized",
    ),
)


def chooser_markdown_table() -> str:
    """Render the front-door decision matrix (:data:`CHOOSER`) as Markdown.

    Two sub-tables, ``common`` then ``specialized``, so each specialist family has
    an explicit "start here when…" route rather than being buried in the catalog.
    Raises ``KeyError`` if a row names a model absent from :data:`REGISTRY`, which
    is the drift guard: the table cannot ship a model that has been renamed or
    removed.
    """
    def _table(header: str, rows: tuple[ChooserRow, ...]) -> list[str]:
        out = [header, "|---|---|---|---|---|"]
        for r in rows:
            REGISTRY[r.primary]  # KeyError if the model no longer exists
            also = f"`{REGISTRY[r.also].name}`" if r.also else "—"
            out.append(
                f"| {r.goal} | `{REGISTRY[r.primary].name}` | {also} | {r.calls} | {r.note} |"
            )
        return out

    common = tuple(r for r in CHOOSER if r.section == "common")
    special = tuple(r for r in CHOOSER if r.section == "specialized")
    lines = ["**Common openings**", ""]
    lines += _table("| If your goal is… | Start with | Also consider | First calls | Note |", common)
    lines += ["", "**Specialized approaches.** Start here when your design calls for one.", ""]
    lines += _table("| If your data or goal is… | Start with | Also consider | First calls | Note |", special)
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
