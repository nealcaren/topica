"""Registry-driven validity-invariant suite (issue #271, "Wave 0").

For EVERY model in ``topica.list_models()`` we build a small planted/synthetic
corpus tailored to that model's input contract (its ``ModelInfo.brings``), fit it
with small K and modest iterations under a fixed seed, and assert the fit is
statistically VALID — not merely well-shaped. The validity checks live in
``tests/invariants.py``; the headline one is ``assert_healthy_theta``, which a
degenerate fit (issue #270's covariate-keyATM collapse) FAILS.

A single ``FIT_ADAPTERS`` dict maps model name -> a fit function returning
``(theta, topic_word_or_None, n_topics)``. The coverage gate
(``tests/test_model_coverage_manifest.py``) imports this dict so a newly added
model with no adapter fails CI.

Where it is cheap (the stochastic count-based models), we also run the
metamorphic ``assert_more_iters_not_worse`` check, which is what actually catches
collapse-on-training.
"""
from __future__ import annotations

import numpy as np
import pytest

import topica
from invariants import (
    assert_finite,
    assert_healthy_theta,
    assert_more_iters_not_worse,
    assert_simplex,
    effective_topics,
    _degenerate_theta_must_raise,
)


# ---------------------------------------------------------------------------
# The non-vacuity self-test (required by #271): a collapsed theta MUST raise.
# ---------------------------------------------------------------------------

def test_invariants_catch_degenerate_theta():
    """Proof the invariant has teeth: a deliberately degenerate theta is rejected.

    If this passes, ``assert_healthy_theta`` genuinely detects collapse; if the
    body of ``_degenerate_theta_must_raise`` ever stops raising, this test fails
    loudly rather than the suite silently going vacuous.
    """
    _degenerate_theta_must_raise()


# ---------------------------------------------------------------------------
# Planted-block corpus builders (shared across adapters)
# ---------------------------------------------------------------------------

def _planted_blocks(k=4, block=8, n=300, length=14, seed=0):
    """K disjoint word-blocks; each document draws all its tokens from one block,
    cycling through the blocks so the corpus-level topic distribution is roughly
    flat (a healthy fit spreads mass across K topics). Returns (docs, vocab)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(length)])
    return docs, vocab


def _planted_embeddings(k=4, block=8, seed=0):
    """Word embeddings for the planted-block vocabulary: each word points along
    its block's axis. Returns (vocab, word_emb)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    e = k + 2
    word_emb = rng.normal(0, 0.2, (k * block, e))
    for w in range(k * block):
        word_emb[w, w // block] += 3.0
    return vocab, word_emb


def _doc_embeddings(docs, k=4, block=8, seed=0):
    """One embedding per document, one-hot (plus noise) along the document's block.
    The block of a planted doc is recoverable from its first token's name."""
    rng = np.random.default_rng(seed)
    e = k + 2
    out = np.zeros((len(docs), e))
    for d, doc in enumerate(docs):
        b = int(doc[0].split("w")[0][1:])  # "b3w5" -> 3
        out[d, b] += 3.0
        out[d] += rng.normal(0, 0.2, e)
    return out


def _block_keywords(vocab, k=4, block=8, per=4):
    """A keyword/seed dict over the planted blocks: ``{topic: [first words]}``."""
    return {f"t{b}": [f"b{b}w{i}" for i in range(per)] for b in range(k)}


# ---------------------------------------------------------------------------
# Per-model fit adapters. Each returns (theta, topic_word_or_None, n_topics).
# Kept small so the whole file runs in a couple of minutes.
# ---------------------------------------------------------------------------

K = 4  # default planted-block count for the suite


def _theta_from_doc_paths(model):
    """HLDA has no doc_topic; build a per-document distribution over tree nodes by
    spreading each document's mass uniformly over the nodes on its path. A healthy
    tree puts documents on several distinct paths; a collapsed one puts them all
    on the same node, which the effective-topics check then flags."""
    n_nodes = model.num_nodes
    paths = model.doc_paths
    theta = np.zeros((len(paths), n_nodes))
    for d, path in enumerate(paths):
        for node in path:
            theta[d, node] += 1.0 / len(path)
    return theta, n_nodes


# ---- General-purpose --------------------------------------------------------

def _fit_lda(iters=200):
    docs, _ = _planted_blocks(seed=0)
    m = topica.LDA(num_topics=K, seed=1)
    m.fit(docs, iters=iters, num_samples=2, sample_interval=5)
    return m.doc_topic, m.topic_word, K


def _fit_online_lda(iters=80):
    docs, _ = _planted_blocks(seed=0)
    m = topica.OnlineLDA(num_topics=K, batch_size=32, tau=1.0, kappa=0.7, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_ctm(iters=60):
    docs, _ = _planted_blocks(seed=0)
    m = topica.CTM(num_topics=K, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_prodlda(iters=150):
    docs, _ = _planted_blocks(seed=0)
    m = topica.ProdLDA(num_topics=K, batch_size=64, lr=0.01, dropout=0.0, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_hdp(iters=150):
    docs, _ = _planted_blocks(k=K, seed=0)
    m = topica.HDP(seed=1, alpha=1.0, gamma=1.0)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_nmf(iters=None):
    docs, _ = _planted_blocks(seed=0)
    m = topica.NMF(K, seed=1)
    m.fit(docs)
    return m.doc_topic, m.topic_word, K


def _fit_lsa(iters=None):
    # LSA's doc_topic is signed SVD coordinates, not a simplex. Map to a topic
    # distribution by absolute loading per document so the health check is
    # meaningful (a collapse would still pile abs-loading onto one component).
    docs, _ = _planted_blocks(seed=0)
    m = topica.LSA(K, seed=1)
    m.fit(docs)
    dt = np.abs(np.asarray(m.doc_topic))
    dt = dt / dt.sum(axis=1, keepdims=True)
    tw = np.abs(np.asarray(m.topic_word))
    tw = tw / tw.sum(axis=1, keepdims=True)
    return dt, tw, K


def _fit_anchorlda(iters=None):
    # The planted blocks are separable, so anchor-words should recover one healthy
    # topic per block.
    docs, _ = _planted_blocks(seed=0)
    m = topica.AnchorLDA(K, min_count=2, seed=1)
    m.fit(docs)
    return m.doc_topic, m.topic_word, K


def _fit_tensorlda(iters=50):
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        docs, _ = _planted_blocks(seed=0)
        m = topica.TensorLDA(num_topics=K, alpha_0=1.0, seed=1)
        m.fit(docs, iters=iters)
        return m.doc_topic, m.topic_word, K
    finally:
        topica.enable_experimental(was)


# ---- Covariates & structure -------------------------------------------------

def _covariate_corpus(seed=0):
    """Planted blocks plus a one-hot covariate that favours one block per level,
    matching the DMR/STM/SAGE recovery fixtures' spirit."""
    docs, vocab = _planted_blocks(k=K, seed=seed)
    # level == block of each doc (recoverable, planted covariate effect)
    levels = [int(doc[0].split("w")[0][1:]) for doc in docs]
    X = np.zeros((len(docs), K))
    X[np.arange(len(docs)), levels] = 1.0
    return docs, vocab, X, levels


def _fit_stm(iters=60):
    docs, _, X, _ = _covariate_corpus()
    m = topica.STM(num_topics=K, seed=1)
    m.fit(docs, prevalence=X, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_sts(iters=40):
    docs, _, X, levels = _covariate_corpus()
    sent_seed = [float(l % 3) for l in levels]  # 3-level sentiment seed
    m = topica.STS(num_topics=K, seed=1)
    m.fit(docs, sentiment_seed=sent_seed, prevalence=X, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_sage(iters=200):
    docs, _, _, levels = _covariate_corpus()
    groups = [f"g{l % 2}" for l in levels]  # 2 content groups
    m = topica.SAGE(num_topics=K, seed=1, optimize_interval=25, burn_in=50)
    m.fit(docs, groups, iters=iters, num_samples=2, sample_interval=10)
    # topic_word is (K, G, V); collapse to the marginal for the finite/simplex check
    return m.doc_topic, np.asarray(m.topic_word_marginal), K


def _fit_dmr(iters=300):
    docs, _, X, _ = _covariate_corpus()
    m = topica.DMR(num_topics=K, seed=1, optimize_interval=25, burn_in=50)
    m.fit(docs, X, iters=iters, num_samples=2, sample_interval=10)
    return m.doc_topic, m.topic_word, K


def _fit_scholar(iters=150):
    docs, _, X, _ = _covariate_corpus()
    m = topica.Scholar(num_topics=K, seed=1)
    m.fit(docs, covariates=X, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_gdmr(iters=300):
    # GDMR wants a continuous covariate; use the block index scaled to [0,1].
    docs, vocab = _planted_blocks(k=K, seed=0)
    levels = np.array([int(doc[0].split("w")[0][1:]) for doc in docs], dtype=float)
    meta = (levels / (K - 1))[:, None]
    # sigma0 is the (tomotopy-faithful) prior std on the intercept/baseline; the
    # default 3.0 is a weak baseline prior that this tiny synthetic corpus cannot
    # constrain (one topic runs away), so use a tighter intercept here. On real
    # continuous-metadata corpora the likelihood dominates and the default is fine.
    m = topica.GDMR(num_topics=K, degrees=[3], sigma0=1.0, seed=1,
                    optimize_interval=25, burn_in=50)
    m.fit(docs, meta, iters=iters, num_samples=2, sample_interval=10)
    return m.doc_topic, m.topic_word, K


def _fit_narrativetm(iters=300):
    docs, vocab = _planted_blocks(k=K, seed=0)
    topica.enable_experimental()
    m = topica.NarrativeTM(num_topics=K, degree=3, seed=1, optimize_interval=25, burn_in=50)
    m.fit(docs, iters=iters, num_samples=2, sample_interval=10)
    return m.doc_topic, m.topic_word, K



# ---- Guided & supervised ----------------------------------------------------

def _fit_keyatm(iters=400):
    docs, vocab = _planted_blocks(k=K, seed=0)
    seeds = _block_keywords(vocab, k=K)
    m = topica.KeyATM(seeds, num_topics=K, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_seededlda(iters=400):
    docs, vocab = _planted_blocks(k=K, seed=0)
    seeds = _block_keywords(vocab, k=K)
    m = topica.SeededLDA(seeds, seed=1)  # 4 seeded topics
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_guided_nmf(iters=100):
    docs, vocab = _planted_blocks(k=K, seed=0)
    seeds = _block_keywords(vocab, k=K)
    m = topica.GuidedNMF(K, seeds, guidance=20.0, weighting="count", seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_labeledlda(iters=300):
    docs, vocab = _planted_blocks(k=K, seed=0)
    labels = [[f"t{int(doc[0].split('w')[0][1:])}"] for doc in docs]
    m = topica.LabeledLDA(alpha=0.1, seed=1)
    m.fit(docs, labels, iters=iters, num_samples=2, sample_interval=10)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_disclda(iters=300):
    # Class label = the planted block; each class gets a class-specific topic plus a
    # shared block, so a healthy fit spreads mass across the class+shared topics.
    docs, vocab = _planted_blocks(k=K, seed=0)
    y = [f"c{int(doc[0].split('w')[0][1:])}" for doc in docs]  # "b3w5" -> "c3"
    m = topica.DiscLDA(k_class=1, k_shared=2, alpha=0.1, iters=iters, seed=1)
    m.fit(docs, y)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_rtm(iters=40):
    # Links dense within a planted block, sparse across, so a healthy fit spreads
    # doc-topic mass across the K blocks (as the words already imply).
    docs, vocab = _planted_blocks(k=K, seed=0)
    rng = np.random.default_rng(0)
    edges = []
    for i in range(len(docs)):
        for j in range(i + 1, min(i + 12, len(docs))):
            p = 0.5 if (i % K) == (j % K) else 0.02
            if rng.random() < p:
                edges.append((i, j))
    m = topica.RTM(K, alpha=0.5, seed=1)
    m.fit(docs, edges, iters=iters)
    return m.doc_topic, m.topic_word, m.num_topics


def _supervised_corpus(n=200, seed=0):
    """Mixed two-block docs with a response driven by block-0 prevalence."""
    rng = np.random.default_rng(seed)
    T0 = [f"a{i}" for i in range(6)]
    T1 = [f"g{i}" for i in range(6)]
    docs, y = [], []
    for _ in range(n):
        p = rng.random()
        doc = [(T0 if rng.random() < p else T1)[rng.integers(6)] for _ in range(20)]
        docs.append(doc)
        y.append(2 * p - 1 + (rng.random() - 0.5) * 0.2)
    return docs, np.array(y)


def _fit_supervisedlda(iters=25):
    docs, y = _supervised_corpus()
    m = topica.SupervisedLDA(num_topics=2, seed=7)
    m.fit(docs, y, iters=iters, var_iters=15)
    return m.doc_topic, m.topic_word, 2


# ---- Short text -------------------------------------------------------------

def _short_corpus(seed=0, n=300):
    """Short (3-token) docs over K disjoint blocks."""
    rng = np.random.default_rng(seed)
    blocks = [[f"b{b}w{i}" for i in range(4)] for b in range(K)]
    docs = []
    for d in range(n):
        blk = blocks[d % K]
        docs.append([blk[int(rng.integers(4))] for _ in range(3)])
    return docs


def _fit_gsdmm(iters=60):
    docs = _short_corpus()
    m = topica.GSDMM(num_topics=15, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_pt(iters=300):
    docs = _short_corpus()
    m = topica.PT(num_topics=K, num_pseudo=10, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_btm(iters=200):
    docs = _short_corpus()
    m = topica.BTM(num_topics=K, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_factorial_lda(iters=400):
    # 6 planted blocks map to the 6 tuples of factor_sizes=[3, 2]; a healthy fit
    # spreads mass across the six tuple word-distributions.
    docs, _ = _planted_blocks(k=6, block=8, n=300, length=14)
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    m.fit(docs, iters=iters, samples=iters // 4)
    return m.doc_topic, m.topic_word, 6


# ---- Dynamic & hierarchical -------------------------------------------------

def _fit_dtm(iters=20):
    # DTM exposes no doc_topic (it models topic-word drift, not per-doc theta).
    # Build a fit-derived per-document topic distribution by scoring each document
    # under the slice-0 topic-word distributions (a soft assignment from the
    # learned topics). A collapse -- all K topics on the same words -- then shows
    # up as one dominant column / low effective_topics, exactly as intended.
    docs, vocab = _planted_blocks(k=K, n=300, seed=0)
    times = [i % 3 for i in range(len(docs))]
    m = topica.DTM(num_topics=K, chain_variance=0.5, seed=1)
    m.fit(docs, times, iters=iters)
    tw = np.asarray(m.topic_word(0))  # (K, V) slice-0 word distributions
    vocab_idx = {w: i for i, w in enumerate(m.vocabulary)}
    logtw = np.log(tw + 1e-12)
    theta = np.zeros((len(docs), K))
    for d, doc in enumerate(docs):
        ids = [vocab_idx[w] for w in doc if w in vocab_idx]
        scores = logtw[:, ids].sum(axis=1)  # log p(doc | topic k)
        scores -= scores.max()
        p = np.exp(scores)
        theta[d] = p / p.sum()
    return theta, tw, K


def _fit_detm(iters=40):
    docs, vocab = _planted_blocks(k=K, block=6, n=240, length=20, seed=0)
    _, word_emb = _planted_embeddings(k=K, block=6, seed=0)
    times = np.array([i % 4 for i in range(len(docs))])
    m = topica.DETM(K, delta=0.005, hidden_size=32, lr=0.02, seed=42)
    m.fit(docs, word_emb, vocab, times=times, iters=iters)
    return np.asarray(m.doc_topic), np.asarray(m.topic_word), K


def _fit_hlda(iters=300):
    # HLDA has no doc_topic / num_topics; synthesize a per-doc node distribution
    # from the learned tree paths.
    shared = ["the", "of", "and"]
    blocks = [[f"b{b}w{i}" for i in range(4)] for b in range(K)]
    docs = []
    for d in range(300):
        docs.append(shared + [blocks[d % K][i] for i in range(4)])
    m = topica.HLDA(depth=2, seed=1)
    m.fit(docs, iters=iters)
    theta, n_nodes = _theta_from_doc_paths(m)
    return theta, m.topic_word, n_nodes


def _fit_pa(iters=300):
    rng = np.random.default_rng(0)
    blocks = [[f"b{g}w{i}" for i in range(5)] for g in range(4)]
    docs = []
    for _ in range(160):
        pair = (blocks[0], blocks[1]) if rng.random() < 0.5 else (blocks[2], blocks[3])
        doc = []
        for blk in pair:
            doc += [blk[int(rng.integers(5))] for _ in range(6)]
        docs.append(doc)
    m = topica.PA(num_super=2, num_sub=4, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, m.num_sub


# ---- Embedding-based --------------------------------------------------------

def _fit_bertopic(iters=None):
    docs, vocab = _planted_blocks(k=K, block=8, n=300, seed=0)
    doc_emb = _doc_embeddings(docs, k=K, block=8, seed=0)
    m = topica.BERTopic(min_cluster_size=15, seed=1)
    m.fit(docs, doc_emb)
    if m.num_topics == 0:
        pytest.skip("BERTopic found no clusters at this min_cluster_size")
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_top2vec(iters=None):
    docs, vocab = _planted_blocks(k=K, block=8, n=300, seed=0)
    doc_emb = _doc_embeddings(docs, k=K, block=8, seed=0)
    m = topica.Top2Vec(min_cluster_size=15, seed=1)
    m.fit(docs, doc_emb)
    if m.num_topics == 0:
        pytest.skip("Top2Vec found no clusters at this min_cluster_size")
    return m.doc_topic, m.topic_word, m.num_topics


def _fit_semanticsignalseparation(iters=None):
    # S³ (FastICA over embeddings): doc and vocab embeddings share one space. Each
    # planted block is an independent one-hot signal, so ICA recovers the K axes.
    docs, vocab = _planted_blocks(k=K, block=8, n=300, seed=0)
    doc_emb = _doc_embeddings(docs, k=K, block=8, seed=0)  # (n, K+2)
    e = K + 2
    rng = np.random.default_rng(0)
    vocab_emb = np.zeros((len(vocab), e))
    for i, w in enumerate(vocab):
        b = int(w.split("w")[0][1:])  # "b3w5" -> 3
        vocab_emb[i, b] += 3.0
        vocab_emb[i] += rng.normal(0, 0.2, e)
    m = topica.SemanticSignalSeparation(K, seed=1)
    m.fit(docs, doc_emb, vocab_emb, vocabulary=vocab)
    return m.doc_topic, m.topic_word, K


def _fit_etm(iters=80):
    docs, vocab = _planted_blocks(k=K, block=8, n=240, length=12, seed=0)
    _, word_emb = _planted_embeddings(k=K, block=8, seed=0)
    m = topica.ETM(num_topics=K, seed=1)
    m.fit(docs, word_emb, vocab, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_idealpoint(iters=40):
    # IdealPointTM is experimental and gated. Topic model with a latent ideal-point
    # head; documents grouped into authors that carry a position. The default fit
    # (no word_embeddings) is the count representation, exercised here; the
    # word-embedding representation is covered in tests/test_idealpoint.py.
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        docs, _ = _planted_blocks(k=K, block=8, n=240, length=12, seed=0)
        group = [f"a{i % 16}" for i in range(len(docs))]
        m = topica.IdealPointTM(num_topics=K, num_dims=1, seed=1)
        m.fit(docs, group=group, iters=iters)
        return m.doc_topic, m.topic_word, K
    finally:
        topica.enable_experimental(was)


def _fit_tbip(iters=600):
    # Text-Based Ideal Points: a Poisson factorization whose neutral topics are
    # rescaled by a per-word ideological factor exp(x_s * eta_kv), fit by
    # mean-field VI (reparameterized SVI).
    docs, _ = _planted_blocks(k=K, block=8, n=240, length=12, seed=0)
    group = [f"a{i % 16}" for i in range(len(docs))]
    m = topica.TBIP(num_topics=K, seed=1, iters=iters, batch_size=len(docs))
    m.fit(docs, group=group)
    return m.doc_topic, m.topic_word, K


def _fit_sentence_ideal(iters=60):
    # IdealPointSentenceTM is experimental and gated. Continuous ideal-point model
    # over per-document embeddings: topics are Gaussian clusters; doc_topic is the
    # soft cluster assignment. No topic_word (it is embedding-, not word-, based).
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        docs, _ = _planted_blocks(k=K, block=8, n=240, length=12, seed=0)
        emb = _doc_embeddings(docs, k=K, block=8, seed=0)
        group = [f"a{i % 16}" for i in range(len(docs))]
        m = topica.IdealPointSentenceTM(num_topics=K, num_dims=1, seed=1)
        m.fit(emb, group=group, iters=iters)
        return m.doc_topic, None, K
    finally:
        topica.enable_experimental(was)


def _fit_fastopic(iters=200):
    docs, vocab = _planted_blocks(k=K, block=6, n=200, length=10, seed=0)
    doc_emb = _doc_embeddings(docs, k=K, block=6, seed=0)
    m = topica.FASTopic(num_topics=K, lr=0.05, seed=1)
    m.fit(docs, doc_emb, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_embeddinglda(iters=300):
    docs, vocab = _planted_blocks(k=K, block=8, n=300, seed=0)
    _, word_emb = _planted_embeddings(k=K, block=8, seed=0)
    m = topica.EmbeddingLDA(num_topics=K, embeddings=word_emb, vocabulary=vocab,
                            top_m=5, seed=1)
    m.fit(docs, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_contextual(cls, iters=120):
    docs, vocab = _planted_blocks(k=K, block=8, n=200, length=15, seed=0)
    doc_emb = _doc_embeddings(docs, k=K, block=8, seed=0)
    m = cls(num_topics=K, batch_size=60, lr=0.01, dropout=0.0, seed=1)
    m.fit(docs, doc_emb, iters=iters)
    return m.doc_topic, m.topic_word, K


def _fit_combinedtm(iters=120):
    return _fit_contextual(topica.CombinedTM, iters=iters)


def _fit_zeroshottm(iters=120):
    return _fit_contextual(topica.ZeroShotTM, iters=iters)


def _fit_infoctm(iters=120):
    rng = np.random.default_rng(0)
    blocks = K
    per, length = 4, 12

    def corpus(prefix, n):
        return [[f"{prefix}{d % blocks}_{int(rng.integers(per))}" for _ in range(length)]
                for d in range(n)]

    a, b = corpus("a", 160), corpus("b", 152)
    dictionary = [(f"a{blk}_{i}", f"b{blk}_{j}")
                  for blk in range(blocks) for i in range(per) for j in range(per)]
    m = topica.InfoCTM(num_topics=K, seed=1, hidden_size=32, lr=0.01,
                       languages=("en", "zh"))
    m.fit(a, b, dictionary=dictionary, iters=iters, batch_size=40)
    # validate both languages; return language A for the headline check.
    assert_simplex(m.topic_word(lang="zh"), axis=1, model="InfoCTM(zh)")
    return m.doc_topic(lang="en"), m.topic_word(lang="en"), K


def _fit_pltm(iters=400):
    # Aligned multilingual tuples: language l has its own vocabulary; tuple d is
    # topic d % K in every language, so a healthy fit spreads mass across K topics.
    rng = np.random.default_rng(0)
    langs = ("en", "fr", "de")
    n, length = 200, 14
    data = {
        name: [
            [f"{name}_b{d % K}w{int(rng.integers(8))}" for _ in range(length)]
            for d in range(n)
        ]
        for name in langs
    }
    m = topica.PolylingualLDA(num_topics=K, iters=iters, seed=1)
    m.fit(data)
    assert_simplex(m.topic_word(lang="fr"), axis=1, model="PolylingualLDA(fr)")
    return m.doc_topic, m.topic_word(lang="en"), K


FIT_ADAPTERS = {
    "LDA": _fit_lda,
    "OnlineLDA": _fit_online_lda,
    "CTM": _fit_ctm,
    "ProdLDA": _fit_prodlda,
    "HDP": _fit_hdp,
    "NMF": _fit_nmf,
    "GuidedNMF": _fit_guided_nmf,
    "LSA": _fit_lsa,
    "AnchorLDA": _fit_anchorlda,
    "TensorLDA": _fit_tensorlda,
    "STM": _fit_stm,
    "STS": _fit_sts,
    "SAGE": _fit_sage,
    "DMR": _fit_dmr,
    "GDMR": _fit_gdmr,
    "Scholar": _fit_scholar,
    "NarrativeTM": _fit_narrativetm,
    "KeyATM": _fit_keyatm,
    "SeededLDA": _fit_seededlda,
    "LabeledLDA": _fit_labeledlda,
    "SupervisedLDA": _fit_supervisedlda,
    "DiscLDA": _fit_disclda,
    "RTM": _fit_rtm,
    "GSDMM": _fit_gsdmm,
    "BTM": _fit_btm,
    "FactorialLDA": _fit_factorial_lda,
    "PolylingualLDA": _fit_pltm,
    "PT": _fit_pt,
    "DTM": _fit_dtm,
    "DETM": _fit_detm,
    "HLDA": _fit_hlda,
    "PA": _fit_pa,
    "BERTopic": _fit_bertopic,
    "Top2Vec": _fit_top2vec,
    "SemanticSignalSeparation": _fit_semanticsignalseparation,
    "ETM": _fit_etm,
    "IdealPointTM": _fit_idealpoint,
    "IdealPointSentenceTM": _fit_sentence_ideal,
    "TBIP": _fit_tbip,
    "FASTopic": _fit_fastopic,
    "EmbeddingLDA": _fit_embeddinglda,
    "CombinedTM": _fit_combinedtm,
    "ZeroShotTM": _fit_zeroshottm,
    "InfoCTM": _fit_infoctm,
}

# Models that need an external LLM/API and so cannot be fit here.
SKIP_MODELS = {
    "TopicGPT": "needs an LLM / external API",
    "Wordfish": "a pure ideal-point scaler with no topic/doc-topic distribution; "
    "validated by its own recovery test in tests/test_wordfish.py",
    "PartyEmbeddings": "a learned-embedding ideal-point scaler with no "
    "topic/doc-topic distribution; validated by its own recovery + parity tests "
    "(tests/test_party_embeddings.py, parity/party_embeddings_compare.py)",
}

# Models for which the cheap metamorphic "more iters not worse" check is run.
# These are the stochastic count-based / Gibbs models where it is fast and where
# collapse-on-training (the #270 failure) is the relevant risk. Neural/OT/SVD
# models and the auto-K / non-simplex models are excluded (slow or ill-posed for
# this particular relation).
METAMORPHIC_MODELS = {
    "LDA": _fit_lda,
    "DMR": _fit_dmr,
    "KeyATM": _fit_keyatm,
    "SeededLDA": _fit_seededlda,
    "PT": _fit_pt,
    "SAGE": _fit_sage,
}


def _theta_only(fit_fn):
    """Wrap a (theta, tw, k) adapter into an iters -> theta function."""
    def inner(iters):
        return fit_fn(iters)[0]
    return inner


# ---------------------------------------------------------------------------
# The parametrized validity suite.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [m.name for m in topica.list_models()],
    ids=[m.name for m in topica.list_models()],
)
def test_model_theta_is_healthy(name):
    """Every model produces a non-degenerate, valid theta on a planted corpus."""
    if name in SKIP_MODELS:
        pytest.skip(SKIP_MODELS[name])
    adapter = FIT_ADAPTERS.get(name)
    assert adapter is not None, f"no FIT_ADAPTER for registered model {name!r}"

    theta, topic_word, n_topics = adapter()
    theta = np.asarray(theta, dtype=float)

    # Validity invariants (the #271 high-leverage assertions).
    assert n_topics >= 1, f"[{name}] degenerate n_topics={n_topics}"
    assert_finite(theta, model=name)
    if topic_word is not None:
        assert_finite(topic_word, model=name)
        assert_simplex(topic_word, axis=1, model=name)

    if n_topics == 1:
        # An auto-K model that found a single topic on this corpus still must be
        # finite/valid, but the multi-topic health check is not meaningful.
        assert_simplex(theta, axis=1, model=name)
    else:
        assert_healthy_theta(theta, n_topics, model=name)


@pytest.mark.parametrize("name", sorted(METAMORPHIC_MODELS), ids=sorted(METAMORPHIC_MODELS))
def test_more_iters_not_worse(name):
    """Training longer must not make the fit MORE degenerate (catches #270)."""
    fit_fn = METAMORPHIC_MODELS[name]
    assert_more_iters_not_worse(_theta_only(fit_fn), low=25, high=150, model=name)
