"""Embedding-guided LDA: clustering the vocabulary embeddings into seed sets,
recovering the embedding structure, delegating the SeededLDA surface, and
(crucially) letting the data override a misleading embedding seed."""

import numpy as np
import pytest

import topica
from topica.embedding import embedding_seeds


def _blocks(n_blocks=3, per_block=8, e_dim=16, seed=0):
    """Vocabulary of disjoint blocks; each block's embeddings cluster around a
    distinct random center. Returns (vocab, embeddings, blocks)."""
    rng = np.random.default_rng(seed)
    blocks = [[f"{chr(97 + b)}{i}" for i in range(per_block)] for b in range(n_blocks)]
    vocab = [w for blk in blocks for w in blk]
    centers = rng.normal(size=(n_blocks, e_dim)) * 3
    emb = np.zeros((len(vocab), e_dim))
    for i in range(len(vocab)):
        emb[i] = centers[i // per_block] + rng.normal(size=e_dim) * 0.5
    return vocab, emb, blocks


def _corpus(blocks, n_docs=300, doc_len=6, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.choice(blocks[d % len(blocks)], doc_len)) for d in range(n_docs)]


def test_recovers_embedding_clusters():
    vocab, emb, blocks = _blocks()
    docs = _corpus(blocks)
    m = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, top_m=5, seed=1)
    m.fit(docs, iters=300)
    # Each recovered topic's top words come from a single embedding block.
    for t in range(3):
        prefixes = [w[0] for w, _ in m.top_words(6)[t]]
        dominant = max(set(prefixes), key=prefixes.count)
        assert prefixes.count(dominant) >= 5, f"topic {t} mixes blocks: {prefixes}"


def test_seeds_disjoint_and_sized():
    vocab, emb, _ = _blocks()
    seeds = embedding_seeds(emb, vocab, num_topics=3, top_m=4, seed=1)
    assert len(seeds) == 3
    all_seeds = [w for ws in seeds.values() for w in ws]
    assert len(all_seeds) == len(set(all_seeds))  # disjoint
    assert all(len(ws) <= 4 for ws in seeds.values())
    assert set(all_seeds) <= set(vocab)


def test_delegates_to_seeded_lda():
    vocab, emb, blocks = _blocks()
    m = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=1)
    m.fit(_corpus(blocks), iters=100)
    assert m.num_topics == 3
    assert m.topic_word.shape[0] == 3
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert len(m.top_words(5)) == 3  # delegated method


def test_deterministic():
    vocab, emb, blocks = _blocks()
    docs = _corpus(blocks)
    a = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=7)
    b = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=7)
    assert a.seeds == b.seeds
    a.fit(docs, iters=100)
    b.fit(docs, iters=100)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_data_overrides_misleading_seed():
    # A decoy word 'z' is embedded near block-a (so it seeds topic-a), but in the
    # data it only ever appears with block-b words. The fit should follow the
    # co-occurrence and place 'z' with block b, not its embedding seed.
    vocab, emb, blocks = _blocks(n_blocks=2, per_block=8, seed=3)
    rng = np.random.default_rng(3)
    a_center = emb[:8].mean(axis=0)
    vocab = vocab + ["z"]
    emb = np.vstack([emb, a_center + rng.normal(size=emb.shape[1]) * 0.3])  # near block a

    # Docs: block-a docs (pure a), and block-b docs that always include 'z'.
    docs = []
    for d in range(300):
        if d % 2 == 0:
            docs.append(list(rng.choice(blocks[0], 6)))
        else:
            docs.append(list(rng.choice(blocks[1], 5)) + ["z"])

    m = topica.EmbeddingLDA(num_topics=2, embeddings=emb, vocabulary=vocab, top_m=8, seed=1)
    # 'z' is seeded into whichever topic its embedding (block a) lands in.
    z_seed_topic = next(k for k, ws in m.seeds.items() if "z" in ws)
    m.fit(docs, iters=400)

    vocab_fitted = list(m.vocabulary)
    z = vocab_fitted.index("z")
    b0 = vocab_fitted.index("b0")
    # The topic that most claims 'z' should be the one that claims block b, not a.
    z_topic = int(np.argmax(m.topic_word[:, z]))
    b_topic = int(np.argmax(m.topic_word[:, b0]))
    assert z_topic == b_topic, "data co-occurrence should override the embedding seed"


def test_document_prior_separates_identical_bag_of_words():
    # Every document has the SAME mixed bag of words (block a + block b), so
    # co-occurrence alone cannot tell the two topics apart per document. Half the
    # documents carry an embedding near cluster a, half near cluster b. The
    # document-embedding prior should pull each group toward its anchored topic.
    rng = np.random.default_rng(5)
    a = [f"a{i}" for i in range(8)]
    b = [f"b{i}" for i in range(8)]
    vocab = a + b
    e_dim = 16
    centers = rng.normal(size=(2, e_dim)) * 4
    emb = np.array([centers[i // 8] + rng.normal(size=e_dim) * 0.3 for i in range(16)])

    # Identical-content docs: 3 a-words + 3 b-words each.
    docs, doc_emb = [], []
    for d in range(200):
        docs.append(list(rng.choice(a, 3)) + list(rng.choice(b, 3)))
        # alternate the document embedding between the two cluster centers.
        doc_emb.append(centers[d % 2] + rng.normal(size=e_dim) * 0.3)
    doc_emb = np.array(doc_emb)

    m = topica.EmbeddingLDA(
        num_topics=2, embeddings=emb, vocabulary=vocab, top_m=8, doc_anchor=5.0, seed=1
    )
    m.fit(docs, doc_embeddings=doc_emb, iters=400)

    # Identify which fitted topic is the "a" topic by its top words.
    a_topic = 0 if all(w[0] == "a" for w, _ in m.top_words(4)[0]) else 1
    theta = m.doc_topic
    a_docs = theta[0::2, a_topic]  # embedding near center 0
    b_docs = theta[1::2, a_topic]  # embedding near center 1
    # Documents anchored toward cluster 0 lean more on the a-topic than those
    # anchored toward cluster 1, despite identical word content.
    assert a_docs.mean() - b_docs.mean() > 0.2, (a_docs.mean(), b_docs.mean())


def test_document_prior_shape_and_inspection():
    vocab, emb, blocks = _blocks(n_blocks=3)
    m = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=1)
    doc_emb = np.random.default_rng(0).normal(size=(10, emb.shape[1]))
    prior = m.document_topic_prior(doc_emb)
    assert prior.shape == (10, 3)
    assert (prior >= m.alpha - 1e-9).all()  # base alpha plus a non-negative boost


def test_v1_mode_unchanged_without_doc_embeddings():
    vocab, emb, blocks = _blocks()
    a = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=2)
    a.fit(_corpus(blocks), iters=100)  # no doc_embeddings -> V1 path
    b = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=2)
    b.fit(_corpus(blocks), iters=100, doc_embeddings=None)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_convergence_signal_passthrough():
    # #661: EmbeddingLDA.fit forwards convergence_tol/check_every to SeededLDA, so
    # fit_history is populated and `converged` becomes a real early-stop verdict.
    vocab, emb, blocks = _blocks()
    docs = _corpus(blocks, n_docs=400)

    # Default: full run, no early stop, but the trace is still recorded.
    m = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=3)
    m.fit(docs, iters=200, check_every=25)
    assert m.converged is False
    assert m.fit_history and m.fit_history[-1][0] == 200

    # A tolerance turns on early stopping: it must converge before iters here.
    m2 = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=3)
    m2.fit(docs, iters=2000, convergence_tol=1e-4, check_every=25)
    assert m2.converged is True
    assert m2.fit_history[-1][0] < 2000


def test_doc_embeddings_dim_validation():
    vocab, emb, blocks = _blocks()
    m = topica.EmbeddingLDA(num_topics=3, embeddings=emb, vocabulary=vocab, seed=1)
    with pytest.raises(ValueError):
        m.document_topic_prior(np.zeros((5, emb.shape[1] + 1)))  # wrong E


def test_validation_errors():
    vocab, emb, _ = _blocks()
    with pytest.raises(ValueError):
        embedding_seeds(emb, vocab[:-1], 3)  # mismatched vocab length
    with pytest.raises(ValueError):
        embedding_seeds(emb, vocab, 1)  # num_topics < 2
    with pytest.raises(ValueError):
        embedding_seeds(emb, vocab, len(vocab) + 1)  # more topics than words
    with pytest.raises(ValueError):
        embedding_seeds(emb, vocab, 3, top_m=0)
    with pytest.raises(ValueError):
        embedding_seeds(np.zeros((len(vocab),)), vocab, 3)  # not 2-D
