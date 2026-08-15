"""FASTopic: the optimal-transport embedding topic model."""

import numpy as np
import pytest

import topica


def _planted(k=3, block=6, h=8, n=150, seed=0):
    """K word-blocks; each document draws from one block and embeds on that
    block's axis. Returns (token_docs, doc_emb, vocab, truth)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs, doc_emb, truth = [], [], []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(10)])
        e = np.zeros(h)
        e[b] = 3.0
        e += rng.normal(0, 0.3, h)
        doc_emb.append(e)
        truth.append(b)
    return docs, np.array(doc_emb), vocab, np.array(truth)


def test_fit_basic_shapes():
    docs, doc_emb, vocab, _ = _planted()
    m = topica.FASTopic(num_topics=3, lr=0.05, seed=1)
    m.fit(docs, doc_emb, iters=120)
    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.topic_embeddings.shape[0] == 3
    assert m.word_embeddings.shape == (len(vocab), m.topic_embeddings.shape[1])
    # theta and beta rows are distributions.
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)


def test_loss_decreases():
    docs, doc_emb, _, _ = _planted()
    m = topica.FASTopic(num_topics=3, lr=0.05, seed=1)
    m.fit(docs, doc_emb, iters=150)
    assert len(m.loss_history) >= 2
    assert m.loss_history[-1] < m.loss_history[0]


def test_recovers_planted_blocks():
    docs, doc_emb, vocab, _ = _planted()
    m = topica.FASTopic(num_topics=3, lr=0.05, seed=2)
    m.fit(docs, doc_emb, iters=250)
    covered = set()
    for t in range(3):
        words = [w for w, _ in m.top_words(3, topic=t, weights=True)]
        blocks = {w.split("w")[0] for w in words}  # "b0w3" -> "b0"
        assert len(blocks) == 1, f"topic {t} mixes blocks: {words}"
        covered |= blocks
    assert len(covered) == 3


def test_transform_held_out():
    docs, doc_emb, vocab, _ = _planted(h=8)
    m = topica.FASTopic(num_topics=3, lr=0.05, seed=3)
    m.fit(docs, doc_emb, iters=200)
    # One clean embedding per block; each should land on a distinct topic.
    new = np.array([np.eye(8)[b] * 3.0 for b in range(3)])
    theta = m.transform(doc_embeddings=new)
    assert theta.shape == (3, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert len(set(theta.argmax(axis=1))) == 3


def test_fit_transform_matches_doc_topic():
    docs, doc_emb, _, _ = _planted()
    m = topica.FASTopic(num_topics=3, lr=0.05, seed=4)
    theta = m.fit_transform(docs, doc_emb)
    assert np.allclose(theta, m.doc_topic)


def test_determinism():
    docs, doc_emb, _, _ = _planted()
    a = topica.FASTopic(num_topics=3, lr=0.05, seed=7)
    a.fit(docs, doc_emb, iters=80)
    b = topica.FASTopic(num_topics=3, lr=0.05, seed=7)
    b.fit(docs, doc_emb, iters=80)
    assert np.allclose(a.topic_word, b.topic_word)
    assert np.allclose(a.doc_topic, b.doc_topic)


def test_errors():
    docs, doc_emb, _, _ = _planted()
    with pytest.raises(ValueError):
        topica.FASTopic(num_topics=1)
    with pytest.raises(ValueError):
        topica.FASTopic(num_topics=3, theta_temp=0.0)
    # #517 guard-parity: the *_tol params must be finite and >= 0.
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            topica.FASTopic(num_topics=3, convergence_tol=bad)
        with pytest.raises(ValueError):
            topica.FASTopic(num_topics=3, sinkhorn_tol=bad)
    m = topica.FASTopic(num_topics=3, seed=1)
    with pytest.raises(Exception):
        _ = m.topic_word  # not fitted
    with pytest.raises(ValueError):
        m.fit(docs, doc_emb[:10])  # row/doc mismatch
    # #517 guard-parity: the fit-time convergence_tol override is guarded too.
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            m.fit(docs, doc_emb, convergence_tol=bad)
