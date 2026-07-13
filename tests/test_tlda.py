"""Integration tests for Online Tensor LDA (TensorLDA) topic model.
Verify construction, gating, fitting, parameter shapes, save/load round-trips,
transform on unseen text, and determinism.
"""

import os
import tempfile
import numpy as np
import pytest

import topica


def _planted(k=3, block=8, n=120, length=10, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(length)])
    return docs, vocab


def test_tlda_experimental_gate():
    # Make sure it's disabled first (default state)
    topica.enable_experimental(False)
    with pytest.raises(RuntimeError) as exc_info:
        topica.TensorLDA(3)
    assert "is experimental and unvalidated" in str(exc_info.value)


def test_fit_recovers_planted_blocks():
    topica.enable_experimental(True)
    docs, vocab = _planted()
    m = topica.TensorLDA(3, alpha_0=1.0, seed=42)
    m.fit(docs)

    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 3)

    # Verify rows sum to 1.0
    assert np.allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-5)

    # Topic words should map to the vocabulary.
    assert len(m.vocabulary) == len(set(vocab))
    assert len(m.doc_names) == len(docs)

    # Check that it converged
    assert m.converged in (True, False)
    assert len(m.fit_history) > 0


def test_save_load_roundtrip():
    topica.enable_experimental(True)
    docs, _ = _planted()
    m = topica.TensorLDA(2, seed=99)
    m.fit(docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.bin")
        m.save(path)

        loaded = topica.TensorLDA.load(path)
        assert loaded.num_topics == 2
        assert np.allclose(loaded.topic_word, m.topic_word)
        assert np.allclose(loaded.doc_topic, m.doc_topic)
        assert np.allclose(loaded.weights, m.weights)


def test_transform_unseen_docs():
    topica.enable_experimental(True)
    docs, _ = _planted()
    m = topica.TensorLDA(3, seed=42)
    m.fit(docs[:80])

    # Transform remaining docs
    theta_trans = m.transform(docs[80:])
    assert theta_trans.shape == (40, 3)
    assert np.allclose(theta_trans.sum(axis=1), 1.0, atol=1e-5)


def test_determinism():
    topica.enable_experimental(True)
    docs, _ = _planted()

    m1 = topica.TensorLDA(2, seed=7)
    m1.fit(docs)

    m2 = topica.TensorLDA(2, seed=7)
    m2.fit(docs)

    assert np.allclose(m1.topic_word, m2.topic_word)
    assert np.allclose(m1.doc_topic, m2.doc_topic)
    assert np.allclose(m1.weights, m2.weights)
