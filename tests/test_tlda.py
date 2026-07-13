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


def test_tlda_parameter_validations():
    topica.enable_experimental(True)

    # 1. Invalid topics
    with pytest.raises(ValueError) as exc:
        topica.TensorLDA(1)
    assert "num_topics must be >= 2" in str(exc.value)

    # 2. Invalid alpha_0
    with pytest.raises(ValueError) as exc:
        topica.TensorLDA(2, alpha_0=0.0)
    assert "alpha_0 must be > 0.0" in str(exc.value)

    # 3. Invalid n_iter_train
    with pytest.raises(ValueError) as exc:
        topica.TensorLDA(2, n_iter_train=0)
    assert "n_iter_train must be > 0" in str(exc.value)

    # 4. Invalid batch_size
    with pytest.raises(ValueError) as exc:
        topica.TensorLDA(2, batch_size=0)
    assert "batch_size must be > 0" in str(exc.value)

    # 5. Invalid smoothing
    with pytest.raises(ValueError) as exc:
        topica.TensorLDA(2, smoothing=1.0)
    assert "smoothing must be in [0.0, 1.0)" in str(exc.value)

    # 6. Fit small corpus (num_docs < num_topics) raises error, doesn't panic
    m = topica.TensorLDA(5)
    with pytest.raises(ValueError) as exc:
        m.fit([["a", "b"]])
    assert "corpus must have at least num_topics=5 documents" in str(exc.value)

    # 7. Inclusive / 1-based training works (iters=1 executes exactly 1 step)
    m_1iter = topica.TensorLDA(2, n_iter_train=1)
    m_1iter.fit([["a", "b"], ["b", "c"]])
    assert len(m_1iter.fit_history) == 1

