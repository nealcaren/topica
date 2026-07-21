"""Integration tests for the Biterm Topic Model (BTM).

Verify construction/validation, fitting, parameter shapes and simplices, the
global theta and biterm count, save/load round-trips, transform on unseen text,
the background variant, and determinism.
"""

import os
import tempfile

import numpy as np
import pytest

import topica


def _planted(k=3, block=8, n=200, length=5, seed=0):
    """Short-text planted corpus: each doc is `length` words from one of k blocks."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs = [[f"b{d % k}w{int(rng.integers(block))}" for _ in range(length)] for d in range(n)]
    return docs, vocab


def test_btm_recovers_planted_blocks():
    docs, vocab = _planted()
    m = topica.BTM(num_topics=3, seed=7, iters=200)
    m.fit(docs)

    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 3)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert np.isclose(m.theta.sum(), 1.0, atol=1e-9)
    assert m.num_biterms > 0

    blocks = {vocab[int(np.argmax(m.topic_word[j]))].split("w")[0] for j in range(3)}
    assert len(blocks) == 3, f"topics should map to distinct blocks, got {blocks}"


def test_btm_alpha_default_is_50_over_k():
    # alpha=None resolves to 50/k; the fit should succeed and match an explicit pass.
    docs, _ = _planted(seed=1)
    a = topica.BTM(num_topics=5, seed=3, iters=60)
    a.fit(docs)
    b = topica.BTM(num_topics=5, alpha=50.0 / 5, seed=3, iters=60)
    b.fit(docs)
    assert np.allclose(a.topic_word, b.topic_word)


def test_btm_transform_unseen_docs():
    docs, _ = _planted(seed=2)
    m = topica.BTM(num_topics=3, seed=5, iters=100)
    m.fit(docs)
    dt = m.transform(docs[:10])
    assert dt.shape == (10, 3)
    assert np.allclose(dt.sum(axis=1), 1.0, atol=1e-6)
    # an out-of-vocabulary-only document falls back to a valid (uniform) simplex
    oov = m.transform([["ZZZ_not_in_vocab"]])
    assert oov.shape == (1, 3)
    assert np.isclose(oov.sum(), 1.0, atol=1e-6)


def test_btm_background_variant_runs():
    docs, _ = _planted(seed=4)
    m = topica.BTM(num_topics=3, background=True, seed=6, iters=80)
    m.fit(docs)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)


def test_btm_save_load_roundtrip():
    docs, _ = _planted(seed=5)
    m = topica.BTM(num_topics=3, seed=9, iters=80)
    m.fit(docs)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "btm.bin")
        m.save(p)
        m2 = topica.BTM.load(p)
    assert np.allclose(m.topic_word, m2.topic_word)
    assert np.allclose(m.doc_topic, m2.doc_topic)
    assert np.allclose(m.theta, m2.theta)
    assert m2.num_biterms == m.num_biterms
    assert m2.vocabulary == m.vocabulary


def test_btm_determinism():
    docs, _ = _planted(seed=6)
    m1 = topica.BTM(num_topics=3, seed=11, iters=100)
    m1.fit(docs)
    m2 = topica.BTM(num_topics=3, seed=11, iters=100)
    m2.fit(docs)
    assert np.array_equal(m1.topic_word, m2.topic_word)
    assert np.array_equal(m1.doc_topic, m2.doc_topic)
    assert np.array_equal(m1.theta, m2.theta)


def test_btm_window_changes_biterms():
    # A larger window forms more biterms from the same corpus.
    docs, _ = _planted(seed=7, length=8)
    small = topica.BTM(num_topics=3, window=2, seed=1, iters=20)
    small.fit(docs)
    wide = topica.BTM(num_topics=3, window=8, seed=1, iters=20)
    wide.fit(docs)
    assert wide.num_biterms > small.num_biterms


def test_btm_parameter_validations():
    with pytest.raises(ValueError):
        topica.BTM(num_topics=1)
    with pytest.raises(ValueError):
        topica.BTM(num_topics=3, beta=0.0)
    with pytest.raises(ValueError):
        topica.BTM(num_topics=3, alpha=-1.0)
    with pytest.raises(ValueError):
        topica.BTM(num_topics=3, iters=0)
    with pytest.raises(ValueError):
        topica.BTM(num_topics=3, window=1)
    # vocabulary smaller than num_topics is rejected at fit
    with pytest.raises(ValueError):
        topica.BTM(num_topics=5).fit([["a", "b"], ["a", "b"]])
