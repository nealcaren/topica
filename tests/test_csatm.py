"""Tests for CSATM — Conversational Structure Aware and Context Sensitive Topic
Model (Sun, Loparo & Kolacinski, IEEE ICSC 2020, arXiv:2002.02353; issue #811).

Follows the four idioms in CONTRIBUTING-MODELS.md (shapes/normalization,
planted-data recovery, determinism, save-load + bad-params) plus an
analysis-surface check and CSATM-specific structural tests (popularity from the
reply tree, transitivity smoothing, and the LDA-reduction identity).
"""
import numpy as np
import pytest

import topica


def _planted_docs():
    """Two disjoint-vocabulary blocks; each planted topic should own one block."""
    a = [["cat", "dog", "pet", "cat", "dog", "pet"] for _ in range(15)]
    b = [["star", "moon", "sky", "star", "moon", "sky"] for _ in range(15)]
    return a + b


def test_shapes_and_normalization():
    docs = _planted_docs()
    m = topica.CSATM(2, seed=0).fit(docs, iters=50)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 2)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)


def test_recovers_planted_topics():
    docs = _planted_docs()
    m = topica.CSATM(2, seed=3).fit(docs, iters=300)
    blocks = [{"cat", "dog", "pet"}, {"star", "moon", "sky"}]
    tops = [{w for w in m.top_words(3, topic=t)} for t in range(2)]
    owned = [max(range(2), key=lambda b: len(tops[t] & blocks[b])) for t in range(2)]
    assert set(owned) == {0, 1}


def test_determinism():
    docs = _planted_docs()
    a = topica.CSATM(2, seed=1).fit(docs, iters=100)
    b = topica.CSATM(2, seed=1).fit(docs, iters=100)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    # A different seed must differ, so the test cannot pass trivially.
    c = topica.CSATM(2, seed=999).fit(docs, iters=100)
    assert not np.array_equal(a.topic_word, c.topic_word)


def test_save_load_roundtrip(tmp_path):
    docs = _planted_docs()
    m = topica.CSATM(2, seed=0).fit(docs, iters=50)
    p = str(tmp_path / "m.tt")
    m.save(p)
    loaded = topica.CSATM.load(p)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert np.array_equal(m.doc_topic, loaded.doc_topic)


def test_bad_params():
    with pytest.raises(ValueError):
        topica.CSATM(0)
    with pytest.raises(ValueError):
        topica.CSATM(2, alpha=0.0)
    with pytest.raises(ValueError):
        topica.CSATM(2, beta=-1.0)
    with pytest.raises(ValueError):
        topica.CSATM(2, lambda_=0.0)
    with pytest.raises(ValueError):
        topica.CSATM(2, weight_seq="not-a-sequence")


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.CSATM(2).fit([])


def test_analysis_surface():
    docs = _planted_docs()
    m = topica.CSATM(2, seed=0).fit(docs, iters=50)
    # The four required attributes power the model-neutral analysis surface.
    assert topica.inspect.topic_table(m) is not None
    assert m.num_topics == 2
    assert len(m.vocabulary) == m.topic_word.shape[1]


# --- CSATM-specific structure ------------------------------------------------

def _threaded_corpus():
    """Two threads, each a root post + a depth-3 reply chain. Returns
    (docs, parents). Doc order: [root0, c0a, c0b, root1, c1a, c1b]."""
    docs = [
        ["cat", "dog", "pet"],   # 0 root (thread 0)
        ["cat", "pet", "dog"],   # 1 -> 0
        ["dog", "cat", "pet"],   # 2 -> 1
        ["star", "moon", "sky"], # 3 root (thread 1)
        ["moon", "sky", "star"], # 4 -> 3
        ["sky", "star", "moon"], # 5 -> 4
    ]
    parents = [-1, 0, 1, -1, 3, 4]
    return docs, parents


def test_popularity_from_reply_tree():
    docs, parents = _threaded_corpus()
    m = topica.CSATM(2, seed=0).fit(docs, parents, iters=30)
    pop = m.popularity
    assert pop.shape == (6,)
    # Arithmetic default (c=1, d=0.5): a chain node with a single child at level 2
    # and a grandchild at level 3 -> w1 + w2 + w3 = 1 + 0.5 + 0 = 1.5; a leaf -> 1.
    assert pop[0] == pytest.approx(1.5)  # root: self + child(0.5) + grandchild(0)
    assert pop[1] == pytest.approx(1.5)  # middle
    assert pop[2] == pytest.approx(1.0)  # leaf
    assert pop[5] == pytest.approx(1.0)  # leaf


def test_transitivity_changes_doc_topic():
    docs, parents = _threaded_corpus()
    m = topica.CSATM(2, seed=0).fit(docs, parents, iters=100)
    # The smoothed doc_topic differs from the raw Gibbs theta for non-root nodes
    # (root nodes have a self-only path, so they are unchanged).
    assert not np.allclose(m.doc_topic, m.doc_topic_raw)
    # Root nodes (self-only transitivity path) are unchanged.
    np.testing.assert_allclose(m.doc_topic[0], m.doc_topic_raw[0], atol=1e-12)
    np.testing.assert_allclose(m.doc_topic[3], m.doc_topic_raw[3], atol=1e-12)


def test_no_parents_reduces_to_flat_popularity():
    docs = _planted_docs()
    m = topica.CSATM(2, seed=0, lambda_=1.0).fit(docs, iters=50)
    # Every document is a root: popularity 1 and no transitivity smoothing.
    assert np.allclose(m.popularity, 1.0)
    np.testing.assert_allclose(m.doc_topic, m.doc_topic_raw, atol=1e-12)


def test_parents_length_validated():
    docs, _ = _threaded_corpus()
    with pytest.raises(ValueError):
        topica.CSATM(2, seed=0).fit(docs, [-1, 0])  # wrong length
