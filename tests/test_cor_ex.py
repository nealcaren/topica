"""Tests for CorEx: information-theoretic Correlation Explanation topic modeling
(Gallagher et al., TACL 2017), validated against the corextopic package.

CorEx is non-generative: doc_topic is per-topic independent probabilities (rows do
NOT sum to 1) and topic_word is alpha*mis (not a distribution). Tests cover the
standard surface, planted recovery, determinism, anchoring, held-out transform, and
save/load.
"""

import numpy as np
import pytest

import topica


def _planted_docs(reps=30):
    """Three co-occurring word blocks -> three latent topics."""
    blocks = [["a", "b", "c", "d"], ["m", "n", "o", "p"], ["x", "y", "z", "q"]]
    docs = []
    for _ in range(reps):
        for blk in blocks:
            docs.append(list(blk))
    return docs


def test_shapes_and_nonsimplex():
    # Include docs that span two blocks so multiple topics co-activate and the
    # per-topic probabilities do NOT sum to 1 (CorEx topics are not a mixture).
    docs = _planted_docs()
    docs += [["a", "b", "m", "n"], ["m", "n", "x", "y"], ["a", "b", "x", "y"]] * 10
    m = topica.CorEx(3, seed=0).fit(docs, iters=100)
    V = len(m.vocabulary)
    assert m.topic_word.shape == (3, V)
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.mis.shape == (3, V)
    assert m.alpha.shape == (3, V)
    assert np.all((m.doc_topic >= 0) & (m.doc_topic <= 1))
    # doc_topic is NOT a simplex: some documents co-activate multiple topics, so
    # their per-topic probabilities sum to more than 1.
    assert m.doc_topic.sum(axis=1).max() > 1.05
    assert len(m.clusters) == V
    assert m.labels.shape == (len(docs), 3)


def test_recovers_planted_blocks():
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=200)
    # Each block's words should share one cluster; 3 distinct clusters overall.
    vocab = list(m.vocabulary)
    cl = {w: c for w, c in zip(vocab, m.clusters)}
    for blk in (["a", "b", "c", "d"], ["m", "n", "o", "p"], ["x", "y", "z", "q"]):
        present = [cl[w] for w in blk if w in cl]
        assert len(set(present)) == 1, f"block {blk} split across clusters {present}"
    assert len(set(m.clusters)) == 3
    assert m.total_correlation > 0


def test_determinism():
    docs = _planted_docs()
    a = topica.CorEx(3, seed=1).fit(docs, iters=80)
    b = topica.CorEx(3, seed=1).fit(docs, iters=80)
    assert np.array_equal(a.mis, b.mis)
    assert np.array_equal(a.alpha, b.alpha)
    assert a.clusters == b.clusters


def test_determinism_across_threads():
    docs = _planted_docs()
    a = topica.CorEx(3, seed=2).fit(docs, iters=80, num_threads=1)
    b = topica.CorEx(3, seed=2).fit(docs, iters=80, num_threads=4)
    assert np.array_equal(a.mis, b.mis)
    assert a.clusters == b.clusters


def test_total_correlation_and_history():
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=100)
    assert np.isclose(m.total_correlation, sum(m.topic_tc))
    assert len(m.tc_history) == m.iters_run
    assert len(m.fit_history) == m.iters_run
    assert m.fit_history[0][0] == 1


def test_anchoring_places_words():
    docs = _planted_docs()
    anchors = {"first": ["a"], "third": ["x"]}
    m = topica.CorEx(3, anchor_words=anchors, anchor_strength=2.0, seed=0).fit(docs, iters=200)
    vocab = list(m.vocabulary)
    cl = {w: c for w, c in zip(vocab, m.clusters)}
    # Anchor group order = topic order: "first" -> topic 0, "third" -> topic 1.
    assert cl["a"] == 0
    assert cl["x"] == 1
    # The anchored word's block should share its topic.
    assert cl["b"] == 0 and cl["c"] == 0
    assert cl["y"] == 1 and cl["z"] == 1


def test_transform_heldout():
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=100)
    held = [["a", "b", "c", "d"], ["x", "y", "z", "q"]]
    p = np.asarray(m.transform(held))
    assert p.shape == (2, 3)
    assert np.all((p >= 0) & (p <= 1))


def test_transform_corpus_matches_tokens():
    """transform() must remap a held-out Corpus onto the TRAINING vocabulary by word,
    not consume its own column ids (Gate-B fix). A Corpus and the equivalent token
    lists must give identical predictions even when the Corpus vocabulary order
    differs from training."""
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=100)
    held = [["a", "b", "c", "d"], ["x", "y", "z", "q"], ["m", "n", "o", "p"]]
    # A Corpus with a deliberately different (reversed) vocabulary order.
    rev_vocab = list(reversed(list(m.vocabulary)))
    held_corpus = topica.Corpus.from_documents(held, vocabulary=rev_vocab)
    p_tokens = np.asarray(m.transform(held))
    p_corpus = np.asarray(m.transform(held_corpus))
    assert np.allclose(p_tokens, p_corpus), "Corpus transform must match token-list transform"


def test_top_words_and_coherence():
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=100)
    tw = m.top_words(3, topic=0)
    assert len(tw) == 3 and isinstance(tw[0], str)
    tw_w = m.top_words(3, topic=0, weights=True)
    assert len(tw_w) == 3 and isinstance(tw_w[0], tuple)
    assert m.coherence(4).shape == (3,)


def test_save_load_roundtrip(tmp_path):
    docs = _planted_docs()
    m = topica.CorEx(3, seed=0).fit(docs, iters=80)
    p = str(tmp_path / "corex.topica")
    m.save(p)
    loaded = topica.CorEx.load(p)
    assert np.array_equal(m.mis, loaded.mis)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert loaded.clusters == m.clusters
    assert np.isclose(loaded.total_correlation, m.total_correlation)
    assert loaded.settings == m.settings
    # transform round-trips EXACTLY after load (all params persisted).
    held = [["a", "b", "c", "d"], ["x", "y", "z", "q"]]
    assert np.allclose(np.asarray(loaded.transform(held)), np.asarray(m.transform(held)))


def test_rejects_more_anchor_groups_than_topics():
    with pytest.raises(ValueError):
        topica.CorEx(1, anchor_words={"a": ["x"], "b": ["y"]})


def test_rejects_unmatched_anchor_group():
    docs = _planted_docs()
    with pytest.raises((ValueError, RuntimeError)):
        topica.CorEx(3, anchor_words={"ghost": ["zzz_absent"]}).fit(docs, iters=5)


def test_rejects_fraction_count():
    with pytest.raises(ValueError):
        topica.CorEx(3, count="fraction")


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.CorEx(2).fit([])
