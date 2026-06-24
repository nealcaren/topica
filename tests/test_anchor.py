"""Tests for AnchorLDA, the experimental anchor-words spectral estimator
(Arora et al. 2013)."""

import os

import numpy as np
import pytest

import topica


@pytest.fixture(autouse=True)
def _experimental_on():
    """AnchorLDA is gated; enable it for these tests and restore the gate after."""
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        yield
    finally:
        topica.enable_experimental(was)


def _separable_corpus(k=4, words_per_topic=12, n_docs=400, doc_len=40, seed=0):
    """A separable corpus: each topic owns an anchor word that appears only in
    it, plus shared filler. The planted topic assignment is recoverable."""
    rng = np.random.default_rng(seed)
    vocab_blocks = [[f"t{t}_w{j}" for j in range(words_per_topic)] for t in range(k)]
    anchors = [f"t{t}_anchor" for t in range(k)]
    docs, truth = [], []
    for _ in range(n_docs):
        t = int(rng.integers(k))
        truth.append(t)
        words = list(rng.choice(vocab_blocks[t], doc_len - 2))
        words += [anchors[t], anchors[t]]  # the anchor occurs only in topic t
        docs.append(words)
    return docs, truth, anchors


class TestGate:
    def test_refused_without_optin(self):
        topica.enable_experimental(False)
        try:
            with pytest.raises(RuntimeError, match="experimental"):
                topica.AnchorLDA(5)
        finally:
            topica.enable_experimental(True)

    def test_constructs_when_enabled(self):
        m = topica.AnchorLDA(5)
        assert m.num_topics == 5


class TestFit:
    def test_shapes_and_attrs(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        v = len(m.vocabulary)
        assert np.asarray(m.topic_word).shape == (4, v)
        assert np.asarray(m.doc_topic).shape == (len(docs), 4)
        assert len(m.anchors) == 4

    def test_kl_default_is_iterative(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)  # recover="kl"
        assert m.fit_history, "KL recovery should record an objective trace"
        assert isinstance(m.converged, bool)
        # The KL objective is non-increasing along the recorded trace.
        objs = [o for _, o in m.fit_history]
        assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:]))

    def test_l2_is_noniterative(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, recover="l2", min_count=2, seed=0).fit(docs)
        assert m.fit_history == []
        assert m.converged is None

    def test_bad_recover_raises(self):
        with pytest.raises(ValueError, match="recover must be"):
            topica.AnchorLDA(4, recover="nope")

    def test_both_recoveries_recover_separable(self):
        # Both recovery methods should recover the planted separable topics.
        docs, _, anchors = _separable_corpus(k=4, seed=1)
        for rec in ("kl", "l2"):
            m = topica.AnchorLDA(4, recover=rec, min_count=2, seed=0).fit(docs)
            assert set(m.anchors) == set(anchors), rec
            for t in range(4):
                blocks = {w.split("_")[0] for w, _ in m.top_words(8, topic=t)}
                assert len(blocks) == 1, (rec, t)

    def test_topic_word_rows_are_distributions(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        rowsums = np.asarray(m.topic_word).sum(axis=1)
        assert np.allclose(rowsums, 1.0, atol=1e-6)
        assert (np.asarray(m.topic_word) >= 0).all()

    def test_recovers_separable_topics(self):
        # On a separable corpus each planted anchor word should be selected, and
        # each recovered topic should be dominated by one block's vocabulary.
        docs, truth, anchors = _separable_corpus(k=4, seed=1)
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        # Every planted anchor is chosen as some topic's anchor.
        assert set(m.anchors) == set(anchors)
        # Each topic's top words come overwhelmingly from a single block.
        for t in range(4):
            tops = [w for w, _ in m.top_words(8, topic=t)]
            blocks = {w.split("_")[0] for w in tops}
            assert len(blocks) == 1, (t, tops)

    def test_deterministic(self):
        docs, _, _ = _separable_corpus()
        a = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        b = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        assert np.array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))
        assert a.anchors == b.anchors

    def test_accepts_corpus_object(self):
        docs, _, _ = _separable_corpus()
        corpus = topica.Corpus.from_documents(docs)
        m = topica.AnchorLDA(4, seed=0).fit(corpus)
        assert np.asarray(m.doc_topic).shape[1] == 4

    def test_too_few_vocab_raises(self):
        docs = [["a", "b"], ["b", "c"]]
        with pytest.raises(ValueError, match="num_topics"):
            topica.AnchorLDA(10, min_count=1).fit(docs)


class TestSurface:
    def test_conformance(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        assert topica.check_conformance(m) == []

    def test_coherence_returns_per_topic(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        cv = m.coherence(10)
        assert np.asarray(cv).shape == (4,)

    def test_topic_names_roundtrip(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        m.topic_names = ["a", "b", "c", "d"]
        assert m.topic_names == ["a", "b", "c", "d"]
        with pytest.raises(ValueError):
            m.topic_names = ["only", "three", "names"]

    def test_unfitted_access_raises(self):
        m = topica.AnchorLDA(4)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_word

    def test_save_load_roundtrip(self, tmp_path):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        p = os.path.join(tmp_path, "anchor")
        m.save(p)
        loaded = topica.AnchorLDA.load(p)
        assert np.array_equal(np.asarray(m.topic_word), np.asarray(loaded.topic_word))
        assert m.anchors == loaded.anchors
        assert m.vocabulary == loaded.vocabulary


class TestRegistry:
    def test_registered_as_experimental(self):
        info = {m.name: m for m in topica.list_models()}
        assert "AnchorLDA" in info
        assert info["AnchorLDA"].experimental is True
