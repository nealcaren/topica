"""Phase-final conformance tests: coherence, doc_names, save/load, and iters
for the neural and cluster models (ETM, FASTopic, ProdLDA, BERTopic, Top2Vec).

Save/load scratch goes to the OS default temporary directory (portable across
macOS, Linux, and Windows CI), not a hardcoded /private/tmp.
"""

import inspect
import os
import tempfile

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _planted_text(k=3, block=6, n=90, seed=0):
    """Tiny planted corpus: K word-blocks, each document draws from one block."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(8)])
    return docs, vocab


def _planted_etm(k=3, block=6, e=4, n=90, seed=0):
    """Planted corpus plus axis-aligned word embeddings for ETM."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    word_emb = np.zeros((k * block, e))
    for w in range(k * block):
        word_emb[w, w // block] = 3.0
        word_emb[w] += rng.normal(0, 0.1, e)
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(8)])
    return docs, vocab, word_emb


def _planted_fastopic(k=3, block=6, h=6, n=90, seed=0):
    """Planted corpus plus axis-aligned doc embeddings for FASTopic/cluster."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs, doc_emb = [], []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(8)])
        e = np.zeros(h)
        e[b] = 3.0
        e += rng.normal(0, 0.2, h)
        doc_emb.append(e)
    return docs, np.array(doc_emb), vocab


def _accepts_kwarg(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ---------------------------------------------------------------------------
# ETM
# ---------------------------------------------------------------------------

class TestETM:
    def _model(self, seed=1):
        docs, vocab, word_emb = _planted_etm()
        m = topica.models.ETM(num_topics=3, seed=seed)
        m.fit(docs, word_emb, vocab, iters=30)
        return m, docs, vocab, word_emb

    def test_iters_kwarg(self):
        assert _accepts_kwarg(topica.models.ETM.fit, "iters")

    def test_no_epochs_in_constructor(self):
        sig = inspect.signature(topica.models.ETM.__init__)
        assert "epochs" not in sig.parameters
        assert "em_iters" not in sig.parameters

    def test_coherence_returns_k_finite_floats(self):
        m, *_ = self._model()
        c = m.coherence(5)
        assert c.shape == (3,)
        assert np.all(np.isfinite(c))

    def test_doc_names_length(self):
        m, docs, *_ = self._model()
        dn = m.doc_names
        assert len(dn) == len(docs)

    def test_save_load_roundtrip(self):
        m, docs, vocab, word_emb = self._model()
        with tempfile.NamedTemporaryFile(suffix=".topica", delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = topica.models.ETM.load(path)
            assert np.allclose(m.topic_word, m2.topic_word, atol=1e-12)
            assert np.allclose(m.doc_topic, m2.doc_topic, atol=1e-12)
            assert m2.topic_names == m.topic_names
        finally:
            os.unlink(path)

    def test_save_load_vae(self):
        docs, vocab, word_emb = _planted_etm()
        m = topica.models.ETM(num_topics=3, inference="vae", hidden_size=32, batch_size=32, seed=1)
        m.fit(docs, word_emb, vocab, iters=10)
        with tempfile.NamedTemporaryFile(suffix=".topica", delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = topica.models.ETM.load(path)
            assert np.allclose(m.topic_word, m2.topic_word, atol=1e-12)
            assert np.allclose(m.doc_topic, m2.doc_topic, atol=1e-12)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# FASTopic
# ---------------------------------------------------------------------------

class TestFASTopic:
    def _model(self, seed=1):
        docs, doc_emb, vocab = _planted_fastopic()
        m = topica.models.FASTopic(num_topics=3, lr=0.05, seed=seed)
        m.fit(docs, doc_emb, iters=50)
        return m, docs, doc_emb, vocab

    def test_iters_kwarg(self):
        assert _accepts_kwarg(topica.models.FASTopic.fit, "iters")

    def test_no_epochs_in_constructor(self):
        sig = inspect.signature(topica.models.FASTopic.__init__)
        assert "epochs" not in sig.parameters

    def test_coherence_returns_k_finite_floats(self):
        m, *_ = self._model()
        c = m.coherence(5)
        assert c.shape == (3,)
        assert np.all(np.isfinite(c))

    def test_doc_names_length(self):
        m, docs, *_ = self._model()
        dn = m.doc_names
        assert len(dn) == len(docs)

    def test_save_load_roundtrip(self):
        m, *_ = self._model()
        with tempfile.NamedTemporaryFile(suffix=".topica", delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = topica.models.FASTopic.load(path)
            assert np.allclose(m.topic_word, m2.topic_word, atol=1e-12)
            assert np.allclose(m.doc_topic, m2.doc_topic, atol=1e-12)
            assert m2.topic_names == m.topic_names
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ProdLDA
# ---------------------------------------------------------------------------

class TestProdLDA:
    def _model(self, seed=1):
        docs, vocab = _planted_text()
        m = topica.models.ProdLDA(num_topics=3, batch_size=30, lr=0.01, dropout=0.0, seed=seed)
        m.fit(docs, iters=50)
        return m, docs, vocab

    def test_iters_kwarg(self):
        assert _accepts_kwarg(topica.models.ProdLDA.fit, "iters")

    def test_no_epochs_in_constructor(self):
        sig = inspect.signature(topica.models.ProdLDA.__init__)
        assert "epochs" not in sig.parameters

    def test_save_load_roundtrip(self):
        m, *_ = self._model()
        with tempfile.NamedTemporaryFile(suffix=".topica", delete=False) as f:
            path = f.name
        try:
            m.save(path)
            m2 = topica.models.ProdLDA.load(path)
            assert np.allclose(m.topic_word, m2.topic_word, atol=1e-12)
            assert np.allclose(m.doc_topic, m2.doc_topic, atol=1e-12)
            assert m2.topic_names == m.topic_names
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

class TestBERTopic:
    def _model(self, seed=1):
        docs, doc_emb, vocab = _planted_fastopic()
        m = topica.models.BERTopic(min_cluster_size=5, seed=seed)
        m.fit(docs, doc_emb)
        return m, docs, doc_emb, vocab

    def test_coherence_returns_k_finite_floats(self):
        m, *_ = self._model()
        k = m.num_topics
        if k == 0:
            pytest.skip("clustering found no topics")
        c = m.coherence(5)
        assert c.shape == (k,)
        assert np.all(np.isfinite(c))


# ---------------------------------------------------------------------------
# Top2Vec
# ---------------------------------------------------------------------------

class TestTop2Vec:
    def _model(self, seed=1):
        docs, doc_emb, vocab = _planted_fastopic()
        m = topica.models.Top2Vec(min_cluster_size=5, seed=seed)
        m.fit(docs, doc_emb)
        return m, docs, doc_emb, vocab

    def test_coherence_returns_k_finite_floats(self):
        m, *_ = self._model()
        k = m.num_topics
        if k == 0:
            pytest.skip("clustering found no topics")
        c = m.coherence(5)
        assert c.shape == (k,)
        assert np.all(np.isfinite(c))
