"""transform / fit_transform for the embedding models, and BERTopic's c-TF-IDF
options."""

import numpy as np
import pytest

import topica


def _setup(k=3, block=6, e=8, n=120, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    v = len(vocab)
    word_emb = np.array([[3.0 if d == w // block else 0.0 for d in range(e)] for w in range(v)])
    word_emb += rng.normal(0, 0.2, (v, e))
    docs = [[f"b{d % k}w{int(rng.integers(block))}" for _ in range(10)] for d in range(n)]
    doc_emb = np.array([word_emb[vocab.index(docs[d][0])] + rng.normal(0, 0.3, e) for d in range(n)])
    return docs, vocab, word_emb, doc_emb


def test_etm_transform_roundtrip():
    docs, vocab, word_emb, _ = _setup()
    m = topica.ETM(num_topics=3, seed=1)
    th = m.fit_transform(docs, word_emb, vocab, iters=30)
    assert th.shape == (len(docs), 3)
    assert np.allclose(th.sum(axis=1), 1.0)
    held = m.transform(docs[:5])
    assert held.shape == (5, 3) and np.allclose(held.sum(axis=1), 1.0)
    # fit_transform equals the model's own doc_topic.
    assert np.allclose(th, m.doc_topic)


def test_top2vec_transform():
    docs, vocab, word_emb, doc_emb = _setup()
    m = topica.Top2Vec(min_cluster_size=8, seed=1)
    th = m.fit_transform(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    assert th.shape[0] == len(docs)
    new = m.transform(docs[:5], doc_emb[:5])
    assert new.shape == (5, m.num_topics) and np.allclose(new.sum(axis=1), 1.0)


def test_bertopic_transform_and_ctfidf_options():
    docs, _, _, doc_emb = _setup()
    m = topica.BERTopic(min_cluster_size=8, seed=1)
    th = m.fit_transform(docs, doc_emb)
    assert th.shape[0] == len(docs)
    new = m.transform(docs[:5])
    assert new.shape == (5, m.num_topics)
    # c-TF-IDF options run and keep topics block-pure.
    mb = topica.BERTopic(min_cluster_size=8, bm25=True, reduce_frequent=True, seed=1)
    mb.fit(docs, doc_emb)
    for t in range(mb.num_topics):
        blocks = {w.split("w")[0] for w, _ in mb.top_words(4, topic=t, weights=True)}
        assert len(blocks) == 1


def _fit_bertopic_no_clusters():
    """Return a fitted BERTopic that found zero clusters (min_cluster_size too
    large for the data), which is the condition that triggered issue #103."""
    import warnings
    rng = np.random.default_rng(0)
    # Structureless (random) embeddings over enough documents that HDBSCAN
    # runs cleanly but finds no cluster at this min_cluster_size, so num_topics
    # is 0. (A handful of documents with min_cluster_size above the dataset
    # size instead panics inside the MST crate; see the follow-up issue.)
    docs = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"]] * 30
    emb = rng.standard_normal((len(docs), 4))
    m = topica.BERTopic(min_cluster_size=40, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(docs, emb)
    return m, docs


def test_bertopic_zero_clusters_transform_raises():
    m, docs = _fit_bertopic_no_clusters()
    if m.num_topics > 0:
        pytest.skip("clustering found topics with this seed; skip zero-cluster guard test")
    with pytest.raises(RuntimeError, match="num_topics=0"):
        m.transform(docs)


def test_bertopic_zero_clusters_top_words_raises():
    m, _ = _fit_bertopic_no_clusters()
    if m.num_topics > 0:
        pytest.skip("clustering found topics with this seed; skip zero-cluster guard test")
    with pytest.raises(RuntimeError, match="num_topics=0"):
        m.top_words(5)


# ---------------------------------------------------------------------------
# #106: all four embedding models share transform(data, doc_embeddings=None)
# ---------------------------------------------------------------------------

class TestEmbeddingTransformSignature:
    """Verify the canonical transform(data, doc_embeddings=None) signature
    for the four embedding models: correct inputs work, missing required
    inputs raise ValueError."""

    @pytest.fixture(scope="class")
    def setup(self):
        docs, vocab, word_emb, doc_emb = _setup()
        return docs, vocab, word_emb, doc_emb

    def test_etm_transform_with_data_only(self, setup):
        docs, vocab, word_emb, _ = setup
        m = topica.ETM(num_topics=3, seed=1)
        m.fit(docs, word_emb, vocab, iters=30)
        result = m.transform(docs[:5])
        assert result.shape == (5, 3)

    def test_etm_transform_ignores_doc_embeddings(self, setup):
        """ETM.transform does not use doc_embeddings; passing it is a no-op."""
        docs, vocab, word_emb, doc_emb = setup
        m = topica.ETM(num_topics=3, seed=1)
        m.fit(docs, word_emb, vocab, iters=30)
        result_a = m.transform(docs[:5])
        result_b = m.transform(docs[:5], doc_emb[:5])
        np.testing.assert_array_equal(result_a, result_b)

    def test_bertopic_transform_with_data_only(self, setup):
        docs, _, _, doc_emb = setup
        m = topica.BERTopic(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb)
        if m.num_topics == 0:
            pytest.skip("no clusters found; skip transform test")
        result = m.transform(docs[:5])
        assert result.shape == (5, m.num_topics)

    def test_bertopic_transform_ignores_doc_embeddings(self, setup):
        """BERTopic.transform does not use doc_embeddings; passing it is a no-op."""
        docs, _, _, doc_emb = setup
        m = topica.BERTopic(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb)
        if m.num_topics == 0:
            pytest.skip("no clusters found; skip transform test")
        result_a = m.transform(docs[:5])
        result_b = m.transform(docs[:5], doc_emb[:5])
        np.testing.assert_array_equal(result_a, result_b)

    def test_top2vec_transform_with_both_args(self, setup):
        docs, vocab, word_emb, doc_emb = setup
        m = topica.Top2Vec(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
        if m.num_topics == 0:
            pytest.skip("no clusters found; skip transform test")
        result = m.transform(docs[:5], doc_emb[:5])
        assert result.shape == (5, m.num_topics)

    def test_top2vec_transform_missing_doc_embeddings_raises(self, setup):
        docs, vocab, word_emb, doc_emb = setup
        m = topica.Top2Vec(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
        if m.num_topics == 0:
            pytest.skip("no clusters found; skip ValueError guard test")
        with pytest.raises(ValueError, match="doc_embeddings"):
            m.transform(docs[:5])

    def test_fastopic_transform_with_doc_embeddings(self, setup):
        docs, _, _, doc_emb = setup
        m = topica.FASTopic(num_topics=3, seed=1)
        m.fit(docs, doc_emb, iters=20)
        result = m.transform(doc_embeddings=doc_emb[:5])
        assert result.shape == (5, 3)

    def test_fastopic_transform_data_positional_doc_embeddings_kwarg(self, setup):
        docs, _, _, doc_emb = setup
        m = topica.FASTopic(num_topics=3, seed=1)
        m.fit(docs, doc_emb, iters=20)
        # data is first positional but ignored; doc_embeddings is second or kwarg
        result = m.transform(None, doc_emb[:5])
        assert result.shape == (5, 3)

    def test_fastopic_transform_missing_doc_embeddings_raises(self, setup):
        docs, _, _, doc_emb = setup
        m = topica.FASTopic(num_topics=3, seed=1)
        m.fit(docs, doc_emb, iters=20)
        with pytest.raises(ValueError, match="doc_embeddings"):
            m.transform(docs[:5])
