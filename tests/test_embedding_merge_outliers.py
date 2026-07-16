"""Manual topic merging and outlier reduction for the clustering models."""

import numpy as np
import pytest

import topica


def _data(k=4, block=6, e=8, seed=0, with_noise=True):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    v = len(vocab)
    word_emb = np.array([[3.0 if d == w // block else 0.0 for d in range(e)] for w in range(v)])
    word_emb += rng.normal(0, 0.2, (v, e))
    docs, doc_emb = [], []
    for d in range(160):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(8)])
        doc_emb.append(word_emb[vocab.index(docs[-1][0])] + rng.normal(0, 0.4, e))
    if with_noise:
        # a handful of scattered documents far from every cluster
        for _ in range(12):
            docs.append([f"b{int(rng.integers(k))}w{int(rng.integers(block))}" for _ in range(8)])
            doc_emb.append(rng.normal(0, 1.0, e) * 6.0)
    return docs, vocab, word_emb, np.array(doc_emb)


@pytest.mark.parametrize("kind", ["bertopic", "top2vec"])
def test_merge_topics(kind):
    docs, vocab, word_emb, doc_emb = _data(with_noise=False)
    if kind == "bertopic":
        m = topica.models.BERTopic(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb)
    else:
        m = topica.models.Top2Vec(min_cluster_size=8, seed=1)
        m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    before = m.num_topics
    assert before >= 3
    m.merge_topics([[0, 1]])
    assert m.num_topics == before - 1
    assert m.doc_topic.shape == (len(docs), m.num_topics)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.topic_word.shape[0] == m.num_topics


@pytest.mark.parametrize("kind", ["bertopic", "top2vec"])
def test_reduce_outliers(kind):
    docs, vocab, word_emb, doc_emb = _data(with_noise=True)
    if kind == "bertopic":
        m = topica.models.BERTopic(min_cluster_size=8, min_samples=4, seed=1)
        m.fit(docs, doc_emb)
    else:
        m = topica.models.Top2Vec(min_cluster_size=8, min_samples=4, seed=1)
        m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    n_out = sum(1 for l in m.labels if l < 0)
    if n_out == 0:
        pytest.skip("clustering left no outliers to reduce")
    reassigned = m.reduce_outliers()
    assert reassigned == n_out
    assert all(l >= 0 for l in m.labels)
    assert m.topic_word.shape[0] == m.num_topics
