"""GMM soft doc_topic for BERTopic (#357).

With clusterer="gmm", `doc_topic` is the GMM posterior responsibilities — a soft
mixture membership — instead of the c-TF-IDF approximate distribution. Documents
that sit between topics get a blend; the hard `labels` remain the row argmax.
"""

import numpy as np
import pytest

import topica


def _blobs(k=4, per=60, dim=10, spread=1.1, sep=1.2, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, (k, dim))
    emb, docs = [], []
    for c in range(k):
        for _ in range(per):
            emb.append(centers[c] + rng.normal(0, spread, dim))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 5, 4)])
    return docs, np.array(emb)


def test_gmm_doc_topic_is_soft_and_consistent():
    docs, emb = _blobs()
    m = topica.BERTopic(clusterer="gmm", num_clusters=4, n_components=10, seed=1)
    m.fit(docs, emb)
    dt = np.asarray(m.doc_topic)
    # Shape matches the topic count; rows are a distribution.
    assert dt.shape == (len(docs), m.num_topics)
    assert np.allclose(dt.sum(axis=1), 1.0)
    # The hard labels are the soft argmax.
    assert np.array_equal(dt.argmax(axis=1), np.asarray(m.labels))


def test_gmm_doc_topic_has_genuine_mixtures():
    # On overlapping topics, some documents get a real blend (not one-hot).
    docs, emb = _blobs(spread=1.1, sep=1.2)
    m = topica.BERTopic(clusterer="gmm", num_clusters=4, n_components=10, seed=1)
    m.fit(docs, emb)
    dt = np.asarray(m.doc_topic)
    max_prob = dt.max(axis=1)
    assert (max_prob < 0.9).sum() >= 5, "expected some multi-topic documents"
    assert max_prob.min() < 0.8


def test_gmm_with_nr_topics_falls_back_to_ctfidf():
    # With topic reduction, the soft responsibilities can't compose through the
    # hard c-TF-IDF merge, so doc_topic falls back to the c-TF-IDF distribution
    # (still a valid (n, num_topics) distribution).
    docs, emb = _blobs()
    m = topica.BERTopic(clusterer="gmm", num_clusters=6, nr_topics=3, n_components=10, seed=1)
    m.fit(docs, emb)
    dt = np.asarray(m.doc_topic)
    assert m.num_topics == 3
    assert dt.shape == (len(docs), 3)
    assert np.allclose(dt.sum(axis=1), 1.0)


def test_non_gmm_doc_topic_unchanged():
    # kmeans keeps the c-TF-IDF approximate distribution; still a valid distribution
    # whose argmax is the label, but not the GMM responsibilities.
    docs, emb = _blobs()
    m = topica.BERTopic(clusterer="kmeans", num_clusters=4, n_components=10, seed=1)
    m.fit(docs, emb)
    dt = np.asarray(m.doc_topic)
    assert dt.shape[1] == m.num_topics
    assert np.allclose(dt.sum(axis=1), 1.0)


def test_gmm_soft_is_deterministic():
    docs, emb = _blobs()
    a = topica.BERTopic(clusterer="gmm", num_clusters=4, n_components=10, seed=1)
    a.fit(docs, emb)
    b = topica.BERTopic(clusterer="gmm", num_clusters=4, n_components=10, seed=1)
    b.fit(docs, emb)
    assert np.array_equal(np.asarray(a.doc_topic), np.asarray(b.doc_topic))
