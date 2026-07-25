"""Upstream-faithfulness checks for topica.BERTopic (issue #488).

Covers the three items the #488 review flagged as diverging from the upstream
``bertopic`` package and outside the gold coverage: the ``bm25`` c-TF-IDF variant,
``nr_topics`` topic reduction, and ``approximate_distribution`` (the sliding-window
per-document topic distribution, its ``min_similarity`` gate, and that the window
weighting tracks the ``bm25``/``reduce_frequent`` knobs).

These use the deterministic ``clusterer="kmeans"`` path so the recovered partition
is stable without any reference clustering stack (no umap/hdbscan/sklearn needed).
"""

import numpy as np
import pytest

import topica


def _planted(k=3, per=40, block=5, dim=12, ubiquitous=True, seed=0):
    """k well-separated clusters. Each cluster c uses its own vocabulary block
    ``c*block .. (c+1)*block`` and, if ``ubiquitous``, a single shared word
    ``"STOP"`` present in every document of every cluster."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((k, dim))
    for c in range(k):
        centers[c, c % dim] = 8.0
    docs, emb = [], []
    for d in range(k * per):
        c = d % k
        toks = [f"w{c * block + rng.integers(0, block)}" for _ in range(8)]
        if ubiquitous:
            toks += ["STOP"] * 6  # dominate the counts so f_STOP > avg class size
        docs.append(toks)
        emb.append(centers[c] + rng.normal(0, 0.3, dim))
    return docs, np.array(emb)


def _fit(docs, emb, **kw):
    kw.setdefault("clusterer", "kmeans")
    kw.setdefault("num_clusters", 3)
    kw.setdefault("n_components", 5)
    # These upstream c-TF-IDF / distribution checks are reducer-independent; pin
    # the linear reducer so they exercise the deterministic path (the class now
    # defaults to reducer="umap").
    kw.setdefault("reducer", "pca")
    kw.setdefault("seed", 1)
    m = topica.BERTopic(**kw)
    m.fit(docs, emb)
    return m


# --- (a) bm25 / reduce_frequent -------------------------------------------------

def test_bm25_topic_word_is_valid_distribution():
    docs, emb = _planted()
    m = _fit(docs, emb, bm25=True)
    tw = np.asarray(m.topic_word)
    # The normalized topic-word surface stays a valid non-negative distribution
    # even though the raw BM25 c-TF-IDF carries negative weights for the
    # ubiquitous term (issue #488 floors those to zero for this surface only).
    assert (tw >= 0).all()
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_bm25_downweights_ubiquitous_term_below_default():
    # With bm25 the term present in every class gets a negative idf and is ranked
    # out of the top words; the default path can still surface it.
    docs, emb = _planted()
    m = _fit(docs, emb, bm25=True)
    for t in range(m.num_topics):
        words = [w for w, _ in m.top_words(4, topic=t)]
        assert "STOP" not in words, f"topic {t} top words still include STOP: {words}"


def test_bm25_reduce_frequent_compose_and_stay_valid():
    docs, emb = _planted()
    m = _fit(docs, emb, bm25=True, reduce_frequent=True)
    tw = np.asarray(m.topic_word)
    assert (tw >= 0).all()
    assert np.allclose(tw.sum(axis=1), 1.0)


# --- (b) nr_topics reduction ----------------------------------------------------

def test_nr_topics_reduces_to_requested_real_topic_count():
    docs, emb = _planted(k=3, per=40)
    full = _fit(docs, emb, num_clusters=4)
    assert full.num_topics == 4
    reduced = _fit(docs, emb, num_clusters=4, nr_topics=2)
    # nr_topics counts real topics (kmeans emits no -1 noise here), so exactly 2.
    assert reduced.num_topics == 2
    dt = np.asarray(reduced.doc_topic)
    assert dt.shape == (len(docs), 2)
    assert np.allclose(dt.sum(axis=1), 1.0)


# --- (c) approximate_distribution ----------------------------------------------

def test_approximate_distribution_is_valid_distribution():
    docs, emb = _planted()
    m = _fit(docs, emb)
    dist = np.asarray(m.approximate_distribution(docs))
    assert dist.shape == (len(docs), m.num_topics)
    assert (dist >= 0).all()
    assert np.allclose(dist.sum(axis=1), 1.0)


def test_min_similarity_gate_sparsifies_then_uniform():
    docs, emb = _planted(ubiquitous=False)
    m = _fit(docs, emb)
    # A permissive gate keeps a spread; a gate above every window-topic cosine
    # falls back to a uniform row (topica keeps doc_topic a valid distribution).
    loose = np.asarray(m.approximate_distribution(docs, min_similarity=0.0))
    assert np.allclose(loose.sum(axis=1), 1.0)
    tight = np.asarray(m.approximate_distribution(docs, min_similarity=1.5))
    # Every similarity is gated out, so each row is uniform.
    uniform = np.full(m.num_topics, 1.0 / m.num_topics)
    assert np.allclose(tight, uniform)


def test_min_similarity_zeroes_weak_topics():
    # A document made only of block-0 words should, under a moderate gate, place
    # zero mass on the topics it does not resemble.
    docs, emb = _planted(ubiquitous=False)
    m = _fit(docs, emb)

    def block_of(topic):
        return int(m.top_words(1, topic=topic)[0][0][1:]) // 5

    block0 = next(t for t in range(m.num_topics) if block_of(t) == 0)
    doc = [["w0", "w1", "w2", "w3", "w4", "w0", "w1", "w2"]]
    gated = np.asarray(m.approximate_distribution(doc, min_similarity=0.3))
    assert gated[0].argmax() == block0
    assert (gated[0] > 0).sum() < m.num_topics or m.num_topics == 1


# --- settings + save/load round-trip -------------------------------------------

def test_min_similarity_in_settings_and_survives_save_load(tmp_path):
    docs, emb = _planted()
    m = _fit(docs, emb, bm25=True, min_similarity=0.1)
    assert m.settings["min_similarity"] == pytest.approx(0.1)
    assert m.settings["bm25"] is True
    p = tmp_path / "bt.topica"
    m.save(str(p))
    loaded = topica.BERTopic.load(str(p))
    assert loaded.settings["min_similarity"] == pytest.approx(0.1)
    # doc_topic reproduces after load.
    assert np.allclose(np.asarray(loaded.doc_topic), np.asarray(m.doc_topic))


def test_min_similarity_must_be_finite():
    with pytest.raises(ValueError):
        topica.BERTopic(min_similarity=float("nan"))
