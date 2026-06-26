"""TBIP: Text-Based Ideal Points (Vafa, Naidu & Blei 2020). EXPERIMENTAL, gated.

It is a topic model (covered by the registry-driven topic-health invariants); these
tests check the ideal-point head specifically -- the experimental gate, getter
shapes, determinism, save/load, and the group= default.
"""
import numpy as np
import pytest

import topica


@pytest.fixture(autouse=True)
def _experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    yield
    topica.enable_experimental(was)


def _planted(n_authors=12, vocab=24, docs_per=6, seed=0):
    """Authors carry a planted ideal point; eta gives words a within-topic position
    tilt. Counts are sampled from the TBIP generative model."""
    rng = np.random.default_rng(seed)
    k = 3
    block = vocab // k
    x = rng.uniform(-1.0, 1.0, n_authors)
    beta = rng.gamma(0.3, 1 / 0.3, size=(k, vocab)) + 0.05
    eta = 0.1 * rng.standard_normal((k, vocab))
    for kk in range(k):
        for v in range(kk * block, (kk + 1) * block):
            eta[kk, v] = 1.5 if v % 2 == 0 else -1.5

    docs, group = [], []
    for a in range(n_authors):
        for _ in range(docs_per):
            theta = rng.gamma(0.3, 1 / 0.3, size=k) + 0.05
            rate = (theta[:, None] * beta * np.exp(x[a] * eta)).sum(0)
            counts = rng.poisson(rate)
            doc = []
            for v, c in enumerate(counts):
                doc.extend([f"w{v}"] * int(c))
            if not doc:
                doc = [f"w{a % vocab}"]
            docs.append(doc)
            group.append(f"a{a}")
    return docs, group, x


def test_requires_experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(False)
    try:
        with pytest.raises(Exception):
            topica.TBIP(num_topics=3)
    finally:
        topica.enable_experimental(was)


def test_shapes_and_getters():
    docs, group, _ = _planted(seed=1)
    m = topica.TBIP(num_topics=3, seed=0, iters=200, batch_size=64)
    m.fit(docs, group=group)
    assert m.num_topics == 3
    assert m.num_authors == 12
    assert m.ideal_points.shape == (12,)
    assert m.topic_word.shape == (3, len(m.vocabulary))
    assert m.ideological_topics.shape == (3, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 3)
    # topic_word rows are simplices
    assert np.allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert len(m.author_names) == 12
    pos, neg = None, None  # top_words returns the standard structure
    tw = m.top_words(5)
    assert len(tw) == 3
    assert len(m.fit_history) == 200
    assert m.iters_run == 200


def test_recovers_positions():
    docs, group, x_true = _planted(seed=2, n_authors=15, docs_per=8)
    m = topica.TBIP(num_topics=3, seed=0, iters=1500, batch_size=len(docs))
    m.fit(docs, group=group)
    pos = dict(zip(m.author_names, m.ideal_points))
    recovered = np.array([pos[f"a{a}"] for a in range(15)])
    r = abs(np.corrcoef(recovered, x_true)[0, 1])
    assert r > 0.7, f"ideal-point recovery r={r:.3f}"


def test_determinism():
    docs, group, _ = _planted(seed=3)
    a = topica.TBIP(num_topics=3, seed=0, iters=150, batch_size=64)
    a.fit(docs, group=group)
    b = topica.TBIP(num_topics=3, seed=0, iters=150, batch_size=64)
    b.fit(docs, group=group)
    assert np.array_equal(a.ideal_points, b.ideal_points)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_group_default_is_per_document():
    """With no group=, each document is its own author."""
    docs, _, _ = _planted(seed=4)
    m = topica.TBIP(num_topics=3, seed=0, iters=100, batch_size=64)
    m.fit(docs)
    assert m.num_authors == len(docs)
    assert m.ideal_points.shape == (len(docs),)


def test_save_load(tmp_path):
    docs, group, _ = _planted(seed=5)
    m = topica.TBIP(num_topics=3, seed=0, iters=150, batch_size=64)
    m.fit(docs, group=group)
    p = tmp_path / "tbip.topica"
    m.save(str(p))
    m2 = topica.TBIP.load(str(p))
    assert np.array_equal(m.ideal_points, m2.ideal_points)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.ideological_topics, m2.ideological_topics)
    assert m.author_names == m2.author_names
    assert m.vocabulary == m2.vocabulary
