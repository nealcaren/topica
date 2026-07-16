"""IdealPointTM, count representation (fit without word_embeddings). EXPERIMENTAL,
gated.

It is a topic model (covered by the registry-driven topic-health invariants); these
tests check the ideal-point head specifically — that it recovers planted positions
from counts, orients to anchors, reports representation="counts", and round-trips
(save tag 33).
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


def _planted(n_authors=30, vocab=40, docs_per=12, length=40, seed=0):
    """Two topics over disjoint vocab halves; topic 0 has a within-topic position
    split. Authors carry a planted position; documents mix the two topics."""
    rng = np.random.default_rng(seed)
    half = vocab // 2
    theta = rng.uniform(-1.0, 1.0, n_authors)

    def beta(topic, x):
        eta = np.full(vocab, -3.0)
        if topic == 0:
            eta[:half] = 0.5
            # within-topic discrimination on topic 0's vocab
            for v in range(half):
                eta[v] += x * (2.0 if v % 2 == 0 else -2.0)
        else:
            eta[half:] = 0.5
        e = np.exp(eta - eta.max())
        return e / e.sum()

    docs, group = [], []
    for a in range(n_authors):
        for _ in range(docs_per):
            doc = []
            for _ in range(length):
                t = rng.integers(0, 2)
                v = rng.choice(vocab, p=beta(t, theta[a]))
                doc.append(f"w{v}")
            docs.append(doc)
            group.append(f"a{a}")
    return docs, group, theta


def test_requires_experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(False)
    try:
        with pytest.raises(Exception):
            topica.models.IdealPointTM(num_topics=2)
    finally:
        topica.enable_experimental(was)


def test_recovers_positions():
    docs, group, theta = _planted(seed=1)
    m = topica.models.IdealPointTM(num_topics=2, num_dims=1, seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a29": 1.0}, iters=40)

    assert m.representation == "counts"
    assert m.num_authors == 30
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    recovered = np.array([pos[f"a{a}"] for a in range(30)])
    r = abs(np.corrcoef(recovered, theta)[0, 1])
    assert r > 0.8, f"position recovery r={r:.3f}"
    # positions standardized
    assert abs(recovered.mean()) < 1e-6


def test_topics_and_shapes():
    docs, group, _ = _planted(seed=2)
    m = topica.models.IdealPointTM(num_topics=2, seed=1)
    m.fit(docs, group=group, iters=30)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape[1] == 2
    # rows of topic_word are simplices
    assert np.allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)
    assert m.topic_discrimination.shape == (2,)
    pos, neg = m.position_shift(0, n=5)
    assert len(pos) == 5 and len(neg) == 5


def test_anchors_orient_sign():
    docs, group, _ = _planted(seed=3)
    m = topica.models.IdealPointTM(num_topics=2, seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a29": 1.0})
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    assert pos["a0"] < pos["a29"]


def test_determinism():
    docs, group, _ = _planted(seed=4)
    a = topica.models.IdealPointTM(num_topics=2, seed=1)
    a.fit(docs, group=group, iters=20)
    b = topica.models.IdealPointTM(num_topics=2, seed=1)
    b.fit(docs, group=group, iters=20)
    assert np.array_equal(a.author_positions, b.author_positions)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_save_load(tmp_path):
    docs, group, _ = _planted(seed=5)
    m = topica.models.IdealPointTM(num_topics=2, seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a29": 1.0})
    p = tmp_path / "iplda.topica"
    m.save(str(p))
    m2 = topica.models.IdealPointTM.load(str(p))
    assert np.array_equal(m.author_positions, m2.author_positions)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert m.author_names == m2.author_names
