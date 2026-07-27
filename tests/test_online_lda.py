"""Tests for OnlineLDA — online (streaming) variational-Bayes LDA (issue #572).

Follows the standard idioms: shapes/normalization, planted-data recovery,
determinism, save/load round-trip, the streaming partial_fit / transform surface,
and parameter validation.
"""
import random

import numpy as np
import pytest

import topica

_BLOCKS = [
    ["sport", "ball", "team", "game", "score"],
    ["bank", "money", "loan", "rate", "cash"],
    ["film", "movie", "actor", "scene", "plot"],
]


def _planted_docs(n=300, doc_len=25, mix=0.85, seed=0):
    """Each doc draws `doc_len` tokens, mostly from its own block."""
    rng = random.Random(seed)
    vocab = sum(_BLOCKS, [])
    docs = []
    for i in range(n):
        block = _BLOCKS[i % len(_BLOCKS)]
        docs.append(
            [rng.choice(block) if rng.random() < mix else rng.choice(vocab) for _ in range(doc_len)]
        )
    return docs


def _toy_docs():
    return [["a", "b", "c"], ["a", "b", "b"], ["c", "c", "d"], ["d", "e", "f"]]


def test_shapes_and_normalization():
    m = topica.OnlineLDA(2, batch_size=2, seed=0).fit(_toy_docs(), iters=20)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape == (len(_toy_docs()), 2)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert (m.topic_word >= 0).all()


def test_determinism():
    a = topica.OnlineLDA(3, batch_size=32, seed=1).fit(_planted_docs(), iters=40)
    b = topica.OnlineLDA(3, batch_size=32, seed=1).fit(_planted_docs(), iters=40)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_recovers_planted_topics():
    m = topica.OnlineLDA(3, batch_size=32, tau=1.0, kappa=0.7, seed=42).fit(
        _planted_docs(), iters=80
    )
    # Every planted block should be the top-5 of exactly one recovered topic.
    recovered = set()
    for t in range(3):
        top = {w for w, _ in m.top_words(5, topic=t)}
        for b, block in enumerate(_BLOCKS):
            if set(block) <= top:
                recovered.add(b)
    assert recovered == {0, 1, 2}, f"only recovered {recovered}"


def test_partial_fit_streams_and_advances_schedule():
    docs = _planted_docs()
    m = topica.OnlineLDA(3, batch_size=32, seed=7).fit(docs[:100], iters=20)
    updates_before = m.updates
    theta = m.partial_fit(docs[100:132])
    assert theta.shape == (32, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert m.updates == updates_before + 1
    # A second minibatch advances the step index again.
    m.partial_fit(docs[132:164])
    assert m.updates == updates_before + 2


def test_transform_does_not_mutate_model():
    docs = _planted_docs()
    m = topica.OnlineLDA(3, batch_size=32, seed=3).fit(docs, iters=40)
    before = m.topic_word.copy()
    updates_before = m.updates
    theta = m.transform(docs[:20])
    assert theta.shape == (20, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.array_equal(m.topic_word, before)
    assert m.updates == updates_before


def test_partial_fit_requires_fit_first():
    with pytest.raises((RuntimeError, ValueError)):
        topica.OnlineLDA(2).partial_fit(_toy_docs())


def test_save_load_round_trip(tmp_path):
    docs = _planted_docs(n=120)
    m = topica.OnlineLDA(3, batch_size=32, seed=5).fit(docs, iters=30)
    path = str(tmp_path / "online.topica")
    m.save(path)
    loaded = topica.OnlineLDA.load(path)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert np.array_equal(m.doc_topic, loaded.doc_topic)
    assert loaded.updates == m.updates
    assert loaded.settings == m.settings
    # A loaded model can resume streaming on the same schedule.
    t1 = m.partial_fit(docs[:32])
    t2 = loaded.partial_fit(docs[:32])
    assert np.array_equal(t1, t2)
    assert np.array_equal(m.topic_word, loaded.topic_word)


def test_fit_history_is_a_bound_trace():
    m = topica.OnlineLDA(3, batch_size=32, seed=0).fit(_planted_docs(), iters=15)
    hist = m.fit_history
    assert len(hist) == 15
    passes = [p for p, _ in hist]
    assert passes == list(range(1, 16))


def test_convergence_tol_early_stops():
    m = topica.OnlineLDA(3, batch_size=32, seed=0).fit(
        _planted_docs(), iters=200, convergence_tol=1e-3
    )
    assert m.converged
    assert len(m.fit_history) < 200


def test_settings_keys_match_constructor():
    s = topica.OnlineLDA(4, alpha_sum=2.0, beta=0.02, tau=8.0, kappa=0.6).settings
    assert s["num_topics"] == 4
    assert s["alpha_sum"] == 2.0
    assert s["beta"] == 0.02
    assert s["tau"] == 8.0
    assert s["kappa"] == 0.6


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.OnlineLDA(2).fit([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": 0.0},
        {"beta": -1.0},
        {"kappa": 0.4},  # must be in (0.5, 1]
        {"kappa": 1.5},
        {"tau": -1.0},
        {"batch_size": 0},
        {"inner_iters": 0},
        {"alpha_sum": 0.0},
    ],
)
def test_rejects_bad_params(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        topica.OnlineLDA(3, **kwargs)
