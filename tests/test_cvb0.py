"""CVB0 (collapsed variational Bayes, zeroth-order) inference for LDA, exposed
as LDA(sampler="cvb0").

CVB0 is a deterministic, non-sampling backend for the *same* LDA model: each
(doc, word) cell keeps a soft topic responsibility updated from expected counts.
These tests check it recovers known topics, produces valid distributions, is
exactly reproducible for a seed, exposes no MCMC θ-draws (it is not a sampler),
and supports the convergence-tol early stop.
"""

import numpy as np
import numpy.testing as npt
import pytest

import topica

PETS = ["cat", "dog", "fish", "pet", "paw", "tail"]
SPACE = ["planet", "star", "moon", "sun", "orbit", "comet"]
TWO_TOPIC_DOCS = [list(PETS)] * 40 + [list(SPACE)] * 40


def _fit(docs, k=2, iters=200, **kw):
    m = topica.LDA(k, seed=1, sampler="cvb0", optimize_interval=0, **kw)
    m.fit(docs, iters=iters)
    return m


def test_recovers_two_topics():
    m = _fit(TWO_TOPIC_DOCS)
    sets = {frozenset(w for w, _ in ws) for ws in m.top_words(6, weights=True)}
    assert {frozenset(PETS), frozenset(SPACE)} == sets


def test_valid_distributions():
    m = _fit(TWO_TOPIC_DOCS)
    npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
    npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)
    assert (m.topic_word >= 0).all() and (m.doc_topic >= 0).all()


def test_deterministic_for_seed():
    a = _fit(TWO_TOPIC_DOCS)
    b = _fit(TWO_TOPIC_DOCS)
    npt.assert_array_equal(a.topic_word, b.topic_word)
    npt.assert_array_equal(a.doc_topic, b.doc_topic)


def test_no_theta_draws():
    # CVB0 is deterministic variational inference, not MCMC: no posterior draws.
    m = _fit(TWO_TOPIC_DOCS)
    assert m.theta_draws is None


def test_convergence_tol_early_stops():
    m = topica.LDA(2, seed=1, sampler="cvb0", optimize_interval=0)
    m.fit(TWO_TOPIC_DOCS, iters=1000, convergence_tol=1e-4, check_every=5)
    assert m.converged


def test_aliases_and_bad_name():
    for name in ("cvb0", "cvb"):
        m = topica.LDA(2, seed=1, sampler=name)
        m.fit(TWO_TOPIC_DOCS, iters=30)
        assert m.topic_word.shape[0] == 2
    with pytest.raises(ValueError):
        topica.LDA(2, sampler="banana")


def test_recovers_more_topics_at_larger_k():
    n_blocks, wpb = 4, 5
    vocab = [f"w{i}" for i in range(n_blocks * wpb)]
    docs = []
    for d in range(200):
        b = d % n_blocks
        block = vocab[b * wpb : (b + 1) * wpb]
        docs.append(block + block)
    m = _fit(docs, k=n_blocks, iters=150)
    blocks = [set(vocab[b * wpb : (b + 1) * wpb]) for b in range(n_blocks)]
    covered = set()
    for t in range(m.num_topics):
        top = {w for w, _ in m.top_words(wpb, topic=t, weights=True)}
        for bi, blk in enumerate(blocks):
            if blk <= top:
                covered.add(bi)
    assert covered == set(range(n_blocks)), f"only recovered {covered}"


def test_save_load_round_trip(tmp_path):
    m = _fit(TWO_TOPIC_DOCS)
    path = str(tmp_path / "cvb0.bin")
    m.save(path)
    reloaded = topica.LDA.load(path)
    npt.assert_allclose(m.topic_word, reloaded.topic_word)
