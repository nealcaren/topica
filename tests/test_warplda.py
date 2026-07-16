"""WarpLDA (cache-efficient two-pass Metropolis-Hastings) sampler for the LDA
model (Chen et al., 2016).

WarpLDA is a *different sampler for the same model* as the default SparseLDA,
so the tests check (a) it recovers known topics, (b) it produces valid
distributions, (c) it is deterministic, (d) it accepts the sampler aliases, and
(e) it round-trips through save/load and held-out transform like any other LDA.
Its headline property — flat per-sweep cost in K, so it wins at large K — is a
benchmark concern, not asserted here (timing is too flaky for CI).
"""

import numpy as np
import numpy.testing as npt
import pytest

import topica

# Two cleanly separated topics; the sampler must recover them.
PETS = ["cat", "dog", "fish", "pet", "paw", "tail"]
SPACE = ["planet", "star", "moon", "sun", "orbit", "comet"]
TWO_TOPIC_DOCS = [list(PETS)] * 40 + [list(SPACE)] * 40


def _fit(docs, k=2, iters=300, **kw):
    opts = dict(num_topics=k, seed=1, sampler="warp", optimize_interval=0)
    opts.update(kw)
    m = topica.models.LDA(**opts)
    m.fit(docs, iters=iters, num_samples=5, sample_interval=10)
    return m


def test_recovers_two_topics():
    m = _fit(TWO_TOPIC_DOCS)
    sets = {frozenset(w for w, _ in ws) for ws in m.top_words(6)}
    assert {frozenset(PETS), frozenset(SPACE)} == sets


def test_phi_and_theta_are_valid_distributions():
    m = _fit(TWO_TOPIC_DOCS)
    phi, theta = m.topic_word, m.doc_topic
    npt.assert_allclose(phi.sum(axis=1), 1.0)
    npt.assert_allclose(theta.sum(axis=1), 1.0)
    assert (phi >= 0).all() and (theta >= 0).all()


def test_deterministic_with_fixed_seed():
    a = _fit(TWO_TOPIC_DOCS)
    b = _fit(TWO_TOPIC_DOCS)
    npt.assert_array_equal(a.topic_word, b.topic_word)
    npt.assert_array_equal(a.doc_topic, b.doc_topic)


def test_sampler_aliases_accepted():
    for name in ("warp", "warplda"):
        m = topica.models.LDA(num_topics=2, sampler=name)
        m.fit(TWO_TOPIC_DOCS, iters=50)
        assert m.topic_word.shape[0] == 2


def test_unknown_sampler_rejected():
    with pytest.raises(ValueError):
        topica.models.LDA(num_topics=2, sampler="banana")


def test_recovers_more_topics_at_larger_k():
    # Four disjoint blocks; warp should put each block on its own topic. Exercises
    # the regime warp is built for (its per-sweep cost is flat in K).
    n_blocks, wpb = 4, 5
    vocab = [f"w{i}" for i in range(n_blocks * wpb)]
    docs = []
    for d in range(200):
        b = d % n_blocks
        block = vocab[b * wpb : (b + 1) * wpb]
        docs.append(block + block)
    m = _fit(docs, k=n_blocks, iters=300)
    blocks = [set(vocab[b * wpb : (b + 1) * wpb]) for b in range(n_blocks)]
    covered = set()
    for t in range(m.num_topics):
        top = {w for w, _ in m.top_words(wpb, topic=t)}
        for bi, blk in enumerate(blocks):
            if blk <= top:
                covered.add(bi)
    assert covered == set(range(n_blocks)), f"only recovered {covered}"


def test_save_load_and_transform_round_trip(tmp_path):
    m = _fit(TWO_TOPIC_DOCS)
    path = str(tmp_path / "warplda_roundtrip.tt")
    m.save(path)
    reloaded = topica.models.LDA.load(path)
    npt.assert_allclose(m.topic_word, reloaded.topic_word)

    theta = m.transform([list(PETS), list(SPACE)])
    assert theta.shape == (2, 2)
    npt.assert_allclose(theta.sum(axis=1), 1.0)
    pet_topic = int(np.argmax([phi_t[m.vocabulary.index("cat")] for phi_t in m.topic_word]))
    assert int(theta[0].argmax()) == pet_topic
    assert int(theta[1].argmax()) != pet_topic
