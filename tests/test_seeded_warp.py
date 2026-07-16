"""WarpLDA backend for SeededLDA (sampler="warp").

SeededLDA's default sparse sweep scores all K topics per token (its asymmetric
β is not sparsity-aware), so it scales poorly in K; the WarpLDA backend (seeded
word phase) is O(1) per token and flat in K. These tests check the warp path
recovers seeded topics, produces valid distributions, is deterministic, honours
the seeds, and rejects the unsupported doc_topic_prior combination.
"""

import numpy as np
import numpy.testing as npt
import pytest

import topica

_BLOCKS = [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]]
_SEEDS = {"t0": ["a"], "t1": ["e"], "t2": ["i"]}


def _docs(n=180):
    return [_BLOCKS[d % 3] + _BLOCKS[d % 3] for d in range(n)]


def _fit(docs, iters=300, **kw):
    m = topica.models.SeededLDA(_SEEDS, seed=1, sampler="warp", **kw)
    m.fit(docs, iters=iters)
    return m


def _recovered(m):
    bsets = [set(b) for b in _BLOCKS]
    covered = set()
    for t in range(m.num_topics):
        top = {w for w, _ in m.top_words(4, topic=t)}
        for bi, bs in enumerate(bsets):
            if bs <= top:
                covered.add(bi)
    return covered


def test_recovers_seeded_blocks():
    m = _fit(_docs())
    assert _recovered(m) == {0, 1, 2}


def test_valid_distributions():
    m = _fit(_docs())
    npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
    npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.topic_word.shape == (3, 12)


def test_deterministic():
    a = _fit(_docs(), iters=120)
    b = _fit(_docs(), iters=120)
    npt.assert_array_equal(a.topic_word, b.topic_word)


def test_seed_word_lands_on_its_topic():
    # Each seed word should be most probable in (or strongly associated with) its
    # own seeded topic: the topic whose top words are that seed's block.
    m = _fit(_docs())
    vocab = m.vocabulary
    tw = m.topic_word
    for ti, (_, words) in enumerate(_SEEDS.items()):
        seed_w = words[0]
        # the topic where this seed word is most probable
        best_t = int(np.argmax(tw[:, vocab.index(seed_w)]))
        block = set(_BLOCKS[ti])
        top = {w for w, _ in m.top_words(4, topic=best_t)}
        assert block <= top


def test_aliases_and_bad_name():
    for name in ("warp", "warplda"):
        m = topica.models.SeededLDA(_SEEDS, seed=1, sampler=name)
        m.fit(_docs(60), iters=40)
        assert m.topic_word.shape[0] == 3
    with pytest.raises(ValueError):
        topica.models.SeededLDA(_SEEDS, sampler="banana")


def test_doc_topic_prior_rejected_for_warp():
    docs = _docs(60)
    prior = np.full((len(docs), 3), 0.1)
    m = topica.models.SeededLDA(_SEEDS, seed=1, sampler="warp")
    with pytest.raises(ValueError):
        m.fit(docs, iters=20, doc_topic_prior=prior)


def _fit_cvb0(docs, iters=200, **kw):
    m = topica.models.SeededLDA(_SEEDS, seed=1, sampler="cvb0", **kw)
    m.fit(docs, iters=iters)
    return m


def test_cvb0_recovers_and_valid():
    m = _fit_cvb0(_docs())
    assert _recovered(m) == {0, 1, 2}
    npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
    npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.topic_word.shape == (3, 12)


def test_cvb0_deterministic_no_draws():
    a = _fit_cvb0(_docs(), iters=120)
    b = _fit_cvb0(_docs(), iters=120)
    npt.assert_array_equal(a.topic_word, b.topic_word)
    assert a.theta_draws is None


def test_cvb0_rejects_doc_topic_prior():
    docs = _docs(60)
    prior = np.full((len(docs), 3), 0.1)
    m = topica.models.SeededLDA(_SEEDS, seed=1, sampler="cvb0")
    with pytest.raises(ValueError):
        m.fit(docs, iters=20, doc_topic_prior=prior)
