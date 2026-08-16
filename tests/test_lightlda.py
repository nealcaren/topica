"""LightLDA (alias-table Metropolis-Hastings) sampler for the LDA model.

LightLDA is a *different sampler for the same model* as the default SparseLDA,
so the tests check (a) it recovers known topics, (b) it produces valid
distributions, (c) it is deterministic, (d) its topic quality matches the
SparseLDA sampler on a real corpus, and (e) it round-trips through save/load and
held-out transform like any other LDA.
"""

import csv
import os
import tempfile

import numpy as np
import pytest

import topica
from topica import Corpus


# Two cleanly separated topics; both samplers must recover them.
PETS = ["cat", "dog", "fish", "pet", "paw", "tail"]
SPACE = ["planet", "star", "moon", "sun", "orbit", "comet"]
TWO_TOPIC_DOCS = [list(PETS)] * 40 + [list(SPACE)] * 40


def _fit(sampler, docs, k=2, **kw):
    opts = dict(num_topics=k, seed=1, sampler=sampler, optimize_interval=0)
    opts.update(kw)
    m = topica.LDA(**opts)
    m.fit(docs, iters=300, num_samples=5, sample_interval=10)
    return m


def _topic_sets(model, n=5):
    return [frozenset(w for w, _ in ws) for ws in model.top_words(n, weights=True)]


def test_recovers_two_topics():
    m = _fit("lightlda", TWO_TOPIC_DOCS)
    sets = _topic_sets(m, n=6)
    # Each recovered topic should be (essentially) one of the planted ones.
    assert {frozenset(PETS), frozenset(SPACE)} == set(sets)


def test_phi_and_theta_are_valid_distributions():
    m = _fit("lightlda", TWO_TOPIC_DOCS)
    phi, theta = m.topic_word, m.doc_topic
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert (phi >= 0).all() and (theta >= 0).all()


def test_deterministic_with_fixed_seed():
    a = _fit("lightlda", TWO_TOPIC_DOCS)
    b = _fit("lightlda", TWO_TOPIC_DOCS)
    assert np.allclose(a.topic_word, b.topic_word)
    assert np.allclose(a.doc_topic, b.doc_topic)


def test_mh_steps_validation():
    # mh_steps must be >= 1 for the alias sampler.
    with pytest.raises(ValueError):
        topica.LDA(num_topics=2, sampler="lightlda", mh_steps=0)
    # unknown sampler name is rejected.
    with pytest.raises(ValueError):
        topica.LDA(num_topics=2, sampler="banana")


def test_sampler_aliases_accepted():
    # Friendly aliases resolve to the same backend.
    for name in ["lightlda", "light", "alias"]:
        m = topica.LDA(num_topics=2, sampler=name)
        m.fit(TWO_TOPIC_DOCS, iters=50)
        assert m.topic_word.shape[0] == 2
    for name in ["sparse", "mallet"]:
        topica.LDA(num_topics=2, sampler=name)


def test_save_load_and_transform_round_trip():
    m = _fit("lightlda", TWO_TOPIC_DOCS)
    path = os.path.join(tempfile.gettempdir(), "lightlda_roundtrip.tt")
    try:
        m.save(path)
        reloaded = topica.LDA.load(path)
        assert np.allclose(m.topic_word, reloaded.topic_word)
    finally:
        if os.path.exists(path):
            os.remove(path)

    theta = m.transform([list(PETS), list(SPACE)])
    assert theta.shape == (2, 2)
    assert np.allclose(theta.sum(axis=1), 1.0)
    # The pet snippet loads on the pet topic, the space snippet on the other.
    pet_topic = int(np.argmax([phi_t[m.vocabulary.index("cat")] for phi_t in m.topic_word]))
    assert int(theta[0].argmax()) == pet_topic
    assert int(theta[1].argmax()) != pet_topic


@pytest.mark.skipif(
    not os.path.exists("examples/poliblog.csv"), reason="poliblog corpus not present"
)
def test_topic_quality_matches_sparse_on_real_corpus():
    rows = list(csv.DictReader(open("examples/poliblog.csv")))
    docs = [r["text"].split() for r in rows]
    corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)

    def mean_cv(sampler):
        m = topica.LDA(num_topics=15, seed=1, sampler=sampler)
        m.fit(corpus, iters=400)
        return float(np.mean(topica.coherence(m, docs, coherence_type="c_v", topn=10)))

    sparse_cv = mean_cv("sparse")
    light_cv = mean_cv("lightlda")
    # The two samplers target the same posterior; coherence should be within a
    # small tolerance, not systematically worse for the alias sampler.
    assert light_cv >= sparse_cv - 0.05
