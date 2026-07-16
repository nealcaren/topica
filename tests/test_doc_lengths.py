"""Fitted Gibbs models retain their own per-document token lengths (issue #32),
so the Dirichlet-approximation path of composition_theta is self-sufficient
without re-threading the Corpus.
"""

import numpy as np
import pytest

import topica
from topica.effects import composition_theta


def _toy_docs(n=40, vocab=12, lo=8, hi=15, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocab)]
    return [list(rng.choice(words, size=int(rng.integers(lo, hi)))) for _ in range(n)]


def _fit_each(corpus, *, keep_draws):
    out = {}
    m = topica.models.LDA(num_topics=4, seed=1)
    m.fit(corpus, iters=120, keep_theta_draws=keep_draws)
    out["LDA"] = m
    m = topica.models.SeededLDA({"a": ["w0", "w1"], "b": ["w5", "w6"]}, residual=2, seed=1)
    m.fit(corpus, iters=120, keep_theta_draws=keep_draws)
    out["SeededLDA"] = m
    m = topica.models.KeyATM({"a": ["w0", "w1"], "b": ["w5", "w6"]}, num_topics=4, seed=1)
    m.fit(corpus, iters=120, keep_theta_draws=keep_draws)
    out["KeyATM"] = m
    return out


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_doc_lengths_matches_corpus(name):
    docs = _toy_docs()
    corpus = topica.Corpus.from_documents(docs)
    model = _fit_each(corpus, keep_draws=False)[name]
    lengths = np.asarray(model.doc_lengths)
    d = np.asarray(model.doc_topic).shape[0]
    assert lengths.shape == (d,), name
    # Raw token counts, in doc_topic row order, matching the Corpus.
    assert np.array_equal(lengths, np.asarray(corpus.doc_lengths)), name


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_composition_self_sufficient_without_corpus(name):
    # With draws disabled, composition_theta falls back to the Dirichlet
    # approximation. It must now work with no corpus= and match the corpus path
    # byte-for-byte (same seed, same recovered N_d).
    docs = _toy_docs()
    corpus = topica.Corpus.from_documents(docs)
    model = _fit_each(corpus, keep_draws=False)[name]
    assert model.theta_draws is None, name

    without = composition_theta(model, nsims=20, seed=7)
    withc = composition_theta(model, corpus=corpus, nsims=20, seed=7)
    assert np.array_equal(without, withc), name
    d, k = np.asarray(model.doc_topic).shape
    assert without.shape == (20, d, k), name


def test_corpus_takes_precedence_over_retained_lengths():
    # Passing corpus= still wins, so an alternate corpus can be used deliberately.
    docs = _toy_docs()
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.LDA(num_topics=4, seed=1)
    m.fit(corpus, iters=120, keep_theta_draws=False)

    # A corpus with deliberately different (here, longer) documents changes the
    # Dirichlet concentration, so the draws differ from the retained-length path.
    longer = [d * 3 for d in docs]
    alt = topica.Corpus.from_documents(longer)
    a = composition_theta(m, corpus=alt, nsims=20, seed=7)
    b = composition_theta(m, nsims=20, seed=7)
    assert not np.array_equal(a, b)


def test_doc_lengths_dynamic_keyatm_in_caller_order():
    # The dynamic model fits on time-sorted docs but reports in caller order;
    # doc_lengths must line up with that (and with doc_topic).
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(12)]
    docs = [list(rng.choice(vocab, size=int(rng.integers(8, 16)))) for _ in range(30)]
    timestamps = list(rng.permutation(np.repeat(np.arange(6), 5)))
    m = topica.models.KeyATM({"a": ["w0", "w1"]}, num_topics=3, seed=1)
    m.fit(docs, iters=120, timestamps=timestamps, num_states=2, keep_theta_draws=False)

    lengths = np.asarray(m.doc_lengths)
    assert np.array_equal(lengths, np.asarray([len(d) for d in docs]))
    # And the fallback composition path works with no corpus.
    out = composition_theta(m, nsims=10, seed=0)
    assert out.shape == (10, len(docs), 3)
