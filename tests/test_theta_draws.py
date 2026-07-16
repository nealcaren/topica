"""Retained MCMC theta draws for the Gibbs family (issue #31).

LDA, KeyATM, and SeededLDA snapshot thinned, post-burn-in theta during the fit
loop and expose them as ``model.theta_draws`` (num_draws, num_docs, num_topics).
``composition_theta`` prefers these real cross-sweep posterior samples over the
within-document Dirichlet approximation, falling back cleanly when a model was
fit with ``keep_theta_draws=False``.
"""

import numpy as np
import pytest

import topica
from topica.effects import composition_theta, dirichlet_theta_samples


def _toy_docs(n=40, vocab=12, lo=8, hi=15, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocab)]
    return [list(rng.choice(words, size=int(rng.integers(lo, hi)))) for _ in range(n)]


# A fitted model of each instrumented Gibbs family on a shared toy corpus.
def _fit_each(docs, *, keep=True, num_draws=25):
    out = {}
    m = topica.models.LDA(num_topics=4, seed=1)
    m.fit(docs, iters=120, keep_theta_draws=keep, num_theta_draws=num_draws)
    out["LDA"] = m
    m = topica.models.SeededLDA({"a": ["w0", "w1"], "b": ["w5", "w6"]}, residual=2, seed=1)
    m.fit(docs, iters=120, keep_theta_draws=keep, num_theta_draws=num_draws)
    out["SeededLDA"] = m
    m = topica.models.KeyATM({"a": ["w0", "w1"], "b": ["w5", "w6"]}, num_topics=4, seed=1)
    m.fit(docs, iters=120, keep_theta_draws=keep, num_theta_draws=num_draws)
    out["KeyATM"] = m
    return out


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_theta_draws_shape_and_simplex(name):
    docs = _toy_docs()
    model = _fit_each(docs, num_draws=20)[name]
    td = np.asarray(model.theta_draws)
    d, k = np.asarray(model.doc_topic).shape
    assert td.shape == (20, d, k), name
    assert td.dtype == np.float32, name
    # Each draw's rows are probability simplices.
    assert np.allclose(td.sum(axis=2), 1.0, atol=1e-4), name
    assert (td >= 0).all(), name
    # Genuine cross-sweep variation, not a repeated point estimate.
    assert td.std(axis=0).max() > 0.0, name


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_keep_theta_draws_false_disables(name):
    docs = _toy_docs()
    model = _fit_each(docs, keep=False)[name]
    assert model.theta_draws is None, name


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_composition_theta_uses_draws_without_corpus(name):
    # When draws are present, composition_theta needs no corpus and returns the
    # draws resampled to nsims.
    docs = _toy_docs()
    model = _fit_each(docs, num_draws=30)[name]
    out = composition_theta(model, nsims=12)
    d, k = np.asarray(model.doc_topic).shape
    assert out.shape == (12, d, k), name
    assert np.allclose(out.sum(axis=2), 1.0, atol=1e-4), name


@pytest.mark.parametrize("name", ["LDA", "SeededLDA", "KeyATM"])
def test_fallback_identical_to_dirichlet_when_absent(name):
    # keep_theta_draws=False -> composition_theta reproduces the Dirichlet
    # approximation byte-for-byte (same seed, same lengths).
    docs = _toy_docs()
    model = _fit_each(docs, keep=False)[name]
    lengths = np.array([len(d) for d in docs], dtype=float)
    fallback = composition_theta(model, corpus=docs, nsims=15, seed=3)
    ref = dirichlet_theta_samples(
        np.asarray(model.doc_topic, dtype=float), lengths, nsims=15, seed=3
    )
    assert np.array_equal(fallback, ref), name


def test_real_draws_capture_cross_sweep_variance():
    # In a poorly-identified regime (longer documents, no topic structure) the
    # within-document Dirichlet noise (~1/N_d) is small, so the real MCMC draws
    # carry strictly more cross-draw variance than the approximation -- the gap
    # issue #31 is closing. (For short documents the ordering reverses, because
    # the 1/N_d term dominates; that is expected, not a regression.)
    rng = np.random.default_rng(2)
    vocab = [f"w{i}" for i in range(24)]
    docs = [list(rng.choice(vocab, size=int(rng.integers(160, 240)))) for _ in range(40)]
    m = topica.models.LDA(num_topics=4, seed=2)
    m.fit(docs, iters=300, num_theta_draws=40)

    real = np.asarray(m.theta_draws, dtype=float)
    lengths = np.array([len(d) for d in docs], dtype=float)
    approx = dirichlet_theta_samples(
        np.asarray(m.doc_topic, dtype=float), lengths, nsims=real.shape[0], seed=0
    )
    assert real.var(axis=0).mean() > approx.var(axis=0).mean()


def test_real_draws_reflect_identifiability():
    # Real draws track model confidence: a cleanly separated corpus pins theta,
    # so cross-draw variance is far smaller than the length-only approximation
    # (which is blind to identifiability). This is the behavior change that lets
    # composition SEs fall below the approximation for well-identified data.
    rng = np.random.default_rng(0)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    docs = []
    for _ in range(80):
        block = a if rng.random() < 0.5 else b
        docs.append(list(rng.choice(block, size=40)))
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=250, num_theta_draws=40)

    real = np.asarray(m.theta_draws, dtype=float)
    lengths = np.array([len(d) for d in docs], dtype=float)
    approx = dirichlet_theta_samples(
        np.asarray(m.doc_topic, dtype=float), lengths, nsims=real.shape[0], seed=0
    )
    assert real.var(axis=0).mean() < approx.var(axis=0).mean()


def test_num_theta_draws_bounds_count():
    docs = _toy_docs()
    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(docs, iters=200, num_theta_draws=10)
    assert np.asarray(m.theta_draws).shape[0] == 10


def test_keyatm_dynamic_draws_match_doc_order():
    # The dynamic model fits on time-sorted documents; its draws must be unsorted
    # back to the caller's document order, so the draw mean tracks doc_topic.
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(12)]
    docs = [list(rng.choice(vocab, size=12)) for _ in range(30)]
    timestamps = list(rng.permutation(np.repeat(np.arange(6), 5)))
    m = topica.models.KeyATM({"a": ["w0", "w1"]}, num_topics=3, seed=1)
    m.fit(docs, iters=120, timestamps=timestamps, num_states=2)

    td = np.asarray(m.theta_draws, dtype=float)
    d, k = np.asarray(m.doc_topic).shape
    assert td.shape == (25, d, k)
    assert np.allclose(td.sum(axis=2), 1.0, atol=1e-4)
    # Per-document draw mean is close to the reported doc_topic (same ordering).
    assert np.abs(td.mean(axis=0) - np.asarray(m.doc_topic)).mean() < 0.1
