"""The model-neutral prevalence tools: top-level/effects aliases resolve to the
same functions as their old model-namespaced homes, and estimate_effect runs the
method of composition on Gibbs models via dirichlet_theta_samples."""

import numpy as np
import pytest

import topica


def test_aliases_point_to_one_implementation():
    assert topica.estimate_effect is topica.effects.estimate_effect
    assert topica.estimate_effect is topica.stm.estimate_effect
    assert topica.by_strata is topica.effects.by_strata is topica.keyatm.by_strata
    assert topica.top_topics is topica.keyatm.top_topics
    assert topica.posterior_theta_samples is topica.stm.posterior_theta_samples


def _corpus(n=80):
    a = ["econ", "tax", "jobs", "budget"]
    b = ["war", "troop", "iraq", "border"]
    docs = [list(np.tile(a if i % 2 == 0 else b, 3)) for i in range(n)]
    x = np.array([[float(i % 2)] for i in range(n)])
    return docs, x


def test_dirichlet_theta_samples_shape_and_determinism():
    docs, _ = _corpus()
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=150)
    lengths = np.array([len(d) for d in docs])
    a = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=15, seed=0)
    b = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=15, seed=0)
    assert a.shape == (15, len(docs), 2)
    assert np.allclose(a.sum(axis=2), 1.0)
    assert np.array_equal(a, b)  # deterministic for a fixed seed
    # Longer documents give tighter (lower-variance) draws.
    short = topica.dirichlet_theta_samples(m.doc_topic, np.full(len(docs), 5.0), nsims=40, seed=0)
    long = topica.dirichlet_theta_samples(m.doc_topic, np.full(len(docs), 500.0), nsims=40, seed=0)
    assert long.var(axis=0).mean() < short.var(axis=0).mean()


def test_estimate_effect_on_gibbs_draws_and_point():
    docs, x = _corpus()
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    lengths = np.array([len(d) for d in docs])
    draws = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=20, seed=0)

    moc = topica.estimate_effect(draws, x, feature_names=["grp"])
    ols = topica.estimate_effect(m.doc_topic, x, feature_names=["grp"])
    assert len(moc) == 2 and len(ols) == 2
    # Method of composition propagates topic uncertainty, so its standard errors
    # are at least as large as the point-estimate OLS ones.
    moc_se = moc[0].as_dict()["grp"]["se"]
    ols_se = ols[0].as_dict()["grp"]["se"]
    assert moc_se >= ols_se - 1e-9


def test_prevalence_ci_is_model_neutral():
    # The draws-based credible-band primitive that time_prevalence_ci wraps works
    # on any model, not just the dynamic keyATM. Here: a plain LDA grouped by a
    # binary covariate, off retained MCMC draws.
    docs, x = _corpus()
    groups = ["even" if xi[0] == 0 else "odd" for xi in x]
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200, keep_theta_draws=True, num_theta_draws=20)

    out = topica.prevalence_ci(m, groups, ci=0.9)
    assert out["labels"] == ["even", "odd"]
    for key in ("mean", "ci_low", "ci_high", "sd"):
        assert out[key].shape == (2, 2)
    assert np.all(out["ci_low"] <= out["mean"] + 1e-9)
    assert np.all(out["mean"] <= out["ci_high"] + 1e-9)
    # Normalized prevalence rows sum to 1.
    assert np.allclose(out["mean"].sum(axis=1), 1.0)
    # Explicit label ordering is honored.
    flipped = topica.prevalence_ci(m, groups, labels=["odd", "even"])
    assert flipped["labels"] == ["odd", "even"]
    assert np.allclose(flipped["mean"][0], out["mean"][1])


def test_prevalence_ci_falls_back_without_retained_draws():
    # No retained draws: it still works via the Dirichlet approximation, which
    # needs the corpus for document lengths.
    docs, x = _corpus()
    groups = [int(xi[0]) for xi in x]
    m = topica.models.LDA(num_topics=2, seed=2)
    m.fit(docs, iters=150)  # keep_theta_draws defaults off here
    out = topica.prevalence_ci(m, groups, corpus=docs, nsims=15, seed=0)
    assert out["mean"].shape == (2, 2)
    with pytest.raises(ValueError, match="one label per document"):
        topica.prevalence_ci(m, groups[:-1], corpus=docs, nsims=10)


def test_dirichlet_validation():
    with pytest.raises(ValueError):
        topica.dirichlet_theta_samples(np.zeros((4, 3)), np.zeros(5))  # mismatched lengths
    with pytest.raises(ValueError):
        topica.dirichlet_theta_samples(np.zeros(4), np.zeros(4))  # theta not 2-D
