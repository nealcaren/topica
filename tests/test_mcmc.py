"""Single-chain MCMC diagnostics (topica.mcmc).

The estimators are validated against their defining properties: an i.i.d. chain
has an integrated autocorrelation time near 1 (ESS ~ N), an AR(1) chain recovers
its theoretical tau = (1+phi)/(1-phi), and a constant chain carries no
information (tau = inf, ESS = 0). The model-facing wrapper is checked against a
real fitted Gibbs model and against the guards for the non-Gibbs / no-draws
cases.
"""

import warnings

import numpy as np
import pytest

import topica
from topica.models import LDA


def test_autocorrelation_of_iid_is_flat():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    acf = topica.diagnostics.autocorrelation(x, max_lag=10)
    assert acf[0] == pytest.approx(1.0)
    # lag>=1 autocorrelation of white noise is ~0 (1/sqrt(N) noise band)
    assert np.all(np.abs(acf[1:]) < 0.1)


def test_autocorrelation_recovers_ar1_coefficient():
    rng = np.random.default_rng(1)
    phi = 0.8
    x = np.empty(20000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    acf = topica.diagnostics.autocorrelation(x, max_lag=3)
    assert acf[1] == pytest.approx(phi, abs=0.03)
    assert acf[2] == pytest.approx(phi**2, abs=0.05)


def test_iac_time_iid_near_one():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    tau = topica.diagnostics.integrated_autocorr_time(x)
    assert 0.8 < tau < 1.3


def test_iac_time_ar1_matches_theory():
    rng = np.random.default_rng(3)
    phi = 0.8
    x = np.empty(50000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    tau = topica.diagnostics.integrated_autocorr_time(x)
    theory = (1 + phi) / (1 - phi)  # = 9
    assert tau == pytest.approx(theory, rel=0.15)


def test_ess_iid_close_to_n():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    assert topica.diagnostics.effective_sample_size(x) == pytest.approx(3000, rel=0.15)


def test_ess_never_exceeds_n():
    # tau is floored at 1, so ESS <= N for any positively-correlated chain
    rng = np.random.default_rng(5)
    phi = 0.9
    x = np.empty(8000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    assert topica.diagnostics.effective_sample_size(x) <= x.size


def test_constant_chain_has_zero_ess():
    const = np.ones(200)
    assert np.isinf(topica.diagnostics.integrated_autocorr_time(const))
    assert topica.diagnostics.effective_sample_size(const) == 0.0


def test_ess_2d_is_columnwise():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2000, 4))
    ess = topica.diagnostics.effective_sample_size(x)
    assert ess.shape == (4,)
    assert np.all(ess == pytest.approx(2000, rel=0.2))


def test_autocorrelation_short_and_constant_inputs():
    assert topica.diagnostics.autocorrelation([1.0]).tolist() == [1.0]
    acf = topica.diagnostics.autocorrelation(np.ones(10), max_lag=3)
    assert acf[0] == 1.0
    assert np.all(acf[1:] == 0.0)


@pytest.fixture(scope="module")
def fitted_lda():
    docs = [
        ["apple", "banana", "apple", "fruit"],
        ["dog", "cat", "pet", "dog"],
        ["apple", "fruit", "banana", "juice"],
        ["cat", "pet", "kitten", "dog"],
    ] * 20
    corpus = topica.Corpus.from_documents(docs)
    model = LDA(num_topics=3, seed=0)
    model.fit(corpus, iters=300, num_theta_draws=60)
    return model


def test_mcmc_diagnostics_on_fitted_model(fitted_lda):
    d = topica.diagnostics.mcmc_diagnostics(fitted_lda)
    assert d.model == "LDA"
    assert d.inference == "gibbs"
    assert d.n_draws == 60
    assert d.theta_ess.shape == (80, 3)
    # ESS is bounded by the number of retained draws
    assert d.theta_ess_min > 0
    assert d.theta_ess.max() <= d.n_draws + 1e-6
    assert d.loglik_tau is not None and d.loglik_tau >= 1.0
    assert d.loglik_ess is not None
    assert "LDA" in d.summary()


def test_mcmc_diagnostics_requires_theta_draws():
    docs = [["a", "b", "c"], ["d", "e", "f"]] * 20
    corpus = topica.Corpus.from_documents(docs)
    model = LDA(num_topics=2, seed=0)
    model.fit(corpus, iters=40, keep_theta_draws=False)
    with pytest.raises(ValueError, match="theta_draws"):
        topica.diagnostics.mcmc_diagnostics(model)


def test_mcmc_diagnostics_warns_for_variational_model():
    docs = [["a", "b", "c"], ["d", "e", "f"]] * 20
    corpus = topica.Corpus.from_documents(docs)
    model = topica.models.CTM(num_topics=2, seed=0)
    model.fit(corpus, iters=15)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            topica.diagnostics.mcmc_diagnostics(model)
    assert any("not Gibbs" in str(w.message) for w in caught)


def test_mcmc_diagnostics_warn_false_is_silent(fitted_lda):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        topica.diagnostics.mcmc_diagnostics(fitted_lda, warn=False)
