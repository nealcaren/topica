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
from topica import LDA


def test_autocorrelation_of_iid_is_flat():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    acf = topica.autocorrelation(x, max_lag=10)
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
    acf = topica.autocorrelation(x, max_lag=3)
    assert acf[1] == pytest.approx(phi, abs=0.03)
    assert acf[2] == pytest.approx(phi**2, abs=0.05)


def test_iac_time_iid_near_one():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    tau = topica.integrated_autocorr_time(x)
    assert 0.8 < tau < 1.3


def test_iac_time_ar1_matches_theory():
    rng = np.random.default_rng(3)
    phi = 0.8
    x = np.empty(50000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    tau = topica.integrated_autocorr_time(x)
    theory = (1 + phi) / (1 - phi)  # = 9
    assert tau == pytest.approx(theory, rel=0.15)


def test_ess_iid_close_to_n():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    assert topica.effective_sample_size(x) == pytest.approx(3000, rel=0.15)


def test_ess_never_exceeds_n():
    # tau is floored at 1, so ESS <= N for any positively-correlated chain
    rng = np.random.default_rng(5)
    phi = 0.9
    x = np.empty(8000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    assert topica.effective_sample_size(x) <= x.size


def test_constant_chain_has_zero_ess():
    const = np.ones(200)
    assert np.isinf(topica.integrated_autocorr_time(const))
    assert topica.effective_sample_size(const) == 0.0


def test_ess_2d_is_columnwise():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2000, 4))
    ess = topica.effective_sample_size(x)
    assert ess.shape == (4,)
    assert np.all(ess == pytest.approx(2000, rel=0.2))


def test_autocorrelation_short_and_constant_inputs():
    assert topica.autocorrelation([1.0]).tolist() == [1.0]
    acf = topica.autocorrelation(np.ones(10), max_lag=3)
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
    d = topica.mcmc_diagnostics(fitted_lda)
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
        topica.mcmc_diagnostics(model)


def test_mcmc_diagnostics_warns_for_variational_model():
    docs = [["a", "b", "c"], ["d", "e", "f"]] * 20
    corpus = topica.Corpus.from_documents(docs)
    model = topica.CTM(num_topics=2, seed=0)
    model.fit(corpus, iters=15)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            topica.mcmc_diagnostics(model)
    assert any("not Gibbs" in str(w.message) for w in caught)


def test_mcmc_diagnostics_warn_false_is_silent(fitted_lda):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        topica.mcmc_diagnostics(fitted_lda, warn=False)


# ---------------------------------------------------------------------------
# Multi-chain diagnostics: R-hat (Gelman-Rubin) and cross-chain ESS.
# ---------------------------------------------------------------------------


def test_rhat_of_well_mixed_chains_is_near_one():
    rng = np.random.default_rng(10)
    chains = rng.standard_normal((4, 4000))
    assert topica.rhat(chains) == pytest.approx(1.0, abs=0.01)


def test_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(11)
    # four chains centered far apart never mixed to a common target
    chains = rng.standard_normal((4, 4000)) + np.array([[0.0], [5.0], [10.0], [15.0]])
    assert topica.rhat(chains) > 1.2


def test_rhat_of_identical_constant_chains_is_one():
    assert topica.rhat(np.ones((3, 200))) == pytest.approx(1.0)


def test_rhat_split_detects_within_chain_drift():
    # two chains, each a slow linear ramp: no between-chain spread but each half
    # disagrees with the other, which split-R-hat is designed to catch.
    ramp = np.linspace(0.0, 10.0, 2000)
    chains = np.vstack([ramp, ramp])
    assert topica.rhat(chains, split=True) > 1.1
    # without splitting the drift is invisible (chain means are identical)
    assert topica.rhat(chains, split=False) == pytest.approx(1.0, abs=0.05)


def test_rhat_accepts_unequal_length_chains():
    rng = np.random.default_rng(12)
    a = rng.standard_normal(3000)
    b = rng.standard_normal(2000)
    r = topica.rhat([a, b])  # truncated to the shorter
    assert r == pytest.approx(1.0, abs=0.03)


def test_rhat_requires_two_chains():
    with pytest.raises(ValueError, match="two chains"):
        topica.rhat(np.zeros((1, 100)))


def test_ndtri_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    from topica.mcmc import _ndtri

    p = np.array([1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 0.9999])
    assert np.allclose(_ndtri(p), scipy_stats.norm.ppf(p), atol=1e-6)


@pytest.fixture(scope="module")
def lda_chains():
    rng = np.random.default_rng(0)
    vocabs = [
        ["cat", "dog", "pet", "fur", "paw"],
        ["stock", "bond", "market", "trade", "bank"],
        ["star", "planet", "orbit", "moon", "galaxy"],
    ]
    docs = []
    for _ in range(120):
        topic = rng.integers(0, 3)
        docs.append(list(rng.choice(vocabs[topic], size=25)))
    corpus = topica.Corpus.from_documents(docs)
    chains = []
    for seed in (1, 2, 3, 4):
        model = LDA(num_topics=3, seed=seed)
        model.fit(corpus, iters=400, num_theta_draws=80)
        chains.append(model)
    return chains


def test_multichain_diagnostics_on_lda(lda_chains):
    d = topica.multichain_diagnostics(lda_chains)
    assert d.model == "LDA"
    assert d.inference == "gibbs"
    assert d.n_chains == 4
    # log-likelihood R-hat is computed on the permutation-invariant trace
    assert d.loglik_rhat is not None and d.loglik_rhat >= 1.0
    assert d.loglik_ess is not None
    # per-topic R-hat over the three aligned topics
    assert d.topic_rhat.shape == (3,)
    assert d.topic_ess.shape == (3,)
    assert d.topic_alignment.shape == (3,)
    # a clean three-topic corpus aligns its topics across seeds
    assert d.topic_alignment.min() > 0.5
    assert "LDA" in d.summary()
    assert isinstance(d.converged, bool)


def test_multichain_diagnostics_requires_two_chains(lda_chains):
    with pytest.raises(ValueError, match="two chains"):
        topica.multichain_diagnostics(lda_chains[:1])


def test_multichain_diagnostics_without_theta_draws_warns():
    docs = [["a", "b", "c"], ["d", "e", "f"]] * 20
    corpus = topica.Corpus.from_documents(docs)
    chains = []
    for seed in (1, 2):
        model = LDA(num_topics=2, seed=seed)
        model.fit(corpus, iters=80, keep_theta_draws=False)
        chains.append(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = topica.multichain_diagnostics(chains)
    assert any("theta_draws" in str(w.message) for w in caught)
    assert d.topic_rhat is None
    # the log-likelihood R-hat still lands from the retained trace
    assert d.loglik_rhat is not None


def test_multichain_diagnostics_warns_for_variational_models():
    docs = [["a", "b", "c"], ["d", "e", "f"]] * 20
    corpus = topica.Corpus.from_documents(docs)
    chains = []
    for seed in (1, 2):
        model = topica.CTM(num_topics=2, seed=seed)
        model.fit(corpus, iters=15)
        chains.append(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        topica.multichain_diagnostics(chains)
    assert any("not Gibbs" in str(w.message) for w in caught)
