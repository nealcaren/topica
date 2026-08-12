"""Method-of-composition uncertainty for estimate_effect, the STM/CTM
variational posterior, and spline/interaction design helpers.
"""

import numpy as np
import pytest

from topica import STM, CTM, stm


@pytest.fixture(scope="module")
def treated_model():
    """A 2-topic STM where `treatment` cleanly flips topic prevalence."""
    docs, treat = [], []
    for i in range(300):
        t = i % 2
        docs.append(["threat", "fear", "danger"] * 2 if t else ["calm", "neutral", "ok"] * 2)
        treat.append(float(t))
    X = np.array(treat).reshape(-1, 1)
    m = STM(num_topics=2, seed=1)
    m.fit(docs, X, prevalence_names=["treatment"], iters=60)
    return m, X


class TestVariationalPosterior:
    def test_eta_shapes(self, treated_model):
        m, _ = treated_model
        d = m.doc_topic.shape[0]
        km1 = m.num_topics - 1
        assert m.eta_mean.shape == (d, km1)
        assert m.eta_cov.shape == (d, km1, km1)

    def test_eta_cov_psd(self, treated_model):
        m, _ = treated_model
        # Each per-doc covariance is symmetric positive semi-definite.
        c = m.eta_cov[0]
        assert np.allclose(c, c.T)
        assert np.all(np.linalg.eigvalsh(c) > -1e-8)

    def test_ctm_also_exposes_posterior(self):
        docs = [["a", "b", "c"]] * 30 + [["x", "y", "z"]] * 30
        m = CTM(num_topics=2, seed=1)
        m.fit(docs, iters=30)
        assert m.eta_mean.shape == (60, 1)
        assert m.eta_cov.shape == (60, 1, 1)


class TestPosteriorSamples:
    def test_shape_and_simplex(self, treated_model):
        m, _ = treated_model
        draws = stm.posterior_theta_samples(m, nsims=20, seed=1)
        assert draws.shape == (20, m.doc_topic.shape[0], m.num_topics)
        np.testing.assert_allclose(draws.sum(axis=2), 1.0, atol=1e-9)
        assert np.all(draws >= 0)

    def test_deterministic_for_seed(self, treated_model):
        m, _ = treated_model
        a = stm.posterior_theta_samples(m, nsims=10, seed=7)
        b = stm.posterior_theta_samples(m, nsims=10, seed=7)
        assert np.array_equal(a, b)

    def test_mean_near_point_theta(self, treated_model):
        m, _ = treated_model
        draws = stm.posterior_theta_samples(m, nsims=200, seed=3)
        # The sample mean of θ should be close to the point estimate.
        assert np.allclose(draws.mean(axis=0), m.doc_topic, atol=0.05)


class TestMethodOfComposition:
    def test_pooled_matches_point_coef_but_wider_se(self, treated_model):
        m, X = treated_model
        pt = stm.estimate_effect(m.doc_topic, X, feature_names=["treatment"])
        draws = stm.posterior_theta_samples(m, nsims=30, seed=1)
        moc = stm.estimate_effect(draws, X, feature_names=["treatment"])
        ti = pt[0].feature_names.index("treatment")
        # Point estimate of the coefficient is essentially unchanged...
        assert abs(pt[0].coef[ti] - moc[0].coef[ti]) < 0.05
        # ...but method-of-composition propagates topic uncertainty, so its SE is
        # at least as large (here the point OLS SE collapses toward zero).
        assert moc[0].se[ti] >= pt[0].se[ti]
        assert moc[0].se[ti] > 0

    def test_treatment_significant(self, treated_model):
        m, X = treated_model
        draws = stm.posterior_theta_samples(m, nsims=30, seed=2)
        moc = stm.estimate_effect(draws, X, feature_names=["treatment"])
        ti = moc[0].feature_names.index("treatment")
        # One topic up, one down, both significant under pooled uncertainty.
        zs = [e.z[ti] for e in moc]
        assert max(abs(z) for z in zs) > 1.96

    def test_pooled_shapes(self, treated_model):
        m, X = treated_model
        draws = stm.posterior_theta_samples(m, nsims=15, seed=1)
        moc = stm.estimate_effect(draws, X, feature_names=["treatment"])
        assert len(moc) == m.num_topics
        for e in moc:
            assert e.coef.shape == (2,)  # intercept + treatment
            assert np.all(e.se >= 0)

    def test_bad_ndim_raises(self, treated_model):
        m, X = treated_model
        with pytest.raises(ValueError):
            stm.estimate_effect(np.zeros((2, 2, 2, 2)), X)


class TestUncertaintySelector:
    """#715-#2: R stm's thetaPosterior Global/Local/None, defaulting to Global
    (R estimateEffect's default) for estimate_effect's method-of-composition."""

    def test_none_draws_are_the_point_theta(self, treated_model):
        m, _ = treated_model
        draws = stm.posterior_theta_samples(m, nsims=5, seed=0, uncertainty="none")
        assert draws.shape == (5, m.doc_topic.shape[0], m.num_topics)
        # No topic-model uncertainty: every draw is the point theta.
        for s in range(5):
            np.testing.assert_allclose(draws[s], np.asarray(m.doc_topic), atol=1e-12)

    def test_global_covariance_is_mean_eta_cov(self, treated_model):
        # R's Global shared covariance (Sigma - crossprod(lambda-mu)/N) equals the
        # mean per-doc variational covariance at sigma.prior=0. Verify the draws'
        # empirical within-doc covariance matches mean(eta_cov), shared across docs.
        m, _ = treated_model
        shared = np.asarray(m.eta_cov, dtype=np.float64).mean(axis=0)  # (K-1,K-1)
        draws = stm.posterior_theta_samples(m, nsims=4000, seed=0, uncertainty="global")
        # Map theta back to eta (ref cat 0): eta_k = log(theta_k) - log(theta_ref).
        logit = np.log(draws[..., :-1]) - np.log(draws[..., -1:])  # (S, D, K-1)
        emp = np.cov(logit[:, 0, :], rowvar=False)  # doc 0's draw covariance
        np.testing.assert_allclose(np.atleast_2d(emp), shared, rtol=0.15, atol=0.02)

    def test_se_ordering_none_local_global(self, treated_model):
        m, X = treated_model
        ses = {}
        for unc in ("none", "local", "global"):
            eff = stm.estimate_effect(
                m, X=X, feature_names=["treatment"], nsims=80, seed=0, uncertainty=unc
            )
            ti = eff[0].feature_names.index("treatment")
            ses[unc] = eff[0].se[ti]
        # Global propagates the most uncertainty, none the least.
        assert ses["none"] <= ses["local"] <= ses["global"]

    def test_global_is_the_default(self, treated_model):
        m, X = treated_model
        default = stm.estimate_effect(m, X=X, feature_names=["treatment"], nsims=80, seed=0)
        explicit = stm.estimate_effect(
            m, X=X, feature_names=["treatment"], nsims=80, seed=0, uncertainty="global"
        )
        np.testing.assert_allclose(default[0].se, explicit[0].se)

    def test_invalid_uncertainty_rejected(self, treated_model):
        m, X = treated_model
        with pytest.raises(ValueError, match="uncertainty must be"):
            stm.estimate_effect(m, X=X, nsims=5, uncertainty="bogus")
        with pytest.raises(ValueError, match="uncertainty must be"):
            stm.posterior_theta_samples(m, nsims=5, uncertainty="bogus")


class TestSpline:
    def test_basis_shape_and_names(self):
        x = np.linspace(0, 10, 100)
        basis, names = stm.spline(x, df=4)
        assert basis.shape == (100, 4)
        assert len(names) == 4

    def test_recovers_nonlinear_trend(self):
        # A spline design should fit a quadratic far better than a linear term.
        rng = np.random.default_rng(0)
        x = np.linspace(-3, 3, 200)
        y = x**2 + rng.normal(0, 0.1, size=200)
        # Linear fit R^2
        Xl = np.column_stack([np.ones_like(x), x])
        bl = np.linalg.lstsq(Xl, y, rcond=None)[0]
        r2_lin = 1 - ((y - Xl @ bl) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # Spline fit R^2
        basis, _ = stm.spline(x, df=5)
        Xs = np.column_stack([np.ones_like(x), basis])
        bs = np.linalg.lstsq(Xs, y, rcond=None)[0]
        r2_spl = 1 - ((y - Xs @ bs) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        assert r2_spl > 0.99 and r2_spl > r2_lin + 0.1

    def test_df_too_small_raises(self):
        with pytest.raises(ValueError):
            stm.spline(np.arange(10.0), df=1)


class TestInteraction:
    def test_product_columns(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        prod, names = stm.interaction(a, b)
        np.testing.assert_allclose(prod[:, 0], a * b)
        assert len(names) == 1

    def test_multi_column_interaction(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0], [6.0]])
        prod, names = stm.interaction(a, b)
        assert prod.shape == (2, 2)  # 2 cols of a x 1 col of b
        assert len(names) == 2

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError):
            stm.interaction(np.arange(3.0), np.arange(4.0))
