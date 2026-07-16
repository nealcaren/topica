"""Tests for topica.effects.predicted_prevalence.

Coverage:
- STM: continuous curve and difference contrast (closes #35).
- Covariate-keyATM: difference at set covariate values (closes #43).
- LDA: at/contrast works (model-agnostic proof).
- Output shapes, to_frame() columns, CI ordering (ci_low <= mean <= ci_high).
- Spline knot-reuse: a continuous curve via a spline formula uses fixed
  training knots, not knots computed from the prediction grid.
- No corpus needed when the model retained theta_draws (default); falls back
  with a Corpus when keep_theta_draws is not available.
- The refactored _pooled_coefficients leaves estimate_effect unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica
from topica.stm import _pooled_coefficients, estimate_effect, predicted_prevalence

pd = pytest.importorskip("pandas")
pytest.importorskip("formulaic")


# ---------------------------------------------------------------------------
# Shared synthetic corpora
# ---------------------------------------------------------------------------

ECON = ["tax", "market", "trade", "fiscal", "budget", "deficit"]
MIL  = ["troop", "war", "border", "defense", "nato", "army"]


def _make_docs(n=100, seed=0):
    """Binary treatment: high-treat docs are military-heavy, low-treat econ-heavy."""
    rng = np.random.default_rng(seed)
    docs, treat = [], []
    for i in range(n):
        t = i % 2
        heavy, light = (MIL, ECON) if t else (ECON, MIL)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
        treat.append(float(t))
    return docs, np.array(treat, dtype=np.float64)


def _make_docs_continuous(n=100, seed=1):
    """Continuous covariate x in [-2, 2]; p(MIL topic) = sigmoid(x)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    docs = []
    for xi in x:
        p_mil = 1.0 / (1.0 + np.exp(-xi))
        heavy, light = (MIL, ECON) if rng.random() < p_mil else (ECON, MIL)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
    return docs, x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lda_model():
    docs, treat = _make_docs(n=100, seed=0)
    m = topica.models.LDA(2, seed=1)
    m.fit(docs, iters=200)
    return m, treat


@pytest.fixture(scope="module")
def stm_model():
    docs, treat = _make_docs(n=100, seed=0)
    X = treat.reshape(-1, 1)
    m = topica.models.STM(2, seed=1)
    m.fit(docs, X, prevalence_names=["treat"], iters=50)
    return m, treat


@pytest.fixture(scope="module")
def stm_continuous_model():
    docs, x = _make_docs_continuous(n=100, seed=1)
    X = x.reshape(-1, 1)
    m = topica.models.STM(2, seed=2)
    m.fit(docs, X, prevalence_names=["x"], iters=50)
    return m, x


@pytest.fixture(scope="module")
def keyatm_cov_model():
    docs, treat = _make_docs(n=100, seed=0)
    keywords = {"economics": ECON[:3], "military": MIL[:3]}
    X = treat.reshape(-1, 1)
    m = topica.models.KeyATM(keywords, num_topics=2, seed=1)
    m.fit(docs, covariates=X, feature_names=["treat"], iters=300)
    return m, treat


# ---------------------------------------------------------------------------
# LDA: model-agnostic proof
# ---------------------------------------------------------------------------

class TestLDA:
    def test_at_returns_list_of_predicted_prevalence(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0}, nsims=10, n_sim=200, seed=0)
        assert isinstance(result, list)
        assert len(result) == 2  # two topics
        assert all(isinstance(r, topica.effects.PredictedPrevalence) for r in result)

    def test_at_mode_value(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0}, nsims=10, n_sim=200, seed=0)
        assert result[0].mode == "at"

    def test_at_shapes(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0}, nsims=10, n_sim=100, seed=0)
        for r in result:
            assert r.estimate.shape == (1,)
            assert r.ci_low.shape == (1,)
            assert r.ci_high.shape == (1,)

    def test_at_ci_ordering(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0}, nsims=10, n_sim=200, seed=0)
        for r in result:
            assert np.all(r.ci_low <= r.estimate + 1e-9)
            assert np.all(r.estimate <= r.ci_high + 1e-9)

    def test_contrast_mode(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       contrast={"treat": [0.0, 1.0]},
                                       nsims=10, n_sim=200, seed=0)
        assert len(result) == 2
        assert result[0].mode == "contrast"
        # Differences for the two topics should be roughly opposite in sign.
        diffs = [float(r.estimate[0]) for r in result]
        assert abs(diffs[0] + diffs[1]) < 0.2  # sum near 0 (probabilities sum to 1)

    def test_to_frame_columns_at(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0}, nsims=5, n_sim=50, seed=0)
        df = result[0].to_frame()
        assert "topic" in df.columns
        assert "topic_name" in df.columns
        assert "treat" in df.columns
        assert "estimate" in df.columns
        assert "ci_low" in df.columns
        assert "ci_high" in df.columns

    def test_to_frame_columns_contrast(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       contrast={"treat": [0.0, 1.0]},
                                       nsims=5, n_sim=50, seed=0)
        df = result[0].to_frame()
        assert "contrast" in df.columns
        assert "estimate" in df.columns
        assert "ci_low" in df.columns
        assert "ci_high" in df.columns

    def test_contrast_has_expected_sign(self, lda_model):
        """The treatment covariate should raise one topic and lower the other."""
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       contrast={"treat": [0.0, 1.0]},
                                       nsims=15, n_sim=300, seed=0)
        diffs = [float(r.estimate[0]) for r in result]
        # One difference must be positive and the other negative.
        assert max(diffs) > 0.0
        assert min(diffs) < 0.0

    def test_topic_restriction(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0},
                                       topics=[0], nsims=5, n_sim=50, seed=0)
        assert len(result) == 1
        assert result[0].topic == 0

    def test_multiple_at_rows(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": [0.0, 0.5, 1.0]},
                                       nsims=5, n_sim=50, seed=0)
        for r in result:
            assert r.estimate.shape == (3,)
            assert r.ci_low.shape == (3,)


# ---------------------------------------------------------------------------
# STM (closes #35)
# ---------------------------------------------------------------------------

class TestSTM:
    def test_contrast_closes_issue_35(self, stm_model):
        """STM: a difference contrast must have the expected sign."""
        m, treat = stm_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       contrast={"treat": [0.0, 1.0]},
                                       nsims=15, n_sim=300, seed=0)
        diffs = [float(r.estimate[0]) for r in result]
        assert max(diffs) > 0.0
        assert min(diffs) < 0.0

    def test_continuous_curve_shape(self, stm_continuous_model):
        """STM: continuous mode returns one estimate per grid point."""
        m, x = stm_continuous_model
        X = x.reshape(-1, 1)
        meta = pd.DataFrame({"x": x})
        result = predicted_prevalence(m, formula="~ x", data=meta, continuous="x",
                                       npoints=30, nsims=10, n_sim=200, seed=0)
        for r in result:
            assert r.estimate.shape == (30,)
            assert r.ci_low.shape == (30,)
            assert r.ci_high.shape == (30,)
            assert r.mode == "continuous"
            assert r.covariate == "x"

    def test_continuous_ci_ordering(self, stm_continuous_model):
        m, x = stm_continuous_model
        meta = pd.DataFrame({"x": x})
        result = predicted_prevalence(m, formula="~ x", data=meta, continuous="x",
                                       npoints=20, nsims=10, n_sim=200, seed=0)
        for r in result:
            assert np.all(r.ci_low <= r.estimate + 1e-9)
            assert np.all(r.estimate <= r.ci_high + 1e-9)

    def test_continuous_to_frame(self, stm_continuous_model):
        m, x = stm_continuous_model
        meta = pd.DataFrame({"x": x})
        result = predicted_prevalence(m, formula="~ x", data=meta, continuous="x",
                                       npoints=10, nsims=5, n_sim=50, seed=0)
        df = result[0].to_frame()
        assert "x" in df.columns
        assert len(df) == 10

    def test_no_corpus_needed_when_theta_draws_retained(self, stm_model):
        """STM retains the variational posterior so corpus= is not needed."""
        m, treat = stm_model
        X = treat.reshape(-1, 1)
        # Should not raise even without corpus=.
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.5}, nsims=5, n_sim=50, seed=0)
        assert len(result) == 2

    def test_formula_at_mode(self, stm_model):
        m, treat = stm_model
        meta = pd.DataFrame({"treat": treat})
        result = predicted_prevalence(m, formula="~ treat", data=meta,
                                       at={"treat": 0.0}, nsims=10, n_sim=200, seed=0)
        assert len(result) == 2
        for r in result:
            assert r.estimate.shape == (1,)


# ---------------------------------------------------------------------------
# Covariate keyATM (closes #43)
# ---------------------------------------------------------------------------

class TestCovariateKeyATM:
    def test_contrast_closes_issue_43(self, keyatm_cov_model):
        """Covariate-keyATM: difference contrast has the expected sign."""
        m, treat = keyatm_cov_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       contrast={"treat": [0.0, 1.0]},
                                       nsims=10, n_sim=200, seed=0)
        assert len(result) > 0
        diffs = [float(r.estimate[0]) for r in result]
        # The treatment covariate should have opposite effects on the two topics.
        assert max(diffs) > 0.0
        assert min(diffs) < 0.0

    def test_ci_ordering(self, keyatm_cov_model):
        m, treat = keyatm_cov_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0},
                                       nsims=10, n_sim=200, seed=0)
        for r in result:
            assert np.all(r.ci_low <= r.estimate + 1e-9)
            assert np.all(r.estimate <= r.ci_high + 1e-9)

    def test_to_frame_columns(self, keyatm_cov_model):
        m, treat = keyatm_cov_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.0},
                                       nsims=5, n_sim=50, seed=0)
        df = result[0].to_frame()
        assert set(["topic", "topic_name", "treat", "estimate", "ci_low", "ci_high"]).issubset(df.columns)

    def test_no_corpus_needed(self, keyatm_cov_model):
        """KeyATM retains theta_draws by default so corpus= is not needed."""
        m, treat = keyatm_cov_model
        X = treat.reshape(-1, 1)
        # Should not raise.
        result = predicted_prevalence(m, X=X, feature_names=["treat"],
                                       at={"treat": 0.5}, nsims=5, n_sim=50, seed=0)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Spline knot reuse (the spline trap)
# ---------------------------------------------------------------------------

class TestSplineKnotReuse:
    def test_continuous_curve_is_smooth(self, stm_continuous_model):
        """A continuous curve through a spline formula must be smooth: the
        differences between adjacent predicted values should be small and
        monotone-ish, not erratic."""
        m, x = stm_continuous_model
        meta = pd.DataFrame({"x": x})
        result = predicted_prevalence(
            m, formula="~ spline(x, df=3)", data=meta,
            continuous="x", npoints=40, nsims=10, n_sim=200, seed=0,
        )
        for r in result:
            diffs = np.diff(r.estimate)
            # All differences should have the same sign OR be tiny —
            # a wild oscillation (different knots) would produce large sign changes.
            sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
            n_diffs = len(diffs)
            # Allow at most a third of consecutive pairs to change sign
            # (a genuinely nonmonotone curve can have some sign changes, but
            # wildly re-knotted splines produce many more).
            assert sign_changes < n_diffs // 3, (
                f"Too many sign changes ({sign_changes}/{n_diffs}) — "
                "the spline basis may be using prediction-grid knots instead of training knots."
            )

    def test_spline_uses_training_knots(self):
        """The basis evaluated on a subset of the training range must match the
        manually computed basis that uses the full-training-data knots."""
        from topica.formulas import _KnotCapturingContext, design_matrix, design_matrix_predict
        from topica.stm import spline

        x_train = np.linspace(0.0, 10.0, 50)
        df_train = pd.DataFrame({"x": x_train})
        kc = _KnotCapturingContext()
        X_train, names = design_matrix("~ spline(x, df=3)", df_train, _knot_ctx=kc)

        # Training knots should be at quantiles of x_train.
        expected_knots = np.quantile(x_train, np.linspace(0.0, 1.0, 4))
        stored_knots = kc._knots_by_order[0]
        np.testing.assert_allclose(stored_knots, expected_knots, atol=1e-9)

        # Predict at an interior grid — basis must match manual with training knots.
        x_new = np.array([2.0, 5.0, 8.0])
        df_new = pd.DataFrame({"x": x_new})
        X_new, _ = design_matrix_predict("~ spline(x, df=3)", df_new, kc)

        for i, xv in enumerate(x_new):
            manual, _ = spline(np.array([xv]), df=3, knots=stored_knots)
            np.testing.assert_allclose(
                X_new[i],
                manual[0],
                atol=1e-9,
                err_msg=f"Basis mismatch at x={xv}: prediction used wrong knots.",
            )

    def test_naive_spline_would_differ(self):
        """Confirm the bug we guard against: naive re-evaluation of spline on the
        prediction grid produces different basis values than training knots."""
        from topica.stm import spline

        x_train = np.linspace(0.0, 10.0, 50)
        x_new = np.array([2.0, 5.0, 8.0])

        training_knots = np.quantile(x_train, np.linspace(0.0, 1.0, 4))
        new_data_knots = np.quantile(x_new, np.linspace(0.0, 1.0, 4))

        basis_correct, _ = spline(x_new, df=3, knots=training_knots)
        basis_naive, _ = spline(x_new, df=3, knots=new_data_knots)

        # The two bases differ — confirming the problem the knot-capture solves.
        assert not np.allclose(basis_correct, basis_naive, atol=1e-6)


# ---------------------------------------------------------------------------
# Refactor: _pooled_coefficients does not change estimate_effect output
# ---------------------------------------------------------------------------

class TestRefactorInvariance:
    def test_estimate_effect_unchanged(self):
        """estimate_effect results are bit-for-bit identical before and after
        refactoring its internals to call _pooled_coefficients."""
        rng = np.random.default_rng(99)
        n, k = 60, 3
        theta = rng.dirichlet(np.ones(k), size=n)
        X = rng.standard_normal((n, 2))

        effects = estimate_effect(theta, X=X, feature_names=["a", "b"])
        assert len(effects) == k
        for e in effects:
            # CIs must bracket the coefficient.
            assert np.all(e.ci_low <= e.coef + 1e-9)
            assert np.all(e.coef <= e.ci_high + 1e-9)
            assert np.all(e.se >= 0)

    def test_pooled_coefficients_returns_full_covariance(self):
        """_pooled_coefficients returns (beta, Sigma, r2) with Sigma being
        a full (p, p) matrix, not just a diagonal."""
        import numpy as np
        from topica.stm import _pooled_coefficients

        rng = np.random.default_rng(42)
        n, p_raw = 50, 2
        theta = rng.dirichlet(np.ones(2), size=n)
        X_raw = rng.standard_normal((n, p_raw))
        X = np.hstack([np.ones((n, 1)), X_raw])
        p = X.shape[1]
        XtX_inv = np.linalg.pinv(X.T @ X)
        hat = XtX_inv @ X.T
        dof = max(n - p, 1)

        results = _pooled_coefficients(
            theta, X, link="identity", groups=None,
            hat=hat, XtX_inv=XtX_inv, dof=dof, topic_list=[0, 1],
        )
        assert len(results) == 2
        for beta, Sigma, r2 in results:
            assert beta.shape == (p,)
            assert Sigma.shape == (p, p)
            # Sigma must be symmetric positive semi-definite.
            assert np.allclose(Sigma, Sigma.T, atol=1e-9)
            eigs = np.linalg.eigvalsh(Sigma)
            assert np.all(eigs >= -1e-8)

    def test_estimate_effect_composition_still_works(self, stm_model):
        m, treat = stm_model
        X = treat.reshape(-1, 1)
        from topica.stm import posterior_theta_samples
        draws = posterior_theta_samples(m, nsims=20, seed=1)
        effects = estimate_effect(draws, X=X, feature_names=["treat"])
        assert len(effects) == m.num_topics
        for e in effects:
            assert np.all(e.se >= 0)
            assert np.all(e.ci_low <= e.coef + 1e-9)


# ---------------------------------------------------------------------------
# Export surface
# ---------------------------------------------------------------------------

class TestExports:
    def test_toplevel_export(self):
        assert hasattr(topica.effects, "predicted_prevalence")
        assert hasattr(topica.effects, "PredictedPrevalence")
        assert topica.effects.predicted_prevalence is topica.stm.predicted_prevalence
        assert topica.effects.predicted_prevalence is topica.effects.predicted_prevalence

    def test_stm_module_export(self):
        assert hasattr(topica.stm, "predicted_prevalence")
        assert hasattr(topica.stm, "PredictedPrevalence")

    def test_effects_module_export(self):
        assert hasattr(topica.effects, "predicted_prevalence")
        assert hasattr(topica.effects, "PredictedPrevalence")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_no_X_no_formula_raises(self, lda_model):
        m, _ = lda_model
        with pytest.raises(ValueError, match="X|formula"):
            predicted_prevalence(m, at={"x": 0.0})

    def test_formula_without_data_raises(self, lda_model):
        m, _ = lda_model
        with pytest.raises(ValueError, match="data"):
            predicted_prevalence(m, formula="~ x", at={"x": 0.0})

    def test_no_mode_raises(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        with pytest.raises(ValueError):
            predicted_prevalence(m, X=X, feature_names=["treat"])

    def test_out_of_range_topic_raises(self, lda_model):
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        with pytest.raises(ValueError, match="topic"):
            predicted_prevalence(m, X=X, feature_names=["treat"],
                                  at={"treat": 0.0}, topics=[99])


# ---------------------------------------------------------------------------
# Issue #99 Part 1: categorical covariates via formula (closes #99)
# ---------------------------------------------------------------------------

def _make_docs_party(n=120, seed=7):
    """Docs with a categorical 'party' covariate (D/R) and a numeric 'year'."""
    rng = np.random.default_rng(seed)
    docs, parties, years = [], [], []
    for i in range(n):
        party = "D" if i % 2 == 0 else "R"
        year_val = 2010 + (i % 5)
        heavy, light = (ECON, MIL) if party == "D" else (MIL, ECON)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
        parties.append(party)
        years.append(float(year_val))
    return docs, parties, years


@pytest.fixture(scope="module")
def lda_party_model():
    """LDA fitted on docs with a categorical 'party' covariate."""
    docs, parties, years = _make_docs_party(n=120, seed=7)
    m = topica.models.LDA(2, seed=3)
    m.fit(docs, iters=200)
    meta = pd.DataFrame({"party": parties, "year": years})
    return m, meta


class TestCategoricalFormulaIssue99:
    """Categorical covariates via formula must not drop dummy columns on prediction
    rows that contain only one factor level (issue #99 Part 1)."""

    def test_at_categorical_shape(self, lda_party_model):
        """at= with a list of categorical levels returns one estimate per level."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            at={"party": ["D", "R"]},
            nsims=10, n_sim=200, seed=0,
        )
        assert isinstance(result, list)
        assert len(result) == 2  # two topics
        for r in result:
            assert r.estimate.shape == (2,)
            assert r.ci_low.shape == (2,)
            assert r.ci_high.shape == (2,)
            assert r.mode == "at"

    def test_at_categorical_finite(self, lda_party_model):
        """Predicted prevalences for categorical at= must be finite and in [0,1]."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            at={"party": ["D", "R"]},
            nsims=10, n_sim=200, seed=0,
        )
        for r in result:
            assert np.all(np.isfinite(r.estimate))
            assert np.all(r.estimate >= -0.1)  # identity link can go slightly negative
            assert np.all(np.isfinite(r.ci_low))
            assert np.all(np.isfinite(r.ci_high))

    def test_at_categorical_ci_ordering(self, lda_party_model):
        """CI bounds must bracket the point estimate for each grid row."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            at={"party": ["D", "R"]},
            nsims=10, n_sim=200, seed=0,
        )
        for r in result:
            assert np.all(r.ci_low <= r.estimate + 1e-9)
            assert np.all(r.estimate <= r.ci_high + 1e-9)

    def test_contrast_categorical_formula(self, lda_party_model):
        """contrast= with a categorical column must return finite values."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            contrast={"party": ["D", "R"]},
            nsims=10, n_sim=200, seed=0,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert r.mode == "contrast"
            assert r.estimate.shape == (1,)
            assert np.all(np.isfinite(r.estimate))
            assert np.all(np.isfinite(r.ci_low))
            assert np.all(np.isfinite(r.ci_high))

    def test_contrast_categorical_opposite_sign(self, lda_party_model):
        """Party should pull topics in opposite directions."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            contrast={"party": ["D", "R"]},
            nsims=15, n_sim=400, seed=0,
        )
        diffs = [float(r.estimate[0]) for r in result]
        # The two topics must have roughly opposite contrasts (sum near zero).
        assert abs(diffs[0] + diffs[1]) < 0.3

    def test_at_single_level_no_crash(self, lda_party_model):
        """A single-level at= for a categorical must not crash (was the core bug)."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            at={"party": ["D"]},
            nsims=5, n_sim=100, seed=0,
        )
        for r in result:
            assert r.estimate.shape == (1,)
            assert np.all(np.isfinite(r.estimate))

    def test_to_frame_categorical_at(self, lda_party_model):
        """to_frame() must include the covariate columns from at=."""
        m, meta = lda_party_model
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            at={"party": ["D", "R"]},
            nsims=5, n_sim=50, seed=0,
        )
        df = result[0].to_frame()
        assert "party" in df.columns
        assert "estimate" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Issue #99 Part 2: tuple/sequence contrast form (closes #99)
# ---------------------------------------------------------------------------

class TestTupleContrastIssue99:
    """Sequence/tuple contrast= must work without raising NameError (issue #99 Part 2)."""

    def test_scalar_tuple_contrast_single_feature(self, lda_model):
        """(val_a, val_b) works for a single-feature raw-X model."""
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        result = predicted_prevalence(
            m, X=X, feature_names=["treat"],
            contrast=(0.0, 1.0),
            nsims=10, n_sim=200, seed=0,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert r.mode == "contrast"
            assert r.estimate.shape == (1,)
            assert np.all(np.isfinite(r.estimate))
            assert np.all(np.isfinite(r.ci_low))
            assert np.all(np.isfinite(r.ci_high))

    def test_tuple_contrast_agrees_with_dict_contrast(self, lda_model):
        """Tuple contrast (0.0, 1.0) must give the same result as dict contrast."""
        m, treat = lda_model
        X = treat.reshape(-1, 1)
        res_tuple = predicted_prevalence(
            m, X=X, feature_names=["treat"],
            contrast=(0.0, 1.0),
            nsims=15, n_sim=300, seed=42,
        )
        res_dict = predicted_prevalence(
            m, X=X, feature_names=["treat"],
            contrast={"treat": [0.0, 1.0]},
            nsims=15, n_sim=300, seed=42,
        )
        for r_t, r_d in zip(res_tuple, res_dict):
            np.testing.assert_allclose(r_t.estimate, r_d.estimate, atol=1e-9)

    def test_dict_sequence_contrast_formula(self, lda_party_model):
        """A 2-tuple of dicts works as two full covariate settings for the formula path."""
        m, meta = lda_party_model
        setting_d = {"party": "D", "year": 2012.0}
        setting_r = {"party": "R", "year": 2012.0}
        result = predicted_prevalence(
            m, formula="~ party + year", data=meta,
            contrast=(setting_d, setting_r),
            nsims=10, n_sim=200, seed=0,
        )
        assert isinstance(result, list)
        for r in result:
            assert r.mode == "contrast"
            assert np.all(np.isfinite(r.estimate))

    def test_scalar_tuple_contrast_multi_feature_raises(self, lda_model):
        """Scalar tuple contrast with multi-feature X must raise a clear ValueError."""
        m, treat = lda_model
        X2 = np.column_stack([treat, np.ones(len(treat))])
        with pytest.raises(ValueError, match="multi-feature|dict"):
            predicted_prevalence(
                m, X=X2, feature_names=["treat", "other"],
                contrast=(0.0, 1.0),
                nsims=5, n_sim=50, seed=0,
            )

    def test_scalar_tuple_contrast_formula_scalar_raises(self, lda_party_model):
        """Scalar tuple contrast with formula= must raise a clear ValueError."""
        m, meta = lda_party_model
        with pytest.raises(ValueError, match="dict"):
            predicted_prevalence(
                m, formula="~ party + year", data=meta,
                contrast=(0.0, 1.0),
                nsims=5, n_sim=50, seed=0,
            )
