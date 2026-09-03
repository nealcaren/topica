"""B-spline prevalence terms: topica.design.bs / s, the s()/bs() formula terms,
and STM's fit-time formula= path (issue #867).

The numerical checks here are self-contained (no R): a B-spline basis is a
partition of unity on its support, the column count is exactly ``df``, and the
smooth ``s()`` term reproduces ``bs(df=min(10, n_unique-1))``. Column-identical
parity against R ``splines::bs`` / ``stm:::s`` lives in
``parity/stm_bspline_867.py`` (R-gated, skips cleanly without Rscript).
"""

import numpy as np
import pytest

import topica
from topica import stm

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# bs(): the B-spline basis helper
# ---------------------------------------------------------------------------

def test_bs_shape_and_names():
    x = np.linspace(0, 10, 60)
    basis, names = topica.design.bs(x, df=10, name="day")
    assert basis.shape == (60, 10)  # df columns, no intercept
    assert names == [f"bs(day, df=10)[{j}]" for j in range(10)]


def test_bs_partition_of_unity_interior():
    # splines::bs drops the first basis column (intercept=FALSE), so the row sums
    # are <= 1; adding an intercept column back recovers the partition of unity.
    x = np.linspace(0, 100, 200)
    basis, _ = topica.design.bs(x, df=8)
    # Reconstruct the full (intercept) basis: full row-sum is exactly 1.
    order = 4
    interior, (b0, b1) = stm._bs_knots(x, 8, 3)
    knots = np.concatenate([np.repeat(b0, order), interior, np.repeat(b1, order)])
    full = stm._bspline_derivs(x, knots, order, 0)[0]
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-12)
    # The dropped-first-column basis equals full[:, 1:].
    assert np.allclose(basis, full[:, 1:], atol=1e-12)


def test_bs_boundary_endpoint_included():
    # x at the right boundary knot must land in the final basis function.
    x = np.linspace(0, 5, 30)
    basis, _ = topica.design.bs(x, df=6)
    assert basis[-1, -1] == pytest.approx(1.0)


def test_bs_shoves_knot_coinciding_with_boundary():
    # Spike at the max: the 75% quantile lands on the right boundary knot, so R
    # (and now topica) shoves it 1/8 of the gap inside instead of leaving a
    # zero-width span. R's value for this exact case is 9.5.
    x = np.array([1.0] * 5 + list(range(2, 11)) + [10.0] * 5)
    interior, (b0, b1) = stm._bs_knots(x, 6, 3)
    assert b0 == 1.0 and b1 == 10.0
    assert interior.max() < b1  # no interior knot sits on the boundary
    assert interior[-1] == pytest.approx(9.5)
    # Basis is still well-formed (df columns, finite).
    basis, _ = topica.design.bs(x, df=6)
    assert basis.shape == (len(x), 6)
    assert np.all(np.isfinite(basis))


def test_bs_rejects_nonfinite_and_constant():
    with pytest.raises(ValueError, match="non-finite"):
        topica.design.bs(np.array([1.0, 2.0, np.nan, 4.0]), df=5)
    with pytest.raises(ValueError, match="non-finite"):
        topica.design.s(np.array([1.0, np.inf, 3.0]))
    with pytest.raises(ValueError, match="constant"):
        topica.design.bs(np.array([3.0, 3.0, 3.0, 3.0]), df=5)


def test_fit_formula_rejects_nonfinite_covariate():
    pytest.importorskip("formulaic")
    rng = np.random.default_rng(0)
    docs = [[f"w{rng.integers(0, 20)}" for _ in range(20)] for _ in range(30)]
    day = rng.uniform(0, 100, 30)
    day[3] = np.nan
    meta = pd.DataFrame({"day": day})
    # A NaN covariate must fail loudly, not silently fit an all-zero design.
    # (formulaic wraps the ValueError from s(), but the message is preserved.)
    with pytest.raises(Exception) as excinfo:
        topica.STM(3, seed=13).fit(docs, formula="~ s(day)", data=meta, iters=10)
    assert "non-finite" in str(excinfo.value)


def test_bs_replay_with_frozen_knots_matches_training():
    x = np.sort(np.random.default_rng(0).uniform(0, 50, 80))
    train, _ = topica.design.bs(x, df=10)
    interior, (b0, b1) = stm._bs_knots(x, 10, 3)
    # Re-applying with the captured knots on the SAME points reproduces the basis.
    replay, _ = topica.design.bs(x, knots=interior, boundary_knots=(b0, b1))
    assert np.allclose(train, replay, atol=1e-12)
    # And on a new in-range grid, the basis still sums (with intercept) to 1.
    grid = np.linspace(b0, b1, 15)
    g, _ = topica.design.bs(grid, knots=interior, boundary_knots=(b0, b1))
    assert g.shape == (15, 10)


# ---------------------------------------------------------------------------
# s(): R stm's smooth term (df = min(10, n_unique - 1))
# ---------------------------------------------------------------------------

def test_s_default_df_from_cardinality():
    # Continuous covariate -> df = 10.
    cont = np.linspace(0, 1, 40)
    basis, names = topica.design.s(cont)
    assert basis.shape[1] == 10
    assert names[0].startswith("s(x, df=10)")
    # Low-cardinality covariate -> df = min(10, n_unique - 1).
    depth = np.array([0, 1, 2, 3, 4, 5, 6] * 6, dtype=float)  # 7 unique
    b2, n2 = topica.design.s(depth)
    assert b2.shape[1] == 6
    assert n2[0].startswith("s(x, df=6)")


def test_s_matches_bs_with_resolved_df():
    depth = np.array([0, 1, 2, 3, 4, 5, 6] * 5, dtype=float)
    bs_basis, _ = topica.design.bs(depth, df=6)
    s_basis, _ = topica.design.s(depth)
    assert np.allclose(bs_basis, s_basis, atol=1e-12)


# ---------------------------------------------------------------------------
# formula terms s(...) / bs(...) with knot capture and prediction replay
# ---------------------------------------------------------------------------

def test_formula_s_bs_columns():
    pytest.importorskip("formulaic")
    df = pd.DataFrame({"day": np.linspace(0, 100, 50), "z": np.linspace(-1, 1, 50)})
    X, names = topica.design_matrix("~ s(day) + bs(z, df=5)", df)
    assert X.shape == (50, 15)  # 10 + 5, no intercept
    assert not any(n.lower() == "intercept" for n in names)


def test_formula_prediction_replays_training_knots():
    pytest.importorskip("formulaic")
    from topica.formulas import _KnotCapturingContext, design_matrix, design_matrix_predict

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"day": np.sort(rng.uniform(0, 100, 60))})
    kc = _KnotCapturingContext()
    Xtr, _ = design_matrix("~ s(day)", df, _knot_ctx=kc)
    # Predicting on the same rows must reproduce the training design exactly
    # (same knots, not re-estimated from the prediction frame).
    Xpr, _ = design_matrix_predict("~ s(day)", df, kc)
    assert np.allclose(Xtr, Xpr, atol=1e-12)


# ---------------------------------------------------------------------------
# STM.fit(formula=, data=): the fit-time path
# ---------------------------------------------------------------------------

@pytest.fixture
def smooth_corpus():
    rng = np.random.default_rng(3)
    K, V, D = 4, 30, 150
    vocab = [f"w{i}" for i in range(V)]
    day = rng.uniform(0, 100, D)
    docs = []
    for d in range(D):
        frac = 0.5 + 0.4 * np.sin(day[d] / 100 * np.pi)
        topic = 0 if rng.random() < frac else 1
        docs.append([vocab[(topic * 7 + rng.integers(0, 7)) % V] for _ in range(30)])
    return docs, pd.DataFrame({"day": day})


def test_fit_formula_equals_raw_bs_design(smooth_corpus):
    pytest.importorskip("formulaic")
    docs, meta = smooth_corpus
    m_form = topica.STM(4, seed=13).fit(docs, formula="~ s(day)", data=meta, iters=50)
    X = topica.design.s(meta["day"].to_numpy())[0]
    m_raw = topica.STM(4, seed=13).fit(docs, prevalence=X, iters=50)
    # Identical design -> identical fit.
    assert m_form.bound == pytest.approx(m_raw.bound, abs=1e-9)
    # 1 intercept + 10 spline columns, K-1 outcome columns.
    assert np.asarray(m_form.prevalence_effects).shape == (11, 3)


def test_fit_formula_reproducible_from_same_frame(smooth_corpus):
    # The spline knots are deterministic in `data`, so estimate_effect /
    # predicted_prevalence rebuild the fitted design from the same formula+frame.
    pytest.importorskip("formulaic")
    docs, meta = smooth_corpus
    m = topica.STM(4, seed=13).fit(docs, formula="~ s(day)", data=meta, iters=30)
    X_fit, _ = topica.design_matrix("~ s(day)", meta)
    X_again, _ = topica.design_matrix("~ s(day)", meta)
    assert np.array_equal(X_fit, X_again)
    # And the effects path accepts the same formula+data without error.
    eff = stm.estimate_effect(m, formula="~ s(day)", data=meta, nsims=15, corpus=docs)
    assert len(eff) == 4


def test_fit_formula_error_paths(smooth_corpus):
    pytest.importorskip("formulaic")
    docs, meta = smooth_corpus
    with pytest.raises(ValueError, match="requires data="):
        topica.STM(4).fit(docs, formula="~ s(day)")
    X = topica.design.s(meta["day"].to_numpy())[0]
    with pytest.raises(ValueError, match="either formula= or prevalence"):
        topica.STM(4).fit(docs, prevalence=X, formula="~ s(day)", data=meta)


def test_stm_is_subclass_of_core():
    # The public STM is the Python wrapper; instances are still core STM.
    assert issubclass(topica.STM, topica._topica.STM)
    m = topica.STM(3)
    assert isinstance(m, topica._topica.STM)
    assert type(m).__name__ == "STM"
