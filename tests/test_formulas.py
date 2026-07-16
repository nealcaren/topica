"""R-style formula interface: design_matrix and estimate_effect(formula=...)."""

import numpy as np
import pytest

import topica
from topica import stm

pd = pytest.importorskip("pandas")
pytest.importorskip("formulaic")


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "party": ["D", "R"] * 10,
            "year": list(range(2000, 2010)) * 2,
            "x": np.linspace(0, 1, 20),
        }
    )


def test_design_matrix_strips_intercept_and_expands(frame):
    X, names = topica.design.design_matrix("~ party * x", frame)
    # No standalone intercept column (estimate_effect adds its own).
    assert not any(n.lower() == "intercept" for n in names)
    assert names == ["party[T.R]", "x", "party[T.R]:x"]
    assert X.shape == (20, 3)


def test_design_matrix_spline_columns(frame):
    X, names = topica.design.design_matrix("~ spline(year, df=3)", frame)
    assert len(names) == 3
    assert all("spline(year" in n for n in names)
    assert X.shape == (20, 3)


def test_formula_path_matches_manual_X(frame):
    theta = np.random.default_rng(0).dirichlet([1, 1, 1], size=20)
    X, names = topica.design.design_matrix("~ party + x", frame)
    manual = stm.estimate_effect(theta, X, feature_names=names)
    viaform = stm.estimate_effect(theta, data=frame, formula="~ party + x")
    for a, b in zip(manual, viaform):
        assert a.feature_names == b.feature_names
        assert np.allclose(a.coef, b.coef)
        assert np.allclose(a.se, b.se)


def test_string_cluster_matches_array_cluster(frame):
    frame = frame.copy()
    frame["blog"] = ["a", "b", "c", "d", "e"] * 4
    theta = np.random.default_rng(1).dirichlet([1, 1], size=20)
    by_name = stm.estimate_effect(theta, data=frame, formula="~ party", cluster="blog")
    X, names = topica.design.design_matrix("~ party", frame)
    by_array = stm.estimate_effect(
        theta, X, feature_names=names, cluster=frame["blog"].to_numpy()
    )
    assert by_name[0].feature_names == ["intercept", "party[T.R]"]
    assert np.allclose(by_name[0].se, by_array[0].se)  # same clustered SEs


def test_errors():
    theta = np.random.default_rng(0).dirichlet([1, 1], size=4)
    with pytest.raises(ValueError):
        stm.estimate_effect(theta)  # no X, no formula
    with pytest.raises(ValueError):
        stm.estimate_effect(theta, formula="~ x")  # formula without data
