"""Random-intercept prevalence terms in estimate_effect (lme4-style (1 | group)).

Functional checks plus an exact parity leg against R lme4 (skipped when Rscript /
lme4 are unavailable).
"""

import shutil
import subprocess

import numpy as np
import pytest

from topica import stm


def _sim(n=300, q=12, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, q, size=n)
    x = rng.normal(size=n)
    b = rng.normal(0.0, 0.5, size=q)
    y = 0.2 + 0.4 * x + b[group] + rng.normal(0.0, 0.3, size=n)
    theta = np.column_stack([y, 1.0 - y])
    import pandas as pd

    data = pd.DataFrame({"x": x, "grp": group})
    return theta, data, x, y, group


def test_random_intercept_fits_and_reports_varcomp():
    theta, data, *_ = _sim()
    eff = stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | grp)")
    e = eff[0]
    assert e.feature_names == ["intercept", "x"]
    assert np.all(np.isfinite(e.coef)) and np.all(e.se > 0)
    assert np.isnan(e.r_squared)  # r^2 undefined for a mixed model
    assert set(e.varcomp) == {"grp", "residual"}
    assert e.varcomp["grp"] >= 0 and e.varcomp["residual"] > 0
    # recovers the fixed effect roughly (intercept ~0.2, x ~0.4)
    assert abs(e.coef[1] - 0.4) < 0.1


def test_random_intercept_widens_se_vs_ignoring_grouping():
    # A random intercept that absorbs between-group variation should not report
    # smaller SEs than plain OLS that ignores it.
    theta, data, *_ = _sim()
    ols = stm.estimate_effect(theta, formula="~ x", data=data)[0]
    re = stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | grp)")[0]
    assert re.se[0] >= ols.se[0] - 1e-9  # intercept SE not understated


def test_random_intercept_method_of_composition():
    # 3-D posterior draws route through the per-draw REML + Rubin pooling.
    theta, data, *_ = _sim()
    rng = np.random.default_rng(1)
    draws = np.stack([theta + rng.normal(0, 0.02, size=theta.shape) for _ in range(8)])
    eff = stm.estimate_effect(draws, formula="~ x", data=data, random="(1 | grp)")
    assert len(eff) == 2
    assert np.all(eff[0].se > 0)
    assert eff[0].varcomp is not None


def test_random_intercept_error_paths():
    theta, data, *_ = _sim()
    with pytest.raises(NotImplementedError, match="random slopes"):
        stm.estimate_effect(theta, formula="~ x", data=data, random="(x | grp)")
    with pytest.raises(ValueError, match="link='identity'"):
        stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | grp)", link="logit")
    with pytest.raises(ValueError, match="cluster= or weights="):
        stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | grp)",
                            cluster=np.zeros(len(data)))
    with pytest.raises(ValueError, match="'nope' column"):
        stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | nope)")
    with pytest.raises(ValueError, match="single grouping factor"):
        stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | a | b)")


def _lme4_available():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("lme4", quietly=TRUE))'],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and "TRUE" in r.stdout


@pytest.mark.parity
@pytest.mark.skipif(not _lme4_available(), reason="Rscript with lme4 not available")
def test_random_intercept_matches_lme4():
    import os
    import tempfile

    import pandas as pd

    theta, data, x, y, group = _sim(seed=3)
    eff = stm.estimate_effect(theta, formula="~ x", data=data, random="(1 | grp)")[0]

    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "d.csv")
    pd.DataFrame({"y": y, "x": x, "grp": group}).to_csv(p, index=False)
    script = f"""
    ok <- suppressMessages(require(lme4)); if (!ok) quit(status = 42)
    d <- read.csv("{p}"); d$grp <- factor(d$grp)
    m <- lmer(y ~ x + (1 | grp), data = d, REML = TRUE,
              control = lmerControl(calc.derivs = FALSE))
    fe <- fixef(m); se <- sqrt(diag(as.matrix(vcov(m))))
    vc <- as.data.frame(VarCorr(m))
    cat(fe[1], fe[2], se[1], se[2],
        vc$sdcor[vc$grp == "grp"], vc$sdcor[vc$grp == "Residual"])
    """
    res = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    r_int, r_x, r_se_int, r_se_x, r_sd_grp, r_sd_resid = map(float, res.stdout.split())

    assert e_close(eff.coef[0], r_int)
    assert e_close(eff.coef[1], r_x)
    assert e_close(eff.se[0], r_se_int)
    assert e_close(eff.se[1], r_se_x)
    assert e_close(eff.varcomp["grp"], r_sd_grp, tol=1e-3)
    assert e_close(eff.varcomp["residual"], r_sd_resid, tol=1e-3)


def e_close(a, b, tol=1e-4):
    return abs(a - b) <= tol * (1.0 + abs(b))
