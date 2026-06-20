"""Parity: topica's survey-weighted ``estimate_effect`` and ``average_marginal_effects``
against faSTM's exact effect-layer formulas, reimplemented in base R.

faSTM (the R reimplementation of STM on the same topica-core engine) adds two
covariate-effect features topica did not have: survey weights (weighted least
squares) and average marginal effects (``ame``). Both live in faSTM's pure-R
``estimateEffect`` layer:

  - weighted ``.ols``: ``beta = (X'WX)^-1 X'W y``; classical vcov
    ``(rss/df)(X'WX)^-1`` with ``rss = sum(w e^2)``; the cluster-robust meat uses
    weighted score rows ``X * (w e)`` with the Stata/CR1 finite-sample factor.
  - ``ame``: average the design-change vector over the data (factor:
    level-vs-reference contrast; continuous: central difference), then
    ``est = cv . beta``, ``se = sqrt(cv' Sigma cv)``.

The effect layer is engine-independent: given the SAME theta and design, faSTM and
topica must compute the SAME numbers (topic numbering, which differs between
independent fits, never enters). So this check fixes a deterministic
(theta, design, weights, cluster) in Python, computes topica's values, and
compares them to faSTM's formulas evaluated in R on the identical inputs. The R
side is base R only (no faSTM/stm install needed) and transcribes faSTM's source
verbatim, so a match is a faithful parity with faSTM's algorithm.

Skips cleanly (exit 0) if Rscript is not on PATH.

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python parity/weights_ame_faSTM_compare.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import topica

TOL = 1e-8


def _make_data(seed: int = 7, D: int = 400, K: int = 4):
    """A deterministic (theta, design frame, weights, cluster)."""
    rng = np.random.default_rng(seed)
    year = rng.normal(size=D)
    party = np.where(rng.integers(0, 2, D) == 1, "R", "D")
    # Topic 0 depends on both covariates; the rest split the remainder.
    t0 = np.clip(0.30 + 0.06 * year + 0.13 * (party == "R")
                 + 0.04 * rng.normal(size=D), 0.01, 0.99)
    rest = rng.dirichlet(np.ones(K - 1), size=D) * (1 - t0)[:, None]
    theta = np.column_stack([t0, rest])
    weights = rng.uniform(0.5, 2.0, D)
    cluster = rng.integers(0, 40, D)
    return theta, year, party, weights, cluster


def _r_reference(tmp: str) -> dict:
    """Run faSTM's effect formulas (base R) on the exported inputs; read results."""
    r_code = r"""
    args <- commandArgs(trailingOnly = TRUE)
    d <- args[1]
    th  <- as.matrix(read.csv(file.path(d, "theta.csv")))      # D x K (topic 0 = col 1)
    cov <- read.csv(file.path(d, "cov.csv"), stringsAsFactors = TRUE)
    w   <- read.csv(file.path(d, "w.csv"))$w
    cl  <- read.csv(file.path(d, "cluster.csv"))$cluster

    # faSTM's weighted .ols (verbatim): classical when no cluster, else CR1 sandwich.
    ols <- function(X, y, w = NULL, cluster = NULL) {
      n <- nrow(X); p <- ncol(X)
      fit <- if (is.null(w)) lm.fit(X, y) else lm.wfit(X, y, w)
      resid <- fit$residuals
      df <- n - p
      XtXi <- if (is.null(w)) chol2inv(qr.R(qr(X))) else chol2inv(chol(crossprod(X * sqrt(w))))
      if (is.null(cluster)) {
        rss <- if (is.null(w)) sum(resid^2) else sum(w * resid^2)
        vcov <- (rss / df) * XtXi
      } else {
        sc <- X * (if (is.null(w)) resid else w * resid)
        G <- split(seq_len(n), cluster)
        meat <- Reduce(`+`, lapply(G, function(g) tcrossprod(colSums(sc[g, , drop = FALSE]))))
        ng <- length(G)
        adj <- (ng / (ng - 1)) * ((n - 1) / (n - p))
        vcov <- adj * (XtXi %*% meat %*% XtXi)
      }
      list(coef = fit$coefficients, vcov = vcov)
    }

    # Design with an intercept (topica prepends one; formulaic treatment-codes party).
    X <- model.matrix(~ year + party, cov)     # cols: (Intercept), year, partyR
    y <- th[, 1]                               # topic 0

    w_ols  <- ols(X, y, w = w)                 # weighted, no cluster
    w_clus <- ols(X, y, w = w, cluster = cl)   # weighted + cluster

    # ame on the weighted fit (continuous year: central diff; factor party: contrast).
    co <- w_ols$coef; V <- w_ols$vcov
    hh <- 0.01 * sd(cov$year)
    nd1 <- cov; nd1$year <- cov$year + hh
    nd0 <- cov; nd0$year <- cov$year - hh
    cv_year <- colMeans((model.matrix(~ year + party, nd1) -
                         model.matrix(~ year + party, nd0)) / (2 * hh))
    ame_year <- sum(cv_year * co)
    se_year  <- sqrt(as.numeric(t(cv_year) %*% V %*% cv_year))

    lv <- levels(cov$party); ref <- lv[1]
    nd_r <- cov; nd_r$party <- factor("R", levels = lv)
    nd_d <- cov; nd_d$party <- factor(ref, levels = lv)
    cv_party <- colMeans(model.matrix(~ year + party, nd_r) -
                         model.matrix(~ year + party, nd_d))
    ame_party <- sum(cv_party * co)
    se_party  <- sqrt(as.numeric(t(cv_party) %*% V %*% cv_party))

    out <- c(w_ols$coef, sqrt(diag(w_ols$vcov)),
             sqrt(diag(w_clus$vcov)),
             ame_year, se_year, ame_party, se_party)
    names(out) <- c("b_int","b_year","b_party","se_int","se_year","se_party",
                    "sec_int","sec_year","sec_party",
                    "ame_year","ame_year_se","ame_party","ame_party_se")
    write.csv(data.frame(name = names(out), value = as.numeric(out)),
              file.path(d, "r_out.csv"), row.names = FALSE)
    """
    script = os.path.join(tmp, "ref.R")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(r_code)
    subprocess.run(["Rscript", script, tmp], check=True, capture_output=True, text=True)
    out = {}
    with open(os.path.join(tmp, "r_out.csv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            name, value = line.strip().split(",")
            out[name.strip('"')] = float(value)
    return out


def main() -> int:
    if shutil.which("Rscript") is None:
        print("SKIP: Rscript not on PATH (parity check needs R for the faSTM reference)")
        return 0

    import pandas as pd

    theta, year, party, weights, cluster = _make_data()
    cov = pd.DataFrame({"year": year, "party": party})

    # topica side.
    eff_w = topica.estimate_effect(theta, formula="~ year + party", data=cov, weights=weights)[0]
    eff_wc = topica.estimate_effect(theta, formula="~ year + party", data=cov,
                                    weights=weights, cluster=cluster)[0]
    j = {n: eff_w.feature_names.index(n) for n in ("intercept", "year", "party[T.R]")}
    ame_y = topica.average_marginal_effects(theta, "year", formula="~ year + party",
                                            data=cov, weights=weights).to_frame()
    ame_p = topica.average_marginal_effects(theta, "party", formula="~ year + party",
                                            data=cov, weights=weights).to_frame()
    y_row = ame_y.query("topic == 0 and term == 'year'").iloc[0]
    p_row = ame_p.query("topic == 0 and term == 'partyR'").iloc[0]

    tp = {
        "b_int": eff_w.coef[j["intercept"]], "b_year": eff_w.coef[j["year"]],
        "b_party": eff_w.coef[j["party[T.R]"]],
        "se_int": eff_w.se[j["intercept"]], "se_year": eff_w.se[j["year"]],
        "se_party": eff_w.se[j["party[T.R]"]],
        "sec_int": eff_wc.se[j["intercept"]], "sec_year": eff_wc.se[j["year"]],
        "sec_party": eff_wc.se[j["party[T.R]"]],
        "ame_year": y_row.ame, "ame_year_se": y_row.se,
        "ame_party": p_row.ame, "ame_party_se": p_row.se,
    }

    tmp = tempfile.mkdtemp(prefix="topica_faSTM_parity_")
    try:
        np.savetxt(os.path.join(tmp, "theta.csv"), theta, delimiter=",",
                   header=",".join(f"t{i}" for i in range(theta.shape[1])), comments="")
        cov.to_csv(os.path.join(tmp, "cov.csv"), index=False)
        pd.DataFrame({"w": weights}).to_csv(os.path.join(tmp, "w.csv"), index=False)
        pd.DataFrame({"cluster": cluster}).to_csv(os.path.join(tmp, "cluster.csv"), index=False)
        ref = _r_reference(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"{'quantity':<14} {'topica':>14} {'faSTM (R)':>14} {'|diff|':>10}")
    worst = 0.0
    for k in ("b_int", "b_year", "b_party", "se_year", "se_party",
              "sec_year", "sec_party", "ame_year", "ame_year_se",
              "ame_party", "ame_party_se"):
        diff = abs(tp[k] - ref[k])
        worst = max(worst, diff)
        flag = "" if diff <= TOL else "  <-- MISMATCH"
        print(f"{k:<14} {tp[k]:>14.8f} {ref[k]:>14.8f} {diff:>10.2e}{flag}")

    print(f"\nmax |diff| = {worst:.2e}  (tol {TOL:.0e})")
    if worst <= TOL:
        print("OK: topica's weighted estimate_effect and AME match faSTM's formulas.")
        return 0
    print("FAIL: a quantity diverged from faSTM beyond tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
