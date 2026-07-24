"""Cross-implementation validation: topica's native STS κ estimator vs R glmnet.

topica's `kappa_estimation="lasso"` and `"adjusted"` both reduce to the same
kernel: an L1-penalized Poisson regression over a λ path with AIC-selected
penalty, solved on an aggregated `(word × group·topic)` design with a
document-fixed-effect offset. That is exactly what CRAN `sts`'s `opt.kappa.R`
delegates to `glmnet::glmnet(..., family="poisson", alpha=1)`.

This script validates the native Rust kernel head-to-head with glmnet on the
actual STS κ design shape (the `"adjusted"` vs `"lasso"` difference is only the
upstream aggregation, checked by Rust unit tests; here we validate the shared
solver). On that design the two agree essentially bit-for-bit; the tolerance
below allows the occasional case where AIC selection lands on an adjacent λ. The
native solver is an independent coordinate-descent implementation, so this is a
"tracks glmnet closely," not a "bit-identical," claim.

Shells out to `Rscript` with the `glmnet` package. Skips (exit 0) when Rscript or
glmnet is unavailable, so it is safe to schedule as a CI / integration job. Run
directly:

    python parity/sts_kappa_glmnet.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from topica import _topica

NLAMBDA = 250
LAMBDA_MIN_RATIO = 0.001
COSINE_FLOOR = 0.999
MAXABS_CEIL = 0.10


def glmnet_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    try:
        out = subprocess.run(
            [rscript, "-e", 'cat(requireNamespace("glmnet", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().upper().startswith("TRUE")


def make_design(seed: int, k: int = 3, g: int = 4):
    """The actual STS κ aggregated design: one row per (group, topic). Columns are
    `[K topic dummies | K sentiment slopes]`, block-diagonal — row `(g, t)` carries
    a 1 in topic-dummy column `t` and the group-mean sentiment `alpha_agg[g][t]` in
    slope column `K+t`. The offset is the doc-topic log-mass and the response is the
    Poisson-sampled aggregated count. This mirrors `opt_kappa`'s design exactly."""
    rng = np.random.default_rng(seed)
    n, p = g * k, 2 * k
    alpha_agg = rng.normal(size=(g, k))
    kappa_t = rng.normal(scale=0.5, size=k)
    kappa_s = rng.normal(scale=0.5, size=k)
    word_rate = rng.uniform(1.0, 5.0, size=(g, k))
    x = np.zeros((n, p))
    offset = np.zeros(n)
    y = np.zeros(n)
    for gg in range(g):
        for t in range(k):
            r = gg * k + t
            x[r, t] = 1.0
            x[r, k + t] = alpha_agg[gg, t]
            offset[r] = np.log(max(word_rate[gg, t], 1e-9))
            eta = kappa_t[t] + kappa_s[t] * alpha_agg[gg, t] + offset[r]
            y[r] = rng.poisson(np.exp(np.clip(eta, -6, 6)))
    return x, y, offset


_R_DRIVER = r"""
suppressMessages(library(glmnet))
x   <- as.matrix(read.csv(file.path(dir, "x.csv"), header = FALSE))
y   <- scan(file.path(dir, "y.csv"), quiet = TRUE)
off <- scan(file.path(dir, "offset.csv"), quiet = TRUE)
mod <- glmnet(x = x, y = y, family = "poisson", offset = off,
              standardize = FALSE, intercept = FALSE, alpha = 1,
              lambda.min.ratio = %(lmr)g, nlambda = %(nl)d,
              maxit = 100000, thresh = 1e-7)
dev <- (1 - mod$dev.ratio) * mod$nulldev
ic  <- dev + 2 * mod$df            # AIC, exactly as opt.kappa.R
sel <- which.min(ic)
coef <- as.numeric(mod$beta[, sel])
writeLines(sprintf("%%.10g", coef), file.path(dir, "coef.csv"))
"""


def r_glmnet_coef(x, y, offset):
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "x.csv"), x, delimiter=",")
        np.savetxt(os.path.join(d, "y.csv"), y)
        np.savetxt(os.path.join(d, "offset.csv"), offset)
        driver = ("dir <- commandArgs(trailingOnly = TRUE)[1]\n") + (
            _R_DRIVER % {"lmr": LAMBDA_MIN_RATIO, "nl": NLAMBDA}
        )
        script = os.path.join(d, "driver.R")
        with open(script, "w") as f:
            f.write(driver)
        subprocess.run(
            [shutil.which("Rscript"), script, d], check=True, capture_output=True, text=True
        )
        return np.loadtxt(os.path.join(d, "coef.csv"))


def aligned_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
    return float(a @ b / (na * nb))


def run() -> None:
    if not glmnet_available():
        print("Rscript or the glmnet package is unavailable. Skipping κ parity check.")
        return

    ok = True
    for seed in range(5):
        x, y, offset = make_design(seed)
        native = np.asarray(
            _topica._sts_poisson_lasso(x.tolist(), y.tolist(), offset.tolist(), NLAMBDA, LAMBDA_MIN_RATIO)
        )
        ref = r_glmnet_coef(x, y, offset)
        cos = aligned_cosine(native, ref)
        maxabs = float(np.max(np.abs(native - ref)))
        status = "OK" if (cos >= COSINE_FLOOR and maxabs <= MAXABS_CEIL) else "FAIL"
        if status == "FAIL":
            ok = False
        print(
            f"seed {seed}: cosine={cos:.4f} max|Δ|={maxabs:.4f}  "
            f"native={np.round(native, 3)} glmnet={np.round(ref, 3)}  [{status}]"
        )

    assert ok, (
        "native STS κ solver diverged from R glmnet beyond tolerance "
        f"(cosine>={COSINE_FLOOR}, max|Δ|<={MAXABS_CEIL})"
    )
    print("SUCCESS: native STS κ solver matches R glmnet within tolerance.")


if __name__ == "__main__":
    run()
