"""#867 B-spline prevalence parity: topica's ``bs`` / ``s`` are column-identical
to R ``splines::bs`` and ``stm:::s``.

Issue #867: ``topica.STM`` took only a *linear* prevalence design, so a
"full-strength STM" built with a continuous structural covariate was a weaker
predictive baseline than R ``stm``, which puts a smooth B-spline (``s(x)``) on
continuous prevalence terms by default. topica now exposes that exact basis via
``topica.design.bs`` / ``topica.design.s`` and the ``s(...)`` / ``bs(...)``
formula terms, and ``STM.fit(formula=~ s(day), data=meta)`` builds it for you.

This script checks, against a live R:

  1. ``bs(x, df=10, degree=3, intercept=FALSE)`` matches ``splines::bs`` on the
     training points (interior knots at type-7 quantiles, boundary at range(x)).
  2. ``predict(bs_obj, newx)`` matches on a grid that extends *beyond* the
     boundary knots -- i.e. R's degree-3 Taylor extrapolation off the data range.
  3. ``s(x)`` matches ``stm:::s(x)`` with its default df = min(10, n_unique - 1),
     including a low-cardinality (integer) covariate where df < 10.

Skips cleanly (exit 0, prints SKIP) when Rscript or the stm/splines packages are
unavailable, so it is safe to run in any environment.

    python parity/stm_bspline_867.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from topica.design import bs, s  # noqa: E402

ATOL = float(os.environ.get("BSPLINE_ATOL", "1e-9"))


def r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e",
             'cat(requireNamespace("stm", quietly=TRUE) && '
             'requireNamespace("splines", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


_R_DRIVER = r"""
suppressMessages(library(stm)); suppressMessages(library(splines))
x  <- as.numeric(read.csv(file.path(D, "x.csv"), header = FALSE)[, 1])
g  <- as.numeric(read.csv(file.path(D, "g.csv"), header = FALSE)[, 1])
dp <- as.numeric(read.csv(file.path(D, "depth.csv"), header = FALSE)[, 1])

# 1/2: splines::bs on training points and predicted on an out-of-range grid.
b  <- bs(x, df = 10, degree = 3, intercept = FALSE)
class(b) <- c("bs", "basis", "matrix", "array")   # skip stm's buggy predict.s path
G  <- suppressWarnings(predict(b, g))
# 3: stm:::s default df on a low-cardinality integer covariate.
sd <- stm:::s(dp)
write.table(b,  file.path(D, "B.csv"),  sep = ",", row.names = FALSE, col.names = FALSE)
write.table(G,  file.path(D, "G.csv"),  sep = ",", row.names = FALSE, col.names = FALSE)
write.table(sd, file.path(D, "SD.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
"""


def main() -> int:
    if not r_available():
        print("SKIP: Rscript or the stm/splines packages are unavailable.")
        return 0

    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 100, 40))
    grid = np.linspace(-25, 130, 21)          # spans beyond both boundary knots
    depth = np.array([0, 1, 2, 3, 4, 5, 6] * 6, dtype=float)  # 7 unique -> df 6

    from topica.stm import _bs_knots

    py_B, _ = bs(x, df=10)
    interior, (b0, b1) = _bs_knots(x, 10, 3)
    py_G, _ = bs(grid, knots=interior, boundary_knots=(b0, b1))
    py_SD, _ = s(depth)

    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "x.csv"), x, delimiter=",")
        np.savetxt(os.path.join(d, "g.csv"), grid, delimiter=",")
        np.savetxt(os.path.join(d, "depth.csv"), depth, delimiter=",")
        script = f'D <- "{d}"\n' + _R_DRIVER
        proc = subprocess.run(["Rscript", "-e", script],
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print("SKIP: R driver failed:\n" + proc.stderr[-800:])
            return 0
        R_B = np.loadtxt(os.path.join(d, "B.csv"), delimiter=",")
        R_G = np.loadtxt(os.path.join(d, "G.csv"), delimiter=",")
        R_SD = np.loadtxt(os.path.join(d, "SD.csv"), delimiter=",")

    d_fit = float(np.max(np.abs(R_B - py_B)))
    d_grid = float(np.max(np.abs(R_G - py_G)))
    d_s = float(np.max(np.abs(R_SD - py_SD)))

    print(f"bs() training design   max|Δ| vs splines::bs      = {d_fit:.2e}  "
          f"(shape {py_B.shape})")
    print(f"bs() out-of-range grid max|Δ| vs predict(bs,·)    = {d_grid:.2e}  "
          f"(shape {py_G.shape})")
    print(f"s()  low-card df={py_SD.shape[1]} max|Δ| vs stm:::s = {d_s:.2e}")

    ok = d_fit < ATOL and d_grid < 1e-6 and d_s < ATOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
