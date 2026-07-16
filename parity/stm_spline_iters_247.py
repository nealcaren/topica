"""#247 EM-iteration anchor: why a prevalence spline costs topica more EM
iterations than R `stm`, and what it does NOT cost.

Issue #247 observed that fitting STM with a day spline (``~ rating + s(day)``)
on poliblog takes topica noticeably more EM iterations to converge than R `stm`,
and asked whether that also means the library's speed estimates are misleading
(a fixed-iteration benchmark compares the two engines at equal iteration counts,
but they need different counts to converge).

This script reproduces the gap and isolates its cause with three measurements,
all on the bundled poliblog5k (5,000 docs), Spectral init, K=20, emtol=1e-5:

  1. Cross-engine, rating only      -> topica ~22 vs R stm ~20  (close)
  2. Cross-engine, rating + spline  -> topica ~32 vs R stm ~20  (the gap)
  3. QR-orthonormalized spline      -> topica ~35  (NOT conditioning)

History: before topica's ``gamma_prior="pooled"`` was made empirical-Bayes (issue
#247), the spline case took ~37 iterations. The prevalence M-step was a *fixed*
1e-6 ridge (~OLS), while R stm's Pooled prior is empirical-Bayes adaptive
shrinkage (`vb.variational.reg`). topica now ports that VB regression faithfully
(see topica-core ``fit_gamma_vb``, golden-tested against stm 1.3.8), removing the
fixed-ridge component of the gap (~37 -> ~32).

Two things this script still demonstrates:
  * Conditioning is NOT the cause -- orthonormalizing the design (condition number
    3e4 -> 1) does not help, because QR preserves the column space, so the
    projected prevalence mean mu = X.gamma is unchanged.
  * A residual gap to stm remains (~32 vs ~20): with a wide prevalence design the
    bound increases monotonically but in smaller per-iteration steps near the
    optimum, so it sits just above emtol for several extra iterations. This is a
    convergence-rate difference, not a correctness issue -- the fitted topics match
    stm (see parity/stm_poliblog5k_compare.py).

Implication for benchmarks: time STM to convergence, not at a fixed iteration
count, when a spline (or any wide prevalence design) is present.

Shells out to ``Rscript`` with ``stm`` for the cross-engine rows; the topica-only
rows (1-spline gap within topica, plus the QR control) always run. Skips the R
rows (still prints topica rows) if Rscript/stm is unavailable. Run:

    python parity/stm_spline_iters_247.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.environ.get("POLIBLOG5K_CSV") or os.path.join(
    ROOT, "benchmarks", "poliblog5k_prepped.csv")

K = int(os.environ.get("POLIBLOG5K_K", "20"))
SPLINE_DF = int(os.environ.get("POLIBLOG5K_SPLINE_DF", "10"))
EM_ITERS = int(os.environ.get("POLIBLOG5K_EM_ITERS", "200"))
EM_TOL = float(os.environ.get("POLIBLOG5K_EM_TOL", "1e-5"))


def r_stm_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


def load_and_prep():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    docs = [r["text"].split() for r in rows]
    lib = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])
    keep = np.array([len(d) > 0 for d in docs])
    docs = [d for d, k in zip(docs, keep) if k]
    return docs, lib[keep], day[keep]


_R_DRIVER = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks)))
vmap  <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d) {
  tb <- table(d); idx <- as.integer(vmap[names(tb)]); o <- order(idx)
  matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow = 2)
})
X <- as.matrix(read.csv(file.path(dir, "design.csv")))
set.seed(1)
f <- stm(documents, vocab, K = KVAL, prevalence = X, init.type = "Spectral",
         verbose = FALSE, emtol = EMTOL, max.em.its = EMITS)
cat("RITERS", length(f$convergence$bound), "\n")
"""


def r_stm_iters(docs, X, names) -> int:
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + names)
            for row in np.column_stack([np.ones(len(docs)), X]):
                w.writerow(list(row))
        script = (f'dir <- "{d}"\nKVAL <- {K}\nEMTOL <- {EM_TOL}\n'
                  f'EMITS <- {EM_ITERS}\n' + _R_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, timeout=7200)
        for line in proc.stdout.splitlines():
            if line.startswith("RITERS"):
                return int(line.split()[1])
        raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr[-2000:]}")


def topica_iters(docs, X, names) -> tuple[int, bool]:
    from topica.models import STM
    m = STM(num_topics=K, init="spectral", seed=1)
    m.fit(docs, X, prevalence_names=names, iters=EM_ITERS, convergence_tol=EM_TOL)
    return len(m.bound_history), bool(m.converged)


def run(verbose: bool = True) -> dict:
    from topica.stm import spline

    docs, lib, day = load_and_prep()
    spl, _ = spline(day, df=SPLINE_DF)

    X_rating = lib.reshape(-1, 1)
    n_rating = ["ratingLiberal"]
    X_spline = np.column_stack([lib, spl])
    n_spline = ["ratingLiberal"] + [f"day_s{j}" for j in range(spl.shape[1])]
    X_qr = np.linalg.qr(X_spline)[0]  # same column space, orthonormal
    n_qr = [f"q{j}" for j in range(X_qr.shape[1])]

    have_r = r_stm_available()
    result = {
        "n_docs": len(docs), "K": K, "spline_df": SPLINE_DF,
        "em_tol": EM_TOL, "have_r": have_r,
        "cond_spline": float(np.linalg.cond(X_spline)),
        "cond_qr": float(np.linalg.cond(X_qr)),
    }

    rows = []
    for label, X, names, run_r in [
        ("rating only", X_rating, n_rating, True),
        ("rating + spline", X_spline, n_spline, True),
        ("rating + spline (QR-ortho)", X_qr, n_qr, False),
    ]:
        ti, conv = topica_iters(docs, X, names)
        ri = r_stm_iters(docs, X, names) if (have_r and run_r) else None
        rows.append((label, ti, conv, ri))
    result["rows"] = rows

    if verbose:
        print(f"poliblog5k: {result['n_docs']} docs, K={K}, spline_df={SPLINE_DF}, "
              f"emtol={EM_TOL}")
        print(f"design condition number: spline={result['cond_spline']:.2e}  "
              f"QR-ortho={result['cond_qr']:.2e}")
        print(f"R stm: {'available' if have_r else 'UNAVAILABLE (R rows skipped)'}\n")
        print(f"{'design':30s} {'topica':>8s} {'R stm':>8s}")
        for label, ti, conv, ri in rows:
            tcol = f"{ti}{'' if conv else '*'}"
            rcol = str(ri) if ri is not None else "-"
            print(f"{label:30s} {tcol:>8s} {rcol:>8s}")
        print("\n* = hit iteration cap without converging")
        print("Reading: pooled is now empirical-Bayes (matches stm's vb.variational.reg),")
        print("which removed the fixed-ridge part of the gap (~37 -> ~32 with spline).")
        print("QR-orthonormalizing (cond -> 1) does not help -> not conditioning; the")
        print("residual gap is a convergence-rate difference on wide designs, not a bug.")
    return result


if __name__ == "__main__":
    import sys
    if not os.path.exists(CSV):
        print(f"SKIP: {CSV} not found (set POLIBLOG5K_CSV)")
        sys.exit(0)
    run(verbose=True)
