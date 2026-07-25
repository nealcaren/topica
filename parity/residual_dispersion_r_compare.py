"""Cross-implementation validation: topica's residual dispersion vs. R `stm`'s
`checkResiduals` (Taddy 2012).

Unlike a fit comparison, this isolates the *metric*: we fit one STM model in R,
call `stm::checkResiduals` on it, then export that model's exact `theta`, `beta`
(= `exp(logbeta)`), and integer-coded documents and feed the identical arrays to
topica-core's `inspect::residual_dispersion` (via `topica.check_residuals`'s
binding). Because both sides consume the same fitted quantities, the dispersion,
degrees of freedom, and chi-squared statistic should agree to floating-point
noise. Any gap is a faithfulness bug in topica's port of stm's residual formula
or its degrees-of-freedom accounting (`df = Nhat - V - n*(K-1) - K*(V-1)`).

Shells out to `Rscript` with the `stm` package. Skips (exit 0) if R or `stm` is
unavailable. Run directly:

    python parity/residual_dispersion_r_compare.py
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
GADARIAN = os.path.join(ROOT, "examples", "gadarian.csv")
STOPLIST = os.path.join(ROOT, "examples", "english-stoplist.txt")


def r_stm_available() -> bool:
    """True iff `Rscript` is on PATH and the `stm` package loads."""
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


R_SCRIPT = r"""
suppressMessages(library(stm))
args <- commandArgs(trailingOnly=TRUE)
csv <- args[1]; stoplist <- args[2]; outdir <- args[3]
d <- read.csv(csv, stringsAsFactors=FALSE)
stops <- readLines(stoplist, warn=FALSE)
proc <- textProcessor(d$open.ended.response, metadata=d,
                      customstopwords=stops, verbose=FALSE)
out <- prepDocuments(proc$documents, proc$vocab, proc$meta, verbose=FALSE)
set.seed(2138)
mod <- stm(out$documents, out$vocab, K=3, prevalence=~treatment,
           data=out$meta, init.type="Spectral", max.em.its=25, verbose=FALSE)
rc <- checkResiduals(mod, out$documents, tol=0.01)

theta <- mod$theta                       # N x K
beta  <- exp(mod$beta$logbeta[[1]])      # K x V
write.table(theta, file.path(outdir, "theta.csv"), sep=",",
            row.names=FALSE, col.names=FALSE)
write.table(beta,  file.path(outdir, "beta.csv"),  sep=",",
            row.names=FALSE, col.names=FALSE)
# documents: one line per doc, space-separated 0-based token ids (repeated by count)
con <- file(file.path(outdir, "docs.txt"), "w")
for (doc in out$documents) {
  ids <- rep(doc[1, ] - 1L, doc[2, ])
  writeLines(paste(ids, collapse=" "), con)
}
close(con)
writeLines(sprintf("%.10f %.10f %.10f", rc$dispersion, rc$df, rc$pvalue),
           file.path(outdir, "rc.txt"))
"""


def main() -> int:
    if not r_stm_available():
        print("SKIP: Rscript or the `stm` package is unavailable.")
        return 0
    if not os.path.exists(GADARIAN):
        print("SKIP: examples/gadarian.csv not found.")
        return 0

    from topica._topica import inspect_residual_dispersion
    from topica.validation import _chisq_sf

    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "run.R")
        with open(script, "w") as f:
            f.write(R_SCRIPT)
        proc = subprocess.run(
            ["Rscript", script, GADARIAN, STOPLIST, td],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            print("SKIP: R run failed:\n" + proc.stderr[-2000:])
            return 0

        theta = np.loadtxt(os.path.join(td, "theta.csv"), delimiter=",", ndmin=2)
        beta = np.loadtxt(os.path.join(td, "beta.csv"), delimiter=",", ndmin=2)
        with open(os.path.join(td, "docs.txt")) as f:
            docs = [[int(x) for x in line.split()] for line in f if line.strip()]
        with open(os.path.join(td, "rc.txt")) as f:
            r_disp, r_df, r_pval = (float(x) for x in f.read().split())

    disp, df, num_params, statistic, nhat = inspect_residual_dispersion(
        beta.tolist(), theta.tolist(), docs, 0.01
    )
    pval = _chisq_sf(statistic, df) if df > 0 else float("nan")

    print(f"R   stm::checkResiduals: dispersion={r_disp:.6f} df={r_df:.0f} pvalue={r_pval:.4g}")
    print(f"topica residual_dispersion: dispersion={disp:.6f} df={df:.0f} pvalue={pval:.4g}")
    ok_disp = np.isclose(disp, r_disp, rtol=1e-4, atol=1e-6)
    ok_df = np.isclose(df, r_df, atol=0.5)
    print(f"dispersion match: {ok_disp}   df match: {ok_df}")
    if not (ok_disp and ok_df):
        raise SystemExit("MISMATCH: topica residual dispersion diverges from stm.")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
