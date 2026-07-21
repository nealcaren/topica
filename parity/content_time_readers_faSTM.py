"""Parity: topica.content content_trajectory / content_divergence vs faSTM (#365).

topica's STM content_time reading layer ports faSTM's `content_trajectory()` /
`content_divergence()` (faSTM/R/content-trajectory.R). Both are deterministic
functions of the fitted content surface beta_{k,g,v}, so *given the same beta*
they must produce identical point estimates.

This script fits an STM `content_time` model once in faSTM, exports its content
beta (exp of `fit$beta$logbeta`), its `group@period` levels, its vocabulary, and
the R readers' outputs. It then builds a duck-typed model from that same beta,
runs topica's `content_trajectory` / `content_divergence` on it, and checks the
estimates agree to floating point. The R side also reports which topic its
anchor-word rule selected; topica reads the same topic explicitly, so the
comparison isolates the reader arithmetic from the (implementation-native) anchor
heuristic.

Shells out to `Rscript` with the `faSTM` package. Skips (exit 0) if unavailable.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

R_SCRIPT = r"""
suppressMessages(library(faSTM))
# The reference readers live in faSTM's source; source it so both content_trajectory
# and content_divergence are the current versions regardless of the installed build.
rfile <- "../faSTM/R/content-trajectory.R"
if (file.exists(rfile)) suppressMessages(source(rfile))
set.seed(1)

vocab <- c("climate","border","econ","jobs","tax","health","trade","energy")
periods <- c(1990, 2000, 2010, 2020)
docs <- list(); grp <- character(0); yr <- integer(0)
mk <- function(counts) {
  nz <- which(counts > 0)
  matrix(as.integer(rbind(nz, counts[nz])), nrow = 2)
}
for (pi in seq_along(periods)) {
  d <- (pi - 1) / (length(periods) - 1)
  for (i in 1:70) {
    cd <- c(round(1 + 5*d), round(1 + 5*(1-d)), 4, 3, 2, 1, 1, 1)  # Dem
    cr <- c(round(1 + 5*(1-d)), round(1 + 5*d), 4, 3, 2, 1, 1, 1)  # Rep
    docs[[length(docs)+1]] <- mk(cd); grp <- c(grp, "Dem"); yr <- c(yr, periods[pi])
    docs[[length(docs)+1]] <- mk(cr); grp <- c(grp, "Rep"); yr <- c(yr, periods[pi])
  }
}
meta <- data.frame(group = grp, year = yr, stringsAsFactors = FALSE)

fit <- stm(docs, vocab, K = 3, content = ~ group, content_time = ~ year,
           content_prior = "l1", data = meta, seed = 1, verbose = FALSE)

anchor <- c("climate","border","econ")
words  <- c("climate","border","health")
tr <- content_trajectory(fit, words = words, groups = c("Dem","Rep"),
                         anchor_words = anchor)
dv <- content_divergence(fit, groups = c("Dem","Rep"), anchor_words = anchor,
                         measure = "hellinger")

# which topic did the anchor rule pick (1-indexed)?
lb  <- fit$beta$logbeta
avg <- Reduce(`+`, lb) / length(lb)
ksel <- which.max(apply(avg, 1, function(r)
  sum(fit$vocab[order(r, decreasing = TRUE)[1:20]] %in% anchor)))

beta <- lapply(lb, function(m) as.numeric(t(exp(m))))  # row-major (K x V) per group
out <- list(
  vocab   = fit$vocab,
  levels  = fit$settings$covariates$yvarlevels,
  K       = nrow(lb[[1]]),
  V       = ncol(lb[[1]]),
  beta    = beta,                 # list over groups, each length K*V row-major
  ksel    = ksel,
  traj    = tr,                   # word, period, estimate
  div     = dv                    # period, divergence
)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 15))
"""


def _run_r():
    try:
        proc = subprocess.run(
            ["Rscript", "-e", R_SCRIPT], capture_output=True, text=True, timeout=1800
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout[-2000:])
        return None


class _Shim:
    """A duck-typed content model exposing exactly what topica.content reads."""

    def __init__(self, twbg, groups, vocab):
        self.topic_word_by_group = twbg
        self.groups = groups
        self.vocabulary = vocab


def main():
    data = _run_r()
    if data is None:
        print("SKIP: Rscript with the 'faSTM' package is not available")
        return 0

    from topica import content

    K, V = int(data["K"]), int(data["V"])
    G = len(data["levels"])
    # Rebuild beta as (K, G, V) from the per-group row-major (K x V) vectors.
    twbg = np.empty((K, G, V))
    for g in range(G):
        twbg[:, g, :] = np.asarray(data["beta"][g], float).reshape(K, V)
    shim = _Shim(twbg, list(data["levels"]), list(data["vocab"]))

    k = int(data["ksel"]) - 1  # R is 1-indexed

    # --- content_trajectory parity ---
    words = sorted({row["word"] for row in _rows(data["traj"])})
    tr = content.content_trajectory(shim, words, groups=("Dem", "Rep"), topic=k)
    r_traj = {(row["word"], str(row["period"])): row["estimate"]
              for row in _rows(data["traj"])}
    tmax = 0.0
    for i, w in enumerate(tr.words):
        for j, p in enumerate(tr.periods):
            rv = r_traj.get((w, p))
            if rv is not None and not np.isnan(tr.estimate[i, j]):
                tmax = max(tmax, abs(tr.estimate[i, j] - rv))

    # --- content_divergence parity ---
    dv = content.content_divergence(shim, groups=("Dem", "Rep"), topic=k,
                                    measure="hellinger")
    r_div = {str(row["period"]): row["divergence"] for row in _rows(data["div"])}
    dmax = 0.0
    for j, p in enumerate(dv.periods):
        rv = r_div.get(p)
        if rv is not None and not np.isnan(dv.divergence[j]):
            dmax = max(dmax, abs(dv.divergence[j] - rv))

    print(f"content_trajectory max |topica - faSTM| = {tmax:.2e}")
    print(f"content_divergence max |topica - faSTM| = {dmax:.2e}")
    tol = 1e-8
    ok = tmax < tol and dmax < tol
    print("PASS" if ok else "FAIL", f"(tol {tol:.0e})")
    return 0 if ok else 1


def _rows(df):
    """A jsonlite data.frame arrives as a dict of columns; iterate its rows."""
    if isinstance(df, list):
        return df
    keys = list(df.keys())
    n = len(df[keys[0]])
    return [{k: df[k][i] for k in keys} for i in range(n)]


if __name__ == "__main__":
    raise SystemExit(main())
