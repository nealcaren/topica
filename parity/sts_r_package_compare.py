"""Portable end-to-end validation: topica STS vs a live R `sts`-package fit.

Companion to ``sts_r_compare.py``. That script compares topica against the
authors' *frozen* published fit (``Poliblogs_results.RDS``), which needs the
replication package on disk (``STS_REPL_DIR``). This one is fully portable: it
fits the **R `sts` package** live on the poliblog corpus that ships with `stm`
(``data(poliblog5k)``), so it needs no external data — only the CRAN `sts` and
`stm` packages.

The R `sts` package (Chen & Mankad 2024) and topica's STS are independent
implementations of the same model. The comparison is statistical, like the
STM/CTM R parity: the two use different initializations (R its anchor init,
topica its own), so we do not expect a bit-identical fit — we check that the
aligned topic-word distributions at mean sentiment agree well. topica is fit with
``reference="cran"`` to match R `sts`'s public default (``kappaEstimation=
"adjusted"`` plus the reference kappa damping).

The kappa-solver kernel is separately validated against glmnet in
``sts_kappa_glmnet.py``; this validates the whole EM end to end. Shells out to
``Rscript`` with the `sts` and `stm` packages; skips (exit 0) when any is
unavailable, so it is safe to schedule as an integration job. Run directly:

    python parity/sts_r_package_compare.py

Environment overrides: STS_NDOCS, STS_K, STS_MAXITER, STS_SEED, STS_COSINE_FLOOR.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from itertools import permutations

import numpy as np

import topica

# Defaults match a validated run: 300 poliblog docs, K=5, 8 EM iterations gives an
# aligned topic-word cosine of 0.89 vs the R sts package (floor 0.80). Larger NDOCS
# / K / MAXITER strengthen the test at higher cost.
NDOCS = int(os.environ.get("STS_NDOCS", "300"))
K = int(os.environ.get("STS_K", "5"))
MAXITER = int(os.environ.get("STS_MAXITER", "8"))
SEED = int(os.environ.get("STS_SEED", "1"))
COSINE_FLOOR = float(os.environ.get("STS_COSINE_FLOOR", "0.80"))


def r_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    try:
        out = subprocess.run(
            [
                rscript,
                "-e",
                'cat(requireNamespace("sts",quietly=TRUE) && requireNamespace("stm",quietly=TRUE))',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().upper().endswith("TRUE")


_R_DRIVER = r"""
suppressMessages({library(sts); library(stm)})
args   <- commandArgs(trailingOnly = TRUE)
dir    <- args[1]; ndocs <- as.integer(args[2]); K <- as.integer(args[3])
maxit  <- as.integer(args[4]); seed <- as.integer(args[5])
data(poliblog5k)
idx  <- seq_len(min(ndocs, length(poliblog5k.docs)))
meta <- poliblog5k.meta[idx, ]
meta$sent <- ifelse(meta$rating == "Liberal", 1, -1)
out  <- prepDocuments(poliblog5k.docs[idx], poliblog5k.voc, meta, verbose = FALSE)
fit  <- sts(prevalence_sentiment = ~rating, initializationVar = ~sent, corpus = out,
            K = K, maxIter = maxit, initialization = "anchor",
            kappaEstimation = "adjusted", verbose = FALSE, stmSeed = seed)

# Topic-word at mean sentiment (alpha^(s)=0): softmax(mv + kappa_t[,k]).
bw <- apply(fit$kappa$kappa_t, 2, function(kt) { e <- exp(fit$mv + kt); e / sum(e) })
write.table(t(bw), file.path(dir, "r_beta.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)                  # K x V
writeLines(fit$vocab, file.path(dir, "r_vocab.txt"))              # V

# Emit the exact tokenized corpus + sentiment so topica fits identical data.
con <- file(file.path(dir, "docs.txt"), "w")
for (d in out$documents) {
    toks <- rep(out$vocab[d[1, ]], d[2, ])
    writeLines(paste(toks, collapse = " "), con)
}
close(con)
writeLines(as.character(out$meta$sent), file.path(dir, "sent.txt"))
writeLines(as.character(ifelse(out$meta$rating == "Liberal", 1L, 0L)),
           file.path(dir, "rating.txt"))
cat("ok\n")
"""


def align_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine over the best one-to-one topic alignment (rows of a to b)."""
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-300)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-300)
    sim = an @ bn.T
    k = sim.shape[0]
    best = max(permutations(range(k)), key=lambda p: sum(sim[i, p[i]] for i in range(k)))
    return float(np.mean([sim[i, best[i]] for i in range(k)]))


def run(verbose: bool = True) -> dict:
    with tempfile.TemporaryDirectory() as d:
        driver = os.path.join(d, "driver.R")
        with open(driver, "w") as f:
            f.write(_R_DRIVER)
        proc = subprocess.run(
            [shutil.which("Rscript"), driver, d, str(NDOCS), str(K), str(MAXITER), str(SEED)],
            capture_output=True,
            text=True,
            timeout=2400,
        )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")
        r_beta = np.loadtxt(os.path.join(d, "r_beta.csv"), delimiter=",")
        with open(os.path.join(d, "r_vocab.txt")) as f:
            r_vocab = [w.strip() for w in f]
        with open(os.path.join(d, "docs.txt")) as f:
            docs = [line.split() for line in f]
        sent = [float(x) for x in open(os.path.join(d, "sent.txt")).read().split()]
        rating = [[float(x)] for x in open(os.path.join(d, "rating.txt")).read().split()]

    model = topica.STS(num_topics=K, seed=SEED)
    model.fit(docs, sentiment_seed=sent, prevalence=rating, iters=MAXITER, reference="cran")
    t_beta = np.asarray(model.topic_word)
    t_vocab = list(model.vocabulary)

    # Reindex both topic-word matrices onto the shared vocabulary.
    t_set = set(t_vocab)
    shared = [w for w in r_vocab if w in t_set]
    r_idx = {w: i for i, w in enumerate(r_vocab)}
    t_idx = {w: i for i, w in enumerate(t_vocab)}
    r_aligned = r_beta[:, [r_idx[w] for w in shared]]
    t_aligned = t_beta[:, [t_idx[w] for w in shared]]

    cos = align_cosine(r_aligned, t_aligned)
    metrics = {
        "cosine": cos,
        "n_docs": len(docs),
        "vocab_r": len(r_vocab),
        "vocab_shared": len(shared),
        "num_topics": K,
    }
    if verbose:
        print(f"docs={len(docs)}  R vocab={len(r_vocab)}  shared={len(shared)}  K={K}  iters={MAXITER}")
        print(f"topica-STS(reference='cran') vs R-sts aligned topic-word cosine: {cos:.4f} "
              f"(floor {COSINE_FLOOR})")
    return metrics


def main() -> int:
    if not r_available():
        print("Rscript or the sts/stm packages are unavailable. Skipping R STS-package parity check.")
        return 0
    m = run()
    assert m["cosine"] >= COSINE_FLOOR, (
        f"topica STS diverged from R sts: aligned topic-word cosine {m['cosine']:.4f} < {COSINE_FLOOR}"
    )
    print("OK: topica STS recovers the same poliblog topics as the R sts package.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
