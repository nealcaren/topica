"""Parity: topica's spectral-init recovery vs R stm's recoverL2 (issue #234).

topica's `recover()` (src/spectral.rs) is the Arora anchor-word recovery step. This
checks it reproduces R `stm`'s reference recovery on an identical corpus: it drives
R to tokenize gadarian, compute its spectral topic-word matrix via the QP reference
path (`recoverEG=FALSE` — the converged optimum, not stm's non-converging default
exponentiated-gradient), and emit the prepped documents; then it fits topica's
spectral init (CTM, iters=0) on those same documents and reports the Hungarian-
matched topic-word cosine. Pre-#234 this was ~0.37 (topica's EG diverged at the
fixed eta=50); the fix targets ~1.0.

Skips cleanly if Rscript or the stm/Matrix packages are unavailable.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


R_SCRIPT = r"""
ok <- requireNamespace("stm", quietly=TRUE) && requireNamespace("Matrix", quietly=TRUE)
if (!ok) { cat("SKIP: stm/Matrix not installed\n"); quit(status=0) }
suppressMessages({library(stm); library(Matrix)})
args <- commandArgs(trailingOnly=TRUE); outdir <- args[1]; K <- as.integer(args[2])
data("gadarian", package="stm")
proc <- textProcessor(gadarian$open.ended.response, metadata=gadarian, verbose=FALSE)
out  <- prepDocuments(proc$documents, proc$vocab, proc$meta, verbose=FALSE)
docs <- out$documents; vocab <- out$vocab; V <- length(vocab)
# documents as word-string lists (same tokens topica will see)
lines <- vapply(docs, function(m) paste(rep(vocab[m[1,]], m[2,]), collapse=" "), character(1))
writeLines(lines, file.path(outdir, "docs.txt"))
# stm spectral topic-word via the QP reference path (the converged optimum)
rows<-integer(0);cols<-integer(0);vals<-integer(0)
for(i in seq_along(docs)){m<-docs[[i]];rows<-c(rows,rep(i,ncol(m)));cols<-c(cols,m[1,]);vals<-c(vals,m[2,])}
mat <- sparseMatrix(i=rows,j=cols,x=vals,dims=c(length(docs),V))
Q <- stm:::gram(mat); Qsums <- rowSums(Q); Qbar <- Q/Qsums
anchors <- stm:::fastAnchor(Qbar, K, verbose=FALSE)
beta <- stm:::recoverL2(Qbar, anchors, Qsums/sum(Qsums), verbose=FALSE, recoverEG=FALSE)$A
writeLines(vocab, file.path(outdir, "vocab.txt"))
write.table(beta, file.path(outdir, "beta.csv"), sep=",", row.names=FALSE, col.names=FALSE)
cat("OK\n")
"""


def _hungarian_cosine(A, B):
    from scipy.optimize import linear_sum_assignment
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


def main():
    if shutil.which("Rscript") is None:
        print("SKIP: Rscript not found")
        return 0
    import topica
    K = 5
    with tempfile.TemporaryDirectory() as d:
        rs = os.path.join(d, "gen.R")
        with open(rs, "w") as f:
            f.write(R_SCRIPT)
        res = subprocess.run(["Rscript", rs, d, str(K)], capture_output=True, text=True)
        if "SKIP" in res.stdout:
            print(res.stdout.strip())
            return 0
        if res.returncode != 0 or not os.path.exists(os.path.join(d, "beta.csv")):
            print("SKIP: R step failed\n" + res.stdout + res.stderr)
            return 0
        docs = [ln.split() for ln in open(os.path.join(d, "docs.txt")) if ln.strip()]
        stm_vocab = [w.strip() for w in open(os.path.join(d, "vocab.txt"))]
        stm_beta = np.loadtxt(os.path.join(d, "beta.csv"), delimiter=",")  # (K, V_stm)

    m = topica.CTM(num_topics=K, seed=1, init="spectral")
    m.fit(docs, iters=0)  # iters=0 returns the pure spectral-init topic-word
    tb = np.asarray(m.topic_word)
    tvocab = list(m.vocabulary)

    # align both to the shared vocabulary
    stm_idx = {w: i for i, w in enumerate(stm_vocab)}
    shared = [w for w in tvocab if w in stm_idx]
    tcols = [tvocab.index(w) for w in shared]
    scols = [stm_idx[w] for w in shared]
    A = tb[:, tcols]
    B = stm_beta[:, scols]
    cos = _hungarian_cosine(B, A)
    print(f"shared vocab: {len(shared)} words | topica V={len(tvocab)} stm V={len(stm_vocab)}")
    print(f"spectral recover() vs R stm recoverL2 (QP path): matched cosine = {cos:.4f}")
    ok = cos >= 0.95
    print("PASS" if ok else "FAIL (expected >= 0.95)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
