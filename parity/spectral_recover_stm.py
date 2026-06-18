"""Parity: topica's spectral init vs R stm's spectral start (issues #234, #240).

This is an END-TO-END check of topica's whole spectral pipeline — its own
`cooccurrence()` + `fast_anchor_words()` + `recover()` (src/spectral.rs) — against
R `stm`'s, NOT just the `recover()` step. It drives R to tokenize gadarian via
`textProcessor`/`prepDocuments`, compute the spectral topic-word matrix through the
QP reference path (`recoverEG=FALSE` — the converged optimum, not stm's
non-converging default exponentiated-gradient), and emit those exact prepped
documents; then it fits topica's spectral init (`CTM(init="spectral").fit(iters=0)`,
which returns the raw spectral β unchanged) on the SAME documents. Because no stm
intermediates are substituted, agreement here means topica's cooccurrence, anchors,
and recovery all reproduce stm's given identical input.

On gadarian K=5 the per-topic (same-order) cosine is ~1.0: topica's default
spectral start equals stm's. (Pre-#234 the recover() step alone was ~0.37 because
its exponentiated gradient diverged at a fixed eta=50.) #240 note: an end-to-end
gap observed downstream comes from feeding a *differently prepared* corpus, not
from the spectral algorithm — given identical prepped input, topica matches stm.

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

    # CTM(init="spectral").fit(iters=0) returns the pure spectral-init topic-word
    # *from topica's own full pipeline* (its cooccurrence() + fast_anchor_words() +
    # recover()), with no EM and no stm intermediates substituted. So this is an
    # end-to-end check, not just a recover() check — given the SAME stm-prepped
    # documents, topica's whole spectral start should equal stm's.
    m = topica.CTM(num_topics=K, seed=1, init="spectral")
    m.fit(docs, iters=0)
    tb = np.asarray(m.topic_word)
    tvocab = list(m.vocabulary)

    # align both to the shared vocabulary
    stm_idx = {w: i for i, w in enumerate(stm_vocab)}
    shared = [w for w in tvocab if w in stm_idx]
    tcols = [tvocab.index(w) for w in shared]
    scols = [stm_idx[w] for w in shared]
    A = tb[:, tcols]
    B = stm_beta[:, scols]
    # Per-topic (diagonal, un-permuted) cosine: tests that topica reproduces stm's
    # spectral topics in the SAME order, not merely the same set. Hungarian is the
    # weaker fallback in case a future change permutes topic order.
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    diag_cos = float(np.mean(np.sum(An * Bn, axis=1)))
    matched_cos = _hungarian_cosine(B, A)
    print(f"shared vocab: {len(shared)} words | topica V={len(tvocab)} stm V={len(stm_vocab)}")
    print(f"full spectral_init vs R stm spectral start (cooccurrence+anchor+recover):")
    print(f"  per-topic (same-order) cosine = {diag_cos:.4f}")
    print(f"  matched (Hungarian)    cosine = {matched_cos:.4f}")
    ok = diag_cos >= 0.95 and matched_cos >= 0.95
    print("PASS" if ok else "FAIL (expected >= 0.95)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
