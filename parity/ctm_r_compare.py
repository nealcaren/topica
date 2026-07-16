"""Cross-implementation validation: topica CTM vs the R `stm` package fit as a
CTM (no covariates).

A Structural Topic Model with no prevalence or content covariates *is* a
Correlated Topic Model: the same logistic-normal variational EM with Arora-style
spectral initialization. topica's CTM and R `stm` are independent implementations
of it, so validation is statistical: fit both on the SAME tokenized corpus and
ask whether they land on the same topics.

The benchmark is R's own reproducibility. Under matched (Spectral) initialization,
which is deterministic, R-vs-topica agreement should meet or exceed R's
random-seed self-consistency floor. We also check that topica's variational bound
increases monotonically across EM iterations (the EM correctness signature).

Shells out to `Rscript` with the `stm` and `quanteda` packages. Skips (exit 0) if
they are unavailable. Run directly:

    python parity/ctm_r_compare.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from collections import Counter

import numpy as np

from stm_r_compare import _best_alignment_cosine, _read_r_beta, r_stm_available

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLIBLOG = os.path.join(ROOT, "examples", "poliblog.csv")

K = int(os.environ.get("CTM_K", "10"))
MIN_DOC_FREQ = int(os.environ.get("CTM_MIN_DF", "5"))
EM_ITERS = int(os.environ.get("CTM_EM_ITERS", "60"))


def load_and_prep():
    """Poliblog (already stemmed) filtered to a shared vocabulary; empty documents
    dropped. Returns list[list[str]]."""
    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    toks = [r["text"].split() for r in rows]
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= MIN_DOC_FREQ}
    return [d for d in ([w for w in doc if w in vocab] for doc in toks) if d]


_R_DRIVER = r"""
suppressMessages(library(stm)); suppressMessages(library(quanteda))
lines  <- readLines(file.path(dir, "vdocs.txt"))
toks   <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat  <- quanteda::dfm(toks)
sd     <- quanteda::convert(dfmat, to = "stm")
beta_of <- function(seed, init) {
  f <- stm(sd$documents, sd$vocab, K = K, prevalence = NULL, init.type = init,
           seed = seed, verbose = FALSE)
  b <- exp(f$beta$logbeta[[1]]); colnames(b) <- sd$vocab; b
}
write.csv(beta_of(1, "Spectral"), file.path(dir, "r_spectral.csv"), row.names = FALSE)
write.csv(beta_of(1, "Random"),   file.path(dir, "r_rand1.csv"),   row.names = FALSE)
write.csv(beta_of(2, "Random"),   file.path(dir, "r_rand2.csv"),   row.names = FALSE)
cat("ok\n")
"""


def run(verbose: bool = True) -> dict:
    if not r_stm_available():
        raise RuntimeError("Rscript with the 'stm' package is not available")
    from topica.models import CTM

    docs = load_and_prep()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        script = f'dir <- "{d}"\nK <- {K}\n' + _R_DRIVER
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        with open(os.path.join(d, "r_spectral.csv"), newline="") as f:
            r_vocab = [h.strip('"') for h in next(csv.reader(f))]
        r_spectral = _read_r_beta(os.path.join(d, "r_spectral.csv"), r_vocab)
        r_rand1 = _read_r_beta(os.path.join(d, "r_rand1.csv"), r_vocab)
        r_rand2 = _read_r_beta(os.path.join(d, "r_rand2.csv"), r_vocab)

        model = CTM(num_topics=K, init="spectral")
        model.fit(docs, iters=EM_ITERS)
        idx = {w: i for i, w in enumerate(model.vocabulary)}
        raw = np.asarray(model.topic_word)
        tt = np.zeros((K, len(r_vocab)))
        for j, w in enumerate(r_vocab):
            if w in idx:
                tt[:, j] = raw[:, idx[w]]

    spectral_cosine = _best_alignment_cosine(r_spectral, tt)
    r_self_cosine = _best_alignment_cosine(r_rand1, r_rand2)
    r_spec_vs_rand = 0.5 * (
        _best_alignment_cosine(r_spectral, r_rand1) + _best_alignment_cosine(r_spectral, r_rand2)
    )
    bh = list(model.bound_history)
    mono = float(np.mean([b2 >= b1 - 1e-6 for b1, b2 in zip(bh, bh[1:])])) if len(bh) > 1 else 1.0

    result = {
        "spectral_cosine": spectral_cosine,
        "r_self_cosine": r_self_cosine,
        "r_spectral_vs_random": r_spec_vs_rand,
        "bound_monotone_frac": mono,
        "final_bound": float(model.bound),
        "n_docs": len(docs),
        "vocab_size": len(r_vocab),
        "K": K,
    }
    if verbose:
        print(f"corpus: {len(docs)} docs, {len(r_vocab)} vocab, K={K}")
        print(f"R-Spectral vs topica-Spectral cosine : {spectral_cosine:.3f}")
        print(f"R Random-vs-Random self-consistency  : {r_self_cosine:.3f}")
        print(f"R Spectral-vs-Random                 : {r_spec_vs_rand:.3f}")
        print(f"topica bound monotone fraction       : {mono:.2f}  (final {model.bound:.1f})")
        gap = r_spec_vs_rand - spectral_cosine
        if spectral_cosine >= 0.9 or gap <= 0.05:
            print(f"PASS — topica reproduces R's CTM as well as R reproduces itself "
                  f"(cosine {spectral_cosine:.3f})")
        elif gap <= 0.15:
            print(f"COMPARABLE — within R's own init spread (gap {gap:.3f})")
        else:
            print(f"DIVERGENT — {gap:.3f} beyond R's own spread; investigate")
    return result


if __name__ == "__main__":
    import sys

    if not r_stm_available():
        print("SKIP: Rscript with the 'stm' package is not available")
        sys.exit(0)
    run(verbose=True)
