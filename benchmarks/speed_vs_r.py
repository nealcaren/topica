"""Wall-clock speed: topica vs. the R reference packages (`stm`, `keyATM`).

Fits the SAME corpus and settings in R and in topica, timing only the fit call
in each (R-process startup, library load, and preprocessing are excluded), and
prints a speedup table. Both engines are pinned to the same number of
iterations so the comparison is per-unit-work, not convergence-dependent:

  - STM:    fixed EM iterations (R `max.em.its` + `emtol=0`; topica `convergence_tol=0`).
  - keyATM: fixed Gibbs sweeps (the natural iteration unit for both).

topica STM is single-threaded (variational EM); keyATM is timed single-threaded
(the fair R comparison) and multi-threaded (topica's AD-LDA parallel path, which
R has no equivalent of).

Shells out to `Rscript` with `stm` and `keyATM`. Run:

    python benchmarks/speed_vs_r.py

Env knobs: STM_K, STM_EM_ITERS, KEYATM_K, KEYATM_ITERS, KEYATM_THREADS.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "parity"))

import keyatm_r_compare as KA  # noqa: E402
import stm_poliblog_compare as STM  # noqa: E402

STM_K = int(os.environ.get("STM_K", "20"))
STM_EM_ITERS = int(os.environ.get("STM_EM_ITERS", "30"))
KEYATM_K = int(os.environ.get("KEYATM_K", "10"))
KEYATM_ITERS = int(os.environ.get("KEYATM_ITERS", "1000"))
KEYATM_THREADS = int(os.environ.get("KEYATM_THREADS", "4"))
LDA_K = int(os.environ.get("LDA_K", "20"))
LDA_ITERS = int(os.environ.get("LDA_ITERS", "1000"))
LDA_THREADS = int(os.environ.get("LDA_THREADS", "4"))


def _rscript(body: str, timeout=3600) -> str:
    proc = subprocess.run(
        ["Rscript", "-e", body], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0 or "ERR" in proc.stdout:
        raise RuntimeError(f"R failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


# --- STM ---------------------------------------------------------------------

_R_STM = r"""
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
t <- system.time({
  f <- stm(documents, vocab, K = KVAL, prevalence = X, init.type = "Spectral",
           max.em.its = NITERS, emtol = 0, verbose = FALSE)
})
cat("R_TIME", as.numeric(t["elapsed"]), "\n")
"""


def bench_stm() -> dict:
    docs, rating, day, _ = STM.load_and_prep()
    from topica import STM as TopicaSTM
    from topica.stm import spline

    spline_basis, _ = spline(day, df=10)
    X = np.column_stack([rating, spline_basis])
    feat = ["ratingLiberal"] + [f"day_s{j}" for j in range(spline_basis.shape[1])]
    design = np.column_stack([np.ones(len(docs)), X])

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + feat)
            w.writerows(design.tolist())
        out = _rscript(f'dir <- "{d}"\nKVAL <- {STM_K}\nNITERS <- {STM_EM_ITERS}\n' + _R_STM)
    r_time = float([ln for ln in out.splitlines() if ln.startswith("R_TIME")][0].split()[1])

    # topica: same fixed EM iterations (convergence_tol=0 disables early stop).
    t0 = time.perf_counter()
    TopicaSTM(num_topics=STM_K, init="spectral").fit(
        docs, X, prevalence_names=feat, iters=STM_EM_ITERS, convergence_tol=0.0
    )
    tt_time = time.perf_counter() - t0
    return {
        "n_docs": len(docs), "vocab": len({w for dd in docs for w in dd}),
        "k": STM_K, "iters": STM_EM_ITERS, "r": r_time, "tt": tt_time,
    }


# --- keyATM ------------------------------------------------------------------

_R_KEYATM = r"""
suppressMessages({library(keyATM); library(quanteda)})
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
kdocs <- keyATM_read(texts = dfmat)
kw <- jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE)
kw <- lapply(kw, function(x) unlist(x))
t <- system.time({
  out <- keyATM(docs = kdocs, model = "base", no_keyword_topics = NREG, keywords = kw,
                options = list(seed = 1, iterations = NITERS, verbose = FALSE))
})
cat("R_TIME", as.numeric(t["elapsed"]), "\n")
"""


def bench_keyatm() -> dict:
    docs, keywords = KA.load_and_prep()
    from topica import KeyATM

    num_kw = len(keywords)
    nreg = KEYATM_K - num_kw
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        json.dump(keywords, open(os.path.join(d, "keywords.json"), "w"))
        out = _rscript(
            f'dir <- "{d}"\nNREG <- {nreg}\nNITERS <- {KEYATM_ITERS}\n' + _R_KEYATM
        )
    r_time = float([ln for ln in out.splitlines() if ln.startswith("R_TIME")][0].split()[1])

    def topica_fit(threads):
        t0 = time.perf_counter()
        KeyATM(keywords, num_topics=KEYATM_K, seed=1).fit(
            docs, iters=KEYATM_ITERS, num_threads=threads
        )
        return time.perf_counter() - t0

    return {
        "n_docs": len(docs), "vocab": len({w for dd in docs for w in dd}),
        "k": KEYATM_K, "kw": num_kw, "iters": KEYATM_ITERS,
        "r": r_time, "tt1": topica_fit(1), "ttN": topica_fit(KEYATM_THREADS),
    }


# --- LDA vs. Java MALLET -----------------------------------------------------

def mallet_available() -> bool:
    return shutil.which("mallet") is not None


def bench_lda() -> dict:
    """Time MALLET's `train-topics` vs. topica LDA, both with hyperparameter
    optimization off and the same K / iterations (MALLET's import step, which
    topica does not need, is excluded from the timing)."""
    docs = STM.load_and_prep()[0]
    from topica import LDA

    mallet = shutil.which("mallet")
    d = tempfile.mkdtemp()
    try:
        txt = os.path.join(d, "tok.txt")
        with open(txt, "w") as f:
            for i, toks in enumerate(docs):
                f.write(f"doc{i}\t{' '.join(toks)}\n")
        mal = os.path.join(d, "tok.mallet")
        subprocess.run(
            [mallet, "import-file", "--input", txt, "--output", mal, "--keep-sequence",
             "--token-regex", r"\S+", "--line-regex", r"^(\S+)\t(.*)$",
             "--name", "1", "--data", "2", "--label", "0"],
            check=True, capture_output=True, text=True,
        )
        t0 = time.perf_counter()
        subprocess.run(
            [mallet, "train-topics", "--input", mal, "--num-topics", str(LDA_K),
             "--num-iterations", str(LDA_ITERS), "--random-seed", "1",
             "--optimize-interval", "0", "--num-threads", "1"],
            check=True, capture_output=True, text=True,
        )
        mallet_time = time.perf_counter() - t0
    finally:
        shutil.rmtree(d, ignore_errors=True)

    def topica_fit(threads):
        t0 = time.perf_counter()
        LDA(num_topics=LDA_K, seed=1, optimize_interval=0, num_threads=threads).fit(
            docs, iters=LDA_ITERS
        )
        return time.perf_counter() - t0

    return {
        "n_docs": len(docs), "k": LDA_K, "iters": LDA_ITERS,
        "mallet": mallet_time, "tt1": topica_fit(1), "ttN": topica_fit(LDA_THREADS),
    }


def main():
    if not STM.r_stm_available():
        print("SKIP: Rscript with the 'stm' package not available")
        return
    print("Benchmarking (this fits each model in R and topica)...\n")

    s = bench_stm()
    rows = [
        ("STM", "R stm", f"K={s['k']}, {s['iters']} EM its, spectral",
         s["r"], s["tt"], s["r"] / s["tt"], "single-thread (variational EM)"),
    ]

    if mallet_available():
        ll = bench_lda()
        rows.append((
            "LDA", "MALLET", f"K={ll['k']}, {ll['iters']} Gibbs its",
            ll["mallet"], ll["tt1"], ll["mallet"] / ll["tt1"],
            f"1 thread; {LDA_THREADS}-thread: {ll['ttN']:.1f}s "
            f"({ll['mallet'] / ll['ttN']:.1f}x)",
        ))

    if KA.r_keyatm_available():
        k = bench_keyatm()
        rows.append((
            "keyATM", "R keyATM", f"K={k['k']} ({k['kw']} kw), {k['iters']} sweeps",
            k["r"], k["tt1"], k["r"] / k["tt1"],
            f"1 thread; {KEYATM_THREADS}-thread: {k['ttN']:.1f}s "
            f"({k['r'] / k['ttN']:.1f}x)",
        ))
        corpus = f"{k['n_docs']} docs, {k['vocab']} vocab"
    else:
        corpus = f"{s['n_docs']} docs, {s['vocab']} vocab"

    print(f"corpus: {corpus} (poliblog)\n")
    print(f"| {'Model':<7} | {'Reference':<9} | {'Settings':<30} | {'ref':>7} | {'topica':>7} | {'speedup':>7} | notes |")
    print(f"|{'-'*9}|{'-'*11}|{'-'*32}|{'-'*9}|{'-'*9}|{'-'*9}|-------|")
    for name, ref, settings, rt, tt, sp, notes in rows:
        print(f"| {name:<7} | {ref:<9} | {settings:<30} | {rt:6.1f}s | {tt:6.1f}s | {sp:5.1f}x | {notes} |")


if __name__ == "__main__":
    main()
