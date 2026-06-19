"""Benchmark topica's STM per-iteration fit cost against R ``stm``.

This is the *mechanism* benchmark: it isolates per-iteration engine efficiency by
running both engines a fixed number of EM iterations from a common Spectral init
(R with ``max.em.its=iters, emtol=0`` so it does not stop early; topica with
``convergence_tol=0``), on synthetic corpora, on a realistic covariate-rich
design (``~ rating + s(day)``, a day spline). It answers "how fast is one EM
iteration," not "how long to converge."

The wall-clock a user actually waits is time *to convergence*, reported on real
text by ``speed_vs_size.py`` / ``speed_vs_r.py`` (the headline). The two differ
because topica converges in somewhat more EM iterations than R ``stm`` on a wide
spline design (issue #247), so this per-iteration ratio is larger than the
to-convergence ratio; we report it as the mechanism behind the headline. Synthetic
corpora are used here precisely because per-iteration cost is design-controlled
and reproducible; convergence behavior is only meaningful on real text, which is
why the to-convergence numbers live in the real-data scripts.

R ``stm`` is single-threaded by design; topica's variational E-step uses all
cores by default — set ``RAYON_NUM_THREADS=1`` for the single-core comparison.

Run::

    python benchmarks/bench_stm.py                      # topica on all cores
    RAYON_NUM_THREADS=1 python benchmarks/bench_stm.py  # single-threaded

The synthetic corpora are fixed-seed, so results are reproducible (timings vary
with hardware).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from topica import STM
from topica.stm import spline

# (num_docs, vocab, num_topics) — a small/moderate/large-vocab sweep.
CONFIGS = [
    (1000, 500, 10),
    (2000, 2000, 10),
    (5000, 5000, 20),
]
EM_ITERS = 30          # fixed iteration budget, both engines (per-iteration cost)
SPLINE_DF = 10         # day-spline basis dimension (a realistic covariate design)
TOKENS_PER_DOC = 60


def synthetic_corpus(v, d, k_true, seed=0, length=TOKENS_PER_DOC):
    """A fixed-seed corpus of `d` docs over vocab `v` from `k_true` planted
    topics, with a realistic prevalence design: a binary ``rating`` correlated
    with topic 0 and a continuous ``day`` whose spline basis enters the model."""
    rng = np.random.default_rng(seed)
    beta = np.zeros((k_true, v))
    band = v // k_true
    for kk in range(k_true):
        cols = np.arange(kk * band, (kk + 1) * band) % v
        beta[kk, cols] = 1.0
        beta[kk] += 0.01
        beta[kk] /= beta[kk].sum()
    docs, rating, day = [], [], []
    for _ in range(d):
        theta = rng.dirichlet(np.ones(k_true) * 0.3)
        z = rng.choice(k_true, size=length, p=theta)
        docs.append([f"w{int(rng.choice(v, p=beta[zz]))}" for zz in z])
        rating.append(1.0 if theta[0] > 0.2 else 0.0)  # correlated with topic 0
        day.append(float(rng.integers(1, 366)))
    basis, _ = spline(np.array(day), df=SPLINE_DF)
    x = np.column_stack([np.array(rating), basis])
    names = ["rating"] + [f"day_s{j}" for j in range(basis.shape[1])]
    return docs, x, names


def time_topica(docs, x, names, k, iters):
    t0 = time.perf_counter()
    m = STM(num_topics=k, init="spectral", seed=1)
    # convergence_tol=0 disables early stop, so the full iters budget runs.
    m.fit(docs, x, prevalence_names=names, iters=iters, convergence_tol=0.0)
    return time.perf_counter() - t0


def r_stm_available():
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


_R_DRIVER = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "docs.txt")); toks <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks))); vmap <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d) {
  tb <- table(d); idx <- as.integer(vmap[names(tb)]); o <- order(idx)
  matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow = 2)
})
X <- as.matrix(read.csv(file.path(dir, "design.csv")))
el <- system.time(
  fit <- stm(documents, vocab, K = K, prevalence = X,
             init.type = "Spectral", max.em.its = ITERS, emtol = 0, verbose = FALSE)
)["elapsed"]
cat(sprintf("ELAPSED %f\n", el))
"""


def time_r_stm(docs, x, names, k, iters):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "docs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + names)
            for row in np.column_stack([np.ones(len(docs)), x]):
                w.writerow(list(row))
        script = f'dir <- "{d}"\nK <- {k}\nITERS <- {iters}\n' + _R_DRIVER
        out = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=1800
        )
        for line in out.stdout.splitlines():
            if line.startswith("ELAPSED"):
                return float(line.split()[1])
        raise RuntimeError(f"R stm failed:\n{out.stdout}\n{out.stderr}")


def main():
    threads = os.environ.get("RAYON_NUM_THREADS", f"all ({os.cpu_count()} cores)")
    have_r = r_stm_available()
    print(f"topica threads: {threads};  design: ~rating + s(day, df={SPLINE_DF});  "
          f"fixed {EM_ITERS} EM iterations (per-iteration cost);  "
          f"R stm: {'available' if have_r else 'not found (topica only)'}\n")
    header = f"{'docs':>6} {'vocab':>6} {'K':>3} | {'topica':>10} {'ms/it':>6}"
    if have_r:
        header += f" {'R stm':>10} {'ms/it':>7} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for d, v, k in CONFIGS:
        docs, x, names = synthetic_corpus(v, d, k_true=max(k, 20))
        tt = time_topica(docs, x, names, k, EM_ITERS)
        row = f"{d:>6} {v:>6} {k:>3} | {tt:>9.2f}s {tt / EM_ITERS * 1000:>6.0f}"
        if have_r:
            rt = time_r_stm(docs, x, names, k, EM_ITERS)
            row += f" {rt:>9.2f}s {rt / EM_ITERS * 1000:>7.0f} {rt / tt:>7.1f}x"
        print(row)
    print(f"\nspeedup = per-iteration cost ratio (both run {EM_ITERS} fixed iters); "
          "to-convergence wall-clock is in speed_vs_size.py")


if __name__ == "__main__":
    main()
