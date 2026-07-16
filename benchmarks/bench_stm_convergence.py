"""STM wall-clock TO CONVERGENCE on real text: topica vs R ``stm``.

This is the *headline* STM benchmark: the time a user actually waits. Both engines
run to their own convergence (the same relative-bound ``emtol``) on a realistic
covariate-rich prevalence design, on real corpora, and we report wall-clock plus
the EM iteration count each engine needed. topica's per-iteration cost is far
lower (see the controlled synthetic comparison in ``bench_stm.py``), but on a wide
spline design it converges in somewhat more EM iterations than R ``stm`` (issue
#247), so the per-iteration speedup overstates this wall-clock-to-convergence
speedup. This script measures the honest, user-facing number.

Corpora (set ``CORPUS``):
  - ``poliblog5k`` (default): the bundled 5k poliblog (``~ rating + s(day)``).
    Build it once with the R snippet in ``speed_vs_size.py`` /
    ``benchmarks/export_poliblog5k`` -> ``benchmarks/poliblog5k_prepped.csv``.
  - ``congress``: the medium ~25k Congress subset (``~ party + s(congress)``).
    Build it with ``python benchmarks/export_congress.py`` (needs the ECTM
    corpus; see that script) -> ``benchmarks/congress_prepped.csv``.

topica is timed single-core (``num_threads=1``, the apples-to-apples comparison to
single-threaded R ``stm``) and at a representative consumer core cap (``THREADS``,
default 16 — not every core of a big HPC node, which would report a speedup nobody
on a normal machine sees).

Run::

    python benchmarks/bench_stm_convergence.py                  # poliblog5k
    CORPUS=congress python benchmarks/bench_stm_convergence.py  # ~25k Congress

Writes ``benchmarks/stm_convergence_results.json``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-corpus config: csv path, group covariate (binary) and its positive level,
# continuous covariate for the spline, and the model's K.
CORPORA = {
    "poliblog5k": dict(csv="poliblog5k_prepped.csv", group="rating", pos="Liberal",
                       cont="day", spline_df=10),
    "congress": dict(csv="congress_prepped.csv", group="party", pos="D",
                     cont="congress", spline_df=6),  # 12 sessions -> a smaller basis
}
CORPUS = os.environ.get("CORPUS", "poliblog5k")
K = int(os.environ.get("STM_K", "20"))
EM_TOL = float(os.environ.get("STM_EM_TOL", "1e-5"))
MAX_EM_ITERS = int(os.environ.get("STM_MAX_EM_ITERS", "500"))
# The "all-cores" run is capped at a representative consumer core count rather
# than every core of a big HPC node (a 44-core node flatters topica with speedups
# nobody on a normal machine sees). Default 16; override with THREADS.
THREADS = int(os.environ.get("THREADS", str(min(16, os.cpu_count() or 16))))


def corpus_path(cfg):
    return os.path.join(HERE, cfg["csv"])


def load(cfg):
    path = corpus_path(cfg)
    rows = list(csv.DictReader(open(path, newline="")))
    docs = [r["text"].split() for r in rows]
    group = np.array([1.0 if r[cfg["group"]] == cfg["pos"] else 0.0 for r in rows])
    cont = np.array([float(r[cfg["cont"]]) for r in rows])
    keep = np.array([len(d) > 0 for d in docs])
    docs = [d for d, k in zip(docs, keep) if k]
    return docs, group[keep], cont[keep]


def design(group, cont, cfg):
    from topica.stm import spline
    basis, _ = spline(cont, df=cfg["spline_df"])
    x = np.column_stack([group, basis])
    names = [cfg["group"]] + [f"{cfg['cont']}_s{j}" for j in range(basis.shape[1])]
    return x, names


def time_topica(docs, x, names, num_threads):
    from topica.models import STM
    m = STM(num_topics=K, init="spectral", seed=1)
    t0 = time.perf_counter()
    m.fit(docs, x, prevalence_names=names, iters=MAX_EM_ITERS,
          convergence_tol=EM_TOL, num_threads=num_threads)
    return time.perf_counter() - t0, len(m.bound_history), bool(m.converged)


def r_stm_available():
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
                             capture_output=True, text=True, timeout=60)
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
t <- system.time(
  fit <- stm(documents, vocab, K = K, prevalence = X, init.type = "Spectral",
             max.em.its = ITERS, emtol = EMTOL, verbose = FALSE)
)["elapsed"]
cat(sprintf("R_RESULT %f %d\n", as.numeric(t), length(fit$convergence$bound)))
"""


def time_r_stm(docs, x, names):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "docs.txt"), "w") as f:
            f.write("\n".join(" ".join(x) for x in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + names)
            for row in np.column_stack([np.ones(len(docs)), x]):
                w.writerow(list(row))
        script = (f'dir <- "{d}"\nK <- {K}\nITERS <- {MAX_EM_ITERS}\n'
                  f'EMTOL <- {EM_TOL}\n' + _R_DRIVER)
        out = subprocess.run(["Rscript", "-e", script], capture_output=True,
                             text=True, timeout=14400)
        for line in out.stdout.splitlines():
            if line.startswith("R_RESULT"):
                p = line.split()
                return float(p[1]), int(p[2])
        raise RuntimeError(f"R stm failed:\n{out.stdout}\n{out.stderr[-2000:]}")


def main():
    cfg = CORPORA[CORPUS]
    if not os.path.exists(corpus_path(cfg)):
        print(f"SKIP: {corpus_path(cfg)} not found "
              f"(build it — see this script's docstring; CORPUS={CORPUS})")
        return
    docs, group, cont = load(cfg)
    x, names = design(group, cont, cfg)
    have_r = r_stm_available()
    print(f"corpus={CORPUS}: {len(docs)} docs, K={K}, "
          f"~{cfg['group']} + s({cfg['cont']}, df={cfg['spline_df']}), "
          f"to convergence (emtol={EM_TOL})\n")

    ts, its, cs = time_topica(docs, x, names, 1)
    tm, itm, cm = time_topica(docs, x, names, THREADS)
    res = {"corpus": CORPUS, "n_docs": len(docs), "K": K, "em_tol": EM_TOL,
           "threads": THREADS,
           "topica_single_s": ts, "topica_single_iters": its,
           "topica_multi_s": tm, "topica_multi_iters": itm,
           "topica_converged": cs and cm}
    print(f"topica single-core   : {ts:8.2f}s  ({its} iters)")
    print(f"topica {THREADS}-core{'':<{max(0,6-len(str(THREADS)))}}: {tm:8.2f}s  ({itm} iters)")
    if have_r:
        rt, ri = time_r_stm(docs, x, names)
        res.update({"r_s": rt, "r_iters": ri,
                    "speedup_single": rt / ts, "speedup_multi": rt / tm})
        print(f"R stm (1 thread)     : {rt:8.2f}s  ({ri} iters)")
        print(f"\nto-convergence speedup:  single {rt / ts:.1f}x   {THREADS}-core {rt / tm:.1f}x")
    else:
        print("\nR stm not available — topica numbers only")
    out = os.path.join(HERE, "stm_convergence_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
