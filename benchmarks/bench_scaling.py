"""Topic-count (K) scaling benchmark for the logistic-normal variational fit.

Shows where topica 0.17.0's memory and speed options matter: the per-document
variational covariance is O(N*K^2), so it dominates at large K. This sweeps K at a
fixed corpus and measures, for each variant, the fit time and peak RSS:

  * laplace, keep_eta_cov=True   -- full covariance stored (grows ~K^2)
  * laplace, keep_eta_cov=False  -- covariance recomputed on demand (flat in K)
  * diagonal, keep_eta_cov=True  -- mean-field covariance, skips the O(K^3) inverse

Each fit runs in a subprocess wrapped with /usr/bin/time so the RSS is just that
fit's footprint. Fixed-seed synthetic corpus, so it is reproducible (absolute
timings vary with hardware).

Run::

    python benchmarks/bench_scaling.py            # sweep + render fig_scaling.pdf
    python benchmarks/bench_scaling.py --render    # re-render from saved JSON
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
RESULTS = os.path.join(HERE, "scaling_results.json")

N_DOCS = 5000
VOCAB = 1000
DOC_LEN = 60
ITERS = 10
KS = [20, 50, 100, 200]
VARIANTS = ["laplace_keep", "laplace_nokeep", "diagonal_keep"]


def _corpus(n=N_DOCS, vocab=VOCAB, doc_len=DOC_LEN, seed=0):
    """Seeded synthetic corpus with a continuous prevalence covariate: two word
    blocks, the covariate tilts a document toward one block."""
    rng = np.random.default_rng(seed)
    half = vocab // 2
    a = [f"a{i}" for i in range(half)]
    b = [f"b{i}" for i in range(vocab - half)]
    docs, xs = [], []
    for _ in range(n):
        x = rng.random()
        xs.append(x)
        p = 0.85 if x > 0.5 else 0.15
        words = [(rng.choice(a) if rng.random() < p else rng.choice(b)) for _ in range(doc_len)]
        docs.append(list(words))
    return docs, np.array(xs).reshape(-1, 1)


def _worker(k: int, variant: str) -> None:
    """Fit one (K, variant) and print FIT_TIME; invoked under /usr/bin/time."""
    import topica

    docs, x = _corpus()
    mode = "diagonal" if variant.startswith("diagonal") else "laplace"
    keep = not variant.endswith("nokeep")
    t = time.perf_counter()
    m = topica.models.STM(num_topics=k, seed=1, variational=mode)
    m.fit(docs, prevalence=x, iters=ITERS, keep_eta_cov=keep)
    _ = m.topic_word  # touch the result
    print(f"FIT_TIME {time.perf_counter() - t:.4f}")


def _peak_rss_mb(argv: list[str]) -> tuple[str, float]:
    cmd = (["/usr/bin/time", "-l"] if sys.platform == "darwin"
           else ["/usr/bin/time", "-v"]) + argv
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and "FIT_TIME" not in proc.stdout:
        raise RuntimeError(f"worker failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    rss = float("nan")
    for line in proc.stderr.splitlines():
        if "maximum resident set size" in line.lower():
            try:
                rss = (int(line.split()[0]) / 1024 / 1024 if sys.platform == "darwin"
                       else int(line.split()[-1]) / 1024)
            except (ValueError, IndexError):
                pass
            break
    fit_time = float("nan")
    for line in proc.stdout.splitlines():
        if line.startswith("FIT_TIME"):
            fit_time = float(line.split()[1])
    return fit_time, rss


def sweep() -> dict:
    out: dict = {"n_docs": N_DOCS, "vocab": VOCAB, "iters": ITERS, "ks": KS, "variants": {}}
    py = sys.executable
    for variant in VARIANTS:
        out["variants"][variant] = {"time": [], "rss": []}
        for k in KS:
            argv = [py, os.path.abspath(__file__), "--worker", str(k), variant]
            ft, rss = _peak_rss_mb(argv)
            out["variants"][variant]["time"].append(ft)
            out["variants"][variant]["rss"].append(rss)
            print(f"  K={k:4d} {variant:15s} time={ft:7.2f}s rss={rss:7.0f}MB")
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results written to {RESULTS}")
    return out


def render(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = data["ks"]
    v = data["variants"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    # Memory: keep vs no-keep (covariance storage).
    ax1.plot(ks, v["laplace_keep"]["rss"], "o-", label="keep_eta_cov=True")
    ax1.plot(ks, v["laplace_nokeep"]["rss"], "s--", label="keep_eta_cov=False")
    ax1.set_xlabel("number of topics K")
    ax1.set_ylabel("peak RSS (MB)")
    ax1.set_title("Memory")
    ax1.legend(frameon=False, fontsize=8)

    # Speed: laplace vs diagonal (the O(K^3) inverse).
    ax2.plot(ks, v["laplace_keep"]["time"], "o-", label="laplace")
    ax2.plot(ks, v["diagonal_keep"]["time"], "^--", label="diagonal")
    ax2.set_xlabel("number of topics K")
    ax2.set_ylabel("fit time (s)")
    ax2.set_title("Speed")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    dest = os.path.join(PAPER, "fig_scaling.pdf")
    fig.savefig(dest)
    print(f"Figure written to {dest}")


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]), sys.argv[3])
        return
    if "--render" in sys.argv:
        with open(RESULTS) as f:
            render(json.load(f))
        return
    print(f"K-scaling sweep: N={N_DOCS}, vocab={VOCAB}, iters={ITERS}, K in {KS}")
    render(sweep())


if __name__ == "__main__":
    main()
