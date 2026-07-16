"""LDA-vs-tomotopy K-crossover sweep.

Regenerates the K-crossover table in ``docs/benchmarks.md``: topica's sparse
SparseLDA sampler vs tomotopy's dense Eigen sampler across a sweep of K, at a
fixed corpus size, single-threaded. tomotopy scores all K topics per token so
its time rises with K; topica visits only the topics a word occupies, so its
time is roughly flat. The two curves cross at some K -- below it tomotopy wins,
above it topica wins.

Both engines are timed together in the same harness (each fit in its own
subprocess via bench.peak_rss_mb), so the crossover is self-consistent on
whatever machine runs it. Methodology matches ``bench.py``'s matrix LDA leg:
end-to-end from token lists (tomotopy's timer includes the add_doc loop;
topica's fit() includes its internal corpus build).

Usage
-----
    python benchmarks/k_crossover.py

Env knobs
---------
    KX_SIZE   corpus size subsampled from poliblog5k (default 2000)
    KX_ITERS  Gibbs iterations    (default 500)
    KX_KS     comma-separated K values (default 20,50,100,200,400)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402  (reuse the harness loader/subsample/timer)

SIZE = int(os.environ.get("KX_SIZE", "2000"))
ITERS = int(os.environ.get("KX_ITERS", "500"))
KS = [int(k) for k in os.environ.get("KX_KS", "20,50,100,200,400").split(",")]

PYTHON_PATH = str(bench.ROOT / "python")
OUT_JSON = os.path.join(HERE, "k_crossover_results.json")
OUT_MD = os.path.join(HERE, "k_crossover.md")


def _topica_script(docs_path: str, k: int) -> str:
    return (
        "import json, sys, time\n"
        f"sys.path.insert(0, {PYTHON_PATH!r})\n"
        "import topica\n"
        f"docs = json.load(open({docs_path!r}))['docs']\n"
        "t0 = time.perf_counter()\n"
        f"topica.models.LDA(num_topics={k}, seed=1, optimize_interval=0,"
        " num_threads=1).fit(docs, iters=" + str(ITERS) + ")\n"
        "print('FIT_TIME', time.perf_counter() - t0)\n"
    )


def _tomo_script(docs_path: str, k: int) -> str:
    return (
        "import json, time\n"
        "import tomotopy as tp\n"
        f"docs = json.load(open({docs_path!r}))['docs']\n"
        f"mdl = tp.LDAModel(k={k}, seed=1)\n"
        "t0 = time.perf_counter()\n"
        "for d in docs:\n"
        "    mdl.add_doc(d)\n"
        f"mdl.train({ITERS}, workers=1)\n"
        "print('FIT_TIME', time.perf_counter() - t0)\n"
    )


def main() -> None:
    import tempfile

    bench._ensure_csv()
    toks, rating_full, day_full = bench._load_full()
    docs, _, _, vocab = bench._subsample(toks, rating_full, day_full, SIZE)
    print(
        f"poliblog5k subsampled to {len(docs)} docs ({len(vocab)} vocab); "
        f"iters={ITERS}; K sweep={KS}\n",
        flush=True,
    )

    results = []
    with tempfile.TemporaryDirectory() as d:
        docs_path = os.path.join(d, "docs.json")
        json.dump({"docs": docs}, open(docs_path, "w"))
        for k in KS:
            print(f"  [k-crossover] K={k} ...", flush=True)
            tt, _ = bench._run_subprocess_fit(_topica_script(docs_path, k))
            tm, _ = bench._run_subprocess_fit(_tomo_script(docs_path, k))
            # "faster" reads as "<winner> N× faster" (N = slower/faster >= 1).
            faster = (
                f"tomotopy {tt / tm:.2f}x" if tm < tt else f"topica {tm / tt:.2f}x"
            )
            results.append({"K": k, "topica": tt, "tomotopy": tm, "faster": faster})
            print(f"    topica {tt:.2f}s  tomotopy {tm:.2f}s  ({faster})", flush=True)

    json.dump({"size": len(docs), "iters": ITERS, "rows": results},
              open(OUT_JSON, "w"), indent=2)

    lines = ["| K | topica | tomotopy | faster |", "|---|---|---|---|"]
    for r in results:
        win = (
            f"tomotopy {r['topica'] / r['tomotopy']:.2f}×"
            if r["tomotopy"] < r["topica"]
            else f"**topica {r['tomotopy'] / r['topica']:.2f}×**"
        )
        lines.append(
            f"| {r['K']} | {r['topica']:.1f}s | {r['tomotopy']:.1f}s | {win} |"
        )
    md = "\n".join(lines) + "\n"
    open(OUT_MD, "w").write(md)
    print("\n" + md, flush=True)
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
