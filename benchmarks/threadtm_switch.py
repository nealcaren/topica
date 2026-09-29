"""Wall-clock cost of ``ThreadTM(switch=True)`` relative to the base LDA it wraps (#907).

Times three fits on one corpus at default settings: the base ``LDA(K).fit``, the pooled
``ThreadTM(K).fit`` and the switch ``ThreadTM(K, switch=True).fit``, and reports each as a
multiple of the LDA time. ``--profile`` adds a cProfile of the switch fit (cumulative time of
the calibration's hot functions).

The corpus is either a reply-tree JSONL (``--corpus``, gzipped or not; one object per line with
``text`` and ``parent``, a 0-based parent row or -1 for a root, as in the threadtm-paper
``reddit_recent`` corpora) or, by default, a synthetic corpus of threads whose replies inherit
their parent's topic mix half the time.

    python benchmarks/threadtm_switch.py
    python benchmarks/threadtm_switch.py --corpus .../NeutralPolitics/corpus.jsonl.gz --k 30
"""

from __future__ import annotations

import argparse
import cProfile
import gzip
import json
import pstats
import re
import time
import warnings

import numpy as np

import topica
from topica import threads

HOT = ("_calibrate_switch", "_coord_search", "thread_ll", "token_mix", "_inherit_prob",
       "_interp_cols", "share_tables", "_sequential_log_ml", "_transform_switch", "fit")


def load_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    docs = [re.findall(r"[a-z']+", threads.strip_quotes(r["text"]).lower()) for r in rows]
    return docs, [int(r["parent"]) for r in rows]


def synthetic(n_threads=1500, k=10, v=400, seed=13):
    """Threads of one root and 4-10 replies; each reply inherits its parent's mix with
    probability 0.5, else draws a fresh one."""
    rng = np.random.default_rng(seed)
    beta = rng.dirichlet(np.full(v, 0.05), k)
    docs, parents = [], []
    for _ in range(n_threads):
        mixes = [rng.dirichlet(np.full(k, 0.1))]
        root = len(docs)
        par = [-1]
        for _ in range(rng.integers(4, 11)):
            p = int(rng.integers(0, len(mixes)))
            mix = (rng.dirichlet(50 * mixes[p] + 0.01) if rng.random() < 0.5
                   else rng.dirichlet(np.full(k, 0.1)))
            mixes.append(mix)
            par.append(root + p)
        for i, mix in enumerate(mixes):
            n = 120 if i == 0 else int(rng.integers(20, 60))
            z = rng.choice(k, n, p=mix)
            docs.append([f"w{rng.choice(v, p=beta[t])}" for t in z])
        parents.extend(par)
    return docs, parents


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", help="reply-tree JSONL (text, parent); default synthetic")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-cf", type=int, default=5)
    ap.add_argument("--skip-pooled", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    topica.enable_experimental()
    docs, parents = load_jsonl(args.corpus) if args.corpus else synthetic(seed=args.seed)
    ckw = {"min_cf": args.min_cf, "stopwords": "english"}
    print(f"{len(docs):,} documents, {sum(p < 0 for p in parents):,} threads, K = {args.k}")

    corpus = topica.Corpus.from_documents(docs, **ckw)
    t_lda, _ = timed(lambda: topica.LDA(args.k, seed=args.seed).fit(corpus))
    rows = [("LDA(K).fit", t_lda)]
    if not args.skip_pooled:
        t, _ = timed(lambda: topica.ThreadTM(args.k, seed=args.seed).fit(
            docs, parents, corpus_kwargs=ckw))
        rows.append(("ThreadTM(K).fit (pooled)", t))

    def switch():
        return topica.ThreadTM(args.k, seed=args.seed, switch=True).fit(
            docs, parents, corpus_kwargs=ckw)
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    t, _ = timed(switch)
    if prof:
        prof.disable()
    rows.append(("ThreadTM(K, switch=True).fit", t))

    print(f"\n| run | wall | vs LDA |\n|---|---|---|")
    for name, secs in rows:
        print(f"| `{name}` | {secs:.1f} s | {secs / t_lda:.1f}x |")

    if prof:
        stats = pstats.Stats(prof).stats
        print("\n| function | cum time | calls |\n|---|---|---|")
        hot = [(fn[2] if fn[2] != "fit" else f"fit ({fn[0].rsplit('/', 1)[-1]}:{fn[1]})",
                v[3], v[1]) for fn, v in stats.items()
               if fn[2] in HOT and "threads.py" in fn[0]]
        for name, cum, calls in sorted(hot, key=lambda r: -r[1]):
            print(f"| `{name}` | {cum:.1f} s | {calls:,} |")


if __name__ == "__main__":
    main()
