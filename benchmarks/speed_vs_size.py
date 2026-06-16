"""Wall-clock speed vs. the R/Java references ACROSS CORPUS SIZES.

Same methodology as ``speed_vs_r.py`` (fit the identical corpus and settings in
each engine, time only the fit call, pin iterations), run at several corpus
sizes so the trend that matters for the approximate parallel Gibbs samplers is
visible: **multithread speedup grows with corpus size.** The per-sweep
count-table merge is fixed overhead, so larger corpora amortize it and
parallelize better, and the small-corpus ``speed_vs_r.py`` table understates the
multithread numbers a large-corpus user sees.

The sizes are seeded subsamples of stm's ``poliblog5k`` (exported once to
``benchmarks/poliblog5k_prepped.csv``), so the corpus is consistently
preprocessed and the run is reproducible. STM is variational EM (single-threaded,
vs R ``stm``); keyATM and LDA are collapsed Gibbs, timed single- and
multi-threaded (topica's approximate AD-LDA path) against R ``keyATM`` and Java
MALLET.

    python benchmarks/speed_vs_size.py

Env knobs: SIZES (2000,3500,5000), STM_K, STM_EM_ITERS, KEYATM_K, KEYATM_ITERS,
LDA_K, LDA_ITERS, THREADS (8), MIN_DF (3).
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
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "parity"))

import speed_vs_r as SV  # noqa: E402  (reuses _rscript, _R_STM, _R_KEYATM, mallet_available)
import keyatm_r_compare as KA  # noqa: E402  (KEYWORD_SETS)

PREPPED = os.path.join(HERE, "poliblog5k_prepped.csv")
SIZES = [int(x) for x in os.environ.get("SIZES", "2000,3500,5000").split(",")]
STM_K = int(os.environ.get("STM_K", "20"))
STM_EM_ITERS = int(os.environ.get("STM_EM_ITERS", "30"))
KEYATM_K = int(os.environ.get("KEYATM_K", "10"))
KEYATM_ITERS = int(os.environ.get("KEYATM_ITERS", "1000"))
LDA_K = int(os.environ.get("LDA_K", "20"))
LDA_ITERS = int(os.environ.get("LDA_ITERS", "1000"))
THREADS = int(os.environ.get("THREADS", "8"))
MIN_DF = int(os.environ.get("MIN_DF", "3"))


def load_full():
    if not os.path.exists(PREPPED):
        raise SystemExit(
            f"missing {PREPPED}\nExport it once from R:\n"
            '  suppressMessages(library(stm)); data(poliblog5k, package="stm")\n'
            '  voc <- poliblog5k.voc\n'
            '  txt <- vapply(poliblog5k.docs, function(d) '
            'paste(rep(voc[d[1,]], d[2,]), collapse=" "), character(1))\n'
            '  out <- data.frame(rating=poliblog5k.meta$rating, day=poliblog5k.meta$day,\n'
            '                    blog=poliblog5k.meta$blog, text=txt, stringsAsFactors=FALSE)\n'
            '  write.csv(out[nchar(out$text)>0,], "benchmarks/poliblog5k_prepped.csv", row.names=FALSE)'
        )
    rows = list(csv.DictReader(open(PREPPED, newline="")))
    toks = [r["text"].split() for r in rows]
    rating = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])
    return toks, rating, day


def subsample(toks, rating, day, n, seed=0):
    """A seeded n-document subsample, pruned to terms in >= MIN_DF documents."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(toks), size=min(n, len(toks)), replace=False)
    sub = [toks[i] for i in idx]
    dfc = Counter()
    for d in sub:
        dfc.update(set(d))
    vocab = {w for w, c in dfc.items() if c >= MIN_DF}
    docs, keep = [], []
    for d in sub:
        dd = [w for w in d if w in vocab]
        keep.append(len(dd) > 0)
        if dd:
            docs.append(dd)
    keep = np.array(keep)
    return docs, rating[idx][keep], day[idx][keep], vocab


def _r_time(out: str) -> float:
    return float([ln for ln in out.splitlines() if ln.startswith("R_TIME")][0].split()[1])


def bench_stm(docs, rating, day):
    from topica import STM
    from topica.stm import spline

    sb, _ = spline(day, df=10)
    X = np.column_stack([rating, sb])
    feat = ["ratingLiberal"] + [f"day_s{j}" for j in range(sb.shape[1])]
    design = np.column_stack([np.ones(len(docs)), X])
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(x) for x in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + feat)
            w.writerows(design.tolist())
        out = SV._rscript(f'dir <- "{d}"\nKVAL <- {STM_K}\nNITERS <- {STM_EM_ITERS}\n' + SV._R_STM)
    r = _r_time(out)
    t0 = time.perf_counter()
    STM(num_topics=STM_K, init="spectral").fit(
        docs, X, prevalence_names=feat, iters=STM_EM_ITERS, convergence_tol=0.0
    )
    return {"r": r, "tt1": time.perf_counter() - t0}


def bench_keyatm(docs, vocab):
    from topica import KeyATM

    kws = {n: [w for w in ws if w in vocab] for n, ws in KA.KEYWORD_SETS.items()}
    kws = {n: ws for n, ws in kws.items() if ws}
    nreg = KEYATM_K - len(kws)
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(x) for x in docs) + "\n")
        json.dump(kws, open(os.path.join(d, "keywords.json"), "w"))
        out = SV._rscript(f'dir <- "{d}"\nNREG <- {nreg}\nNITERS <- {KEYATM_ITERS}\n' + SV._R_KEYATM)
    r = _r_time(out)

    def fit(t):
        t0 = time.perf_counter()
        KeyATM(kws, num_topics=KEYATM_K, seed=1).fit(docs, iters=KEYATM_ITERS, num_threads=t)
        return time.perf_counter() - t0

    return {"r": r, "tt1": fit(1), "ttN": fit(THREADS)}


def bench_lda(docs):
    from topica import LDA

    mallet = shutil.which("mallet")
    r = None
    if mallet:
        d = tempfile.mkdtemp(dir="/private/tmp")
        try:
            txt = os.path.join(d, "tok.txt")
            with open(txt, "w") as f:
                for i, t in enumerate(docs):
                    f.write(f"doc{i}\t{' '.join(t)}\n")
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
            r = time.perf_counter() - t0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def fit(t):
        t0 = time.perf_counter()
        LDA(num_topics=LDA_K, seed=1, optimize_interval=0, num_threads=t).fit(docs, iters=LDA_ITERS)
        return time.perf_counter() - t0

    return {"r": r, "tt1": fit(1), "ttN": fit(THREADS)}


def main():
    toks, rating, day = load_full()
    print(f"poliblog5k: {len(toks)} docs available | sizes={SIZES} | threads={THREADS}\n")
    res = []
    for n in SIZES:
        docs, rt, dy, vocab = subsample(toks, rating, day, n)
        nd = len(docs)
        print(f"== N~{n}  ({nd} docs, {len(vocab)} vocab) ==", flush=True)
        s = bench_stm(docs, rt, dy)
        print(f"  STM    vs R stm   : {s['r']/s['tt1']:.1f}x single-thread", flush=True)
        k = bench_keyatm(docs, vocab)
        print(f"  keyATM vs R keyATM: {k['r']/k['tt1']:.1f}x ST, {k['r']/k['ttN']:.1f}x MT "
              f"(topica {k['tt1']/k['ttN']:.1f}x thread scaling)", flush=True)
        la = bench_lda(docs)
        if la["r"]:
            print(f"  LDA    vs MALLET  : {la['r']/la['tt1']:.2f}x ST, {la['r']/la['ttN']:.2f}x MT "
                  f"(topica {la['tt1']/la['ttN']:.1f}x thread scaling)", flush=True)
        else:
            print(f"  LDA    (no MALLET): topica {la['tt1']/la['ttN']:.1f}x thread scaling", flush=True)
        res.append({"n_docs": nd, "vocab": len(vocab), "stm": s, "keyatm": k, "lda": la})

    # Summary table: the thread-scaling trend across sizes (the headline).
    print("\nThread scaling (topica single / multi), by corpus size:")
    print(f"| {'docs':>6} | {'LDA ST/MT':>9} | {'keyATM ST/MT':>12} |")
    print(f"|{'-'*8}|{'-'*11}|{'-'*14}|")
    for r in res:
        lda_sc = f"{r['lda']['tt1']/r['lda']['ttN']:.1f}x"
        ka_sc = f"{r['keyatm']['tt1']/r['keyatm']['ttN']:.1f}x"
        print(f"| {r['n_docs']:>6} | {lda_sc:>9} | {ka_sc:>12} |")

    json.dump(res, open(os.path.join(HERE, "speed_vs_size.json"), "w"), indent=2)
    print(f"\nWrote {os.path.join(HERE, 'speed_vs_size.json')}")


if __name__ == "__main__":
    main()
