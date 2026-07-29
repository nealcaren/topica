"""Reproduce the topica paper's worked-example artifacts and validation checks.

This drives the worked example (Section 7) and the cross-implementation
validation (Section 5). The speed comparisons of Section 6 are reproduced
separately by the timing scripts in `benchmarks/` (bench_stm.py, bench.py,
k_crossover.py), which need a tuned, quiet machine to be meaningful.

Run from the repo root:

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/replication.py
    # ... --quick   to skip the slow cross-implementation checks (no R/Java needed)

Three parts:

  1. The *spanning comparison* (Worked example / breadth). Several models from
     different families -- count-based (LDA, CTM, STM) and embedding-based
     (BERTopic on deterministic LSA vectors) -- are fit on the SAME corpus and then
     scored by the SAME diagnostics in ONE loop: the framework's central claim made
     concrete. Reproduces with only numpy/scikit-learn (no sentence-transformers).

  2. The STM covariate-effect forest plot (`fig_poliblog_effect.pdf`) with honest
     method-of-composition intervals.

  3. *Cross-implementation validation* -- the part that keeps the validation claims
     honest. We do not just assert that topica reproduces MALLET, R `stm`, and R
     `keyATM`; we RUN those reference packages on the same inputs and report the
     agreement, by driving the scripts in `parity/` and `benchmarks/`. Each is
     skipped (clearly, not silently) if its toolchain (the `mallet` CLI, or
     `Rscript` with `stm`/`keyATM`) is not installed -- so the script runs anywhere,
     and reports exactly which reference comparisons actually executed.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import topica
from topica import Corpus, stm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 15
SEED = 1


def load_poliblog():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "examples", "poliblog.csv"))))
    docs = [r["text"].split() for r in rows]
    conservative = np.array(
        [r["rating"] == "Conservative" for r in rows], float
    ).reshape(-1, 1)
    return docs, conservative


def lsa_embeddings(docs, dim=50, seed=0):
    """Deterministic document vectors via TF-IDF + truncated SVD (LSA).

    Stands in for any sentence embedder so the embedding-based model is fully
    reproducible. topica takes the vectors; where they come from is the user's call.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(tokenizer=lambda d: d, preprocessor=lambda d: d,
                            token_pattern=None).fit_transform(docs)
    svd = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(tfidf)
    return normalize(svd)  # L2-normalize so HDBSCAN sees comparable distances


def score(model, texts):
    """Every model is scored by the SAME diagnostics -- this is the whole point."""
    coh = float(np.mean(topica.coherence(model, texts, coherence_type="c_v")))
    exc = float(np.mean(topica.exclusivity(model)))
    div = float(topica.topic_diversity(model))
    return coh, exc, div


def spanning_comparison(docs, conservative):
    print("=" * 72)
    print("Spanning comparison: different model families, one scoring loop")
    print("=" * 72)

    fits = []

    def run(name, build, fit):
        t0 = time.perf_counter()
        m = build()
        fit(m)
        dt = time.perf_counter() - t0
        coh, exc, div = score(m, docs)
        fits.append((name, m.num_topics, coh, exc, div, dt))
        print(f"  fit {name:10s} K={m.num_topics:<3d} {dt:5.1f}s")
        return m

    run("LDA", lambda: topica.LDA(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, iters=500))
    run("CTM", lambda: topica.CTM(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, iters=25))
    run("STM", lambda: topica.STM(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, conservative, prevalence_names=["conservative"], iters=25))
    try:
        emb = lsa_embeddings(docs)
        run("BERTopic", lambda: topica.BERTopic(seed=42),
            lambda m: m.fit_transform(docs, emb))
    except Exception as e:  # embedding extra not available: report and continue
        print(f"  (embedding model skipped: {type(e).__name__}: {e})")

    print("\n  model       K   coherence(c_v)  exclusivity  diversity")
    print("  " + "-" * 56)
    for name, k, coh, exc, div, _ in fits:
        print(f"  {name:10s} {k:<3d}    {coh:8.3f}      {exc:7.3f}    {div:6.3f}")
    return fits


def effect_figure(docs, conservative):
    print("\n" + "=" * 72)
    print("STM covariate-effect figure (fig_poliblog_effect.pdf)")
    print("=" * 72)
    import topica.viz as viz

    corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)
    print(f"  {corpus.num_docs} docs, vocab {corpus.num_words}")

    model = topica.STM(num_topics=K, seed=SEED)
    model.fit(docs, conservative, prevalence_names=["conservative"], iters=25)

    labeled = stm.label_topics(model.topic_word, model.vocabulary, n=3)
    topica.set_topic_labels(
        model, {t: ", ".join(w for w, _ in labeled[t]["frex"]) for t in range(K)}
    )
    panel = viz.effect_plot(model, corpus, X=conservative,
                            feature_names=["conservative"], nsims=100)
    out = os.path.join(HERE, "fig_poliblog_effect.pdf")
    panel.to_png(out)
    print(f"  wrote {out}")

    # The one-call report: a composite figure off the shared contract that shows
    # the reporting layer (topics-by-prevalence, coherence/exclusivity, the
    # honest correlation, and prevalence by group).
    report = topica.plot_report(model, texts=docs, groups=conservative.ravel(),
                                n=8, figsize=(11, 7),
                                title="topica report: poliblog STM (K=15)")
    report_out = os.path.join(HERE, "fig_poliblog_report.pdf")
    report.savefig(report_out, bbox_inches="tight")
    print(f"  wrote {report_out}")

    df = panel.to_frame().sort_values("coef")
    print("\n  Per-topic effect of conservative rating on prevalence:")
    for _, r in df.iterrows():
        flag = "" if r["reliable"] else "  (unreliable)"
        print(f"    topic {int(r['topic']):2d} {r['label'][:30]:<30} "
              f"coef={r['coef']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{flag}")


def _missing_r_packages(pkgs):
    """Return the subset of `pkgs` that Rscript cannot load (best effort).

    Lets a missing `quanteda`/`jsonlite` surface as a clean SKIP rather than an
    opaque exit-1 from the comparison script. If the probe itself fails, assume
    nothing is missing and let the script report its own error.
    """
    if not pkgs:
        return []
    expr = ("p <- c(%s); ok <- sapply(p, requireNamespace, quietly=TRUE); "
            "cat(paste(p[!ok], collapse=' '))") % \
        ", ".join(f'"{p}"' for p in pkgs)
    try:
        proc = subprocess.run(["Rscript", "-e", expr],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return (proc.stdout or "").split()


def cross_implementation():
    """Run the reference packages themselves and report the agreement.

    Each entry names the script, the executables it needs, the R packages it needs
    (probed when Rscript is present), and a one-line note. Missing toolchains are
    reported, never hidden -- the paper's validation claims only stand for the rows
    that actually ran here.
    """
    print("\n" + "=" * 72)
    print("Cross-implementation validation: topica vs. the reference packages")
    print("=" * 72)

    # (label, script, executables, R packages)
    checks = [
        ("LDA / LabeledLDA / DMR vs. Java MALLET", "parity/mallet_parity.py",
         ["mallet", "java"], []),
        ("STM vs. R stm (poliblog vignette)", "parity/stm_poliblog_compare.py",
         ["Rscript"], ["stm"]),
        ("STM content-covariate (SAGE) vs. R stm", "parity/stm_content_r_compare.py",
         ["Rscript"], ["stm"]),
        ("KeyATM vs. R keyATM", "parity/keyatm_r_compare.py", ["Rscript"],
         ["keyATM", "quanteda"]),
        ("STS vs. authors' replication package (sets STS_REPL_DIR)",
         "parity/sts_r_compare.py", ["Rscript"], ["stm"]),
        ("Fit-time speedups vs. R stm / keyATM", "benchmarks/speed_vs_r.py",
         ["Rscript"], ["stm", "keyATM", "quanteda", "jsonlite"]),
    ]

    ran, skipped = [], []
    for label, script, exes, r_pkgs in checks:
        missing = [e for e in exes if shutil.which(e) is None]
        print(f"\n--- {label}\n    ({script})")
        if missing:
            print(f"    SKIPPED: not installed: {', '.join(missing)}")
            skipped.append(label)
            continue
        missing_pkgs = _missing_r_packages(r_pkgs) if r_pkgs else []
        if missing_pkgs:
            print(f"    SKIPPED: missing R package(s): {', '.join(missing_pkgs)}")
            skipped.append(label)
            continue
        proc = subprocess.run([sys.executable, script], cwd=ROOT_DIR,
                              capture_output=True, text=True)
        out = (proc.stdout or "").strip().splitlines()
        for line in out[-12:]:            # the scripts print a short summary; echo its tail
            print("    " + line)
        if proc.returncode != 0:
            print(f"    (exit {proc.returncode}); stderr tail:")
            for line in (proc.stderr or "").strip().splitlines()[-4:]:
                print("    ! " + line)
        ran.append(label)

    print("\n" + "-" * 72)
    print(f"Reference comparisons run here: {len(ran)}/{len(checks)}.")
    if skipped:
        print("Skipped (install the toolchain to reproduce): " + "; ".join(skipped))


def main():
    ap = argparse.ArgumentParser(description="Reproduce the topica paper's artifacts.")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow cross-implementation checks (no R/Java needed)")
    args = ap.parse_args()

    docs, conservative = load_poliblog()
    spanning_comparison(docs, conservative)
    try:
        effect_figure(docs, conservative)
    except ImportError as e:  # matplotlib/pandas not installed: report and continue
        print(f"\n  (effect figure skipped: {type(e).__name__}: {e};"
              " install matplotlib and pandas to render it)")
    if not args.quick:
        cross_implementation()


if __name__ == "__main__":
    main()
