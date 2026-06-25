"""Parity check: topica `Wordfish` vs R `quanteda.textmodels::textmodel_wordfish`.

Builds a fixed synthetic corpus sampled from the Wordfish model, fits both topica
and quanteda on the same author x word count matrix, and reports the correlation
between their estimated positions (and each against the planted truth). A faithful
port recovers the same one-dimensional scale as quanteda (|r| ~ 1).

Shells out to `Rscript` with `quanteda.textmodels`. Skips cleanly (exit 0) if
Rscript or the package is unavailable, mirroring the other `parity/` scripts.

    python parity/wordfish_r_compare.py
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import topica

R_WORDFISH = r"""
suppressMessages({library(quanteda); library(quanteda.textmodels)})
args <- commandArgs(trailingOnly = TRUE)
m <- read.csv(args[1], row.names = 1, check.names = FALSE)
d <- as.dfm(as.matrix(m))
wf <- textmodel_wordfish(d)
out <- data.frame(speakerid = rownames(m), theta = as.numeric(wf$theta))
write.csv(out, args[2], row.names = FALSE)
"""


def _planted(n_authors=50, n_words=80, docs_per=4, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.0, 1.0, n_authors)
    beta = np.linspace(-1.0, 1.0, n_words)
    psi = np.log(rng.uniform(3.0, 12.0, n_words))
    docs, group = [], []
    for a in range(n_authors):
        rates = np.exp(psi + beta * theta[a]) / docs_per
        for _ in range(docs_per):
            counts = rng.poisson(rates)
            doc = []
            for j, c in enumerate(counts):
                doc.extend([f"w{j}"] * int(c))
            rng.shuffle(doc)
            docs.append(doc)
            group.append(f"a{a:03d}")
    return docs, group, theta


def _pearson(x, y):
    return abs(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> int:
    if shutil.which("Rscript") is None:
        print("SKIP: Rscript not available")
        return 0
    probe = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("quanteda.textmodels", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    if "TRUE" not in probe.stdout:
        print("SKIP: quanteda.textmodels not installed")
        return 0

    docs, group, theta_true = _planted()
    topica.enable_experimental(True)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a000": -1.0, "a049": 1.0}, iters=200)
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    authors = sorted(pos)
    topica_theta = [pos[a] for a in authors]

    # Build the author x word count matrix for quanteda.
    vocab = m.vocabulary
    vidx = {w: j for j, w in enumerate(vocab)}
    counts = {a: np.zeros(len(vocab)) for a in authors}
    for doc, a in zip(docs, group):
        for w in doc:
            if w in vidx:
                counts[a][vidx[w]] += 1.0

    with tempfile.TemporaryDirectory() as td:
        infile = Path(td) / "counts.csv"
        outfile = Path(td) / "theta.csv"
        with open(infile, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["speakerid", *vocab])
            for a in authors:
                wr.writerow([a, *[int(x) for x in counts[a]]])
        proc = subprocess.run(
            ["Rscript", "-e", R_WORDFISH, str(infile), str(outfile)],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if proc.returncode != 0 or not outfile.exists():
            print("SKIP: quanteda wordfish run failed:\n", proc.stderr)
            return 0
        rows = list(csv.DictReader(open(outfile, encoding="utf-8")))
    r_theta = {row["speakerid"]: float(row["theta"]) for row in rows}
    quanteda_theta = [r_theta[a] for a in authors]

    planted = [theta_true[int(a[1:])] for a in authors]
    r_vs_quanteda = _pearson(topica_theta, quanteda_theta)
    r_topica_truth = _pearson(topica_theta, planted)
    r_quanteda_truth = _pearson(quanteda_theta, planted)

    print(f"topica Wordfish vs quanteda textmodel_wordfish : |r| = {r_vs_quanteda:.4f}")
    print(f"topica   vs planted truth                      : |r| = {r_topica_truth:.4f}")
    print(f"quanteda vs planted truth                      : |r| = {r_quanteda_truth:.4f}")
    if r_vs_quanteda < 0.95:
        print(f"FAIL: topica and quanteda disagree (|r|={r_vs_quanteda:.4f} < 0.95)")
        return 1
    print("PASS: topica Wordfish matches quanteda's scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
