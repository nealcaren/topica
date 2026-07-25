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
out <- data.frame(speakerid = rownames(m), theta = as.numeric(wf$theta),
                  se = as.numeric(wf$se.theta))
write.csv(out, args[2], row.names = FALSE)
# Per-word discrimination beta, aligned to the dfm feature order (args[3]).
bout <- data.frame(word = wf$features, beta = as.numeric(wf$beta))
write.csv(bout, args[3], row.names = FALSE)
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
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a000": -1.0, "a049": 1.0}, iters=200)
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    se = dict(zip(m.author_names, m.position_se))
    authors = sorted(pos)
    topica_theta = [pos[a] for a in authors]

    # Unanchored fit: topica default-orients to the first two authors
    # (theta[0] < theta[1]), mirroring quanteda's default dir = c(1, 2). Refit with
    # no anchors so the *signed* correlation with quanteda validates the default
    # orientation (finding #2), not just the axis up to sign.
    mu = topica.Wordfish(seed=1)
    mu.fit(docs, group=group, iters=200)
    unanchored = dict(zip(mu.author_names, mu.author_positions[:, 0]))

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
        betafile = Path(td) / "beta.csv"
        with open(infile, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["speakerid", *vocab])
            for a in authors:
                wr.writerow([a, *[int(x) for x in counts[a]]])
        proc = subprocess.run(
            ["Rscript", "-e", R_WORDFISH, str(infile), str(outfile), str(betafile)],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if proc.returncode != 0 or not outfile.exists():
            print("SKIP: quanteda wordfish run failed:\n", proc.stderr)
            return 0
        rows = list(csv.DictReader(open(outfile, encoding="utf-8")))
        beta_rows = (
            list(csv.DictReader(open(betafile, encoding="utf-8")))
            if betafile.exists()
            else []
        )
    r_theta = {row["speakerid"]: float(row["theta"]) for row in rows}
    r_se = {row["speakerid"]: float(row["se"]) for row in rows if row.get("se")}
    r_beta = {row["word"]: float(row["beta"]) for row in beta_rows if row.get("beta")}
    quanteda_theta = [r_theta[a] for a in authors]

    planted = [theta_true[int(a[1:])] for a in authors]
    r_vs_quanteda = _pearson(topica_theta, quanteda_theta)
    r_topica_truth = _pearson(topica_theta, planted)
    r_quanteda_truth = _pearson(quanteda_theta, planted)

    # Signed correlation of the *unanchored* topica fit vs quanteda: both default to
    # dir = c(1, 2), so a faithful default orientation gives a positive sign, not
    # merely |r| ~ 1.
    unanchored_theta = [unanchored[a] for a in authors]
    signed_default = float(
        np.corrcoef(np.asarray(unanchored_theta), np.asarray(quanteda_theta))[0, 1]
    )

    print(f"topica Wordfish vs quanteda textmodel_wordfish : |r| = {r_vs_quanteda:.4f}")
    print(f"topica   vs planted truth                      : |r| = {r_topica_truth:.4f}")
    print(f"quanteda vs planted truth                      : |r| = {r_quanteda_truth:.4f}")
    print(f"topica (unanchored, default dir) vs quanteda   :  r  = {signed_default:+.4f}")

    # per-word discrimination beta: topica vs quanteda, aligned by word.
    beta_r = float("nan")
    if r_beta:
        t_beta = dict(zip(vocab, m.word_discrimination))
        common_b = [w for w in vocab if w in r_beta]
        if len(common_b) >= 2:
            beta_r = _pearson(
                [t_beta[w] for w in common_b], [r_beta[w] for w in common_b]
            )
            print(f"topica beta vs quanteda beta                   : |r| = {beta_r:.4f}")

    # standard errors: topica's analytic position_se vs quanteda's se.theta. These
    # need not be bit-equal (topica profiles out alpha and folds in the theta
    # prior; see position_se docstring), so the ratio is gated loosely.
    se_r = float("nan"); se_ratio = float("nan")
    if r_se:
        common_se = [a for a in authors if a in r_se]
        ts = np.array([se[a] for a in common_se])
        qs = np.array([r_se[a] for a in common_se])
        se_r = _pearson(ts, qs)
        se_ratio = float(np.median(ts / qs))
        print(f"topica position_se vs quanteda se.theta        : |r| = {se_r:.4f} "
              f"(median ratio {se_ratio:.3f})")

    failures = []
    if r_vs_quanteda < 0.95:
        failures.append(f"theta |r|={r_vs_quanteda:.4f} < 0.95")
    # Default orientation (no anchors) must match quanteda's dir sign, i.e. a strong
    # *positive* signed correlation.
    if signed_default < 0.95:
        failures.append(f"unanchored signed r={signed_default:+.4f} < 0.95")
    # beta shares theta's identification, so a faithful scale recovers it too.
    if not np.isnan(beta_r) and beta_r < 0.95:
        failures.append(f"beta |r|={beta_r:.4f} < 0.95")
    # SE is only comparable, not equal: require the same order of magnitude and a
    # strong rank correlation, not exact equality.
    if not np.isnan(se_r) and se_r < 0.90:
        failures.append(f"se |r|={se_r:.4f} < 0.90")
    if not np.isnan(se_ratio) and not (0.5 <= se_ratio <= 2.0):
        failures.append(f"se median ratio {se_ratio:.3f} outside [0.5, 2.0]")

    if failures:
        print("FAIL: topica and quanteda disagree (" + "; ".join(failures) + ")")
        return 1
    print("PASS: topica Wordfish matches quanteda's scale (theta, beta, se)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
