"""Cross-implementation validation: topica STM vs. R `stm` on the poliblog
vignette (Roberts, Stewart & Tingley's JSS example).

Same model as the gadarian check (logistic-normal variational EM, prevalence
covariates, Arora-style spectral init), but the canonical vignette corpus:
2,000 political-blog posts from 2008, already stemmed/stopworded, with a
``rating`` (Conservative/Liberal) covariate and a ``day`` spline — the
``prevalence = ~ rating + s(day)`` design from the paper.

The two engines share no code and no RNG, so they are never byte-identical;
"identical results" means *statistical* agreement on the fitted topics. To keep
the comparison clean we build ONE numeric design matrix (rating dummy + a single
day-spline basis) in Python and hand the SAME matrix to both engines, so any
gap is the inference engine, not a basis or coding difference. We feed both the
SAME integer-coded documents and vocabulary, both initialized Spectral.

The yardstick is R's own reproducibility: poliblog is far less multimodal than
gadarian K=3, so under matched Spectral init topica should land essentially on
R's solution, and well inside R's Spectral-vs-Random basin spread.

Shells out to `Rscript` with `stm`; skips (exit 0) if unavailable. Run:

    python parity/stm_poliblog_compare.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLIBLOG = os.path.join(ROOT, "examples", "poliblog.csv")

K = int(os.environ.get("POLIBLOG_K", "20"))
SPLINE_DF = int(os.environ.get("POLIBLOG_SPLINE_DF", "10"))
MIN_DOC_FREQ = int(os.environ.get("POLIBLOG_MIN_DF", "3"))
EM_ITERS = int(os.environ.get("POLIBLOG_EM_ITERS", "200"))


def r_stm_available() -> bool:
    """True iff `Rscript` is on PATH and the `stm` package loads."""
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


def load_and_prep():
    """Load the (already preprocessed) poliblog corpus.

    Returns (docs, rating_lib, day, vocab) where docs is list[list[str]],
    rating_lib is the 0/1 Liberal dummy (Conservative = baseline, matching R's
    alphabetical factor coding), and day is the raw day-of-year covariate.
    """
    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    toks = [r["text"].split() for r in rows]
    rating_lib = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])

    # Light frequency prune (the text is already stemmed/stopworded); both
    # engines get the identical surviving vocabulary, so this only sets corpus
    # size, not the parity.
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= MIN_DOC_FREQ}
    toks = [[w for w in d if w in vocab] for d in toks]
    keep = np.array([len(d) > 0 for d in toks])
    docs = [d for d, k in zip(toks, keep) if k]
    return docs, rating_lib[keep], day[keep], sorted({w for d in docs for w in d})


# R driver. Reads space-joined docs + a numeric design matrix (intercept
# already included) and fits stm with that matrix as `prevalence`, Spectral
# init plus two Random seeds. Exports each topic-word matrix (K x V).
_R_DRIVER = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks)))
vmap  <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d) {
  tb <- table(d); idx <- as.integer(vmap[names(tb)]); o <- order(idx)
  matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow = 2)
})
X <- as.matrix(read.csv(file.path(dir, "design.csv")))  # intercept + covariates
beta_of <- function(seed, init) {
  set.seed(seed)
  f <- stm(documents, vocab, K = KVAL, prevalence = X,
           init.type = init, verbose = FALSE)
  b <- exp(f$beta$logbeta[[1]]); colnames(b) <- vocab; b
}
write.csv(beta_of(1, "Spectral"), file.path(dir, "r_spectral.csv"), row.names = FALSE)
write.csv(beta_of(11, "Random"),  file.path(dir, "r_rand1.csv"),    row.names = FALSE)
write.csv(beta_of(22, "Random"),  file.path(dir, "r_rand2.csv"),    row.names = FALSE)
write(vocab, file.path(dir, "r_vocab.txt"))
cat("ok\n")
"""


def _read_r_beta(path, vocab):
    """Read an R topic-word CSV (K rows, vocab-named columns) into a K x |vocab|
    array aligned to `vocab` (the R column order)."""
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        cols = [h.strip('"') for h in header]
        rows = [[float(x) for x in row] for row in rdr]
    mat = np.array(rows)
    idx = {w: i for i, w in enumerate(cols)}
    out = np.zeros((mat.shape[0], len(vocab)))
    for j, w in enumerate(vocab):
        if w in idx:
            out[:, j] = mat[:, idx[w]]
    return out


def _best_alignment_cosine(a, b, return_pairs=False):
    """Mean cosine of the best one-to-one topic alignment between two K x V
    topic-word matrices (greedy; good enough for diagnostic K)."""
    k = a.shape[0]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    used = set()
    total = 0.0
    pairs = []
    for i in np.argsort(-sim.max(axis=1)):
        for j in np.argsort(-sim[i]):
            if j not in used:
                used.add(j)
                total += sim[i, j]
                pairs.append((int(i), int(j), float(sim[i, j])))
                break
    return (total / k, pairs) if return_pairs else total / k


def run(verbose: bool = True) -> dict:
    if not r_stm_available():
        raise RuntimeError("Rscript with the 'stm' package is not available")

    from topica import STM
    from topica.stm import spline

    docs, rating_lib, day, _ = load_and_prep()

    # ONE design matrix, shared by both engines. topica auto-prepends an
    # intercept; R's matrix-prevalence path does not, so we write the intercept
    # column for R and pass the bare covariates to topica.
    spline_basis, _ = spline(day, df=SPLINE_DF)
    X = np.column_stack([rating_lib, spline_basis])
    feat_names = ["ratingLiberal"] + [f"day_s{j}" for j in range(spline_basis.shape[1])]
    design_with_intercept = np.column_stack([np.ones(len(docs)), X])

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + feat_names)
            for row in design_with_intercept:
                w.writerow(list(row))

        script = f'dir <- "{d}"\nKVAL <- {K}\n' + _R_DRIVER
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=1800
        )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        r_vocab = open(os.path.join(d, "r_vocab.txt")).read().split()
        r_spectral = _read_r_beta(os.path.join(d, "r_spectral.csv"), r_vocab)
        r_rand1 = _read_r_beta(os.path.join(d, "r_rand1.csv"), r_vocab)
        r_rand2 = _read_r_beta(os.path.join(d, "r_rand2.csv"), r_vocab)

        model = STM(num_topics=K, init="spectral")
        model.fit(docs, X, prevalence_names=feat_names, iters=EM_ITERS, convergence_tol=1e-5)
        tt_converged = bool(model.converged)
        tt_em_iters = len(model.bound_history)
        tt_vocab = list(model.vocabulary)
        tt_beta_raw = np.asarray(model.topic_word)

        tt_idx = {w: i for i, w in enumerate(tt_vocab)}
        tt_beta = np.zeros((tt_beta_raw.shape[0], len(r_vocab)))
        for j, w in enumerate(r_vocab):
            if w in tt_idx:
                tt_beta[:, j] = tt_beta_raw[:, tt_idx[w]]

    spectral_cosine, pairs = _best_alignment_cosine(r_spectral, tt_beta, return_pairs=True)
    r_self_cosine = _best_alignment_cosine(r_rand1, r_rand2)
    r_spec_vs_rand = 0.5 * (
        _best_alignment_cosine(r_spectral, r_rand1)
        + _best_alignment_cosine(r_spectral, r_rand2)
    )

    result = {
        "spectral_cosine": spectral_cosine,
        "r_self_cosine": r_self_cosine,
        "r_spec_vs_rand": r_spec_vs_rand,
        "pairs": pairs,
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
        "K": K,
        "topica_converged": tt_converged,
        "topica_em_iters": tt_em_iters,
    }
    if verbose:
        print(f"corpus: {result['n_docs']} docs, {result['vocab_size']} vocab, K={K}")
        conv = "converged" if tt_converged else "hit cap"
        print(f"topica EM: {conv} after {tt_em_iters} iterations (em_tol=1e-5)")
        print(f"R-Spectral vs topica-Spectral cosine      : {spectral_cosine:.3f}")
        print(f"R-Spectral vs R-Random (within-R basins)   : {r_spec_vs_rand:.3f}")
        print(f"R Random-vs-Random self-consistency        : {r_self_cosine:.3f}")
        per = sorted((c for _, _, c in pairs))
        print(f"per-topic cosine: min {per[0]:.3f}  median {per[len(per)//2]:.3f}  max {per[-1]:.3f}")
        gap = r_spec_vs_rand - spectral_cosine
        if spectral_cosine >= 0.9:
            verdict = f"PASS — topica reproduces R's Spectral solution (cosine {spectral_cosine:.3f})"
        elif gap <= 0.05:
            verdict = "PASS — topica's Spectral is as close to R's as R's own init variants are"
        elif gap <= 0.15:
            verdict = f"COMPARABLE — within R's own basin spread (gap {gap:.3f})"
        else:
            verdict = f"DIVERGENT — {gap:.3f} further than R's own basins differ; investigate"
        print(verdict)
    return result


if __name__ == "__main__":
    import sys

    if not r_stm_available():
        print("SKIP: Rscript with the 'stm' package is not available")
        sys.exit(0)
    run(verbose=True)
