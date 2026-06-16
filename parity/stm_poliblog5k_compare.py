"""Cross-implementation validation: topica STM vs R `stm` on the FULL poliblog5k
corpus (5,000 docs), to test whether the aligned topic-word cosine rises with n
relative to the small-corpus parity checks (gadarian K=3, 339 docs).

Mirrors parity/stm_poliblog_compare.py's methodology (one shared numeric design
matrix, identical integer-coded docs, Spectral + two Random R seeds, greedy
one-to-one topic alignment, mean cosine) but feeds the bundled poliblog5k.

Run:
    POLIBLOG5K_K=15 python parity/stm_poliblog5k_compare.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.environ.get("POLIBLOG5K_CSV") or os.path.join(
    ROOT, "benchmarks", "poliblog5k_prepped.csv")

K = int(os.environ.get("POLIBLOG5K_K", "15"))
SPLINE_DF = int(os.environ.get("POLIBLOG5K_SPLINE_DF", "10"))
EM_ITERS = int(os.environ.get("POLIBLOG5K_EM_ITERS", "200"))
EM_TOL = float(os.environ.get("POLIBLOG5K_EM_TOL", "1e-5"))
LIMIT = int(os.environ.get("POLIBLOG5K_LIMIT", "0"))  # 0 = all docs


def r_stm_available() -> bool:
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


def load_and_prep():
    """Load the already-prepped poliblog5k. Vocab is already pruned by stm, so we
    do not prune further. Returns (docs, rating_lib, day)."""
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if LIMIT:
        rows = rows[:LIMIT]
    docs = [r["text"].split() for r in rows]
    rating_lib = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])
    keep = np.array([len(d) > 0 for d in docs])
    docs = [d for d, k in zip(docs, keep) if k]
    return docs, rating_lib[keep], day[keep]


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
X <- as.matrix(read.csv(file.path(dir, "design.csv")))
beta_of <- function(seed, init) {
  set.seed(seed)
  f <- stm(documents, vocab, K = KVAL, prevalence = X,
           init.type = init, verbose = FALSE,
           emtol = EMTOL, max.em.its = EMITS)
  b <- exp(f$beta$logbeta[[1]]); colnames(b) <- vocab; b
}
write.csv(beta_of(1, "Spectral"), file.path(dir, "r_spectral.csv"), row.names = FALSE)
write.csv(beta_of(11, "Random"),  file.path(dir, "r_rand1.csv"),    row.names = FALSE)
write.csv(beta_of(22, "Random"),  file.path(dir, "r_rand2.csv"),    row.names = FALSE)
write(vocab, file.path(dir, "r_vocab.txt"))
cat("ok\n")
"""


def _read_r_beta(path, vocab):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        cols = [h.strip('"') for h in next(rdr)]
        rows = [[float(x) for x in row] for row in rdr]
    mat = np.array(rows)
    idx = {w: i for i, w in enumerate(cols)}
    out = np.zeros((mat.shape[0], len(vocab)))
    for j, w in enumerate(vocab):
        if w in idx:
            out[:, j] = mat[:, idx[w]]
    return out


def _best_alignment_cosine(a, b, return_pairs=False):
    k = a.shape[0]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    used = set(); total = 0.0; pairs = []
    for i in np.argsort(-sim.max(axis=1)):
        for j in np.argsort(-sim[i]):
            if j not in used:
                used.add(j); total += sim[i, j]
                pairs.append((int(i), int(j), float(sim[i, j]))); break
    return (total / k, pairs) if return_pairs else total / k


def run(verbose: bool = True) -> dict:
    if not r_stm_available():
        raise RuntimeError("Rscript with the 'stm' package is not available")

    from topica import STM
    from topica.stm import spline

    docs, rating_lib, day = load_and_prep()

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

        script = (f'dir <- "{d}"\nKVAL <- {K}\nEMTOL <- {EM_TOL}\n'
                  f'EMITS <- {EM_ITERS}\n' + _R_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, timeout=7200)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        r_vocab = open(os.path.join(d, "r_vocab.txt")).read().split()
        r_spectral = _read_r_beta(os.path.join(d, "r_spectral.csv"), r_vocab)
        r_rand1 = _read_r_beta(os.path.join(d, "r_rand1.csv"), r_vocab)
        r_rand2 = _read_r_beta(os.path.join(d, "r_rand2.csv"), r_vocab)

        model = STM(num_topics=K, init="spectral")
        model.fit(docs, X, prevalence_names=feat_names, iters=EM_ITERS,
                  convergence_tol=EM_TOL)
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
        + _best_alignment_cosine(r_spectral, r_rand2))
    tt_vs_rrand = 0.5 * (
        _best_alignment_cosine(tt_beta, r_rand1)
        + _best_alignment_cosine(tt_beta, r_rand2))

    result = {
        "spectral_cosine": spectral_cosine, "r_self_cosine": r_self_cosine,
        "r_spec_vs_rand": r_spec_vs_rand, "tt_vs_rrand": tt_vs_rrand,
        "pairs": pairs, "vocab_size": len(r_vocab), "n_docs": len(docs), "K": K,
        "topica_converged": tt_converged, "topica_em_iters": tt_em_iters,
    }
    if verbose:
        print(f"corpus: {result['n_docs']} docs, {result['vocab_size']} vocab, K={K}, "
              f"EM_ITERS={EM_ITERS}, EM_TOL={EM_TOL}")
        conv = "converged" if tt_converged else "hit cap"
        print(f"topica EM: {conv} after {tt_em_iters} iterations")
        print(f"R-Spectral vs topica-Spectral cosine       : {spectral_cosine:.3f}")
        print(f"R-Spectral vs R-Random (within-R basins)    : {r_spec_vs_rand:.3f}")
        print(f"R Random-vs-Random self-consistency         : {r_self_cosine:.3f}")
        print(f"topica-Spectral vs R-Random                 : {tt_vs_rrand:.3f}")
        per = sorted(c for _, _, c in pairs)
        print(f"per-topic cosine: min {per[0]:.3f}  median {per[len(per)//2]:.3f}  max {per[-1]:.3f}")
        print(f"all pairs: {[round(c, 3) for c in per]}")
    return result


if __name__ == "__main__":
    import sys
    if not r_stm_available():
        print("SKIP: Rscript with the 'stm' package is not available")
        sys.exit(0)
    if not os.path.exists(CSV):
        print(f"SKIP: {CSV} not found (gitignored; build it with "
              "benchmarks/export_poliblog5k.R or set POLIBLOG5K_CSV)")
        sys.exit(0)
    run(verbose=True)
