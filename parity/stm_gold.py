"""Committed-gold parity for topica STM vs the R `stm` package (issue #271, Wave 1).

R's `stm` and topica are independent implementations of the same model (logistic-
normal variational EM with prevalence covariates and Arora-style spectral
initialization). They share no code and no RNG, so agreement is *statistical*: fit
both on the SAME tokenized corpus + design matrix and ask whether they land on the
same topics.

The corpus is a fixed-seed 2,000-document subsample of the poliblog vignette
(Roberts, Stewart & Tingley's JSS example), at the canonical
``prevalence = ~ rating + s(day)`` design and K=20. Unlike the multimodal
gadarian K=3 corpus, poliblog K=20 is well-identified: under matched Spectral
init topica lands essentially on R's solution (aligned topic-word cosine ~0.9),
far above gadarian's ~0.55. So the absolute cosine — not just the gap to R's own
seed-to-seed noise floor — is a meaningful validation.

We build ONE numeric design matrix (rating dummy + a single day-spline basis) in
Python and hand the SAME matrix to both engines, so any gap is the inference
engine, not a basis or coding difference.

Two phases:

  * ``--regenerate`` (needs Rscript + the ``stm`` package): fits R `stm` once,
    computes R's own seed-to-seed noise floor, and writes the committed gold
    (``parity/stm_gold.npz`` + ``.json``). The gold stores R's beta, the vocab,
    the exact tokenized corpus + design matrix (for an offline refit), and the
    noise floor.
  * default (no R): loads the committed gold, fits topica STM on the same corpus +
    vocab, aligns to R's beta, and checks the bar.

Run directly::

    python parity/stm_gold.py               # offline compare against committed gold
    python parity/stm_gold.py --regenerate  # run R once, write the gold
"""

from __future__ import annotations

import csv
import datetime
import io
import sys

import numpy as np

import harness

NAME = "stm"
K = 20
ITERS = 200
CONV_TOL = 1e-5
SPLINE_DF = 10
PREVALENCE = "~ rating + s(day)"
# Multimodality margin: topica may land in a neighboring basin no further from R's
# Spectral solution than R's own Spectral-vs-Random basins differ, plus this slack.
# On well-identified poliblog the absolute cosine clears the bar comfortably.
MARGIN = 0.15

# R driver: reads space-joined docs + a numeric design matrix (intercept already
# included) and fits stm with that matrix as `prevalence`, Spectral plus two
# Random seeds. Exports each K x V topic-word matrix (vocab-named columns).
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


def _r_version() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["Rscript", "-e",
             'cat(as.character(getRversion()), as.character(packageVersion("stm")))'],
            capture_output=True, text=True, timeout=60,
        )
        rv, sv = out.stdout.strip().split()
        return f"R {rv} / stm {sv}"
    except Exception:
        return "R / stm (version unknown)"


def _design(rating_lib, day):
    """The ONE design matrix shared by both engines and its feature names.

    topica auto-prepends an intercept and takes the bare covariates; R's
    matrix-prevalence path takes the intercept column explicitly. Returns
    ``(X_bare, feat_names, design_with_intercept)``.
    """
    from topica.stm import spline

    spline_basis, _ = spline(day, df=SPLINE_DF)
    X = np.column_stack([rating_lib, spline_basis])
    feat_names = ["ratingLiberal"] + [f"day_s{j}" for j in range(spline_basis.shape[1])]
    design_with_intercept = np.column_stack([np.ones(len(rating_lib)), X])
    return X, feat_names, design_with_intercept


def regenerate() -> None:
    if not harness.r_available("stm"):
        print("Rscript with the 'stm' package not available; cannot regenerate.")
        sys.exit(1)

    docs, rating_lib, day, _ = harness.poliblog_corpus()
    X, feat_names, design = _design(rating_lib, day)

    vdocs = harness.docs_to_lines(docs)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["intercept"] + feat_names)
    for row in design:
        w.writerow(list(row))
    design_csv = buf.getvalue()

    out = harness.run_rscript(
        _R_DRIVER.replace("KVAL", str(K)),
        files={"vdocs.txt": vdocs, "design.csv": design_csv},
        reads=["r_spectral.csv", "r_rand1.csv", "r_rand2.csv", "r_vocab.txt"],
        timeout=1800,
    )
    r_vocab = out["r_vocab.txt"].split()
    r_spectral = harness.read_r_beta_csv(out["r_spectral.csv"], r_vocab)
    r_rand1 = harness.read_r_beta_csv(out["r_rand1.csv"], r_vocab)
    r_rand2 = harness.read_r_beta_csv(out["r_rand2.csv"], r_vocab)

    # R's own seed-to-seed noise floor (Random-vs-Random) and the fair Spectral
    # basin yardstick (Spectral-vs-Random).
    noise_floor, _ = harness.align_cosine(r_rand1, r_rand2)
    spec_vs_rand = 0.5 * (
        harness.align_cosine(r_spectral, r_rand1)[0]
        + harness.align_cosine(r_spectral, r_rand2)[0]
    )

    # topica fit summary captured at regenerate time for the provenance log.
    t_cos = _topica_cosine(docs, X, feat_names, r_vocab, r_spectral)

    harness.save_gold(
        NAME,
        # Only r_spectral is needed offline (and for the non-vacuous shuffle
        # check); the Random betas were used here to compute the noise floor /
        # Spectral-vs-Random bar, which are recorded in the JSON, so they're not
        # committed. beta as float32 keeps the fixture small without affecting
        # the cosine alignment.
        arrays={
            "r_spectral": r_spectral.astype(np.float32),
            "vocab": np.array(r_vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "design": design.astype(np.float64),
            "feat_names": np.array(feat_names, dtype=object),
        },
        meta={
            "reference": _r_version(),
            "model": "STM",
            "corpus": (
                f"poliblog (examples/poliblog.csv), fixed-seed {harness.POLIBLOG_N_DOCS}-doc "
                f"subsample (seed {harness.POLIBLOG_SEED}), vignette preprocessing"
            ),
            "num_docs": len(docs),
            "vocab_size": len(r_vocab),
            "K": K,
            "formula": PREVALENCE,
            "spline_df": SPLINE_DF,
            "init": "Spectral",
            "seeds": {"spectral": 1, "random_1": 11, "random_2": 22},
            "topica_iters": ITERS,
            "convergence_tol": CONV_TOL,
            "date": datetime.date.today().isoformat(),
            "noise_floor_random_vs_random": noise_floor,
            "r_spectral_vs_random": spec_vs_rand,
            "margin": MARGIN,
            "topica_vs_r_spectral_cosine": t_cos,
            "pass_bar": "topica cosine >= r_spectral_vs_random - margin",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name}")
    print(f"  corpus: {len(docs)} docs, {len(r_vocab)} vocab, K={K}")
    print(f"  R noise floor (rand-vs-rand): {noise_floor:.4f}")
    print(f"  R Spectral-vs-Random:        {spec_vs_rand:.4f}")
    print(f"  topica-vs-R Spectral cosine: {t_cos:.4f}")


def _topica_cosine(docs, X, feat_names, r_vocab, r_spectral) -> float:
    from topica.models import STM

    model = STM(num_topics=K, init="spectral")
    model.fit(docs, X, prevalence_names=feat_names, iters=ITERS, convergence_tol=CONV_TOL)
    t_beta = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)
    cos, _ = harness.align_cosine(r_spectral, t_beta)
    return cos


def _to_r_vocab(raw: np.ndarray, vocab: list[str], r_vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


def run(verbose: bool = True) -> dict:
    """Offline compare: load committed gold, fit topica, return metrics."""
    arrays, meta = harness.load_gold(NAME)
    r_spectral = arrays["r_spectral"]
    r_vocab = list(arrays["vocab"])
    spec_vs_rand = float(meta.get("r_spectral_vs_random", 0.0))
    noise_floor = float(meta.get("noise_floor_random_vs_random", 0.0))

    # Refit on the EXACT corpus + design matrix frozen in the gold (offline; no R).
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    X = arrays["design"][:, 1:]  # drop the intercept column (topica re-adds it)
    feat_names = list(arrays["feat_names"])

    from topica.models import STM

    model = STM(num_topics=K, init="spectral")
    model.fit(docs, X, prevalence_names=feat_names, iters=ITERS, convergence_tol=CONV_TOL)
    t_beta = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)

    spectral_cosine, _ = harness.align_cosine(r_spectral, t_beta)
    jaccard = harness.top_word_jaccard(r_spectral, t_beta, n=10)
    bar = spec_vs_rand - MARGIN

    result = {
        "spectral_cosine": spectral_cosine,
        "top_word_jaccard": jaccard,
        "r_spectral_vs_random": spec_vs_rand,
        "noise_floor": noise_floor,
        "bar": bar,
        "margin_over_bar": spectral_cosine - bar,
        "passes": bool(spectral_cosine >= bar),
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
    }
    if verbose:
        print(f"corpus: {result['n_docs']} docs, {result['vocab_size']} vocab (gold: {meta.get('reference')})")
        print(f"  topica-vs-R Spectral cosine : {spectral_cosine:.4f}")
        print(f"  top-10 Jaccard              : {jaccard:.4f}")
        print(f"  R Spectral-vs-Random        : {spec_vs_rand:.4f}")
        print(f"  R rand-vs-rand noise floor  : {noise_floor:.4f}")
        print(f"  bar (spec_vs_rand - {MARGIN}) : {bar:.4f}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} (margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
