"""Committed-gold parity for topica STM vs the R `stm` package (issue #271, Wave 1).

R's `stm` and topica are independent implementations of the same model (logistic-
normal variational EM with prevalence covariates and Arora-style spectral
initialization). They share no code and no RNG, so agreement is *statistical*: fit
both on the SAME tokenized gadarian corpus + metadata and ask whether they land on
the same topics.

The benchmark is R's own reproducibility. gadarian K=3 is multimodal (many local
optima), so two R runs from different random seeds only agree to a topic-word
cosine well below 1. Under matched (Spectral) init, topica-vs-R agreement should
meet or exceed R's own Spectral-vs-Random basin spread — i.e. topica is no further
from R's Spectral solution than R's own init variants are from each other.

Two phases:

  * ``--regenerate`` (needs Rscript + the ``stm`` package): fits R `stm` once,
    computes R's own seed-to-seed noise floor, and writes the committed gold
    (``parity/stm_gold.npz`` + ``.json``).
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
K = 3
ITERS = 80
PREVALENCE = "~treatment + pid_rep"
# Multimodality margin: topica may land in a neighboring basin no further from R's
# Spectral solution than R's own Spectral-vs-Random basins differ, plus this slack.
MARGIN = 0.15

# R driver: fits stm Spectral + two Random seeds, exports K x V beta and vocab.
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
meta <- read.csv(file.path(dir, "vmeta.csv"))
beta_of <- function(seed, init) {
  set.seed(seed)
  f <- stm(documents, vocab, K = 3, prevalence = ~treatment + pid_rep,
           data = meta, init.type = init, verbose = FALSE)
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


def regenerate() -> None:
    if not harness.r_available("stm"):
        print("Rscript with the 'stm' package not available; cannot regenerate.")
        sys.exit(1)

    docs, treatment, pid, _ = harness.gadarian_corpus()

    vdocs = "\n".join(" ".join(doc) for doc in docs) + "\n"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["treatment", "pid_rep"])
    for t, p in zip(treatment, pid):
        w.writerow([t, p])
    vmeta = buf.getvalue()

    out = harness.run_rscript(
        _R_DRIVER,
        files={"vdocs.txt": vdocs, "vmeta.csv": vmeta},
        reads=["r_spectral.csv", "r_rand1.csv", "r_rand2.csv", "r_vocab.txt"],
        timeout=600,
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
    t_cos = _topica_cosine(docs, treatment, pid, r_vocab, r_spectral)

    harness.save_gold(
        NAME,
        arrays={
            "r_spectral": r_spectral,
            "r_rand1": r_rand1,
            "r_rand2": r_rand2,
            "vocab": np.array(r_vocab, dtype=object),
        },
        meta={
            "reference": _r_version(),
            "model": "STM",
            "corpus": "gadarian (examples/gadarian.csv), vignette preprocessing",
            "num_docs": len(docs),
            "vocab_size": len(r_vocab),
            "K": K,
            "formula": PREVALENCE,
            "init": "Spectral",
            "seeds": {"spectral": 1, "random_1": 11, "random_2": 22},
            "topica_iters": ITERS,
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
    print(f"  R noise floor (rand-vs-rand): {noise_floor:.4f}")
    print(f"  R Spectral-vs-Random:        {spec_vs_rand:.4f}")
    print(f"  topica-vs-R Spectral cosine: {t_cos:.4f}")


def _topica_cosine(docs, treatment, pid, r_vocab, r_spectral) -> float:
    from topica import STM

    X = np.column_stack([treatment, pid])
    model = STM(num_topics=K, init="spectral")
    model.fit(docs, X, prevalence_names=["treatment", "pid_rep"], iters=ITERS)
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

    docs, treatment, pid, _ = harness.gadarian_corpus()
    from topica import STM

    X = np.column_stack([treatment, pid])
    model = STM(num_topics=K, init="spectral")
    model.fit(docs, X, prevalence_names=["treatment", "pid_rep"], iters=ITERS)
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
