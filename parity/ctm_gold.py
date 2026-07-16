"""Committed-gold parity for topica CTM vs the R `stm` package fit as a CTM
(no covariates) — issue #271, Wave 1.

A Structural Topic Model with no prevalence or content covariates *is* a Correlated
Topic Model: the same logistic-normal variational EM with Arora-style spectral
initialization. topica's CTM and R `stm` are independent implementations, so
validation is statistical: fit both on the SAME tokenized corpus and ask whether
they land on the same topics.

The corpus is the same fixed-seed 2,000-document poliblog subsample as the STM
gold (Roberts, Stewart & Tingley's JSS example), at K=20 but with no covariates.
Unlike the multimodal gadarian K=3 corpus, poliblog K=20 is well-identified:
under matched Spectral init topica lands essentially on R's solution (aligned
topic-word cosine ~0.9), far above gadarian's ~0.55. So the absolute cosine — not
just the gap to R's own seed-to-seed noise floor — is a meaningful validation.

Two phases:

  * ``--regenerate`` (needs Rscript + the ``stm`` package): fits R `stm` (no
    covariates) once, computes R's noise floor, writes ``parity/ctm_gold.npz`` +
    ``.json`` (R's beta, vocab, the exact tokenized corpus, the noise floor).
  * default (no R): loads the committed gold, fits topica CTM, checks the bar.

Run directly::

    python parity/ctm_gold.py
    python parity/ctm_gold.py --regenerate
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness

NAME = "ctm"
K = 20
ITERS = 200
MARGIN = 0.15

# R driver: fits stm with prevalence=NULL (a CTM) Spectral + two Random seeds.
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
beta_of <- function(seed, init) {
  f <- stm(documents, vocab, K = KVAL, prevalence = NULL, init.type = init,
           seed = seed, verbose = FALSE)
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


def _to_r_vocab(raw: np.ndarray, vocab: list[str], r_vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


def _topica_cosine(docs, r_vocab, r_spectral) -> float:
    from topica.models import CTM

    model = CTM(num_topics=K, init="spectral")
    model.fit(docs, iters=ITERS)
    t_beta = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)
    return harness.align_cosine(r_spectral, t_beta)[0]


def regenerate() -> None:
    if not harness.r_available("stm"):
        print("Rscript with the 'stm' package not available; cannot regenerate.")
        sys.exit(1)

    docs, _, _, _ = harness.poliblog_corpus()
    vdocs = harness.docs_to_lines(docs)

    out = harness.run_rscript(
        _R_DRIVER.replace("KVAL", str(K)),
        files={"vdocs.txt": vdocs},
        reads=["r_spectral.csv", "r_rand1.csv", "r_rand2.csv", "r_vocab.txt"],
        timeout=1800,
    )
    r_vocab = out["r_vocab.txt"].split()
    r_spectral = harness.read_r_beta_csv(out["r_spectral.csv"], r_vocab)
    r_rand1 = harness.read_r_beta_csv(out["r_rand1.csv"], r_vocab)
    r_rand2 = harness.read_r_beta_csv(out["r_rand2.csv"], r_vocab)

    noise_floor, _ = harness.align_cosine(r_rand1, r_rand2)
    spec_vs_rand = 0.5 * (
        harness.align_cosine(r_spectral, r_rand1)[0]
        + harness.align_cosine(r_spectral, r_rand2)[0]
    )
    t_cos = _topica_cosine(docs, r_vocab, r_spectral)

    harness.save_gold(
        NAME,
        # Only r_spectral is needed offline (and for the non-vacuous shuffle
        # check); the Random betas were used here for the noise floor /
        # Spectral-vs-Random bar (recorded in the JSON), so they're not
        # committed. beta as float32 keeps the fixture small.
        arrays={
            "r_spectral": r_spectral.astype(np.float32),
            "vocab": np.array(r_vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
        },
        meta={
            "reference": _r_version(),
            "model": "CTM (R stm with prevalence=NULL)",
            "corpus": (
                f"poliblog (examples/poliblog.csv), fixed-seed {harness.POLIBLOG_N_DOCS}-doc "
                f"subsample (seed {harness.POLIBLOG_SEED}), vignette preprocessing"
            ),
            "num_docs": len(docs),
            "vocab_size": len(r_vocab),
            "K": K,
            "formula": None,
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
    print(f"  corpus: {len(docs)} docs, {len(r_vocab)} vocab, K={K}")
    print(f"  R noise floor (rand-vs-rand): {noise_floor:.4f}")
    print(f"  R Spectral-vs-Random:        {spec_vs_rand:.4f}")
    print(f"  topica-vs-R Spectral cosine: {t_cos:.4f}")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    r_spectral = arrays["r_spectral"]
    r_vocab = list(arrays["vocab"])
    spec_vs_rand = float(meta.get("r_spectral_vs_random", 0.0))
    noise_floor = float(meta.get("noise_floor_random_vs_random", 0.0))

    # Refit on the EXACT corpus frozen in the gold (offline; no R).
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    from topica.models import CTM

    model = CTM(num_topics=K, init="spectral")
    model.fit(docs, iters=ITERS)
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
