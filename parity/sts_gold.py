"""Committed-gold parity for topica STS vs the R `stm` package (issue #271, final wave).

topica's STS (Structural Topic and Sentiment-Discourse model, Chen & Mankad 2024)
extends STM with a per-document, per-topic continuous sentiment-discourse latent
that modulates the topic-word distribution. The live ``parity/sts_r_compare.py``
validates topica-STS against the authors' *published* poliblog STS fit (a 12 MB
full-corpus ``Poliblogs_results.RDS``, ~13k docs) — far too big to commit, which is
why this gold was deferred to the final wave.

REFERENCE PATH (recorded honestly in the JSON). The authors' reference is a single
pre-fit RDS on the full corpus: you cannot re-fit it on a subsample, so it cannot
be the offline reference for a *slim* corpus. We therefore freeze the structural
cross-implementation reference that CAN be re-fit on the slim corpus: **R `stm` on
the identical fixed-seed slim subsample**. STS reduces to STM structurally (STS =
STM + a sentiment direction), so "topica-STS aligns to R-stm-on-slim about as well
as topica-STM does" is the faithful slim-corpus bar. The authors'-RDS full-corpus
comparison stays live in ``parity/sts_r_compare.py`` (which needs the 12 MB package).

The corpus is the SAME fixed-seed 2,000-doc poliblog subsample the STM gold uses
(``harness.poliblog_corpus``), at K=5 (the paper's poliblog K) with the Liberal
rating dummy driving both prevalence and the sentiment seed — exactly the
``X <- X_seed <- model.matrix(~rating)[,-1]`` design of the replication script.

Two phases (mirrors parity/stm_gold.py):

  * ``--regenerate`` (needs Rscript + the ``stm`` package): fits R `stm` once on
    the slim corpus, computes R's own seed-to-seed noise floor, and writes the
    committed gold (``parity/sts_gold.npz`` + ``.json``). The gold stores R's beta,
    the vocab, the exact tokenized corpus + sentiment seed (for an offline refit),
    and the noise floor.
  * default (no R): loads the committed gold, fits topica STS AND topica STM on the
    same corpus, aligns STS's mean-sentiment topic-word to R-stm's beta, and checks
    (a) STS clears the cross-impl bar, and (b) STS agrees with topica's own STM.

Run directly::

    python parity/sts_gold.py               # offline compare against committed gold
    python parity/sts_gold.py --regenerate  # run R once, write the gold
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness

NAME = "sts"
K = 5
STS_ITERS = 50
STM_ITERS = 80
CONV_TOL = 1e-5
# STS reads its recognizable topic at the mean sentiment (the reference's "Avg
# alpha" print.topWords view), not at neutral. Margin: topica-STS may sit no
# further from R-stm than R-stm's own Spectral-vs-Random basins differ, plus slack.
MARGIN = 0.15
# topica-STS should agree with topica's OWN STM at least this well (STS extends STM).
STS_VS_STM_MIN = 0.85

# R driver: read space-joined docs + the 0/1 rating dummy, fit stm with rating as a
# single-column prevalence design (Spectral + two Random seeds), export each K x V
# topic-word matrix (vocab-named columns) plus the vocab.
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
rating <- read.csv(file.path(dir, "rating.csv"))$rating
meta <- data.frame(rating = rating)
beta_of <- function(seed, init) {
  set.seed(seed)
  f <- stm(documents, vocab, K = KVAL, prevalence = ~rating,
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


def _to_r_vocab(raw: np.ndarray, vocab: list[str], r_vocab: list[str]) -> np.ndarray:
    raw = np.asarray(raw)
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


def _beta_at_mean_sentiment(sts) -> np.ndarray:
    """topica STS topic-word read at each topic's mean sentiment-discourse alpha^(s),
    (K, V) on the model's own vocabulary — the reference's representative-topic view.
    The topic signal lives in the sentiment direction, so the recognizable topic is
    beta_k evaluated at mean_d alpha^(s)_{d,k}, not at neutral sentiment."""
    mean_s = np.asarray(sts.sentiment).mean(axis=0)
    return np.vstack([
        np.asarray(sts.topic_word_at(float(mean_s[k])))[k]
        for k in range(len(mean_s))
    ])


def _fit_topica(docs, rating, r_vocab):
    """Fit topica STS + STM on the slim corpus; return their R-vocab-aligned betas."""
    from topica.models import STM, STS

    X = rating.reshape(-1, 1)
    sts = STS(num_topics=K, init="spectral")
    sts.fit(docs, sentiment_seed=rating.tolist(), prevalence=X,
            prevalence_names=["rating"], iters=STS_ITERS,
            convergence_tol=CONV_TOL, kappa_estimation="lasso")
    stm = STM(num_topics=K, init="spectral")
    stm.fit(docs, X, prevalence_names=["rating"], iters=STM_ITERS,
            convergence_tol=CONV_TOL)

    t_sts = _to_r_vocab(_beta_at_mean_sentiment(sts), list(sts.vocabulary), r_vocab)
    t_stm = _to_r_vocab(np.asarray(stm.topic_word), list(stm.vocabulary), r_vocab)
    return t_sts, t_stm


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not harness.r_available("stm"):
        print("Rscript with the 'stm' package not available; cannot regenerate.")
        sys.exit(1)

    docs, rating_lib, _day, _ = harness.poliblog_corpus()

    vdocs = harness.docs_to_lines(docs)
    rating_csv = "rating\n" + "\n".join(str(int(v)) for v in rating_lib) + "\n"

    out = harness.run_rscript(
        _R_DRIVER.replace("KVAL", str(K)),
        files={"vdocs.txt": vdocs, "rating.csv": rating_csv},
        reads=["r_spectral.csv", "r_rand1.csv", "r_rand2.csv", "r_vocab.txt"],
        timeout=1800,
    )
    r_vocab = out["r_vocab.txt"].split()
    r_spectral = harness.read_r_beta_csv(out["r_spectral.csv"], r_vocab)
    r_rand1 = harness.read_r_beta_csv(out["r_rand1.csv"], r_vocab)
    r_rand2 = harness.read_r_beta_csv(out["r_rand2.csv"], r_vocab)

    # R's own seed-to-seed noise floor + the fair Spectral basin yardstick.
    noise_floor, _ = harness.align_cosine(r_rand1, r_rand2)
    spec_vs_rand = 0.5 * (
        harness.align_cosine(r_spectral, r_rand1)[0]
        + harness.align_cosine(r_spectral, r_rand2)[0]
    )

    # topica fit summary captured at regenerate time for the provenance log.
    t_sts, t_stm = _fit_topica(docs, rating_lib, r_vocab)
    sts_vs_r, _ = harness.align_cosine(r_spectral, t_sts)
    stm_vs_r, _ = harness.align_cosine(r_spectral, t_stm)
    sts_vs_stm, _ = harness.align_cosine(t_stm, t_sts)

    harness.save_gold(
        NAME,
        # Only r_spectral is needed offline (+ the non-vacuous shuffle check); the
        # Random betas only fed the noise floor / Spectral-vs-Random bar (recorded
        # in the JSON), so they are not committed. beta as float32 keeps the fixture
        # small without affecting the cosine alignment.
        arrays={
            "r_spectral": r_spectral.astype(np.float32),
            "vocab": np.array(r_vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "rating": rating_lib.astype(np.float64),
        },
        meta={
            "reference": _r_version(),
            "model": "STS (Chen & Mankad 2024) vs R stm-on-slim",
            "reference_path": (
                "R stm baseline on the slim subsample (the authors' poliblog STS "
                "RDS is a full-corpus pre-fit that cannot be re-fit on a subsample; "
                "STS reduces to STM structurally, so R-stm-on-slim is the slim-corpus "
                "cross-impl reference. The authors'-RDS full-corpus comparison stays "
                "in parity/sts_r_compare.py)"
            ),
            "corpus": (
                f"poliblog (examples/poliblog.csv), fixed-seed {harness.POLIBLOG_N_DOCS}-doc "
                f"subsample (seed {harness.POLIBLOG_SEED}), vignette preprocessing — the "
                "same slim corpus as the STM gold"
            ),
            "num_docs": len(docs),
            "vocab_size": len(r_vocab),
            "K": K,
            "design": "~rating (Liberal dummy) drives both prevalence and the sentiment seed",
            "kappa_estimation": "lasso (matches the reference glmnet estimator)",
            "init": "Spectral",
            "seeds": {"spectral": 1, "random_1": 11, "random_2": 22},
            "sts_iters": STS_ITERS,
            "stm_iters": STM_ITERS,
            "convergence_tol": CONV_TOL,
            "topic_word_view": "mean sentiment-discourse alpha^(s) (the reference print.topWords 'Avg alpha' view)",
            "date": datetime.date.today().isoformat(),
            "noise_floor_random_vs_random": noise_floor,
            "r_spectral_vs_random": spec_vs_rand,
            "margin": MARGIN,
            "sts_vs_stm_min": STS_VS_STM_MIN,
            "topica_sts_vs_r_cosine": sts_vs_r,
            "topica_stm_vs_r_cosine": stm_vs_r,
            "topica_sts_vs_stm_cosine": sts_vs_stm,
            "pass_bar": (
                "topica-STS cosine vs R-stm >= max(topica-STM cosine vs R-stm, "
                "r_spectral_vs_random) - margin; AND topica-STS vs topica-STM "
                ">= sts_vs_stm_min (STS extends STM)"
            ),
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  corpus: {len(docs)} docs, {len(r_vocab)} vocab, K={K}")
    print(f"  R noise floor (rand-vs-rand): {noise_floor:.4f}")
    print(f"  R Spectral-vs-Random:        {spec_vs_rand:.4f}")
    print(f"  topica-STS vs R-stm:         {sts_vs_r:.4f}")
    print(f"  topica-STM vs R-stm:         {stm_vs_r:.4f}  <- cross-impl baseline")
    print(f"  topica-STS vs topica-STM:    {sts_vs_stm:.4f}  <- STS extends STM")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    r_spectral = np.asarray(arrays["r_spectral"], dtype=np.float64)
    r_vocab = [str(w) for w in arrays["vocab"]]
    spec_vs_rand = float(meta.get("r_spectral_vs_random", 0.0))
    noise_floor = float(meta.get("noise_floor_random_vs_random", 0.0))

    docs = harness.lines_to_docs(str(arrays["corpus"]))
    rating = np.asarray(arrays["rating"], dtype=np.float64)

    t_sts, t_stm = _fit_topica(docs, rating, r_vocab)

    sts_vs_r, _ = harness.align_cosine(r_spectral, t_sts)
    stm_vs_r, _ = harness.align_cosine(r_spectral, t_stm)
    sts_vs_stm, _ = harness.align_cosine(t_stm, t_sts)
    jaccard = harness.top_word_jaccard(r_spectral, t_sts, n=10)

    # Bar: STS should sit as close to R-stm as topica's own (already validated) STM
    # does — accounting for the irreducible cross-impl gap — within the Spectral
    # basin slack. AND STS must agree with topica's STM (it structurally extends it).
    cross_impl_bar = max(stm_vs_r, spec_vs_rand) - MARGIN
    passes = (
        sts_vs_r >= cross_impl_bar
        and sts_vs_stm >= STS_VS_STM_MIN
    )
    result = {
        "sts_vs_r_cosine": sts_vs_r,
        "stm_vs_r_cosine": stm_vs_r,
        "sts_vs_stm_cosine": sts_vs_stm,
        "top_word_jaccard": jaccard,
        "r_spectral_vs_random": spec_vs_rand,
        "noise_floor": noise_floor,
        "cross_impl_bar": cross_impl_bar,
        "sts_vs_stm_min": STS_VS_STM_MIN,
        "margin_over_bar": sts_vs_r - cross_impl_bar,
        "passes": bool(passes),
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
    }
    if verbose:
        print(f"corpus: {result['n_docs']} docs, {result['vocab_size']} vocab (gold: {meta.get('reference')})")
        print(f"  reference path: {meta.get('reference_path', '?')[:70]}...")
        print(f"  topica-STS vs R-stm  : {sts_vs_r:.4f}  (top-10 Jaccard {jaccard:.4f})")
        print(f"  topica-STM vs R-stm  : {stm_vs_r:.4f}  <- cross-impl baseline")
        print(f"  topica-STS vs -STM   : {sts_vs_stm:.4f}  (>= {STS_VS_STM_MIN}; STS extends STM)")
        print(f"  R Spectral-vs-Random : {spec_vs_rand:.4f}  (noise floor {noise_floor:.4f})")
        print(f"  cross-impl bar       : {cross_impl_bar:.4f}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} (margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
