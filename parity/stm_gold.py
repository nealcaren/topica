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
# Random seeds. Exports each K x V topic-word matrix (vocab-named columns), plus,
# for the Spectral fit, the posterior-mean theta (N x K), the topic-correlation
# matrix (K x K), and the per-topic prevalence effect: the coefficient on the
# rating covariate (design column 2) from regressing each topic's theta on the
# SAME design X (a matched-design point regression, so any gap is the fit, not a
# spline-basis or formula-coding difference).
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
fit_of <- function(seed, init) {
  set.seed(seed)
  stm(documents, vocab, K = KVAL, prevalence = X, init.type = init, verbose = FALSE)
}
beta_mat <- function(f) { b <- exp(f$beta$logbeta[[1]]); colnames(b) <- vocab; b }
fs <- fit_of(1, "Spectral")
write.csv(beta_mat(fs), file.path(dir, "r_spectral.csv"), row.names = FALSE)
write.csv(fs$theta,     file.path(dir, "r_theta.csv"),    row.names = FALSE)
write.csv(topicCorr(fs)$cor, file.path(dir, "r_topiccorr.csv"), row.names = FALSE)
rating_coef <- sapply(seq_len(KVAL),
                      function(k) lm.fit(X, fs$theta[, k])$coefficients[2])
write.csv(data.frame(topic = seq_len(KVAL), rating_coef = rating_coef),
          file.path(dir, "r_effect.csv"), row.names = FALSE)
write.csv(beta_mat(fit_of(11, "Random")), file.path(dir, "r_rand1.csv"), row.names = FALSE)
write.csv(beta_mat(fit_of(22, "Random")), file.path(dir, "r_rand2.csv"), row.names = FALSE)
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
        reads=["r_spectral.csv", "r_rand1.csv", "r_rand2.csv", "r_vocab.txt",
               "r_theta.csv", "r_topiccorr.csv", "r_effect.csv"],
        timeout=1800,
    )
    r_vocab = out["r_vocab.txt"].split()
    r_spectral = harness.read_r_beta_csv(out["r_spectral.csv"], r_vocab)
    r_rand1 = harness.read_r_beta_csv(out["r_rand1.csv"], r_vocab)
    r_rand2 = harness.read_r_beta_csv(out["r_rand2.csv"], r_vocab)
    r_theta = _read_float_csv(out["r_theta.csv"])                 # (N, K)
    r_corr = _read_float_csv(out["r_topiccorr.csv"])              # (K, K)
    r_effect = _read_float_csv(out["r_effect.csv"])[:, 1]         # rating_coef column

    # R's own seed-to-seed noise floor (Random-vs-Random) and the fair Spectral
    # basin yardstick (Spectral-vs-Random).
    noise_floor, _ = harness.align_cosine(r_rand1, r_rand2)
    spec_vs_rand = 0.5 * (
        harness.align_cosine(r_spectral, r_rand1)[0]
        + harness.align_cosine(r_spectral, r_rand2)[0]
    )

    # topica fit + the full aligned parity (theta / topic-corr / rating effect),
    # captured at regenerate time for the provenance log.
    model = _fit_topica(docs, X, feat_names)
    parity = _aligned_parity(
        model, r_vocab, r_spectral, r_theta, r_corr, r_effect, X, feat_names
    )

    harness.save_gold(
        NAME,
        # Only r_spectral is needed offline for the beta bar (and the non-vacuous
        # shuffle check); the Random betas fed the noise floor / Spectral-vs-Random
        # bar recorded in the JSON. r_theta/r_topiccorr/r_effect back the
        # whole-model parity (theta, Sigma-as-correlation, gamma-as-effect).
        # Floats stored as float32 to keep the fixture small.
        arrays={
            "r_spectral": r_spectral.astype(np.float32),
            "vocab": np.array(r_vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "design": design.astype(np.float64),
            "feat_names": np.array(feat_names, dtype=object),
            "r_theta": r_theta.astype(np.float32),
            "r_topiccorr": r_corr.astype(np.float32),
            "r_effect": r_effect.astype(np.float64),
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
            "topica_vs_r_spectral_cosine": parity["beta_cosine"],
            "theta_cosine": parity["theta_cosine"],
            "topic_corr_cosine": parity["topic_corr_cosine"],
            "effect_corr": parity["effect_corr"],
            "effect_sign_agree": parity["effect_sign_agree"],
            "effect_sign_total": parity["effect_sign_total"],
            "pass_bar": "topica cosine >= r_spectral_vs_random - margin",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name}")
    print(f"  corpus: {len(docs)} docs, {len(r_vocab)} vocab, K={K}")
    print(f"  R noise floor (rand-vs-rand): {noise_floor:.4f}")
    print(f"  R Spectral-vs-Random:        {spec_vs_rand:.4f}")
    print(f"  topica-vs-R Spectral cosine: {parity['beta_cosine']:.4f}")
    print(f"  theta cosine:                {parity['theta_cosine']:.4f}")
    print(f"  topic-correlation cosine:    {parity['topic_corr_cosine']:.4f}")
    print(f"  rating-effect correlation:   {parity['effect_corr']:.4f}")
    print(f"  effect sign agreement:       {parity['effect_sign_agree']}/{K}")


def _read_float_csv(text: str) -> np.ndarray:
    """Parse a headered all-numeric CSV (R ``write.csv`` output) into a float array."""
    return np.genfromtxt(io.StringIO(text), delimiter=",", skip_header=1)


def _fit_topica(docs, X, feat_names):
    from topica import STM

    model = STM(num_topics=K, init="spectral")
    model.fit(docs, X, prevalence_names=feat_names, iters=ITERS, convergence_tol=CONV_TOL)
    return model


def _topica_rating_effect(model, X, feat_names) -> np.ndarray:
    """Per-topic coefficient on the rating covariate from the matched-design point
    regression (topica's estimate_effect on the point theta), for the effect gold."""
    from topica.stm import estimate_effect

    effs = estimate_effect(np.asarray(model.doc_topic), X, feature_names=feat_names)
    ri = effs[0].feature_names.index(feat_names[0])  # rating is the first covariate
    return np.array([e.coef[ri] for e in effs])


def _aligned_parity(model, r_vocab, r_spectral, r_theta, r_corr, r_effect, X, feat_names):
    """Align topica's topics onto R's (by beta cosine) and compare theta, the
    topic-correlation matrix, and the per-topic rating effect. gamma and Sigma
    live in the (K-1) reference space and are not simply permutable, so we validate
    their interpretable K-space forms: the composition effect (for gamma) and the
    topic correlation (for Sigma)."""
    t_beta = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)
    beta_cos, perm = harness.align_cosine(r_spectral, t_beta)

    # theta: mean per-document cosine between R's and topica's (aligned) doc-topic rows.
    t_theta = np.asarray(model.doc_topic)[:, perm]
    tn = t_theta / (np.linalg.norm(t_theta, axis=1, keepdims=True) + 1e-12)
    rn = r_theta / (np.linalg.norm(r_theta, axis=1, keepdims=True) + 1e-12)
    theta_cos = float(np.mean(np.sum(rn * tn, axis=1)))

    # topic correlation (Sigma's interpretable form): off-diagonal cosine.
    t_corr = np.asarray(model.topic_correlation)[np.ix_(perm, perm)]
    iu = np.triu_indices(K, k=1)
    a, b = r_corr[iu], t_corr[iu]
    corr_cos = float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

    # rating effect (gamma's interpretable form): correlation + sign agreement across topics.
    t_eff = _topica_rating_effect(model, X, feat_names)[perm]
    if np.std(r_effect) > 0 and np.std(t_eff) > 0:
        eff_corr = float(np.corrcoef(r_effect, t_eff)[0, 1])
    else:
        eff_corr = float("nan")
    sign_agree = int(np.sum(np.sign(r_effect) == np.sign(t_eff)))

    return {
        "beta_cosine": beta_cos,
        "theta_cosine": theta_cos,
        "topic_corr_cosine": corr_cos,
        "effect_corr": eff_corr,
        "effect_sign_agree": sign_agree,
        "effect_sign_total": K,
    }


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

    model = _fit_topica(docs, X, feat_names)
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

    # Whole-model parity, gated when the extended gold is present (theta,
    # Sigma-as-topic-correlation, gamma-as-rating-effect). These back the doc
    # claims the beta-only gold used to leave unverified.
    if "r_theta" in arrays:
        parity = _aligned_parity(
            model, r_vocab, r_spectral,
            np.asarray(arrays["r_theta"], dtype=np.float64),
            np.asarray(arrays["r_topiccorr"], dtype=np.float64),
            np.asarray(arrays["r_effect"], dtype=np.float64),
            X, feat_names,
        )
        result.update(parity)
        # Bars: theta and topic-correlation are high-agreement K-space quantities;
        # the effect correlation is the substantive-conclusion bar. Set below the
        # regenerate-time values (theta 0.967, topic-corr 0.983, effect 0.977) with
        # generous headroom for cross-platform EM drift (the topica refit can land
        # in a slightly different basin under a different BLAS).
        result["theta_passes"] = bool(parity["theta_cosine"] >= 0.85)
        result["corr_passes"] = bool(parity["topic_corr_cosine"] >= 0.85)
        result["effect_passes"] = bool(parity["effect_corr"] >= 0.80)
        result["passes"] = bool(
            result["passes"] and result["theta_passes"]
            and result["corr_passes"] and result["effect_passes"]
        )

    if verbose:
        print(f"corpus: {result['n_docs']} docs, {result['vocab_size']} vocab (gold: {meta.get('reference')})")
        print(f"  topica-vs-R Spectral cosine : {spectral_cosine:.4f}")
        print(f"  top-10 Jaccard              : {jaccard:.4f}")
        if "theta_cosine" in result:
            print(f"  theta cosine                : {result['theta_cosine']:.4f} (bar 0.85)")
            print(f"  topic-correlation cosine    : {result['topic_corr_cosine']:.4f} (bar 0.85)")
            print(f"  rating-effect correlation   : {result['effect_corr']:.4f} (bar 0.80)")
            print(f"  effect sign agreement       : {result['effect_sign_agree']}/{result['effect_sign_total']}")
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
