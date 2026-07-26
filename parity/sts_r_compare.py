"""Cross-implementation validation: topica's STS vs the authors' reference R
implementation of the Structural Topic and Sentiment-Discourse model (Chen &
Mankad 2024, *Management Science*).

Both are the same model -- Laplace variational EM over the 2K-1 prevalence +
sentiment-discourse latent, with the topic-word distribution modulated by a
continuous per-document sentiment. They are independent implementations, so
validation is statistical: fit both on the SAME tokenized corpus and ask whether
they recover the same topics.

This uses the authors' published *political-blog* fit (``Poliblogs_results.RDS``,
K=5) from their replication package -- the same corpus the rest of the paper's
worked example runs on. R regenerates the exact corpus (``textProcessor`` +
``prepDocuments(lower.thresh = 30)`` on ``Data/poliblogs2008.csv``), reads the
fitted RDS, and also fits R ``stm`` on the same corpus as a calibration baseline.
topica then fits both STS and STM and we align everything.

The representative topic-word is read at the **mean** sentiment, not at neutral
(sentiment = 0). STS parks topic signal in the sentiment direction kappa^(s) when
the sentiment seed correlates with content, so the topic a reader recognizes is
softmax(m_v + kappa^(t)_k + kappa^(s)_k * mean_d alpha^(s)_{d,k}), the "Avg alpha"
view the reference's ``print.topWords`` reports. (Read at neutral, the two engines
agree far less, because each splits the content/sentiment decomposition its own
way; read at the mean, they recover the same topics.)

The check is benchmarked against the STM baseline rather than an absolute
threshold: topica-STS should sit about as close to R-STS as topica's already
validated STM sits to R-STM, and STS should agree with topica's own STM
(confirming STS extends STM). On this well-conditioned corpus all three land in
the mid-0.90s. topica fits with its ``"lasso"`` kappa estimator (the default is
``"ridge"``) to match the reference's glmnet estimator.

Point the script at the replication package via ``STS_REPL_DIR`` (default
``~/Downloads/mnsc.2022.00261``). Skips (exit 0) if Rscript, the ``stm`` package,
or the replication results are unavailable. Run directly:

    python parity/sts_r_compare.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from stm_r_compare import _best_alignment_cosine, _read_r_beta

REPL_DIR = os.environ.get(
    "STS_REPL_DIR", os.path.expanduser("~/Downloads/mnsc.2022.00261")
)

# The regenerated corpus can drift by a word or two from the fitted vocabulary
# across stm/SnowballC versions; compare on the shared vocab and only bail if the
# overlap is too small to be the same corpus.
MIN_VOCAB_OVERLAP = 0.98


def available() -> tuple[bool, str]:
    """Whether Rscript, ``stm``, and the replication results are all present."""
    if shutil.which("Rscript") is None:
        return False, "Rscript not on PATH"
    rds = os.path.join(REPL_DIR, "Results", "Poliblogs_results.RDS")
    if not os.path.isfile(rds):
        return False, f"replication results not found (set STS_REPL_DIR); looked in {REPL_DIR}"
    data = os.path.join(REPL_DIR, "Data", "poliblogs2008.csv")
    if not os.path.isfile(data):
        return False, f"poliblog source data not found (set STS_REPL_DIR); looked in {REPL_DIR}"
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False, "could not run Rscript"
    if not out.stdout.strip().endswith("TRUE"):
        return False, "the 'stm' R package is not installed"
    return True, ""


# R driver: regenerate the poliblog corpus exactly as the paper does, read the
# published STS fit, and export (a) the corpus as token lists, (b) the metadata,
# (c) the reference mean-sentiment topic-word matrix, and (d) an R stm fit on the
# same corpus as the calibration baseline.
_R_DRIVER = r"""
suppressMessages(library(stm))
repl <- Sys.getenv("STS_REPL_DIR")
data <- read.csv(file.path(repl, "Data", "poliblogs2008.csv"))
temp <- textProcessor(documents = data$documents, metadata = data,
                      onlycharacter = TRUE, verbose = FALSE)
out  <- prepDocuments(temp$documents, temp$vocab, temp$meta,
                      lower.thresh = 30, verbose = FALSE)
fit  <- readRDS(file.path(repl, "Results", "Poliblogs_results.RDS"))

shared <- intersect(fit$vocab, out$vocab)
overlap <- length(shared) / length(fit$vocab)
cat(sprintf("vocab regen=%d fit=%d shared=%d overlap=%.4f\n",
            length(out$vocab), length(fit$vocab), length(shared), overlap))
if (overlap < __MIN_OVERLAP__) { cat("vocab-mismatch\n"); quit(status = 0) }

# Corpus as token lists: vocab word repeated by its count, one doc per line.
toks <- sapply(out$documents, function(m) paste(rep(out$vocab[m[1, ]], m[2, ]), collapse = " "))
writeLines(toks, file.path(dir, "docs.txt"))

# The rating covariate (Conservative/Liberal) drives both prevalence and the
# sentiment seed, exactly as the replication script does:
#   X <- X_seed <- model.matrix(~out$meta$rating)[,-1]
write.csv(data.frame(rating = as.integer(out$meta$rating == "Liberal")),
          file.path(dir, "meta.csv"), row.names = FALSE)

# Reference topic-word at the mean sentiment (print.topWords' "Avg alpha" view):
# softmax(m_v + kappa^(t)_{.,k} + kappa^(s)_{.,k} * mean_d alpha^(s)_{d,k}).
K <- ncol(fit$kappa$kappa_t)
as_mean <- apply(fit$alpha[, 1:K + K - 1, drop = FALSE], 2, mean)
beta <- t(sapply(1:K, function(k) {
  e <- exp(fit$mv + fit$kappa$kappa_t[, k] + fit$kappa$kappa_s[, k] * as_mean[k]); e / sum(e)
}))
colnames(beta) <- fit$vocab
write.csv(beta, file.path(dir, "r_sts_beta.csv"), row.names = FALSE)

# R STM on the same corpus -- the calibration baseline. STS extends STM, and
# topica's STM is already validated against R's, so "topica-STS vs R-STS" should
# be about as close as "topica-STM vs R-STM".
fstm <- stm(out$documents, out$vocab, K, prevalence = ~rating,
            data = data.frame(rating = out$meta$rating),
            init.type = "Spectral", verbose = FALSE)
bstm <- exp(fstm$beta$logbeta[[1]]); colnames(bstm) <- out$vocab
write.csv(bstm, file.path(dir, "r_stm_beta.csv"), row.names = FALSE)

writeLines(fit$vocab, file.path(dir, "r_vocab.txt"))
cat("ok\n")
"""


def _to_r_vocab(raw, vocab, r_vocab):
    """A topic-word matrix on `vocab` reindexed onto R's vocabulary order."""
    raw = np.asarray(raw)
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


def _beta_at_mean_sentiment(sts):
    """topica STS topic-word read at each topic's mean sentiment (the reference's
    representative-topic view), (K, V) on the model's own vocabulary.

    The topic signal lives in the sentiment direction, so the recognizable topic
    is beta_k evaluated at mean_d alpha^(s)_{d,k}, not at neutral sentiment."""
    mean_s = np.asarray(sts.sentiment).mean(axis=0)  # per-topic mean alpha^(s)
    return np.vstack([
        np.asarray(sts.topic_word_at(float(mean_s[k])))[k]
        for k in range(len(mean_s))
    ]), mean_s


def run(verbose: bool = True) -> dict:
    """Fit topica STS on the published poliblog corpus and compare it to the
    reference STS -- calibrated against the topica-vs-R STM baseline and topica's
    own STM, so the irreducible cross-implementation gap is accounted for."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)

    from topica import STM, STS

    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "STS_REPL_DIR": REPL_DIR}
        driver = _R_DRIVER.replace("__MIN_OVERLAP__", repr(MIN_VOCAB_OVERLAP))
        script = f'dir <- "{d}"\n' + driver
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=1200, env=env
        )
        if "vocab-mismatch" in proc.stdout:
            raise RuntimeError(
                "the regenerated poliblog vocabulary overlaps the fitted RDS by "
                f"less than {MIN_VOCAB_OVERLAP:.0%} (likely a different stm/SnowballC "
                "version); cannot compare on a shared vocab"
            )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")
        if verbose:
            for line in proc.stdout.strip().splitlines():
                if line.startswith("vocab "):
                    print("  " + line)

        docs = [line.split() for line in open(os.path.join(d, "docs.txt")) if line.strip()]
        with open(os.path.join(d, "meta.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        rating = np.array([float(r["rating"]) for r in rows])
        r_vocab = open(os.path.join(d, "r_vocab.txt")).read().split()
        r_sts = _read_r_beta(os.path.join(d, "r_sts_beta.csv"), r_vocab)
        r_stm = _read_r_beta(os.path.join(d, "r_stm_beta.csv"), r_vocab)

    # topica STS and STM on the same corpus, same K, same rating design.
    K = r_sts.shape[0]
    X = rating.reshape(-1, 1)
    sts = STS(num_topics=K, init="spectral")
    # reference="paper" replicates Chen & Mankad's STS.R init faithfully: prevalence
    # latents at 0 and Σ = diag(20) (their stm(max.em.its=0) returns eta=0,
    # invsigma=diag(1/20)), the "lasso" (plain group-mean) κ estimator, and the
    # centered-seed sentiment init. This is the profile the reference RDS was produced
    # with, so it is what we validate against.
    sts.fit(docs, sentiment_seed=rating.tolist(), prevalence=X,
            prevalence_names=["rating"],
            iters=50, kappa_estimation="lasso", reference="paper")
    stm = STM(num_topics=K, init="spectral")
    stm.fit(docs, X, prevalence_names=["rating"], iters=80)

    beta_mean, _ = _beta_at_mean_sentiment(sts)
    t_sts = _to_r_vocab(beta_mean, list(sts.vocabulary), r_vocab)
    t_stm = _to_r_vocab(stm.topic_word, list(stm.vocabulary), r_vocab)

    sts_vs_ref = _best_alignment_cosine(r_sts, t_sts)
    stm_vs_ref = _best_alignment_cosine(r_stm, t_stm)  # the cross-impl baseline
    sts_vs_stm = _best_alignment_cosine(t_sts, t_stm)  # internal consistency
    ref_ceiling = _best_alignment_cosine(r_sts, r_stm)  # R-STS vs R-STM
    jaccard = _aligned_top_word_jaccard(r_sts, t_sts, topn=10)

    rng = np.random.default_rng(0)
    chance = float(np.mean([
        _best_alignment_cosine(r_sts[:, rng.permutation(r_sts.shape[1])], t_sts)
        for _ in range(5)
    ]))

    metrics = {
        "sts_vs_ref": sts_vs_ref,
        "stm_vs_ref": stm_vs_ref,
        "sts_vs_stm": sts_vs_stm,
        "ref_ceiling": ref_ceiling,
        "top_word_jaccard": jaccard,
        "chance_cosine": chance,
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
        "num_topics": K,
    }
    if verbose:
        print(f"docs={len(docs)}  vocab={len(r_vocab)}  K={K}")
        print(f"topica-STS vs R-STS:  {sts_vs_ref:.3f}  (chance {chance:.3f}, top-10 Jaccard {jaccard:.3f})")
        print(f"topica-STM vs R-STM:  {stm_vs_ref:.3f}  <- cross-implementation baseline")
        print(f"R-STS   vs R-STM:     {ref_ceiling:.3f}  <- same-ecosystem ceiling")
        print(f"topica-STS vs topica-STM: {sts_vs_stm:.3f}  <- STS extends STM")
    return metrics


def _aligned_top_word_jaccard(a, b, topn=10):
    """Mean top-`topn` Jaccard overlap of the best one-to-one topic alignment
    between two K×V topic-word matrices (aligned by cosine)."""
    from topica.validation import _hungarian

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cost = 1.0 - an @ bn.T
    scores = []
    for i, j in _hungarian(cost):
        sa = set(np.argsort(a[i])[::-1][:topn])
        sb = set(np.argsort(b[j])[::-1][:topn])
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


def main() -> int:
    ok, why = available()
    if not ok:
        print(f"skipping STS R parity: {why}")
        return 0
    m = run()
    # Validation is topic-level and benchmark-relative: (1) STS clearly beats
    # chance, (2) STS aligns to R about as well as topica's already validated STM
    # does, and (3) STS agrees with topica's own STM -- confirming it extends it.
    assert m["sts_vs_ref"] > m["chance_cosine"] + 0.3, (
        f"STS-vs-reference {m['sts_vs_ref']:.3f} not clearly above chance {m['chance_cosine']:.3f}"
    )
    assert m["sts_vs_ref"] > m["stm_vs_ref"] - 0.1, (
        f"STS aligns to R ({m['sts_vs_ref']:.3f}) much worse than STM does "
        f"({m['stm_vs_ref']:.3f})"
    )
    assert m["sts_vs_stm"] > 0.85, (
        f"topica STS does not agree with topica STM ({m['sts_vs_stm']:.3f}); STS should extend STM"
    )
    print("OK: topica STS recovers the published poliblog topics, as faithfully as STM does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
