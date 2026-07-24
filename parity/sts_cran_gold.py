"""Committed R-`sts` gold for topica STS's reference-compatible profile (#454).

topica's STS ships a reference-fidelity profile (`reference="cran"`) that matches
the CRAN `sts` package's public default: `kappaEstimation="adjusted"`, the
reference `diag(1/20)` sentiment prior, anchor initialization, and half-step kappa
damping. That profile already has a *live* end-to-end check
(`parity/sts_r_package_compare.py`), but it shells out to Rscript and skips when
the toolchain is absent, so it does not gate CI.

This freezes that comparison into an offline gold, the way `seededlda_gold.py`
does for SeededLDA. From a fixed 300-doc poliblog corpus (stm's `poliblog5k`,
prepped by `prepDocuments`) we freeze:

  1. R `sts`'s topic-word distribution at mean sentiment (softmax(mv + kappa_t))
     at two `stmSeed`s -> an R self-consistency cosine floor, and
  2. the exact tokenized corpus + sentiment + rating R fit on, so topica refits
     the identical data offline.

The offline test refits topica STS with `reference="cran"` on the frozen corpus
and asserts its topic-word distribution at mean sentiment clears R's own
two-seed cosine floor (minus a margin) -- the same "at least as consistent as R
is with itself" bar the SeededLDA / keyATM / STM golds use. The kappa solver is
separately validated against glmnet in `sts_kappa_glmnet.py`; this pins the whole
adjusted-profile EM end to end without an R toolchain at test time.

Two phases::

    python parity/sts_cran_gold.py --regenerate   # needs Rscript + sts + stm
    python parity/sts_cran_gold.py                # offline compare against the gold
"""

from __future__ import annotations

import datetime
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "sts_cran"

N_DOCS = 300
K = 5
MAXITER = 8
SEEDS = (1, 2)  # two R fits -> the self-consistency (noise) floor
# R's adjusted-profile fit is near-identical across seeds (self cosine ~0.998), so
# the margin here is really absorbing the topica-vs-R cross-implementation gap
# (different initializations). 0.20 puts the bar at ~0.80 — the same pre-registered
# floor the live `sts_r_package_compare.py` uses and has validated — giving topica's
# ~0.89 a platform-robust headroom while the shuffle check keeps the gate honest.
MARGIN = 0.20
TOPICA_SEED = 1


def _driver() -> str:
    """R driver: prep the corpus once, fit R `sts` (adjusted profile) at two
    seeds, and emit both topic-word-at-mean-sentiment matrices plus the exact
    corpus/sentiment/rating so topica can refit the identical data."""
    return f"""
suppressMessages({{library(sts); library(stm)}})
data(poliblog5k)
idx  <- seq_len(min({N_DOCS}, length(poliblog5k.docs)))
meta <- poliblog5k.meta[idx, ]
meta$sent <- ifelse(meta$rating == "Liberal", 1, -1)
out  <- prepDocuments(poliblog5k.docs[idx], poliblog5k.voc, meta, verbose = FALSE)

fit_beta <- function(s) {{
  fit <- sts(prevalence_sentiment = ~rating, initializationVar = ~sent,
             corpus = out, K = {K}, maxIter = {MAXITER}, initialization = "anchor",
             kappaEstimation = "adjusted", verbose = FALSE, stmSeed = s)
  # Topic-word at mean sentiment (alpha^(s)=0): softmax(mv + kappa_t[,k]).  V x K.
  apply(fit$kappa$kappa_t, 2, function(kt) {{ e <- exp(fit$mv + kt); e / sum(e) }})
}}
bw1 <- fit_beta({SEEDS[0]}); bw2 <- fit_beta({SEEDS[1]})
write.table(t(bw1), file.path(dir, "beta1.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)                  # K x V
write.table(t(bw2), file.path(dir, "beta2.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
writeLines(out$vocab, file.path(dir, "vocab.txt"))                # V

# The exact tokenized corpus + sentiment/rating R fit on.
con <- file(file.path(dir, "docs.txt"), "w")
for (d in out$documents) {{
    toks <- rep(out$vocab[d[1, ]], d[2, ])
    writeLines(paste(toks, collapse = " "), con)
}}
close(con)
writeLines(as.character(out$meta$sent), file.path(dir, "sent.txt"))
writeLines(as.character(ifelse(out$meta$rating == "Liberal", 1L, 0L)),
           file.path(dir, "rating.txt"))
cat("ok\\n")
"""


def _fit_topica(docs, sent, rating, K, seed):
    """Fit topica STS with the reference-compatible ("cran") profile; return its
    topic-word-at-mean-sentiment matrix and vocabulary."""
    import topica

    model = topica.STS(num_topics=K, seed=seed)
    model.fit(
        docs,
        sentiment_seed=sent,
        prevalence=[[r] for r in rating],
        iters=MAXITER,
        reference="cran",
    )
    return np.asarray(model.topic_word), list(model.vocabulary)


def _shared_cosine(r_beta, r_vocab, t_beta, t_vocab):
    """Align R and topica topic-word matrices onto the shared vocabulary and
    return the mean cosine over the best one-to-one topic alignment."""
    t_set = set(t_vocab)
    shared = [w for w in r_vocab if w in t_set]
    r_idx = {w: i for i, w in enumerate(r_vocab)}
    t_idx = {w: i for i, w in enumerate(t_vocab)}
    r_aligned = r_beta[:, [r_idx[w] for w in shared]]
    t_aligned = t_beta[:, [t_idx[w] for w in shared]]
    cos, _ = harness.align_cosine(r_aligned, t_aligned)
    return cos, len(shared)


def _r_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["Rscript", "-e", 'cat(as.character(packageVersion("sts")))'],
            capture_output=True, text=True, timeout=60,
        )
        return f"R sts {out.stdout.strip()}"
    except Exception:
        return "R sts (version unknown)"


def regenerate() -> None:
    if not harness.r_available("sts") or not harness.r_available("stm"):
        raise SystemExit("regenerate needs Rscript with the sts + stm packages")

    out = harness.run_rscript(
        _driver(), {}, ["beta1.csv", "beta2.csv", "vocab.txt", "docs.txt", "sent.txt", "rating.txt"],
        timeout=3600,
    )
    vocab = [w.strip() for w in out["vocab.txt"].splitlines() if w.strip()]
    beta1 = np.loadtxt(io.StringIO(out["beta1.csv"]), delimiter=",")
    beta2 = np.loadtxt(io.StringIO(out["beta2.csv"]), delimiter=",")
    docs = harness.lines_to_docs(out["docs.txt"])
    sent = [float(x) for x in out["sent.txt"].split()]
    rating = [float(x) for x in out["rating.txt"].split()]

    # R's own two-seed floor, then topica against seed-1's fit.
    r_self, _ = _shared_cosine(beta1, vocab, beta2, vocab)
    t_beta, t_vocab = _fit_topica(docs, sent, rating, K, TOPICA_SEED)
    topica_cos, n_shared = _shared_cosine(beta1, vocab, t_beta, t_vocab)

    harness.save_gold(
        NAME,
        arrays={
            "beta1": beta1,
            "vocab": np.array(vocab, dtype=object),
            "docs": np.array([" ".join(d) for d in docs], dtype=object),
            "sent": np.array(sent, dtype=float),
            "rating": np.array(rating, dtype=float),
        },
        meta={
            "reference": _r_version(),
            "model": "STS reference profile (CRAN sts, kappaEstimation='adjusted')",
            "topica_profile": "reference='cran'",
            "corpus": f"stm poliblog5k first {N_DOCS} docs, prepDocuments",
            "num_topics": K,
            "maxiter": MAXITER,
            "r_seeds": list(SEEDS),
            "topica_seed": TOPICA_SEED,
            "num_docs": len(docs),
            "vocab_size": len(vocab),
            "vocab_shared": n_shared,
            "margin": MARGIN,
            "r_self_cosine": r_self,
            "topica_cosine": topica_cos,
            "date": datetime.date.today().isoformat(),
            "pass_bar": "topica reference='cran' topic-word cosine (at mean sentiment) "
            ">= r_self_cosine - margin",
        },
    )
    print(f"regenerated {NAME} gold:")
    print(f"  R self cosine {r_self:.4f}  topica {topica_cos:.4f}  "
          f"(bar {r_self - MARGIN:.4f}, shared vocab {n_shared})")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    beta1 = arrays["beta1"]
    vocab = [str(w) for w in arrays["vocab"]]
    docs = [str(s).split() for s in arrays["docs"]]
    sent = [float(x) for x in arrays["sent"]]
    rating = [float(x) for x in arrays["rating"]]
    r_self = float(meta["r_self_cosine"])
    margin = float(meta["margin"])
    bar = r_self - margin

    t_beta, t_vocab = _fit_topica(docs, sent, rating, int(meta["num_topics"]), TOPICA_SEED)
    cos, n_shared = _shared_cosine(beta1, vocab, t_beta, t_vocab)

    result = {
        "cosine": cos,
        "r_self_cosine": r_self,
        "bar": bar,
        "margin_over_bar": cos - bar,
        "vocab_shared": n_shared,
        "passes": bool(cos >= bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}  ({meta.get('model')})")
        print("STS reference='cran' topic-word at mean sentiment vs R sts:")
        print(f"  topica {cos:.4f}  (R self {r_self:.4f}, bar {bar:.4f}, "
              f"shared vocab {n_shared})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
