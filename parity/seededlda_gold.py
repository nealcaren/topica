"""Committed R-`seededlda` gold for topica SeededLDA (#456).

This REPLACES the earlier planted self-consistency gold. The koheiw/`seededlda`
R package is an installed, runnable reference, so we validate topica against it
directly. We freeze, from a fixed poliblog subsample:

  1. R seededlda's seeded topic-word phi at two seeds -> a keyword-phi cosine
     noise floor (how well R agrees with itself across seeds), and
  2. R `seededlda:::tfm`'s exact seed-pseudocount matrix (`count * weight * 100`).

The offline test then refits topica with the reference-faithful default
(`seed_prior="frequency"`, alpha=0.5, beta=0.1) on the same corpus and asserts:

  * topica's `seed_prior_matrix` reproduces R's `tfm()` EXACTLY (the seed-mass
    construction the #456 review flagged), and
  * topica's seeded-topic phi clears R's own two-seed cosine floor (minus a
    small multimodality margin) -- the same "at least as consistent as R is with
    itself" bar the keyATM / STM gold use.

Runs in CI WITHOUT Rscript: the reference fit is frozen in the committed
``parity/seededlda_gold.npz`` + ``.json``.

Two phases::

    python parity/seededlda_gold.py --regenerate   # needs Rscript + seededlda + quanteda
    python parity/seededlda_gold.py                # offline compare against the gold
"""

from __future__ import annotations

import csv
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "seededlda"

# Keyword sets (stemmed poliblog vocabulary), one seeded topic each; plus one
# residual (unseeded) topic, matching R's residual=1 default label "other".
SEEDS: dict[str, list[str]] = {
    "econ": ["tax", "economi", "econom", "market", "spend", "budget"],
    "elect": ["obama", "mccain", "vote", "voter", "campaign", "elect"],
    "social": ["abort", "gay", "marriag", "religi", "church", "famili"],
    "war": ["iraq", "iraqi", "war", "troop", "militari", "surg"],
}
N_SEEDED = len(SEEDS)
RESIDUAL = 1
N_DOCS = 400
CORPUS_SEED = 271
WEIGHT = 0.01
ALPHA = 0.5
BETA = 0.1
ITERS = 200
MARGIN = 0.15


def _dict_r() -> str:
    body = ",\n".join(
        f'  {k}=c({", ".join(chr(34) + w + chr(34) for w in ws)})' for k, ws in SEEDS.items()
    )
    return "dictionary(list(\n" + body + "\n))"


def _driver() -> str:
    return f"""
suppressMessages({{library(seededlda); library(quanteda)}})
lines <- readLines(file.path(dir, "docs.txt"))
toks <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
dict <- {_dict_r()}
fit_one <- function(s) {{
  set.seed(s)
  textmodel_seededlda(dfmat, dict, residual = {RESIDUAL}, weight = {WEIGHT},
                      max_iter = {ITERS}, alpha = {ALPHA}, beta = {BETA},
                      verbose = FALSE)
}}
m1 <- fit_one(1); m2 <- fit_one(2)
# phi rows are in dictionary order (seeded topics first, then the residual).
write.csv(m1$phi, file.path(dir, "phi1.csv"), row.names = FALSE)
write.csv(m2$phi, file.path(dir, "phi2.csv"), row.names = FALSE)
tf <- seededlda:::tfm(dfmat, dict, weight = {WEIGHT}, residual = {RESIDUAL})
write.csv(as.matrix(tf), file.path(dir, "tfm.csv"))
cat("ok\\n")
"""


def _parse_tfm(text: str, vocab: list[str]) -> np.ndarray:
    """R `tfm` CSV (rows = topic labels, cols = features) -> (N_SEEDED, |vocab|)
    aligned to `vocab`, seeded topics only (dictionary order)."""
    rows = list(csv.reader(text.splitlines()))
    cols = [c.strip('"') for c in rows[0][1:]]
    cidx = {w: i for i, w in enumerate(cols)}
    by_label = {r[0].strip('"'): np.array([float(x) for x in r[1:]]) for r in rows[1:]}
    out = np.zeros((N_SEEDED, len(vocab)))
    for k, name in enumerate(SEEDS):
        rrow = by_label[name]
        for j, w in enumerate(vocab):
            if w in cidx:
                out[k, j] = rrow[cidx[w]]
    return out


def _fit_topica(docs, vocab):
    """Fit topica SeededLDA (reference-faithful frequency profile); return the
    seeded topic-word phi (N_SEEDED x |vocab|, aligned to `vocab`) and the seed
    pseudocount matrix (same shape)."""
    import topica

    m = topica.SeededLDA(
        SEEDS, residual=RESIDUAL, seed_prior="frequency",
        alpha=ALPHA, beta=BETA, weight=WEIGHT, seed=1,
    )
    m.fit(docs, iters=ITERS)
    tv = {w: i for i, w in enumerate(m.vocabulary)}
    tn = {n: i for i, n in enumerate(m.topic_names)}

    def _align(mat_kv):
        out = np.zeros((N_SEEDED, len(vocab)))
        for k, name in enumerate(SEEDS):
            row = mat_kv[tn[name]]
            for j, w in enumerate(vocab):
                if w in tv:
                    out[k, j] = row[tv[w]]
        return out

    phi = _align(np.asarray(m.topic_word))
    seed_mat = _align(np.asarray(m.seed_prior_matrix))
    return phi, seed_mat


def _r_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["Rscript", "-e", 'cat(as.character(packageVersion("seededlda")))'],
            capture_output=True, text=True, timeout=60,
        )
        return f"R seededlda {out.stdout.strip()}"
    except Exception:
        return "R seededlda (version unknown)"


def regenerate() -> None:
    if not harness.r_available("seededlda") or not harness.r_available("quanteda"):
        raise SystemExit("regenerate needs Rscript with the seededlda + quanteda packages")

    docs, _rating, _day, vocab = harness.poliblog_corpus(n_docs=N_DOCS, seed=CORPUS_SEED)
    out = harness.run_rscript(
        _driver(),
        {"docs.txt": harness.docs_to_lines(docs)},
        ["phi1.csv", "phi2.csv", "tfm.csv"],
        timeout=1800,
    )
    phi1 = harness.read_r_beta_csv(out["phi1.csv"], vocab)
    phi2 = harness.read_r_beta_csv(out["phi2.csv"], vocab)
    r_tfm = _parse_tfm(out["tfm.csv"], vocab)

    kw = slice(0, N_SEEDED)
    r_self, _ = harness.align_cosine(phi1[kw], phi2[kw])

    topica_phi, topica_seed_mat = _fit_topica(docs, vocab)
    topica_cos, _ = harness.align_cosine(phi1[kw], topica_phi)
    tfm_max_abs = float(np.abs(topica_seed_mat - r_tfm).max())

    harness.save_gold(
        NAME,
        arrays={
            "phi1": phi1,
            "tfm": r_tfm,
            "vocab": np.array(vocab, dtype=object),
        },
        meta={
            "reference": _r_version(),
            "model": "SeededLDA (koheiw/seededlda)",
            "corpus": f"poliblog subsample ({N_DOCS} docs, seed {CORPUS_SEED})",
            "seeds": SEEDS,
            "num_seeded": N_SEEDED,
            "residual": RESIDUAL,
            "weight": WEIGHT,
            "alpha": ALPHA,
            "beta": BETA,
            "iters": ITERS,
            "vocab_size": len(vocab),
            "num_docs": len(docs),
            "seed_prior": "frequency",
            "margin": MARGIN,
            "keyword_r_self_cosine": r_self,
            "topica_keyword_cosine": topica_cos,
            "tfm_max_abs_diff": tfm_max_abs,
            "date": datetime.date.today().isoformat(),
            "pass_bar": "topica seed_prior_matrix == R tfm (exact) AND "
            "topica keyword phi cosine >= keyword_r_self_cosine - margin",
        },
    )
    print(f"regenerated {NAME} gold:")
    print(f"  R self keyword cosine {r_self:.4f}  topica {topica_cos:.4f}  "
          f"(bar {r_self - MARGIN:.4f})")
    print(f"  tfm exact match: max |Δ| = {tfm_max_abs:.2e}")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    phi1 = arrays["phi1"]
    r_tfm = arrays["tfm"]
    vocab = [str(w) for w in arrays["vocab"]]
    r_self = float(meta["keyword_r_self_cosine"])
    margin = float(meta["margin"])
    bar = r_self - margin

    docs, _rating, _day, live_vocab = harness.poliblog_corpus(n_docs=N_DOCS, seed=CORPUS_SEED)
    assert live_vocab == vocab, "corpus vocabulary drifted from the committed gold"
    topica_phi, topica_seed_mat = _fit_topica(docs, vocab)

    kw = slice(0, N_SEEDED)
    cos, _ = harness.align_cosine(phi1[kw], topica_phi)
    tfm_max_abs = float(np.abs(topica_seed_mat - r_tfm).max())

    tfm_exact = tfm_max_abs < 1e-6
    cosine_ok = cos >= bar
    result = {
        "keyword_cosine": cos,
        "keyword_r_self_cosine": r_self,
        "bar": bar,
        "margin_over_bar": cos - bar,
        "tfm_max_abs_diff": tfm_max_abs,
        "tfm_exact": tfm_exact,
        "passes": bool(tfm_exact and cosine_ok),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print("seed-mass construction (topica seed_prior_matrix vs R tfm):")
        print(f"  max |Δ| = {tfm_max_abs:.2e}  -> {'EXACT' if tfm_exact else 'MISMATCH'}")
        print("seeded topic-word phi:")
        print(f"  topica {cos:.4f}  (R self {r_self:.4f}, bar {bar:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
