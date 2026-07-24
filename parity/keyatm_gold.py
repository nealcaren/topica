"""Committed-gold parity for topica KeyATM vs the R `keyATM` package (issue #271, Wave 1).

R's `keyATM` and topica are independent implementations of the keyword-assisted
topic model (Eshima, Imai & Sasaki 2024). They share no code and no RNG, and
both initialize at random, so agreement is *statistical*: fit both on the SAME
tokenized corpus + keyword sets and ask whether they recover the same topics,
benchmarked against R's own seed-to-seed reproducibility.

keyATM's signature is keyword anchoring, so the sharp, fair comparison is the
*keyword* topics: their content is pinned by the supplied keywords, so they
should agree across implementations at least as well as R agrees with itself.
The committed gold freezes R's keyword-topic-word distributions (`phi`), the
vocab, the keyword sets, the config, and R's own seed-to-seed keyword-phi noise
floor, for all three reference variants:

  1. base       — keyword-anchored, no covariates (the core model).
  2. covariate  — document-topic prior is a Dirichlet-multinomial regression on
                  the binary `rating` covariate. This is the model fixed in #270
                  (theta was collapsing onto one topic before R's covariate
                  standardization + lambda bounds were added); a committed gold
                  vs R locks that fix against the reference.
  3. dynamic    — Chib's change-point HMM over a shared, binned time index.
                  The gold locks both keyword-topic phi and the observable
                  per-topic prevalence-trend signs against R's own spread.

Two phases (mirrors parity/stm_gold.py exactly):

  * ``--regenerate`` (needs Rscript + the ``keyATM`` package): fits R `keyATM`
    twice per variant, computes R's keyword-phi seed-to-seed noise floor, and
    writes the committed gold (``parity/keyatm_gold.npz`` + ``.json``).
  * default (no R): loads the committed gold, fits topica KeyATM on the same
    corpus + keywords, aligns to R's phi, and checks the bar.

The pass-bar logic is taken verbatim from the live scripts
``keyatm_r_compare.py`` / ``keyatm_models_r_compare.py`` (NOT reinvented):
topica's keyword-topic phi must align to R's at least as well as R's own
seed-to-seed runs do, minus a small multimodality margin.

Run directly::

    python parity/keyatm_gold.py               # offline compare against committed gold
    python parity/keyatm_gold.py --regenerate  # run R once, write the gold
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import harness

# The live scripts hold the corpora, keyword sets, R drivers, and the alignment
# metric; reuse them so the gold reproduces their coverage rather than forking it.
import keyatm_r_compare as base_live
import keyatm_models_r_compare as cov_live

NAME = "keyatm"

# Iterations for the gold. Fewer than the live default keeps regeneration quick;
# the keyword topics stabilize early (the live test runs 400). Both engines use
# the same count, so the comparison stays fair.
ITERS = 600

# Multimodality margin, taken from the live scripts' pass-bar (keyword_cosine >=
# keyword_r_self_cosine - 0.15).
MARGIN = 0.15

NUM_REGULAR_BASE = base_live.NUM_REGULAR  # 6
NUM_REGULAR_COV = cov_live.NUM_REGULAR    # 4


# --------------------------------------------------------------------------- #
# R version string
# --------------------------------------------------------------------------- #
def _r_version() -> str:
    try:
        out = subprocess.run(
            ["Rscript", "-e",
             'cat(as.character(getRversion()), as.character(packageVersion("keyATM")))'],
            capture_output=True, text=True, timeout=60,
        )
        rv, sv = out.stdout.strip().split()
        return f"R {rv} / keyATM {sv}"
    except Exception:
        return "R / keyATM (version unknown)"


# --------------------------------------------------------------------------- #
# vocab alignment helper
# --------------------------------------------------------------------------- #
def _to_r_vocab(raw: np.ndarray, vocab: list[str], r_vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


# --------------------------------------------------------------------------- #
# R drivers (kept here, model-specific, per the harness convention)
# --------------------------------------------------------------------------- #
_R_BASE_DRIVER = r"""
suppressMessages(library(keyATM)); suppressMessages(library(quanteda))
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("need jsonlite")
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
docs  <- keyATM_read(texts = dfmat)
keywords <- lapply(jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE), unlist)
phi_of <- function(seed) {
  out <- keyATM(docs = docs, model = "base", no_keyword_topics = NREG, keywords = keywords,
                options = list(seed = seed, iterations = ITERS, verbose = FALSE))
  out$phi
}
write.csv(phi_of(1), file.path(dir, "r_phi1.csv"))
write.csv(phi_of(2), file.path(dir, "r_phi2.csv"))
cat("ok\n")
"""

_R_COV_DRIVER = r"""
suppressMessages(library(keyATM)); suppressMessages(library(quanteda))
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("need jsonlite")
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
docs  <- keyATM_read(texts = dfmat)
keywords <- lapply(jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE), unlist)
rating <- scan(file.path(dir, "rating.txt"), quiet = TRUE)
fit_cov <- function(seed) {
  keyATM(docs = docs, model = "covariates", no_keyword_topics = NREG, keywords = keywords,
         model_settings = list(covariates_data = data.frame(rating = rating),
                               covariates_formula = ~ rating),
         options = list(seed = seed, iterations = ITERS, verbose = FALSE))
}
c1 <- fit_cov(1); c2 <- fit_cov(2)
write.csv(c1$phi, file.path(dir, "cov_phi1.csv"))
write.csv(c2$phi, file.path(dir, "cov_phi2.csv"))
write.csv(c1$theta, file.path(dir, "cov_theta1.csv"), row.names = FALSE)
write.csv(c2$theta, file.path(dir, "cov_theta2.csv"), row.names = FALSE)
cat("ok\n")
"""

_R_DYNAMIC_DRIVER = r"""
suppressMessages(library(keyATM)); suppressMessages(library(quanteda))
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("need jsonlite")
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
docs  <- keyATM_read(texts = dfmat)
keywords <- lapply(jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE), unlist)
tindex <- as.integer(scan(file.path(dir, "time.txt"), quiet = TRUE))
fit_dyn <- function(seed) {
  keyATM(docs = docs, model = "dynamic", no_keyword_topics = NREG, keywords = keywords,
         model_settings = list(time_index = tindex, num_states = NSTATES),
         options = list(seed = seed, iterations = ITERS, verbose = FALSE))
}
d1 <- fit_dyn(1); d2 <- fit_dyn(2)
write.csv(d1$phi, file.path(dir, "dyn_phi1.csv"))
write.csv(d2$phi, file.path(dir, "dyn_phi2.csv"))
write.csv(d1$theta, file.path(dir, "dyn_theta1.csv"), row.names = FALSE)
write.csv(d2$theta, file.path(dir, "dyn_theta2.csv"), row.names = FALSE)
cat("ok\n")
"""


def _read_phi_csv(path, r_vocab):
    return base_live._read_r_phi(path, r_vocab)


def _read_theta_csv(path):
    return cov_live._read_theta(path)


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not base_live.r_keyatm_available():
        print("Rscript with the 'keyATM' package not available; cannot regenerate.")
        sys.exit(1)

    arrays: dict = {}
    meta_models: dict = {}

    # ----- base model ----- #
    docs, keywords = base_live.load_and_prep()
    num_keyword = len(keywords)
    K = num_keyword + NUM_REGULAR_BASE

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(keywords, f)
        script = (f'dir <- "{d}"\nNREG <- {NUM_REGULAR_BASE}\nITERS <- {ITERS}\n'
                  + _R_BASE_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R base driver failed:\n{proc.stdout}\n{proc.stderr}")
        import csv as _csv
        with open(os.path.join(d, "r_phi1.csv"), newline="") as f:
            r_vocab = [h.strip('"') for h in next(_csv.reader(f))[1:]]
        base_phi1 = _read_phi_csv(os.path.join(d, "r_phi1.csv"), r_vocab)
        base_phi2 = _read_phi_csv(os.path.join(d, "r_phi2.csv"), r_vocab)

    kw = slice(0, num_keyword)
    base_r_self = base_live._best_alignment_cosine(base_phi1[kw], base_phi2[kw])
    base_tt = _fit_topica_base(docs, keywords, K, r_vocab)
    base_tt_cos = base_live._best_alignment_cosine(base_phi1[kw], base_tt[kw])

    arrays["base_phi1"] = base_phi1
    arrays["base_vocab"] = np.array(r_vocab, dtype=object)
    meta_models["base"] = {
        "num_topics": K,
        "num_keyword": num_keyword,
        "num_regular": NUM_REGULAR_BASE,
        "keywords": keywords,
        "num_docs": len(docs),
        "vocab_size": len(r_vocab),
        "seeds": {"phi1": 1, "phi2": 2},
        "keyword_r_self_cosine": base_r_self,
        "topica_keyword_cosine": base_tt_cos,
    }

    # ----- covariate model ----- #
    cdocs, ckeywords, rating, _time = cov_live.load_with_covariates()
    cnum_keyword = len(ckeywords)
    cK = cnum_keyword + NUM_REGULAR_COV

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in cdocs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(ckeywords, f)
        np.savetxt(os.path.join(d, "rating.txt"), rating)
        script = (f'dir <- "{d}"\nNREG <- {NUM_REGULAR_COV}\nITERS <- {ITERS}\n'
                  + _R_COV_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R covariate driver failed:\n{proc.stdout}\n{proc.stderr}")
        import csv as _csv
        with open(os.path.join(d, "cov_phi1.csv"), newline="") as f:
            cr_vocab = [h.strip('"') for h in next(_csv.reader(f))[1:]]
        cov_phi1 = _read_phi_csv(os.path.join(d, "cov_phi1.csv"), cr_vocab)
        cov_phi2 = _read_phi_csv(os.path.join(d, "cov_phi2.csv"), cr_vocab)
        cov_th1 = _read_theta_csv(os.path.join(d, "cov_theta1.csv"))
        cov_th2 = _read_theta_csv(os.path.join(d, "cov_theta2.csv"))

    ckw = slice(0, cnum_keyword)
    cov_r_self = cov_live._best_alignment_cosine(cov_phi1[ckw], cov_phi2[ckw])
    # R's own rating-effect sign reproducibility on the keyword topics.
    sgn_r1 = cov_live._group_sign(cov_th1, rating)[ckw]
    sgn_r2 = cov_live._group_sign(cov_th2, rating)[ckw]
    cov_sign_r_self = float((sgn_r1 == sgn_r2).mean())

    cov_tt, cov_th_tt = _fit_topica_cov(cdocs, ckeywords, cK, rating, cr_vocab)
    cov_tt_cos = cov_live._best_alignment_cosine(cov_phi1[ckw], cov_tt[ckw])
    sgn_tt = cov_live._group_sign(cov_th_tt, rating)[ckw]
    cov_sign_tt = float((sgn_r1 == sgn_tt).mean())

    arrays["cov_phi1"] = cov_phi1
    arrays["cov_vocab"] = np.array(cr_vocab, dtype=object)
    arrays["cov_rating_sign_r"] = sgn_r1.astype(float)
    meta_models["covariate"] = {
        "num_topics": cK,
        "num_keyword": cnum_keyword,
        "num_regular": NUM_REGULAR_COV,
        "keywords": ckeywords,
        "covariate_formula": "~ rating",
        "covariate": "rating (Conservative=1, Liberal=0)",
        "num_docs": len(cdocs),
        "vocab_size": len(cr_vocab),
        "seeds": {"phi1": 1, "phi2": 2},
        "keyword_r_self_cosine": cov_r_self,
        "topica_keyword_cosine": cov_tt_cos,
        "rating_sign_r_self": cov_sign_r_self,
        "topica_rating_sign_agree": cov_sign_tt,
        "note": "model fixed in #270 (covariate standardization + lambda bounds)",
    }

    # ----- dynamic model ----- #
    # Reuse the covariate corpus because it supplies the shared, sorted time
    # index used by the live dynamic parity check.  Two R seeds give both a
    # keyword-phi noise floor and a trend-sign reproducibility floor.
    ddocs, dkeywords, _rating, time = cov_live.load_with_covariates()
    dnum_keyword = len(dkeywords)
    dK = dnum_keyword + NUM_REGULAR_COV
    num_states = 5

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in ddocs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(dkeywords, f)
        np.savetxt(os.path.join(d, "time.txt"), time, fmt="%d")
        script = (f'dir <- "{d}"\nNREG <- {NUM_REGULAR_COV}\nITERS <- {ITERS}\n'
                  f"NSTATES <- {num_states}\n" + _R_DYNAMIC_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True,
                              text=True, timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R dynamic driver failed:\n{proc.stdout}\n{proc.stderr}")
        import csv as _csv
        with open(os.path.join(d, "dyn_phi1.csv"), newline="") as f:
            dr_vocab = [h.strip('\"') for h in next(_csv.reader(f))[1:]]
        dyn_phi1 = _read_phi_csv(os.path.join(d, "dyn_phi1.csv"), dr_vocab)
        dyn_phi2 = _read_phi_csv(os.path.join(d, "dyn_phi2.csv"), dr_vocab)
        dyn_th1 = _read_theta_csv(os.path.join(d, "dyn_theta1.csv"))
        dyn_th2 = _read_theta_csv(os.path.join(d, "dyn_theta2.csv"))

    dkw = slice(0, dnum_keyword)
    dyn_r_self = cov_live._best_alignment_cosine(dyn_phi1[dkw], dyn_phi2[dkw])
    trend_r1 = cov_live._trend_sign(dyn_th1, time)[dkw]
    trend_r2 = cov_live._trend_sign(dyn_th2, time)[dkw]
    dyn_trend_r_self = float((trend_r1 == trend_r2).mean())
    dyn_tt, dyn_th_tt = _fit_topica_dynamic(
        ddocs, dkeywords, dK, time, num_states, dr_vocab,
    )
    dyn_tt_cos = cov_live._best_alignment_cosine(dyn_phi1[dkw], dyn_tt[dkw])
    dyn_trend_tt = cov_live._trend_sign(dyn_th_tt, time)[dkw]
    dyn_trend_tt_agree = float((trend_r1 == dyn_trend_tt).mean())

    arrays["dyn_phi1"] = dyn_phi1
    arrays["dyn_vocab"] = np.array(dr_vocab, dtype=object)
    arrays["dyn_trend_sign_r"] = trend_r1.astype(float)
    meta_models["dynamic"] = {
        "num_topics": dK,
        "num_keyword": dnum_keyword,
        "num_regular": NUM_REGULAR_COV,
        "num_states": num_states,
        "num_time_bins": cov_live.NUM_TIME_BINS,
        "keywords": dkeywords,
        "time_index": "poliblog day rank, sorted and binned",
        "num_docs": len(ddocs),
        "vocab_size": len(dr_vocab),
        "seeds": {"phi1": 1, "phi2": 2},
        "keyword_r_self_cosine": dyn_r_self,
        "topica_keyword_cosine": dyn_tt_cos,
        "trend_sign_r_self": dyn_trend_r_self,
        "topica_trend_sign_agree": dyn_trend_tt_agree,
    }

    harness.save_gold(
        NAME,
        arrays=arrays,
        meta={
            "reference": _r_version(),
            "model": "KeyATM (base + covariate + dynamic)",
            "corpus": "poliblog (examples/poliblog.csv), already stemmed",
            "topica_iters": ITERS,
            "margin": MARGIN,
            "date": datetime.date.today().isoformat(),
            "pass_bar": "topica keyword-phi cosine >= keyword_r_self_cosine - margin",
            "models": meta_models,
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  base      — R self {base_r_self:.4f}  topica {base_tt_cos:.4f}")
    print(f"  covariate — R self {cov_r_self:.4f}  topica {cov_tt_cos:.4f}  "
          f"sign agree {cov_sign_tt:.2f} (R self {cov_sign_r_self:.2f})")
    print(f"  dynamic   — R self {dyn_r_self:.4f}  topica {dyn_tt_cos:.4f}  "
          f"trend agree {dyn_trend_tt_agree:.2f} (R self {dyn_trend_r_self:.2f})")


# --------------------------------------------------------------------------- #
# topica fits (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _fit_topica_base(docs, keywords, K, r_vocab):
    from topica import KeyATM

    model = KeyATM(keywords, num_topics=K, seed=1)
    model.fit(docs, iters=ITERS)
    return _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)


def _fit_topica_cov(docs, keywords, K, rating, r_vocab):
    from topica import KeyATM

    model = KeyATM(keywords, num_topics=K, seed=1)
    model.fit(docs, iters=ITERS, covariates=rating.reshape(-1, 1),
              feature_names=["rating"])
    phi = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)
    return phi, np.asarray(model.doc_topic)


def _fit_topica_dynamic(docs, keywords, K, time, num_states, r_vocab):
    from topica import KeyATM

    model = KeyATM(keywords, num_topics=K, seed=1)
    model.fit(docs, iters=ITERS, timestamps=time.tolist(), num_states=num_states)
    phi = _to_r_vocab(np.asarray(model.topic_word), list(model.vocabulary), r_vocab)
    return phi, np.asarray(model.doc_topic)


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    models = meta.get("models", {})
    margin = float(meta.get("margin", MARGIN))
    result: dict = {}

    # ----- base ----- #
    bm = models["base"]
    base_phi1 = arrays["base_phi1"]
    base_vocab = list(arrays["base_vocab"])
    num_keyword = int(bm["num_keyword"])
    kw = slice(0, num_keyword)

    docs, keywords = base_live.load_and_prep()
    base_tt = _fit_topica_base(docs, keywords, int(bm["num_topics"]), base_vocab)
    base_cos = base_live._best_alignment_cosine(base_phi1[kw], base_tt[kw])
    base_r_self = float(bm["keyword_r_self_cosine"])
    base_bar = base_r_self - margin
    result["base"] = {
        "keyword_cosine": base_cos,
        "keyword_r_self_cosine": base_r_self,
        "bar": base_bar,
        "margin_over_bar": base_cos - base_bar,
        "passes": bool(base_cos >= base_bar),
    }

    # ----- covariate ----- #
    cm = models["covariate"]
    cov_phi1 = arrays["cov_phi1"]
    cov_vocab = list(arrays["cov_vocab"])
    cnum_keyword = int(cm["num_keyword"])
    ckw = slice(0, cnum_keyword)
    sgn_r = arrays["cov_rating_sign_r"]

    cdocs, ckeywords, rating, _time = cov_live.load_with_covariates()
    cov_tt, cov_th_tt = _fit_topica_cov(cdocs, ckeywords, int(cm["num_topics"]),
                                        rating, cov_vocab)
    cov_cos = cov_live._best_alignment_cosine(cov_phi1[ckw], cov_tt[ckw])
    cov_r_self = float(cm["keyword_r_self_cosine"])
    cov_bar = cov_r_self - margin
    sgn_tt = cov_live._group_sign(cov_th_tt, rating)[ckw]
    cov_sign_agree = float((sgn_r == sgn_tt).mean())
    cov_sign_r_self = float(cm.get("rating_sign_r_self", 1.0))
    result["covariate"] = {
        "keyword_cosine": cov_cos,
        "keyword_r_self_cosine": cov_r_self,
        "bar": cov_bar,
        "margin_over_bar": cov_cos - cov_bar,
        "rating_sign_agree": cov_sign_agree,
        "rating_sign_r_self": cov_sign_r_self,
        # The #270 fix means theta no longer collapses; the rating effect must
        # agree in sign with R at least as often as R agrees with itself.
        "passes": bool(cov_cos >= cov_bar
                       and cov_sign_agree >= cov_sign_r_self - 1e-9),
    }

    # ----- dynamic ----- #
    dm = models["dynamic"]
    dyn_phi1 = arrays["dyn_phi1"]
    dyn_vocab = list(arrays["dyn_vocab"])
    dnum_keyword = int(dm["num_keyword"])
    dkw = slice(0, dnum_keyword)
    trend_r = arrays["dyn_trend_sign_r"]
    ddocs, dkeywords, _rating, time = cov_live.load_with_covariates()
    dyn_tt, dyn_th_tt = _fit_topica_dynamic(
        ddocs, dkeywords, int(dm["num_topics"]), time,
        int(dm["num_states"]), dyn_vocab,
    )
    dyn_cos = cov_live._best_alignment_cosine(dyn_phi1[dkw], dyn_tt[dkw])
    dyn_r_self = float(dm["keyword_r_self_cosine"])
    dyn_bar = dyn_r_self - margin
    dyn_trend = cov_live._trend_sign(dyn_th_tt, time)[dkw]
    dyn_trend_agree = float((trend_r == dyn_trend).mean())
    dyn_trend_r_self = float(dm["trend_sign_r_self"])
    result["dynamic"] = {
        "keyword_cosine": dyn_cos,
        "keyword_r_self_cosine": dyn_r_self,
        "bar": dyn_bar,
        "margin_over_bar": dyn_cos - dyn_bar,
        "trend_sign_agree": dyn_trend_agree,
        "trend_sign_r_self": dyn_trend_r_self,
        "passes": bool(dyn_cos >= dyn_bar
                       and dyn_trend_agree >= dyn_trend_r_self - 1e-9),
    }

    result["passes"] = (result["base"]["passes"]
                        and result["covariate"]["passes"]
                        and result["dynamic"]["passes"])

    if verbose:
        print(f"gold: {meta.get('reference')}")
        b = result["base"]
        print("base model:")
        print(f"  keyword phi  — topica {b['keyword_cosine']:.4f}  "
              f"(R self {b['keyword_r_self_cosine']:.4f}, bar {b['bar']:.4f})")
        print(f"  verdict: {'PASS' if b['passes'] else 'FAIL'} "
              f"(margin {b['margin_over_bar']:+.4f})")
        c = result["covariate"]
        print("covariate model (#270 fix):")
        print(f"  keyword phi  — topica {c['keyword_cosine']:.4f}  "
              f"(R self {c['keyword_r_self_cosine']:.4f}, bar {c['bar']:.4f})")
        print(f"  rating sign  — agree {c['rating_sign_agree']:.2f}  "
              f"(R self {c['rating_sign_r_self']:.2f})")
        print(f"  verdict: {'PASS' if c['passes'] else 'FAIL'} "
              f"(margin {c['margin_over_bar']:+.4f})")
        d = result["dynamic"]
        print("dynamic model:")
        print(f"  keyword phi  — topica {d['keyword_cosine']:.4f}  "
              f"(R self {d['keyword_r_self_cosine']:.4f}, bar {d['bar']:.4f})")
        print(f"  trend sign   — agree {d['trend_sign_agree']:.2f}  "
              f"(R self {d['trend_sign_r_self']:.2f})")
        print(f"  verdict: {'PASS' if d['passes'] else 'FAIL'} "
              f"(margin {d['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
