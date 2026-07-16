"""Cross-implementation validation for the keyATM *covariate* and *dynamic*
models: topica.models.KeyATM vs the R `keyATM` package.

The base model is covered in `keyatm_r_compare.py`. This extends the same
statistical-equivalence approach to the two model variants, on the poliblog
corpus (which carries a binary `rating` covariate and a numeric `day`):

  - covariate model: the document-topic prior is a Dirichlet-multinomial
    regression on `rating`. We hand both engines the same docs, keywords, and
    covariate, then check (a) the keyword-topic phi alignment and (b) that the
    *sign* of each topic's rating effect agrees. The effect is read off the
    observable -- the Conservative-minus-Liberal difference in mean theta -- so
    it does not depend on either engine's internal lambda parameterization.

  - dynamic model: a Chib (1998) change-point HMM lets prevalence shift over
    time. We bin `day` into a shared time index and check the keyword-topic phi
    alignment and that each topic's prevalence *trend* over time agrees in sign.

Both are benchmarked against R's own seed-to-seed spread, since keyATM is
random-init Gibbs. Skips (exit 0) if R or the `keyATM` package is unavailable.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from collections import Counter

import numpy as np

from keyatm_r_compare import (
    KEYWORD_SETS,
    MIN_DOC_FREQ,
    _best_alignment_cosine,
    _read_r_phi,
    r_keyatm_available,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLIBLOG = os.path.join(ROOT, "examples", "poliblog.csv")

ITERS = int(os.environ.get("KEYATM_ITERS", "1000"))
NUM_REGULAR = int(os.environ.get("KEYATM_REGULAR", "4"))
NUM_TIME_BINS = int(os.environ.get("KEYATM_TIME_BINS", "8"))


def load_with_covariates():
    """Poliblog docs filtered to the shared vocab, with the aligned binary rating
    (Conservative=1) and a time index (day binned into NUM_TIME_BINS). Drops empty
    documents, keeping covariates in step. Returns (docs, keywords, rating, time)."""
    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    toks_all = [r["text"].split() for r in rows]
    df = Counter()
    for d in toks_all:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= MIN_DOC_FREQ}

    docs, rating, day = [], [], []
    for r, t in zip(rows, toks_all):
        kept = [w for w in t if w in vocab]
        if kept:
            docs.append(kept)
            rating.append(1.0 if r["rating"] == "Conservative" else 0.0)
            day.append(float(r["day"]))
    keywords = {n: [w for w in ws if w in vocab] for n, ws in KEYWORD_SETS.items()}
    keywords = {n: ws for n, ws in keywords.items() if ws}

    # keyATM's dynamic model needs documents ordered by time with a contiguous,
    # non-decreasing index, so sort the whole corpus by day and bin into states.
    order = np.argsort(day, kind="stable")
    docs = [docs[i] for i in order]
    rating = np.array(rating)[order]
    day = np.array(day)[order]
    ranks = np.arange(len(day))
    time = (ranks * NUM_TIME_BINS // len(day) + 1).astype(int)
    return docs, keywords, rating, time


def _group_sign(theta, group):
    """Per topic, sign of mean(theta[group==1]) - mean(theta[group==0])."""
    hi = theta[group == 1].mean(axis=0)
    lo = theta[group == 0].mean(axis=0)
    return np.sign(hi - lo)


def _trend_sign(theta, time):
    """Per topic, sign of the slope of mean theta against the time index."""
    ts = np.unique(time)
    means = np.array([theta[time == t].mean(axis=0) for t in ts])  # T x K
    tc = ts - ts.mean()
    slope = (tc[:, None] * (means - means.mean(0))).sum(0) / (tc**2).sum()
    return np.sign(slope)


_R_DRIVER = r"""
suppressMessages(library(keyATM)); suppressMessages(library(quanteda))
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
docs  <- keyATM_read(texts = dfmat)
keywords <- lapply(jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE), unlist)
rating <- scan(file.path(dir, "rating.txt"), quiet = TRUE)
tindex <- as.integer(scan(file.path(dir, "time.txt"), quiet = TRUE))

fit_cov <- function(seed) {
  keyATM(docs = docs, model = "covariates", no_keyword_topics = NREG, keywords = keywords,
         model_settings = list(covariates_data = data.frame(rating = rating),
                               covariates_formula = ~ rating),
         options = list(seed = seed, iterations = ITERS, verbose = FALSE))
}
fit_dyn <- function(seed) {
  keyATM(docs = docs, model = "dynamic", no_keyword_topics = NREG, keywords = keywords,
         model_settings = list(time_index = tindex, num_states = NSTATES),
         options = list(seed = seed, iterations = ITERS, verbose = FALSE))
}
c1 <- fit_cov(1); c2 <- fit_cov(2); d1 <- fit_dyn(1)
write.csv(c1$phi, file.path(dir, "cov_phi1.csv"))
write.csv(c2$phi, file.path(dir, "cov_phi2.csv"))
write.csv(c1$theta, file.path(dir, "cov_theta1.csv"), row.names = FALSE)
write.csv(c2$theta, file.path(dir, "cov_theta2.csv"), row.names = FALSE)
write.csv(d1$phi, file.path(dir, "dyn_phi1.csv"))
write.csv(d1$theta, file.path(dir, "dyn_theta1.csv"), row.names = FALSE)
cat("ok\n")
"""


def _read_theta(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)
        return np.array([[float(x) for x in row] for row in rdr])


def run(verbose: bool = True) -> dict:
    if not r_keyatm_available():
        raise RuntimeError("Rscript with the 'keyATM' package is not available")
    from topica.models import KeyATM

    docs, keywords, rating, time = load_with_covariates()
    num_keyword = len(keywords)
    num_topics = num_keyword + NUM_REGULAR

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(keywords, f)
        np.savetxt(os.path.join(d, "rating.txt"), rating)
        np.savetxt(os.path.join(d, "time.txt"), time, fmt="%d")

        script = (
            f'dir <- "{d}"\nNREG <- {NUM_REGULAR}\nITERS <- {ITERS}\nNSTATES <- 5\n'
            + _R_DRIVER
        )
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        with open(os.path.join(d, "cov_phi1.csv"), newline="") as f:
            r_vocab = [h.strip('"') for h in next(csv.reader(f))[1:]]
        cov_phi1 = _read_r_phi(os.path.join(d, "cov_phi1.csv"), r_vocab)
        cov_phi2 = _read_r_phi(os.path.join(d, "cov_phi2.csv"), r_vocab)
        cov_th1 = _read_theta(os.path.join(d, "cov_theta1.csv"))
        cov_th2 = _read_theta(os.path.join(d, "cov_theta2.csv"))
        dyn_phi1 = _read_r_phi(os.path.join(d, "dyn_phi1.csv"), r_vocab)
        dyn_th1 = _read_theta(os.path.join(d, "dyn_theta1.csv"))

    def align_to_r(model):
        idx = {w: i for i, w in enumerate(model.vocabulary)}
        raw = np.asarray(model.topic_word)
        out = np.zeros((raw.shape[0], len(r_vocab)))
        for j, w in enumerate(r_vocab):
            if w in idx:
                out[:, j] = raw[:, idx[w]]
        return out

    # --- covariate model ---
    cm = KeyATM(keywords, num_topics=num_topics, seed=1)
    cm.fit(docs, iters=ITERS, covariates=rating.reshape(-1, 1), feature_names=["rating"])
    cov_phi_tt = align_to_r(cm)
    cov_th_tt = np.asarray(cm.doc_topic)

    kw = slice(0, num_keyword)
    cov_kw_cos = _best_alignment_cosine(cov_phi1[kw], cov_phi_tt[kw])
    cov_kw_self = _best_alignment_cosine(cov_phi1[kw], cov_phi2[kw])
    sgn_r = _group_sign(cov_th1, rating)[kw]
    sgn_tt = _group_sign(cov_th_tt, rating)[kw]
    sgn_r2 = _group_sign(cov_th2, rating)[kw]
    cov_sign_agree = float((sgn_r == sgn_tt).mean())
    cov_sign_self = float((sgn_r == sgn_r2).mean())

    # --- dynamic model ---
    dm = KeyATM(keywords, num_topics=num_topics, seed=1)
    dm.fit(docs, iters=ITERS, timestamps=time.tolist(), num_states=5)
    dyn_phi_tt = align_to_r(dm)
    dyn_th_tt = np.asarray(dm.doc_topic)
    dyn_kw_cos = _best_alignment_cosine(dyn_phi1[kw], dyn_phi_tt[kw])
    trend_r = _trend_sign(dyn_th1, time)[kw]
    trend_tt = _trend_sign(dyn_th_tt, time)[kw]
    dyn_trend_agree = float((trend_r == trend_tt).mean())

    result = {
        "n_docs": len(docs), "vocab_size": len(r_vocab),
        "num_topics": num_topics, "num_keyword": num_keyword,
        "cov_keyword_cosine": cov_kw_cos, "cov_keyword_r_self": cov_kw_self,
        "cov_rating_sign_agree": cov_sign_agree, "cov_rating_sign_r_self": cov_sign_self,
        "dyn_keyword_cosine": dyn_kw_cos, "dyn_trend_sign_agree": dyn_trend_agree,
    }
    if verbose:
        print(f"corpus: {len(docs)} docs, {len(r_vocab)} vocab, {num_topics} topics "
              f"({num_keyword} keyword + {NUM_REGULAR} regular)")
        print("covariate model:")
        print(f"  keyword phi   — R vs topica: {cov_kw_cos:.3f}  (R vs R: {cov_kw_self:.3f})")
        print(f"  rating effect — sign agreement R vs topica: {cov_sign_agree:.2f}  (R vs R: {cov_sign_self:.2f})")
        print("dynamic model:")
        print(f"  keyword phi   — R vs topica: {dyn_kw_cos:.3f}")
        print(f"  time trend    — sign agreement R vs topica: {dyn_trend_agree:.2f}")
    return result


if __name__ == "__main__":
    import sys

    if not r_keyatm_available():
        print("SKIP: Rscript with the 'keyATM' package is not available")
        sys.exit(0)
    run(verbose=True)
