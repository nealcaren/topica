#!/usr/bin/env python
"""Does group-differentiated discourse land WITHIN a topic or SPLIT into topics?

ECTM can book a group's distinctive language two ways: as within-topic content
(kappa) variation -- one topic, worded differently by group -- or as separate,
group-skewed topics the prevalence model then sorts groups across. Which one it
picks is arbitrated by the kappa prior. This script measures the behaviour on the
planted gentrification data and sweeps the two levers that move it:

  * content_prior_var -- the kappa L2 prior variance. Higher = looser = the
    content model absorbs more group vocabulary within a topic.
  * num_topics (K)     -- more topics gives the model room to split.

Two instruments (the seed of the #324 content-covariate diagnostics):

  within_polarization(model)  per-topic Jensen-Shannon divergence across groups of
                              the period-averaged content beta_{k,g}. HIGH on a
                              topic = the group difference lives INSIDE it.
  split_pairs(model, groups)  near-duplicate topic pairs (high baseline-beta
                              cosine) that are pulled apart by group prevalence --
                              the fragmentation signature.

Run scripts/generate_gentrification_data.py first to build the dataset.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import topica
from topica.content import topic_polarization, split_topics
from topica.models import ECTM

CSV = "examples/gentrification_debates_temporal.csv"
# planted developer housing drift markers (see verify_synthetic_ectm.py)
MARKERS = ["upscale", "premium", "subsidized", "inclusionary"]


def housing_topic(model) -> int:
    """The topic carrying the planted developer housing-drift marker words."""
    vocab = model.vocabulary
    mi = [vocab.index(w) for w in MARKERS if w in vocab]
    G = model.groups.index("developer_official") if "developer_official" in model.groups else 0
    mass = np.zeros(model.num_topics)
    for t in range(model.num_periods):
        mass += np.asarray(model.content_word_dist(G, t))[:, mi].sum(axis=1)
    return int(np.argmax(mass))


# --------------------------------------------------------------------------- #
# Fit + sweep.
# --------------------------------------------------------------------------- #
def build_corpus():
    df = pd.read_csv(CSV)
    df["period"] = df["period"].astype(str)
    corpus = topica.prep.from_dataframe(
        df, text_col="text",
        metadata_cols=["neighborhood_status", "speaker_role", "period"],
        stopwords=topica.prep.stopwords("en"))
    X, feat = topica.design.one_hot(corpus.metadata["neighborhood_status"], drop_first=True)
    content = list(corpus.metadata["speaker_role"])
    times = list(corpus.metadata["period"])
    return corpus, X, feat, content, times


def fit(corpus, X, feat, content, times, *, K, cpv, seed):
    m = ECTM(num_topics=K, seed=seed)
    m.fit(corpus, times=times, content=content, prevalence=X, prevalence_names=feat,
          iters=200, content_prior_var=cpv)
    return m


def metrics(model, content):
    pol = topic_polarization(model)
    ht = housing_topic(model)
    sp = split_topics(model, content)
    return {
        "housing_pol": float(pol[ht]),      # within-topic group JSD on housing topic
        "mean_pol": float(pol.mean()),      # mean within-topic JSD
        "n_split": len(sp),                 # fragmentation pairs
    }


def main():
    if not os.path.exists(CSV):
        print(f"{CSV} not found; run scripts/generate_gentrification_data.py first.")
        return
    topica.enable_experimental()
    corpus, X, feat, content, times = build_corpus()
    print(f"corpus: {corpus.num_docs} docs, {corpus.num_words} words, "
          f"groups={sorted(set(content))}, periods={sorted(set(times))}\n")

    SEEDS = [1, 7, 42]
    CPVS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    KS = [6, 8, 10, 12, 14]

    def cell(K, cpv):
        rows = [metrics(fit(corpus, X, feat, content, times, K=K, cpv=cpv, seed=s), content)
                for s in SEEDS]
        return {k: np.mean([r[k] for r in rows]) for k in rows[0]}

    print("=" * 68)
    print(f"A)  content_prior_var sweep at K=10  (mean over seeds {SEEDS})")
    print("=" * 68)
    print(f"{'cpv':>6} | {'housing_pol':>12} | {'mean_pol':>10} | {'n_split':>8}")
    print("-" * 46)
    A = {}
    for cpv in CPVS:
        r = cell(10, cpv); A[cpv] = r
        print(f"{cpv:>6} | {r['housing_pol']:>12.3f} | {r['mean_pol']:>10.3f} | {r['n_split']:>8.2f}")

    print("\n" + "=" * 68)
    print(f"B)  K sweep at content_prior_var=1.0  (mean over seeds {SEEDS})")
    print("=" * 68)
    print(f"{'K':>6} | {'housing_pol':>12} | {'mean_pol':>10} | {'n_split':>8}")
    print("-" * 46)
    B = {}
    for K in KS:
        r = cell(K, 1.0); B[K] = r
        print(f"{K:>6} | {r['housing_pol']:>12.3f} | {r['mean_pol']:>10.3f} | {r['n_split']:>8.2f}")

    _plot(A, B)


def _plot(A, B):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available; skipping plot)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cpvs = sorted(A)
    ax1.plot(cpvs, [A[c]["housing_pol"] for c in cpvs], "o-", label="housing-topic polarization")
    ax1.plot(cpvs, [A[c]["mean_pol"] for c in cpvs], "s--", label="mean polarization", alpha=.6)
    ax1b = ax1.twinx()
    ax1b.plot(cpvs, [A[c]["n_split"] for c in cpvs], "^-", color="crimson", label="split pairs")
    ax1.set_xscale("log"); ax1.set_xlabel("content_prior_var (log)")
    ax1.set_ylabel("within-topic group JSD"); ax1b.set_ylabel("split pairs", color="crimson")
    ax1.set_title("A. kappa prior: within-topic vs split (K=10)")
    ax1.legend(loc="upper left", fontsize=8)
    ks = sorted(B)
    ax2.plot(ks, [B[k]["housing_pol"] for k in ks], "o-", label="housing-topic polarization")
    ax2b = ax2.twinx()
    ax2b.plot(ks, [B[k]["n_split"] for k in ks], "^-", color="crimson", label="split pairs")
    ax2.set_xlabel("num_topics K"); ax2.set_ylabel("within-topic group JSD")
    ax2b.set_ylabel("split pairs", color="crimson")
    ax2.set_title("B. K sweep (content_prior_var=1.0)")
    fig.tight_layout()
    out = "scripts/ectm_split_vs_within.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
