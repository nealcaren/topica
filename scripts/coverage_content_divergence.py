"""Synthetic coverage grid for the STM content_time bootstrap CIs (#365).

The #340 objection to ECTM was that its cluster bootstrap could not preserve the
design, so its intervals targeted an "uncharacterized, coverage-conditioned
population." The content_time reader bootstrap resamples the *document* (the real
sampling unit) and refits, so its percentile CI should have close to nominal
frequentist coverage of the estimand.

This harness checks that. It draws `M` independent datasets from a fixed
document-generating process with a known per-period Dem/Rep wording gap, fits an
STM `content_time` model on each, and on each computes the point divergence plus a
document-bootstrap CI. The estimand's center is the across-dataset mean point
estimate; empirical coverage is the fraction of datasets whose CI brackets it, at
each period. Coverage should sit near the nominal level.

Run: python scripts/coverage_content_divergence.py   (a few minutes; it refits STM
M*(1+B) times). Not a gated test — it is a reproducibility harness.
"""

from __future__ import annotations

import numpy as np

import topica
from topica import content

PERIODS = [1990, 2000, 2010, 2020]
ANCHOR = ["climate", "border"]
LEVEL = 0.90
M = 40          # independent datasets
B = 40          # bootstrap replicates per dataset
PER_CELL = 40   # documents per (group, period) cell
ITERS = 120


def draw_dataset(rng):
    """One dataset: 2 groups x 4 periods, a topic whose Dem/Rep wording gap grows
    across periods, with genuine per-document multinomial sampling noise."""
    docs, grp, per = [], [], []
    vocab_topic = ["climate", "border", "econ", "jobs", "tax"]
    for pi, yr in enumerate(PERIODS):
        gap = 0.05 + 0.15 * pi / (len(PERIODS) - 1)  # 0.05 -> 0.20
        pd_ = np.array([0.30 + gap, 0.30 - gap, 0.13, 0.14, 0.13])
        pr = np.array([0.30 - gap, 0.30 + gap, 0.13, 0.14, 0.13])
        pd_ = pd_ / pd_.sum(); pr = pr / pr.sum()
        for _ in range(PER_CELL):
            nd = rng.integers(12, 20)
            docs.append([vocab_topic[i] for i in rng.choice(5, nd, p=pd_)])
            grp.append("Dem"); per.append(yr)
            docs.append([vocab_topic[i] for i in rng.choice(5, nd, p=pr)])
            grp.append("Rep"); per.append(yr)
    return docs, grp, per


def one_run(seed):
    rng = np.random.default_rng(seed)
    docs, grp, per = draw_dataset(rng)
    m = topica.STM(num_topics=2, seed=1)
    m.fit(docs, content=grp, content_time=per, content_prior="l1", iters=ITERS)
    point = content.content_divergence(m, groups=("Dem", "Rep"), anchor_words=ANCHOR)
    fk = dict(num_topics=2, seed=1, content=grp, content_time=per,
              content_prior="l1", iters=ITERS)
    boot = content.content_divergence(m, groups=("Dem", "Rep"), anchor_words=ANCHOR,
                                      ci=True, corpus=docs, fit_kwargs=fk, B=B,
                                      level=LEVEL, seed=seed + 1000)
    return point.divergence, boot.ci_low, boot.ci_high


def main():
    points, lows, highs = [], [], []
    for s in range(M):
        p, lo, hi = one_run(s)
        points.append(p); lows.append(lo); highs.append(hi)
        print(f"  dataset {s + 1}/{M} done", flush=True)
    points = np.array(points); lows = np.array(lows); highs = np.array(highs)
    center = np.nanmean(points, axis=0)  # estimand center per period
    covered = (lows <= center) & (center <= highs)  # (M, P)
    per_period = np.nanmean(covered, axis=0)
    overall = float(np.nanmean(covered))
    print("\nnominal level:", LEVEL)
    print("per-period coverage:", np.round(per_period, 3).tolist())
    print("overall coverage:   ", round(overall, 3))
    ok = abs(overall - LEVEL) <= 0.12  # Monte-Carlo tolerance at M=40
    print("PASS" if ok else "FAIL", "(overall within 0.12 of nominal)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
