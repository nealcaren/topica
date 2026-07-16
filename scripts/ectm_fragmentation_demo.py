#!/usr/bin/env python
"""A synthetic case engineered to INDUCE fragmentation, to exercise split_topics.

The gentrification data mostly shows the suppress -> within-topic transition; the
"one discourse fragments into parallel group-topics" failure barely fires there.
This plants a case that provokes it, then shows the content_prior_var lever moving
the fit from SPLIT (low prior) to WITHIN-topic (high prior).

The generator plants a single *contested theme*: a shared core vocabulary both
groups use, plus group-specific accent words. Background topics are shared equally
by both groups, so only the contested theme can go group-skewed.

The demo contrasts the two levers that decide split vs within-topic:

  Design A -- group in the CONTENT model only. content_prior_var moves the fit
    along suppress -> within-topic (polarization rises); no fragmentation.
  Design B -- group ALSO in the PREVALENCE design. Now the model is rewarded for
    group-specific topics and the contested theme fragments into two
    near-duplicate, oppositely group-skewed topics -- and content_prior_var can no
    longer undo it. So the split lever is the DESIGN, not the prior.
"""

from __future__ import annotations

import numpy as np

import topica
from topica.models import ECTM
from topica.content import topic_polarization, split_topics

CORE = ["policy", "council", "meeting", "proposal", "vote", "budget"]
ACCENT_A = ["growth", "investment", "jobs", "development"]
ACCENT_B = ["displacement", "eviction", "tenants", "affordable"]
BG = {
    "b1": ["school", "teacher", "student", "classroom"],
    "b2": ["park", "trail", "trees", "recreation"],
    "b3": ["transit", "bus", "commute", "station"],
    "b4": ["police", "safety", "crime", "patrol"],
}
PERIODS = ["t0", "t1", "t2"]


def generate(n=600, seed=0):
    """Return (docs, groups, times). Contested theme is group-worded; background
    topics are shared equally, so only the theme can go group-skewed."""
    rng = np.random.default_rng(seed)
    bg_keys = list(BG)
    docs, groups, times = [], [], []
    for i in range(n):
        g = "A" if i % 2 == 0 else "B"
        doc = []
        accent = ACCENT_A if g == "A" else ACCENT_B
        for _ in range(6):
            doc += list(rng.choice(CORE, size=2))
            doc += list(rng.choice(accent, size=1))
        # Background: every doc draws from ALL background topics (group-neutral).
        for _ in range(4):
            topic = rng.choice(bg_keys)
            doc += list(rng.choice(BG[topic], size=3))
        docs.append(doc)
        groups.append(g)
        times.append(PERIODS[i % 3])
    return docs, groups, times


def fit(docs, groups, times, *, cpv, group_in_prevalence, K=6, seed=1):
    corpus = topica.Corpus.from_documents(docs)
    kwargs = {}
    if group_in_prevalence:
        X, feat = topica.design.one_hot(groups, drop_first=True)
        kwargs = dict(prevalence=X, prevalence_names=feat)
    m = ECTM(num_topics=K, seed=seed)
    m.fit(corpus, times=times, content=groups, iters=200, content_prior_var=cpv, **kwargs)
    return m


def report(m, groups, label):
    pol = topic_polarization(m)
    splits = split_topics(m, groups, cos_thresh=0.5, skew_thresh=0.6)
    print(f"  {label:<44} max_pol={pol.max():.3f}  split_pairs={len(splits)}")
    return len(splits)


def main():
    topica.enable_experimental()
    docs, groups, times = generate()
    print(f"generated {len(docs)} docs, groups={sorted(set(groups))}; "
          "contested theme = shared core + group accents; background shared\n")

    print("Design A -- group in CONTENT only (content_prior_var is the lever):")
    for cpv in (0.1, 1.0, 6.0):
        report(fit(docs, groups, times, cpv=cpv, group_in_prevalence=False), groups,
               f"content_prior_var={cpv}")

    print("\nDesign B -- group ALSO in PREVALENCE (fragmentation is provoked):")
    for cpv in (0.1, 1.0, 6.0):
        report(fit(docs, groups, times, cpv=cpv, group_in_prevalence=True), groups,
               f"content_prior_var={cpv}")

    print("\nReading: in Design A the prior slides suppress -> within-topic "
          "(max_pol rises) with no fragmentation. In Design B, putting the group "
          "in the prevalence design provokes group-skewed near-duplicate topics "
          "(split_pairs > 0) that the prior cannot undo -- the split lever is the "
          "DESIGN, not content_prior_var.")


if __name__ == "__main__":
    main()
