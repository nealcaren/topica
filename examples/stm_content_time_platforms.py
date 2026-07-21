"""STM content_time on U.S. party platforms, 1948-2024: how Democrats and
Republicans word the same policy topics, and how those partisan vocabularies
evolve across 20 presidential elections.

Each national platform (Democratic and Republican, from the American Presidency
Project) is split into paragraph-documents, so every (party, election-year) cell
is well populated. Party is the content **group**; the election year is the
ordered **content_time** covariate -- 20 periods over 76 years. STM crosses the
two into a base-by-period content design tied across adjacent periods by a
first-order random walk (`content_smooth`), with a sparse L1 prior on the content
deviations (`content_prior="l1"`). The reading layer -- `content_trajectory` and
`content_divergence` in `topica.content` -- reads the smoothed surface back out,
with a design-preserving (document / cluster) bootstrap for confidence bands.

The corpus ships gzipped in examples/platforms_data/ (rebuild it with
`python examples/prep_platforms.py`). Run:

    python examples/stm_content_time_platforms.py

Headline result: inside a single stable "environment" topic, `climate` and
`clean` enter the Democratic vocabulary after 2000 while Republicans never adopt
them, and the whole-distribution partisan divergence widens -- evolution that a
fixed-vocabulary dynamic model cannot represent.
"""
import gzip
import json
import os
import re
from collections import Counter

import numpy as np

import topica
from topica.content import content_divergence, content_trajectory

DATA = os.path.join(os.path.dirname(__file__), "platforms_data", "platforms.json.gz")

STOP = set((
    "the of to and a in that is for on it as with be by this are at from or an was not have has "
    "had will would shall can could may should do does did but if their our your his her its they "
    "them we us he she you i me my which who what when where how all any some no nor so than then "
    "there here more most other such only own same too very just also into out up down over under "
    "again about against between through during before after above below off one two many must make "
    "made being been am were these those this their them "
    # platform boilerplate
    "democrat democrats republican republicans party platform pledge believe support americans "
    "american america people nation national country government public new also need ensure continue "
    "work years year must commit committed"
).split())


def load():
    rows = json.load(gzip.open(DATA, "rt"))
    docs, party, year = [], [], []
    for r in rows:
        toks = [w for w in re.findall(r"[a-z]{3,}", r["text"].lower()) if w not in STOP]
        docs.append(toks)
        party.append(r["party"])
        year.append(int(r["year"]))
    # vocabulary trim: drop very rare and ubiquitous words
    dfc = Counter()
    for d in docs:
        dfc.update(set(d))
    n = len(docs)
    keep = {w for w, c in dfc.items() if c >= 10 and c <= 0.4 * n}
    docs = [[w for w in d if w in keep] for d in docs]
    idx = [i for i, d in enumerate(docs) if len(d) >= 8]
    return ([docs[i] for i in idx], [party[i] for i in idx], [year[i] for i in idx], len(keep))


def main():
    docs, party, year, vocab = load()
    print(f"{len(docs)} platform paragraphs | parties D/R | {len(set(year))} elections "
          f"{min(year)}-{max(year)} | vocab {vocab}")

    # Prevalence design: party, a smooth year trend, and their interaction. Passing
    # it as prevalence= lets each topic's *attention* depend on party and time
    # (the "how often" half), the complement to the content ("how worded") half.
    party_col, pn = topica.one_hot(party)                 # indicator(Republican)
    yr_basis, sn = topica.spline(np.asarray(year, float), df=4)
    inter, _ = topica.interaction(party_col, yr_basis, name="party_year")
    X = np.column_stack([party_col, yr_basis, inter])
    names = list(pn) + list(sn) + [f"party_year_{i}" for i in range(inter.shape[1])]

    model = topica.STM(num_topics=18, seed=1)
    model.fit(docs, prevalence=X, prevalence_names=names,
              content=party, content_time=year,
              content_prior="l1", content_prior_var=1.0, content_smooth=6.0,
              iters=150)
    print(f"fitted: {model} | converged={model.converged}\n")

    # The two content groups and the ordered periods, read straight off the surface.
    groups = ("D", "R")

    def avg_div(k):
        d = content_divergence(model, groups=groups, topic=k, measure="tv")
        return float(np.nanmean(d.divergence))

    print("=== Most partisan-worded topics (avg D-R total variation) ===")
    for k in sorted(range(model.num_topics), key=avg_div, reverse=True)[:4]:
        print(f"\nTopic #{k} (avgTV={avg_div(k):.3f}): "
              f"{' '.join(w for w, _ in model.top_words(9, topic=k))}")

    # Find the environment topic and trace its evolution.
    vi = {w: i for i, w in enumerate(model.vocabulary)}
    if "climate" in vi:
        # dominant topic for "climate": the topic whose global row weights it most
        tw = np.asarray(model.topic_word)
        env = int(tw[:, vi["climate"]].argmax())
        print(f"\n=== Environment topic #{env}: vocabulary evolution a fixed model holds constant ===")

        div = content_divergence(model, groups=groups, topic=env, measure="tv")
        print("D-R vocabulary distance (total variation) by election:")
        for yr, d in zip(div.periods, div.divergence):
            bar = "" if not np.isfinite(d) else "#" * int(d * 40)
            print(f"  {yr}  {d:.2f}  {bar}")

        traj = content_trajectory(model, ["climate", "conservation", "clean", "pollution"],
                                  groups=groups, topic=env)
        cols = traj.periods[::4]
        ci = [traj.periods.index(c) for c in cols]
        print("\nD-R word-probability contrast (x1000), every 4th election "
              "(positive = Democratic):")
        print("           " + " ".join(f"{c:>5}" for c in cols))
        for wi, w in enumerate(traj.words):
            row = [1000 * traj.estimate[wi, j] for j in ci]
            print(f"  {w:<11}" + " ".join(f"{v:5.1f}" for v in row))

        # Design-preserving CI: resample whole platforms (party x year clusters),
        # refit, realign the topic by anchor words, percentile bands.
        print("\nWhole-distribution D-R divergence with a cluster bootstrap CI "
              "(resampling platforms):")
        cluster = [f"{p}{y}" for p, y in zip(party, year)]
        fit_kwargs = dict(num_topics=18, prevalence=X, prevalence_names=names,
                          content=party, content_time=year,
                          content_prior="l1", content_prior_var=1.0, content_smooth=6.0,
                          iters=80)
        anchor = [w for w, _ in model.top_words(8, topic=env)]
        dci = content_divergence(model, groups=groups, anchor_words=anchor, measure="hellinger",
                                 ci=True, corpus=docs, fit_kwargs=fit_kwargs,
                                 cluster=cluster, B=40, seed=0)
        print("           divergence   [95% CI]")
        for j, yr in enumerate(dci.periods[::4]):
            k = dci.periods.index(yr)
            lo = dci.ci_low[k] if dci.ci_low is not None else float("nan")
            hi = dci.ci_high[k] if dci.ci_high is not None else float("nan")
            print(f"  {yr}    {dci.divergence[k]:.3f}     [{lo:.3f}, {hi:.3f}]")
        print("\n  (Bands widen at the first/last election, where the random walk is least")
        print("   constrained -- the honest uncertainty a point trajectory hides.)")


if __name__ == "__main__":
    main()
