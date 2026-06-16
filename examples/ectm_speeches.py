"""ECTM on congressional speeches, 1948-2008: how Democrats and Republicans word
the same topics, and how the partisan gap evolves -- recovering the documented
rise of partisan language in Congress.

Speeches come from the Eugleo/us-congressional-speeches-subset corpus; speaker
party is joined from Voteview. Party is the content **group**; a 4-year window is
the **period** (16 periods). Build the corpus once (downloads ~2.9 GB of shards):

    python examples/prep_speeches.py
    python examples/ectm_speeches.py

Headline result: the mean Democrat-Republican vocabulary distance across topics
is flat through the 1950s-80s (~0.19) and rises in the mid-1990s (~0.22), the
same break Gentzkow, Shapiro & Taddy (2019) find in the partisanship of speech.
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

import topica
from topica.ectm import content_contrast, content_divergence, prevalence_by_group

DATA = os.path.join(os.path.dirname(__file__), "speech_data", "speeches.parquet")

STOP = set((
    "the of to and a in that is for on it as with be by this are at from or an was not have has had "
    "will would shall can could may should do does did but if their our your his her its they them we "
    "us he she you i me my which who what when where how all any some no nor so than then there here "
    "more most other such only own same too very just also into out up down over under again about "
    "against between through during before after above below off one two many must make made being been "
    "am were these those "
    # congressional procedural boilerplate
    "mr madam speaker chairman gentleman gentlewoman gentlemen yield time amendment bill act section "
    "president senator representative house senate floor consent unanimous clerk vote ask reserve "
    "balance point order chair colleague distinguished members member committee congress chamber"
).split())


def load():
    if not os.path.exists(DATA):
        raise SystemExit("Run `python examples/prep_speeches.py` first to build examples/speech_data/.")
    df = pd.read_parquet(DATA)
    docs = [[w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP] for t in df.text]
    dfc = Counter()
    for d in docs:
        dfc.update(set(d))
    n = len(docs)
    keep = {w for w, c in dfc.items() if c >= 25 and c <= 0.4 * n}
    docs = [[w for w in d if w in keep] for d in docs]
    idx = [i for i, d in enumerate(docs) if len(d) >= 10]
    return [docs[i] for i in idx], df.party.iloc[idx].tolist(), df.period.iloc[idx].tolist(), len(keep)


def main():
    docs, party, period, vocab = load()
    print(f"{len(docs)} speeches | parties D/R | {len(set(period))} periods "
          f"{min(period)}-{max(period)} | vocab {vocab}")

    # Prevalence design (party * smooth time) so attention can vary by party and
    # period -- the "how often" half, with SEs available via predicted_prevalence.
    party_col, pn = topica.one_hot(party)
    t_basis, sn = topica.spline(np.asarray(period, float), df=4)
    inter, _ = topica.interaction(party_col, t_basis, name="party_time")
    X = np.column_stack([party_col, t_basis, inter])
    names = list(pn) + list(sn) + [f"party_time_{i}" for i in range(inter.shape[1])]

    model = topica.ECTM(num_topics=16, seed=1)
    model.fit(docs, times=period, content=party, prevalence=X, prevalence_names=names,
              iters=120, period_smooth=6.0, interaction_shrink=1.3)
    print(f"fitted: {model} | converged={model.converged}\n")

    P = model.periods
    # Overall partisanship of language = mean D-R distance across topics, per period.
    print("Mean D-R vocabulary distance across all topics, by period (the polarization curve):")
    for p in P:
        v = np.mean([dict(content_divergence(model, k, "D", "R"))[p] for k in range(model.num_topics)])
        print(f"  {p}  {v:.3f}  {'#' * int(v * 60)}")

    def avg_div(k):
        return np.mean([d for _, d in content_divergence(model, k, "D", "R")])

    print("\n=== Most partisan-worded topics ===")
    for k in sorted(range(model.num_topics), key=avg_div, reverse=True)[:3]:
        print(f"\nTopic #{k} (avgTV={avg_div(k):.3f}): "
              f"{' '.join(w for w, _ in model.top_words(9, topic=k))}")
        for yr in (P[0], P[len(P) // 2], P[-1]):
            c = content_contrast(model, k, "D", "R", yr, n=5)
            print(f"   {yr}  D: {', '.join(w for w, _ in c['toward_D'])}"
                  f"   R: {', '.join(w for w, _ in c['toward_R'])}")

    # The other half: how often each party discusses the most divergent topic.
    top = max(range(model.num_topics), key=avg_div)
    att = prevalence_by_group(model, party, period, topic=top) * 100
    gi = {g: i for i, g in enumerate(model.groups)}
    print(f"\n=== Attention (prevalence) for topic #{top}, by party (%), every 4th period ===")
    print("        " + " ".join(f"{c:>6}" for c in P[::4]))
    for g in model.groups:
        print(f"  {g}:   " + " ".join(f"{v:6.1f}" for v in att[gi[g]][::4]))
    print("(for attention gaps with standard errors, pass this prevalence design to "
          "topica.stm.predicted_prevalence / estimate_effect)")


if __name__ == "__main__":
    main()
