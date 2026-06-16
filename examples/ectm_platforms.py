"""ECTM on U.S. party platforms, 1948-2024: how Democrats and Republicans word
the same policy topics, and how those partisan vocabularies evolve across 20
presidential elections.

Each national platform (Democratic and Republican, from the American Presidency
Project) is split into paragraph-documents, so every (party, election-year) cell
is well populated. Party is the content **group**; the election year is the
**period** -- 20 periods over 76 years, the kind of long span where ECTM's
content evolution is visible.

The corpus ships gzipped in examples/platforms_data/ (rebuild it with
`python examples/prep_platforms.py`). Run:

    python examples/ectm_platforms.py

Headline result: inside a single stable "environment" topic, `climate` and
`clean` enter the Democratic vocabulary after 2000 while Republicans never adopt
them, `conservation` fades, and the partisan gap widens -- evolution that
keyATM's fixed-vocabulary dynamic model cannot represent.
"""
import gzip
import json
import os
import re
from collections import Counter

import numpy as np

import topica
from topica.ectm import (content_contrast, content_divergence, content_trajectory,
                         prevalence_by_group)

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
    # (the "how often" half), and using the same design in predicted_prevalence
    # below gives method-of-composition standard errors.
    party_col, pn = topica.one_hot(party)                 # indicator(Republican)
    yr_basis, sn = topica.spline(np.asarray(year, float), df=4)
    inter, _ = topica.interaction(party_col, yr_basis, name="party_year")
    X = np.column_stack([party_col, yr_basis, inter])
    names = list(pn) + list(sn) + [f"party_year_{i}" for i in range(inter.shape[1])]

    model = topica.ECTM(num_topics=18, seed=1)
    model.fit(docs, times=year, content=party, prevalence=X, prevalence_names=names,
              iters=150, period_smooth=6.0, interaction_shrink=1.2)
    print(f"fitted: {model} | converged={model.converged}\n")

    P = model.periods

    def avg_div(k):
        return np.mean([d for _, d in content_divergence(model, k, "D", "R")])

    print("=== Most partisan-worded topics (avg D-R total variation) ===")
    for k in sorted(range(model.num_topics), key=avg_div, reverse=True)[:4]:
        print(f"\nTopic #{k} (avgTV={avg_div(k):.3f}): "
              f"{' '.join(w for w, _ in model.top_words(9, topic=k))}")
        for yr in (P[0], P[len(P) // 2], P[-1]):
            c = content_contrast(model, k, "D", "R", yr, n=5)
            print(f"   {yr}  D: {', '.join(w for w, _ in c['toward_D'])}"
                  f"   R: {', '.join(w for w, _ in c['toward_R'])}")

    # Find the environment topic and trace its evolution.
    vi = {w: i for i, w in enumerate(model.vocabulary)}
    if "climate" in vi:
        env = max(range(model.num_topics),
                  key=lambda k: sum(np.asarray(model.content_word_dist("D", t))[k, vi["climate"]]
                                    for t in range(model.num_periods)))
        print(f"\n=== Environment topic #{env}: vocabulary evolution keyATM holds fixed ===")
        print("D-R vocabulary distance (total variation) by election:")
        for yr, d in content_divergence(model, env, "D", "R"):
            print(f"  {yr}  {d:.2f}  {'#' * int(d * 40)}")
        print("\nWord probability in the topic (x1000), every 4th election:")
        cols = P[::4]
        print("           " + " ".join(f"{c:>5}" for c in cols))
        for w in ("climate", "conservation", "clean", "pollution"):
            if w not in vi:
                continue
            for pt in ("D", "R"):
                traj = [1000 * model.content_word_dist(pt, t)[env, vi[w]] for t in range(len(P))][::4]
                print(f"  {w + ' (' + pt + ')':<11}" + " ".join(f"{v:5.1f}" for v in traj))

        # --- Does the topic model add value over a naive cross-tab? ---
        # Naive baseline: the raw share of `climate` among tokens of documents
        # whose dominant topic is this one, by party-year. ECTM should agree on
        # the trend but partial-pool the sparse early cells (where the raw share is
        # 0 or undefined) and isolate the word *within* the topic.
        assign = model.doc_topic.argmax(1)
        cols = P[::4]

        def raw_climate(pt, ylabel):
            tok = [t for d, pp, yy, a in zip(docs, party, year, assign)
                   if a == env and pp == pt and yy == int(ylabel) for t in d]
            return 1000 * tok.count("climate") / len(tok) if tok else float("nan")

        ectm = [1000 * model.content_word_dist("D", t)[env, vi["climate"]] for t in range(len(P))]
        print("\nECTM (RW-pooled) vs naive within-topic cross-tab, 'climate' for Democrats (x1000):")
        print("           " + " ".join(f"{c:>6}" for c in cols))
        print("  ECTM:    " + " ".join(f"{ectm[i * 4]:6.1f}" for i in range(len(cols))))
        print("  raw:     " + " ".join(f"{raw_climate('D', c):6.1f}" for c in cols))
        print("  (trend agrees; ECTM adds period pooling on sparse cells, topic separation, "
              "and the\n   uncertainty band -- it is the regularized, inferential version of the cross-tab.)")

        # Per-word content contrast with analytic standard errors (last election).
        from topica.ectm import content_contrast_se
        dl = [len(d) for d in docs]
        print(f"\nContent contrast with analytic SEs, {P[-1]} (word: D-R x1000 +/- SE, z):")
        for w, c, se in content_contrast_se(model, env, "D", "R", P[-1], party, year, dl, n=5):
            z = c / se if se > 0 else 0.0
            print(f"  {w:12} {1000 * c:+7.1f} +/- {1000 * se:4.1f}  z={z:+.1f}")
        print("  (instant, conservative -- ignores period pooling; for the pooled band use "
              "content_trajectory_ci)")

        # --- The other half: how OFTEN each party discusses the topic ---
        print(f"\n=== Environment topic #{env}: ATTENTION (prevalence), the other half ===")
        att = prevalence_by_group(model, party, year, topic=env) * 100  # (num_groups, num_periods) %
        gi = {g: i for i, g in enumerate(model.groups)}
        print("Share of platform devoted to the topic (%), every 4th election:")
        print("        " + " ".join(f"{c:>6}" for c in P[::4]))
        for g in model.groups:
            print(f"  {g}:   " + " ".join(f"{v:6.1f}" for v in att[gi[g]][::4]))
        try:
            import pandas as pd
            from topica.stm import predicted_prevalence
            data = pd.DataFrame({"party": party, "year": year})
            res = predicted_prevalence(model, formula="~ C(party) * spline(year, df=4)",
                                       data=data, contrast={"party": ["D", "R"]},
                                       topics=[env], nsims=25, n_sim=2000, seed=0)
            e = res[0]
            gap, lo, hi = (100 * float(np.ravel(x)[0]) for x in (e.estimate, e.ci_low, e.ci_high))
            print(f"\nD-R attention gap (predicted, method-of-composition 95% CI): "
                  f"{gap:+.2f} points  [{lo:+.2f}, {hi:+.2f}]")
            print(f"Takeaway: the ATTENTION gap is about a percentage point ({abs(gap):.1f}), while the")
            print("WORDING gap (climate, above) swings ~20-30 per-mille and grows -- an order of")
            print("magnitude larger. The partisan divergence on the environment is overwhelmingly in")
            print("HOW it is discussed, not HOW MUCH -- which a prevalence-only model cannot see.")
        except ImportError:
            print("\n(install topica[formula] to get the attention gap with standard errors)")


if __name__ == "__main__":
    main()
