"""Party and time in congressional press releases — a Structural Topic Model tour.

U.S. House members put out a steady stream of press releases, and what they
choose to talk about is shaped by two things this tutorial makes explicit: *who*
is speaking (party) and *when* (year). That is exactly the setting the Structural
Topic Model was built for — prevalence covariates that are a group *and* a smooth
trend in time. We ask two questions a political sociologist would put in a paper:

    - Which topics are distinctively Democratic vs Republican?
    - Which topics rose or fell across 2013-2024, and by how much (with CIs)?

The corpus is 3,120 House releases, 260 per year for twelve years (2013-2024),
balanced by party. topica bundles a clean copy as ``load_congress()``, but the
point of this tutorial is the *raw-to-result* pipeline, so we build the frame
from the raw JSONL that Derek Willis's ``congress-press`` project publishes
(https://github.com/dwillis/congress-press, MIT). If the network is unavailable
we fall back to the bundled sample; the analysis is identical either way.

We move through the workflow in research order:

    1. Acquire   : fetch raw JSONL, or load the bundled clean sample.
    2. Preprocess: strip HTML/boilerplate, tokenize, prune the vocabulary.
    3. Choose K  : search_k, a defensible coherence/exclusivity frontier.
    4. STM       : prevalence ~ party + spline(year).
    5. Label     : a FREX topic table.
    6. Effects   : party contrast + a 2013->2024 time trend, with honest CIs.

Everything is tuned to run in a couple of minutes; comments note where to scale
K / iterations / nsims up for a real analysis.

Run:  .venv-dev/bin/python examples/congress_tutorial.py
"""

import json
import urllib.request

import numpy as np
import pandas as pd

import topica

RAW_BASE = "https://raw.githubusercontent.com/dwillis/congress-press/main/data"
# A small raw slice: two months across the twelve years. Widen (more months /
# years, higher per-file cap) for a real study; congress-press has 2001-present.
RAW_YEARS = range(2013, 2025)
RAW_MONTHS = ("04", "10")
PER_FILE_CAP = 300


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_from_raw(seed=13):
    """Fetch raw press releases from congress-press and assemble the analysis
    frame. This is the step the bundled loader hides: real text arrives with
    datelines, markup, and mixed metadata, House-labelled by the source's member
    records. We keep House + two-party releases with a usable body."""
    import random

    rng = random.Random(seed)
    rows = []
    for year in RAW_YEARS:
        for month in RAW_MONTHS:
            url = f"{RAW_BASE}/{year}/{year}-{month}.jsonl"
            raw = urllib.request.urlopen(url, timeout=90).read().decode("utf-8", "ignore")
            keep = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip the occasional truncated line in the raw feed
                member = rec.get("member") or {}
                text = rec.get("text") or ""
                if member.get("chamber") != "House":
                    continue
                if member.get("party") not in ("Democrat", "Republican"):
                    continue
                if not (200 <= len(text) <= 20000):
                    continue
                keep.append({
                    "text": text,
                    "date": rec.get("date"),
                    "year": year,
                    "party": member.get("party"),
                    "state": member.get("state"),
                    "member": member.get("name"),
                })
            if len(keep) > PER_FILE_CAP:
                keep = rng.sample(keep, PER_FILE_CAP)
            rows.extend(keep)
            print(f"  {year}-{month}: kept {len(keep)}")
    return pd.DataFrame(rows)


def load_data():
    banner("1. Acquire the corpus (raw JSONL, or the bundled clean sample)")
    try:
        df = build_from_raw()
        print(f"Built {len(df)} releases from raw congress-press JSONL.")
    except Exception as exc:  # offline / rate-limited: use the bundled copy
        print(f"Raw fetch unavailable ({type(exc).__name__}); using load_congress().")
        df = topica.datasets.load_congress()
        print(f"Loaded the bundled sample: {len(df)} releases.")
    print("party:", df["party"].value_counts().to_dict())
    print("years:", sorted(df["year"].unique()))
    return df


def main():
    df = load_data()

    banner("2. Preprocess: strip web boilerplate, tokenize, prune")
    # Press releases carry HTML and URLs (href/http/aspx) that would otherwise
    # form a junk topic; strip_html=True removes them before tokenizing. We keep
    # metadata aligned so the covariates never drift from the surviving docs.
    corpus = topica.from_dataframe(
        df,
        text_col="text",
        strip_html=True,
        stopwords=topica.ENGLISH_STOPWORDS,
        min_doc_freq=10,       # a term must appear in >= 10 releases
        max_doc_fraction=0.4,  # drop boilerplate common to most releases
    )
    print(f"Corpus: {corpus.num_docs} docs, {corpus.num_words} vocabulary terms")

    banner("3. Choose K at the coherence/exclusivity frontier")
    # A small grid for the tutorial; widen ks and raise iters for real work.
    X_party, party_names = topica.one_hot(corpus.metadata["party"], reference="Democrat")
    scan = topica.search_k(
        corpus, [10, 15, 20, 25], model="stm",
        prevalence=X_party, iters=60, seed=13,
    )
    k = scan.best_k()  # frontier knee; warns if it lands on a grid edge
    print(scan.to_frame().to_string(index=False))
    print(f"Chosen K = {k}")

    banner("4. Fit the STM: prevalence ~ party + year")
    # Build the design by hand (no formulaic dependency): the party contrast plus a
    # centered linear year, whose coefficient reads directly as a per-year trend.
    # For curvature, swap in topica.spline(year, df=4) — it returns (basis, names)
    # you hstack the same way; we use a linear term so the effect stays a single
    # interpretable slope.
    year = corpus.metadata["year"].to_numpy(dtype=float)
    year_c = (year - year.mean()).reshape(-1, 1)  # center so the intercept is mid-period
    X = np.hstack([X_party, year_c])
    names = party_names + ["year"]
    model = topica.STM(num_topics=k, seed=13)
    model.fit(corpus, prevalence=X, prevalence_names=names, iters=200)
    print(f"Fitted STM(K={k}); variational bound {model.bound:,.0f}")

    banner("5. A FREX topic table")
    labels = topica.label_topics(model.topic_word, corpus.vocabulary, n=8, corpus=corpus)
    lab_words = [", ".join(w for w, _ in labels[t]["frex"][:4]) for t in range(len(labels))]
    for t, lab in enumerate(labels):
        # label_topics rows are dicts: pick a labeling, THEN read its (word, score) pairs
        print(f"  T{t:>2}: " + ", ".join(w for w, _ in lab["frex"]))

    # estimate_effect propagates topic-estimation uncertainty (method of
    # composition). We read every effect by NAME, never coef[0] (the intercept).
    effects = topica.estimate_effect(model, X=X, feature_names=names, nsims=50, seed=0)

    banner("6a. Party effect: which topics are Republican vs Democratic")
    # Reference level is Democrat, so a positive 'Republican' coefficient = the
    # topic is more prevalent in Republican releases.
    party = sorted(
        (eff.topic, *(_row(eff.effect_of("Republican"))) ) for eff in effects
    )
    print("Most Democratic (negative) and most Republican (positive) topics:")
    for topic, coef, lo, hi, p in party[:3] + party[-3:]:
        tag = "Rep" if coef > 0 else "Dem"
        print(f"  T{topic:>2} [{tag}] {coef:+.4f}  CI[{lo:+.4f}, {hi:+.4f}]  "
              f"p={p:.1e}  ({lab_words[topic]})")

    banner("6b. Time trend: prevalence change per year, and across 2013->2024")
    # The 'year' coefficient is the per-year change in the topic's prevalence
    # logit; scaling by the 11-year span gives the total shift, CI and all.
    span = 2024 - 2013
    trend = sorted((eff.topic, *_row(eff.effect_of("year"))) for eff in effects)
    print("Topics that fell, then rose, across the period (per-year slope x11):")
    for topic, coef, lo, hi, p in trend[:3] + trend[-3:]:
        arrow = "up" if coef > 0 else "down"
        print(f"  T{topic:>2} [{arrow}] {coef * span:+.4f} over 2013-2024  "
              f"CI[{lo * span:+.4f}, {hi * span:+.4f}]  p={p:.1e}  ({lab_words[topic]})")

    banner("Done")
    print("Bundled quick path:  df = topica.datasets.load_congress()")
    print("Reproduce:           seed=13 throughout; widen ks / iters / nsims for a paper.")


def _row(effect):
    """(coef, ci_low, ci_high, pvalue) from an effect_of(...) dict row."""
    return effect["coef"], effect["ci_low"], effect["ci_high"], effect["pvalue"]


if __name__ == "__main__":
    main()
