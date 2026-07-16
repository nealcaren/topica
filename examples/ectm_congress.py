"""ECTM on the Congressional-bills corpus from the keyATM paper (Eshima, Imai &
Sasaki 2024): how the House and Senate word the same policy topics, and how that
chamber difference has evolved across 14 Congresses (101st-114th, 1989-2017).

The corpus is the 4,421 floor-vote bills used in the keyATM base/dynamic models
(Harvard Dataverse doi:10.7910/DVN/RKNNVL). We treat the **chamber** (House /
Senate) as the content group and the **Congress** as the time period -- 14
periods, so ECTM's random-walk pooling across adjacent Congresses earns its keep
on the thinner Senate cells.

First build the data once (needs R + quanteda):
    Rscript examples/prep_congress.R
Then run:
    python examples/ectm_congress.py
"""
import csv
import os

import numpy as np
from scipy.io import mmread

import topica
from topica.ectm import content_contrast, content_divergence

topica.enable_experimental()  # ECTM is experimental and gated; opt in to use it

DATA = os.path.join(os.path.dirname(__file__), "congress_data")
TOKENS_PER_BILL = 250  # subsample: bills average ~5,500 tokens; 250 keeps the fit quick


def load():
    if not os.path.exists(os.path.join(DATA, "counts.mtx")):
        raise SystemExit("Run `Rscript examples/prep_congress.R` first to build examples/congress_data/.")
    counts = mmread(os.path.join(DATA, "counts.mtx")).tocsr()
    vocab = open(os.path.join(DATA, "vocab.txt")).read().split("\n")
    meta = list(csv.DictReader(open(os.path.join(DATA, "meta.csv"))))
    rng = np.random.default_rng(0)
    docs = []
    for d in range(counts.shape[0]):
        row = counts.getrow(d)
        idx, cnt = row.indices, row.data.astype(float)
        if cnt.sum() == 0:
            docs.append([])
            continue
        draw = rng.multinomial(min(int(cnt.sum()), TOKENS_PER_BILL), cnt / cnt.sum())
        docs.append([vocab[idx[j]] for j in np.nonzero(draw)[0] for _ in range(int(draw[j]))])
    chambers = [m["chamber"] for m in meta]
    congress = [int(m["congress"]) for m in meta]
    cap = [m["cap_topic"] for m in meta]
    return docs, chambers, congress, cap


def label(model, topic, cap, doc_topic):
    """Name a fitted topic by the most common CAP gold topic among the bills that
    load most heavily on it."""
    top_docs = np.argsort(doc_topic[:, topic])[::-1][:40]
    votes = [cap[d] for d in top_docs if cap[d] != "Unlabeled"]
    return max(set(votes), key=votes.count) if votes else "?"


def main():
    docs, chambers, congress, cap = load()
    print(f"{len(docs)} bills | House {chambers.count('House')} / Senate {chambers.count('Senate')} "
          f"| Congress {min(congress)}-{max(congress)}")

    model = topica.models.ECTM(num_topics=15, seed=1)
    model.fit(docs, times=congress, content=chambers, iters=120,
              period_smooth=8.0, interaction_shrink=2.0)
    print(f"fitted: {model} | converged={model.converged}\n")

    dt = model.doc_topic
    labels = [label(model, k, cap, dt) for k in range(model.num_topics)]

    # The topic the chambers word most differently, averaged over Congresses.
    def avg_div(k):
        return np.mean([d for _, d in content_divergence(model, k, "House", "Senate")])

    topic = max(range(model.num_topics), key=avg_div)
    print(f"Topic the House and Senate word most differently: #{topic} "
          f"(CAP gold label: {labels[topic]})")
    print("  shared top words:", " ".join(w for w, _ in model.top_words(10, topic=topic)))

    print("\nHouse vs Senate vocabulary distance (total variation) by Congress:")
    for cong, dist in content_divergence(model, topic, "House", "Senate"):
        print(f"  {cong}  {dist:.3f}  {'#' * int(round(dist * 50))}")

    # How each chamber worded the topic, early (101st) vs late (114th).
    for cong in (model.periods[0], model.periods[-1]):
        con = content_contrast(model, topic, "House", "Senate", cong, n=6)
        print(f"\nDistinctive words, Congress {cong} (topic #{topic}):")
        print("  House: ", ", ".join(w for w, _ in con["toward_House"]))
        print("  Senate:", ", ".join(w for w, _ in con["toward_Senate"]))


if __name__ == "__main__":
    main()
