"""ECTM on the 2008 political-blog corpus: how Conservative and Liberal bloggers
worded the campaign's topics, and how that partisan vocabulary gap moved across
the year.

The corpus (``examples/poliblog.csv``, 2,000 already-stemmed posts) carries a
``rating`` (Conservative / Liberal) and a ``day`` in 2008. We treat ``rating`` as
the content **group** and the calendar **quarter** as the time period, then read
the per-(group, quarter) topic-word distributions ECTM fits.

Run from the repo root:  python examples/ectm_poliblog.py
"""
import csv
import os

import numpy as np

import topica
from topica.ectm import content_contrast, content_divergence, content_trajectory

topica.enable_experimental()  # ECTM is experimental and gated; opt in to use it


def load():
    path = os.path.join(os.path.dirname(__file__), "poliblog.csv")
    rows = list(csv.DictReader(open(path)))
    docs = [r["text"].split() for r in rows]
    groups = [r["rating"] for r in rows]
    # Bin day (2..366) into the four quarters of 2008.
    quarters = [f"Q{(int(r['day']) - 1) // 92 + 1}" for r in rows]
    return docs, groups, quarters


def main():
    docs, groups, quarters = load()
    print(f"{len(docs)} posts | groups={sorted(set(groups))} | periods={sorted(set(quarters))}")

    model = topica.ECTM(num_topics=10, seed=42)
    model.fit(
        docs,
        times=quarters,
        content=groups,
        iters=200,
        period_smooth=5.0,       # random-walk pooling across adjacent quarters
        interaction_shrink=2.0,  # pull the changing group gap toward zero unless earned
    )
    print(f"fitted: {model} | converged={model.converged}\n")

    # Pick the topic with the largest Conservative-vs-Liberal vocabulary gap,
    # averaged over quarters (the most partisan-worded topic).
    def avg_divergence(k):
        return np.mean([d for _, d in content_divergence(model, k, "Conservative", "Liberal")])

    topic = max(range(model.num_topics), key=avg_divergence)
    print(f"Most partisan-worded topic: #{topic}")
    print("  shared top words:", " ".join(w for w, _ in model.top_words(8, topic=topic)))

    # How far apart the two sides' wording of this topic is, quarter by quarter.
    print("\nConservative vs Liberal vocabulary distance (total variation), by quarter:")
    for q, dist in content_divergence(model, topic, "Conservative", "Liberal"):
        bar = "#" * int(round(dist * 40))
        print(f"  {q}  {dist:.3f}  {bar}")

    # The words that most distinguished each side in the final quarter.
    print(f"\nMost distinctive words in Q4 (topic #{topic}):")
    contrast = content_contrast(model, topic, "Conservative", "Liberal", "Q4", n=6)
    print("  Conservative:", ", ".join(f"{w}" for w, _ in contrast["toward_Conservative"]))
    print("  Liberal:    ", ", ".join(f"{w}" for w, _ in contrast["toward_Liberal"]))

    # Trace one word's partisan contrast across the year. Use the word with the
    # biggest Q4 gap toward Conservatives.
    word = contrast["toward_Conservative"][0][0]
    print(f"\nPartisan contrast for '{word}' across 2008 (Conservative - Liberal prob):")
    for q, val in content_trajectory(model, topic, word, contrast=("Conservative", "Liberal")):
        print(f"  {q}  {val:+.4f}")


if __name__ == "__main__":
    main()
