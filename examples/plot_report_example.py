"""Generate the plot_report figure shown in the documentation.

Fits an LDA on the political-blog corpus and renders topica.plot_report with all
five panels (prevalence, quality, correlation, topics over time, prevalence by
class), then saves it to docs/images/model_report.png. Reproducible from the seed;
re-run to refresh the docs image.

    python examples/plot_report_example.py
"""

import csv
import os

import numpy as np

import topica

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLIBLOG = os.path.join(ROOT, "examples", "poliblog.csv")
OUT = os.path.join(ROOT, "docs", "images", "model_report.png")


def main():
    import matplotlib

    matplotlib.use("Agg")

    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    docs = [r["text"].split() for r in rows]
    texts = [r["text"] for r in rows]
    rating = [r["rating"] for r in rows]
    # Bin the 360 days into 6 evenly sized periods for a readable time axis.
    day = np.array([float(r["day"]) for r in rows])
    period = (day.argsort().argsort() * 6 // len(day) + 1)
    period = [f"P{p}" for p in period]

    model = topica.models.LDA(num_topics=8, seed=1)
    model.fit(docs, iters=800)

    fig = topica.plot_report(
        model, texts=texts, timestamps=period, groups=rating, n=6,
        title="LDA on political blogs (K=8)",
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
