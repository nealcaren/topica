# ng20 topic-cluster / embedding-classifier experiment

Do an unsupervised topic model's cluster assignments survive a supervised
classifier trained on sentence embeddings — and are the documents that "come up
in a different cluster" a meaningful set?

The pipeline: fit topica LDA (K topics) on the token texts, take each doc's
argmax topic as its cluster; train a cross-validated logistic-regression
classifier on the sentence embeddings to recover those clusters; then study the
documents where the two independent views (word counts vs. embeddings) disagree,
using the held-out true newsgroup only as an external referee.

## Layout

- `experiment.py` — the shared pipeline (`run(npz, K, outdir, ...)`).
- **5 groups** (bundled `examples/ng20_minilm.npz`, K=5):
  - `run_experiment.py` — `python experiments/ng20_topic_classify/run_experiment.py`
  - `report.md`, plus `results.json` / `disagreements.csv` / `fig_*.png` here.
- **All 20 groups** (K=20, embeddings built locally):
  - `prepare_full_ng20.py` — download 20NG, apply the bundled preprocessing
    recipe, compute MiniLM embeddings → `ng20_full_minilm.npz` (git-ignored).
  - `run_full.py` — runs the pipeline, writes into `full/`.
  - `report_full.md`, plus `full/results.json` / `full/disagreements.csv` /
    `full/fig_*.png`.
- `report.html` — combined visual writeup (both scales).

## Reproduce

```
python experiments/ng20_topic_classify/run_experiment.py          # 5 groups
python experiments/ng20_topic_classify/prepare_full_ng20.py       # build full npz
python experiments/ng20_topic_classify/run_full.py                # 20 groups
```

Deps beyond topica: `numpy scikit-learn scipy matplotlib` (+ `sentence-transformers`
and `torch` for `prepare_full_ng20.py`). Scripts import `experiment.py`, so run
them from this directory or with it on `PYTHONPATH`.

## Headline

|  | 5 groups | 20 groups |
|---|---:|---:|
| embeddings → LDA cluster (CV) | 0.850 | 0.729 |
| embeddings → true group (ceiling) | 0.899 | 0.725 |
| shuffled-label floor / base rate | 0.43 / 0.45 | 0.26 / 0.27 |
| disagreements | 15.0% | 27.1% |

Clusters are strongly learnable from embeddings at both scales. Disagreements are
a meaningful set of ambiguous, boundary-straddling docs; at 5 groups the
embedding pick also matches the true label more often than the LDA cluster it
overrides, while at 20 groups that clean win dilutes (junk posts + overlapping
official groups). See `report.md` / `report_full.md`.
