# ng20 topic-cluster / embedding-classifier experiment

Do an unsupervised topic model's cluster assignments survive a supervised
classifier trained on sentence embeddings — and are the documents that "come up
in a different cluster" a meaningful set?

- `run_experiment.py` — end-to-end pipeline (LDA clusters -> embedding
  classifier -> disagreement analysis + controls). Run from repo root:
  `python experiments/ng20_topic_classify/run_experiment.py`
- `report.md` — full writeup and interpretation.
- `results.json`, `disagreements.csv`, `fig_confusion.png`, `fig_entropy.png` —
  generated outputs.

Deps beyond topica: `numpy scikit-learn scipy matplotlib`.
