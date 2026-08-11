"""5-group run: topic-model clusters vs. embedding classifier on the bundled
``examples/ng20_minilm.npz`` (comp.graphics, rec.sport.baseball, sci.med,
sci.space, talk.politics.guns). See report.md for the writeup and experiment.py
for the pipeline. K = 5 = number of source newsgroups.

    python experiments/ng20_topic_classify/run_experiment.py
"""

from pathlib import Path

from experiment import run

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent.parent / "examples" / "ng20_minilm.npz"

if __name__ == "__main__":
    run(NPZ, K=5, outdir=HERE, seed=13, k_sweep=(5, 8, 10, 15), tag="ng20-5group")
