"""Full 20-group run: topic-model clusters vs. embedding classifier on all 20
newsgroups. Requires ng20_full_minilm.npz — build it first with
prepare_full_ng20.py. Writes artifacts into ./full/. K = 20 = number of groups.

    python experiments/ng20_topic_classify/prepare_full_ng20.py
    python experiments/ng20_topic_classify/run_full.py
"""

from pathlib import Path

from experiment import run

HERE = Path(__file__).resolve().parent
NPZ = HERE / "ng20_full_minilm.npz"

if __name__ == "__main__":
    if not NPZ.exists():
        raise SystemExit(f"missing {NPZ}; run prepare_full_ng20.py first")
    run(NPZ, K=20, outdir=HERE / "full", seed=13,
        k_sweep=(10, 20, 30, 40), tag="ng20-20group")
