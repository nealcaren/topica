"""Offline planted-gold test for topica TensorLDA (Wave 2).
Loads the committed gold (``parity/tlda_gold.npz`` + ``.json``), refits topica on the
same fixed-seed planted corpus, and asserts (1) the refit reproduces the frozen
topic-word matrix in cosine (determinism), (2) the planted structure is recovered,
and (3) the Wave 0 validity invariants hold.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import tlda_gold  # noqa: E402

NAME = "tlda"


def test_tlda_gold_present():
    npz, js = harness.gold_paths(NAME)
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/tlda_gold.py --regenerate`"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_tlda_matches_committed_gold():
    r = tlda_gold.run(verbose=False)
    assert r["passes"], (
        f"topica TensorLDA refit-vs-gold cosine {r['cosine']:.4f} below bar "
        f"{r['cosine_bar']:.2f} or recovery failed {r['recovery']}; details: {r}"
    )


def test_tlda_gold_is_non_vacuous():
    arrays, meta = harness.load_gold(NAME)
    gold_tw = arrays["topic_word"]
    bar = float(meta["cosine_bar"])

    rng = np.random.default_rng(0)
    shuffled = gold_tw[:, rng.permutation(gold_tw.shape[1])]
    cos, _ = harness.align_cosine(gold_tw, shuffled)
    assert cos < bar, (
        f"shuffled TensorLDA topic-word cosine {cos:.4f} should be below the bar "
        f"{bar:.2f}; the gate is vacuous"
    )
