"""Offline gold-fixture parity for topica LabeledLDA vs Java MALLET (#271, Wave 1).

Loads the committed gold (``parity/labeledlda_gold.npz`` + ``.json``), fits topica
LabeledLDA on the frozen multi-label corpus, aligns its topics to MALLET's frozen
run BY LABEL NAME, and asserts the mean per-label topic-word cosine clears
MALLET's own seed-to-seed floor (minus a small margin).

This runs in CI WITHOUT MALLET / Java: the reference fit is frozen in the
committed gold, so no ``mallet`` or ``java`` is shelled out at test time. The
topica refit is fast (<1s), so it stays in the default suite.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import labeledlda_gold  # noqa: E402


def test_labeledlda_gold_present():
    npz, js = harness.gold_paths("labeledlda")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/labeledlda_gold.py "
        "--regenerate` (needs the mallet jars + javac)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_labeledlda_gold_shapes():
    arrays, meta = harness.load_gold("labeledlda")
    mal_phi = arrays["mallet_phi"]
    labels = arrays["labels"]
    words = arrays["words"]
    assert mal_phi.shape == (len(labels), len(words))
    assert int(meta["num_labels"]) == len(labels)


def test_labeledlda_matches_committed_gold():
    r = labeledlda_gold.run(verbose=False)
    assert r["passes"], (
        f"topica LabeledLDA vs MALLET mean per-label cosine {r['cosine']:.4f} below "
        f"bar {r['cosine_bar']:.4f} (MALLET self {r['mallet_self_cosine']:.4f} - "
        f"margin); details: {r}"
    )


def test_labeledlda_gold_is_non_vacuous():
    """A column-shuffled MALLET topic-word matrix must drop the mean per-label
    cosine below the bar — proving the gate discriminates a correct fit."""
    arrays, meta = harness.load_gold("labeledlda")
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    self_cos = float(meta["mallet_self_cosine"])
    bar = self_cos - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = mal_phi[:, rng.permutation(mal_phi.shape[1])]
    cos = float(np.mean(labeledlda_gold._label_cosines(mal_phi, shuffled)))
    assert cos < bar, (
        f"column-shuffled mean per-label cosine {cos:.4f} should be below the bar "
        f"{bar:.4f}; the gate is vacuous"
    )
