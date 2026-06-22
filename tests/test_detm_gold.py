"""Offline planted-gold test for topica DETM (issue #271, Wave 2).

DETM has NO external reference implementation, so this is a PLANTED
self-consistency gold (see ``parity/detm_gold.py``). It loads the committed gold
(``parity/detm_gold.npz`` + ``.json``), refits topica on the same fixed-seed
planted corpus, and asserts (1) the refit reproduces the frozen topic-word matrix
in cosine (determinism), (2) the planted structure is recovered, and (3) the
Wave 0 validity invariants hold. The shuffle check proves the gate is non-vacuous.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import detm_gold  # noqa: E402

NAME = "detm"


def test_detm_gold_present():
    npz, js = harness.gold_paths(NAME)
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/detm_gold.py --regenerate`"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_detm_matches_committed_gold():
    # Asserts determinism (refit-vs-gold cosine), planted recovery, and the
    # Wave 0 validity invariants (run inside wave2.run).
    r = detm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica DETM refit-vs-gold cosine {r['cosine']:.4f} below bar "
        f"{r['cosine_bar']:.2f} or recovery failed {r['recovery']}; details: {r}"
    )


def test_detm_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FALL BELOW the cosine bar (and lose its
    top-word overlap), proving the gate discriminates a correct fit from a wrong
    one even when the softmax rows are near-flat."""
    arrays, meta = harness.load_gold(NAME)
    gold_tw = arrays["topic_word"]
    bar = float(meta["cosine_bar"])

    rng = np.random.default_rng(0)
    shuffled = gold_tw[:, rng.permutation(gold_tw.shape[1])]
    cos, _ = harness.align_cosine(gold_tw, shuffled)
    jacc = harness.top_word_jaccard(gold_tw, shuffled, n=5)
    assert cos < bar, (
        f"shuffled DETM topic-word cosine {cos:.4f} should be below the bar "
        f"{bar:.2f}; the gate is vacuous"
    )
    # Belt-and-suspenders for near-flat softmax rows: the shuffle must also wreck
    # the top-word overlap a genuine recovery keeps high.
    assert jacc < 0.5, (
        f"shuffled DETM top-word jaccard {jacc:.4f} too high; gate is weak"
    )
