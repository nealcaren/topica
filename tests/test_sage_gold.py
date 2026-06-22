"""Offline gold-fixture parity for topica SAGE (issue #271, Wave 1).

SAGE has NO external reference implementation (no gensim/tomotopy/R equivalent),
so this is a PLANTED self-consistency gold, not a cross-implementation one. It
loads the committed gold (``parity/sage_gold.npz`` + ``.json``), refits topica
SAGE on the same fixed-seed bilingual corpus, and asserts (a) the refit reproduces
the frozen group-specific topic-word distributions in cosine and (b) the planted
bilingual structure is recovered (each topic's en top-words are English, de
top-words German). The shuffle check proves the gate is non-vacuous.

No reference toolchain is touched at test time; the topica refit is fast (~few s).
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import sage_gold  # noqa: E402


def test_sage_gold_present():
    npz, js = harness.gold_paths("sage")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/sage_gold.py --regenerate`"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_sage_matches_committed_gold():
    r = sage_gold.run(verbose=False)
    assert r["passes"], (
        f"topica SAGE refit-vs-gold cosine {r['cosine']:.4f} below bar {r['bar']:.2f} "
        f"or recovery invariants failed {r['recovery_invariants']}; details: {r}"
    )


def test_sage_gold_is_non_vacuous():
    """A shuffled group-specific topic-word matrix must FALL BELOW the cosine bar —
    proving the gate discriminates a correct fit from a wrong one."""
    import numpy as np

    arrays, meta = harness.load_gold("sage")
    gold_kg = arrays["topic_word_kg"]            # (K, G, V)
    bar = float(meta["cosine_bar"])

    flat = sage_gold._flatten_kg(gold_kg)         # (K*G, V)
    rng = np.random.default_rng(0)
    shuffled = flat[:, rng.permutation(flat.shape[1])]
    cos, _ = harness.align_cosine(flat, shuffled)
    assert cos < bar, (
        f"shuffled SAGE topic-word cosine {cos:.4f} should be below the bar "
        f"{bar:.2f}; the gate is vacuous"
    )
