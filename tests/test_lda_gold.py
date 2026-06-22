"""Offline gold-fixture parity for topica LDA vs Java MALLET (#271, Wave 1).

Loads the committed gold (``parity/lda_gold.npz`` + ``.json``), fits topica LDA on
the frozen planted corpus, aligns its topics to MALLET's frozen run, and asserts
the aligned topic-word cosine AND top-word jaccard clear MALLET's own
seed-to-seed floor (minus a small margin).

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
import lda_gold  # noqa: E402


def test_lda_gold_present():
    npz, js = harness.gold_paths("lda")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/lda_gold.py --regenerate` "
        "(needs the mallet CLI)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_lda_gold_shapes():
    arrays, meta = harness.load_gold("lda")
    k = int(arrays["k"])
    mal_phi = arrays["mallet_phi"]
    vocab = arrays["vocab"]
    assert mal_phi.shape == (k, len(vocab))
    assert int(meta["K"]) == k


def test_lda_matches_committed_gold():
    r = lda_gold.run(verbose=False)
    assert r["passes"], (
        f"topica LDA vs MALLET cosine {r['cosine']:.4f} (bar {r['cosine_bar']:.4f}) "
        f"or jaccard {r['jaccard']:.4f} (bar {r['jaccard_bar']:.4f}) below MALLET's "
        f"own seed-to-seed floor; details: {r}"
    )


def test_lda_gold_is_non_vacuous():
    """A column-shuffled MALLET matrix scrambles each topic's top words, so the
    aligned top-word jaccard must FALL BELOW the bar — proving the gate
    discriminates a correct fit from a wrong one."""
    arrays, meta = harness.load_gold("lda")
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    self_jacc = float(meta["mallet_self_jaccard"])
    margin = float(meta["margin"])
    jacc_bar = self_jacc - margin

    rng = np.random.default_rng(0)
    shuffled = mal_phi[:, rng.permutation(mal_phi.shape[1])]
    jacc = harness.top_word_jaccard(mal_phi, shuffled, n=int(meta["top_n"]))
    assert jacc < jacc_bar, (
        f"column-shuffled jaccard {jacc:.4f} should be below the bar {jacc_bar:.4f}; "
        "the gate is vacuous"
    )
