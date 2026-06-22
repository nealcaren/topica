"""Offline gold-fixture parity for topica InfoCTM vs a paper-derived PyTorch reference (#271, Wave 1).

Loads the committed gold (``parity/infoctm_gold.npz`` + ``.json``), fits topica
InfoCTM on the SAME matched bilingual planted-block corpus + block-aligned
dictionary, and asserts topica's own cross-lingual topic alignment clears the live
``infoctm_compare`` bar (>= 0.8): topic ``k`` must concentrate on the same planted
block in both languages. Per-language top-word block agreement against the frozen
reference is reported as a diagnostic but is not the bar — the live script holds the
two implementations to cross-lingual alignment, not topic-word cosine, because
topica is sharper (lower entropy) by design.

This runs in CI WITHOUT torch: the reference fit and the corpus / dictionary are
frozen in the committed gold, so no torch is imported at test time (asserted
explicitly below). The topica refit (A=300 / B=270 docs, K=5, 200 iters) is fast
(~5s), so it stays in the default suite alongside the gold-present / shape /
non-vacuous checks.
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import infoctm_gold  # noqa: E402


def test_infoctm_gold_present():
    npz, js = harness.gold_paths("infoctm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/infoctm_gold.py --regenerate` "
        "(needs torch)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_infoctm_gold_shape():
    """Fast default check: the two frozen reference topic-word matrices are
    (K, V_lang) and the dictionary is a list of word pairs."""
    arrays, meta = harness.load_gold("infoctm")
    assert arrays["topic_word_a"].shape == (meta["num_topics"], meta["vocab_size_a"])
    assert arrays["topic_word_b"].shape == (meta["num_topics"], meta["vocab_size_b"])
    assert arrays["dictionary"].shape[1] == 2


def test_infoctm_no_torch_at_test_time():
    """The committed gold must validate with no deep-learning framework present."""
    assert "torch" not in sys.modules, (
        "torch was imported at test time; the gold must validate offline"
    )


def test_infoctm_gold_is_non_vacuous():
    """A within-language shuffle that scrambles which block each topic's top word
    falls in must drop the cross-lingual alignment below the bar — proving the gate
    discriminates a correct fit. Runs by default (no fit)."""
    import numpy as np

    arrays, meta = harness.load_gold("infoctm")
    beta_a = arrays["topic_word_a"].astype(np.float64)
    beta_b = arrays["topic_word_b"].astype(np.float64)
    bar = float(meta["align_bar"])

    # Frozen reference is perfectly aligned; shuffling language B's vocab columns
    # destroys the block correspondence, so alignment must fall below the bar.
    aligned = infoctm_gold._alignment(beta_a, beta_b)
    assert aligned >= bar, "frozen reference should itself be aligned"

    rng = np.random.default_rng(0)
    beta_b_shuf = beta_b[:, rng.permutation(beta_b.shape[1])]
    scrambled = infoctm_gold._alignment(beta_a, beta_b_shuf)
    assert scrambled < bar, (
        f"vocab-shuffled InfoCTM alignment {scrambled:.2f} should be below the bar "
        f"{bar:.2f}; the gate is vacuous"
    )


def test_infoctm_matches_committed_gold():
    """Refit topica InfoCTM and check its cross-lingual alignment clears the bar."""
    r = infoctm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica InfoCTM cross-lingual alignment {r['topica_alignment']:.2f} below "
        f"bar {r['bar']:.2f}; details: {r}"
    )
