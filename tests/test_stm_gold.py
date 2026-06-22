"""Offline gold-fixture parity for topica STM vs R `stm` (issue #271, Wave 1).

Loads the committed gold (``parity/stm_gold.npz`` + ``.json``), fits topica STM on
the same gadarian corpus + vocab, aligns to R's Spectral beta, and asserts the
aligned cosine clears R's own Spectral-vs-Random basin spread (minus a small
multimodality margin). This runs in CI WITHOUT Rscript: the reference fit is frozen
in the committed gold, so no R toolchain is touched at test time.
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import stm_gold  # noqa: E402


def test_stm_gold_present():
    npz, js = harness.gold_paths("stm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/stm_gold.py --regenerate` "
        "(needs R + the stm package)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_stm_matches_committed_gold():
    r = stm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica STM aligned cosine {r['spectral_cosine']:.4f} below bar {r['bar']:.4f} "
        f"(R Spectral-vs-Random {r['r_spectral_vs_random']:.4f} - margin); details: {r}"
    )


def test_stm_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FAIL the cosine bar — proving the gate
    actually discriminates a correct fit from a wrong one."""
    import numpy as np

    arrays, meta = harness.load_gold("stm")
    r_spectral = arrays["r_spectral"]
    bar = float(meta["r_spectral_vs_random"]) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = r_spectral[:, rng.permutation(r_spectral.shape[1])]
    cos, _ = harness.align_cosine(r_spectral, shuffled)
    assert cos < bar, (
        f"shuffled beta cosine {cos:.4f} should be below the bar {bar:.4f}; "
        "the gate is vacuous"
    )
