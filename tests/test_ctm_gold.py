"""Offline gold-fixture parity for topica CTM vs R `stm` fit as a CTM
(no covariates) — issue #271, Wave 1.

Loads the committed gold (``parity/ctm_gold.npz`` + ``.json``), fits topica CTM on
the SAME corpus that R was fit on (a fixed-seed poliblog subsample, frozen in the
gold), aligns to R's Spectral beta, and asserts the aligned cosine clears R's own
Spectral-vs-Random basin spread (minus a small multimodality margin). On
well-identified poliblog K=20 the absolute cosine is high (~0.9), a meaningful
validation. Runs in CI WITHOUT Rscript — the reference fit and exact corpus are
frozen in the committed gold. The topica refit is fast (~2s), so it stays in the
default suite (not marked slow).
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import ctm_gold  # noqa: E402
import harness  # noqa: E402


def test_ctm_gold_present():
    npz, js = harness.gold_paths("ctm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/ctm_gold.py --regenerate` "
        "(needs R + the stm package)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_ctm_matches_committed_gold():
    r = ctm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica CTM aligned cosine {r['spectral_cosine']:.4f} below bar {r['bar']:.4f} "
        f"(R Spectral-vs-Random {r['r_spectral_vs_random']:.4f} - margin); details: {r}"
    )


def test_ctm_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FAIL the cosine bar."""
    import numpy as np

    arrays, meta = harness.load_gold("ctm")
    r_spectral = arrays["r_spectral"]
    bar = float(meta["r_spectral_vs_random"]) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = r_spectral[:, rng.permutation(r_spectral.shape[1])]
    cos, _ = harness.align_cosine(r_spectral, shuffled)
    assert cos < bar, (
        f"shuffled beta cosine {cos:.4f} should be below the bar {bar:.4f}; "
        "the gate is vacuous"
    )
