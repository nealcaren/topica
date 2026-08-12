"""Offline gold-fixture parity for topica STM vs R `stm` (issue #271, Wave 1).

Loads the committed gold (``parity/stm_gold.npz`` + ``.json``), fits topica STM on
the SAME corpus + design matrix that R was fit on (a fixed-seed poliblog subsample,
frozen in the gold), aligns to R's Spectral beta, and asserts the aligned cosine
clears R's own Spectral-vs-Random basin spread (minus a small multimodality
margin). On well-identified poliblog K=20 the absolute cosine is high (~0.9), so
the bar is cleared by a wide margin — a meaningful validation, unlike the
multimodal gadarian corpus where R barely reproduced itself.

This runs in CI WITHOUT Rscript: the reference fit and the exact corpus are frozen
in the committed gold, so no R toolchain is touched at test time. The topica refit
is fast (~2s), so it stays in the default suite (not marked slow).
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


def test_stm_whole_model_parity():
    """Beyond topic-word beta, the committed gold now also gates the whole-model
    agreement with R (issue #715-#1): the doc-topic theta, the topic-correlation
    matrix (Sigma's interpretable K-space form), and the per-topic prevalence
    effect (gamma's interpretable form, the substantive conclusion). gamma/Sigma
    are not compared in the raw (K-1) reference space, where two independent fits'
    relabeled topics do not align, but through these aligned K-space quantities."""
    r = stm_gold.run(verbose=False)
    if "theta_cosine" not in r:
        import pytest

        pytest.skip("committed gold predates the whole-model parity arrays; regenerate")
    assert r["theta_passes"], f"theta cosine {r['theta_cosine']:.4f} < 0.85; {r}"
    assert r["corr_passes"], f"topic-correlation cosine {r['topic_corr_cosine']:.4f} < 0.85; {r}"
    assert r["effect_passes"], f"rating-effect correlation {r['effect_corr']:.4f} < 0.80; {r}"
    # The prevalence effect (the substantive conclusion) agrees in sign on the
    # large majority of topics.
    assert r["effect_sign_agree"] >= int(0.75 * r["effect_sign_total"]), r


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
