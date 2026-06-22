"""Offline gold-fixture parity for topica DMR vs Java MALLET (#271, Wave 1).

Loads the committed gold (``parity/dmr_gold.npz`` + ``.json``), fits topica DMR on
the frozen two-cluster corpus + binary covariate, aligns its topics to MALLET's
frozen run, and asserts (a) the aligned topic-word cosine clears MALLET's own
seed-to-seed floor (minus a small margin) and (b) the covariate's effect
(space-minus-animal ``is_space`` weight) is strongly positive in BOTH engines.

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
import dmr_gold  # noqa: E402


def test_dmr_gold_present():
    npz, js = harness.gold_paths("dmr")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/dmr_gold.py --regenerate` "
        "(needs the mallet jars + javac)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_dmr_gold_shapes():
    arrays, meta = harness.load_gold("dmr")
    mal_phi = arrays["mallet_phi"]
    words = arrays["words"]
    lam = arrays["mallet_lambda"]
    assert mal_phi.shape == (int(meta["K"]), len(words))
    assert lam.shape == (int(meta["K"]), 2)  # intercept + is_space


def test_dmr_matches_committed_gold():
    r = dmr_gold.run(verbose=False)
    assert r["passes"], (
        f"topica DMR vs MALLET cosine {r['cosine']:.4f} (bar {r['cosine_bar']:.4f}) "
        f"or covariate effect MALLET {r['mallet_effect']:+.2f} / topica "
        f"{r['topica_effect']:+.2f} (both must exceed 1.0); details: {r}"
    )


def test_dmr_gold_is_non_vacuous():
    """A column-shuffled MALLET topic-word matrix must drop the aligned cosine
    below the bar — proving the topic gate discriminates a correct fit."""
    arrays, meta = harness.load_gold("dmr")
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    self_cos = float(meta["mallet_self_cosine"])
    bar = self_cos - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = mal_phi[:, rng.permutation(mal_phi.shape[1])]
    cos = float(
        (harness._row_normalize(mal_phi) @ harness._row_normalize(shuffled).T)
        .max(axis=1).mean()
    )
    assert cos < bar, (
        f"column-shuffled cosine {cos:.4f} should be below the bar {bar:.4f}; "
        "the gate is vacuous"
    )
