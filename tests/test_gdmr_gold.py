"""Offline gold-fixture parity for topica GDMR vs tomotopy ``GDMRModel`` (#271, Wave 1).

Loads the committed gold (``parity/gdmr_gold.npz`` + ``.json``), fits topica GDMR
on the same synthetic two-cluster corpus + metadata, aligns its space topic to
tomotopy's frozen run, and asserts (a) the space-topic ``tdf`` curve correlates
with tomotopy's at least as well as tomotopy reproduces itself across seeds (minus
a small margin) and (b) the space-topic word distribution matches in cosine.

This runs in CI WITHOUT tomotopy: the reference fit is frozen in the committed
gold, so no tomotopy is imported at test time. The topica refit is fast (~few s),
so it stays in the default suite.
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import gdmr_gold  # noqa: E402


def test_gdmr_gold_present():
    npz, js = harness.gold_paths("gdmr")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/gdmr_gold.py --regenerate` "
        "(needs tomotopy)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_gdmr_matches_committed_gold():
    r = gdmr_gold.run(verbose=False)
    assert r["passes"], (
        f"topica GDMR space-topic tdf r {r['tdf_r']:.4f} below bar {r['bar']:.4f} "
        f"(tomotopy self {r['tomotopy_tdf_self_r']:.4f} - margin) or word cosine "
        f"{r['word_cosine']:.4f} < 0.9; details: {r}"
    )


def test_gdmr_gold_is_non_vacuous():
    """A reversed tdf curve (space topic responding the wrong way to metadata)
    must FALL BELOW the bar — proving the gate discriminates a correct fit."""
    import numpy as np

    arrays, meta = harness.load_gold("gdmr")
    gold_tdf = arrays["space_tdf"]
    self_r = float(meta["tomotopy_tdf_self_r"])
    bar = self_r - float(meta["margin"])

    # Reversing the curve flips its monotone response; r must drop below the bar.
    reversed_tdf = gold_tdf[::-1]
    r = float(np.corrcoef(gold_tdf, reversed_tdf)[0, 1])
    assert r < bar, (
        f"reversed tdf curve r {r:.4f} should be below the bar {bar:.4f}; "
        "the gate is vacuous"
    )
