"""Offline gold-fixture parity for topica SITS vs the reference Java sampler (#906).

Loads the committed gold (``parity/sits_gold.npz`` + ``.json``): six reseeded chains
of Rossiter's fork of the SITS Java sampler, as published and with its one-line
initialization fix, on a corpus simulated from the SITS generative process. Fits six
topica chains in each mode (default vs the fixed fork, ``compat="rossiter2022"`` vs
the published fork) and asserts agreement on eligible-turn shift rates (per speaker
and overall), per-turn shift probabilities, and aligned topic-word distributions,
each against the Java chains' own spread.

Runs in CI without Java: the reference chains are frozen in the committed gold.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import sits_compare  # noqa: E402


def test_sits_gold_present():
    npz, js = harness.gold_paths("sits")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/sits_compare.py --regenerate` "
        "(needs git + javac)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_sits_gold_shapes():
    arrays, meta = harness.load_gold("sits")
    n = len(arrays["speakers"])
    for tag in ("published", "fixed"):
        assert arrays[f"{tag}_shift_prob"].shape == (meta["chains"], n)
        assert arrays[f"{tag}_phi"].shape == (meta["chains"], sits_compare.K, sits_compare.V)


def test_sits_matches_committed_gold():
    r = sits_compare.run(verbose=False)
    assert r["passes"], f"topica SITS disagrees with the Java reference: {r}"


def test_sits_gold_is_non_vacuous():
    """The eligible-rate gate must reject a sampler that never shifts, and the
    fork's bug must be visible: published and fixed Java differ beyond the bar."""
    arrays, _ = harness.load_gold("sits")
    turns = harness.lines_to_docs(str(arrays["turns"]))
    speakers, convs = arrays["speakers"], arrays["convs"]
    _, eligible = sits_compare._masks(turns, convs)
    java = arrays["fixed_shift_prob"]
    jr = np.array([sits_compare._speaker_rates(p, speakers, eligible) for p in java])
    never = java.copy()
    never[:, eligible] = 0.0
    nr = np.array([sits_compare._speaker_rates(p, speakers, eligible) for p in never])
    assert np.abs(sits_compare._z(nr, jr)).max() > sits_compare.Z_BAR
    pub = arrays["published_shift_prob"]
    po = pub[:, eligible].mean(axis=1)[:, None]
    fo = java[:, eligible].mean(axis=1)[:, None]
    assert abs(float(sits_compare._z(po, fo)[0])) > sits_compare.Z_BAR
