"""#611: topica's HDP (resample_conc=True default) reproduces the blei-lab/hdp
concentration-update *equations*. `parity/hdp_blei_parity.py` runs an independent
NumPy implementation of the Escobar-West/Teh (2006) updates alongside topica on a
20-newsgroups subset; on the same corpus the two land in the same regime (topic
count, learned alpha/gamma, background-topic share), far from the degenerate
collapse that fixed low concentrations give. Both are direct-assignment samplers,
so this validates topica's equations and self-consistency — not a topic-count
match against the reference table-based (CRF) samplers.

Skipped when scikit-learn (the corpus source) is unavailable, per the parity/
convention; the pure-NumPy oracle itself has no external dependency.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn", reason="20NG corpus source not installed")

_PARITY = Path(__file__).resolve().parents[1] / "parity" / "hdp_blei_parity.py"


def _load_parity():
    spec = importlib.util.spec_from_file_location("hdp_blei_parity", _PARITY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hdp_blei_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_topica_matches_independent_concentration_oracle():
    mod = _load_parity()
    r = mod.run(iters=80, n_docs=60)
    o, t = r["oracle"], r["topica"]

    # Both discover a rich topic set, nothing like the fixed-conc collapse.
    fixed_k = r["topica_fixed"]["K"]
    assert t["K"] > 5 * fixed_k, (t["K"], fixed_k)
    assert o["K"] > 5 * fixed_k, (o["K"], fixed_k)

    # topica tracks the independent Escobar-West/Teh oracle (different RNG ->
    # statistical match, generous bands). Topic count within 40%.
    assert 0.6 <= t["K"] / o["K"] <= 1.4, (t["K"], o["K"])
    # Learned concentrations within a factor of two either way.
    assert 0.5 <= t["alpha"] / o["alpha"] <= 2.0, (t["alpha"], o["alpha"])
    assert 0.5 <= t["gamma"] / o["gamma"] <= 2.0, (t["gamma"], o["gamma"])
    # Background-topic share matches within 10 points.
    assert abs(t["top_share"] - o["top_share"]) < 0.10, (t["top_share"], o["top_share"])


@pytest.mark.slow
def test_estimated_conc_default_avoids_collapse():
    # The headline of the default change: with resampling on (the default) the
    # model does not collapse to a dominant background topic the way fixed low
    # concentrations do.
    mod = _load_parity()
    r = mod.run(iters=80, n_docs=60)
    assert r["topica"]["K"] >= 20
    assert r["topica"]["top_share"] < 0.6
    # The fixed-conc reference genuinely collapses: few topics AND one topic
    # holding most of the token mass.
    fixed = r["topica_fixed"]
    assert fixed["K"] <= 15
    assert fixed["top_share"] > 0.5, fixed
