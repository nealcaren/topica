"""#433: HDP's resampled-concentration cap was hard-coded at 2.0, pinning any
posterior with mass above it to a spurious atom (biasing gamma/alpha and K
downward). The cap is now a configurable `concentration_max` (default 2.0, so
existing behavior is unchanged) that users can raise for corpora that legitimately
support larger concentrations."""
import numpy as np
import pytest
import topica


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    blocks = [[f"w{b * 6 + i}" for i in range(6)] for b in range(6)]
    return [list(rng.choice(blocks[d % 6], size=12)) for d in range(300)]


def test_concentration_max_is_exposed_and_defaults_to_two():
    m = topica.HDP(resample_conc=True)
    assert m.settings["concentration_max"] == 2.0


def test_concentration_max_survives_save_load(tmp_path):
    m = topica.HDP(resample_conc=True, concentration_max=25.0, seed=3)
    m.fit(_corpus(), iters=20)
    p = str(tmp_path / "hdp.topica")
    m.save(p)
    assert topica.HDP.load(p).settings["concentration_max"] == 25.0


def test_raising_the_cap_lets_the_learned_concentration_exceed_two():
    # A blocky corpus with resampling on: the default cap holds every resampled
    # gamma at or below 2.0, while a raised cap lets it float above. The learned
    # concentrations are recorded in concentration_history as (iter, alpha, gamma).
    capped = topica.HDP(resample_conc=True, concentration_max=2.0, gamma=1.0, seed=1)
    capped.fit(_corpus(), iters=60)
    freed = topica.HDP(resample_conc=True, concentration_max=1e6, gamma=1.0, seed=1)
    freed.fit(_corpus(), iters=60)
    capped_max = max(g for _, _, g in capped.concentration_history)
    freed_max = max(g for _, _, g in freed.concentration_history)
    assert capped_max <= 2.0 + 1e-9
    assert freed_max > 2.0 + 1e-6, (capped_max, freed_max)


def test_invalid_concentration_max_is_rejected():
    for bad in (0.0, 1e-4, float("nan"), float("inf"), -5.0):
        with pytest.raises(ValueError):
            topica.HDP(concentration_max=bad)


def test_runtime_signature_default_matches_the_stub():
    # The pyo3 signature must render a literal 2.0 (not Ellipsis from a Rust const
    # expression), so introspection/docs agree with _topica.pyi's `= 2.0`.
    import inspect

    default = inspect.signature(topica.HDP).parameters["concentration_max"].default
    assert default == 2.0, repr(default)


def test_cap_bounds_alpha_not_only_gamma():
    # The document-level alpha is also clamped by the cap. (That a raised cap lets
    # alpha exceed 2.0 needs a state whose second-level posterior wants alpha > 2,
    # which this blocky corpus does not produce — that direction is covered by the
    # controlled Rust test `raising_the_cap_removes_the_spurious_atom_at_two`.)
    # Here we just confirm alpha never breaches the default cap.
    capped = topica.HDP(resample_conc=True, concentration_max=2.0, seed=2)
    capped.fit(_corpus(), iters=60)
    assert max(a for _, a, _ in capped.concentration_history) <= 2.0 + 1e-9
