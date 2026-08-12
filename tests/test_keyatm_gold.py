"""Offline gold-fixture parity for topica KeyATM vs R `keyATM` (issue #271, Wave 1).

Loads the committed gold (``parity/keyatm_gold.npz`` + ``.json``), fits topica
KeyATM on the same poliblog corpus + keyword sets, aligns to R's keyword-topic
``phi``, and asserts the aligned cosine clears R's own seed-to-seed keyword-phi
noise floor (minus a small multimodality margin). The covariate and dynamic
variants also check their observable prevalence effects (rating difference and
time-trend sign) agree with R at least as often as R agrees with itself.

This runs in CI WITHOUT Rscript: the reference fit is frozen in the committed
gold, so no R toolchain is touched at test time.
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import keyatm_gold  # noqa: E402


def test_keyatm_gold_present():
    npz, js = harness.gold_paths("keyatm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/keyatm_gold.py --regenerate` "
        "(needs R + the keyATM package)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_keyatm_base_matches_committed_gold():
    r = keyatm_gold.run(verbose=False)
    b = r["base"]
    assert b["passes"], (
        f"topica KeyATM base keyword cosine {b['keyword_cosine']:.4f} below bar "
        f"{b['bar']:.4f} (R self {b['keyword_r_self_cosine']:.4f} - margin); details: {b}"
    )


def test_keyatm_covariate_matches_committed_gold():
    """Covariate model — the #270 fix. Locks both the keyword-phi alignment and
    the rating-effect sign agreement against R `keyATM`."""
    r = keyatm_gold.run(verbose=False)
    c = r["covariate"]
    assert c["passes"], (
        f"topica KeyATM covariate keyword cosine {c['keyword_cosine']:.4f} "
        f"(bar {c['bar']:.4f}) / rating-sign agree {c['rating_sign_agree']:.2f} "
        f"(R self {c['rating_sign_r_self']:.2f}); details: {c}"
    )


def test_keyatm_dynamic_matches_committed_gold():
    """Dynamic change-point HMM — locks keyword phi and prevalence trends
    against R `keyATM`'s own two-seed reproducibility."""
    r = keyatm_gold.run(verbose=False)
    d = r["dynamic"]
    assert d["passes"], (
        f"topica KeyATM dynamic keyword cosine {d['keyword_cosine']:.4f} "
        f"(bar {d['bar']:.4f}) / trend-sign agree {d['trend_sign_agree']:.2f} "
        f"(R self {d['trend_sign_r_self']:.2f}); details: {d}"
    )


def test_keyatm_covariate_magnitude_parity():
    """#716-#4: beyond the rating-effect SIGN, the gold gates the MAGNITUDE of the
    covariate coefficients λ (topica MAP vs R posterior-mean, correlation on the
    keyword topics) and the keyword-switch π."""
    import pytest

    arrays, _ = harness.load_gold("keyatm")
    if "cov_lambda_r" not in arrays:
        pytest.skip("committed gold predates the λ/π magnitude arrays; regenerate")
    c = keyatm_gold.run(verbose=False)["covariate"]
    assert c["lambda_corr"] >= keyatm_gold.LAMBDA_BAR, (
        f"covariate λ magnitude correlation {c['lambda_corr']:.3f} < "
        f"{keyatm_gold.LAMBDA_BAR}; details: {c}")
    assert c["pi_corr"] >= keyatm_gold.PI_BAR, (
        f"keyword-switch π correlation {c['pi_corr']:.3f} < {keyatm_gold.PI_BAR}; {c}")


def test_keyatm_dynamic_state_path_parity():
    """#716-#4: the dynamic gold gates the HMM state path — topica's per-period
    state vs R's, label-invariant (adjusted Rand index on the change-point
    structure), not just the prevalence-trend sign."""
    import pytest

    arrays, _ = harness.load_gold("keyatm")
    if "dyn_state_r" not in arrays:
        pytest.skip("committed gold predates the state-path array; regenerate")
    d = keyatm_gold.run(verbose=False)["dynamic"]
    assert d["state_ari"] >= keyatm_gold.STATE_ARI_BAR, (
        f"dynamic HMM state-path ARI {d['state_ari']:.3f} < "
        f"{keyatm_gold.STATE_ARI_BAR}; details: {d}")


def test_keyatm_magnitude_and_state_gates_are_non_vacuous():
    """The λ / π / state-path bars must FAIL on a mismatched pairing — otherwise
    they rubber-stamp any fit. Permute the reference's topics/periods and confirm
    each metric collapses below its bar."""
    import numpy as np
    import pytest

    arrays, meta = harness.load_gold("keyatm")
    if "cov_lambda_r" not in arrays:
        pytest.skip("committed gold predates the magnitude/state arrays; regenerate")
    cm = meta["models"]["covariate"]
    nk = int(cm["num_keyword"])
    rng = np.random.default_rng(0)

    # λ / π: with only a handful of keyword topics a single permutation is noisy,
    # so average |correlation| over many derangements — a genuine mismatch must
    # sit well below the bar on average (not accidentally clear it).
    lam = np.asarray(arrays["cov_lambda_r"])[:nk]
    pi = np.asarray(arrays["cov_pi_r"])[:nk]

    def _mean_shuffled_corr(x):
        vals = []
        for _ in range(200):
            p = rng.permutation(nk)
            vals.append(abs(keyatm_gold._corr(x, x[p])))
        return float(np.nanmean(vals))

    assert _mean_shuffled_corr(lam) < keyatm_gold.LAMBDA_BAR
    assert _mean_shuffled_corr(pi) < keyatm_gold.PI_BAR

    # state path: a shuffled period assignment must drop the ARI below its bar.
    st = np.asarray(arrays["dyn_state_r"]).astype(int)
    aris = []
    for _ in range(200):
        aris.append(harness.adjusted_rand_index(st, st[rng.permutation(len(st))]))
    assert float(np.mean(aris)) < keyatm_gold.STATE_ARI_BAR


def test_keyatm_gold_is_non_vacuous():
    """A shuffled keyword-topic phi must FALL BELOW the bar for both variants —
    proving the gate discriminates a correct fit from a wrong one."""
    import numpy as np

    arrays, meta = harness.load_gold("keyatm")
    models = meta["models"]
    margin = float(meta["margin"])
    rng = np.random.default_rng(0)

    for key, prefix in (("base", "base"), ("covariate", "cov"), ("dynamic", "dyn")):
        m = models[key]
        phi1 = arrays[f"{prefix}_phi1"]
        nk = int(m["num_keyword"])
        kw = phi1[:nk]
        bar = float(m["keyword_r_self_cosine"]) - margin
        shuffled = kw[:, rng.permutation(kw.shape[1])]
        cos = keyatm_gold.base_live._best_alignment_cosine(kw, shuffled)
        assert cos < bar, (
            f"{key}: shuffled keyword phi cosine {cos:.4f} should be below the bar "
            f"{bar:.4f}; the gate is vacuous"
        )
