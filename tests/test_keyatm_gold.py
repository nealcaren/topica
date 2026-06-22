"""Offline gold-fixture parity for topica KeyATM vs R `keyATM` (issue #271, Wave 1).

Loads the committed gold (``parity/keyatm_gold.npz`` + ``.json``), fits topica
KeyATM on the same poliblog corpus + keyword sets, aligns to R's keyword-topic
``phi``, and asserts the aligned cosine clears R's own seed-to-seed keyword-phi
noise floor (minus a small multimodality margin). The covariate variant also
checks the rating-effect sign agrees with R at least as often as R agrees with
itself — the observable that the #270 fix restored (theta no longer collapses
onto one topic).

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


def test_keyatm_gold_is_non_vacuous():
    """A shuffled keyword-topic phi must FALL BELOW the bar for both variants —
    proving the gate discriminates a correct fit from a wrong one."""
    import numpy as np

    arrays, meta = harness.load_gold("keyatm")
    models = meta["models"]
    margin = float(meta["margin"])
    rng = np.random.default_rng(0)

    for key, prefix in (("base", "base"), ("covariate", "cov")):
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
