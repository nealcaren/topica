"""Offline same-algorithm gold test for topica's collapsed-Gibbs RTM (#424).

R lda's `rtm.em` is collapsed Gibbs (it wraps `rtm.collapsed.gibbs.sampler`) — the
same algorithm as topica's `inference="gibbs"` backend — so it is an authoritative
oracle, not a directional baseline. This loads the committed gold (R lda's seeded
topic-word phi at two seeds on a fixed planted network) and asserts topica's
Gibbs refit clears R's own seed-to-seed self-consistency floor. Runs in CI without
Rscript. The shuffle check proves the gate is non-vacuous.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import rtm_gibbs_gold  # noqa: E402

NAME = "rtm_gibbs"


def test_rtm_gibbs_gold_present():
    npz, js = harness.gold_paths(NAME)
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/rtm_gibbs_gold.py --regenerate`"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_rtm_gibbs_matches_committed_gold():
    r = rtm_gibbs_gold.run(verbose=False)
    assert r["link_beta_all_negative"], (
        "RTM Gibbs link coefficient should be negative like R lda's estimate.params"
    )
    assert r["passes"], (
        f"topica Gibbs RTM topic cosine {r['topic_cosine']:.4f} below bar "
        f"{r['bar']:.4f} (R self {r['topic_r_self_cosine']:.4f}); details: {r}"
    )


def test_rtm_gibbs_gold_is_non_vacuous():
    """A shuffled topic-word matrix must fall below the cosine bar, proving the
    gate discriminates a correct fit from a wrong one."""
    arrays, meta = harness.load_gold(NAME)
    r_phi = arrays["phi1"]
    bar = float(meta["topic_r_self_cosine"]) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = r_phi[:, rng.permutation(r_phi.shape[1])]
    cos, _ = harness.align_cosine(r_phi, shuffled)
    assert cos < bar, (
        f"shuffled RTM Gibbs cosine {cos:.4f} should be below the bar {bar:.4f}; "
        f"the gate is vacuous"
    )
