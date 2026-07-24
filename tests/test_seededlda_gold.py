"""Offline R-`seededlda` gold test for topica SeededLDA (#456).

SeededLDA now has a real external reference: the koheiw/`seededlda` R package.
This loads the committed gold (``parity/seededlda_gold.npz`` + ``.json``) — R's
seeded topic-word phi at two seeds and R's exact ``tfm`` seed matrix — refits
topica with the reference-faithful default (``seed_prior="frequency"``), and
asserts (1) topica's ``seed_prior_matrix`` reproduces R's ``tfm`` construction
(``count * weight * 100``) EXACTLY, and (2) topica's seeded-topic phi clears R's
own two-seed cosine floor (minus a margin). The shuffle check proves the cosine
gate is non-vacuous.

Runs in CI WITHOUT Rscript: the reference fit is frozen in the committed gold.
"""

import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import seededlda_gold  # noqa: E402

NAME = "seededlda"


def test_seededlda_gold_present():
    npz, js = harness.gold_paths(NAME)
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/seededlda_gold.py --regenerate`"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_seededlda_matches_committed_gold():
    # (1) topica's seed-mass construction reproduces R's tfm exactly; (2) topica's
    # seeded-topic phi clears R's own two-seed cosine floor minus the margin.
    r = seededlda_gold.run(verbose=False)
    assert r["tfm_exact"], (
        f"topica seed_prior_matrix does not reproduce R tfm (max |Δ| = "
        f"{r['tfm_max_abs_diff']:.2e}); the count*weight*100 seed-mass construction "
        f"has drifted from the seededlda package"
    )
    assert r["passes"], (
        f"topica SeededLDA keyword cosine {r['keyword_cosine']:.4f} below bar "
        f"{r['bar']:.4f} (R self {r['keyword_r_self_cosine']:.4f}); details: {r}"
    )


def test_seededlda_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FALL BELOW the cosine bar (and lose its
    top-word overlap), proving the gate discriminates a correct fit from a wrong
    one."""
    arrays, meta = harness.load_gold(NAME)
    n_seeded = int(meta["num_seeded"])
    r_phi = arrays["phi1"][:n_seeded]
    bar = float(meta["keyword_r_self_cosine"]) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = r_phi[:, rng.permutation(r_phi.shape[1])]
    cos, _ = harness.align_cosine(r_phi, shuffled)
    jacc = harness.top_word_jaccard(r_phi, shuffled, n=5)
    assert cos < bar, (
        f"shuffled SeededLDA keyword cosine {cos:.4f} should be below the bar "
        f"{bar:.4f}; the gate is vacuous"
    )
    assert jacc < 0.5, (
        f"shuffled SeededLDA top-word jaccard {jacc:.4f} too high; gate is weak"
    )
