"""Cross-implementation validation of the STM content model against R `stm`.

Guards the content path (where the topic-collapse bug lived) against the
reference implementation: both engines must SEPARATE the two topics, and their
per-group word distributions must agree. Skips when Rscript / `stm` is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import stm_content_r_compare  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not stm_content_r_compare.r_stm_available(),
    reason="Rscript with the 'stm' package not available",
)
def test_content_model_matches_r_stm():
    r = stm_content_r_compare.run(verbose=False)
    for g in r["cosine"]:
        # Both engines must separate the topics (collapse would be ~1.0).
        assert r["r_topic_sep"][g] < 0.5, r
        assert r["tt_topic_sep"][g] < 0.5, r
        # Per-group word distributions agree with R (default L2 fit).
        assert r["cosine"][g] > 0.7, r


@pytest.mark.skipif(
    not stm_content_r_compare.r_stm_available(),
    reason="Rscript with the 'stm' package not available",
)
def test_content_matched_l1_prior_tracks_r_separation():
    """#715-#3: R defaults to kappa.prior="L1"; topica defaults to L2. Under a
    matched L1 prior topica's topic-separation lines up with R's (both sparse and
    low), where the default L2 fit sits higher — proving the L1 content path is
    faithful and the default gap is the prior, not the inference."""
    r = stm_content_r_compare.run(verbose=False)
    for g in r["cosine_l1"]:
        # Matched L1 tracks R's low separation as closely as R matches itself,
        # and clearly closer than the default L2 fit does.
        assert r["tt_topic_sep_l1"][g] < 0.5, r
        assert abs(r["tt_topic_sep_l1"][g] - r["r_topic_sep"][g]) < 0.05, r
        assert r["tt_topic_sep_l1"][g] <= r["tt_topic_sep"][g], r
        assert r["cosine_l1"][g] > 0.7, r
