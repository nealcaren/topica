"""HDBSCAN clustering parity: topica (petal-clustering) vs the reference `hdbscan`
package on a frozen embedding fixture (#489, last open item).

Isolates the density-clustering step from UMAP so a real difference in the
clusterer would show as a partition disagreement, not reducer noise. On the
frozen well-separated fixture the two implementations must produce the same
partition (ARI == 1) and recover the planted truth equally well. Skips cleanly
when the reference `hdbscan` package is unavailable.

See ``parity/top2vec_hdbscan_parity.py`` for the overlapping-regime sweep, where
the two diverge by border-point / cluster-selection variance (petal's
bcubed+stability vs the reference's excess-of-mass) rather than any topica defect.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import top2vec_hdbscan_parity as hp  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not hp.hdbscan_available(),
    reason="reference `hdbscan` package not installed",
)
def test_topica_hdbscan_matches_reference():
    m = hp.run(mcs=15, verbose=False)

    # Same discovered topic count on the well-separated fixture.
    assert m["topica_num_topics"] == m["ref_num_topics"] == hp.N_CLUSTERS, m
    # The two clusterers produce the same partition on separable structure.
    assert m["cross_ari"] >= 0.98, m
    # Both recover the planted truth, and neither is meaningfully worse.
    assert m["topica_truth_ari"] >= 0.98, m
    assert abs(m["topica_truth_ari"] - m["ref_truth_ari"]) <= 0.02, m
