"""topica's in-house HDBSCAN* (`src/hdbscan.rs`, replacing petal-clustering)
reproduces the reference `hdbscan` package's exact ``generic`` algorithm on a
frozen real-world hard case (issue #555).

The fixture is a real UMAP projection of 20-Newsgroups MiniLM embeddings
(``parity/hdbscan_hardcase_555.npz``: 2594x5 points + frozen ``generic`` labels at
min_cluster_size 10/15/20). petal-clustering scored ARI 0.12-0.16 here and
collapsed BERTopic to 2 topics; the reimplementation matches ``generic`` at
ARI >= 0.98. Offline by construction: the labels are frozen, so nothing here
imports `hdbscan`/`umap`/`sklearn` (which CI lacks).
"""

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "parity" / "hdbscan_hardcase_555.npz"


def _adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index, pure numpy (avoids a scikit-learn import in CI)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]
    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)
    cont = np.zeros((a_idx.max() + 1, b_idx.max() + 1), dtype=np.int64)
    np.add.at(cont, (a_idx, b_idx), 1)
    comb2 = lambda x: (x * (x - 1)) // 2  # noqa: E731
    sum_comb = comb2(cont).sum()
    sum_a = comb2(cont.sum(axis=1)).sum()
    sum_b = comb2(cont.sum(axis=0)).sum()
    total = comb2(np.array([n], dtype=np.int64)).sum()
    expected = sum_a * sum_b / total if total else 0.0
    maxi = (sum_a + sum_b) / 2.0
    if maxi == expected:
        return 1.0
    return float((sum_comb - expected) / (maxi - expected))


def _num_clusters(labels: np.ndarray) -> int:
    return len({int(x) for x in labels if x >= 0})


@pytest.mark.skipif(not FIXTURE.exists(), reason="hardcase fixture missing")
@pytest.mark.parametrize("mcs", [10, 15, 20])
def test_hdbscan_matches_generic_on_hardcase(mcs):
    from topica._topica import _hdbscan_labels_debug

    data = np.load(FIXTURE)
    points = data["points"].astype(np.float64)
    ref = data[f"generic_labels_mcs{mcs}"]

    got = np.asarray(_hdbscan_labels_debug(points.tolist(), mcs, mcs))

    # Same discovered cluster count and a near-identical partition as the reference
    # `generic` HDBSCAN. petal-clustering scored ARI 0.12-0.16 / 2 clusters here.
    assert _num_clusters(got) == _num_clusters(ref), (
        f"mcs={mcs}: topica found {_num_clusters(got)} clusters, "
        f"reference generic found {_num_clusters(ref)}"
    )
    ari = _adjusted_rand(ref, got)
    assert ari >= 0.98, f"mcs={mcs}: ARI vs reference generic = {ari:.3f} (< 0.98)"
