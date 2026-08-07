"""topica's HDBSCAN* matches *every* exact MST path of the reference `hdbscan`
package — ``generic``, ``prims_kdtree``, and ``boruvka_kdtree`` — on the frozen
20NG/MiniLM hard case (issue #603, follow-up to #555/#602).

#603 asked for an optional Boruvka MST so ``topica.BERTopic`` would match the
``bertopic`` package's default clusterer, which finds 13 topics here where
``generic``/``prims_kdtree``/topica find 2. The hypothesis was MST tie-breaking
under equal edge weights. It isn't: the reference's default is
``approx_min_span_tree=True``, under which Boruvka skips resetting its dual-tree
bounds and returns a spanning tree that is **not minimal** (767.049 vs the exact
766.384 at ``min_samples=15``). Run Boruvka with ``approx_min_span_tree=False``
and it returns the exact MST and agrees with topica at ARI 1.00.

So there is no third valid MST to match, and topica ships no Boruvka path. These
tests pin both halves of that conclusion: topica reproduces all three exact
paths, and the remaining gap is a strictly heavier tree, not a tie-break.

Offline by construction: the reference labels and MST weights are frozen in
``parity/hdbscan_mst_parity_603.npz`` (regenerate with
``python parity/hdbscan_mst_parity_603.py --regenerate``), so nothing here imports
`hdbscan`/`sklearn`, which CI lacks.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "parity" / "hdbscan_hardcase_555.npz"
GOLD = ROOT / "parity" / "hdbscan_mst_parity_603.npz"

MCS_VALUES = [10, 15, 20]
EXACT_PATHS = ["generic", "prims", "boruvka_exact"]

pytestmark = pytest.mark.skipif(
    not (POINTS.exists() and GOLD.exists()), reason="hdbscan parity fixtures missing"
)


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


def _topica_labels(mcs: int) -> np.ndarray:
    from topica._topica import _hdbscan_labels_debug

    points = np.load(POINTS)["points"].astype(np.float64)
    return np.asarray(_hdbscan_labels_debug(points.tolist(), mcs, mcs))


@pytest.mark.parametrize("mcs", MCS_VALUES)
@pytest.mark.parametrize("path", EXACT_PATHS)
def test_matches_every_exact_reference_mst_path(path, mcs):
    """topica == the reference under each of its three exact MST algorithms."""
    ref = np.load(GOLD)[f"{path}_labels_mcs{mcs}"]
    got = _topica_labels(mcs)

    assert _num_clusters(got) == _num_clusters(ref), (
        f"mcs={mcs}: topica found {_num_clusters(got)} clusters, "
        f"reference {path} found {_num_clusters(ref)}"
    )
    ari = _adjusted_rand(ref, got)
    assert ari >= 0.98, f"mcs={mcs}: ARI vs reference {path} = {ari:.3f} (< 0.98)"


@pytest.mark.parametrize("mcs", MCS_VALUES)
def test_reference_exact_paths_agree_on_the_mst_weight(mcs):
    """generic / prims / exact-boruvka find the same-weight tree — no ties in play.

    They agree to summation-order noise (~1e-13 relative), which is what "these are
    all the same minimum spanning tree" looks like in floating point.
    """
    gold = np.load(GOLD)
    exact = float(gold[f"generic_mst_weight_mcs{mcs}"])
    for path in EXACT_PATHS:
        w = float(gold[f"{path}_mst_weight_mcs{mcs}"])
        assert abs(w - exact) <= 1e-9 * exact, (
            f"mcs={mcs}: reference {path} MST weight {w:.9f} differs from generic's "
            f"{exact:.9f} by more than summation noise"
        )


@pytest.mark.parametrize("mcs", MCS_VALUES)
def test_reference_default_tree_is_not_minimal(mcs):
    """The `bertopic` default (`approx_min_span_tree=True`) returns a heavier tree.

    This is the whole of the residual `topica.BERTopic` vs `bertopic` topic-count
    gap on tie-heavy data: an approximation the reference documents in
    `_hdbscan_boruvka.pyx`, not a different-but-equal MST.
    """
    gold = np.load(GOLD)
    exact = float(gold[f"generic_mst_weight_mcs{mcs}"])
    approx = float(gold[f"boruvka_approx_mst_weight_mcs{mcs}"])
    assert approx > exact * (1.0 + 1e-6), (
        f"mcs={mcs}: approximate default tree ({approx:.6f}) is not heavier than "
        f"the exact MST ({exact:.6f})"
    )


def test_approximation_is_what_splits_the_hard_case():
    """At mcs=15 the approximate tree alone produces the 13-topic split.

    Guards the #603 headline: exact Boruvka gives topica's 2 topics, and only the
    approximate default gives 13. If a future `hdbscan` release makes its default
    exact, this fails and the gold should be regenerated.
    """
    gold = np.load(GOLD)
    assert _num_clusters(gold["boruvka_exact_labels_mcs15"]) == 2
    assert _num_clusters(gold["boruvka_approx_labels_mcs15"]) == 13
    assert _num_clusters(_topica_labels(15)) == 2
