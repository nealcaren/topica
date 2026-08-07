"""Committed-gold parity for topica's HDBSCAN* MST against *every* MST algorithm
the reference ``hdbscan`` package offers (issue #603, follow-up to #555/#602).

WHAT THIS SETTLES. #603 asked whether topica should grow a Boruvka MST path so
``topica.BERTopic`` matches the ``bertopic`` package's default clusterer on
tie-heavy data, where the reference disagrees *with itself*: ``generic`` and
``prims_kdtree`` find 2 topics on the frozen 20NG/MiniLM hard case at
``min_cluster_size=min_samples=15``, while the default ``boruvka_kdtree`` finds 13.
The issue's hypothesis was MST tie-breaking under equal edge weights.

That hypothesis is wrong, and this gold is the evidence. The reference's default
is ``approx_min_span_tree=True``, and with that flag Boruvka deliberately does not
reset its dual-tree traversal bounds unless a round joins no components
(``hdbscan/_hdbscan_boruvka.pyx``, "This doesn't produce a true min spanning tree,
but only an approximation"). The tree it returns is **not minimal**: on the frozen
hard case at ``min_samples=15`` it weighs 767.049 against the exact 766.384. Run
the same Boruvka with ``approx_min_span_tree=False`` and it returns the exact MST
and 2 topics, agreeing with ``generic``, ``prims_kdtree``, and topica.

So there are no ties to break, and there is no third valid MST to match: topica
already reproduces every *exact* MST path of the reference, Boruvka included. The
residual difference against the ``bertopic`` package is the reference default's
approximation, not a topica clustering bug — which is why topica does not ship a
Boruvka path. This gold pins that conclusion so it cannot silently rot.

Two phases (mirrors ``parity/bertopic_gold.py``):

  * ``--regenerate`` (needs hdbscan + scikit-learn): runs the reference's four MST
    paths over the frozen points from ``parity/hdbscan_hardcase_555.npz`` and
    freezes their labels, cluster/noise counts, and total MST weights into
    ``parity/hdbscan_mst_parity_603.npz`` + ``.json``. The points are NOT copied —
    they stay in the #555 fixture, which this gold reads.
  * default (no hdbscan / scikit-learn): loads the committed gold, reruns topica's
    clusterer on the same points, and checks it matches the exact paths while the
    approximate default is a strictly heavier tree.

Run directly::

    python parity/hdbscan_mst_parity_603.py               # offline compare
    python parity/hdbscan_mst_parity_603.py --regenerate  # run the reference, write the gold

The offline assertions also run in CI as ``tests/test_hdbscan_boruvka_parity.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POINTS_FIXTURE = HERE / "hdbscan_hardcase_555.npz"
GOLD_NPZ = HERE / "hdbscan_mst_parity_603.npz"
GOLD_JSON = HERE / "hdbscan_mst_parity_603.json"

# min_cluster_size values frozen in the #555 fixture; min_samples == min_cluster_size.
MCS_VALUES = (10, 15, 20)

# (gold key, reference `algorithm=`, approx_min_span_tree)
VARIANTS = (
    ("generic", "generic", False),
    ("prims", "prims_kdtree", False),
    ("boruvka_exact", "boruvka_kdtree", False),
    # The reference's default: algorithm="best" picks boruvka_kdtree, and
    # HDBSCAN(approx_min_span_tree=True) is the default. This is what the
    # `bertopic` package runs.
    ("boruvka_approx", "boruvka_kdtree", True),
)


def load_points() -> np.ndarray:
    if not POINTS_FIXTURE.exists():
        raise SystemExit(f"missing points fixture: {POINTS_FIXTURE}")
    return np.load(POINTS_FIXTURE)["points"].astype(np.float64)


def num_clusters(labels) -> int:
    return len({int(x) for x in np.asarray(labels) if x >= 0})


def adjusted_rand(a, b) -> float:
    """Adjusted Rand Index, pure numpy (no scikit-learn import in CI)."""
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


def regenerate() -> None:
    from hdbscan import HDBSCAN  # only needed at --regenerate time

    points = load_points()
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for key, algorithm, approx in VARIANTS:
        for mcs in MCS_VALUES:
            model = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=mcs,
                algorithm=algorithm,
                approx_min_span_tree=approx,
                gen_min_span_tree=True,
            ).fit(points)
            labels = np.asarray(model.labels_, dtype=np.int64)
            weight = float(model.minimum_spanning_tree_.to_numpy()[:, 2].sum())
            arrays[f"{key}_labels_mcs{mcs}"] = labels
            arrays[f"{key}_mst_weight_mcs{mcs}"] = np.array(weight, dtype=np.float64)
            summary[f"{key}_mcs{mcs}"] = {
                "algorithm": algorithm,
                "approx_min_span_tree": approx,
                "n_clusters": num_clusters(labels),
                "n_noise": int((labels < 0).sum()),
                "mst_total_weight": weight,
            }
            print(
                f"  {key:<15} mcs={mcs:<3} k={num_clusters(labels):<3} "
                f"noise={int((labels < 0).sum()):<5} mst_weight={weight:.6f}"
            )

    # Sanity: the reference's bare default really is the approximate Boruvka path.
    for mcs in MCS_VALUES:
        default = np.asarray(
            HDBSCAN(min_cluster_size=mcs, min_samples=mcs).fit_predict(points),
            dtype=np.int64,
        )
        got = arrays[f"boruvka_approx_labels_mcs{mcs}"]
        assert np.array_equal(default, got), (
            f"mcs={mcs}: HDBSCAN() default labels differ from the explicit "
            "boruvka_kdtree + approx_min_span_tree=True run"
        )

    np.savez_compressed(GOLD_NPZ, **arrays)
    GOLD_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLD_NPZ} and {GOLD_JSON}")


def compare() -> None:
    from topica._topica import _hdbscan_labels_debug

    points = load_points()
    gold = np.load(GOLD_NPZ)

    for mcs in MCS_VALUES:
        got = np.asarray(_hdbscan_labels_debug(points.tolist(), mcs, mcs))
        exact_w = float(gold[f"generic_mst_weight_mcs{mcs}"])
        approx_w = float(gold[f"boruvka_approx_mst_weight_mcs{mcs}"])

        print(f"min_cluster_size = min_samples = {mcs}")
        print(f"  topica                k={num_clusters(got):<3}")
        for key, _algorithm, _approx in VARIANTS:
            ref = gold[f"{key}_labels_mcs{mcs}"]
            w = float(gold[f"{key}_mst_weight_mcs{mcs}"])
            print(
                f"  {key:<21} k={num_clusters(ref):<3} "
                f"mst_weight={w:.6f}  ARI(topica) = {adjusted_rand(ref, got):.4f}"
            )

        for key in ("generic", "prims", "boruvka_exact"):
            ref = gold[f"{key}_labels_mcs{mcs}"]
            ari = adjusted_rand(ref, got)
            assert num_clusters(got) == num_clusters(ref), (
                f"mcs={mcs}: topica found {num_clusters(got)} clusters, "
                f"reference {key} found {num_clusters(ref)}"
            )
            assert ari >= 0.98, f"mcs={mcs}: ARI vs reference {key} = {ari:.3f} (< 0.98)"
            # The exact paths agree on the MST weight to summation-order noise
            # (~1e-13 relative); only the approximate default is really heavier.
            w = float(gold[f"{key}_mst_weight_mcs{mcs}"])
            assert abs(w - exact_w) <= 1e-9 * exact_w, (
                f"mcs={mcs}: reference {key} MST weight {w:.9f} differs from "
                f"generic's {exact_w:.9f} by more than summation noise"
            )
        assert approx_w > exact_w * (1.0 + 1e-6), (
            f"mcs={mcs}: the approximate default tree ({approx_w:.6f}) is not heavier "
            f"than the exact MST ({exact_w:.6f})"
        )
    print("OK: topica matches every exact reference MST path; the default is approximate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="run the reference hdbscan package and rewrite the committed gold",
    )
    args = parser.parse_args()
    if args.regenerate:
        regenerate()
    else:
        compare()
