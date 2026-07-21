"""topica.agreement: external validation of a partition against gold labels.

Values are checked against hand-computable cases and metric invariants, not
against scikit-learn (CI has no sklearn); the sklearn cross-check lives in the dev
environment and is documented in the module.
"""

import numpy as np
import pytest

import topica


def test_identical_partition_scores_one():
    gold = np.array([0, 0, 1, 1, 2, 2])
    r = topica.agreement(gold.copy(), gold)
    for k in ("ari", "nmi", "homogeneity", "completeness", "v_measure", "purity"):
        assert r[k] == pytest.approx(1.0), k


def test_metrics_are_label_agnostic():
    # Relabeling the predicted clusters must not change any score.
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    relabeled = np.array([7, 7, 3, 3, 9, 9])  # same partition, different ids
    a = topica.agreement(pred, gold)
    b = topica.agreement(relabeled, gold)
    assert a == b


def test_purity_hand_computed():
    # cluster 0 = {gold 0,0,1} -> majority 2; cluster 1 = {gold 1,1} -> 2; (2+2)/5.
    pred = np.array([0, 0, 0, 1, 1])
    gold = np.array([0, 0, 1, 1, 1])
    assert topica.agreement(pred, gold)["purity"] == pytest.approx(0.8)


def test_pure_but_incomplete_split():
    # Every doc its own cluster: perfectly homogeneous, not complete.
    pred = np.array([0, 1, 2, 3])
    gold = np.array([0, 0, 1, 1])
    r = topica.agreement(pred, gold)
    assert r["homogeneity"] == pytest.approx(1.0)
    assert r["completeness"] < 1.0


def test_complete_but_inhomogeneous():
    # One giant cluster: perfectly complete, zero homogeneity.
    pred = np.array([0, 0, 0, 0])
    gold = np.array([0, 0, 1, 1])
    r = topica.agreement(pred, gold)
    assert r["completeness"] == pytest.approx(1.0)
    assert r["homogeneity"] == pytest.approx(0.0)
    assert r["v_measure"] == pytest.approx(0.0)


def test_ari_bounded_and_below_one_when_imperfect():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 5, 300)
    gold = rng.integers(0, 5, 300)
    ari = topica.agreement(pred, gold)["ari"]
    assert -0.5 <= ari < 0.2  # independent labelings sit near 0


def test_noise_keep_vs_drop():
    # Two clean clusters plus two -1 noise docs whose gold classes differ.
    pred = np.array([0, 0, 1, 1, -1, -1])
    gold = np.array([0, 0, 1, 1, 0, 1])
    keep = topica.agreement(pred, gold, noise="keep")
    drop = topica.agreement(pred, gold, noise="drop")
    # Dropping the noise leaves a perfect 2-cluster partition.
    assert drop["ari"] == pytest.approx(1.0)
    assert drop["purity"] == pytest.approx(1.0)
    # Keeping -1 as its own topic is penalized.
    assert keep["ari"] < 1.0


def test_partial_gold_via_mask():
    pred = np.array([0, 0, 1, 1, 2, 2])
    gold = np.array([0, 0, 1, 1, -1, -1])  # last two unlabeled
    mask = gold != -1
    r = topica.agreement(pred[mask], gold[mask])
    assert r["ari"] == pytest.approx(1.0)


def test_input_validation():
    with pytest.raises(ValueError, match="same length"):
        topica.agreement([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="empty"):
        topica.agreement([], [])
    with pytest.raises(ValueError, match="noise must be"):
        topica.agreement([0, 1], [0, 1], noise="ignore")
    with pytest.raises(ValueError, match="after dropping"):
        topica.agreement([-1, -1], [0, 1], noise="drop")


def test_accepts_lists_and_model_labels_shape():
    # Plain Python lists work (coerced), same as arrays.
    r = topica.agreement([0, 0, 1, 1], [1, 1, 0, 0])
    assert r["ari"] == pytest.approx(1.0)
