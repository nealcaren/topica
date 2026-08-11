"""Fold-engine tests for the cross-validation framework (#701, PR1).

Covers the Gate A guards: deterministic seeds across interpreter restarts, group
isolation + dominant-group error, temporal ordering with atomic ties, exactly-once
test coverage, oof_mask, and the per-fold invariants.
"""

import subprocess
import sys

import numpy as np
import pytest

from topica.crossval import make_folds, Folds


# --------------------------------------------------------------------------- kfold


def test_kfold_covers_every_doc_exactly_once():
    f = make_folds(50, strategy="kfold", folds=5, seed=13)
    assert len(f) == 5
    tested = np.concatenate([test for _, test in f])
    assert np.array_equal(np.sort(tested), np.arange(50))
    assert f.oof_mask.all()


def test_kfold_train_test_disjoint_and_nonempty():
    f = make_folds(37, strategy="kfold", folds=4, seed=7)
    for train, test in f:
        assert train.dtype == np.int64 and test.dtype == np.int64
        assert train.size and test.size
        assert not np.intersect1d(train, test).size
        assert train.size + test.size == 37


def test_kfold_deterministic_same_seed():
    a = make_folds(40, folds=5, seed=13)
    b = make_folds(40, folds=5, seed=13)
    for (t1, e1), (t2, e2) in zip(a, b):
        assert np.array_equal(t1, t2) and np.array_equal(e1, e2)
    assert a.fold_seeds == b.fold_seeds


def test_kfold_different_seed_differs():
    a = make_folds(40, folds=5, seed=13)
    b = make_folds(40, folds=5, seed=99)
    assert a.fold_seeds != b.fold_seeds


def test_seeds_reproducible_across_interpreter_restart():
    """The Gate A-B5 guard: SeedSequence, not hash() — stable across processes."""
    code = (
        "from topica.crossval import make_folds;"
        "f=make_folds(30, folds=5, seed=13);"
        "print(f.fold_seeds);"
        "print([e.tolist() for _,e in f])"
    )
    out1 = subprocess.check_output([sys.executable, "-c", code], text=True)
    out2 = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out1 == out2


# --------------------------------------------------------------------------- grouped


def test_grouped_isolates_groups():
    groups = np.repeat(np.arange(10), 5)  # 10 groups of 5 docs
    f = make_folds(50, strategy="grouped", folds=5, groups=groups, seed=13)
    for train, test in f:
        assert not (set(groups[train]) & set(groups[test]))
    tested = np.concatenate([test for _, test in f])
    assert np.array_equal(np.sort(tested), np.arange(50))


def test_grouped_requires_groups():
    with pytest.raises(ValueError, match="requires groups"):
        make_folds(20, strategy="grouped", folds=4)


def test_grouped_too_few_groups():
    groups = np.repeat(np.arange(3), 10)
    with pytest.raises(ValueError, match="at least folds"):
        make_folds(30, strategy="grouped", folds=5, groups=groups)


def test_grouped_dominant_group_warns_on_imbalance():
    # One group is 60% of the corpus: folds are valid (group 0 = one whole fold)
    # but heavily imbalanced, which must be surfaced loudly.
    groups = np.array([0] * 60 + list(range(1, 41)))
    with pytest.warns(UserWarning, match="imbalanced"):
        f = make_folds(100, strategy="grouped", folds=5, groups=groups)
    # Still a valid, group-disjoint, exactly-once split.
    for train, test in f:
        assert not (set(groups[train]) & set(groups[test]))
    tested = np.concatenate([test for _, test in f])
    assert np.array_equal(np.sort(tested), np.arange(100))


def test_grouped_impossible_empty_fold_errors():
    # Group 0 swallows so much that fewer than K groups remain to fill the bins.
    groups = np.array([0] * 96 + [1, 2, 3, 4])
    with pytest.raises(ValueError, match="empty fold|at least folds"):
        make_folds(100, strategy="grouped", folds=6, groups=groups)


def test_grouped_missing_group_id_errors():
    groups = np.array([0.0, 1.0, np.nan] + [2.0] * 17)
    with pytest.raises(ValueError, match="missing group"):
        make_folds(20, strategy="grouped", folds=3, groups=groups)


# --------------------------------------------------------------------------- temporal


def test_temporal_test_strictly_after_train():
    times = np.arange(60)
    f = make_folds(60, strategy="temporal", folds=5, times=times)
    for train, test in f:
        if train.size:
            assert times[train].max() < times[test].min()


def test_temporal_expanding_window_grows():
    times = np.arange(60)
    f = make_folds(60, strategy="temporal", folds=5, times=times)
    sizes = [train.size for train, _ in f]
    assert sizes == sorted(sizes)  # monotonically nondecreasing


def test_temporal_initial_window_not_tested():
    times = np.arange(60)
    f = make_folds(60, strategy="temporal", folds=5, times=times)
    # The earliest docs (block 0) are never in any test set.
    assert not f.oof_mask.all()
    tested = np.concatenate([test for _, test in f])
    assert 0 not in tested


def test_temporal_ties_stay_together():
    # Two big tie groups: timestamp 0 (30 docs) then timestamp 1..K.
    times = np.array([0] * 30 + list(range(1, 31)))
    f = make_folds(60, strategy="temporal", folds=5, times=times)
    for train, test in f:
        if train.size:
            # No shared timestamp across the boundary.
            assert times[train].max() < times[test].min()


def test_temporal_rolling_window():
    times = np.arange(60)
    exp = make_folds(60, strategy="temporal", folds=5, times=times, window=None)
    roll = make_folds(60, strategy="temporal", folds=5, times=times, window=1)
    # Rolling window trains on fewer docs in the later folds than expanding.
    assert roll[-1][0].size < exp[-1][0].size


def test_temporal_requires_times():
    with pytest.raises(ValueError, match="requires times"):
        make_folds(20, strategy="temporal", folds=4)


def test_temporal_too_few_distinct_times():
    times = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="distinct timestamps"):
        make_folds(6, strategy="temporal", folds=5, times=times)


# --------------------------------------------------------------------------- misc


def test_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        make_folds(20, strategy="bogus", folds=4)


def test_folds_exceed_docs():
    with pytest.raises(ValueError, match="exceeds n_docs"):
        make_folds(3, folds=5)


def test_repr():
    f = make_folds(50, folds=5, seed=13)
    assert "Folds(" in repr(f) and "kfold" in repr(f)
