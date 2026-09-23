"""Tests for topica.evaluate.topic_groups, the TopicCheck-style N-run group table
(issue #803)."""

import numpy as np
import pytest

import topica
from topica.evaluate import TopicGroups, topic_groups


def _planted(k=4, v=40, seed=0):
    """K topics on disjoint word blocks, row-normalized."""
    rng = np.random.default_rng(seed)
    beta = np.full((k, v), 1e-3)
    block = v // k
    for t in range(k):
        beta[t, t * block:(t + 1) * block] += rng.random(block) + 0.5
    return beta / beta.sum(axis=1, keepdims=True)


def test_permuted_copies_form_solid_groups():
    """Runs that are row-permutations of one another group into K solid groups, and
    the assignment grid recovers each permutation."""
    beta = _planted()
    perms = [np.arange(4), np.array([2, 0, 3, 1]), np.array([3, 2, 1, 0])]
    runs = [beta[p] for p in perms]
    g = topic_groups(runs, threshold=0.9)
    assert isinstance(g, TopicGroups)
    assert len(g.groups) == 4
    assert np.all(g.solidity == 1.0)
    grid = g.assignments()
    for _, row in grid.iterrows():
        # every run's topic in this group is the same underlying planted topic
        originals = {int(perms[r][int(row[f"run {r}"])]) for r in range(3)}
        assert len(originals) == 1


def test_up_to_one_topic_per_run():
    """A run that splits one theme into two near-identical topics contributes only
    one of them to the theme's group (TopicCheck's up-to-one constraint)."""
    beta = _planted()
    split = beta.copy()
    split[3] = 0.98 * beta[0] + 0.02 * beta[3]   # a near-duplicate of topic 0
    split /= split.sum(axis=1, keepdims=True)
    g = topic_groups([beta, beta.copy(), split], threshold=0.5)
    for grp in g.groups:
        assert len(grp["members"]) == len(set(grp["members"]))  # one topic per run
        assert grp["solidity"] <= 1.0
    # the duplicate is left in a group of its own rather than merged with topic 0
    solo = [grp for grp in g.groups if grp["members"] == {2: 3}]
    assert len(solo) == 1 and solo[0]["solidity"] == pytest.approx(1 / 3)


def test_cut_matches_a_fresh_fit_and_is_monotone():
    rng = np.random.default_rng(1)
    runs = [_planted(seed=s) + rng.random((4, 40)) * 0.02 for s in range(4)]
    g = topic_groups(runs, threshold=0.5)
    counts = []
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        cut = g.cut(t)
        fresh = topic_groups(runs, threshold=t)
        assert [x["members"] for x in cut.groups] == [x["members"] for x in fresh.groups]
        counts.append(len(cut.groups))
    assert counts == sorted(counts)  # higher threshold, more (tighter) groups
    sims = g.merge_similarities
    assert np.all(np.diff(sims) <= 1e-12)  # merge similarities never increase


def test_runs_may_differ_in_k():
    beta = _planted(k=4)
    runs = [beta, beta[:3], beta[[1, 2]]]
    g = topic_groups(runs, threshold=0.9)
    sizes = sorted(len(x["members"]) for x in g.groups)
    assert sizes == [1, 2, 3, 3]
    assert g.assignments().shape == (4, 3)


def test_js_metric():
    beta = _planted()
    g = topic_groups([beta, beta[::-1].copy()], threshold=0.9, metric="js")
    assert len(g.groups) == 4 and np.all(g.solidity == 1.0)


def test_fitted_models_give_words_and_weights(toy_corpus):
    runs = [topica.LDA(3, seed=s).fit(toy_corpus, iters=100) for s in (1, 2, 3)]
    g = topic_groups(runs, threshold=0.5)
    words = g.top_words(5)
    assert all(isinstance(w, str) for w in words[0])
    weights = [x["weight"] for x in g.groups]
    assert all(w is not None and 0.0 <= w <= 1.0 for w in weights)
    # the weights of all groups sum to one: every run's prevalence is split across groups
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    frame = g.to_frame()
    assert list(frame.columns) == ["solidity", "runs", "weight", "similarity", "top_words", "members"]
    assert "| group | runs | solidity | top words |" in g.to_markdown()
    assert str(g).startswith("TopicGroups:")


def test_validation():
    beta = _planted()
    with pytest.raises(ValueError, match="at least two runs"):
        topic_groups([beta])
    with pytest.raises(ValueError, match="one vocabulary"):
        topic_groups([beta, beta[:, :30]])
    with pytest.raises(ValueError, match="threshold"):
        topic_groups([beta, beta], threshold=1.5)
    with pytest.raises(ValueError, match="metric"):
        topic_groups([beta, beta], metric="euclid")


def _reference_merges(dist, run_of):
    """The plain O(T^3) constrained average linkage: full-matrix argmin every merge."""
    n = dist.shape[0]
    D = dist.astype(float).copy()
    D[run_of[:, None] == run_of[None, :]] = np.inf
    np.fill_diagonal(D, np.inf)
    runs = [{int(r)} for r in run_of]
    sizes = np.ones(n)
    alive = np.ones(n, dtype=bool)
    out = []
    while True:
        a, b = divmod(int(np.argmin(D)), n)
        d = D[a, b]
        if not np.isfinite(d):
            return out
        row = (sizes[a] * D[a] + sizes[b] * D[b]) / (sizes[a] + sizes[b])
        runs[a] |= runs[b]
        for c in np.flatnonzero(alive):
            if c != a and runs[c] & runs[a]:
                row[c] = np.inf
        row[~alive] = np.inf
        row[a] = np.inf
        D[a, :] = D[:, a] = row
        D[b, :] = D[:, b] = np.inf
        alive[b] = False
        sizes[a] += sizes[b]
        out.append((a, b, float(d)))


def test_cached_merges_match_the_full_scan_reference():
    """The row-minimum cache must reproduce the plain full-scan merge sequence exactly,
    ties included (quantized rows create many equal distances)."""
    from topica._topic_groups import _constrained_merges, _similarity

    rng = np.random.default_rng(7)
    for trial in range(120):
        ks = rng.integers(1, 6, size=rng.integers(2, 6))
        if trial % 2:
            betas = [rng.integers(0, 3, size=(k, 10)).astype(float) + 1e-9 for k in ks]
        else:
            betas = [rng.random((k, 10)) for k in ks]
        pooled = np.vstack(betas)
        run_of = np.concatenate([np.full(k, r) for r, k in enumerate(ks)])
        dist = 1.0 - _similarity(pooled, "cosine")
        assert _constrained_merges(dist, run_of) == _reference_merges(dist, run_of)


def test_flat_alias_is_the_function():
    assert callable(topica.topic_groups)
    assert topica.topic_groups is topica.evaluate.topic_groups


def test_rejects_nan_and_negative_input():
    beta = _planted()
    bad = beta.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        topic_groups([beta, bad])
    with pytest.raises(ValueError, match="finite and non-negative"):
        topic_groups([beta, -beta])


def test_mixed_prevalence_warns(toy_corpus):
    fitted = topica.LDA(3, seed=1).fit(toy_corpus, iters=50)
    raw = np.asarray(fitted.topic_word)
    with pytest.warns(UserWarning, match="no document-topic matrix"):
        g = topic_groups([fitted, raw])
    assert all(x["weight"] is None for x in g.groups)
