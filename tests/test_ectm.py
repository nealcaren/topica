"""Behavior tests for ECTM (Evolving Content Topic Model)."""
import numpy as np
import pytest

import topica
from topica.ectm import content_contrast, content_divergence, content_trajectory, content_words


def _corpus(reps=60, drift=True):
    """Two groups over three periods, vocab {a,b,x,y}. Group A always uses {a,b}.
    Group B starts on {a,b} and (if drift) moves onto {x,y} by the last period —
    a group-by-time content contrast that grows. With drift=False both groups
    use {a,b} every period (no contrast)."""
    docs, groups, times = [], [], []
    if drift:
        gb = {0: ["a", "b", "a", "b"], 1: ["a", "b", "x", "y"], 2: ["x", "y", "x", "y"]}
    else:
        gb = {0: ["a", "b", "a", "b"], 1: ["a", "b", "a", "b"], 2: ["a", "b", "a", "b"]}
    for _ in range(reps):
        for per in range(3):
            docs.append(["a", "b", "a", "b"]); groups.append("A"); times.append(2000 + per)
            docs.append(gb[per]); groups.append("B"); times.append(2000 + per)
    return docs, groups, times


def _fit(seed=1, drift=True, **kw):
    docs, groups, times = _corpus(drift=drift)
    m = topica.ECTM(num_topics=2, seed=seed)
    m.fit(docs, times=times, content=groups, iters=60,
          period_smooth=5.0, interaction_shrink=2.0, **kw)
    return m


# --- The four idioms -------------------------------------------------------

def test_shapes_and_normalization():
    m = _fit()
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape == (360, 2)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    assert m.num_groups == 2 and m.num_periods == 3
    assert m.groups == ["A", "B"]
    assert m.periods == ["2000", "2001", "2002"]
    # per-cell content distributions are normalized
    for g in range(2):
        for t in range(3):
            cw = m.content_word_dist(g, t)
            assert cw.shape == (2, len(m.vocabulary))
            np.testing.assert_allclose(cw.sum(axis=1), 1.0, atol=1e-9)


def test_determinism():
    a, b = _fit(seed=3), _fit(seed=3)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.content_word_dist(1, 2), b.content_word_dist(1, 2))
    c = _fit(seed=4)
    assert not np.array_equal(a.content_word_dist(1, 2), c.content_word_dist(1, 2))


def test_save_load_roundtrip(tmp_path):
    m = _fit()
    p = str(tmp_path / "m.tt")
    m.save(p)
    loaded = topica.ECTM.load(p)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert loaded.periods == m.periods and loaded.groups == m.groups
    assert np.array_equal(m.content_word_dist("B", 2), loaded.content_word_dist("B", 2))


def test_bad_params():
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=1)
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2, sigma_shrink=2.0)
    docs, groups, times = _corpus(reps=4)
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2).fit(docs, times=times, content=groups, interaction_shrink=0.0)
    # times / content length mismatch
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2).fit(docs, times=times[:-1], content=groups)


# --- Synthetic recovery scenarios (from ECTM.md) ---------------------------

def test_recovers_growing_contrast():
    """Scenario 4: a group difference that grows over time."""
    m = _fit(drift=True)
    vocab = m.vocabulary
    xi, yi = vocab.index("x"), vocab.index("y")
    # topic where B carries the {x,y} mass in the last period
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, xi] + m.content_word_dist("B", 2)[k, yi])

    def gap(per):
        b = m.content_word_dist("B", per)[k, xi] + m.content_word_dist("B", per)[k, yi]
        a = m.content_word_dist("A", per)[k, xi] + m.content_word_dist("A", per)[k, yi]
        return b - a

    assert gap(2) > gap(0) + 0.2, "B-vs-A contrast should grow across periods"
    assert gap(2) > 0.3


def test_no_contrast_when_groups_identical():
    """Scenario 1: no group differences — divergence stays small every period."""
    m = _fit(drift=False)
    for k in range(2):
        for _, dist in content_divergence(m, k, "A", "B"):
            assert dist < 0.15, "identical groups should have near-zero divergence"


def test_content_helpers_surface():
    m = _fit(drift=True)
    vocab = m.vocabulary
    xi = vocab.index("x")
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, xi])
    # content_words returns ranked (word, prob)
    cw = content_words(m, k, "B", 2, n=3)
    assert len(cw) == 3 and all(isinstance(w, str) for w, _ in cw)
    # content_contrast returns both directions
    con = content_contrast(m, k, "B", "A", 2, n=3)
    assert "toward_B" in con and "toward_A" in con
    # the growing word should head the toward-B list in the last period
    assert "x" in {w for w, _ in con["toward_B"]} or "y" in {w for w, _ in con["toward_B"]}
    # content_trajectory: the B-A contrast for x grows
    traj = content_trajectory(m, k, "x", contrast=("B", "A"))
    assert [p for p, _ in traj] == m.periods
    assert traj[-1][1] > traj[0][1]
    # content_divergence in [0,1], one per period
    div = content_divergence(m, k, "A", "B")
    assert len(div) == m.num_periods and all(0.0 <= d <= 1.0 for _, d in div)


def test_analysis_surface():
    m = _fit()
    assert topica.summary(m) is not None
    assert topica.topic_table(m) is not None
    assert m.coherence(5).shape == (2,)
