"""IdealPointSentenceTM: continuous ideal-point model over embeddings. EXPERIMENTAL.

It has a doc_topic distribution (soft cluster assignment), so it is covered by the
topic-health invariants; these tests check the ideal-point head: recovering planted
positions from embeddings, anchor orientation, determinism, and round-trip.
"""
import numpy as np
import pytest

import topica


@pytest.fixture(autouse=True)
def _experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    yield
    topica.enable_experimental(was)


def _planted(n_authors=40, dim=8, obs_per=20, seed=0):
    """Two Gaussian clusters; topic 0's centroid shifts along the author position."""
    rng = np.random.default_rng(seed)
    mu = np.zeros((2, dim))
    mu[0, 0] = 3.0
    mu[1, 1] = 3.0
    v = np.zeros((2, dim))
    v[0, 2] = 2.0
    v[0, 3] = -2.0  # topic 0 discriminates
    theta = rng.uniform(-1.0, 1.0, n_authors)
    emb, group = [], []
    for a in range(n_authors):
        for _ in range(obs_per):
            t = rng.integers(0, 2)
            mean = mu[t] + theta[a] * v[t]
            emb.append(mean + rng.normal(0, 0.5, dim))
            group.append(f"a{a}")
    return np.array(emb), group, theta


def test_requires_experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(False)
    try:
        with pytest.raises(Exception):
            topica.IdealPointSentenceTM(num_topics=2)
    finally:
        topica.enable_experimental(was)


def test_recovers_positions():
    emb, group, theta = _planted(seed=1)
    m = topica.IdealPointSentenceTM(num_topics=2, num_dims=1, seed=1)
    m.fit(emb, group=group, anchors={"a0": -1.0, "a39": 1.0}, iters=80)

    assert m.num_authors == 40
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    recovered = np.array([pos[f"a{a}"] for a in range(40)])
    r = abs(np.corrcoef(recovered, theta)[0, 1])
    assert r > 0.8, f"position recovery r={r:.3f}"
    assert abs(recovered.mean()) < 1e-6


def test_shapes_and_topics():
    emb, group, _ = _planted(seed=2)
    m = topica.IdealPointSentenceTM(num_topics=2, seed=1)
    m.fit(emb, group=group, iters=50)
    assert m.doc_topic.shape == (emb.shape[0], 2)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert m.topic_centroids.shape == (2, emb.shape[1])
    assert m.topic_discrimination.shape == (2,)
    c = m.position_centroid(0, [1.0])
    assert c.shape == (emb.shape[1],)


def test_position_se_is_well_shaped_and_shrinks_with_data():
    # The position SE is the exact Laplace posterior SE of the linear-Gaussian
    # position system: aligned to author_positions, finite/positive, capped by the
    # prior SD (sqrt(x_prior_variance)=1), and smaller for authors with more
    # observations. Author a0 is made data-rich by replicating its observations.
    emb, group, _ = _planted(seed=6)
    emb = np.asarray(emb)
    a0 = [i for i, g in enumerate(group) if g == "a0"]
    emb = np.concatenate([emb] + [emb[a0]] * 6, axis=0)
    group = list(group) + ["a0"] * (len(a0) * 6)

    m = topica.IdealPointSentenceTM(num_topics=2, num_dims=1, seed=1)
    m.fit(emb, group=group, iters=50)
    se = m.position_se
    assert se.shape == m.author_positions.shape == (m.num_authors, 1)
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)
    assert np.all(se <= 1.0 + 1e-9)
    names = list(m.author_names)
    rich = names.index("a0")
    others = [i for i in range(len(names)) if i != rich]
    assert se[rich, 0] < np.median(se[others, 0])


def test_position_se_survives_save_load(tmp_path):
    emb, group, _ = _planted(seed=7)
    m = topica.IdealPointSentenceTM(num_topics=2, num_dims=1, seed=1)
    m.fit(emb, group=group, iters=40)
    path = tmp_path / "s.topica"
    m.save(str(path))
    m2 = topica.IdealPointSentenceTM.load(str(path))
    assert np.array_equal(m.position_se, m2.position_se)


def test_anchors_orient_sign():
    emb, group, _ = _planted(seed=3)
    m = topica.IdealPointSentenceTM(num_topics=2, seed=1)
    m.fit(emb, group=group, anchors={"a0": -1.0, "a39": 1.0})
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    assert pos["a0"] < pos["a39"]


def test_determinism():
    emb, group, _ = _planted(seed=4)
    a = topica.IdealPointSentenceTM(num_topics=2, seed=1)
    a.fit(emb, group=group, iters=40)
    b = topica.IdealPointSentenceTM(num_topics=2, seed=1)
    b.fit(emb, group=group, iters=40)
    assert np.array_equal(a.author_positions, b.author_positions)


def test_ll_history_is_monotone(): # noqa: D103
    # Regression for #499: the reported per-iteration log-likelihood must include
    # the sigma^2-dependent Gaussian normalizer -(D/2) ln(2 pi sigma^2). sigma^2 is
    # re-estimated every sweep (it shrinks sharply early), so without the normalizer
    # the reported ll_history is true_ll plus a per-iteration-varying offset and can
    # DECREASE while the model improves. With it, ll_history is the actual EM
    # objective and is monotone non-decreasing across iterations.
    emb, group, _ = _planted(seed=7)
    m = topica.IdealPointSentenceTM(num_topics=2, num_dims=1, seed=1)
    m.fit(emb, group=group, iters=60)
    hist = m.fit_history
    lls = [ll for _, ll in hist]
    assert len(lls) >= 3
    # Every EM step must not decrease the reported log-likelihood (tiny tolerance
    # for f64 rounding in the parallel reductions).
    diffs = np.diff(lls)
    assert np.all(diffs >= -1e-6), (
        f"ll_history not monotone non-decreasing; min step = {diffs.min():.3e}"
    )


def test_save_load(tmp_path):
    emb, group, _ = _planted(seed=5)
    m = topica.IdealPointSentenceTM(num_topics=2, seed=1)
    m.fit(emb, group=group, anchors={"a0": -1.0, "a39": 1.0})
    p = tmp_path / "sitm.topica"
    m.save(str(p))
    m2 = topica.IdealPointSentenceTM.load(str(p))
    assert np.array_equal(m.author_positions, m2.author_positions)
    assert np.array_equal(m.topic_centroids, m2.topic_centroids)
    assert m.author_names == m2.author_names
