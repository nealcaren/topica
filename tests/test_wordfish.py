"""Wordfish: the word-frequency ideal-point baseline.

Wordfish has no topic distribution, so it is exempt from the registry-driven
topic-health invariant suite; its validity is checked here: it recovers planted
positions and word discriminations from counts sampled from its own model.
"""
import math

import numpy as np

import topica


def _planted(n_authors=40, n_words=60, docs_per=3, seed=0):
    """Counts sampled from the Wordfish model, expanded to token lists."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.0, 1.0, n_authors)
    beta = np.linspace(-1.0, 1.0, n_words)
    psi = np.log(rng.uniform(3.0, 12.0, n_words))
    docs, group = [], []
    for a in range(n_authors):
        rates = np.exp(psi + beta * theta[a]) / docs_per
        for _ in range(docs_per):
            counts = rng.poisson(rates)
            doc = []
            for j, c in enumerate(counts):
                doc.extend([f"w{j}"] * int(c))
            rng.shuffle(doc)
            docs.append(doc)
            group.append(f"a{a}")
    return docs, group, theta, beta


def test_recovers_positions_and_discrimination():
    docs, group, theta, beta = _planted(seed=1)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0}, iters=100)

    assert m.num_authors == 40
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    recovered = np.array([pos[f"a{a}"] for a in range(40)])
    r = abs(np.corrcoef(recovered, theta)[0, 1])
    assert r > 0.9, f"position recovery r={r:.3f}"

    # word discrimination recovers the planted beta (by vocabulary order)
    vocab = m.vocabulary
    bhat = dict(zip(vocab, m.word_discrimination))
    planted = np.array([beta[int(w[1:])] for w in vocab])
    got = np.array([bhat[w] for w in vocab])
    rb = abs(np.corrcoef(got, planted)[0, 1])
    assert rb > 0.8, f"discrimination recovery r={rb:.3f}"

    # positions standardized
    assert abs(recovered.mean()) < 1e-6


def test_position_se():
    # SE is finite/positive, aligned to author_positions, and smaller for authors with
    # more text (more information -> tighter estimate).
    docs, group, theta, _ = _planted(n_authors=40, docs_per=3, seed=1)
    # give the first 20 authors much more text than the last 20
    docs2, group2 = [], []
    for d, g in zip(docs, group):
        reps = 5 if int(g[1:]) < 20 else 1
        for _ in range(reps):
            docs2.append(d)
            group2.append(g)
    m = topica.Wordfish(seed=1)
    m.fit(docs2, group=group2, anchors={"a0": -1.0, "a39": 1.0})
    se = m.position_se
    assert se.shape == (m.num_authors,)
    assert np.all(np.isfinite(se)) and np.all(se > 0)
    se_by = dict(zip(m.author_names, se))
    se_more = np.mean([se_by[f"a{a}"] for a in range(20)])
    se_less = np.mean([se_by[f"a{a}"] for a in range(20, 40)])
    assert se_more < se_less, f"more text should give smaller SE: {se_more:.3f} vs {se_less:.3f}"


def test_anchors_orient_sign():
    docs, group, theta, _ = _planted(seed=2)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0})
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    assert pos["a0"] < pos["a39"], "anchors did not orient the axis"


def test_discriminating_words():
    docs, group, _, _ = _planted(seed=3)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0})
    pos, neg = m.discriminating_words(5)
    assert len(pos) == 5 and len(neg) == 5
    # positive-end words have higher beta than negative-end words
    assert pos[0][1] > neg[0][1]


def test_determinism():
    docs, group, _, _ = _planted(seed=4)
    a = topica.Wordfish(seed=1)
    a.fit(docs, group=group)
    b = topica.Wordfish(seed=1)
    b.fit(docs, group=group)
    assert np.array_equal(a.author_positions, b.author_positions)


def test_save_load(tmp_path):
    docs, group, _, _ = _planted(seed=5)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0})
    p = tmp_path / "wf.topica"
    m.save(str(p))
    m2 = topica.Wordfish.load(str(p))
    assert np.array_equal(m.author_positions, m2.author_positions)
    assert m.author_names == m2.author_names
    assert m.vocabulary == m2.vocabulary


def test_inf_prior_is_flat():
    # math.inf priors must be accepted (no regularization) and still fit.
    docs, group, theta, _ = _planted(seed=6)
    m = topica.Wordfish(beta_prior_sd=math.inf, theta_prior_sd=math.inf, seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0})
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    recovered = np.array([pos[f"a{a}"] for a in range(40)])
    assert abs(np.corrcoef(recovered, theta)[0, 1]) > 0.85
