"""Wordfish: the word-frequency ideal-point baseline.

Wordfish has no topic distribution, so it is exempt from the registry-driven
topic-health invariant suite; its validity is checked here: it recovers planted
positions and word discriminations from counts sampled from its own model.
"""
import math

import numpy as np
import pytest

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


def test_determinism_with_multiple_anchors():
    # Anchors arrive as a dict (a HashMap on the Rust side); the orientation must
    # be a deterministic function of the corpus + anchors regardless of the dict's
    # iteration order (#411 — anchor pairs are sorted before the sign check).
    docs, group, _, _ = _planted(seed=4)
    anchors = {"a0": -1.0, "a10": -0.5, "a20": 0.5, "a39": 1.0}
    a = topica.Wordfish(seed=1)
    a.fit(docs, group=group, anchors=anchors)
    b = topica.Wordfish(seed=1)
    b.fit(docs, group=group, anchors=dict(reversed(list(anchors.items()))))
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


def _contaminated(seed=0, n_authors=50, docs_per=14, doc_len=40, p_nu=0.45, p_id=0.40):
    """A weak ideology signal plus a DOMINANT, control-aligned nuisance axis. Plain
    Wordfish is hijacked by the nuisance; the control covariate should absorb it and
    recover ideology."""
    rng = np.random.default_rng(seed)
    ideo = np.linspace(-1.0, 1.0, n_authors)
    ctrl = np.array([i % 2 for i in range(n_authors)])
    iL = [f"iL{i}" for i in range(8)]; iR = [f"iR{i}" for i in range(8)]
    nA = [f"nA{i}" for i in range(8)]; nB = [f"nB{i}" for i in range(8)]
    fill = [f"f{i}" for i in range(30)]
    docs, group, control = [], [], []
    for a in range(n_authors):
        pr_right = (ideo[a] + 1) / 2
        for _ in range(docs_per):
            d = []
            for _ in range(doc_len):
                u = rng.random()
                if u < p_nu:
                    d.append((nB if ctrl[a] == 1 else nA)[rng.integers(8)])
                elif u < p_nu + p_id:
                    d.append((iR if rng.random() < pr_right else iL)[rng.integers(8)])
                else:
                    d.append(fill[rng.integers(30)])
            docs.append(d); group.append(f"a{a:02d}"); control.append(f"c{ctrl[a]}")
    return docs, group, control, ideo


def _ideology_recovery(m, ideo):
    pos = m.author_positions[:, 0]
    truth = np.array([ideo[int(n[1:])] for n in m.author_names])
    return abs(np.corrcoef(pos, truth)[0, 1])


def test_control_covariate_rescues_axis():
    docs, group, control, ideo = _contaminated(seed=0)
    plain = topica.Wordfish(min_count=1, seed=1)
    plain.fit(docs, group=group, iters=120)
    ctrl = topica.Wordfish(min_count=1, seed=1)
    ctrl.fit(docs, group=group, control=control, iters=120)
    r_plain = _ideology_recovery(plain, ideo)
    r_ctrl = _ideology_recovery(ctrl, ideo)
    assert r_plain < 0.4, f"nuisance should hijack plain Wordfish, got {r_plain:.3f}"
    assert r_ctrl > 0.8, f"control should recover ideology, got {r_ctrl:.3f}"
    # the absorbed effect is exposed
    assert ctrl.control_names == ["c0", "c1"]
    assert ctrl.control_word_offsets.shape == (2, len(ctrl.vocabulary))
    # baseline level row is held at zero
    assert np.allclose(ctrl.control_word_offsets[0], 0.0)


def test_control_none_matches_plain():
    # control=None must give exactly the historical Wordfish fit.
    docs, group, _, _ = _planted(seed=2)
    a = topica.Wordfish(seed=1); a.fit(docs, group=group, iters=60)
    b = topica.Wordfish(seed=1); b.fit(docs, group=group, control=None, iters=60)
    assert np.array_equal(a.author_positions, b.author_positions)


def test_control_save_load(tmp_path):
    docs, group, control, _ = _contaminated(seed=3)
    m = topica.Wordfish(min_count=1, seed=1)
    m.fit(docs, group=group, control=control, iters=40)
    p = tmp_path / "wfc.topica"
    m.save(str(p))
    m2 = topica.Wordfish.load(str(p))
    assert np.array_equal(m.author_positions, m2.author_positions)
    assert np.array_equal(m.control_word_offsets, m2.control_word_offsets)
    assert m.control_names == m2.control_names


def test_control_validation():
    docs, group, control, _ = _contaminated(seed=4)
    m = topica.Wordfish(min_count=1, seed=1)
    # wrong length
    with pytest.raises(Exception):
        m.fit(docs, group=group, control=control[:-5], iters=10)
    # not constant within an author: flip one document's control within author a00
    bad = list(control)
    first = group.index("a00")
    bad[first] = "cX" if bad[first] != "cX" else "cY"
    with pytest.raises(Exception):
        m.fit(docs, group=group, control=bad, iters=10)
