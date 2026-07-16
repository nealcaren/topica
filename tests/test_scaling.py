"""Intrinsic ideal-point diagnostics: bimodality + split-half reliability."""
import numpy as np
import pytest

import topica


def test_bimodality_separates_one_vs_two_modes():
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0, 1, 400)
    bimodal = np.concatenate([rng.normal(-2, 0.4, 200), rng.normal(2, 0.4, 200)])
    bc_uni = topica.scaling.bimodality(unimodal)
    bc_bi = topica.scaling.bimodality(bimodal)
    assert bc_uni < 0.555 < bc_bi, f"uni={bc_uni:.3f} bi={bc_bi:.3f}"
    # accepts an (n, 1) column too (e.g. author_positions)
    assert topica.scaling.bimodality(bimodal.reshape(-1, 1)) == pytest.approx(bc_bi)


def test_bimodality_needs_enough_points():
    with pytest.raises(ValueError):
        topica.scaling.bimodality([0.1, 0.2, 0.3])


def test_polarization_centroid_distance():
    pos = np.array([-1.0, -0.9, -1.1, 1.0, 0.9, 1.1])
    lab = ["L", "L", "L", "R", "R", "R"]
    # raw = distance between camp means (centroids at -1 and +1)
    assert topica.scaling.polarization(pos, lab) == pytest.approx(2.0, abs=1e-9)
    # accepts an (n, 1) column (e.g. author_positions)
    assert topica.scaling.polarization(pos.reshape(-1, 1), lab) == pytest.approx(2.0, abs=1e-9)
    # rises as the camps move apart
    near = topica.scaling.polarization([-0.2, -0.2, 0.2, 0.2], ["L", "L", "R", "R"])
    far = topica.scaling.polarization([-2.0, -2.0, 2.0, 2.0], ["L", "L", "R", "R"])
    assert far > near


def test_polarization_multidim_is_euclidean():
    pos = np.array([[-1.0, 0.0], [-1.0, 0.0], [2.0, 4.0], [2.0, 4.0]])
    lab = ["L", "L", "R", "R"]
    assert topica.scaling.polarization(pos, lab) == pytest.approx(5.0, abs=1e-9)  # 3-4-5


def test_polarization_normalize_is_scale_free():
    # Doubling the axis scale doubles the raw distance but leaves the normalized
    # (effect-size) form unchanged.
    pos = np.array([-1.0, -0.8, 0.8, 1.0])
    lab = ["L", "L", "R", "R"]
    raw1 = topica.scaling.polarization(pos, lab)
    raw2 = topica.scaling.polarization(pos * 2, lab)
    assert raw2 == pytest.approx(2 * raw1)
    assert topica.scaling.polarization(pos, lab, normalize=True) == pytest.approx(
        topica.scaling.polarization(pos * 2, lab, normalize=True)
    )


def test_polarization_more_than_two_camps():
    pos = np.array([-2.0, -2.0, 0.0, 0.0, 2.0, 2.0])
    lab = ["A", "A", "B", "B", "C", "C"]
    # mean of pairwise centroid distances: |A-B|=2, |A-C|=4, |B-C|=2 -> 8/3
    assert topica.scaling.polarization(pos, lab) == pytest.approx(8.0 / 3.0, abs=1e-9)


def test_polarization_ci_propagates_position_se():
    # The point estimate matches polarization(); tight SEs give a narrow interval
    # bracketing the estimate, large SEs widen it toward (and past) zero.
    pos = np.array([-1.0, -0.9, -1.1, 1.0, 0.9, 1.1])
    lab = ["L", "L", "L", "R", "R", "R"]
    point = topica.scaling.polarization(pos, lab)

    tight = topica.scaling.polarization_ci(pos, lab, np.full(6, 0.02), n_sim=2000, seed=0)
    assert tight.estimate == pytest.approx(point, abs=1e-9)
    assert tight.lo <= point <= tight.hi
    assert tight.se > 0.0

    wide = topica.scaling.polarization_ci(pos, lab, np.full(6, 1.0), n_sim=2000, seed=0)
    assert wide.se > tight.se
    assert wide.hi - wide.lo > tight.hi - tight.lo


def test_polarization_ci_matches_model_surface():
    # End to end: an ideal-point model's position_se feeds polarization_ci directly.
    topica.enable_experimental(True)
    rng = np.random.default_rng(0)
    docs, group, lab = [], [], []
    for camp in (0, 1):
        for a in range(5):
            p = np.ones(20)
            p[:10] += 4 if camp == 0 else 0
            p[10:] += 0 if camp == 0 else 4
            p /= p.sum()
            for _ in range(8):
                docs.append(list(rng.choice([f"w{i}" for i in range(20)], size=25, p=p)))
                group.append(f"{camp}_{a}")
    m = topica.models.IdealPointTM(3, num_dims=1, seed=1)
    m.fit(docs, group=group)
    camp_of = {n: n.split("_")[0] for n in m.author_names}
    labels = [camp_of[n] for n in m.author_names]
    res = topica.scaling.polarization_ci(m.author_positions, labels, m.position_se, n_sim=500, seed=0)
    assert res.estimate == pytest.approx(
        topica.scaling.polarization(m.author_positions, labels), abs=1e-9
    )
    assert res.lo <= res.estimate <= res.hi


def test_polarization_errors():
    with pytest.raises(ValueError):
        topica.scaling.polarization([0.1, 0.2, 0.3], ["L", "R"])  # length mismatch
    with pytest.raises(ValueError):
        topica.scaling.polarization([0.1, 0.2], ["L", "L"])  # one camp


def test_split_half_reliability_mechanics():
    # A toy "fit" with a fixed per-author trait plus tiny per-call noise: reliability
    # should be near 1. Exercises the splitting/correlation without a heavy model.
    rng = np.random.default_rng(1)
    n_authors = 30
    trait = {f"a{i}": rng.normal() for i in range(n_authors)}
    group = [f"a{i}" for i in range(n_authors) for _ in range(8)]  # 8 units each

    def fit(idx):
        authors = sorted({group[i] for i in idx})
        # position = the author's planted trait + small noise (deterministic-ish)
        pos = [trait[a] + rng.normal(0, 1e-3) for a in authors]
        return authors, np.array(pos)

    r = topica.scaling.split_half_reliability(fit, group, seed=0)
    assert r > 0.99, f"reliability={r:.3f}"


def test_split_half_reliability_low_for_noise():
    # If "fit" returns pure noise unrelated across halves, reliability ~ 0.
    rng = np.random.default_rng(2)
    group = [f"a{i}" for i in range(40) for _ in range(6)]

    def fit(idx):
        authors = sorted({group[i] for i in idx})
        return authors, rng.normal(size=len(authors))

    r = topica.scaling.split_half_reliability(fit, group, seed=0, repeats=3)
    assert r < 0.5, f"noise reliability={r:.3f}"


def test_position_intervals_mechanics():
    # Authors with different numbers of units; per-unit values are the author's trait
    # plus noise. More units -> a tighter bootstrap SE; estimates track the trait.
    rng = np.random.default_rng(0)
    group, val, trait = [], [], {}
    for a in range(20):
        trait[a] = rng.normal()
        nunits = 4 if a % 2 == 0 else 16
        for _ in range(nunits):
            group.append(f"a{a}")
            val.append(trait[a] + rng.normal(0, 1.0))
    val = np.array(val)

    def fit(idx):
        idx = list(idx)
        acc = {}
        for i in idx:
            acc.setdefault(group[i], []).append(val[i])
        authors = sorted(acc)
        return authors, np.array([np.mean(acc[a]) for a in authors])

    res = topica.scaling.position_intervals(fit, group, n_boot=60, seed=0)
    assert set(res) == {f"a{a}" for a in range(20)}
    est = np.array([res[f"a{a}"].estimate for a in range(20)])
    tru = np.array([trait[a] for a in range(20)])
    assert abs(np.corrcoef(est, tru)[0, 1]) > 0.9
    # the lo/hi bracket the estimate; SE positive
    for a in range(20):
        pi = res[f"a{a}"]
        assert pi.lo <= pi.estimate <= pi.hi and pi.se > 0
    se_few = np.mean([res[f"a{a}"].se for a in range(0, 20, 2)])    # 4-unit authors
    se_many = np.mean([res[f"a{a}"].se for a in range(1, 20, 2)])   # 16-unit authors
    assert se_few > se_many, f"more data should mean smaller SE: {se_few:.3f} vs {se_many:.3f}"


def test_position_intervals_with_wordfish():
    topica.enable_experimental(True)
    try:
        rng = np.random.default_rng(1)
        theta = np.linspace(-1, 1, 24)
        docs, group = [], []
        for a in range(24):
            for _ in range(6):
                doc = []
                for j in range(50):
                    lam = np.exp(1.0 + ((j % 2) * 2 - 1) * theta[a] * (j < 25))
                    doc += [f"w{j}"] * int(rng.poisson(max(lam, 0.05)))
                docs.append(doc)
                group.append(f"a{a}")

        def fit(idx):
            m = topica.models.Wordfish(seed=1)
            m.fit([docs[i] for i in idx], group=[group[i] for i in idx],
                  anchors={"a0": -1.0, "a23": 1.0}, iters=80)
            return m.author_names, m.author_positions[:, 0]

        res = topica.scaling.position_intervals(fit, group, n_boot=10, seed=0)
        assert all(np.isfinite(pi.se) and pi.se >= 0 for pi in res.values())
    finally:
        topica.enable_experimental(False)


def test_split_half_reliability_with_idealpointlda():
    # Integration: a planted IdealPointTM (counts) corpus should be reliable.
    topica.enable_experimental(True)
    try:
        rng = np.random.default_rng(3)
        n_authors, vocab, half = 24, 40, 20
        theta = rng.uniform(-1, 1, n_authors)

        def beta(topic, x):
            eta = np.full(vocab, -3.0)
            if topic == 0:
                eta[:half] = 0.5
                for v in range(half):
                    eta[v] += x * (2.0 if v % 2 == 0 else -2.0)
            else:
                eta[half:] = 0.5
            e = np.exp(eta - eta.max())
            return e / e.sum()

        docs, group = [], []
        for a in range(n_authors):
            for _ in range(10):  # 10 docs each -> splittable
                doc = []
                for _ in range(40):
                    t = rng.integers(0, 2)
                    doc.append(f"w{rng.choice(vocab, p=beta(t, theta[a]))}")
                docs.append(doc)
                group.append(f"a{a}")

        def fit(idx):
            m = topica.models.IdealPointTM(num_topics=2, num_dims=1, seed=1)
            m.fit([docs[i] for i in idx], group=[group[i] for i in idx], iters=30)
            return m.author_names, m.author_positions[:, 0]

        r = topica.scaling.split_half_reliability(fit, group, seed=0)
        assert r > 0.6, f"planted reliability too low: {r:.3f}"
    finally:
        topica.enable_experimental(False)
