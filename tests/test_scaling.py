"""Intrinsic ideal-point diagnostics: bimodality + split-half reliability."""
import numpy as np
import pytest

import topica


def test_bimodality_separates_one_vs_two_modes():
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0, 1, 400)
    bimodal = np.concatenate([rng.normal(-2, 0.4, 200), rng.normal(2, 0.4, 200)])
    bc_uni = topica.bimodality(unimodal)
    bc_bi = topica.bimodality(bimodal)
    assert bc_uni < 0.555 < bc_bi, f"uni={bc_uni:.3f} bi={bc_bi:.3f}"
    # accepts an (n, 1) column too (e.g. author_positions)
    assert topica.bimodality(bimodal.reshape(-1, 1)) == pytest.approx(bc_bi)


def test_bimodality_needs_enough_points():
    with pytest.raises(ValueError):
        topica.bimodality([0.1, 0.2, 0.3])


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

    r = topica.split_half_reliability(fit, group, seed=0)
    assert r > 0.99, f"reliability={r:.3f}"


def test_split_half_reliability_low_for_noise():
    # If "fit" returns pure noise unrelated across halves, reliability ~ 0.
    rng = np.random.default_rng(2)
    group = [f"a{i}" for i in range(40) for _ in range(6)]

    def fit(idx):
        authors = sorted({group[i] for i in idx})
        return authors, rng.normal(size=len(authors))

    r = topica.split_half_reliability(fit, group, seed=0, repeats=3)
    assert r < 0.5, f"noise reliability={r:.3f}"


def test_split_half_reliability_with_idealpointlda():
    # Integration: a planted IdealPointLDA corpus should be reliable.
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
            m = topica.IdealPointLDA(num_topics=2, num_dims=1, seed=1)
            m.fit([docs[i] for i in idx], group=[group[i] for i in idx], iters=30)
            return m.author_names, m.author_positions[:, 0]

        r = topica.split_half_reliability(fit, group, seed=0)
        assert r > 0.6, f"planted reliability too low: {r:.3f}"
    finally:
        topica.enable_experimental(False)
