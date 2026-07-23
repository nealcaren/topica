"""#428 finding 2: a Dirichlet-prior ProdLDA transforms via the normalized Weibull
median it trains under (not softmax(mu)). The log-variance batchnorm this needs is
now persisted, so the Dirichlet transform is identical before and after save/load."""
import numpy as np
import pytest
import topica


def _blocky(seed=0, n=150, k=3, block=8):
    rng = np.random.default_rng(seed)
    v = k * block
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"w{b * block + int(rng.integers(block))}" for _ in range(15)])
    return docs, v


def test_dirichlet_transform_is_stable_across_save_load(tmp_path):
    docs, _ = _blocky()
    m = topica.ProdLDA(num_topics=3, prior="dirichlet", seed=1)
    m.fit(docs, iters=40)
    before = np.asarray(m.transform(docs[:8]))

    p = str(tmp_path / "prodlda_dir.topica")
    m.save(p)
    loaded = topica.ProdLDA.load(p)
    after = np.asarray(loaded.transform(docs[:8]))

    # bn_lv round-tripped, so the Dirichlet median transform is reproduced exactly.
    assert np.allclose(before, after, atol=1e-9), np.abs(before - after).max()
    # settings still report the Dirichlet prior.
    assert loaded.settings["prior"] == "dirichlet"


def test_dirichlet_and_laplace_transforms_differ(tmp_path):
    # Sanity: the two priors give different point estimates for the same corpus,
    # so the Dirichlet path is not silently collapsing to the laplace softmax(mu).
    docs, _ = _blocky(seed=2)
    dir_m = topica.ProdLDA(num_topics=3, prior="dirichlet", seed=7)
    dir_m.fit(docs, iters=40)
    lap_m = topica.ProdLDA(num_topics=3, prior="laplace", seed=7)
    lap_m.fit(docs, iters=40)
    d = np.asarray(dir_m.transform(docs[:8]))
    l = np.asarray(lap_m.transform(docs[:8]))
    assert np.abs(d - l).max() > 1e-3
    # both are valid simplices
    assert np.allclose(d.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(l.sum(axis=1), 1.0, atol=1e-6)
