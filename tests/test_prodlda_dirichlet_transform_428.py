"""#428 finding 2: a Dirichlet-prior ProdLDA transforms via the normalized Weibull
median it trains under, not softmax(mu). The log-variance batchnorm this needs is
now persisted, so the transform survives save/load.

These tests are *discriminating*: against a pre-fix binary (Dirichlet transform =
softmax(mu)) the frozen expected values are off by ~0.05-0.14 per element (measured),
far outside the tolerance, so the tests fail. The math itself is guarded by the Rust
unit test `dirichlet_transform_uses_the_weibull_median_not_softmax_mu`.
"""
import numpy as np
import topica

# Fixed corpus + fit used to freeze the expected median theta below.
_VOCAB = ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"]


def _corpus():
    rng = np.random.default_rng(0)
    return [list(rng.choice(_VOCAB, size=12)) for _ in range(80)]


def _fit():
    m = topica.ProdLDA(num_topics=3, prior="dirichlet", seed=0)
    m.fit(_corpus(), iters=30)
    return m


# transform(docs[:4]) captured from the fixed (seed=0, iters=30) Dirichlet fit.
# These are the normalized Weibull median; softmax(mu) (the pre-fix transform)
# gives materially different values (per-element gap up to ~0.07).
_EXPECTED_MEDIAN = np.array(
    [
        [0.312899, 0.299055, 0.388046],
        [0.356359, 0.259517, 0.384124],
        [0.297734, 0.286162, 0.416104],
        [0.358843, 0.277953, 0.363204],
    ]
)
# Loose enough to absorb cross-platform float drift, tight enough that the pre-fix
# softmax(mu) output (off by 0.05-0.14) cannot pass.
_ATOL = 1e-2


def test_dirichlet_transform_is_the_weibull_median():
    th = np.asarray(_fit().transform(_corpus()[:4]))
    assert np.allclose(th, _EXPECTED_MEDIAN, atol=_ATOL), np.abs(th - _EXPECTED_MEDIAN).max()
    np.testing.assert_allclose(th.sum(axis=1), 1.0, atol=1e-6)


def test_dirichlet_median_transform_survives_save_load(tmp_path):
    # bn_lv is persisted, so the loaded model reproduces the median transform. A
    # pre-fix binary would neither persist bn_lv nor produce the median, so the
    # loaded transform would be softmax(mu) and miss the frozen values.
    m = _fit()
    docs4 = _corpus()[:4]
    before = np.asarray(m.transform(docs4))
    p = str(tmp_path / "prodlda_dir.topica")
    m.save(p)
    loaded = topica.ProdLDA.load(p)
    after = np.asarray(loaded.transform(docs4))

    assert np.allclose(before, after, atol=1e-9), np.abs(before - after).max()
    assert np.allclose(after, _EXPECTED_MEDIAN, atol=_ATOL), np.abs(after - _EXPECTED_MEDIAN).max()
    assert loaded.settings["prior"] == "dirichlet"
