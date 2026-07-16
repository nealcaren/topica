"""The covariate-design helpers `spline` and `interaction` are general (they
build numpy design-matrix blocks usable by any covariate model), so they are
exported at the top level as `topica.design.spline` / `topica.design.interaction`, not only
under `topica.stm`. See the #137 follow-up.
"""

import numpy as np

import topica


def test_spline_interaction_exported_at_top_level():
    assert hasattr(topica.design, "spline"), "topica.design.spline should be a top-level export"
    assert hasattr(topica.design, "interaction"), "topica.design.interaction should be top level"
    # Same object as the stm-namespaced helper (re-export, not a copy).
    assert topica.design.spline is topica.stm.spline
    assert topica.design.interaction is topica.stm.interaction
    assert "spline" in topica.design.__all__
    assert "interaction" in topica.design.__all__


def test_spline_block_drives_a_non_stm_covariate_model():
    """spline builds a basis usable in any model's design matrix — here DMR,
    which is not the STM the helper was first written for."""
    rng = np.random.default_rng(0)
    docs = [["cat", "dog", "pet"] if i % 2 else ["tax", "vote", "law"]
            for i in range(60)]
    year = np.linspace(2000, 2020, len(docs))

    basis, names = topica.design.spline(year, df=3)
    assert basis.shape == (len(docs), 3)
    assert len(names) == 3

    m = topica.models.DMR(num_topics=2, seed=1)
    m.fit(docs, basis, feature_names=names, iters=40)
    assert m.num_topics == 2
    tw = np.asarray(m.topic_word)
    assert tw.shape[0] == 2 and np.isfinite(tw).all()


def test_interaction_block_shapes():
    a = np.array([0.0, 1.0, 0.0, 1.0])
    b = np.array([1.0, 1.0, 0.0, 0.0])
    prod, names = topica.design.interaction(a, b)
    np.testing.assert_array_equal(prod.ravel(), a * b)
    assert names == ["interaction"]
