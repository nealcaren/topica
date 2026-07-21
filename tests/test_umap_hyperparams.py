"""UMAP hyperparameters exposed on BERTopic/Top2Vec and topica.project (#346).

The in-house UMAP reducer already accepted min_dist/spread/n_epochs/
negative_sample_rate/repulsion_strength/metric; these tests check they are now
reachable from Python, change the layout, keep the reference defaults
(non-breaking), and are validated.
"""

import warnings

import numpy as np
import pytest

import topica


def _umap_available():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topica.project(np.random.default_rng(0).normal(0, 1, (20, 6)), 2, method="umap")
        return True
    except RuntimeError:
        return False


umap_only = pytest.mark.skipif(not _umap_available(), reason="build without the umap feature")


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(0, 1, (200, 20))


@umap_only
def test_project_umap_params_change_layout(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = topica.project(X, 5, method="umap", seed=1)
        far = topica.project(X, 5, method="umap", min_dist=0.8, seed=1)
        euc = topica.project(X, 5, method="umap", metric="euclidean", seed=1)
        again = topica.project(X, 5, method="umap", seed=1)
    assert not np.allclose(base, far), "min_dist should change the layout"
    assert not np.allclose(base, euc), "metric should change the layout"
    assert np.allclose(base, again), "same params + seed must reproduce"


@umap_only
def test_project_umap_defaults_are_the_reference(X):
    # Leaving the kwargs alone must equal passing the documented reference defaults.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implicit = topica.project(X, 5, method="umap", seed=3)
        explicit = topica.project(
            X, 5, method="umap", min_dist=0.0, spread=1.0, n_epochs=0,
            negative_sample_rate=5, repulsion_strength=1.0, metric="cosine", seed=3,
        )
    assert np.allclose(implicit, explicit)


@umap_only
def test_bertopic_umap_params_fit_and_reproduce():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, (5, 20))
    emb, docs = [], []
    for c in range(5):
        for _ in range(50):
            emb.append(centers[c] + rng.normal(0, 1, 20))
            docs.append([f"w{c}"])
    emb = np.array(emb)
    kw = dict(reducer="umap", clusterer="kmeans", num_clusters=5, min_dist=0.5,
              metric="euclidean", seed=1)
    a = topica.BERTopic(**kw)
    a.fit(docs, emb)
    b = topica.BERTopic(**kw)
    b.fit(docs, emb)
    assert list(a.labels) == list(b.labels)
    assert a.num_topics == 5


def test_umap_param_validation():
    for bad in (
        dict(min_dist=-0.1),
        dict(spread=0.0),
        dict(repulsion_strength=0.0),
        dict(negative_sample_rate=0),
        dict(metric="cosin"),
    ):
        with pytest.raises(ValueError):
            topica.BERTopic(reducer="umap", **bad)
        with pytest.raises(ValueError):
            topica.Top2Vec(reducer="umap", **bad)


def test_umap_params_accepted_but_harmless_under_pca():
    # The kwargs exist regardless of reducer; under pca they're simply ignored.
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (60, 10))
    docs = [["a", "b"] for _ in range(60)]
    m = topica.BERTopic(reducer="pca", min_dist=0.5, metric="euclidean", seed=1)
    m.fit(docs, emb)  # must not raise
    assert m.num_topics >= 0
