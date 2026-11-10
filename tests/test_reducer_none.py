"""`reducer="none"` / `n_components=0`: cluster the raw embeddings on the unit sphere.

Without the L2 normalization, rows with larger norms dominate the Euclidean
clusterer and the partition tracks vector length rather than direction.
"""

import warnings

import numpy as np
import pytest

import topica


def _blobs(seed=0, n_per=40, dim=64, k=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    emb = np.vstack([c + 0.3 * rng.normal(size=(n_per, dim)) for c in centers])
    docs = [[f"w{g}", f"w{g}x", "common"] for g in range(k) for _ in range(n_per)]
    return docs, emb


@pytest.mark.parametrize("cls", [topica.BERTopic, topica.Top2Vec])
def test_raw_clustering_ignores_row_norms(cls):
    docs, emb = _blobs()
    scaled = emb * np.random.default_rng(1).uniform(0.2, 5.0, size=(len(emb), 1))
    kw = dict(reducer="none", clusterer="kmeans", num_clusters=3, seed=13)
    a = np.asarray(cls(**kw).fit(docs, emb).labels)
    b = np.asarray(cls(**kw).fit(docs, scaled).labels)
    np.testing.assert_array_equal(a, b)


def test_reducer_none_matches_n_components_zero():
    docs, emb = _blobs()
    kw = dict(clusterer="kmeans", num_clusters=3, seed=13)
    a = topica.BERTopic(reducer="none", **kw).fit(docs, emb)
    b = topica.BERTopic(n_components=0, **kw).fit(docs, emb)
    np.testing.assert_array_equal(np.asarray(a.labels), np.asarray(b.labels))
    assert a.settings["reducer"] == "none"
    assert b.settings["reducer"] == "none"
    assert topica.BERTopic().settings["reducer"] == "umap"
    assert topica.BERTopic(reducer="pca").settings["reducer"] == "pca"


@pytest.mark.parametrize("cls", [topica.BERTopic, topica.Top2Vec])
def test_hdbscan_on_raw_embeddings_warns(cls):
    docs, emb = _blobs()
    with pytest.warns(UserWarning, match="unreduced 64-dimensional"):
        cls(reducer="none", diagnostics=False).fit(docs, emb)
    # Low-dimensional input (e.g. already reduced by the user) does not warn.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*unreduced.*")
        cls(reducer="none", diagnostics=False, min_cluster_size=5).fit(docs, emb[:, :5])


def test_unknown_reducer_lists_none():
    with pytest.raises(ValueError, match="'none'"):
        topica.BERTopic(reducer="tsne")
