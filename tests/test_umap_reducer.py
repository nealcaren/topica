"""The optional UMAP reducer for the embedding models.

UMAP is shipped in the wheel as an opt-in (`reducer="umap"`); PCA stays the
default. topica's in-house UMAP reducer is seeded, so the whole fit is
reproducible for a fixed `seed` — it must run, NOT warn about non-determinism,
and give identical results across runs.
"""

import warnings

import numpy as np
import pytest

import topica


def _manifold(seed=0, per=40, k=4):
    rng = np.random.default_rng(seed)
    docs, emb = [], []
    for c in range(k):
        centre = np.array([np.cos(c * 1.4), np.sin(c * 1.4)]) * 5.0
        for _ in range(per):
            p = centre + rng.normal(0, 0.4, 2)
            emb.append(np.concatenate([p, rng.normal(0, 0.3, 6)]))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 5, 8)])
    return docs, np.array(emb)


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
def test_umap_reducer_runs_without_warning(model_cls):
    docs, emb = _manifold()
    cls = getattr(topica, model_cls)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = cls(reducer="umap", min_cluster_size=15, n_neighbors=15, seed=1)
        m.fit(docs, emb)
    # discovers some topics on a clearly-clustered manifold
    assert m.num_topics >= 2
    # the in-house UMAP is seeded, so no non-reproducibility warning is emitted
    assert not any("not reproducible" in str(x.message) for x in w)


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
def test_umap_fit_is_reproducible(model_cls):
    docs, emb = _manifold()
    cls = getattr(topica, model_cls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = cls(reducer="umap", min_cluster_size=15, n_neighbors=15, seed=1)
        a.fit(docs, emb)
        b = cls(reducer="umap", min_cluster_size=15, n_neighbors=15, seed=1)
        b.fit(docs, emb)
    # seeded reducer + deterministic clusterer => identical fits
    assert a.num_topics == b.num_topics
    assert list(a.labels) == list(b.labels)


def test_pca_default_is_silent():
    docs, emb = _manifold()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = topica.BERTopic(min_cluster_size=15, seed=1)  # reducer="pca" default
        m.fit(docs, emb)
    assert not any("reproducible" in str(x.message) for x in w)


def test_transform_is_deterministic_even_with_umap_fit():
    docs, emb = _manifold()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.Top2Vec(reducer="umap", min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    # the prediction phase never re-runs the reducer, so it is reproducible
    a = m.transform(docs[:6], emb[:6])
    b = m.transform(docs[:6], emb[:6])
    assert np.allclose(a, b)


def test_unknown_reducer_errors():
    with pytest.raises(ValueError, match="unknown reducer"):
        topica.BERTopic(reducer="tsne")
