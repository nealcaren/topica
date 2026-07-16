"""The optional UMAP reducer for the embedding models.

UMAP is shipped in the wheel as an opt-in (`reducer="umap"`); PCA stays the
deterministic default. The UMAP *discovery* fit is intentionally not asserted to
be reproducible (the umap-rs optimizer's negative sampling is unseeded), but it
must run, warn, and leave the transform / prediction phase deterministic.
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
def test_umap_reducer_runs_and_warns(model_cls):
    docs, emb = _manifold()
    cls = getattr(topica.models, model_cls)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = cls(reducer="umap", min_cluster_size=15, n_neighbors=15, seed=1)
        m.fit(docs, emb)
    # discovers some topics on a clearly-clustered manifold
    assert m.num_topics >= 2
    # the documented non-determinism is surfaced as a warning
    assert any("not reproducible" in str(x.message) for x in w)


def test_pca_default_is_silent():
    docs, emb = _manifold()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = topica.models.BERTopic(min_cluster_size=15, seed=1)  # reducer="pca" default
        m.fit(docs, emb)
    assert not any("reproducible" in str(x.message) for x in w)


def test_transform_is_deterministic_even_with_umap_fit():
    docs, emb = _manifold()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.models.Top2Vec(reducer="umap", min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    # the prediction phase never re-runs the reducer, so it is reproducible
    a = m.transform(docs[:6], emb[:6])
    b = m.transform(docs[:6], emb[:6])
    assert np.allclose(a, b)


def test_unknown_reducer_errors():
    with pytest.raises(ValueError, match="unknown reducer"):
        topica.models.BERTopic(reducer="tsne")
