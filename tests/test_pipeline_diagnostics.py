"""Post-fit degenerate-pipeline warnings for BERTopic/Top2Vec (#356).

The reduce->cluster pipeline fails silently: a bad configuration still returns a
model. These tests check that the common degenerate signatures (collapse to 1-2
topics, a high `-1` noise fraction, gross over-splitting) emit a warning naming a
fix, that the warning is opt-out via `diagnostics=False`, and that a healthy fit
stays quiet.
"""

import warnings

import numpy as np
import pytest

import topica


def _fit_capture(model, docs, emb):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(docs, emb)
    return [str(x.message) for x in w]


def test_collapse_warns():
    # Two clean blobs -> HDBSCAN finds only 2 topics on a 300-doc corpus.
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (150, 6)), rng.normal(8, 0.3, (150, 6))])
    docs = [[f"w{i % 6}"] for i in range(300)]
    msgs = _fit_capture(topica.BERTopic(min_cluster_size=15, seed=1), docs, emb)
    assert any("only 2 topic" in m for m in msgs), msgs


def test_collapse_not_warned_for_fixed_k():
    # kmeans with num_clusters=2 legitimately returns 2 topics — not a collapse.
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (150, 6)), rng.normal(8, 0.3, (150, 6))])
    docs = [[f"w{i % 6}"] for i in range(300)]
    msgs = _fit_capture(
        topica.BERTopic(clusterer="kmeans", num_clusters=2, seed=1), docs, emb
    )
    assert not any("only 2 topic" in m for m in msgs), msgs


def test_high_noise_warns():
    # dim == n_components -> no PCA/normalize; wide scatter stays HDBSCAN noise.
    rng = np.random.default_rng(7)
    emb = np.vstack([
        rng.normal([0, 0, 0, 0, 0], 0.15, (60, 5)),
        rng.normal([30, 30, 0, 0, 0], 0.15, (60, 5)),
        rng.uniform(-60, 60, (220, 5)),
    ])
    docs = [[f"w{i % 6}"] for i in range(len(emb))]
    m = topica.BERTopic(n_components=5, min_cluster_size=20, min_samples=8, seed=1)
    msgs = _fit_capture(m, docs, emb)
    assert any("unassigned" in x for x in msgs), msgs


def test_over_split_warns():
    # Many tiny well-separated blobs -> HDBSCAN over-splits far past n/10.
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 6, (60, 20))
    emb, docs = [], []
    for c in range(60):
        for _ in range(8):
            emb.append(centers[c] + rng.normal(0, 0.15, 20))
            docs.append([f"w{c}"])
    emb = np.array(emb)
    msgs = _fit_capture(topica.BERTopic(min_cluster_size=3, seed=1), docs, emb)
    assert any("over-split" in m for m in msgs), msgs


def test_diagnostics_false_suppresses():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (150, 6)), rng.normal(8, 0.3, (150, 6))])
    docs = [[f"w{i % 6}"] for i in range(300)]
    msgs = _fit_capture(
        topica.BERTopic(min_cluster_size=15, diagnostics=False, seed=1), docs, emb
    )
    assert not any("only 2 topic" in m or "over-split" in m or "unassigned" in m for m in msgs), msgs


def test_healthy_fit_is_quiet():
    # Six clean, well-populated blobs: a sensible topic count, no noise.
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, (6, 15))
    emb, docs = [], []
    for c in range(6):
        for _ in range(50):
            emb.append(centers[c] + rng.normal(0, 0.6, 15))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 5, 4)])
    emb = np.array(emb)
    m = topica.BERTopic(clusterer="leiden", seed=1)
    msgs = _fit_capture(m, docs, emb)
    assert not any(
        "only" in x or "over-split" in x or "unassigned" in x for x in msgs
    ), (m.num_topics, msgs)


def test_top2vec_also_diagnoses():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (150, 6)), rng.normal(8, 0.3, (150, 6))])
    docs = [[f"w{i % 6}"] for i in range(300)]
    msgs = _fit_capture(topica.Top2Vec(min_cluster_size=15, seed=1), docs, emb)
    assert any("Top2Vec" in m and "only 2 topic" in m for m in msgs), msgs
