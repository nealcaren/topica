"""CETopic TFIDF×IDF_i topic-word weighting on topica.BERTopic (issue #581).

CETopic (Zhang, Fang, Chen & Namazi-Rad, "Is Neural Topic Modelling Better than
Clustering?", NAACL 2022) is contextual embeddings → UMAP → K-Means → per-cluster
word weighting; topica's ``BERTopic`` already expresses the pipeline, so the port is
the one weighting formula, exposed as ``weighting="tfidf-idf"``.

These use the deterministic ``clusterer="kmeans"`` path (no umap/hdbscan needed) so
the recovered partition is stable, and pin the port against an independent
scikit-learn reference computed on the very same clusters.
"""

import numpy as np
import pytest

import topica


def _planted(k=3, per=40, block=5, dim=12, ubiquitous=True, seed=0):
    """k well-separated clusters. Each cluster c uses its own vocabulary block and,
    if ``ubiquitous``, a shared word ``"STOP"`` in every document — the cross-cluster
    word the TFIDF×IDF_i penalty is meant to demote."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((k, dim))
    for c in range(k):
        centers[c, c % dim] = 8.0
    docs, emb = [], []
    for d in range(k * per):
        c = d % k
        toks = [f"w{c * block + rng.integers(0, block)}" for _ in range(8)]
        if ubiquitous:
            toks += ["STOP"] * 6
        docs.append(toks)
        emb.append(centers[c] + rng.normal(0, 0.3, dim))
    return docs, np.array(emb)


def _fit(docs, emb, **kw):
    kw.setdefault("clusterer", "kmeans")
    kw.setdefault("num_clusters", 3)
    kw.setdefault("n_components", 5)
    kw.setdefault("reducer", "pca")
    kw.setdefault("seed", 1)
    m = topica.BERTopic(**kw)
    m.fit(docs, emb)
    return m


# --- basic validity -----------------------------------------------------------

def test_tfidf_idf_topic_word_is_valid_distribution():
    docs, emb = _planted()
    m = _fit(docs, emb, weighting="tfidf-idf")
    tw = np.asarray(m.topic_word)
    assert tw.shape == (m.num_topics, len(m.vocabulary))
    assert (tw >= 0).all()
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_tfidf_idf_recovers_block_pure_topics():
    docs, emb = _planted(ubiquitous=False)
    m = _fit(docs, emb, weighting="tfidf-idf")
    for t in range(m.num_topics):
        blocks = {int(w[1:]) // 5 for w, _ in m.top_words(4, topic=t, weights=True)}
        assert len(blocks) == 1, f"topic {t} mixes planted blocks: {blocks}"


def test_tfidf_idf_changes_the_representation():
    # The weighting knob must actually take effect: on the same (deterministic
    # kmeans) clusters, TFIDF×IDF_i produces a different topic-word matrix than the
    # default c-TF-IDF. (On a clean planted corpus the *top words* can still agree —
    # the paper's diversity gains are marginal effects on real text — so we assert
    # the matrices differ, not that one wins.)
    docs, emb = _planted(ubiquitous=True)
    base = _fit(docs, emb)  # c-tf-idf default
    ceto = _fit(docs, emb, weighting="tfidf-idf")
    assert list(base.labels) == list(ceto.labels), "same clusters expected"
    assert not np.allclose(
        np.asarray(base.topic_word), np.asarray(ceto.topic_word)
    ), "weighting='tfidf-idf' should change topic_word from c-TF-IDF"


# --- parity against the scikit-learn reference (hyintell/topicx) ---------------

def _reference_tfidf_idfi(docs, labels, vocab):
    """Recompute CETopic's TFIDF×IDF_i with the reference's exact scikit-learn
    pipeline (two TfidfTransformers at their defaults, then l1-normalize)."""
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize

    idx = {w: i for i, w in enumerate(vocab)}
    V, K = len(vocab), int(max(labels)) + 1
    X_origin = np.zeros((len(docs), V))
    X_per = np.zeros((K, V))
    for d, doc in enumerate(docs):
        c = labels[d]
        for w in doc:
            if w in idx:
                X_origin[d, idx[w]] += 1.0
                X_per[c, idx[w]] += 1.0

    global_tfidf = TfidfTransformer().fit_transform(X_origin).toarray()  # per-row l2
    avg = np.zeros((K, V))
    for c in range(K):
        rows = global_tfidf[np.asarray(labels) == c]
        avg[c] = rows.mean(axis=0) if len(rows) else 0.0
    idfi = TfidfTransformer().fit(X_per).idf_
    scores = avg * idfi
    return normalize(scores, axis=1, norm="l1")


def test_tfidf_idf_matches_sklearn_reference_on_same_clusters():
    # scikit-learn is a dev-only reference toolchain; CI runs without it, so this
    # parity test skips cleanly there rather than erroring the whole test job.
    pytest.importorskip("sklearn")
    docs, emb = _planted(ubiquitous=True, seed=3)
    m = _fit(docs, emb, weighting="tfidf-idf")
    labels = list(m.labels)
    assert min(labels) >= 0, "kmeans should leave no -1 noise"
    want = _reference_tfidf_idfi(docs, labels, list(m.vocabulary))
    got = np.asarray(m.topic_word)
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-9), (
        f"max abs diff {np.abs(got - want).max()}"
    )


# --- aliases, settings, save/load, validation ---------------------------------

@pytest.mark.parametrize("alias", ["tfidf-idf", "tfidf_idfi", "TFIDF-IDF", "cetopic"])
def test_weighting_aliases_accepted(alias):
    assert topica.BERTopic(weighting=alias).settings["weighting"] == "tfidf-idf"


def test_default_weighting_is_ctfidf():
    docs, emb = _planted()
    m = _fit(docs, emb)  # no weighting= argument
    assert m.settings["weighting"] == "c-tf-idf"


def test_weighting_survives_save_load(tmp_path):
    docs, emb = _planted()
    m = _fit(docs, emb, weighting="tfidf-idf")
    assert m.settings["weighting"] == "tfidf-idf"
    p = tmp_path / "cetopic.topica"
    m.save(str(p))
    loaded = topica.BERTopic.load(str(p))
    assert loaded.settings["weighting"] == "tfidf-idf"
    assert np.allclose(np.asarray(loaded.topic_word), np.asarray(m.topic_word))


def test_merge_topics_keeps_tfidf_idf_weighting(tmp_path):
    docs, emb = _planted(k=3, per=40)
    m = _fit(docs, emb, num_clusters=4, weighting="tfidf-idf")
    assert m.num_topics == 4
    m.merge_topics([[0, 1]])
    tw = np.asarray(m.topic_word)
    assert (tw >= 0).all()
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_invalid_weighting_raises():
    with pytest.raises(ValueError):
        topica.BERTopic(weighting="bogus")
