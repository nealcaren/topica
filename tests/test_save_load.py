"""Model serialization: every model saves to disk and loads back identically,
and a reloaded LDA can still infer topics for new documents.
"""

import os
import tempfile

import numpy as np
import pytest

import topica

DOCS = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path / "model.tt")


def _fit_all():
    """Return one fitted instance of every model."""
    X = np.array([[0.0]] * 30 + [[1.0]] * 30)
    y = np.array([0.0] * 30 + [1.0] * 30)
    groups = ["a"] * 30 + ["b"] * 30
    times = [0] * 30 + [1] * 30

    lda = topica.models.LDA(num_topics=2, seed=1); lda.fit(DOCS, iters=200)
    dmr = topica.models.DMR(num_topics=2, seed=1); dmr.fit(DOCS, X, feature_names=["g"])
    lab = topica.models.LabeledLDA(seed=1); lab.fit(DOCS, [["x"]] * 60)
    sage = topica.models.SAGE(num_topics=2, seed=1); sage.fit(DOCS, groups)
    ctm = topica.models.CTM(num_topics=2, seed=1); ctm.fit(DOCS, iters=20)
    stm = topica.models.STM(num_topics=2, seed=1); stm.fit(DOCS, X, prevalence_names=["g"], iters=20)
    hdp = topica.models.HDP(seed=1); hdp.fit(DOCS, iters=40)
    dtm = topica.models.DTM(num_topics=2, seed=1); dtm.fit(DOCS, times, iters=8)
    slda = topica.models.SupervisedLDA(num_topics=2, seed=1); slda.fit(DOCS, y, iters=10)
    return {
        "LDA": lda, "DMR": dmr, "LabeledLDA": lab, "SAGE": sage, "CTM": ctm,
        "STM": stm, "HDP": hdp, "DTM": dtm, "SupervisedLDA": slda,
    }


@pytest.mark.parametrize("name", ["LDA", "DMR", "LabeledLDA", "SAGE", "CTM", "STM", "HDP", "DTM", "SupervisedLDA"])
def test_roundtrip(name, tmp):
    m = _fit_all()[name]
    m.save(tmp)
    assert os.path.getsize(tmp) > 0
    loaded = type(m).load(tmp)

    if name == "DTM":
        for t in range(m.num_times):
            assert np.array_equal(m.topic_word(t), loaded.topic_word(t))
    elif name == "SAGE":
        assert np.array_equal(m.topic_word, loaded.topic_word)  # (K, G, V)
    else:
        assert np.array_equal(m.topic_word, loaded.topic_word)
        assert np.array_equal(m.doc_topic, loaded.doc_topic)
    assert list(m.vocabulary) == list(loaded.vocabulary)


def test_lda_transform_after_load(tmp):
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=200)
    before = m.transform([["cat", "dog"], ["star", "moon"]])
    m.save(tmp)
    loaded = topica.models.LDA.load(tmp)
    after = loaded.transform([["cat", "dog"], ["star", "moon"]])
    assert np.array_equal(before, after)


def test_lda_save_state_mallet_format(tmp_path):
    import gzip

    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=200)
    path = str(tmp_path / "state.gz")
    m.save_state(path)

    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    assert lines[0] == "#doc source pos typeindex type topic"
    assert lines[1].startswith("#alpha :")
    assert lines[2].startswith("#beta :")

    rows = lines[3:]
    # One row per token in the training corpus.
    assert len(rows) == sum(len(doc) for doc in DOCS)
    # Each row: doc source pos typeindex type topic; topic in [0, K).
    for row in rows[:5]:
        parts = row.split()
        assert len(parts) == 6
        assert 0 <= int(parts[5]) < 3
    # typeindex matches the vocabulary entry named in the row.
    vocab = list(m.vocabulary)
    for row in rows:
        parts = row.split()
        assert vocab[int(parts[3])] == parts[4]


def test_lda_save_state_after_load(tmp_path):
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=150)
    model_path = str(tmp_path / "m.tt")
    m.save(model_path)
    loaded = topica.models.LDA.load(model_path)
    state_path = str(tmp_path / "s.gz")
    loaded.save_state(state_path)  # token-level state survives a round-trip
    assert os.path.getsize(state_path) > 0


def test_save_state_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        topica.models.LDA(num_topics=2).save_state(str(tmp_path / "s.gz"))


def test_stm_posterior_survives(tmp):
    m = topica.models.STM(num_topics=2, seed=1)
    X = np.array([[0.0]] * 30 + [[1.0]] * 30)
    m.fit(DOCS, X, prevalence_names=["g"], iters=20)
    m.save(tmp)
    loaded = topica.models.STM.load(tmp)
    assert np.array_equal(m.eta_mean, loaded.eta_mean)
    assert np.array_equal(m.eta_cov, loaded.eta_cov)
    assert np.array_equal(m.prevalence_effects, loaded.prevalence_effects)


def test_slda_predict_after_load(tmp):
    m = topica.models.SupervisedLDA(num_topics=2, seed=1)
    y = np.array([0.0] * 30 + [1.0] * 30)
    m.fit(DOCS, y, iters=10)
    m.save(tmp)
    loaded = topica.models.SupervisedLDA.load(tmp)
    assert np.allclose(m.coefficients, loaded.coefficients)
    assert np.allclose(m.predict(DOCS), loaded.predict(DOCS))


def test_save_unfitted_raises(tmp):
    with pytest.raises(RuntimeError):
        topica.models.LDA(num_topics=2).save(tmp)


def test_load_garbage_raises(tmp_path):
    bad = str(tmp_path / "bad.tt")
    with open(bad, "wb") as f:
        f.write(b"not a model")
    with pytest.raises(ValueError):
        topica.models.LDA.load(bad)
