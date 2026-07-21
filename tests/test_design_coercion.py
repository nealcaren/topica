"""Design-matrix coercion: a covariate matrix no longer has to be pre-cast with
``.to_numpy(float)``. A numeric DataFrame/Series (or an int array) goes straight
into ``STM.fit`` and ``estimate_effect``; a non-numeric column raises a directive
error pointing at the design-matrix helpers.
"""

import numpy as np
import pandas as pd
import pytest

import topica
from topica import STM, estimate_effect


@pytest.fixture(scope="module")
def corpus_meta():
    docs, treat = [], []
    for i in range(120):
        t = i % 2
        docs.append(["threat", "fear", "danger"] * 2 if t else ["calm", "neutral", "ok"] * 2)
        treat.append(t)
    corpus = topica.Corpus.from_documents(docs)
    meta = pd.DataFrame({"treatment": treat})  # int64 column
    return corpus, meta


def test_stm_fit_accepts_int_dataframe(corpus_meta):
    corpus, meta = corpus_meta
    a = STM(num_topics=3, seed=42)
    a.fit(corpus, meta[["treatment"]].to_numpy(float), prevalence_names=["treatment"], iters=20)
    b = STM(num_topics=3, seed=42)
    b.fit(corpus, meta[["treatment"]], prevalence_names=["treatment"], iters=20)
    # Coercing internally must not change the fit.
    assert np.array_equal(np.asarray(a.doc_topic), np.asarray(b.doc_topic))
    assert np.array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))


def test_stm_fit_accepts_series(corpus_meta):
    corpus, meta = corpus_meta
    m = STM(num_topics=3, seed=42)
    m.fit(corpus, meta["treatment"], prevalence_names=["treatment"], iters=20)
    assert m.doc_topic.shape == (corpus.num_docs, 3)


def test_stm_fit_categorical_raises_directive(corpus_meta):
    corpus, meta = corpus_meta
    bad = meta[["treatment"]].copy()
    bad["party"] = np.where(bad["treatment"] > 0, "dem", "rep")
    with pytest.raises(ValueError) as exc:
        STM(num_topics=3, seed=42).fit(corpus, bad, iters=5)
    msg = str(exc.value)
    assert "party" in msg
    assert "design_matrix" in msg or "one_hot" in msg


def test_estimate_effect_infers_names_from_dataframe(corpus_meta):
    corpus, meta = corpus_meta
    m = STM(num_topics=3, seed=42)
    m.fit(corpus, meta[["treatment"]], prevalence_names=["treatment"], iters=20)
    eff = estimate_effect(m, meta[["treatment"]])
    assert eff[0].feature_names == ["intercept", "treatment"]


def test_estimate_effect_categorical_raises_directive(corpus_meta):
    corpus, meta = corpus_meta
    m = STM(num_topics=3, seed=42)
    m.fit(corpus, meta[["treatment"]], prevalence_names=["treatment"], iters=20)
    bad = meta[["treatment"]].copy()
    bad["party"] = np.where(bad["treatment"] > 0, "dem", "rep")
    with pytest.raises(ValueError) as exc:
        estimate_effect(m, bad)
    assert "party" in str(exc.value)
