"""Evaluation metrics ported from OCTIS and TopMost (#786 recon follow-up):
topic significance (KL to a null), downstream classification quality on theta,
and dynamic coherence/diversity scored per time slice.
"""

import numpy as np
import pytest

import topica


def _two_block_corpus(n=200, seed=0):
    """Two well-separated word blocks, with a binary label and a 4-slice time
    index per document, so the metrics have real signal to find."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(40)]
    docs, labels, times = [], [], []
    for d in range(n):
        b = d % 2
        pool = range(0, 20) if b == 0 else range(20, 40)
        docs.append([vocab[i] for i in rng.choice(list(pool), 15)])
        labels.append(b)
        times.append(d % 4)
    return topica.Corpus.from_documents(docs), labels, times


# --- topic significance ----------------------------------------------------


@pytest.mark.parametrize("kind", ["uniform", "vacuous", "background"])
def test_topic_significance_returns_finite_scalar(kind):
    c, _, _ = _two_block_corpus()
    m = topica.LDA(4, seed=13).fit(c, iters=60)
    score = topica.evaluate.topic_significance(m, kind=kind)
    assert isinstance(score, float) and np.isfinite(score)


def test_topic_significance_per_topic_length():
    c, _, _ = _two_block_corpus()
    m = topica.LDA(4, seed=13).fit(c, iters=60)
    per = topica.evaluate.topic_significance(m, kind="vacuous", per_topic=True)
    assert len(per) == 4 and all(np.isfinite(per))


def test_topic_significance_flags_a_uniform_topic():
    # A hand-built model surface: three sharp topics and one uniform (junk) topic.
    # The uniform topic must score lower on KL-uniform than the sharp ones.
    v = 40

    class Fake:
        num_topics = 4
        vocabulary = [f"w{i}" for i in range(v)]

        @property
        def topic_word(self):
            phi = np.full((4, v), 1e-6)
            phi[0, 0:5] = 1.0
            phi[1, 5:10] = 1.0
            phi[2, 10:15] = 1.0
            phi[3, :] = 1.0  # uniform junk topic
            return phi / phi.sum(axis=1, keepdims=True)

        @property
        def doc_topic(self):
            return np.full((10, 4), 0.25)

    per = topica.evaluate.topic_significance(Fake(), kind="uniform", per_topic=True)
    assert per[3] == min(per), "the uniform topic should score lowest on KL-uniform"


def test_topic_significance_bad_kind_raises():
    c, _, _ = _two_block_corpus()
    m = topica.LDA(3, seed=13).fit(c, iters=30)
    with pytest.raises(ValueError, match="kind"):
        topica.evaluate.topic_significance(m, kind="nonsense")


# --- classification quality ------------------------------------------------


def test_classification_quality_recovers_separable_labels():
    pytest.importorskip("sklearn")
    c, labels, _ = _two_block_corpus()
    m = topica.LDA(4, seed=13).fit(c, iters=80)
    out = topica.evaluate.classification_quality(m, labels)
    assert set(out) == {"accuracy", "macro_f1"}
    assert 0.0 <= out["accuracy"] <= 1.0 and 0.0 <= out["macro_f1"] <= 1.0
    # the two word blocks are cleanly separable, so topics should predict the label
    assert out["accuracy"] > 0.8


def test_classification_quality_label_length_mismatch_raises():
    pytest.importorskip("sklearn")
    c, labels, _ = _two_block_corpus()
    m = topica.LDA(3, seed=13).fit(c, iters=30)
    with pytest.raises(ValueError, match="documents"):
        topica.evaluate.classification_quality(m, labels[:-1])


# --- dynamic coherence / diversity -----------------------------------------


def test_coherence_over_time_on_dtm():
    c, _, times = _two_block_corpus()
    dm = topica.DTM(3, seed=13).fit(c, np.array(times), iters=15)
    mean = topica.evaluate.coherence_over_time(dm, c, times, n=8)
    per = topica.evaluate.coherence_over_time(dm, c, times, n=8, per_slice=True)
    assert np.isfinite(mean)
    assert len(per) == dm.num_times and all(np.isfinite(per))


def test_diversity_over_time_on_dtm():
    c, _, times = _two_block_corpus()
    dm = topica.DTM(3, seed=13).fit(c, np.array(times), iters=15)
    mean = topica.evaluate.diversity_over_time(dm, n=8)
    assert 0.0 <= mean <= 1.0


def test_dynamic_metrics_reject_static_model():
    c, _, times = _two_block_corpus()
    m = topica.LDA(3, seed=13).fit(c, iters=20)
    with pytest.raises(TypeError, match="dynamic"):
        topica.evaluate.coherence_over_time(m, c, times)
    with pytest.raises(TypeError, match="dynamic"):
        topica.evaluate.diversity_over_time(m)


def test_metrics_reachable_from_root_and_namespace():
    for name in ["topic_significance", "classification_quality",
                 "coherence_over_time", "diversity_over_time"]:
        assert hasattr(topica, name)
        assert hasattr(topica.evaluate, name)
