"""Tests for TopicsOverTime (issue #694).

Follows the four idioms in CONTRIBUTING-MODELS.md (shapes/normalization,
planted-data recovery, determinism, save-load + bad-params) plus the temporal
analysis surface (topic_time / peak / mean) and edge cases.
"""

import os
import tempfile

import numpy as np
import pytest

import topica


def _toy_docs():
    return [["a", "b", "c"], ["a", "b", "b"], ["c", "c", "d"], ["d", "e", "f"]]


def _planted():
    """K=3 disjoint word blocks, each clustered in its own time window: block k's
    documents sit near t = k (so the temporal signal is real and separable)."""
    rng = np.random.default_rng(0)
    blocks = [["w0", "w1", "w2", "w3", "w4"],
              ["w5", "w6", "w7", "w8", "w9"],
              ["w10", "w11", "w12", "w13", "w14"]]
    docs, times = [], []
    for k, block in enumerate(blocks):
        for _ in range(40):
            docs.append([block[i % 5] for i in range(15)])
            times.append(float(k) + rng.uniform(-0.2, 0.2))
    return docs, times, 3


def test_shapes_and_normalization():
    docs, times, k = _planted()
    m = topica.TopicsOverTime(k, seed=0).fit(docs, times=times, iters=50)
    V = len(m.vocabulary)
    assert m.topic_word.shape == (k, V)
    assert m.doc_topic.shape == (len(docs), k)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    # temporal surface
    assert m.topic_time.shape == (k, 2)
    assert m.topic_time_peak.shape == (k,)
    assert m.topic_time_mean.shape == (k,)
    assert (m.topic_time > 0).all()


def test_determinism():
    docs, times, k = _planted()
    a = topica.TopicsOverTime(k, seed=1).fit(docs, times=times, iters=30)
    b = topica.TopicsOverTime(k, seed=1).fit(docs, times=times, iters=30)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert np.array_equal(a.topic_time, b.topic_time)


def test_recovers_planted_topics():
    docs, times, k = _planted()
    m = topica.TopicsOverTime(k, seed=13).fit(docs, times=times, iters=300)
    # each recovered topic concentrates on one 5-word block
    for row in m.topic_word:
        assert np.sort(row)[-5:].sum() > 0.9
    # the three recovered peaks/means span the time range (temporal signal recovered)
    means = np.sort(m.topic_time_mean)
    assert means[0] < 0.7 and means[-1] > 1.3  # windows centered near 0, 1, 2


def test_times_and_timestamps_alias_agree():
    docs, times, k = _planted()
    a = topica.TopicsOverTime(k, seed=2).fit(docs, times=times, iters=30)
    b = topica.TopicsOverTime(k, seed=2).fit(docs, timestamps=times, iters=30)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.topic_time, b.topic_time)


def test_peak_reports_in_original_units():
    docs, times, k = _planted()
    m = topica.TopicsOverTime(k, seed=0).fit(docs, times=times, iters=100)
    lo, hi = m.time_range
    assert lo == pytest.approx(min(times))
    assert hi == pytest.approx(max(times))
    # every finite peak lies inside the observed timestamp range
    peaks = m.topic_time_peak
    finite = peaks[~np.isnan(peaks)]
    assert (finite >= lo - 1e-9).all() and (finite <= hi + 1e-9).all()


def test_constant_time_reduces_to_lda_with_warning():
    docs, _times, k = _planted()
    times = [5.0] * len(docs)  # one timestamp → no temporal signal
    with pytest.warns(UserWarning):
        m = topica.TopicsOverTime(k, seed=0).fit(docs, times=times, iters=40)
    # uniform Beta fallback (psi1=psi2=1) and no defined peak
    assert np.allclose(m.topic_time, 1.0)
    assert np.isnan(m.topic_time_peak).all()
    # topics still recover (the word model is unaffected)
    for row in m.topic_word:
        assert np.sort(row)[-5:].sum() > 0.9


def test_save_load_roundtrip():
    docs, times, k = _planted()
    m = topica.TopicsOverTime(k, seed=7).fit(docs, times=times, iters=30)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tot.bin")
        m.save(path)
        loaded = topica.TopicsOverTime.load(path)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert np.array_equal(m.topic_time, loaded.topic_time)
    assert np.array_equal(np.nan_to_num(m.topic_time_peak),
                          np.nan_to_num(loaded.topic_time_peak))
    assert m.time_range == loaded.time_range


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2).fit([], times=[])


def test_rejects_missing_times():
    with pytest.raises((ValueError, TypeError)):
        topica.TopicsOverTime(2).fit(_toy_docs())


def test_rejects_length_mismatch():
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2).fit(_toy_docs(), times=[0.0, 1.0])


def test_rejects_both_times_and_timestamps():
    docs = _toy_docs()
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2).fit(docs, times=[0, 1, 2, 3], timestamps=[0, 1, 2, 3])


def test_rejects_nonfinite_times():
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2).fit(_toy_docs(), times=[0.0, float("nan"), 1.0, 2.0])


def test_rejects_bad_hyperparams():
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(0)
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2, beta=0.0)
    with pytest.raises((ValueError, RuntimeError)):
        topica.TopicsOverTime(2, alpha=-1.0)


def test_top_words_and_coherence():
    docs, times, k = _planted()
    m = topica.TopicsOverTime(k, seed=0).fit(docs, times=times, iters=50)
    tw = m.top_words(3, topic=0)
    assert len(tw) == 3 and isinstance(tw[0][0], str)
    coh = m.coherence(5)
    assert coh.shape == (k,)
