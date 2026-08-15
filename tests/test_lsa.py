"""LSA / LSI on topica's truncated-SVD core (reusing NMF's randomized SVD): it
recovers planted block structure, exposes the standard fitted surface plus signed
singular_values, ranks top words by absolute loading, round-trips through
save/load, is deterministic under a seed, supports both weightings, and validates
its inputs. Unlike the probabilistic models, doc_topic rows are SIGNED and do not
sum to 1."""

import numpy as np
import pytest

import topica


def _planted(k=3, block=8, n=240, length=15, seed=0):
    """K word-blocks; each document draws its tokens from one block. Returns
    (docs, vocab)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs = []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(length)])
    return docs, vocab


def test_construction_defaults():
    m = topica.LSA(3)
    assert m.num_topics == 3
    assert "LSA(num_topics=3" in repr(m)


def test_fit_recovers_planted_blocks():
    docs, vocab = _planted()
    m = topica.LSA(3)
    m.fit(docs)

    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    # doc_topic is signed coordinates, NOT a simplex: rows generally do not sum to 1.
    assert not np.allclose(m.doc_topic.sum(axis=1), 1.0)

    # Each component's top-|loading| words come from a single planted block; all
    # blocks covered. (LSA components are signed/orthogonal, so we check block
    # purity of the dominant absolute loadings.)
    vocab_arr = list(m.vocabulary)
    covered = set()
    tw = m.topic_word
    for t in range(3):
        order = np.argsort(np.abs(tw[t]))[::-1][:4]
        blocks = {int(vocab_arr[w].split("w")[0][1:]) for w in order}
        assert len(blocks) == 1, f"component {t} top loadings mix blocks"
        covered.add(next(iter(blocks)))
    assert covered == {0, 1, 2}


def test_fitted_surface():
    docs, vocab = _planted()
    m = topica.LSA(3)
    m.fit(docs)

    assert isinstance(m.topic_word, np.ndarray)
    assert isinstance(m.doc_topic, np.ndarray)
    assert list(m.topic_names) == ["topic_0", "topic_1", "topic_2"]
    assert sorted(m.vocabulary) == sorted(set(vocab))
    assert len(m.doc_names) == len(docs)
    # Direct solve: no iterative trace, no convergence flag.
    assert m.fit_history == []
    assert m.converged is None
    coh = m.coherence(5)
    assert coh.shape == (3,)


def test_singular_values():
    docs, _ = _planted()
    m = topica.LSA(4)
    m.fit(docs)
    sv = m.singular_values
    assert sv.shape == (4,)
    # Non-increasing and non-negative.
    assert np.all(np.diff(sv) <= 1e-9)
    assert np.all(sv >= -1e-12)


def test_top_words():
    docs, _ = _planted()
    m = topica.LSA(3)
    m.fit(docs)
    allw = m.top_words(5)
    assert len(allw) == 3
    assert all(len(row) == 5 for row in allw)
    one = m.top_words(5, topic=0, weights=True)
    assert len(one) == 5
    assert all(isinstance(w, str) and isinstance(v, float) for w, v in one)
    # Ranked by ABSOLUTE loading: the magnitudes are non-increasing.
    mags = [abs(v) for _, v in one]
    assert mags == sorted(mags, reverse=True)
    with pytest.raises(Exception):
        m.top_words(5, topic=99)


def test_save_load_roundtrip(tmp_path):
    docs, _ = _planted()
    m = topica.LSA(3, weighting="count", seed=7)
    m.fit(docs)
    path = str(tmp_path / "lsa.bin")
    m.save(path)

    loaded = topica.LSA.load(path)
    assert loaded.num_topics == 3
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
    assert np.array_equal(loaded.singular_values, m.singular_values)
    assert list(loaded.topic_names) == list(m.topic_names)


@pytest.mark.parametrize("weighting", ["tfidf", "count"])
def test_weighting_values(weighting):
    docs, vocab = _planted()
    m = topica.LSA(3, weighting=weighting)
    m.fit(docs)
    assert m.topic_word.shape == (3, len(set(vocab)))
    assert m.singular_values.shape == (3,)


def test_determinism_same_seed():
    docs, _ = _planted()
    for weighting in ("tfidf", "count"):
        a = topica.LSA(3, weighting=weighting, seed=11)
        a.fit(docs)
        b = topica.LSA(3, weighting=weighting, seed=11)
        b.fit(docs)
        assert np.array_equal(a.topic_word, b.topic_word)
        assert np.array_equal(a.doc_topic, b.doc_topic)
        assert np.array_equal(a.singular_values, b.singular_values)


def test_input_validation():
    with pytest.raises(Exception):
        topica.LSA(1)  # K < 2
    with pytest.raises(Exception):
        topica.LSA(3, weighting="nonsense")

    # K > min(num_docs, vocab): vocabulary size 6 here.
    docs, _ = _planted(k=3, block=2)
    m = topica.LSA(20)
    with pytest.raises(Exception):
        m.fit(docs)

    empty = topica.LSA(3)
    with pytest.raises(Exception):
        empty.fit([])


def test_num_threads_is_deterministic_resource_knob():
    """num_threads bounds the truncated-SVD matmul pool; LSA is a direct solve, so
    the output must be identical across worker counts. None/0 = all cores."""
    docs, _vocab = _planted(seed=2)
    base = topica.LSA(3, seed=1).fit(docs, num_threads=1)
    for nt in (2, 4, 8, 0, None):
        m = topica.LSA(3, seed=1).fit(docs, num_threads=nt)
        assert np.array_equal(base.topic_word, m.topic_word), f"num_threads={nt}"
