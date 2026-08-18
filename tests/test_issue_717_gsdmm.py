"""Regression tests for the GSDMM Gate B follow-ups (#717).

Covers the auto-K honesty warnings (F2), the tiny-cluster warning (F7), the
``max_topics`` cap accessor (F9), the edge-case guards (F12), ``find_thoughts``
accepting a model/Corpus (F4), and a stronger planted-recovery check with ARI +
inferred-K tolerance + the in-sample ``doc_topic`` signature (F6). The bit-exact
equivalence of the sparse ``doc_cluster_dist`` rewrite (F11) is locked in the
Rust unit test ``gsdmm::tests::doc_cluster_dist_matches_dense_reference``.
"""

import warnings

import numpy as np
import pytest

import topica


def _disjoint_blocks(seed=0, n_per=80, n_blocks=4):
    """Well-separated short-text corpus: each block has its own 4-word vocabulary,
    so a faithful GSDMM recovers the blocks cleanly. Returns (docs, labels)."""
    rng = np.random.default_rng(seed)
    vocab = [[f"b{b}w{w}" for w in range(4)] for b in range(n_blocks)]
    docs, labels = [], []
    for b in range(n_blocks):
        for _ in range(n_per):
            docs.append([vocab[b][int(rng.integers(4))] for _ in range(3)])
            labels.append(b)
    order = rng.permutation(len(docs))
    return [docs[i] for i in order], np.array([labels[i] for i in order])


# --- F9: max_topics vs num_topics ------------------------------------------

def test_max_topics_reports_the_cap_num_topics_reports_discovered():
    docs, _ = _disjoint_blocks()
    m = topica.GSDMM(num_topics=12, seed=13)
    assert m.max_topics == 12  # the cap, available before fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(docs, iters=50)
    assert m.max_topics == 12  # unchanged: still the cap
    assert m.num_topics <= m.max_topics  # discovered count is the (smaller) inference
    # settings mirrors __init__ (issue #400): the key stays num_topics = the cap.
    assert m.settings["num_topics"] == 12


# --- F2: auto-K honesty warning --------------------------------------------

def _cap_warned(records):
    return any("at or near the num_topics" in str(x.message) for x in records)


def test_warns_when_discovered_count_pins_at_cap():
    # Many distinct 3-token docs over a big vocab with a small cap: the count is
    # limited by the cap, not inferred from the data, so fit() must warn.
    rng = np.random.default_rng(0)
    docs = [[f"w{int(rng.integers(200))}" for _ in range(3)] for _ in range(300)]
    m = topica.GSDMM(num_topics=6, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=30)
    assert m.num_topics >= int(np.ceil(0.9 * m.max_topics))
    assert _cap_warned(w)


def test_no_cap_warning_when_count_settles_below_cap():
    docs, _ = _disjoint_blocks()
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=60)
    assert m.num_topics < m.max_topics
    assert not _cap_warned(w)


def test_no_warnings_when_iters_zero():
    # iters=0 leaves the uniform-random init (num_used==k_max, stray singletons);
    # those are artifacts, not discoveries, so no honesty warning should fire.
    docs, _ = _disjoint_blocks(n_per=20)
    m = topica.GSDMM(num_topics=10, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=0)
    assert not _cap_warned(w)
    assert not any("tiny cluster" in str(x.message).lower() for x in w)


# --- long-document appropriateness warning ---------------------------------

def test_warns_when_documents_are_long():
    # GSDMM is a short-text model; warn when documents are long (multi-topic).
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(80)]
    docs = [[vocab[int(rng.integers(80))] for _ in range(60)] for _ in range(60)]
    m = topica.GSDMM(num_topics=8, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=20)
    assert any("average" in str(x.message) and "short" in str(x.message) for x in w)


def test_no_long_doc_warning_on_short_text():
    docs, _ = _disjoint_blocks()  # 3-token docs
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=40)
    assert not any("single-topic documents" in str(x.message) for x in w)


# --- F7: tiny-cluster warning fires only on real fragmentation -------------

def test_warns_when_fit_fragments_into_tiny_clusters():
    # Fully disjoint singleton-vocab docs: GSDMM keeps them apart, so the fit
    # fragments into many 1-2 doc clusters and must warn.
    docs = [[f"d{i}_a", f"d{i}_b"] for i in range(14)]
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=40)
    assert any("fragmented into" in str(x.message) for x in w)


def test_no_tiny_warning_on_a_healthy_fit():
    # A few big clusters and at most one stray singleton must NOT warn (the old
    # behaviour fired on every stray singleton — that was noise).
    docs, _ = _disjoint_blocks(n_per=80, n_blocks=4)
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, iters=80)
    assert not any("fragmented into" in str(x.message) for x in w)


# --- F12: edge-case guards -------------------------------------------------

def test_empty_corpus_raises_clearly():
    with pytest.raises(ValueError):
        topica.GSDMM(4, seed=1).fit([[], [], []], iters=5)


def test_beta_overflow_raises_directively():
    huge = float(np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="beta.*too large|overflow"):
        topica.GSDMM(4, beta=huge, seed=1).fit([["a", "b"], ["c", "d"]] * 5, iters=5)


# --- F4: find_thoughts accepts a model and a Corpus ------------------------

def test_find_thoughts_accepts_model_and_corpus():
    docs, _ = _disjoint_blocks(n_per=20)
    corpus = topica.Corpus.from_documents(docs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GSDMM(8, seed=13).fit(corpus, iters=50)
    # The natural call: model as first arg, Corpus as texts. Must not crash.
    res = topica.find_thoughts(m, corpus, topic=0, n=3)
    assert len(res) == 3
    idx, prop, text = res[0]
    assert 0.0 <= prop <= 1.0
    assert text is not None  # the Corpus supplied the (tokenized) document
    # A raw doc_topic array + Corpus texts works the same way.
    res2 = topica.find_thoughts(m.doc_topic, corpus, topic=0, n=3)
    assert [r[0] for r in res2] == [r[0] for r in res]


# --- F6: stronger planted recovery -----------------------------------------

def test_planted_recovery_ari_and_inferred_k():
    pytest.importorskip("sklearn")
    from sklearn.metrics import adjusted_rand_score

    docs, labels = _disjoint_blocks(n_per=80, n_blocks=4)
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(docs, iters=80)
    # Hard labels recover the planted blocks on cleanly separable text.
    ari = adjusted_rand_score(labels, np.asarray(m.doc_cluster))
    assert ari >= 0.9, f"planted ARI {ari:.3f} too low"
    # The inferred count lands at/near the planted 4 (not the 15 cap).
    assert 4 <= m.num_topics <= 7, f"discovered K={m.num_topics} off planted 4"


def test_doc_topic_is_in_sample_and_over_peaked():
    # doc_topic is the in-sample Eq. 4 plug-in: it is heavily over-peaked and its
    # argmax is not guaranteed to equal the hard doc_cluster (F1/F8 signature).
    docs, _ = _disjoint_blocks(n_per=80, n_blocks=4)
    m = topica.GSDMM(num_topics=15, seed=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(docs, iters=80)
    theta = np.asarray(m.doc_topic)
    assert np.median(theta.max(axis=1)) > 0.9  # over-peaked, as documented
    # Rows still sum to 1 (a valid distribution over discovered clusters).
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-9)
