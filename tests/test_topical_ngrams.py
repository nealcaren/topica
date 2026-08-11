"""Topical N-Grams (Wang, McCallum & Wei 2007): joint topic + phrase discovery.

Checks phrase recovery on a planted-collocation corpus, that phrases include the
head word (not just the x=1 run), boundary handling (a dropped token breaks a
phrase), reproducibility, the tamer-vs-MALLET default behaviour, and save/load.
"""
import numpy as np
import pytest

import topica


def _planted(seed=0, n_docs=250, doc_len=40):
    """Each doc mixes filler unigrams with two-word collocations; the collocation
    words never appear alone, so a faithful TNG surfaces them as phrases."""
    rng = np.random.default_rng(seed)
    collocs = [("machine", "learning"), ("neural", "network"),
               ("climate", "change"), ("supreme", "court")]
    filler = [f"f{i}" for i in range(20)]
    docs = []
    for _ in range(n_docs):
        doc = []
        while len(doc) < doc_len:
            if rng.random() < 0.5:
                c = collocs[rng.integers(len(collocs))]
                doc.extend([c[0], c[1]])
            else:
                doc.append(filler[rng.integers(len(filler))])
        docs.append(doc)
    return docs, collocs


def test_recovers_planted_collocations():
    docs, collocs = _planted(seed=1)
    m = topica.TopicalNGrams(num_topics=6, seed=13).fit(docs, iters=300)
    top = {p for p, _ in m.top_phrases(15)}
    planted = {f"{a} {b}" for a, b in collocs}
    found = len(planted & top)
    assert found >= 3, f"recovered only {found}/4 planted collocations: {sorted(top)[:8]}"
    # phrases carry both words (the head is included, not dropped)
    assert all(len(p.split()) >= 2 for p, _ in m.top_phrases(10))


def test_standard_topic_surface():
    docs, _ = _planted(seed=2)
    m = topica.TopicalNGrams(num_topics=5, seed=13).fit(docs, iters=100)
    assert m.topic_word.shape == (5, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 5)
    # topic_word rows are a proper simplex
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    # top_words and top_phrases are separate views
    assert isinstance(m.top_words(5), list)
    assert m.num_phrases > 0
    # coherence returns one score per topic
    assert m.coherence(10).shape == (5,)


def test_deterministic():
    docs, _ = _planted(seed=3)
    a = topica.TopicalNGrams(num_topics=5, seed=13).fit(docs, iters=100)
    b = topica.TopicalNGrams(num_topics=5, seed=13).fit(docs, iters=100)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert a.top_phrases(10) == b.top_phrases(10)


def test_pruned_word_excluded_from_phrases():
    # A word pruned by min_count is out of vocabulary and can never appear in a
    # phrase (and, by breaking adjacency, cannot glue its neighbours together).
    docs = [["alpha", "rare_singleton", "beta"]]  # rare_singleton occurs once
    docs += [["alpha", "beta", "gamma", "gamma"] for _ in range(20)]
    m = topica.TopicalNGrams(num_topics=2, seed=13, min_count=2).fit(docs, iters=100)
    assert "rare_singleton" not in m.vocabulary
    phrases = {p for p, _ in m.top_phrases(20)}
    assert not any("rare_singleton" in p for p in phrases)


def test_per_topic_vs_global_phrases():
    docs, _ = _planted(seed=4)
    m = topica.TopicalNGrams(num_topics=6, seed=13).fit(docs, iters=200)
    glob = m.top_phrases(20)
    # global phrases are unique (pooled across topics)
    texts = [p for p, _ in glob]
    assert len(texts) == len(set(texts))
    # a per-topic query returns that topic's phrases (subset of the vocabulary of phrases)
    t0 = m.top_phrases(10, topic=0)
    assert all(isinstance(p, str) and isinstance(w, float) for p, w in t0)


def test_tamer_default_vs_mallet_prior():
    # The balanced default discovers discrete collocations; MALLET's 0.2/1000 forces
    # long runs, so its top "phrases" are much longer on average.
    docs, _ = _planted(seed=5)
    tame = topica.TopicalNGrams(num_topics=6, seed=13).fit(docs, iters=200)
    mallet = topica.TopicalNGrams(num_topics=6, seed=13, delta1=0.2, delta2=1000.0).fit(
        docs, iters=200
    )
    tame_len = np.mean([len(p.split()) for p, _ in tame.top_phrases(10)])
    mallet_len = np.mean([len(p.split()) for p, _ in mallet.top_phrases(10)])
    assert mallet_len > tame_len, f"mallet {mallet_len} not longer than tame {tame_len}"


def test_save_load_round_trip(tmp_path):
    docs, _ = _planted(seed=6)
    m = topica.TopicalNGrams(num_topics=5, seed=13).fit(docs, iters=100)
    p = tmp_path / "tng.model"
    m.save(str(p))
    loaded = topica.TopicalNGrams.load(str(p))
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert loaded.top_phrases(10) == m.top_phrases(10)
    assert loaded.vocabulary == m.vocabulary
    assert loaded.settings == m.settings


def test_settings_keys():
    m = topica.TopicalNGrams(10, alpha_sum=20.0, delta1=0.5)
    assert set(m.settings) == {
        "num_topics", "alpha_sum", "beta", "gamma", "delta1", "delta2",
        "min_count", "seed",
    }
    assert m.settings["seed"] == 13
    assert m.settings["num_topics"] == 10


def test_bad_hyperparameters_rejected():
    with pytest.raises(ValueError):
        topica.TopicalNGrams(10, delta1=0.0)
    with pytest.raises(ValueError):
        topica.TopicalNGrams(0)
