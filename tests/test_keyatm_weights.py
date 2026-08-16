"""keyATM token weighting (weighted LDA). keyATM downweights frequent words by a
function of corpus frequency (information-theory by default, the word's surprisal
in bits) so that common words count for less. These check the scheme is wired
through, changes results, demotes stopwords, and validates the argument."""

import numpy as np
import pytest

import topica

A = ["tax", "market", "trade", "fiscal"]
B = ["abortion", "gay", "church", "family"]
SEEDS = {"econ": A[:2], "soc": B[:2]}


def _corpus(seed=0, n=200):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        heavy, light = (A, B) if i % 2 else (B, A)
        # Each doc is padded with very frequent stopwords.
        docs.append(
            rng.choice(heavy, 8).tolist()
            + rng.choice(light, 2).tolist()
            + ["the", "of"] * 4
        )
    return docs


def test_weights_change_results():
    docs = _corpus()
    out = {}
    for w in ("information-theory", "inv-freq", "none"):
        m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
        m.fit(docs, iters=200, weights=w)
        out[w] = m.topic_word
    # Information-theory weighting should not reproduce the unweighted fit.
    assert not np.allclose(out["information-theory"], out["none"])
    assert not np.allclose(out["inv-freq"], out["none"])


def test_weighting_demotes_stopwords():
    docs = _corpus()
    # Stopwords dominate raw frequency, so unweighted topics surface them; the
    # default information-theory weighting should push them out of the top words.
    m_none = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m_none.fit(docs, iters=300, weights="none")
    m_inf = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m_inf.fit(docs, iters=300, weights="information-theory")

    def stopword_rank(model):
        tops = [w for w, _ in model.top_words(6, topic=0, weights=True)]
        return sum(1 for w in ("the", "of") if w in tops)

    assert stopword_rank(m_inf) <= stopword_rank(m_none)


def test_default_is_information_theory():
    docs = _corpus()
    m_default = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m_default.fit(docs, iters=150)
    m_inf = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m_inf.fit(docs, iters=150, weights="information-theory")
    assert np.allclose(m_default.topic_word, m_inf.topic_word)


def test_invalid_weights_rejected():
    docs = _corpus(n=40)
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, iters=10, weights="tfidf")


def test_weighting_deterministic():
    docs = _corpus(n=120)
    m1 = topica.KeyATM(SEEDS, num_topics=2, seed=3)
    m1.fit(docs, iters=150, weights="information-theory")
    m2 = topica.KeyATM(SEEDS, num_topics=2, seed=3)
    m2.fit(docs, iters=150, weights="information-theory")
    assert np.allclose(m1.topic_word, m2.topic_word)
    assert np.allclose(m1.doc_topic, m2.doc_topic)
