"""CVB0 backend for the base keyATM model (sampler="cvb0").

CVB0 is a deterministic, non-MCMC alternative to keyATM's collapsed-Gibbs
sampler: each (document, word) cell keeps a soft responsibility over the
(topic, keyword-switch) states. It is *not* the R-keyATM estimator (it does not
preserve R-parity), so it is an opt-in backend; these tests check it recovers a
keyword topic, produces valid outputs, is deterministic, and is restricted to
the base model (covariate/dynamic variants stay Gibbs-only).
"""

import numpy as np
import numpy.testing as npt
import pytest

import topica

_BS, _NB = 10, 3  # block size, num blocks


def _corpus():
    docs = []
    for b in range(_NB):
        for d in range(100):
            doc = [(b * _BS + (i + d) % _BS) for i in range(5)]
            doc.append(((b + 1) % _NB) * _BS + d % _BS)  # one noise token
            docs.append([f"w{x}" for x in doc])
    keywords = {"A": ["w0", "w1"], "B": ["w10", "w11"]}
    return docs, keywords


def _fit(sampler="cvb0", seed=42, iters=150, **kw):
    docs, keywords = _corpus()
    m = topica.models.KeyATM(keywords, num_topics=3, beta=0.1, beta_keyword=0.5, seed=seed,
                      sampler=sampler, **kw)
    m.fit(docs, iters=iters)
    return m


def test_recovers_keyword_topic():
    # Topic 0's keyword set is block A (words w0..w9); its regular distribution
    # should be dominated by block-A words.
    m = _fit()
    vocab = m.vocabulary
    block_a = sum(m.topic_word[0][vocab.index(f"w{i}")] for i in range(_BS) if f"w{i}" in vocab)
    assert block_a > 0.5


def test_valid_distributions_and_no_draws():
    m = _fit()
    npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
    npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.theta_draws is None
    assert len(m.keyword_rate) == 3


def test_deterministic():
    a = _fit(seed=4, iters=80)
    b = _fit(seed=4, iters=80)
    npt.assert_array_equal(a.topic_word, b.topic_word)
    npt.assert_array_equal(a.doc_topic, b.doc_topic)


def test_matches_gibbs_topic_structure():
    # CVB0 is a different estimator, but on the same corpus it should land each
    # keyword topic on the same block as the Gibbs sampler (the dominant block
    # per topic agrees).
    docs, _ = _corpus()
    vocab_blocks = lambda m: [
        int(np.argmax([
            sum(m.topic_word[k][m.vocabulary.index(f"w{b*_BS+i}")]
                for i in range(_BS) if f"w{b*_BS+i}" in m.vocabulary)
            for b in range(_NB)
        ]))
        for k in range(2)
    ]
    assert vocab_blocks(_fit("sparse")) == vocab_blocks(_fit("cvb0"))


def test_aliases_and_bad_name():
    for name in ("cvb0", "cvb"):
        m = _fit(sampler=name, iters=20)
        assert m.num_topics == 3
    with pytest.raises(ValueError):
        topica.models.KeyATM({"A": ["w0"]}, num_topics=2, sampler="banana")


def test_cvb0_rejects_covariate_and_dynamic():
    docs, keywords = _corpus()
    m = topica.models.KeyATM(keywords, num_topics=3, seed=1, sampler="cvb0")
    with pytest.raises(ValueError):
        m.fit(docs, iters=10, covariates=np.ones((len(docs), 1)))
    m2 = topica.models.KeyATM(keywords, num_topics=3, seed=1, sampler="cvb0")
    with pytest.raises(ValueError):
        m2.fit(docs, iters=10, timestamps=list(range(len(docs))))
