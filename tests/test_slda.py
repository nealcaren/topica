"""Supervised LDA (Blei & McAuliffe 2007) — topics shaped to predict a
per-document real-valued response. These tests check that the topics separate a
response-driving signal, that the regression coefficients rank topics correctly,
that prediction works on held-out documents, and that outputs/validation behave.
"""

import numpy as np
import pytest

from topica import SupervisedLDA, Corpus

T0 = ["a", "b", "c", "d", "e", "f"]
T1 = ["g", "h", "i", "j", "k", "l"]


def _supervised_corpus(n=200, seed=0):
    """Each doc mixes two disjoint-vocabulary topics in proportion p; the
    response is 2p-1 plus small noise, so topic-0 prevalence drives y up."""
    rng = np.random.default_rng(seed)
    docs, y = [], []
    for _ in range(n):
        p = rng.random()
        doc = [(T0 if rng.random() < p else T1)[rng.integers(6)] for _ in range(20)]
        docs.append(doc)
        y.append(2 * p - 1 + (rng.random() - 0.5) * 0.2)
    return docs, np.array(y)


@pytest.fixture(scope="module")
def fitted():
    docs, y = _supervised_corpus()
    m = SupervisedLDA(num_topics=2, seed=7)
    m.fit(docs, y, iters=25, var_iters=15)
    return m, docs, y


class TestSupervision:
    def test_topics_separate_vocabularies(self, fitted):
        m, _, _ = fitted
        tw = m.topic_word
        vocab = m.vocabulary
        idx0 = [vocab.index(w) for w in T0]
        idx1 = [vocab.index(w) for w in T1]
        # Each topic concentrates on one of the two blocks.
        t0_block = 0 if tw[0][idx0].sum() > tw[1][idx0].sum() else 1
        t1_block = 0 if tw[0][idx1].sum() > tw[1][idx1].sum() else 1
        assert t0_block != t1_block

    def test_coefficients_rank_topics(self, fitted):
        m, _, _ = fitted
        tw = m.topic_word
        vocab = m.vocabulary
        idx0 = [vocab.index(w) for w in T0]
        # The topic owning the T0 vocabulary (which drives y up) should have the
        # larger regression coefficient.
        t0_topic = 0 if tw[0][idx0].sum() > tw[1][idx0].sum() else 1
        coefs = m.coefficients
        assert coefs[t0_topic] > coefs[1 - t0_topic]

    def test_prediction_correlates(self, fitted):
        m, docs, y = fitted
        pred = m.predict(docs)
        assert pred.shape == (len(docs),)
        corr = np.corrcoef(pred, y)[0, 1]
        assert corr > 0.7

    def test_predict_heldout(self, fitted):
        m, _, _ = fitted
        held, yh = _supervised_corpus(n=80, seed=99)
        pred = m.predict(held)
        assert np.corrcoef(pred, yh)[0, 1] > 0.6


class TestUncertainty:
    def test_coefficient_se_shape_and_significance(self, fitted):
        m, _, _ = fitted
        se = m.coefficient_se
        assert se is not None
        assert se.shape == m.coefficients.shape == (2,)
        assert np.all(np.isfinite(se)) and np.all(se > 0.0)
        # On this well-separated corpus both coefficients are far from zero.
        z = m.coefficients / se
        assert np.all(np.abs(z) > 2.0)

    def test_predict_return_std(self, fitted):
        m, docs, _ = fitted
        mean, std = m.predict(docs[:10], return_std=True)
        point = m.predict(docs[:10])
        assert mean.shape == std.shape == (10,)
        np.testing.assert_allclose(mean, point)
        # Predictive SD is at least the residual SD (topic uncertainty adds to it).
        assert np.all(std >= np.sqrt(m.sigma2) - 1e-9)
        assert np.all(np.isfinite(std))

    def test_coefficient_se_survives_save_load(self, fitted, tmp_path):
        m, _, _ = fitted
        path = tmp_path / "slda.topica"
        m.save(str(path))
        m2 = SupervisedLDA.load(str(path))
        np.testing.assert_array_equal(m.coefficient_se, m2.coefficient_se)


class TestOutputs:
    def test_shapes(self, fitted):
        m, docs, _ = fitted
        assert m.topic_word.shape == (2, len(m.vocabulary))
        assert m.doc_topic.shape == (len(docs), 2)
        assert m.coefficients.shape == (2,)

    def test_doc_topic_normalized(self, fitted):
        m, _, _ = fitted
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_sigma2_positive(self, fitted):
        m, _, _ = fitted
        assert m.sigma2 > 0

    def test_top_words_and_coherence(self, fitted):
        m, _, _ = fitted
        assert len(m.top_words(5, topic=0)) == 5
        assert m.coherence().shape == (2,)


class TestApi:
    def test_deterministic(self):
        docs, y = _supervised_corpus()
        a = SupervisedLDA(num_topics=2, seed=3)
        a.fit(docs, y, iters=10, var_iters=10)
        b = SupervisedLDA(num_topics=2, seed=3)
        b.fit(docs, y, iters=10, var_iters=10)
        assert np.array_equal(a.coefficients, b.coefficients)
        assert np.array_equal(a.topic_word, b.topic_word)

    def test_accepts_corpus_object(self):
        docs, y = _supervised_corpus(n=80)
        c = Corpus.from_documents(docs)
        m = SupervisedLDA(num_topics=2, seed=1)
        m.fit(c, y, iters=8)
        assert m.topic_word.shape[0] == 2

    def test_predict_ignores_oov(self, fitted):
        m, _, _ = fitted
        # Unknown words are dropped; a doc of pure T0 vocab still predicts high.
        pred = m.predict([T0 * 3, ["ZZZ", "QQQ"] + T1 * 3])
        assert pred.shape == (2,)

    def test_y_length_mismatch_raises(self):
        docs, y = _supervised_corpus(n=40)
        with pytest.raises(ValueError):
            SupervisedLDA(num_topics=2).fit(docs, y[:-1])

    def test_unfitted_raises(self):
        m = SupervisedLDA(num_topics=2)
        with pytest.raises(RuntimeError):
            _ = m.coefficients

    def test_bad_hyperparams_raise(self):
        with pytest.raises(ValueError):
            SupervisedLDA(num_topics=1)
        with pytest.raises(ValueError):
            SupervisedLDA(num_topics=2, alpha=0.0)
