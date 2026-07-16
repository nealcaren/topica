"""LDAvis relevance + pyLDAvis export, Taddy residual check, and topic
alignment / stability utilities (topica.stm)."""

import numpy as np
import pytest

from topica.models import LDA
from topica import stm, validation as diagnostics


@pytest.fixture(scope="module")
def two_topic():
    docs = [["cat", "dog", "pet", "cat", "dog"]] * 40 + [["star", "moon", "sky", "star", "moon"]] * 40
    m = LDA(num_topics=2, seed=1)
    m.fit(docs, iters=400)
    return m, docs


class TestRelevance:
    def test_lambda_one_is_probability_order(self, two_topic):
        m, _ = two_topic
        # lambda=1 -> rank by p(w|t); should select the same top words as prob.
        rel = stm.relevance(m.topic_word, m.vocabulary, lam=1.0, n=3)
        probs = m.top_words(3)
        for t in range(2):
            assert {w for w, _ in rel[t]} == {w for w, _ in probs[t]}

    def test_single_topic_and_range(self, two_topic):
        m, _ = two_topic
        one = stm.relevance(m.topic_word, m.vocabulary, topic=0, n=3)
        assert len(one) == 3
        with pytest.raises(ValueError):
            stm.relevance(m.topic_word, m.vocabulary, topic=9)

    def test_term_frequency_marginal(self, two_topic):
        m, docs = two_topic
        tf = np.zeros(len(m.vocabulary))
        vindex = {w: i for i, w in enumerate(m.vocabulary)}
        for d in docs:
            for w in d:
                tf[vindex[w]] += 1
        rel = stm.relevance(m.topic_word, m.vocabulary, lam=0.4, n=3, term_frequency=tf)
        assert len(rel) == 2


class TestPyLDAvis:
    def test_inputs_well_formed(self, two_topic):
        m, docs = two_topic
        viz = stm.prepare_pyldavis(m, docs)
        # pyLDAvis isn't installed in CI -> returns the inputs container.
        assert isinstance(viz, stm.PyLDAvisInputs)
        assert viz.topic_term_dists.shape == (2, len(m.vocabulary))
        assert viz.doc_topic_dists.shape == (80, 2)
        assert viz.doc_lengths.shape == (80,)
        assert viz.term_frequency.shape == (len(m.vocabulary),)
        # topic-term and doc-topic rows are distributions.
        np.testing.assert_allclose(viz.topic_term_dists.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(viz.doc_topic_dists.sum(axis=1), 1.0, atol=1e-9)
        assert tuple(viz.unpack()[0].shape) == (2, len(m.vocabulary))

    def test_doc_mismatch_raises(self, two_topic):
        m, docs = two_topic
        with pytest.raises(ValueError):
            stm.prepare_pyldavis(m, docs[:-1])


class TestCheckResiduals:
    def test_returns_finite_dispersion(self, two_topic):
        m, docs = two_topic
        rc = stm.check_residuals(m, docs)
        assert np.isfinite(rc.dispersion)
        assert 0.0 <= rc.pvalue <= 1.0
        assert rc.df > 0

    def test_chisq_sf_matches_known_values(self):
        # chi-square survival function sanity checks.
        assert abs(diagnostics._chisq_sf(0.0, 4) - 1.0) < 1e-9
        # median of chi^2_4 is ~3.357 -> SF ~ 0.5
        assert abs(diagnostics._chisq_sf(3.357, 4) - 0.5) < 0.02
        assert diagnostics._chisq_sf(1000.0, 10) < 1e-6

    def test_doc_mismatch_raises(self, two_topic):
        m, docs = two_topic
        with pytest.raises(ValueError):
            stm.check_residuals(m, docs[:10])


class TestAlignment:
    def test_matches_swapped_topics(self, two_topic):
        m, docs = two_topic
        b = LDA(num_topics=2, seed=2)
        b.fit(docs, iters=400)
        pairs = stm.align_topics(m, b)
        assert len(pairs) == 2
        # one-to-one: each a-topic and b-topic used once.
        assert sorted(i for i, _, _ in pairs) == [0, 1]
        assert sorted(j for _, j, _ in pairs) == [0, 1]
        # matched topics are near-identical: two independent fits of the same
        # well-identified corpus converge to the same topics (to ~1e-6), though
        # not bit-for-bit across different seeds.
        assert all(dist < 1e-4 for _, _, dist in pairs)

    def test_js_metric(self, two_topic):
        m, docs = two_topic
        b = LDA(num_topics=2, seed=3)
        b.fit(docs, iters=400)
        pairs = stm.align_topics(m, b, metric="js")
        assert len(pairs) == 2

    def test_vocab_mismatch_raises(self):
        a = np.array([[0.5, 0.5], [0.5, 0.5]])
        b = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError):
            stm.align_topics(a, b)

    def test_hungarian_optimal(self):
        # Cost matrix whose optimal assignment is the anti-diagonal.
        cost = np.array([[9.0, 1.0], [1.0, 9.0]])
        pairs = diagnostics._hungarian(cost)
        assert sorted(pairs) == [(0, 1), (1, 0)]
        cost2 = np.array([[1.0, 9.0], [9.0, 1.0]])
        assert sorted(diagnostics._hungarian(cost2)) == [(0, 0), (1, 1)]


class TestStability:
    def test_identical_fits_perfectly_stable(self, two_topic):
        m, docs = two_topic
        b = LDA(num_topics=2, seed=2)
        b.fit(docs, iters=400)
        s = stm.topic_stability([m, b], topn=3)
        assert s == 1.0  # the two clean topics recur exactly

    def test_in_unit_range(self, two_topic):
        m, docs = two_topic
        runs = []
        for seed in (1, 2, 3):
            r = LDA(num_topics=2, seed=seed)
            r.fit(docs, iters=300)
            runs.append(r)
        s = stm.topic_stability(runs, topn=3)
        assert 0.0 <= s <= 1.0

    def test_needs_two_runs(self, two_topic):
        m, _ = two_topic
        with pytest.raises(ValueError):
            stm.topic_stability([m])
