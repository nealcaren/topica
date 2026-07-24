"""Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024).

STS extends STM with a per-document, per-topic continuous sentiment-discourse
latent driven by covariates. These tests exercise the Python binding end to end:
shapes, topic recovery, the sentiment outputs, the covariate regressions, and the
shared analysis surface.
"""

import numpy as np
import pytest

import topica


A = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
B = ["star", "moon", "sky", "sun", "comet", "orbit"]


def _planted(n=84, seed=0):
    """Two disjoint-vocabulary topics; each document is drawn from one. The
    prevalence covariate is the topic indicator; the sentiment seed is a separate
    3-level signal *independent* of the topics (so prevalence and sentiment are
    not confounded — using the topic indicator for both is unidentifiable)."""
    rng = np.random.default_rng(seed)
    docs, sent_seed, prev, truth = [], [], [], []
    for d in range(n):
        a = d % 2 == 0
        vocab = A if a else B
        docs.append([vocab[int(rng.integers(len(vocab)))] for _ in range(12)])
        sent_seed.append(float(d % 3))  # 3 aggregation groups, orthogonal to topic
        prev.append([0.0 if a else 1.0])
        truth.append(0 if a else 1)
    return docs, sent_seed, prev, np.array(truth)


def _fit(**kw):
    docs, sent_seed, prev, truth = _planted()
    m = topica.STS(num_topics=2, seed=1)
    m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=kw.get("iters", 25))
    return m, docs, truth


class TestFit:
    def test_shapes(self):
        m, docs, _ = _fit()
        v = len(set(w for d in docs for w in d))
        assert m.topic_word.shape == (2, v)
        assert m.doc_topic.shape == (len(docs), 2)
        assert m.sentiment.shape == (len(docs), 2)
        assert m.prevalence_effects.shape == (2, 1)  # (intercept+indicator) x (K-1)
        assert m.sentiment_effects.shape == (2, 2)    # (intercept+indicator) x K
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)

    def test_recovers_topics(self):
        m, docs, truth = _fit()
        # Map each topic to a planted block by its actual top words (the corpus
        # assigns vocabulary ids in its own order, so word strings are the anchor).
        top_words = [{w for w, _ in m.top_words(4)[t]} for t in range(2)]
        a_set = set(A)
        topic_for_a = 0 if len(top_words[0] & a_set) >= len(top_words[1] & a_set) else 1
        theta = np.asarray(m.doc_topic)
        dominant = (theta[:, 1] > theta[:, 0]).astype(int)
        expected = np.where(truth == 0, topic_for_a, 1 - topic_for_a)
        assert (dominant == expected).mean() > 0.9

    def test_sentiment_and_bound(self):
        m, _, _ = _fit()
        # Sentiment latents are non-trivial and the bound trajectory is recorded.
        assert np.abs(np.asarray(m.sentiment)).max() > 1e-6
        assert len(m.bound_history) >= 1
        assert np.isfinite(m.bound)

    def test_topic_word_at_levels(self):
        m, _, _ = _fit()
        for level in (-2.0, 0.0, 2.0):
            b = np.asarray(m.topic_word_at(level))
            assert b.shape == m.topic_word.shape
            np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-6)

    def test_deterministic(self):
        docs, sent_seed, prev, _ = _planted()
        m1 = topica.STS(num_topics=2, seed=7)
        m2 = topica.STS(num_topics=2, seed=7)
        m1.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15)
        m2.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15)
        np.testing.assert_allclose(m1.topic_word, m2.topic_word)
        np.testing.assert_allclose(m1.sentiment, m2.sentiment)


class TestAnalysisSurface:
    def test_flows_into_coherence_and_top_words(self):
        m, docs, _ = _fit()
        cv = topica.coherence(m, docs, coherence_type="c_v", topn=4)
        assert np.asarray(cv).shape == (2,)
        rows = m.top_words(3)
        assert len(rows) == 2 and all(isinstance(w, str) for w, _ in rows[0])

    def test_no_prevalence_still_fits(self):
        docs, sent_seed, _, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, iters=15)  # no prevalence design
        assert m.doc_topic.shape == (len(docs), 2)
        assert m.sentiment.shape == (len(docs), 2)


class TestPersistenceAndTransform:
    def test_doc_names(self):
        docs, sent_seed, prev, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15)
        assert m.doc_names == [f"doc_{i}" for i in range(len(docs))]

    def test_transform_shape_and_normalization(self):
        docs, sent_seed, prev, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=20)
        held = [A[:4], B[:4], ["???", "out-of-vocab"]]
        th = m.transform(held)
        assert th.shape == (3, 2)
        np.testing.assert_allclose(th.sum(axis=1), 1.0, atol=1e-9)
        # an all-OOV document falls back to a uniform prevalence
        np.testing.assert_allclose(th[2], [0.5, 0.5], atol=1e-9)
        # a clearly topic-A document leans to the topic that owns the A block
        a_top = max(range(2), key=lambda t: th[0, t])
        b_top = max(range(2), key=lambda t: th[1, t])
        assert a_top != b_top

    def test_save_load_round_trip(self, tmp_path):
        docs, sent_seed, prev, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=20)
        p = str(tmp_path / "model.sts")
        m.save(p)
        m2 = topica.STS.load(p)
        for attr in ("topic_word", "doc_topic", "sentiment", "eta_mean", "eta_cov"):
            np.testing.assert_array_equal(
                np.asarray(getattr(m, attr)), np.asarray(getattr(m2, attr))
            )
        assert m2.prevalence_effects.shape == m.prevalence_effects.shape
        # the reloaded model transforms identically
        held = [A[:4], B[:4]]
        np.testing.assert_allclose(m2.transform(held), m.transform(held), atol=1e-12)


class TestErrors:
    def test_num_topics_too_small(self):
        with pytest.raises(ValueError):
            topica.STS(num_topics=1)

    def test_seed_length_mismatch(self):
        docs, _, prev, _ = _planted()
        m = topica.STS(num_topics=2)
        with pytest.raises(ValueError, match="sentiment_seed"):
            m.fit(docs, sentiment_seed=[0.0, 1.0], prevalence=prev)

    def test_effects_require_prevalence(self):
        docs, sent_seed, _, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, iters=10)
        with pytest.raises(RuntimeError, match="prevalence"):
            _ = m.prevalence_effects

    def test_getters_require_fit(self):
        m = topica.STS(num_topics=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_word


def _sep(m, truth):
    tw = m.topic_word
    a_topic = 0 if int(np.argmax(tw[0])) < len(A) else 1
    dom = np.asarray(m.doc_topic).argmax(1)
    expected = np.where(truth == 0, a_topic, 1 - a_topic)
    return float((dom == expected).mean())


class TestKappaEstimationAndReference:
    """The κ estimators ("ridge"/"lasso"/"adjusted") and the reference-fidelity
    profiles ("paper"/"cran"), added for #454."""

    @pytest.mark.parametrize("mode", ["ridge", "lasso", "adjusted"])
    def test_each_estimator_fits_and_separates(self, mode):
        docs, sent_seed, prev, truth = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=25, kappa_estimation=mode)
        assert np.isfinite(m.topic_word).all()
        assert _sep(m, truth) > 0.9

    @pytest.mark.parametrize("ref", ["paper", "cran"])
    def test_reference_profiles_fit_finite_and_separate(self, ref):
        docs, sent_seed, prev, truth = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=25, reference=ref)
        assert np.isfinite(m.topic_word).all()
        assert np.isfinite(m.sentiment).all()
        assert _sep(m, truth) > 0.9

    def test_cran_profile_changes_the_fit(self):
        # The reference profile must actually alter the solution vs the topica-native
        # default (adjusted aggregation + damping + reference init).
        docs, sent_seed, prev, _ = _planted()
        base = topica.STS(num_topics=2, seed=1)
        base.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=20)  # ridge, none
        cran = topica.STS(num_topics=2, seed=1)
        cran.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=20, reference="cran")
        assert not np.allclose(base.topic_word, cran.topic_word)

    def test_reference_profile_is_deterministic(self):
        docs, sent_seed, prev, _ = _planted()
        m1 = topica.STS(num_topics=2, seed=1)
        m2 = topica.STS(num_topics=2, seed=1)
        m1.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15, reference="cran")
        m2.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15, reference="cran")
        assert np.allclose(m1.topic_word, m2.topic_word)

    def test_bad_estimator_value_raises(self):
        docs, sent_seed, prev, _ = _planted()
        with pytest.raises(ValueError, match="kappa_estimation must be"):
            topica.STS(2).fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=2,
                              kappa_estimation="bogus")

    def test_bad_reference_value_raises(self):
        docs, sent_seed, prev, _ = _planted()
        with pytest.raises(ValueError, match="reference must be"):
            topica.STS(2).fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=2,
                              reference="bogus")

    @pytest.mark.parametrize(
        "ref,bad_kappa",
        [
            ("cran", "lasso"),
            ("paper", "adjusted"),
            # An explicit "ridge" must also conflict: because kappa_estimation is a
            # sentinel Option, an explicit "ridge" is distinguishable from unset and
            # is not silently overwritten by the profile's estimator.
            ("cran", "ridge"),
            ("paper", "ridge"),
        ],
    )
    def test_conflicting_estimator_and_profile_raises(self, ref, bad_kappa):
        docs, sent_seed, prev, _ = _planted()
        with pytest.raises(ValueError, match="estimator"):
            topica.STS(2).fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=2,
                              reference=ref, kappa_estimation=bad_kappa)

    @pytest.mark.parametrize("ref,ok_kappa", [("cran", "adjusted"), ("paper", "lasso")])
    def test_matching_explicit_estimator_and_profile_is_allowed(self, ref, ok_kappa):
        # Passing the profile's own estimator explicitly is redundant but not a
        # conflict, so it must fit rather than raise.
        docs, sent_seed, prev, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=10,
              reference=ref, kappa_estimation=ok_kappa)
        assert np.isfinite(m.topic_word).all()
