"""Universal `transform` / held-out inference across model families.

Every model that exposes ``transform`` should:
  * accept a Corpus or list[list[str]] of *new* documents,
  * return a ``(num_docs, num_topics)`` array whose rows sum to 1,
  * drop out-of-vocabulary tokens (and give an all-OOV doc the prior),
  * separate documents drawn from clearly distinct vocabularies.

The variational models (CTM/STM) additionally reproduce their own training θ
when transform is run on the training documents — the held-out E-step is the
same Laplace variational inference used at fit time.
"""

import numpy as np
import pytest

import topica


A = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
B = ["star", "moon", "sky", "sun", "comet", "orbit"]


def _two_topic_corpus(n=120, seed=0):
    rng = np.random.default_rng(seed)
    docs, is_a = [], []
    for _ in range(n):
        a = rng.random() < 0.5
        v = A if a else B
        docs.append([v[int(rng.integers(len(v)))] for _ in range(12)])
        is_a.append(a)
    return docs, np.array(is_a)


NEW = [
    ["cat", "dog", "pet", "vet"],   # topic A
    ["moon", "sun", "orbit", "comet"],  # topic B
    ["zzz_oov_only"],               # all out-of-vocabulary
]


def _check_basic(theta, k=2):
    assert theta.shape == (3, k)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-6)
    # The two in-vocabulary docs load on different topics.
    assert theta[0].argmax() != theta[1].argmax()
    # The all-OOV doc falls back to a near-uniform prior.
    assert abs(theta[2].max() - theta[2].min()) < 0.4


class TestVariationalTransform:
    def test_ctm(self):
        docs, _ = _two_topic_corpus()
        m = topica.models.CTM(num_topics=2, seed=1)
        m.fit(docs, iters=60)
        _check_basic(m.transform(NEW))

    def test_ctm_reproduces_training_theta(self):
        # The held-out E-step on the training docs matches the stored θ.
        docs, _ = _two_topic_corpus()
        m = topica.models.CTM(num_topics=2, seed=1)
        m.fit(docs, iters=60)
        np.testing.assert_allclose(m.transform(docs), m.doc_topic, atol=1e-3)

    def test_stm_prevalence(self):
        docs, is_a = _two_topic_corpus()
        x = is_a.astype(float).reshape(-1, 1)
        m = topica.models.STM(num_topics=2, seed=1)
        m.fit(docs, prevalence=x)
        _check_basic(m.transform(NEW))

    def test_save_load_parity(self, tmp_path):
        docs, _ = _two_topic_corpus()
        m = topica.models.CTM(num_topics=2, seed=1)
        m.fit(docs, iters=40)
        p = str(tmp_path / "ctm.tt")
        m.save(p)
        loaded = topica.models.CTM.load(p)
        np.testing.assert_array_equal(m.transform(NEW), loaded.transform(NEW))


class TestGibbsTransform:
    def test_lda(self):
        docs, _ = _two_topic_corpus()
        m = topica.models.LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        _check_basic(m.transform(NEW))

    def test_hdp(self):
        docs, _ = _two_topic_corpus()
        # Default 0.1/0.1 is tuned for real-scale corpora; this small two-topic
        # corpus needs explicit concentration to instantiate both topics.
        m = topica.models.HDP(seed=1, alpha=1.0, gamma=1.0)
        m.fit(docs, iters=300)
        theta = m.transform(NEW)
        k = m.num_topics
        assert theta.shape == (3, k)
        np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-6)
        assert theta[0].argmax() != theta[1].argmax()

    def test_labeled_lda(self):
        docs, is_a = _two_topic_corpus()
        labels = [["animal"] if a else ["space"] for a in is_a]
        m = topica.models.LabeledLDA(seed=1)
        m.fit(docs, labels)
        _check_basic(m.transform(NEW))

    def test_supervised_lda(self):
        docs, is_a = _two_topic_corpus()
        y = np.where(is_a, 1.0, -1.0)
        m = topica.models.SupervisedLDA(num_topics=2, seed=1)
        m.fit(docs, y)
        _check_basic(m.transform(NEW))

    def test_dmr_baseline_and_covariate(self):
        docs, is_a = _two_topic_corpus()
        x = is_a.astype(float).reshape(-1, 1)
        m = topica.models.DMR(num_topics=2, seed=1)
        m.fit(docs, x)
        # Intercept-only baseline prior. (DMR's learned prior is asymmetric, so
        # the all-OOV doc returns that skewed prior rather than a uniform one —
        # check separation on the in-vocabulary docs only.)
        t = m.transform(NEW)
        assert t.shape == (3, 2)
        np.testing.assert_allclose(t.sum(axis=1), 1.0, atol=1e-6)
        assert t[0].argmax() != t[1].argmax()
        # Supplying held-out covariates (no intercept) is accepted and shifts θ.
        x_new = np.array([[0.0], [1.0], [0.0]])
        t = m.transform(NEW, x_new)
        assert t.shape == (3, 2)
        np.testing.assert_allclose(t.sum(axis=1), 1.0, atol=1e-6)

    def test_dmr_covariate_shape_validation(self):
        docs, is_a = _two_topic_corpus()
        x = is_a.astype(float).reshape(-1, 1)
        m = topica.models.DMR(num_topics=2, seed=1)
        m.fit(docs, x)
        with pytest.raises(ValueError):
            m.transform(NEW, np.zeros((2, 1)))  # wrong number of rows
        with pytest.raises(ValueError):
            m.transform(NEW, np.zeros((3, 2)))  # wrong number of covariates


def test_transform_accepts_corpus_object():
    docs, _ = _two_topic_corpus()
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    corpus = topica.Corpus.from_documents(NEW)
    theta = m.transform(corpus)
    assert theta.shape[0] == len(NEW)


def test_transform_deterministic():
    docs, _ = _two_topic_corpus()
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    np.testing.assert_array_equal(m.transform(NEW, seed=7), m.transform(NEW, seed=7))


# ---------------------------------------------------------------------------
# #104: iters= is canonical; iterations= is a deprecated alias
# ---------------------------------------------------------------------------

class TestTransformIterationsAlias:
    """transform(iters=N) and transform(iterations=N) give identical results;
    the deprecated form emits a DeprecationWarning."""

    def setup_method(self):
        docs, _ = _two_topic_corpus()
        self.m = topica.models.LDA(num_topics=2, seed=1)
        self.m.fit(docs, iters=200)

    def test_lda_iters_works(self):
        theta = self.m.transform(NEW, iters=20, seed=0)
        assert theta.shape == (len(NEW), 2)
        np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-6)

    def test_lda_iterations_deprecated_warns(self):
        with pytest.warns(DeprecationWarning, match="iterations="):
            theta = self.m.transform(NEW, iterations=20, seed=0)
        assert theta.shape == (len(NEW), 2)

    def test_lda_iters_and_iterations_same_result(self):
        """Both aliases must produce numerically identical output for the same seed."""
        theta_new = self.m.transform(NEW, iters=20, seed=0)
        with pytest.warns(DeprecationWarning):
            theta_old = self.m.transform(NEW, iterations=20, seed=0)
        np.testing.assert_array_equal(theta_new, theta_old)

    def test_dmr_iterations_deprecated_warns(self):
        """Check a second Gibbs model (DMR) also emits the warning."""
        docs, is_a = _two_topic_corpus()
        x = is_a.astype(float).reshape(-1, 1)
        m = topica.models.DMR(num_topics=2, seed=1)
        m.fit(docs, x)
        with pytest.warns(DeprecationWarning, match="iterations="):
            theta = m.transform(NEW, iterations=20, seed=0)
        assert theta.shape == (len(NEW), 2)
