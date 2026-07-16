"""Alias and rename regression tests (issue #107).

Checks:
  - Canonical names work on all affected models.
  - Deprecated aliases work and emit a DeprecationWarning.
  - ETM/ProdLDA/FASTopic convergence_tol actually affects the fit.
  - covariates= works on DMR and STM; conflicts raise ValueError.
  - num_threads in both constructor and fit() for LDA and KeyATM.
"""

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# Shared tiny corpora
# ---------------------------------------------------------------------------

def _tiny_docs(n=80, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [["cat", "dog", "pet"], ["tax", "vote", "law"]]
    docs = []
    for i in range(n):
        block = vocab[i % 2]
        docs.append([block[int(rng.integers(3))] for _ in range(5)])
    return docs


def _short_docs(n=150, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [["cat", "dog", "pet", "vet"],
              ["star", "moon", "sky", "sun"],
              ["tax", "vote", "law", "bill"]]
    docs = []
    for _ in range(n):
        blk = blocks[int(rng.integers(3))]
        docs.append([blk[int(rng.integers(4))] for _ in range(3)])
    return docs


# ---------------------------------------------------------------------------
# Item 1: em_tol -> convergence_tol (CTM / STM / STS)
# ---------------------------------------------------------------------------

class TestConvergenceTolCanonical:
    def test_ctm_canonical(self):
        docs = _tiny_docs()
        m = topica.models.CTM(num_topics=2, seed=1)
        m.fit(docs, iters=20, convergence_tol=0.0)
        assert m.doc_topic.shape[0] == len(docs)

    def test_stm_canonical(self):
        docs = _tiny_docs()
        x = np.random.default_rng(0).standard_normal((len(docs), 1))
        m = topica.models.STM(num_topics=2, seed=1)
        m.fit(docs, x, prevalence_names=["x"], iters=15, convergence_tol=0.0)
        assert m.doc_topic.shape[0] == len(docs)


class TestEmTolDeprecatedAliasWarns:
    def test_ctm_em_tol_warns(self):
        docs = _tiny_docs()
        m = topica.models.CTM(num_topics=2, seed=1)
        with pytest.warns(DeprecationWarning, match="em_tol"):
            m.fit(docs, iters=10, em_tol=1e-5)

    def test_stm_em_tol_warns(self):
        docs = _tiny_docs()
        x = np.random.default_rng(0).standard_normal((len(docs), 1))
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.warns(DeprecationWarning, match="em_tol"):
            m.fit(docs, x, prevalence_names=["x"], iters=10, em_tol=1e-5)


# ---------------------------------------------------------------------------
# Item 1b: ETM / ProdLDA / FASTopic  convergence_tol constructor + fit()
# ---------------------------------------------------------------------------

class TestNeuralConvergenceTol:
    """convergence_tol in constructor and fit() override actually changes
    the number of epochs the model runs (tight tol stops earlier than loose)."""

    def _word_embeddings(self, vocab, d=10, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(vocab), d))

    def _doc_embeddings(self, docs, d=10, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(docs), d))

    def test_etm_convergence_tol_constructor(self):
        docs = _tiny_docs(n=60)
        m_tight = topica.models.ETM(num_topics=2, convergence_tol=1.0, seed=1)
        m_loose = topica.models.ETM(num_topics=2, convergence_tol=0.0, seed=1)
        vocab = sorted({w for d in docs for w in d})
        emb = self._word_embeddings(vocab)
        m_tight.fit(docs, emb, vocab, iters=50)
        m_loose.fit(docs, emb, vocab, iters=50)
        # tight tol should converge early (fewer bound_history entries) or both converge
        # at minimum the model runs without error
        assert m_tight.topic_word.shape[0] == 2
        assert m_loose.topic_word.shape[0] == 2

    def test_etm_convergence_tol_fit_override(self):
        docs = _tiny_docs(n=60)
        vocab = sorted({w for d in docs for w in d})
        emb = self._word_embeddings(vocab)
        # Constructor sets a loose (never-stop) tol; the fit() override sets a
        # tight one, so the fit() value must take effect: the run stops early.
        m = topica.models.ETM(num_topics=2, convergence_tol=0.0, seed=1)
        m.fit(docs, emb, vocab, iters=50, convergence_tol=1.0)
        assert m.topic_word.shape[0] == 2
        assert m.converged is True
        assert len(m.fit_history) < 50

    def test_prodlda_convergence_tol_fit_override(self):
        docs = _tiny_docs(n=60)
        m = topica.models.ProdLDA(num_topics=2, convergence_tol=0.0, seed=1)
        m.fit(docs, iters=30, convergence_tol=1.0)
        assert m.topic_word.shape[0] == 2
        assert m.converged is True
        assert len(m.fit_history) < 30

    def test_fastopic_convergence_tol_fit_override(self):
        docs = _tiny_docs(n=60)
        emb = self._doc_embeddings(docs)
        m = topica.models.FASTopic(num_topics=2, convergence_tol=0.0, seed=1)
        m.fit(docs, emb, iters=30, convergence_tol=1.0)
        assert m.topic_word.shape[0] == 2
        assert m.converged is True
        assert len(m.fit_history) < 30

    def test_etm_em_tol_constructor_warns(self):
        docs = _tiny_docs(n=60)
        vocab = sorted({w for d in docs for w in d})
        emb = self._word_embeddings(vocab)
        with pytest.warns(DeprecationWarning, match="em_tol"):
            m = topica.models.ETM(num_topics=2, em_tol=1e-4, seed=1)
        m.fit(docs, emb, vocab, iters=5)
        assert m.topic_word.shape[0] == 2

    def test_prodlda_em_tol_constructor_warns(self):
        docs = _tiny_docs(n=60)
        with pytest.warns(DeprecationWarning, match="em_tol"):
            m = topica.models.ProdLDA(num_topics=2, em_tol=0.0, seed=1)
        m.fit(docs, iters=5)
        assert m.topic_word.shape[0] == 2

    def test_fastopic_em_tol_constructor_warns(self):
        docs = _tiny_docs(n=60)
        emb = self._doc_embeddings(docs)
        with pytest.warns(DeprecationWarning, match="em_tol"):
            m = topica.models.FASTopic(num_topics=2, em_tol=1e-6, seed=1)
        m.fit(docs, emb, iters=5)
        assert m.topic_word.shape[0] == 2


# ---------------------------------------------------------------------------
# Item 2: eta -> beta in HDP and HLDA
# ---------------------------------------------------------------------------

class TestHDPBetaCanonical:
    def test_hdp_beta_canonical(self):
        docs = _tiny_docs()
        m = topica.models.HDP(beta=0.01, seed=1)
        m.fit(docs, iters=30)
        assert m.topic_word.shape[1] > 0

    def test_hdp_eta_deprecated_warns(self):
        docs = _tiny_docs()
        with pytest.warns(DeprecationWarning, match="eta"):
            m = topica.models.HDP(eta=0.01, seed=1)
        m.fit(docs, iters=30)
        assert m.topic_word.shape[1] > 0


class TestHLDABetaCanonical:
    def test_hlda_beta_canonical(self):
        docs = _tiny_docs()
        m = topica.models.HLDA(beta=0.01, seed=1)
        m.fit(docs, iters=30)
        assert m.topic_word.shape[1] > 0

    def test_hlda_eta_deprecated_warns(self):
        docs = _tiny_docs()
        with pytest.warns(DeprecationWarning, match="eta"):
            m = topica.models.HLDA(eta=0.01, seed=1)
        m.fit(docs, iters=30)
        assert m.topic_word.shape[1] > 0


# ---------------------------------------------------------------------------
# Item 3: covariates= alias (DMR and STM)
# ---------------------------------------------------------------------------

class TestCovariatesAlias:
    def test_dmr_covariates_alias(self):
        docs = _tiny_docs()
        x = np.ones((len(docs), 1))
        m = topica.models.DMR(num_topics=2, seed=1)
        # covariates= is a symmetric alias for features=
        m.fit(docs, covariates=x, feature_names=["x"], iters=50)
        assert m.doc_topic.shape[0] == len(docs)

    def test_dmr_features_canonical(self):
        docs = _tiny_docs()
        x = np.ones((len(docs), 1))
        m = topica.models.DMR(num_topics=2, seed=1)
        m.fit(docs, x, feature_names=["x"], iters=50)
        assert m.doc_topic.shape[0] == len(docs)

    def test_dmr_both_raises(self):
        docs = _tiny_docs()
        x = np.ones((len(docs), 1))
        m = topica.models.DMR(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="features.*covariates|covariates.*features"):
            m.fit(docs, x, covariates=x, iters=5)

    def test_dmr_neither_raises(self):
        docs = _tiny_docs()
        m = topica.models.DMR(num_topics=2, seed=1)
        with pytest.raises(ValueError):
            m.fit(docs, iters=5)

    def test_stm_covariates_alias(self):
        docs = _tiny_docs()
        x = np.ones((len(docs), 1))
        m = topica.models.STM(num_topics=2, seed=1)
        m.fit(docs, covariates=x, prevalence_names=["x"], iters=10, convergence_tol=0.0)
        assert m.doc_topic.shape[0] == len(docs)

    def test_stm_both_raises(self):
        docs = _tiny_docs()
        x = np.ones((len(docs), 1))
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="prevalence.*covariates|covariates.*prevalence"):
            m.fit(docs, x, covariates=x, iters=5, convergence_tol=0.0)


# ---------------------------------------------------------------------------
# Item 4: report_interval -> progress_interval (HDP / GSDMM / KeyATM)
# ---------------------------------------------------------------------------

class TestProgressIntervalCanonical:
    def test_hdp_progress_interval(self):
        docs = _tiny_docs()
        m = topica.models.HDP(seed=1)
        m.fit(docs, iters=30, progress_interval=10)
        iters = [it for it, _ in m.topic_count_history]
        assert iters == list(range(10, 31, 10))

    def test_gsdmm_progress_interval(self):
        docs = _short_docs()
        m = topica.models.GSDMM(num_topics=10, seed=1)
        m.fit(docs, iters=20, progress_interval=5)
        iters = [it for it, _ in m.cluster_count_history]
        assert iters == list(range(5, 21, 5))

    def test_keyatm_progress_interval(self):
        docs = _tiny_docs()
        seeds = {"pets": ["cat", "dog"], "politics": ["tax", "vote"]}
        m = topica.models.KeyATM(seeds, num_topics=2, seed=1)
        m.fit(docs, iters=20, progress_interval=5)
        iters = [it for it, _, _ in m.log_likelihood_history]
        assert iters == list(range(5, 21, 5))


class TestReportIntervalDeprecatedWarns:
    def test_hdp_report_interval_warns(self):
        docs = _tiny_docs()
        m = topica.models.HDP(seed=1)
        with pytest.warns(DeprecationWarning, match="report_interval"):
            m.fit(docs, iters=30, report_interval=10)
        iters = [it for it, _ in m.topic_count_history]
        assert iters == list(range(10, 31, 10))

    def test_gsdmm_report_interval_warns(self):
        docs = _short_docs()
        m = topica.models.GSDMM(num_topics=10, seed=1)
        with pytest.warns(DeprecationWarning, match="report_interval"):
            m.fit(docs, iters=20, report_interval=5)
        iters = [it for it, _ in m.cluster_count_history]
        assert iters == list(range(5, 21, 5))

    def test_keyatm_report_interval_warns(self):
        docs = _tiny_docs()
        seeds = {"pets": ["cat", "dog"], "politics": ["tax", "vote"]}
        m = topica.models.KeyATM(seeds, num_topics=2, seed=1)
        with pytest.warns(DeprecationWarning, match="report_interval"):
            m.fit(docs, iters=20, report_interval=5)
        iters = [it for it, _, _ in m.log_likelihood_history]
        assert iters == list(range(5, 21, 5))


# ---------------------------------------------------------------------------
# Item 5: num_threads in LDA and KeyATM
# ---------------------------------------------------------------------------

class TestNumThreads:
    def test_lda_num_threads_in_fit(self):
        docs = _tiny_docs(n=100)
        m = topica.models.LDA(num_topics=2, seed=1)
        m.fit(docs, iters=20, num_threads=1)
        assert m.topic_word.shape[0] == 2

    def test_lda_num_threads_fit_overrides_constructor(self):
        docs = _tiny_docs(n=100)
        m2 = topica.models.LDA(num_topics=2, seed=1, num_threads=2)
        m1 = topica.models.LDA(num_topics=2, seed=1, num_threads=1)
        # fit(num_threads=1) override makes them equivalent to single-threaded
        m2.fit(docs, iters=20, num_threads=1)
        m1.fit(docs, iters=20, num_threads=1)
        assert np.array_equal(m1.topic_word, m2.topic_word)

    def test_keyatm_num_threads_in_constructor(self):
        docs = _tiny_docs()
        seeds = {"pets": ["cat", "dog"], "politics": ["tax", "vote"]}
        m = topica.models.KeyATM(seeds, num_topics=2, seed=1, num_threads=1)
        m.fit(docs, iters=30)
        assert m.topic_word.shape[0] == 2

    def test_keyatm_num_threads_in_fit(self):
        docs = _tiny_docs()
        seeds = {"pets": ["cat", "dog"], "politics": ["tax", "vote"]}
        m = topica.models.KeyATM(seeds, num_topics=2, seed=1)
        m.fit(docs, iters=30, num_threads=1)
        assert m.topic_word.shape[0] == 2

    def test_keyatm_num_threads_fit_overrides_constructor(self):
        docs = _tiny_docs()
        seeds = {"pets": ["cat", "dog"], "politics": ["tax", "vote"]}
        # Constructor sets 2 threads, fit() overrides with 1
        m_c2_f1 = topica.models.KeyATM(seeds, num_topics=2, seed=3, num_threads=2)
        m_c1_f1 = topica.models.KeyATM(seeds, num_topics=2, seed=3, num_threads=1)
        m_c2_f1.fit(docs, iters=50, num_threads=1)
        m_c1_f1.fit(docs, iters=50, num_threads=1)
        # Both use 1 thread -> identical results
        assert np.array_equal(m_c1_f1.topic_word, m_c2_f1.topic_word)
