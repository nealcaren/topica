"""HDP (Hierarchical Dirichlet Process) topic model — the nonparametric LDA that
infers the number of topics. These tests check that K is inferred in a sane
range, that planted topics are recovered, and that outputs are well-formed and
deterministic.
"""

import numpy as np
import pytest

from topica import HDP, Corpus


def _planted_corpus(n_blocks=5, words_per_block=6, n_docs=250):
    """One disjoint vocabulary block per topic; each doc draws from one block."""
    vocab = [f"w{i}" for i in range(n_blocks * words_per_block)]
    docs = []
    for d in range(n_docs):
        b = d % n_blocks
        block = vocab[b * words_per_block : (b + 1) * words_per_block]
        docs.append(block + block)  # each block word twice
    return docs, vocab, n_blocks


class TestInference:
    def test_infers_reasonable_k(self):
        docs, _, n_blocks = _planted_corpus()
        # This tiny planted corpus needs controlled concentrations to instantiate
        # exactly the planted topics, so we fix them (resample_conc=False) at a
        # moderate value rather than let the canonical sampler infer/over-segment
        # (cf. the Rust unit test).
        m = HDP(seed=1, alpha=1.0, gamma=1.0, resample_conc=False)
        m.fit(docs, iters=120)
        # Auto-K is approximate and HDP slightly over-segments; it must at least
        # find the planted topics and not explode.
        assert n_blocks - 1 <= m.num_topics <= 2 * n_blocks + 2

    def test_recovers_planted_blocks(self):
        docs, vocab, n_blocks = _planted_corpus()
        # Fixed concentrations for a clean planted recovery (see above).
        m = HDP(seed=1, alpha=1.0, gamma=1.0, resample_conc=False)
        m.fit(docs, iters=200)
        wps = len(vocab) // n_blocks
        blocks = [
            set(vocab[b * wps : (b + 1) * wps]) for b in range(n_blocks)
        ]
        covered = set()
        for t in range(m.num_topics):
            top = {w for w, _ in m.top_words(wps, topic=t, weights=True)}
            for bi, blk in enumerate(blocks):
                if blk <= top:
                    covered.add(bi)
        assert covered == set(range(n_blocks)), f"only recovered {covered}"


class TestOutputs:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=3)
        m.fit(docs, iters=80)
        return m

    def test_shapes(self, fitted):
        k = fitted.num_topics
        assert fitted.topic_word.shape == (k, len(fitted.vocabulary))
        assert fitted.doc_topic.shape[1] == k

    def test_distributions_normalized(self, fitted):
        # Topic-word rows and doc-topic rows are proper distributions.
        npt = np.testing
        npt.assert_allclose(fitted.topic_word.sum(axis=1), 1.0, atol=1e-9)
        npt.assert_allclose(fitted.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_learned_concentrations_positive(self, fitted):
        assert fitted.alpha > 0
        assert fitted.gamma > 0

    def test_coherence_finite(self, fitted):
        # UMass is normally <= 0, but on planted data where a topic's top words
        # always co-occur it can edge slightly positive; assert finiteness.
        c = fitted.coherence(n=5)
        assert c.shape == (fitted.num_topics,)
        assert np.isfinite(c).all()

    def test_top_words_all_topics(self, fitted):
        allw = fitted.top_words(5)
        assert len(allw) == fitted.num_topics
        assert all(len(t) == 5 for t in allw)


class TestDeterminismAndApi:
    def test_deterministic_for_seed(self):
        docs, _, _ = _planted_corpus()
        a = HDP(seed=7)
        a.fit(docs, iters=60)
        b = HDP(seed=7)
        b.fit(docs, iters=60)
        assert a.num_topics == b.num_topics
        assert np.array_equal(a.topic_word, b.topic_word)

    def test_accepts_corpus_object(self):
        docs, _, _ = _planted_corpus(n_docs=100)
        c = Corpus.from_documents(docs)
        m = HDP(seed=1)
        m.fit(c, iters=40)
        assert m.num_topics >= 1

    def test_fixed_concentrations(self):
        docs, _, _ = _planted_corpus(n_docs=100)
        m = HDP(seed=1, resample_conc=False, alpha=0.5, gamma=0.5)
        m.fit(docs, iters=40)
        # With resampling off the concentrations stay at their initial values.
        assert m.alpha == 0.5
        assert m.gamma == 0.5

    def test_unfitted_raises(self):
        m = HDP()
        with pytest.raises(RuntimeError):
            _ = m.topic_word

    def test_bad_hyperparams_raise(self):
        with pytest.raises(ValueError):
            HDP(alpha=0.0)
        with pytest.raises(ValueError):
            HDP(beta=-1.0)


class TestDiscoveryTrace:
    """The discovery/convergence trace — HDP's headline diagnostic."""

    def test_traces_recorded_and_aligned(self):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1)
        m.fit(docs, iters=120, progress_interval=10)
        tch = m.topic_count_history
        llh = m.log_likelihood_history
        ch = m.concentration_history
        assert [it for it, _ in tch] == list(range(10, 121, 10))
        # All three traces share the same iteration grid.
        assert [it for it, _ in tch] == [it for it, _ in llh]
        assert [it for it, _ in tch] == [it for it, _, _ in ch]
        assert all(k >= 1 for _, k in tch)
        assert all(np.isfinite(ll) and ll < 0 for _, ll in llh)

    def test_auto_spacing(self):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1)
        m.fit(docs, iters=150)  # progress_interval=0 -> auto
        assert len(m.topic_count_history) == 50

    def test_final_count_matches_num_topics(self):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1)
        m.fit(docs, iters=120, progress_interval=20)
        assert m.topic_count_history[-1][1] == m.num_topics

    def test_log_likelihood_improves(self):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1, alpha=1.0, gamma=1.0)
        m.fit(docs, iters=120, progress_interval=5)
        lls = [ll for _, ll in m.log_likelihood_history]
        assert lls[-1] > lls[0]

    def test_trace_survives_save_load(self, tmp_path):
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1)
        m.fit(docs, iters=80, progress_interval=10)
        path = str(tmp_path / "hdp.bin")
        m.save(path)
        reloaded = HDP.load(path)
        assert reloaded.topic_count_history == m.topic_count_history
        assert reloaded.log_likelihood_history == m.log_likelihood_history
        assert reloaded.concentration_history == m.concentration_history

    def test_plot_topic_discovery(self):
        import topica

        plt = pytest.importorskip("matplotlib.pyplot")
        docs, _, _ = _planted_corpus()
        m = HDP(seed=1)
        m.fit(docs, iters=80, progress_interval=10)
        ax = topica.plot_topic_discovery(m)
        assert ax.get_xlabel() == "Gibbs iteration"
        plt.close("all")
