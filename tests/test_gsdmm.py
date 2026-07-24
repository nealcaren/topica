"""GSDMM (Movie Group Process) short-text clustering: inference of K, output
shapes, one-topic-per-document assignment, determinism, and save/load."""

import numpy as np
import pytest

import topica


def _short_corpus(seed=0, n=150):
    rng = np.random.default_rng(seed)
    blocks = [["cat", "dog", "pet", "vet"],
              ["star", "moon", "sky", "sun"],
              ["tax", "vote", "law", "bill"]]
    docs = []
    for _ in range(n):
        blk = blocks[int(rng.integers(3))]
        docs.append([blk[int(rng.integers(4))] for _ in range(3)])  # 3-token docs
    return docs


class TestGSDMM:
    def test_infers_fewer_than_k_max(self):
        docs = _short_corpus()
        m = topica.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        # Empty clusters die out -> effective K is below the cap.
        assert 0 < m.num_topics <= 15

    def test_output_shapes(self):
        docs = _short_corpus()
        m = topica.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        k = m.num_topics
        assert m.topic_word.shape == (k, len(m.vocabulary))
        assert m.doc_topic.shape == (len(docs), k)
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_hard_assignment(self):
        docs = _short_corpus()
        m = topica.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        dc = m.doc_cluster
        assert dc.shape == (len(docs),)
        assert dc.min() >= 0 and dc.max() < m.num_topics

    def test_recovers_blocks(self):
        docs = _short_corpus()
        m = topica.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=60)
        blocks = [{"cat", "dog", "pet", "vet"},
                  {"star", "moon", "sky", "sun"},
                  {"tax", "vote", "law", "bill"}]
        covered = set()
        for t in range(m.num_topics):
            top = {w for w, _ in m.top_words(4, topic=t)}
            for b, blk in enumerate(blocks):
                if len(top & blk) >= 3:    # a cluster cleanly owns a block
                    covered.add(b)
        assert covered == {0, 1, 2}        # all three blocks recovered

    def test_deterministic(self):
        docs = _short_corpus()
        a = topica.GSDMM(num_topics=12, seed=3); a.fit(docs, iters=30)
        b = topica.GSDMM(num_topics=12, seed=3); b.fit(docs, iters=30)
        assert a.num_topics == b.num_topics
        assert np.array_equal(a.topic_word, b.topic_word)
        assert np.array_equal(a.doc_cluster, b.doc_cluster)

    def test_save_load(self, tmp_path):
        docs = _short_corpus()
        m = topica.GSDMM(num_topics=12, seed=1); m.fit(docs, iters=30)
        p = str(tmp_path / "gsdmm.tt"); m.save(p)
        ld = topica.GSDMM.load(p)
        assert ld.num_topics == m.num_topics
        assert np.array_equal(ld.topic_word, m.topic_word)
        assert np.array_equal(ld.doc_cluster, m.doc_cluster)

    def test_bad_params(self):
        with pytest.raises(ValueError):
            topica.GSDMM(num_topics=1)
        with pytest.raises(ValueError):
            topica.GSDMM(num_topics=10, alpha=0.0)


class TestClusterDiscovery:
    def test_count_collapses_from_k_max(self):
        docs = _short_corpus(n=300)
        m = topica.GSDMM(num_topics=12, seed=1)
        m.fit(docs, iters=40, progress_interval=5)
        cch = m.cluster_count_history
        assert [it for it, _ in cch] == list(range(5, 41, 5))
        # The Movie Group Process starts near the cap and collapses.
        assert cch[0][1] > cch[-1][1]
        assert cch[-1][1] == m.num_topics

    def test_log_likelihood_stabilizes(self):
        docs = _short_corpus(n=300)
        m = topica.GSDMM(num_topics=12, seed=1)
        m.fit(docs, iters=60, progress_interval=5)
        lls = [ll for _, ll in m.log_likelihood_history]
        assert all(np.isfinite(ll) and ll < 0 for ll in lls)
        # The cluster-fit score settles once the clustering stops moving (it can
        # dip as clusters merge, then plateau): the tail should be near-constant.
        tail = lls[-3:]
        assert max(tail) - min(tail) < 0.1

    def test_auto_spacing(self):
        docs = _short_corpus(n=150)
        m = topica.GSDMM(num_topics=10, seed=1)
        m.fit(docs, iters=100)  # auto cadence
        assert len(m.cluster_count_history) == 50

    def test_trace_survives_save_load(self, tmp_path):
        docs = _short_corpus(n=150)
        m = topica.GSDMM(num_topics=10, seed=1)
        m.fit(docs, iters=30, progress_interval=5)
        path = str(tmp_path / "g.bin")
        m.save(path)
        ld = topica.GSDMM.load(path)
        assert ld.cluster_count_history == m.cluster_count_history
        assert ld.log_likelihood_history == m.log_likelihood_history


class TestTransform:
    """Held-out cluster assignment via the Movie-Group-Process conditional (#490)."""

    def _fit(self, seed=1, n=300):
        docs = _short_corpus(n=n, seed=seed)
        m = topica.GSDMM(num_topics=12, seed=seed).fit(docs, iters=40)
        return m, docs

    def test_shape_and_rows_sum_to_one(self):
        m, _ = self._fit()
        held = [["cat", "dog", "pet"], ["star", "sky"], ["tax", "vote", "law"]]
        t = m.transform(held)
        assert t.shape == (3, m.num_topics)
        assert np.allclose(t.sum(axis=1), 1.0)
        assert np.all(t >= 0.0)

    def test_in_sample_matches_doc_topic(self):
        # transform on the training docs reproduces the training-time scoring,
        # so its argmax equals doc_topic's argmax (both are doc_cluster_dist).
        m, docs = self._fit()
        assert np.array_equal(m.transform(docs).argmax(1), np.asarray(m.doc_topic).argmax(1))

    def test_pure_heldout_doc_lands_on_its_block(self):
        m, _ = self._fit()
        # A doc made only of one block's words concentrates on one cluster, and the
        # three blocks map to three distinct clusters.
        pure = [["cat", "dog", "pet", "vet"], ["star", "moon", "sky"], ["tax", "law", "bill"]]
        t = m.transform(pure)
        assert t.max(axis=1).min() > 0.9  # each row confident
        assert len(set(t.argmax(axis=1))) == 3  # three distinct clusters

    def test_oov_and_empty_fall_back_to_size_prior(self):
        m, _ = self._fit()
        t = m.transform([["zzzz", "qqqq"], []])  # all-OOV, then empty
        assert np.allclose(t.sum(axis=1), 1.0)
        assert np.all(np.isfinite(t))
        # No in-vocabulary evidence -> the posterior reduces to the clusters' size
        # prior, which is the SAME for every evidence-free document.
        assert np.allclose(t[0], t[1])

    def test_deterministic_and_leaves_fit_intact(self):
        m, _ = self._fit()
        held = [["cat", "dog"], ["star", "sun"]]
        before = np.asarray(m.doc_topic).copy()
        t1 = m.transform(held)
        t2 = m.transform(held)
        assert np.array_equal(t1, t2)
        assert np.array_equal(np.asarray(m.doc_topic), before)  # not mutated

    def test_accepts_corpus(self):
        m, _ = self._fit()
        c = topica.Corpus.from_documents([["cat", "dog", "pet"], ["tax", "vote"]])
        t = m.transform(c)
        assert t.shape == (2, m.num_topics)
        assert np.allclose(t.sum(axis=1), 1.0)

    def test_survives_save_load(self, tmp_path):
        m, _ = self._fit()
        held = [["cat", "dog", "pet"], ["star", "sky"]]
        expected = m.transform(held)
        path = str(tmp_path / "g.bin")
        m.save(path)
        ld = topica.GSDMM.load(path)
        assert np.allclose(ld.transform(held), expected)

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            topica.GSDMM(num_topics=5).transform([["cat"]])
