"""Tests for AnchorLDA, the anchor-words spectral estimator (Arora et al. 2013)."""

import os

import numpy as np
import pytest

import topica


def _separable_corpus(k=4, words_per_topic=12, n_docs=400, doc_len=40, seed=0):
    """A separable corpus: each topic owns an anchor word that appears only in
    it, plus shared filler. The planted topic assignment is recoverable."""
    rng = np.random.default_rng(seed)
    vocab_blocks = [[f"t{t}_w{j}" for j in range(words_per_topic)] for t in range(k)]
    anchors = [f"t{t}_anchor" for t in range(k)]
    docs, truth = [], []
    for _ in range(n_docs):
        t = int(rng.integers(k))
        truth.append(t)
        words = list(rng.choice(vocab_blocks[t], doc_len - 2))
        words += [anchors[t], anchors[t]]  # the anchor occurs only in topic t
        docs.append(words)
    return docs, truth, anchors


class TestConstruction:
    def test_constructs_ungated(self):
        # AnchorLDA is validated (reference parity); no experimental opt-in needed.
        m = topica.AnchorLDA(5)
        assert m.num_topics == 5


class TestFit:
    def test_shapes_and_attrs(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        v = len(m.vocabulary)
        assert np.asarray(m.topic_word).shape == (4, v)
        assert np.asarray(m.doc_topic).shape == (len(docs), 4)
        assert len(m.anchors) == 4

    def test_kl_default_is_iterative(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)  # recover="kl"
        assert m.fit_history, "KL recovery should record an objective trace"
        assert isinstance(m.converged, bool)
        # The KL objective is non-increasing along the recorded trace.
        objs = [o for _, o in m.fit_history]
        assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:]))

    def test_l2_is_iterative(self):
        # recover="l2" is now a vectorized exponentiated-gradient solve (#622), so
        # it reports a convergence trace like the KL path.
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, recover="l2", min_count=2, seed=0).fit(docs)
        assert len(m.fit_history) > 0
        assert m.converged in (True, False)

    def test_bad_recover_raises(self):
        with pytest.raises(ValueError, match="recover must be"):
            topica.AnchorLDA(4, recover="nope")

    def test_both_recoveries_recover_separable(self):
        # Both recovery methods should recover the planted separable topics.
        docs, _, anchors = _separable_corpus(k=4, seed=1)
        for rec in ("kl", "l2"):
            m = topica.AnchorLDA(4, recover=rec, min_count=2, seed=0).fit(docs)
            assert set(m.anchors) == set(anchors), rec
            for t in range(4):
                blocks = {w.split("_")[0] for w, _ in m.top_words(8, topic=t)}
                assert len(blocks) == 1, (rec, t)

    def test_topic_word_rows_are_distributions(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        rowsums = np.asarray(m.topic_word).sum(axis=1)
        assert np.allclose(rowsums, 1.0, atol=1e-6)
        assert (np.asarray(m.topic_word) >= 0).all()

    def test_recovers_separable_topics(self):
        # On a separable corpus each planted anchor word should be selected, and
        # each recovered topic should be dominated by one block's vocabulary.
        docs, truth, anchors = _separable_corpus(k=4, seed=1)
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        # Every planted anchor is chosen as some topic's anchor.
        assert set(m.anchors) == set(anchors)
        # Each topic's top words come overwhelmingly from a single block.
        for t in range(4):
            tops = [w for w, _ in m.top_words(8, topic=t)]
            blocks = {w.split("_")[0] for w in tops}
            assert len(blocks) == 1, (t, tops)

    def test_deterministic(self):
        docs, _, _ = _separable_corpus()
        a = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        b = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        assert np.array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))
        assert a.anchors == b.anchors

    def test_accepts_corpus_object(self):
        docs, _, _ = _separable_corpus()
        corpus = topica.Corpus.from_documents(docs)
        m = topica.AnchorLDA(4, seed=0).fit(corpus)
        assert np.asarray(m.doc_topic).shape[1] == 4

    def test_too_few_vocab_raises(self):
        docs = [["a", "b"], ["b", "c"]]
        with pytest.raises(ValueError, match="num_topics"):
            topica.AnchorLDA(10, min_count=1).fit(docs)


class TestSurface:
    def test_conformance(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        assert topica.check_conformance(m) == []

    def test_coherence_returns_per_topic(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        cv = m.coherence(10)
        assert np.asarray(cv).shape == (4,)

    def test_topic_names_roundtrip(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        m.topic_names = ["a", "b", "c", "d"]
        assert m.topic_names == ["a", "b", "c", "d"]
        with pytest.raises(ValueError):
            m.topic_names = ["only", "three", "names"]

    def test_unfitted_access_raises(self):
        m = topica.AnchorLDA(4)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_word

    def test_save_load_roundtrip(self, tmp_path):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        p = os.path.join(tmp_path, "anchor")
        m.save(p)
        loaded = topica.AnchorLDA.load(p)
        assert np.array_equal(np.asarray(m.topic_word), np.asarray(loaded.topic_word))
        assert m.anchors == loaded.anchors
        assert m.vocabulary == loaded.vocabulary


class TestRegistry:
    def test_registered_on_validated_roster(self):
        info = {m.name: m for m in topica.list_models()}
        assert "AnchorLDA" in info
        assert info["AnchorLDA"].experimental is False


def _overlap_corpus(k=30, block=8, shared=15, n=1500, length=40, seed=0):
    """A high-K corpus whose topics share a heavy common-word background. Each
    document is one block's distinctive words plus a unique per-block anchor word,
    diluted by ~50% shared common words. The common words are the most frequent
    tokens, so they dominate a plain LDA topic's mass; anchor-words divides word
    frequency out (Bayes inversion), so its topics still lead with the block's
    distinctive words. This is the regime where the high-K theory bites."""
    rng = np.random.default_rng(seed)
    blocks = [[f"b{b}_w{j}" for j in range(block)] for b in range(k)]
    anchors = [f"b{b}_anchor" for b in range(k)]
    common = [f"common_{j}" for j in range(shared)]
    docs = []
    for d in range(n):
        b = d % k
        docs.append([anchors[b], anchors[b]]
                    + list(rng.choice(blocks[b], 18))
                    + list(rng.choice(common, 20)))
    return docs, set(anchors), k


def _is_common(w):
    return w.startswith("common_")


def _anchors_surfaced(model, true_anchors, n=10):
    """Fraction of true anchor words that appear in some topic's top-n."""
    v = list(model.vocabulary)
    tw = np.asarray(model.topic_word)
    top = set()
    for t in range(tw.shape[0]):
        top |= {v[i] for i in np.argsort(tw[t])[::-1][:n]}
    return len(top & true_anchors) / len(true_anchors)


def _distinctive_lead(model):
    """Fraction of topics whose single top word is not a shared common word."""
    v = list(model.vocabulary)
    tw = np.asarray(model.topic_word)
    return float(np.mean([not _is_common(v[int(np.argmax(tw[t]))])
                          for t in range(tw.shape[0])]))


class TestHighKTheory:
    """At high K with overlapping (shared-background) vocabulary, anchor-words
    recovers the per-topic distinctive words that a random-init Gibbs LDA, at a
    comparable budget, loses to frequency domination. See PR #290 discussion."""

    K = 30

    @pytest.fixture(scope="class")
    def fits(self):
        docs, true_anchors, k = _overlap_corpus(k=self.K, seed=0)
        anchor = topica.AnchorLDA(k, seed=0).fit(docs)
        lda = topica.LDA(num_topics=k, seed=0)
        lda.fit(docs, iters=100)
        return anchor, lda, true_anchors

    def test_anchor_recovers_distinctive_structure(self, fits):
        anchor, _, true_anchors = fits
        # Anchor-words should surface (nearly) every planted anchor and lead every
        # topic with a distinctive (non-common) word despite the shared background.
        assert _anchors_surfaced(anchor, true_anchors) >= 0.9
        assert _distinctive_lead(anchor) >= 0.9

    def test_anchor_beats_lda_at_high_k(self, fits):
        anchor, lda, true_anchors = fits
        a_surf, l_surf = _anchors_surfaced(anchor, true_anchors), _anchors_surfaced(lda, true_anchors)
        a_lead, l_lead = _distinctive_lead(anchor), _distinctive_lead(lda)
        # The theory: a plain Gibbs LDA at high K loses distinctive words to the
        # frequent common background, while anchor-words does not.
        assert l_surf < 0.8, f"LDA unexpectedly recovered anchors (surf={l_surf})"
        assert a_surf >= l_surf + 0.2, f"anchor {a_surf} not clearly above LDA {l_surf}"
        assert a_lead >= l_lead + 0.15, f"anchor {a_lead} not clearly above LDA {l_lead}"

    def test_anchor_deterministic_at_high_k(self):
        docs, _, k = _overlap_corpus(k=self.K, seed=0)
        a = topica.AnchorLDA(k, seed=0).fit(docs)
        b = topica.AnchorLDA(k, seed=0).fit(docs)
        assert np.array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))


class TestTopWordRanking:
    """The Bayes inversion weights beta by raw word frequency, so the default
    FREX ranking matters for anchor-words: it removes pervasive words that a
    plain probability ranking surfaces across many topics (see PR #290)."""

    @staticmethod
    def _leaky_corpus(k=20, block=6, n_common=4, common_tok=20, n=1000, seed=0):
        """A corpus with a few ultra-frequent shared words. Because anchor's
        Bayes inversion weights beta by word frequency, those words top every
        topic under a probability ranking."""
        rng = np.random.default_rng(seed)
        blocks = [[f"b{b}_w{j}" for j in range(block)] for b in range(k)]
        anchors = [f"b{b}_anchor" for b in range(k)]
        common = [f"common_{j}" for j in range(n_common)]
        docs = []
        for d in range(n):
            b = d % k
            docs.append([anchors[b], anchors[b]]
                        + list(rng.choice(blocks[b], 28 - common_tok))
                        + list(rng.choice(common, common_tok)))
        return docs, k

    @staticmethod
    def _led_by_common(m, method):
        return float(np.mean([m.top_words(1, topic=t, method=method)[0][0].startswith("common_")
                              for t in range(m.num_topics)]))

    def test_methods_return_pairs(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        for method in ("frex", "prob", "lift"):
            tw = m.top_words(5, topic=0, method=method)
            assert len(tw) == 5 and all(isinstance(w, str) for w, _ in tw)

    def test_bad_method_raises(self):
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, seed=0).fit(docs)
        with pytest.raises(ValueError, match="method must be"):
            m.top_words(5, topic=0, method="nope")

    def test_frex_default_removes_frequent_word_leakage(self):
        # With the exact inversion (frequency_temper=1), raw probability leads
        # (nearly) every topic with a shared frequent word; the default FREX
        # ranking does not. (frequency_temper<1 also fixes it, hence =1 here to
        # isolate the ranking effect.)
        docs, k = self._leaky_corpus(seed=0)
        m = topica.AnchorLDA(k, frequency_temper=1.0, seed=0).fit(docs)
        assert self._led_by_common(m, "prob") >= 0.8
        assert self._led_by_common(m, "frex") <= 0.2
        assert self._led_by_common(m, "lift") <= 0.2

    def test_frex_is_default(self):
        docs, k = self._leaky_corpus(seed=0)
        m = topica.AnchorLDA(k, seed=0).fit(docs)
        default = [m.top_words(8, topic=t) for t in range(k)]
        frex = [m.top_words(8, topic=t, method="frex") for t in range(k)]
        assert default == frex


class TestFrequencyTemper:
    def test_default_is_half(self):
        m = topica.AnchorLDA(4)
        assert m.frequency_temper == 0.5

    def test_temper_reduces_frequent_word_leakage(self):
        # Tempering the inversion (gamma<1) removes the frequent-word domination
        # from the topic-word matrix itself, seen via a plain probability ranking.
        docs, k = TestTopWordRanking._leaky_corpus(seed=0)
        led = lambda m: TestTopWordRanking._led_by_common(m, "prob")
        exact = topica.AnchorLDA(k, frequency_temper=1.0, seed=0).fit(docs)
        tempered = topica.AnchorLDA(k, frequency_temper=0.5, seed=0).fit(docs)
        assert led(exact) >= 0.8
        assert led(tempered) <= 0.2

    def test_gamma_one_is_exact_bayes(self):
        # frequency_temper=1 reproduces the exact Bayes inversion beta ∝ p(t|w)·p(w).
        docs, _, _ = _separable_corpus()
        a = topica.AnchorLDA(4, frequency_temper=1.0, min_count=2, seed=0).fit(docs)
        b = topica.AnchorLDA(4, frequency_temper=0.5, min_count=2, seed=0).fit(docs)
        # Different exponents give different topic-word matrices.
        assert not np.allclose(np.asarray(a.topic_word), np.asarray(b.topic_word))


class TestKLStability:
    """RecoverKL must not diverge to NaN on sparse co-occurrence (large vocab,
    short documents), where words that never co-occur with an anchor drive recon
    to its floor and explode the exponentiated-gradient step (see 20NG)."""

    @staticmethod
    def _sparse_corpus(k=20, block=30, n=2000, length=5, seed=0):
        rng = np.random.default_rng(seed)
        blocks = [[f"b{b}_w{j}" for j in range(block)] for b in range(k)]
        anchors = [f"b{b}_a" for b in range(k)]
        return [[anchors[d % k]] + list(rng.choice(blocks[d % k], length - 1))
                for d in range(n)]

    def test_no_nan_on_sparse_cooccurrence(self):
        docs = self._sparse_corpus()
        m = topica.AnchorLDA(20, min_count=1, seed=0).fit(docs)  # recover="kl"
        tw = np.asarray(m.topic_word)
        dt = np.asarray(m.doc_topic)
        assert np.isfinite(tw).all(), "topic_word has NaN/inf"
        assert np.isfinite(dt).all(), "doc_topic has NaN/inf"
        # And the topics are non-degenerate (the NaN collapse made them identical).
        tops = [[w for w, _ in m.top_words(8, topic=t)] for t in range(20)]
        flat = [w for t in tops for w in t]
        assert len(set(flat)) / len(flat) > 0.5


class TestAnchorDocFreq:
    def test_default_is_small_fraction(self):
        assert topica.AnchorLDA(4).anchor_min_doc_freq == 0.01

    def test_restriction_excludes_rare_anchor_words(self):
        # A distinctive but very rare word (one document) should be eligible as an
        # anchor only when the doc-frequency floor is off.
        rng = np.random.default_rng(0)
        common = [f"w{j}" for j in range(40)]
        docs = [list(rng.choice(common, 15)) for _ in range(400)]
        docs[0] = docs[0] + ["rareanchor", "rareanchor"]   # appears in 1 doc
        off = topica.AnchorLDA(5, min_count=1, anchor_min_doc_freq=0.0, seed=0).fit(docs)
        on = topica.AnchorLDA(5, min_count=1, anchor_min_doc_freq=0.05, seed=0).fit(docs)
        assert "rareanchor" in off.anchors
        assert "rareanchor" not in on.anchors

    def test_too_few_candidates_falls_back(self):
        # An impossibly high floor leaves < num_topics candidates; the fit should
        # still succeed (restriction dropped) rather than fail.
        docs, _, _ = _separable_corpus()
        m = topica.AnchorLDA(4, min_count=2, anchor_min_doc_freq=0.999, seed=0).fit(docs)
        assert len(m.anchors) == 4
        assert np.isfinite(np.asarray(m.topic_word)).all()


def test_no_cooccurrence_evidence_raises():
    # Every document has a single token, so no document has >= 2 in-vocab tokens
    # and the co-occurrence estimator Q has no evidence. Previously this hit a
    # divide-by-zero in _build_q and returned an all-NaN (but fitted-looking)
    # model; it must raise a clear ValueError instead (#449).
    docs = [[f"w{i % 4}"] for i in range(40)]
    with pytest.raises(ValueError, match="co-occurrence"):
        topica.AnchorLDA(2, seed=0).fit(docs)
