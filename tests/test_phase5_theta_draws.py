"""Phase 5: theta_draws and doc_lengths on the remaining Dirichlet-family models.

Covers DMR, SAGE, LabeledLDA, HDP, PT, PA, and SupervisedLDA.
"""

import numpy as np
import pytest

import topica
from topica.effects import composition_theta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _toy_docs(n=60, vocab=16, lo=8, hi=18, seed=7):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocab)]
    return [list(rng.choice(words, size=int(rng.integers(lo, hi)))) for _ in range(n)]


def _toy_corpus(seed=7):
    return topica.Corpus.from_documents(_toy_docs(seed=seed))


# ---------------------------------------------------------------------------
# DMR
# ---------------------------------------------------------------------------

def _fit_dmr(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    n = corpus.num_docs
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, 2)).tolist()
    m = topica.models.DMR(num_topics=4, seed=1)
    m.fit(corpus, features, iters=120,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestDMR:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_theta_draws_vary_across_draws(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus, num_theta_draws=10)
        td = np.asarray(m.theta_draws)
        assert td.std() > 1e-4

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths_shape(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_dmr(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# SAGE
# ---------------------------------------------------------------------------

def _fit_sage(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    n = corpus.num_docs
    groups = ["a" if i < n // 2 else "b" for i in range(n)]
    m = topica.models.SAGE(num_topics=4, seed=1)
    m.fit(corpus, groups, iters=120,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestSAGE:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_sage(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_sage(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_sage(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_sage(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_sage(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# LabeledLDA
# ---------------------------------------------------------------------------

def _fit_labeled(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    n = corpus.num_docs
    labels = [["topic_a"] if i % 3 != 0 else ["topic_b"] for i in range(n)]
    m = topica.models.LabeledLDA(alpha=0.1, beta=0.01, seed=1)
    m.fit(corpus, labels, iters=120,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestLabeledLDA:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_labeled(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_labeled(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_labeled(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_labeled(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_labeled(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# HDP
# ---------------------------------------------------------------------------

def _fit_hdp(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    m = topica.models.HDP(seed=1)
    m.fit(corpus, iters=80,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestHDP:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_hdp(corpus, num_theta_draws=8)
        assert m.theta_draws is not None
        td = np.asarray(m.theta_draws)
        k = m.num_topics
        d = np.asarray(m.doc_topic).shape[0]
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_hdp(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_hdp(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_hdp(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_self_sufficient_for_composition(self):
        corpus = _toy_corpus()
        m = _fit_hdp(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# PT (Pseudo-document Topic Model)
# ---------------------------------------------------------------------------

def _fit_pt(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    m = topica.models.PT(num_topics=4, num_pseudo=20, seed=1)
    m.fit(corpus, iters=200,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestPT:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_pt(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_pt(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_pt(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_pt(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_pt(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# PA (Pachinko Allocation Model)
# ---------------------------------------------------------------------------

def _fit_pa(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    m = topica.models.PA(num_super=2, num_sub=4, seed=1)
    m.fit(corpus, iters=300,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestPA:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_pa(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_pa(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_pa(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_pa(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_pa(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])


# ---------------------------------------------------------------------------
# SupervisedLDA
# ---------------------------------------------------------------------------

def _fit_slda(corpus, *, keep_theta_draws=True, num_theta_draws=10):
    n = corpus.num_docs
    rng = np.random.default_rng(42)
    y = rng.standard_normal(n).tolist()
    m = topica.models.SupervisedLDA(num_topics=4, seed=1)
    m.fit(corpus, y, iters=15,
          keep_theta_draws=keep_theta_draws, num_theta_draws=num_theta_draws)
    return m


class TestSupervisedLDA:
    def test_theta_draws_shape(self):
        corpus = _toy_corpus()
        m = _fit_slda(corpus, num_theta_draws=8)
        td = np.asarray(m.theta_draws)
        d, k = np.asarray(m.doc_topic).shape
        assert td.shape == (8, d, k)
        assert td.dtype == np.float32

    def test_theta_draws_rows_sum_to_one(self):
        corpus = _toy_corpus()
        m = _fit_slda(corpus, num_theta_draws=5)
        td = np.asarray(m.theta_draws)
        row_sums = td.sum(axis=2)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_keep_false_returns_none(self):
        corpus = _toy_corpus()
        m = _fit_slda(corpus, keep_theta_draws=False)
        assert m.theta_draws is None

    def test_doc_lengths(self):
        corpus = _toy_corpus()
        m = _fit_slda(corpus)
        dl = m.doc_lengths
        assert len(dl) == corpus.num_docs
        assert all(n > 0 for n in dl)

    def test_composition_theta_works(self):
        corpus = _toy_corpus()
        m = _fit_slda(corpus, num_theta_draws=5)
        out = composition_theta(m, corpus, nsims=3)
        assert out.shape == (3, corpus.num_docs, np.asarray(m.doc_topic).shape[1])
