"""Tests for make_heldout / eval_heldout and the un-gated search_k perplexity.

Issue #38: R stm-style within-corpus word-heldout diagnostic, model-agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica
from topica.diagnostics import Heldout, HeldoutResult


# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------
# Two clearly separated topic clusters with 8 words each.
# Large enough that models can reliably learn structure in few iterations.

_A = [f"a{i}" for i in range(8)]
_B = [f"b{i}" for i in range(8)]


def _planted(n_per=60, doc_len=20, seed=0):
    """Return two-topic corpus: first half draws from A, second from B."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_per):
        docs.append(list(rng.choice(_A, size=doc_len, replace=True)))
    for _ in range(n_per):
        docs.append(list(rng.choice(_B, size=doc_len, replace=True)))
    return docs


@pytest.fixture(scope="module")
def corpus():
    return _planted(n_per=60, doc_len=20, seed=0)


# ---------------------------------------------------------------------------
# make_heldout: structure and correctness
# ---------------------------------------------------------------------------

class TestMakeHeldout:
    def test_documents_length_unchanged(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        assert len(h.documents) == len(corpus)

    def test_sampled_docs_are_shorter(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        for doc_idx, _ in h.missing:
            assert len(h.documents[doc_idx]) < len(corpus[doc_idx])

    def test_unsampled_docs_unchanged(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        sampled = set(h.doc_indices.tolist())
        for i, orig in enumerate(corpus):
            if i not in sampled:
                assert h.documents[i] == orig

    def test_missing_holds_exactly_removed_tokens(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        for doc_idx, held_tokens in h.missing:
            orig = corpus[doc_idx]
            retained = h.documents[doc_idx]
            # retained + held_tokens should reconstruct orig (as a multiset)
            assert sorted(retained + held_tokens) == sorted(orig)

    def test_deterministic_for_fixed_seed(self, corpus):
        h1 = topica.diagnostics.make_heldout(corpus, seed=7)
        h2 = topica.diagnostics.make_heldout(corpus, seed=7)
        assert h1.doc_indices.tolist() == h2.doc_indices.tolist()
        for (i1, t1), (i2, t2) in zip(h1.missing, h2.missing):
            assert i1 == i2
            assert t1 == t2

    def test_different_seeds_differ(self, corpus):
        h1 = topica.diagnostics.make_heldout(corpus, seed=0)
        h2 = topica.diagnostics.make_heldout(corpus, seed=99)
        # With high probability the sampled sets differ for different seeds
        assert h1.doc_indices.tolist() != h2.doc_indices.tolist()

    def test_prop_docs_honored(self, corpus):
        D = len(corpus)
        for prop in (0.2, 0.5, 0.8):
            h = topica.diagnostics.make_heldout(corpus, prop_docs=prop, seed=0)
            expected = int(np.floor(prop * D))
            # Allow for a few docs dropped (too short) but never more than expected
            assert len(h.doc_indices) <= expected
            # And at least 90% of expected should make it (our docs are long)
            assert len(h.doc_indices) >= max(1, int(0.9 * expected))

    def test_prop_words_honored(self, corpus):
        for prop in (0.2, 0.5, 0.8):
            h = topica.diagnostics.make_heldout(corpus, prop_words=prop, seed=0)
            for doc_idx, held_tokens in h.missing:
                n_orig = len(corpus[doc_idx])
                expected_hold = int(np.floor(prop * n_orig))
                assert len(held_tokens) == expected_hold

    def test_short_docs_handled_without_error(self):
        # Some docs too short to split (1 token); make_heldout must not raise
        docs = [["only"], ["one"], ["token"]] + _planted(n_per=10, doc_len=15, seed=1)
        h = topica.diagnostics.make_heldout(docs, prop_docs=0.5, seed=0)
        # The single-token docs are skipped; the rest work fine
        assert h.documents is not None

    def test_accepts_corpus_object(self, corpus):
        c = topica.Corpus.from_documents(corpus)
        h = topica.diagnostics.make_heldout(c, seed=0)
        assert len(h.documents) == len(corpus)

    def test_doc_indices_sorted(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        idx = h.doc_indices.tolist()
        assert idx == sorted(idx)

    def test_returns_heldout_dataclass(self, corpus):
        h = topica.diagnostics.make_heldout(corpus, seed=0)
        assert isinstance(h, Heldout)
        assert hasattr(h, "documents")
        assert hasattr(h, "missing")
        assert hasattr(h, "doc_indices")


# ---------------------------------------------------------------------------
# eval_heldout: model-agnostic scoring
# ---------------------------------------------------------------------------

def _fit_and_eval(model_cls, docs, heldout, **fit_kwargs):
    """Fit model_cls on heldout.documents and return eval_heldout result."""
    m = model_cls(2, seed=1)
    m.fit(heldout.documents, **fit_kwargs)
    return topica.diagnostics.eval_heldout(m, heldout, seed=0)


class TestEvalHeldout:
    @pytest.fixture(scope="class")
    def heldout(self):
        return topica.diagnostics.make_heldout(_planted(n_per=60, doc_len=20, seed=0), seed=0)

    def test_returns_heldoutresult_dataclass(self, heldout):
        m = topica.models.LDA(2, seed=1)
        m.fit(heldout.documents, iters=100)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert isinstance(result, HeldoutResult)

    def test_lda_finite_negative_loglik(self, heldout):
        m = topica.models.LDA(2, seed=1)
        m.fit(heldout.documents, iters=100)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert np.isfinite(result.mean_per_doc_loglik)
        assert result.mean_per_doc_loglik < 0.0

    def test_stm_finite_negative_loglik(self, heldout):
        D = len(heldout.documents)
        rng = np.random.default_rng(3)
        prevalence = rng.normal(size=(D, 2))
        m = topica.models.STM(2, seed=1)
        m.fit(heldout.documents, prevalence, iters=20)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert np.isfinite(result.mean_per_doc_loglik)
        assert result.mean_per_doc_loglik < 0.0

    def test_ctm_finite_negative_loglik(self, heldout):
        m = topica.models.CTM(2, seed=1)
        m.fit(heldout.documents, iters=20)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert np.isfinite(result.mean_per_doc_loglik)
        assert result.mean_per_doc_loglik < 0.0

    def test_dmr_finite_negative_loglik(self, heldout):
        D = len(heldout.documents)
        rng = np.random.default_rng(5)
        feats = rng.normal(size=(D, 2))
        m = topica.models.DMR(2, seed=1)
        m.fit(heldout.documents, feats, iters=100)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert np.isfinite(result.mean_per_doc_loglik)
        assert result.mean_per_doc_loglik < 0.0

    def test_shapes_and_counts_consistent(self, heldout):
        m = topica.models.LDA(2, seed=1)
        m.fit(heldout.documents, iters=100)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert result.n_docs == len(result.per_doc_loglik)
        assert result.n_docs > 0
        assert result.n_tokens > 0
        # total_loglik must equal sum of per_doc_loglik
        np.testing.assert_allclose(result.total_loglik, result.per_doc_loglik.sum(), rtol=1e-9)
        # mean must match
        np.testing.assert_allclose(
            result.mean_per_doc_loglik,
            result.per_doc_loglik.mean(),
            rtol=1e-9,
        )

    def test_rejects_bertopic(self, heldout):
        rng = np.random.default_rng(0)
        D = len(heldout.documents)
        emb = rng.normal(size=(D, 8))
        m = topica.models.BERTopic(min_cluster_size=5, seed=1)
        m.fit(heldout.documents, emb)
        with pytest.raises(ValueError, match="generative|no held-out|class-based"):
            topica.diagnostics.eval_heldout(m, heldout)

    def test_rejects_top2vec(self, heldout):
        rng = np.random.default_rng(0)
        D = len(heldout.documents)
        emb = rng.normal(size=(D, 8))
        m = topica.models.Top2Vec(min_cluster_size=5, seed=1)
        m.fit(heldout.documents, emb)
        with pytest.raises(ValueError, match="generative|no held-out|class-based"):
            topica.diagnostics.eval_heldout(m, heldout)

    def test_per_doc_loglik_all_negative(self, heldout):
        m = topica.models.LDA(2, seed=1)
        m.fit(heldout.documents, iters=100)
        result = topica.diagnostics.eval_heldout(m, heldout)
        assert np.all(result.per_doc_loglik < 0.0)


# ---------------------------------------------------------------------------
# Round-trip test: make_heldout -> fit -> eval_heldout
# ---------------------------------------------------------------------------

def test_roundtrip_lda():
    docs = _planted(n_per=60, doc_len=20, seed=0)
    h = topica.diagnostics.make_heldout(docs, prop_docs=0.3, prop_words=0.5, seed=42)
    m = topica.models.LDA(2, seed=1)
    m.fit(h.documents, iters=150)
    result = topica.diagnostics.eval_heldout(m, h)
    assert isinstance(result, HeldoutResult)
    assert result.n_docs > 0
    assert np.isfinite(result.mean_per_doc_loglik)
    assert result.mean_per_doc_loglik < 0.0


def test_roundtrip_stm():
    docs = _planted(n_per=60, doc_len=20, seed=1)
    h = topica.diagnostics.make_heldout(docs, prop_docs=0.4, prop_words=0.5, seed=7)
    rng = np.random.default_rng(7)
    prevalence = rng.normal(size=(len(h.documents), 2))
    m = topica.models.STM(2, seed=1)
    m.fit(h.documents, prevalence, iters=20)
    result = topica.diagnostics.eval_heldout(m, h)
    assert isinstance(result, HeldoutResult)
    assert result.mean_per_doc_loglik < 0.0


def test_roundtrip_ctm():
    docs = _planted(n_per=60, doc_len=20, seed=2)
    h = topica.diagnostics.make_heldout(docs, prop_docs=0.4, prop_words=0.5, seed=13)
    m = topica.models.CTM(2, seed=1)
    m.fit(h.documents, iters=20)
    result = topica.diagnostics.eval_heldout(m, h)
    assert isinstance(result, HeldoutResult)
    assert result.mean_per_doc_loglik < 0.0


# ---------------------------------------------------------------------------
# search_k: un-gated held-out perplexity for model="stm" and model="lda"
# ---------------------------------------------------------------------------

class TestSearchKHeldout:
    @pytest.fixture(scope="class")
    def small_corpus_held_prevalence(self):
        rng = np.random.default_rng(0)
        docs = [list(rng.choice(["alpha", "beta", "gamma"], size=10, replace=True))
                for _ in range(20)]
        held = [list(rng.choice(["alpha", "beta", "gamma"], size=10, replace=True))
                for _ in range(6)]
        # A simple binary covariate aligned to docs (needed for STM)
        prevalence = rng.normal(size=(len(docs), 1))
        return docs, held, prevalence

    def test_lda_reports_perplexity_with_held_out(self, small_corpus_held_prevalence):
        docs, held, _ = small_corpus_held_prevalence
        rows = topica.select.search_k(docs, [2], iters=80, held_out=held, seed=0)
        assert "perplexity" in rows[0]
        assert np.isfinite(rows[0]["perplexity"]) and rows[0]["perplexity"] > 1.0

    def test_stm_reports_perplexity_with_held_out(self, small_corpus_held_prevalence):
        docs, held, prevalence = small_corpus_held_prevalence
        rows = topica.select.search_k(docs, [2], model="stm", prevalence=prevalence,
                               iters=10, held_out=held, seed=0)
        assert "perplexity" in rows[0]
        assert np.isfinite(rows[0]["perplexity"]) and rows[0]["perplexity"] > 1.0

    def test_lda_no_held_out_no_perplexity(self, small_corpus_held_prevalence):
        docs, _, _ = small_corpus_held_prevalence
        rows = topica.select.search_k(docs, [2], iters=80, seed=0)
        assert "perplexity" not in rows[0]

    def test_stm_no_held_out_no_perplexity(self, small_corpus_held_prevalence):
        docs, _, prevalence = small_corpus_held_prevalence
        rows = topica.select.search_k(docs, [2], model="stm", prevalence=prevalence,
                               iters=10, seed=0)
        assert "perplexity" not in rows[0]

    def test_existing_coherence_metric_label_intact(self, small_corpus_held_prevalence):
        docs, held, prevalence = small_corpus_held_prevalence
        for m_type, prev in (("lda", None), ("stm", prevalence)):
            rows = topica.select.search_k(docs, [2], model=m_type, prevalence=prev,
                                   held_out=held, iters=10, seed=0)
            assert rows[0]["coherence_metric"] == "u_mass"


# ---------------------------------------------------------------------------
# Issue #55: search_k(held_out=Heldout) used to raise TypeError
# ---------------------------------------------------------------------------

class TestSearchKWithHeldoutObject:
    """Regression tests for the Heldout-dispatch bug in search_k (issue #55).

    Before the fix, passing a Heldout dataclass as held_out raised
    ``TypeError: 'list' object is not callable`` because search_k called
    ``perplexity(m, held_out)`` which tried ``held_out.documents()`` — but
    ``.documents`` on a Heldout is a plain list, not a method.
    """

    @pytest.fixture(scope="class")
    def heldout_fixture(self):
        rng = np.random.default_rng(0)
        docs = [list(rng.choice(["alpha", "beta", "gamma", "delta"], size=12, replace=True))
                for _ in range(30)]
        ho = topica.diagnostics.make_heldout(docs, prop_docs=0.5, prop_words=0.5, seed=1)
        prevalence = rng.normal(size=(len(docs), 1))
        return docs, ho, prevalence

    def test_lda_heldout_object_no_typeerror(self, heldout_fixture):
        """search_k with a Heldout object must not raise TypeError."""
        docs, ho, _ = heldout_fixture
        rows = topica.select.search_k(ho.documents, [2], iters=80, held_out=ho, seed=0)
        assert len(rows) == 1

    def test_lda_heldout_object_uses_heldout_loglik_key(self, heldout_fixture):
        """Heldout path stores the metric under 'heldout_loglik', not 'perplexity'."""
        docs, ho, _ = heldout_fixture
        rows = topica.select.search_k(ho.documents, [2], iters=80, held_out=ho, seed=0)
        row = rows[0]
        assert "heldout_loglik" in row, "expected 'heldout_loglik' key in result row"
        assert "perplexity" not in row, "'perplexity' key must not appear for Heldout path"

    def test_lda_heldout_loglik_is_finite_negative(self, heldout_fixture):
        docs, ho, _ = heldout_fixture
        rows = topica.select.search_k(ho.documents, [2], iters=80, held_out=ho, seed=0)
        val = rows[0]["heldout_loglik"]
        assert np.isfinite(val), f"heldout_loglik not finite: {val}"
        assert val < 0.0, f"mean per-doc log-likelihood should be negative, got {val}"

    def test_stm_heldout_object_no_typeerror(self, heldout_fixture):
        """STM path also dispatches correctly through the Heldout branch."""
        docs, ho, prevalence = heldout_fixture
        rows = topica.select.search_k(
            ho.documents, [2], model="stm", prevalence=prevalence,
            iters=10, held_out=ho, seed=0,
        )
        assert len(rows) == 1
        row = rows[0]
        assert "heldout_loglik" in row
        assert "perplexity" not in row
        assert np.isfinite(row["heldout_loglik"])
        assert row["heldout_loglik"] < 0.0

    def test_multiple_ks_all_rows_have_heldout_loglik(self, heldout_fixture):
        """Every row produced for different K values must carry the metric."""
        docs, ho, _ = heldout_fixture
        rows = topica.select.search_k(ho.documents, [2, 3], iters=50, held_out=ho, seed=0)
        assert len(rows) == 2
        for row in rows:
            assert "heldout_loglik" in row
            assert np.isfinite(row["heldout_loglik"])

    def test_legacy_corpus_list_still_produces_perplexity(self, heldout_fixture):
        """The legacy path (token lists, not Heldout) still yields 'perplexity'."""
        docs, ho, _ = heldout_fixture
        rng = np.random.default_rng(42)
        held_docs = [list(rng.choice(["alpha", "beta", "gamma", "delta"], size=12, replace=True))
                     for _ in range(8)]
        rows = topica.select.search_k(docs, [2], iters=80, held_out=held_docs, seed=0)
        row = rows[0]
        assert "perplexity" in row
        assert "heldout_loglik" not in row
        assert np.isfinite(row["perplexity"])
        assert row["perplexity"] > 1.0


# ---------------------------------------------------------------------------
# plot_search_k with heldout_loglik metric
# ---------------------------------------------------------------------------

def test_plot_search_k_accepts_heldout_loglik_metric():
    """plot_search_k must plot 'heldout_loglik' rows without raising."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")

    rows = [
        {"k": 2, "coherence": -10.0, "exclusivity": 0.6, "heldout_loglik": -5.2},
        {"k": 3, "coherence": -12.0, "exclusivity": 0.5, "heldout_loglik": -4.8},
    ]
    ax = topica.viz.plot_search_k(rows, metrics=("coherence", "heldout_loglik"))
    assert ax is not None


def test_plot_search_k_mixed_rows_no_error():
    """plot_search_k must silently skip metrics absent from all rows."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")

    rows = [
        {"k": 2, "coherence": -10.0, "exclusivity": 0.6},
        {"k": 3, "coherence": -12.0, "exclusivity": 0.5},
    ]
    # asking for heldout_loglik when no row has it should raise (no metrics left)
    with pytest.raises(ValueError):
        topica.viz.plot_search_k(rows, metrics=("heldout_loglik",))


# ---------------------------------------------------------------------------
# Optional: better-fit model should score higher held-out LL
# ---------------------------------------------------------------------------

def test_better_fit_scores_higher_loglik():
    """A well-fit K=2 model should score higher mean held-out LL than K=1 on
    clearly separated two-topic data.  Use wide margins to keep this robust."""
    docs = _planted(n_per=80, doc_len=20, seed=0)
    h = topica.diagnostics.make_heldout(docs, prop_docs=0.5, prop_words=0.5, seed=0)

    m_good = topica.models.LDA(2, seed=1)
    m_good.fit(h.documents, iters=300)

    m_bad = topica.models.LDA(1, seed=1)
    m_bad.fit(h.documents, iters=300)

    r_good = topica.diagnostics.eval_heldout(m_good, h)
    r_bad = topica.diagnostics.eval_heldout(m_bad, h)

    # K=2 should have meaningfully higher (less negative) mean held-out LL
    assert r_good.mean_per_doc_loglik > r_bad.mean_per_doc_loglik
