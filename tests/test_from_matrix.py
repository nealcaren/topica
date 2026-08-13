"""``Corpus.from_matrix`` — the document x term count-matrix entry point (#575).

The contract exists because the caller's column indices are usually load-bearing
(an SAE feature id, an external vectorizer's vocabulary), unlike ``from_documents``
which derives and frequency-sorts the vocabulary from the data. Each guarantee below
is one a Mechanistic Topic Model depends on.
"""

import numpy as np
import pytest

import topica

COUNTS = np.array([[2, 0, 1, 0],
                   [0, 0, 0, 0],     # an all-zero document
                   [1, 1, 0, 0],
                   [3, 0, 0, 0]], dtype=np.int64)
NAMES = ["a", "b", "c", "zero_col"]   # zero_col never occurs


class TestColumnContract:
    def test_column_order_is_the_callers(self):
        # from_documents frequency-sorts; from_matrix must not, or topic_word[:, j]
        # stops lining up with the caller's column j.
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        assert list(c.vocabulary) == NAMES

    def test_all_zero_columns_are_kept(self):
        # The reference's feature width is fixed by its filter intersection, so a
        # silently dropped empty column would reindex every downstream feature id.
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        assert c.num_words == COUNTS.shape[1]
        assert "zero_col" in list(c.vocabulary)

    def test_topic_word_width_matches_the_input(self):
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        m = topica.LDA(num_topics=2, seed=13).fit(c, iters=30)
        assert np.asarray(m.topic_word).shape[1] == COUNTS.shape[1]

    def test_default_feature_names(self):
        c = topica.Corpus.from_matrix(COUNTS)
        assert list(c.vocabulary) == ["f0", "f1", "f2", "f3"]


class TestRowContract:
    def test_empty_rows_are_kept_and_indices_are_identity(self):
        # doc_topic rows must stay aligned with the caller's external metadata.
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        assert c.num_docs == COUNTS.shape[0]
        assert list(c.kept_indices) == list(range(COUNTS.shape[0]))

    def test_doc_lengths_are_the_row_sums(self):
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        assert list(c.doc_lengths) == [int(r.sum()) for r in COUNTS]
        assert c.total_tokens == int(COUNTS.sum())


class TestUbiquitousFilter:
    def test_max_doc_fraction_prunes_and_reports_kept_features(self):
        # App. A.1's filter; the one operation that may change the width.
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES, max_doc_fraction=0.5)
        assert list(c.vocabulary) == ["b", "c", "zero_col"]   # "a" is in 3/4 docs
        assert c.kept_features == [1, 2, 3]

    def test_identity_kept_features_when_unfiltered(self):
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES)
        assert c.kept_features == list(range(COUNTS.shape[1]))

    def test_zero_fraction_keeps_only_never_occurring_terms(self):
        # max_doc_fraction is an UPPER bound: "drop terms in more than this
        # fraction of documents". A term in 0% of documents is not ubiquitous, so
        # the all-zero column is exactly what survives at 0.0.
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES, max_doc_fraction=0.0)
        assert list(c.vocabulary) == ["zero_col"]

    def test_pruning_everything_raises(self):
        # No zero columns here, so a 0.0 bound leaves nothing.
        dense = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(ValueError, match="pruned every term"):
            topica.Corpus.from_matrix(dense, max_doc_fraction=0.0)


class TestNTokens:
    def test_n_tokens_is_stored_separately_from_the_row_sum(self):
        # For thresholded SAE activations the row sum counts feature activations,
        # NOT tokens; mETM's Binomial n needs the true count (paper eq. 12).
        c = topica.Corpus.from_matrix(COUNTS, feature_names=NAMES,
                                      n_tokens=[100, 50, 80, 90])
        assert c.n_tokens == [100, 50, 80, 90]
        assert c.n_tokens != list(c.doc_lengths)

    def test_absent_by_default(self):
        assert topica.Corpus.from_matrix(COUNTS).n_tokens is None

    def test_other_constructors_report_none(self):
        c = topica.Corpus.from_documents([["a", "b"], ["c"]])
        assert c.n_tokens is None and c.kept_features is None


class TestValidation:
    @pytest.mark.parametrize("kwargs, match", [
        (dict(feature_names=["a"]), "feature_names has 1"),
        (dict(doc_names=["x"]), "doc_names has 1"),
        (dict(doc_labels=["x"]), "doc_labels has 1"),
        (dict(n_tokens=[1]), "n_tokens has 1"),
        (dict(max_doc_fraction=2.0), r"max_doc_fraction must be in \[0.0, 1.0\]"),
    ])
    def test_shape_and_range_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            topica.Corpus.from_matrix(COUNTS, **kwargs)

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            topica.Corpus.from_matrix(np.array([[-1, 0]], dtype=np.int64))

    def test_empty_matrix_rejected(self):
        with pytest.raises(ValueError, match="at least one document and one term"):
            topica.Corpus.from_matrix(np.zeros((0, 3), dtype=np.int64))


class TestEquivalence:
    def test_matches_the_pseudo_token_expansion(self):
        # from_matrix must be exactly the bag-of-words expansion, since that is
        # what the MTM prototype used and what the parity numbers were measured on.
        rng = np.random.default_rng(13)
        counts = rng.integers(0, 5, size=(30, 12)).astype(np.int64)
        names = [f"f{j}" for j in range(12)]
        docs = [[names[j] for j in np.repeat(np.arange(12), row)] for row in counts]

        a = topica.Corpus.from_matrix(counts, feature_names=names)
        b = topica.Corpus.from_documents(docs, vocabulary=names)
        # Same vocabulary, and identical per-document token multisets.
        assert list(a.vocabulary) == list(b.vocabulary)
        assert sorted(a.doc_lengths) == sorted(b.doc_lengths)
        assert a.total_tokens == b.total_tokens
