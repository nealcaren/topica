"""Tests for new LDA inference APIs and module-level tokenize.

Covers:
- LDA.transform()
- LDA.top_documents()
- LDA.topic_divergence (property)
- LDA.similar_documents()
- topica.tokenize()
"""

import numpy as np
import numpy.testing as npt
import pytest

from topica.models import LDA
from topica import Corpus, tokenize


# ---------------------------------------------------------------------------
# Shared well-separated two-cluster corpus (~40 animal + ~40 space docs)
# Built once per session for speed.
# ---------------------------------------------------------------------------

ANIMAL_WORDS = ["cat", "dog", "fish", "bird", "horse", "rabbit", "hamster", "turtle"]
SPACE_WORDS = ["planet", "star", "moon", "rocket", "galaxy", "asteroid", "comet", "nebula"]


def _make_cluster_docs():
    rng = np.random.default_rng(42)
    animal_docs = [
        [ANIMAL_WORDS[rng.integers(len(ANIMAL_WORDS))] for _ in range(int(rng.integers(5, 12)))]
        for _ in range(40)
    ]
    space_docs = [
        [SPACE_WORDS[rng.integers(len(SPACE_WORDS))] for _ in range(int(rng.integers(5, 12)))]
        for _ in range(40)
    ]
    return animal_docs, space_docs


@pytest.fixture(scope="module")
def cluster_model():
    """Fitted LDA on 40 animal + 40 space docs. Topic 0 = animal, topic 1 = space."""
    animal_docs, space_docs = _make_cluster_docs()
    all_docs = animal_docs + space_docs
    model = LDA(2, seed=1, optimize_interval=0)
    model.fit(all_docs, iters=300, num_samples=3, sample_interval=10)
    # Sanity: topics are distinct
    vocab = model.vocabulary
    animal_t = int(model.topic_word[:, vocab.index("cat")].argmax())
    space_t = int(model.topic_word[:, vocab.index("planet")].argmax())
    assert animal_t != space_t, "Fixture: clusters not separated — adjust seed or iterations"
    return model, animal_t, space_t


@pytest.fixture(scope="module")
def animal_topic(cluster_model):
    _, at, _ = cluster_model
    return at


@pytest.fixture(scope="module")
def space_topic(cluster_model):
    _, _, st = cluster_model
    return st


@pytest.fixture(scope="module")
def model(cluster_model):
    m, _, _ = cluster_model
    return m


# ---------------------------------------------------------------------------
# Unfitted guards
# ---------------------------------------------------------------------------

class TestUnfittedGuards:
    def test_transform_raises_before_fit(self):
        m = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.transform([["cat", "dog"]])

    def test_top_documents_raises_before_fit(self):
        m = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.top_documents(0)

    def test_topic_divergence_raises_before_fit(self):
        m = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_divergence  # property, no parentheses

    def test_similar_documents_raises_before_fit(self):
        m = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.similar_documents(0)


# ---------------------------------------------------------------------------
# LDA.transform()
# ---------------------------------------------------------------------------

class TestTransform:
    def test_shape(self, model):
        new_docs = [["cat", "dog"], ["planet", "moon"]]
        result = model.transform(new_docs, seed=0)
        assert result.shape == (2, 2)

    def test_rows_sum_to_one(self, model):
        new_docs = [["cat", "dog"], ["planet", "moon"], ["fish", "bird"]]
        result = model.transform(new_docs, seed=0)
        npt.assert_allclose(result.sum(axis=1), np.ones(3), atol=1e-6)

    def test_animal_docs_assign_animal_topic(self, model, animal_topic):
        new_animal = [["cat", "dog", "fish"], ["bird", "horse", "cat"]]
        result = model.transform(new_animal, seed=0)
        for row in result:
            assert row.argmax() == animal_topic

    def test_space_docs_assign_space_topic(self, model, space_topic):
        new_space = [["planet", "star", "moon"], ["rocket", "galaxy", "comet"]]
        result = model.transform(new_space, seed=0)
        for row in result:
            assert row.argmax() == space_topic

    def test_deterministic_with_seed(self, model):
        new_docs = [["cat", "dog", "fish"], ["planet", "rocket"]]
        t1 = model.transform(new_docs, seed=0)
        t2 = model.transform(new_docs, seed=0)
        assert np.array_equal(t1, t2)

    def test_accepts_corpus_input(self, model, animal_topic):
        new_docs = [["cat", "dog", "fish"], ["bird", "horse"]]
        corpus = Corpus.from_documents(new_docs)
        result = model.transform(corpus, seed=0)
        assert result.shape == (2, 2)
        npt.assert_allclose(result.sum(axis=1), np.ones(2), atol=1e-6)
        assert result[0].argmax() == animal_topic

    def test_corpus_and_list_give_same_result(self, model):
        new_docs = [["cat", "dog", "fish"], ["planet", "moon"]]
        corpus = Corpus.from_documents(new_docs)
        from_list = model.transform(new_docs, seed=0)
        from_corpus = model.transform(corpus, seed=0)
        assert np.array_equal(from_list, from_corpus)

    def test_oov_only_doc_is_finite_and_sums_to_one(self, model):
        """A doc with only OOV tokens should return the prior theta (finite, sums to 1)."""
        oov_doc = [["xyz123", "zzz", "unknown_word_xyz"]]
        result = model.transform(oov_doc, seed=0)
        assert result.shape == (1, 2)
        assert np.all(np.isfinite(result))
        npt.assert_allclose(result.sum(axis=1), np.ones(1), atol=1e-6)

    def test_values_in_zero_one(self, model):
        new_docs = [["cat", "dog"], ["planet", "star"], ["fish"]]
        result = model.transform(new_docs, seed=0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


# ---------------------------------------------------------------------------
# LDA.top_documents()
# ---------------------------------------------------------------------------

class TestTopDocuments:
    def test_returns_list_of_tuples(self, model):
        result = model.top_documents(0, n=5)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            name, weight = item
            assert isinstance(name, str)
            assert isinstance(weight, float)

    def test_length_at_most_n(self, model):
        for n in [1, 5, 10]:
            result = model.top_documents(0, n=n)
            assert len(result) <= n

    def test_weights_descending(self, model):
        result = model.top_documents(0, n=10)
        weights = [w for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_weights_in_zero_one(self, model):
        result = model.top_documents(0, n=10)
        for _, w in result:
            assert 0 <= w <= 1

    def test_names_are_valid_doc_names(self, model):
        valid = set(model.doc_names)
        for topic in range(2):
            result = model.top_documents(topic, n=10)
            for name, _ in result:
                assert name in valid

    def test_animal_topic_top_docs_are_animal_docs(self, model, animal_topic):
        """Top docs for animal topic should be from the first 40 (indices 0-39)."""
        result = model.top_documents(animal_topic, n=10)
        doc_names = model.doc_names
        # All top-10 animal docs should have index < 40
        for name, _ in result:
            idx = doc_names.index(name)
            assert idx < 40, f"Expected animal doc (idx<40), got {name} at idx {idx}"

    def test_space_topic_top_docs_are_space_docs(self, model, space_topic):
        """Top docs for space topic should be from the last 40 (indices 40-79)."""
        result = model.top_documents(space_topic, n=10)
        doc_names = model.doc_names
        for name, _ in result:
            idx = doc_names.index(name)
            assert idx >= 40, f"Expected space doc (idx>=40), got {name} at idx {idx}"

    def test_out_of_range_raises_value_error(self, model):
        with pytest.raises(ValueError):
            model.top_documents(99)

    def test_default_n_returns_at_most_10(self, model):
        result = model.top_documents(0)
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# LDA.topic_divergence (property)
# ---------------------------------------------------------------------------

class TestTopicDivergence:
    def test_is_property_not_method(self, model):
        """Access without parentheses — it is a property."""
        D = model.topic_divergence  # no ()
        assert isinstance(D, np.ndarray)

    def test_shape(self, model):
        D = model.topic_divergence
        assert D.shape == (2, 2)

    def test_zero_diagonal(self, model):
        D = model.topic_divergence
        npt.assert_allclose(np.diag(D), np.zeros(2), atol=1e-10)

    def test_symmetric(self, model):
        D = model.topic_divergence
        npt.assert_allclose(D, D.T, atol=1e-10)

    def test_values_in_zero_one(self, model):
        D = model.topic_divergence
        assert np.all(D >= 0)
        assert np.all(D <= 1)

    def test_high_divergence_between_well_separated_topics(self, model):
        """Two clearly distinct topics (animal vs space) should have high divergence."""
        D = model.topic_divergence
        # Off-diagonal should exceed 0.5 for well-separated clusters
        assert D[0, 1] > 0.5, f"Expected D[0,1] > 0.5 for well-separated clusters, got {D[0,1]}"


# ---------------------------------------------------------------------------
# LDA.similar_documents()
# ---------------------------------------------------------------------------

class TestSimilarDocuments:
    def test_returns_list_of_tuples(self, model):
        result = model.similar_documents(0, n=5)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            name, div = item
            assert isinstance(name, str)
            assert isinstance(div, float)

    def test_length_at_most_n(self, model):
        for n in [1, 5, 10]:
            result = model.similar_documents(0, n=n)
            assert len(result) <= n

    def test_ascending_divergence(self, model):
        result = model.similar_documents(0, n=10)
        divs = [d for _, d in result]
        assert divs == sorted(divs)

    def test_query_doc_excluded(self, model):
        """The query document itself should not appear in the results."""
        query_name = model.doc_names[0]
        result = model.similar_documents(0, n=20)
        names = [n for n, _ in result]
        assert query_name not in names

    def test_animal_doc_neighbors_are_animal_docs(self, model, animal_topic):
        """Doc 0 is an animal doc; its nearest neighbors should be other animal docs."""
        result = model.similar_documents(0, n=5)
        doc_names = model.doc_names
        for name, _ in result:
            idx = doc_names.index(name)
            assert idx < 40, f"Expected animal neighbor (idx<40), got {name} at idx {idx}"

    def test_names_are_valid_doc_names(self, model):
        valid = set(model.doc_names)
        result = model.similar_documents(0, n=10)
        for name, _ in result:
            assert name in valid

    def test_divergence_values_in_zero_one(self, model):
        result = model.similar_documents(0, n=10)
        for _, d in result:
            assert 0 <= d <= 1

    def test_out_of_range_raises_value_error(self, model):
        with pytest.raises(ValueError):
            model.similar_documents(999)

    def test_default_n_returns_at_most_10(self, model):
        result = model.similar_documents(0)
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# topica.tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_returns_list_of_strings(self):
        result = tokenize("hello world")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_lowercases_by_default(self):
        result = tokenize("Hello WORLD Foo")
        assert result == ["hello", "world", "foo"]

    def test_no_lowercase_when_disabled(self):
        result = tokenize("Hello WORLD", lowercase=False)
        assert "Hello" in result
        assert "WORLD" in result

    def test_stopwords_removed(self):
        result = tokenize("the quick brown fox", stopwords=["the", "fox"])
        assert "the" not in result
        assert "fox" not in result
        assert "quick" in result
        assert "brown" in result

    def test_min_length_filter(self):
        result = tokenize("ab abc abcd", min_length=3)
        assert "ab" not in result
        assert "abc" in result
        assert "abcd" in result

    def test_bare_punctuation_dropped(self):
        """Default regex drops bare punctuation and numbers."""
        result = tokenize("hello, world! 123 test.")
        assert "," not in result
        assert "!" not in result
        assert "123" not in result

    def test_hyphenated_words_kept(self):
        """Default regex keeps hyphenated tokens."""
        result = tokenize("well-known state-of-the-art")
        assert "well-known" in result

    def test_abbreviation_kept(self):
        """Default regex keeps abbreviations like U.S.A -> u.s.a (lowercased)."""
        result = tokenize("U.S.A is great")
        assert "u.s.a" in result

    def test_custom_token_regex(self):
        result = tokenize("hello world 123 abc", token_regex=r"[a-zA-Z]+")
        assert "123" not in result
        assert "hello" in result
        assert "world" in result

    def test_invalid_regex_raises_value_error(self):
        with pytest.raises(ValueError):
            tokenize("test", token_regex="[invalid")

    def test_empty_string_returns_empty_list(self):
        result = tokenize("")
        assert result == []

    def test_all_stopwords_returns_empty_list(self):
        result = tokenize("the a an", stopwords=["the", "a", "an"])
        assert result == []
