"""Vocabulary control on Corpus: ``max_features``, a fixed ``vocabulary=``, and
``Corpus.transform`` for held-out documents (scikit-learn / gensim parity)."""

import numpy as np
import pytest

import topica


# --- max_features ---------------------------------------------------------


def test_max_features_keeps_top_n_by_total_frequency():
    # totals: cat=5, dog=4, fish=3, bird=1, tree=1
    docs = [
        ["cat", "cat", "dog", "fish"],
        ["dog", "dog", "bird"],
        ["cat", "fish", "fish", "tree"],
        ["dog", "cat", "cat"],
    ]
    c = topica.Corpus.from_documents(docs, max_features=3)
    assert set(c.vocabulary) == {"cat", "dog", "fish"}
    assert c.num_words == 3


def test_max_features_larger_than_vocab_is_a_noop():
    docs = [["a", "b"], ["b", "c"]]
    c = topica.Corpus.from_documents(docs, max_features=100)
    assert set(c.vocabulary) == {"a", "b", "c"}


def test_max_features_tie_break_is_deterministic_first_appearance():
    # "alpha" and "omega" both have total frequency 1; "alpha" appears first, so
    # with max_features=1 it wins the tie (ties break by ascending vocab id).
    docs = [["alpha", "omega"]]
    c = topica.Corpus.from_documents(docs, max_features=1)
    assert c.vocabulary == ["alpha"]


def test_max_features_applies_after_rm_top():
    # totals: the=10, cat=4, dog=3, fish=2. rm_top=1 removes "the"; max_features=2
    # then keeps the top 2 survivors: cat, dog.
    docs = [
        ["the"] * 10,
        ["cat"] * 4,
        ["dog"] * 3,
        ["fish"] * 2,
    ]
    c = topica.Corpus.from_documents(docs, rm_top=1, max_features=2)
    assert set(c.vocabulary) == {"cat", "dog"}


def test_max_features_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        topica.Corpus.from_documents([["a"]], max_features=0)


def test_preprocessing_reports_max_features():
    c = topica.Corpus.from_documents([["a", "b"], ["b", "c"]], max_features=2)
    assert c.preprocessing["max_features"] == 2
    assert c.preprocessing["vocabulary"] is False


# --- fixed vocabulary -----------------------------------------------------


def test_fixed_vocabulary_preserves_order_and_full_width():
    docs = [["dog", "cat"], ["cat", "cat"]]
    V = ["dog", "cat", "zebra"]  # "zebra" never appears
    c = topica.Corpus.from_documents(docs, vocabulary=V)
    assert c.vocabulary == V  # order preserved, full width kept
    # word_counts parallel to vocabulary: dog=1, cat=3, zebra=0
    assert list(c.word_counts) == [1, 3, 0]
    assert c.preprocessing["vocabulary"] is True


def test_fixed_vocabulary_drops_oov_and_empty_docs_with_kept_indices():
    docs = [["dog", "xxx"], ["yyy", "zzz"], ["cat"]]
    c = topica.Corpus.from_documents(docs, vocabulary=["dog", "cat"])
    # doc 1 is all out-of-vocabulary and is dropped; alignment via kept_indices
    assert c.num_docs == 2
    assert c.kept_indices == [0, 2]


def test_fixed_vocabulary_all_oov_raises():
    with pytest.raises(ValueError, match="none of the tokens"):
        topica.Corpus.from_documents([["xxx"], ["yyy"]], vocabulary=["dog", "cat"])


def test_fixed_vocabulary_validation():
    with pytest.raises(ValueError, match="empty"):
        topica.Corpus.from_documents([["a"]], vocabulary=[])
    with pytest.raises(ValueError, match="duplicate"):
        topica.Corpus.from_documents([["a"]], vocabulary=["a", "a"])
    with pytest.raises(ValueError, match="empty-string"):
        topica.Corpus.from_documents([["a"]], vocabulary=["a", ""])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_doc_freq": 2},
        {"max_doc_fraction": 0.5},
        {"min_cf": 3},
        {"rm_top": 1},
        {"max_features": 5},
    ],
)
def test_fixed_vocabulary_mutually_exclusive_with_pruning(kwargs):
    with pytest.raises(ValueError, match="fixed vocabulary"):
        topica.Corpus.from_documents([["a", "b"]], vocabulary=["a", "b"], **kwargs)


@pytest.mark.parametrize("frac", [2.0, float("inf"), float("nan")])
def test_fixed_vocabulary_rejects_out_of_range_max_doc_fraction(frac):
    # A non-default max_doc_fraction (including nan/inf/>1) alongside vocabulary=
    # is still a conflict, not silently ignored. (Codex review, finding 2.)
    with pytest.raises(ValueError, match="fixed vocabulary"):
        topica.Corpus.from_documents([["a", "b"]], vocabulary=["a", "b"], max_doc_fraction=frac)


def test_empty_string_token_never_enters_vocabulary():
    # An empty-string "token" is not a word and must not become a vocab type, so a
    # later transform against that vocabulary does not spuriously fail.
    # (Codex review, finding 1.)
    c = topica.Corpus.from_documents([["", "a"], ["a", "b"]])
    assert "" not in c.vocabulary
    held = c.transform([["a"]])  # would raise if "" were a vocab term
    assert held.vocabulary == c.vocabulary


def test_fixed_vocabulary_allows_default_pruning_args():
    # Passing the *default* pruning values alongside vocabulary is a no-op, not an
    # error (only a meaningfully-active pruning argument conflicts).
    c = topica.Corpus.from_documents(
        [["a", "b"]], vocabulary=["a", "b"], min_doc_freq=1, max_doc_fraction=1.0
    )
    assert c.vocabulary == ["a", "b"]


# --- transform (held-out) -------------------------------------------------


def test_transform_shares_vocabulary_exactly():
    corpus = topica.Corpus.from_documents([["a", "b", "c"], ["b", "c", "d"]])
    held = corpus.transform([["a", "zzz"], ["d"]])
    assert held.vocabulary == corpus.vocabulary  # same terms, order, and width


def test_transform_drops_oov_tokens_and_empty_docs():
    corpus = topica.Corpus.from_documents([["dog", "cat"]])
    held = corpus.transform([["dog"], ["xxx", "yyy"], ["cat", "dog"]])
    assert held.num_docs == 2  # the all-OOV middle doc is dropped
    assert held.kept_indices == [0, 2]


def test_transform_all_oov_raises():
    corpus = topica.Corpus.from_documents([["dog", "cat"]])
    with pytest.raises(ValueError, match="none of the tokens"):
        corpus.transform([["xxx"], ["yyy"]])


def test_transform_keeps_topic_word_alignment():
    train = [["dog", "cat", "fish"] * 3, ["bird", "dog", "cat"] * 3, ["fish", "fish", "tree"] * 3]
    corpus = topica.Corpus.from_documents(train)
    model = topica.LDA(num_topics=2, seed=0).fit(corpus)
    held = corpus.transform([["dog", "cat"], ["fish", "tree"]])
    # topic_word columns align to the transformed corpus vocabulary...
    assert model.topic_word.shape[1] == held.num_words
    # ...so the fitted model can score the held-out documents.
    theta = model.transform(held)
    assert theta.shape == (held.num_docs, 2)
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_zero_frequency_vocab_column_keeps_coherence_finite():
    # A fixed vocabulary can carry a term that never appears (doc_freq 0). Fitting
    # and scoring must not divide by zero / take log(0). (Gemini review, open q b.)
    docs = [["dog", "cat", "fish"] * 3, ["cat", "fish", "bird"] * 3, ["dog", "dog", "tree"] * 3]
    c = topica.Corpus.from_documents(docs, vocabulary=["dog", "cat", "fish", "bird", "tree", "ghost"])
    assert c.word_counts[c.vocabulary.index("ghost")] == 0
    model = topica.LDA(num_topics=2, seed=0).fit(c)
    assert np.isfinite(model.coherence()).all()


# --- from_dataframe passthrough -------------------------------------------


def test_from_dataframe_max_features_and_vocabulary_passthrough():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "text": ["cat cat dog", "dog dog bird", "cat fish fish", "dog cat cat"],
            "year": [2000, 2001, 2002, 2003],
        }
    )
    c = topica.from_dataframe(df, text_col="text", max_features=2)
    assert c.num_words == 2

    cv = topica.from_dataframe(df, text_col="text", vocabulary=["cat", "dog"])
    assert cv.vocabulary == ["cat", "dog"]
    # metadata stays aligned to the surviving rows
    assert cv.metadata is not None
    assert len(cv.metadata) == cv.num_docs
