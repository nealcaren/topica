"""Covariate-safe corpus construction: kept_indices, metadata, from_dataframe.

Topic-model pruning drops documents that empty out. These tests pin down that
the corpus reports which originals survived and keeps metadata aligned, so an
STM prevalence design can't silently misalign with the text.
"""

import numpy as np
import pytest

import topica

pd = pytest.importorskip("pandas")


# Docs 1 ("zzz") and 3 ("qqq") empty out under min_doc_freq=2.
DOCS = [["cat", "dog"], ["zzz"], ["cat", "dog", "fish"], ["qqq"], ["dog", "fish"]]


def test_kept_indices_identity_when_nothing_dropped():
    c = topica.Corpus.from_documents(DOCS)  # no pruning
    assert c.kept_indices == list(range(len(DOCS)))


def test_kept_indices_tracks_dropped_docs():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert c.num_docs == 3
    assert c.kept_indices == [0, 2, 4]


def test_align_numpy_array():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    X = np.arange(10, 15)
    assert topica.prep.align(X, c).tolist() == [10, 12, 14]


def test_align_list_and_dataframe():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert topica.prep.align(list("abcde"), c) == ["a", "c", "e"]
    df = pd.DataFrame({"y": [0, 1, 2, 3, 4]})
    aligned = topica.prep.align(df, c)
    assert list(aligned["y"]) == [0, 2, 4]
    assert list(aligned.index) == [0, 1, 2]  # reset


def test_from_dataframe_aligns_metadata():
    df = pd.DataFrame(
        {
            "text": ["cat dog", "zzz", "cat dog fish", "qqq", "dog fish"],
            "year": [2000, 2001, 2002, 2003, 2004],
            "party": ["D", "R", "D", "R", "D"],
        }
    )
    c = topica.prep.from_dataframe(df, text_col="text", min_doc_freq=2)
    assert c.num_docs == 3
    # text_col excluded by default; surviving rows only.
    assert list(c.metadata.columns) == ["year", "party"]
    assert list(c.metadata["year"]) == [2000, 2002, 2004]


def test_from_dataframe_explicit_columns_and_stm_payoff():
    df = pd.DataFrame(
        {
            "speech": ["cat dog", "zzz", "cat dog fish", "qqq", "dog fish"],
            "year": [2000, 2001, 2002, 2003, 2004],
            "party": ["D", "R", "D", "R", "D"],
        }
    )
    c = topica.prep.from_dataframe(df, text_col="speech", metadata_cols=["party"], min_doc_freq=2)
    assert list(c.metadata.columns) == ["party"]
    # The aligned metadata feeds an STM prevalence design with no manual hstack.
    X = c.metadata["party"].eq("D").astype(float).values.reshape(-1, 1)
    model = topica.models.STM(num_topics=2, seed=1)
    model.fit(c, X, prevalence_names=["is_D"], iters=10)
    assert model.doc_topic.shape == (3, 2)


def test_metadata_is_settable():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert c.metadata is None
    c.metadata = pd.DataFrame({"k": [1, 2, 3]})
    assert list(c.metadata["k"]) == [1, 2, 3]
