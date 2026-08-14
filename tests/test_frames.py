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
    assert topica.align(X, c).tolist() == [10, 12, 14]


def test_align_list_and_dataframe():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert topica.align(list("abcde"), c) == ["a", "c", "e"]
    df = pd.DataFrame({"y": [0, 1, 2, 3, 4]})
    aligned = topica.align(df, c)
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
    c = topica.from_dataframe(df, text_col="text", min_doc_freq=2)
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
    c = topica.from_dataframe(df, text_col="speech", metadata_cols=["party"], min_doc_freq=2)
    assert list(c.metadata.columns) == ["party"]
    # The aligned metadata feeds an STM prevalence design with no manual hstack.
    X = c.metadata["party"].eq("D").astype(float).values.reshape(-1, 1)
    model = topica.STM(num_topics=2, seed=1)
    model.fit(c, X, prevalence_names=["is_D"], iters=10)
    assert model.doc_topic.shape == (3, 2)


def test_metadata_is_settable():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert c.metadata is None
    c.metadata = pd.DataFrame({"k": [1, 2, 3]})
    assert list(c.metadata["k"]) == [1, 2, 3]


# -- metadata survives save/load via a sidecar (issue #730) ----------------------


def test_metadata_round_trips_through_save_load(tmp_path):
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    c.metadata = pd.DataFrame({"year": [2000, 2002, 2004], "party": ["D", "R", "D"]})
    p = tmp_path / "corpus.bin"
    c.save(str(p))
    assert (tmp_path / "corpus.bin.meta").exists()  # sidecar written
    loaded = topica.Corpus.load(str(p))
    assert loaded.metadata is not None
    assert loaded.metadata.equals(c.metadata)


def test_save_without_metadata_writes_no_sidecar_and_clears_stale(tmp_path):
    p = tmp_path / "corpus.bin"
    # First save WITH metadata leaves a sidecar...
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    c.metadata = pd.DataFrame({"k": [1, 2, 3]})
    c.save(str(p))
    assert (tmp_path / "corpus.bin.meta").exists()
    # ...re-saving a corpus with no metadata must remove the stale sidecar, so a
    # later load doesn't reattach the wrong covariates.
    plain = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    plain.save(str(p))
    assert not (tmp_path / "corpus.bin.meta").exists()
    assert topica.Corpus.load(str(p)).metadata is None


# -- scikit-learn min_df / max_df aliases (issue #647 discoverability) -----------


def _alias_frame():
    # 8 docs; "common" in all 8, "rare" in 1, plus per-doc fillers.
    rows = [f"common tok{i} tok{(i + 1) % 8} extra{i}" for i in range(8)]
    rows[0] += " rare"
    return pd.DataFrame({"text": rows})


class TestDfAliases:
    def test_int_min_df_equals_min_doc_freq(self):
        df = _alias_frame()
        a = topica.from_dataframe(df, text_col="text", min_df=2)
        b = topica.from_dataframe(df, text_col="text", min_doc_freq=2)
        assert sorted(a.vocabulary) == sorted(b.vocabulary)
        assert "rare" not in a.vocabulary  # df 1 < 2 -> dropped

    def test_float_max_df_equals_max_doc_fraction(self):
        df = _alias_frame()
        with pytest.warns(UserWarning):  # "common" is in all 8 docs
            a = topica.from_dataframe(df, text_col="text", max_df=0.5)
        with pytest.warns(UserWarning):
            b = topica.from_dataframe(df, text_col="text", max_doc_fraction=0.5)
        assert sorted(a.vocabulary) == sorted(b.vocabulary)
        assert "common" not in a.vocabulary

    def test_float_min_df_is_a_proportion(self):
        df = _alias_frame()  # 8 docs; 0.25 -> ceil(2) docs
        a = topica.from_dataframe(df, text_col="text", min_df=0.25)
        b = topica.from_dataframe(df, text_col="text", min_doc_freq=2)
        assert sorted(a.vocabulary) == sorted(b.vocabulary)

    def test_int_max_df_is_an_absolute_count(self):
        df = _alias_frame()  # 8 docs; max_df=4 -> fraction 0.5
        with pytest.warns(UserWarning):
            a = topica.from_dataframe(df, text_col="text", max_df=4)
        with pytest.warns(UserWarning):
            b = topica.from_dataframe(df, text_col="text", max_doc_fraction=0.5)
        assert sorted(a.vocabulary) == sorted(b.vocabulary)

    def test_conflicting_min_aliases_raise(self):
        df = _alias_frame()
        with pytest.raises(ValueError, match="min_df or min_doc_freq"):
            topica.from_dataframe(df, text_col="text", min_df=2, min_doc_freq=2)

    def test_conflicting_max_aliases_raise(self):
        df = _alias_frame()
        with pytest.raises(ValueError, match="max_df or max_doc_fraction"):
            topica.from_dataframe(df, text_col="text", max_df=0.5, max_doc_fraction=0.5)

    @pytest.mark.parametrize("bad", [1.5, -0.1])
    def test_out_of_range_float_max_df_raises(self, bad):
        df = _alias_frame()
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            topica.from_dataframe(df, text_col="text", max_df=bad)

    def test_bool_min_df_is_rejected(self):
        # bool is an int subclass; True must not be read as min_df=1.
        df = _alias_frame()
        with pytest.raises(ValueError, match="min_df must be an int or a float"):
            topica.from_dataframe(df, text_col="text", min_df=True)
