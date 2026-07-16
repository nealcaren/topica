"""Polars is accepted anywhere pandas is, in the frames and formula helpers.

topica's DataFrame surface (``from_dataframe``, ``align``, ``design_matrix``) is
duck-typed: a Polars frame goes in, a Polars frame comes back, and the result
matches what pandas would produce. Skips cleanly when Polars is not installed.
"""

import numpy as np
import pytest

import topica

pl = pytest.importorskip("polars")
pd = pytest.importorskip("pandas")

# Docs "zzz" and "qqq" empty out under min_doc_freq=2, so two rows are dropped.
ROWS = {
    "text": ["cat dog cat", "zzz", "cat dog fish", "qqq", "dog fish dog"],
    "party": ["D", "R", "D", "R", "D"],
    "year": [2000, 2001, 2002, 2003, 2004],
}


def test_from_dataframe_polars_in_polars_out():
    cp = topica.prep.from_dataframe(pl.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    assert isinstance(cp.metadata, pl.DataFrame)
    # Two docs dropped; metadata follows the surviving rows in order.
    assert cp.kept_indices == [0, 2, 4]
    assert list(cp.metadata["year"]) == [2000, 2002, 2004]
    assert set(cp.metadata.columns) == {"party", "year"}


def test_from_dataframe_matches_pandas():
    cp = topica.prep.from_dataframe(pl.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    cd = topica.prep.from_dataframe(pd.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    assert cp.kept_indices == cd.kept_indices
    assert list(cp.metadata["year"]) == list(cd.metadata["year"])
    assert list(cp.metadata["party"]) == list(cd.metadata["party"])


def test_align_polars_series_and_frame():
    c = topica.Corpus.from_documents(
        [["cat", "dog"], ["zzz"], ["cat", "dog", "fish"], ["qqq"], ["dog", "fish"]],
        min_doc_freq=2,
    )
    assert c.kept_indices == [0, 2, 4]
    s = pl.Series("party", ["D", "R", "D", "R", "D"])
    assert list(topica.prep.align(s, c)) == ["D", "D", "D"]
    df = pl.DataFrame({"a": [10, 11, 12, 13, 14]})
    aligned = topica.prep.align(df, c)
    assert isinstance(aligned, pl.DataFrame)
    assert list(aligned["a"]) == [10, 12, 14]


def test_design_matrix_from_polars_matches_pandas():
    cp = topica.prep.from_dataframe(pl.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    cd = topica.prep.from_dataframe(pd.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    pytest.importorskip("formulaic")
    Xp, np_names = topica.design.design_matrix("~ party", cp.metadata)
    Xd, pd_names = topica.design.design_matrix("~ party", cd.metadata)
    assert np_names == pd_names
    assert np.allclose(Xp, Xd)


def test_design_matrix_spline_from_polars():
    cp = topica.prep.from_dataframe(pl.DataFrame(ROWS), text_col="text", min_doc_freq=2)
    pytest.importorskip("formulaic")
    X, names = topica.design.design_matrix("~ spline(year, df=3)", cp.metadata)
    assert X.shape == (3, 3)
    assert len(names) == 3
