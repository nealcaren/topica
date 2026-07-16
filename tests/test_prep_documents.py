"""Tests for prep_documents and plot_removed (issue #41).

prep_documents applies a document-frequency threshold AND keeps a metadata
frame row-aligned through document removal. The alignment invariant is the
key correctness property: len(meta) == len(corpus.doc_lengths) after the call,
and the surviving metadata rows correspond to the surviving documents.
"""

import pytest

import topica

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Corpus fixture
# ---------------------------------------------------------------------------
# Documents 1 ("zzz") and 3 ("qqq") appear only once, so lower_thresh=2 makes
# them rare. After the vocabulary is pruned, docs 1 and 3 become empty and are
# dropped.  The surviving documents are indices 0, 2, 4.
DOCS = [
    ["cat", "dog"],          # 0  — survives
    ["zzz"],                 # 1  — "zzz" drops under thresh 2 → doc becomes empty
    ["cat", "dog", "fish"],  # 2  — survives
    ["qqq"],                 # 3  — "qqq" drops under thresh 2 → doc becomes empty
    ["dog", "fish"],         # 4  — survives
]

META_DF = pd.DataFrame(
    {
        "year":  [2000, 2001, 2002, 2003, 2004],
        "party": ["D",  "R",  "D",  "R",  "D"],
    }
)


@pytest.fixture
def base_corpus():
    return topica.Corpus.from_documents(DOCS)


# ---------------------------------------------------------------------------
# prep_documents — core alignment guarantee
# ---------------------------------------------------------------------------

def test_prep_documents_drops_empty_docs(base_corpus):
    """After applying lower_thresh=2, only 3 of 5 documents survive."""
    fc, _ = topica.prep.prep_documents(base_corpus, lower_thresh=2)
    assert fc.num_docs == 3


def test_prep_documents_metadata_row_count_equals_kept_docs(base_corpus):
    """The alignment trap: meta rows must equal kept-doc count, not original count."""
    fc, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=2)
    assert len(fmeta) == fc.num_docs
    assert len(fmeta) == len(fc.doc_lengths)


def test_prep_documents_metadata_values_follow_correct_rows(base_corpus):
    """Surviving metadata rows correspond to the surviving documents (0, 2, 4)."""
    fc, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=2)
    assert list(fmeta["year"]) == [2000, 2002, 2004]
    assert list(fmeta["party"]) == ["D", "D", "D"]


def test_prep_documents_index_reset_after_drop(base_corpus):
    """Returned pandas metadata has a clean 0-based integer index."""
    _, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=2)
    assert list(fmeta.index) == [0, 1, 2]


def test_prep_documents_no_thresh_returns_all(base_corpus):
    """lower_thresh=1 (the default) keeps everything."""
    fc, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=1)
    assert fc.num_docs == 5
    assert len(fmeta) == 5


def test_prep_documents_returns_none_meta_when_no_meta(base_corpus):
    """When neither meta kwarg nor corpus.metadata is set, second return is None."""
    fc, fmeta = topica.prep.prep_documents(base_corpus, lower_thresh=2)
    assert fmeta is None
    assert fc.num_docs == 3


def test_prep_documents_uses_corpus_metadata_attribute(base_corpus):
    """corpus.metadata is used when meta kwarg is omitted."""
    base_corpus.metadata = META_DF
    fc, fmeta = topica.prep.prep_documents(base_corpus, lower_thresh=2)
    assert len(fmeta) == 3
    assert list(fmeta["year"]) == [2000, 2002, 2004]


def test_prep_documents_kwarg_meta_overrides_corpus_metadata(base_corpus):
    """Explicit meta= takes precedence over corpus.metadata."""
    # Set corpus.metadata to something different
    base_corpus.metadata = META_DF.assign(year=range(100, 105))
    fc, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=2)
    assert list(fmeta["year"]) == [2000, 2002, 2004]


def test_prep_documents_list_meta(base_corpus):
    """Plain list metadata is aligned correctly."""
    meta = ["a", "b", "c", "d", "e"]
    _, fmeta = topica.prep.prep_documents(base_corpus, meta=meta, lower_thresh=2)
    assert fmeta == ["a", "c", "e"]


# ---------------------------------------------------------------------------
# prep_documents — composes with from_dataframe / align
# ---------------------------------------------------------------------------

def test_prep_documents_composes_with_from_dataframe():
    """prep_documents works on a corpus built by from_dataframe."""
    df = pd.DataFrame(
        {
            "text": ["cat dog", "zzz", "cat dog fish", "qqq", "dog fish"],
            "year": [2000, 2001, 2002, 2003, 2004],
        }
    )
    c = topica.prep.from_dataframe(df, text_col="text")
    fc, fmeta = topica.prep.prep_documents(c, lower_thresh=2)
    assert fc.num_docs == len(fmeta)


def test_prep_documents_doc_lengths_aligned_with_meta(base_corpus):
    """doc_lengths length equals metadata length after filtering."""
    fc, fmeta = topica.prep.prep_documents(base_corpus, meta=META_DF, lower_thresh=2)
    assert len(fc.doc_lengths) == len(fmeta)


# ---------------------------------------------------------------------------
# prep_documents — upper_thresh
# ---------------------------------------------------------------------------

def test_prep_documents_upper_thresh_removes_common_terms(base_corpus):
    """upper_thresh drops terms above the count ceiling, reducing vocabulary."""
    fc_no_upper, _ = topica.prep.prep_documents(base_corpus, lower_thresh=1)
    # "dog" appears in docs 0, 2, 4 (3 out of 5). upper_thresh=2 removes it.
    fc_upper, _ = topica.prep.prep_documents(base_corpus, lower_thresh=1, upper_thresh=2)
    assert fc_upper.num_words < fc_no_upper.num_words


# ---------------------------------------------------------------------------
# plot_removed
# ---------------------------------------------------------------------------

def test_plot_removed_returns_axes(base_corpus):
    """plot_removed returns a matplotlib Axes."""
    mpl = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    ax = topica.viz.plot_removed(base_corpus, thresholds=range(1, 4))
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_plot_removed_accepts_ax_argument(base_corpus):
    """plot_removed draws into a provided Axes and returns it."""
    mpl = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    returned = topica.viz.plot_removed(base_corpus, thresholds=[1, 2, 3], ax=ax)
    assert returned is ax
    plt.close("all")
