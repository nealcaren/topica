"""The model-neutral fitted-model analysis surface (``topica.analysis``)."""

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# A real fitted LDA over a tiny two-theme corpus.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lda_corpus():
    pets = [["cat", "dog", "pet", "cat", "dog", "fur"]]
    sky = [["star", "moon", "sky", "star", "moon", "night"]]
    docs = pets * 20 + sky * 20
    texts = ["cat dog pet cat dog fur"] * 20 + ["star moon sky star moon night"] * 20
    timestamps = ([2019] * 10 + [2020] * 10) + ([2019] * 10 + [2020] * 10)
    groups = (["pets"] * 20) + (["sky"] * 20)
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    return m, docs, texts, timestamps, groups


# A duck-typed clustering model: just the public attributes the surface reads,
# including ``labels`` with a few -1 (outlier) documents.
class FakeClusterModel:
    def __init__(self):
        self.num_topics = 3
        self.vocabulary = ["alpha", "beta", "gamma", "delta", "eps"]
        self.topic_names = ["Topic A", "Topic B", "Topic C"]
        self.topic_word = np.array([
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
        ])
        # 6 documents: topics 0,0,1,2,2 plus two outliers (-1).
        self.doc_topic = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
            [0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
        ])
        self.labels = [0, 0, 1, 2, 2, -1, -1]


@pytest.fixture
def fake_cluster():
    return FakeClusterModel()


# ---------------------------------------------------------------------------
# topic_sizes
# ---------------------------------------------------------------------------

def test_topic_sizes_lda(lda_corpus):
    m, docs, *_ = lda_corpus
    sizes = topica.topic_sizes(m)
    assert sizes["size"].shape == (2,)
    assert sizes["mass"].shape == (2,)
    assert int(sizes["size"].sum()) == len(docs)
    assert sizes["outliers"] == 0
    np.testing.assert_allclose(sizes["mass"].sum(), len(docs), rtol=1e-6)


def test_topic_sizes_clustering_counts_labels_and_outliers(fake_cluster):
    sizes = topica.topic_sizes(fake_cluster)
    np.testing.assert_array_equal(sizes["size"], np.array([2, 1, 2]))
    assert sizes["outliers"] == 2


# ---------------------------------------------------------------------------
# topic_labels / set_topic_labels
# ---------------------------------------------------------------------------

def test_topic_labels_default_and_override(fake_cluster):
    assert topica.topic_labels(fake_cluster) == ["Topic A", "Topic B", "Topic C"]
    topica.set_topic_labels(fake_cluster, {1: "Renamed"})
    labels = topica.topic_labels(fake_cluster)
    assert labels[1] == "Renamed"
    assert labels[0] == "Topic A"  # untouched
    assert labels[2] == "Topic C"


def test_set_topic_labels_roundtrip_lda(lda_corpus):
    m, *_ = lda_corpus
    topica.set_topic_labels(m, {0: "Pets", 1: "Sky"})
    assert topica.topic_labels(m) == ["Pets", "Sky"]


# ---------------------------------------------------------------------------
# representative_docs
# ---------------------------------------------------------------------------

def test_representative_docs_one_topic(lda_corpus):
    m, docs, texts, *_ = lda_corpus
    reps = topica.representative_docs(m, texts, topic=0, n=3)
    assert isinstance(reps, list)
    assert len(reps) == 3
    assert all(isinstance(t, str) for t in reps)


def test_representative_docs_all_topics(lda_corpus):
    m, docs, texts, *_ = lda_corpus
    reps = topica.representative_docs(m, texts, n=4)
    assert set(reps.keys()) == {0, 1}
    assert all(len(v) == 4 for v in reps.values())


# ---------------------------------------------------------------------------
# topic_info
# ---------------------------------------------------------------------------

def test_topic_info_lda_one_row_per_topic(lda_corpus):
    m, docs, texts, *_ = lda_corpus
    rows = topica.topic_info(m, texts, n=5)
    assert len(rows) == 2
    assert [r["topic"] for r in rows] == [0, 1]
    for r in rows:
        assert set(r) >= {"topic", "label", "size", "prevalence", "top_words",
                          "representative_docs"}
        assert len(r["top_words"]) <= 5
        assert len(r["representative_docs"]) == 5


def test_topic_info_no_texts_omits_reps(lda_corpus):
    m, *_ = lda_corpus
    rows = topica.topic_info(m)
    assert all("representative_docs" not in r for r in rows)


def test_topic_info_labels_override(lda_corpus):
    m, docs, texts, *_ = lda_corpus
    rows = topica.topic_info(m, labels=["X", "Y"])
    assert [r["label"] for r in rows] == ["X", "Y"]


def test_topic_info_clustering_appends_outlier_row(fake_cluster):
    rows = topica.topic_info(fake_cluster, n=3)
    # 3 topics + one outlier row.
    assert len(rows) == 4
    assert [r["topic"] for r in rows] == [0, 1, 2, -1]
    outlier = rows[-1]
    assert outlier["topic"] == -1
    assert outlier["size"] == 2
    assert outlier["top_words"] == []
    # Real topics carry words pulled from the topic_word rows.
    assert rows[0]["top_words"][0] == "alpha"


# ---------------------------------------------------------------------------
# topics_over_time
# ---------------------------------------------------------------------------

def test_topics_over_time_rows_sum_to_one(lda_corpus):
    m, docs, texts, timestamps, _ = lda_corpus
    out = topica.topics_over_time(m, timestamps)
    assert out["labels"] == [2019, 2020]
    assert out["prevalence"].shape == (2, 2)
    np.testing.assert_allclose(out["prevalence"].sum(axis=1), 1.0, atol=1e-9)


def test_topics_over_time_unnormalized(lda_corpus):
    m, docs, texts, timestamps, _ = lda_corpus
    out = topica.topics_over_time(m, timestamps, normalize=False)
    # Each row is a mean of theta rows, so it already sums close to 1 but is
    # not forced to; the labels still align.
    assert out["prevalence"].shape == (2, 2)


def test_topics_over_time_length_mismatch(lda_corpus):
    m, *_ = lda_corpus
    with pytest.raises(ValueError):
        topica.topics_over_time(m, [2019, 2020])


# ---------------------------------------------------------------------------
# topics_per_class
# ---------------------------------------------------------------------------

def test_topics_per_class(lda_corpus):
    m, docs, texts, timestamps, groups = lda_corpus
    strata = topica.topics_per_class(m, groups)
    assert {s.stratum for s in strata} == {"pets", "sky"}
    for s in strata:
        assert s.mean.shape == (2,)
        assert s.n == 20


# ---------------------------------------------------------------------------
# plot_report (a composite matplotlib figure; skips if matplotlib is absent)
# ---------------------------------------------------------------------------

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def test_plot_report_full(lda_corpus):
    import matplotlib.pyplot as plt

    m, docs, texts, timestamps, groups = lda_corpus
    fig = topica.plot_report(m, texts=texts, timestamps=timestamps, groups=groups, n=5)
    assert isinstance(fig, plt.Figure)
    # All five panels apply here, so there is more than one drawn axis.
    titles = {ax.get_title() for ax in fig.get_axes()}
    assert "Topics by prevalence" in titles
    assert any("quality" in t for t in titles)
    assert any("time" in t.lower() for t in titles)
    assert any("class" in t.lower() for t in titles)
    plt.close(fig)


def test_plot_report_minimal(lda_corpus):
    import matplotlib.pyplot as plt

    m, *_ = lda_corpus
    # No texts/timestamps/groups: the prevalence panel is always present.
    fig = topica.plot_report(m)
    titles = {ax.get_title() for ax in fig.get_axes()}
    assert "Topics by prevalence" in titles
    # The time/class panels need their inputs, so they are absent here.
    assert not any("time" in t.lower() for t in titles)
    assert not any("class" in t.lower() for t in titles)
    plt.close(fig)


def test_plot_report_saves(tmp_path, lda_corpus):
    import matplotlib.pyplot as plt

    m, docs, texts, *_ = lda_corpus
    fig = topica.plot_report(m, texts=texts)
    out = tmp_path / "report.png"
    fig.savefig(out, dpi=60)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_plot_report_correlation_masks_diagonal(lda_corpus):
    """The correlation panel masks the always-1.0 self-correlation diagonal and
    scales its colors to the off-diagonal range, instead of drawing the raw
    matrix on a saturated +/-1 scale (regression for the plot_report sub-panel,
    which the original viz.TopicCorrelation fix had missed)."""
    import matplotlib.pyplot as plt

    m, docs, texts, *_ = lda_corpus
    fig = topica.plot_report(m, texts=texts)
    corr_axes = [ax for ax in fig.get_axes() if ax.get_title() == "Topic correlation"]
    if not corr_axes:
        plt.close(fig)
        pytest.skip("correlation panel not drawn for this model")
    im = corr_axes[0].images[0]
    data = im.get_array()
    # Diagonal is masked (drawn as neutral background), not a saturated 1.0.
    if np.ma.isMaskedArray(data):
        assert np.ma.getmaskarray(data).diagonal().all()
    else:
        assert np.all(np.isnan(np.asarray(data, dtype=float).diagonal()))
    # Color range comes from the off-diagonal magnitudes, not a hard +/-1.
    vmin, vmax = im.get_clim()
    assert vmin == pytest.approx(-vmax)
    plt.close(fig)
