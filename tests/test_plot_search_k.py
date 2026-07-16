"""plot_search_k builds a coherence/exclusivity-vs-K figure from search_k rows."""

import pytest

import topica

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless


ROWS = [
    {"k": 10, "coherence": -51.1, "exclusivity": 0.579},
    {"k": 20, "coherence": -57.9, "exclusivity": 0.532},
    {"k": 30, "coherence": -59.8, "exclusivity": 0.556},
]


def test_returns_axes_with_two_metric_lines():
    ax = topica.viz.plot_search_k(ROWS)
    assert ax.get_xlabel() == "number of topics (K)"
    assert list(ax.get_xticks()) == [10, 20, 30]
    # primary axis carries the first metric (coherence)
    assert ax.lines and len(ax.lines[0].get_xdata()) == 3


def test_single_metric_and_unsorted_input():
    ax = topica.viz.plot_search_k(list(reversed(ROWS)), metrics=("coherence",))
    line = ax.lines[0]
    assert list(line.get_xdata()) == [10, 20, 30]  # sorted by K


def test_missing_metric_raises():
    with pytest.raises(ValueError):
        topica.viz.plot_search_k(ROWS, metrics=("perplexity",))


def test_real_search_k_output_plots():
    docs = [["cat", "dog", "fish"]] * 30 + [["sun", "moon", "star"]] * 30
    rows = topica.select.search_k(docs, ks=[2, 3], iters=80)
    ax = topica.viz.plot_search_k(rows)
    assert ax is not None
