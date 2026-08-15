"""#742: the dict-returning diagnostics converge on the library's ``.to_frame()``
idiom without breaking the dict contract.

``quality_frontier``, ``bootstrap_stability``, ``visualize_keywords``, and
``time_prevalence_ci`` return ``FrameDict`` subclasses — still dicts (indexing,
``.get``, ``==`` with a plain dict, ``**`` unpacking all unchanged) that also
offer ``.to_frame()`` like ``SearchKResult`` and the effect/robustness results.
"""

import numpy as np
import pytest

import topica
from topica import keyatm
from topica._results import (
    BootstrapStability,
    KeywordDiagnostics,
    QualityFrontier,
    TimePrevalenceCI,
)

pd = pytest.importorskip("pandas")

A = ["tax", "market", "trade"]
B = ["war", "troop", "militari"]


def _corpus(seed=0, n=120):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        heavy, light = (A, B) if i % 2 else (B, A)
        docs.append(rng.choice(heavy, 6).tolist() + rng.choice(light, 2).tolist())
    return docs


def test_result_classes_exported_at_top_level():
    # #752: the FrameDict result types are reachable as topica.<Name> for
    # isinstance checks, not buried in topica._results.
    import topica
    for name in ("FrameDict", "QualityFrontier", "BootstrapStability",
                 "KeywordDiagnostics", "TimePrevalenceCI"):
        assert hasattr(topica, name), f"topica.{name} not exported"
        assert name in topica.__all__, f"{name} missing from __all__"
    assert topica.BootstrapStability is BootstrapStability


def test_framedict_is_still_a_dict():
    # Non-breaking: a FrameDict compares equal to the plain dict and unpacks.
    fd = BootstrapStability({"mean": 0.7, "topic": [0, 1], "stability": [0.6, 0.8]})
    assert fd == {"mean": 0.7, "topic": [0, 1], "stability": [0.6, 0.8]}
    assert isinstance(fd, dict)
    assert fd["mean"] == 0.7
    assert fd.get("missing") is None
    assert dict(**fd).keys() == fd.keys()


def test_quality_frontier_to_frame():
    docs = _corpus()
    model = topica.LDA(num_topics=3, seed=13).fit(docs)
    qf = topica.quality_frontier(model)
    assert isinstance(qf, QualityFrontier)
    # dict access unchanged
    assert set(qf) == {"topic", "coherence", "exclusivity", "prevalence"}
    df = qf.to_frame()
    assert list(df.columns) == ["topic", "coherence", "exclusivity", "prevalence"]
    assert len(df) == 3  # one row per topic


def test_bootstrap_stability_to_frame():
    docs = _corpus()
    bs = topica.bootstrap_stability(docs, k=3, n_boot=3, seed=13)
    assert isinstance(bs, BootstrapStability)
    assert "mean" in bs and "reference" in bs  # dict access unchanged
    df = bs.to_frame()
    # scalar `mean` and the `reference` model do not leak into the per-topic frame
    assert list(df.columns) == ["topic", "stability"]
    assert len(df) == 3


def test_visualize_keywords_to_frame_stacks_sets():
    docs = _corpus()
    kw = {"econ": ["tax", "market", "absent"], "war": ["war"]}
    vis = keyatm.visualize_keywords(docs, kw)
    assert isinstance(vis, KeywordDiagnostics)
    assert set(vis) == {"econ", "war"}  # dict access unchanged
    df = vis.to_frame()
    assert list(df.columns) == ["set", "keyword", "count", "proportion", "doc_freq", "in_vocab"]
    # one row per keyword across both sets, tagged by set name
    assert len(df) == 4
    assert set(df["set"]) == {"econ", "war"}
    assert df[df["keyword"] == "absent"]["count"].iloc[0] == 0


def test_time_prevalence_ci_to_frame_melts():
    # The wrapping is exercised by the dynamic-keyATM tests; here we check the
    # melt directly so the (period, topic) tidy shape is covered cheaply.
    T, K = 3, 2
    tpc = TimePrevalenceCI({
        "labels": ["2013", "2014", "2015"],
        "mean": np.arange(T * K, dtype=float).reshape(T, K),
        "ci_low": np.zeros((T, K)),
        "ci_high": np.ones((T, K)),
        "sd": np.full((T, K), 0.1),
    })
    assert isinstance(tpc, TimePrevalenceCI)
    assert list(tpc["labels"]) == ["2013", "2014", "2015"]  # dict access unchanged
    df = tpc.to_frame()
    assert list(df.columns) == ["period", "topic", "mean", "ci_low", "ci_high", "sd"]
    assert len(df) == T * K
    # period 2014 (index 1), topic 1 -> mean value 3.0
    row = df[(df["period"] == "2014") & (df["topic"] == 1)].iloc[0]
    assert row["mean"] == 3.0 and row["ci_high"] == 1.0
