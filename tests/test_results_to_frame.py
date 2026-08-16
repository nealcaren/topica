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
    # Curated root (#757): reachable as topica.<Name> for isinstance checks (no
    # longer required in __all__, which now lists only the curated surface).
    for name in ("FrameDict", "QualityFrontier", "BootstrapStability",
                 "KeywordDiagnostics", "TimePrevalenceCI"):
        assert hasattr(topica, name), f"topica.{name} not exported"
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


# ---------------------------------------------------------------------------
# #758: sample-user API-consistency follow-ups
# ---------------------------------------------------------------------------


def _fit_lda(seed=13, k=2):
    corpus = topica.Corpus.from_documents(_corpus(seed=seed), min_doc_freq=1)
    return corpus, topica.LDA(num_topics=k, seed=seed).fit(corpus, iters=60)


def test_corpus_len_matches_num_docs():
    # #758: len(corpus) works and equals num_docs.
    corpus = topica.Corpus.from_documents(_corpus(n=37), min_doc_freq=1)
    assert len(corpus) == corpus.num_docs == 37


def test_corpus_from_dataframe_classmethod_alias():
    # #758: Corpus.from_dataframe(df, ...) is the pandas-native alias for the
    # module-level topica.from_dataframe and builds the same corpus.
    df = pd.DataFrame({"text": [" ".join(d) for d in _corpus(n=20)],
                       "party": (["D", "R"] * 10)})
    via_method = topica.Corpus.from_dataframe(df, text_col="text")
    via_func = topica.from_dataframe(df, text_col="text")
    assert isinstance(via_method, topica.Corpus)
    assert len(via_method) == len(via_func) == 20
    assert via_method.vocabulary == via_func.vocabulary


def test_model_topic_table_method_matches_function():
    # #758: m.topic_table() mirrors the top-level topica.topic_table(m).
    _, m = _fit_lda()
    from_method = m.topic_table()
    from_func = topica.topic_table(m)
    assert [r["topic"] for r in from_method] == [r["topic"] for r in from_func]
    assert [r["frex"] for r in from_method] == [r["frex"] for r in from_func]
    # and it carries the same .to_frame() the function's result does
    assert from_method.to_frame().equals(from_func.to_frame())


def test_topic_table_method_absent_on_scaling_models():
    # #758: models with no topic-word matrix (scaling / embedding) do not get it.
    assert not hasattr(topica.Wordfish, "topic_table")


def test_estimate_effect_returns_effectlist_with_to_frame():
    # #758: the estimate_effect container has .to_frame() like its siblings, and
    # is still a plain list.
    corpus, m = _fit_lda()
    X = np.array([[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(len(corpus))])
    eff = topica.estimate_effect(m, X=X, feature_names=["D", "R"], add_intercept=False)
    assert isinstance(eff, topica.EffectList)
    assert isinstance(eff, list)  # non-breaking: still indexes / iterates
    df = eff.to_frame()
    # one row per (topic, feature); matches the manual concat it replaces.
    manual = pd.concat([e.to_frame() for e in eff], ignore_index=True)
    assert df.equals(manual)
    assert set(["topic", "feature", "coef", "se"]).issubset(df.columns)


def test_effectlist_to_frame_empty():
    empty = topica.EffectList()
    assert empty.to_frame().empty


def test_heldout_corpus_attribute_gives_directive_hint():
    # #758: heldout.corpus (a common mis-reach) points at .documents / eval_heldout.
    corpus = topica.Corpus.from_documents(_corpus(n=20), min_doc_freq=1)
    ho = topica.make_heldout(corpus, seed=13)
    with pytest.raises(AttributeError, match=r"\.documents"):
        _ = ho.corpus


def test_converged_flags_point_at_stop_reason():
    # #758: a tab-completion user landing on the getter docstring finds the
    # stop_reason() pointer without hunting through conventions.md.
    assert "stop_reason" in (topica.LDA.converged.__doc__ or "")
    assert "stop_reason" in (topica.LDA.early_stopped.__doc__ or "")


def test_topic_table_not_bound_to_time_sliced_models():
    # #758 review: DTM's topic_word is time-sliced (a method, not a (K,V)
    # property), so a flat topic_table is ill-defined; the method is not bound
    # rather than bound-and-always-raising.
    assert not hasattr(topica.DTM, "topic_table")
    # a dynamic model that DOES expose a plain (K,V) topic_word still gets it
    assert hasattr(topica.TopicsOverTime, "topic_table")
