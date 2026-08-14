"""keyATM workflow helpers (topica.keyatm): top_topics, by_strata,
visualize_keywords, refine_keywords — the keyATM-flavored post-fit functions."""

import numpy as np
import pytest

import topica
from topica import keyatm

A = ["tax", "market", "trade"]
B = ["war", "troop", "militari"]


def _corpus(seed=0, n=160):
    rng = np.random.default_rng(seed)
    docs, strata = [], []
    for i in range(n):
        lab = i % 2
        heavy, light = (A, B) if lab else (B, A)
        docs.append(rng.choice(heavy, 6).tolist() + rng.choice(light, 2).tolist())
        strata.append("L" if lab else "C")
    return docs, strata


def test_visualize_keywords_counts_and_order():
    docs, _ = _corpus()
    kw = {"econ": ["tax", "market", "absent"], "war": ["war"]}
    vis = keyatm.visualize_keywords(docs, kw)
    assert set(vis) == {"econ", "war"}
    # Sorted by descending proportion; the absent keyword has count 0.
    props = [r["proportion"] for r in vis["econ"]]
    assert props == sorted(props, reverse=True)
    absent = [r for r in vis["econ"] if r["keyword"] == "absent"][0]
    assert absent["count"] == 0 and absent["doc_freq"] == 0


def test_visualize_keywords_corpus_aware_of_pruning():
    # #743: scoring against the built Corpus reflects vocabulary pruning. A seed
    # dropped by max_doc_fraction shows count 0 / in_vocab False and warns, while
    # scoring the raw docs reports the misleading pre-pruning count.
    docs = [["health", "care", "x"], ["health", "tax", "y"], ["health", "budget", "z"]] * 5
    seeds = {"Health": ["health", "care"], "Tax": ["tax"]}
    raw = keyatm.visualize_keywords(docs, seeds)
    raw_health = [r for r in raw["Health"] if r["keyword"] == "health"][0]
    assert raw_health["count"] > 0 and raw_health["in_vocab"]  # raw: looks fine

    c = topica.Corpus.from_documents(docs, max_doc_fraction=0.5)  # prunes "health"
    assert "health" not in c.vocabulary
    with pytest.warns(UserWarning, match="not in the corpus vocabulary"):
        vis = keyatm.visualize_keywords(c, seeds)
    corp_health = [r for r in vis["Health"] if r["keyword"] == "health"][0]
    assert corp_health["count"] == 0 and corp_health["in_vocab"] is False


def test_refine_keywords_corpus_drops_pruned_seed():
    # #743: refine_keywords on the built Corpus drops a pruned seed too.
    docs = [["health", "care", "x"], ["health", "tax", "y"]] * 6
    c = topica.Corpus.from_documents(docs, max_doc_fraction=0.5)  # prunes "health"
    refined, dropped = keyatm.refine_keywords(c, {"Health": ["health", "care"]})
    assert "health" in dropped["Health"] and refined["Health"] == ["care"]


def test_refine_keywords_drops_rare_and_empty_sets():
    docs, _ = _corpus()
    kw = {"econ": ["tax", "market", "absent"], "empty": ["nope1", "nope2"]}
    refined, dropped = keyatm.refine_keywords(docs, kw, min_count=2)
    assert refined == {"econ": ["tax", "market"]}
    assert dropped["econ"] == ["absent"]
    assert "empty" not in refined  # whole set removed (no surviving keyword)


def test_top_topics_shape_and_sorting():
    docs, _ = _corpus()
    m = topica.KeyATM({"econ": A[:2], "war": B[:2]}, num_topics=4, seed=1)
    m.fit(docs, iters=150)
    tt = keyatm.top_topics(m, n=2)
    assert len(tt) == len(docs)
    for row in tt:
        assert len(row) == 2
        assert row[0][1] >= row[1][1]  # sorted by proportion
        assert isinstance(row[0][0], str)  # topic names, not indices


def test_top_topics_from_raw_theta():
    theta = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    tt = keyatm.top_topics(theta, n=1, topic_names=["a", "b", "c"])
    assert tt[0][0] == ("b", pytest.approx(0.7))
    assert tt[1][0] == ("a", pytest.approx(0.6))


def test_by_strata_recovers_group_structure():
    docs, strata = _corpus()
    m = topica.KeyATM({"econ": A[:2], "war": B[:2]}, num_topics=4, seed=1)
    m.fit(docs, iters=200)
    res = keyatm.by_strata(m, strata)
    levels = {s.stratum: s for s in res}
    assert set(levels) == {"C", "L"}
    for s in res:
        assert s.n == 80
        assert s.mean.shape == (4,)
        assert np.all(s.ci_low <= s.mean + 1e-9)
        assert np.all(s.ci_high >= s.mean - 1e-9)
    # econ (topic 0) and war (topic 1) should separate the two strata.
    econ_gap = abs(levels["L"].mean[0] - levels["C"].mean[0])
    assert econ_gap > 0.1
    assert "econ" in res[0].as_dict()


def test_by_strata_validates_length():
    theta = np.full((5, 3), 1 / 3)
    with pytest.raises(ValueError):
        keyatm.by_strata(theta, ["a", "b"])


def test_all_out_of_vocab_seeded_topic_raises(recwarn):
    # A seeded topic whose keywords are all out-of-vocabulary would leave a hole
    # in the keyword-topic prefix; the fit must raise cleanly, not panic (#418).
    docs, _ = _corpus()
    m = topica.KeyATM({"econ": A[:2], "ghost": ["zzzznotaword"]}, num_topics=3, seed=1)
    with pytest.raises(ValueError, match="no keywords left in the vocabulary"):
        m.fit(docs, iters=3)


def test_duplicate_keyword_string_fits_without_panicking():
    # A repeated keyword string (or two surface forms that resolve to the same
    # vocabulary id) would reach the core's "keyword id listed more than once"
    # invariant and panic across the FFI boundary. topica dedupes the resolved ids
    # first, so a duplicate is harmless and the fit completes cleanly (#418).
    docs, _ = _corpus()
    m = topica.KeyATM({"econ": ["tax", "tax", "market"], "war": B}, num_topics=3, seed=1)
    m.fit(docs, iters=5)  # must not raise or panic
    assert np.isfinite(np.asarray(m.topic_word)).all()
