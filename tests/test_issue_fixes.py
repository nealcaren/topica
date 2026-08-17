"""Regression tests for v0.7.0 user-testing fixes."""

import warnings

import numpy as np
import pytest

import topica


def test_english_stopwords_and_tokenize_accepts_iterable():
    # #3: a bundled stopword frozenset, and tokenize accepts any iterable.
    assert isinstance(topica.ENGLISH_STOPWORDS, frozenset)
    assert "the" in topica.ENGLISH_STOPWORDS and "cat" not in topica.ENGLISH_STOPWORDS
    toks = topica.tokenize("The cat and the dog ran", stopwords=topica.ENGLISH_STOPWORDS)
    assert toks == ["cat", "dog", "ran"]
    # a plain set and a list work too
    assert topica.tokenize("a big cat", stopwords={"a"}) == ["big", "cat"]
    assert topica.tokenize("a big cat", stopwords=["a", "big"]) == ["cat"]


def test_corpus_documents_round_trip():
    # #11: Corpus can recover its token lists.
    docs = [["cat", "dog"], ["star", "moon", "star"]]
    c = topica.Corpus.from_documents(docs)
    assert c.documents() == docs


def test_prepare_pyldavis_accepts_corpus():
    # #11: prepare_pyldavis takes a Corpus (no manual re-tokenizing).
    docs = [["cat", "dog", "pet"]] * 10 + [["star", "moon", "sky"]] * 10
    c = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(c, iters=100)
    out = topica.prepare_pyldavis(m, c)  # must not raise on a Corpus
    assert out is not None


def test_keyatm_warns_on_oov_keywords():
    # #9: out-of-vocabulary keywords warn instead of silently doing nothing.
    docs = [["health", "care", "doctor"]] * 10 + [["tax", "econ", "budget"]] * 10
    c = topica.Corpus.from_documents(docs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        topica.KeyATM({"Health": ["health", "zzznotaword"]}, num_topics=2, seed=1).fit(c, iters=20)
    msgs = " ".join(str(x.message) for x in w)
    assert "zzznotaword" in msgs and "vocabulary" in msgs


def test_empty_clustering_warns_and_diagnostics_guard():
    # #6: degenerate embeddings -> 0 clusters: a warning, and diagnostics raise a
    # clear error instead of leaking numpy's "Mean of empty slice".
    rng = np.random.default_rng(0)
    docs = [["a", "b", "c"]] * 30
    emb = rng.normal(0, 1e-3, (30, 5))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = topica.BERTopic(min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    if m.num_topics == 0:
        assert any("no clusters" in str(x.message) for x in w)
        with pytest.raises(ValueError, match="no topics"):
            topica.label_topics(m.topic_word, m.vocabulary)


def test_citation_handle():
    # #14: programmatic citation.
    assert "Caren" in topica.__citation__ and "topica" in topica.__citation__


@pytest.mark.parametrize(
    "make",
    [
        lambda: topica.LDA(-3),
        lambda: topica.DMR(-1),
        lambda: topica.CTM(-2),
        lambda: topica.STM(-2),
        lambda: topica.GSDMM(-1),
        lambda: topica.PA(-1, 3),
        lambda: topica.PA(3, -1),
        lambda: topica.PT(-2),
        lambda: topica.PT(2, num_pseudo=-1),
        lambda: topica.HLDA(depth=-1),
        lambda: topica.KeyATM({"a": ["x"]}, num_topics=-5),
    ],
)
def test_negative_count_is_value_error(make):
    # #13: a negative count raises a clean ValueError, not a raw OverflowError.
    with pytest.raises(ValueError):
        make()


def test_zero_num_topics_still_guarded():
    # #13: the existing zero guard keeps working.
    with pytest.raises(ValueError, match="num_topics must be >= 1"):
        topica.LDA(0)


def test_flexible_first_arg_accepts_model_or_matrix():
    # #10: frex/relevance/label_topics/topic_correlation/find_thoughts accept a
    # fitted model (the failing convention) as well as the raw matrix.
    docs = [["cat", "dog", "pet", "vet"]] * 12 + [["star", "moon", "sky", "sun"]] * 12
    m = topica.LDA(2, seed=1)
    m.fit(docs, iters=150)
    texts = [" ".join(d) for d in docs]

    # model-first (previously raised "float() argument ... not 'topica.LDA'")
    assert topica.topic_correlation(m).cor.shape == (2, 2)
    assert len(topica.find_thoughts(m, texts, topic=0)) == 3
    assert len(topica.frex(m)) == 2
    assert set(topica.label_topics(m)[0]) == {"prob", "frex", "lift", "score"}
    assert len(topica.relevance(m, topic=0)) == m.topic_word.shape[1]  # capped at vocab

    # matrix-first still works (backward compatible)
    assert topica.topic_correlation(m.doc_topic).cor.shape == (2, 2)
    assert len(topica.frex(m.topic_word, m.vocabulary)) == 2

    # a bare matrix with no vocabulary gives a clear message, not a cryptic one
    with pytest.raises(ValueError, match="vocabulary is required"):
        topica.frex(m.topic_word)


def test_search_k_labels_its_coherence_metric():
    # #14: search_k reports UMass; label it so its scale isn't confused with c_v.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    rows = topica.search_k(docs, [2, 3], iters=60, num_samples=1)
    assert all(r["coherence_metric"] == "u_mass" for r in rows)


def test_search_k_best_k_and_directions():
    # #153: best_k optimizes in the correct direction; coherence is negative, so
    # the maximum (least-negative) K is best, not the minimum.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    res = topica.search_k(docs, [2, 3], iters=60, num_samples=1)
    # still behaves as the list of rows it always was
    assert isinstance(res, list) and len(res) == 2
    assert res.directions["coherence"] == "maximize"
    assert res.directions["exclusivity"] == "maximize"
    # explicit coherence still maximizes (least-negative), but warns about
    # monotonicity (#167)
    expected = max(res, key=lambda r: r["coherence"])["k"]
    with pytest.warns(UserWarning, match="monotone"):
        assert res.best_k("coherence") == expected
    # asking for an absent held-out metric is a clear error, not a silent wrong pick
    with pytest.raises(ValueError):
        res.best_k("heldout_loglik")


def test_search_k_best_k_defaults_to_frontier():
    # #167: with no held-out set, best_k() picks the coherence/exclusivity
    # frontier (a knee), not bare coherence (which is monotone in K and would
    # return the grid floor). The frontier default must not emit the bare-
    # coherence monotonicity warning. (A #715 grid-boundary warning may still
    # fire when the knee lands on the smallest/largest scanned K — that is the
    # intended new signal to widen the grid, not the thing this test guards.)
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    res = topica.search_k(docs, [2, 3, 4], iters=60, num_samples=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = res.best_k()
    assert not any("monotone" in str(w.message).lower() for w in caught)
    assert chosen == res.best_k("frontier")
    assert chosen in {r["k"] for r in res}


def test_frontier_k_is_zscore_argmax():
    # #167: the frontier is argmax over K of z(coherence)+z(exclusivity), both
    # maximize. Verify against a hand-built result reproducing the issue's table.
    from topica.validation import SearchKResult
    rows = SearchKResult([
        {"k": 40, "coherence": -108.4, "exclusivity": 0.636},
        {"k": 60, "coherence": -114.4, "exclusivity": 0.652},
        {"k": 80, "coherence": -118.4, "exclusivity": 0.657},
        {"k": 100, "coherence": -125.2, "exclusivity": 0.660},
    ])
    # coherence-max would pick the grid floor (40); the frontier picks the knee.
    assert rows.best_k("frontier") == 60
    assert rows.best_k() == 60  # frontier is the no-held-out default


def test_frontier_needs_two_k():
    from topica.validation import SearchKResult
    one = SearchKResult([{"k": 5, "coherence": -10.0, "exclusivity": 0.5}])
    with pytest.raises(ValueError, match="at least two"):
        one.best_k("frontier")
    # a single-K grid falls back to coherence without the monotonicity warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert one.best_k() == 5


def test_best_k_is_nan_safe():
    # A degenerate fit can yield NaN coherence. Raw max()/z-score would let the
    # NaN row win (`x > nan` is always False); selection must ignore NaN rows.
    from topica.validation import SearchKResult
    rows = SearchKResult([
        {"k": 2, "coherence": float("nan"), "exclusivity": 0.1, "coherence_metric": "u_mass"},
        {"k": 5, "coherence": -10.0, "exclusivity": 0.8, "coherence_metric": "u_mass"},
        {"k": 10, "coherence": -5.0, "exclusivity": 0.9, "coherence_metric": "u_mass"},
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert rows.best_k("coherence") == 10   # not the NaN row (k=2)
    assert rows._frontier_k() == 10             # coherence not silently poisoned
    allnan = SearchKResult([{"k": 2, "coherence": float("nan"), "exclusivity": float("nan")}])
    with pytest.raises(ValueError, match="no finite value"):
        allnan.best_k("exclusivity")


def test_frontier_excludes_degenerate_nan_k():
    # A NaN in either frontier metric marks a degenerate fit. Even if that K has an
    # extreme value on the *other* metric, the frontier must not recommend it
    # (nan_to_num would otherwise give it a neutral z and let the extreme win).
    from topica.validation import SearchKResult
    rows = SearchKResult([
        {"k": 2, "coherence": float("nan"), "exclusivity": 1.0},   # degenerate
        {"k": 5, "coherence": -5.0, "exclusivity": 0.5},
        {"k": 10, "coherence": -6.0, "exclusivity": 0.5},
    ])
    assert rows._frontier_k() in {5, 10}   # never the NaN row
    assert rows._frontier_k() == 5         # the knee among the valid Ks


def test_dispersion_is_a_diagnostic_not_selectable():
    # dispersion is reported per K but is not a selection metric (it falls
    # monotonically with K), so best_k must reject it rather than pick a K by it.
    from topica.validation import SearchKResult
    rows = SearchKResult([
        {"k": 2, "coherence": -5.0, "exclusivity": 0.5, "dispersion": 3.0},
        {"k": 3, "coherence": -6.0, "exclusivity": 0.5, "dispersion": 1.2},
    ])
    with pytest.raises(ValueError, match="unknown metric"):
        rows.best_k("dispersion")


def test_frontier_tie_breaks_toward_smaller_k():
    # Equal scores must resolve to the smaller (simpler) K regardless of grid order.
    from topica.validation import SearchKResult
    for ks in ([50, 10], [10, 50]):
        rows = SearchKResult([{"k": k, "coherence": -5.0, "exclusivity": 0.5} for k in ks])
        assert rows._frontier_k() == 10


def test_best_k_coherence_warning_matches_coherence_type():
    # The monotonicity caveat is UMass-specific; for c_v the warning must not claim
    # UMass monotonicity (c_v is not monotone in K).
    from topica.validation import SearchKResult
    cv = SearchKResult([
        {"k": 2, "coherence": 0.4, "exclusivity": 0.5, "coherence_metric": "c_v"},
        {"k": 3, "coherence": 0.6, "exclusivity": 0.5, "coherence_metric": "c_v"},
    ])
    with pytest.warns(UserWarning, match="'c_v' coherence alone"):
        cv.best_k("coherence")
    umass = SearchKResult([
        {"k": 2, "coherence": -10.0, "exclusivity": 0.5, "coherence_metric": "u_mass"},
        {"k": 3, "coherence": -12.0, "exclusivity": 0.5, "coherence_metric": "u_mass"},
    ])
    with pytest.warns(UserWarning, match="monotone"):
        umass.best_k("coherence")


def test_search_k_n_jobs_matches_serial():
    # #631: parallel per-K fits (n_jobs>1) must give results identical to the
    # serial path -- each K keeps its own fixed seed -- and preserve row order.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12
    ks = [2, 3, 4]
    serial = topica.search_k(docs, ks, iters=60, num_samples=1, n_jobs=1)
    for n_jobs in (2, -1, None):
        par = topica.search_k(docs, ks, iters=60, num_samples=1, n_jobs=n_jobs)
        assert [r["k"] for r in par] == ks                      # order preserved
        assert list(par) == list(serial)                        # every key, bit-identical
        assert par.best_k() == serial.best_k()
    # A single-K grid must behave identically no matter what n_jobs asks for.
    one = topica.search_k(docs, [3], iters=60, num_samples=1, n_jobs=1)
    assert list(topica.search_k(docs, [3], iters=60, num_samples=1, n_jobs=8)) == list(one)


def test_search_k_reports_residual_dispersion_and_dedupes_ks():
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    with pytest.warns(UserWarning, match="duplicate"):
        rows = topica.search_k(docs, [2, 2, 3], iters=60, num_samples=1)
    assert [r["k"] for r in rows] == [2, 3]                # duplicate K dropped
    assert all("dispersion" in r and "dispersion_pvalue" in r for r in rows)
    assert all(np.isfinite(r["dispersion"]) for r in rows)


def test_extra_criteria_detect_redundant_topics():
    # #633: the discriminating behavior the criteria exist for. Four near-orthogonal
    # topics vs the same four plus a duplicate fifth: the redundant topic must raise
    # cao_juan (topics more similar) and lower deveaud (topics less distinct).
    from topica.validation import _cao_juan, _deveaud
    V = 8
    base = np.zeros((4, V))
    for i in range(4):
        base[i, 2 * i:2 * i + 2] = 0.5
    dup = np.vstack([base, base[0]])  # 5th topic duplicates the 1st
    assert _cao_juan(dup) > _cao_juan(base)
    assert _deveaud(dup) < _deveaud(base)
    # exact sentinels: identical topics -> cosine 1 / JSD 0; disjoint -> cosine 0
    same = np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    orth = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert _cao_juan(same) == pytest.approx(1.0)
    assert _cao_juan(orth) == pytest.approx(0.0)
    assert _deveaud(same) == pytest.approx(0.0)
    assert _deveaud(orth) == pytest.approx(np.log(2))
    # Three mutually-disjoint topics: all 3 pairs have JSD ln2, cosine 0. The
    # criterion is the *mean* over pairs (== ln2 / 0), not the sum (3*ln2 / 0) --
    # this pins the pair-count normalization for K>2.
    tri = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert _deveaud(tri) == pytest.approx(np.log(2))
    assert _cao_juan(tri) == pytest.approx(0.0)
    # single topic -> nan (no pairs)
    assert np.isnan(_cao_juan(np.array([[1.0, 0.0]])))
    assert np.isnan(_deveaud(np.array([[1.0, 0.0]])))


def test_search_k_criteria_are_opt_in_and_selectable():
    # #633: criteria default off (out of the frontier default) and are added only
    # when requested; then they are selectable and carry SEs under num_seeds>1.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12
    plain = topica.search_k(docs, [2, 3], iters=60, num_samples=1, num_seeds=1)
    assert "deveaud" not in plain[0] and "cao_juan" not in plain[0]

    res = topica.search_k(docs, [2, 3, 4], iters=60, num_samples=1, num_seeds=1,
                          criteria=["deveaud", "cao_juan"])
    assert all("deveaud" in r and "cao_juan" in r for r in res)
    assert res.directions["deveaud"] == "maximize"
    assert res.directions["cao_juan"] == "minimize"
    assert res.best_k("deveaud") in {r["k"] for r in res}
    assert res.best_k("cao_juan") in {r["k"] for r in res}
    # criteria stay out of the default selection (no held-out -> frontier)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert res.best_k() == res.best_k("frontier")

    with pytest.raises(ValueError, match="unknown criteria"):
        topica.search_k(docs, [2], criteria=["bogus"])

    multi = topica.search_k(docs, [2, 3], iters=60, num_samples=1, num_seeds=3,
                            criteria=["deveaud"])
    assert "deveaud_se" in multi[0]


def test_search_k_num_seeds_reports_se_and_is_backward_compatible():
    # #632: num_seeds=1 (default) reports no standard-error columns; num_seeds>1
    # adds <metric>_se for every varying metric and keeps <metric> as the mean.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12
    single = topica.search_k(docs, [2, 3], iters=60, num_samples=1)
    assert not any(k.endswith("_se") for k in single[0])

    multi = topica.search_k(docs, [2, 3], iters=60, num_samples=1, num_seeds=4)
    for base in ("coherence", "exclusivity", "dispersion"):
        assert base + "_se" in multi[0]
        assert all(r[base + "_se"] >= 0 for r in multi)
    # dispersion p-value gets no SE (an SE on a p-value is not meaningful)
    assert "dispersion_pvalue_se" not in multi[0]
    # means still drive the point-estimate selectors
    assert isinstance(multi.best_k(), int)


def test_search_k_num_seeds_se_math_is_correct():
    # #632: pin the actual SE formula (sample std, ddof=1, over sqrt(n)) and the
    # mean, by recomputing them from the same per-seed fits search_k runs.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12
    base, nseed = 7, 5
    res = topica.search_k(docs, [2, 3], seed=base, num_seeds=nseed,
                          iters=60, num_samples=1)
    row = next(r for r in res if r["k"] == 2)
    cohs = []
    for s in range(nseed):
        m = topica.LDA(2, seed=base + s)
        m.fit(docs, iters=60, num_samples=1)
        cohs.append(float(np.mean(m.coherence(10))))
    assert row["coherence"] == pytest.approx(float(np.mean(cohs)))
    assert row["coherence_se"] == pytest.approx(
        float(np.std(cohs, ddof=1) / np.sqrt(nseed)))
    # num_seeds must be >= 1
    with pytest.raises(ValueError, match="num_seeds must be"):
        topica.search_k(docs, [2], num_seeds=0)


def test_search_k_one_se_minimize_and_within_band():
    # 1se on a minimize metric (perplexity) and exact band membership.
    from topica.validation import SearchKResult
    # maximize: optimum k=4 (mean 10, se 1); k=2 within band (>=9), k=3 not (8.5)
    rows = SearchKResult([
        {"k": 2, "heldout_loglik": 9.2, "heldout_loglik_se": 0.5},
        {"k": 3, "heldout_loglik": 8.5, "heldout_loglik_se": 0.5},
        {"k": 4, "heldout_loglik": 10.0, "heldout_loglik_se": 1.0},
    ])
    assert rows.best_k("heldout_loglik") == 4              # plain optimum
    assert rows.best_k("heldout_loglik", rule="1se") == 2  # simplest within 10-1=9
    # minimize: optimum k=4 (mean 5, se 1); k=2 within band (<=6), so pick k=2
    perp = SearchKResult([
        {"k": 2, "perplexity": 5.8, "perplexity_se": 0.4},
        {"k": 3, "perplexity": 7.0, "perplexity_se": 0.4},
        {"k": 4, "perplexity": 5.0, "perplexity_se": 1.0},
    ])
    assert perp.best_k("perplexity") == 4
    assert perp.best_k("perplexity", rule="1se") == 2      # within 5+1=6


def test_best_k_elbow_rule():
    # #637: the elbow of a diminishing-returns curve, where the plain optimum is
    # the grid edge. The poliblog held-out curve rises fast then flattens; the
    # elbow sits well inside the grid, not at the argmax.
    from topica.validation import SearchKResult
    ks = [10, 20, 30, 40, 60, 80, 100, 120]
    ll = [-712.5, -708.2, -705.6, -704.4, -702.9, -701.9, -701.4, -701.1]
    res = SearchKResult([{"k": k, "heldout_loglik": v} for k, v in zip(ks, ll)])
    assert res.best_k("heldout_loglik") == 120                 # plain optimum = edge
    assert res.best_k("heldout_loglik", rule="elbow") == 40    # the knee, interior
    # a minimize metric (mirror curve) gives the same interior knee
    perp = SearchKResult([{"k": k, "perplexity": -v} for k, v in zip(ks, ll)])
    assert perp.best_k("perplexity", rule="elbow") == 40
    # guards: elbow needs >=3 K, and is undefined for the frontier
    two = SearchKResult([{"k": 2, "heldout_loglik": 1.0}, {"k": 3, "heldout_loglik": 2.0}])
    with pytest.raises(ValueError, match="at least three"):
        two.best_k("heldout_loglik", rule="elbow")
    with pytest.raises(ValueError, match="not defined for the frontier"):
        res_ce = SearchKResult([{"k": 2, "coherence": -5.0, "exclusivity": 0.5},
                                {"k": 3, "coherence": -6.0, "exclusivity": 0.6}])
        res_ce.best_k("frontier", rule="elbow")


def test_best_k_reconstruction_error_selectable(recwarn):
    # #730: reconstruction_error (an NMF/LSA scree column) used to raise
    # "unknown metric"; now it is selectable. It is monotone-decreasing in K, so
    # rule='best' hits the largest K (grid edge) and warns, while rule='elbow'
    # finds the interior knee.
    from topica.validation import SearchKResult
    ks = [2, 4, 6, 8, 10, 12]
    err = [3.0, 2.0, 1.6, 1.4, 1.3, 1.25]  # falls fast then flattens (a scree)
    res = SearchKResult([{"k": k, "reconstruction_error": e} for k, e in zip(ks, err)])
    assert res.best_k("reconstruction_error") == 12               # smallest error = edge
    assert any("grid boundary" in str(w.message) for w in recwarn)
    assert res.best_k("reconstruction_error", rule="elbow") == 6  # the knee, interior


def test_best_k_unknown_metric_lists_present_columns():
    # #730: the "unknown metric" error must enumerate the columns actually in
    # these results, not just every recognized metric name.
    from topica.validation import SearchKResult
    res = SearchKResult([{"k": 2, "coherence": -5.0, "exclusivity": 0.5},
                         {"k": 3, "coherence": -6.0, "exclusivity": 0.6}])
    with pytest.raises(ValueError, match=r"selectable metrics in these results.*coherence"):
        res.best_k("recon_err")  # unrecognized name -> lists present columns


def test_best_k_configurable_frontier():
    # #637: frontier weights/metrics reshape the knee; default is unchanged.
    from topica.validation import SearchKResult
    ks = [10, 20, 30, 40, 50, 60]
    coh = [-8, -5, -6, -7, -9, -11]        # coherence peaks at K=20
    exc = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]   # exclusivity peaks at K=60
    res = SearchKResult([{"k": k, "coherence": c, "exclusivity": e}
                         for k, c, e in zip(ks, coh, exc)])
    assert res.best_k("frontier") == 20                          # equal weight
    assert res.best_k("frontier", weights=(1, 5)) == 60          # exclusivity-heavy
    assert res.best_k("frontier", weights=(5, 1)) == 20          # coherence-heavy
    # a different metric set is honored (and directions applied)
    res2 = SearchKResult([{"k": k, "coherence": c, "exclusivity": e,
                           "cao_juan": 0.5 - 0.001 * k}
                          for k, c, e in zip(ks, coh, exc)])
    assert res2.best_k("frontier", frontier_metrics=["coherence", "cao_juan"]) in set(ks)
    # a custom frontier is best-only; validation errors are clear
    with pytest.raises(ValueError, match="rule='best' only"):
        res.best_k("frontier", weights=(1, 1), rule="1se")
    with pytest.raises(ValueError, match="length"):
        res.best_k("frontier", weights=(1, 1, 1))
    with pytest.raises(ValueError, match="unknown frontier metric"):
        res.best_k("frontier", frontier_metrics=["coherence", "bogus"])
    with pytest.raises(ValueError, match="finite and non-negative"):
        res.best_k("frontier", weights=(-1, 1))
    # footgun guard: frontier kwargs with a non-frontier metric must raise, not
    # be silently ignored (including the default metric resolving to a scalar).
    with pytest.raises(ValueError, match="only apply to metric='frontier'"):
        res.best_k("coherence", weights=(1, 1))


def test_best_k_elbow_warns_on_convex_curve():
    # An accelerating (convex) curve has no diminishing-returns knee; the elbow
    # rule warns and falls back to the smallest K rather than inventing one.
    from topica.validation import SearchKResult
    convex = SearchKResult([{"k": k, "heldout_loglik": float(k * k)}
                            for k in (10, 20, 30, 40)])
    with pytest.warns(UserWarning, match="not well defined"):
        assert convex.best_k("heldout_loglik", rule="elbow") == 10
    # inf in a metric is treated as non-finite and dropped, not fed to the elbow
    withinf = SearchKResult([{"k": 10, "heldout_loglik": float("inf")},
                             {"k": 20, "heldout_loglik": -5.0},
                             {"k": 30, "heldout_loglik": -4.0},
                             {"k": 40, "heldout_loglik": -3.9}])
    assert withinf.best_k("heldout_loglik") == 40  # inf row ignored, not the max


def test_frontier_best_and_1se_share_one_curve():
    # best_k('frontier') and best_k(rule='1se') must band around the SAME frontier
    # curve when seeds are present, so 1se is never stricter than best.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12 + [["a", "b", "c"]] * 12
    res = topica.search_k(docs, [2, 3, 4, 5], iters=60, num_samples=1, num_seeds=4)
    assert res.best_k(rule="1se") <= res.best_k()   # guaranteed, same curve
    # an all-degenerate frontier falls back to the smallest K, not a crash
    from topica.validation import SearchKResult
    deg = SearchKResult([{"k": 2}, {"k": 3}])
    deg._frontier_mean = np.array([-np.inf, -np.inf])
    deg._frontier_se = np.array([np.nan, np.nan])
    assert deg._frontier_k_1se() == 2


def test_search_k_best_k_one_se_rule():
    # #632: the 1-SE rule needs num_seeds>1 and picks the simplest K within one SE
    # of the optimum; it must reject a single-seed result.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12 + \
           [["red", "blue", "green"]] * 12
    single = topica.search_k(docs, [2, 3, 4], iters=60, num_samples=1)
    with pytest.raises(ValueError, match="num_seeds"):
        single.best_k(rule="1se")

    multi = topica.search_k(docs, [2, 3, 4], iters=60, num_samples=1, num_seeds=4)
    ks = {r["k"] for r in multi}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bare coherence warns by design
        for m in ("frontier", "coherence", "exclusivity"):
            k = multi.best_k(m, rule="1se") if m != "frontier" else multi.best_k(rule="1se")
            assert k in ks
    # 1-SE is never stricter (larger K) than the plain optimum for the frontier:
    # it can only move toward a simpler (<=) K within the error band.
    assert multi.best_k(rule="1se") <= multi.best_k()

    with pytest.raises(ValueError, match="rule must be"):
        multi.best_k(rule="bogus")


def test_search_k_best_k_defaults_to_heldout_when_present():
    # #153: with a held-out set, best_k defaults to the held-out log-likelihood
    # (maximize) rather than coherence.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    held = [["cat", "dog"], ["star", "moon"]]
    res = topica.search_k(docs, [2, 3], iters=60, num_samples=1, held_out=held)
    assert res.directions["perplexity"] == "minimize"
    assert res.best_k() == min(res, key=lambda r: r["perplexity"])["k"]


def _planted_embeddings(seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    word_emb = np.vstack([rng.normal([3, 0], 0.2, (8, 2)),
                          rng.normal([-3, 0], 0.2, (8, 2))])
    idx = {w: i for i, w in enumerate(vocab)}
    docs = [[f"a{i}" for i in rng.integers(0, 8, 6)] for _ in range(30)] + \
           [[f"b{i}" for i in rng.integers(0, 8, 6)] for _ in range(30)]
    doc_emb = np.array([word_emb[[idx[w] for w in d]].mean(0) for d in docs])
    doc_emb = doc_emb + rng.normal(0, 0.05, doc_emb.shape)
    return docs, vocab, word_emb, doc_emb


def test_top2vec_centroid_default_and_kwarg():
    # #8: topic_neighbors(0, n=8) must not raise (topic is the first positional);
    # and top_words defaults to the centroid view when word_embeddings are present,
    # giving Top2Vec a headline distinct from BERTopic's c-TF-IDF.
    docs, vocab, word_emb, doc_emb = _planted_embeddings()
    tv = topica.Top2Vec(min_cluster_size=8, seed=1)
    tv.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)

    neigh = [w for w, _ in tv.topic_neighbors(0, n=4)]  # previously raised
    assert len(neigh) == 4

    centroid = [w for w, _ in tv.top_words(4, topic=0, weights=True)]              # default
    assert centroid == neigh                                        # centroid view
    ctfidf = [w for w, _ in tv.top_words(4, topic=0, representation="c-tf-idf", weights=True)]
    assert isinstance(ctfidf, list)
    assert tv.topic_word.shape[0] == tv.num_topics                  # matrix stays c-TF-IDF


def test_top2vec_centroid_requires_word_vectors():
    # #8: without word_embeddings, top_words falls back to c-TF-IDF and an explicit
    # centroid request gives a clear error.
    docs, _, _, doc_emb = _planted_embeddings()
    tv = topica.Top2Vec(min_cluster_size=8, seed=1)
    tv.fit(docs, doc_emb)
    assert isinstance(tv.top_words(3, topic=0), list)  # c-TF-IDF default, no raise
    with pytest.raises(ValueError, match="word_embeddings"):
        tv.top_words(3, topic=0, representation="centroid")


def _three_blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[4, 0], [-4, 0], [0, 4]], float)
    doc_emb, docs = [], []
    for c in range(3):
        for _ in range(25):
            doc_emb.append(centers[c] + rng.normal(0, 0.3, 2))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 5, 6)])
    return docs, np.array(doc_emb)


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
@pytest.mark.parametrize("clusterer", ["kmeans", "gmm", "agglomerative"])
def test_swappable_clusterer_assigns_every_doc(model_cls, clusterer):
    # #7/#352: kmeans / gmm / agglomerative assign every document (no -1 noise
    # bucket) to a fixed number of clusters, unlike HDBSCAN.
    docs, doc_emb = _three_blobs()
    cls = getattr(topica, model_cls)
    m = cls(min_cluster_size=8, clusterer=clusterer, num_clusters=3, seed=1)
    m.fit(docs, doc_emb)
    assert m.num_topics == 3
    assert -1 not in set(m.labels)  # no noise bucket


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
@pytest.mark.parametrize("clusterer", ["louvain", "leiden"])
def test_graph_clusterer_auto_k_assigns_every_doc(model_cls, clusterer):
    # #352: louvain / leiden are auto-K (like hdbscan, no num_clusters) but assign
    # every document (no -1 noise bucket). On three separated blobs they recover
    # the three topics without being told the count.
    docs, doc_emb = _three_blobs()
    cls = getattr(topica, model_cls)
    m = cls(min_cluster_size=8, clusterer=clusterer, seed=1)  # no num_clusters
    m.fit(docs, doc_emb)
    assert m.num_topics == 3
    assert -1 not in set(m.labels)  # graph clusterers assign everything
    # deterministic for a fixed seed
    m2 = cls(min_cluster_size=8, clusterer=clusterer, seed=1)
    m2.fit(docs, doc_emb)
    assert list(m.labels) == list(m2.labels)


def test_clusterer_validation():
    # #7/#352: clear errors for the new knobs.
    with pytest.raises(ValueError, match="needs num_clusters"):
        topica.BERTopic(clusterer="kmeans")
    with pytest.raises(ValueError, match="unknown clusterer"):
        topica.Top2Vec(clusterer="dbscan")
    with pytest.raises(ValueError, match="num_clusters must be >= 1"):
        topica.Top2Vec(clusterer="kmeans", num_clusters=-2)
    # louvain / leiden are auto-K: they must NOT require num_clusters.
    topica.BERTopic(clusterer="louvain")
    topica.Top2Vec(clusterer="leiden")


def _overlapping_blobs(k=8, per=40, dim=15, spread=1.6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (k, dim))
    emb, docs = [], []
    for c in range(k):
        for _ in range(per):
            emb.append(centers[c] + rng.normal(0, spread, dim))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 6, 6)])
    return docs, np.array(emb)


@pytest.mark.parametrize("clusterer", ["louvain", "leiden"])
def test_resolution_steers_topic_count(clusterer):
    # #358: higher resolution -> more, smaller topics for the graph clusterers.
    docs, emb = _overlapping_blobs()
    # Pin the linear reducer: this checks the resolution knob's effect on the graph
    # clusterer, which is independent of the reducer (BERTopic now defaults to umap).
    lo = topica.BERTopic(clusterer=clusterer, resolution=0.5, min_cluster_size=5, reducer="pca", seed=1)
    lo.fit(docs, emb)
    hi = topica.BERTopic(clusterer=clusterer, resolution=3.0, min_cluster_size=5, reducer="pca", seed=1)
    hi.fit(docs, emb)
    assert hi.num_topics > lo.num_topics


def test_knn_neighbors_steers_topic_count():
    # #358: smaller knn_neighbors -> more, tighter communities. The effect is
    # weaker than resolution, so use a fine-grained corpus where it bites.
    docs, emb = _overlapping_blobs(k=20, per=40, dim=30, spread=1.4)
    # Pin the linear reducer: this checks the knn_neighbors knob on the graph
    # clusterer, independent of the reducer (Top2Vec now defaults to umap).
    small = topica.Top2Vec(clusterer="leiden", knn_neighbors=5, min_cluster_size=5, reducer="pca", seed=1)
    small.fit(docs, emb)
    large = topica.Top2Vec(clusterer="leiden", knn_neighbors=30, min_cluster_size=5, reducer="pca", seed=1)
    large.fit(docs, emb)
    assert small.num_topics > large.num_topics


def test_resolution_ignored_by_nongraph_and_validated():
    # #358: resolution/knn_neighbors are accepted but ignored for non-graph
    # clusterers, and invalid values error at construction.
    docs, emb = _overlapping_blobs()
    m = topica.BERTopic(
        clusterer="kmeans", num_clusters=4, resolution=3.0, knn_neighbors=8, seed=1
    )
    m.fit(docs, emb)
    assert m.num_topics == 4
    for bad in (dict(resolution=0.0), dict(resolution=-1.0), dict(knn_neighbors=0)):
        with pytest.raises(ValueError):
            topica.BERTopic(clusterer="leiden", **bad)


def test_report_is_callable():
    # #12: report(model) works as a one-call overview (alias for summary).
    assert callable(topica.report)
    docs = [["cat", "dog"], ["star", "moon"]] * 8
    m = topica.LDA(2, seed=1)
    m.fit(docs, iters=50)
    assert topica.report(m) == topica.summary(m)
    assert "num_topics" in topica.report(m)


def test_search_k_result_to_frame():
    # #733 Tier 3: SearchKResult gained to_frame(), like the effect/robustness
    # results (previously only effects_across_k had it).
    from topica.validation import SearchKResult
    rows = SearchKResult([{"k": 2, "coherence": -5.0, "exclusivity": 0.5},
                          {"k": 3, "coherence": -6.0, "exclusivity": 0.6}])
    df = rows.to_frame()
    assert list(df["k"]) == [2, 3]
    assert {"coherence", "exclusivity"}.issubset(df.columns)


def test_best_k_accepts_coherence_metric_label_alias():
    # #733 Tier 3: the result advertises coherence_metric ("semcoh"/"u_mass");
    # best_k now accepts that label as an alias for "coherence" instead of raising.
    from topica.validation import SearchKResult
    for label in ("semcoh", "u_mass"):
        rows = SearchKResult([{"k": 2, "coherence": -5.0, "exclusivity": 0.5,
                               "coherence_metric": label},
                              {"k": 3, "coherence": -6.0, "exclusivity": 0.6,
                               "coherence_metric": label}])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # bare-coherence monotone warning
            assert rows.best_k(label) == rows.best_k("coherence")
    # a label that is NOT this result's coherence_metric still errors
    rows = SearchKResult([{"k": 2, "coherence": -5.0, "exclusivity": 0.5,
                           "coherence_metric": "u_mass"}])
    with pytest.raises(ValueError, match="unknown metric"):
        rows.best_k("semcoh")


def test_topic_labels_shape_and_guard():
    # #733 Tier 3: label_topics rows are TopicLabels dicts; an integer index (the
    # common "iterate for words" mistake) raises a directive error, dict access works.
    from topica.validation import TopicLabels
    tl = TopicLabels({"prob": [("a", 0.3)], "frex": [("b", 0.9)],
                      "lift": [("c", 1.1)], "score": [("d", 0.2)]})
    assert tl["frex"] == [("b", 0.9)]                  # dict access intact
    assert [w for w, _ in tl["frex"]] == ["b"]
    assert set(tl.keys()) == {"prob", "frex", "lift", "score"}
    with pytest.raises(TypeError, match="dict keyed by"):
        tl[0]                                          # int index -> directive error
    assert "TopicLabels" in repr(tl)


def test_predicted_prevalence_scalar_helpers():
    # #733 Tier 3: a contrast is a single value but estimate is a 1-element array
    # (f-string on it crashes). .value / .ci give floats; multi-point raises.
    import numpy as np
    from topica.stm import PredictedPrevalence
    pp = PredictedPrevalence(topic=0, topic_name="t", mode="contrast", grid=["a", "b"],
                             estimate=np.array([-0.096]), ci_low=np.array([-0.14]),
                             ci_high=np.array([-0.05]))
    assert f"{pp.value:+.3f}" == "-0.096"
    assert pp.ci == pytest.approx((-0.14, -0.05))
    multi = PredictedPrevalence(topic=0, topic_name="t", mode="continuous",
                                grid=[{}, {}], estimate=np.array([0.1, 0.2]),
                                ci_low=np.array([0.0, 0.1]), ci_high=np.array([0.2, 0.3]))
    with pytest.raises(ValueError, match="single grid point"):
        multi.value


def test_lda_rejects_num_topics_over_vocab():
    # #741: K larger than the vocabulary produces only degenerate topics, so LDA
    # now guards it the way NMF/CTM do instead of fitting silently. K == vocab is
    # still allowed.
    tiny = [["a", "b"], ["b", "c"], ["a", "c"]]  # 3 distinct word types
    with pytest.raises(ValueError, match="exceeds the vocabulary size"):
        topica.LDA(num_topics=100, seed=13).fit(tiny, iters=10)
    m = topica.LDA(num_topics=3, seed=13).fit(tiny, iters=10)  # K == vocab: fine
    assert np.asarray(m.topic_word).shape == (3, 3)


def _two_topic_corpus():
    a = [["cat", "dog", "pet", "vet", "paw"]] * 30
    b = [["star", "moon", "sky", "sun", "orbit"]] * 30
    return topica.Corpus.from_documents(a + b)


def test_bootstrap_stability_fit_kwargs_dict_and_inline_merge():
    # #740: fit_kwargs= (a dict, like cross_validate) is documented but used to be
    # **kwargs, so passing it crashed. Both the dict form and inline keywords now
    # work and merge.
    c = _two_topic_corpus()
    r_dict = topica.bootstrap_stability(c, k=2, n_boot=2, fit_kwargs={"iters": 40})
    r_inline = topica.bootstrap_stability(c, k=2, n_boot=2, iters=40)
    assert 0.0 <= r_dict["mean"] <= 1.0
    assert r_dict["mean"] == r_inline["mean"]  # same effective fit args -> same result


def test_bootstrap_stability_factory_only_without_k():
    # #742/#745: a supplied model_factory owns the topic count, so factory-only
    # usage (no k=, no reference=) must work rather than raising "pass k ...".
    c = _two_topic_corpus()
    r = topica.bootstrap_stability(
        c, n_boot=1, model_factory=lambda seed: topica.LDA(2, seed=seed), iters=30)
    assert r["stability"].shape == (2,)  # K read back off the fitted reference
    # still errors when there is genuinely no way to know K
    with pytest.raises(ValueError, match="pass k"):
        topica.bootstrap_stability(c, n_boot=1, iters=30)


def test_bootstrap_stability_warns_on_k_with_model_factory():
    # #740: k= is silently ignored when model_factory= is given (the factory owns
    # the topic count and its argument is the seed). Warn instead of building the
    # wrong K.
    c = _two_topic_corpus()
    with pytest.warns(UserWarning, match="k= is ignored when model_factory"):
        topica.bootstrap_stability(
            c, k=8, n_boot=1,
            model_factory=lambda seed: topica.LDA(2, seed=seed), iters=30)


def test_list_models_is_sortable():
    # #742: ModelInfo had no __lt__, so sorted(list_models()) raised TypeError.
    models = topica.list_models()
    ordered = sorted(models)
    assert [m.name for m in ordered] == sorted(m.name for m in models)


def test_stopwords_frozenset_composes_with_corpus_builder():
    # #742: topica.stopwords()/ENGLISH_STOPWORDS is a frozenset, but the corpus
    # builders demanded a Sequence, so the library's own accessor did not compose
    # with its own builder. Any iterable (set/frozenset/tuple/list) is now accepted.
    sw = topica.stopwords("english")
    assert isinstance(sw, frozenset)
    c = topica.Corpus.from_documents([["the", "cat", "runs"], ["a", "dog", "barks"]],
                                     stopwords=sw)
    assert "the" not in c.vocabulary and "cat" in c.vocabulary
    # a bare set and a tuple work too
    topica.Corpus.from_documents([["x", "y"]], stopwords={"x"})
    topica.Corpus.from_documents([["x", "y"]], stopwords=("x",))


def test_predicted_prevalence_contrast_bad_shape_is_actionable():
    # #742: a 3-tuple contrast raised a terse "must be a 2-item dict or 2-element
    # sequence"; the message now names both accepted forms.
    docs = [["cat", "dog", "pet", "vet", "paw"]] * 20 + \
           [["star", "moon", "sky", "sun", "orbit"]] * 20
    import pandas as pd
    df = pd.DataFrame({"text": [" ".join(d) for d in docs],
                       "party": (["D"] * 20) + (["R"] * 20)})
    m = topica.LDA(3, seed=13).fit(
        topica.from_dataframe(df, text_col="text", metadata_cols=["party"]), iters=40)
    with pytest.raises(ValueError, match="one-key dict.*or a 2-element sequence"):
        topica.predicted_prevalence(m, formula="party", data=df,
                                    contrast=("R", 1.0, 0.0))


def test_diagnostics_no_spurious_warning_after_top_words_bare_strings():
    # #761: diagnostics() unpacked top_words as (word, weight) pairs, but since
    # #752 top_words returns bare strings, so it warned and fell back on every call.
    import warnings
    docs = [["housing", "rent", "tenant", "landlord", "evict"] * 3,
            ["health", "clinic", "doctor", "patient", "care"] * 3,
            ["school", "teacher", "student", "class", "grade"] * 3] * 20
    m = topica.LDA(3, seed=13).fit(topica.Corpus.from_documents(docs), iters=80)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = topica.diagnostics(m, texts=docs)
    assert len(rows) == 3


def test_perplexity_rejects_heldout_object():
    # #767: perplexity() must not silently accept a Heldout. make_heldout withholds
    # words in place; that word-holdout is scored by eval_heldout, not by
    # perplexity's own document-completion split. Passing the object would score the
    # reduced training corpus under a different scheme than the user intends, so we
    # redirect (mirroring the Heldout.corpus guard from #765). The earlier #761/#762
    # accept behavior is deliberately reversed here.
    docs = [["housing", "rent", "tenant", "landlord", "evict"] * 3,
            ["health", "clinic", "doctor", "patient", "care"] * 3,
            ["school", "teacher", "student", "class", "grade"] * 3] * 20
    c = topica.Corpus.from_documents(docs)
    m = topica.LDA(3, seed=13).fit(c, iters=80)
    ho = topica.make_heldout(c)

    with pytest.raises(TypeError) as exc:
        topica.perplexity(m, ho)
    # The error must hand the user both correct paths.
    assert "eval_heldout" in str(exc.value)
    assert "ho.documents" in str(exc.value)

    # A plain Corpus is still accepted, and the explicit escape hatch
    # (document-completion perplexity on the reduced corpus) still returns a number.
    assert float(topica.perplexity(m, c)) > 0
    assert float(topica.perplexity(m, ho.documents)) > 0


def test_topic_stability_fitted_runs_with_num_topics_is_directive():
    # #761: passing fitted runs AND num_topics= tripped the corpus overload, which
    # then tried to .fit() the model list. It should raise a directive TypeError.
    docs = [["housing", "rent", "tenant", "landlord", "evict"] * 3,
            ["health", "clinic", "doctor", "patient", "care"] * 3,
            ["school", "teacher", "student", "class", "grade"] * 3] * 20
    c = topica.Corpus.from_documents(docs)
    runs = [topica.LDA(3, seed=s).fit(c, iters=40) for s in (1, 2)]
    with pytest.raises(TypeError, match="list of fitted models"):
        topica.topic_stability(runs, num_topics=3)
    # the plain fitted-runs path still works
    assert 0.0 <= float(topica.topic_stability(runs)) <= 1.0


def test_stability_helpers_discoverable_under_robustness():
    # #775 T4.1: "does a topic survive a different seed?" is a robustness question,
    # so users reach for topica.robustness.topic_stability first. Re-export the
    # stability helpers there (evaluate stays their home) so the guess resolves.
    assert topica.robustness.topic_stability is topica.evaluate.topic_stability
    assert topica.robustness.bootstrap_stability is topica.evaluate.bootstrap_stability
    assert {"topic_stability", "bootstrap_stability"} <= set(topica.robustness.__all__)


def test_bootstrap_stability_rejects_fitted_model_directively():
    # #775 T2.2: bootstrap_stability refits from the corpus, so bootstrap_stability(
    # model) and bootstrap_stability(model, corpus) — the natural guesses — used to
    # die on an opaque positional-arg error. Redirect to docs-first + reference=.
    docs = [["cat", "dog", "pet"]] * 15 + [["star", "moon", "sky"]] * 15
    m = topica.LDA(2, seed=1).fit(docs, iters=50)
    for call in (lambda: topica.evaluate.bootstrap_stability(m),
                 lambda: topica.evaluate.bootstrap_stability(m, docs)):
        with pytest.raises(TypeError, match="does not take a fitted model|reference="):
            call()
    # a second positional that is NOT a model still gets a clear keyword-only error
    with pytest.raises(TypeError, match="one positional argument"):
        topica.evaluate.bootstrap_stability(docs, docs)
    # the correct call still works
    r = topica.evaluate.bootstrap_stability(docs, num_topics=2, n_boot=3, iters=40)
    assert 0.0 <= r["mean"] <= 1.0


def test_topic_stability_per_topic_vector():
    # #775 T3.3: per_topic=True returns a per-topic vector (index-aligned to
    # runs[0]) alongside the scalar default, so a lone 0.46 isn't the only signal.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    runs = [topica.LDA(3, seed=s).fit(docs, iters=60) for s in (1, 2, 3)]
    overall = topica.topic_stability(runs)
    per = topica.topic_stability(runs, per_topic=True)
    assert isinstance(overall, float)
    assert per.shape == (3,)
    assert np.all((per >= 0.0) & (per <= 1.0))
