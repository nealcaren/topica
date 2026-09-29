"""Tests for SITS, parametric Speaker Identity for Topic Segmentation (issue #906).

Shapes and normalization, planted recovery of per-speaker shift propensities and
segment boundaries, determinism, save/load, argument checks, the reference's
bookkeeping (forced turns, short turns, compat mode), and the analysis surface.
The exact-posterior check (enumeration on a tiny conversation) lives in the Rust
unit tests in src/sits.rs.
"""
import warnings

import numpy as np
import pytest

import topica

BLOCKS = [[f"a{i}" for i in range(12)], [f"b{i}" for i in range(12)], [f"c{i}" for i in range(12)]]


def _planted(num_convs=10, turns_per_conv=24, seed=0, short_every=0):
    """Conversations where speaker "shifter" always moves to a new word block and
    "follower" always stays on the current one."""
    rng = np.random.default_rng(seed)
    turns, speakers, convs = [], [], []
    for c in range(num_convs):
        block = rng.integers(3)
        for t in range(turns_per_conv):
            spk = "shifter" if t % 4 == 0 else "follower"
            if t > 0 and spk == "shifter":
                block = (block + 1 + rng.integers(2)) % 3
            n = 2 if (short_every and t % short_every == short_every - 1 and spk == "follower") else 10
            turns.append(list(rng.choice(BLOCKS[block], size=n)))
            speakers.append(spk)
            convs.append(f"c{c}")
    return turns, speakers, convs


def _fit(seed=1, iters=3000, **kw):
    turns, speakers, convs = _planted()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return topica.SITS(3, seed=seed, **kw).fit(
            turns, speakers, conversations=convs, iters=iters)


@pytest.fixture(scope="module")
def fitted():
    return _fit()


def test_shapes_and_normalization(fitted):
    m = fitted
    t = len(m.shift_prob)
    assert m.topic_word.shape == (3, len(m.vocabulary))
    assert m.doc_topic.shape == (t, 3)
    assert m.turn_topic.shape == (t, 3)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    assert m.speakers == ["follower", "shifter"]
    assert m.shift_propensity.shape == (2,)
    assert m.shift_propensity_interval(0.9).shape == (2, 2)
    assert m.shift_propensity_draws.shape[1] == 2
    assert m.num_draws == 1500 and m.burn_in == 1500
    assert len(m.shift_trace) == 3000
    # the default warm start holds every eligible turn as a shift for 1,000 sweeps
    assert np.all(m.shift_trace[:1000] == m.eligible.sum())
    assert m.speaker_turn_counts.sum() == t


def test_recovers_planted_shifters_and_boundaries(fitted):
    m = fitted
    rate = dict(zip(m.speakers, m.eligible_shift_rate))
    assert rate["shifter"] > 0.85
    assert rate["follower"] < 0.1
    # the shifter's turns are (almost) all segment starts
    turns, speakers, _ = _planted()
    shifter = np.array([s == "shifter" for s in speakers])
    assert m.shift_prob[shifter].mean() > 0.85
    # each topic owns one word block
    owner = [max(range(3), key=lambda b: len(set(m.top_words(6, topic=k)) & set(BLOCKS[b])))
             for k in range(3)]
    assert sorted(owner) == [0, 1, 2]


def test_readsits_score_counts_all_turns(fitted):
    """shift_propensity is Rossiter's (γ + Σl)/(2γ + n) over all of a speaker's
    turns, the openers included; eligible_shift_rate leaves them out."""
    m = fitted
    g = m.settings["gamma"]
    counts = m.speaker_turn_counts
    elig = m.speaker_eligible_counts
    shifts = m.eligible_shift_rate * elig
    openers = np.zeros(2)
    conv = m.conversation_index
    first = np.r_[True, conv[1:] != conv[:-1]]
    turns, speakers, _ = _planted()
    for i, s in enumerate(m.speakers):
        openers[i] = sum(1 for t, sp in enumerate(speakers) if sp == s and first[t])
    expect = (g + shifts + openers) / (2 * g + counts)
    np.testing.assert_allclose(m.shift_propensity, expect, atol=1e-12)
    # draws average to the reported mean only up to thinning; the interval brackets it
    lo, hi = m.shift_propensity_interval(0.99).T
    assert np.all(lo <= m.shift_propensity + 1e-12) and np.all(m.shift_propensity <= hi + 1e-12)


def test_determinism_and_seed_sensitivity():
    a, b = _fit(seed=4, iters=80), _fit(seed=4, iters=80)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert np.array_equal(a.shift_prob, b.shift_prob)
    assert np.array_equal(a.shift_trace, b.shift_trace)
    c = _fit(seed=5, iters=80)
    assert not np.array_equal(a.shift_trace, c.shift_trace)


def test_save_load_roundtrip(fitted, tmp_path):
    p = str(tmp_path / "sits.tt")
    fitted.save(p)
    m = topica.SITS.load(p)
    for attr in ["topic_word", "doc_topic", "shift_prob", "shift_propensity",
                 "eligible_shift_rate", "segments", "shift_trace", "eligible"]:
        assert np.array_equal(getattr(m, attr), getattr(fitted, attr), equal_nan=True), attr
    assert m.speakers == fitted.speakers
    assert m.settings == fitted.settings
    assert m.geweke_z == fitted.geweke_z


def test_short_and_empty_turns_are_kept_and_never_shift():
    turns, speakers, convs = _planted(short_every=3)
    turns[5] = []  # an empty turn still counts toward its speaker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(3, seed=2).fit(turns, speakers, conversations=convs, iters=200)
    assert len(m.shift_prob) == len(turns)
    short = np.array([0 < len(t) < 5 or len(t) == 0 for t in turns])
    assert not m.eligible[short].any()
    assert np.all(m.shift_prob[short & ~np.r_[True, np.array(convs[1:]) != np.array(convs[:-1])]] == 0)
    assert m.speaker_turn_counts.sum() == len(turns)
    # sampler counts equal the recorded shifts in the default mode
    assert m.num_phantom == 0


def test_openers_always_shift(fitted):
    first = np.r_[True, fitted.conversation_index[1:] != fitted.conversation_index[:-1]]
    assert np.all(fitted.shift_prob[first] == 1.0)
    assert not fitted.eligible[first].any()
    seg = fitted.segments
    assert np.all(np.diff(seg) >= 0) and seg[0] == 0


def test_short_turn_warning():
    turns, speakers, convs = _planted(short_every=2)
    with pytest.warns(UserWarning, match="min_shift_tokens"):
        topica.SITS(3, seed=0).fit(turns, speakers, conversations=convs, iters=20)


def test_too_short_chain_warns_instead_of_staying_silent():
    turns, speakers, convs = _planted()
    with pytest.warns(UserWarning, match="too few to check convergence"):
        m = topica.SITS(3, seed=0).fit(turns, speakers, conversations=convs, iters=100)
    assert m.geweke_z is None


def test_five_token_turns_are_eligible_four_are_not():
    """Rossiter's test is len >= min_shift_tokens (the original code used > 5)."""
    turns = [["a"] * 6, ["b"] * 5, ["c"] * 4, ["d"] * 5, ["e"] * 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(2, seed=0).fit(turns, [0, 1, 0, 1, 0],
                                       conversations=[0, 0, 0, 0, 0], iters=30)
    assert m.eligible.tolist() == [False, True, False, True, True]
    assert m.short_turn_share == 0.25


def test_compat_reproduces_phantom_boundaries():
    turns, speakers, convs = _planted(short_every=2)
    with pytest.warns(UserWarning, match="rossiter2022"):
        m = topica.SITS(3, compat="rossiter2022", init_shift_rate=1 / 3, seed=3).fit(
            turns, speakers, conversations=convs, iters=100)
    assert m.num_phantom > 0
    # phantom shifts stay in the sampler's shift counts for the whole chain, on top
    # of the conversation openers
    num_convs = len(set(convs))
    assert m.sampler_shift_counts[:, 1].sum() >= m.num_phantom + num_convs
    # the recorded shifts never include a short turn (select short turns by the
    # conversation boundaries, not by shift_prob, which would hide a phantom at 1)
    conv = m.conversation_index
    opener = np.r_[True, conv[1:] != conv[:-1]]
    short = ~m.eligible & ~opener
    assert short.sum() > 0
    assert np.all(m.shift_prob[short] == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = topica.SITS(3, init="random", init_shift_rate=1 / 3, seed=3).fit(
            turns, speakers, conversations=convs, iters=100)
    assert d.num_phantom == 0


def test_compat_requires_integer_init_denominator():
    with pytest.raises(ValueError, match="1/I"):
        topica.SITS(3, compat="rossiter2022", init_shift_rate=0.3)
    topica.SITS(3, compat="rossiter2022", init_shift_rate=0.0)  # I = 1: never
    topica.SITS(3, compat="rossiter2022", init_shift_rate=0.25)
    with pytest.raises(ValueError, match="compat must be"):
        topica.SITS(3, compat="rossiter", init_shift_rate=0.25)
    # a typo is reported as a typo, not as a missing init_shift_rate
    with pytest.raises(ValueError, match="compat must be"):
        topica.SITS(3, compat="rossiter")


def test_compat_requires_an_explicit_init_rate():
    """The default 0.1 would silently replicate an I = 10 run Rossiter never made."""
    with pytest.raises(ValueError, match="explicit init_shift_rate"):
        topica.SITS(3, compat="rossiter2022")
    assert topica.SITS(3, init="random").settings["init_shift_rate"] == 0.1
    assert topica.SITS(3).settings["init"] == "lda"
    assert topica.SITS(3, compat="rossiter2022", init_shift_rate=0.25).settings["init"] == "random"


def test_init_and_warmup_arguments():
    with pytest.raises(ValueError, match="init_shift_rate applies"):
        topica.SITS(3, init_shift_rate=0.2)
    with pytest.raises(ValueError, match="init='random'"):
        topica.SITS(3, compat="rossiter2022", init="lda", init_shift_rate=0.25)
    with pytest.raises(ValueError, match="init must be"):
        topica.SITS(3, init="warm")
    turns, speakers, convs = _planted(num_convs=2)
    with pytest.raises(ValueError, match="at least warmup"):
        topica.SITS(3, warmup=500).fit(turns, speakers, conversations=convs, iters=600)


def test_lda_warm_start_holds_shifts_during_warmup():
    turns, speakers, convs = _planted()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(3, warmup=100, seed=4).fit(
            turns, speakers, conversations=convs, iters=400, burn_in=200)
    assert np.all(m.shift_trace[:100] == m.eligible.sum())
    rate = dict(zip(m.speakers, m.eligible_shift_rate))
    assert rate["shifter"] > 0.85 and rate["follower"] < 0.1


def test_argument_checks():
    turns, speakers, convs = _planted(num_convs=2)
    with pytest.raises(ValueError):
        topica.SITS(0)
    for bad in [dict(alpha=0.0), dict(beta=-1.0), dict(gamma=float("nan")),
                dict(init_shift_rate=1.5), dict(gamma=1e308), dict(beta=1e-320),
                dict(alpha=1e12)]:
        with pytest.raises(ValueError):
            topica.SITS(3, **bad)
    m = topica.SITS(3)
    with pytest.raises(ValueError, match="conversations"):
        m.fit(turns, speakers)
    with pytest.raises(ValueError, match="speakers"):
        m.fit(turns, conversations=convs)
    with pytest.raises(ValueError, match="entries"):
        m.fit(turns, speakers[:-1], conversations=convs)
    with pytest.raises(ValueError, match="burn_in"):
        m.fit(turns, speakers, conversations=convs, iters=10, burn_in=10)
    # a conversation id that comes back after another conversation is an error
    shuffled = convs[:5] + ["c1"] * 3 + convs[8:]
    with pytest.raises(ValueError, match="reappears"):
        m.fit(turns, speakers, conversations=shuffled)
    with pytest.raises(ValueError, match="not both"):
        m.fit(turns, speakers, authors=speakers, conversations=convs)
    with pytest.raises((ValueError, RuntimeError)):
        m.fit([], [], conversations=[])
    with pytest.raises(RuntimeError, match="not fitted"):
        topica.SITS(3).shift_prob


def test_authors_alias_and_int_labels():
    turns, speakers, convs = _planted(num_convs=3)
    ids = [0 if s == "follower" else 1 for s in speakers]
    conv_ids = [int(c[1:]) for c in convs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = topica.SITS(3, seed=1).fit(turns, authors=ids, conversations=conv_ids, iters=40)
        b = topica.SITS(3, seed=1).fit(turns, ids, conversations=conv_ids, iters=40)
    assert a.speakers == [0, 1]
    assert np.array_equal(a.shift_prob, b.shift_prob)


def test_integer_speakers_sort_numerically_and_align():
    turns, _, convs = _planted(num_convs=3)
    ids = [i % 12 for i in range(len(turns))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(3, seed=1).fit(turns, ids, conversations=convs, iters=30)
    assert m.speakers == list(range(12))
    np.testing.assert_array_equal(m.speaker_turn_counts, np.bincount(ids))
    np.testing.assert_array_equal(np.asarray(m.speakers)[m.speaker_index], ids)


@pytest.mark.parametrize("bad", [
    lambda s: [1 if x == "shifter" else "1" for x in s],   # 1 and "1" would merge
    lambda s: [None] + s[1:],
    lambda s: [True] + s[1:],
    lambda s: [1.5] + s[1:],
])
def test_ambiguous_labels_are_rejected(bad):
    turns, speakers, convs = _planted(num_convs=2)
    with pytest.raises(ValueError):
        topica.SITS(3).fit(turns, bad(speakers), conversations=convs, iters=5)


def test_a_bare_string_is_not_split_into_speakers():
    with pytest.raises(ValueError, match="single string"):
        topica.SITS(2).fit([["a"], ["b"]], "ab", conversations=[0, 0], iters=5)


def test_a_pruned_corpus_is_refused():
    turns, speakers, convs = _planted(num_convs=2)
    turns[3] = ["a_rare_word"]  # emptied by min_doc_freq=2, so the Corpus drops it
    corpus = topica.Corpus.from_documents(turns, min_doc_freq=2)
    assert len(corpus.kept_indices) < len(turns)
    with pytest.raises(ValueError, match="dropped turns"):
        topica.SITS(3).fit(corpus, speakers, conversations=convs, iters=5)


def test_a_corpus_that_dropped_trailing_turns_is_refused():
    """Dropping only the last turns leaves kept_indices an identity prefix."""
    corpus = topica.Corpus.from_documents([["x"] * 5, ["x"] * 5, ["rare"]], min_doc_freq=2)
    assert corpus.kept_indices == [0, 1]
    with pytest.raises(ValueError, match="dropped turns"):
        topica.SITS(2).fit(corpus, ["A", "A"], conversations=[0, 0], iters=5)


def test_settings_round_trip():
    for kw in [{}, dict(init="random"), dict(compat="rossiter2022", init_shift_rate=1 / 3)]:
        m = topica.SITS(2, **kw)
        assert topica.SITS(**m.settings).settings == m.settings
    assert topica.SITS(2).settings["init_shift_rate"] is None
    m = _fit(seed=1, iters=300)
    assert m.settings["warmup"] is None  # as passed, so a refit resolves it afresh
    assert m.warmup_used == 150          # min(1000, burn_in) as actually run
    assert m.iters == 300
    topica.SITS(**m.settings)
    assert topica.SITS(2, init="random", warmup=0).settings["warmup"] is None


def test_speaker_table_pools_chains():
    a = _fit(seed=1, iters=300)
    b = _fit(seed=2, iters=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = topica.sits.speaker_table([a, b], level=0.9)
        one = topica.sits.speaker_table(a)
    assert list(t["speaker"]) == a.speakers
    assert (t["openers"] + t["short_turns"] + t["eligible"] == t["turns"]).all()
    np.testing.assert_allclose(t["forced_share"],
                               (t["openers"] + t["short_turns"]) / t["turns"])
    assert (t["shift_propensity_lo"] <= t["shift_propensity_hi"]).all()
    assert np.isfinite(t["rhat"]).all()
    assert t.attrs["chains"] == 2
    assert np.isnan(one["rhat"]).all()


@pytest.mark.parametrize("kw", [dict(gamma=2.0), dict(min_shift_tokens=100),
                                dict(init="random")])
def test_speaker_table_refuses_chains_with_other_settings(kw):
    a = _fit(seed=1, iters=300)
    c = _fit(seed=2, iters=300, **kw)
    with pytest.raises(ValueError, match="chain 1 (was fitted with different settings|differs)"):
        topica.sits.speaker_table([a, c])


def test_speaker_table_refuses_chains_with_other_conversations():
    """Same speakers and turn counts but a different conversation split."""
    turns = [["w"] * 3] * 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = topica.SITS(2, seed=1).fit(turns, ["A"] * 3, conversations=[0, 0, 0], iters=300)
        b = topica.SITS(2, seed=2).fit(turns, ["A"] * 3, conversations=[0, 1, 2], iters=300)
    with pytest.raises(ValueError, match="conversation_index"):
        topica.sits.speaker_table([a, b])


def test_speaker_table_warnings():
    a = _fit(seed=1, iters=300)
    with pytest.warns(UserWarning, match="share seed"):
        topica.sits.speaker_table([a, a])
    # half of the follower's turns are short: forced share above 0.5 warns
    turns, speakers, convs = _planted(short_every=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(3, seed=1).fit(turns, speakers, conversations=convs, iters=300)
    t = None
    with pytest.warns(UserWarning, match="forced_share"):
        t = topica.sits.speaker_table(m)
    assert (t["forced_share"] > 0.5).any()


def test_record_fit_refuses_a_list_and_records_the_run(fitted):
    turns, _, _ = _planted()
    with pytest.raises(TypeError, match="one fitted model"):
        topica.provenance.record_fit([fitted, fitted], turns)
    man = topica.provenance.record_fit(fitted, turns)
    rec = man.model["fit_record"]
    assert rec["iters"] == 3000 and rec["burn_in"] == 1500
    assert rec["num_speakers"] == 2 and rec["num_conversations"] == 10
    assert "shift_propensity" in man.model["output_fingerprints"]


def test_single_turn_conversations_and_k1():
    turns = [["a", "b", "c", "d", "e"]] * 6
    speakers = ["x", "y", "x", "y", "x", "y"]
    convs = [0, 1, 1, 2, 3, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.SITS(1, seed=0).fit(turns, speakers, conversations=convs, iters=50)
    assert m.shift_prob[0] == 1.0 and m.shift_prob[3] == 1.0
    np.testing.assert_allclose(m.doc_topic, 1.0)


def test_analysis_surface(fitted):
    turns, _, _ = _planted()
    assert len(topica.summary(fitted)) > 0
    assert len(topica.topic_table(fitted)) == 3
    assert fitted.coherence(5).shape == (3,)
    assert np.isfinite(topica.coherence(fitted, turns)).all()


def test_speaker_table_refuses_chains_on_other_texts_or_lengths():
    turns, speakers, convs = _planted()
    other = [[w for w in t if w != "a0"] or ["b0"] * 10 for t in turns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = topica.SITS(3, seed=1).fit(turns, speakers, conversations=convs, iters=300)
        b = topica.SITS(3, seed=2).fit(other, speakers, conversations=convs, iters=300)
        c = topica.SITS(3, seed=3).fit(turns, speakers, conversations=convs, iters=400,
                                       burn_in=150)
        d = topica.SITS(3, seed=4).fit(turns, speakers, conversations=convs, iters=300,
                                       sample_interval=10)
    assert np.array_equal(a.eligible, b.eligible)
    with pytest.raises(ValueError, match="different turns"):
        topica.sits.speaker_table([a, b])
    with pytest.raises(ValueError, match="iters"):
        topica.sits.speaker_table([a, c])
    with pytest.raises(ValueError, match="stored"):
        topica.sits.speaker_table([a, d])


def test_speaker_table_warns_on_infinite_rhat():
    turns = [["x"] * 5] * 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = [topica.SITS(1, init="random", gamma=1e-10, seed=s).fit(
            turns, ["A", "B", "B"], conversations=[0, 0, 0], iters=40) for s in (0, 5)]
    # a near-zero gamma freezes the two chains at opposite extremes: R-hat is inf
    assert sorted(m.eligible_shift_rate[1] for m in ms) == [0.0, 1.0]
    with pytest.warns(UserWarning, match="R-hat"):
        topica.sits.speaker_table(ms)


def test_spearman_handles_ties_and_constants():
    from topica.sits import _spearman
    assert _spearman(np.array([1., 1., 2.]), np.array([2., 1., 3.])) == pytest.approx(0.8660254)
    assert np.isnan(_spearman(np.array([1., 1., 1.]), np.array([1., 2., 3.])))


def test_prep_documents_keeps_the_record_of_dropped_turns():
    from topica.frames import prep_documents
    docs = [["x"] * 5, ["rare"], ["x"] * 5, ["x"] * 5]
    pruned = topica.Corpus.from_documents(docs, min_doc_freq=2)
    assert pruned.dropped_documents is True
    assert topica.Corpus.from_documents(docs).dropped_documents is False
    derived = prep_documents(pruned, lower_thresh=1)
    derived = derived[0] if isinstance(derived, tuple) else derived
    assert derived.dropped_documents is True
    with pytest.raises(ValueError, match="dropped turns"):
        topica.SITS(2).fit(derived, ["A"] * 3, conversations=[0] * 3, iters=5)


def test_manifest_verifies_shift_propensity(fitted):
    turns, _, _ = _planted()
    man = topica.provenance.record_fit(fitted, turns)
    other = _fit(seed=2)
    res = man.verify(model=other)
    assert "model_shift_propensity" in str(res)


def test_find_thoughts_treats_a_dataframe_as_a_matrix():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"turn_topic": [0.1, 0.9], "other": [0.9, 0.1]})
    out = topica.inspect.find_thoughts(frame, topic=0, n=1)
    assert out[0][0] == 1  # column 0 ("turn_topic") peaks at row 1
