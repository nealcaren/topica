"""Behavior tests for ECTM (Evolving Content Topic Model)."""
import numpy as np
import pytest

import topica
from topica.ectm import content_contrast, content_divergence, content_trajectory, content_words


@pytest.fixture(autouse=True)
def _enable_experimental():
    """ECTM is experimental and gated; opt in for these tests and restore after."""
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    yield
    topica.enable_experimental(was)


def test_ectm_is_gated_without_optin():
    """Without opt-in, construction and load raise a clear, actionable error."""
    topica.enable_experimental(False)
    try:
        with pytest.raises(RuntimeError, match="experimental"):
            topica.ECTM(num_topics=3)
        with pytest.raises(RuntimeError, match="experimental"):
            topica.ECTM.load("does-not-exist.topica")
    finally:
        topica.enable_experimental(True)
    topica.ECTM(num_topics=3)  # opted back in: constructs fine


def _corpus(reps=60, drift=True):
    """Two groups over three periods, vocab {a,b,x,y}. Group A always uses {a,b}.
    Group B starts on {a,b} and (if drift) moves onto {x,y} by the last period —
    a group-by-time content contrast that grows. With drift=False both groups
    use {a,b} every period (no contrast)."""
    docs, groups, times = [], [], []
    if drift:
        gb = {0: ["a", "b", "a", "b"], 1: ["a", "b", "x", "y"], 2: ["x", "y", "x", "y"]}
    else:
        gb = {0: ["a", "b", "a", "b"], 1: ["a", "b", "a", "b"], 2: ["a", "b", "a", "b"]}
    for _ in range(reps):
        for per in range(3):
            docs.append(["a", "b", "a", "b"]); groups.append("A"); times.append(2000 + per)
            docs.append(gb[per]); groups.append("B"); times.append(2000 + per)
    return docs, groups, times


def _fit(seed=1, drift=True, init="spectral", **kw):
    docs, groups, times = _corpus(drift=drift)
    m = topica.ECTM(num_topics=2, seed=seed, init=init)
    m.fit(docs, times=times, content=groups, iters=60,
          period_smooth=5.0, interaction_shrink=2.0, **kw)
    return m


def _placebo_model(docs, groups, times):
    m = topica.ECTM(num_topics=2, seed=1)
    m.fit(docs, times=times, content=groups, iters=60)
    return m


# --- The four idioms -------------------------------------------------------

def test_shapes_and_normalization():
    m = _fit()
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape == (360, 2)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    assert m.num_groups == 2 and m.num_periods == 3
    assert m.groups == ["A", "B"]
    assert m.periods == ["2000", "2001", "2002"]
    # per-cell content distributions are normalized
    for g in range(2):
        for t in range(3):
            cw = m.content_word_dist(g, t)
            assert cw.shape == (2, len(m.vocabulary))
            np.testing.assert_allclose(cw.sum(axis=1), 1.0, atol=1e-9)


def test_spectral_init_is_seed_independent():
    # The default spectral base (#220) is deterministic, so the fit is bit-exact:
    # different seeds give identical topics and content. No random-base collapse.
    a, b = _fit(seed=3), _fit(seed=4)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.content_word_dist(1, 2), b.content_word_dist(1, 2))


def test_random_init_is_seeded_not_fixed():
    # init="random" keeps the pre-#220 behavior: same seed reproduces, different
    # seeds differ.
    a, b = _fit(seed=3, init="random"), _fit(seed=3, init="random")
    assert np.array_equal(a.content_word_dist(1, 2), b.content_word_dist(1, 2))
    c = _fit(seed=4, init="random")
    assert not np.array_equal(a.content_word_dist(1, 2), c.content_word_dist(1, 2))


def test_save_load_roundtrip(tmp_path):
    m = _fit()
    p = str(tmp_path / "m.tt")
    m.save(p)
    loaded = topica.ECTM.load(p)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert loaded.periods == m.periods and loaded.groups == m.groups
    assert np.array_equal(m.content_word_dist("B", 2), loaded.content_word_dist("B", 2))


def test_bad_params():
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=1)
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2, sigma_shrink=2.0)
    docs, groups, times = _corpus(reps=4)
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2).fit(docs, times=times, content=groups, interaction_shrink=0.0)
    # times / content length mismatch
    with pytest.raises(ValueError):
        topica.ECTM(num_topics=2).fit(docs, times=times[:-1], content=groups)


# --- Synthetic recovery scenarios (from ECTM.md) ---------------------------

def test_recovers_growing_contrast():
    """Scenario 4: a group difference that grows over time."""
    m = _fit(drift=True)
    vocab = m.vocabulary
    xi, yi = vocab.index("x"), vocab.index("y")
    # topic where B carries the {x,y} mass in the last period
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, xi] + m.content_word_dist("B", 2)[k, yi])

    def gap(per):
        b = m.content_word_dist("B", per)[k, xi] + m.content_word_dist("B", per)[k, yi]
        a = m.content_word_dist("A", per)[k, xi] + m.content_word_dist("A", per)[k, yi]
        return b - a

    assert gap(2) > gap(0) + 0.2, "B-vs-A contrast should grow across periods"
    assert gap(2) > 0.3


def test_no_contrast_when_groups_identical():
    """Scenario 1: no group differences — divergence stays small every period."""
    m = _fit(drift=False)
    for k in range(2):
        for _, dist in content_divergence(m, k, "A", "B"):
            assert dist < 0.15, "identical groups should have near-zero divergence"


def test_content_helpers_surface():
    m = _fit(drift=True)
    vocab = m.vocabulary
    xi = vocab.index("x")
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, xi])
    # content_words returns ranked (word, prob)
    cw = content_words(m, k, "B", 2, n=3)
    assert len(cw) == 3 and all(isinstance(w, str) for w, _ in cw)
    # content_contrast returns both directions
    con = content_contrast(m, k, "B", "A", 2, n=3)
    assert "toward_B" in con and "toward_A" in con
    # the growing word should head the toward-B list in the last period
    assert "x" in {w for w, _ in con["toward_B"]} or "y" in {w for w, _ in con["toward_B"]}
    # content_trajectory: the B-A contrast for x grows
    traj = content_trajectory(m, k, "x", contrast=("B", "A"))
    assert [p for p, _ in traj] == m.periods
    assert traj[-1][1] > traj[0][1]
    # content_divergence in [0,1], one per period
    div = content_divergence(m, k, "A", "B")
    assert len(div) == m.num_periods and all(0.0 <= d <= 1.0 for _, d in div)


def test_prevalence_helpers():
    docs, groups, times = _corpus(drift=True)
    m = topica.ECTM(num_topics=2, seed=1)
    m.fit(docs, times=times, content=groups, iters=40)
    from topica.ectm import prevalence_by_group, prevalence_contrast
    pv = prevalence_by_group(m, groups, times)
    assert pv.shape == (m.num_groups, m.num_periods, m.num_topics)
    # populated cells are topic distributions (sum to 1 over topics)
    finite = pv[~np.isnan(pv).any(axis=2)]
    np.testing.assert_allclose(finite.sum(axis=1), 1.0, atol=1e-6)
    one = prevalence_by_group(m, groups, times, topic=0)
    assert one.shape == (m.num_groups, m.num_periods)
    pc = prevalence_contrast(m, 0, "A", "B", groups, times)
    assert len(pc) == m.num_periods and [p for p, _ in pc] == m.periods


def test_content_contrast_se():
    from topica.ectm import content_contrast_se
    docs, groups, times = _corpus(drift=True)
    m = topica.ECTM(num_topics=2, seed=1)
    m.fit(docs, times=times, content=groups, iters=40)
    dl = [len(d) for d in docs]
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, m.vocabulary.index("x")])
    res = content_contrast_se(m, k, "B", "A", 2, groups, times, dl, n=4)  # period index 2
    assert len(res) == 4
    for w, c, se in res:
        assert isinstance(w, str) and se >= 0
    full = dict((w, (c, se)) for w, c, se in
                content_contrast_se(m, k, "B", "A", 2, groups, times, dl, n=len(m.vocabulary)))
    # the drifting word leans toward B with a finite SE in the last period
    assert full["x"][0] > 0 and full["x"][1] > 0


def test_content_trajectory_ci():
    from topica.ectm import content_trajectory_ci
    docs, groups, times = _corpus(drift=True)

    def refit(d, g, p):
        m = topica.ECTM(num_topics=2, seed=5)
        m.fit(d, times=p, content=g, iters=30)
        return m

    m0 = refit(docs, groups, times)
    k = max(range(2), key=lambda k: m0.content_word_dist("B", 2)[k, m0.vocabulary.index("x")])
    anchor = [w for w, _ in m0.top_words(4, topic=k)]
    band = content_trajectory_ci(refit, docs, groups, times, anchor_words=anchor,
                                 word="x", contrast=("B", "A"),
                                 clusters=list(zip(groups, times)), n_boot=4, seed=0)
    assert len(band) == m0.num_periods
    assert [p for p, *_ in band] == m0.periods  # sorted in period order
    for _, mean, lo, hi in band:
        assert lo <= mean <= hi


def test_content_placebo_separates_drift_from_floor():
    """The placebo (issue #230): a real group-by-time contrast clears the
    finite-sample floor (small p), identical groups do not (large p)."""
    from topica.ectm import content_placebo

    # drift: B moves onto {x,y}; the contrast is real -> observed >> floor.
    docs, groups, times = _corpus(drift=True)
    m = _placebo_model(docs, groups, times)
    res = content_placebo(m, docs, groups, times, n_perm=20, iters=40, seed=0)
    assert res.observed.shape == (2,)
    assert res.null.shape == (20, 2)
    assert res.floor.shape == (2,) and res.pval.shape == (2,)
    assert res.group_a == "A" and res.group_b == "B"
    # the topic carrying the drift should sit clearly above its floor and be sig.
    k = int(np.argmax(res.observed))
    assert res.observed[k] > res.floor[k]
    assert res.pval[k] <= 0.1
    rows = res.as_dict()
    assert len(rows) == 2 and {"topic", "observed", "floor", "pvalue"} <= set(rows[0])


def test_content_placebo_null_when_groups_identical():
    """No real contrast -> observed near the floor, p-value not significant."""
    from topica.ectm import content_placebo

    docs, groups, times = _corpus(drift=False)
    m = _placebo_model(docs, groups, times)
    res = content_placebo(m, docs, groups, times, n_perm=20, iters=40, seed=0)
    # observed should be in the bulk of the null, not its tail
    for t in range(2):
        assert res.pval[t] > 0.1


# --- Minibatch / stochastic VI (issue #231) --------------------------------

def _fit_svi(seed=1, drift=True, **kw):
    # This synthetic corpus is tiny (vocab 4, 360 docs), so SVI needs many epochs,
    # a gentle schedule, and frequent content updates (content_every=1) to converge
    # to the batch fit; on real corpora the defaults (content_every=0, once per
    # epoch) recover at ~30 epochs (see the paper's batch-vs-SVI check).
    docs, groups, times = _corpus(drift=drift)
    m = topica.ECTM(num_topics=2, seed=seed)
    m.fit(docs, times=times, content=groups, iters=150, inference="svi",
          batch_size=48, tau=4.0, kappa=0.6, content_every=1,
          period_smooth=5.0, interaction_shrink=2.0, **kw)
    return m


def test_svi_shapes_and_normalization():
    m = _fit_svi()
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert m.doc_topic.shape == (360, 2)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    for g in range(2):
        for t in range(3):
            np.testing.assert_allclose(m.content_word_dist(g, t).sum(axis=1), 1.0, atol=1e-9)


def test_svi_recovers_growing_contrast():
    """SVI recovers the same qualitative structure as the batch fit: the drift
    topic carries a large A-vs-B content gap that grows across periods."""
    m = _fit_svi(drift=True)
    vocab = m.vocabulary
    xi, yi = vocab.index("x"), vocab.index("y")
    k = max(range(2), key=lambda k: m.content_word_dist("B", 2)[k, xi] + m.content_word_dist("B", 2)[k, yi])

    def gap(per):
        b = m.content_word_dist("B", per)[k, xi] + m.content_word_dist("B", per)[k, yi]
        a = m.content_word_dist("A", per)[k, xi] + m.content_word_dist("A", per)[k, yi]
        return b - a

    assert gap(2) > gap(0) + 0.2, "B-vs-A contrast should grow across periods"
    assert gap(2) > 0.3
    # the drift topic clears a clear divergence, batch-style
    div = max(np.mean([d for _, d in content_divergence(m, t, "A", "B")]) for t in range(2))
    assert div > 0.3


def test_svi_no_contrast_when_groups_identical():
    m = _fit_svi(drift=False)
    for k in range(2):
        for _, dist in content_divergence(m, k, "A", "B"):
            assert dist < 0.2, "identical groups should have near-zero divergence under SVI"


def test_svi_is_seed_reproducible_not_bit_exact():
    """SVI samples minibatches from the model seed: same seed reproduces exactly,
    a different seed differs (unlike the deterministic spectral batch fit)."""
    a, b = _fit_svi(seed=1), _fit_svi(seed=1)
    assert np.array_equal(a.content_word_dist(1, 2), b.content_word_dist(1, 2))
    c = _fit_svi(seed=2)
    assert not np.array_equal(a.content_word_dist(1, 2), c.content_word_dist(1, 2))


def test_svi_with_prevalence():
    """SVI threads a prevalence design through the blended ridge gamma."""
    docs, groups, times = _corpus(drift=True)
    party, _ = topica.one_hot(groups)
    m = topica.ECTM(num_topics=2, seed=1)
    m.fit(docs, times=times, content=groups, prevalence=party,
          iters=30, inference="svi", batch_size=48, tau=16.0, kappa=0.7)
    assert m.prevalence_effects.shape[1] == 1  # K-1
    assert m.doc_topic.shape == (360, 2)


def test_svi_content_convergence_diagnostic():
    """SVI exposes a content-convergence diagnostic; batch leaves it empty."""
    m = _fit_svi(drift=True)
    assert len(m.content_shift_history) >= 1
    assert m.content_converged  # the gentle toy schedule settles
    b = _fit(drift=True)
    assert b.content_shift_history == [] and b.content_converged


def test_svi_under_convergence_warns():
    """A starved content schedule (one solve from zero) is flagged loudly, not
    silently returned as a collapsed content model."""
    import warnings
    docs, groups, times = _corpus(drift=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = topica.ECTM(num_topics=2, seed=1)
        # content_every huge => only the end-of-fit flush solves kappa once
        m.fit(docs, times=times, content=groups, iters=1, inference="svi",
              batch_size=48, content_every=10**6)
    assert not m.content_converged
    assert any("may not have converged" in str(x.message) for x in w)


def test_svi_save_load_roundtrip(tmp_path):
    m = _fit_svi()
    p = str(tmp_path / "svi.tt")
    m.save(p)
    loaded = topica.ECTM.load(p)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert np.array_equal(m.content_word_dist("B", 2), loaded.content_word_dist("B", 2))


def test_analysis_surface():
    m = _fit()
    assert topica.summary(m) is not None
    assert topica.topic_table(m) is not None
    assert m.coherence(5).shape == (2,)


# --- Seeded content (experimental): anchor a topic's baseline vocabulary -----

def test_seeds_none_is_bit_exact_with_baseline():
    """seeds=None must not perturb the fit (the seed prior mean is all zeros)."""
    base = _fit(seed=3)
    seeded_none = _fit(seed=3, seeds=None)
    assert np.allclose(np.asarray(base.topic_word),
                       np.asarray(seeded_none.topic_word))


def test_seeds_consolidate_vocabulary_into_the_seeded_topic():
    """The E-step seed boost forces the seed words' tokens onto the seeded topic,
    so that topic owns almost all of the seed vocabulary's probability mass --
    the anchoring that lets ECTM read a pre-specified issue off a known topic."""
    # A corpus with three latent themes, so seeding must actively consolidate.
    themes = [["tax", "budget", "fiscal"], ["war", "troops", "army"],
              ["health", "clinic", "patient"]]
    docs, groups, times = [], [], []
    for i in range(180):
        docs.append(themes[i % 3] * 3)
        groups.append("A" if i % 2 else "B")
        times.append(2000 + i % 3)

    def seed_share(seeds):
        m = topica.models.ECTM(num_topics=6, seed=1)
        m.fit(docs, times=times, content=groups, iters=120, seeds=seeds, seed_strength=6.0)
        beta = np.asarray(m.topic_word)
        sw = [m.vocabulary.index(w) for w in ("tax", "budget", "fiscal")]
        best_share = (beta[:, sw].sum(axis=0) / beta[:, sw].sum()).max()   # unseeded
        t0_share = beta[0][sw].sum() / beta[:, sw].sum()                    # seeded topic 0
        return best_share, t0_share

    best_unseeded, _ = seed_share(None)
    _, seeded_t0 = seed_share({0: ["tax", "budget", "fiscal"]})
    # seeding consolidates the fiscal vocabulary onto topic 0
    assert seeded_t0 > best_unseeded
    assert seeded_t0 > 0.6


def test_seeds_reject_out_of_range_topic():
    docs, groups, times = _corpus()
    m = topica.models.ECTM(num_topics=2, seed=1)
    with pytest.raises(Exception):
        m.fit(docs, times=times, content=groups, iters=20, seeds={5: ["a"]})


def test_seeds_ignore_unknown_words():
    """Words outside the vocabulary are silently skipped (no crash)."""
    m = _fit(seed=1, seeds={0: ["x", "notaword"]}, seed_strength=5.0)
    assert np.asarray(m.topic_word).shape[0] == 2
