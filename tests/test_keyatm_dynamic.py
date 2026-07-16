"""keyATM dynamic model: a Chib (1998) change-point HMM lets topic prevalence
shift over time across ``num_states`` latent regimes. These check that a planted
change point is recovered, that the HMM bookkeeping is well formed, that the
document order is restored after the internal time-sort, and the output API."""

import numpy as np
import pytest

import topica

ECON = ["tax", "market", "trade", "fiscal", "budget", "deficit"]
SOC = ["abortion", "gay", "marriage", "church", "family", "prayer"]
SEEDS = {"economic": ECON[:4], "social": SOC[:4]}


def _corpus(seed=0, n_years=12, change_at=6, per_year=40):
    """Early years lean economic, later years lean social; change at `change_at`."""
    rng = np.random.default_rng(seed)
    docs, years = [], []
    for t in range(n_years):
        soc_share = 0.15 if t < change_at else 0.85
        for _ in range(per_year):
            heavy = SOC if rng.random() < soc_share else ECON
            light = ECON if heavy is SOC else SOC
            docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
            years.append(2000 + t)
    return docs, years


@pytest.fixture(scope="module")
def dyn_model():
    docs, years = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, timestamps=years, num_states=2, iters=400)
    return m, docs, years


def test_time_outputs_shapes(dyn_model):
    m, _, _ = dyn_model
    assert m.time_labels == [str(2000 + t) for t in range(12)]
    assert len(m.time_state) == 12
    assert m.time_prevalence.shape == (12, 2)
    assert m.transition_matrix.shape == (2, 2)
    # prevalence rows are distributions
    assert np.allclose(m.time_prevalence.sum(axis=1), 1.0)


def test_recovers_change_point(dyn_model):
    m, _, _ = dyn_model
    si = m.topic_names.index("social")
    tp = m.time_prevalence[:, si]
    early, late = tp[:6].mean(), tp[6:].mean()
    # The social topic should rise substantially in the later regime.
    assert late - early > 0.3
    # The first and last segments must sit in different HMM states.
    assert m.time_state[0] != m.time_state[-1]


def test_state_path_is_monotone_left_to_right(dyn_model):
    m, _, _ = dyn_model
    # Chib change-point HMM: the state index never decreases over time, and the
    # path visits every state exactly once in order.
    states = m.time_state
    assert states == sorted(states)
    assert states[0] == 0
    assert states[-1] == 1
    assert set(states) == {0, 1}


def test_transition_matrix_is_left_to_right(dyn_model):
    m, _, _ = dyn_model
    P = m.transition_matrix
    # Only the diagonal and super-diagonal carry mass; last state is absorbing.
    assert P[1, 0] == 0.0
    assert np.isclose(P[1, 1], 1.0)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 1] > 0.0  # some probability of advancing


def test_doc_order_preserved_after_internal_sort():
    # Feed documents in shuffled time order; theta must come back aligned to the
    # ORIGINAL document order, not the internally time-sorted order.
    docs, years = _corpus(seed=3)
    order = np.random.default_rng(9).permutation(len(docs))
    docs_sh = [docs[i] for i in order]
    years_sh = [years[i] for i in order]

    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs_sh, timestamps=years_sh, num_states=2, iters=300)
    th = m.doc_topic
    assert th.shape == (len(docs), 2)
    si = m.topic_names.index("social")
    # Documents from 2000-2005 (economic regime) should have lower social mass
    # than 2006-2011 documents, even though they were fed shuffled.
    early = np.array([th[i, si] for i, y in enumerate(years_sh) if y < 2006])
    late = np.array([th[i, si] for i, y in enumerate(years_sh) if y >= 2006])
    assert late.mean() > early.mean() + 0.2


def test_deterministic(dyn_model):
    m, docs, years = dyn_model
    m2 = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m2.fit(docs, timestamps=years, num_states=2, iters=400)
    assert np.allclose(m.time_prevalence, m2.time_prevalence)
    assert m.time_state == m2.time_state
    assert np.allclose(m.doc_topic, m2.doc_topic)


def test_string_timestamps_sorted():
    docs, years = _corpus(seed=1, n_years=4, change_at=2, per_year=30)
    labels = [f"era_{y}" for y in years]  # string timestamps
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, timestamps=labels, num_states=2, iters=200)
    assert m.time_labels == ["era_2000", "era_2001", "era_2002", "era_2003"]


def test_noncontiguous_unsorted_timestamps_keep_theta_aligned():
    # Topica must not require a contiguous, pre-sorted time index: the caller may
    # pass arbitrary timestamp values with gaps, in any order. build_time_index
    # maps the distinct values to a contiguous index, and the fit sorts internally
    # and scatters theta back to the input document order. Plant a block per
    # document independent of its timestamp, so we can check the per-document
    # alignment exactly (not just a regime mean).
    rng = np.random.default_rng(0)
    gap_years = [2001.0, 2004.0, 2009.0, 2013.0, 2020.0]  # non-contiguous, with gaps
    docs, ts, block = [], [], []
    for i in range(120):
        is_econ = i % 2 == 0
        heavy, light = (ECON, SOC) if is_econ else (SOC, ECON)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
        ts.append(float(rng.choice(gap_years)))  # random, unsorted, gapped
        block.append(0 if is_econ else 1)
    block = np.array(block)

    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, timestamps=ts, num_states=2, iters=400)

    # The distinct values define the (contiguous) time order; labels are sorted.
    assert m.time_labels == ["2001", "2004", "2009", "2013", "2020"]
    assert m.time_prevalence.shape == (5, 2)

    # theta is aligned to the INPUT document order: each doc loads on its own
    # keyword topic (economic=0, social=1), regardless of timestamp order/gaps.
    th = np.asarray(m.doc_topic)
    econ_i = m.topic_names.index("economic")
    soc_i = m.topic_names.index("social")
    pred = np.where(th[:, soc_i] > th[:, econ_i], 1, 0)
    assert (pred == block).mean() > 0.95


def test_base_model_has_no_time_outputs():
    docs, _ = _corpus(seed=2, n_years=3, per_year=20)
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=100)
    assert m.time_state == []
    assert m.time_labels == []
    with pytest.raises(RuntimeError):
        _ = m.time_prevalence
    with pytest.raises(RuntimeError):
        _ = m.transition_matrix


def test_timestamps_and_covariates_mutually_exclusive():
    docs, years = _corpus(seed=4, n_years=3, per_year=20)
    x = np.zeros((len(docs), 1))
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, timestamps=years, covariates=x, iters=10)


def test_num_states_validated():
    docs, years = _corpus(seed=5, n_years=3, per_year=20)
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, timestamps=years, num_states=5, iters=10)  # only 3 timestamps


# ---------------------------------------------------------------------------
# time_prevalence_ci: per-period credible intervals from the HMM posterior
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dyn_model_ci():
    """Small dynamic KeyATM with theta_draws retained (the default)."""
    docs, years = _corpus(seed=7, n_years=6, change_at=3, per_year=30)
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=2)
    m.fit(docs, timestamps=years, num_states=2, iters=300, keep_theta_draws=True)
    return m, docs, years


def test_time_prevalence_ci_shapes(dyn_model_ci):
    m, _, years = dyn_model_ci
    result = topica.time_prevalence_ci(m, years)
    T = len(m.time_labels)
    K = m.num_topics
    assert result["labels"] == m.time_labels
    assert result["mean"].shape == (T, K)
    assert result["ci_low"].shape == (T, K)
    assert result["ci_high"].shape == (T, K)
    assert result["sd"].shape == (T, K)


def test_time_prevalence_ci_ordering(dyn_model_ci):
    m, _, years = dyn_model_ci
    result = topica.time_prevalence_ci(m, years)
    assert result["labels"] == m.time_labels


def test_time_prevalence_ci_bounds(dyn_model_ci):
    m, _, years = dyn_model_ci
    result = topica.time_prevalence_ci(m, years)
    # ci_low <= mean <= ci_high everywhere (elementwise, up to floating-point noise)
    assert np.all(result["ci_low"] <= result["mean"] + 1e-12)
    assert np.all(result["mean"] <= result["ci_high"] + 1e-12)
    # sd is non-negative
    assert np.all(result["sd"] >= 0.0)


def test_time_prevalence_ci_requires_draws():
    """Raises a clear error when theta_draws were not retained."""
    docs, years = _corpus(seed=8, n_years=4, change_at=2, per_year=20)
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=3)
    m.fit(docs, timestamps=years, num_states=2, iters=200, keep_theta_draws=False)
    assert m.theta_draws is None
    with pytest.raises(ValueError, match="keep_theta_draws=True"):
        topica.time_prevalence_ci(m, years)


def test_time_prevalence_ci_requires_dynamic_model():
    """Raises for a non-dynamic (base) KeyATM."""
    docs, _ = _corpus(seed=9, n_years=3, per_year=20)
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=4)
    m.fit(docs, iters=100)
    assert m.time_labels == []
    with pytest.raises(ValueError, match="dynamic KeyATM"):
        topica.time_prevalence_ci(m, [0] * len(docs))


def test_time_prevalence_ci_wrong_length(dyn_model_ci):
    """Raises when timestamps length does not match number of documents."""
    m, _, years = dyn_model_ci
    with pytest.raises(ValueError, match="timestamps"):
        topica.time_prevalence_ci(m, years[:-5])


def test_topics_over_time_dynamic_keyatm_has_ci(dyn_model_ci):
    """TopicsOverTime for a dynamic KeyATM uses posterior bands automatically."""
    pytest.importorskip("matplotlib")
    import topica.viz as viz

    m, _, years = dyn_model_ci
    panel = viz.topics_over_time(m, years)
    assert panel.has_ci is True
    df = panel.to_frame()
    assert "ci_low" in df.columns and "ci_high" in df.columns
    # All lower bounds must be <= point estimates <= upper bounds
    assert (df["ci_low"] <= df["prevalence"] + 1e-12).all()
    assert (df["prevalence"] <= df["ci_high"] + 1e-12).all()


def test_topics_over_time_non_dynamic_no_ci():
    """TopicsOverTime for a plain LDA has no CI without nsims."""
    pytest.importorskip("matplotlib")
    import topica.viz as viz

    rng = np.random.default_rng(0)
    docs = [list(rng.choice(["a", "b", "c", "d"], size=8)) for _ in range(60)]
    years = [2000 + i % 3 for i in range(60)]
    m = topica.models.LDA(2, seed=1)
    m.fit(docs, iters=100)
    panel = viz.topics_over_time(m, years)
    assert panel.has_ci is False
