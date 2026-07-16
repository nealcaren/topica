"""keyATM covariate model: the document-topic prior is a DMR on covariates,
alpha_{d,k} = exp(x_d . lambda_k), matching the keyATM R package. These check
that a planted covariate effect on topic prevalence is recovered, that the base
model is unaffected, and the output API."""

import numpy as np
import pytest

import topica

ECON = ["tax", "market", "trade", "fiscal", "budget", "deficit"]
SOC = ["abortion", "gay", "marriage", "church", "family", "prayer"]
SEEDS = {"economic": ECON[:4], "social": SOC[:4]}


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    docs, party = [], []
    for i in range(300):
        is_d = i % 2 == 0
        heavy, light = (SOC, ECON) if is_d else (ECON, SOC)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
        party.append(1.0 if is_d else 0.0)
    return docs, np.array(party)


@pytest.fixture(scope="module")
def cov_model():
    docs, party = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, covariates=party.reshape(-1, 1), feature_names=["is_D"], iters=600)
    return m, docs, party


def test_feature_effects_shape_and_names(cov_model):
    m, _, _ = cov_model
    assert m.feature_names == ["intercept", "is_D"]
    assert m.feature_effects.shape == (2, 2)  # (K, intercept + 1 covariate)


def test_recovers_covariate_effect_on_prevalence(cov_model):
    m, _, _ = cov_model
    si, ei = m.topic_names.index("social"), m.topic_names.index("economic")
    # Being a Democrat raises the social topic relative to the economic topic.
    assert m.feature_effects[si, 1] - m.feature_effects[ei, 1] > 1.0


def test_feature_effect_se_shape_and_finite(cov_model):
    m, _, _ = cov_model
    se = m.feature_effect_se
    assert se.shape == m.feature_effects.shape == (2, 2)
    # No clamped coefficients in this well-identified design, so all finite & > 0.
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)


def test_strong_effect_is_significant(cov_model):
    """The planted party->social effect is large, so it sits many SEs from zero."""
    m, _, _ = cov_model
    si = m.topic_names.index("social")
    z = m.feature_effects[si, 1] / m.feature_effect_se[si, 1]
    assert abs(z) > 2.0, f"z={z:.2f} for a planted effect"


def test_feature_effect_se_base_model_raises():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=200)
    with pytest.raises(RuntimeError):
        _ = m.feature_effect_se


def test_theta_tracks_covariate(cov_model):
    m, _, party = cov_model
    th = m.doc_topic
    si = m.topic_names.index("social")
    assert th[party == 1, si].mean() > th[party == 0, si].mean() + 0.2


def test_deterministic(cov_model):
    m, docs, party = cov_model
    m2 = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m2.fit(docs, covariates=party.reshape(-1, 1), feature_names=["is_D"], iters=600)
    assert np.allclose(m.feature_effects, m2.feature_effects)


def test_base_model_has_no_feature_effects():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=200)
    assert m.feature_names == []
    with pytest.raises(RuntimeError):
        _ = m.feature_effects


def test_covariate_row_count_validated():
    docs, party = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, covariates=party[:-1].reshape(-1, 1), iters=10)


# ---------------------------------------------------------------------------
# Issue #270 regression: a high-dimensional covariate design (many one-hot
# levels, e.g. ~C(year)) must not drive the covariate prior to a degenerate
# theta on a single topic. The guards (covariate standardization + lambda
# bounded to +/-5 under the N(0,1) prior, matching R keyATM) prevent the
# runaway. See also issue #271 (hardening the suite with validity invariants).
# ---------------------------------------------------------------------------

def _effective_topics(theta):
    """exp(H(mean theta)): how many topics carry real mass (1.0 == collapsed)."""
    mean_t = np.asarray(theta).mean(axis=0)
    mean_t = mean_t / mean_t.sum()
    return float(np.exp(-(mean_t * np.log(mean_t + 1e-12)).sum()))


def _onehot_corpus(n, k, levels, vocab=120, topic_alpha=0.5, seed=0):
    """Overlapping-topic corpus with a high-dim one-hot covariate, each level
    favouring one topic (a planted, recoverable covariate effect)."""
    rng = np.random.default_rng(seed)
    topics = rng.dirichlet(np.full(vocab, topic_alpha), size=k)
    level_pref = np.array([g % k for g in range(levels)])
    docs, lev = [], rng.integers(0, levels, size=n)
    for d in range(n):
        base = np.full(k, 0.3)
        base[level_pref[lev[d]]] += 2.0
        theta = rng.dirichlet(base)
        toks = [f"w{int(rng.choice(vocab, p=topics[rng.choice(k, p=theta)]))}"
                for _ in range(rng.integers(35, 55))]
        docs.append(toks)
    keywords = {f"kt{j}": [f"w{int(w)}" for w in np.argsort(topics[j])[::-1][:5]]
                for j in range(min(4, k))}
    X = np.zeros((n, levels))
    X[np.arange(n), lev] = 1.0
    return docs, X, keywords, level_pref


def test_high_dim_covariate_fit_is_healthy_and_recovers_effect():
    """#270/#271: covariate keyATM on a 15-level one-hot design stays healthy
    (no collapse), keyword_rate leaves its 0.5 init, lambda is finite, and the
    planted level->topic effect is recovered directionally."""
    k, levels = 8, 15
    docs, X, keywords, level_pref = _onehot_corpus(3000, k, levels, seed=1)
    m = topica.models.KeyATM(keywords, num_topics=k, seed=1, num_threads=2)
    m.fit(docs, covariates=X, iters=400)

    theta = m.doc_topic
    # Invariants (the #271 high-leverage assertions): not degenerate.
    mean_mass = np.asarray(theta).mean(axis=0)
    mean_mass = mean_mass / mean_mass.sum()
    assert mean_mass.max() < 0.6, f"one topic dominates: {mean_mass.max():.3f}"
    assert _effective_topics(theta) > k * 0.5, "theta collapsed (too few effective topics)"

    fe = np.asarray(m.feature_effects)
    assert np.isfinite(fe).all(), "feature_effects not finite"

    kr = np.asarray(m.keyword_rate)[:4]  # keyword topics
    assert np.all((kr > 0.0) & (kr < 1.0)), f"keyword_rate out of (0,1): {kr}"
    assert np.any(np.abs(kr - 0.5) > 1e-3), "keyword_rate stuck at 0.5 init"

    # Directional recovery: a level's planted topic gets a higher coefficient on
    # that level's column than the average topic does.
    planted = np.array([fe[level_pref[g], g + 1] for g in range(levels)])
    assert planted.mean() > fe[:, 1:].mean(), "planted covariate effect not recovered"


@pytest.mark.slow
def test_covariate_collapse_regression_at_scale():
    """#270: at scale (the regime that exposed the bug) the covariate fit must
    not collapse onto one topic. Opt-in (slow); set TOPICA_SLOW_TESTS=1."""
    import os

    if not os.environ.get("TOPICA_SLOW_TESTS"):
        pytest.skip("slow scale test; set TOPICA_SLOW_TESTS=1 to run")
    k, levels = 31, 30
    docs, X, keywords, _ = _onehot_corpus(50000, k, levels, topic_alpha=0.7, seed=2)
    m = topica.models.KeyATM(keywords, num_topics=k, seed=1, num_threads=8)
    m.fit(docs, covariates=X, iters=500)
    eff = _effective_topics(m.doc_topic)
    assert eff > k * 0.4, f"covariate theta collapsed at scale: eff#topics={eff:.1f}/{k}"
