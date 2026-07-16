"""keyATM convergence trace (``log_likelihood_history``) — the per-iteration
per-token log-likelihood keyATM's ``plot_modelfit`` / ``model_fit`` reports. This
is the Gibbs analog of STM's variational-bound trajectory: it lets users judge
whether the sampler has mixed, instead of trusting a fixed iteration count."""

import os
import tempfile

import numpy as np
import pytest

import topica

A = ["tax", "market", "trade", "fiscal"]
B = ["abortion", "gay", "church", "family"]
SEEDS = {"econ": A[:2], "soc": B[:2]}


def _corpus(seed=0, n=200):
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for i in range(n):
        lab = i % 2
        heavy, light = (A, B) if lab else (B, A)
        docs.append(rng.choice(heavy, 8).tolist() + rng.choice(light, 2).tolist())
        labels.append(float(lab))
    return docs, np.array(labels)


def test_trace_recorded_auto_spacing():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=200)
    h = m.log_likelihood_history
    assert len(h) == 50  # auto: ~50 evenly spaced points
    iters = [it for it, _, _ in h]
    assert iters == sorted(iters)
    assert iters[-1] == 200  # last sweep always recorded
    # (iteration, collapsed log-likelihood < 0, perplexity > 1) — keyATM model_fit.
    assert all(np.isfinite(ll) and ll < 0 and ppl > 1 for _, ll, ppl in h)


def test_trace_explicit_interval():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=100, progress_interval=25)
    assert [it for it, _, _ in m.log_likelihood_history] == [25, 50, 75, 100]


def test_trace_rises_from_random_start():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=200, progress_interval=5)
    lls = [ll for _, ll, _ in m.log_likelihood_history]
    # The fit should improve overall as Gibbs moves away from the random start.
    assert lls[-1] > lls[0]


def test_trace_for_covariate_and_dynamic():
    docs, labels = _corpus()
    cov = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    cov.fit(docs, iters=150, covariates=labels.reshape(-1, 1), feature_names=["x"])
    assert len(cov.log_likelihood_history) > 0
    assert cov.log_likelihood_history[-1][0] == 150

    dyn = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    dyn.fit(docs, iters=120, timestamps=labels.astype(int), num_states=2)
    assert len(dyn.log_likelihood_history) > 0
    assert all(np.isfinite(ll) for _, ll, _ in dyn.log_likelihood_history)


def test_alpha_and_pi_traces():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=4, seed=1)
    m.fit(docs, iters=200, progress_interval=20)
    # plot_alpha: base model estimates an asymmetric alpha that moves over sweeps.
    assert len(m.alpha_history) == 10
    it, alpha = m.alpha_history[-1]
    assert it == 200 and len(alpha) == 4 and all(a > 0 for a in alpha)
    assert not np.allclose(m.alpha_history[0][1], m.alpha_history[-1][1])
    # plot_pi: keyword topics carry a nonzero switch rate, regular topics 0.
    it, pi = m.pi_history[-1]
    assert it == 200 and len(pi) == 4
    assert pi[0] > 0 and pi[1] > 0 and pi[2] == 0 and pi[3] == 0


def test_weighted_lda_is_keyword_free():
    docs, _ = _corpus()
    w = topica.models.KeyATM.weighted_lda(num_topics=3, seed=1)
    w.fit(docs, iters=150, weights="information-theory")
    assert w.topic_word.shape[0] == 3
    assert np.allclose(w.keyword_rate, 0.0)  # no keyword topics
    assert len(w.pi_history) == 0  # plot_pi not applicable
    assert len(w.alpha_history) > 0  # alpha still estimated
    assert len(w.log_likelihood_history) > 0


def test_weighted_lda_validates_num_topics():
    with pytest.raises(ValueError):
        topica.models.KeyATM.weighted_lda(num_topics=1)


def test_trace_survives_save_load():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=100)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.bin")
        m.save(path)
        reloaded = topica.models.KeyATM.load(path)
    assert reloaded.log_likelihood_history == m.log_likelihood_history


def test_history_requires_fit():
    m = topica.models.KeyATM(SEEDS, num_topics=2)
    with pytest.raises(RuntimeError):
        _ = m.log_likelihood_history


def test_estimate_alpha_false_keeps_symmetric_prior():
    # estimate_alpha=False skips the per-sweep alpha slice sampler (the speed
    # knob); alpha stays symmetric, so the no-alpha trace is not recorded.
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=4, seed=1, estimate_alpha=False)
    m.fit(docs, iters=150)
    # With a fixed symmetric prior there is no learned asymmetric alpha to trace.
    assert m.alpha_history == []
    # The fit still produces well-formed topics and pi/log-likelihood traces.
    assert m.topic_word.shape[0] == 4
    assert len(m.log_likelihood_history) > 0
    assert len(m.pi_history) > 0


def test_estimate_alpha_default_is_true():
    docs, _ = _corpus()
    m = topica.models.KeyATM(SEEDS, num_topics=4, seed=1)  # default
    m.fit(docs, iters=150)
    # The default learns an asymmetric alpha, so the alpha trace is populated.
    assert len(m.alpha_history) > 0


def test_estimate_alpha_deterministic():
    docs, _ = _corpus()
    a = topica.models.KeyATM(SEEDS, num_topics=4, seed=2, estimate_alpha=False)
    a.fit(docs, iters=100)
    b = topica.models.KeyATM(SEEDS, num_topics=4, seed=2, estimate_alpha=False)
    b.fit(docs, iters=100)
    assert np.array_equal(a.topic_word, b.topic_word)
