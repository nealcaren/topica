"""Regression tests for the silent-degradation / API-drift batch (issues #24–#27).

Each of these was a result quietly less trustworthy than the API implied — a
double-counted prior, a misaligned document returned without error, a swallowed
exception, or a type stub / docstring that disagreed with runtime. The shared
theme: failures should be loud (raise or warn), never silent.
"""

import warnings

import numpy as np
import pytest

import topica
from topica import effects, validation
from topica.viz.capability import capabilities


def _toy_corpus(n=40, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(14)]
    docs = [list(rng.choice(vocab, size=12)) for _ in range(n)]
    return docs, topica.Corpus.from_documents(docs)


# --- #26: dirichlet_theta_samples double-counted the prior on the prior>0 path ---

def test_dirichlet_theta_samples_prior_not_double_counted():
    D, K = 1500, 10
    theta = np.full((D, K), 0.2 / (K - 1))
    theta[:, 0] = 0.8
    lengths = np.full(D, 10.0)
    draws = effects.dirichlet_theta_samples(theta, lengths, nsims=200, seed=1, prior=1.0)
    # The posterior mean for topic 0 should recover 0.8; the old +prior/K addend
    # biased it toward uniform (~0.74).
    assert abs(draws.mean(axis=(0, 1))[0] - 0.8) < 0.02


def test_dirichlet_theta_samples_default_prior_unchanged():
    theta = np.array([[0.7, 0.3], [0.4, 0.6]])
    lengths = np.array([10.0, 10.0])
    draws = effects.dirichlet_theta_samples(theta, lengths, nsims=300, seed=2)
    assert np.allclose(draws.mean(axis=0), theta, atol=0.05)


# --- #24: display functions must guard against misaligned texts/groups ----------

def _pruned_setup():
    docs = [["the", "cat", "sat", "on", "mat"], ["quuxfrobnitz"],
            ["the", "dog", "ate", "the", "cat"], ["cat", "mat", "dog", "sat"]]
    texts = ["Text0", "DROPPED1", "Text2", "Text3"]
    corpus = topica.Corpus.from_documents(docs, min_doc_freq=2)  # drops doc 1
    m = topica.LDA(num_topics=2, seed=42)
    m.fit(corpus, iters=100)
    return m, corpus, texts


def test_find_thoughts_rejects_misaligned_texts():
    m, _, texts = _pruned_setup()
    with pytest.raises(ValueError, match="rows"):
        topica.find_thoughts(m.doc_topic, texts, topic=0, n=2)


def test_document_intrusion_rejects_misaligned_texts():
    m, _, texts = _pruned_setup()
    with pytest.raises(ValueError, match="rows"):
        topica.document_intrusion(m.doc_topic, texts, n_docs=1)


def test_find_thoughts_accepts_aligned_texts():
    m, corpus, texts = _pruned_setup()
    aligned = [texts[i] for i in corpus.kept_indices]
    out = topica.find_thoughts(m.doc_topic, aligned, topic=0, n=2)
    assert all(t in aligned for _, _, t in out)


def test_plot_report_warns_on_misaligned_groups():
    pytest.importorskip("matplotlib")
    m, _, _ = _pruned_setup()
    with pytest.warns(UserWarning, match="class.*rows|rows"):
        fig = topica.plot_report(m, groups=["a", "b", "a", "b"])  # 4 vs 3 kept docs
    # The misaligned panel is dropped, not silently drawn from the wrong rows.
    assert fig is not None


# --- #25: swallowed exceptions / ignored params ---------------------------------

def test_bootstrap_refit_hook_typeerror_is_not_swallowed():
    """A TypeError raised inside a 2-arg refit hook must propagate, not be
    misread as an arity mismatch and silently retried at the default seed."""
    _, corpus = _toy_corpus()
    model = topica.LDA(num_topics=2, seed=1)
    model.fit(corpus, iters=80)

    def broken_refit(picks, seed):
        raise TypeError("genuine bug inside the hook")

    with pytest.raises(TypeError, match="genuine bug"):
        topica.standard_errors(
            model, corpus, of="top_words", method="bootstrap",
            n_boot=2, refit=broken_refit,
        )


def test_quality_frontier_warns_when_coherence_type_needs_texts():
    _, corpus = _toy_corpus()
    m = topica.LDA(num_topics=3, seed=1)
    m.fit(corpus, iters=80)
    with pytest.warns(UserWarning, match="needs texts"):
        validation.quality_frontier(m, coherence_type="c_v")  # no texts


# --- #27: API surface drift -----------------------------------------------------

def test_coherence_works_for_sage_via_marginal():
    """SAGE's top_words takes `topic` first and its topic_word is 3-D; coherence
    must still work by falling back to the group-marginal matrix."""
    docs, _ = _toy_corpus()
    s = topica.SAGE(num_topics=3, seed=1, optimize_interval=25, burn_in=20)
    s.fit(docs, ["g"] * len(docs), iters=80, num_samples=2, sample_interval=10)
    coh = topica.coherence(s, docs, coherence_type="u_mass")
    assert coh.shape == (3,) and np.all(np.isfinite(coh))
    assert topica.exclusivity(s).shape == (3,)


def test_coherence_rejects_dtm_with_clear_message():
    """DTM's topic_word is time-sliced; the static surface must say so, not coerce
    a bound method into an object array."""
    docs, _ = _toy_corpus()
    dtm = topica.DTM(num_topics=2, seed=1)
    dtm.fit(docs, [0] * 20 + [1] * 20, iters=3)
    with pytest.raises(ValueError, match="time-sliced"):
        topica.coherence(dtm, docs)


def test_capability_no_doc_topic_models_are_not_soft_theta():
    docs, _ = _toy_corpus()
    dtm = topica.DTM(num_topics=2, seed=1)
    dtm.fit(docs, [0] * 20 + [1] * 20, iters=3)
    h = topica.HLDA(depth=2, seed=1)
    h.fit(docs, iters=50)
    assert capabilities(dtm).soft_theta is False
    assert capabilities(h).soft_theta is False


def test_dmr_stub_matches_runtime():
    """The DMR type stub was copied from STM. Guard the runtime contract the
    corrected stub now documents: fit(data, features, ...) and feature_effects."""
    docs, _ = _toy_corpus()
    X = np.random.default_rng(0).normal(size=(len(docs), 1))
    m = topica.DMR(num_topics=3, seed=1, optimize_interval=25, burn_in=20)
    m.fit(docs, X, feature_names=["x"], iters=80, num_samples=2, sample_interval=10)
    assert np.asarray(m.feature_effects).shape[0] == 3
    assert not hasattr(m, "prevalence_effects")


# --- Gate B cross-cutting footguns (#713/#714/#715/#717) -----------------------

def test_composition_theta_is_top_level_exported():
    """#713-#1: posterior_theta_samples' error message steers users to
    topica.composition_theta, so it must exist at the top level (not only under
    topica.effects), and be the same callable."""
    assert hasattr(topica, "composition_theta")
    assert topica.composition_theta is effects.composition_theta


def test_estimate_effect_warns_on_rank_deficient_design():
    """#713-#2/#714/#715-#6: a full dummy set plus the default add_intercept=True
    is a collinear double-intercept design. The pseudoinverse solves it silently
    with an arbitrary split; estimate_effect must warn that the coefficients are
    not identified."""
    rng = np.random.default_rng(0)
    n, k = 40, 3
    theta = rng.dirichlet(np.ones(k), size=n)
    g = np.array([0, 1] * (n // 2))
    X_full = np.column_stack([g == 0, g == 1]).astype(float)  # collinear with intercept
    with pytest.warns(UserWarning, match="rank-deficient"):
        effects.estimate_effect(theta, X=X_full, feature_names=["lvl0", "lvl1"])
    # A full-rank design (drop_first) must not warn.
    X_ok = (g == 1).astype(float)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        effects.estimate_effect(theta, X=X_ok, feature_names=["lvl1"])


def test_best_k_frontier_warns_at_grid_boundary():
    """#715-#5: the default frontier best_k returned a grid-edge K silently, so
    the pick flips with grid resolution. It must warn when the knee is at the
    smallest or largest scanned K, and stay quiet on an interior optimum."""
    boundary = validation.SearchKResult([
        {"k": 3, "coherence": -70.0, "exclusivity": 0.50},
        {"k": 5, "coherence": -68.0, "exclusivity": 0.55},
        {"k": 7, "coherence": -66.0, "exclusivity": 0.60},
        {"k": 10, "coherence": -60.0, "exclusivity": 0.95},  # best on both -> boundary
    ])
    with pytest.warns(UserWarning, match="boundary"):
        assert boundary.best_k("frontier") == 10
    interior = validation.SearchKResult([
        {"k": 3, "coherence": -70.0, "exclusivity": 0.50},
        {"k": 5, "coherence": -60.0, "exclusivity": 0.95},  # best interior
        {"k": 7, "coherence": -66.0, "exclusivity": 0.60},
        {"k": 10, "coherence": -69.0, "exclusivity": 0.52},
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert interior.best_k("frontier") == 5
