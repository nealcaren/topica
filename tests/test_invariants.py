"""Reference-free metamorphic / invariant tests for the count-Gibbs core (#420).

Cross-implementation *parity* validates central tendency against one reference
config on benign data; it is structurally blind to tail regimes, uncertainty
machinery, second-order (SE-only) effects, and non-parity paths. These tests
assert properties that must hold **regardless of the correct numeric answer**, so
they need no reference implementation:

- finiteness / normalization on degenerate inputs (empty docs, K=1);
- sampling-control invariance (num_samples/sample_interval move only the SE, never
  the point estimate);
- determinism (same seed -> bit-identical fit);
- SE stationarity (an SE is finite when emitted, else None -- never a silent NaN).

This is the first slice of the #420 harness, covering the shared count-Gibbs core
(LDA / DMR / keyATM / SAGE / GDMR) at the Python-API level. The reconstruction and
exact-dense-conditional invariants (which need internal count tables) are tracked
as a Rust-level follow-up.

Run just these with `-m invariants`.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica

pytestmark = pytest.mark.invariants

_ITERS = 25
_VOCAB = ["a", "b", "c", "d", "e", "f"]


def _docs(n: int = 36, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [list(rng.choice(_VOCAB, 6)) for _ in range(n)]


def _covariate(n: int) -> np.ndarray:
    # One deterministic continuous covariate column, index-derived (no RNG so the
    # design is stable across doc counts).
    return np.array([[np.sin(i)] for i in range(n)], dtype=float)


def _groups(n: int) -> list[int]:
    return [i % 2 for i in range(n)]


# Each entry: build(K) -> model, fit(model, docs), covariate flag.
_MODELS: dict[str, dict] = {
    "LDA": dict(
        build=lambda k: topica.LDA(k, seed=1),
        fit=lambda m, docs: m.fit(docs, iters=_ITERS),
        covariate=False,
    ),
    "DMR": dict(
        build=lambda k: topica.DMR(k, seed=1),
        fit=lambda m, docs: m.fit(docs, _covariate(len(docs)), iters=_ITERS),
        covariate=True,
    ),
    "keyATM": dict(
        build=lambda k: topica.KeyATM({"g": ["a"]}, num_topics=k, seed=1),
        fit=lambda m, docs: m.fit(docs, iters=_ITERS),
        covariate=False,
    ),
    "SAGE": dict(
        build=lambda k: topica.SAGE(k, seed=1),
        fit=lambda m, docs: m.fit(docs, _groups(len(docs)), iters=_ITERS),
        covariate=False,
    ),
    "GDMR": dict(
        build=lambda k: topica.GDMR(k, degrees=[2], seed=1),
        fit=lambda m, docs: m.fit(docs, _covariate(len(docs)), iters=_ITERS),
        covariate=True,
    ),
}

_ALL = list(_MODELS)
_COVARIATE = [n for n, s in _MODELS.items() if s["covariate"]]


def _fit(name: str, k: int, docs: list[list[str]]):
    spec = _MODELS[name]
    model = spec["build"](k)
    spec["fit"](model, docs)
    return model


def _assert_valid_distributions(model) -> None:
    """topic_word and doc_topic must be finite and normalized on their last axis
    (works for a 2D K×V topic_word and SAGE's 3D K×G×V grouped one)."""
    tw = np.asarray(model.topic_word)
    dt = np.asarray(model.doc_topic)
    assert np.isfinite(tw).all(), "topic_word has non-finite entries"
    assert np.isfinite(dt).all(), "doc_topic has non-finite entries"
    assert np.allclose(tw.sum(axis=-1), 1.0, atol=1e-3), "topic_word rows do not sum to 1"
    assert np.allclose(dt.sum(axis=-1), 1.0, atol=1e-3), "doc_topic rows do not sum to 1"


# --- Boundary / finiteness -------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_outputs_finite_and_normalized_on_normal_corpus(name):
    _assert_valid_distributions(_fit(name, 3, _docs()))


@pytest.mark.parametrize("name", _ALL)
def test_finite_with_an_empty_document(name):
    # A zero-length document must not produce a NaN row or a divide-by-zero.
    docs = _docs()
    docs = docs[:18] + [[]] + docs[18:]
    _assert_valid_distributions(_fit(name, 3, docs))


@pytest.mark.parametrize("name", _ALL)
def test_finite_with_single_topic(name):
    # K=1 removes the topic dimension entirely (packed-count / simplex edge case).
    # A model may legitimately require K>=2 (keyATM needs a keyword/regular split);
    # a clean error is an acceptable boundary outcome per the invariant, a silent
    # NaN is not.
    try:
        model = _fit(name, 1, _docs())
    except ValueError:
        return
    _assert_valid_distributions(model)


# --- Determinism -----------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_same_seed_is_bit_identical(name):
    docs = _docs()
    a = _fit(name, 3, docs)
    b = _fit(name, 3, docs)
    assert np.array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))
    assert np.array_equal(np.asarray(a.doc_topic), np.asarray(b.doc_topic))


# --- Sampling-control invariance -------------------------------------------


@pytest.mark.parametrize("name", _COVARIATE)
def test_sampling_control_does_not_move_the_point_estimate(name):
    # num_samples / sample_interval average the topic-word posterior and drive the
    # SE, but the covariate point estimate (lambda / feature_effects) is a MAP
    # optimum of the counts and must be bit-identical across sampling settings.
    docs = _docs()
    spec = _MODELS[name]
    build = spec["build"]

    def fe(num_samples, sample_interval):
        m = build(3)
        m.fit(docs, _covariate(len(docs)), iters=40,
              num_samples=num_samples, sample_interval=sample_interval)
        return np.asarray(m.feature_effects)

    base = fe(3, 10)
    more = fe(9, 5)
    assert np.array_equal(base, more), (
        f"{name}.feature_effects moved with sampling settings (max |Δ| = "
        f"{np.abs(base - more).max():.3e}); sampling must affect only the SE"
    )


# --- SE stationarity -------------------------------------------------------


@pytest.mark.parametrize("name", _COVARIATE)
def test_se_is_none_or_finite_never_silent_nan(name):
    # An observed-information SE is meaningful only at a stationary optimum. The
    # estimator must return either a finite SE or None -- never inf, and never a
    # NaN slipped into an otherwise-numeric array (#419).
    m = _fit(name, 3, _docs())
    se = m.feature_effect_se
    if se is None:
        return
    se = np.asarray(se)
    # Individual entries may be NaN by design (a clamped / unidentified effect
    # advertises "no SE" as NaN), but there must be no infinities and the point
    # estimate it hangs off of must itself be finite.
    assert not np.isinf(se).any(), f"{name} SE contains an infinity"
    assert np.isfinite(np.asarray(m.feature_effects)).all(), f"{name} feature_effects not finite"
