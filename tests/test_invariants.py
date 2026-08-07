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
- SE stationarity (when lambda is optimized the SE is emitted and free of any
  infinity; an individual entry may be an advertised NaN for a clamped effect).

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
        # optimize_interval/burn_in are set below _ITERS so lambda actually
        # optimizes within the short fits these invariants use -- otherwise
        # feature_effects stays at its zero init and feature_effect_se is None,
        # which would make the covariate invariants vacuous.
        build=lambda k: topica.DMR(k, seed=1, optimize_interval=10, burn_in=10),
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
        build=lambda k: topica.GDMR(
            k, degrees=[2], seed=1, optimize_interval=10, burn_in=10
        ),
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
    # Guard against a vacuous pass: if lambda never optimized (burn_in >= iters),
    # feature_effects would be an all-zero init and the equality below would hold
    # trivially. The build sets optimize_interval/burn_in below iters, so lambda
    # is a real MAP optimum here -- assert it actually moved off the init.
    assert np.any(base != 0.0), (
        f"{name}.feature_effects is all-zero -- lambda never optimized, so this "
        f"invariant is vacuous (raise iters or lower burn_in)"
    )
    assert np.array_equal(base, more), (
        f"{name}.feature_effects moved with sampling settings (max |Δ| = "
        f"{np.abs(base - more).max():.3e}); sampling must affect only the SE"
    )


# --- SE stationarity -------------------------------------------------------


@pytest.mark.parametrize("name", _COVARIATE)
def test_se_emitted_at_the_optimum_has_no_infinity(name):
    # An observed-information SE is meaningful only at a stationary optimum. The
    # build sets optimize_interval/burn_in below iters, so lambda IS optimized
    # here and the estimator must actually emit an SE (not None -- otherwise this
    # invariant never exercises the #419 code path). When emitted, it may carry a
    # NaN in an individual entry by design -- a clamped / unidentified effect
    # advertises "no SE for this cell" as NaN -- but it must never contain an
    # infinity, and the point estimate it hangs off of must itself be finite.
    m = _fit(name, 3, _docs())
    se = m.feature_effect_se
    assert se is not None, (
        f"{name}.feature_effect_se is None even though lambda was optimized "
        f"(optimize_interval/burn_in < iters); the SE path is untested"
    )
    se = np.asarray(se)
    assert not np.isinf(se).any(), f"{name} SE contains an infinity"
    assert np.isfinite(np.asarray(m.feature_effects)).all(), f"{name} feature_effects not finite"


# --- Extreme-covariate tail (the unbounded exp) ----------------------------


def _covariate_extreme(n: int, scale: float) -> np.ndarray:
    """One continuous covariate column at a large magnitude, to stress the DMR/GDMR
    prior's ``exp(x . lambda)`` toward overflow (#419)."""
    return np.array([[scale * np.sin(i)] for i in range(n)], dtype=float)


@pytest.mark.parametrize("name", _COVARIATE)
@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_extreme_covariate_scale_stays_finite(name, scale):
    # A large-magnitude covariate pushes exp(x . lambda) toward inf; the prior must
    # stay finite and produce valid topics (or reject the design cleanly), never a
    # silent NaN. Parity fixtures are curated and never hit this regime (#419).
    docs = _docs()
    model = _MODELS[name]["build"](3)
    try:
        model.fit(docs, _covariate_extreme(len(docs), scale), iters=_ITERS)
    except (ValueError, OverflowError):
        return  # a clean rejection of a pathological design satisfies the invariant
    _assert_valid_distributions(model)
    assert np.isfinite(np.asarray(model.feature_effects)).all(), (
        f"{name} feature_effects went non-finite at covariate scale {scale:g}"
    )
    se = model.feature_effect_se
    if se is not None:
        assert not np.isinf(np.asarray(se)).any(), (
            f"{name} SE contains an infinity at covariate scale {scale:g}"
        )


# --- Multithread (approximate parallel) path -------------------------------


def _build_threaded(name: str, k: int, num_threads: int):
    """Rebuild a model like its registry entry but with ``num_threads`` set. Raises
    TypeError if the constructor has no such argument (the caller skips)."""
    builders = {
        "LDA": lambda: topica.LDA(k, seed=1, num_threads=num_threads),
        "DMR": lambda: topica.DMR(k, seed=1, optimize_interval=10, burn_in=10,
                                  num_threads=num_threads),
        "keyATM": lambda: topica.KeyATM({"g": ["a"]}, num_topics=k, seed=1,
                                        num_threads=num_threads),
        "GDMR": lambda: topica.GDMR(k, degrees=[2], seed=1, optimize_interval=10,
                                    burn_in=10, num_threads=num_threads),
        "SAGE": lambda: topica.SAGE(k, seed=1, num_threads=num_threads),
    }
    return builders[name]()


@pytest.mark.parametrize("name", _ALL)
def test_multithread_path_stays_valid(name):
    # The approximate parallel (AD-LDA-style) sampler is a non-parity path: parity
    # always runs single-thread. Its count-table merge must still produce finite,
    # normalized, non-collapsed topics -- a merge bug shows up as NaN or a degenerate
    # single-topic result, not as a parity drift.
    docs = _docs(48)
    try:
        model = _build_threaded(name, 3, num_threads=2)
    except TypeError:
        pytest.skip(f"{name} takes no num_threads")
    _MODELS[name]["fit"](model, docs)
    _assert_valid_distributions(model)
    # not collapsed to a single topic (a classic parallel-merge failure mode):
    # more than one topic carries appreciable mean mass across documents.
    mean_theta = np.asarray(model.doc_topic).mean(axis=0)
    assert int((mean_theta > 0.01).sum()) > 1, (
        f"{name} multithread fit collapsed to ~1 used topic"
    )
