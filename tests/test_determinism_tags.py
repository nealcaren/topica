"""Enforce each model's registry determinism tag against its actual behavior.

The contract (``topica.registry``):

- ``bit-exact``        -> output is identical regardless of seed (deterministic
                          initialization: CTM/STM/STS spectral anchor words,
                          NMF/LSA via SVD).
- ``seed-reproducible``-> output is identical for a fixed seed and thread count,
                          run twice, but may differ across seeds (the model
                          reaches the RNG at initialization).

This test exists because issue #216 was a determinism-claim violation: five
models (DTM, ETM, FASTopic, CombinedTM, ZeroShotTM) were tagged ``bit-exact``
but reach the RNG only at init, so they are really ``seed-reproducible`` — which
also matches their reference implementations (all random-init). The test reads
the *current* registry tag and asserts the matching behavior, so re-mislabeling
any model flips this red.

Only models with a builder below are checked; extend ``BUILDERS`` to cover more.
"""

import numpy as np
import pytest

import topica

# ---------------------------------------------------------------------------
# Shared tiny inputs
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_VOCAB = [f"w{i}" for i in range(40)]
_V = len(_VOCAB)
_DOCS = [[_VOCAB[int(_RNG.integers(0, _V))] for _ in range(30)] for _ in range(60)]
_WORD_EMB = _RNG.standard_normal((_V, 16)).astype(np.float64)
_DOC_EMB = _RNG.standard_normal((len(_DOCS), 16)).astype(np.float64)
_G = np.asarray([_RNG.integers(0, 2) for _ in range(len(_DOCS))], dtype=float)
_PREV = np.column_stack([np.ones(len(_DOCS)), _G])

_K = 4
_ITERS = 15


def _signature(model) -> np.ndarray:
    """A topic-word array we can compare for equality across two fits."""
    tw = model.topic_word
    if callable(tw):  # DTM exposes topic_word(t) per time slice
        tw = tw(0)
    return np.asarray(tw)


# Each builder fits a model from scratch at the given seed and returns its
# topic-word signature. Keep the inputs identical across calls so the only
# difference is the seed.

def _ctm(seed):
    m = topica.CTM(num_topics=_K, seed=seed); m.fit(_DOCS, iters=_ITERS); return m

def _nmf(seed):
    m = topica.NMF(num_topics=_K, seed=seed); m.fit(_DOCS, iters=_ITERS); return m

def _lsa(seed):
    m = topica.LSA(num_topics=_K, seed=seed); m.fit(_DOCS); return m

def _stm(seed):
    m = topica.STM(_K, seed=seed, init="spectral"); m.fit(_DOCS, _PREV, iters=_ITERS); return m

def _dtm(seed):
    times = [int(_RNG.integers(0, 3)) for _ in range(len(_DOCS))]
    # times must be identical across calls -> derive deterministically
    times = [i % 3 for i in range(len(_DOCS))]
    m = topica.DTM(num_topics=_K, seed=seed); m.fit(_DOCS, times, iters=10); return m

def _etm(seed):
    m = topica.ETM(num_topics=_K, seed=seed); m.fit(_DOCS, _WORD_EMB, _VOCAB, iters=_ITERS); return m

def _fastopic(seed):
    m = topica.FASTopic(num_topics=_K, seed=seed); m.fit(_DOCS, _DOC_EMB, iters=_ITERS); return m

def _prodlda(seed):
    m = topica.ProdLDA(num_topics=_K, seed=seed); m.fit(_DOCS, iters=_ITERS); return m

def _combinedtm(seed):
    m = topica.CombinedTM(num_topics=_K, seed=seed); m.fit(_DOCS, _DOC_EMB, iters=_ITERS); return m

def _zeroshottm(seed):
    m = topica.ZeroShotTM(num_topics=_K, seed=seed); m.fit(_DOCS, _DOC_EMB, iters=_ITERS); return m

def _tensorlda(seed):
    topica.enable_experimental(True)
    m = topica.TensorLDA(num_topics=_K, seed=seed); m.fit(_DOCS, iters=_ITERS); return m


BUILDERS = {
    "CTM": _ctm,
    "NMF": _nmf,
    "LSA": _lsa,
    "STM": _stm,
    "DTM": _dtm,
    "ETM": _etm,
    "FASTopic": _fastopic,
    "ProdLDA": _prodlda,
    "CombinedTM": _combinedtm,
    "ZeroShotTM": _zeroshottm,
    "TensorLDA": _tensorlda,
}


def _tag(name: str) -> str:
    (meta,) = [m for m in topica.list_models() if m.name == name]
    return meta.determinism


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_determinism_matches_registry_tag(name):
    """Behavior must match the model's registry determinism tag."""
    tag = _tag(name)
    build = BUILDERS[name]

    if tag == "bit-exact":
        # Identical regardless of seed.
        a = _signature(build(1))
        b = _signature(build(2))
        assert np.array_equal(a, b), (
            f"{name} is tagged bit-exact but two different seeds produced "
            f"different topic_word matrices -- it is at most seed-reproducible."
        )
    elif tag == "seed-reproducible":
        # Identical for a fixed seed, run twice.
        a = _signature(build(7))
        b = _signature(build(7))
        assert np.array_equal(a, b), (
            f"{name} is tagged seed-reproducible but two runs at the same seed "
            f"produced different topic_word matrices (broken determinism)."
        )
    else:
        pytest.skip(f"{name} tag {tag!r} not exercised here")


@pytest.mark.parametrize(
    "name",
    sorted(n for n in BUILDERS if _tag(n) == "seed-reproducible"),
)
def test_seed_reproducible_models_actually_use_the_seed(name):
    """A seed-reproducible model should respond to the seed (else it would be
    bit-exact). This guards against silently dropping the seed."""
    a = _signature(BUILDERS[name](1))
    b = _signature(BUILDERS[name](2))
    assert not np.array_equal(a, b), (
        f"{name} is tagged seed-reproducible but ignores the seed (identical "
        f"output for seeds 1 and 2) -- it may actually be bit-exact."
    )
