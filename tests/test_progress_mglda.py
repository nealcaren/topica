"""Live fit-progress callback for MGLDA, DiscLDA, and FactorialLDA (#786).

Each model's fit(progress=cb) must call cb(iteration, total, info) once per
sweep, drive iteration up to total, abort on KeyboardInterrupt, reject a
non-callable/non-bool progress, and leave the fit bit-identical.
"""

import random

import numpy as np
import pytest

import topica

# ~200 short docs over a ~40-word vocab, two planted themes.
_RNG = random.Random(13)
_THEME_A = [f"a{i}" for i in range(20)]
_THEME_B = [f"b{i}" for i in range(20)]


def _make_docs(n=200):
    docs = []
    for d in range(n):
        pool = _THEME_A if d % 2 == 0 else _THEME_B
        docs.append([_RNG.choice(pool) for _ in range(8)])
    return docs


DOCS = _make_docs()
# Sentence-segmented view for MGLDA: split each doc into two sentences.
SDOCS = [[d[: len(d) // 2], d[len(d) // 2 :]] for d in DOCS]
Y = [d % 2 for d in range(len(DOCS))]


def _fit(model_name, progress, iters=30):
    """Fit one model with progress=progress at a small iters; return the model."""
    c = topica.Corpus.from_documents(DOCS)
    if model_name == "MGLDA":
        return topica.MGLDA(5, 5).fit(SDOCS, iters=iters, progress=progress)
    if model_name == "DiscLDA":
        return topica.DiscLDA(2, 4).fit(c, Y, iters=iters, progress=progress)
    if model_name == "FactorialLDA":
        return topica.FactorialLDA([2, 3]).fit(
            c, iters=iters, samples=10, progress=progress
        )
    raise AssertionError(model_name)


def _fitted_array(model_name, progress, iters=30):
    return np.asarray(_fit(model_name, progress, iters=iters).topic_word)


MODELS = ["MGLDA", "DiscLDA", "FactorialLDA"]


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_fires_with_total_equals_iters(model_name):
    iters = 30
    calls = []
    _fit(model_name, lambda it, total, info: calls.append((it, total)), iters=iters)
    assert calls, f"{model_name}: progress never fired"
    # A model that runs its full budget hits (iters, iters) on the last call.
    assert calls[-1] == (iters, iters), f"{model_name}: last call {calls[-1]}"
    assert all(total == iters for _, total in calls)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_keyboardinterrupt_aborts(model_name):
    def boom(it, total, info):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model_name, boom, iters=30)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_typeerror_on_non_callable(model_name):
    with pytest.raises(TypeError):
        _fit(model_name, 5, iters=30)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_does_not_change_fit(model_name):
    baseline = _fitted_array(model_name, None, iters=30)
    with_cb = _fitted_array(model_name, lambda *a: None, iters=30)
    np.testing.assert_array_equal(baseline, with_cb)
