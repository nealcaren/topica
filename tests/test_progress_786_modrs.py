"""progress= for the mod.rs-bound iterative models PT, STS, SupervisedLDA (#786).

Same contract as the other progress-enabled models: an opt-in (iteration, total,
info) callback, KeyboardInterrupt aborts, non-callable raises TypeError, and the
fit is bit-identical with and without the callback.
"""

import numpy as np
import pytest

import topica

rng = np.random.default_rng(0)
VOCAB = [f"w{i}" for i in range(40)]
DOCS = [[VOCAB[i] for i in rng.integers(0, 40, 20)] for _ in range(200)]
SENT = [i % 5 for i in range(200)]
Y = [float(i % 2) for i in range(200)]


def _corpus():
    return topica.Corpus.from_documents(DOCS)


def _fit(model, progress, iters):
    c = _corpus()
    if model == "PT":
        return topica.PT(5, seed=13).fit(c, iters=iters, progress=progress)
    if model == "STS":
        return topica.STS(5, seed=13).fit(c, SENT, iters=iters, progress=progress)
    if model == "SupervisedLDA":
        return topica.SupervisedLDA(5, seed=13).fit(c, Y, iters=iters, progress=progress)
    if model == "SupervisedLDA-gibbs":
        return topica.SupervisedLDA(5, seed=13, inference="gibbs").fit(
            c, Y, iters=iters, progress=progress
        )
    raise ValueError(model)


MODELS = ["PT", "STS", "SupervisedLDA", "SupervisedLDA-gibbs"]


@pytest.mark.parametrize("model", MODELS)
def test_progress_callback_fires(model):
    calls = []
    _fit(model, lambda it, total, info: calls.append((it, total)), iters=30)
    assert calls
    # totals are the iter budget, except a convergence "snap" which pegs the bar
    # to 100% by reporting (current, current).
    assert all(total in (30, it) for it, total in calls)


@pytest.mark.parametrize("model", MODELS)
def test_keyboardinterrupt_aborts(model):
    def boom(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model, boom, iters=500)


@pytest.mark.parametrize("model", MODELS)
def test_non_callable_progress_raises(model):
    with pytest.raises(TypeError):
        _fit(model, 5, iters=20)


@pytest.mark.parametrize("model", MODELS)
def test_progress_does_not_change_fit(model):
    a = _fit(model, None, iters=40)
    b = _fit(model, lambda *args: None, iters=40)
    assert np.allclose(np.asarray(a.topic_word), np.asarray(b.topic_word))
