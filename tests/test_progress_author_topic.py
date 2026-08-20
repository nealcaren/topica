"""Fit-progress callback for AuthorTopic, TopicalNGrams, TopicsOverTime (#786).

Each model's ``fit(progress=cb)`` must call ``cb(iteration, total, info)`` once
per sweep, drive iteration up to the ``iters`` budget, abort on a callback that
raises ``KeyboardInterrupt``, reject a non-callable ``progress``, and leave the
fit bit-identical to a ``progress=None`` run at the same seed.
"""

import numpy as np
import pytest

import topica

# ~200 short token lists over a ~40-word vocab, two planted themes.
_RNG = np.random.RandomState(0)
_BLOCK_A = [f"a{i}" for i in range(20)]
_BLOCK_B = [f"b{i}" for i in range(20)]
DOCS = []
for d in range(200):
    block = _BLOCK_A if d % 2 == 0 else _BLOCK_B
    DOCS.append([block[_RNG.randint(0, len(block))] for _ in range(6)])

AUTHORS = [[f"a{d % 8}"] for d in range(len(DOCS))]
TIMES = [float(d % 5) for d in range(len(DOCS))]


def _corpus():
    return topica.Corpus.from_documents(DOCS)


def _fit(model_name, iters, progress):
    """Fit one of the three models with the given progress callback."""
    c = _corpus()
    if model_name == "AuthorTopic":
        return topica.AuthorTopic(5).fit(c, AUTHORS, iters=iters, progress=progress)
    if model_name == "TopicalNGrams":
        return topica.TopicalNGrams(5).fit(c, iters=iters, progress=progress)
    if model_name == "TopicsOverTime":
        return topica.TopicsOverTime(5).fit(
            c, timestamps=TIMES, iters=iters, progress=progress
        )
    raise AssertionError(model_name)


MODELS = ["AuthorTopic", "TopicalNGrams", "TopicsOverTime"]


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_callback_fires(model_name):
    iters = 30
    calls = []
    _fit(model_name, iters, lambda it, total, info: calls.append((it, total)))
    assert calls, f"{model_name}: progress callback never fired"
    assert all(total == iters for _, total in calls), (
        f"{model_name}: every call must report total == iters ({iters}): {calls[:3]}"
    )
    # Iterations are 1-based and reach the budget.
    assert calls[0][0] == 1
    assert calls[-1][0] == iters


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_abort_raises(model_name):
    def boom(it, total, info):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model_name, 40, boom)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_type_validation(model_name):
    with pytest.raises(TypeError):
        _fit(model_name, 20, 5)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_determinism(model_name):
    a = _fit(model_name, 30, None)
    b = _fit(model_name, 30, lambda *args: None)
    assert np.array_equal(a.topic_word, b.topic_word), (
        f"{model_name}: progress callback perturbed the fit"
    )
