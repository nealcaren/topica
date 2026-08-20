"""Fit-progress callback for Wordfish and TBIP (#786).

Each model's ``fit(progress=cb)`` must call ``cb(iteration, total, metric)`` once
per main-loop iteration, drive iteration up to the ``iters`` budget (a Wordfish
convergence snap may report ``(n, n)``), abort on a callback that raises
``KeyboardInterrupt``, reject a non-callable ``progress``, and leave the fit
bit-identical to a ``progress=None`` run at the same seed.
"""

import numpy as np
import pytest

import topica

# ~200 short token lists over a ~40-word vocab, two planted themes so the
# Poisson scaling has a real axis to find.
_RNG = np.random.RandomState(0)
_BLOCK_A = [f"a{i}" for i in range(20)]
_BLOCK_B = [f"b{i}" for i in range(20)]
DOCS = []
for d in range(200):
    # Mix the two blocks with a doc-specific tilt so authors differ.
    tilt = (d % 10) / 10.0
    doc = []
    for _ in range(8):
        block = _BLOCK_A if _RNG.rand() < tilt else _BLOCK_B
        doc.append(block[_RNG.randint(0, len(block))])
    DOCS.append(doc)

GROUPS = [f"g{i % 4}" for i in range(len(DOCS))]


def _fit(model_name, iters, progress):
    if model_name == "Wordfish":
        return topica.Wordfish().fit(DOCS, iters=iters, progress=progress)
    if model_name == "TBIP":
        return topica.TBIP(5).fit(DOCS, group=GROUPS, iters=iters, progress=progress)
    raise AssertionError(model_name)


def _array(model_name, model):
    if model_name == "Wordfish":
        return model.author_positions
    return model.topic_word


MODELS = ["Wordfish", "TBIP"]


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_callback_fires(model_name):
    iters = 12
    calls = []
    _fit(model_name, iters, lambda it, total, metric: calls.append((it, total)))
    assert calls, f"{model_name}: progress callback never fired"
    # Every reported total is the iter budget, except a convergence snap that
    # reports (n, n).
    assert all(total == iters or it == total for it, total in calls), (
        f"{model_name}: unexpected totals: {calls[:5]}"
    )
    assert calls[0][0] == 1


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_abort_raises(model_name):
    def boom(it, total, metric):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model_name, 40, boom)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_type_validation(model_name):
    with pytest.raises(TypeError):
        _fit(model_name, 20, 5)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_determinism(model_name):
    a = _array(model_name, _fit(model_name, 12, None))
    b = _array(model_name, _fit(model_name, 12, lambda *args: None))
    assert np.array_equal(a, b), f"{model_name}: progress callback perturbed the fit"
