"""progress= forwarding for the pure-Python wrapper models (#786).

GDMR, NarrativeTM, and EmbeddingLDA are Python wrappers over compiled cores
(DMR / GDMR->DMR / SeededLDA) that already emit progress. These tests confirm
the wrappers forward `progress=` so the shared bar, the interrupt abort, and the
type validation all reach the user.
"""

import os
import warnings

import numpy as np
import pytest

import topica

os.environ.setdefault("TOPICA_EXPERIMENTAL", "1")
try:
    topica.enable_experimental()
except Exception:
    pass

rng = np.random.default_rng(0)
VOCAB = [f"w{i}" for i in range(40)]
DOCS = [[VOCAB[i] for i in rng.integers(0, 40, 20)] for _ in range(200)]
COV = rng.normal(size=200)


def _corpus():
    return topica.Corpus.from_documents(DOCS)


def _fit(model, progress, iters):
    c = _corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "GDMR":
            topica.GDMR(5, degrees=[3]).fit(c, COV, iters=iters, progress=progress)
        elif model == "NarrativeTM":
            topica.NarrativeTM(5).fit(c, iters=iters, progress=progress)
        elif model == "EmbeddingLDA":
            emb = rng.normal(size=(c.num_words, 16))
            topica.EmbeddingLDA(5, embeddings=emb, vocabulary=list(c.vocabulary)).fit(
                c, iters=iters, progress=progress
            )
        else:  # pragma: no cover
            raise ValueError(model)


MODELS = ["GDMR", "NarrativeTM", "EmbeddingLDA"]


@pytest.mark.parametrize("model", MODELS)
def test_wrapper_forwards_progress_callback(model):
    calls = []
    _fit(model, lambda it, total, info: calls.append(it), iters=30)
    assert calls  # the underlying core's progress reached our callback


@pytest.mark.parametrize("model", MODELS)
def test_wrapper_keyboardinterrupt_aborts(model):
    def boom(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model, boom, iters=500)


@pytest.mark.parametrize("model", MODELS)
def test_wrapper_non_callable_progress_raises(model):
    with pytest.raises(TypeError):
        _fit(model, 5, iters=20)
