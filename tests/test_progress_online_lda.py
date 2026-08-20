"""Fit progress callback (`progress=`) for OnlineLDA, RTM, and TensorLDA (#786).

Each model gets the same four checks:
  1. a valid callback fires with the 3-arg contract (iter, total, info), the
     recorded totals are the iteration budget (except a convergence snap, which
     reports ``(n, n)``);
  2. a callback raising ``KeyboardInterrupt`` aborts the fit with
     ``KeyboardInterrupt``;
  3. a non-callable, non-bool ``progress`` (here ``5``) is a ``TypeError``;
  4. determinism: ``progress=lambda *a: None`` gives the same fitted topic-word
     matrix as ``progress=None`` for the same seed.
"""

import os

import numpy as np
import pytest

import topica

# TensorLDA is experimental-gated; enable it so the sweep can construct it.
os.environ.setdefault("TOPICA_EXPERIMENTAL", "1")
try:
    topica.enable_experimental()
except Exception:
    pass


def _planted_docs(n_docs=200, n_blocks=4, words_per_block=10, doc_len=12, seed=0):
    """A small planted-block corpus: each doc draws mostly from one word block."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(n_blocks * words_per_block)]
    docs = []
    for d in range(n_docs):
        b = d % n_blocks
        toks = []
        for _ in range(doc_len):
            if rng.random() < 0.85:
                w = b * words_per_block + rng.integers(0, words_per_block)
            else:
                w = rng.integers(0, len(vocab))
            toks.append(vocab[int(w)])
        docs.append(toks)
    return docs


DOCS = _planted_docs()
LINKS = [(i, i + 1) for i in range(0, 100, 2)]
ITERS = 12


def _fit(model_name, progress):
    """Fit one of the three models with the given `progress` value; return it."""
    if model_name == "OnlineLDA":
        return topica.OnlineLDA(4, seed=13).fit(DOCS, iters=ITERS, progress=progress)
    if model_name == "RTM":
        return topica.RTM(4, seed=13).fit(DOCS, LINKS, iters=ITERS, progress=progress)
    if model_name == "TensorLDA":
        return topica.TensorLDA(4, seed=13).fit(DOCS, iters=ITERS, progress=progress)
    raise AssertionError(model_name)


MODELS = ["OnlineLDA", "RTM", "TensorLDA"]


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_callback_fires(model_name):
    calls = []

    def cb(it, total, info):
        calls.append((it, total, info))

    _fit(model_name, cb)

    assert calls, f"{model_name}: progress callback never fired"
    for it, total, info in calls:
        assert isinstance(info, dict)
        assert 1 <= it <= total
        # Every call reports the iteration budget as the total, except a
        # convergence snap, which reports (n, n).
        assert total == ITERS or it == total, (it, total)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_keyboardinterrupt_aborts(model_name):
    def cb(it, total, info):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model_name, cb)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_non_callable_is_typeerror(model_name):
    with pytest.raises(TypeError):
        _fit(model_name, 5)


@pytest.mark.parametrize("model_name", MODELS)
def test_progress_is_deterministic(model_name):
    noop = _fit(model_name, lambda *a: None)
    none = _fit(model_name, None)
    np.testing.assert_array_equal(noop.topic_word, none.topic_word)
