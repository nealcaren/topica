"""Interrupt propagation + progress= validation for the shared emitters (#794).

Two guarantees, fixed once in the shared progress helpers (`deliver_progress`,
`resolve_progress`) and therefore covering every progress-enabled model:

1. A ``KeyboardInterrupt`` / ``SystemExit`` raised in the callback -- or a real
   Ctrl-C (SIGINT) that arrives while the GIL-released fit is running -- aborts
   the fit and propagates, so a long fit can actually be interrupted. An
   ordinary ``Exception`` from a buggy callback is still swallowed (a broken
   progress display must never kill a fit, #785).
2. A non-callable ``progress=`` raises ``TypeError`` up front instead of a
   silent no-op fit.
"""

import os
import signal
import sys

import pytest

import topica

# One small corpus, reused. Two clear word-blocks so every model has something
# to fit; token-list input keeps the models that take a Corpus or a list happy.
DOCS = [["tax", "budget", "vote", "senate"]] * 60 + [
    ["health", "care", "clinic", "nurse"]
] * 60


def _fit(model_name, progress, iters):
    """Kick off a fit with `progress` for a representative model per emit path.

    Covers the three shared emitters: `emit_progress` (LDA direct-in-loop, CTM
    closure), `emit_progress_bare` (BTM), and `emit_progress_ll_ppl` (keyATM),
    plus the custom GSDMM closure.
    """
    if model_name == "LDA":
        topica.LDA(2, seed=13).fit(DOCS, iters=iters, progress_interval=1, progress=progress)
    elif model_name == "CTM":
        topica.CTM(2, seed=13).fit(DOCS, iters=iters, progress=progress)
    elif model_name == "BTM":
        topica.BTM(2, seed=13).fit(DOCS, iters=iters, progress=progress)
    elif model_name == "GSDMM":
        topica.GSDMM(4, seed=13).fit(DOCS, iters=iters, progress_interval=1, progress=progress)
    elif model_name == "keyATM":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topica.KeyATM({"fiscal": ["tax", "budget"]}, num_topics=2, seed=13).fit(
                DOCS, iters=iters, progress_interval=1, progress=progress
            )
    else:  # pragma: no cover
        raise ValueError(model_name)


MODELS = ["LDA", "CTM", "BTM", "GSDMM", "keyATM"]


@pytest.mark.parametrize("model", MODELS)
def test_non_callable_progress_raises_typeerror(model):
    with pytest.raises(TypeError, match="callable"):
        _fit(model, progress=5, iters=20)


@pytest.mark.parametrize("model", MODELS)
def test_keyboardinterrupt_in_callback_aborts_and_propagates(model):
    calls = {"n": 0}

    def cb(*args):
        calls["n"] += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _fit(model, progress=cb, iters=500)
    # aborted at the first tick, not run to completion
    assert calls["n"] == 1


@pytest.mark.parametrize("model", MODELS)
def test_systemexit_in_callback_propagates(model):
    def cb(*args):
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        _fit(model, progress=cb, iters=500)


@pytest.mark.parametrize("model", MODELS)
def test_ordinary_exception_in_callback_is_swallowed(model):
    def cb(*args):
        raise ValueError("buggy user callback")

    # A raising-but-ordinary callback must not abort the fit (#785).
    _fit(model, progress=cb, iters=20)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGINT) is not a portable Ctrl-C simulation on Windows "
    "(it maps to TerminateProcess, not a KeyboardInterrupt); the check_signals "
    "abort path itself is platform-independent and covered by the callback-raises "
    "cases above.",
)
def test_real_sigint_during_fit_is_surfaced():
    # The reported bug: with progress default-on, a Ctrl-C during a slow fit was
    # swallowed at the callback and the fit ran to completion. The callback here
    # sends a real SIGINT to this process (without raising); the abort must come
    # from `py.check_signals()` surfacing the pending signal at the next tick.
    calls = {"n": 0}

    def cb(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            os.kill(os.getpid(), signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        _fit("LDA", progress=cb, iters=1000)
    assert calls["n"] >= 2


def test_progress_none_and_valid_callback_still_complete():
    # Regression guard: the abort plumbing does not disturb ordinary fits.
    seen = []
    _fit("LDA", progress=lambda it, total, info: seen.append(it), iters=20)
    assert seen  # ran to completion, callback fired
