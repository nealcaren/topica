"""`topica.stop_reason` — human-readable training-stop reason (issue #267).

Distinguishes the two cases `converged` alone leaves implicit: the run stopped
early on `convergence_tol` (a floor), or it ran the full `iters` budget (a
ceiling)."""

import numpy as np

import topica


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    blocks = [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"]]
    docs = []
    for i in range(120):
        b = blocks[i % len(blocks)]
        docs.append([b[rng.integers(0, len(b))] for _ in range(15)])
    return docs


def test_stop_reason_max_iters_when_tol_disabled():
    m = topica.LDA(num_topics=3, seed=1)
    m.fit(_corpus(), iters=40, convergence_tol=0.0)
    assert not m.converged
    msg = topica.stop_reason(m)
    assert "iteration cap" in msg and "without early stopping" in msg
    # reports the iteration count it ran
    assert "40" in msg


def test_stop_reason_converged_when_tol_loose():
    m = topica.LDA(num_topics=3, seed=1)
    # A very loose tolerance trips on the first recorded relative change.
    m.fit(_corpus(), iters=400, convergence_tol=10.0, check_every=10)
    assert m.converged
    msg = topica.stop_reason(m)
    assert msg.startswith("converged")
    assert "convergence_tol" in msg


def test_stop_reason_variational_model():
    # STM/CTM expose a variational bound trace; the helper reads it the same way.
    rng = np.random.default_rng(0)
    docs = _corpus(1)
    cov = rng.integers(0, 2, len(docs)).astype(float).reshape(-1, 1)
    m = topica.STM(num_topics=3, seed=1)
    m.fit(docs, cov, prevalence_names=["g"], iters=20, convergence_tol=0.0)
    msg = topica.stop_reason(m)
    assert "iteration cap" in msg or msg.startswith("converged")
