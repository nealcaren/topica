"""The Rust co-occurrence core behind coherence() must match the pure-Python
fallback exactly, and must be the path used when the extension is present."""

import numpy as np
import pytest

import topica
import topica._topica as ext


DOCS = (
    [["cat", "dog", "fish", "pet", "vet", "paw"]] * 25
    + [["planet", "star", "moon", "sun", "orbit", "comet"]] * 25
    + [["cat", "star", "dog", "moon", "pet", "orbit"]] * 10  # some cross-talk
)


def _fit():
    m = topica.models.LDA(num_topics=4, seed=1)
    m.fit(DOCS, iters=200)
    return m


def _both_paths(model, ct, **kw):
    fast = topica.coherence(model, DOCS, coherence_type=ct, **kw)
    saved = ext.window_cooccurrence
    try:
        del ext.window_cooccurrence  # force the pure-Python fallback
        slow = topica.coherence(model, DOCS, coherence_type=ct, **kw)
    finally:
        ext.window_cooccurrence = saved
    return fast, slow


def test_rust_function_is_exposed():
    assert hasattr(ext, "window_cooccurrence")


@pytest.mark.parametrize("ct", ["u_mass", "c_uci", "c_npmi", "c_v"])
def test_rust_matches_python_fallback(ct):
    m = _fit()
    fast, slow = _both_paths(m, ct)
    assert np.allclose(np.nan_to_num(fast), np.nan_to_num(slow), atol=1e-9)


def test_window_size_override_matches():
    m = _fit()
    fast, slow = _both_paths(m, "c_v", window_size=5)
    assert np.allclose(np.nan_to_num(fast), np.nan_to_num(slow), atol=1e-9)


def test_window_cooccurrence_basic_semantics():
    # Two docs; relevant words 0 and 1, sentinel for the rest.
    S = (1 << 32) - 1
    docs = [[0, S, 1], [0, 1, 1]]
    # Whole-document window (window=0): occ = #docs containing the word,
    # co = #docs containing both.
    occ, co, nw = ext.window_cooccurrence(docs, 2, [(0, 1)], 0)
    assert occ == [2.0, 2.0]
    assert co == [2.0]
    assert nw == 2.0

    # Width-2 sliding window on a single 3-token doc [0, S, 1]: windows are
    # positions {0,1} and {1,2}. Word 0 is in window 0 only; word 1 in window 1
    # only; they never share a window.
    occ, co, nw = ext.window_cooccurrence([[0, S, 1]], 2, [(0, 1)], 2)
    assert nw == 2.0
    assert occ == [1.0, 1.0]
    assert co == [0.0]
