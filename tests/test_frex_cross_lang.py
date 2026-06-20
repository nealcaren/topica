"""Cross-language FREX/score/lift parity: topica-core's Rust ``inspect`` module
vs the pure-Python ``topica.validation`` scorers (issue #260).

The Rust ``inspect`` module (``frex_scores`` / ``lift_scores`` / ``score_scores``)
is the stm-faithful port that faSTM and the Stata plugin use; topica's Python
layer has its own implementations in ``topica.validation``. These tests pin down
where the two agree so the definitions cannot silently drift, and document (via
xfail) the one place they intentionally differ today.

Findings locked in here:

- **FREX** (no James-Stein shrinkage, the comparable mode): **bit-for-bit
  identical**. Same ECDF/average rank on continuous data, same harmonic mean.
- **score**: identical up to the log floor (Rust floors at ``f64::EPSILON``,
  Python at ``1e-12``), which only moves negligible-probability words; the
  selected top words match exactly.
- **lift**: *different by definition* — Python ``label_topics`` reports a
  mean-beta ratio ``beta / mean_k beta``; stm/Rust reports the empirical-frequency
  log-lift ``log(beta) - log(wordfreq)``. Captured as an xfail until unified.

These call the internal ``topica._topica.inspect_*`` bindings (added for exactly
this comparison) and need no R or external data.
"""

import numpy as np
import pytest

from topica import _topica
from topica import validation as val


def _continuous_beta(seed=0, K=6, V=200):
    """A K x V topic-word matrix with no ties (so ranks are unambiguous)."""
    rng = np.random.default_rng(seed)
    beta = rng.gamma(0.3, size=(K, V))
    beta /= beta.sum(axis=1, keepdims=True)
    vocab = [f"w{i}" for i in range(V)]
    return beta, vocab


def _py_frex_matrix(beta, vocab, w=0.5):
    """Full (K, V) FREX score matrix from the Python scorer (n=V returns all)."""
    K, V = beta.shape
    top = val.frex(beta, vocab, w=w, n=V)
    mat = np.zeros((K, V))
    for t in range(K):
        for word, score in top[t]:
            mat[t, int(word[1:])] = score
    return mat


def test_frex_matches_rust_inspect_bitwise():
    beta, vocab = _continuous_beta()
    rust = np.array(_topica.inspect_frex_scores(beta.tolist(), [], 0.5))
    py = _py_frex_matrix(beta, vocab, w=0.5)
    # Continuous beta -> no ties -> the two rank methods coincide exactly.
    np.testing.assert_allclose(rust, py, atol=1e-12, rtol=0)


@pytest.mark.parametrize("w", [0.3, 0.5, 0.7])
def test_frex_top_words_agree(w):
    beta, vocab = _continuous_beta(seed=1)
    rust = np.array(_topica.inspect_frex_scores(beta.tolist(), [], w))
    py = _py_frex_matrix(beta, vocab, w=w)
    for t in range(beta.shape[0]):
        r_top = set(np.argsort(rust[t])[::-1][:10])
        p_top = set(np.argsort(py[t])[::-1][:10])
        assert r_top == p_top


def test_frex_shrinkage_is_an_option_only_in_rust():
    # Passing word_counts engages stm's James-Stein exclusivity shrinkage, which
    # changes the scores; the Python scorer has no such mode (no-shrink only).
    beta, _ = _continuous_beta(seed=2)
    rng = np.random.default_rng(2)
    wc = rng.integers(1, 500, beta.shape[1]).tolist()
    no_shrink = np.array(_topica.inspect_frex_scores(beta.tolist(), [], 0.5))
    shrink = np.array(_topica.inspect_frex_scores(beta.tolist(), wc, 0.5))
    assert not np.allclose(no_shrink, shrink)


def test_score_top_words_agree():
    # Top-word agreement on a realistic (sparse) beta: ranking is robust to the
    # differing log floors (which only touch negligible-probability words).
    beta, vocab = _continuous_beta(seed=3)
    rust = np.array(_topica.inspect_score_scores(beta.tolist()))
    log_phi = np.log(np.clip(beta, 1e-12, None))
    py = beta * (log_phi - log_phi.mean(axis=0))
    for t in range(beta.shape[0]):
        assert (set(np.argsort(rust[t])[::-1][:10])
                == set(np.argsort(py[t])[::-1][:10]))


def test_score_matches_rust_inspect_on_dense_beta():
    # With no near-zero entries, the two log floors (Rust f64::EPSILON, Python
    # 1e-12) never bite, so the score formula matches bit-for-bit.
    rng = np.random.default_rng(5)
    beta = rng.uniform(0.5, 1.5, size=(6, 200))
    beta /= beta.sum(axis=1, keepdims=True)  # all entries ~1/V, far above any floor
    rust = np.array(_topica.inspect_score_scores(beta.tolist()))
    log_phi = np.log(beta)
    py = beta * (log_phi - log_phi.mean(axis=0))
    np.testing.assert_allclose(rust, py, atol=1e-12, rtol=0)


@pytest.mark.xfail(reason="Python label_topics lift = beta/mean_k beta; stm/Rust "
                          "lift = log(beta) - log(empirical word freq). Different "
                          "definitions until unified (#260).", strict=True)
def test_lift_matches_rust_inspect():
    beta, _ = _continuous_beta(seed=4)
    rng = np.random.default_rng(4)
    wc = rng.integers(1, 500, beta.shape[1]).tolist()
    rust = np.array(_topica.inspect_lift_scores(beta.tolist(), wc))
    marginal = beta.mean(axis=0)
    py = beta / np.where(marginal > 0, marginal, 1e-12)
    np.testing.assert_allclose(rust, py, atol=1e-8)
