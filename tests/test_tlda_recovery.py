"""Known-truth acceptance criteria for experimental TensorLDA topic recovery.

These are the explicit, CI-runnable floors for TensorLDA's method-of-moments
*topic recovery* on a fixed LDA simulation with known topic-word distributions
and unequal Dirichlet weights. They guard the square-rank configuration
(``n_eigenvec == num_topics``) against regression at its current baseline.

Prevalence calibration is a deliberately separate question: the planted weights
and document-topic correlations are computed and surfaced here as diagnostics
but are NOT asserted, because TensorLDA's ``weights`` are not yet validated as
calibrated prevalence estimates (see ``docs/replications/tlda.md``). The
rectangular case ``n_eigenvec > num_topics`` is a Topica extension and is
likewise reported, not gated.

The simulation mirrors ``parity/tlda_true_lda_compare.py`` (same seed and
config) so the numbers correspond; that script is the fuller human-readable
report. This module is hermetic: it imports only topica and numpy.
"""

from itertools import permutations

import numpy as np
import pytest

import topica

# Square-rank topic-recovery floors. Current baseline on this fixed simulation is
# mean aligned cosine 0.529 with a per-topic minimum of 0.443; these floors sit
# safely below to catch regression without being brittle.
MEAN_COSINE_FLOOR = 0.50
MIN_TOPIC_COSINE_FLOOR = 0.35


def _simulate(seed: int = 2026):
    """Sample documents from ordinary LDA with known beta and unequal weights."""
    rng = np.random.default_rng(seed)
    k, block, v, d, length = 4, 20, 80, 2500, 100
    weights = np.array([0.50, 0.25, 0.15, 0.10])
    beta = np.full((k, v), 0.15 / v)
    for topic in range(k):
        beta[topic, topic * block:(topic + 1) * block] += 0.85 / block
    beta /= beta.sum(axis=1, keepdims=True)

    theta = rng.dirichlet(weights, size=d)
    vocab = np.array([f"w{i}" for i in range(v)])
    docs = []
    for mixture in theta:
        assignments = rng.choice(k, size=length, p=mixture)
        docs.append([str(vocab[rng.choice(v, p=beta[topic])]) for topic in assignments])
    return docs, beta, theta, weights


def _align(cosine: np.ndarray) -> np.ndarray:
    """Permutation of fitted topics maximizing total cosine against the truth."""
    k = cosine.shape[0]
    best = max(permutations(range(k)), key=lambda p: sum(cosine[i, p[i]] for i in range(k)))
    return np.asarray(best)


def _fit_and_score(rank: int):
    docs, beta, theta_true, weights_true = _simulate()
    topica.enable_experimental(True)
    model = topica.TensorLDA(
        beta.shape[0], alpha_0=1.0, n_eigenvec=rank, n_iter_train=500,
        n_iter_test=50, learning_rate=0.01, batch_size=25, seed=2026,
    )
    model.fit(docs)
    learned = np.asarray(model.topic_word)
    cosine = beta @ learned.T / np.maximum(
        np.linalg.norm(beta, axis=1)[:, None] * np.linalg.norm(learned, axis=1)[None, :],
        1e-300,
    )
    fit_idx = _align(cosine)
    diag = cosine[np.arange(beta.shape[0]), fit_idx]
    return diag, model, fit_idx, theta_true, weights_true


def test_square_rank_topic_recovery_meets_floor():
    """Acceptance gate: square-rank (R=K) topic-word recovery must clear the floor."""
    diag, _model, _fit_idx, _theta_true, _weights_true = _fit_and_score(rank=4)
    mean_cos = float(diag.mean())
    min_cos = float(diag.min())

    assert mean_cos >= MEAN_COSINE_FLOOR, (
        f"square-rank mean aligned topic cosine {mean_cos:.4f} regressed below "
        f"the {MEAN_COSINE_FLOOR} acceptance floor; per-topic {np.round(diag, 4)}"
    )
    assert min_cos >= MIN_TOPIC_COSINE_FLOOR, (
        f"a topic was recovered at cosine {min_cos:.4f}, below the "
        f"{MIN_TOPIC_COSINE_FLOOR} per-topic floor; per-topic {np.round(diag, 4)}"
    )


def test_prevalence_is_reported_but_not_gated():
    """Weight/theta recovery is surfaced as a diagnostic, deliberately un-asserted.

    This documents the separation in code: TensorLDA's ``weights`` are not yet a
    validated prevalence estimator, so recovery here is allowed to be weak. The
    test only checks that the diagnostic quantities are well-formed and finite.
    """
    _diag, model, fit_idx, _theta_true, weights_true = _fit_and_score(rank=4)
    recovered_weights = np.asarray(model.weights)[fit_idx]

    assert recovered_weights.shape == weights_true.shape
    assert np.all(np.isfinite(recovered_weights))
    # Not asserted: agreement with weights_true. See module docstring.


@pytest.mark.parametrize("rank", [8])
def test_rectangular_rank_runs_as_extension(rank):
    """n_eigenvec > num_topics is a Topica extension: it must run and stay finite.

    No parity claim is made against the upstream square-rank method here.
    """
    diag, _model, _fit_idx, _theta_true, _weights_true = _fit_and_score(rank=rank)
    assert diag.shape == (4,)
    assert np.all(np.isfinite(diag))
