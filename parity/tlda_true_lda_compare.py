"""Known-truth LDA recovery report for experimental TensorLDA.

This is the fuller human-readable report over the same fixed simulation. The
square-rank topic-recovery floor is enforced as an acceptance gate in
``tests/test_tlda_recovery.py`` (CI-runnable, hermetic); this script prints the
full diagnostics -- per-topic cosines at square and rectangular whitening ranks,
plus the weight and document-topic recovery that remain a separate, ungated
prevalence question.

    python parity/tlda_true_lda_compare.py
"""

from itertools import permutations

import numpy as np

import topica


def align(cosine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the permutation of fitted topics maximizing total cosine."""
    k = cosine.shape[0]
    best = max(permutations(range(k)), key=lambda p: sum(cosine[i, p[i]] for i in range(k)))
    return np.arange(k), np.asarray(best)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency (inputs have no ties)."""
    a_rank = np.argsort(np.argsort(a)).astype(float)
    b_rank = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def simulate(seed: int = 2026):
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


def evaluate(rank: int, docs, beta, theta_true, weights_true) -> None:
    k = beta.shape[0]
    model = topica.TensorLDA(
        k, alpha_0=1.0, n_eigenvec=rank, n_iter_train=500,
        n_iter_test=50, learning_rate=0.01, batch_size=25, seed=2026,
    )
    model.fit(docs)
    learned = np.asarray(model.topic_word)
    cosine = beta @ learned.T / np.maximum(
        np.linalg.norm(beta, axis=1)[:, None] * np.linalg.norm(learned, axis=1)[None, :],
        1e-300,
    )
    truth_idx, fit_idx = align(cosine)
    recovered_weights = np.asarray(model.weights)[fit_idx]
    theta = np.asarray(model.doc_topic)[:, fit_idx]
    theta_corr = [np.corrcoef(theta_true[:, j], theta[:, j])[0, 1] for j in range(k)]

    print(f"R={rank}")
    print("  topic cosines:     ", np.round(cosine[truth_idx, fit_idx], 4))
    print("  mean cosine:       ", round(float(cosine[truth_idx, fit_idx].mean()), 4))
    print("  true weights:      ", weights_true)
    print("  recovered weights: ", np.round(recovered_weights, 4))
    print("  weight rank corr.: ", round(rank_correlation(weights_true, recovered_weights), 4))
    print("  theta correlations:", np.round(theta_corr, 4))
    print("  theta MAE:         ", round(float(np.abs(theta_true - theta).mean()), 4))


def main() -> None:
    docs, beta, theta, weights = simulate()
    topica.enable_experimental(True)
    print("LDA simulation: D=2500, V=80, K=4, length=100, alpha=[0.5, 0.25, 0.15, 0.1]")
    evaluate(4, docs, beta, theta, weights)
    evaluate(8, docs, beta, theta, weights)


if __name__ == "__main__":
    main()
