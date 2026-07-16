"""Exact contracts for SeededLDA and GSDMM.

These models do not have a convenient byte-identical external reference runner
in the way topica's MALLET-backed LDA path does. The reference contracts we can
test exactly are the advertised algorithmic formulas:

* SeededLDA follows the seededlda convention that a seed word receives
  ``weight * 100`` extra topic-word prior mass in its seed topic, and seeded
  tokens initialize into that topic.
* GSDMM follows the Movie Group Process equations for smoothed cluster-word
  distributions and in-sample soft document-cluster probabilities.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import topica


def test_seededlda_zero_sweep_seed_prior_is_exact() -> None:
    docs = [["tax"], ["iraq"]]
    model = topica.models.SeededLDA(
        {"econ": ["tax"], "war": ["iraq"]},
        alpha=0.1,
        beta=0.01,
        weight=0.01,
        seed=123,
    )
    model.fit(docs, iters=0)

    # weight=0.01 means a 1.0 extra pseudocount on each seed word in its topic.
    # With one seeded token assigned to each seeded topic and V=2:
    #   phi(seed topic, seed word) = (1 + beta + 1.0) / (1 + 2*beta + 1.0)
    #   phi(seed topic, other word) = beta / (1 + 2*beta + 1.0)
    expected_phi = np.array(
        [
            [2.01 / 2.02, 0.01 / 2.02],
            [0.01 / 2.02, 2.01 / 2.02],
        ]
    )
    expected_theta = np.array(
        [
            [1.1 / 1.2, 0.1 / 1.2],
            [0.1 / 1.2, 1.1 / 1.2],
        ]
    )

    assert model.topic_names == ["econ", "war"]
    assert model.vocabulary == ["tax", "iraq"]
    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.doc_topic, expected_theta, rtol=0, atol=1e-12)


def test_seededlda_weight_zero_reduces_to_symmetric_word_prior_at_initialization() -> None:
    docs = [["tax"], ["iraq"]]
    model = topica.models.SeededLDA(
        {"econ": ["tax"], "war": ["iraq"]},
        alpha=0.1,
        beta=0.01,
        weight=0.0,
        seed=123,
    )
    model.fit(docs, iters=0)

    # Seeded initialization still puts each seed token in its named topic, but
    # with weight=0 there is no extra seed-word pseudocount.
    expected_phi = np.array(
        [
            [1.01 / 1.02, 0.01 / 1.02],
            [0.01 / 1.02, 1.01 / 1.02],
        ]
    )
    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)


def _manual_gsdmm_counts(docs: list[list[str]], clusters: np.ndarray, vocab: list[str]):
    word_index = {w: i for i, w in enumerate(vocab)}
    k = int(clusters.max()) + 1
    m = np.zeros(k, dtype=float)
    n = np.zeros(k, dtype=float)
    nw = np.zeros((k, len(vocab)), dtype=float)
    encoded_docs: list[list[int]] = []
    for doc, cluster in zip(docs, clusters):
        kk = int(cluster)
        ids = [word_index[w] for w in doc]
        encoded_docs.append(ids)
        m[kk] += 1
        n[kk] += len(ids)
        for wid in ids:
            nw[kk, wid] += 1
    return encoded_docs, m, n, nw


def _manual_gsdmm_doc_topic(
    encoded_docs: list[list[int]],
    m: np.ndarray,
    n: np.ndarray,
    nw: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    k, v = nw.shape
    out = np.zeros((len(encoded_docs), k), dtype=float)
    vbeta = v * beta
    for d, ids in enumerate(encoded_docs):
        counts = Counter(ids)
        logp = np.zeros(k, dtype=float)
        for kk in range(k):
            lp = np.log(m[kk] + alpha)
            for wid, count in counts.items():
                base = nw[kk, wid] + beta
                for j in range(count):
                    lp += np.log(base + j)
            for i in range(len(ids)):
                lp -= np.log(n[kk] + vbeta + i)
            logp[kk] = lp
        probs = np.exp(logp - logp.max())
        out[d] = probs / probs.sum()
    return out


def test_gsdmm_public_outputs_follow_movie_group_process_formulas() -> None:
    docs = [["cat", "cat"], ["dog", "dog"], ["cat", "dog"]]
    alpha = 0.1
    beta = 0.1
    model = topica.models.GSDMM(num_topics=3, alpha=alpha, beta=beta, seed=2)
    model.fit(docs, iters=0)

    clusters = np.asarray(model.doc_cluster)
    encoded_docs, m, n, nw = _manual_gsdmm_counts(docs, clusters, model.vocabulary)

    expected_phi = (nw + beta) / (n[:, None] + len(model.vocabulary) * beta)
    expected_theta = _manual_gsdmm_doc_topic(
        encoded_docs,
        m,
        n,
        nw,
        alpha=alpha,
        beta=beta,
    )

    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.doc_topic, expected_theta, rtol=0, atol=1e-12)


def test_gsdmm_trace_records_effective_cluster_count_and_formula_likelihood() -> None:
    docs = [["cat", "cat"], ["dog", "dog"], ["cat", "dog"]]
    model = topica.models.GSDMM(num_topics=3, alpha=0.1, beta=0.1, seed=2)
    model.fit(docs, iters=1, progress_interval=1)

    clusters = np.asarray(model.doc_cluster)
    encoded_docs, _, n, nw = _manual_gsdmm_counts(docs, clusters, model.vocabulary)
    expected_ll = 0.0
    total_tokens = 0
    for ids, cluster in zip(encoded_docs, clusters):
        kk = int(cluster)
        denom = n[kk] + len(model.vocabulary) * 0.1
        for wid in ids:
            expected_ll += np.log((nw[kk, wid] + 0.1) / denom)
            total_tokens += 1
    expected_ll /= total_tokens

    assert model.cluster_count_history == [(1, model.num_topics)]
    assert len(model.log_likelihood_history) == 1
    iteration, observed_ll = model.log_likelihood_history[0]
    assert iteration == 1
    assert observed_ll == expected_ll

