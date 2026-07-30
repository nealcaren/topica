"""Cross-implementation validation: topica's ``ensemble(method="stable")`` vs
gensim's ``EnsembleLda`` (Brigl 2019).

The "stable" method is a numpy-native procedure derived from gensim's CBDBSCAN
stable-topic clustering: pool the topics from every run, compute an asymmetric
masked-cosine distance between them, run Checkback DBSCAN to find dense cores, and
keep the clusters with enough cores as stable topics (averaging their members).

Unlike the R parity checks, gensim is a Python package, so the comparison is exact
rather than statistical: we drive *both* implementations from the SAME pooled
topic-term array (so no LDA training noise enters) and check that on a
well-separated input they agree to floating-point precision on (1) the asymmetric
distance matrix and (2) the resulting stable topics.

topica deliberately diverges from gensim on two edge cases where gensim
degenerates; the second half of this script pins those divergences down so they
stay intentional rather than accidental:

  * small-vocabulary rank masking — gensim's ``rank_masking`` yields an all-False
    mask when ``int(V * threshold) == 0``; topica keeps at least one term.
  * scan-order-dependent core validation — gensim seeds a core's own label into
    ``neighboring_labels`` only reciprocally, so a core with no core neighbor is
    left order-dependently unvalidated; topica seeds the self-label.

Skips (exit 0) if gensim is unavailable. Run directly:

    python parity/ensemblelda_gensim_compare.py
"""

from __future__ import annotations

import sys

import numpy as np


def gensim_available():
    try:
        import gensim.models.ensemblelda  # noqa: F401
        return True
    except Exception:
        return False


def _synthetic_ttda(m=5, K=4, V=30, seed=0):
    """A pooled topic-term array with clean cluster structure: K disjoint-block
    prototypes, lightly perturbed in each of m runs. Rows are normalized."""
    rng = np.random.default_rng(seed)
    protos = np.zeros((K, V))
    for k in range(K):
        protos[k, k * (V // K):(k + 1) * (V // K)] = 1.0
    protos /= protos.sum(1, keepdims=True)
    runs = []
    for _ in range(m):
        b = protos + rng.normal(0, 0.01, protos.shape)
        b = np.clip(b, 1e-6, None)
        b /= b.sum(1, keepdims=True)
        runs.append(b)
    return runs


def _gensim_stable(ttda, eps, min_samples, min_cores):
    """Run gensim's own clustering pipeline on a fixed ttda and return its stable
    topics — the reference output."""
    from gensim.models.ensemblelda import (
        CBDBSCAN, _aggregate_topics, _calculate_asymmetric_distance_matrix_chunk,
        _group_by_labels, _is_valid_core, _validate_clusters, mass_masking,
    )

    D = _calculate_asymmetric_distance_matrix_chunk(ttda, ttda, 0, mass_masking, 0.95)
    cb = CBDBSCAN(eps=eps, min_samples=min_samples)
    cb.fit(D)
    clusters = _aggregate_topics(_group_by_labels(cb.results))
    valid = {c.label for c in _validate_clusters(clusters, min_cores)}
    for t in cb.results:
        t.valid_neighboring_labels = {lbl for lbl in t.neighboring_labels if lbl in valid}
    mask = np.vectorize(_is_valid_core)(cb.results)
    labels = np.array([t.label for t in cb.results])[mask]
    stable = np.vstack([ttda[mask][labels == lbl].mean(0) for lbl in np.unique(labels)])
    return D, stable


def main():
    if not gensim_available():
        print("gensim not available; skipping EnsembleLda parity check.")
        return 0

    import topica
    from topica.ensemble import _asymmetric_distance

    m, K = 5, 4
    runs = _synthetic_ttda(m=m, K=K)
    ttda = np.vstack(runs)
    eps = 0.1
    min_samples = int(m / 2)
    min_cores = min(3, max(1, int(m / 4 + 1)))

    D_gensim, stable_gensim = _gensim_stable(ttda, eps, min_samples, min_cores)
    D_topica = _asymmetric_distance(ttda, "mass", 0.95)
    res = topica.ensemble(
        runs, method="stable", eps=eps, min_samples=min_samples,
        min_cores=min_cores, masking="mass", masking_threshold=0.95,
    )

    d_err = float(np.abs(D_gensim - D_topica).max())
    print(f"asymmetric distance matrix: max abs diff = {d_err:.2e}")
    assert d_err < 1e-9, "asymmetric distance matrices disagree"

    print(f"stable topic count: gensim={stable_gensim.shape[0]} topica={res.topic_word.shape[0]}")
    assert stable_gensim.shape[0] == res.topic_word.shape[0] == K, "stable topic count mismatch"

    pairs = topica.align_topics(stable_gensim, res.topic_word, metric="cosine")
    t_err = max(d for _, _, d in pairs)
    print(f"stable topics: max aligned cosine distance = {t_err:.2e}")
    assert t_err < 1e-9, "stable topics disagree"

    print("OK: on a well-separated input, ensemble(method='stable') matches gensim.")

    _check_intended_divergences()
    return 0


def _check_intended_divergences():
    """Pin down the two edge cases where topica intentionally improves on gensim,
    so a future refactor cannot silently drop the improvement or the divergence."""
    from gensim.models.ensemblelda import rank_masking
    from topica.ensemble import _rank_mask, _cbdbscan

    # (1) Small-vocabulary rank masking. V=8, default threshold 0.11 -> gensim's
    #     int(8 * 0.11) == 0 gives an all-False mask; topica keeps >= 1 term.
    a = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
    g_mask = rank_masking(a, 0.11)
    t_mask = _rank_mask(a, 0.11)
    print(f"small-V rank mask: gensim keeps {int(g_mask.sum())}, topica keeps {int(t_mask.sum())}")
    assert g_mask.sum() == 0, "expected gensim to degenerate to an empty mask here"
    assert t_mask.sum() >= 1, "topica must keep at least one term"

    # ...and where gensim's rank mask is non-empty (normal vocabulary), topica
    # matches it bit-for-bit -- the divergence is confined to the degenerate case.
    rng = np.random.default_rng(0)
    for _ in range(500):
        V = int(rng.integers(20, 200))
        row = rng.random(V)
        row /= row.sum()
        for thr in (0.11, 0.3, 0.5):
            g = rank_masking(row, thr)
            if g.sum() == 0:
                continue
            assert np.array_equal(g, _rank_mask(row, thr)), (
                "topica rank mask must equal gensim where gensim is non-empty"
            )
    print("large-V rank mask: topica == gensim wherever gensim is non-empty")

    # (2) Scan-order-dependent core validation. A chain 0-1-2 (0<->1, 1<->2 close,
    #     0<->2 far) has exactly one core (topic 1) whose neighbors are both
    #     non-core, so gensim leaves its neighboring_labels empty; topica seeds the
    #     self-label, so the isolated-core count sees it.
    D = np.array([[0.0, 0.05, 0.20], [0.05, 0.0, 0.05], [0.20, 0.05, 0.0]])
    results = _cbdbscan(D, eps=0.1, min_samples=2)
    core = next(i for i, r in enumerate(results) if r.is_core)
    print(f"isolated core: topica neighboring_labels = {results[core].neighboring_labels}")
    assert results[core].neighboring_labels == {results[core].label}, (
        "topica must seed a core's own label so validation is scan-order-invariant"
    )

    print("OK: both intended divergences from gensim hold.")


if __name__ == "__main__":
    sys.exit(main())
