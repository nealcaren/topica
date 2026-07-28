"""Cross-implementation check for topica's SemanticSignalSeparation (S³) against
the reference algorithm from turftopic (Kardos et al., ACL 2025).

turftopic's ``SemanticSignalSeparation`` is a thin wrapper over scikit-learn's
``FastICA``: it decomposes the document embeddings with FastICA (logcosh,
parallel, unit-variance whitening, ``max_iter=200``), then projects the
vocabulary embeddings through the fitted decomposition (``axial_components_``),
scores them by cosine to the standard basis (``angular_components_``), and by
default combines the two (``square(axial) * angular``). We reproduce exactly that
pipeline with scikit-learn as the oracle (turftopic itself pulls in torch +
sentence-transformers, which we do not need since topica takes embeddings
directly), and hold topica to a topic-aligned-similarity bar.

FastICA is stochastic (random ``w_init``) and identified only up to sign and
permutation of the axes, so we align axes by Hungarian assignment on the absolute
cosine of the signed component matrices and calibrate against the reference's own
seed-to-seed spread: topica should match the reference about as well as the
reference matches itself across seeds.

Skips cleanly when scikit-learn (or scipy) is unavailable.
"""

from __future__ import annotations

import numpy as np

N_DOCS = 300
EMB_DIM = 10
K = 4  # number of independent axes / topics
N_VOCAB = 40
SEED = 0


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def make_data(seed: int = SEED):
    """Planted independent-source embeddings + a vocabulary in the same space.

    K independent non-Gaussian sources are mixed into EMB_DIM dimensions; each
    vocabulary word is a noisy draw of one source direction, so the ICA axes have
    a known word structure.
    """
    rng = np.random.default_rng(seed)
    # Independent uniform sources (non-Gaussian => ICA-recoverable).
    sources = rng.uniform(-1.0, 1.0, size=(N_DOCS, K))
    mixing = rng.standard_normal((K, EMB_DIM))
    doc_emb = sources @ mixing + 0.02 * rng.standard_normal((N_DOCS, EMB_DIM))
    # Vocabulary: each word aligns with one source's mixing direction.
    words_per_axis = N_VOCAB // K
    vocab_emb = []
    vocab = []
    for k in range(K):
        for j in range(words_per_axis):
            w = mixing[k] + 0.15 * rng.standard_normal(EMB_DIM)
            vocab_emb.append(w)
            vocab.append(f"axis{k}_w{j}")
    return doc_emb, np.array(vocab_emb), vocab


def reference_components(doc_emb, vocab_emb, k, seed, feature_importance="combined"):
    """turftopic's S³ math via scikit-learn FastICA: returns the signed (K, V)
    ``components_`` and the (D, K) source scores."""
    from sklearn.decomposition import FastICA
    from sklearn.metrics.pairwise import cosine_similarity

    ica = FastICA(n_components=k, max_iter=200, random_state=seed)
    sources = ica.fit_transform(doc_emb)  # (D, K)
    axial = ica.transform(vocab_emb).T  # (K, V)
    if feature_importance == "axial":
        comps = axial
    else:
        axis_vectors = np.eye(k)
        angular = cosine_similarity(axis_vectors, axial.T)  # (K, V)
        if feature_importance == "angular":
            comps = angular
        else:  # combined
            comps = np.square(axial) * angular
    return comps, sources


def aligned_abs_cosine(a, b):
    """Mean per-axis |cosine| after Hungarian alignment of signed (K, V) matrices."""
    from scipy.optimize import linear_sum_assignment

    def norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    an, bn = norm(a), norm(b)
    sim = np.abs(an @ bn.T)  # (K, K) absolute cosine (sign-invariant)
    r, c = linear_sum_assignment(-sim)
    return sim[r, c].mean(), sim[r, c]


def main() -> None:
    if not sklearn_available():
        print("SKIP: scikit-learn/scipy not available")
        return
    import topica

    doc_emb, vocab_emb, vocab = make_data()
    # Token documents drawn from the vocabulary, so topica's corpus vocabulary
    # equals `vocab` and the vocab-embedding alignment is exact. S³ reads topic
    # structure from the embeddings, not these tokens.
    rng = np.random.default_rng(SEED + 7)
    docs = [[vocab[i] for i in rng.integers(0, len(vocab), 12)] for _ in range(N_DOCS)]

    # Reference seed-to-seed floor: how well does the reference match ITSELF
    # across two different random inits?
    ref_a, _ = reference_components(doc_emb, vocab_emb, K, seed=1)
    ref_b, _ = reference_components(doc_emb, vocab_emb, K, seed=2)
    floor, _ = aligned_abs_cosine(ref_a, ref_b)

    # topica vs the reference (reference seed 1).
    m = topica.SemanticSignalSeparation(K, seed=1).fit(
        docs, doc_emb, vocab_emb, vocabulary=vocab
    )
    topica_comps = np.asarray(m.components)
    # topica realigns the vocabulary to the corpus order; reorder the reference's
    # columns to match before comparing (columns are the V features of each axis).
    pos = {w: i for i, w in enumerate(vocab)}
    cols = [pos[w] for w in m.vocabulary]
    ref_a_aligned = ref_a[:, cols]
    mean_cos, per_axis = aligned_abs_cosine(topica_comps, ref_a_aligned)

    print(f"reference seed-to-seed floor (mean |cos|): {floor:.3f}")
    print(f"topica vs reference    (mean |cos|):       {mean_cos:.3f}")
    print(f"per-axis |cos|: {np.round(per_axis, 3).tolist()}")

    # The port should land inside the reference's own noise floor (within a small
    # margin), i.e. it matches the reference about as well as the reference
    # matches itself.
    assert mean_cos >= floor - 0.15, (
        f"topica-vs-reference {mean_cos:.3f} is well below the reference "
        f"seed-to-seed floor {floor:.3f}: a fidelity gap"
    )
    print("PASS: topica S³ reproduces the reference within its own seed-to-seed spread")


if __name__ == "__main__":
    main()
