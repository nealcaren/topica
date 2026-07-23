"""Parity comparison for topica's TensorLDA vs upstream TensorLy TLDA.

The upstream reference (https://github.com/tensorly/tlda) is not a topica
dependency. Point ``TOPICA_TLDA_REF`` at a checkout to run the comparison;
without it the script reports that the reference is unavailable and exits 0, so
it is safe to schedule as a CI / integration job. See ``parity/tlda_ref.py`` and
``docs/replications/tlda.md`` for the one-time setup.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import topica  # noqa: E402
from tlda_ref import load_reference_tlda  # noqa: E402

TLDA = load_reference_tlda()
HAS_REF = TLDA is not None

def run_comparison():
    if not HAS_REF:
        print("Reference TLDA not available in python environment. Skipping parity check.")
        return

    topica.enable_experimental(True)

    # Generate synthetic document-term count matrix
    np.random.seed(42)
    n_docs = 100
    vocab_size = 12
    x = np.random.randint(0, 5, size=(n_docs, vocab_size)).astype(float)

    # Convert X matrix to token lists for topica
    docs = []
    for i in range(n_docs):
        doc = []
        for w in range(vocab_size):
            count = int(x[i, w])
            doc.extend([f"word_{w}"] * count)
        docs.append(doc)

    # Initialize and fit reference model
    ref = TLDA(
        n_topic=3,
        alpha_0=1.0,
        n_iter_train=50,
        n_iter_test=30,
        learning_rate=0.01,
        third_order_cumulant_batch=10,
        smoothing=0.01,
        random_seed=42,
    )
    ref.fit(x)

    # Initialize and fit topica TensorLDA
    m = topica.TensorLDA(3, alpha_0=1.0, n_iter_train=50, n_iter_test=30, learning_rate=0.01, seed=42)
    m.fit(docs)

    # Trigger unwhitening in the reference model
    ref_factors = ref.unwhitened_factors

    print("Ref PCA components shape:", ref.second_order.projection_weights_.shape)
    print("Ref explained variance (lambdas):", ref.second_order.pca.explained_variance_)
    print("Ref whitening weights (explained_variance * (N-1)/N):", ref.second_order.whitening_weights_)
    
    # Compare weights
    print("Ref weights:", ref.weights_)
    print("Topica weights:", m.weights)

    # Compare unwhitened raw factors (or beta topic_word)
    ref_beta = ref_factors.T
    top_beta = m.topic_word
    print("Ref beta shape:", ref_beta.shape)
    print("Topica beta shape:", top_beta.shape)

    matched_indices = []
    for ref_t in ref_beta:
        best_idx = -1
        best_cos = -1.0
        for i, top_t in enumerate(top_beta):
            cos = np.dot(ref_t, top_t) / (np.linalg.norm(ref_t) * np.linalg.norm(top_t))
            if cos > best_cos:
                best_cos = cos
                best_idx = i
        matched_indices.append((best_idx, best_cos))

    print("Matched topic indices and cosine similarities:")
    for idx, (top_idx, cos) in enumerate(matched_indices):
        print(f"Ref topic {idx} -> Topica topic {top_idx}: cosine={cos:.6f}")
        assert cos > 0.85, f"Parity mismatch: cosine similarity is only {cos:.6f}"

    # Compare doc_topic predictions
    ref_theta = ref.transform(x, predict=True)
    # Align doc_topic columns based on matches
    aligned_ref_theta = np.zeros_like(ref_theta)
    for ref_idx, (top_idx, _) in enumerate(matched_indices):
        aligned_ref_theta[:, top_idx] = ref_theta[:, ref_idx]

    # Normalize aligned ref theta rows
    aligned_ref_theta /= aligned_ref_theta.sum(axis=1, keepdims=True)

    # Compare with topica's doc_topic
    print("Comparing doc_topic predictions...")
    mae = np.mean(np.abs(aligned_ref_theta - m.doc_topic))
    print(f"Mean Absolute Error (MAE) of doc_topic: {mae:.6f}")
    assert mae < 0.45, f"Doc topic prediction mismatch: MAE={mae:.6f}"

    print("SUCCESS: Parity checks passed completely!")


def run_streaming_comparison():
    """topica's streaming ``partial_fit`` vs the reference online ``partial_fit``.

    Both build the whitening in one pass over the batches, then train the factors
    over several passes -- never holding the whole count matrix. We check that the
    two implementations recover the same topic-word distributions (aligned cosine).
    """
    if not HAS_REF:
        print("Reference TLDA not available. Skipping streaming parity check.")
        return

    topica.enable_experimental(True)

    rng = np.random.default_rng(0)
    k, block, n_docs, length = 3, 8, 180, 40
    vocab_size = k * block
    x = np.zeros((n_docs, vocab_size))
    for d in range(n_docs):
        b = d % k
        for _ in range(length):
            x[d, b * block + rng.integers(0, block)] += 1

    vocab = [f"word_{w}" for w in range(vocab_size)]
    docs = []
    for d in range(n_docs):
        doc = []
        for w in range(vocab_size):
            doc.extend([f"word_{w}"] * int(x[d, w]))
        docs.append(doc)

    batch_idx = [list(range(i, min(i + 60, n_docs))) for i in range(0, n_docs, 60)]
    n_iter_train = 40

    ref = TLDA(
        n_topic=k, alpha_0=1.0, n_iter_train=n_iter_train, n_iter_test=20,
        learning_rate=0.01, pca_batch_size=60, third_order_cumulant_batch=10,
        smoothing=0.01, theta=1.0, random_seed=42,
    )
    for _ in range(1 + n_iter_train):
        for bi, idx in enumerate(batch_idx):
            ref.partial_fit(x[idx], bi)
    ref_beta = np.clip(np.asarray(ref.unwhitened_factors), 0, None)
    ref_beta = (ref_beta / (ref_beta.sum(0, keepdims=True) + 1e-12)).T  # k x V

    m = topica.TensorLDA(k, alpha_0=1.0, learning_rate=0.01, batch_size=10,
                         pca_batch_size=60, seed=42)
    for _ in range(1 + n_iter_train):
        for bi, idx in enumerate(batch_idx):
            m.partial_fit([docs[i] for i in idx], bi, vocabulary=vocab)
    m.finalize()
    top_beta = m.topic_word  # k x V

    used = set()
    print("Streaming topica-vs-ref aligned topic-word cosine:")
    for i, tt in enumerate(top_beta):
        best_j, best_cos = -1, -1.0
        for j, rt in enumerate(ref_beta):
            if j in used:
                continue
            cos = np.dot(tt, rt) / (np.linalg.norm(tt) * np.linalg.norm(rt) + 1e-12)
            if cos > best_cos:
                best_cos, best_j = cos, j
        used.add(best_j)
        print(f"  topica topic {i} -> ref topic {best_j}: cosine={best_cos:.6f}")
        assert best_cos > 0.99, f"Streaming parity mismatch: cosine {best_cos:.6f}"

    print("SUCCESS: Streaming parity check passed!")


if __name__ == "__main__":
    if not HAS_REF:
        print(
            "Reference TensorLy TLDA not found. Set TOPICA_TLDA_REF to a checkout "
            "of https://github.com/tensorly/tlda to run this comparison. Skipping."
        )
        sys.exit(0)
    run_comparison()
    print()
    run_streaming_comparison()
