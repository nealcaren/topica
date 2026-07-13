"""Parity comparison script for topica's TensorLDA vs the reference Python TLDA package.
"""

import sys
sys.path.append("/Users/nealcaren/.gemini/antigravity/brain/d6ed7e63-64cd-4bfd-bf91-4edae64b50f4/scratch")

import numpy as np
import topica

try:
    from tlda_wrapper import TLDA
    HAS_REF = True
except ImportError:
    HAS_REF = False

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
    assert mae < 1e-3, f"Doc topic prediction mismatch: MAE={mae:.6f}"

    print("SUCCESS: Parity checks passed completely!")

if __name__ == "__main__":
    run_comparison()
