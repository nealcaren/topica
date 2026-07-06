#!/usr/bin/env python
"""
Verification script for NarrativeTM.
Generates a small synthetic corpus with distinct beginning-middle-end topic structures
and fits NarrativeTM to verify it recovers the global trajectories.
"""

from __future__ import annotations
import os
import numpy as np
import topica

def main():
    print("Generating synthetic narrative corpus...")
    # Word pools for 3 stages
    beg_words = ["intro", "background", "start", "hook", "opening", "preface"]
    mid_words = ["body", "development", "climax", "conflict", "clash", "crisis"]
    end_words = ["conclusion", "ending", "resolution", "summary", "closing", "epilogue"]

    rng = np.random.default_rng(42)
    docs = []

    for _ in range(80):
        # Build a document with a clear beginning, middle, and end structure
        doc = []
        # Beginning: 15-20 words from beg_words
        doc.extend(rng.choice(beg_words, size=rng.integers(15, 20)).tolist())
        # Middle: 20-25 words from mid_words
        doc.extend(rng.choice(mid_words, size=rng.integers(20, 25)).tolist())
        # End: 15-20 words from end_words
        doc.extend(rng.choice(end_words, size=rng.integers(15, 20)).tolist())
        docs.append(doc)

    print(f"Generated {len(docs)} documents. Average length: {np.mean([len(d) for d in docs]):.1f} tokens.")

    # Enable experimental models
    print("Enabling experimental models...")
    topica.enable_experimental()

    # Fit NarrativeTM
    print("Fitting NarrativeTM (K=3, degree=3)...")
    model = topica.NarrativeTM(
        num_topics=3,
        degree=3,
        segment_by="chunk",
        chunk_size=15,
        seed=42,
        optimize_interval=10,
        burn_in=20,
    )
    model.fit(docs, iters=80)

    print("\nModel vocabulary size:", len(model.vocabulary))
    print("\nRecovered Topics (Top Words):")
    for k in range(3):
        top = " ".join([w for w, _ in model.top_words(5, topic=k)])
        print(f"Topic {k}: {top}")

    # Evaluate global trajectory at t = 0.0, 0.5, 1.0
    print("\nEvaluating Global Trajectories:")
    t_points = [0.0, 0.5, 1.0]
    traj = model.global_trajectory(t_points)
    
    print(f"{'Position (t)':<12} | {'Topic 0':<10} | {'Topic 1':<10} | {'Topic 2':<10}")
    print("-" * 51)
    for i, t in enumerate(t_points):
        print(f"{t:<12.1f} | {traj[i, 0]:<10.4f} | {traj[i, 1]:<10.4f} | {traj[i, 2]:<10.4f}")

    # Reconstructed doc_topic shape
    print("\nReconstructed doc_topic shape:", model.doc_topic.shape)
    print("Sample document topic mixtures (first 3 documents):")
    for i in range(3):
        theta_str = " ".join(f"{v:.4f}" for v in model.doc_topic[i])
        print(f"Doc {i}: {theta_str}")

    # Save & Load verification
    temp_path = "scratch_narrative_model.topica"
    print(f"\nSaving model to {temp_path}...")
    model.save(temp_path)

    print(f"Loading model from {temp_path}...")
    loaded_model = topica.NarrativeTM.load(temp_path)
    print("Loaded model:", loaded_model)

    # Clean up temp files
    for p in (temp_path, temp_path + "._inner_gdmr", temp_path + "._inner_gdmr._inner_dmr"):
        if os.path.exists(p):
            os.remove(p)
    print("Cleaned up temp model files.")

    # Check trajectory agreement
    loaded_traj = loaded_model.global_trajectory(t_points)
    np.testing.assert_allclose(traj, loaded_traj, rtol=1e-7)
    print("\nVerification SUCCESS: Save/load and trajectory evaluations match perfectly!")

if __name__ == "__main__":
    main()
