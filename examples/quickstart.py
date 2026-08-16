"""topica quickstart — two-theme toy corpus demo.

This script is self-contained: it builds a small corpus entirely in memory,
fits an LDA model, inspects results, and demonstrates Corpus save/load. No
external files or downloads required.

Run from the project root:
    python examples/quickstart.py
"""

import os
import tempfile

import topica
from topica import LDA, Corpus

print(f"topica version: {topica.__version__}")
print()

# ---------------------------------------------------------------------------
# 1. Build a toy corpus — two clearly distinct themes
# ---------------------------------------------------------------------------

# Theme 1: pets/animals
animal_docs = [["cat", "dog", "fish", "cat", "dog"] for _ in range(15)]
# Theme 2: astronomy/space
space_docs  = [["planet", "star", "moon", "rocket", "planet"] for _ in range(15)]

# Combine into a flat list of pre-tokenized documents (list[list[str]])
documents: list[list[str]] = animal_docs + space_docs

# You can also attach names; here we just use the defaults.
print(f"Corpus: {len(documents)} documents, two themes (animals vs. space)")
print()

# ---------------------------------------------------------------------------
# 2. Fit the LDA model
# ---------------------------------------------------------------------------

# The simple way: sensible defaults, no tuning required.
print("Training …")
model = LDA(num_topics=2, seed=13)   # seed -> deterministic, reproducible fit
model.fit(documents)
print()

# --- Advanced: the same fit with the tuning knobs spelled out --------------
# You need none of these for good topics; they are here so you know what exists.
# Uncomment to try.
#
# def on_progress(iteration: int, ll_per_token: float) -> None:
#     # called every `progress_interval` iterations
#     print(f"  iter {iteration:4d}  LL/token = {ll_per_token:.4f}")
#
# model = LDA(
#     num_topics=2,
#     seed=13,
#     alpha_sum=2.0,         # document-topic prior concentration (default: num_topics)
#     beta=0.01,             # topic-word smoothing
#     optimize_interval=50,  # re-optimize alpha/beta every 50 iters after burn-in
#     burn_in=200,           # iters before hyperparameter optimization begins
# )
# model.fit(
#     documents,
#     iters=500,             # total Gibbs sweeps
#     num_samples=5,         # average this many snapshots for the final phi/theta
#     sample_interval=25,    # sweeps between snapshots
#     progress=on_progress,
#     progress_interval=100,
# )

# ---------------------------------------------------------------------------
# 3. Inspect result shapes
# ---------------------------------------------------------------------------

print(f"topic_word shape : {model.topic_word.shape}   (num_topics × num_words)")
print(f"doc_topic  shape : {model.doc_topic.shape}   (num_docs × num_topics)")
print(f"vocabulary       : {model.vocabulary}")
print(f"log-likelihood   : {model.log_likelihood():.4f}")
print()

# Verify θ rows sum to 1
row_sums = model.doc_topic.sum(axis=1)
print(f"doc_topic row sums — min={row_sums.min():.6f}  max={row_sums.max():.6f}  (should be 1.0)")
print()

# ---------------------------------------------------------------------------
# 4. Top words per topic
# ---------------------------------------------------------------------------

print("Top 5 words per topic:")
for topic_idx, words in enumerate(model.top_words(5, weights=True)):
    words_str = "  ".join(f"{w}({p:.3f})" for w, p in words)
    print(f"  Topic {topic_idx}: {words_str}")
print()

# Single-topic convenience (weights=True for the (word, probability) pairs)
print("Top 3 words for topic 0 only:")
for word, prob in model.top_words(3, topic=0, weights=True):
    print(f"  {word:12s} {prob:.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Document-topic assignments for a few documents
# ---------------------------------------------------------------------------

print("Document-topic assignments (first 3 animal docs, first 3 space docs):")
dt = model.doc_topic
for i in [0, 1, 2, 15, 16, 17]:
    theme = "animal" if i < 15 else "space "
    t0, t1 = dt[i, 0], dt[i, 1]
    dominant = 0 if t0 >= t1 else 1
    print(f"  doc {i:2d} ({theme})  topic0={t0:.3f}  topic1={t1:.3f}  → dominant topic {dominant}")
print()

# ---------------------------------------------------------------------------
# 6. Corpus.from_documents with filtering, then save / load round-trip
# ---------------------------------------------------------------------------

# Build a Corpus explicitly so we can demonstrate filtering and save/load.
corpus = Corpus.from_documents(
    documents,
    doc_names=[f"doc{i:03d}" for i in range(len(documents))],
    min_doc_freq=2,          # drop words that appear in fewer than 2 docs
    max_doc_fraction=0.95,   # drop words that appear in >95% of docs
)
print(f"Corpus object: {corpus}")
print(f"  num_docs={corpus.num_docs}  num_words={corpus.num_words}  "
      f"total_tokens={corpus.total_tokens}")
print(f"  vocabulary: {corpus.vocabulary}")
print()

# Save to a temp file and reload.
with tempfile.NamedTemporaryFile(suffix=".corp", delete=False) as fh:
    corp_path = fh.name

try:
    corpus.save(corp_path)
    corpus2 = Corpus.load(corp_path)
    assert corpus2.num_docs   == corpus.num_docs,   "num_docs mismatch after round-trip"
    assert corpus2.num_words  == corpus.num_words,  "num_words mismatch after round-trip"
    assert corpus2.vocabulary == corpus.vocabulary, "vocabulary mismatch after round-trip"
    print(f"Corpus save/load round-trip: OK  ({corp_path})")
finally:
    os.unlink(corp_path)

print()
print("Quickstart complete.")
