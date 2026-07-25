"""Cross-implementation check for topica's Top2Vec against BERTopic.

topica's Top2Vec and BERTopic are independent embedding-clustering topic models.
They share the same shape (reduce the document embeddings, density-cluster, read
topics off the clusters) but not the same implementation: topica reduces with its
in-house UMAP, BERTopic with umap-learn. Exact agreement is therefore impossible, so we hold them to a
statistical-equivalence bar on a controlled task.

We plant well-separated document clusters (each with its own vocabulary block),
hand BOTH models the SAME document embeddings, and ask how well each recovers the
planted structure. The bar, following the keyATM parity test, is that topica
recovers the ground truth at least as well as BERTopic does (within a margin),
and that the two implementations agree with each other. We also report top-word
purity: each topic's top c-TF-IDF words should come from a single planted block.

Skips cleanly when BERTopic (and its UMAP/HDBSCAN deps) is unavailable.
"""

from __future__ import annotations

import numpy as np

N_CLUSTERS = 4
N_DOCS = 320
EMB_DIM = 12
BLOCK = 6  # vocabulary words per cluster
SEED = 0


def bertopic_available() -> bool:
    try:
        import bertopic  # noqa: F401
        import hdbscan  # noqa: F401
        import umap  # noqa: F401
    except Exception:
        return False
    return True


def make_data(seed: int = SEED):
    """Planted clusters: each document belongs to one cluster, embeds near that
    cluster's center, and uses only that cluster's vocabulary block. Returns
    (token_docs, joined_texts, doc_emb, word_emb, vocabulary, truth)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(N_CLUSTERS, EMB_DIM)) * 8.0
    vocab = [f"c{c}w{i}" for c in range(N_CLUSTERS) for i in range(BLOCK)]

    docs, texts, doc_emb, truth = [], [], [], []
    for d in range(N_DOCS):
        c = d % N_CLUSTERS
        block = [f"c{c}w{i}" for i in range(BLOCK)]
        toks = list(rng.choice(block, 8))
        docs.append(toks)
        texts.append(" ".join(toks))
        doc_emb.append(centers[c] + rng.normal(size=EMB_DIM) * 0.6)
        truth.append(c)
    # Word embeddings: each word near its block's center.
    word_emb = []
    for c in range(N_CLUSTERS):
        for _ in range(BLOCK):
            word_emb.append(centers[c] + rng.normal(size=EMB_DIM) * 0.6)

    return (
        docs,
        texts,
        np.array(doc_emb),
        np.array(word_emb),
        vocab,
        np.array(truth),
    )


def _ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def _block_purity(top_words_per_topic) -> float:
    """Fraction of topics whose top words all come from one planted block."""
    pure = 0
    for words in top_words_per_topic:
        blocks = {w.split("w")[0] for w in words}  # "c0w3" -> "c0"
        if len(blocks) == 1:
            pure += 1
    return pure / max(1, len(top_words_per_topic))


def run(verbose: bool = True) -> dict:
    import topica
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    docs, texts, doc_emb, word_emb, vocab, truth = make_data()
    min_cluster = 15

    # topica: shared embeddings, in-house UMAP reducer (the default).
    tv = topica.Top2Vec(n_components=5, min_cluster_size=min_cluster, seed=1)
    tv.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    tv_labels = np.array(tv.labels)
    # Compare the class-based TF-IDF words (what BERTopic also reports); Top2Vec's
    # default top_words is now the centroid view when word_embeddings are present.
    tv_words = [
        [w for w, _ in tv.top_words(BLOCK, topic=t, representation="c-tf-idf")]
        for t in range(tv.num_topics)
    ]

    # BERTopic: same embeddings, UMAP reducer, HDBSCAN with a fixed seed.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster, prediction_data=True)
    bt = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=False)
    bt_topics, _ = bt.fit_transform(texts, embeddings=doc_emb)
    bt_labels = np.array(bt_topics)
    bt_info = bt.get_topics()  # {topic_id: [(word, score)]}
    bt_words = [
        [w for w, _ in bt_info[t][:BLOCK]] for t in sorted(bt_info) if t != -1
    ]

    # Recovery of the planted truth (non-noise docs for each model).
    tv_mask = tv_labels >= 0
    bt_mask = bt_labels >= 0
    metrics = {
        "topica_num_topics": int(tv.num_topics),
        "bertopic_num_topics": int(len([t for t in set(bt_labels) if t >= 0])),
        "true_num_topics": N_CLUSTERS,
        "topica_truth_ari": _ari(truth[tv_mask], tv_labels[tv_mask]),
        "bertopic_truth_ari": _ari(truth[bt_mask], bt_labels[bt_mask]),
        "cross_ari": _ari(tv_labels, bt_labels),
        "topica_block_purity": _block_purity(tv_words),
        "bertopic_block_purity": _block_purity(bt_words),
        "topica_assigned_frac": float(tv_mask.mean()),
        "bertopic_assigned_frac": float(bt_mask.mean()),
    }
    if verbose:
        for k, v in metrics.items():
            print(f"  {k:24s} {v}")
    return metrics


if __name__ == "__main__":
    if not bertopic_available():
        print("BERTopic not installed; skipping.")
    else:
        run(verbose=True)
