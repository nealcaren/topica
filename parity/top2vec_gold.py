"""Committed-gold parity for topica Top2Vec vs BERTopic (#271, Wave 1).

topica's Top2Vec and BERTopic are independent embedding-clustering topic models:
they share the shape (reduce the document embeddings, density-cluster, read topics
off the clusters) but not the same implementation — topica reduces with its
in-house UMAP, BERTopic with umap-learn. Exact agreement is impossible, so — exactly as in the live script
``parity/top2vec_compare.py`` — we hold them to a statistical-equivalence bar on a
controlled planted-cluster task: well-separated document clusters (each with its own
vocabulary block), the SAME document embeddings handed to both models, and we ask how
well each recovers the planted structure.

HONEST NOTE ON THE BAR. Unlike the topic-word-cosine golds (FASTopic, CombinedTM,
…), a clustering model is gated on a *clustering-agreement* metric, not topic-word
cosine: the adjusted Rand index (ARI) of each model's partition against the planted
truth, plus the cross-ARI between the two implementations and per-topic block purity.
This mirrors the keyATM-style "topica recovers the truth at least as well as the
reference (within a margin)" bar the live ``tests/test_top2vec_parity.py`` already
uses — it is NOT a topic-word cosine.

The corpus + embeddings + config are taken verbatim from ``parity/top2vec_compare.py``
(``make_data(seed=0)``: 4 planted clusters, 320 docs, 12-dim synthetic embeddings, 6
vocabulary words per block). Because the embeddings are synthetic and seeded they are
fully reproducible; we still freeze them into the npz so the offline topica refit is
byte-identical to what BERTopic saw.

Two phases (mirrors parity/combinedtm_gold.py):

  * ``--regenerate`` (needs bertopic + umap + hdbscan): builds the planted data, fits
    BERTopic on the shared embeddings, and freezes BERTopic's recovered labels + its
    truth-recovery ARI / block purity, the planted truth, the synthetic doc and word
    embeddings, the vocab, and the corpus into the committed gold
    (``parity/top2vec_gold.npz`` + ``.json``).
  * default (no bertopic / umap / hdbscan): loads the committed gold, fits topica
    Top2Vec on the same frozen embeddings, and checks topica recovers the planted
    truth at least as well as the frozen BERTopic run (within margin), agrees with
    BERTopic's frozen partition, and keeps its top-word block purity.

Run directly::

    python parity/top2vec_gold.py               # offline compare against committed gold
    python parity/top2vec_gold.py --regenerate  # run bertopic once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import top2vec_compare as tc  # noqa: E402

NAME = "top2vec"

N_CLUSTERS = tc.N_CLUSTERS  # 4
BLOCK = tc.BLOCK            # 6
MIN_CLUSTER = 15
# Bars, verbatim from the live tests/test_top2vec_parity.py.
ARI_MARGIN = 0.2           # topica truth-ARI >= bertopic truth-ARI - margin
CROSS_ARI_MIN = 0.4
TRUTH_ARI_MIN = 0.5
PURITY_MIN = 0.9


def _ari(a, b) -> float:
    # numpy-only ARI (harness) so the offline gold test needs no scikit-learn,
    # which CI does not install (sklearn is only a regenerate-time reference).
    return harness.adjusted_rand_index(a, b)


def _topica_fit(docs, doc_emb, word_emb, vocab):
    """Fit topica Top2Vec on the shared embeddings; return
    ``(labels, ctfidf_words, centroid_words, num_topics)``.

    We score both representations: the shared c-TF-IDF words (comparable to
    BERTopic) and the distinctly-Top2Vec **centroid** words (#489 finding 5) —
    the vocabulary nearest each topic's embedding, which no sibling model
    produces and which the earlier gold never validated."""
    import topica

    tv = topica.Top2Vec(n_components=5, min_cluster_size=MIN_CLUSTER, seed=1)
    tv.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    labels = np.array(tv.labels)
    ctfidf_words = [
        [w for w, _ in tv.top_words(BLOCK, topic=t, representation="c-tf-idf")]
        for t in range(tv.num_topics)
    ]
    centroid_words = [
        [w for w, _ in tv.top_words(BLOCK, topic=t, representation="centroid")]
        for t in range(tv.num_topics)
    ]
    return labels, ctfidf_words, centroid_words, int(tv.num_topics)


def _block_purity(top_words_per_topic) -> float:
    return tc._block_purity(top_words_per_topic)


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not tc.bertopic_available():
        print("bertopic / umap / hdbscan not available; cannot regenerate.")
        sys.exit(1)

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    docs, texts, doc_emb, word_emb, vocab, truth = tc.make_data()

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    hdbscan_model = HDBSCAN(min_cluster_size=MIN_CLUSTER, prediction_data=True)
    bt = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=False
    )
    bt_topics, _ = bt.fit_transform(texts, embeddings=doc_emb)
    bt_labels = np.array(bt_topics)
    bt_info = bt.get_topics()
    bt_words = [[w for w, _ in bt_info[t][:BLOCK]] for t in sorted(bt_info) if t != -1]

    bt_mask = bt_labels >= 0
    bt_truth_ari = _ari(truth[bt_mask], bt_labels[bt_mask])
    bt_purity = _block_purity(bt_words)
    bt_num_topics = int(len([t for t in set(bt_labels) if t >= 0]))

    # topica summary at regenerate time for the provenance log.
    t_labels, t_words, t_centroid_words, t_num = _topica_fit(docs, doc_emb, word_emb, vocab)
    t_mask = t_labels >= 0
    t_truth_ari = _ari(truth[t_mask], t_labels[t_mask])
    cross_ari = _ari(t_labels, bt_labels)

    harness.save_gold(
        NAME,
        arrays={
            "bertopic_labels": bt_labels.astype(np.int64),
            "truth": truth.astype(np.int64),
            "doc_embeddings": doc_emb.astype(np.float64),
            "word_embeddings": word_emb.astype(np.float64),
            "vocab": np.array(vocab, dtype=object),
            "corpus": harness.docs_to_lines(docs),
        },
        meta={
            "reference": _bertopic_version(),
            "model": "Top2Vec (Angelov 2020) vs BERTopic (Grootendorst 2022)",
            "corpus": (
                f"synthetic planted clusters from parity/top2vec_compare.py make_data(): "
                f"{N_CLUSTERS} clusters, {tc.N_DOCS} docs, {tc.EMB_DIM}-dim embeddings, "
                f"{BLOCK} words/block; embeddings frozen into the npz"
            ),
            "num_docs": tc.N_DOCS,
            "vocab_size": len(vocab),
            "emb_dim": tc.EMB_DIM,
            "n_clusters": N_CLUSTERS,
            "block": BLOCK,
            "min_cluster_size": MIN_CLUSTER,
            "bertopic_truth_ari": bt_truth_ari,
            "bertopic_block_purity": bt_purity,
            "bertopic_num_topics": bt_num_topics,
            "topica_truth_ari": t_truth_ari,
            "topica_block_purity": _block_purity(t_words),
            "topica_centroid_block_purity": _block_purity(t_centroid_words),
            "topica_num_topics": t_num,
            "cross_ari": cross_ari,
            "ari_margin": ARI_MARGIN,
            "cross_ari_min": CROSS_ARI_MIN,
            "truth_ari_min": TRUTH_ARI_MIN,
            "purity_min": PURITY_MIN,
            "date": datetime.date.today().isoformat(),
            "metric_kind": "clustering-agreement (ARI), NOT topic-word cosine",
            "pass_bar": (
                "topica truth-ARI >= max(TRUTH_ARI_MIN, frozen-bertopic truth-ARI - "
                "ARI_MARGIN); cross-ARI vs frozen bertopic labels >= CROSS_ARI_MIN; "
                "topica block purity >= PURITY_MIN (verbatim from tests/test_top2vec_parity.py)"
            ),
            "kind": "cross-implementation (BERTopic reference, clustering bar)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  bertopic truth ARI : {bt_truth_ari:.4f}  block purity {bt_purity:.4f}")
    print(f"  topica   truth ARI : {t_truth_ari:.4f}  cross ARI {cross_ari:.4f}")


def _bertopic_version() -> str:
    try:
        import bertopic
        return f"bertopic {getattr(bertopic, '__version__', '?')} (UMAP+HDBSCAN)"
    except Exception:
        return "bertopic (version unknown)"


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    bt_labels = arrays["bertopic_labels"].astype(np.int64)
    truth = arrays["truth"].astype(np.int64)
    doc_emb = arrays["doc_embeddings"].astype(np.float64)
    word_emb = arrays["word_embeddings"].astype(np.float64)
    vocab = [str(w) for w in arrays["vocab"]]
    docs = harness.lines_to_docs(str(arrays["corpus"]))

    bt_truth_ari = float(meta["bertopic_truth_ari"])

    t_labels, t_words, t_centroid_words, t_num = _topica_fit(docs, doc_emb, word_emb, vocab)
    t_mask = t_labels >= 0
    t_truth_ari = _ari(truth[t_mask], t_labels[t_mask])
    cross_ari = _ari(t_labels, bt_labels)
    purity = _block_purity(t_words)
    centroid_purity = _block_purity(t_centroid_words)

    ari_bar = max(TRUTH_ARI_MIN, bt_truth_ari - ARI_MARGIN)
    passes = (
        t_truth_ari >= ari_bar
        and cross_ari >= CROSS_ARI_MIN
        and purity >= PURITY_MIN
        and centroid_purity >= PURITY_MIN
        and 2 <= t_num <= 8
    )
    result = {
        "topica_truth_ari": t_truth_ari,
        "bertopic_truth_ari": bt_truth_ari,
        "cross_ari": cross_ari,
        "topica_block_purity": purity,
        "topica_centroid_block_purity": centroid_purity,
        "topica_num_topics": t_num,
        "ari_bar": ari_bar,
        "margin_over_ari_bar": t_truth_ari - ari_bar,
        "passes": bool(passes),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}  (metric: {meta.get('metric_kind')})")
        print(f"  topica truth ARI {t_truth_ari:.4f} (bar {ari_bar:.4f}, "
              f"bertopic {bt_truth_ari:.4f}); cross ARI {cross_ari:.4f}; "
              f"block purity {purity:.4f} (centroid {centroid_purity:.4f}); topics {t_num}")
        print(f"  verdict: {'PASS' if passes else 'FAIL'}")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
