"""Committed-gold parity for topica BERTopic vs the `bertopic` package (#271, final wave).

topica's ``BERTopic`` is an independent reimplementation of the BERTopic algorithm
(Grootendorst 2022): reduce the document embeddings, density-cluster, and define each
topic by class-based TF-IDF over its documents' words. It shares the algorithm's shape
with the ``bertopic`` package but not the reducer — topica uses a pure-Rust randomized
PCA + HDBSCAN, while the package uses UMAP + HDBSCAN. Exact agreement is impossible, so
— exactly as the live ``parity/top2vec_compare.py`` does — we hold them to a
statistical-equivalence bar on a controlled planted-cluster task and ask how well each
recovers the planted structure.

HONEST NOTE ON THE BAR. BERTopic is a *clustering* topic model, so the gate is a
clustering-agreement metric, NOT a topic-word cosine: the adjusted Rand index (ARI) of
each model's partition against the planted truth, the cross-ARI between the two
implementations, and per-topic c-TF-IDF block purity. This mirrors the Top2Vec gold
(``parity/top2vec_gold.py``) and the live ``tests/test_top2vec_parity.py`` bar.

The corpus + embeddings + config are taken verbatim from ``parity/top2vec_compare.py``
(``make_data(seed=0)``: 4 planted clusters, 320 docs, 12-dim synthetic embeddings, 6
vocabulary words per block). The embeddings are synthetic and seeded, so fully
reproducible; we still freeze them into the npz so the offline topica refit is
byte-identical to what the ``bertopic`` package saw.

topica's BERTopic.fit() needs NO CI-absent backend: its reducer ("pca") and clusterer
("hdbscan") are implemented in Rust, and you bring the document embeddings yourself —
so the offline gold test runs in CI with numpy/scipy/topica only. (The package's
BERTopic, by contrast, pulls in umap + hdbscan + scikit-learn; those are imported ONLY
at --regenerate time.)

Two phases (mirrors parity/top2vec_gold.py):

  * ``--regenerate`` (needs bertopic + umap + hdbscan): builds the planted data, fits
    the `bertopic` package on the shared embeddings, and freezes its recovered labels +
    truth-recovery ARI / block purity, the planted truth, the synthetic doc and word
    embeddings, the vocab, and the corpus into the committed gold
    (``parity/bertopic_gold.npz`` + ``.json``).
  * default (no bertopic / umap / hdbscan): loads the committed gold, fits topica
    BERTopic on the same frozen embeddings, and checks topica recovers the planted
    truth at least as well as the frozen package run (within margin), agrees with its
    frozen partition, and keeps its top-word block purity.

Run directly::

    python parity/bertopic_gold.py               # offline compare against committed gold
    python parity/bertopic_gold.py --regenerate  # run the bertopic package once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import top2vec_compare as tc  # noqa: E402

NAME = "bertopic"

N_CLUSTERS = tc.N_CLUSTERS  # 4
BLOCK = tc.BLOCK            # 6
MIN_CLUSTER = 15
# Bars, verbatim from the live tests/test_top2vec_parity.py (same clustering gate).
ARI_MARGIN = 0.2           # topica truth-ARI >= reference truth-ARI - margin
CROSS_ARI_MIN = 0.4
TRUTH_ARI_MIN = 0.5
PURITY_MIN = 0.9


def _ari(a, b) -> float:
    # numpy-only ARI (harness) so the offline gold test needs no scikit-learn,
    # which CI does not install (sklearn is only a regenerate-time reference).
    return harness.adjusted_rand_index(a, b)


def _block_purity(top_words_per_topic) -> float:
    return tc._block_purity(top_words_per_topic)


def _topica_fit(docs, doc_emb):
    """Fit topica BERTopic on the shared embeddings; return (labels, top_words, num)."""
    import topica

    # Pin reducer="pca" so this gold stays the deterministic, Rust-PCA fixture the
    # provenance below documents, independent of BERTopic's default reducer (which
    # is now "umap"); the committed frozen provenance was computed under PCA.
    bt = topica.BERTopic(n_components=5, min_cluster_size=MIN_CLUSTER, reducer="pca", seed=42)
    bt.fit(docs, doc_emb)
    labels = np.array(bt.labels)
    words = [
        [w for w, _ in bt.top_words(BLOCK, topic=t)]
        for t in range(bt.num_topics)
    ]
    return labels, words, int(bt.num_topics)


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
    ref_labels = np.array(bt_topics)
    bt_info = bt.get_topics()
    ref_words = [[w for w, _ in bt_info[t][:BLOCK]] for t in sorted(bt_info) if t != -1]

    ref_mask = ref_labels >= 0
    ref_truth_ari = _ari(truth[ref_mask], ref_labels[ref_mask])
    ref_purity = _block_purity(ref_words)
    ref_num_topics = int(len([t for t in set(ref_labels) if t >= 0]))

    # topica summary at regenerate time for the provenance log.
    t_labels, t_words, t_num = _topica_fit(docs, doc_emb)
    t_mask = t_labels >= 0
    t_truth_ari = _ari(truth[t_mask], t_labels[t_mask])
    cross_ari = _ari(t_labels, ref_labels)

    harness.save_gold(
        NAME,
        arrays={
            "reference_labels": ref_labels.astype(np.int64),
            "truth": truth.astype(np.int64),
            "doc_embeddings": doc_emb.astype(np.float64),
            "word_embeddings": word_emb.astype(np.float64),
            "vocab": np.array(vocab, dtype=object),
            "corpus": harness.docs_to_lines(docs),
        },
        meta={
            "reference": _bertopic_version(),
            "model": "BERTopic (Grootendorst 2022): topica reimpl vs the bertopic package",
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
            "reference_truth_ari": ref_truth_ari,
            "reference_block_purity": ref_purity,
            "reference_num_topics": ref_num_topics,
            "topica_truth_ari": t_truth_ari,
            "topica_block_purity": _block_purity(t_words),
            "topica_num_topics": t_num,
            "cross_ari": cross_ari,
            "ari_margin": ARI_MARGIN,
            "cross_ari_min": CROSS_ARI_MIN,
            "truth_ari_min": TRUTH_ARI_MIN,
            "purity_min": PURITY_MIN,
            "date": datetime.date.today().isoformat(),
            "metric_kind": "clustering-agreement (ARI), NOT topic-word cosine",
            "topica_backend": (
                "topica.BERTopic.fit needs NO CI-absent package (Rust PCA+HDBSCAN; "
                "you supply doc_embeddings); the bertopic package needs umap+hdbscan+sklearn"
            ),
            "pass_bar": (
                "topica truth-ARI >= max(TRUTH_ARI_MIN, frozen-reference truth-ARI - "
                "ARI_MARGIN); cross-ARI vs frozen reference labels >= CROSS_ARI_MIN; "
                "topica block purity >= PURITY_MIN (verbatim from tests/test_top2vec_parity.py)"
            ),
            "kind": "cross-implementation (bertopic package reference, clustering bar)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  bertopic-pkg truth ARI : {ref_truth_ari:.4f}  block purity {ref_purity:.4f}")
    print(f"  topica       truth ARI : {t_truth_ari:.4f}  cross ARI {cross_ari:.4f}")


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
    ref_labels = arrays["reference_labels"].astype(np.int64)
    truth = arrays["truth"].astype(np.int64)
    doc_emb = arrays["doc_embeddings"].astype(np.float64)
    docs = harness.lines_to_docs(str(arrays["corpus"]))

    ref_truth_ari = float(meta["reference_truth_ari"])

    t_labels, t_words, t_num = _topica_fit(docs, doc_emb)
    t_mask = t_labels >= 0
    t_truth_ari = _ari(truth[t_mask], t_labels[t_mask])
    cross_ari = _ari(t_labels, ref_labels)
    purity = _block_purity(t_words)

    ari_bar = max(TRUTH_ARI_MIN, ref_truth_ari - ARI_MARGIN)
    passes = (
        t_truth_ari >= ari_bar
        and cross_ari >= CROSS_ARI_MIN
        and purity >= PURITY_MIN
        and 2 <= t_num <= 8
    )
    result = {
        "topica_truth_ari": t_truth_ari,
        "reference_truth_ari": ref_truth_ari,
        "cross_ari": cross_ari,
        "topica_block_purity": purity,
        "topica_num_topics": t_num,
        "ari_bar": ari_bar,
        "margin_over_ari_bar": t_truth_ari - ari_bar,
        "passes": bool(passes),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}  (metric: {meta.get('metric_kind')})")
        print(f"  topica truth ARI {t_truth_ari:.4f} (bar {ari_bar:.4f}, "
              f"reference {ref_truth_ari:.4f}); cross ARI {cross_ari:.4f}; "
              f"block purity {purity:.4f}; topics {t_num}")
        print(f"  verdict: {'PASS' if passes else 'FAIL'}")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
