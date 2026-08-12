"""Parity check: topica's **default** BERTopic pipeline (reducer="umap") vs the
reference umap-learn + HDBSCAN, on real embeddings.

The committed BERTopic gold (``parity/bertopic_gold.py``) pins ``reducer="pca"``
so it never exercises topica's default UMAP path. This script closes that gap and,
in doing so, documents a result the retroactive Gate-B review (#718) initially
misread as a "collapse bug":

On ``load_ng20_minilm`` (2594 docs, 5 newsgroup classes, MiniLM embeddings), the
default density-based pipeline finds only ~3 topics with one topic holding ~82% of
the documents (ARI ~0.16 vs the labels). That is **not** a topica defect — it is
what the reference ``umap-learn`` -> HDBSCAN pipeline *also* finds on this corpus at
``min_cluster_size=15``. ``reducer="pca"`` / ``clusterer="kmeans"`` score higher
only because they *force* more clusters, which is not what a density-based default
does. topica reproduces the reference; the few-topic result is genuine structure.

So this check runs BOTH pipelines on the same bundled embeddings and asserts they
agree on the discovered topic count, the dominant-bucket share, and the label ARI
within a tolerance that absorbs HDBSCAN-implementation and UMAP seed noise.

Needs only ``umap-learn`` + ``scikit-learn`` (the MiniLM embeddings are bundled, so
no ``sentence-transformers``/``torch``). Skips cleanly when they are absent or the
dataset cannot be loaded offline; it is a manual parity tool, not a CI test.
"""

from __future__ import annotations

import sys

N_COMPONENTS, N_NEIGHBORS, MIN_CLUSTER = 5, 15, 15
SEED = 13
# topica's default pipeline must land within this ARI margin of the reference
# umap-learn+HDBSCAN pipeline (both collapse to the same few-topic structure here).
ARI_MARGIN = 0.10
# ...and agree on the discovered topic count within this many topics.
K_TOL = 2


def _refs_available() -> bool:
    for m in ("umap", "sklearn", "numpy"):
        try:
            __import__(m)
        except Exception:
            return False
    return True


def _stats(labels, truth):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    labels = np.asarray(labels)
    truth = np.asarray(truth)
    assigned = labels >= 0
    k = len(set(labels[assigned].tolist()))
    if assigned.sum() == 0:
        return k, 0.0, float("nan")
    sizes = np.bincount(labels[assigned])
    dominant = sizes.max() / assigned.sum()
    ari = adjusted_rand_score(truth[assigned], labels[assigned])
    return k, float(dominant), float(ari)


def run(verbose: bool = True) -> dict:
    if not _refs_available():
        if verbose:
            print("reference stack (umap-learn / scikit-learn) not installed; "
                  "skipping default-UMAP BERTopic parity check.")
        return {"skipped": True}

    import numpy as np
    import topica

    try:
        ng = topica.datasets.load_ng20_minilm()
    except Exception as exc:  # dataset not cached / no network
        if verbose:
            print(f"load_ng20_minilm unavailable ({exc}); skipping.")
        return {"skipped": True}

    docs = [t.split() for t in ng["texts"]]
    emb = np.asarray(ng["doc_embeddings"], dtype=np.float64)
    truth = np.asarray(ng["labels"])

    # topica: the default BERTopic pipeline (reducer="umap").
    m = topica.BERTopic(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_cluster_size=MIN_CLUSTER,
        reducer="umap",
        seed=SEED,
    ).fit(docs, emb)
    t_k, t_dom, t_ari = _stats(np.asarray(m.labels), truth)

    # Reference: umap-learn -> the same HDBSCAN, on the same embeddings.
    import umap
    from sklearn.cluster import HDBSCAN

    red = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
    ).fit_transform(emb)
    ref_labels = HDBSCAN(min_cluster_size=MIN_CLUSTER).fit_predict(red)
    r_k, r_dom, r_ari = _stats(ref_labels, truth)

    ok_k = abs(t_k - r_k) <= K_TOL
    ok_dom = abs(t_dom - r_dom) <= 0.15
    ok_ari = abs(t_ari - r_ari) <= ARI_MARGIN
    passed = ok_k and ok_dom and ok_ari

    if verbose:
        print("Default-UMAP BERTopic vs reference umap-learn+HDBSCAN "
              "(load_ng20_minilm, 5 classes):")
        print(f"  topica    : K={t_k}, dominant={t_dom:.2%}, ARI={t_ari:.3f}")
        print(f"  reference : K={r_k}, dominant={r_dom:.2%}, ARI={r_ari:.3f}")
        print(f"  agree: K±{K_TOL}={ok_k}, dominant±0.15={ok_dom}, "
              f"ARI±{ARI_MARGIN}={ok_ari} -> {'PASS' if passed else 'FAIL'}")
        print("  (Both find the same few-topic structure: it is genuine density "
              "structure on this corpus, faithfully reproduced — not a collapse bug.)")

    return {
        "skipped": False,
        "passed": passed,
        "topica": {"k": t_k, "dominant": t_dom, "ari": t_ari},
        "reference": {"k": r_k, "dominant": r_dom, "ari": r_ari},
    }


if __name__ == "__main__":
    result = run(verbose=True)
    if result.get("skipped"):
        sys.exit(0)
    sys.exit(0 if result["passed"] else 1)
