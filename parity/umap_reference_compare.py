"""Parity check: topica's in-house UMAP reducer vs the reference umap-learn.

Issue #343 reported that the old `reducer="umap"` (delegating to the umap-rs
crate) recovered ~0.11-0.19 ARI worse cluster structure than umap-learn on real
sentence embeddings, because umap-rs's layout gradient carried an extra
`dist_squared` factor in its attractive/repulsive denominators. topica now ships
a from-scratch UMAP faithful to umap-learn (src/umap.rs); this script measures
that it reaches umap-learn's cluster quality on the reporter's setup.

Setup: 20 Newsgroups, a balanced 6-category slice, all-mpnet-base-v2 embeddings,
UMAP to 5-D (cosine, n_neighbors=15, min_dist=0.0), the SAME Euclidean HDBSCAN
(min_cluster_size=30) on both reducers' output, mean over 3 seeds. Metric is the
adjusted Rand index of the discovered partition vs the newsgroup labels.

Skips cleanly if the reference stack (sentence-transformers / umap-learn /
hdbscan / sklearn) is not installed; it is a manual parity tool, not a CI test.
"""

from __future__ import annotations

import sys

CATS = [
    "rec.sport.hockey", "sci.space", "talk.politics.guns",
    "comp.graphics", "sci.med", "soc.religion.christian",
]
N_PER = 250
N_COMPONENTS, N_NEIGHBORS, MIN_CLUSTER = 5, 15, 30
SEEDS = [0, 1, 2]
# topica must land within this ARI margin of umap-learn (it typically matches or
# slightly exceeds it; the margin absorbs seed-to-seed and HDBSCAN-impl noise).
MARGIN = 0.05


def _available() -> bool:
    import importlib.util as u
    return all(
        u.find_spec(m)
        for m in ("sklearn", "sentence_transformers", "umap", "hdbscan", "numpy")
    )


def _load():
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer

    ng = fetch_20newsgroups(
        subset="all", categories=CATS,
        remove=("headers", "footers", "quotes"), random_state=42,
    )
    rng = np.random.default_rng(42)
    texts, labels = [], []
    for ci in range(len(CATS)):
        idx = np.where(np.array(ng.target) == ci)[0]
        keep = rng.choice(idx, size=min(N_PER, len(idx)), replace=False)
        for i in keep:
            texts.append(ng.data[i]); labels.append(ci)
    model = SentenceTransformer("all-mpnet-base-v2")
    emb = model.encode(texts, batch_size=64).astype("float64")
    return emb, np.array(labels)


def _ari(truth, pred):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    pred = np.asarray(pred)
    m = pred >= 0
    return adjusted_rand_score(truth[m], pred[m]) if m.sum() else 0.0


def main() -> int:
    if not _available():
        print("reference stack (sentence-transformers/umap-learn/hdbscan/sklearn) "
              "not installed; skipping UMAP parity check.")
        return 0

    import numpy as np
    import topica
    from hdbscan import HDBSCAN
    from umap import UMAP

    emb, truth = _load()

    def cluster(red):
        return HDBSCAN(min_cluster_size=MIN_CLUSTER).fit_predict(red)

    topica_ari, ref_ari = [], []
    for s in SEEDS:
        t_red = np.asarray(topica.project(emb, N_COMPONENTS, method="umap",
                                          n_neighbors=N_NEIGHBORS, seed=s))
        r_red = UMAP(n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                     min_dist=0.0, metric="cosine", random_state=s).fit_transform(emb)
        topica_ari.append(_ari(truth, cluster(t_red)))
        ref_ari.append(_ari(truth, cluster(r_red)))

    t_mean, r_mean = float(np.mean(topica_ari)), float(np.mean(ref_ari))
    print(f"topica in-house UMAP ARI: {t_mean:.3f}")
    print(f"reference umap-learn ARI: {r_mean:.3f}")
    print(f"gap (topica - reference): {t_mean - r_mean:+.3f}  (margin {MARGIN})")

    if t_mean < r_mean - MARGIN:
        print("FAIL: topica trails umap-learn by more than the margin.")
        return 1
    print("PASS: topica matches umap-learn cluster quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
