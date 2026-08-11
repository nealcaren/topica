"""Reusable pipeline: topic-model clusters as labels, embeddings as features.

See report.md for the full writeup. This module holds the pipeline so both the
5-group run (``run_experiment.py``) and the full 20-group run (``run_full.py``)
share one implementation.

Given an .npz with ``texts`` (space-joined in-vocab tokens), ``labels`` (true
group), and ``doc_embeddings`` (n, d):

1. Fit topica LDA (K topics) on the token texts -> cluster label z per doc.
2. Cross-validated logistic regression on the embeddings -> predict z (z_hat).
3. Characterize the documents where the two views disagree (z_hat != z), using
   the held-out true label only as an external referee.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def hungarian_map(true_labels: np.ndarray, cluster_labels: np.ndarray, n_clusters: int):
    """Best cluster->class assignment (maximizes matched count). Returns
    (mapping dict cluster->class, accuracy under that mapping)."""
    from scipy.optimize import linear_sum_assignment

    classes = sorted(set(true_labels))
    class_idx = {c: i for i, c in enumerate(classes)}
    cost = np.zeros((n_clusters, len(classes)))
    for cl, tr in zip(cluster_labels, true_labels):
        cost[cl, class_idx[tr]] += 1
    row, col = linear_sum_assignment(-cost)  # maximize overlap
    mapping = {int(r): classes[c] for r, c in zip(row, col)}
    matched = sum(cost[r, c] for r, c in zip(row, col))
    return mapping, matched / len(true_labels)


def run(npz_path, K, outdir, *, seed=13, k_sweep=(), tag=""):
    """Run the full pipeline and write artifacts into ``outdir``.

    Returns the results dict (also written to ``results.json``).
    """
    import topica
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        accuracy_score,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"topica version: {topica.__version__}   [{tag or npz_path}]")

    data = np.load(npz_path, allow_pickle=True)
    texts = data["texts"]
    true_labels = np.asarray(data["labels"])
    emb = data["doc_embeddings"].astype(np.float32)
    n = len(texts)
    n_groups = len(set(true_labels))
    print(f"{n} docs, {emb.shape[1]}-d embeddings, {n_groups} true groups, K={K}")

    # --- Step 1: LDA topic model -> cluster label per doc --------------------
    documents = [t.split() for t in texts]
    lda = topica.LDA(num_topics=K, seed=seed)
    lda.fit(documents)
    doc_topic = np.asarray(lda.doc_topic)
    z = doc_topic.argmax(axis=1)
    p = np.clip(doc_topic, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(K)
    conf = doc_topic.max(axis=1)
    top_words = [[w for w, _ in lda.top_words(10)[k]] for k in range(K)]
    print("\nLDA topics (top words):")
    for k in range(K):
        print(f"  cluster {k:2d}: {' '.join(top_words[k][:8])}")

    lda_map, lda_acc = hungarian_map(true_labels, z, K)
    ari = adjusted_rand_score(true_labels, z)
    nmi = normalized_mutual_info_score(true_labels, z)
    print(f"\nLDA clusters vs. true groups: ARI={ari:.3f} NMI={nmi:.3f} "
          f"best-map acc={lda_acc:.3f}")

    # --- Step 2: classify embeddings -> LDA cluster (out-of-fold) ------------
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    z_hat = cross_val_predict(clf, emb, z, cv=cv)
    recover_acc = accuracy_score(z, z_hat)
    print(f"\nEmbeddings -> LDA cluster (5-fold CV) accuracy: {recover_acc:.3f}")

    # --- Step 3: study the disagreements ------------------------------------
    disagree = z_hat != z
    n_dis = int(disagree.sum())
    print(f"\nDisagreements: {n_dis} / {n} = {n_dis / n:.1%}")

    ent_agree = float(ent[~disagree].mean())
    ent_disagree = float(ent[disagree].mean())
    conf_agree = float(conf[~disagree].mean())
    conf_disagree = float(conf[disagree].mean())
    print(f"  mean LDA entropy   agree={ent_agree:.3f}  disagree={ent_disagree:.3f}")
    print(f"  mean LDA max-theta agree={conf_agree:.3f}  disagree={conf_disagree:.3f}")

    z_as_ng = np.array([lda_map[int(c)] for c in z])
    zhat_as_ng = np.array([lda_map[int(c)] for c in z_hat])
    on_dis = disagree
    lda_true_on_dis = float((z_as_ng[on_dis] == true_labels[on_dis]).mean())
    clf_true_on_dis = float((zhat_as_ng[on_dis] == true_labels[on_dis]).mean())
    lda_all_true = float((z_as_ng == true_labels).mean())
    clf_all_true = float((zhat_as_ng == true_labels).mean())
    print(f"  on disagreements, match TRUE group: LDA={lda_true_on_dis:.3f} "
          f"embed={clf_true_on_dis:.3f}")

    # --- Controls ------------------------------------------------------------
    y_true_hat = cross_val_predict(clf, emb, true_labels, cv=cv)
    embed_true_acc = float(accuracy_score(true_labels, y_true_hat))
    rng = np.random.default_rng(seed)
    z_perm = rng.permutation(z)
    z_perm_hat = cross_val_predict(clf, emb, z_perm, cv=cv)
    perm_acc = float(accuracy_score(z_perm, z_perm_hat))
    base_rate = float(max(np.bincount(z)) / n)
    print(f"\nControls: embed->TRUE={embed_true_acc:.3f}  "
          f"embed->SHUFFLED={perm_acc:.3f} (base {base_rate:.3f})")

    k_sweep_out = []
    for k in k_sweep:
        m = topica.LDA(num_topics=k, seed=seed)
        m.fit(documents)
        zk = np.asarray(m.doc_topic).argmax(axis=1)
        zk_hat = cross_val_predict(clf, emb, zk, cv=cv)
        k_sweep_out.append({
            "K": int(k),
            "cv_recovery_acc": float(accuracy_score(zk, zk_hat)),
            "frac_disagree": float((zk_hat != zk).mean()),
            "ari_to_true": float(adjusted_rand_score(true_labels, zk)),
        })
        print(f"  K={k:2d}: recovery={k_sweep_out[-1]['cv_recovery_acc']:.3f} "
              f"disagree={k_sweep_out[-1]['frac_disagree']:.3f} "
              f"ARI={k_sweep_out[-1]['ari_to_true']:.3f}")

    # --- Figures -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf_mat = np.zeros((K, K), dtype=int)
    for a, b in zip(z, z_hat):
        conf_mat[a, b] += 1
    fig, ax = plt.subplots(figsize=(5.6, 5.0) if K <= 6 else (7.6, 6.8))
    im = ax.imshow(conf_mat, cmap="Blues")
    if K <= 8:
        for i in range(K):
            for j in range(K):
                ax.text(j, i, conf_mat[i, j], ha="center", va="center",
                        color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black",
                        fontsize=9)
    ax.set_xlabel("Embedding-predicted cluster (CV)")
    ax.set_ylabel("LDA-assigned cluster")
    ax.set_title("LDA cluster vs. embedding classifier")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "fig_confusion.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.hist(ent[~disagree], bins=25, density=True, alpha=0.6,
            label=f"agree (n={n - n_dis})")
    ax.hist(ent[disagree], bins=25, density=True, alpha=0.6,
            label=f"disagree (n={n_dis})")
    ax.set_xlabel("LDA topic entropy (0 = confident, 1 = uniform)")
    ax.set_ylabel("density")
    ax.set_title("Where the two views disagree, LDA was less sure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_entropy.png", dpi=140)
    plt.close(fig)

    # --- disagreements.csv ---------------------------------------------------
    order = np.argsort(-conf * disagree)
    with open(outdir / "disagreements.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "doc_idx", "lda_cluster", "lda_topwords",
                    "lda_maxtheta", "lda_entropy", "embed_pred_cluster",
                    "embed_topwords", "true_group", "lda_cluster_as_group",
                    "embed_pred_as_group", "text"])
        rank = 0
        for i in order:
            if not disagree[i]:
                continue
            rank += 1
            w.writerow([
                rank, int(i), int(z[i]), " ".join(top_words[z[i]][:6]),
                f"{conf[i]:.3f}", f"{ent[i]:.3f}", int(z_hat[i]),
                " ".join(top_words[z_hat[i]][:6]), true_labels[i],
                z_as_ng[i], zhat_as_ng[i], texts[i][:300],
            ])

    results = {
        "tag": tag,
        "n_docs": int(n),
        "embed_dim": int(emb.shape[1]),
        "n_true_groups": int(n_groups),
        "K": int(K),
        "seed": seed,
        "lda_top_words": {str(k): top_words[k] for k in range(K)},
        "lda_cluster_map_to_group": {str(k): v for k, v in lda_map.items()},
        "lda_vs_true": {"ari": float(ari), "nmi": float(nmi),
                        "best_map_accuracy": float(lda_acc)},
        "embed_recovers_lda_cluster": {
            "cv_accuracy": float(recover_acc),
            "n_disagree": n_dis,
            "frac_disagree": float(n_dis / n)},
        "disagreement_profile": {
            "mean_entropy_agree": ent_agree,
            "mean_entropy_disagree": ent_disagree,
            "mean_maxtheta_agree": conf_agree,
            "mean_maxtheta_disagree": conf_disagree},
        "external_referee_true_group": {
            "on_disagreements_lda_matches_true": lda_true_on_dis,
            "on_disagreements_embed_matches_true": clf_true_on_dis,
            "overall_lda_matches_true": lda_all_true,
            "overall_embed_matches_true": clf_all_true},
        "controls": {
            "embed_recovers_true_group_cv_acc": embed_true_acc,
            "embed_recovers_shuffled_cluster_cv_acc": perm_acc,
            "base_rate": base_rate},
        "k_sweep": k_sweep_out,
    }
    with open(outdir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote artifacts to", outdir)
    return results
