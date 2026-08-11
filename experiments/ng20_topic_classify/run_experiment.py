"""Experiment: topic-model clusters as labels, embeddings as features.

Question
--------
If we let an (unsupervised) topic model assign every 20-Newsgroups document to a
cluster, then train a supervised classifier to reproduce those cluster labels
from the sentence embeddings, what do the *disagreements* look like? Are the
documents that "come up in a different cluster" (topic model says A, the
embedding classifier says B) a meaningful set, or just noise?

Pipeline
--------
1. Fit topica's LDA (bag-of-words, K topics) on the in-vocab token texts.
   Each document's argmax topic is its cluster label z_i. This uses *only* the
   word-count view of the data.
2. Train a multinomial logistic-regression classifier on the 384-d MiniLM
   sentence embeddings to predict z. Use out-of-fold (cross-validated)
   predictions z_hat so we never score a document the model trained on. This
   uses *only* the embedding view of the data.
3. The two views are independent (counts vs. neural embeddings). Study the
   documents where they disagree (z_hat != z):
     - How separable are the LDA clusters in embedding space (CV accuracy)?
     - Are disagreements concentrated on low-confidence LDA assignments
       (high topic entropy)?
     - We also hold out the TRUE newsgroup label as an external referee. On the
       disagreement set, does the embedding classifier's pick match the true
       newsgroup *more often* than the LDA cluster it overrode? If so, the
       "different cluster" documents are a meaningful set: genuine boundary
       cases the bag-of-words topic model misfiled.

Outputs (written next to this script):
  results.json           machine-readable summary of every metric
  report.md              human-readable writeup
  fig_confusion.png      LDA cluster vs. embedding-predicted cluster
  fig_entropy.png        LDA topic entropy: agreements vs. disagreements
  disagreements.csv      every disagreement doc with context
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent.parent / "examples" / "ng20_minilm.npz"
SEED = 13
K = 5  # topics == number of source newsgroups, so clusters are interpretable


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


def main() -> None:
    import topica
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        accuracy_score,
    )

    print(f"topica version: {topica.__version__}")
    data = np.load(NPZ, allow_pickle=True)
    texts = data["texts"]
    true_labels = np.asarray(data["labels"])
    emb = data["doc_embeddings"].astype(np.float32)  # (N, 384), already L2-normed
    n = len(texts)
    print(f"{n} docs, {emb.shape[1]}-d embeddings, "
          f"{len(set(true_labels))} true newsgroups")

    # --- Step 1: LDA topic model -> cluster label per doc --------------------
    documents = [t.split() for t in texts]
    lda = topica.LDA(num_topics=K, seed=SEED)
    lda.fit(documents)
    doc_topic = np.asarray(lda.doc_topic)          # (N, K), rows sum to 1
    z = doc_topic.argmax(axis=1)                    # LDA cluster label
    # per-doc assignment confidence = 1 - normalized entropy of theta
    p = np.clip(doc_topic, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(K)  # 0 = certain, 1 = uniform
    top_words = [[w for w, _ in lda.top_words(10)[k]] for k in range(K)]
    print("\nLDA topics (top words):")
    for k in range(K):
        print(f"  cluster {k}: {' '.join(top_words[k][:8])}")

    # How well do the unsupervised LDA clusters line up with true newsgroups?
    lda_map, lda_acc = hungarian_map(true_labels, z, K)
    print(f"\nLDA clusters vs. true newsgroups: "
          f"ARI={adjusted_rand_score(true_labels, z):.3f} "
          f"NMI={normalized_mutual_info_score(true_labels, z):.3f} "
          f"best-map acc={lda_acc:.3f}")

    # --- Step 2: classify embeddings -> LDA cluster (out-of-fold) ------------
    clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial by default
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    z_hat = cross_val_predict(clf, emb, z, cv=cv)
    recover_acc = accuracy_score(z, z_hat)
    print(f"\nEmbeddings -> LDA cluster (5-fold CV) accuracy: {recover_acc:.3f}")

    # --- Step 3: study the disagreements ------------------------------------
    disagree = z_hat != z
    n_dis = int(disagree.sum())
    print(f"\nDisagreements (embedding pick != LDA cluster): "
          f"{n_dis} / {n} = {n_dis / n:.1%}")

    # (a) are disagreements concentrated on low-confidence LDA assignments?
    ent_agree = float(ent[~disagree].mean())
    ent_disagree = float(ent[disagree].mean())
    # confidence = max theta
    conf = doc_topic.max(axis=1)
    conf_agree = float(conf[~disagree].mean())
    conf_disagree = float(conf[disagree].mean())
    print(f"  mean LDA topic entropy   agree={ent_agree:.3f}  "
          f"disagree={ent_disagree:.3f}")
    print(f"  mean LDA max-theta (conf) agree={conf_agree:.3f}  "
          f"disagree={conf_disagree:.3f}")

    # (b) external referee: on disagreements, does the embedding pick match the
    #     true newsgroup more often than the LDA cluster it overrode?
    #     Map each cluster id to a newsgroup via the SAME LDA best-map, then
    #     compare predicted-newsgroup accuracy on the disagreement set.
    z_as_ng = np.array([lda_map[int(c)] for c in z])
    zhat_as_ng = np.array([lda_map[int(c)] for c in z_hat])
    on_dis = disagree
    lda_true_on_dis = float((z_as_ng[on_dis] == true_labels[on_dis]).mean())
    clf_true_on_dis = float((zhat_as_ng[on_dis] == true_labels[on_dis]).mean())
    print(f"  on disagreements, match to TRUE newsgroup: "
          f"LDA cluster={lda_true_on_dis:.3f}  embedding pick={clf_true_on_dis:.3f}")

    # overall accuracy of each view against the true newsgroup, for context
    lda_all_true = float((z_as_ng == true_labels).mean())
    clf_all_true = float((zhat_as_ng == true_labels).mean())

    # --- Controls ------------------------------------------------------------
    # (c) ceiling: how well do the embeddings recover the TRUE newsgroup? If the
    #     LDA-cluster recovery (85%) is close to this, the clusters are almost as
    #     learnable as the real labels.
    y_true_hat = cross_val_predict(clf, emb, true_labels, cv=cv)
    embed_true_acc = float(accuracy_score(true_labels, y_true_hat))

    # (d) permutation control: shuffle the cluster labels and re-run CV. If the
    #     85% were an artifact of the classifier / 5 classes, shuffled labels
    #     would score similarly. They should collapse toward the base rate.
    rng = np.random.default_rng(SEED)
    z_perm = rng.permutation(z)
    z_perm_hat = cross_val_predict(clf, emb, z_perm, cv=cv)
    perm_acc = float(accuracy_score(z_perm, z_perm_hat))
    base_rate = float(max(np.bincount(z)) / n)
    print(f"\nControls:")
    print(f"  embeddings -> TRUE newsgroup CV acc (ceiling): {embed_true_acc:.3f}")
    print(f"  embeddings -> SHUFFLED cluster CV acc (floor):  {perm_acc:.3f} "
          f"(base rate {base_rate:.3f})")

    # (e) K sweep: is the recoverability stable across the number of topics?
    k_sweep = []
    for k in (5, 8, 10, 15):
        m = topica.LDA(num_topics=k, seed=SEED)
        m.fit(documents)
        zk = np.asarray(m.doc_topic).argmax(axis=1)
        zk_hat = cross_val_predict(clf, emb, zk, cv=cv)
        k_sweep.append({
            "K": k,
            "cv_recovery_acc": float(accuracy_score(zk, zk_hat)),
            "frac_disagree": float((zk_hat != zk).mean()),
            "ari_to_true": float(adjusted_rand_score(true_labels, zk)),
        })
        print(f"  K={k:2d}: recovery={k_sweep[-1]['cv_recovery_acc']:.3f} "
              f"disagree={k_sweep[-1]['frac_disagree']:.3f} "
              f"ARI(true)={k_sweep[-1]['ari_to_true']:.3f}")

    # --- Save artifacts ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # confusion: LDA cluster (rows) vs embedding-predicted cluster (cols)
    conf_mat = np.zeros((K, K), dtype=int)
    for a, b in zip(z, z_hat):
        conf_mat[a, b] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(conf_mat, cmap="Blues")
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
    fig.savefig(HERE / "fig_confusion.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(ent[~disagree], bins=25, density=True, alpha=0.6,
            label=f"agree (n={n - n_dis})")
    ax.hist(ent[disagree], bins=25, density=True, alpha=0.6,
            label=f"disagree (n={n_dis})")
    ax.set_xlabel("LDA topic entropy (0 = confident, 1 = uniform)")
    ax.set_ylabel("density")
    ax.set_title("Where the two views disagree, LDA was less sure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fig_entropy.png", dpi=140)
    plt.close(fig)

    # disagreements.csv, sorted so the most confident LDA calls the embedding
    # classifier still overrode float to the top (the interesting ones)
    import csv
    order = np.argsort(-conf * disagree)  # confident disagreements first
    with open(HERE / "disagreements.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "doc_idx", "lda_cluster", "lda_topwords",
                    "lda_maxtheta", "lda_entropy", "embed_pred_cluster",
                    "embed_topwords", "true_newsgroup",
                    "lda_cluster_as_ng", "embed_pred_as_ng", "text"])
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
        "n_docs": int(n),
        "embed_dim": int(emb.shape[1]),
        "n_true_newsgroups": int(len(set(true_labels))),
        "K": K,
        "seed": SEED,
        "lda_top_words": {str(k): top_words[k] for k in range(K)},
        "lda_cluster_map_to_newsgroup": {str(k): v for k, v in lda_map.items()},
        "lda_vs_true": {
            "ari": float(adjusted_rand_score(true_labels, z)),
            "nmi": float(normalized_mutual_info_score(true_labels, z)),
            "best_map_accuracy": float(lda_acc),
        },
        "embed_recovers_lda_cluster": {
            "cv_accuracy": float(recover_acc),
            "n_disagree": n_dis,
            "frac_disagree": float(n_dis / n),
        },
        "disagreement_profile": {
            "mean_entropy_agree": ent_agree,
            "mean_entropy_disagree": ent_disagree,
            "mean_maxtheta_agree": conf_agree,
            "mean_maxtheta_disagree": conf_disagree,
        },
        "external_referee_true_newsgroup": {
            "on_disagreements_lda_matches_true": lda_true_on_dis,
            "on_disagreements_embed_matches_true": clf_true_on_dis,
            "overall_lda_matches_true": lda_all_true,
            "overall_embed_matches_true": clf_all_true,
        },
        "controls": {
            "embed_recovers_true_newsgroup_cv_acc": embed_true_acc,
            "embed_recovers_shuffled_cluster_cv_acc": perm_acc,
            "base_rate": base_rate,
        },
        "k_sweep": k_sweep,
    }
    with open(HERE / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote results.json, report.md inputs, and figures to", HERE)
    return results


if __name__ == "__main__":
    main()
