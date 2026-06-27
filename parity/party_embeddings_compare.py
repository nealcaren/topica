"""Parity check: topica PartyEmbeddings vs the gensim Doc2Vec (PV-DM) reference.

Party Embeddings (Rheault & Cochrane 2020) is gensim `Doc2Vec(dm=1)` with
party-period document tags, scaled by PCA of the tag vectors. gensim is the
reference engine. Both are stochastic SGD with independent RNGs, so the target is
*correlation*, not equality: on a corpus with a planted 1-D party ordering, each
engine's first principal component should recover the plant, and topica's scale
should agree with gensim's about as well as the plant allows.

The gensim reference's own run-to-run agreement on this corpus is |r| ~ 0.99
(measured cross-seed), so the bar is set conservatively below that floor to absorb
the implementation difference (topica is a from-scratch PV-DM, not gensim).

Skips cleanly when gensim is not installed.

    python parity/party_embeddings_compare.py
"""
import shutil  # noqa: F401  (kept for parity-script symmetry)
import sys

import numpy as np

import topica

try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sklearn.decomposition import PCA
    HAVE_GENSIM = True
except Exception:
    HAVE_GENSIM = False


def planted(n_groups=12, docs_per=60, doc_len=18, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.linspace(-1.0, 1.0, n_groups)
    left = [f"L{i}" for i in range(12)]
    right = [f"R{i}" for i in range(12)]
    filler = [f"f{i}" for i in range(40)]
    docs, groups, planted = [], [], {}
    for g in range(n_groups):
        label = f"P{g:02d}"
        planted[label] = float(pos[g])
        pr_right = (pos[g] + 1) / 2.0
        for _ in range(docs_per):
            doc = []
            for _ in range(doc_len):
                if rng.random() < 0.45:
                    pool = right if rng.random() < pr_right else left
                    doc.append(pool[rng.integers(len(pool))])
                else:
                    doc.append(filler[rng.integers(len(filler))])
            docs.append(doc)
            groups.append(label)
    return docs, groups, planted


def gensim_scale(docs, groups, labels, *, vector_size=64, window=5, seed=0):
    tagged = [TaggedDocument(words=d, tags=[g]) for d, g in zip(docs, groups)]
    m = Doc2Vec(dm=1, vector_size=vector_size, window=window, min_count=1,
                negative=5, sample=1e-3, epochs=40, seed=seed, workers=1, hs=0)
    m.build_vocab(tagged)
    m.train(tagged, total_examples=m.corpus_count, epochs=m.epochs)
    tv = np.vstack([m.dv[g] for g in labels])
    pc1 = PCA(n_components=2, svd_solver="full").fit_transform(
        tv - tv.mean(axis=0, keepdims=True))[:, 0]
    return pc1


def main():
    if not HAVE_GENSIM:
        print("SKIP: gensim / scikit-learn not installed (reference unavailable)")
        return 0

    docs, groups, plant_map = planted()
    labels = sorted(set(groups))
    plant = np.array([plant_map[g] for g in labels])

    # topica
    pe = topica.PartyEmbeddings(num_dims=2, vector_size=64, window=5, min_count=1,
                                negative=5, sample=1e-3, learning_rate=0.05, seed=0)
    pe.fit(docs, group=groups, iters=40)
    order = {n: i for i, n in enumerate(pe.author_names)}
    topica_pc1 = np.array([pe.author_positions[order[g], 0] for g in labels])

    # gensim reference
    gens_pc1 = gensim_scale(docs, groups, labels)

    def orient(x):
        return x if np.corrcoef(x, plant)[0, 1] >= 0 else -x

    topica_pc1 = orient(topica_pc1)
    gens_pc1 = orient(gens_pc1)

    r_topica = abs(np.corrcoef(topica_pc1, plant)[0, 1])
    r_gensim = abs(np.corrcoef(gens_pc1, plant)[0, 1])
    r_cross = abs(np.corrcoef(topica_pc1, gens_pc1)[0, 1])

    print(f"topica recovers planted scale:  |r| = {r_topica:.3f}")
    print(f"gensim recovers planted scale:  |r| = {r_gensim:.3f}")
    print(f"topica vs gensim agreement:     |r| = {r_cross:.3f}")

    ok = r_topica > 0.85 and r_cross > 0.80
    if ok:
        print("PASS: topica PartyEmbeddings matches the gensim PV-DM reference scale")
        return 0
    print("FAIL: parity below threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
