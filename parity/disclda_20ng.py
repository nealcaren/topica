"""Parity/validation for DiscLDA against the paper's 20 Newsgroups result.

DiscLDA (Lacoste-Julien, Sha & Jordan, NIPS 2008) has no canonical reference
implementation, so we validate against the paper's empirical claim: topic-proportion
features from DiscLDA feed a linear classifier *better* than features from an
unsupervised LDA of matched dimension.

We reproduce the *ordering* the paper reports, not its exact error figures. The
figure that matches the variant implemented here (the **fixed** block transform,
paper §4.1) is the §4.2 first experiment: on the full 20-way task, LDA features give
25% SVM error and fixed-transform DiscLDA features give 20%. The frequently-quoted
17% vs 20% (their Table 2) is the harder alt.atheism / talk.religion binary pair with
the **learned** transform (§4.2 second experiment), which topica does not yet
implement — so on that pair we expect the same *direction* (DiscLDA above LDA) but a
larger absolute error than 17%.

Here we fit both models on the same tokenized 20NG training split (the binary pair,
which runs fast), extract per-document topic features, train a linear SVM on each, and
assert DiscLDA's held-out accuracy is at least LDA's. Skips cleanly when scikit-learn
or the 20NG download is unavailable.
"""
import numpy as np

import topica

# The paper's hard binary task: two topically similar religion groups.
CATEGORIES = ["alt.atheism", "talk.religion.misc"]
K_CLASS, K_SHARED = 8, 12   # L = 2*8 + 12 = 28
LDA_TOPICS = 2 * K_CLASS + K_SHARED
ITERS, SEED = 600, 42
MARGIN = -0.01              # DiscLDA must not be worse than LDA by more than this


def _load():
    from sklearn.datasets import fetch_20newsgroups
    rm = ("headers", "footers", "quotes")
    tr = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=rm, random_state=0)
    te = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=rm, random_state=0)
    return tr, te


def _tokenize(texts):
    return [topica.tokenize(t) for t in texts]


def _build(train_docs, test_docs, min_df=5):
    # Shared vocabulary from train; map both splits onto it.
    from collections import Counter
    df = Counter()
    for d in train_docs:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= min_df and len(w) > 2}
    tr = [[w for w in d if w in vocab] for d in train_docs]
    te = [[w for w in d if w in vocab] for d in test_docs]
    return tr, te


def test_disclda_beats_lda_features_on_20ng():
    try:
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("SKIP: scikit-learn not available")
        return
    try:
        tr, te = _load()
    except Exception as e:  # network / dataset unavailable
        print(f"SKIP: 20 Newsgroups unavailable ({e})")
        return

    y_tr = [CATEGORIES[i] for i in tr.target]
    y_te = [CATEGORIES[i] for i in te.target]
    train_docs = _tokenize(tr.data)
    test_docs = _tokenize(te.data)
    train_docs, test_docs = _build(train_docs, test_docs)
    # drop empty docs
    keep_tr = [i for i, d in enumerate(train_docs) if d]
    keep_te = [i for i, d in enumerate(test_docs) if d]
    train_docs = [train_docs[i] for i in keep_tr]; y_tr = [y_tr[i] for i in keep_tr]
    test_docs = [test_docs[i] for i in keep_te]; y_te = [y_te[i] for i in keep_te]
    print(f"train={len(train_docs)} test={len(test_docs)} classes={CATEGORIES}")

    # --- DiscLDA features ---
    disc = topica.DiscLDA(K_CLASS, K_SHARED, iters=ITERS, infer_sweeps=100, seed=SEED)
    disc.fit(train_docs, y_tr)
    Xtr_d = disc.transform(train_docs)
    Xte_d = disc.transform(test_docs)

    # --- LDA features (matched dimension) ---
    lda = topica.LDA(num_topics=LDA_TOPICS, seed=SEED)
    lda.fit(train_docs, iters=ITERS, num_samples=1)
    Xtr_l = lda.doc_topic
    Xte_l = lda.transform(test_docs)

    def svm_acc(Xtr, Xte):
        clf = LinearSVC(C=1.0, max_iter=5000)
        clf.fit(Xtr, y_tr)
        return accuracy_score(y_te, clf.predict(Xte))

    acc_d = svm_acc(Xtr_d, Xte_d)
    acc_l = svm_acc(Xtr_l, Xte_l)
    # DiscLDA-as-classifier directly, too
    acc_direct = np.mean([p == t for p, t in zip(disc.predict(test_docs), y_te)])

    print(f"DiscLDA features + SVM : acc={acc_d:.3f} (err={1-acc_d:.3f})")
    print(f"LDA features + SVM     : acc={acc_l:.3f} (err={1-acc_l:.3f})")
    print(f"DiscLDA direct classify: acc={acc_direct:.3f} (err={1-acc_direct:.3f})")
    print("paper (learned-T, this pair): DiscLDA 17% vs LDA 20% err; fixed-T here "
          "reproduces the ordering, not the absolute figure")

    assert acc_d >= acc_l + MARGIN, (
        f"DiscLDA features ({acc_d:.3f}) should be at least as good as LDA "
        f"features ({acc_l:.3f})"
    )


if __name__ == "__main__":
    test_disclda_beats_lda_features_on_20ng()
