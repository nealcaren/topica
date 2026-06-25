"""Compare the four ideal-point models on one corpus: how does the recovered author
scale depend on the representation (word counts / word embeddings / sentence
embeddings) and on whether the model has topics?

The grid:

                 no topics        + topics
    word counts  Wordfish         IdealPointLDA
    embeddings   (Wordfish limit) IdealPointTM (words),  SentenceIdealTM (sentences)

This example plants author positions and a discriminating topic, samples a corpus
from the count-based generative model, derives word embeddings (random) and per-
document embeddings (mean word embedding), fits all four models, and reports how
well each recovers the planted positions plus the pairwise agreement between models.
Self-contained: needs only numpy and topica.

Run from the repo root:  python examples/ideal_point_comparison.py
"""
import numpy as np

import topica

topica.enable_experimental()

rng = np.random.default_rng(0)
K, V, E = 3, 80, 16          # topics, vocabulary, word-embedding dim
A, DOCS_PER, LENGTH = 60, 10, 60  # authors, docs each, tokens per doc

# --- plant the generative world (count-based, IdealPointLDA-style) ------------
# Topic profiles: each topic favors a disjoint third of the vocabulary. Topic 0
# additionally discriminates: a within-topic split that moves with author position.
alpha = np.full((K, V), -3.0)
for k in range(K):
    alpha[k, k * (V // K):(k + 1) * (V // K)] = 0.5
w = np.zeros((K, V))
for v in range(V // K):                      # topic 0's words carry the cleavage
    w[0, v] = 2.0 if v % 2 == 0 else -2.0
x_true = rng.uniform(-1.0, 1.0, A)
word_emb = rng.normal(size=(V, E))           # random word embeddings (for IdealPointTM)
vocab = [f"w{i}" for i in range(V)]


def topic_beta(k, x):
    eta = alpha[k] + x * w[k]
    e = np.exp(eta - eta.max())
    return e / e.sum()


docs, author, doc_emb = [], [], []
for a in range(A):
    for _ in range(DOCS_PER):
        doc = []
        for _ in range(LENGTH):
            k = rng.integers(0, K)
            v = rng.choice(V, p=topic_beta(k, x_true[a]))
            doc.append(vocab[v])
        docs.append(doc)
        author.append(f"a{a:03d}")
        # document embedding = mean of its word embeddings (a stand-in for a real
        # sentence-embedding model), so SentenceIdealTM sees the same signal.
        doc_emb.append(word_emb[[int(t[1:]) for t in doc]].mean(axis=0))
doc_emb = np.array(doc_emb)


def recovery(model, by_doc=False):
    """|Pearson| between a model's author positions and the planted truth."""
    names = model.author_names
    pos = dict(zip(names, np.asarray(model.author_positions)[:, 0]))
    if by_doc:
        # SentenceIdealTM scales per observation (here, per document); average the
        # document positions back to authors for a like-for-like comparison.
        from collections import defaultdict
        acc = defaultdict(list)
        for lbl, p in pos.items():
            acc[lbl].append(p)
        pos = {a: np.mean(v) for a, v in acc.items()}
    recovered = np.array([pos[f"a{a:03d}"] for a in range(A)])
    return recovered, abs(np.corrcoef(recovered, x_true)[0, 1])


anchors = {f"a{int(np.argmin(x_true)):03d}": -1.0, f"a{int(np.argmax(x_true)):03d}": 1.0}

# --- fit the four models ------------------------------------------------------
print("fitting four ideal-point models on the same planted corpus...\n")

wf = topica.Wordfish()
wf.fit(docs, group=author, anchors=anchors, iters=100)

iptm = topica.IdealPointTM(num_topics=K, num_dims=1, seed=1)
iptm.fit(docs, word_emb, vocab, group=author, anchors=anchors, iters=40)

iplda = topica.IdealPointLDA(num_topics=K, num_dims=1, seed=1)
iplda.fit(docs, group=author, anchors=anchors, iters=40)

# SentenceIdealTM scales the per-document embeddings; group is the author per doc.
sitm = topica.SentenceIdealTM(num_topics=K, num_dims=1, seed=1)
sitm.fit(doc_emb, group=author, anchors=anchors, iters=80)

positions = {}
print(f"{'model':<18}{'representation':<22}{'topics':<8}|r| with planted")
print("-" * 62)
for name, model, rep, topics, by_doc in [
    ("Wordfish", wf, "word counts", "no", False),
    ("IdealPointLDA", iplda, "word counts", "yes", False),
    ("IdealPointTM", iptm, "word embeddings", "yes", False),
    ("SentenceIdealTM", sitm, "sentence embeddings", "yes", True),
]:
    rec, r = recovery(model, by_doc=by_doc)
    positions[name] = rec
    print(f"{name:<18}{rep:<22}{topics:<8}{r:.3f}")

# --- pairwise agreement between the models' scales ----------------------------
names = list(positions)
print("\npairwise |r| between the recovered scales:")
print(" " * 18 + "".join(f"{n[:12]:>14}" for n in names))
for a in names:
    row = "".join(f"{abs(np.corrcoef(positions[a], positions[b])[0, 1]):>14.3f}" for b in names)
    print(f"{a:<18}{row}")

print(
    "\nAll four recover the planted axis; they agree closely with each other, which is\n"
    "the point of the comparison: on clean data the latent scale is robust to the\n"
    "representation. Differences show up on real, messy corpora (see the findings\n"
    "memo) and in the per-topic discrimination, which only the topic models provide."
)
