"""IdealPointTM walkthrough: one fit gives you topics AND an unsupervised scaling of
authors along a latent axis (an "ideal point"), plus a per-topic account of how word
choice shifts along that axis.

This example is self-contained: it plants author positions and a single discriminating
topic, samples documents from the model, then fits IdealPointTM and shows that it
recovers the planted positions and reads off the discriminating vocabulary. On real
data you would bring your own tokenized documents, an author label per document, and
word embeddings aligned to the vocabulary (training word2vec on the corpus works well).

A practical note from validation: IdealPointTM works on *expressive* text (messaging,
opinion, manifestos) grouped by author. On procedural text (e.g. legislative floor
speech) the latent axis carries little signal. Keep the corpus clean (one language, low
boilerplate): the single position axis will otherwise lock onto the largest off-topic
axis. IdealPointTM is experimental and gated.

Run from the repo root:  python examples/idealpoint.py
"""
import os
import tempfile

import numpy as np

import topica

topica.enable_experimental()  # IdealPointTM is experimental; opt in to use it

rng = np.random.default_rng(0)
K, V, E = 3, 60, 8            # topics, vocabulary, embedding dimension
A, DOCS_PER, LENGTH = 40, 12, 50  # authors, documents each, tokens per document

# --- plant a generative world -------------------------------------------------
# Word embeddings you would normally bring (here random). Topic embeddings place
# each topic in that space; topic 0 also has a discrimination direction w, so an
# author's position shifts word choice WITHIN topic 0. Topics 1 and 2 are neutral.
rho = rng.normal(size=(V, E))
alpha = rng.normal(size=(K, E))
w0 = rng.normal(size=E) * 2.5
x_true = rng.uniform(-1.0, 1.0, size=A)        # the latent positions to recover
vocab = [f"w{i}" for i in range(V)]


def beta(author_pos, k):
    logits = rho @ alpha[k] + (author_pos * (rho @ w0) if k == 0 else 0.0)
    e = np.exp(logits - logits.max())
    return e / e.sum()


docs, author = [], []
for a in range(A):
    for _ in range(DOCS_PER):
        doc = []
        for _ in range(LENGTH):
            k = rng.integers(0, K)
            doc.append(vocab[rng.choice(V, p=beta(x_true[a], k))])
        docs.append(doc)
        author.append(f"author_{a}")

# --- fit ----------------------------------------------------------------------
# anchors only fix the arbitrary sign/scale: name two authors known to sit at
# opposite ends. Here we use the two with the most extreme planted positions.
lo, hi = f"author_{int(np.argmin(x_true))}", f"author_{int(np.argmax(x_true))}"
m = topica.IdealPointTM(num_topics=K, num_dims=1, seed=1)
m.fit(docs, word_embeddings=rho, vocabulary=vocab, group=author, anchors={lo: -1.0, hi: 1.0}, iters=40)

# --- 1. the scale: author positions ------------------------------------------
pos = dict(zip(m.author_names, m.author_positions[:, 0]))
recovered = np.array([pos[f"author_{a}"] for a in range(A)])
r = np.corrcoef(recovered, x_true)[0, 1]
print(f"recovered {m.num_authors} author positions; correlation with planted = {r:+.3f}")

# --- 2. the topics: the other half of the double job --------------------------
print("\ntop words per topic (at the neutral position):")
for k, words in enumerate(m.top_words(6)):
    print(f"  topic {k}: {', '.join(words)}")

# --- 3. discrimination: which topic carries the latent axis -------------------
disc = m.topic_discrimination
print(f"\ntopic discrimination ||W_k||: {np.round(disc, 2)}")
print(f"  most discriminating topic = {int(np.argmax(disc))} (planted: topic 0)")

# --- 4. position_shift: how words move along the axis within that topic -------
k = int(np.argmax(disc))
high, low = m.position_shift(k, n=6)
print(f"\nwithin topic {k}, words at the + end: {', '.join(wd for wd, _ in high)}")
print(f"within topic {k}, words at the - end: {', '.join(wd for wd, _ in low)}")

# --- 5. persistence -----------------------------------------------------------
path = os.path.join(tempfile.mkdtemp(), "idealpoint.topica")
m.save(path)
reloaded = topica.IdealPointTM.load(path)
print(f"\nsaved and reloaded; positions identical: "
      f"{np.array_equal(m.author_positions, reloaded.author_positions)}")
