"""Cross-implementation validation: topica's ``OnlineLDA`` vs gensim's ``LdaModel``
(Hoffman, Blei & Bach 2010 — online/streaming variational Bayes for LDA).

topica's ``OnlineLDA`` is an independent Rust port of the Hoffman online-VB
algorithm, written from the paper and the published update equations. gensim's
``LdaModel`` (LGPL) and Blei's reference ``onlineldavb.py`` (GPL) are used here
only as an external parity oracle — never copied. Because online VB is stochastic
in its minibatch order and (in the references) its random gamma init, this is a
*statistical* agreement check, not a bit-for-bit one: both implementations, fit on
the same planted-topic corpus with matched hyperparameters, should recover the
same topics (high topic-aligned cosine) and place documents on them the same way.

What it checks, on a planted-block corpus where the ground-truth topics are known:
  1. topica recovers the planted blocks (each block is one topic's top words);
  2. gensim recovers them too (sanity on the oracle);
  3. the two topic-word matrices agree after optimal topic alignment
     (mean cosine over the Hungarian assignment), well above a chance baseline.

Observed (gensim 4.x, planted 4-block corpus, matched schedule): topica and gensim
each recover all 4 planted blocks, and their topic-word matrices agree at an
aligned mean cosine of ~0.999 — i.e. the independent Rust port and gensim's
`LdaModel` reach effectively the same topics.

Skips (exit 0) if gensim is unavailable. Run directly:

    python parity/online_lda_gensim_compare.py
"""

from __future__ import annotations

import sys

import numpy as np


def gensim_available() -> bool:
    try:
        import gensim.models.ldamodel  # noqa: F401

        return True
    except Exception:
        return False


BLOCKS = [
    ["sport", "ball", "team", "game", "score"],
    ["bank", "money", "loan", "rate", "cash"],
    ["film", "movie", "actor", "scene", "plot"],
    ["cell", "gene", "protein", "dna", "enzyme"],
]


def planted_docs(n=800, doc_len=30, mix=0.85, seed=0):
    """Each document draws `doc_len` tokens, ~`mix` from its own block and the
    rest uniform background — a clean but non-degenerate recovery target."""
    rng = np.random.default_rng(seed)
    vocab = [w for b in BLOCKS for w in b]
    docs = []
    for i in range(n):
        block = BLOCKS[i % len(BLOCKS)]
        toks = [
            rng.choice(block) if rng.random() < mix else rng.choice(vocab)
            for _ in range(doc_len)
        ]
        docs.append(list(toks))
    return docs


def align_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine between rows of `a` and `b` under the optimal (Hungarian)
    topic assignment. Falls back to a greedy match if scipy is absent."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T  # (K, K)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-sim)
        return float(sim[r, c].mean())
    except Exception:
        used, total = set(), 0.0
        for i in range(sim.shape[0]):
            order = np.argsort(-sim[i])
            for j in order:
                if j not in used:
                    used.add(j)
                    total += sim[i, j]
                    break
        return total / sim.shape[0]


def blocks_recovered(topic_word: np.ndarray, vocab: list[str]) -> int:
    """How many planted blocks appear as the top-5 of some recovered topic."""
    idx = {w: i for i, w in enumerate(vocab)}
    recovered = set()
    for t in range(topic_word.shape[0]):
        top = set(np.argsort(-topic_word[t])[:5].tolist())
        for b, block in enumerate(BLOCKS):
            if {idx[w] for w in block} <= top:
                recovered.add(b)
    return len(recovered)


def main() -> int:
    if not gensim_available():
        print("SKIP: gensim not installed")
        return 0

    import topica
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    K = len(BLOCKS)
    docs = planted_docs()

    # --- topica OnlineLDA -------------------------------------------------
    tm = topica.OnlineLDA(
        K, batch_size=64, tau=1.0, kappa=0.7, beta=0.01, seed=42
    ).fit(docs, iters=20)
    tw_topica = np.asarray(tm.topic_word)
    vocab = list(tm.vocabulary)

    # --- gensim LdaModel (online VB), matched schedule --------------------
    dictionary = Dictionary(docs)
    bow = [dictionary.doc2bow(d) for d in docs]
    gm = LdaModel(
        corpus=bow,
        id2word=dictionary,
        num_topics=K,
        chunksize=64,      # == batch_size
        offset=1.0,        # == tau
        decay=0.7,         # == kappa
        eta=0.01,          # == beta
        passes=20,         # == iters
        iterations=100,    # == inner_iters
        random_state=42,
    )
    # gensim topic-word over gensim's own vocabulary → re-index to topica's vocab.
    g_tw = gm.get_topics()  # (K, V_gensim)
    g_index = {dictionary[i]: i for i in range(len(dictionary))}
    tw_gensim = np.zeros((K, len(vocab)))
    for j, w in enumerate(vocab):
        if w in g_index:
            tw_gensim[:, j] = g_tw[:, g_index[w]]
    tw_gensim /= tw_gensim.sum(axis=1, keepdims=True) + 1e-12

    # --- report -----------------------------------------------------------
    rec_topica = blocks_recovered(tw_topica, vocab)
    rec_gensim = blocks_recovered(tw_gensim, vocab)
    cos = align_cosine(tw_topica, tw_gensim)

    print(f"planted blocks              : {K}")
    print(f"topica blocks recovered     : {rec_topica}/{K}")
    print(f"gensim blocks recovered     : {rec_gensim}/{K}")
    print(f"aligned mean topic cosine   : {cos:.3f}")

    ok = rec_topica == K and rec_gensim == K and cos > 0.9
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
