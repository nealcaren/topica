"""Parity check: topica `KeyNMF` vs the turftopic reference and a correct oracle.

KeyNMF's turftopic implementation has stage-1 keyword-extraction bugs (a `zip` that
scrambles word->importance pairs when a selected similarity is <= 0, and an
off-by-one that drops a keyword). topica implements the *correct* method, so this
harness validates at two honest levels:

1. **Stage-1 exact vs a correct numpy oracle** — topica's extracted keywords must
   equal a straightforward cosine / top-N / positive computation (NOT turftopic's
   buggy extractor).
2. **Method-level vs turftopic** — same corpus + MiniLM embeddings + vocabulary on
   both sides; topica's topics must align with turftopic's (topic-word cosine after
   normalizing both) and recover the planted structure about as well.

Needs sentence-transformers + turftopic. Skips cleanly (exit 0) otherwise.

    python parity/keynmf_compare.py
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

import topica

warnings.filterwarnings("ignore")


def _available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import turftopic  # noqa: F401
    except Exception:
        return False
    return True


def _corpus(seed=0):
    """A topic-structured corpus of short docs from four themes."""
    rng = np.random.default_rng(seed)
    themes = {
        "tech": "machine learning neural network data model algorithm software training".split(),
        "climate": "climate change global warming carbon emissions policy environment energy".split(),
        "law": "court judge legal ruling case law justice constitution rights trial".split(),
        "health": "patient hospital disease treatment medical doctor clinical health therapy".split(),
    }
    docs = []
    for _ in range(160):
        t = rng.choice(list(themes))
        docs.append(list(rng.choice(themes[t], 6, replace=False)))
    vocab = sorted({w for ws in themes.values() for w in ws})
    return docs, vocab


def _umass(topic_word: np.ndarray, docs, vocab, top=10) -> float:
    """Mean u_mass coherence of a topic-word matrix, computed in numpy over `docs`
    so topica and turftopic topics are scored on exactly the same footing."""
    vidx = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    # document-frequency and pairwise co-document-frequency over the vocabulary.
    df = np.zeros(v)
    codf = np.zeros((v, v))
    for d in docs:
        present = sorted({vidx[w] for w in d if w in vidx})
        for a in present:
            df[a] += 1.0
        for ii in range(len(present)):
            for jj in range(ii):
                a, b = present[ii], present[jj]
                codf[a, b] += 1.0
                codf[b, a] += 1.0
    scores = []
    for k in range(topic_word.shape[0]):
        top_ids = np.argsort(topic_word[k])[::-1][:top]
        s = 0.0
        for ii in range(1, len(top_ids)):
            for jj in range(ii):
                a, b = int(top_ids[ii]), int(top_ids[jj])
                s += np.log((codf[a, b] + 1.0) / (df[b] + 1e-12))
        scores.append(s)
    return float(np.mean(scores))


def _align_cosine(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    r, c = linear_sum_assignment(-sim)
    return float(sim[r, c].mean())


def main() -> int:
    if not _available():
        print("SKIP: sentence-transformers / turftopic not available")
        return 0
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from turftopic import KeyNMF as TurftopicKeyNMF

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    docs, vocab = _corpus()
    texts = [" ".join(d) for d in docs]
    doc_emb = np.asarray(enc.encode(texts))
    word_emb = np.asarray(enc.encode(vocab))
    k, top_n = 4, 10

    # topica
    m = topica.KeyNMF(num_topics=k, top_n=top_n, seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    topica_tw = np.asarray(m.topic_word)

    # (1) stage-1 exact vs numpy oracle. Compute in float64 to match topica's internal
    # up-cast (encoder output is float32); otherwise a sim sitting exactly on the >0 or
    # top_n boundary could disagree by rounding alone. We check BOTH the selected word
    # set AND each stored importance magnitude (the NMF input), since a selection-only
    # check would miss a wrong-metric or wrong-value regression.
    vidx = {w: i for i, w in enumerate(vocab)}
    we64 = word_emb.astype(np.float64)
    de64 = doc_emb.astype(np.float64)
    wn = we64 / np.linalg.norm(we64, axis=1, keepdims=True)
    mism = 0
    valerr = 0.0
    for d in range(len(docs)):
        present = sorted({vidx[w] for w in docs[d] if w in vidx})
        dn = de64[d] / np.linalg.norm(de64[d])
        sims = {i: float(dn @ wn[i]) for i in present}
        ranked = sorted([i for i in present if sims[i] > 0], key=lambda i: -sims[i])[:top_n]
        oracle = set(ranked)
        got_pairs = m.keywords(d)
        got = {vidx[w] for w, _ in got_pairs}
        if got != oracle:
            mism += 1
        else:
            for w, imp in got_pairs:
                valerr = max(valerr, abs(imp - sims[vidx[w]]))
    print(f"stage-1 keyword extraction vs numpy oracle : {len(docs) - mism}/{len(docs)} docs exact")
    print(f"stage-1 importance value max abs error     : {valerr:.2e}")

    # (2) method-level vs turftopic (same vocab via a fixed-vocabulary vectorizer)
    tt = TurftopicKeyNMF(
        k, top_n=top_n, encoder=enc, random_state=0,
        vectorizer=CountVectorizer(vocabulary=vocab),
    )
    tt.fit(texts, embeddings=doc_emb)
    # align turftopic components to topica's vocab order
    tt_vocab = list(tt.get_vocab())
    tt_idx = {w: i for i, w in enumerate(tt_vocab)}
    tt_tw = np.zeros((k, len(vocab)))
    for j, w in enumerate(vocab):
        if w in tt_idx:
            tt_tw[:, j] = tt.components_[:, tt_idx[w]]
    cos = _align_cosine(topica_tw, tt_tw)
    print(f"topica vs turftopic aligned topic-word cosine: {cos:.4f}")

    # coherence competitiveness — score BOTH sides' topics with the same numpy u_mass
    # over the shared corpus, so "competitive" is an actual head-to-head, not a
    # one-sided print. (topica's own m.coherence agrees with this to rounding.)
    coh_topica = _umass(topica_tw, docs, vocab, top=top_n)
    coh_tt = _umass(tt_tw, docs, vocab, top=top_n)
    print(f"u_mass coherence  topica={coh_topica:.3f}  turftopic={coh_tt:.3f}")

    failures = []
    if mism > 0:
        failures.append(f"stage-1 extraction mismatched on {mism} docs (must be 0)")
    if valerr > 1e-6:
        failures.append(f"stage-1 importance values differ from oracle cosine by {valerr:.2e}")
    # turftopic is a noisy oracle (its stage-1 is buggy) and its NMF is a different
    # (non-convex) solver, so we require strong alignment rather than exact parity.
    # Measured ~0.98 on this corpus; guard a floor well below that but high enough to
    # catch a real regression (a broken extractor or mis-aligned vocab drops it to ~0).
    if cos < 0.9:
        failures.append(f"topica-vs-turftopic aligned cosine {cos:.3f} < 0.9")
    # topica must be competitive on coherence, not markedly worse (u_mass is negative;
    # allow a 20% slack on the magnitude to absorb the different NMF solvers).
    if coh_topica < coh_tt - 0.2 * abs(coh_tt):
        failures.append(
            f"topica u_mass {coh_topica:.3f} materially worse than turftopic {coh_tt:.3f}"
        )

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: topica KeyNMF extracts keywords correctly and aligns with turftopic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
