"""Parity: topica ``AnchorLDA`` vs the ``anchor-topic`` reference (Arora et al. 2013).

Both implement the same anchor-words spectral estimator: the unbiased per-document
co-occurrence matrix ``Q = (h hᵀ - diag(h)) / (n(n-1))``, greedy farthest-point
anchor selection, RecoverKL by exponentiated gradient, and the Bayes inversion to
``p(word | topic)``. topica runs here in its exact-Arora configuration
(``recover="kl"``, ``frequency_temper=1.0``) so the two estimators are comparable.

Anchor-words is only *provably exact under separability*, so we validate the parts
that are well-defined and compare the rest honestly:

1. **Co-occurrence Q is bit-identical.** topica's ``_build_q`` and the reference's
   ``computeQ`` agree to machine epsilon on the same word x document counts.
2. **End-to-end recovery matches on a separable corpus.** Where the method's
   guarantee holds, the full pipelines (Q + anchor selection + RecoverKL + Bayes
   inversion) align to ~1.0 mean cosine.
3. **topica's recovery is at least as converged.** On real text the two diverge,
   but topica reaches a *lower* (better) RecoverKL objective than the reference on
   the identical Q and anchors -- the reference's exponentiated gradient stops
   sooner. topica is the more-optimal solver of the same objective, not a different
   model.

Reference: ``pip install anchor-topic`` (forest-snow). Skips cleanly if absent.
"""

import numpy as np

import topica
import topica.anchor as TA

# anchor-topic 0.1.2 predates NumPy 1.24 and calls the removed ``numpy.int`` alias
# in its greedy anchor search. Restore it (== builtin int) so the reference runs
# unmodified under a current NumPy.
if not hasattr(np, "int"):
    np.int = int

K = 6
N_CONTENT = 8
N_SHARED = 12
N_DOCS = 600
DOC_LEN = 70
THRESHOLD = 0.005


def _planted(seed=20260729):
    """Separable corpus: each topic owns one anchor word plus content words, over a
    shared background. The separability the anchor-words guarantee needs holds by
    construction."""
    rng = np.random.default_rng(seed)
    anchors = [f"anchor_t{t}" for t in range(K)]
    content = {t: [f"t{t}_w{i}" for i in range(N_CONTENT)] for t in range(K)}
    shared = [f"shared_{i}" for i in range(N_SHARED)]

    def topic_dist(t):
        words = anchors[t : t + 1] + content[t] + shared
        w = np.array([6.0] + [3.0] * N_CONTENT + [0.6] * N_SHARED)
        return words, w / w.sum()

    dists = [topic_dist(t) for t in range(K)]
    docs = []
    for d in range(N_DOCS):
        words, p = dists[d % K]
        docs.append(list(rng.choice(words, size=DOC_LEN, p=p)))
    return docs


def _dtm_word_doc(docs, vocab):
    """Word x document counts (scipy CSC) over exactly ``vocab``."""
    import scipy.sparse as sp

    idx = {w: i for i, w in enumerate(vocab)}
    rows, cols, data = [], [], []
    for j, doc in enumerate(docs):
        counts = {}
        for tok in doc:
            i = idx.get(tok)
            if i is not None:
                counts[i] = counts.get(i, 0) + 1
        for i, c in counts.items():
            rows.append(i)
            cols.append(j)
            data.append(float(c))
    return sp.csc_matrix((data, (rows, cols)), shape=(len(vocab), len(docs)))


def _mean_aligned_cosine(a, b):
    from scipy.optimize import linear_sum_assignment

    def unit(m):
        m = np.asarray(m, dtype=float)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    sim = unit(a) @ unit(b).T
    r, c = linear_sum_assignment(-sim)
    return float(sim[r, c].mean())


def _fit_topica(docs):
    topica.enable_experimental()  # no-op once AnchorLDA is promoted off the gate
    m = topica.AnchorLDA(
        K,
        recover="kl",
        frequency_temper=1.0,  # exact Arora Bayes inversion (gamma = 1)
        min_count=1,
        anchor_min_doc_freq=THRESHOLD,
        seed=42,
    )
    m.fit(docs)
    return m


def _reference_Q(docs, vocab):
    import anchor_topic.cooccur as cooccur

    return cooccur.computeQ(_dtm_word_doc(docs, vocab))


def _mean_kl(Q, C, anchors):
    """Mean reconstruction KL(Q_i || C_i @ Q_anchors) -- the RecoverKL objective."""
    qa = Q[anchors]
    recon = np.clip(np.asarray(C) @ qa, 1e-12, None)
    mask = Q > 0
    return float((Q[mask] * np.log(Q[mask] / recon[mask])).sum()) / Q.shape[0]


def test_cooccurrence_is_bit_identical():
    """topica's Q equals the reference's computeQ (after its own row-normalization)."""
    import pytest

    try:
        _reference_Q([["a", "b"]], ["a", "b"])
    except ImportError:
        pytest.skip("anchor-topic reference not installed (pip install anchor-topic)")

    docs = _planted()
    m = _fit_topica(docs)
    vocab = list(m.vocabulary)
    counts = TA._doc_term(docs, {w: i for i, w in enumerate(vocab)})
    Qt, _ = TA._build_q(counts)

    Qr = _reference_Q(docs, vocab)
    rs = Qr.sum(axis=1)
    rs[rs == 0] = 1.0
    Qr_norm = Qr / rs[:, None]
    assert np.abs(Qt - Qr_norm).max() < 1e-12


def test_planted_end_to_end_parity():
    """Full pipelines agree on the separable regime where the guarantee holds."""
    import pytest

    try:
        from anchor_topic.topics import model_topics
    except ImportError:
        pytest.skip("anchor-topic reference not installed (pip install anchor-topic)")

    docs = _planted()
    m = _fit_topica(docs)
    vocab = list(m.vocabulary)
    A_ref, _Q, _anchors = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos = _mean_aligned_cosine(m.topic_word, np.asarray(A_ref).T)
    assert cos > 0.99, f"planted end-to-end mean aligned cosine {cos:.3f}"


def test_topica_recovery_at_least_as_converged():
    """On the identical Q and anchors, topica's RecoverKL objective is no worse than
    the reference's -- the reference under-converges; topica is the tighter solver."""
    import pytest

    try:
        import anchor_topic.recover as recover
    except ImportError:
        pytest.skip("anchor-topic reference not installed (pip install anchor-topic)")

    docs = _planted()
    m = _fit_topica(docs)
    vocab = list(m.vocabulary)
    idx = {w: i for i, w in enumerate(vocab)}
    anchors = [idx[w] for w in m.anchors]

    Qt, _ = TA._build_q(TA._doc_term(docs, idx))
    Ct, _hist, _conv = TA._recover_kl(Qt, anchors, iters=5000, eta=1.0, tol=1e-12)

    X = Qt[anchors] / Qt[anchors].sum(axis=1)[:, None]
    XX = X @ X.T
    Cr = np.array(
        [recover.exponentiated_gradient(Qt[w], X, XX, 2e-7, K) for w in range(Qt.shape[0])]
    )

    kl_topica = _mean_kl(Qt, Ct, anchors)
    kl_ref = _mean_kl(Qt, Cr, anchors)
    assert kl_topica <= kl_ref + 1e-9, f"topica KL {kl_topica:.4f} > reference {kl_ref:.4f}"


if __name__ == "__main__":
    from anchor_topic.topics import model_topics
    import anchor_topic.recover as recover

    docs = _planted()
    m = _fit_topica(docs)
    vocab = list(m.vocabulary)
    idx = {w: i for i, w in enumerate(vocab)}
    anchors = [idx[w] for w in m.anchors]

    # (1) co-occurrence parity
    Qt, _ = TA._build_q(TA._doc_term(docs, idx))
    Qr = _reference_Q(docs, vocab)
    rs = Qr.sum(axis=1)
    rs[rs == 0] = 1.0
    q_diff = np.abs(Qt - Qr / rs[:, None]).max()

    # (2) end-to-end planted parity
    A_ref, _Q, ref_anchors = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos = _mean_aligned_cosine(m.topic_word, np.asarray(A_ref).T)

    # (3) recovery convergence on identical Q + anchors
    Ct, _h, _c = TA._recover_kl(Qt, anchors, iters=5000, eta=1.0, tol=1e-12)
    X = Qt[anchors] / Qt[anchors].sum(axis=1)[:, None]
    XX = X @ X.T
    Cr = np.array([recover.exponentiated_gradient(Qt[w], X, XX, 2e-7, K) for w in range(Qt.shape[0])])
    kl_t, kl_r = _mean_kl(Qt, Ct, anchors), _mean_kl(Qt, Cr, anchors)

    print(f"vocab size:                    {len(vocab)}")
    print(f"(1) Q max|diff| vs reference:  {q_diff:.2e}   (bit-identical co-occurrence)")
    print(f"(2) planted end-to-end cosine: {cos:.3f}      (separable regime)")
    print(f"(3) RecoverKL objective        topica {kl_t:.4f} vs reference {kl_r:.4f}"
          f"  -> topica {'tighter' if kl_t <= kl_r else 'looser'}")
