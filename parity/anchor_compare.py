"""Parity: topica ``AnchorLDA`` vs the ``anchor-topic`` reference (Arora et al. 2013).

``anchor-topic`` (forest-snow, ``pip install anchor-topic==0.1.2``) is a faithful
implementation of the Arora-family **RecoverL2** anchor-words estimator: the
unbiased per-document co-occurrence matrix ``Q = (h hᵀ - diag(h)) / (n(n-1))``,
greedy farthest-point anchor selection, an exponentiated-gradient solve of the
**squared-error** objective ``‖Q_i - C_i Q_anchors‖²`` (its ``recover.py`` docstring
says "L2 divergence"), and the untempered Bayes inversion to ``p(word | topic)``.
topica implements the same estimator (``recover="l2"``) plus a RecoverKL variant and
an optional frequency-tempered inversion.

Anchor-words is only *provably exact under separability*, and the two libraries use
different (both valid) greedy anchor selectors, so end-to-end agreement is
corpus-dependent. We therefore validate the parts that are algorithm-defined and
scope the rest honestly:

1. **Co-occurrence Q agrees to numerical precision.** topica's ``_build_q`` and the
   reference's ``computeQ`` (both row-normalized) agree to ~1e-16 on the same counts.
2. **Given the same anchors, the RecoverL2 recovery is reference-exact.** Feeding
   both solvers the identical Q and anchor set, topica's ``recover="l2"`` matches the
   reference's L2 recovery to cosine ~1.0 and the same L2 objective -- on planted
   *and* real text. This is the load-bearing faithfulness check: it isolates the
   estimator from anchor-selection differences.
3. **Exact-Arora config matches end-to-end on separable data.** With
   ``recover="l2", frequency_temper=1.0`` (the exact Arora inversion), the full
   pipelines agree to mean cosine ~1.0 where the method's guarantee holds.
4. **The default is a bounded, documented extension.** The constructor default
   ``frequency_temper=0.5`` tempers frequent-word dominance in the Bayes inversion
   for more distinctive topics; it deviates from the reference-exact inversion by a
   small, bounded amount (~0.99 cosine), by design -- it is a topica enhancement,
   not the reference configuration.

Skips cleanly if ``anchor-topic`` is not installed.
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


def _gadarian():
    """Real open-ended survey responses (bundled, offline)."""
    df = topica.datasets.load_gadarian()
    return topica.from_dataframe(
        df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
    )


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


def _unit(m):
    m = np.asarray(m, dtype=float)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def _mean_aligned_cosine(a, b):
    from scipy.optimize import linear_sum_assignment

    sim = _unit(a) @ _unit(b).T
    r, c = linear_sum_assignment(-sim)
    return float(sim[r, c].mean())


def _fit(docs, *, recover="l2", frequency_temper=1.0, k=K, min_count=1, thr=THRESHOLD):
    m = topica.AnchorLDA(
        k,
        recover=recover,
        frequency_temper=frequency_temper,
        min_count=min_count,
        anchor_min_doc_freq=thr,
        seed=42,
    )
    m.fit(docs)
    return m


def _reference_recover_C(Q, anchors, k):
    """The reference's per-word RecoverL2 solution C = p(topic | word) on a given
    row-normalized Q and anchor set (exactly what ``recover.computeA`` runs)."""
    import anchor_topic.recover as recover

    X = Q[anchors] / Q[anchors].sum(axis=1)[:, None]
    XX = X @ X.T
    return np.array(
        [recover.exponentiated_gradient(Q[w], X, XX, 2e-7, k) for w in range(Q.shape[0])]
    )


def _l2_objective(Q, C, anchors):
    """Mean squared reconstruction error ‖Q_i - C_i Q_anchors‖² -- the RecoverL2
    objective both solvers minimize."""
    recon = np.asarray(C) @ Q[anchors]
    return float(((Q - recon) ** 2).sum()) / Q.shape[0]


def _skip_if_no_reference():
    import pytest

    try:
        import anchor_topic  # noqa: F401
    except ImportError:
        pytest.skip("anchor-topic reference not installed (pip install anchor-topic==0.1.2)")


def test_cooccurrence_agrees_to_precision():
    """topica's Q equals the reference's computeQ (both row-normalized) to ~1e-16."""
    _skip_if_no_reference()
    import anchor_topic.cooccur as cooccur

    docs = _planted()
    m = _fit(docs)
    vocab = list(m.vocabulary)
    Qt, _ = TA._build_q(TA._doc_term(docs, {w: i for i, w in enumerate(vocab)}))

    Qr = cooccur.computeQ(_dtm_word_doc(docs, vocab))
    rs = Qr.sum(axis=1)
    rs[rs == 0] = 1.0
    assert np.abs(Qt - Qr / rs[:, None]).max() < 1e-12


def _recovery_given_anchors_parity(docs):
    """Given identical Q + anchors, topica RecoverL2 == the reference's RecoverL2."""
    m = _fit(docs, recover="l2")
    vocab = list(m.vocabulary)
    idx = {w: i for i, w in enumerate(vocab)}
    anchors = [idx[w] for w in m.anchors]
    token_lists = docs.documents() if hasattr(docs, "documents") else docs
    Qt, _ = TA._build_q(TA._doc_term(token_lists, idx))

    Ct = TA._recover_l2(Qt, anchors)
    Cr = _reference_recover_C(Qt, anchors, len(anchors))
    cos = float((_unit(Ct) * _unit(Cr)).sum(axis=1).mean())
    return cos, _l2_objective(Qt, Ct, anchors), _l2_objective(Qt, Cr, anchors)


def test_recovery_given_anchors_is_reference_exact_planted():
    """Load-bearing: same anchors + Q -> topica RecoverL2 matches the reference."""
    _skip_if_no_reference()
    cos, kl_t, kl_r = _recovery_given_anchors_parity(_planted())
    assert cos > 0.999, f"recovered-topic cosine {cos:.4f}"
    assert abs(kl_t - kl_r) < 1e-4, f"L2 objective topica {kl_t:.6f} vs reference {kl_r:.6f}"


def test_recovery_given_anchors_is_reference_exact_real_text():
    """The same recovery parity holds on real text (bundled Gadarian survey)."""
    _skip_if_no_reference()
    cos, _kl_t, _kl_r = _recovery_given_anchors_parity(_gadarian())
    assert cos > 0.999, f"real-text recovered-topic cosine {cos:.4f}"


def test_exact_arora_end_to_end_parity_on_separable_data():
    """Exact config (recover='l2', frequency_temper=1.0) matches the full reference
    pipeline where the separability guarantee holds."""
    _skip_if_no_reference()
    from anchor_topic.topics import model_topics

    docs = _planted()
    m = _fit(docs, recover="l2", frequency_temper=1.0)
    vocab = list(m.vocabulary)
    A_ref, _Q, _a = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos = _mean_aligned_cosine(m.topic_word, np.asarray(A_ref).T)
    assert cos > 0.99, f"exact-Arora end-to-end cosine {cos:.3f}"


def test_default_tempering_is_a_bounded_extension():
    """The constructor default (frequency_temper=0.5) is a deliberate, bounded
    deviation from the reference-exact inversion -- not reference-exact, not wild."""
    _skip_if_no_reference()
    from anchor_topic.topics import model_topics

    docs = _planted()
    default = _fit(docs, recover="kl", frequency_temper=0.5)
    vocab = list(default.vocabulary)
    A_ref, _Q, _a = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos = _mean_aligned_cosine(default.topic_word, np.asarray(A_ref).T)
    assert 0.95 < cos < 1.0, f"default-vs-reference cosine {cos:.4f} (expected a bounded gap)"


if __name__ == "__main__":
    import anchor_topic.cooccur as cooccur
    from anchor_topic.topics import model_topics

    docs = _planted()
    m = _fit(docs, recover="l2")
    vocab = list(m.vocabulary)

    Qt, _ = TA._build_q(TA._doc_term(docs, {w: i for i, w in enumerate(vocab)}))
    Qr = cooccur.computeQ(_dtm_word_doc(docs, vocab))
    rs = Qr.sum(axis=1)
    rs[rs == 0] = 1.0
    q_diff = np.abs(Qt - Qr / rs[:, None]).max()

    cos_p, l2t, l2r = _recovery_given_anchors_parity(_planted())
    cos_r, _, _ = _recovery_given_anchors_parity(_gadarian())

    A_ref, _Q, _a = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos_exact = _mean_aligned_cosine(m.topic_word, np.asarray(A_ref).T)
    default = _fit(docs, recover="kl", frequency_temper=0.5)
    A_ref2, _Q2, _a2 = model_topics(_dtm_word_doc(docs, list(default.vocabulary)), K, THRESHOLD, seed=1)
    cos_default = _mean_aligned_cosine(default.topic_word, np.asarray(A_ref2).T)

    print(f"(1) Q max|diff| vs reference:            {q_diff:.2e}  (row-normalized, ~precision)")
    print(f"(2) recovery | same anchors, planted:    cos {cos_p:.4f}  L2 topica {l2t:.5f} / ref {l2r:.5f}")
    print(f"(2) recovery | same anchors, real text:  cos {cos_r:.4f}")
    print(f"(3) exact-Arora end-to-end (separable):  cos {cos_exact:.3f}")
    print(f"(4) default temper=0.5 vs reference:     cos {cos_default:.3f}  (bounded topica extension)")
