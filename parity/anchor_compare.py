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
3. **Untempered config matches end-to-end on the planted fixture.** With
   ``recover="l2", frequency_temper=1.0`` (the exact Arora inversion), the full
   pipelines agree to mean cosine ~1.0 on the planted separable corpus. Anchor
   selection is a substantive pipeline stage the two libraries implement
   differently, so this end-to-end match is fixture-specific, not general.
4. **The default tempering measurably changes the output, by design.** The
   constructor default ``frequency_temper=0.5`` tempers frequent-word dominance in
   the Bayes inversion for more distinctive topics. Isolating just the exponent
   (same solver, same anchors), ``0.5`` deviates from the exact ``1.0`` inversion by
   a corpus-dependent amount (cosine ~0.99 on the planted fixture, ~0.81 on real
   text) -- a deliberate topica enhancement, not the reference-exact configuration.

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

    Ct, _hist, _conv = TA._recover_l2(Qt, anchors)
    Cr = _reference_recover_C(Qt, anchors, len(anchors))
    cos = float((_unit(Ct) * _unit(Cr)).sum(axis=1).mean())
    return cos, _l2_objective(Qt, Ct, anchors), _l2_objective(Qt, Cr, anchors)


def _assert_recovery_parity(docs, label):
    cos, l2_t, l2_r = _recovery_given_anchors_parity(docs)
    assert cos > 0.999, f"{label} recovered-topic cosine {cos:.6f}"
    # Same objective to a tight relative tolerance (0.1%). Two independent solvers
    # -- topica's NNLS RecoverL2 and the reference's exponentiated gradient -- reach
    # essentially the same L2 minimum; a materially worse implementation would not.
    rel = abs(l2_t - l2_r) / max(l2_r, 1e-12)
    assert rel < 1e-3, f"{label} L2 objective topica {l2_t:.6g} vs reference {l2_r:.6g} (rel {rel:.2e})"


def test_recovery_given_anchors_is_reference_exact_planted():
    """Load-bearing: same anchors + Q -> topica RecoverL2 matches the reference."""
    _skip_if_no_reference()
    _assert_recovery_parity(_planted(), "planted")


def test_recovery_given_anchors_is_reference_exact_real_text():
    """The same recovery parity holds on real text (bundled Gadarian survey)."""
    _skip_if_no_reference()
    _assert_recovery_parity(_gadarian(), "real-text")


def test_untempered_l2_end_to_end_matches_on_planted_fixture():
    """On the planted separable fixture, the untempered config (recover='l2',
    frequency_temper=1.0) matches the full reference pipeline. This is fixture-
    specific: the two libraries use different greedy anchor selectors, so end-to-end
    agreement is corpus-dependent (see the module docstring)."""
    _skip_if_no_reference()
    from anchor_topic.topics import model_topics

    docs = _planted()
    m = _fit(docs, recover="l2", frequency_temper=1.0)
    vocab = list(m.vocabulary)
    A_ref, _Q, _a = model_topics(_dtm_word_doc(docs, vocab), K, THRESHOLD, seed=1)
    cos = _mean_aligned_cosine(m.topic_word, np.asarray(A_ref).T)
    assert cos > 0.99, f"untempered end-to-end cosine on planted fixture {cos:.3f}"


def _isolated_tempering_cosine(docs):
    """Effect of frequency_temper alone: same L2 solver, same (seeded) anchors,
    exponent 1.0 vs 0.5. Aligned topic-word cosine between the two."""
    exact = _fit(docs, recover="l2", frequency_temper=1.0)
    tempered = _fit(docs, recover="l2", frequency_temper=0.5)
    return _mean_aligned_cosine(exact.topic_word, tempered.topic_word)


def test_default_tempering_measurably_changes_output():
    """Isolating just the frequency-temper exponent (same solver, same anchors), the
    default 0.5 measurably departs from the exact 1.0 inversion -- confirming the
    default is NOT the reference-exact configuration. The magnitude is corpus-
    dependent (~0.99 on this planted fixture, lower on real text), so we assert only
    that it is a real, non-degenerate change, not a bound."""
    _skip_if_no_reference()
    cos = _isolated_tempering_cosine(_planted())
    assert 0.5 < cos < 0.999, f"isolated tempering cosine {cos:.4f} (expected a real, non-degenerate change)"


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
    temper_planted = _isolated_tempering_cosine(_planted())
    temper_real = _isolated_tempering_cosine(_gadarian())

    print(f"(1) Q max|diff| vs reference:               {q_diff:.2e}  (row-normalized, ~precision)")
    print(f"(2) recovery | same anchors, planted:       cos {cos_p:.6f}  L2 topica {l2t:.6g} / ref {l2r:.6g}")
    print(f"(2) recovery | same anchors, real text:     cos {cos_r:.6f}")
    print(f"(3) untempered end-to-end (planted fixture): cos {cos_exact:.3f}  (fixture-specific)")
    print(f"(4) isolated tempering 1.0 vs 0.5:          cos {temper_planted:.3f} planted / {temper_real:.3f} real"
          f"  (topica extension, corpus-dependent)")
