"""Tests for embedding_regression (conText port): à la carte embeddings +
covariate regression on meaning, after Rodriguez, Spirling & Stewart (2023)."""

import numpy as np
import pytest

import topica
from topica.embedding_regression import alc_embeddings, compute_transform


def _pretrained(V=60, D=20, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(V)]
    emb = rng.normal(size=(V, D))
    return words, emb


def _planted(n=200, seed=1):
    """Group 0 draws words from the first half of the vocab, group 1 from the
    second half: the covariate genuinely shifts *which words* (meaning) are used."""
    words, emb = _pretrained()
    V = len(words)
    rng = np.random.default_rng(seed)
    half, other = list(range(V // 2)), list(range(V // 2, V))
    docs, g = [], []
    for i in range(n):
        gi = i % 2
        pool = half if gi == 0 else other
        docs.append([words[j] for j in rng.choice(pool, 8)])
        g.append(float(gi))
    return docs, np.array(g), (emb, words)


def _null(n=200, seed=2):
    """Covariate unrelated to the text: both groups draw from the whole vocab."""
    words, emb = _pretrained()
    rng = np.random.default_rng(seed)
    docs = [[words[j] for j in rng.integers(0, len(words), 8)] for _ in range(n)]
    g = np.array([float(i % 2) for i in range(n)])
    return docs, g, (emb, words)


# --- inference behavior -----------------------------------------------------
def test_planted_signal_is_significant():
    docs, g, pre = _planted()
    r = topica.embedding_regression(docs, g, pre, names=["group"],
                                    permutations=200, bootstrap=0, seed=1)
    assert r.normed_estimate[0] > 0.5
    assert r.p_value[0] < 0.05


def test_null_covariate_is_not_significant():
    docs, g, pre = _null()
    r = topica.embedding_regression(docs, g, pre, names=["group"],
                                    permutations=200, bootstrap=0, seed=1)
    assert r.p_value[0] > 0.1


def test_permutation_pvalue_is_calibrated_under_null():
    """Across independent null draws the permutation p-values should not be tiny."""
    words, emb = _pretrained()
    rng = np.random.default_rng(0)
    ps = []
    for s in range(8):
        docs = [[words[j] for j in rng.integers(0, len(words), 8)] for _ in range(150)]
        g = np.array([float(i % 2) for i in range(150)])
        r = topica.embedding_regression(docs, g, (emb, words), permutations=100,
                                        bootstrap=0, seed=s)
        ps.append(r.p_value[0])
    assert np.mean(ps) > 0.2  # not systematically significant


def test_nearest_neighbors_separate_the_pools():
    docs, g, pre = _planted()
    r = topica.embedding_regression(docs, g, pre, names=["group"], permutations=0,
                                    bootstrap=0, seed=1)
    nn0 = {w for w, _ in r.nearest_neighbors({"group": 0}, n=8)}
    nn1 = {w for w, _ in r.nearest_neighbors({"group": 1}, n=8)}
    first = {f"w{i}" for i in range(30)}
    # group 0's neighbors lean to the first-half pool more than group 1's do
    assert len(nn0 & first) > len(nn1 & first)


# --- ALC / dem mechanics ----------------------------------------------------
def test_alc_additive_is_mean_of_context_vectors():
    words, emb = _pretrained(V=5, D=4)
    docs = [["w0", "w1"], ["w2", "w3", "w4"]]
    Y, kept = alc_embeddings(docs, (emb, words), transform=None)
    assert list(kept) == [0, 1]
    assert np.allclose(Y[0], (emb[0] + emb[1]) / 2)
    assert np.allclose(Y[1], (emb[0 + 2] + emb[3] + emb[4]) / 3)


def test_alc_focal_term_uses_windowed_context_excluding_focal():
    words, emb = _pretrained(V=6, D=4)
    docs = [["w0", "target", "w1", "w2"]]  # 'target' not in vocab; window picks neighbors
    words2 = words + ["target"]
    emb2 = np.vstack([emb, np.zeros((1, emb.shape[1]))])
    Y, kept = alc_embeddings(docs, (emb2, words2), transform=None, target="target", window=1)
    # window=1 around 'target' -> context {w0, w1}, focal excluded
    assert np.allclose(Y[0], (emb[0] + emb[1]) / 2)


def test_alc_drops_docs_without_invocab_context():
    words, emb = _pretrained(V=5, D=4)
    docs = [["w0"], ["zzz", "qqq"], ["w2"]]  # middle doc all OOV
    Y, kept = alc_embeddings(docs, (emb, words), transform=None)
    assert list(kept) == [0, 2]
    assert Y.shape == (2, 4)


def test_covariates_realigned_to_kept_docs():
    words, emb = _pretrained(V=5, D=4)
    docs = [["w0"], ["zzz"], ["w2"], ["w3"]]  # doc 1 dropped
    g = np.array([0.0, 1.0, 0.0, 1.0])
    r = topica.embedding_regression(docs, g, (emb, words), permutations=0,
                                    bootstrap=0, seed=0)
    # regression ran on 3 kept docs without a row/length mismatch
    assert r.coefficients.shape == (1, 4)


# --- transform estimation ---------------------------------------------------
def test_compute_transform_shape():
    words, emb = _pretrained(V=40, D=8)
    rng = np.random.default_rng(0)
    docs = [[words[j] for j in rng.integers(0, len(words), 20)] for _ in range(200)]
    A = compute_transform(docs, (emb, words), window=4, min_count=5)
    assert A.shape == (8, 8)


def test_compute_transform_raises_when_too_few_words():
    words, emb = _pretrained(V=10, D=8)
    docs = [["w0", "w1"]]
    with pytest.raises(ValueError, match="min_count"):
        compute_transform(docs, (emb, words), min_count=1000)


# --- determinism & CIs ------------------------------------------------------
def test_determinism():
    docs, g, pre = _planted()
    a = topica.embedding_regression(docs, g, pre, permutations=100, bootstrap=50, seed=3)
    b = topica.embedding_regression(docs, g, pre, permutations=100, bootstrap=50, seed=3)
    assert np.allclose(a.normed_estimate, b.normed_estimate)
    assert np.allclose(a.p_value, b.p_value)
    assert np.allclose(a.normed_ci, b.normed_ci)


def test_bootstrap_ci_brackets_estimate():
    docs, g, pre = _planted()
    r = topica.embedding_regression(docs, g, pre, names=["group"], permutations=0,
                                    bootstrap=100, seed=1)
    assert r.normed_ci[0, 0] <= r.normed_estimate[0] <= r.normed_ci[0, 1]


@pytest.mark.parametrize("stat", ["norm", "squared", "squared_deflated"])
def test_statistic_variants_run(stat):
    docs, g, pre = _planted()
    r = topica.embedding_regression(docs, g, pre, statistic=stat, permutations=50,
                                    bootstrap=0, seed=1)
    assert np.isfinite(r.normed_estimate[0])


# --- input validation -------------------------------------------------------
def test_covariate_row_mismatch_raises():
    docs, g, pre = _planted(n=50)
    with pytest.raises(ValueError, match="one row per document"):
        topica.embedding_regression(docs, g[:10], pre, permutations=0, bootstrap=0)


def test_names_length_mismatch_raises():
    docs, g, pre = _planted(n=50)
    with pytest.raises(ValueError, match="names"):
        topica.embedding_regression(docs, g, pre, names=["a", "b"], permutations=0,
                                    bootstrap=0)


def test_instance_mode_one_row_per_mention():
    words, emb = _pretrained(V=30, D=8)
    # doc 0 mentions the target twice, doc 1 once
    docs = [["a", "tgt", "b", "tgt", "c"], ["d", "tgt", "e"]]
    words2 = words + ["tgt", "a", "b", "c", "d", "e"]
    emb2 = np.vstack([emb, np.random.default_rng(0).normal(size=(6, 8))])
    Y, doc_ids = alc_embeddings(docs, (emb2, words2), target="tgt", window=1,
                                aggregate="instance")
    assert Y.shape[0] == 3          # 2 mentions in doc 0 + 1 in doc 1
    assert list(doc_ids) == [0, 0, 1]
    Yd, did = alc_embeddings(docs, (emb2, words2), target="tgt", window=1,
                             aggregate="document")
    assert Yd.shape[0] == 2         # pooled per document
    assert list(did) == [0, 1]


def test_categorical_covariate_dummy_coded():
    docs, g, pre = _planted()
    party = np.array(["D" if gi == 0 else "R" for gi in g])
    r = topica.embedding_regression(docs, party, pre, names=["party"],
                                    permutations=100, bootstrap=0, seed=1)
    assert r.names == ["party_R"]     # first level D dropped as reference
    assert r.reference_level == "D"
    assert r.p_value[0] < 0.05


def test_squared_deflated_is_default_and_centered_under_null():
    docs, g, pre = _null()
    r = topica.embedding_regression(docs, g, pre, permutations=100, bootstrap=0, seed=1)
    assert r.statistic == "squared_deflated"
    assert abs(r.normed_estimate[0]) < 0.2   # deflated null ~ 0 (not a positive floor)


def test_multiple_covariates():
    docs, g, pre = _planted()
    rng = np.random.default_rng(5)
    X = np.column_stack([g, rng.normal(size=len(g))])  # real + noise covariate
    r = topica.embedding_regression(docs, X, pre, names=["group", "noise"],
                                    permutations=100, bootstrap=0, seed=1)
    assert r.coefficients.shape[0] == 2
    assert r.p_value[0] < 0.05  # real covariate
    assert r.p_value[1] > 0.1   # noise covariate
