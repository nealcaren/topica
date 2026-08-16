"""Tests for GuidedNMF: seed-word-guided semi-supervised NMF (Vendrow, Haddock,
Rebrova & Needell, ICASSP 2021), validated against the ssnmf package.

Covers the four CONTRIBUTING-MODELS idioms (shapes/normalization, planted-data
recovery, determinism, save-load + bad-params) plus the guidance-specific surface
(seed_topic_indices, raw factors) and edge cases.
"""

import numpy as np
import pytest

import topica


def _planted_docs(reps=6):
    """Three word-blocks -> three latent topics; the 4th topic is free."""
    blocks = [
        ["a", "b", "c"],
        ["m", "n", "o"],
        ["x", "y", "z"],
    ]
    docs = []
    for blk in blocks:
        for _ in range(reps):
            docs.append(blk + blk[:2])
    return docs


SEEDS = {"first": ["a", "b"], "third": ["x", "y"]}


def test_shapes_and_normalization():
    docs = _planted_docs()
    m = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=40)
    V = len(m.vocabulary)
    assert m.topic_word.shape == (3, V)
    assert m.doc_topic.shape == (len(docs), 3)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    # Raw factors expose the un-normalized decomposition.
    assert m.factor_a.shape == (len(docs), 3)
    assert m.factor_s.shape == (3, V)
    assert m.factor_b.shape == (2, 3)  # G groups x K topics


def test_determinism():
    docs = _planted_docs()
    a = topica.GuidedNMF(3, SEEDS, weighting="count", seed=1).fit(docs, iters=30)
    b = topica.GuidedNMF(3, SEEDS, weighting="count", seed=1).fit(docs, iters=30)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.factor_s, b.factor_s)


def test_determinism_across_threads():
    docs = _planted_docs()
    a = topica.GuidedNMF(3, SEEDS, weighting="count", seed=2).fit(docs, iters=30, num_threads=1)
    b = topica.GuidedNMF(3, SEEDS, weighting="count", seed=2).fit(docs, iters=30, num_threads=4)
    assert np.array_equal(a.factor_s, b.factor_s)


def test_guidance_steers_seeded_topics():
    docs = _planted_docs()
    m = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=80)
    idx = m.seed_topic_indices
    assert len(idx) == 2
    assert idx[0] != idx[1], "distinct seed groups should steer distinct topics"
    assert list(m.seed_group_names) == ["first", "third"]
    # The topic each group steers should peak on that group's block.
    top0 = [w for w, _ in m.top_words(3, topic=idx[0], weights=True)]
    top1 = [w for w, _ in m.top_words(3, topic=idx[1], weights=True)]
    assert {"a", "b"} & set(top0)
    assert {"x", "y"} & set(top1)


def test_objective_decreases():
    docs = _planted_docs()
    m = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=50)
    hist = np.asarray(m.error_history)
    assert np.all(np.diff(hist) <= 1e-9)


def test_explicit_init_matches_reference_math():
    """One update from a fixed init reproduces ssnmf's supervised-Frobenius update
    to floating-point noise (the parity fixture's strongest check, inlined)."""
    docs = _planted_docs()
    m0 = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=1)
    D, V = m0.factor_a.shape[0], len(m0.vocabulary)
    rng = np.random.RandomState(0)
    A0 = rng.rand(D, 3)
    S0 = rng.rand(3, V)
    B0 = rng.rand(2, 3)
    corpus = topica.Corpus.from_documents(docs, vocabulary=list(m0.vocabulary))
    m = topica.GuidedNMF(
        3, SEEDS, guidance=20.0, weighting="count", init="none",
        init_a=A0, init_s=S0, init_b=B0, convergence_tol=0.0,
    ).fit(corpus, iters=1)
    # Reference one-update in ssnmf's order (A, B, then S), eps = 1e-10.
    X = np.zeros((D, V))
    vidx = {w: i for i, w in enumerate(m0.vocabulary)}
    for d, toks in enumerate(_planted_docs()):
        for t in toks:
            if t in vidx:
                X[d, vidx[t]] += 1.0
    Y = np.zeros((2, V))
    for g, ws in enumerate(SEEDS.values()):
        for w in ws:
            Y[g, vidx[w]] = 1.0
    lam, eps = 20.0, 1e-10
    A1 = A0 * (X @ S0.T) / (A0 @ S0 @ S0.T + eps)
    B1 = B0 * (Y @ S0.T) / (B0 @ S0 @ S0.T + eps)
    S1 = S0 * (A1.T @ X + lam * B1.T @ Y) / (A1.T @ A1 @ S0 + lam * B1.T @ B1 @ S0 + eps)
    assert np.abs(np.asarray(m.factor_s) - S1).max() < 1e-8
    assert np.abs(np.asarray(m.factor_a) - A1).max() < 1e-8


def test_save_load_roundtrip(tmp_path):
    docs = _planted_docs()
    m = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=30)
    p = str(tmp_path / "gnmf.topica")
    m.save(p)
    loaded = topica.GuidedNMF.load(p)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert np.array_equal(m.factor_s, loaded.factor_s)
    assert loaded.seed_topic_indices == m.seed_topic_indices
    assert loaded.settings == m.settings


def test_lam_alias():
    m = topica.GuidedNMF(3, SEEDS, lam=5.0)
    assert m.settings["guidance"] == 5.0


def test_rejects_more_groups_than_topics():
    with pytest.raises(ValueError):
        topica.GuidedNMF(1, SEEDS)  # 2 seed groups, 1 topic


def test_rejects_unmatched_seed_group():
    docs = _planted_docs()
    with pytest.raises((ValueError, RuntimeError)):
        topica.GuidedNMF(3, {"first": ["a"], "ghost": ["zzz_absent"]},
                         weighting="count").fit(docs, iters=5)


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.GuidedNMF(2, {"g": ["a"]}).fit([])


def test_none_init_requires_all_factors():
    with pytest.raises(ValueError):
        topica.GuidedNMF(3, SEEDS, init="none")  # no init_a/s/b


def test_nndsvd_is_seed_independent():
    """init='nndsvd' is a deterministic extension: the seed must not change the fit
    (Gate-B fix: B was previously seeded)."""
    docs = _planted_docs()
    a = topica.GuidedNMF(3, SEEDS, init="nndsvd", weighting="count", seed=1).fit(docs, iters=20)
    b = topica.GuidedNMF(3, SEEDS, init="nndsvd", weighting="count", seed=999).fit(docs, iters=20)
    assert np.array_equal(a.factor_s, b.factor_s)
    assert np.array_equal(a.factor_a, b.factor_a)


def test_overcomplete_k_allowed():
    """K > V is a valid (overcomplete) factorization for random/none init, matching
    the reference (only nndsvd carries the rank guard)."""
    docs = [["a", "b"], ["b", "a"], ["a", "a"], ["b", "b"]] * 4
    m = topica.GuidedNMF(5, {"g": ["a"]}, init="random", weighting="count", seed=1).fit(docs, iters=5)
    assert m.topic_word.shape == (5, len(m.vocabulary))


def test_load_then_refit_errors_clearly():
    """A loaded init='none' model has no stored init factors; re-fitting must raise
    a clear error, not panic (Gate-B fix)."""
    docs = _planted_docs()
    m0 = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=1)
    D, V = m0.factor_a.shape[0], len(m0.vocabulary)
    rng = np.random.RandomState(0)
    corpus = topica.Corpus.from_documents(docs, vocabulary=list(m0.vocabulary))
    m = topica.GuidedNMF(3, SEEDS, weighting="count", init="none",
                         init_a=rng.rand(D, 3), init_s=rng.rand(3, V), init_b=rng.rand(2, 3)).fit(corpus, iters=5)
    import tempfile
    p = tempfile.mktemp(suffix=".topica")
    m.save(p)
    loaded = topica.GuidedNMF.load(p)
    assert np.array_equal(m.factor_s, loaded.factor_s)  # fitted state round-trips
    with pytest.raises(ValueError):
        loaded.fit(corpus, iters=1)  # no stored init factors -> clear error, no panic


def test_zero_topic_row_stays_zero():
    """An extinct topic (all-zero S row) is reported as zeros, not fabricated
    uniform mass (Gate-B fix / Gate-A resolution)."""
    docs = _planted_docs()
    m0 = topica.GuidedNMF(3, SEEDS, weighting="count", seed=0).fit(docs, iters=1)
    D, V = m0.factor_a.shape[0], len(m0.vocabulary)
    S0 = np.ones((3, V))
    S0[1, :] = 0.0  # topic 1 extinct
    A0 = np.ones((D, 3)); A0[:, 1] = 0.0
    B0 = np.ones((2, 3))
    corpus = topica.Corpus.from_documents(docs, vocabulary=list(m0.vocabulary))
    m = topica.GuidedNMF(3, SEEDS, weighting="count", init="none",
                         init_a=A0, init_s=S0, init_b=B0, convergence_tol=0.0).fit(corpus, iters=1)
    tw = np.asarray(m.topic_word)
    assert not np.isnan(tw).any()
    assert np.allclose(tw[1], 0.0)  # extinct topic stays zero, not 1/V


def test_convergence_tol_does_not_stop_prematurely():
    """A modest convergence_tol must not report `converged` after a handful of
    under-converged iterations (sample-user finding: normalizing the relative
    decrease by the initial objective tripped at ~iter 4). The relative decrease is
    measured against the previous objective, like NMF."""
    docs = _planted_docs(reps=10)
    m = topica.GuidedNMF(3, SEEDS, weighting="count", convergence_tol=1e-5, seed=0).fit(docs, iters=100)
    assert m.iters_run > 10, f"stopped too early at iter {m.iters_run}"


def test_tfidf_default_runs():
    docs = _planted_docs()
    m = topica.GuidedNMF(3, SEEDS, seed=0).fit(docs, iters=30)  # default weighting=tfidf
    assert m.settings["weighting"] == "tfidf"
    assert m.topic_word.shape[0] == 3
