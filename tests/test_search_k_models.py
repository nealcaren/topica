"""search_k across model types: the built-in nmf/lsa strings and the generic
fit=(k, seed) -> fitted model hook (model-agnostic, like model_factory on the
other tools)."""

import numpy as np
import pytest

import topica


def _corpus():
    # Two disjoint planted topics so any model recovers a clean structure fast.
    a = [["cat", "dog", "pet", "vet", "paw"]] * 30
    b = [["star", "moon", "sky", "sun", "orbit"]] * 30
    docs = a + b
    return topica.Corpus.from_documents(docs), docs


def test_builtin_nmf_reports_reconstruction_error():
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3, 4], model="nmf", iters=100)
    assert [r["k"] for r in rows] == [2, 3, 4]
    for r in rows:
        assert "coherence" in r and "exclusivity" in r
        assert "reconstruction_error" in r and np.isfinite(r["reconstruction_error"])
    # reconstruction_error is a diagnostic column, not a selectable direction
    assert "reconstruction_error" not in rows.directions
    assert isinstance(rows.best_k("frontier"), int)


def test_builtin_lsa_scans():
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3], model="lsa")
    assert [r["k"] for r in rows] == [2, 3]
    assert all(np.isfinite(r["coherence"]) for r in rows)


def test_fit_hook_scans_any_model():
    c, _ = _corpus()
    rows = topica.search_k(
        c, [2, 3], fit=lambda k, s: topica.NMF(k, seed=s, weighting="count").fit(c, iters=80)
    )
    assert [r["k"] for r in rows] == [2, 3]
    assert all(np.isfinite(r["coherence"]) for r in rows)


def test_fit_hook_with_covariate_model():
    c, docs = _corpus()
    x = np.array([[i < len(docs) // 2] for i in range(len(docs))], float)
    rows = topica.search_k(c, [2, 3], fit=lambda k, s: topica.DMR(k, seed=s).fit(c, x, iters=60))
    assert [r["k"] for r in rows] == [2, 3]


def test_fit_hook_takes_precedence_and_rejects_covariate_args():
    c, _ = _corpus()
    x = np.ones((60, 1))
    with pytest.raises(ValueError, match="fit="):
        topica.search_k(c, [2], fit=lambda k, s: topica.NMF(k, seed=s).fit(c), prevalence=x)


def test_fit_hook_must_return_a_model():
    c, _ = _corpus()
    with pytest.raises(TypeError, match="topic_word"):
        topica.search_k(c, [2], fit=lambda k, s: "not a model")


def test_heldout_rejected_for_transformless_model():
    c, docs = _corpus()
    with pytest.raises(ValueError, match="transform"):
        topica.search_k(c, [2], model="nmf", held_out=docs)


def test_unknown_model_names_the_fit_hook():
    c, _ = _corpus()
    with pytest.raises(ValueError, match="fit="):
        topica.search_k(c, [2], model="bertopic")


def test_nmf_multi_seed_reconstruction_error_has_se():
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3], model="nmf", iters=80, num_seeds=2)
    # multi-seed adds an SE column for numeric metrics
    assert any("reconstruction_error_se" in r for r in rows)
