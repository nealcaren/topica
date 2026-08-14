"""search_k across model types: the built-in nmf/lsa strings and the generic
fit=(k, seed) -> fitted model hook (model-agnostic, like model_factory on the
other tools)."""

import warnings

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
    # reconstruction_error is a scree curve: selectable by name (issue #730), but
    # monotone in K so it stays out of the frontier and the best_k default.
    assert rows.directions["reconstruction_error"] == "minimize"
    assert isinstance(rows.best_k("frontier"), int)
    # reconstruction_error is now selectable by name (rule='best' or 'elbow'),
    # where before it raised "unknown metric".
    assert rows.best_k("reconstruction_error") in (2, 3, 4)
    assert rows.best_k("reconstruction_error", rule="elbow") in (2, 3, 4)


def test_builtin_lsa_scans():
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3], model="lsa")
    assert [r["k"] for r in rows] == [2, 3]
    assert all(np.isfinite(r["coherence"]) for r in rows)


def test_lsa_reports_reconstruction_error_scree():
    # issue #733 Tier 2: LSA now reports a reconstruction_error scree column, like
    # NMF (was omitted though best_k's docstring advertised it as an NMF/LSA column).
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3, 4, 5], model="lsa")
    errs = [r["reconstruction_error"] for r in rows]
    assert all(np.isfinite(e) for e in errs)
    # rank-K Frobenius residual falls monotonically as K grows
    assert all(a >= b for a, b in zip(errs, errs[1:]))
    assert rows.directions["reconstruction_error"] == "minimize"
    assert rows.best_k("reconstruction_error", rule="elbow") in (2, 3, 4, 5)


def test_lsa_reconstruction_error_survives_save_load(tmp_path):
    c, _ = _corpus()
    m = topica.LSA(num_topics=4).fit(c)
    p = tmp_path / "lsa.bin"
    m.save(str(p))
    assert topica.LSA.load(str(p)).reconstruction_error == pytest.approx(
        m.reconstruction_error
    )


def test_lsa_omits_dispersion_signed_factors():
    # LSA's topic_word is a signed SVD factor, so the multinomial residual
    # dispersion test is meaningless (~1e9). It must be skipped, not reported.
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3, 4], model="lsa")
    assert not any("dispersion" in r for r in rows)
    assert "dispersion" not in rows.directions


def test_nmf_omits_dispersion_not_a_generative_count_model():
    # NMF factors a tf-idf matrix, not counts, so Taddy's multinomial residual
    # dispersion is meaningless (non-monotone garbage on real corpora). It must be
    # omitted, like LSA — the column is gated on a generative transform.
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3], model="nmf", iters=100)
    assert not any("dispersion" in r for r in rows)


def test_lda_keeps_dispersion_generative_model():
    c, _ = _corpus()
    rows = topica.search_k(c, [2, 3], model="lda", iters=150)
    for r in rows:
        assert "dispersion" in r and np.isfinite(r["dispersion"])


def test_criteria_omitted_for_signed_lsa_no_warning():
    # deveaud (JS divergence) / cao_juan (cosine) are distribution metrics; LSA's
    # signed loadings make them NaN/artifacts and leak a numpy RuntimeWarning.
    # They must be omitted cleanly, and NMF (non-negative) must keep them.
    c, _ = _corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rl = topica.search_k(c, [2, 3], model="lsa", criteria=("deveaud", "cao_juan"))
    assert not any("deveaud" in r or "cao_juan" in r for r in rl)
    rn = topica.search_k(c, [2, 3], model="nmf", iters=100, criteria=("deveaud", "cao_juan"))
    assert all("deveaud" in r and "cao_juan" in r for r in rn)


def test_fit_hook_without_doc_topic_skips_dispersion():
    # A fit= model exposing topic_word/top_words but no doc_topic must not crash
    # in the dispersion test; the column is simply omitted.
    c, _ = _corpus()

    class TWOnly:
        topic_word = np.abs(np.random.default_rng(0).normal(size=(3, 5)))
        vocabulary = ["cat", "dog", "pet", "vet", "paw"]

        def top_words(self, n, topic=None):
            return [[(f"w{i}", 1.0) for i in range(n)] for _ in range(3)]

    rows = topica.search_k(c, [3], fit=lambda k, s: TWOnly())  # must not crash
    assert "dispersion" not in rows[0]
    assert "coherence" in rows[0] and "exclusivity" in rows[0]


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
    # NMF's default init='nndsvd' is deterministic, so every seed is identical:
    # the SE column is still added (mechanics), but it is 0 and search_k warns
    # that num_seeds adds no robustness information here (issue #733 Tier 1).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = topica.search_k(c, [2, 3], model="nmf", iters=80, num_seeds=2)
    assert any("reconstruction_error_se" in r for r in rows)
    assert all(r["reconstruction_error_se"] == 0.0 for r in rows)
    assert any("identical fit" in str(x.message) for x in w)
