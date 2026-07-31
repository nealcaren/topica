"""Reviewer-safety fixes from the sample-user LDA-workflow audit (issue #647).

Each test pins one finding: a footgun that could put a wrong number in a paper,
an opaque failure, or a silent drop — the things two first-time-user agents hit
running the full LDA workflow end to end.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import topica
from topica.validation import SearchKResult


# --- estimate_effect: intercept footgun + missing p-value -----------------------

def _effects():
    # A tiny (D, K) theta and a one-covariate design; add_intercept=True by default
    # so feature_names becomes ['intercept', 'x'].
    rng = np.random.default_rng(0)
    theta = rng.dirichlet(np.ones(3), size=40)
    x = (np.arange(40) < 20).astype(float)  # binary covariate
    return topica.estimate_effect(theta, X=x, feature_names=["x"])


def test_effect_has_pvalue_aligned_to_features():
    e = _effects()[0]
    assert e.feature_names == ["intercept", "x"]
    p = e.pvalue
    assert p.shape == (2,)
    assert np.all((p >= 0) & (p <= 1))
    # pvalue reaches the tidy frame and the dict summary.
    assert "pvalue" in e.to_frame().columns
    assert "pvalue" in e.as_dict()["x"]


def test_effect_named_access_avoids_intercept_footgun():
    # coef[0] is the INTERCEPT, not the covariate — the silent-wrong-number trap.
    # effect_of / by_feature read the covariate by name instead of by position.
    e = _effects()[0]
    assert e.effect_of("x")["coef"] == pytest.approx(float(e.coef[1]))
    assert e.effect_of("x")["coef"] != pytest.approx(float(e.coef[0]))  # not the intercept
    assert set(e.by_feature) == {"intercept", "x"}
    assert e.by_feature["x"]["pvalue"] == pytest.approx(float(e.pvalue[1]))
    with pytest.raises(KeyError, match="not a covariate"):
        e.effect_of("nope")


def test_effect_pvalue_is_nan_where_z_is_nan():
    e = _effects()[0]
    e.z[1] = np.nan
    assert np.isnan(e.pvalue[1])


# --- best_k: boundary warning on a monotone metric ------------------------------

def test_best_k_warns_at_grid_boundary_for_heldout():
    rows = SearchKResult([
        {"k": 5, "heldout_loglik": -100.0},
        {"k": 10, "heldout_loglik": -90.0},
        {"k": 15, "heldout_loglik": -80.0},  # optimum at the largest K -> boundary
    ])
    with pytest.warns(UserWarning, match="largest K scanned"):
        assert rows.best_k() == 15  # default metric resolves to heldout_loglik


def test_best_k_no_warning_for_interior_optimum():
    rows = SearchKResult([
        {"k": 5, "heldout_loglik": -100.0},
        {"k": 10, "heldout_loglik": -80.0},  # interior optimum
        {"k": 15, "heldout_loglik": -90.0},
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rows.best_k() == 10


# --- coherence_ci: accept (model, corpus); reject swapped args ------------------

def _corpus_and_model():
    docs = [["a", "b", "c"], ["a", "b", "d"], ["c", "d", "e"], ["a", "c", "e"]] * 8
    corpus = topica.Corpus.from_documents(docs)
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(corpus, iters=40)
    return corpus, m


def test_coherence_ci_accepts_model_and_corpus():
    corpus, m = _corpus_and_model()
    ci = topica.coherence_ci(m, corpus, n_boot=20, seed=0)
    assert ci.estimate.shape == (2,)


def test_coherence_ci_rejects_model_as_texts_clearly():
    _, m = _corpus_and_model()
    with pytest.raises(TypeError, match="looks like a fitted model"):
        topica.coherence_ci([["a", "b"]], m, n_boot=5)


# --- from_dataframe: warn when max_doc_fraction drops corpus-defining terms ------

def _frame():
    # "core" appears in every doc (100% > 0.5); the rest vary.
    docs = [
        "core alpha beta", "core alpha gamma", "core beta delta",
        "core gamma delta", "core alpha beta", "core delta gamma",
    ]
    return pd.DataFrame({"text": docs, "grp": list("aabbab")})


def test_from_dataframe_warns_on_high_df_pruning():
    with pytest.warns(UserWarning, match="max_doc_fraction=0.5 drops"):
        topica.from_dataframe(_frame(), text_col="text", max_doc_fraction=0.5)


def test_from_dataframe_no_warning_without_upper_pruning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        topica.from_dataframe(_frame(), text_col="text", min_doc_freq=1)
