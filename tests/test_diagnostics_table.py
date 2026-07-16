"""The one-call diagnostics table: topica.diagnostics(model, texts)."""

import numpy as np
import pytest

import topica


@pytest.fixture(scope="module")
def fitted():
    docs = [["cat", "dog", "pet", "vet", "cat"]] * 15 + \
           [["star", "moon", "sky", "sun", "star"]] * 15
    m = topica.models.LDA(2, seed=1)
    m.fit(docs, iters=300)
    return m, docs


def test_diagnostics_is_a_callable_not_the_module():
    # The module moved to topica.validation; topica.diagnostics is the table.
    assert callable(topica.diagnostics)
    assert topica.validation.__name__ == "topica.validation"
    assert hasattr(topica.validation, "frex")  # module still holds the helpers


def test_diagnostics_table_columns_and_rows(fitted):
    m, docs = fitted
    texts = [" ".join(d) for d in docs]
    df = topica.diagnostics(m, texts)
    # one row per topic, the consolidated columns
    assert len(df) == m.num_topics
    for col in ("label", "size", "prevalence", "coherence", "exclusivity",
                "stability", "top_words", "frex"):
        assert col in df.columns
    # texts -> c_v coherence (bounded), and FREX words are populated
    assert df["coherence"].between(-1.0, 1.0).all()
    assert df["frex"].str.len().gt(0).all()
    # stability is opt-in, so NaN here
    assert df["stability"].isna().all()


def test_diagnostics_umass_fallback_without_texts(fitted):
    m, _ = fitted
    df = topica.diagnostics(m)
    # no reference corpus -> UMass (negative), not c_v
    assert (df["coherence"] < 0).all()


def test_diagnostics_stability_opt_in_aligns_to_model(fitted):
    m, docs = fitted
    df = topica.diagnostics(m, docs, stability=True, n_boot=6)
    # stability is filled, one value per topic, in [0, 1], and matched to THIS model
    assert df["stability"].notna().all()
    assert df["stability"].between(0.0, 1.0).all()


def test_diagnostics_stability_needs_texts(fitted):
    m, _ = fitted
    with pytest.raises(ValueError, match="texts"):
        topica.diagnostics(m, stability=True)


def test_bootstrap_stability_accepts_reference_model(fitted):
    m, docs = fitted
    out = topica.bootstrap_stability(docs, reference=m, n_boot=6, topn=5)
    assert out["reference"] is m
    assert len(out["stability"]) == m.num_topics
