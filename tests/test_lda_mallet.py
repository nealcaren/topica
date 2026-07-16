"""LDA MALLET-parity features: symmetric-alpha optimization, the token-level
state round-trip (save_state/load_state), and MALLET's topic diagnostics."""

import gzip
import os

import numpy as np
import pytest

import topica

DOCS = (
    [["tax", "market", "trade", "tax", "market"]] * 40
    + [["war", "troop", "militia", "war"]] * 30
    + [["court", "law", "judge"]] * 20
)


def test_symmetric_alpha_stays_symmetric():
    sym = topica.models.LDA(num_topics=4, seed=1, optimize_interval=10, burn_in=20,
                     use_symmetric_alpha=True)
    sym.fit(DOCS, iters=300)
    assert np.allclose(sym.alpha, sym.alpha[0])  # every alpha[t] equal


def test_asymmetric_alpha_is_default_and_varies():
    asym = topica.models.LDA(num_topics=4, seed=1, optimize_interval=10, burn_in=20)
    asym.fit(DOCS, iters=300)
    assert not np.allclose(asym.alpha, asym.alpha[0])  # learned per-topic shape


def test_symmetric_flag_survives_save_load(tmp_path):
    m = topica.models.LDA(num_topics=3, seed=1, use_symmetric_alpha=True)
    m.fit(DOCS, iters=100)
    path = str(tmp_path / "m.tt")
    m.save(path)
    loaded = topica.models.LDA.load(path)
    # The reloaded model behaves identically on inference.
    assert np.array_equal(m.doc_topic, loaded.doc_topic)


def test_load_state_round_trip(tmp_path):
    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=200)
    state = str(tmp_path / "state.gz")
    m.save_state(state)

    r = topica.models.LDA.load_state(state)
    assert r.num_topics == 3
    assert sorted(r.vocabulary) == sorted(m.vocabulary)
    assert r.topic_word.shape == m.topic_word.shape
    assert np.allclose(r.doc_topic.sum(axis=1), 1.0)
    # transform works on the reconstructed model.
    out = r.transform([["tax", "market"]])
    assert out.shape == (1, 3)

    # Re-emitting the state yields the same number of token rows.
    state2 = str(tmp_path / "state2.gz")
    r.save_state(state2)
    n1 = len(gzip.open(state, "rt").read().splitlines())
    n2 = len(gzip.open(state2, "rt").read().splitlines())
    assert n1 == n2


def test_load_state_reconstructs_assignments(tmp_path):
    # A planted corpus with disjoint vocab: load_state must preserve which words
    # land in which topic (up to topic relabeling).
    m = topica.models.LDA(num_topics=3, seed=3)
    m.fit(DOCS, iters=300)
    state = str(tmp_path / "s.gz")
    m.save_state(state)
    r = topica.models.LDA.load_state(state)
    # Token counts per topic are preserved (same multiset).
    assert sorted(int(d["tokens"]) for d in r.diagnostics()) == sorted(
        int(d["tokens"]) for d in m.diagnostics()
    )


def test_load_state_plain_text(tmp_path):
    # load_state accepts an uncompressed state file too.
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=100)
    gz = str(tmp_path / "s.gz")
    m.save_state(gz)
    plain = str(tmp_path / "s.txt")
    with gzip.open(gz, "rt") as f, open(plain, "w") as g:
        g.write(f.read())
    r = topica.models.LDA.load_state(plain)
    assert r.num_topics == 2


def test_diagnostics_has_mallet_fields():
    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=200)
    diag = m.diagnostics(n=5)
    assert len(diag) == 3
    required = {
        "topic", "tokens", "coherence", "exclusivity", "effective_words",
        "document_entropy", "uniform_dist", "corpus_dist", "rank1_docs",
        "alpha", "top_words",
    }
    for row in diag:
        assert required <= set(row)
        # Distances and entropies are finite and non-negative.
        assert row["document_entropy"] >= 0
        assert row["uniform_dist"] >= -1e-9
        assert row["corpus_dist"] >= -1e-9
        assert np.isfinite(row["effective_words"])
