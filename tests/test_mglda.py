"""Tests for MGLDA: Multi-Grain LDA (Titov & McDonald, WWW 2008), validated against
tomotopy's MGLDAModel in parity/mglda_gold.py.

MG-LDA learns global (document-level) and local (sliding-window aspect) topics with a
per-token grain switch. Input is sentence-segmented: list[list[list[str]]]
(doc -> sentences -> tokens). doc_topic is an empirical prevalence over the combined
[global | local] topic set (rows sum to 1), not a single Dirichlet theta.
"""

import numpy as np
import pytest

import topica


def _planted(reps=60, seed=0):
    """2 global themes (doc-level) x 3 local aspects (per-sentence), sentence-segmented."""
    rng = np.random.default_rng(seed)
    themes = {0: ["phone", "android", "mobile"], 1: ["laptop", "notebook", "keyboard"]}
    aspects = {0: ["battery", "charge", "power"], 1: ["screen", "display", "bright"],
               2: ["price", "cost", "cheap"]}
    docs = []
    for _ in range(reps):
        g = int(rng.integers(0, 2))
        sents = []
        for _ in range(6):
            a = int(rng.integers(0, 3))
            sent = [themes[g][int(rng.integers(0, 3))] for _ in range(3)]
            sent += [aspects[a][int(rng.integers(0, 3))] for _ in range(3)]
            sents.append(sent)
        docs.append(sents)
    return docs


def test_shapes_and_normalization():
    docs = _planted()
    m = topica.MGLDA(2, 3, window=3, seed=0).fit(docs, iters=200)
    V = len(m.vocabulary)
    assert m.num_topics == 5
    assert m.topic_word.shape == (5, V)
    assert m.global_topic_word.shape == (2, V)
    assert m.local_topic_word.shape == (3, V)
    assert m.doc_topic.shape == (len(docs), 5)
    assert m.global_doc_topic.shape == (len(docs), 2)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert np.allclose(m.global_doc_topic.sum(axis=1), 1.0)
    assert 0.0 <= m.global_fraction <= 1.0


def test_recovers_global_themes():
    # The robust signal: global topics recover the two document-level themes distinctly.
    # (Local-grain survival is data-dependent for MG-LDA — see parity/mglda_gold.py.)
    docs = _planted()
    m = topica.MGLDA(2, 3, window=3, seed=0).fit(docs, iters=300)
    themes = [{"phone", "android", "mobile"}, {"laptop", "notebook", "keyboard"}]
    top_sets = [{w for w, _ in m.top_words(3, topic=t)} for t in range(2)]
    # each global topic matches a distinct planted theme block
    matched = set()
    for ts in top_sets:
        best = max(range(2), key=lambda i: len(ts & themes[i]))
        assert len(ts & themes[best]) >= 2, f"global topic not a theme: {ts}"
        matched.add(best)
    assert matched == {0, 1}, "global topics did not split into distinct themes"


def test_determinism():
    docs = _planted()
    a = topica.MGLDA(2, 3, seed=7).fit(docs, iters=120)
    b = topica.MGLDA(2, 3, seed=7).fit(docs, iters=120)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert a.global_fraction == b.global_fraction


def test_rejects_flat_input():
    # A flat list[list[str]] must be rejected, not silently trained on characters.
    with pytest.raises((ValueError, TypeError)):
        topica.MGLDA(2, 3).fit([["this", "is", "flat"], ["another", "doc"]], iters=5)


def test_rejects_string_document():
    with pytest.raises((ValueError, TypeError)):
        topica.MGLDA(2, 3).fit(["a whole document as a string"], iters=5)


def test_handles_short_and_single_sentence_docs():
    # S < T (short) and single-sentence documents must not crash.
    docs = [
        [["phone", "battery", "good"]],                       # S=1 < T=3
        [["laptop", "screen"], ["price", "cheap"]],           # S=2 < T=3
        [["phone", "android"], ["battery", "power"], ["screen", "bright"]],
    ]
    m = topica.MGLDA(2, 2, window=3, seed=0).fit(docs, iters=30)
    assert m.topic_word.shape[0] == 4
    assert m.doc_topic.shape[0] == 3


def test_doc_alignment_preserved_with_empty_docs():
    # An all-empty (or all-OOV) document must NOT be dropped: doc_topic keeps a 1:1 row
    # correspondence with the input so downstream metadata joins stay aligned.
    docs = [
        [["phone", "android"], ["battery", "power"]],
        [[], []],                       # entirely empty document (kept as a uniform row)
        [["laptop", "keyboard"], ["price", "cheap"]],
    ]
    m = topica.MGLDA(2, 2, window=2, seed=0).fit(docs, iters=20)
    assert m.doc_topic.shape[0] == 3, "empty document must not shift row alignment"
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    # the empty doc's row is the uniform fallback
    assert np.allclose(m.doc_topic[1], 1.0 / 4)


def test_warns_when_local_grain_collapses():
    # On text without within-document aspect locality the grain switch routes ~all
    # tokens global; fit must warn so a prior-dominated local table isn't taken as signal.
    docs = _planted()  # synthetic, global-dominant
    with pytest.warns(UserWarning, match="local grain"):
        topica.MGLDA(2, 3, seed=0).fit(docs, iters=200)


def test_window_one():
    docs = _planted(reps=20)
    m = topica.MGLDA(2, 3, window=1, seed=0).fit(docs, iters=40)
    assert m.num_topics == 5
    assert 0.0 <= m.global_fraction <= 1.0


def test_settings_and_save_load(tmp_path):
    docs = _planted(reps=30)
    m = topica.MGLDA(2, 3, window=2, alpha_global=0.2, beta_local=0.02, gamma=0.5, seed=5)
    m.fit(docs, iters=80)
    assert m.settings == {
        "num_global_topics": 2, "num_local_topics": 3, "window": 2,
        "alpha_global": 0.2, "alpha_local": 0.1, "alpha_mix_global": 0.1,
        "alpha_mix_local": 0.1, "beta_global": 0.01, "beta_local": 0.02,
        "gamma": 0.5, "seed": 5,
    }
    p = str(tmp_path / "m.topica")
    m.save(p)
    L = topica.MGLDA.load(p)
    assert np.array_equal(L.topic_word, m.topic_word)
    assert np.array_equal(L.global_topic_word, m.global_topic_word)
    assert np.array_equal(L.doc_topic, m.doc_topic)
    assert L.global_fraction == m.global_fraction
    assert L.settings == m.settings


def test_fit_history_and_converged():
    docs = _planted(reps=30)
    m = topica.MGLDA(2, 3, seed=0).fit(docs, iters=100)
    assert len(m.fit_history) > 0
    it, ll = m.fit_history[0]
    assert isinstance(it, int) and it > 0 and isinstance(ll, float)
    assert m.converged is False


def test_coherence_shape():
    docs = _planted(reps=30)
    m = topica.MGLDA(2, 3, seed=0).fit(docs, iters=80)
    assert m.coherence(5).shape == (5,)


def test_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        topica.MGLDA(0, 3)
    with pytest.raises(ValueError):
        topica.MGLDA(2, 3, gamma=0.0)
    with pytest.raises(ValueError):
        topica.MGLDA(2, 3, window=0)
