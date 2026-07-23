"""#401 — config-aware (per-instance) determinism.

`topica.effective_determinism(model, fit_settings=)` refines the coarse per-class
registry tag using the instance's configuration. Two kinds of check here:

- the *claim* the helper makes for each config (the rule table), and
- that the claim is *true* at runtime (same seed identical; a downgraded config is
  actually seed-dependent; a thread-independent path really is).

Scope is deliberately config-only: cases whose determinism depends on a runtime
outcome the config cannot see (did spectral init succeed or fall back?) keep the
common-case class plus a caveat note, and are not asserted bit-exact here.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica
from topica import effective_determinism

topica.enable_experimental()

_DOCS = [
    ["a", "b", "c", "d"],
    ["b", "c", "d", "e"],
    ["a", "d", "e", "f"],
    ["c", "e", "f", "a"],
    ["b", "f", "a", "d"],
    ["e", "a", "c", "b"],
] * 4


# --------------------------------------------------------------------------
# The rule table (what the helper claims)
# --------------------------------------------------------------------------

def test_cvb0_is_seed_reproducible_and_thread_independent():
    for m in (
        topica.LDA(3, sampler="cvb0"),
        topica.DMR(3, sampler="cvb0"),
        topica.KeyATM({"a": ["x"]}, num_topics=3, sampler="cvb0"),
    ):
        det = effective_determinism(m)
        assert det["effective"] == "seed-reproducible"
        assert "seed" in det["replay_requires"]
        assert "num_threads" not in det["replay_requires"]  # cvb0 is thread-independent


def test_nmf_init_conditional():
    bit = effective_determinism(topica.NMF(3, init="nndsvd"))
    assert bit["effective"] == "bit-exact"
    assert bit["replay_requires"] == {}
    rnd = effective_determinism(topica.NMF(3, init="random"))
    assert rnd["effective"] == "seed-reproducible"
    assert rnd["replay_requires"].get("seed") == 42


def test_ctm_stm_init_and_svi():
    # random init downgrades
    assert effective_determinism(topica.CTM(3, init="random"))["effective"] == "seed-reproducible"
    assert effective_determinism(topica.STM(3, init="random"))["effective"] == "seed-reproducible"
    # svi (a fit-time arg) downgrades even under spectral init
    svi = effective_determinism(topica.CTM(3), fit_settings={"inference": "svi"})
    assert svi["effective"] == "seed-reproducible"
    # default spectral batch stays bit-exact, with an honest fallback caveat
    spec = effective_determinism(topica.CTM(3))
    assert spec["effective"] == "bit-exact"
    assert any("spectral" in n for n in spec["notes"])


def test_gibbs_thread_conditionality():
    # single-threaded: records num_threads=1
    d1 = effective_determinism(topica.LDA(3))
    assert d1["effective"] == "seed-reproducible"
    assert d1["replay_requires"].get("num_threads") == 1
    # multi-threaded via a fit-time override: replay needs that count, with a note
    d4 = effective_determinism(topica.LDA(3), fit_settings={"num_threads": 4})
    assert d4["replay_requires"].get("num_threads") == 4
    assert any("num_threads=4" in n for n in d4["notes"])


def test_wordfish_unconditionally_bit_exact():
    det = effective_determinism(topica.Wordfish())
    assert det["effective"] == "bit-exact"
    assert det["replay_requires"] == {}


def test_anchorlda_keeps_class_with_caveat():
    det = effective_determinism(topica.AnchorLDA(3))
    assert det["effective"] == "bit-exact"
    assert det["notes"], "AnchorLDA should carry its fallback/backend caveat"


def test_llm_bounded():
    det = effective_determinism(topica.TopicGPT(backend=lambda p: ""))
    assert det["effective"] == "llm-bounded"
    assert det["replay_requires"] == {}


@pytest.mark.parametrize(
    "model,expected",
    [
        (topica.LDA(3), "seed-reproducible"),
        (topica.NMF(3), "bit-exact"),
        (topica.CTM(3), "bit-exact"),
        (topica.STM(3), "bit-exact"),
        (topica.ProdLDA(3), "seed-reproducible"),
        (topica.HDP(), "seed-reproducible"),
    ],
)
def test_default_config_matches_registry_tag(model, expected):
    det = effective_determinism(model)
    assert det["effective"] == expected == det["registry_class"]


# --------------------------------------------------------------------------
# The claims are true at runtime
# --------------------------------------------------------------------------

def test_nmf_nndsvd_seed_independent_random_seed_dependent():
    a = topica.NMF(3, init="nndsvd", seed=1); a.fit(_DOCS)
    b = topica.NMF(3, init="nndsvd", seed=999); b.fit(_DOCS)
    assert np.allclose(a.topic_word, b.topic_word)  # nndsvd: seed-independent
    c = topica.NMF(3, init="random", seed=1); c.fit(_DOCS)
    d = topica.NMF(3, init="random", seed=999); d.fit(_DOCS)
    assert not np.allclose(c.topic_word, d.topic_word)  # random: seed-dependent


def test_cvb0_seed_dependent_but_thread_independent():
    a = topica.LDA(3, sampler="cvb0", seed=1); a.fit(_DOCS, iters=20, num_threads=1)
    b = topica.LDA(3, sampler="cvb0", seed=1); b.fit(_DOCS, iters=20, num_threads=4)
    assert np.allclose(a.topic_word, b.topic_word)  # thread-independent
    c = topica.LDA(3, sampler="cvb0", seed=7); c.fit(_DOCS, iters=20)
    assert not np.allclose(a.topic_word, c.topic_word)  # seed-dependent


# --------------------------------------------------------------------------
# Manifest integration
# --------------------------------------------------------------------------

def test_manifest_records_effective_determinism():
    m = topica.NMF(3, init="random"); m.fit(_DOCS)
    rec = topica.record_fit(m, corpus=topica.Corpus.from_documents(_DOCS))
    assert rec.model["determinism"] == "seed-reproducible"
    detail = rec.model["determinism_detail"]
    assert detail["registry_class"] == "bit-exact"
    assert detail["replay_requires"].get("seed") == 42


def test_manifest_cvb0_detail_has_no_thread_requirement():
    m = topica.LDA(3, sampler="cvb0"); m.fit(_DOCS, iters=10)
    rec = topica.record_fit(m, corpus=topica.Corpus.from_documents(_DOCS))
    assert rec.model["determinism"] == "seed-reproducible"
    assert "num_threads" not in rec.model["determinism_detail"]["replay_requires"]
