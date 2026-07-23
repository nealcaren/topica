"""#410 — the fitted model records the initialization route it actually took,
so config-aware determinism (#401) can make an exact claim instead of a caveat.

`model.initialization` is one of "spectral" / "random-fallback" / "random" /
"provided" (None before fit / for a pre-#410 save). `effective_determinism` reads
it: a confirmed spectral fit is bit-exact with no caveat; a spectral run that
silently fell back to a seeded random init is correctly seed-reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica
from topica import effective_determinism as ed

topica.enable_experimental()

_DOCS = [
    ["a", "b", "c", "d"],
    ["b", "c", "d", "e"],
    ["a", "d", "e", "f"],
    ["c", "e", "f", "a"],
    ["b", "f", "a", "d"],
    ["e", "a", "c", "b"],
] * 4
_X = np.random.RandomState(0).randn(len(_DOCS), 1)
_SENT = [(-1.0 if i % 2 else 1.0) for i in range(len(_DOCS))]
_TIMES = [i % 3 for i in range(len(_DOCS))]

def _fit(m, **kw):
    m.fit(_DOCS, **kw)
    return m


def test_ctm_stm_sts_record_spectral_vs_random():
    assert _fit(topica.CTM(3)).initialization == "spectral"
    assert _fit(topica.CTM(3, init="random")).initialization == "random"
    assert _fit(topica.STM(3), prevalence=_X).initialization == "spectral"
    assert _fit(topica.STM(3, init="random"), prevalence=_X).initialization == "random"
    sts = topica.STS(3)
    sts.fit(_DOCS, _SENT)
    assert sts.initialization == "spectral"
    sts_r = topica.STS(3, init="random")
    sts_r.fit(_DOCS, _SENT)
    assert sts_r.initialization == "random"


def test_dtm_default_is_random_route():
    # DTM's constructor default is init="random" (gensim-style), unlike STM/CTM.
    d = topica.DTM(3)
    d.fit(_DOCS, times=_TIMES, iters=3)
    assert d.initialization == "random"
    ds = topica.DTM(3, init="spectral")
    ds.fit(_DOCS, times=_TIMES, iters=3)
    assert ds.initialization in ("spectral", "random-fallback")


def test_spectral_fallback_is_recorded():
    # K larger than the tiny vocabulary makes spectral recovery return None, so the
    # fit silently falls back to a seeded random init — now recorded, not hidden.
    tiny = [["a", "b"], ["b", "a"]] * 6
    m = topica.CTM(3)
    m.fit(tiny)
    assert m.initialization == "random-fallback"


def test_unfitted_is_none():
    assert topica.CTM(3).initialization is None
    assert topica.STM(3).initialization is None
    assert topica.DTM(3).initialization is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: _fit(topica.CTM(3)),
        lambda: _fit(topica.STM(3), prevalence=_X),
        lambda: _dtm(),
    ],
)
def test_initialization_survives_save_load(make, tmp_path):
    m = make()
    route = m.initialization
    assert route is not None
    p = str(tmp_path / "m.bin")
    m.save(p)
    assert type(m).load(p).initialization == route


def _dtm():
    d = topica.DTM(3)
    d.fit(_DOCS, times=_TIMES, iters=3)
    return d


def test_sts_save_load_route(tmp_path):
    m = topica.STS(3)
    m.fit(_DOCS, _SENT)
    p = str(tmp_path / "sts.bin")
    m.save(p)
    assert topica.STS.load(p).initialization == m.initialization


# --- effective_determinism uses the recorded route (exact, no caveat) ----------

def test_effective_determinism_exact_from_route():
    spec = _fit(topica.CTM(3))
    r = ed(spec)
    assert r["effective"] == "bit-exact"
    assert r["notes"] == []  # confirmed spectral: no fallback caveat

    rnd = _fit(topica.CTM(3, init="random"))
    assert ed(rnd)["effective"] == "seed-reproducible"

    tiny = [["a", "b"], ["b", "a"]] * 6
    fb = topica.CTM(3)
    fb.fit(tiny)
    assert ed(fb)["effective"] == "seed-reproducible"  # fallback detected


def test_unfitted_keeps_config_caveat():
    r = ed(topica.CTM(3))
    assert r["effective"] == "bit-exact"
    assert any("spectral" in n for n in r["notes"])  # caveat retained when route unknown


def test_anchorlda_fallback_flag_and_determinism():
    a = topica.AnchorLDA(3)
    a.fit(_DOCS)
    assert a.anchor_fallback_used is False  # clean corpus, deterministic selection
    assert topica.AnchorLDA(3).anchor_fallback_used is None  # unfitted
    r = ed(a)
    assert r["effective"] == "bit-exact"
    assert not any("fallback" in n for n in r["notes"])  # no fallback caveat when it didn't fire


def test_manifest_records_route_backed_determinism():
    m = _fit(topica.CTM(3))
    rec = topica.record_fit(m, corpus=topica.Corpus.from_documents(_DOCS))
    assert rec.model["determinism"] == "bit-exact"
    assert rec.model["determinism_detail"]["notes"] == []
