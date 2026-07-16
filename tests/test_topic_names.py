"""Tests for the settable topic_names property on every estimator.

Covers:
- Default names after fit: ["topic_0", ..., "topic_{K-1}"]
- Setter round-trip: custom names are retrieved back unchanged
- Wrong-length setter raises ValueError
- Save/load round-trip preserves custom names (for models with save/load)
"""

from __future__ import annotations

import pytest

import topica

DOCS = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default_names(k: int) -> list[str]:
    return [f"topic_{i}" for i in range(k)]


def _fit_lda():
    m = topica.models.LDA(2, seed=1)
    m.fit(DOCS, iters=100)
    return m


def _fit_dmr():
    import numpy as np
    X = [[0.0]] * 30 + [[1.0]] * 30
    m = topica.models.DMR(2, seed=1)
    m.fit(DOCS, X)
    return m


def _fit_sage():
    groups = ["a"] * 30 + ["b"] * 30
    m = topica.models.SAGE(2, seed=1)
    m.fit(DOCS, groups)
    return m


def _fit_ctm():
    m = topica.models.CTM(2, seed=1)
    m.fit(DOCS, iters=20)
    return m


def _fit_stm():
    import numpy as np
    X = [[0.0]] * 30 + [[1.0]] * 30
    m = topica.models.STM(2, seed=1)
    m.fit(DOCS, X, prevalence_names=["g"], iters=20)
    return m


def _fit_hdp():
    m = topica.models.HDP(seed=1)
    m.fit(DOCS, iters=40)
    return m


def _fit_dtm():
    times = [0] * 30 + [1] * 30
    m = topica.models.DTM(2, seed=1)
    m.fit(DOCS, times, iters=8)
    return m


def _fit_slda():
    y = [0.0] * 30 + [1.0] * 30
    m = topica.models.SupervisedLDA(2, seed=1)
    m.fit(DOCS, y, iters=10)
    return m


def _fit_pt():
    m = topica.models.PT(2, num_pseudo=10, seed=1)
    m.fit(DOCS, iters=100)
    return m


def _fit_gsdmm():
    m = topica.models.GSDMM(5, seed=1)
    m.fit(DOCS, iters=20)
    return m


def _fit_pa():
    m = topica.models.PA(2, 4, seed=1)
    m.fit(DOCS, iters=100)
    return m


def _fit_hlda():
    m = topica.models.HLDA(depth=2, seed=1)
    m.fit(DOCS, iters=50)
    return m


def _fit_labeled_lda():
    m = topica.models.LabeledLDA(seed=1)
    m.fit(DOCS, [["x"]] * 60)
    return m


def _fit_seeded_lda():
    m = topica.models.SeededLDA({"animals": ["cat", "dog"], "space": ["star", "moon"]}, seed=1)
    m.fit(DOCS, iters=200)
    return m


def _fit_keyatm():
    m = topica.models.KeyATM({"animals": ["cat", "dog"], "space": ["star", "moon"]}, seed=1)
    m.fit(DOCS, iters=200)
    return m


# ---------------------------------------------------------------------------
# Group 1: models that had no topic_names before Phase 3
# ---------------------------------------------------------------------------

class TestGroup1DefaultNames:
    """Default topic_names are ['topic_0', ..., 'topic_{K-1}'] after fit."""

    def test_lda(self):
        m = _fit_lda()
        assert m.topic_names == _default_names(m.num_topics)

    def test_dmr(self):
        m = _fit_dmr()
        assert m.topic_names == _default_names(m.num_topics)

    def test_sage(self):
        m = _fit_sage()
        assert m.topic_names == _default_names(m.num_topics)

    def test_ctm(self):
        m = _fit_ctm()
        assert m.topic_names == _default_names(m.num_topics)

    def test_stm(self):
        m = _fit_stm()
        assert m.topic_names == _default_names(m.num_topics)

    def test_hdp(self):
        m = _fit_hdp()
        assert m.topic_names == _default_names(m.num_topics)

    def test_dtm(self):
        m = _fit_dtm()
        assert m.topic_names == _default_names(m.num_topics)

    def test_slda(self):
        m = _fit_slda()
        assert m.topic_names == _default_names(m.num_topics)

    def test_pt(self):
        m = _fit_pt()
        assert m.topic_names == _default_names(m.num_topics)

    def test_gsdmm(self):
        m = _fit_gsdmm()
        k = m.num_topics
        assert m.topic_names == _default_names(k)

    def test_pa(self):
        m = _fit_pa()
        assert m.topic_names == _default_names(m.num_topics)

    def test_hlda(self):
        m = _fit_hlda()
        assert m.topic_names == _default_names(m.num_nodes)

    def test_labeled_lda(self):
        m = _fit_labeled_lda()
        assert m.topic_names == _default_names(m.num_topics)


class TestGroup1Setter:
    """Custom names can be set and retrieved."""

    def test_lda_roundtrip(self):
        m = _fit_lda()
        names = ["animals", "space"]
        m.topic_names = names
        assert m.topic_names == names

    def test_dmr_roundtrip(self):
        m = _fit_dmr()
        names = ["animals", "space"]
        m.topic_names = names
        assert m.topic_names == names

    def test_sage_roundtrip(self):
        m = _fit_sage()
        names = ["animals", "space"]
        m.topic_names = names
        assert m.topic_names == names

    def test_ctm_roundtrip(self):
        m = _fit_ctm()
        names = ["animals", "space"]
        m.topic_names = names
        assert m.topic_names == names

    def test_stm_roundtrip(self):
        m = _fit_stm()
        names = ["animals", "space"]
        m.topic_names = names
        assert m.topic_names == names

    def test_hdp_roundtrip(self):
        m = _fit_hdp()
        k = m.num_topics
        names = [f"theme_{i}" for i in range(k)]
        m.topic_names = names
        assert m.topic_names == names

    def test_pa_roundtrip(self):
        m = _fit_pa()
        names = [f"sub_{i}" for i in range(m.num_topics)]
        m.topic_names = names
        assert m.topic_names == names

    def test_hlda_roundtrip(self):
        m = _fit_hlda()
        names = [f"node_{i}" for i in range(m.num_nodes)]
        m.topic_names = names
        assert m.topic_names == names

    def test_labeled_lda_roundtrip(self):
        m = _fit_labeled_lda()
        names = [f"label_{i}" for i in range(m.num_topics)]
        m.topic_names = names
        assert m.topic_names == names


class TestGroup1WrongLength:
    """Setting topic_names with wrong length raises ValueError."""

    def test_lda_wrong_length(self):
        m = _fit_lda()
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = ["only_one"]

    def test_dmr_wrong_length(self):
        m = _fit_dmr()
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = ["too", "many", "names"]

    def test_ctm_wrong_length(self):
        m = _fit_ctm()
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = []

    def test_hlda_wrong_length(self):
        m = _fit_hlda()
        nn = m.num_nodes
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = ["a"] * (nn + 1)  # one too many


class TestGroup1SaveLoad:
    """topic_names survive a save/load round-trip."""

    def test_lda(self, tmp_path):
        m = _fit_lda()
        m.topic_names = ["animals", "space"]
        path = str(tmp_path / "lda.tt")
        m.save(path)
        m2 = topica.models.LDA.load(path)
        assert m2.topic_names == ["animals", "space"]

    def test_pt(self, tmp_path):
        m = _fit_pt()
        m.topic_names = ["animals", "space"]
        path = str(tmp_path / "pt.tt")
        m.save(path)
        m2 = topica.models.PT.load(path)
        assert m2.topic_names == ["animals", "space"]

    def test_gsdmm(self, tmp_path):
        m = _fit_gsdmm()
        k = m.num_topics
        names = [f"cluster_{i}" for i in range(k)]
        m.topic_names = names
        path = str(tmp_path / "gsdmm.tt")
        m.save(path)
        m2 = topica.models.GSDMM.load(path)
        assert m2.topic_names == names

    def test_pa(self, tmp_path):
        m = _fit_pa()
        names = [f"sub_{i}" for i in range(m.num_topics)]
        m.topic_names = names
        path = str(tmp_path / "pa.tt")
        m.save(path)
        m2 = topica.models.PA.load(path)
        assert m2.topic_names == names

    def test_hlda(self, tmp_path):
        m = _fit_hlda()
        names = [f"node_{i}" for i in range(m.num_nodes)]
        m.topic_names = names
        path = str(tmp_path / "hlda.tt")
        m.save(path)
        m2 = topica.models.HLDA.load(path)
        assert m2.topic_names == names


# ---------------------------------------------------------------------------
# Group 2: models that already had read-only topic_names; now settable
# ---------------------------------------------------------------------------

class TestGroup2Setter:
    """Group 2 models already had topic_names; setter now works."""

    def test_seeded_lda_default(self):
        m = _fit_seeded_lda()
        # SeededLDA's natural names come from seed_words keys + residual
        assert "animals" in m.topic_names or m.topic_names[0].startswith("topic_")
        assert len(m.topic_names) == m.num_topics

    def test_seeded_lda_setter(self):
        m = _fit_seeded_lda()
        names = [f"t{i}" for i in range(m.num_topics)]
        m.topic_names = names
        assert m.topic_names == names

    def test_seeded_lda_wrong_length(self):
        m = _fit_seeded_lda()
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = ["only_one"]

    def test_keyatm_default(self):
        m = _fit_keyatm()
        assert len(m.topic_names) == m.num_topics

    def test_keyatm_setter(self):
        m = _fit_keyatm()
        names = [f"t{i}" for i in range(m.num_topics)]
        m.topic_names = names
        assert m.topic_names == names

    def test_keyatm_wrong_length(self):
        m = _fit_keyatm()
        with pytest.raises(ValueError, match="topic_names"):
            m.topic_names = []

    def test_seeded_lda_save_load_setter(self, tmp_path):
        m = _fit_seeded_lda()
        names = [f"custom_{i}" for i in range(m.num_topics)]
        m.topic_names = names
        path = str(tmp_path / "slda.tt")
        m.save(path)
        m2 = topica.models.SeededLDA.load(path)
        assert m2.topic_names == names
