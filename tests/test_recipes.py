"""Smoke tests: every task recipe in examples/recipes/ runs end to end.

The recipes are meant to be transplanted onto a user's own data, so "it runs on
the bundled data" is the contract that keeps them trustworthy. Each is executed
as a script (``runpy.run_path``) and must complete without error and print its
header. The gadarian recipes use bundled data and always run; the DTM recipe
fetches ``load_congress`` and is skipped when that dataset is unavailable
(offline CI), mirroring ``test_dubois_tutorial``.
"""
import io
import os
import runpy
from contextlib import redirect_stdout

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(os.path.dirname(HERE), "examples", "recipes")


def _run(name: str) -> str:
    path = os.path.join(RECIPES, name)
    buf = io.StringIO()
    with redirect_stdout(buf):
        runpy.run_path(path, run_name="__main__")
    return buf.getvalue()


def _dataset_available(loader: str) -> bool:
    try:
        import topica

        getattr(topica.datasets, loader)()
        return True
    except Exception:
        return False


def test_lda_explore_runs():
    out = _run("lda_explore.py")
    assert "frontier-suggested K" in out
    assert "coherence" in out


def test_keyatm_seeded_runs():
    out = _run("keyatm_seeded.py")
    assert "keyword_rate" in out
    assert "Unseeded topics" in out


def test_gsdmm_short_text_runs():
    out = _run("gsdmm_short_text.py")
    assert "clusters retained" in out


def test_robustness_runs():
    out = _run("robustness.py")
    assert "Bootstrap topic stability" in out
    assert "mean stability" in out


def test_provenance_runs(tmp_path, monkeypatch):
    # provenance.py writes a manifest JSON to the cwd; run it in a temp dir so it
    # does not litter the repo.
    monkeypatch.chdir(tmp_path)
    out = _run("provenance.py")
    assert "Analysis card" in out
    assert (tmp_path / "gadarian_manifest.json").exists()


@pytest.mark.skipif(not _dataset_available("load_ng20_minilm"),
                    reason="load_ng20_minilm unavailable (offline)")
def test_bertopic_embeddings_runs():
    out = _run("bertopic_embeddings.py")
    assert "topics discovered" in out


def test_stm_prevalence_groups_runs():
    out = _run("stm_prevalence_groups.py")
    assert "prevalence" in out
    assert "CI excludes zero" in out


def test_stm_content_groups_runs():
    out = _run("stm_content_groups.py")
    assert "content = ~" in out
    assert "divergence" in out


@pytest.mark.skipif(not _dataset_available("load_congress"),
                    reason="load_congress unavailable (offline)")
def test_dtm_over_time_runs():
    out = _run("dtm_over_time.py")
    assert "DTM with K=" in out
    assert "over time:" in out
