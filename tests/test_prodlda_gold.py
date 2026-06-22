"""Offline gold-fixture parity for topica ProdLDA vs a PyTorch AVITM reference (#271, Wave 1).

Loads the committed gold (``parity/prodlda_gold.npz`` + ``.json``), fits topica
ProdLDA on the SAME 20-newsgroups token corpus frozen in the gold, scores its mean
c_npmi with topica's own coherence, and asserts it clears the one-sided lower bar
``reference_mean - 2 * reference_sd`` — i.e. topica is no worse than the PyTorch
AVITM reference within the reference's own seed-to-seed noise. This is the
objective/coherence-parity bar the live ``prodlda_compare.py`` already uses (real
text, no topic-word cosine): in practice topica is *sharper* and lands above the
reference mean, which is a pass, not a failure.

This runs in CI WITHOUT torch / sklearn: the reference fit, its c_npmi noise floor,
and the exact token corpus are frozen in the committed gold, so neither is imported
at test time (asserted explicitly below). The topica ProdLDA refit (~1.5k docs,
V~1.1k, 120 epochs) takes ~15s, so the heavy refit assertion is marked
``@pytest.mark.slow`` (deselected by the default suite); the gold-present / shape /
non-vacuous checks run by default.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import prodlda_gold  # noqa: E402


def test_prodlda_gold_present():
    npz, js = harness.gold_paths("prodlda")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/prodlda_gold.py --regenerate` "
        "(needs torch + sklearn)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_prodlda_gold_shape():
    """Fast default check: the frozen reference topic-word matrix is (K, V), the
    corpus round-trips, and the frozen c_npmi noise floor is recorded."""
    arrays, meta = harness.load_gold("prodlda")
    tw = arrays["topic_word"]
    assert tw.shape == (meta["num_topics"], meta["vocab_size"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]
    assert meta["reference_c_npmi_sd"] > 0.0


def test_prodlda_no_torch_at_test_time():
    """The committed gold must validate with NO deep-learning framework.

    Proven in a fresh subprocess that hard-blocks torch via an import finder, then
    runs the full offline ``prodlda_gold.run()``. A subprocess (not a bare
    ``sys.modules`` check) is used because a sibling test in the same pytest session
    may have already imported torch; the point is that the GOLD PATH needs none."""
    script = textwrap.dedent(
        f"""
        import sys
        BLOCKED = {{"torch"}}
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked {{name}} for offline gold test")
                return None
        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, {str(PARITY)!r})
        import prodlda_gold
        r = prodlda_gold.run(verbose=False)
        assert r["passes"], r
        assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert "OK" in proc.stdout, f"offline gold path imported a blocked module:\n{proc.stderr}"


def test_prodlda_gold_is_non_vacuous():
    """A vocab-shuffled topic-word matrix must lose its top-word overlap — proving
    the frozen reference topics carry real structure, not noise. Runs by default
    (no fit, no torch).

    Top-word Jaccard, not raw softmax cosine, is the discriminator here: the AVITM
    softmax rows are near-flat, so even a column permutation keeps cosine high; what
    carries topic identity is the top-word ranking, which a shuffle destroys."""
    import numpy as np

    arrays, _ = harness.load_gold("prodlda")
    tw = arrays["topic_word"].astype(np.float64)

    assert harness.top_word_jaccard(tw, tw, n=10) == 1.0  # self-overlap is total
    rng = np.random.default_rng(0)
    shuffled = tw[:, rng.permutation(tw.shape[1])]
    jac = harness.top_word_jaccard(tw, shuffled, n=10)
    assert jac < 0.5, (
        f"shuffled ProdLDA top-word Jaccard {jac:.3f} should be well below 0.5; "
        "the frozen reference topics carry no structure"
    )


@pytest.mark.slow
def test_prodlda_matches_committed_gold():
    """Refit topica ProdLDA on the frozen corpus and check the coherence-parity bar.

    Marked ``slow`` (the 120-epoch refit is ~15s); the default suite deselects it.
    Run with ``-m slow``.
    """
    r = prodlda_gold.run(verbose=False)
    assert r["passes"], (
        f"topica ProdLDA c_npmi {r['topica_c_npmi']:.4f} below lower bar "
        f"{r['lower_bar']:.4f} (reference {r['reference_c_npmi_mean']:.4f} "
        f"+/- {r['reference_c_npmi_sd']:.4f}); details: {r}"
    )
