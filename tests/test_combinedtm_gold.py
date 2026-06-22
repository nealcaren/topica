"""Offline gold-fixture parity for topica CombinedTM vs a PyTorch AVITM reference (#271, Wave 1).

Loads the committed gold (``parity/combinedtm_gold.npz`` + ``.json``), fits topica
CombinedTM on the SAME planted-block corpus + per-document embeddings frozen in the
gold, Hungarian-aligns its topic-word matrix to the reference's, and asserts the
aligned cosine clears the reference's own seed-to-seed self-cosine floor (minus a
small margin). On this clean planted-block design the two implementations agree at
~0.97, so the bar is cleared by a wide margin.

This runs in CI WITHOUT torch: the AVITM reference fit and the exact corpus +
embeddings are frozen in the committed gold, so no torch is imported at test time
(asserted explicitly below). The topica refit (600 docs, K=5, 150 epochs) is fast
(~3s), so it stays in the default suite alongside the gold-present / shape /
non-vacuous checks.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import combinedtm_gold  # noqa: E402


def test_combinedtm_gold_present():
    npz, js = harness.gold_paths("combinedtm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/combinedtm_gold.py --regenerate` "
        "(needs torch)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_combinedtm_gold_shape():
    """Fast default check: the frozen reference topic-word matrix is (K, V) and the
    corpus / embeddings round-trip to the documented sizes."""
    arrays, meta = harness.load_gold("combinedtm")
    tw = arrays["topic_word"]
    assert tw.shape == (meta["num_topics"], meta["vocab_size"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]
    assert arrays["embeddings"].shape == (meta["num_docs"], meta["emb_dim"])


def test_combinedtm_no_torch_at_test_time():
    """The committed gold must validate with NO deep-learning framework.

    Proven in a fresh subprocess that hard-blocks torch via an import finder, then
    runs the full offline ``combinedtm_gold.run()``. A subprocess (not a bare
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
        import combinedtm_gold
        r = combinedtm_gold.run(verbose=False)
        assert r["passes"], r
        assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert "OK" in proc.stdout, f"offline gold path imported a blocked module:\n{proc.stderr}"


def test_combinedtm_gold_is_non_vacuous():
    """A vocab-shuffled topic-word matrix must lose its top-word overlap — proving
    the gate discriminates a correct fit from a wrong one. Runs by default (no fit).

    Top-word Jaccard, not raw softmax cosine, is the discriminator here: the AVITM
    softmax rows are near-flat, so even a column permutation keeps cosine high; what
    carries topic identity is the top-word ranking, which a shuffle destroys."""
    import numpy as np

    arrays, _ = harness.load_gold("combinedtm")
    tw = arrays["topic_word"].astype(np.float64)

    assert harness.top_word_jaccard(tw, tw, n=10) == 1.0  # self-overlap is total
    rng = np.random.default_rng(0)
    shuffled = tw[:, rng.permutation(tw.shape[1])]
    jac = harness.top_word_jaccard(tw, shuffled, n=10)
    assert jac < 0.5, (
        f"shuffled CombinedTM top-word Jaccard {jac:.3f} should be well below 0.5; "
        "the gate is vacuous"
    )


def test_combinedtm_matches_committed_gold():
    """Refit topica CombinedTM and compare to the frozen AVITM-reference gold."""
    r = combinedtm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica CombinedTM topic-word cosine {r['cosine']:.4f} below bar "
        f"{r['bar']:.4f} (reference self {r['reference_self_cosine']:.4f}); details: {r}"
    )
