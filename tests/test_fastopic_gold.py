"""Offline gold-fixture parity for topica FASTopic vs the `fastopic` package (#271, Wave 1).

Loads the committed gold (``parity/fastopic_gold.npz`` + ``.json``), fits topica
FASTopic on the SAME 20-newsgroups subsample + frozen MiniLM document embeddings
that the gold was built from, restricts both topic-word matrices to their shared
vocabulary, Hungarian-aligns, and asserts the aligned cosine clears the reference's
own seed-to-seed self-cosine floor (minus a margin). On real MiniLM embeddings the
two implementations agree at ~0.61, a clear step below the reference's own ~0.69
self-cosine (the PCA/Adam vs autodiff-Sinkhorn gap), so the bar is the wider
``self - 0.15`` and is cleared by ~0.06.

This runs in CI WITHOUT fastopic / torch / sentence-transformers: the reference fit
and the exact corpus + embeddings are frozen in the committed gold, so none of those
are imported at test time (asserted explicitly below). The topica refit (~420 docs,
K=10) is ~13s — under the slow threshold, so it stays in the default suite alongside
the gold-present / shape / non-vacuous checks.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import fastopic_gold  # noqa: E402


def test_fastopic_gold_present():
    npz, js = harness.gold_paths("fastopic")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/fastopic_gold.py --regenerate` "
        "(needs fastopic + sentence-transformers)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_fastopic_gold_shape():
    """Fast default check: the frozen reference topic-word matrix is (K, V) and the
    corpus / embeddings round-trip to the documented sizes."""
    arrays, meta = harness.load_gold("fastopic")
    tw = arrays["topic_word"]
    assert tw.shape == (meta["num_topics"], meta["vocab_size"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]
    assert arrays["embeddings"].shape == (meta["num_docs"], meta["emb_dim"])


def test_fastopic_no_heavy_import_at_test_time():
    """The committed gold must validate with NO deep-learning / embedding framework.

    Proven in a fresh subprocess that hard-blocks torch / fastopic /
    sentence_transformers via an import finder, then runs the full offline
    ``fastopic_gold.run()``. A subprocess (not a bare ``sys.modules`` check) is used
    because a sibling live parity test in the same pytest session may have already
    imported these heavy modules; the point is that the GOLD PATH itself needs none."""
    script = textwrap.dedent(
        f"""
        import sys
        BLOCKED = {{"torch", "fastopic", "sentence_transformers", "topmost"}}
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked {{name}} for offline gold test")
                return None
        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, {str(PARITY)!r})
        import fastopic_gold
        r = fastopic_gold.run(verbose=False)
        assert r["passes"], r
        assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert "OK" in proc.stdout, f"offline gold path imported a blocked module:\n{proc.stderr}"


def test_fastopic_gold_is_non_vacuous():
    """A vocab-shuffled topic-word matrix must lose its top-word overlap — proving
    the gate discriminates a correct fit from a wrong one. Runs by default (no fit).

    Top-word Jaccard, not raw softmax cosine, is the discriminator: the FASTopic
    beta rows are near-flat over a 3k-word vocabulary, so a column permutation can
    keep cosine high; what carries topic identity is the top-word ranking, which a
    shuffle destroys."""
    import numpy as np

    arrays, _ = harness.load_gold("fastopic")
    tw = arrays["topic_word"].astype(np.float64)

    assert harness.top_word_jaccard(tw, tw, n=10) == 1.0  # self-overlap is total
    rng = np.random.default_rng(0)
    shuffled = tw[:, rng.permutation(tw.shape[1])]
    jac = harness.top_word_jaccard(tw, shuffled, n=10)
    assert jac < 0.5, (
        f"shuffled FASTopic top-word Jaccard {jac:.3f} should be well below 0.5; "
        "the gate is vacuous"
    )


def test_fastopic_matches_committed_gold():
    """Refit topica FASTopic and compare to the frozen fastopic-reference gold."""
    r = fastopic_gold.run(verbose=False)
    assert r["passes"], (
        f"topica FASTopic topic-word cosine {r['cosine']:.4f} below bar "
        f"{r['bar']:.4f} (reference self {r['reference_self_cosine']:.4f}); details: {r}"
    )
