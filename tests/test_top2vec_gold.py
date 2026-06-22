"""Offline gold-fixture parity for topica Top2Vec vs BERTopic (#271, Wave 1).

Loads the committed gold (``parity/top2vec_gold.npz`` + ``.json``), fits topica
Top2Vec on the SAME synthetic planted-cluster embeddings BERTopic saw, and asserts
topica recovers the planted truth at least as well as the frozen BERTopic run
(within a margin), agrees with BERTopic's frozen partition, and keeps its top-word
block purity.

HONEST NOTE ON THE BAR. This is a *clustering model*, so the gate is a
clustering-agreement metric (adjusted Rand index of the partition vs the planted
truth, cross-ARI between the two implementations, and per-topic block purity) — NOT
a topic-word cosine like the FASTopic / CombinedTM golds. The bars are verbatim from
the live ``tests/test_top2vec_parity.py``. On the clean planted design both
implementations recover the truth perfectly (ARI 1.0).

This runs in CI WITHOUT bertopic / umap / hdbscan: BERTopic's recovered labels and
the exact synthetic embeddings are frozen in the committed gold, so none of those are
imported at test time (asserted explicitly below). The topica refit (320 docs, 4
clusters) is ~1.5s, well inside the default suite.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import top2vec_gold  # noqa: E402


def test_top2vec_gold_present():
    npz, js = harness.gold_paths("top2vec")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/top2vec_gold.py --regenerate` "
        "(needs bertopic + umap + hdbscan)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_top2vec_gold_shape():
    """Fast default check: the frozen arrays round-trip to the documented sizes."""
    arrays, meta = harness.load_gold("top2vec")
    assert arrays["doc_embeddings"].shape == (meta["num_docs"], meta["emb_dim"])
    assert arrays["bertopic_labels"].shape == (meta["num_docs"],)
    assert arrays["truth"].shape == (meta["num_docs"],)
    assert arrays["word_embeddings"].shape == (meta["vocab_size"], meta["emb_dim"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]


def test_top2vec_no_heavy_import_at_test_time():
    """The committed gold must validate with NO clustering-stack framework present.

    Proven in a fresh subprocess that hard-blocks bertopic / umap / hdbscan via an
    import finder, then runs the full offline ``top2vec_gold.run()``. A subprocess
    (not a bare ``sys.modules`` check) is used because the sibling live parity test
    ``test_top2vec_parity.py`` imports bertopic in the same pytest session; the point
    is that the GOLD PATH itself needs none of it."""
    script = textwrap.dedent(
        f"""
        import sys
        BLOCKED = {{"bertopic", "umap", "hdbscan", "torch"}}
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked {{name}} for offline gold test")
                return None
        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, {str(PARITY)!r})
        import top2vec_gold
        r = top2vec_gold.run(verbose=False)
        assert r["passes"], r
        assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert "OK" in proc.stdout, f"offline gold path imported a blocked module:\n{proc.stderr}"


def test_top2vec_gold_is_non_vacuous():
    """A randomly shuffled partition must collapse the truth-recovery ARI below the
    bar — proving the clustering gate discriminates a real partition from noise. Runs
    by default (no fit).

    For a clustering model the discriminator is ARI of the partition, not top-word
    Jaccard: a permuted label vector keeps the same cluster sizes but destroys the
    doc->cluster correspondence, so its ARI against the truth must drop near zero."""
    import numpy as np

    arrays, meta = harness.load_gold("top2vec")
    truth = arrays["truth"].astype(np.int64)
    bt_labels = arrays["bertopic_labels"].astype(np.int64)

    # The frozen BERTopic labels recover the truth (sanity on the gold itself).
    # Uses the harness's numpy-only ARI so the test needs no scikit-learn (CI).
    assert harness.adjusted_rand_index(truth, bt_labels) >= meta["truth_ari_min"]

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(bt_labels)
    jumbled_ari = harness.adjusted_rand_index(truth, shuffled)
    bar = max(meta["truth_ari_min"], meta["bertopic_truth_ari"] - meta["ari_margin"])
    assert jumbled_ari < bar, (
        f"shuffled-partition ARI {jumbled_ari:.3f} should be far below the bar "
        f"{bar:.3f}; the gate is vacuous"
    )


def test_top2vec_matches_committed_gold():
    """Refit topica Top2Vec and compare to the frozen BERTopic-reference gold."""
    r = top2vec_gold.run(verbose=False)
    assert r["passes"], (
        f"topica Top2Vec failed the clustering bars: truth ARI {r['topica_truth_ari']:.4f} "
        f"(bar {r['ari_bar']:.4f}), cross ARI {r['cross_ari']:.4f}, block purity "
        f"{r['topica_block_purity']:.4f}, topics {r['topica_num_topics']}; details: {r}"
    )
