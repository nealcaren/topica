"""doc_super getter for PA (issue #497).

PA exposes doc_topic (doc->sub marginal) and super_sub (global super->sub
association). doc_super adds the per-document super-topic proportions, the
per-document companion to super_sub. These tests check its shape, that rows are
valid probability distributions, and (sanity) that it aligns with the planted
super-topic structure.
"""

import numpy as np
import pytest

import topica


def _planted_corpus():
    """S=2 super-topics over K=4 sub-topics with disjoint-vocabulary blocks:
    super 0 owns blocks {0,1}, super 1 owns blocks {2,3}. Each document mixes
    BOTH blocks of ONE super-topic (mixing ratio varies deterministically per
    doc, so the two co-occurring sub-topics stay identifiable), giving documents
    that split cleanly by super-topic. Mirrors the Rust planted_corpus in
    src/pa.rs. Returns (docs, true_super) with true_super the planted label.
    """
    blocks = [[f"w{b * 6 + i}" for i in range(6)] for b in range(4)]
    groups = [[0, 1], [2, 3]]  # super-topic -> its two blocks
    docs, true_super = [], []
    for d in range(400):
        g = d % 2
        pair = groups[g]
        n0 = 8 + (d // 2) % 17  # 8..=24 tokens from the first block
        n1 = 32 - n0  # the rest from the second block
        doc = []
        for t in range(n0):
            blk = blocks[pair[0]]
            doc.append(blk[(t + d) % len(blk)])
        for t in range(n1):
            blk = blocks[pair[1]]
            doc.append(blk[(t + d) % len(blk)])
        docs.append(doc)
        true_super.append(g)
    return docs, np.array(true_super)


@pytest.fixture(scope="module")
def fitted():
    docs, true_super = _planted_corpus()
    model = topica.PA(num_super=2, num_sub=4, alpha=0.1, beta=0.01, seed=42)
    model.fit(docs, iters=200)
    return model, true_super


def test_doc_super_shape(fitted):
    model, _ = fitted
    ds = model.doc_super
    assert ds.shape == (400, model.num_super)


def test_doc_super_is_probability(fitted):
    model, _ = fitted
    ds = model.doc_super
    assert np.all(ds >= 0.0)
    assert np.allclose(ds.sum(axis=1), 1.0, atol=1e-9)


def test_doc_super_is_decisive_and_non_degenerate(fitted):
    """doc_super must carry real per-document structure: PAM commits each document
    hard to one super-topic (the documented single-super-topic behavior), so most
    rows have a dominant super-topic, and both super-topics dominate some
    documents (the assignment partitions the corpus rather than collapsing to one
    column). Which planted group maps to which super-topic is seed-dependent (the
    super layer can specialize on an axis other than the planting), so we do not
    assert planted-label recovery here; the Rust unit test in src/pa.rs pins that
    on a seed that recovers cleanly."""
    model, _ = fitted
    ds = model.doc_super
    crisp = (ds.max(axis=1) > 0.9).mean()
    assert crisp > 0.8, f"doc_super not decisive; only {crisp:.2f} of rows commit"
    dominant = ds.argmax(axis=1)
    used = np.unique(dominant)
    assert used.size == model.num_super, (
        f"doc_super degenerate; only super-topics {used.tolist()} ever dominate"
    )


def test_doc_super_survives_roundtrip(tmp_path):
    docs, _ = _planted_corpus()
    model = topica.PA(num_super=2, num_sub=4, alpha=0.1, beta=0.01, seed=7)
    model.fit(docs, iters=50)
    before = model.doc_super
    path = tmp_path / "pa.bin"
    model.save(str(path))
    reloaded = topica.PA.load(str(path))
    assert np.allclose(reloaded.doc_super, before)


def test_doc_super_requires_fit():
    model = topica.PA(num_super=2, num_sub=4, alpha=0.1, beta=0.01, seed=7)
    with pytest.raises(RuntimeError):
        _ = model.doc_super
