"""Tests for AuthorRecipientTopic (ART; McCallum, Wang & Corrada-Emmanuel, JAIR 2007).

ART is realized as a faithful wrapper over AuthorTopic with the "author" entity being
the ordered (sender, recipient) pair. The central validation (per the Gate-A review)
is the *isomorphism identity*: on a multi-recipient corpus, ART must equal an
AuthorTopic fit on the hand-built pair labels, bit-for-bit. Plus planted recovery
with unequal recipient-set sizes, the LDA reduction, determinism, and edge cases.
"""

import numpy as np
import pytest

import topica


def _planted_pairs():
    """Three disjoint word blocks. Three senders, each writing one block to its own
    recipient; plus some multi-recipient messages (|r_d| = 2) so the isomorphism test
    exercises the non-degenerate path."""
    blocks = [["a", "b", "c", "d"], ["m", "n", "o", "p"], ["x", "y", "z", "q"]]
    docs, authors, recipients = [], [], []
    for si, blk in enumerate(blocks):
        for _ in range(30):
            docs.append(list(blk) * 3)
            authors.append(f"s{si}")
            recipients.append([f"r{si}"])
    # multi-recipient messages: sender s0 writes block 0 to two recipients
    for _ in range(10):
        docs.append(["a", "b", "c", "d"])
        authors.append("s0")
        recipients.append(["r0", "r1"])
    return docs, authors, recipients


def test_isomorphism_with_authortopic():
    """ART equals AuthorTopic on the equivalent pairs. Independent check: the reference
    AuthorTopic authors are built with a DIFFERENT labeling scheme (``a//r``) that
    induces the *same* token->pair partition in the same within-doc author order, so
    ``topic_word`` / ``doc_topic`` — which depend only on the partition, not the labels
    — match bit-for-bit even though the wrapper's internal ids ("p0", ...) differ.
    (Equality relies on both schemes yielding the same within-doc author ordering,
    which holds here: the fixed sender keeps (s0,r0) before (s0,r1) under both.) This
    would fail if ART mis-partitioned tokens; it does not re-run ART's own code."""
    docs, authors, recipients = _planted_pairs()
    art = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=200)

    at_authors = [[f"{a}//{r}" for r in sorted(set(rset), key=repr)]
                  for a, rset in zip(authors, recipients)]
    # match ART's default hypers (alpha 50/K inherited; beta 0.1, the paper value)
    at = topica.AuthorTopic(3, beta=0.1, seed=0).fit(docs, at_authors, iters=200)

    assert np.array_equal(art.topic_word, at.topic_word)
    assert np.array_equal(art.doc_topic, at.doc_topic)
    # pair_topic aligned by label equals author_topic aligned by its labels
    at_row = {lab: i for i, lab in enumerate(at.authors)}
    for row, (s, r) in enumerate(art.pair_labels):
        assert np.allclose(art.pair_topic[row], at.author_topic[at_row[f"{s}//{r}"]])


def test_shapes_and_normalization():
    docs, authors, recipients = _planted_pairs()
    m = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=150)
    V = len(m.vocabulary)
    P = len(m.pair_labels)
    assert m.topic_word.shape == (3, V)
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.pair_topic.shape == (P, 3)
    assert np.allclose(m.topic_word.sum(1), 1.0)
    assert np.allclose(m.doc_topic.sum(1), 1.0)
    assert np.allclose(m.pair_topic.sum(1), 1.0)
    # directed pairs: (s0,r0), (s0,r1), (s1,r1), (s2,r2) all present and ordered
    assert ("s0", "r0") in m.pair_labels and ("s0", "r1") in m.pair_labels


def test_planted_recovery_unequal_recipient_sizes():
    docs, authors, recipients = _planted_pairs()
    m = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=300)
    # each topic concentrates on its 4-word block
    top4 = np.sort(m.topic_word, axis=1)[:, -4:].sum(axis=1)
    assert np.all(top4 > 0.9), f"topics not block-concentrated: {top4}"
    # the three solo pairs land on three distinct dominant topics
    labels = m.pair_labels
    solo = [labels.index((f"s{i}", f"r{i}")) for i in range(3)]
    dom = m.pair_topic[solo].argmax(axis=1)
    assert len(set(dom.tolist())) == 3, f"pairs not on distinct topics: {dom}"


def test_lda_reduction_unique_pair_per_doc():
    """A *unique* pair per document reduces ART to LDA (as AuthorTopic does with a
    unique author per doc) — not a single pair everywhere (that is a unigram mixture)."""
    docs, _, _ = _planted_pairs()
    authors = [f"s{i}" for i in range(len(docs))]     # unique sender per document
    recipients = [["r"]] * len(docs)                  # -> a unique (sender, recipient) pair per doc
    art = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=250)
    assert len(art.pair_labels) == len(docs)
    lda = topica.LDA(3, seed=0).fit(docs, iters=250)
    # both recover the planted blocks; each topic concentrates on its 4-word block
    for m in (art, lda):
        top4 = np.sort(m.topic_word, axis=1)[:, -4:].sum(axis=1)
        assert np.all(top4 > 0.9), f"did not recover blocks: {top4}"


def test_pair_counts_are_messages():
    docs, authors, recipients = _planted_pairs()
    m = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=50)
    pc = dict(zip(m.pair_labels, np.asarray(m.pair_counts).tolist()))
    # (s0,r0) is on the 30 solo messages PLUS the 10 multi-recipient [r0,r1] messages
    # = 40; (s0,r1) only on those 10; the other solo pairs on 30 each.
    assert pc[("s0", "r0")] == 40
    assert pc[("s0", "r1")] == 10
    assert pc[("s1", "r1")] == 30
    assert pc[("s2", "r2")] == 30


def test_label_types_preserved():
    """Integer sender/recipient labels round-trip as ints in pair_labels, not strings."""
    docs, _, _ = _planted_pairs()
    authors = [d % 3 for d in range(len(docs))]           # int senders
    recipients = [[10 + (d % 3)] for d in range(len(docs))]  # int recipients
    m = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=20)
    for s, r in m.pair_labels:
        assert isinstance(s, int) and isinstance(r, int), f"label types lost: {(s, r)}"
    assert (0, 10) in m.pair_labels


def test_no_collision_across_label_types():
    """(1, "2") and ("1", 2) are distinct pairs and must not be merged (no stringify collision)."""
    docs = [["a", "b", "c"]] * 4
    authors = [1, "1", 1, "1"]
    recipients = [["2"], [2], ["2"], [2]]
    m = topica.ART(2, seed=0).fit(docs, authors=authors, recipients=recipients, iters=10)
    labels = set(m.pair_labels)
    assert (1, "2") in labels and ("1", 2) in labels
    assert len(m.pair_labels) == 2


def test_determinism_with_set_recipients():
    """A set (unordered) recipient container still yields bit-for-bit identical fits."""
    docs, authors, _ = _planted_pairs()
    recips = [{"r0", "r1", "r2"} for _ in docs]
    a = topica.ART(3, seed=7).fit(docs, authors=authors, recipients=recips, iters=80)
    b = topica.ART(3, seed=7).fit(docs, authors=authors, recipients=recips, iters=80)
    assert np.array_equal(a.pair_topic, b.pair_topic)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_determinism():
    docs, authors, recipients = _planted_pairs()
    a = topica.ART(3, seed=7).fit(docs, authors=authors, recipients=recipients, iters=120)
    b = topica.ART(3, seed=7).fit(docs, authors=authors, recipients=recipients, iters=120)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.pair_topic, b.pair_topic)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_empty_recipient_raises():
    docs, authors, recipients = _planted_pairs()
    recipients[5] = []
    with pytest.raises(ValueError, match="empty recipient"):
        topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=5)


def test_bare_string_recipient_raises():
    """A bare string recipient set (instead of a list) is a footgun -> raise, not iterate chars."""
    docs, authors, recipients = _planted_pairs()
    recipients[3] = "alice"   # should have been ["alice"]
    with pytest.raises(ValueError, match="bare string"):
        topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=5)


def test_length_mismatch_raises():
    docs, authors, recipients = _planted_pairs()
    with pytest.raises(ValueError):
        topica.ART(3, seed=0).fit(docs, authors=authors[:-1], recipients=recipients, iters=5)


def test_alpha_default_is_paper_50_over_k():
    """The wrapped AuthorTopic engine defaults alpha = 50/K (the ART paper's prior)."""
    docs, authors, recipients = _planted_pairs()
    m = topica.ART(5, seed=0).fit(docs, authors=authors, recipients=recipients, iters=20)
    assert m._at.settings["alpha"] == pytest.approx(50.0 / 5)


def test_save_load_roundtrip(tmp_path):
    docs, authors, recipients = _planted_pairs()
    m = topica.ART(3, seed=0).fit(docs, authors=authors, recipients=recipients, iters=80)
    p = tmp_path / "art.pkl"
    m.save(str(p))
    m2 = topica.AuthorRecipientTopic.load(str(p))
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.pair_topic, m2.pair_topic)
    assert m.pair_labels == m2.pair_labels
