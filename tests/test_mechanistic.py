"""Mechanistic Topic Model (mLDA) over sparse-autoencoder features (#575).

Two things are checked:

1. ``topica.from_feature_matrix`` turns a document x feature count matrix into a
   Corpus (dense and SciPy-sparse paths agree, and the validation is strict).
2. ``MechanisticLDA`` is a faithful, feature-aware wrapper over the validated
   SparseLDA sampler: fitting it on a feature corpus is *bit-identical* to fitting
   plain ``LDA`` on the equivalent bag-of-words corpus, so the port introduces no
   new inference. The feature-named surface and save/load round-trip are covered.
"""

import numpy as np
import pytest

import topica
from topica import from_feature_matrix
from topica._topica import Corpus, LDA


def _counts():
    # 6 docs x 4 features; a clear two-block structure so topics are recoverable.
    # low=1 so every row has at least one token (no empty documents): the
    # fixed-vocabulary token corpus in the parity test drops empty docs, whereas
    # from_feature_matrix keeps them, so an all-zero row would desync the two.
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, size=(3, 2))
    b = rng.integers(1, 6, size=(3, 2))
    top = np.hstack([a, np.zeros((3, 2), dtype=int)])
    bot = np.hstack([np.zeros((3, 2), dtype=int), b])
    return np.vstack([top, bot]).astype(np.int64)


FEATURES = ["labor", "markets", "climate", "protest"]


# --------------------------------------------------------------------------
# from_feature_matrix
# --------------------------------------------------------------------------

def test_from_feature_matrix_basic():
    counts = _counts()
    c = from_feature_matrix(counts, FEATURES)
    assert c.num_docs == 6
    assert c.num_words == 4
    assert c.vocabulary == FEATURES
    # token counts equal column sums; doc lengths equal row sums.
    assert c.word_counts == [int(x) for x in counts.sum(0)]
    assert c.doc_lengths == [int(x) for x in counts.sum(1)]


def test_from_feature_matrix_default_names():
    c = from_feature_matrix(np.array([[1, 0], [0, 2]]))
    assert c.vocabulary == ["feature_0", "feature_1"]


def test_dense_and_sparse_agree():
    sp = pytest.importorskip("scipy.sparse")
    counts = _counts()
    dense = from_feature_matrix(counts, FEATURES)
    sparse = from_feature_matrix(sp.csr_matrix(counts), FEATURES)
    assert dense.documents() == sparse.documents()
    assert dense.word_counts == sparse.word_counts


def test_doc_ids_and_metadata():
    counts = _counts()
    ids = [f"d{i}" for i in range(6)]
    meta = list(range(6))
    c = from_feature_matrix(counts, FEATURES, doc_ids=ids, metadata=meta)
    assert c.doc_names == ids
    assert c.metadata == meta


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1, -1]]),          # negative
        np.array([[1.5, 0.0]]),       # non-integer
        np.array([1, 2, 3]),          # not 2-D
        np.array([[np.inf, 0.0]]),    # not finite
        np.array([[1e30, 0.0]]),      # exceeds u32 (and would wrap on int cast)
        [[1, None]],                  # non-numeric -> ValueError, not TypeError
    ],
)
def test_from_feature_matrix_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        from_feature_matrix(bad)


def test_feature_names_length_mismatch():
    with pytest.raises(ValueError):
        from_feature_matrix(np.array([[1, 2, 3]]), ["only", "two"])


def test_metadata_row_mismatch():
    with pytest.raises(ValueError):
        from_feature_matrix(_counts(), FEATURES, metadata=[1, 2, 3])


# --------------------------------------------------------------------------
# Parity: mLDA == LDA on the equivalent bag-of-words corpus
# --------------------------------------------------------------------------

def _token_corpus(counts, features):
    """The bag-of-words corpus equivalent to ``counts``: each feature repeated
    ``count`` times, in ascending column order (matching from_counts' expansion),
    against a fixed vocabulary."""
    docs = []
    for row in counts:
        toks = []
        for col in range(len(features)):
            toks.extend([features[col]] * int(row[col]))
        docs.append(toks)
    return Corpus.from_documents(docs, vocabulary=features)


def test_feature_corpus_matches_token_corpus():
    counts = _counts()
    fc = from_feature_matrix(counts, FEATURES)
    tc = _token_corpus(counts, FEATURES)
    assert fc.documents() == tc.documents()


def test_mlda_bit_identical_to_lda_on_tokens():
    counts = _counts()
    fc = from_feature_matrix(counts, FEATURES)
    tc = _token_corpus(counts, FEATURES)

    topica.enable_experimental()
    m = topica.MechanisticLDA(2, seed=13).fit(fc, iters=200, keep_theta_draws=False)

    ref = LDA(2, seed=13)
    ref.fit(tc, iters=200, keep_theta_draws=False)

    # Same sampler, same token stream, same seed -> byte-identical fit.
    np.testing.assert_array_equal(m.topic_feature, ref.topic_word)
    np.testing.assert_array_equal(m.doc_topic, ref.doc_topic)


# --------------------------------------------------------------------------
# MechanisticLDA surface
# --------------------------------------------------------------------------

def test_experimental_gate():
    topica.enable_experimental(False)
    try:
        with pytest.raises(RuntimeError):
            topica.MechanisticLDA(2)
    finally:
        topica.enable_experimental(True)


def test_feature_named_surface_and_aliases():
    topica.enable_experimental()
    m = topica.MechanisticLDA(2, seed=13).fit(from_feature_matrix(_counts(), FEATURES), iters=100)
    # feature-named names and word aliases are the same underlying arrays.
    np.testing.assert_array_equal(m.topic_feature, m.topic_word)
    assert m.feature_names == m.vocabulary == FEATURES
    assert m.top_features(3) == m.top_words(3)
    # top features are (name, weight) pairs drawn from the feature vocabulary.
    for name, weight in m.top_features(4)[0]:
        assert name in FEATURES
        assert 0.0 <= weight <= 1.0
    # delegated LDA surface is reachable through the wrapper.
    assert len(m.coherence(5)) == 2
    assert m.num_topics == 2


def test_fit_accepts_raw_matrix_with_feature_names():
    topica.enable_experimental()
    m = topica.MechanisticLDA(2, seed=13).fit(_counts(), feature_names=FEATURES, iters=50)
    assert m.vocabulary == FEATURES
    # passing feature_names alongside an already-built corpus is an error.
    with pytest.raises(ValueError):
        topica.MechanisticLDA(2).fit(from_feature_matrix(_counts(), FEATURES), feature_names=FEATURES)


def test_save_load_roundtrip(tmp_path):
    topica.enable_experimental()
    m = topica.MechanisticLDA(2, seed=13).fit(from_feature_matrix(_counts(), FEATURES), iters=100)
    path = str(tmp_path / "mlda.bin")
    m.save(path)
    loaded = topica.MechanisticLDA.load(path)
    np.testing.assert_array_equal(loaded.topic_feature, m.topic_feature)
    assert loaded.vocabulary == m.vocabulary
    assert loaded.settings == m.settings


def test_save_load_is_relocatable(tmp_path):
    topica.enable_experimental()
    m = topica.MechanisticLDA(2, seed=13).fit(from_feature_matrix(_counts(), FEATURES), iters=100)
    src = tmp_path / "a"
    src.mkdir()
    m.save(str(src / "mlda.bin"))
    # Move the pair (wrapper + inner LDA) to a new directory and load from there.
    dst = tmp_path / "b"
    dst.mkdir()
    for f in src.iterdir():
        f.rename(dst / f.name)
    loaded = topica.MechanisticLDA.load(str(dst / "mlda.bin"))
    np.testing.assert_array_equal(loaded.topic_feature, m.topic_feature)
