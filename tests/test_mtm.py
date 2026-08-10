"""Mechanistic Topic Models (#575): featurization and the two model wrappers.

Every assertion here is anchored to a specific line of the paper (Zheng et al.
2025, arXiv:2507.23220) or of the reference implementation
(github.com/blei-lab/mechanistic-topic-models, MIT), because the whole value of
this module is that it reproduces *their* recipe rather than a plausible one.

What this file does NOT claim: numeric parity with the reference. That needs
their Gemma-2 activations and SAE. The planted-structure tests below are smoke
tests on a corpus built to be trivially separable, and each one is paired with
the null it has to beat.
"""

import numpy as np
import pytest

import topica
from topica import mtm


@pytest.fixture(scope="module", autouse=True)
def _experimental():
    # Restore whatever the process was in, rather than forcing the gate back on.
    # `enable_experimental` is process-global and other modules turn it on at
    # import time, so a hard reset here breaks them depending on file order.
    previous = topica.experimental_enabled()
    topica.enable_experimental(True)
    yield
    topica.enable_experimental(previous)


def _planted(num_docs=60, num_tokens=40, num_features=30, num_blocks=3, seed=13):
    """A corpus where document ``d`` fires only feature block ``d % num_blocks``."""
    rng = np.random.default_rng(seed)
    width = num_features // num_blocks
    docs, labels = [], []
    for d in range(num_docs):
        b = d % num_blocks
        labels.append(b)
        a = rng.random((num_tokens, num_features)) * 0.5
        a[:, b * width:(b + 1) * width] += 0.8
        docs.append(a)
    return docs, labels


# ---------------------------------------------------------------------------
# §4.1 featurization (eq. 5)
# ---------------------------------------------------------------------------

class TestFeaturize:
    def test_matches_the_reference_expression(self):
        # sae_activations_sum_discrete.py:126-131, transcribed:
        #   sae_activations = sae_activations[:, 1:, :]
        #   torch.where(sae_activations > feature_quantiles, 1, 0).sum(dim=1)
        rng = np.random.default_rng(7)
        acts = rng.random((5, 12, 8))
        thr = rng.random(8)
        expected = np.where(acts[:, 1:, :] > thr, 1, 0).sum(axis=1)
        assert (mtm.featurize(acts, thr).counts == expected).all()

    def test_comparison_is_strictly_greater(self):
        # An SAE emits many exact zeros; `>=` with a low threshold would count
        # every silent token as a firing.
        acts = np.array([[[0.0, 5.0], [0.0, 5.0]]])   # 2 tokens, 2 features
        counts = mtm.featurize(acts, [0.0, 0.0], drop_first_token=False).counts
        assert counts.tolist() == [[0, 2]]

    def test_first_token_is_dropped_and_shortens_n_tokens(self):
        # The SAE was not trained on the BOS position; the reference drops it
        # *before* taking the length (sae_activations_mean.py:118-120).
        acts = np.zeros((1, 4, 2))
        acts[0, 0, :] = 99.0                       # only the first token fires
        fc = mtm.featurize(acts, [1.0, 1.0])
        assert fc.n_tokens.tolist() == [3]
        assert fc.counts.tolist() == [[0, 0]]
        kept = mtm.featurize(acts, [1.0, 1.0], drop_first_token=False)
        assert kept.n_tokens.tolist() == [4]
        assert kept.counts.tolist() == [[1, 1]]

    def test_n_tokens_is_not_the_row_sum(self):
        # The load-bearing distinction: eq. 8 uses N_sae (the row sum), eqs. 12
        # and 14 use N_tok. A token may fire many features or none.
        docs, _ = _planted()
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        assert (fc.n_sae != fc.n_tokens).any()
        assert (fc.n_sae == fc.counts.sum(axis=1)).all()

    def test_ragged_documents(self):
        rng = np.random.default_rng(3)
        docs = [rng.random((n, 4)) for n in (9, 2, 15)]
        fc = mtm.featurize(docs, 0.5)
        assert fc.n_tokens.tolist() == [8, 1, 14]
        assert fc.counts.shape == (3, 4)

    def test_single_token_document_survives_as_empty(self):
        fc = mtm.featurize([np.ones((1, 4))], 0.5)
        assert fc.n_tokens.tolist() == [0] and fc.counts.sum() == 0

    def test_scalar_threshold_broadcasts(self):
        rng = np.random.default_rng(5)
        acts = rng.random((3, 6, 4))
        assert (mtm.featurize(acts, 0.5).counts
                == mtm.featurize(acts, [0.5] * 4).counts).all()

    @pytest.mark.parametrize("thresholds, match", [
        ([0.1, 0.2], "thresholds has 2 entries"),
        (np.zeros((2, 2)), "must be a scalar or 1-D"),
    ])
    def test_threshold_shape_errors(self, thresholds, match):
        with pytest.raises(ValueError, match=match):
            mtm.featurize(np.zeros((2, 3, 4)), thresholds)

    def test_ragged_feature_width_is_an_error(self):
        with pytest.raises(ValueError, match="same number of features"):
            mtm.featurize([np.zeros((3, 4)), np.zeros((3, 5))], 0.5)


class TestFeatureThresholds:
    def test_is_the_pooled_per_feature_quantile(self):
        rng = np.random.default_rng(11)
        docs = [rng.random((10, 5)), rng.random((7, 5))]
        pooled = np.concatenate([docs[0][1:], docs[1][1:]], axis=0)
        assert np.allclose(mtm.feature_thresholds(docs, q=0.8),
                           np.quantile(pooled, 0.8, axis=0))

    def test_roughly_a_1_minus_q_firing_rate(self):
        # The property that makes a within-corpus quantile a usable fallback.
        rng = np.random.default_rng(2)
        docs = [rng.random((200, 6)) for _ in range(10)]
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        assert 0.18 < fc.counts.sum() / (fc.n_tokens.sum() * 6) < 0.22

    def test_q_out_of_range(self):
        with pytest.raises(ValueError, match=r"q must be in \[0.0, 1.0\]"):
            mtm.feature_thresholds(np.zeros((2, 3, 4)), q=1.5)


# ---------------------------------------------------------------------------
# §4.2.3 document embeddings (eq. 14)
# ---------------------------------------------------------------------------

class TestDocumentEmbeddings:
    def test_is_the_decoder_weighted_sum_over_tokens(self):
        rng = np.random.default_rng(17)
        counts = rng.integers(0, 5, size=(6, 9))
        n_tokens = rng.integers(1, 30, size=6)
        directions = rng.normal(size=(9, 4))
        emb = mtm.document_embeddings(counts, directions, n_tokens=n_tokens)
        for d in range(6):
            assert np.allclose(emb[d], (counts[d] @ directions) / n_tokens[d])

    def test_matches_the_reference_decode(self):
        # SubsetSAE.decode_without_bias(z) is `z @ W_dec` (subset_sae.py:145-148),
        # i.e. the reference always embeds in the model's residual-stream space.
        rng = np.random.default_rng(19)
        counts, W = rng.integers(0, 4, size=(5, 7)), rng.normal(size=(7, 3))
        assert np.allclose(mtm.document_embeddings(counts, W), counts @ W)

    def test_one_over_n_tok_is_a_no_op_under_cosine(self):
        # Why the reference can omit it and still match the paper: UMAP's metric
        # is cosine on both sides, and 1/N_tok is a positive per-document scalar.
        rng = np.random.default_rng(23)
        counts = rng.integers(0, 6, size=(8, 10))
        n_tokens = rng.integers(5, 50, size=8)
        W = rng.normal(size=(10, 5))
        a = mtm.document_embeddings(counts, W, n_tokens=n_tokens)
        b = mtm.document_embeddings(counts, W)

        def cosines(e):
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            return e @ e.T

        assert np.allclose(cosines(a), cosines(b))

    def test_zero_token_document_is_a_zero_row_not_a_nan(self):
        emb = mtm.document_embeddings(np.zeros((2, 3)), np.ones((3, 4)),
                                      n_tokens=[0, 5])
        assert np.isfinite(emb).all() and not emb[0].any()

    def test_normalize(self):
        rng = np.random.default_rng(29)
        emb = mtm.document_embeddings(rng.integers(1, 5, (4, 6)),
                                      rng.normal(size=(6, 3)), normalize=True)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)

    def test_feature_width_mismatch_names_the_fix(self):
        with pytest.raises(ValueError, match="kept_features"):
            mtm.document_embeddings(np.zeros((2, 5)), np.zeros((4, 3)))

    def test_n_tokens_shape_error(self):
        with pytest.raises(ValueError, match="n_tokens has shape"):
            mtm.document_embeddings(np.zeros((2, 3)), np.zeros((3, 4)), n_tokens=[1])


# ---------------------------------------------------------------------------
# §4.2.1 mLDA
# ---------------------------------------------------------------------------

class TestMechanisticLDA:
    def test_reference_priors_are_the_defaults(self):
        # mallet_lda_bof_model.py:61-66 -- alpha is a SUM over topics, so K=5
        # gives 1.0 per topic, and MALLET's beta default is 0.01.
        s = mtm.MechanisticLDA(5).settings
        assert (s["alpha_sum"], s["beta"], s["optimize_interval"]) == (5.0, 0.01, 10)
        assert s["max_doc_fraction"] == 0.9        # App. A.1's "crucial" filter
        docs, _ = _planted()
        m = mtm.MechanisticLDA(5).fit(mtm.featurize(docs, 0.9), iters=50)
        assert np.allclose(m.alpha, 1.0) and m.beta == 0.01

    def test_recovers_planted_feature_blocks(self):
        # Smoke test on a trivially separable corpus, with its null below.
        docs, _ = _planted()
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        m = mtm.MechanisticLDA(3, seed=13).fit(
            fc, feature_names=[str(i) for i in range(fc.num_features)], iters=300)
        blocks = {frozenset(range(0, 10)), frozenset(range(10, 20)),
                  frozenset(range(20, 30))}
        found = {frozenset(int(w) for w, _ in m.top_words(5, topic=k)) for k in range(3)}
        assert all(any(f <= b for b in blocks) for f in found)
        assert len({min(f) // 10 for f in found}) == 3   # one topic per block

    def test_the_null_does_not_recover_blocks(self):
        # Same shapes, no planted structure: the test above must be measuring the
        # signal and not the arithmetic.
        rng = np.random.default_rng(13)
        docs = [rng.random((40, 30)) for _ in range(60)]
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        # max_doc_fraction=1.0: with no planted structure every feature fires in
        # every document, so App. A.1's filter would (correctly) prune the whole
        # vocabulary. The null under test here is topic recovery, not the filter.
        m = mtm.MechanisticLDA(3, seed=13, max_doc_fraction=1.0).fit(
            fc, feature_names=[str(i) for i in range(30)], iters=300)
        found = {frozenset(int(w) for w, _ in m.top_words(5, topic=k)) for k in range(3)}
        assert not all(len({int(w) // 10 for w in f}) == 1 for f in found)

    def test_column_order_is_the_callers(self):
        # The from_matrix contract, end to end: topic_word[:, j] is feature j.
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        names = [f"sae{i}" for i in range(fc.num_features)]
        m = mtm.MechanisticLDA(3).fit(fc, feature_names=names, iters=30)
        assert m.vocabulary == names
        assert m.topic_word.shape == (3, fc.num_features)

    def test_max_doc_fraction_prunes_ubiquitous_features(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        counts = fc.counts.copy()
        counts[:, 0] = 7                       # a feature that fires everywhere
        wide = mtm.MechanisticLDA(3, max_doc_fraction=1.0).fit(
            mtm.FeatureCounts(counts, fc.n_tokens), iters=30)
        narrow = mtm.MechanisticLDA(3, max_doc_fraction=0.9).fit(
            mtm.FeatureCounts(counts, fc.n_tokens), iters=30)
        assert 0 in wide.corpus.kept_features
        assert 0 not in narrow.corpus.kept_features

    def test_doc_lengths_are_n_sae_and_n_tokens_survives_on_the_corpus(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        m = mtm.MechanisticLDA(3).fit(fc, iters=30)
        assert m.doc_lengths == fc.n_sae.tolist()
        assert m.corpus.n_tokens == fc.n_tokens.tolist()

    def test_accepts_a_bare_matrix_and_a_corpus(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        a = mtm.MechanisticLDA(3, seed=13).fit(fc.counts, iters=50)
        c = topica.Corpus.from_matrix(fc.counts, max_doc_fraction=0.9)
        b = mtm.MechanisticLDA(3, seed=13).fit(c, iters=50)
        assert np.allclose(a.topic_word, b.topic_word)

    def test_deterministic_from_a_seed(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        a = mtm.MechanisticLDA(3, seed=13).fit(fc, iters=100)
        b = mtm.MechanisticLDA(3, seed=13).fit(fc, iters=100)
        assert np.array_equal(a.topic_word, b.topic_word)
        assert np.array_equal(a.doc_topic, b.doc_topic)

    def test_transform_and_save_round_trip(self, tmp_path):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        m = mtm.MechanisticLDA(3, max_doc_fraction=1.0).fit(fc, iters=50)
        theta = m.transform(fc.counts[:4])
        assert theta.shape == (4, 3)
        assert np.allclose(theta.sum(axis=1), 1.0)
        path = str(tmp_path / "m.bin")
        m.save(path)
        assert np.allclose(mtm.MechanisticLDA.load(path).topic_word, m.topic_word)

    def test_unfitted_access_raises(self):
        m = mtm.MechanisticLDA(3)
        for attr in ("topic_word", "doc_topic", "vocabulary", "corpus"):
            with pytest.raises(RuntimeError, match="not fitted"):
                getattr(m, attr)


# ---------------------------------------------------------------------------
# §4.2.3 mBERTopic
# ---------------------------------------------------------------------------

class TestMechanisticBERTopic:
    def test_reference_min_topic_size_is_the_default(self):
        # mbertopic.py:54 min_topic_size=10; topica's own BERTopic default is 15.
        assert mtm.MechanisticBERTopic().settings["min_cluster_size"] == 10

    def test_recovers_planted_clusters(self):
        docs, labels = _planted()
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        W = np.random.default_rng(31).normal(size=(fc.num_features, 16))
        m = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13).fit(fc, directions=W)
        assert topica.agreement(m.labels, labels)["ari"] > 0.9

    def test_the_null_does_not_recover_clusters(self):
        rng = np.random.default_rng(13)
        docs = [rng.random((40, 30)) for _ in range(60)]
        labels = [d % 3 for d in range(60)]      # labels unrelated to the data
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        W = rng.normal(size=(30, 16))
        m = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13,
                                    max_doc_fraction=1.0).fit(fc, directions=W)
        assert topica.agreement(m.labels, labels)["ari"] < 0.2

    def test_directions_and_doc_embeddings_agree(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        W = np.random.default_rng(37).normal(size=(fc.num_features, 8))
        a = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13, max_doc_fraction=1.0)
        a.fit(fc, directions=W)
        emb = mtm.document_embeddings(fc.counts, W, n_tokens=fc.n_tokens)
        b = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13, max_doc_fraction=1.0)
        b.fit(fc, emb)
        assert np.allclose(a.doc_embeddings, emb)
        assert a.labels == b.labels

    def test_directions_are_sliced_to_the_kept_features(self):
        # A pruned feature must drop out of the embedding too, or the decoder
        # rows stop lining up with the corpus columns.
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        counts = fc.counts.copy()
        counts[:, 0] = 7                        # fires everywhere -> pruned at 0.9
        W = np.random.default_rng(41).normal(size=(fc.num_features, 8))
        m = mtm.MechanisticBERTopic(min_cluster_size=5, max_doc_fraction=0.9)
        m.fit(mtm.FeatureCounts(counts, fc.n_tokens), directions=W)
        kept = m.corpus.kept_features
        assert 0 not in kept
        expected = mtm.document_embeddings(counts[:, kept], W[kept], n_tokens=fc.n_tokens)
        assert np.allclose(m.doc_embeddings, expected)

    def test_exactly_one_embedding_source(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        W = np.zeros((fc.num_features, 4))
        with pytest.raises(ValueError, match="exactly one"):
            mtm.MechanisticBERTopic().fit(fc)
        with pytest.raises(ValueError, match="exactly one"):
            mtm.MechanisticBERTopic().fit(fc, np.zeros((fc.num_docs, 4)), directions=W)

    def test_directions_need_the_counts(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        c = topica.Corpus.from_matrix(fc.counts)
        with pytest.raises(ValueError, match="needs the count matrix"):
            mtm.MechanisticBERTopic().fit(c, directions=np.zeros((fc.num_features, 4)))

    def test_deterministic_from_a_seed(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        W = np.random.default_rng(43).normal(size=(fc.num_features, 8))
        a = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13).fit(fc, directions=W)
        b = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13).fit(fc, directions=W)
        assert a.labels == b.labels
        assert np.array_equal(a.topic_word, b.topic_word)

    def test_topics_are_named_by_features(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, mtm.feature_thresholds(docs, q=0.8))
        names = [f"sae{i}" for i in range(fc.num_features)]
        W = np.random.default_rng(47).normal(size=(fc.num_features, 16))
        m = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13).fit(
            fc, directions=W, feature_names=names)
        assert all(w.startswith("sae") for w, _ in m.top_words(3, topic=0))

    def test_unfitted_access_raises(self):
        m = mtm.MechanisticBERTopic()
        for attr in ("topic_word", "labels", "doc_embeddings", "num_topics"):
            with pytest.raises(RuntimeError, match="not fitted"):
                getattr(m, attr)


# ---------------------------------------------------------------------------
# Tier gating and registry placement
# ---------------------------------------------------------------------------

class TestTier:
    @pytest.mark.parametrize("name", ["MechanisticLDA", "MechanisticBERTopic"])
    def test_is_experimental_and_gated(self, name):
        assert topica.REGISTRY[name].experimental
        topica.enable_experimental(False)
        try:
            with pytest.raises(RuntimeError, match="experimental"):
                getattr(topica, name)(3) if name == "MechanisticLDA" else getattr(topica, name)()
        finally:
            topica.enable_experimental(True)

    @pytest.mark.parametrize("name", ["MechanisticLDA", "MechanisticBERTopic"])
    def test_exported_at_top_level(self, name):
        assert getattr(topica, name) is getattr(mtm, name)
        assert name in topica.__all__


class TestDirectionsAlignment:
    """`directions=` may arrive at full feature width or already sliced."""

    def _setup(self):
        docs, _ = _planted()
        fc = mtm.featurize(docs, 0.9)
        counts = fc.counts.copy()
        counts[:, 0] = 7                       # fires everywhere -> pruned at 0.9
        return mtm.FeatureCounts(counts, fc.n_tokens)

    def test_pre_sliced_directions_are_accepted(self):
        fc = self._setup()
        W = np.random.default_rng(53).normal(size=(fc.num_features, 8))
        full = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13)
        full.fit(fc, directions=W)
        kept = full.corpus.kept_features
        assert len(kept) < fc.num_features
        sliced = mtm.MechanisticBERTopic(min_cluster_size=5, seed=13)
        sliced.fit(fc, directions=W[kept])
        assert np.allclose(full.doc_embeddings, sliced.doc_embeddings)

    def test_wrong_width_names_both_accepted_shapes(self):
        fc = self._setup()
        W = np.zeros((fc.num_features - 5, 8))   # neither full nor kept width
        with pytest.raises(ValueError, match="the full feature width"):
            mtm.MechanisticBERTopic(min_cluster_size=5).fit(fc, directions=W)
