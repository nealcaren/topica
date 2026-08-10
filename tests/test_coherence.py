"""Topic coherence (u_mass / c_uci / c_npmi / c_v) and topic diversity.

The measures are validated by their defining properties: a topic of words that
always co-occur should score higher than a topic of words that never do, the
normalized measures must respect their ranges, and the gensim-style API should
accept both explicit word lists and fitted models.
"""

import numpy as np
import pytest

import topica
from topica import LDA

ALL_TYPES = ["u_mass", "c_uci", "c_npmi", "c_v"]


@pytest.fixture(scope="module")
def reference():
    """A corpus where {a,b,c} always co-occur and the r* words are scattered."""
    rng = np.random.default_rng(0)
    texts = []
    for _ in range(500):
        d = []
        if rng.random() < 0.5:
            d += ["a", "b", "c"] * 3
        d += [f"r{int(rng.integers(200))}" for _ in range(6)]
        rng.shuffle(d)
        texts.append(d)
    return texts


COHERENT = ["a", "b", "c"]
INCOHERENT = ["r1", "r2", "r3"]


class TestCoherenceRanksTopics:
    @pytest.mark.parametrize("ct", ALL_TYPES)
    def test_coherent_beats_incoherent(self, reference, ct):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type=ct, topn=3)
        assert s.shape == (2,)
        assert s[0] > s[1], f"{ct}: coherent {s[0]} !> incoherent {s[1]}"

    def test_per_topic_shape(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT, ["a", "b", "c"]], reference, coherence_type="c_v", topn=3)
        assert s.shape == (3,)


class TestCoherenceAcceptsInputForms:
    """Raw strings and a Corpus must score identically to token lists — never
    silently iterated character-by-character to a degenerate constant (issue #648).
    """

    def test_raw_strings_match_token_lists(self, reference):
        strings = [" ".join(d) for d in reference]
        for ct in ALL_TYPES:
            tok = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type=ct, topn=3)
            s = topica.coherence([COHERENT, INCOHERENT], strings, coherence_type=ct, topn=3)
            np.testing.assert_allclose(s, tok, err_msg=ct)

    def test_raw_strings_are_not_a_degenerate_constant(self, reference):
        # The bug returned 1.0 for every topic (chars never match the vocab).
        strings = [" ".join(d) for d in reference]
        s = topica.coherence([COHERENT, INCOHERENT], strings, coherence_type="c_v", topn=3)
        assert not np.allclose(s, 1.0)
        assert s[0] > s[1]  # coherent still beats incoherent

    def test_corpus_reference_matches_token_lists(self, reference):
        corpus = topica.Corpus.from_documents(reference, min_doc_freq=1)
        tok = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        c = topica.coherence([COHERENT, INCOHERENT], corpus, coherence_type="c_npmi", topn=3)
        np.testing.assert_allclose(c, tok)

    def test_coherence_ci_accepts_strings(self, reference):
        strings = [" ".join(d) for d in reference]
        tok = topica.coherence_ci([COHERENT, INCOHERENT], reference,
                                  coherence_type="c_npmi", topn=3, n_boot=50, seed=0)
        s = topica.coherence_ci([COHERENT, INCOHERENT], strings,
                                coherence_type="c_npmi", topn=3, n_boot=50, seed=0)
        np.testing.assert_allclose(s.estimate, tok.estimate)

    def test_semantic_coherence_accepts_strings(self, reference):
        corpus = topica.Corpus.from_documents(reference, min_doc_freq=1)
        m = topica.LDA(3, seed=0).fit(corpus, iters=30)
        strings = [" ".join(d) for d in reference]
        np.testing.assert_allclose(
            topica.semantic_coherence(m, strings),
            topica.semantic_coherence(m, [list(d) for d in reference]),
        )


class TestCoherenceCI:
    def test_shape_and_estimate_matches_point(self, reference):
        r = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=100, seed=0
        )
        point = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        for arr in (r.estimate, r.se, r.ci_low, r.ci_high):
            assert arr.shape == (2,)
        np.testing.assert_allclose(r.estimate, point)

    def test_interval_brackets_estimate_and_orders(self, reference):
        r = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, seed=1
        )
        assert np.all(r.ci_low <= r.estimate + 1e-9)
        assert np.all(r.estimate <= r.ci_high + 1e-9)
        assert np.all(r.ci_low <= r.ci_high)
        assert np.all(np.isfinite(r.se)) and np.all(r.se >= 0.0)

    def test_wider_ci_widens_band(self, reference):
        wide = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, ci=0.95, seed=2
        )
        narrow = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, ci=0.5, seed=2
        )
        assert float(np.sum(wide.ci_high - wide.ci_low)) > float(np.sum(narrow.ci_high - narrow.ci_low))

    def test_accepts_fitted_model(self, reference):
        m = LDA(num_topics=2, seed=1)
        m.fit(reference, iters=200)
        r = topica.coherence_ci(m, reference, coherence_type="c_npmi", topn=3, n_boot=50, seed=0)
        assert r.estimate.shape == (2,)

    def test_empty_texts_raises(self):
        with pytest.raises(ValueError, match="texts is empty"):
            topica.coherence_ci([COHERENT], [], n_boot=10)


class TestRanges:
    def test_npmi_in_unit_range(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        assert np.all(s >= -1.0001) and np.all(s <= 1.0001)

    def test_cv_nonnegative_ish(self, reference):
        # C_v is a cosine of non-negative-ish context vectors; in [0, 1].
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_v", topn=3)
        assert np.all(s >= -0.01) and np.all(s <= 1.0001)

    def test_umass_nonpositive_for_rare(self, reference):
        s = topica.coherence([INCOHERENT], reference, coherence_type="u_mass", topn=3)
        assert s[0] <= 0.0


class TestApi:
    def test_invalid_type_raises(self, reference):
        with pytest.raises(ValueError):
            topica.coherence([COHERENT], reference, coherence_type="c_bogus")

    def test_accepts_fitted_model(self, reference):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        s = topica.coherence(m, docs, coherence_type="c_npmi", topn=3)
        assert s.shape == (2,)
        assert np.all(np.isfinite(s))

    def test_accepts_word_prob_pairs(self, reference):
        topics = [[("a", 0.5), ("b", 0.3), ("c", 0.2)]]
        s = topica.coherence(topics, reference, coherence_type="c_v", topn=3)
        assert s.shape == (1,)

    def test_window_size_override(self, reference):
        s = topica.coherence([COHERENT], reference, coherence_type="c_npmi", topn=3, window_size=5)
        assert s.shape == (1,) and np.isfinite(s[0])

    def test_default_is_cv(self, reference):
        a = topica.coherence([COHERENT], reference, topn=3)
        b = topica.coherence([COHERENT], reference, coherence_type="c_v", topn=3)
        assert np.allclose(a, b)


class TestDiversity:
    def test_disjoint_is_one(self):
        assert topica.topic_diversity([["a", "b", "c"], ["d", "e", "f"]], topn=3) == 1.0

    def test_identical_is_half(self):
        # two identical 3-word topics: 3 unique / 6 total.
        assert topica.topic_diversity([["a", "b", "c"], ["a", "b", "c"]], topn=3) == 0.5

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        d = topica.topic_diversity(m, topn=3)
        assert 0.0 < d <= 1.0


class TestSemanticDiversity:
    """topic_semantic_diversity (Wu, Nguyen & Luu 2024, Eq. 18): fraction of
    top-word *pairs* that are unique to a single topic."""

    def test_disjoint_is_one(self):
        # No shared words → no shared pairs → every pair unique.
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c"], ["d", "e", "f"]], topn=3
        )
        assert tsd == 1.0

    def test_identical_is_zero(self):
        # Two identical topics: every pair appears in BOTH topics, so no pair
        # occurrence has global count 1 → 0.0. (Note: unlike topic_diversity,
        # which counts unique *words* and gives 0.5 here.)
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c"], ["a", "b", "c"]], topn=3
        )
        assert tsd == 0.0

    def test_partial_overlap_exact(self):
        # A = {a,b,c,d}, B = {a,b,e,f}, topn=4 → 6 pairs each, 12 total.
        # The only shared pair is {a,b} (count 2); it occurs once in A and once
        # in B → 2 non-unique occurrences. Unique = 10 → TSD = 10/12.
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c", "d"], ["a", "b", "e", "f"]], topn=4
        )
        assert tsd == 10 / 12

    def test_topn_two_edge(self):
        # topn=2 → exactly one pair per topic; disjoint pairs → 1.0.
        tsd = topica.topic_semantic_diversity(
            [["a", "b"], ["c", "d"]], topn=2
        )
        assert tsd == 1.0

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        tsd = topica.topic_semantic_diversity(m, topn=3)
        assert 0.0 <= tsd <= 1.0

    def test_topn_below_two_raises(self):
        with pytest.raises(ValueError):
            topica.topic_semantic_diversity([["a", "b", "c"]], topn=1)


class TestInvertedRBO:
    """inverted_rbo (OCTIS InvertedRBO; Bianchi, Terragni & Hovy 2021):
    ``1 - mean pairwise RBO`` over top-word rankings."""

    def test_disjoint_is_one(self):
        # No shared words at any rank → RBO 0 → diversity 1.0.
        assert topica.inverted_rbo(
            [["a", "b", "c"], ["d", "e", "f"]], topn=3
        ) == pytest.approx(1.0)

    def test_identical_is_zero(self):
        # Identical rankings → RBO 1 → diversity 0.0.
        assert topica.inverted_rbo(
            [["a", "b", "c"], ["a", "b", "c"]], topn=3
        ) == pytest.approx(0.0)

    def test_rank_sensitivity(self):
        # Both pairs share {a,b} as a set, so topic_diversity is identical (0.5)
        # for either. RBO instead punishes sharing them at the TOP ranks: the
        # aligned pair overlaps more than the reversed pair, so diversity is
        # strictly lower when the shared words sit at matching ranks.
        aligned = topica.inverted_rbo([["a", "b"], ["a", "b"]], topn=2)
        reversed_ = topica.inverted_rbo([["a", "b"], ["b", "a"]], topn=2)
        assert aligned < reversed_

    def test_single_topic_is_nan(self):
        assert np.isnan(topica.inverted_rbo([["a", "b", "c"]], topn=3))

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        d = topica.inverted_rbo(m, topn=3)
        assert 0.0 <= d <= 1.0

    def test_bad_p_raises(self):
        with pytest.raises(ValueError):
            topica.inverted_rbo([["a", "b"], ["c", "d"]], p=1.0)


class TestEmbeddingCoherence:
    """embedding_coherence (OCTIS we_pairwise / we_centroid; Ding, Nallapati &
    Xiang 2018): top-word proximity in an embedding space."""

    def test_pairwise_identical_vectors_is_one(self):
        # All three words map to the same unit vector → every cosine is 1.
        emb = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]}
        out = topica.embedding_coherence([["a", "b", "c"]], emb, topn=3)
        assert out.shape == (1,)
        assert out[0] == pytest.approx(1.0)

    def test_pairwise_orthogonal_is_zero(self):
        emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        out = topica.embedding_coherence([["a", "b"]], emb, topn=2)
        assert out[0] == pytest.approx(0.0)

    def test_centroid_matches_hand_computed(self):
        # Two orthogonal unit vectors: centroid is (1,1)/sqrt2; each word's
        # cosine to it is 1/sqrt2, so the mean is 1/sqrt2.
        emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        out = topica.embedding_coherence(
            [["a", "b"]], emb, topn=2, method="centroid"
        )
        assert out[0] == pytest.approx(1.0 / np.sqrt(2.0))

    def test_centroid_is_octis_we_centroid_complement(self):
        # Faithfulness pin (non-unit vectors, so raw-sum vs unit-avg centroids
        # differ): topica's centroid similarity == 1 - OCTIS we_centroid, where
        # OCTIS = mean cosine DISTANCE to the centroid of the RAW vectors.
        vecs = {"a": np.array([1.0, 0.0]), "b": np.array([3.0, 3.0])}
        words = ["a", "b"]
        raw = np.array([vecs[w] for w in words])
        centroid = raw.sum(0)
        centroid = centroid / np.linalg.norm(centroid)      # OCTIS: raw sum, normed
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        octis_distance = np.mean([1.0 - u @ centroid for u in unit])
        out = topica.embedding_coherence([words], vecs, topn=2, method="centroid")
        assert out[0] == pytest.approx(1.0 - octis_distance)
        # And it must differ from the naive unit-average centroid (guards the
        # raw-sum construction from silently regressing).
        unit_centroid = unit.mean(0)
        unit_centroid = unit_centroid / np.linalg.norm(unit_centroid)
        naive = float((unit @ unit_centroid).mean())
        assert not np.isclose(out[0], naive)

    def test_matrix_form_matches_dict_form(self):
        vocab = ["a", "b", "c"]
        mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        emb = dict(zip(vocab, mat))
        topics = [["a", "b", "c"]]
        from_mat = topica.embedding_coherence(topics, mat, vocab, topn=3)
        from_dict = topica.embedding_coherence(topics, emb, topn=3)
        np.testing.assert_allclose(from_mat, from_dict)

    def test_unnormalized_vectors_are_normalized(self):
        # Scaling a vector must not change cosine — parallel vectors → 1.0.
        emb = {"a": [2.0, 0.0], "b": [5.0, 0.0]}
        out = topica.embedding_coherence([["a", "b"]], emb, topn=2)
        assert out[0] == pytest.approx(1.0)

    def test_missing_words_dropped_then_nan(self):
        # Only one of the two top words has an embedding → < 2 → nan (+warns).
        emb = {"a": [1.0, 0.0]}
        with pytest.warns(UserWarning, match="top words embedded"):
            out = topica.embedding_coherence([["a", "zzz"]], emb, topn=2)
        assert np.isnan(out[0])

    def test_zero_vector_is_dropped_not_corrupted(self):
        # A zero-vector word carries no direction: it must be DROPPED (treated
        # like OOV), not silently read as cosine ~0. Here 'a' is zero, so only
        # 'b' survives → < 2 → nan, exactly as if 'a' were absent.
        with pytest.warns(UserWarning):
            zeroed = topica.embedding_coherence(
                [["a", "b"]], {"a": [0.0, 0.0], "b": [1.0, 0.0]}, topn=2
            )
        with pytest.warns(UserWarning):
            absent = topica.embedding_coherence(
                [["a", "b"]], {"b": [1.0, 0.0]}, topn=2
            )
        assert np.isnan(zeroed[0]) and np.isnan(absent[0])

    def test_nan_and_inf_vectors_are_dropped(self):
        # A single NaN/inf coordinate makes the word undirected → dropped, so a
        # good sibling word is unaffected rather than the whole topic → nan.
        emb = {"a": [np.nan, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]}
        out = topica.embedding_coherence([["a", "b", "c"]], emb, topn=3)
        assert out[0] == pytest.approx(1.0)  # b,c survive, cos=1
        emb_inf = {"a": [np.inf, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]}
        out_inf = topica.embedding_coherence([["a", "b", "c"]], emb_inf, topn=3)
        assert out_inf[0] == pytest.approx(1.0)

    def test_tiny_vector_normalized_to_unit(self):
        # A tiny-but-nonzero vector still has a direction; it must renormalize to
        # unit length (parallel tiny + unit vector → cosine 1.0), not read as
        # near-zero magnitude.
        emb = {"a": [1e-15, 0.0], "b": [1.0, 0.0]}
        out = topica.embedding_coherence([["a", "b"]], emb, topn=2)
        assert out[0] == pytest.approx(1.0)

    def test_does_not_mutate_caller_embeddings(self):
        vecs = {"a": np.array([2.0, 0.0]), "b": np.array([0.0, 3.0])}
        before = {w: v.copy() for w, v in vecs.items()}
        topica.embedding_coherence([["a", "b"]], vecs, topn=2)
        for w in vecs:
            np.testing.assert_array_equal(vecs[w], before[w])

    def test_partial_coverage_uses_nanmean(self):
        # Corpus score over a partly-covered model: .mean() poisons to nan, but
        # np.nanmean gives the covered topics' mean (documented aggregation).
        emb = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
        topics = [["a", "b", "c"], ["rare1", "rare2", "c"]]  # topic 2 uncovered
        with pytest.warns(UserWarning):
            out = topica.embedding_coherence(topics, emb, topn=3)
        assert np.isnan(out[1]) and np.isnan(np.mean(out))
        assert not np.isnan(np.nanmean(out))

    def test_matrix_without_vocabulary_raises(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            topica.embedding_coherence([["a", "b"]], mat, topn=2)

    def test_matrix_vocab_mismatch_raises(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            topica.embedding_coherence([["a", "b"]], mat, ["a", "b", "c"], topn=2)

    def test_bad_method_raises(self):
        emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        with pytest.raises(ValueError):
            topica.embedding_coherence([["a", "b"]], emb, topn=2, method="cosine")

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        rng = np.random.default_rng(0)
        emb = {w: rng.standard_normal(8) for w in m.vocabulary}
        out = topica.embedding_coherence(m, emb, topn=3)
        assert out.shape == (2,)
        assert np.all((out >= -1.0 - 1e-9) & (out <= 1.0 + 1e-9))


class TestAnalysisContract:
    """Any object exposing the analysis contract -- ``topic_word`` /
    ``doc_topic`` / ``vocabulary`` -- works with the model-agnostic diagnostics,
    even without a ``top_words`` method. This pins the extensibility guarantee:
    a foreign model that presents the two matrices inherits the stack for free.
    """

    def test_duck_typed_model_works_without_top_words(self):
        docs = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)

        class Contract:  # the four members, NO top_words
            topic_word = m.topic_word
            doc_topic = m.doc_topic
            vocabulary = m.vocabulary

        c = Contract()
        # coherence / topic_diversity derive top words from topic_word + vocabulary
        np.testing.assert_allclose(
            topica.coherence(c, docs), topica.coherence(m, docs)
        )
        assert topica.topic_diversity(c) == topica.topic_diversity(m)
        np.testing.assert_allclose(topica.exclusivity(c), topica.exclusivity(m))


class TestUMassExternalCorpus:
    """u_mass with an external reference corpus that does not contain some top
    words must not produce a spuriously large positive score (issue #103)."""

    def test_absent_word_does_not_inflate_umass(self):
        # Reference corpus contains only "a" and "b"; "z" is absent.
        reference = [["a", "b"]] * 20
        topic_present = ["a", "b"]   # both words in reference: normal score
        topic_absent = ["a", "z"]    # "z" absent from reference

        s_present = topica.coherence([topic_present], reference,
                                     coherence_type="u_mass", topn=2)[0]
        s_absent = topica.coherence([topic_absent], reference,
                                    coherence_type="u_mass", topn=2)[0]

        # Before the fix, s_absent was a large positive (log(1/eps)).
        # After the fix: the pair (a, z) is skipped, so s_absent equals
        # nan (no pairs counted) or a non-positive finite value.
        assert np.isnan(s_absent) or s_absent <= 0.0, (
            f"u_mass with absent word should be nan or <= 0, got {s_absent}"
        )

    def test_umass_all_words_present_is_small_and_finite(self):
        # Classic case: topic words all appear in the reference. With numerator
        # +1 smoothing and perfect co-occurrence the score can be a hair above
        # zero (log((occ+1)/occ)), but it is finite and near zero, never the
        # large spurious positive the absent-word path used to produce.
        reference = [["a", "b", "c"]] * 50
        score = topica.coherence([["a", "b", "c"]], reference,
                                 coherence_type="u_mass", topn=3)[0]
        assert np.isfinite(score)
        assert score < 0.1


class TestModelCoherenceMethodConsistency:
    """The model .coherence() method is consistent with the standalone (#686):
    same `n` name, and `coherence_type=` routes through the same implementation."""

    def _fit(self):
        docs = [["a", "b", "c", "x"], ["a", "b", "b", "x"],
                ["c", "c", "d", "y"], ["d", "e", "f", "y"]] * 12
        return topica.LDA(3, seed=13).fit(docs, iters=40), docs

    def test_default_is_umass_backcompat(self):
        m, _ = self._fit()
        assert np.allclose(m.coherence(), m.coherence(coherence_type="u_mass"))

    def test_coherence_type_matches_standalone(self):
        m, docs = self._fit()
        for ct in ("u_mass", "c_v", "c_npmi", "c_uci"):
            method = m.coherence(n=4, coherence_type=ct)
            standalone = topica.coherence(m, docs, coherence_type=ct, n=4)
            assert np.allclose(method, standalone, equal_nan=True), ct

    def test_explicit_texts(self):
        m, _ = self._fit()
        ext = [["a", "b", "c"]] * 30
        got = m.coherence(n=3, coherence_type="u_mass", texts=ext)
        assert got.shape == (3,)

    def test_standalone_n_aliases_topn(self):
        m, docs = self._fit()
        assert np.allclose(topica.coherence(m, docs, n=5, coherence_type="u_mass"),
                           topica.coherence(m, docs, topn=5, coherence_type="u_mass"))
