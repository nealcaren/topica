"""Tests for the topica.stm analysis toolkit.

Covers: estimate_effect, frex, label_topics, topic_correlation,
find_thoughts, search_k, and _normal_ppf.
"""

from __future__ import annotations

import numpy as np
import pytest

from topica import LDA, stm


# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------
# ~160 documents, binary covariate x drives content:
#   x=1  => heavily "space" words  (planet star moon rocket orbit)
#   x=0  => heavily "animal" words (cat dog fish bird bear)
# We fit LDA(num_topics=2, seed=1) at session scope and cache everything.

_SPACE_WORDS = ["planet", "star", "moon", "rocket", "orbit"]
_ANIMAL_WORDS = ["cat", "dog", "fish", "bird", "bear"]


def _make_synthetic_corpus(rng, n_per_class=80):
    """Return (docs, covariate_x, vocab) for the synthetic binary corpus."""
    space_docs = []
    for _ in range(n_per_class):
        # Heavy draw from space words
        toks = list(rng.choice(_SPACE_WORDS, size=20, replace=True)) + list(
            rng.choice(_ANIMAL_WORDS, size=2, replace=True)
        )
        space_docs.append(toks)

    animal_docs = []
    for _ in range(n_per_class):
        toks = list(rng.choice(_ANIMAL_WORDS, size=20, replace=True)) + list(
            rng.choice(_SPACE_WORDS, size=2, replace=True)
        )
        animal_docs.append(toks)

    docs = space_docs + animal_docs
    x = np.array([1] * n_per_class + [0] * n_per_class, dtype=float)
    return docs, x


@pytest.fixture(scope="module")
def synthetic_model_and_x():
    """Session-scoped fitted LDA + covariate for the synthetic corpus."""
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=80)
    model = LDA(num_topics=2, seed=1)
    model.fit(docs, iters=300, num_samples=3, sample_interval=10)
    return model, x


@pytest.fixture(scope="module")
def space_topic_idx(synthetic_model_and_x):
    """Index of the 'space' topic in the fitted 2-topic model."""
    model, _ = synthetic_model_and_x
    vocab = model.vocabulary
    phi = model.topic_word
    # The space topic has higher weight on "planet"
    planet_col = vocab.index("planet")
    return int(phi[:, planet_col].argmax())


# ---------------------------------------------------------------------------
# _normal_ppf
# ---------------------------------------------------------------------------

class TestNormalPPF:
    def test_ppf_0975_approx_196(self):
        val = stm._normal_ppf(0.975)
        assert abs(val - 1.96) < 1e-2

    def test_ppf_0_5_is_zero(self):
        assert abs(stm._normal_ppf(0.5)) < 1e-6

    def test_ppf_out_of_range_raises(self):
        with pytest.raises(ValueError):
            stm._normal_ppf(0.0)
        with pytest.raises(ValueError):
            stm._normal_ppf(1.0)


# ---------------------------------------------------------------------------
# estimate_effect
# ---------------------------------------------------------------------------

class TestEstimateEffect:
    def test_space_topic_positive_z(self, synthetic_model_and_x, space_topic_idx):
        """Space-topic coefficient on x is positive with large z (effect recovery)."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], topics=[space_topic_idx]
        )
        assert len(effects) == 1
        eff = effects[0]
        # intercept prepended, then "x"
        x_idx = eff.feature_names.index("x")
        assert eff.z[x_idx] > 3, f"Expected z>3 for space topic, got z={eff.z[x_idx]:.2f}"

    def test_animal_topic_negative_coef(self, synthetic_model_and_x, space_topic_idx):
        """Animal-topic coefficient on x is negative."""
        model, x = synthetic_model_and_x
        animal_topic = 1 - space_topic_idx
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], topics=[animal_topic]
        )
        eff = effects[0]
        x_idx = eff.feature_names.index("x")
        assert eff.coef[x_idx] < 0

    def test_r_squared_in_range(self, synthetic_model_and_x, space_topic_idx):
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], topics=[space_topic_idx]
        )
        r2 = effects[0].r_squared
        assert 0.0 <= r2 <= 1.0

    def test_coef_length_with_intercept(self, synthetic_model_and_x, space_topic_idx):
        """With intercept, coef/se/z/ci arrays length = n_features + 1."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], add_intercept=True,
            topics=[space_topic_idx]
        )
        eff = effects[0]
        assert len(eff.coef) == 2  # intercept + x
        assert len(eff.se) == 2
        assert len(eff.z) == 2
        assert len(eff.ci_low) == 2
        assert len(eff.ci_high) == 2

    def test_intercept_in_feature_names(self, synthetic_model_and_x, space_topic_idx):
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], add_intercept=True,
            topics=[space_topic_idx]
        )
        assert "intercept" in effects[0].feature_names

    def test_no_intercept_mode(self, synthetic_model_and_x, space_topic_idx):
        """add_intercept=False: no 'intercept' in names; coef length == X columns."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], add_intercept=False,
            topics=[space_topic_idx]
        )
        eff = effects[0]
        assert "intercept" not in eff.feature_names
        assert len(eff.coef) == 1

    def test_intercept_not_doubled_when_x_has_constant(self, synthetic_model_and_x, space_topic_idx):
        """#775 T1.2: if X already carries a constant column, add_intercept=True must
        not prepend a *second* intercept (which splits it across collinear columns
        and shifts every coefficient label). It warns and skips instead."""
        import numpy as np
        model, x = synthetic_model_and_x
        x2 = np.atleast_2d(x)
        if x2.shape[0] == 1:
            x2 = x2.T
        Xc = np.column_stack([np.ones(x2.shape[0]), x2])  # user-supplied intercept
        with pytest.warns(UserWarning, match="already has a constant column"):
            effects = stm.estimate_effect(
                model.doc_topic, Xc, feature_names=["Intercept", "x"],
                add_intercept=True, topics=[space_topic_idx],
            )
        eff = effects[0]
        # exactly one intercept column, and 'x' still at the index the user gave it
        assert [n.lower() for n in eff.feature_names].count("intercept") == 1
        assert eff.feature_names == ["Intercept", "x"]
        assert len(eff.coef) == 2

    def test_all_topics_default(self, synthetic_model_and_x):
        """Default topics=None returns one effect per topic."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(model.doc_topic, x)
        assert len(effects) == model.num_topics

    def test_as_dict_structure(self, synthetic_model_and_x, space_topic_idx):
        """as_dict() returns dict with 'topic', 'r_squared', and per-feature sub-dicts."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], topics=[space_topic_idx]
        )
        d = effects[0].as_dict()
        assert "topic" in d
        assert "r_squared" in d
        # "x" and "intercept" should both be sub-dict keys
        assert "x" in d
        assert isinstance(d["x"], dict)
        assert "coef" in d["x"]
        assert "se" in d["x"]
        assert "z" in d["x"]
        assert "ci" in d["x"]

    def test_row_count_mismatch_raises(self, synthetic_model_and_x):
        model, x = synthetic_model_and_x
        bad_x = x[:-5]  # wrong length
        with pytest.raises(ValueError, match="rows"):
            stm.estimate_effect(model.doc_topic, bad_x)

    def test_to_frame_columns_and_rows(self, synthetic_model_and_x, space_topic_idx):
        """to_frame() gives one row per feature with the tidy column set."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"], topics=[space_topic_idx]
        )
        df = effects[0].to_frame()
        assert list(df.columns) == [
            "topic", "feature", "coef", "se", "z", "pvalue", "ci_low", "ci_high",
            "r_squared"
        ]
        assert len(df) == len(effects[0].feature_names)  # intercept + x
        assert set(df["feature"]) == {"intercept", "x"}
        assert (df["topic"] == space_topic_idx).all()
        # values agree with the underlying arrays
        xrow = df[df["feature"] == "x"].iloc[0]
        j = effects[0].feature_names.index("x")
        assert xrow["coef"] == pytest.approx(float(effects[0].coef[j]))
        assert xrow["ci_low"] == pytest.approx(float(effects[0].ci_low[j]))

    def test_to_frame_concat_one_row_per_topic_feature(self, synthetic_model_and_x):
        """Concatenating per-topic frames yields one row per (topic, feature)."""
        import pandas as pd

        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(model.doc_topic, x, feature_names=["x"])
        table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
        n_feat = len(effects[0].feature_names)
        assert len(table) == len(effects) * n_feat
        assert table.groupby("topic").size().eq(n_feat).all()

    def test_feature_names_length_mismatch_raises(self, synthetic_model_and_x):
        model, x = synthetic_model_and_x
        with pytest.raises(ValueError):
            stm.estimate_effect(model.doc_topic, x, feature_names=["a", "b"])

    def test_topic_out_of_range_raises(self, synthetic_model_and_x):
        model, x = synthetic_model_and_x
        with pytest.raises(ValueError, match="out of range"):
            stm.estimate_effect(model.doc_topic, x, topics=[999])

    def test_multi_column_x(self, synthetic_model_and_x, space_topic_idx):
        """Two-covariate X: shapes correct."""
        model, x = synthetic_model_and_x
        X2 = np.column_stack([x, x ** 2])
        effects = stm.estimate_effect(
            model.doc_topic, X2, feature_names=["x", "x2"],
            topics=[space_topic_idx]
        )
        eff = effects[0]
        # intercept + 2 features
        assert len(eff.coef) == 3
        assert len(eff.feature_names) == 3

    def test_ci_bounds_ordered(self, synthetic_model_and_x, space_topic_idx):
        """ci_low <= ci_high for all coefficients."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, feature_names=["x"],
            topics=[space_topic_idx]
        )
        eff = effects[0]
        assert np.all(eff.ci_low <= eff.ci_high)

    def test_1d_x_accepted(self, synthetic_model_and_x, space_topic_idx):
        """A 1-D X array should be accepted without error."""
        model, x = synthetic_model_and_x
        effects = stm.estimate_effect(
            model.doc_topic, x, topics=[space_topic_idx]
        )
        assert len(effects) == 1


# ---------------------------------------------------------------------------
# frex
# ---------------------------------------------------------------------------

class TestFrex:
    def test_returns_k_lists(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        result = stm.frex(model.topic_word, model.vocabulary)
        assert len(result) == model.num_topics

    def test_each_list_has_n_or_fewer(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        n = 5
        result = stm.frex(model.topic_word, model.vocabulary, n=n)
        for topic_list in result:
            assert len(topic_list) <= n

    def test_words_in_vocabulary(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        vocab_set = set(model.vocabulary)
        for topic_list in stm.frex(model.topic_word, model.vocabulary):
            for word, _ in topic_list:
                assert word in vocab_set

    def test_scores_descending(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        for topic_list in stm.frex(model.topic_word, model.vocabulary):
            scores = [s for _, s in topic_list]
            assert scores == sorted(scores, reverse=True)

    def test_space_topic_frex_words_are_space(self, synthetic_model_and_x, space_topic_idx):
        """FREX top words for the space topic are space words."""
        model, _ = synthetic_model_and_x
        result = stm.frex(model.topic_word, model.vocabulary, n=5)
        space_frex_words = {w for w, _ in result[space_topic_idx]}
        assert len(space_frex_words & set(_SPACE_WORDS)) > 0, (
            f"Expected space FREX words, got {space_frex_words}"
        )

    def test_default_weight(self, synthetic_model_and_x):
        """w=0.5 is the default; explicit call should match."""
        model, _ = synthetic_model_and_x
        r1 = stm.frex(model.topic_word, model.vocabulary, w=0.5)
        r2 = stm.frex(model.topic_word, model.vocabulary)
        for t in range(model.num_topics):
            assert [w for w, _ in r1[t]] == [w for w, _ in r2[t]]


# ---------------------------------------------------------------------------
# label_topics
# ---------------------------------------------------------------------------

class TestLabelTopics:
    def test_returns_k_dicts(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        result = stm.label_topics(model.topic_word, model.vocabulary)
        assert len(result) == model.num_topics

    def test_dict_has_all_keys(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        for d in stm.label_topics(model.topic_word, model.vocabulary):
            assert set(d.keys()) == {"prob", "frex", "lift", "score"}

    def test_each_value_is_list_of_pairs(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        for d in stm.label_topics(model.topic_word, model.vocabulary):
            for key in ("prob", "frex", "lift", "score"):
                for item in d[key]:
                    assert len(item) == 2
                    word, val = item
                    assert isinstance(word, str)
                    assert isinstance(val, float)

    def test_prob_words_are_top_phi(self, synthetic_model_and_x):
        """prob words match descending argsort of phi for each topic."""
        model, _ = synthetic_model_and_x
        result = stm.label_topics(model.topic_word, model.vocabulary, n=5)
        phi = model.topic_word
        vocab = model.vocabulary
        for t, d in enumerate(result):
            expected_idx = np.argsort(phi[t])[::-1][:5]
            expected_words = [vocab[i] for i in expected_idx]
            got_words = [w for w, _ in d["prob"]]
            assert got_words == expected_words

    def test_n_limits_list_length(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        n = 3
        result = stm.label_topics(model.topic_word, model.vocabulary, n=n)
        for d in result:
            for key in ("prob", "frex", "lift", "score"):
                assert len(d[key]) <= n


# ---------------------------------------------------------------------------
# topic_correlation
# ---------------------------------------------------------------------------

class TestTopicCorrelation:
    def test_cor_shape(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        K = model.num_topics
        assert tc.cor.shape == (K, K)

    def test_cor_symmetric(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        np.testing.assert_allclose(tc.cor, tc.cor.T, atol=1e-12)

    def test_cor_diagonal_approx_one(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        np.testing.assert_allclose(np.diag(tc.cor), 1.0, atol=1e-10)

    def test_adjacency_shape(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        K = model.num_topics
        assert tc.adjacency.shape == (K, K)

    def test_adjacency_zero_diagonal(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        assert np.all(np.diag(tc.adjacency) == 0)

    def test_adjacency_binary(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        vals = np.unique(tc.adjacency)
        assert set(vals).issubset({0, 1})

    def test_two_topics_anticorrelated(self, synthetic_model_and_x):
        """With 2 topics (theta sums to 1), the off-diagonal correlation is ≈-1."""
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        assert tc.cor[0, 1] < 0

    def test_two_topics_no_positive_edges(self, synthetic_model_and_x):
        """Anti-correlated 2-topic model has no positive edges above threshold=0.05."""
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic, threshold=0.05)
        assert tc.edges == []

    def test_three_topic_symmetry_and_diagonal(self):
        """3-topic model: adjacency is symmetric, diagonal zero."""
        # Build a quick 3-topic synthetic theta
        rng = np.random.default_rng(7)
        # Dirichlet draws give correlated proportions
        theta = rng.dirichlet([1, 1, 1], size=200)
        tc = stm.topic_correlation(theta, threshold=0.05)
        assert tc.adjacency.shape == (3, 3)
        assert np.all(np.diag(tc.adjacency) == 0)
        np.testing.assert_array_equal(tc.adjacency, tc.adjacency.T)

    def test_edges_list_type(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        tc = stm.topic_correlation(model.doc_topic)
        assert isinstance(tc.edges, list)


# ---------------------------------------------------------------------------
# find_thoughts
# ---------------------------------------------------------------------------

class TestFindThoughts:
    def test_returns_n_or_fewer(self, synthetic_model_and_x, space_topic_idx):
        model, _ = synthetic_model_and_x
        n = 5
        result = stm.find_thoughts(model.doc_topic, topic=space_topic_idx, n=n)
        assert len(result) <= n

    def test_sorted_descending_proportion(self, synthetic_model_and_x, space_topic_idx):
        model, _ = synthetic_model_and_x
        result = stm.find_thoughts(model.doc_topic, topic=space_topic_idx, n=5)
        props = [p for _, p, _ in result]
        assert props == sorted(props, reverse=True)

    def test_doc_index_valid(self, synthetic_model_and_x, space_topic_idx):
        model, _ = synthetic_model_and_x
        n_docs = model.doc_topic.shape[0]
        result = stm.find_thoughts(model.doc_topic, topic=space_topic_idx, n=5)
        for idx, _, _ in result:
            assert 0 <= idx < n_docs

    def test_text_is_none_when_not_provided(self, synthetic_model_and_x, space_topic_idx):
        model, _ = synthetic_model_and_x
        result = stm.find_thoughts(model.doc_topic, topic=space_topic_idx, n=3)
        for _, _, text in result:
            assert text is None

    def test_text_passed_through(self, synthetic_model_and_x, space_topic_idx):
        model, _ = synthetic_model_and_x
        texts = [f"doc_{i}" for i in range(model.doc_topic.shape[0])]
        result = stm.find_thoughts(model.doc_topic, texts=texts, topic=space_topic_idx, n=3)
        for idx, _, text in result:
            assert text == f"doc_{idx}"

    def test_out_of_range_topic_raises(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        with pytest.raises(ValueError, match="out of range"):
            stm.find_thoughts(model.doc_topic, topic=999, n=3)

    def test_negative_topic_raises(self, synthetic_model_and_x):
        model, _ = synthetic_model_and_x
        with pytest.raises(ValueError):
            stm.find_thoughts(model.doc_topic, topic=-1, n=3)

    def test_proportion_matches_theta(self, synthetic_model_and_x, space_topic_idx):
        """Returned proportions match the actual theta values."""
        model, _ = synthetic_model_and_x
        result = stm.find_thoughts(model.doc_topic, topic=space_topic_idx, n=3)
        for idx, prop, _ in result:
            expected = float(model.doc_topic[idx, space_topic_idx])
            assert abs(prop - expected) < 1e-9


# ---------------------------------------------------------------------------
# search_k
# ---------------------------------------------------------------------------

class TestSearchK:
    @pytest.fixture(scope="class")
    def search_k_results(self):
        """Run search_k on a small corpus; keep iterations tiny for speed."""
        rng = np.random.default_rng(99)
        # Small corpus: 40 docs, 2 topics of words
        docs = []
        for _ in range(20):
            docs.append(list(rng.choice(["alpha", "beta", "gamma", "delta", "epsilon"], size=15, replace=True)))
        for _ in range(20):
            docs.append(list(rng.choice(["uno", "dos", "tres", "cuatro", "cinco"], size=15, replace=True)))
        results = stm.search_k(
            docs,
            ks=[2, 3],
            iters=100,
            num_samples=2,
            sample_interval=5,
            seed=42,
        )
        return results

    def test_one_dict_per_k(self, search_k_results):
        assert len(search_k_results) == 2

    def test_dict_has_required_keys(self, search_k_results):
        for row in search_k_results:
            assert "k" in row
            assert "coherence" in row
            assert "exclusivity" in row

    def test_no_perplexity_without_held_out(self, search_k_results):
        for row in search_k_results:
            assert "perplexity" not in row

    def test_k_values_correct(self, search_k_results):
        ks = [r["k"] for r in search_k_results]
        assert ks == [2, 3]

    def test_coherence_is_nonpositive(self, search_k_results):
        """UMass coherence is ≤ 0."""
        for row in search_k_results:
            assert row["coherence"] <= 0.0

    def test_exclusivity_nonnegative(self, search_k_results):
        # stm's exclusivity is a FREX summary over the top words (roughly [0, n]),
        # not a [0, 1] mean.
        for row in search_k_results:
            assert row["exclusivity"] >= 0.0 and np.isfinite(row["exclusivity"])

    def test_with_held_out_has_perplexity(self):
        rng = np.random.default_rng(77)
        docs = []
        held = []
        for _ in range(20):
            docs.append(list(rng.choice(["alpha", "beta", "gamma"], size=10, replace=True)))
        for _ in range(5):
            held.append(list(rng.choice(["alpha", "beta", "gamma"], size=10, replace=True)))
        results = stm.search_k(
            docs,
            ks=[2],
            held_out=held,
            iters=100,
            num_samples=2,
            sample_interval=5,
            seed=42,
        )
        assert "perplexity" in results[0]
        assert results[0]["perplexity"] > 0

    def test_all_floats(self, search_k_results):
        for row in search_k_results:
            assert isinstance(row["coherence"], float)
            assert isinstance(row["exclusivity"], float)

    def test_search_k_with_content_covariates(self):
        rng = np.random.default_rng(101)
        docs = []
        for _ in range(15):
            docs.append(list(rng.choice(["alpha", "beta", "gamma"], size=10, replace=True)))
        for _ in range(15):
            docs.append(list(rng.choice(["uno", "dos", "tres"], size=10, replace=True)))
        
        # Prevalence design (30, 1)
        X = np.array([1.0] * 15 + [0.0] * 15, dtype=np.float64).reshape(-1, 1)
        # Content covariate
        groups = ["en"] * 15 + ["es"] * 15

        # Running search_k with stm model, prevalence, and content
        with pytest.warns(UserWarning, match="Exclusivity and coherence calculations"):
            results = stm.search_k(
                docs,
                ks=[2],
                model="stm",
                prevalence=X,
                content=groups,
                iters=10,
                seed=42
            )
        assert len(results) == 1
        assert results[0]["k"] == 2
        assert "coherence" in results[0]
        assert "exclusivity" in results[0]

        # Also test with non-default coherence type (e.g. c_npmi)
        with pytest.warns(UserWarning, match="Exclusivity and coherence calculations"):
            results_npmi = stm.search_k(
                docs,
                ks=[2],
                model="stm",
                prevalence=X,
                content=groups,
                iters=10,
                seed=42,
                coherence_type="c_npmi"
            )
        assert len(results_npmi) == 1
        assert results_npmi[0]["coherence_metric"] == "c_npmi"
        assert -1.0 <= results_npmi[0]["coherence"] <= 1.0


def test_stm_beta_init_warm_start():
    """STM accepts a caller-supplied base beta (the warm-start hook, issue #234B):
    injecting the spectral init reproduces the default spectral fit."""
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=60)
    X = x.reshape(-1, 1)
    spec = topica.STM(num_topics=2, seed=1)
    spec.fit(docs, prevalence=X, iters=0)  # pure spectral-init topic-word
    warm = topica.STM(num_topics=2, seed=1)
    warm.fit(docs, prevalence=X, iters=10, beta_init=spec.topic_word)
    default = topica.STM(num_topics=2, seed=1)
    default.fit(docs, prevalence=X, iters=10)
    np.testing.assert_allclose(warm.topic_word, default.topic_word, atol=1e-9)


def test_stm_restarts_returns_best_bound():
    """STM.fit(restarts=N) fits N seeded inits and returns the best-bound model
    (the #871 multi-start reliability guard against catastrophic local optima)."""
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=60)
    X = x.reshape(-1, 1)
    single = topica.STM(num_topics=2, seed=1).fit(docs, prevalence=X, iters=30)
    multi = topica.STM(num_topics=2, seed=1).fit(docs, prevalence=X, iters=30, restarts=4)
    assert isinstance(multi, topica.STM)
    assert np.asarray(multi.topic_word).shape == np.asarray(single.topic_word).shape
    # restart 0 reuses the single fit's seed+init, so best-of-N is never worse.
    assert float(multi.bound) >= float(single.bound) - 1e-6


def test_stm_restarts_rejects_beta_init():
    """A fixed beta_init makes every restart identical, so it is rejected."""
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=40)
    X = x.reshape(-1, 1)
    b = topica.STM(num_topics=2, seed=1).fit(docs, prevalence=X, iters=0).topic_word
    with pytest.raises(ValueError, match="restarts"):
        topica.STM(num_topics=2, seed=1).fit(docs, prevalence=X, restarts=3, beta_init=b)


def test_beta_from_reference_aligns_normalizes_and_seeds():
    """beta_from_reference maps an external topic-word matrix onto topica's
    vocabulary (handling permutation + transpose) for beta_init replication (#871)."""
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=40)
    corpus = topica.Corpus.from_documents(docs)
    vocab = list(corpus.vocabulary)
    V = len(vocab)
    ref_vocab = vocab[::-1]  # full permutation: every target word present
    beta = rng.random((2, V))
    beta /= beta.sum(axis=1, keepdims=True)
    out = topica.stm.beta_from_reference(beta, ref_vocab, corpus)
    assert out.shape == (2, V)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    # a target word's column equals the reference column for that same word.
    i = ref_vocab.index(vocab[0])
    np.testing.assert_allclose(out[:, 0], beta[:, i], atol=1e-12)
    # transposed (V x K) input is detected and fixed.
    np.testing.assert_allclose(
        topica.stm.beta_from_reference(beta.T, ref_vocab, corpus), out, atol=1e-12
    )
    # usable directly as a warm start.
    m = topica.STM(num_topics=2, seed=1).fit(docs, prevalence=x.reshape(-1, 1),
                                             iters=5, beta_init=out)
    assert np.asarray(m.topic_word).shape == (2, V)


def test_beta_from_reference_warns_on_low_overlap():
    import topica
    rng = np.random.default_rng(0)
    docs, _ = _make_synthetic_corpus(rng, n_per_class=30)
    corpus = topica.Corpus.from_documents(docs)
    vocab = list(corpus.vocabulary)
    beta = rng.random((2, 2))
    with pytest.warns(UserWarning, match="vocabulary"):
        topica.stm.beta_from_reference(beta, vocab[:2], corpus)


def test_content_kappa_reconstructs_content_beta():
    """SAGE kappa decomposition (issue #237): m + kappa_topic + kappa_cov +
    kappa_interaction, softmax over words, reproduces the per-group topic-word."""
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=60)
    groups = ["a" if xi == 1 else "b" for xi in x]
    m = topica.STM(num_topics=2, seed=1)
    m.fit(docs, content=groups, iters=40)
    ck = m.content_kappa
    K, V, G = m.num_topics, len(m.vocabulary), len(m.groups)
    assert ck["m"].shape == (V,)
    assert ck["kappa_topic"].shape == (K, V)
    assert ck["kappa_cov"].shape == (G, V)
    assert ck["kappa_interaction"].shape == (K, G, V)
    tw = m.topic_word_by_group  # (K, G, V)
    for k in range(K):
        for g in range(G):
            eta = (ck["m"] + ck["kappa_topic"][k] + ck["kappa_cov"][g]
                   + ck["kappa_interaction"][k, g])
            beta = np.exp(eta - eta.max()); beta /= beta.sum()
            np.testing.assert_allclose(beta, tw[k, g], atol=1e-9)


def test_content_kappa_requires_content():
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=40)
    m = topica.STM(num_topics=2, seed=1)
    m.fit(docs, prevalence=x.reshape(-1, 1), iters=20)  # prevalence, no content
    with pytest.raises(RuntimeError, match="without content"):
        _ = m.content_kappa


def test_content_kappa_save_load(tmp_path):
    import topica
    rng = np.random.default_rng(0)
    docs, x = _make_synthetic_corpus(rng, n_per_class=50)
    groups = ["a" if xi == 1 else "b" for xi in x]
    m = topica.STM(num_topics=2, seed=1)
    m.fit(docs, content=groups, iters=30)
    p = str(tmp_path / "m.tt")
    m.save(p)
    loaded = topica.STM.load(p)
    for key in ("m", "kappa_topic", "kappa_cov", "kappa_interaction"):
        np.testing.assert_allclose(m.content_kappa[key], loaded.content_kappa[key])
