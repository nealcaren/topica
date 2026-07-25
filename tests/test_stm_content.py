"""Tests for STM content covariates (topic_word_by_group, groups, word_contrast).

These tests exercise the FULL STM: the existing tests in test_stm_model.py cover
prevalence covariates; this file covers:
  - content-only STM (no prevalence)
  - combined prevalence + content STM
  - group wording recovery (bilingual corpus)
  - validation / error paths
  - unfitted guards for content properties
  - determinism of content-fit models
  - integer group labels
  - content_names ordering

Do NOT break or duplicate the prevalence tests in tests/test_stm_model.py.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from topica import STM, stm

# ---------------------------------------------------------------------------
# Bilingual corpus vocabulary
# ---------------------------------------------------------------------------
# Two topics (weather / food) × two language groups (en / de).
# Each group uses a COMPLETELY DIFFERENT vocabulary for the same topics —
# no shared words between English and German.
# This gives a strong, clean signal for content-covariate recovery.

_EN_WEATHER = ["rain", "sun", "cloud", "wind", "storm"]
_DE_WEATHER = ["regen", "sonne", "wolke", "sturm", "nebel"]
_EN_FOOD    = ["bread", "cheese", "wine", "apple", "meat"]
_DE_FOOD    = ["brot",  "kaese",  "wein",  "apfel", "fleisch"]

_EN_VOCAB = set(_EN_WEATHER) | set(_EN_FOOD)
_DE_VOCAB = set(_DE_WEATHER) | set(_DE_FOOD)


def _make_bilingual_corpus(n_per_cell: int = 50, seed: int = 42):
    """Return (docs, groups) — 4*n_per_cell docs, 2 groups, 2 topics.

    Structure:
      n_per_cell EN weather-heavy docs  (10 weather + 2 food tokens)
      n_per_cell EN food-heavy docs     (10 food + 2 weather tokens)
      n_per_cell DE weather-heavy docs  (10 weather + 2 food tokens)
      n_per_cell DE food-heavy docs     (10 food + 2 weather tokens)
    """
    rng = np.random.default_rng(seed)
    docs, groups = [], []

    for _ in range(n_per_cell):
        docs.append(rng.choice(_EN_WEATHER, size=10).tolist()
                    + rng.choice(_EN_FOOD, size=2).tolist())
        groups.append("en")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_EN_FOOD, size=10).tolist()
                    + rng.choice(_EN_WEATHER, size=2).tolist())
        groups.append("en")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_DE_WEATHER, size=10).tolist()
                    + rng.choice(_DE_FOOD, size=2).tolist())
        groups.append("de")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_DE_FOOD, size=10).tolist()
                    + rng.choice(_DE_WEATHER, size=2).tolist())
        groups.append("de")

    return docs, groups


# ---------------------------------------------------------------------------
# Module-scoped fixture: one content-only bilingual fit, reused by all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bilingual_content_model():
    """STM(2, seed=1) fitted content-only on the bilingual corpus (iters=60)."""
    docs, groups = _make_bilingual_corpus(n_per_cell=50, seed=42)
    m = STM(num_topics=2, seed=1)
    m.fit(docs, content=groups, iters=60)
    return m


# ---------------------------------------------------------------------------
# Unfitted guards for content-specific properties
# ---------------------------------------------------------------------------

class TestSTMContentUnfittedGuards:
    """topic_word_by_group, groups, and word_contrast must raise RuntimeError before fit."""

    def test_topic_word_by_group_raises_before_fit(self):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_word_by_group

    def test_groups_raises_before_fit(self):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.groups

    def test_word_contrast_raises_before_fit(self):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.word_contrast(0, "a", "b")


# ---------------------------------------------------------------------------
# Content-only STM: shapes and basic invariants
# ---------------------------------------------------------------------------

class TestSTMContentOnlyShapes:
    """Output shapes for a content-only STM (no prevalence)."""

    def test_topic_word_shape(self, bilingual_content_model):
        m = bilingual_content_model
        # 2 topics, 20 unique words (10 EN + 10 DE)
        assert m.topic_word.shape == (2, 20)

    def test_topic_word_by_group_shape(self, bilingual_content_model):
        m = bilingual_content_model
        # (num_topics, num_groups, num_words) = (2, 2, 20)
        assert m.topic_word_by_group.shape == (2, 2, 20)

    def test_groups_sorted_de_en(self, bilingual_content_model):
        # Sorted alphabetically: 'de' < 'en'
        assert bilingual_content_model.groups == ["de", "en"]

    def test_doc_topic_shape(self, bilingual_content_model):
        # 4 * 50 = 200 docs
        assert bilingual_content_model.doc_topic.shape == (200, 2)

    def test_doc_topic_rows_sum_to_one(self, bilingual_content_model):
        npt.assert_allclose(
            bilingual_content_model.doc_topic.sum(axis=1),
            np.ones(200),
            atol=1e-5,
        )

    def test_topic_word_by_group_rows_are_valid_distributions(self, bilingual_content_model):
        """Each (topic, group) slice must sum to 1 and be non-negative."""
        tw = bilingual_content_model.topic_word_by_group
        npt.assert_allclose(tw.sum(axis=2), np.ones((2, 2)), atol=1e-5)
        assert (tw >= 0).all()

    def test_vocabulary_length(self, bilingual_content_model):
        m = bilingual_content_model
        assert len(m.vocabulary) == m.topic_word.shape[1]
        assert len(m.vocabulary) == m.topic_word_by_group.shape[2]

    def test_num_topics(self, bilingual_content_model):
        assert bilingual_content_model.num_topics == 2


# ---------------------------------------------------------------------------
# Content-only STM: prevalence_effects raises RuntimeError; feature_names may
# return [] (implementation detail) — we assert prevalence_effects raises.
# ---------------------------------------------------------------------------

class TestSTMContentOnlyPrevalenceGuards:
    """prevalence_effects must raise RuntimeError when fit without prevalence."""

    def test_prevalence_effects_raises_for_content_only(self, bilingual_content_model):
        with pytest.raises(RuntimeError):
            _ = bilingual_content_model.prevalence_effects


# ---------------------------------------------------------------------------
# GROUP WORDING: the core STM content-covariate property
# ---------------------------------------------------------------------------
# The key assertion: per-group slices concentrate on that group's language.
# EN group slice puts >90% of its mass on English words;
# DE group slice puts >90% of its mass on German words.
# This holds for BOTH topics (both topics use language-appropriate vocab).

class TestSTMContentGroupWording:
    """Verify that each group's word distribution uses language-appropriate vocabulary."""

    def _get_vocab_indices(self, vocabulary, word_set):
        """Return indices of words that appear in both the model vocabulary and word_set."""
        return [i for i, w in enumerate(vocabulary) if w in word_set]

    def test_en_group_mass_on_en_words(self, bilingual_content_model):
        """EN group slice must put > 90% of its mass on English words."""
        m = bilingual_content_model
        gi_en = m.groups.index("en")
        en_idx = self._get_vocab_indices(m.vocabulary, _EN_VOCAB)
        for t in range(2):
            en_mass = m.topic_word_by_group[t, gi_en, en_idx].sum()
            assert en_mass > 0.90, (
                f"Topic {t}: EN group mass on EN words is {en_mass:.4f} < 0.90"
            )

    def test_de_group_mass_on_de_words(self, bilingual_content_model):
        """DE group slice must put > 90% of its mass on German words."""
        m = bilingual_content_model
        gi_de = m.groups.index("de")
        de_idx = self._get_vocab_indices(m.vocabulary, _DE_VOCAB)
        for t in range(2):
            de_mass = m.topic_word_by_group[t, gi_de, de_idx].sum()
            assert de_mass > 0.90, (
                f"Topic {t}: DE group mass on DE words is {de_mass:.4f} < 0.90"
            )

    def test_en_group_slice_top_words_are_english(self, bilingual_content_model):
        """Top 5 words of the EN group slice for each topic are all English words."""
        m = bilingual_content_model
        gi_en = m.groups.index("en")
        for t in range(2):
            top5_idx = np.argsort(m.topic_word_by_group[t, gi_en, :])[-5:]
            top5_words = {m.vocabulary[i] for i in top5_idx}
            assert top5_words <= _EN_VOCAB, (
                f"Topic {t}: EN group top-5 words contain non-English words: "
                f"{top5_words - _EN_VOCAB}"
            )

    def test_de_group_slice_top_words_are_german(self, bilingual_content_model):
        """Top 5 words of the DE group slice for each topic are all German words."""
        m = bilingual_content_model
        gi_de = m.groups.index("de")
        for t in range(2):
            top5_idx = np.argsort(m.topic_word_by_group[t, gi_de, :])[-5:]
            top5_words = {m.vocabulary[i] for i in top5_idx}
            assert top5_words <= _DE_VOCAB, (
                f"Topic {t}: DE group top-5 words contain non-German words: "
                f"{top5_words - _DE_VOCAB}"
            )

    def test_word_contrast_en_vs_de_surfaces_english_words(self, bilingual_content_model):
        """word_contrast(t, 'en', 'de') returns English words with positive log-ratio."""
        m = bilingual_content_model
        for t in range(2):
            contrast = m.word_contrast(t, "en", "de", 10)
            contrast_words = {w for w, _ in contrast}
            # All top-contrast words should be English (they favour EN over DE)
            assert contrast_words <= _EN_VOCAB, (
                f"Topic {t}: en-vs-de contrast words contain non-English: "
                f"{contrast_words - _EN_VOCAB}"
            )
            # Top entries should have positive log-ratio
            top3 = [lr for _, lr in contrast[:3]]
            assert all(lr > 0 for lr in top3), (
                f"Topic {t}: expected positive log-ratios for en-vs-de, got {top3}"
            )

    def test_word_contrast_de_vs_en_surfaces_german_words(self, bilingual_content_model):
        """word_contrast(t, 'de', 'en') returns German words with positive log-ratio."""
        m = bilingual_content_model
        for t in range(2):
            contrast = m.word_contrast(t, "de", "en", 10)
            contrast_words = {w for w, _ in contrast}
            assert contrast_words <= _DE_VOCAB, (
                f"Topic {t}: de-vs-en contrast words contain non-German: "
                f"{contrast_words - _DE_VOCAB}"
            )
            top3 = [lr for _, lr in contrast[:3]]
            assert all(lr > 0 for lr in top3), (
                f"Topic {t}: expected positive log-ratios for de-vs-en, got {top3}"
            )

    def test_word_contrast_returns_word_float_tuples(self, bilingual_content_model):
        m = bilingual_content_model
        result = m.word_contrast(0, "en", "de", 5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for w, lr in result:
            assert isinstance(w, str)
            assert isinstance(lr, float)
            assert np.isfinite(lr)

    def test_word_contrast_log_ratios_are_finite(self, bilingual_content_model):
        m = bilingual_content_model
        for t in range(2):
            for a, b in [("en", "de"), ("de", "en")]:
                for _, lr in m.word_contrast(t, a, b, 10):
                    assert np.isfinite(lr)


# ---------------------------------------------------------------------------
# TOPIC SEPARATION under content (regression for the symmetry-collapse bug)
# ---------------------------------------------------------------------------
# Before kappa_t was seeded from the per-topic random beta, supplying a content
# covariate started every topic identical to the background (kappa all zero), a
# symmetric fixed point: theta stayed exactly uniform and all K topics came out
# identical. Group wording worked (kappa_c learned), which is why the existing
# tests above passed. These guard that the topics actually separate.

class TestSTMContentTopicSeparation:
    """A content covariate must not collapse the K topics into one."""

    def test_doc_topic_not_degenerate(self, bilingual_content_model):
        th = bilingual_content_model.doc_topic
        # Uniform-prior collapse showed up as zero variance across documents.
        assert th.std(axis=0).min() > 0.05, "theta collapsed to the uniform prior"
        assert (th.max(axis=1) > 0.6).mean() > 0.5, "no documents have a dominant topic"

    def test_group_marginal_topics_differ(self, bilingual_content_model):
        tw = bilingual_content_model.topic_word
        cos = float(tw[0] @ tw[1] / (np.linalg.norm(tw[0]) * np.linalg.norm(tw[1])))
        assert cos < 0.8, f"topics are nearly identical (cosine {cos:.3f})"

    def test_topics_split_weather_vs_food(self, bilingual_content_model):
        m = bilingual_content_model
        gi_en = m.groups.index("en")
        w_idx = [i for i, w in enumerate(m.vocabulary) if w in set(_EN_WEATHER)]
        f_idx = [i for i, w in enumerate(m.vocabulary) if w in set(_EN_FOOD)]
        leans = []
        for t in range(2):
            wm = m.topic_word_by_group[t, gi_en, w_idx].sum()
            fm = m.topic_word_by_group[t, gi_en, f_idx].sum()
            leans.append("weather" if wm > fm else "food")
        assert set(leans) == {"weather", "food"}, f"both topics lean the same way: {leans}"


# ---------------------------------------------------------------------------
# Combined prevalence + content STM
# ---------------------------------------------------------------------------

class TestSTMCombinedPrevalenceContent:
    """STM with both prevalence and content covariates."""

    @pytest.fixture(scope="class")
    def combined_model_and_x(self):
        """Fit STM with prevalence=X (N,1) and content=groups."""
        rng = np.random.default_rng(42)
        docs, groups = _make_bilingual_corpus(n_per_cell=50, seed=42)

        # Binary prevalence covariate: 1 for EN docs, 0 for DE docs
        x = np.array([1.0] * 100 + [0.0] * 100, dtype=np.float64).reshape(-1, 1)

        m = STM(num_topics=2, seed=1)
        m.fit(docs, x, prevalence_names=["x"], content=groups, iters=60)
        return m, x

    def test_topic_word_by_group_shape(self, combined_model_and_x):
        m, _ = combined_model_and_x
        assert m.topic_word_by_group.shape == (2, 2, 20)

    def test_prevalence_effects_shape(self, combined_model_and_x):
        m, _ = combined_model_and_x
        # F=1 prevalence covariate -> (F+1, num_topics-1) = (2, 1)
        assert m.prevalence_effects.shape == (2, 1)

    def test_groups_present(self, combined_model_and_x):
        m, _ = combined_model_and_x
        assert m.groups == ["de", "en"]

    def test_feature_names(self, combined_model_and_x):
        m, _ = combined_model_and_x
        assert m.feature_names == ["intercept", "x"]

    def test_topic_word_by_group_distributions_valid(self, combined_model_and_x):
        m, _ = combined_model_and_x
        tw = m.topic_word_by_group
        npt.assert_allclose(tw.sum(axis=2), np.ones((2, 2)), atol=1e-5)
        assert (tw >= 0).all()

    def test_estimate_effect_runs_and_returns_finite(self, combined_model_and_x):
        """estimate_effect works on a combined model and returns finite z-scores."""
        m, x = combined_model_and_x
        effects = stm.estimate_effect(m.doc_topic, x, feature_names=["x"])
        for eff in effects:
            d = eff.as_dict()
            assert np.isfinite(d["x"]["coef"]), "coef must be finite"
            assert np.isfinite(d["x"]["z"]),    "z must be finite"

    def test_doc_topic_rows_sum_to_one(self, combined_model_and_x):
        m, _ = combined_model_and_x
        npt.assert_allclose(m.doc_topic.sum(axis=1), np.ones(200), atol=1e-5)


# ---------------------------------------------------------------------------
# Integer group labels: groups == ["0", "1"]
# ---------------------------------------------------------------------------

class TestSTMContentIntegerGroups:
    def test_int_groups_stringified(self):
        """Integer group labels should appear as strings in .groups."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = [0] * 10 + [1] * 10
        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, iters=10)
        assert m.groups == ["0", "1"]

    def test_int_groups_fit_succeeds(self):
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = [0] * 10 + [1] * 10
        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, iters=10)
        assert m.topic_word_by_group.shape == (2, 2, 4)


# ---------------------------------------------------------------------------
# content_names fixes group order
# ---------------------------------------------------------------------------

class TestSTMContentNames:
    def test_content_names_en_de_fixes_order(self):
        """content_names=['en','de'] → groups == ['en','de']."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, content_names=["en", "de"], iters=10)
        assert m.groups == ["en", "de"]

    def test_content_names_de_en_fixes_order(self):
        """content_names=['de','en'] → groups == ['de','en']."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, content_names=["de", "en"], iters=10)
        assert m.groups == ["de", "en"]

    def test_default_order_is_sorted(self):
        """Without content_names, groups are sorted alphabetically."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, iters=10)
        assert m.groups == ["de", "en"]   # sorted


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestSTMContentValidation:
    """Validation errors from the fit() signature."""

    def setup_method(self):
        self.docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        self.groups = ["a"] * 10 + ["b"] * 10

    def test_neither_prevalence_nor_content_raises_value_error(self):
        m = STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="STM needs prevalence and/or content"):
            m.fit(self.docs, iters=5)

    def test_content_length_mismatch_raises_value_error(self):
        m = STM(num_topics=2, seed=1)
        with pytest.raises(ValueError):
            m.fit(self.docs, content=self.groups[:-1], iters=5)  # 19 groups, 20 docs

    def test_group_not_in_content_names_raises_value_error(self):
        m = STM(num_topics=2, seed=1)
        with pytest.raises(ValueError):
            m.fit(self.docs, content=self.groups, content_names=["x", "y"], iters=5)

    def test_word_contrast_bad_topic_raises_value_error(self):
        m = STM(num_topics=2, seed=1)
        m.fit(self.docs, content=self.groups, iters=10)
        with pytest.raises(ValueError):
            m.word_contrast(99, "a", "b")

    def test_word_contrast_unknown_group_raises_value_error(self):
        m = STM(num_topics=2, seed=1)
        m.fit(self.docs, content=self.groups, iters=10)
        with pytest.raises(ValueError):
            m.word_contrast(0, "z", "a")


# ---------------------------------------------------------------------------
# Determinism: same seed → identical topic_word_by_group
# ---------------------------------------------------------------------------

class TestSTMContentDeterminism:
    def test_same_seed_identical_topic_word_by_group(self):
        docs, groups = _make_bilingual_corpus(n_per_cell=20, seed=7)
        m1 = STM(num_topics=2, seed=42)
        m1.fit(docs, content=groups, iters=20)
        m2 = STM(num_topics=2, seed=42)
        m2.fit(docs, content=groups, iters=20)
        assert np.array_equal(m1.topic_word_by_group, m2.topic_word_by_group), (
            "Same seed must produce identical topic_word_by_group"
        )

    def test_same_seed_identical_doc_topic(self):
        docs, groups = _make_bilingual_corpus(n_per_cell=20, seed=7)
        m1 = STM(num_topics=2, seed=42)
        m1.fit(docs, content=groups, iters=20)
        m2 = STM(num_topics=2, seed=42)
        m2.fit(docs, content=groups, iters=20)
        assert np.array_equal(m1.doc_topic, m2.doc_topic), (
            "Same seed must produce identical doc_topic"
        )


# ---------------------------------------------------------------------------
# keep_eta_cov=False recompute for a CONTENT model (#442): _recompute_eta_cov
# must rebuild each document's ν against its own group's β and reproduce the
# stored (keep_eta_cov=True) ν. A fresh fit retains content_beta_estep (the
# group β active during the last E-step), so the match is exact, not just close.
# ---------------------------------------------------------------------------

class TestSTMContentRecomputeEtaCov:
    @staticmethod
    def _fit_pair(variational, iters):
        # Few iters: the final content M-step still moves the per-group β well
        # away from the E-step β, so a recompute against the post-M-step
        # content_beta (the pre-snapshot behavior) would be materially wrong.
        docs, groups = _make_bilingual_corpus(n_per_cell=30, seed=42)
        keep = STM(num_topics=2, seed=1, variational=variational)
        keep.fit(docs, content=groups, iters=iters, keep_eta_cov=True)
        drop = STM(num_topics=2, seed=1, variational=variational)
        drop.fit(docs, content=groups, iters=iters, keep_eta_cov=False)
        return keep, drop

    @pytest.mark.parametrize("variational", ["laplace", "diagonal"])
    def test_recompute_matches_stored_content(self, variational):
        keep, drop = self._fit_pair(variational, iters=4)
        # The recompute path must actually be a content model.
        assert keep.groups == ["de", "en"]
        stored = np.asarray(keep.eta_cov, dtype=np.float64)
        # keep_eta_cov=False makes the getter raise; recompute regenerates ν.
        with pytest.raises(RuntimeError, match="keep_eta_cov"):
            _ = drop.eta_cov
        recomp = np.asarray(drop._recompute_eta_cov(), dtype=np.float64)
        assert recomp.shape == stored.shape
        # Exact reproduction (f32 storage precision): the E-step group-β snapshot
        # rode through to the Python object, so ctm_hpb at the stored λ returns
        # the same ν. A group-averaged or post-M-step β would blow past this.
        npt.assert_allclose(recomp, stored, rtol=0, atol=1e-5)

    def test_refit_does_not_reuse_a_stale_content_snapshot(self):
        # #442: the E-step β snapshot was assigned only under `if !keep_eta_cov` and
        # never cleared, so a keep_eta_cov=False -> True refit on the SAME object
        # left the first fit's group β in place, and _recompute_eta_cov then used it
        # instead of the current fit's β. Swapping the group labels on the refit
        # exposes it: the stale snapshot points at the wrong groups.
        docs, groups = _make_bilingual_corpus(n_per_cell=30, seed=42)
        swapped = ["en" if g == "de" else "de" for g in groups]

        # Same instance: fit keep_eta_cov=False (original labels), then refit
        # keep_eta_cov=True with SWAPPED labels (which clears the snapshot).
        reused = STM(num_topics=2, seed=1)
        reused.fit(docs, content=groups, iters=4, keep_eta_cov=False)
        reused.fit(docs, content=swapped, iters=4, keep_eta_cov=True)

        # A fresh object with only the final (swapped, keep_eta_cov=True) fit is the
        # correct reference — same seed/data/labels, so an identical model state.
        fresh = STM(num_topics=2, seed=1)
        fresh.fit(docs, content=swapped, iters=4, keep_eta_cov=True)

        a = np.asarray(reused._recompute_eta_cov(), dtype=np.float64)
        b = np.asarray(fresh._recompute_eta_cov(), dtype=np.float64)
        # Identical under the fix (both use the current fit's β); under the bug the
        # reused instance recomputes against the stale first-fit β and diverges.
        npt.assert_allclose(a, b, rtol=0, atol=1e-5)


def test_l1_content_prior_is_pure_laplace_not_elastic_net():
    # Under content_prior="l1" the deviation blocks (kappa_cov / kappa_interaction)
    # must carry a *pure* Laplace prior, not a coupled L1+L2 elastic net (#532): the
    # surviving (nonzero) deviations are therefore not ridge-shrunk, so they run
    # larger in magnitude than the l2 fit's deviations at the same content_prior_var.
    # The baseline kappa_topic keeps its L2 (stays dense) under both priors.
    rng = np.random.default_rng(0)
    A = [f"a{i}" for i in range(8)]; B = [f"b{i}" for i in range(8)]; F = [f"f{i}" for i in range(8)]
    docs, grp = [], []
    for _ in range(120):
        docs.append(list(rng.choice(A + F, 20))); grp.append("A")
        docs.append(list(rng.choice(B + F, 20))); grp.append("B")

    def fit(prior):
        m = STM(num_topics=2, seed=1).fit(
            docs, content=grp, content_prior=prior, content_prior_var=0.5, iters=60
        )
        ck = m.content_kappa
        dev = np.concatenate([np.ravel(v) for k, v in ck.items() if k != "kappa_topic"])
        base = np.ravel(ck["kappa_topic"])
        return dev, base

    d_l1, base_l1 = fit("l1")
    d_l2, _ = fit("l2")

    # l1 sparsifies the deviations (exact zeros); l2 does not.
    assert (d_l1 == 0).sum() > 0, "l1 should produce exact zeros on the deviations"
    assert (d_l2 == 0).sum() == 0, "l2 should not force zeros"
    # baseline is L2-regularized (dense) under l1, not sparsified.
    assert (base_l1 == 0).sum() == 0, "kappa_topic must keep its L2 (stay dense) under l1"
    # pure Laplace: surviving deviations are not additionally ridge-shrunk, so their
    # mean magnitude exceeds the (fully ridge-shrunk) l2 deviations.
    mean_l1 = np.abs(d_l1[d_l1 != 0]).mean()
    mean_l2 = np.abs(d_l2).mean()
    assert mean_l1 > mean_l2, f"l1 survivors should exceed l2 magnitudes: {mean_l1:.3f} vs {mean_l2:.3f}"
