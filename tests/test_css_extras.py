"""CSS-workflow additions: Fighting Words, the metadata-preserving document
splitter, the highlighted close-reading export, the coherence-exclusivity
frontier, bootstrap topic stability, and clustered / GLM-link estimate_effect.
"""

import numpy as np
import pytest

import topica
from topica import stm


# ---------------------------------------------------------------------------
# Fighting Words (Monroe-Colaresi-Quinn)
# ---------------------------------------------------------------------------

class TestFightingWords:
    def _corpora(self):
        a = [["tax", "cut", "growth", "jobs", "market"]] * 30
        b = [["climate", "carbon", "green", "planet", "energy"]] * 30
        return a, b

    def test_distinguishes_groups(self):
        a, b = self._corpora()
        scored = topica.fighting_words(a, b, prior=0.05)
        words = [w for w, _ in scored]
        assert words[0] in {"tax", "cut", "growth", "jobs", "market"}     # top -> A
        assert words[-1] in {"climate", "carbon", "green", "planet", "energy"}  # bottom -> B
        # z-scores are sorted descending.
        z = [s for _, s in scored]
        assert z == sorted(z, reverse=True)

    def test_shared_words_are_neutral(self):
        a = [["tax", "the", "of"]] * 40
        b = [["green", "the", "of"]] * 40
        d = dict(topica.fighting_words(a, b, prior=0.05))
        assert abs(d["the"]) < abs(d["tax"])     # shared word near zero

    def test_top_helper_and_informative(self):
        a, b = self._corpora()
        top = topica.top_fighting_words(a, b, n=3)
        assert set(top) == {"a", "b"} and len(top["a"]) == 3
        # Informative prior runs and returns the full vocabulary.
        scored = topica.fighting_words(a, b, informative=True)
        assert len(scored) == 10

    def test_min_count_filter(self):
        a = [["common", "common", "rare_a"]]
        b = [["common", "common", "rare_b"]]
        words = [w for w, _ in topica.fighting_words(a, b, min_count=2)]
        assert words == ["common"]


# ---------------------------------------------------------------------------
# Document splitter
# ---------------------------------------------------------------------------

class TestSplitDocuments:
    def test_propagates_metadata(self):
        long = "word " * 500
        chunks, meta = topica.split_documents([long.strip()], [{"year": 1920, "id": "a"}],
                                          max_words=100, min_words=20)
        assert len(chunks) == len(meta) > 1
        for j, row in enumerate(meta):
            assert row["year"] == 1920 and row["id"] == "a"
            assert row["parent"] == 0 and row["chunk"] == j

    def test_token_input_returns_tokens(self):
        doc = ["w"] * 250
        chunks, meta = topica.split_documents([doc], max_words=100, min_words=20)
        assert all(isinstance(c, list) for c in chunks)
        assert sum(len(c) for c in chunks) == 250          # no text lost

    def test_short_doc_one_chunk(self):
        chunks, meta = topica.split_documents(["just a short sentence."], max_words=100)
        assert len(chunks) == 1 and meta[0]["chunk"] == 0

    def test_runt_tail_merged(self):
        # 230 words, max 100 -> would be 100/100/30; the 30 merges into the prior.
        chunks, _ = topica.split_documents([("w " * 230).strip()], max_words=100,
                                       min_words=50, sentence_aware=False)
        assert all(len(c.split()) >= 50 for c in chunks)

    def test_metadata_length_mismatch(self):
        with pytest.raises(ValueError):
            topica.split_documents(["a", "b"], [{"x": 1}])


# ---------------------------------------------------------------------------
# Model-based helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_topic():
    docs = [["mob", "lynch", "south", "murder"]] * 40 + [["school", "child", "teach", "college"]] * 40
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=400)
    return m, docs


class TestFindThoughtsHtml:
    def test_html_highlights_keywords(self, two_topic):
        m, docs = two_topic
        texts = [" ".join(d) for d in docs]
        html = topica.find_thoughts_html(m, texts, n_docs=2, n_words=4)
        assert "<mark>" in html and "Topic 0" in html and "Topic 1" in html

    def test_markdown_mode(self, two_topic):
        m, docs = two_topic
        texts = [" ".join(d) for d in docs]
        md = topica.find_thoughts_html(m, texts, n_docs=1, markdown=True)
        assert "**" in md and "### Topic" in md

    def test_alignment_checked(self, two_topic):
        m, docs = two_topic
        with pytest.raises(ValueError):
            topica.find_thoughts_html(m, ["only one text"])


class TestQualityFrontier:
    def test_returns_per_topic_arrays(self, two_topic):
        m, _ = two_topic
        qf = topica.quality_frontier(m, n=5)
        for key in ("topic", "coherence", "exclusivity", "prevalence"):
            assert qf[key].shape == (2,)
        np.testing.assert_allclose(qf["prevalence"].sum(), 1.0, atol=1e-6)


class TestBootstrapStability:
    def test_stable_topics_score_high(self, two_topic):
        _, docs = two_topic
        res = topica.bootstrap_stability(docs, k=2, n_boot=4, iters=200, topn=4)
        assert res["stability"].shape == (2,)
        assert 0.0 <= res["mean"] <= 1.0
        # Two clean, well-separated topics should be highly reproducible.
        assert res["mean"] > 0.5

    def test_accepts_corpus_object(self, two_topic):
        # Issue #27: the docstring promises a Corpus is accepted (like its
        # siblings perplexity / prepare_pyldavis), so it must not raise.
        _, docs = two_topic
        corpus = topica.Corpus.from_documents(docs)
        res = topica.bootstrap_stability(corpus, k=2, n_boot=2, iters=80, topn=4)
        assert res["stability"].shape == (2,)
        assert 0.0 <= res["mean"] <= 1.0

    def test_stable_across_changing_vocabulary(self):
        # Each document carries its block's shared words plus a unique filler
        # token, so every bootstrap resample produces a *different* vocabulary.
        # Matching topics by word-index (the original bug) collapses to ~0 here;
        # matching by word string keeps clearly-separated blocks stable.
        rng = np.random.default_rng(0)
        blocks = [["alpha", "bravo", "charlie"], ["xray", "yankee", "zulu"]]
        docs = []
        for i in range(120):
            blk = blocks[i % 2]
            docs.append(blk + [blk[int(rng.integers(3))], f"uniq_{i}"])
        res = topica.bootstrap_stability(docs, k=2, n_boot=6, iters=200, topn=3)
        assert res["mean"] > 0.4          # two clean blocks must stay reproducible


# ---------------------------------------------------------------------------
# estimate_effect: clustered SEs and GLM links
# ---------------------------------------------------------------------------

class TestEstimateEffectExtras:
    def _data(self):
        rng = np.random.default_rng(0)
        D = 200
        groups = np.repeat(np.arange(20), 10)
        x = rng.normal(size=D)
        t0 = np.clip(0.3 + 0.05 * x + 0.1 * rng.normal(size=D), 0.01, 0.99)
        theta = np.column_stack([t0, 1 - t0])
        return theta, x, groups

    def test_cluster_keeps_coef_changes_se(self):
        theta, x, groups = self._data()
        base = stm.estimate_effect(theta, x, feature_names=["x"])[0].as_dict()
        clus = stm.estimate_effect(theta, x, feature_names=["x"], cluster=groups)[0].as_dict()
        assert np.isclose(base["x"]["coef"], clus["x"]["coef"])   # same point estimate
        assert clus["x"]["se"] != base["x"]["se"]                 # different uncertainty

    def test_identity_no_cluster_is_legacy_ols(self):
        theta, x, _ = self._data()
        eff = stm.estimate_effect(theta, x, feature_names=["x"])[0]
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, theta[:, 0], rcond=None)[0]
        np.testing.assert_allclose(eff.coef, beta, atol=1e-10)

    def test_logit_link_runs(self):
        theta, x, _ = self._data()
        eff = stm.estimate_effect(theta, x, feature_names=["x"], link="logit")[0].as_dict()
        assert np.isfinite(eff["x"]["coef"]) and eff["x"]["se"] > 0

    def test_cluster_composes_with_method_of_composition(self):
        theta, x, groups = self._data()
        draws = np.stack([theta + 0.001 * np.random.default_rng(s).normal(size=theta.shape)
                          for s in range(5)])
        eff = stm.estimate_effect(draws, x, feature_names=["x"], cluster=groups)
        assert len(eff) == 2 and np.isfinite(eff[0].as_dict()["x"]["se"])

    def test_bad_link_and_cluster(self):
        theta, x, groups = self._data()
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, link="probit")
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, cluster=groups[:5])

    def test_exposes_full_vcov(self):
        theta, x, _ = self._data()
        eff = stm.estimate_effect(theta, x, feature_names=["x"])[0]
        assert eff.vcov is not None and eff.vcov.shape == (2, 2)
        np.testing.assert_allclose(np.sqrt(np.diag(eff.vcov)), eff.se, atol=1e-12)


# ---------------------------------------------------------------------------
# estimate_effect: survey weights (WLS), and average marginal effects
# ---------------------------------------------------------------------------

class TestSurveyWeights:
    def _data(self):
        rng = np.random.default_rng(3)
        D = 250
        x = rng.normal(size=D)
        g = rng.integers(0, 2, D).astype(float)
        t0 = np.clip(0.3 + 0.05 * x + 0.1 * g + 0.05 * rng.normal(size=D), 0.01, 0.99)
        theta = np.column_stack([t0, 1 - t0])
        w = rng.uniform(0.5, 2.0, D)
        return theta, x, g, w

    def test_weighted_ols_matches_closed_form(self):
        theta, x, g, w = self._data()
        eff = stm.estimate_effect(theta, np.column_stack([x, g]),
                                  feature_names=["x", "g"], weights=w)[0]
        n = len(x)
        X = np.column_stack([np.ones(n), x, g])
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ theta[:, 0])
        resid = theta[:, 0] - X @ beta
        sigma2 = float((w * resid ** 2).sum()) / (n - 3)
        cov = sigma2 * np.linalg.inv(X.T @ W @ X)
        np.testing.assert_allclose(eff.coef, beta, atol=1e-10)
        np.testing.assert_allclose(eff.se, np.sqrt(np.diag(cov)), atol=1e-10)

    def test_unit_weights_equal_unweighted(self):
        theta, x, g, _ = self._data()
        base = stm.estimate_effect(theta, np.column_stack([x, g]), feature_names=["x", "g"])[0]
        wone = stm.estimate_effect(theta, np.column_stack([x, g]), feature_names=["x", "g"],
                                   weights=np.ones(len(x)))[0]
        np.testing.assert_allclose(base.coef, wone.coef, atol=1e-9)
        np.testing.assert_allclose(base.se, wone.se, atol=1e-9)

    def test_weights_compose_with_cluster_and_draws(self):
        theta, x, g, w = self._data()
        groups = np.repeat(np.arange(25), 10)
        draws = np.stack([theta + 0.001 * np.random.default_rng(s).normal(size=theta.shape)
                          for s in range(5)])
        eff = stm.estimate_effect(draws, np.column_stack([x, g]),
                                  feature_names=["x", "g"], weights=w, cluster=groups)
        assert len(eff) == 2 and np.isfinite(eff[0].se).all()

    def test_bad_weights(self):
        theta, x, _, _ = self._data()
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, weights=np.ones(5))
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, weights=-np.ones(len(x)))


class TestAverageMarginalEffects:
    def _data(self):
        import pandas as pd
        rng = np.random.default_rng(4)
        D = 300
        year = rng.normal(size=D)
        party = np.where(rng.integers(0, 2, D) == 1, "R", "D")
        t0 = np.clip(0.3 + 0.05 * year + 0.12 * (party == "R") + 0.05 * rng.normal(size=D),
                     0.01, 0.99)
        theta = np.column_stack([t0, 1 - t0])
        data = pd.DataFrame({"year": year, "party": party})
        return theta, data

    def test_continuous_ame_equals_linear_coef(self):
        theta, data = self._data()
        eff = stm.estimate_effect(theta, formula="~ year + party", data=data)[0]
        res = stm.average_marginal_effects(theta, "year", formula="~ year + party", data=data)
        row = res.to_frame().query("topic == 0 and term == 'year'").iloc[0]
        j = eff.feature_names.index("year")
        assert np.isclose(row.ame, eff.coef[j])
        assert np.isclose(row.se, eff.se[j])

    def test_factor_ame_equals_dummy_coef(self):
        theta, data = self._data()
        eff = stm.estimate_effect(theta, formula="~ year + party", data=data)[0]
        res = stm.average_marginal_effects(theta, "party", formula="~ year + party", data=data)
        row = res.to_frame().query("topic == 0 and term == 'partyR'").iloc[0]
        j = eff.feature_names.index("party[T.R]")
        assert np.isclose(row.ame, eff.coef[j])
        assert np.isclose(row.se, eff.se[j])

    def test_spline_ame_is_finite_and_aliased(self):
        theta, data = self._data()
        res = topica.ame(theta, "year", formula="~ spline(year, df=3) + party", data=data)
        df = res.to_frame()
        assert len(df) == 2 and np.isfinite(df["ame"]).all()
        assert topica.ame is topica.average_marginal_effects

    def test_requires_formula_and_known_covariate(self):
        theta, data = self._data()
        with pytest.raises(ValueError):
            stm.average_marginal_effects(theta, "year", formula=None, data=data)
        with pytest.raises(ValueError):
            stm.average_marginal_effects(theta, "nope", formula="~ year", data=data)
