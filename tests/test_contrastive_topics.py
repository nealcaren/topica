"""Tests for topica.interpret.contrastive_topics — the topic-conditional Fighting Words
contrast that ranks which topics most separate two groups and surfaces the
within-topic words that shift between them.
"""

import math

import numpy as np
import pytest

import topica


def _two_group_corpus(seed=0):
    """A corpus with two topics, each worded differently by two groups.

    Both groups use both topics about equally (so ``usage_diff`` is near zero),
    but within each topic group A and group B use disjoint flavor words. This is
    the case the within-topic ``vocab_shift`` is meant to catch.
    """
    rng = np.random.default_rng(seed)

    def doc(shared, flavor):
        return list(rng.choice(shared, 8)) + list(rng.choice(flavor, 4))

    texts, groups = [], []
    for _ in range(150):
        texts.append(doc(["tax", "budget", "economy"], ["growth", "jobs"]))
        groups.append("A")
        texts.append(doc(["tax", "budget", "economy"], ["inequality", "fairness"]))
        groups.append("B")
        texts.append(doc(["troops", "war", "policy"], ["surge", "strength"]))
        groups.append("A")
        texts.append(doc(["troops", "war", "policy"], ["withdraw", "peace"]))
        groups.append("B")
    return texts, groups


def _fit(texts, k=2, seed=0, iters=300):
    m = topica.models.LDA(num_topics=k, seed=seed)
    m.fit(texts, iters=iters)
    return m


class TestContrastiveTopics:
    def test_recovers_within_topic_word_split(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2, n_words=6)

        assert len(rows) == model.num_topics
        a_flavor = {"growth", "jobs", "surge", "strength"}
        b_flavor = {"inequality", "fairness", "withdraw", "peace"}
        # Every topic should put the A-flavor words on side A and B on side B.
        for r in rows:
            a_words = {w for w, _ in r["a_words"]}
            b_words = {w for w, _ in r["b_words"]}
            assert a_words & a_flavor, r
            assert b_words & b_flavor, r
            # No flavor word lands on the wrong side.
            assert not (a_words & b_flavor)
            assert not (b_words & a_flavor)

    def test_signed_z_scores(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        for r in rows:
            assert all(z > 0 for _, z in r["a_words"])
            assert all(z < 0 for _, z in r["b_words"])

    def test_vocab_shift_positive_when_groups_diverge(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        assert all(r["vocab_shift"] > 0 for r in rows)

    def test_group_order_flips_sign(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        ab = {r["topic"]: r for r in topica.interpret.contrastive_topics(
            model, texts, groups, group_order=("A", "B"), min_count=2)}
        ba = {r["topic"]: r for r in topica.interpret.contrastive_topics(
            model, texts, groups, group_order=("B", "A"), min_count=2)}
        for t, r in ab.items():
            assert r["a_label"] == "A" and r["b_label"] == "B"
            assert math.isclose(r["usage_diff"], -ba[t]["usage_diff"], abs_tol=1e-9)
            # A's distinctive words become B's distinctive words when flipped.
            assert {w for w, _ in r["a_words"]} == {w for w, _ in ba[t]["b_words"]}

    def test_default_order_is_sorted_labels(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        assert rows[0]["a_label"] == "A" and rows[0]["b_label"] == "B"

    def test_sorted_by_absolute_usage_diff(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts, k=4)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        mags = [abs(r["usage_diff"]) for r in rows]
        assert mags == sorted(mags, reverse=True)

    def test_usage_diff_picks_up_prevalence_gap(self):
        # Group A only ever produces the war topic; B only the econ topic.
        rng = np.random.default_rng(1)
        texts, groups = [], []
        for _ in range(120):
            texts.append(list(rng.choice(["troops", "war", "policy", "surge"], 10)))
            groups.append("A")
            texts.append(list(rng.choice(["tax", "budget", "economy", "jobs"], 10)))
            groups.append("B")
        model = _fit(texts, k=2)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        # The most contrastive topic should have a large, non-trivial usage gap.
        assert abs(rows[0]["usage_diff"]) > 0.3
        leans = {r["leans"] for r in rows}
        assert leans == {"A", "B"}

    def test_works_on_dmr_covariate_model(self):
        # Any model exposing doc_topic + vocabulary should work, not just LDA.
        texts, groups = _two_group_corpus()
        x = np.array([[1.0] if g == "A" else [0.0] for g in groups])
        model = topica.models.DMR(num_topics=2, seed=0)
        model.fit(texts, x, iters=300)
        rows = topica.interpret.contrastive_topics(model, texts, groups, min_count=2)
        assert len(rows) == 2
        assert all("a_words" in r and "b_words" in r for r in rows)


class TestContrastiveTopicsValidation:
    def test_rejects_wrong_text_count(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        with pytest.raises(ValueError, match="documents"):
            topica.interpret.contrastive_topics(model, texts[:-1], groups, min_count=2)

    def test_rejects_wrong_group_count(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        with pytest.raises(ValueError, match="labels"):
            topica.interpret.contrastive_topics(model, texts, groups[:-1], min_count=2)

    def test_rejects_non_binary_groups(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        three = list(groups)
        three[0] = "C"
        with pytest.raises(ValueError, match="two distinct groups"):
            topica.interpret.contrastive_topics(model, texts, three, min_count=2)

    def test_rejects_empty_group(self):
        texts, groups = _two_group_corpus()
        model = _fit(texts)
        with pytest.raises(ValueError, match="both groups"):
            topica.interpret.contrastive_topics(
                model, texts, groups, group_order=("A", "Z"), min_count=2)
