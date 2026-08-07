"""Effect robustness across K and across seeds (``topica.effects_across_k`` /
``effects_across_seeds``, issue #644).

The verdict logic is tested directly on constructed rows (deterministic, no fitting),
and the fit/align/estimate plumbing is tested end to end on a planted corpus.
"""

import numpy as np
import pytest

import topica
from topica.robustness import RobustnessResult


def _rows(spec, varied="k"):
    """Build rows from ``{topic: [(setting, coef|None), ...]}``; None = unmatched."""
    out = []
    for topic, entries in spec.items():
        for setting, coef in entries:
            if coef is None:
                out.append({"reference_topic": topic, varied: setting,
                            "matched_topic": None, "matched": False,
                            "similarity": None, "coef": None, "se": None,
                            "ci_low": None, "ci_high": None, "pvalue": None,
                            "sign": None, "significant": None})
            else:
                out.append({"reference_topic": topic, varied: setting,
                            "matched_topic": topic, "matched": True,
                            "similarity": 0.9, "coef": coef, "se": 0.01,
                            "ci_low": coef - 0.02, "ci_high": coef + 0.02,
                            "pvalue": 0.01, "sign": int(np.sign(coef)),
                            "significant": True})
    return out


def _result(spec, varied="k"):
    return RobustnessResult(_rows(spec, varied), feature="x", varied=varied,
                            reference=list(spec.values())[0][0][0], ci=0.95)


# --- verdict logic ------------------------------------------------------------

class TestVerdicts:
    def test_same_sign_everywhere_is_stable(self):
        r = _result({0: [(4, 0.3), (5, 0.28), (6, 0.31)]})
        assert r.stable == [0] and not r.flipped and not r.unmatched

    def test_sign_change_is_flipped(self):
        # The honest red flag: the effect points the other way at another K.
        r = _result({0: [(4, -0.01), (5, 0.03)]})
        assert r.flipped == [0] and not r.stable

    def test_unmatched_anywhere_is_undetermined_not_stable(self):
        # A topic that vanished at one K is NOT robust-confirmed; it is unknown.
        r = _result({0: [(4, 0.3), (5, None)]})
        assert r.unmatched == [0]
        assert 0 not in r.stable and 0 not in r.flipped

    def test_verdicts_partition_every_topic_exactly_once(self):
        r = _result({0: [(4, 0.3), (5, 0.3)],
                     1: [(4, 0.2), (5, -0.2)],
                     2: [(4, 0.1), (5, None)]})
        assert r.verdicts() == {0: "stable", 1: "flipped", 2: "unmatched"}
        buckets = [set(r.stable), set(r.flipped), set(r.unmatched)]
        assert set.union(*buckets) == set(r.topics)
        assert sum(len(b) for b in buckets) == len(r.topics)  # disjoint

    def test_summary_names_settings_and_disclaims_a_test(self):
        r = _result({0: [(4, 0.3), (5, 0.3)]})
        s = r.summary()
        assert "[4, 5]" in s and "'x'" in s
        assert "not a test" in s  # descriptive, never presented as significance

    def test_is_a_list_of_rows(self):
        r = _result({0: [(4, 0.3), (5, 0.3)]})
        assert isinstance(r, list) and len(r) == 2
        assert set(r[0]) >= {"reference_topic", "k", "coef", "matched", "sign"}


# --- end-to-end plumbing ------------------------------------------------------

def _planted(n=300, seed=0):
    """Four themes; group B talks markedly more about theme 0."""
    rng = np.random.default_rng(seed)
    blocks = [[f"t{t}_w{i}" for i in range(12)] for t in range(4)]
    docs, grp = [], []
    for d in range(n):
        g = d % 2
        w = [0.55, 0.15, 0.15, 0.15] if g else [0.25] * 4
        docs.append([rng.choice(blocks[rng.choice(4, p=w)]) for _ in range(14)])
        grp.append(g)
    return docs, np.asarray(grp, dtype=float)[:, None], ["groupB"]


class TestAcrossK:
    def test_reports_every_reference_topic_at_every_k(self):
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50)
        assert r.varied == "k" and r.reference == 4
        # Complete coverage: one row per (reference topic, K) — nothing dropped.
        assert len(r) == 4 * 2
        assert {row["k"] for row in r} == {4, 5}
        assert all(row["reference_topic"] in range(4) for row in r)

    def test_unmatched_reference_topics_are_reported_not_dropped(self):
        # Reference K=6 against a K=3 fit: three topics can have no counterpart.
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [6, 3], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50)
        assert len(r) == 6 * 2  # still one row per (topic, K)
        unmatched = [row for row in r if not row["matched"]]
        assert unmatched, "a K=3 fit cannot match all six reference topics"
        for row in unmatched:
            assert row["coef"] is None and row["matched_topic"] is None
            assert r.verdicts()[row["reference_topic"]] == "unmatched"

    def test_matched_rows_carry_a_usable_effect(self):
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50)
        for row in (x for x in r if x["matched"]):
            assert row["se"] >= 0.0
            assert row["ci_low"] <= row["coef"] <= row["ci_high"]
            assert row["sign"] in (-1, 0, 1)
            assert isinstance(row["significant"], bool)
            assert 0.0 <= row["similarity"] <= 1.0 + 1e-9

    def test_reference_fit_matches_itself_exactly(self):
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50)
        ref = [row for row in r if row["k"] == r.reference]
        assert all(row["matched"] and row["matched_topic"] == row["reference_topic"]
                   for row in ref)

    def test_planted_effect_is_stable_across_k(self):
        # A strongly planted group difference should not flip sign at a nearby K.
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=60)
        assert not r.flipped


class TestAcrossSeeds:
    def test_varies_seed_and_aligns_to_a_reference(self):
        docs, X, names = _planted()
        r = topica.effects_across_seeds(docs, [1, 2], num_topics=4, feature="groupB",
                                        prevalence=X, feature_names=names, iters=50)
        assert r.varied == "seed" and r.reference == 1
        assert {row["seed"] for row in r} == {1, 2}
        assert len(r) == 4 * 2

    def test_accepts_already_fitted_models(self):
        docs, X, names = _planted()
        fits = [topica.STM(num_topics=4, seed=s).fit(docs, X, iters=50) for s in (1, 2)]
        r = topica.effects_across_seeds(docs, [1, 2], num_topics=4, feature="groupB",
                                        prevalence=X, feature_names=names, fits=fits)
        assert len(r) == 4 * 2 and not r.unmatched


class TestApi:
    def test_exported(self):
        for name in ("effects_across_k", "effects_across_seeds", "RobustnessResult"):
            assert hasattr(topica, name) and name in topica.__all__

    def test_empty_settings_raises(self):
        docs, X, names = _planted(n=40)
        with pytest.raises(ValueError, match="empty"):
            topica.effects_across_k(docs, [], feature="groupB", prevalence=X,
                                    feature_names=names)

    def test_missing_design_raises(self):
        docs, _, names = _planted(n=40)
        with pytest.raises(ValueError, match="covariate design"):
            topica.effects_across_k(docs, [3, 4], feature="groupB", feature_names=names)

    def test_reference_outside_the_grid_raises(self):
        docs, X, names = _planted(n=40)
        with pytest.raises(ValueError, match="not among"):
            topica.effects_across_k(docs, [3, 4], feature="groupB", prevalence=X,
                                    feature_names=names, reference=9, iters=20)

    def test_fits_length_mismatch_raises(self):
        docs, X, names = _planted(n=40)
        fits = [topica.STM(num_topics=3, seed=1).fit(docs, X, iters=20)]
        with pytest.raises(ValueError, match="one fitted model per setting"):
            topica.effects_across_k(docs, [3, 4], feature="groupB", prevalence=X,
                                    feature_names=names, fits=fits)

    def test_unknown_model_raises(self):
        docs, X, names = _planted(n=40)
        with pytest.raises(ValueError, match="model must be"):
            topica.effects_across_k(docs, [3], feature="groupB", prevalence=X,
                                    feature_names=names, model="bogus")

    def test_callable_model_factory(self):
        docs, X, names = _planted(n=120)

        def factory(num_topics, seed):
            return topica.STM(num_topics=num_topics, seed=seed).fit(docs, X, iters=30)

        r = topica.effects_across_k(docs, [3, 4], feature="groupB", prevalence=X,
                                    feature_names=names, model=factory)
        assert len(r) == 3 * 2


# --- review follow-ups (#671): honest coverage + clearer STM error ------------
class TestReviewFollowups:
    def test_min_similarity_gate_marks_weak_matches_unmatched(self):
        """The coverage promise: a below-threshold Hungarian pairing must not be
        reported as a confident match. With an impossibly strict threshold every
        non-reference fit fails to match, so no topic can be called stable."""
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50, min_similarity=1.0)
        # Reference fit still self-matches at similarity 1.0; the other fit does not.
        assert r.unmatched == r.topics
        assert r.stable == [] and r.flipped == []
        non_ref = [row for row in r if row["k"] != r.reference]
        assert non_ref and all(not row["matched"] and row["coef"] is None
                               for row in non_ref)

    def test_loose_gate_keeps_every_pairing(self):
        """min_similarity=0.0 restores force-matching (every reference topic keeps a
        counterpart), so the gate is opt-out, not mandatory."""
        docs, X, names = _planted()
        r = topica.effects_across_k(docs, [4, 5], feature="groupB", prevalence=X,
                                    feature_names=names, iters=50, min_similarity=0.0)
        assert not r.unmatched  # K equal + no gate => all matched

    def test_stm_x_only_raises_clear_error(self):
        docs, X, names = _planted()
        with pytest.raises(ValueError, match="prevalence"):
            topica.effects_across_k(docs, [3, 4], feature="groupB", X=X,
                                    feature_names=names, model="stm", iters=50)
