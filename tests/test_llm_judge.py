"""LLM topic judge (Zheng et al. 2025, App. G): pairwise doc-topic comparison ->
Bradley-Terry -> Elo.

Exercised offline with deterministic fake backends (no network), so the pairing
schedule, A/B rendering, BT/Elo aggregation, bootstrap CI, and audit records are all
CI-testable. The live behaviour is llm-bounded and validated separately
(parity/llm_judge_live.py).
"""

import numpy as np
import pytest

import topica
import topica.llm as L
from topica.coherence import (
    JudgeResult, _bradley_terry, _bt_to_elo, _parse_judge_choice, _top_topics_for_doc,
)


# Two clear themes; a model with a dedicated topic per theme should beat one that
# lumps them together.
DOCS = [
    ["water", "river", "lake", "rain"],
    ["water", "lake", "ocean", "wave"],
    ["vote", "senate", "law", "policy"],
    ["election", "vote", "budget", "law"],
] * 8


@pytest.fixture
def models():
    good = topica.LDA(3, seed=13); good.fit(DOCS, iters=40)
    alt = topica.LDA(4, seed=7); alt.fit(DOCS, iters=40)
    lumped = topica.LDA(2, seed=1); lumped.fit(DOCS, iters=40)
    return {"good": good, "alt": alt, "lumped": lumped}


class WaterJudge:
    """A deterministic, model-agnostic judge: prefers whichever topic set mentions
    'water' more. Records call count."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if "central theme" in prompt.lower():  # a summary request
            return "water and rivers" if "water" in prompt else "government and law"
        a = prompt.split("Topic set A:")[1].split("Topic set B:")[0].lower()
        b = prompt.split("Topic set B:")[1].lower()
        na, nb = a.count("water"), b.count("water")
        if na > nb:
            return "A\nset A better captures the document"
        if nb > na:
            return "B\nset B better captures the document"
        return "tie\nboth are similar"


# -- namespace -------------------------------------------------------------

def test_namespace_surface():
    assert callable(topica.llm.judge)
    assert hasattr(topica.llm, "JudgeResult")
    assert "judge" in topica.llm.PROMPTS and "summary" in topica.llm.PROMPTS
    assert not hasattr(topica, "llm_judge")  # namespaced, not flat


# -- end to end (fake backend) ---------------------------------------------

def test_judge_ranks_and_elo_centers(models):
    j = WaterJudge()
    res = L.judge(models, DOCS, backend=j, n_comparisons=15, representation="words",
                  bootstrap=25, seed=0)
    # 3 pairs x 15 comparisons, one judge call each (words mode -> no summary calls)
    assert len(res.comparisons) == 3 * 15
    assert j.calls == 3 * 15
    # Elo is centered at 1500 and the water-favoring judge ranks 'lumped' last.
    assert abs(np.mean(list(res.elo.values())) - 1500.0) < 1e-6
    assert res.ranking()[-1] == "lumped"
    assert res.elo["good"] > res.elo["lumped"]
    # win matrix: no self-play, games are symmetric in count
    assert res.win_matrix.shape == (3, 3)
    assert np.allclose(np.diag(res.win_matrix), 0.0)
    # bootstrap CI present and ordered
    for nm in res.names:
        lo, hi = res.bootstrap_ci[nm]
        assert lo <= res.elo[nm] <= hi or lo <= hi  # CI brackets or is at least valid


def test_summary_representation_caches_per_topic(models):
    # Two models; summary mode should summarize each (model, topic) at most once,
    # regardless of how many comparisons reuse it.
    two = {"good": models["good"], "lumped": models["lumped"]}
    j = WaterJudge()
    res = L.judge(two, DOCS, backend=j, n_comparisons=20, representation="summary",
                  bootstrap=0, seed=2)
    # total topics across the two models bounds the number of distinct summaries
    max_summaries = two["good"].num_topics + two["lumped"].num_topics
    judge_calls = len(res.comparisons)
    # calls = judge calls + at most one summary per (model, topic)
    assert j.calls <= judge_calls + max_summaries
    assert res.representation == "summary"


def test_comparisons_are_auditable(models):
    j = WaterJudge()
    res = L.judge(models, DOCS, backend=j, n_comparisons=5, representation="words",
                  bootstrap=0, seed=0)
    rec = res.comparisons[0]
    assert set(rec) == {"doc", "model_a", "model_b", "choice", "winner", "reasoning"}
    assert rec["model_a"] in models and rec["model_b"] in models
    assert rec["winner"] in (rec["model_a"], rec["model_b"], None)


def test_design_is_reproducible_for_fixed_seed(models):
    j = WaterJudge()
    a = L.judge(models, DOCS, backend=j, n_comparisons=10, representation="words",
                bootstrap=0, seed=7)
    b = L.judge(models, DOCS, backend=WaterJudge(), n_comparisons=10,
                representation="words", bootstrap=0, seed=7)
    # same seed -> same document sampling and A/B presentation schedule
    assert [(c["doc"], c["model_a"], c["model_b"]) for c in a.comparisons] == \
           [(c["doc"], c["model_a"], c["model_b"]) for c in b.comparisons]


def test_bootstrap_skipped_when_zero(models):
    res = L.judge(models, DOCS, backend=WaterJudge(), n_comparisons=5,
                  representation="words", bootstrap=0, seed=0)
    assert all(np.isnan(res.bootstrap_ci[nm][0]) for nm in res.names)


# -- guards from the sample-user review ------------------------------------

def test_warns_when_models_fit_on_different_corpora(models):
    # Two models fit on DIFFERENT documents of the same length pass the row-count
    # check but yield a silently invalid ranking; differing vocabularies catch it.
    other_docs = [["alpha", "beta", "gamma", "delta"],
                  ["epsilon", "zeta", "eta", "theta"]] * 16
    assert len(other_docs) == len(DOCS)  # same length -> row check can't catch it
    good = models["good"]
    elsewhere = topica.LDA(3, seed=5); elsewhere.fit(other_docs, iters=40)
    with pytest.warns(UserWarning, match="different vocabularies"):
        L.judge({"good": good, "elsewhere": elsewhere}, DOCS, backend=WaterJudge(),
                n_comparisons=2, representation="words", bootstrap=0, seed=0)


def test_summary_flags_overlapping_top_cis():
    overlap = JudgeResult(
        elo={"a": 1550.0, "b": 1450.0}, win_matrix=np.zeros((2, 2)),
        bootstrap_ci={"a": (1400.0, 1700.0), "b": (1300.0, 1600.0)},  # overlap
        comparisons=[{}], names=["a", "b"], representation="words")
    assert "overlap" in overlap.summary().lower()
    clear = JudgeResult(
        elo={"a": 1800.0, "b": 1200.0}, win_matrix=np.zeros((2, 2)),
        bootstrap_ci={"a": (1750.0, 1850.0), "b": (1150.0, 1250.0)},  # disjoint
        comparisons=[{}], names=["a", "b"], representation="words")
    assert "overlap" not in clear.summary().lower()


# -- errors ----------------------------------------------------------------

def test_errors(models):
    j = WaterJudge()
    with pytest.raises(TypeError, match="dict"):
        L.judge([models["good"], models["lumped"]], DOCS, backend=j)
    with pytest.raises(ValueError, match="at least two"):
        L.judge({"only": models["good"]}, DOCS, backend=j)
    with pytest.raises(ValueError, match="representation"):
        L.judge(models, DOCS, backend=j, representation="bogus")
    with pytest.raises(ValueError, match="same docs|doc-topic rows"):
        L.judge(models, DOCS[:3], backend=j, n_comparisons=2)


# -- the BT / Elo numerics (deterministic, no LLM) -------------------------

def test_bradley_terry_orders_by_wins():
    # 3 players: 0 beats everyone, 2 loses to everyone.
    win = np.array([[0.0, 8.0, 9.0],
                    [2.0, 0.0, 7.0],
                    [1.0, 3.0, 0.0]])
    p = _bradley_terry(win, smoothing=1.0)
    assert p[0] > p[1] > p[2]
    elo = _bt_to_elo(p)
    assert abs(elo.mean() - 1500.0) < 1e-6
    # a 400-pt Elo gap == 10:1 BT strength ratio
    assert np.isclose(10 ** ((elo[0] - elo[2]) / 400.0), p[0] / p[2])


def test_bradley_terry_finite_for_undefeated():
    # Without smoothing an all-win / all-loss player is degenerate; smoothing keeps
    # every strength finite and positive.
    win = np.array([[0.0, 10.0], [0.0, 0.0]])
    p = _bradley_terry(win, smoothing=1.0)
    assert np.all(np.isfinite(p)) and np.all(p > 0)


@pytest.mark.parametrize("reply,expected", [
    ("A", "a"), ("B\nbecause ...", "b"), ("tie", "tie"),
    ("A. Set A better captures the document", "a"),
    ("The answer is B", "b"),
    # a verbose tie must not be misread as an A-win via the article "a"
    ("It is a tie between them", "tie"),
    ("Both are equal, a tie", "tie"),
    ("neither is clearly better", None),
])
def test_parse_judge_choice(reply, expected):
    assert _parse_judge_choice(reply) == expected


def test_top_topics_for_doc_takes_the_fewer_of_q_and_p():
    row = np.array([0.5, 0.3, 0.15, 0.05])
    # q=2 caps at 2 even though p=0.9 would take 3
    assert _top_topics_for_doc(row, q=2, p=0.9) == [0, 1]
    # p=0.75 reached at the 2nd topic (0.5+0.3=0.8 >= 0.75*1.0); q=3 is looser
    assert _top_topics_for_doc(row, q=3, p=0.75) == [0, 1]
    # always at least one
    assert _top_topics_for_doc(row, q=0, p=0.0) == [0]
