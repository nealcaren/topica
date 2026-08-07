"""LLM top-word refinement (Zheng et al. 2025, App. A.2 / F.1): drop up to m
out-of-place words, keep the top n.

Deterministic fake backends (no network), so the ranking, cap-at-m, majority vote,
and truncation are all CI-testable. The live behaviour is llm-bounded.
"""

import pytest

import topica
import topica.llm as L
from topica.coherence import llm_refine

# Two clean themes, each with planted out-of-place words at known ranks.
TOPICS = [
    ["water", "river", "lake", "ocean", "rain", "flood", "banana", "guitar"],
    ["vote", "senate", "law", "policy", "budget", "election", "telescope", "asphalt"],
]

INTRUDERS = {"banana", "guitar", "telescope", "asphalt"}


def _flagger(intruders=INTRUDERS):
    """A backend that flags any planted intruder present in the prompt."""
    def be(prompt):
        hits = [w for w in intruders if w in prompt.lower()]
        return ", ".join(hits) if hits else "none"
    return be


def test_namespace_surface():
    assert callable(topica.llm.refine)
    assert "refine" in topica.llm.PROMPTS
    assert "refine" in L.__all__


def test_drops_intruders_and_keeps_top_n():
    # n+m = 8 covers both planted intruders (at ranks 6 and 7).
    res = llm_refine(TOPICS, backend=_flagger(), n=6, m=2)
    assert [r["topic"] for r in res] == [0, 1]
    for r in res:
        assert len(r["words"]) == 6           # exactly n back (topic has n+m=8 words)
        assert len(r["dropped"]) <= 2
        # no dropped word survives in the cleaned list
        kept = {w.lower() for w in r["words"]}
        assert not (kept & {w.lower() for w in r["dropped"]})
    # the planted intruders within the top n+m are the ones dropped
    assert set(res[0]["dropped"]) == {"banana", "guitar"}
    assert "banana" not in res[0]["words"] and "guitar" not in res[0]["words"]


def test_none_flagged_returns_top_n_unchanged():
    res = llm_refine(TOPICS, backend=lambda p: "none", n=4, m=2)
    assert res[0]["dropped"] == []
    assert res[0]["words"] == TOPICS[0][:4]   # just the top n, in order


def test_fewer_than_m_flagged_truncates_to_n():
    # only one intruder flagged; n+m=8 words, drop 1 -> 7 remain -> truncate to n=6
    res = llm_refine(TOPICS, backend=_flagger({"banana"}), n=6, m=2)
    assert res[0]["dropped"] == ["banana"]
    assert len(res[0]["words"]) == 6
    assert "banana" not in res[0]["words"]


def test_more_than_m_flagged_drops_only_m_in_rank_order():
    # flag three words; m=2 caps the drop to the two highest-ranked flagged.
    topic = [["a", "bad1", "b", "bad2", "c", "bad3", "d", "e"]]
    res = llm_refine(topic, backend=_flagger({"bad1", "bad2", "bad3"}), n=4, m=2)
    # highest-ranked flagged are bad1 (idx1) and bad2 (idx3); bad3 survives ranking
    assert res[0]["dropped"] == ["bad1", "bad2"]
    assert len(res[0]["words"]) == 4
    assert "bad1" not in res[0]["words"] and "bad2" not in res[0]["words"]


def test_dropped_preserves_rank_order():
    res = llm_refine(TOPICS, backend=_flagger(), n=6, m=2)
    # banana (idx 6) precedes guitar (idx 7) in the topic, so that is the drop order
    assert res[0]["dropped"] == ["banana", "guitar"]


def test_majority_vote_over_samples():
    # A flaky backend flags the intruder only half the time; majority of 4 needs >=2.
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return "banana" if (calls["n"] % 2 == 0 and "banana" in prompt.lower()) else "none"

    res = llm_refine([TOPICS[0]], backend=flaky, n=6, m=2, n_samples=4)
    # 2 of 4 flag banana -> meets the majority threshold (>=2), so it is dropped
    assert "banana" in res[0]["dropped"]
