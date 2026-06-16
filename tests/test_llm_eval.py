"""LLM-based topic evaluation (Stammbach et al. 2023): llm_coherence + llm_intrusion.

Exercised offline with deterministic fake backends (no network), so the
orchestration, parsing, aggregation, and scoring are CI-testable. The live behaviour
is llm-bounded and validated separately (parity/llm_coherence_compare.py).
"""

import numpy as np
import pytest

import topica
from topica import llm_coherence, llm_intrusion, LLM_EVAL_PROMPTS

# Three topics with a clear theme each; the third is deliberately incoherent.
TOPICS = [
    ["water", "river", "lake", "ocean", "rain", "flood", "stream", "wave"],
    ["senate", "election", "vote", "policy", "law", "congress", "budget", "party"],
    ["banana", "telescope", "guitar", "asphalt", "penguin", "invoice", "comet", "yoga"],
]


class Rater:
    """Rates by counting how many words share the first topic's theme. Coherent
    topics (1, 2) get '3', the random topic (3) gets '1'. Records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        water = sum(w in prompt for w in ("water", "river", "lake", "ocean"))
        gov = sum(w in prompt for w in ("senate", "vote", "law", "congress"))
        if water >= 2 or gov >= 2:
            return "I would rate these as very related: 3"
        return "1"


def test_llm_coherence_shape_and_discrimination():
    rater = Rater()
    out = llm_coherence(TOPICS, call=rater)
    assert out.shape == (3,)
    assert (out >= 1).all() and (out <= 3).all()
    # coherent topics score higher than the random one
    assert out[0] == 3.0 and out[1] == 3.0 and out[2] == 1.0
    # one call per topic by default
    assert len(rater.calls) == 3


def test_llm_coherence_n_samples_averages():
    # A backend that alternates 3,1,3 -> mean 2.33 over 3 samples.
    seq = iter(["3", "1", "3"] * 10)
    out = llm_coherence([TOPICS[0]], call=lambda p: next(seq), n_samples=3)
    assert out.shape == (1,)
    assert abs(out[0] - (7 / 3)) < 1e-9


def test_llm_coherence_parses_prose_and_clamps():
    # Replies with out-of-range or prose still parse the first in-range integer.
    out = llm_coherence([TOPICS[0], TOPICS[1]], call=lambda p: "Score: 99? no, a 2 overall.")
    assert np.allclose(out, 2.0)


def test_llm_coherence_accepts_prompts_override():
    seen = {}
    def be(p):
        seen["prompt"] = p
        return "2"
    custom = dict(LLM_EVAL_PROMPTS)
    custom["rating"] = "RATE THESE {dataset}{words}"
    llm_coherence([TOPICS[0]], call=be, prompts=custom, dataset_description="news")
    assert seen["prompt"].startswith("RATE THESE")
    assert "news" in seen["prompt"]


def test_llm_coherence_dataset_description_in_prompt():
    seen = {}
    llm_coherence([TOPICS[0]], call=lambda p: seen.setdefault("p", p) or "3",
                  dataset_description="Wikipedia articles")
    assert "Wikipedia articles" in seen["p"]


# ---------------------------------------------------------------------------
# llm_intrusion (reuses word_intrusion's generated items)
# ---------------------------------------------------------------------------

def _planted_model(seed=0):
    # A simple (K, V) topic-word matrix with three well-separated blocks.
    rng = np.random.default_rng(seed)
    V = 24
    phi = np.full((3, V), 0.001)
    for t in range(3):
        phi[t, t * 8:(t + 1) * 8] = 1.0
    phi = phi / phi.sum(1, keepdims=True)
    vocab = [f"t{t}w{i}" for t in range(3) for i in range(8)]
    return phi, vocab


def test_llm_intrusion_perfect_detector_scores_one():
    phi, vocab = _planted_model()
    # An oracle backend that returns the true intruder for each item.
    items = topica.word_intrusion(phi, vocab, n_words=5, seed=0)
    answer = {", ".join(it["words"]): it["intruder"] for it in items}
    def oracle(prompt):
        for words, intr in answer.items():
            if words in prompt:
                return intr
        return "?"
    res = llm_intrusion(phi, vocab, call=oracle, n_words=5, seed=0)
    assert res["accuracy"] == 1.0
    assert all(e["correct"] for e in res["per_topic"])


def test_llm_intrusion_wrong_picks_score_low():
    phi, vocab = _planted_model()
    # Always returns the first presented word (usually not the intruder).
    res = llm_intrusion(phi, vocab, call=lambda p: p.rsplit("\n", 1)[-1].split(",")[0],
                        n_words=5, seed=0)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert len(res["per_topic"]) == 3


def test_llm_intrusion_majority_vote_over_samples():
    phi, vocab = _planted_model()
    items = topica.word_intrusion(phi, vocab, n_words=5, seed=0)
    intr0 = items[0]["intruder"]
    # 2 of 3 votes are the true intruder -> majority correct for topic 0.
    seq = {}
    def be(prompt):
        if items[0]["intruder"] in prompt and ", ".join(items[0]["words"]) in prompt:
            seq.setdefault("t0", 0)
            seq["t0"] += 1
            return intr0 if seq["t0"] != 2 else items[0]["words"][0]
        return "?"
    res = llm_intrusion(phi, vocab, call=be, n_words=5, seed=0, n_samples=3)
    assert res["per_topic"][0]["picked"] == intr0


def test_match_intruder_is_robust():
    from topica.coherence import _match_intruder
    words = ["river", "lake", "senate", "ocean", "rain", "wave"]
    assert _match_intruder("senate", words) == "senate"
    assert _match_intruder("The intruder is 'senate'.", words) == "senate"
    assert _match_intruder("SENATE", words) == "senate"
    assert _match_intruder("none of these", words) is None


# ---------------------------------------------------------------------------
# Adversarial self-check (gold-free): a competent detector flags a planted outlier
# ---------------------------------------------------------------------------

def test_adversarial_planted_outlier_is_detectable():
    # Topic of clearly-related words plus a blatant outlier; the task generator must
    # be able to surface a detectable intruder, and an oracle must score it.
    phi, vocab = _planted_model()
    items = topica.word_intrusion(phi, vocab, n_words=5, seed=1)
    # Each generated item has exactly one intruder from another block.
    for it in items:
        intr_block = it["intruder"][1]  # "t{block}w{i}"
        topic_block = str(it["topic"])
        assert intr_block != topic_block  # the intruder is genuinely out-of-block


# ---------------------------------------------------------------------------
# Backend resolution / errors
# ---------------------------------------------------------------------------

def test_call_must_be_callable_or_str():
    with pytest.raises(TypeError):
        llm_coherence(TOPICS, call=123)


def test_accepts_fitted_model_surface():
    # llm_coherence reads top_words / topic_word like the other diagnostics.
    docs = [["water", "river", "lake"]] * 20 + [["senate", "vote", "law"]] * 20
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    out = llm_coherence(m, call=lambda p: "2", n_words=5)
    assert out.shape == (2,) and np.allclose(out, 2.0)
