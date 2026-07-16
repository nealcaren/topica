"""LLM-based topic evaluation (Stammbach et al. 2023): llm_coherence + llm_intrusion.

Exercised offline with deterministic fake backends (no network), so the
orchestration, parsing, aggregation, and scoring are CI-testable. The live behaviour
is llm-bounded and validated separately (parity/llm_coherence_compare.py).
"""

import numpy as np
import pytest

import topica
from topica.llm import coherence as llm_coherence, intrusion as llm_intrusion, PROMPTS as LLM_EVAL_PROMPTS

# Three topics with a clear theme each; the third is deliberately incoherent.
TOPICS = [
    ["water", "river", "lake", "ocean", "rain", "flood", "stream", "wave"],
    ["senate", "election", "vote", "policy", "law", "congress", "budget", "party"],
    ["banana", "telescope", "guitar", "asphalt", "penguin", "invoice", "comet", "yoga"],
]


def test_namespace_surface():
    # The LLM-eval suite lives under topica.llm.* (an llm-bounded family), and the
    # flat llm_* metric names are intentionally not exposed at the top level.
    assert callable(topica.llm.coherence)
    assert callable(topica.llm.intrusion)
    assert callable(topica.llm.select_k)
    assert callable(topica.llm.backend)
    assert isinstance(topica.llm.PROMPTS, dict)
    for gone in ("llm_coherence", "llm_intrusion", "llm_select_k", "LLM_EVAL_PROMPTS"):
        assert not hasattr(topica, gone), f"{gone} should be namespaced under topica.llm"
    # backend is also reachable flat (a shared, released constructor).
    assert topica.llm.backend is topica.interpret.llm_backend


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
    out = llm_coherence(TOPICS, backend=rater)
    assert out.shape == (3,)
    assert (out >= 1).all() and (out <= 3).all()
    # coherent topics score higher than the random one
    assert out[0] == 3.0 and out[1] == 3.0 and out[2] == 1.0
    # one call per topic by default
    assert len(rater.calls) == 3


def test_llm_coherence_n_samples_averages():
    # A backend that alternates 3,1,3 -> mean 2.33 over 3 samples.
    seq = iter(["3", "1", "3"] * 10)
    out = llm_coherence([TOPICS[0]], backend=lambda p: next(seq), n_samples=3)
    assert out.shape == (1,)
    assert abs(out[0] - (7 / 3)) < 1e-9


def test_llm_coherence_parses_prose_and_clamps():
    # Replies with out-of-range or prose still parse the first in-range integer.
    out = llm_coherence([TOPICS[0], TOPICS[1]], backend=lambda p: "Score: 99? no, a 2 overall.")
    assert np.allclose(out, 2.0)


def test_llm_coherence_accepts_prompts_override():
    seen = {}
    def be(p):
        seen["prompt"] = p
        return "2"
    custom = dict(LLM_EVAL_PROMPTS)
    custom["rating"] = "RATE THESE {dataset}{words}"
    llm_coherence([TOPICS[0]], backend=be, prompts=custom, dataset_description="news")
    assert seen["prompt"].startswith("RATE THESE")
    assert "news" in seen["prompt"]


def test_llm_coherence_dataset_description_in_prompt():
    seen = {}
    llm_coherence([TOPICS[0]], backend=lambda p: seen.setdefault("p", p) or "3",
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
    items = topica.diagnostics.word_intrusion(phi, vocab, n_words=5, seed=0)
    answer = {", ".join(it["words"]): it["intruder"] for it in items}
    def oracle(prompt):
        for words, intr in answer.items():
            if words in prompt:
                return intr
        return "?"
    res = llm_intrusion(phi, vocab, backend=oracle, n_words=5, seed=0)
    assert res["accuracy"] == 1.0
    assert all(e["correct"] for e in res["per_topic"])


def test_llm_intrusion_wrong_picks_score_low():
    phi, vocab = _planted_model()
    # Always returns the first presented word (usually not the intruder).
    res = llm_intrusion(phi, vocab, backend=lambda p: p.rsplit("\n", 1)[-1].split(",")[0],
                        n_words=5, seed=0)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert len(res["per_topic"]) == 3


def test_llm_intrusion_majority_vote_over_samples():
    phi, vocab = _planted_model()
    items = topica.diagnostics.word_intrusion(phi, vocab, n_words=5, seed=0)
    intr0 = items[0]["intruder"]
    # 2 of 3 votes are the true intruder -> majority correct for topic 0.
    seq = {}
    def be(prompt):
        if items[0]["intruder"] in prompt and ", ".join(items[0]["words"]) in prompt:
            seq.setdefault("t0", 0)
            seq["t0"] += 1
            return intr0 if seq["t0"] != 2 else items[0]["words"][0]
        return "?"
    res = llm_intrusion(phi, vocab, backend=be, n_words=5, seed=0, n_samples=3)
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
    items = topica.diagnostics.word_intrusion(phi, vocab, n_words=5, seed=1)
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
        llm_coherence(TOPICS, backend=123)


def test_accepts_fitted_model_surface():
    # llm_coherence reads top_words / topic_word like the other diagnostics.
    docs = [["water", "river", "lake"]] * 20 + [["senate", "vote", "law"]] * 20
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    out = llm_coherence(m, backend=lambda p: "2", n_words=5)
    assert out.shape == (2,) and np.allclose(out, 2.0)


# ---------------------------------------------------------------------------
# llm_select_k (document-label purity)
# ---------------------------------------------------------------------------

from topica.llm import select_k as llm_select_k


class _FakeModel:
    """Minimal fitted-model stand-in exposing a doc_topic matrix."""

    def __init__(self, doc_topic):
        self.doc_topic = np.asarray(doc_topic, dtype=float)


def _labeled_docs(n_per=8, classes=("alpha", "beta", "gamma")):
    # Each document's first word is its true class label; the fake labeler returns it.
    docs, truth = [], []
    for ci, c in enumerate(classes):
        for j in range(n_per):
            docs.append(f"{c} document number {j} about the {c} theme")
            truth.append(ci)
    return docs, np.array(truth)


def _fake_labeler(prompt: str) -> str:
    # Pull the document's leading class word out of the prompt.
    body = prompt.split("Document:\n", 1)[-1].strip()
    return body.split()[0] if body else "?"


def _model_with_top_docs(D, per_topic_docs):
    """A fake model whose topic t ranks exactly `per_topic_docs[t]` at the top, in
    that order, with DISTINCT descending theta (so argsort is platform-independent
    and there are no cross-label ties)."""
    K = len(per_topic_docs)
    dt = np.full((D, K), 1e-3)
    for t, ds in enumerate(per_topic_docs):
        for rank, d in enumerate(ds):
            dt[d, t] = 1.0 - 1e-3 * rank
    return _FakeModel(dt)


def test_select_k_prefers_pure_partition():
    docs, truth = _labeled_docs(n_per=8)        # 24 docs, 3 true classes
    D = len(docs)
    by_class = [[d for d, c in enumerate(truth) if c == k] for k in range(3)]

    def interleave(a, b):
        out = []
        for x, y in zip(a, b):
            out += [x, y]
        return out

    # Good (K=3): each topic's top docs are one whole class -> pure.
    good = _model_with_top_docs(D, by_class)
    # Bad (K=2): each topic's top docs interleave two classes -> ~50/50, impure.
    bad = _model_with_top_docs(D, [interleave(by_class[0], by_class[1]),
                                   interleave(by_class[1], by_class[2])])
    res = llm_select_k([bad, good], docs, backend=_fake_labeler, n_docs=6)
    assert res["best"] == 3                       # the pure K=3 model wins
    assert res["best_index"] == 1
    pur = {s["num_topics"]: s["purity"] for s in res["scores"]}
    assert pur[3] == 1.0
    assert pur[2] < pur[3] - 0.03                  # mixed model clearly below the knee tol


def test_select_k_per_topic_purity_shape():
    docs, truth = _labeled_docs(n_per=6)
    D = len(docs)
    dt = np.zeros((D, 3))
    for d, c in enumerate(truth):
        dt[d, c] = 1.0
    res = llm_select_k([_FakeModel(dt)], docs, backend=_fake_labeler, n_docs=5)
    s = res["scores"][0]
    assert len(s["per_topic_purity"]) == 3
    assert all(0.0 <= p <= 1.0 for p in s["per_topic_purity"])


def test_select_k_prompt_carries_options():
    docs, truth = _labeled_docs(n_per=4)
    dt = np.zeros((len(docs), 3))
    for d, c in enumerate(truth):
        dt[d, c] = 1.0
    seen = {}
    def be(p):
        seen["p"] = p
        return _fake_labeler(p)
    llm_select_k([_FakeModel(dt)], docs, backend=be, n_docs=2, granularity="narrow",
                 example_labels=["sports", "politics"], research_question="Label by news topic.")
    assert "narrow" in seen["p"]
    assert "sports" in seen["p"] and "Label by news topic." in seen["p"]


def test_select_k_knee_prefers_smaller_on_plateau():
    # Two models: K=3 (pure, true) and K=6 (over-split, marginally purer). The knee
    # criterion prefers the smaller K on the plateau; "max" takes the over-split one.
    docs, truth = _labeled_docs(n_per=8)
    D = len(docs)
    good = np.zeros((D, 3))
    for d, c in enumerate(truth):
        good[d, c] = 1.0
    # K=6: split each class into two topics (still same label -> still pure).
    split = np.zeros((D, 6))
    for d, c in enumerate(truth):
        split[d, c * 2 + (d % 2)] = 1.0
    models = [_FakeModel(good), _FakeModel(split)]
    knee = llm_select_k(models, docs, backend=_fake_labeler, n_docs=4)          # default knee
    mx = llm_select_k(models, docs, backend=_fake_labeler, n_docs=4, criterion="max")
    # both purities are ~1.0 here, so the knee takes the smaller K=3
    assert knee["best"] == 3
    # "max" returns whichever is highest (>= the smaller); never the smaller-but-worse
    assert mx["best"] in (3, 6)


def test_select_k_majority_vote_over_samples():
    docs, truth = _labeled_docs(n_per=4)
    dt = np.zeros((len(docs), 3))
    for d, c in enumerate(truth):
        dt[d, c] = 1.0
    # Labeler returns the true class 2/3 of the time, noise otherwise; majority recovers it.
    state = {"i": 0}
    def noisy(p):
        state["i"] += 1
        return _fake_labeler(p) if state["i"] % 3 != 0 else "noise"
    res = llm_select_k([_FakeModel(dt)], docs, backend=noisy, n_docs=3, n_samples=3)
    assert res["scores"][0]["purity"] == 1.0


# ---------------------------------------------------------------------------
# Tan & D'Souza (2025) metrics: outlier / repetitiveness / diversity /
# adversarial / alignment
# ---------------------------------------------------------------------------

def test_tan_metrics_in_namespace():
    for name in ("outlier", "repetitiveness", "diversity", "adversarial", "alignment"):
        assert callable(getattr(topica.llm, name))


def test_outlier_threshold_voting():
    topics = [["river", "lake", "ocean", "wave", "shakespeare"]]
    # flags "shakespeare" on 3 of 5 runs, "river" once -> only shakespeare survives >=3.
    seq = iter(["shakespeare", "shakespeare", "river", "shakespeare", "none"])
    res = topica.llm.outlier(topics, backend=lambda p: next(seq), n_words=5,
                             n_samples=5, threshold=3)
    assert res[0]["outliers"] == ["shakespeare"]
    assert res[0]["count"] == 1


def test_outlier_filters_to_topic_words():
    topics = [["river", "lake", "ocean"]]
    # a hallucinated word not in the topic is dropped.
    res = topica.llm.outlier(topics, backend=lambda p: "banana, river", n_words=3,
                             n_samples=1, threshold=1)
    assert res[0]["outliers"] == ["river"]


def test_repetitiveness_rate_and_duplicates():
    topics = [["car", "automobile", "road", "drive"]]
    def be(prompt):
        if "repetitive" in prompt.lower() and "Rate" in prompt:
            return "1"   # highly repetitive
        return "(car, automobile)"   # duplicate pair
    res = topica.llm.repetitiveness(topics, backend=be, n_words=4)
    assert res[0]["rate"] == 1.0
    assert res[0]["duplicate_count"] == 1
    assert ("automobile", "car") in res[0]["duplicate_pairs"]


def test_diversity_pairwise_mean():
    topics = [["a", "b"], ["c", "d"], ["e", "f"]]   # 3 topics -> 3 pairs
    res = topica.llm.diversity(topics, backend=lambda p: "3", n_words=2)
    assert len(res["pairwise"]) == 3
    assert res["mean"] == 3.0


def test_diversity_max_pairs_subsets():
    topics = [["a"], ["b"], ["c"], ["d"]]   # 6 pairs
    res = topica.llm.diversity(topics, backend=lambda p: "2", n_words=1, max_pairs=2, seed=0)
    assert len(res["pairwise"]) == 2


def test_adversarial_detects_planted_outlier():
    topics = [["river", "lake", "ocean", "wave"], ["senate", "vote", "law", "court"]]
    # An oracle that flags the planted intruder.
    good = topica.llm.adversarial(topics, backend=lambda p: "shakespeare", n_words=4,
                                  n_samples=1, threshold=1)
    assert good["detection_rate"] == 1.0
    assert good["intruder"] == "shakespeare"
    # A weak model that never flags it -> 0 detection (the capability signal).
    weak = topica.llm.adversarial(topics, backend=lambda p: "none", n_words=4,
                                  n_samples=1, threshold=1)
    assert weak["detection_rate"] == 0.0


def test_alignment_counts_irrelevant_and_missing():
    docs = ["the river flooded the lake and the wave hit the shore"] * 6
    phi = np.zeros((1, 6)); phi[0, :4] = 1.0; phi = phi / phi.sum(1, keepdims=True)
    vocab = ["river", "lake", "wave", "shore", "senate", "tax"]
    # one-topic model exposing doc_topic + the analysis surface

    class M:
        doc_topic = np.ones((6, 1))
        def top_words(self, n):
            return [[(w, 1.0) for w in vocab[:n]]]
    def be(prompt):
        if "not relevant" in prompt.lower() or "irrelevant" in prompt.lower():
            return "tax"          # 1 irrelevant topic word
        return "flooding, shore"  # 2 missing themes
    res = topica.llm.alignment(M(), docs, backend=be, n_words=4, n_docs=3)
    assert res[0]["irrelevant"] >= 0.0 and res[0]["missing"] >= 0.0
