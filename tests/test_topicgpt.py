"""TopicGPT orchestration, exercised with a deterministic fake backend.

No network or API: a fake callable returns canned replies in the published
TopicGPT bracketed-line format (``[1] Label: text``) for the generation,
refinement, and assignment prompts, so the whole generate/refine/assign pipeline
is testable end to end. The fake routes by which template the prompt came from.
"""

from __future__ import annotations

import numpy as np
import pytest

import topica
from topica.topicgpt import TopicGPT, PROMPTS


# Three tiny documents, one per intended topic, with a clear keyword each.
DOCS = [
    "the senate passed a budget bill and the president signed the new law",
    "the team won the championship game with a last second goal by the striker",
    "the new phone ships with a faster chip and a better camera and screen",
]
# Held-out doc that should land on the sports topic.
HELD_OUT = ["the coach praised the goalkeeper after the cup final match"]


class FakeBackend:
    """A deterministic backend that answers each prompt stage in the published
    TopicGPT bracketed-line format (``[1] Label: text``).

    It also records every prompt it sees so tests can assert on call counts and
    on caching (a cached prompt never reaches the backend twice).
    """

    def __init__(self):
        self.calls: list[str] = []
        # Per-document generation replies, keyed by a substring of the document.
        self.gen = {
            "senate": ("Politics", "Government, legislation, and elected officials."),
            "championship": ("Sports", "Competitive games, teams, and athletes."),
            "phone": ("Technology", "Consumer electronics and computing hardware."),
        }

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Assignment shares the "You will receive a document and a..." opening with
        # generation, so test it first by its distinctive [:40] head.
        if prompt.startswith(PROMPTS["assignment"][:40]):
            if "senate" in prompt or "president" in prompt:
                return '[1] Politics: Mentions legislation ("the senate passed a budget bill")'
            if "championship" in prompt or "goalkeeper" in prompt or "cup final" in prompt:
                return '[1] Sports: Mentions a match ("won the championship game")'
            if "phone" in prompt or "chip" in prompt:
                return '[1] Technology: Mentions hardware ("ships with a faster chip")'
            return "[1] Politics: No clear quote ()"
        if prompt.startswith(PROMPTS["generation"][:40]):
            for key, (name, desc) in self.gen.items():
                if key in prompt:
                    return f"[1] {name}: {desc}"
            return "[1] Other: Miscellaneous."
        if prompt.startswith(PROMPTS["refinement"][:40]):
            # Idempotent refinement: return the three topics unchanged.
            return (
                "[1] Politics: Government and legislation.\n"
                "[1] Sports: Games and athletes.\n"
                "[1] Technology: Consumer electronics."
            )
        return "None"


def _fit(**kw) -> tuple[TopicGPT, FakeBackend]:
    be = FakeBackend()
    model = TopicGPT(backend=be, **kw)
    model.fit(DOCS)
    return model, be


# ---------------------------------------------------------------------------
# Core: fit produces a taxonomy + doc_topic + descriptions
# ---------------------------------------------------------------------------

def test_fit_discovers_taxonomy_and_descriptions():
    model, _ = _fit()
    assert model.num_topics == 3
    names = set(model.topic_names)
    assert names == {"Politics", "Sports", "Technology"}
    descs = model.topic_descriptions
    assert len(descs) == 3 and all(isinstance(d, str) and d for d in descs)


def test_doc_topic_is_one_hot_for_hard():
    model, _ = _fit(assignment="hard")
    dt = model.doc_topic
    assert dt.shape == (3, 3)
    # each row sums to 1 and is one-hot
    assert np.allclose(dt.sum(axis=1), 1.0)
    assert set(np.unique(dt)) <= {0.0, 1.0}
    # the three docs land on three distinct topics
    assert len({int(r.argmax()) for r in dt}) == 3


def test_assignments_carry_topic_and_quote():
    model, _ = _fit()
    assert len(model.assignments) == 3
    for a in model.assignments:
        assert 0 <= a.topic_id < model.num_topics
    quotes = [a.quote for a in model.assignments]
    assert any("championship" in q for q in quotes)


# ---------------------------------------------------------------------------
# Synthesized topic_word descriptor
# ---------------------------------------------------------------------------

def test_topic_word_is_valid_descriptor():
    model, _ = _fit()
    tw = model.topic_word
    v = len(model.vocabulary)
    assert tw.shape == (3, v)
    assert (tw >= 0).all()
    # L1-normalized rows (a descriptor that reads like a distribution)
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_top_words_and_salient_keyword():
    model, _ = _fit()
    # the sports topic's top words should surface a sports keyword
    sports = model.topic_names.index("Sports")
    words = [w for w, _ in model.top_words(10, topic=sports)]
    assert any(w in words for w in ("championship", "goal", "striker", "game"))
    # full list form: one row per topic
    allrows = model.top_words(5)
    assert len(allrows) == 3


def test_coherence_runs():
    model, _ = _fit()
    c = model.coherence(5)
    assert np.asarray(c).shape == (3,)
    assert np.all(np.isfinite(c))


def test_composes_with_analysis_surface():
    # coherence() module function and find_thoughts read the standard surface.
    model, _ = _fit()
    c = topica.coherence(model, DOCS, coherence_type="u_mass", topn=5)
    assert np.asarray(c).shape == (3,)
    thoughts = topica.find_thoughts(model.doc_topic, DOCS, topic=0, n=1)
    assert len(thoughts) >= 1


# ---------------------------------------------------------------------------
# transform: held-out assignment
# ---------------------------------------------------------------------------

def test_transform_assigns_heldout():
    model, _ = _fit()
    out = model.transform(HELD_OUT)
    assert out.shape == (1, 3)
    assert np.allclose(out.sum(axis=1), 1.0)
    # the cup-final doc should map to Sports
    assert model.topic_names[int(out[0].argmax())] == "Sports"


# ---------------------------------------------------------------------------
# soft assignment
# ---------------------------------------------------------------------------

def test_soft_assignment_normalizes():
    be = FakeBackend()

    def multi(prompt: str) -> str:
        if prompt.startswith(PROMPTS["assignment"][:40]) and "senate" in prompt:
            return (
                '[1] Politics: Mentions the senate ("senate")\n'
                '[1] Technology: Mentions a law ("law")'
            )
        return be(prompt)

    model = TopicGPT(backend=multi, assignment="soft")
    model.fit(DOCS)
    dt = model.doc_topic
    assert np.allclose(dt.sum(axis=1), 1.0)
    # first doc split across two topics
    assert (dt[0] > 0).sum() == 2


# ---------------------------------------------------------------------------
# Caching avoids duplicate calls + reproducibility
# ---------------------------------------------------------------------------

def test_caching_avoids_duplicate_backend_calls():
    be = FakeBackend()
    model = TopicGPT(backend=be)
    model.fit(DOCS)
    # transform on a doc whose assignment prompt already appeared in fit must hit
    # the cache: the backend call count does not grow.
    before = len(be.calls)
    model.transform([DOCS[0]])   # identical assignment prompt -> cached
    assert len(be.calls) == before


def test_fixed_fake_backend_is_reproducible():
    m1, _ = _fit()
    m2, _ = _fit()
    assert m1.topic_names == m2.topic_names
    assert np.array_equal(m1.doc_topic, m2.doc_topic)
    assert np.array_equal(m1.topic_word, m2.topic_word)


def test_estimated_calls_matches_budget():
    be = FakeBackend()
    model = TopicGPT(backend=be)
    est = model.estimated_calls(DOCS)
    # generation (3) + refinement (1) + assignment (3)
    assert est == 7


def test_sample_limits_generation():
    be = FakeBackend()
    model = TopicGPT(backend=be, sample=1)
    model.fit(DOCS)
    # only the first doc seeds generation, so one topic is discovered
    assert model.num_topics == 1


# ---------------------------------------------------------------------------
# Honest declines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["estimate_effect", "posterior_theta_samples", "ensemble"])
def test_declines_raise(method):
    model, _ = _fit()
    with pytest.raises(NotImplementedError):
        getattr(model, method)()


def test_model_family_is_none():
    model, _ = _fit()
    assert topica.model_family(model) == "none"


# ---------------------------------------------------------------------------
# Backend wiring / errors
# ---------------------------------------------------------------------------

def test_backend_and_model_are_mutually_exclusive():
    with pytest.raises(ValueError):
        TopicGPT(backend=lambda p: "", model="gpt-4o-mini")


def test_no_backend_raises_import_error_with_hint():
    model = TopicGPT()  # neither backend nor model
    with pytest.raises(ImportError) as e:
        model.fit(DOCS)
    assert "topica[llm]" in str(e.value) or "backend" in str(e.value)


def test_unfitted_access_raises():
    model = TopicGPT(backend=lambda p: "{}")
    with pytest.raises(RuntimeError):
        _ = model.doc_topic


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    model, _ = _fit()
    path = str(tmp_path / "tgpt.pkl")
    model.save(path)
    loaded = TopicGPT.load(path)
    assert loaded.topic_names == model.topic_names
    assert np.array_equal(loaded.doc_topic, model.doc_topic)
    assert np.array_equal(loaded.topic_word, model.topic_word)
    assert loaded.topic_descriptions == model.topic_descriptions
    # transform works after re-attaching a backend
    loaded.set_backend(FakeBackend())
    out = loaded.transform(HELD_OUT)
    assert out.shape == (1, model.num_topics)


# ---------------------------------------------------------------------------
# hierarchy
# ---------------------------------------------------------------------------

def test_hierarchical_induces_supertopics():
    be = FakeBackend()
    model = TopicGPT(backend=be, hierarchical=True)
    model.fit(DOCS)
    assert model.hierarchy is not None
    assert "supertopics" in model.hierarchy
    assert any(stage == "hierarchy" for stage, _ in model.stage_log)


# ---------------------------------------------------------------------------
# corpus input + fit_history
# ---------------------------------------------------------------------------

def test_accepts_corpus_input():
    corpus = topica.Corpus.from_documents([d.split() for d in DOCS])
    be = FakeBackend()
    model = TopicGPT(backend=be)
    model.fit(corpus)
    assert model.num_topics == 3


def test_stage_log_logs_stages():
    model, _ = _fit()
    stages = [s for s, _ in model.stage_log]
    assert stages[:3] == ["generation", "refinement", "assignment"]


def test_fit_history_is_empty_no_trace():
    # TopicGPT is a no-trace cluster-style model: fit_history == [], converged None.
    model, _ = _fit()
    assert model.fit_history == []
    assert model.converged is None


# ---------------------------------------------------------------------------
# Custom prompts
# ---------------------------------------------------------------------------

def test_partial_prompt_override_merges_over_defaults():
    # Overriding one stage keeps the published defaults for the others.
    custom_gen = "MY GENERATION {taxonomy} {document}\n[1] Foo: bar"
    be = FakeBackend()
    model = TopicGPT(backend=be, prompts={"generation": custom_gen})
    assert model.prompts["generation"] == custom_gen
    assert model.prompts["refinement"] == PROMPTS["refinement"]
    assert model.prompts["assignment"] == PROMPTS["assignment"]


def test_unknown_prompt_key_raises():
    with pytest.raises(ValueError, match="unknown prompt key"):
        TopicGPT(backend=lambda p: "", prompts={"generaton": "typo {taxonomy} {document}"})


def test_custom_prompt_missing_field_raises():
    # A generation template without {document} would silently drop the document.
    with pytest.raises(ValueError, match="missing required field"):
        TopicGPT(backend=lambda p: "", prompts={"generation": "no fields here {taxonomy}"})


def test_with_prompt_overrides_one_stage_and_drives_fit():
    # A custom generation prompt actually steers the fit: the backend keys off the
    # custom head and returns a single topic.
    custom_gen = "CUSTOM-GEN {taxonomy} :: {document}"

    def be(prompt: str) -> str:
        if prompt.startswith("CUSTOM-GEN"):
            return "[1] OnlyTopic: the only topic"
        if prompt.startswith(PROMPTS["refinement"][:40]):
            return "None"
        return '[1] OnlyTopic: matched ("quote")'

    model = TopicGPT(backend=be).with_prompt("generation", custom_gen)
    model.fit(DOCS)
    assert model.topic_names == ["OnlyTopic"]


def test_with_prompt_after_fit_raises():
    model, _ = _fit()
    with pytest.raises(RuntimeError, match="before fit"):
        model.with_prompt("generation", "x {taxonomy} {document}")
