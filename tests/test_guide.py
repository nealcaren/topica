"""``topica.guide()`` renders from the live registry + signatures, and the
generated docs page stays in sync with it.

The cheat sheet is the entry point an agent or first-time user hits after
``import topica``, so it must (a) stay correct as models change (it is built from
introspection, not a hand list) and (b) not drift from its committed docs page.
"""
import pathlib

import pytest

import topica
from topica._guide import build_guide
from topica.registry import REGISTRY


def test_essentials_has_the_load_bearing_sections():
    text = build_guide()
    for marker in (
        "THE WORKFLOW",
        "EVERY FITTED MODEL EXPOSES THE SAME SURFACE",
        "PICK A MODEL BY GOAL",
        "HELPER NAMESPACES",
        "model.topic_word",
        "model.top_words(n)",
        "topica.guide(",
    ):
        assert marker in text, marker


def test_guide_prints_and_returns_none(capsys):
    assert topica.guide() is None
    out = capsys.readouterr().out
    assert "quick guide" in out


def test_model_card_is_case_insensitive_and_carries_signatures():
    for query in ("STM", "stm"):
        card = build_guide(query)
        assert card.startswith("STM ")
        assert "STM(num_topics" in card  # constructor signature, live
        assert "model.fit(" in card      # fit signature, live
        assert "guides/models.md#stm" in card


def test_helper_card_resolves_signatures():
    # The essentials sheet lists helper names without signatures; guide("<name>")
    # must resolve them (the two a sample-user audit crashed on). See #816 audit.
    for name, ns, needle in [
        ("topic_stability", "evaluate", "topic_stability(runs"),
        ("record_fit", "provenance", "record_fit(model, corpus"),
        ("estimate_effect", "effects", "estimate_effect(doc_topic"),
    ]:
        card = build_guide(name)
        assert card.startswith(f"{ns}.{name} "), card[:60]
        assert needle in card


def test_essentials_points_to_helper_guide():
    text = build_guide()
    assert 'guide("<name>")' in text  # closes the name -> signature loop


def test_unknown_name_suggests_across_models_and_helpers():
    card = build_guide("STMX")
    assert "No model or helper named 'STMX'" in card
    assert "STM" in card  # substring suggestion
    # a helper-side miss suggests helper names too
    card2 = build_guide("stability")
    assert "topic_stability" in card2 or "bootstrap_stability" in card2


def test_full_covers_every_validated_model():
    text = build_guide(full=True)
    validated = [n for n, info in REGISTRY.items() if not info.experimental]
    missing = [n for n in validated if f"\n{n}(" not in text and f"\n{n} " not in text]
    assert not missing, f"full guide omits validated models: {missing}"


def test_docs_page_in_sync_with_guide():
    # The generated block in docs/guides/agent-cheatsheet.md must match the
    # gen_guide render; regenerate with scripts/gen_guide.py if this fails.
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    import gen_guide  # noqa: E402

    text = gen_guide.TARGET.read_text(encoding="utf-8")
    i, j = text.find(gen_guide.BEGIN), text.find(gen_guide.END)
    assert i != -1 and j != -1, "agent-cheatsheet.md marker comments missing"
    assert text[i:j + len(gen_guide.END)] == gen_guide.rendered_block(), (
        "docs/guides/agent-cheatsheet.md is stale; run scripts/gen_guide.py")
