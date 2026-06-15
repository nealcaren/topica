"""AGENTS.md is generated from the topica-analysis skill and must stay in sync
(regenerate with scripts/gen_agents_md.py). Also checks both shipped skills have
valid frontmatter."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "gen_agents_md", ROOT / "scripts" / "gen_agents_md.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agents_md_matches_skill():
    gen = _load_gen()
    assert (ROOT / "AGENTS.md").read_text() == gen.rendered(), (
        "AGENTS.md is stale; run scripts/gen_agents_md.py")


@pytest.mark.parametrize("skill", ["topica-analysis", "add-topic-model"])
def test_skill_has_valid_frontmatter(skill):
    path = ROOT / ".claude" / "skills" / skill / "SKILL.md"
    text = path.read_text()
    assert text.startswith("---\n"), f"{skill}: missing YAML frontmatter"
    end = text.find("\n---", 3)
    assert end != -1, f"{skill}: unterminated frontmatter"
    fm = text[4:end]
    assert "name:" in fm and "description:" in fm, f"{skill}: name/description required"
