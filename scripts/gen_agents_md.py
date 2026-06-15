#!/usr/bin/env python
"""Render the root ``AGENTS.md`` from the ``topica-analysis`` skill.

The skill (``.claude/skills/topica-analysis/SKILL.md``) is the single source of
truth for the "using topica well" guide. ``AGENTS.md`` is its cross-tool ambient
copy (the agents.md open standard, read automatically by Claude Code, Cursor, and
similar) — generated, not hand-edited, so the two cannot drift.

    python scripts/gen_agents_md.py          # rewrite AGENTS.md
    python scripts/gen_agents_md.py --check   # exit 1 if AGENTS.md is stale

Edit the skill, then regenerate. ``tests/test_agents_md_sync.py`` fails if stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / ".claude" / "skills" / "topica-analysis" / "SKILL.md"
AGENTS = ROOT / "AGENTS.md"

BANNER = (
    "<!-- Generated from .claude/skills/topica-analysis/SKILL.md by "
    "`python scripts/gen_agents_md.py`.\n"
    "     This is the cross-tool (agents.md standard) ambient copy of the "
    "topica-analysis skill.\n"
    "     Edit the skill, not this file. -->"
)


def _skill_body() -> str:
    """The SKILL.md text with its YAML frontmatter stripped."""
    text = SKILL.read_text()
    if text.startswith("---"):
        # drop the frontmatter block (between the first two '---' lines)
        end = text.find("\n---", 3)
        text = text[end + 4:].lstrip("\n")
    return text


def rendered() -> str:
    return f"{BANNER}\n\n{_skill_body()}"


def main() -> None:
    check = "--check" in sys.argv
    want = rendered()
    have = AGENTS.read_text() if AGENTS.exists() else ""
    if check:
        if have != want:
            raise SystemExit("AGENTS.md is stale; run scripts/gen_agents_md.py")
        print("AGENTS.md up to date")
    else:
        AGENTS.write_text(want)
        print(f"wrote {AGENTS.name} ({len(want.splitlines())} lines)")


if __name__ == "__main__":
    main()
