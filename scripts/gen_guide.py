#!/usr/bin/env python
"""Render ``topica.guide()`` into the agent cheat-sheet docs page.

``topica._guide.build_guide`` is the single source: the same builder that
``topica.guide()`` prints live. This script injects its one-screen essentials and
its full model reference between the marker comments in
``docs/guides/agent-cheatsheet.md`` so the page cannot drift from the installed
build. ``tests/test_guide.py`` fails if the page is out of sync.

    python scripts/gen_guide.py           # rewrite the page in place
    python scripts/gen_guide.py --check    # exit 1 if the page is stale
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from topica._guide import build_guide  # noqa: E402

BEGIN = "<!-- BEGIN GUIDE (generated from topica.guide via scripts/gen_guide.py; edit _guide.py, not this block) -->"
END = "<!-- END GUIDE -->"
TARGET = ROOT / "docs" / "guides" / "agent-cheatsheet.md"


def rendered_block() -> str:
    essentials = build_guide(show_version=False)
    reference = build_guide(full=True)
    return (
        f"{BEGIN}\n\n"
        "## The one-screen guide\n\n"
        f"```text\n{essentials}\n```\n\n"
        "## Every validated model\n\n"
        f"```text\n{reference}\n```\n"
        f"{END}"
    )


def inject(text: str) -> str:
    i, j = text.find(BEGIN), text.find(END)
    if i == -1 or j == -1:
        raise SystemExit(f"marker comments not found in {TARGET.name}")
    return text[:i] + rendered_block() + text[j + len(END):]


def main() -> None:
    check = "--check" in sys.argv
    old = TARGET.read_text(encoding="utf-8")
    new = inject(old)
    if old == new:
        print("up to date" if check else f"{TARGET.name}: no change")
        return
    if check:
        print(f"stale (run scripts/gen_guide.py): {TARGET.name}")
        raise SystemExit(1)
    TARGET.write_text(new, encoding="utf-8")
    print(f"wrote {TARGET.name}")


if __name__ == "__main__":
    main()
