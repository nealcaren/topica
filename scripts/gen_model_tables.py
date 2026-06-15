#!/usr/bin/env python
"""Render the model roster from ``topica.registry`` into the README and the docs.

The registry (``python/topica/registry.py``) is the single source of truth. This
script injects its grouped Markdown table between the marker comments in
``README.md`` and ``docs/api/models.md`` so the two lists cannot drift. Run it
after editing the registry; ``tests/test_registry.py`` fails if a target file is
out of sync.

    python scripts/gen_model_tables.py          # rewrite the targets in place
    python scripts/gen_model_tables.py --check   # exit 1 if any target is stale
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from topica.registry import markdown_table  # noqa: E402

BEGIN = "<!-- BEGIN MODEL TABLE (generated from topica.registry; edit registry.py, not this block) -->"
END = "<!-- END MODEL TABLE -->"
TARGETS = [ROOT / "README.md", ROOT / "docs" / "guides" / "models.md"]


def _rendered_block() -> str:
    return f"{BEGIN}\n\n{markdown_table(by_group=True)}\n{END}"


def inject(text: str) -> str:
    i, j = text.find(BEGIN), text.find(END)
    if i == -1 or j == -1:
        raise SystemExit("marker comments not found in target")
    return text[:i] + _rendered_block() + text[j + len(END):]


def main() -> None:
    check = "--check" in sys.argv
    stale = []
    for path in TARGETS:
        old = path.read_text(encoding="utf-8")
        new = inject(old)
        if old != new:
            stale.append(path.name)
            if not check:
                path.write_text(new, encoding="utf-8")
    if check and stale:
        print(f"stale (run scripts/gen_model_tables.py): {stale}")
        raise SystemExit(1)
    print("up to date" if check else f"wrote {[t.name for t in TARGETS]}")


if __name__ == "__main__":
    main()
