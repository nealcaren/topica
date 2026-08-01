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

from topica.registry import (  # noqa: E402
    chooser_markdown_table,
    impl_markdown_table,
    markdown_table,
)

BEGIN = "<!-- BEGIN MODEL TABLE (generated from topica.registry; edit registry.py, not this block) -->"
END = "<!-- END MODEL TABLE -->"
TARGETS = [ROOT / "README.md", ROOT / "docs" / "guides" / "models.md"]

# The front-door decision matrix (goal -> model), injected into the README and
# the "What you can do" landing page from the same registry source.
CHOOSER_BEGIN = "<!-- BEGIN CHOOSER (generated from topica.registry CHOOSER; edit registry.py, not this block) -->"
CHOOSER_END = "<!-- END CHOOSER -->"
CHOOSER_TARGETS = [ROOT / "README.md", ROOT / "docs" / "can-do" / "index.md"]

# The contributor implementation map (issue #381): the same registry, rendered
# with the source/binding/feature/validation columns instead of the user-facing
# taxonomy. Injected into its own docs page with its own marker pair.
MAP_BEGIN = "<!-- BEGIN MODEL MAP (generated from topica.registry IMPL; edit registry.py, not this block) -->"
MAP_END = "<!-- END MODEL MAP -->"
MAP_TARGET = ROOT / "docs" / "contributing" / "model-map.md"


def _rendered_block() -> str:
    return f"{BEGIN}\n\n{markdown_table(by_group=True)}\n{END}"


def _rendered_map_block() -> str:
    return f"{MAP_BEGIN}\n\n{impl_markdown_table(by_group=True)}\n{MAP_END}"


def _rendered_chooser_block() -> str:
    return f"{CHOOSER_BEGIN}\n\n{chooser_markdown_table()}\n{CHOOSER_END}"


def _inject(text: str, begin: str, end: str, block: str) -> str:
    i, j = text.find(begin), text.find(end)
    if i == -1 or j == -1:
        raise SystemExit("marker comments not found in target")
    return text[:i] + block + text[j + len(end):]


def inject(text: str) -> str:
    return _inject(text, BEGIN, END, _rendered_block())


def inject_map(text: str) -> str:
    return _inject(text, MAP_BEGIN, MAP_END, _rendered_map_block())


def inject_chooser(text: str) -> str:
    return _inject(text, CHOOSER_BEGIN, CHOOSER_END, _rendered_chooser_block())


def main() -> None:
    check = "--check" in sys.argv
    stale = []
    jobs = (
        [(path, inject) for path in TARGETS]
        + [(MAP_TARGET, inject_map)]
        + [(path, inject_chooser) for path in CHOOSER_TARGETS]
    )
    for path, injector in jobs:
        old = path.read_text(encoding="utf-8")
        new = injector(old)
        if old != new:
            stale.append(path.name)
            if not check:
                path.write_text(new, encoding="utf-8")
    if check and stale:
        print(f"stale (run scripts/gen_model_tables.py): {stale}")
        raise SystemExit(1)
    names = [t.name for t, _ in jobs]
    print("up to date" if check else f"wrote {names}")


if __name__ == "__main__":
    main()
