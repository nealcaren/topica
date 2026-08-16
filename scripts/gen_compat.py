#!/usr/bin/env python
"""Regenerate ``python/topica/_compat.py`` from the workflow-namespace ``__all__``s.

``_compat.LAZY`` maps each legacy top-level name to the namespace that owns it; it
is the single source of truth for ``topica.__getattr__`` and the compat test. This
script rebuilds it so a change to a namespace module's ``__all__`` propagates with
one command. ``--check`` exits non-zero if the committed file is stale (wired into
``scripts/preflight.sh``).

Usage:
    python scripts/gen_compat.py            # rewrite _compat.py
    python scripts/gen_compat.py --check    # verify it is up to date
"""

from __future__ import annotations

import importlib
import pathlib
import sys

NAMESPACES = [
    "data", "design", "select", "inspect", "evaluate",
    "effects", "compare", "embeddings", "provenance",
]

# Names that stay in the curated top-level ``__all__`` are eager and never go
# through ``__getattr__``, so they are excluded from LAZY (in particular
# ``compare`` is a callable module, not the function its namespace exports).
_CURATED = {
    "data", "design", "select", "inspect", "evaluate",
    "effects", "compare", "embeddings", "provenance",
    "Corpus", "tokenize", "from_dataframe",
    "LDA", "STM", "NMF", "KeyATM", "GSDMM", "BERTopic",
    "search_k", "topic_table", "estimate_effect",
    "list_models", "enable_experimental",
}
_TARGET = pathlib.Path(__file__).resolve().parent.parent / "python" / "topica" / "_compat.py"

_HEADER = '''"""Backward-compat map for the curated root (issue #757).

``LAZY`` maps each legacy top-level name to the workflow namespace that now owns
it. It is the single source of truth for :func:`topica.__getattr__` (lazy
resolution of names no longer in the curated ``__all__``) and for the compat
test that every legacy ``topica.X`` still resolves and equals
``topica.<namespace>.X``. Regenerate with ``scripts/gen_compat.py``.
"""

LAZY = {
'''


def build() -> str:
    home: dict[str, str] = {}
    for ns in NAMESPACES:  # priority order: first namespace to export a name wins
        mod = importlib.import_module(f"topica.{ns}")
        for name in getattr(mod, "__all__", []):
            if name in _CURATED:
                continue
            home.setdefault(name, ns)
    body = "".join(f"    {n!r}: {home[n]!r},\n" for n in sorted(home))
    return _HEADER + body + "}\n"


def main() -> int:
    generated = build()
    if "--check" in sys.argv:
        current = _TARGET.read_text() if _TARGET.exists() else ""
        if current != generated:
            print("_compat.py is out of date; run: python scripts/gen_compat.py",
                  file=sys.stderr)
            return 1
        print("_compat.py is up to date")
        return 0
    _TARGET.write_text(generated)
    print(f"wrote {_TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
