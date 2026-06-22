#!/usr/bin/env python
"""Run the test suite with the reference toolchains hard-blocked at import.

The committed gold-fixture tests (``tests/test_*_gold.py``) must validate topica
*offline* — they load a frozen reference result and compare topica against it, so
they must NOT import the reference implementation at test time. CI installs only
topica's light deps (numpy/pandas/scipy/...), so an offline test that reaches for
a reference package (e.g. ``sklearn.metrics.adjusted_rand_score``) turns CI red —
while passing on a dev machine that happens to have it installed. (This bit us in
issue #271; see the post-mortem in the repo memory.)

This runner installs a ``sys.meta_path`` finder that raises ``ImportError`` for
the reference packages, then hands off to pytest. Use it to reproduce CI's
reference-free environment locally before pushing, and in CI as a dedicated guard:

    python scripts/ci_sim.py tests/test_*_gold.py        # the gold subset (fast)
    python scripts/ci_sim.py tests/                      # the whole suite

Any test that imports a blocked package at collection/test time fails loudly,
which is exactly what we want.
"""
from __future__ import annotations

import sys

# Reference / regenerate-only toolchains that CI does not install and that offline
# tests must never import at test time. (They may be imported inside a gold
# script's ``--regenerate`` path, which this runner never triggers.)
BLOCKED = {
    "torch",
    "fastopic",
    "bertopic",
    "tomotopy",
    "gensim",
    "sklearn",
    "umap",
    "hdbscan",
    "sentence_transformers",
    "transformers",
    "topmost",
    "rpy2",
}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(
                f"ci_sim: '{name}' is blocked — offline gold tests must not import "
                f"a reference toolchain at test time (see scripts/ci_sim.py)"
            )
        return None


def main() -> int:
    sys.meta_path.insert(0, _Blocker())
    import pytest

    args = sys.argv[1:] or ["tests/"]
    return int(pytest.main(["-q", "-p", "no:cacheprovider", *args]))


if __name__ == "__main__":
    raise SystemExit(main())
