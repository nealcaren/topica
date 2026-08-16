"""Guards for the curated lazy root (issue #757).

The top-level ``__all__`` was curated to a small surface, with every other legacy
name resolved lazily through its workflow namespace via ``topica.__getattr__``.
These tests pin the compatibility contract: nothing that used to resolve at
``topica.X`` stopped resolving, and each lazily-resolved name is the *same object*
as its namespace member.
"""

import importlib
import inspect as _stdlib_inspect
import pathlib

import topica
from topica import _compat

_LEGACY = pathlib.Path(__file__).parent / "_legacy_surface.txt"
_LEGACY_NAMES = [ln.strip() for ln in _LEGACY.read_text().splitlines() if ln.strip()]

# The curated public surface. Kept in lock-step with topica.__all__.
_CURATED = {
    "data", "design", "select", "inspect", "evaluate",
    "effects", "compare", "embeddings", "provenance",
    "Corpus", "tokenize", "from_dataframe",
    "LDA", "STM", "NMF", "KeyATM", "GSDMM", "BERTopic",
    "list_models", "enable_experimental",
    "__version__", "__citation__",
}


def test_curated_all_is_exactly_the_surface():
    assert set(topica.__all__) == _CURATED
    assert len(topica.__all__) == 22


def test_flagship_models_match_common_start_set():
    # The models kept flat in __all__ are exactly the newcomer starting set.
    models_in_all = {n for n in topica.__all__
                     if isinstance(getattr(topica, n), type)
                     and hasattr(getattr(topica, n), "fit")}
    common_start = {m.name for m in topica.list_models(common_start=True)}
    assert models_in_all == common_start


def test_every_legacy_name_still_resolves():
    # Nothing that resolved at topica.X before the curation stopped resolving.
    missing = [n for n in _LEGACY_NAMES if not hasattr(topica, n)]
    assert not missing, f"legacy names no longer resolve: {missing}"


def test_lazy_names_are_identical_to_their_namespace_member():
    # Each mapped legacy name is the *same object* as topica.<namespace>.<name>.
    bad = []
    for name, ns in _compat.LAZY.items():
        flat = getattr(topica, name)
        namespaced = getattr(getattr(topica, ns), name)
        if flat is not namespaced:
            bad.append((name, ns))
    assert not bad, f"flat alias diverged from namespace object: {bad}"


def test_getattr_serves_a_delisted_name_lazily():
    # Force the eager cache off and confirm __getattr__ reimports from the namespace.
    mod = importlib.import_module("topica")
    mod.__dict__.pop("frex", None)
    assert topica.frex is topica.inspect.frex


def test_star_import_yields_only_the_curated_surface():
    ns = {}
    exec("from topica import *", ns)
    starred = {n for n in ns if not n.startswith("__")}
    # __version__/__citation__ are dunder-named and not star-imported.
    assert starred == _CURATED - {"__version__", "__citation__"}


def test_compat_map_is_in_sync_with_namespaces():
    # _compat.LAZY must match what scripts/gen_compat.py would produce.
    import subprocess
    import sys
    root = pathlib.Path(__file__).parent.parent
    r = subprocess.run([sys.executable, str(root / "scripts" / "gen_compat.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_compare_is_a_callable_module():
    assert callable(topica.compare)
    assert hasattr(topica.compare, "CompareResult")


def test_topica_inspect_is_not_stdlib_inspect():
    assert topica.inspect is not _stdlib_inspect
    assert topica.inspect.__name__ == "topica.inspect"


def test_models_keep_flat_module_qualname():
    # Re-exporting must not restamp __module__/__qualname__ (that is what a pickle
    # or a repr records). The compiled classes stay registered under "topica",
    # where LDA remains resolvable after the curation.
    assert topica.LDA.__module__ == "topica"
    assert topica.LDA.__qualname__ == "LDA"
    assert getattr(topica, "LDA") is topica.LDA
