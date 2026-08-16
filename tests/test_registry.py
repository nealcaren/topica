"""The model registry stays in one-to-one correspondence with the exported
model classes, and its fields use the allowed vocabularies (#182)."""
import inspect

import pytest

import topica
from topica.registry import (
    CHOOSER,
    GROUPS,
    IMPL,
    REGISTRY,
    ModelInfo,
    chooser_markdown_table,
    impl_markdown_table,
    list_models,
    markdown_table,
    validate_impl,
)

# Exported classes that are NOT topic models (so not in the registry).
_NON_MODEL_CLASSES = {"Heldout", "HeldoutResult", "Corpus", "ModelInfo"}

_BRINGS = {"text", "embeddings", "metadata", "seeds", "labels", "times", "llm", "dictionary", "links"}
_INFERENCE = {"gibbs", "variational", "vae", "optimal-transport", "clustering",
              "matrix-factorization", "svd", "ica", "prompting", "em", "neural-embedding",
              "information-theoretic"}
_DETERMINISM = {"bit-exact", "seed-reproducible", "llm-bounded"}


def _exported_model_classes() -> set[str]:
    names = set()
    # Iterate dir(topica), not topica.__all__: the curated __all__ (#757) lists only
    # the flagship models, but every model class stays resolvable at the top level.
    for name in dir(topica):
        obj = getattr(topica, name)
        # Skip aliases (e.g. FLDA -> FactorialLDA, PLTM -> PolylingualLDA): dir()
        # surfaces them, but the registry is keyed by the canonical class name.
        if (inspect.isclass(obj) and hasattr(obj, "fit")
                and name not in _NON_MODEL_CLASSES
                and obj.__name__ == name):
            names.add(name)
    return names


def test_every_exported_model_has_a_registry_entry():
    exported = _exported_model_classes()
    missing = exported - set(REGISTRY)
    assert not missing, f"exported models missing from the registry: {sorted(missing)}"


def test_no_registry_entry_without_an_exported_model():
    exported = _exported_model_classes()
    extra = set(REGISTRY) - exported
    assert not extra, f"registry entries with no exported model: {sorted(extra)}"


def test_registry_name_matches_key_and_resolves():
    for key, info in REGISTRY.items():
        assert info.name == key
        assert hasattr(topica, info.name), f"{info.name} not importable from topica"


def test_registry_fields_use_allowed_vocabularies():
    for info in REGISTRY.values():
        assert info.group in GROUPS, f"{info.name}: bad group {info.group!r}"
        assert info.inference in _INFERENCE, f"{info.name}: bad inference {info.inference!r}"
        assert info.determinism in _DETERMINISM, f"{info.name}: bad determinism"
        assert set(info.brings) <= _BRINGS, f"{info.name}: bad brings {info.brings}"
        assert "text" in info.brings, f"{info.name}: every model brings text"
        assert info.summary and info.doc


def test_list_models_filters():
    assert {m.name for m in list_models(group="short-text")} == {"GSDMM", "PT", "BTM"}
    emb = {m.name for m in list_models(brings="embeddings")}
    assert {"BERTopic", "Top2Vec", "ETM", "FASTopic"} <= emb
    assert all(m.inference == "variational" for m in list_models(inference="variational"))
    assert len(list_models()) == len(REGISTRY)
    with pytest.raises(ValueError):
        list_models(group="not-a-group")


def test_markdown_table_renders_all_models():
    table = markdown_table(by_group=True)
    for name in REGISTRY:
        assert f"`{name}`" in table


def test_impl_map_covers_registry_and_paths_exist():
    # The contributor implementation map (#381): IMPL must cover exactly the
    # registry, and every source/binding/validation path it names and every Cargo
    # feature it names must exist. validate_impl() returns a list of problems.
    problems = validate_impl()
    assert not problems, "IMPL problems:\n  " + "\n  ".join(problems)


def test_impl_map_covers_registry_one_to_one():
    assert set(IMPL) == set(REGISTRY), (
        f"IMPL vs REGISTRY differ: "
        f"only in IMPL={sorted(set(IMPL) - set(REGISTRY))}, "
        f"only in REGISTRY={sorted(set(REGISTRY) - set(IMPL))}"
    )


def test_impl_markdown_table_renders_all_models():
    table = impl_markdown_table(by_group=True)
    for name in REGISTRY:
        assert f"`{name}`" in table


def test_model_map_page_in_sync_with_registry():
    # The generated block in docs/contributing/model-map.md must match the IMPL
    # render; regenerate with scripts/gen_model_tables.py if this fails.
    import pathlib

    BEGIN = ("<!-- BEGIN MODEL MAP (generated from topica.registry IMPL; "
             "edit registry.py, not this block) -->")
    END = "<!-- END MODEL MAP -->"
    expected = f"{BEGIN}\n\n{impl_markdown_table(by_group=True)}\n{END}"
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs/contributing/model-map.md").read_text(encoding="utf-8")
    i, j = text.find(BEGIN), text.find(END)
    assert i != -1 and j != -1, "model-map.md marker comments missing"
    assert text[i:j + len(END)] == expected, (
        "docs/contributing/model-map.md is stale; run scripts/gen_model_tables.py")


def test_readme_and_docs_roster_in_sync_with_registry():
    # The generated blocks in README.md and docs/guides/models.md must match the
    # registry render; regenerate with scripts/gen_model_tables.py if this fails.
    import pathlib

    BEGIN = ("<!-- BEGIN MODEL TABLE (generated from topica.registry; "
             "edit registry.py, not this block) -->")
    END = "<!-- END MODEL TABLE -->"
    expected = f"{BEGIN}\n\n{markdown_table(by_group=True)}\n{END}"
    root = pathlib.Path(__file__).resolve().parent.parent
    for rel in ("README.md", "docs/guides/models.md"):
        text = (root / rel).read_text(encoding="utf-8")
        i, j = text.find(BEGIN), text.find(END)
        assert i != -1 and j != -1, f"{rel}: marker comments missing"
        assert text[i:j + len(END)] == expected, (
            f"{rel} roster is stale; run scripts/gen_model_tables.py")


def test_common_start_is_the_editorial_opening_set():
    # common_start is an editorial flag (where newcomers begin), orthogonal to
    # experimental. Keep the set small and validated (never gated).
    common = {m.name for m in list_models(common_start=True)}
    assert common == {"LDA", "NMF", "STM", "KeyATM", "GSDMM", "BERTopic"}
    assert not any(m.experimental for m in list_models(common_start=True)), (
        "a common starting point must not be experimental/gated")


def test_chooser_references_only_real_models():
    # Every model a chooser row names must exist in the registry (the drift guard
    # that lets the matrix be generated safely) and must not be experimental. Every
    # row is in a known section.
    for r in CHOOSER:
        assert r.section in ("common", "specialized"), f"{r.goal!r}: bad section"
        for name in (r.primary, *( (r.also,) if r.also else () )):
            assert name in REGISTRY, f"chooser row {r.goal!r} names unknown {name!r}"
            assert not REGISTRY[name].experimental, (
                f"chooser row {r.goal!r} points at gated model {name!r}")
    table = chooser_markdown_table()
    for r in CHOOSER:
        assert f"`{r.primary}`" in table


def test_chooser_page_in_sync_with_registry():
    # The generated CHOOSER block in README.md and docs/can-do/index.md must match
    # the render; regenerate with scripts/gen_model_tables.py if this fails.
    import pathlib

    BEGIN = ("<!-- BEGIN CHOOSER (generated from topica.registry CHOOSER; "
             "edit registry.py, not this block) -->")
    END = "<!-- END CHOOSER -->"
    expected = f"{BEGIN}\n\n{chooser_markdown_table()}\n{END}"
    root = pathlib.Path(__file__).resolve().parent.parent
    for rel in ("README.md", "docs/can-do/index.md"):
        text = (root / rel).read_text(encoding="utf-8")
        i, j = text.find(BEGIN), text.find(END)
        assert i != -1 and j != -1, f"{rel}: chooser marker comments missing"
        assert text[i:j + len(END)] == expected, (
            f"{rel} chooser is stale; run scripts/gen_model_tables.py")
