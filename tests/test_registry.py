"""The model registry stays in one-to-one correspondence with the exported
model classes, and its fields use the allowed vocabularies (#182)."""
import inspect

import pytest

import topica
from topica.registry import GROUPS, REGISTRY, ModelInfo, list_models, markdown_table

# Exported classes that are NOT topic models (so not in the registry).
_NON_MODEL_CLASSES = {"Heldout", "HeldoutResult", "Corpus", "ModelInfo"}

_BRINGS = {"text", "embeddings", "metadata", "seeds", "labels", "times", "llm"}
_INFERENCE = {"gibbs", "variational", "vae", "optimal-transport", "clustering",
              "matrix-factorization", "svd", "prompting"}
_DETERMINISM = {"bit-exact", "seed-reproducible", "llm-bounded"}


def _exported_model_classes() -> set[str]:
    names = set()
    for name in topica.__all__:
        obj = getattr(topica, name)
        if (inspect.isclass(obj) and hasattr(obj, "fit")
                and name not in _NON_MODEL_CLASSES):
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
    assert {m.name for m in list_models(group="short-text")} == {"GSDMM", "PT"}
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
        text = (root / rel).read_text()
        i, j = text.find(BEGIN), text.find(END)
        assert i != -1 and j != -1, f"{rel}: marker comments missing"
        assert text[i:j + len(END)] == expected, (
            f"{rel} roster is stale; run scripts/gen_model_tables.py")
