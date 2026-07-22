"""The new-model scaffold generator (scripts/new_model.py, issue #386)."""
import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "new_model",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "new_model.py",
)
new_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(new_model)


@pytest.mark.parametrize(
    "name,snake",
    [
        ("MyModel", "my_model"),
        ("FooBarTM", "foo_bar_tm"),
        ("GSDMM", "gsdmm"),
        ("ETM", "etm"),
        ("Wordfish", "wordfish"),
    ],
)
def test_to_snake(name, snake):
    assert new_model.to_snake(name) == snake


def test_rendered_templates_carry_name_and_sentinel():
    for tpl in (new_model.RUST_ALGO, new_model.RUST_BINDING, new_model.PYTEST):
        out = new_model.render(tpl, "MyModel", "my_model")
        assert "MyModel" in out
        assert "SCAFFOLD(MyModel)" in out
        assert "__NAME__" not in out and "__SNAKE__" not in out


def test_algo_template_is_ndarray_free():
    # ndarray is behind the `embeddings` feature; a default-build model file must
    # not import it. The binding converts via vecs_to_arr2 instead.
    out = new_model.render(new_model.RUST_ALGO, "MyModel", "my_model")
    assert "use ndarray" not in out
    assert "Vec<Vec<f64>>" in out


def test_checklist_names_the_wiring_steps():
    text = new_model.checklist("MyModel", "my_model")
    for needle in ("src/lib.rs", "src/python/mod.rs", "__init__.py", "registry.py",
                   "conformance.py", "test_scaffold_guard.py"):
        assert needle in text
