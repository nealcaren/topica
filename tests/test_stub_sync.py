"""CI guard: the _topica.pyi stub must not drift from the compiled extension.

For every class exported by topica._topica this test asserts:
  - no public (non-dunder) member on the compiled class is absent from the stub,
  - no stub member is absent from the compiled class.

Member-level names only (not full signature equality, which is too brittle for
PyO3 extension types). This is the guard that would have caught the drift fixed
in issue #108.
"""

import ast
import pathlib
import types

import pytest

import topica._topica as _ext


# Members attached to the compiled classes by the ``topica`` package layer (not by
# the extension), so they are intentionally absent from ``_topica.pyi``. Scoped per
# class so the guard still catches a real Rust binding of the same name drifting on
# any other class:
#   - ``topic_table`` is bound onto every topic-word model in ``topica/__init__.py``
#     so ``m.topic_table()`` mirrors ``m.top_words()`` (issue #758);
#   - ``from_dataframe`` is bound onto ``Corpus`` as a classmethod alias for
#     ``topica.from_dataframe`` (issue #758).
_PACKAGE_ATTACHED_BY_CLASS = {"Corpus": {"from_dataframe"}}
_PACKAGE_ATTACHED_DEFAULT = {"topic_table"}


def _package_attached(class_name: str) -> set:
    return _PACKAGE_ATTACHED_BY_CLASS.get(class_name, _PACKAGE_ATTACHED_DEFAULT)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _stub_public_members(stub_path: pathlib.Path) -> dict[str, set[str]]:
    """Parse the .pyi stub and return {class_name: {public_member_name, ...}}."""
    tree = ast.parse(stub_path.read_text())
    classes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        members: set[str] = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_dunder(item.name):
                    members.add(item.name)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name) and not _is_dunder(item.target.id):
                    members.add(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and not _is_dunder(target.id):
                        members.add(target.id)
        classes[node.name] = members
    return classes


def _compiled_public_members(cls: type) -> set[str]:
    """Return the set of public (non-dunder) member names on a compiled class."""
    return {name for name in dir(cls) if not _is_dunder(name)}


# Locate the stub relative to this test file: tests/ -> project root -> python/topica/
_STUB_PATH = pathlib.Path(__file__).parent.parent / "python" / "topica" / "_topica.pyi"

_STUB_MEMBERS = _stub_public_members(_STUB_PATH)

_COMPILED_CLASSES = {
    name: obj
    for name in dir(_ext)
    if isinstance((obj := getattr(_ext, name)), type) and not _is_dunder(name)
}


@pytest.mark.parametrize("class_name", sorted(_COMPILED_CLASSES))
def test_no_members_missing_from_stub(class_name: str) -> None:
    """Every public member on the compiled class must appear in the stub."""
    compiled = _compiled_public_members(_COMPILED_CLASSES[class_name])
    stub = _STUB_MEMBERS.get(class_name, set())
    missing = compiled - stub - _package_attached(class_name)
    assert not missing, (
        f"{class_name}: compiled members missing from stub: {sorted(missing)}"
    )


@pytest.mark.parametrize("class_name", sorted(_STUB_MEMBERS))
def test_no_bogus_members_in_stub(class_name: str) -> None:
    """Every public member declared in the stub must exist on the compiled class."""
    if class_name not in _COMPILED_CLASSES:
        pytest.skip(f"{class_name} not found in compiled module (stub-only class?)")
    compiled = _compiled_public_members(_COMPILED_CLASSES[class_name])
    stub = _STUB_MEMBERS[class_name]
    bogus = stub - compiled
    assert not bogus, (
        f"{class_name}: stub members absent from compiled class: {sorted(bogus)}"
    )
