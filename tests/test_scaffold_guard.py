"""A registered model may not carry a scaffold placeholder (issue #386).

`scripts/new_model.py` stamps every generated template file with a
``SCAFFOLD(<Name>)`` sentinel and wires nothing in, so an un-finished model is
inert (not compiled, not exported, not registered). This guard is the other
half of that contract: once a model IS registered, none of the files that
implement it may still contain the sentinel. So you cannot ship a half-finished
placeholder — the moment a scaffolded model enters the registry, CI fails until
every SCAFFOLD marker is gone.

It reads the implementation map (`topica.registry.IMPL`, issue #381) to know
which files belong to each registered model, so it needs no separate list.
"""
import pathlib

from topica.registry import IMPL

SENTINEL = "SCAFFOLD"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _files_for(info) -> list[str]:
    paths = [info.source]
    if info.binding:
        paths.append(info.binding)
    paths += [p.strip() for p in info.validation.split(",")]
    return paths


def test_no_registered_model_has_a_scaffold_sentinel():
    offenders = []
    for name, info in IMPL.items():
        for rel in _files_for(info):
            path = ROOT / rel
            if not path.exists():
                continue  # existence is checked by test_registry.py
            text = path.read_text(encoding="utf-8")
            if SENTINEL in text:
                offenders.append(f"{name}: {rel} still contains a {SENTINEL} marker")
    assert not offenders, (
        "registered models with unfinished scaffold placeholders:\n  "
        + "\n  ".join(offenders)
    )
