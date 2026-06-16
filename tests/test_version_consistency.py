"""Guard against version drift across the files that declare topica's version.

``Cargo.toml`` (the Rust crate / what the built wheel reports), ``pyproject.toml``
(the PyPI package), and ``CITATION.cff`` (the academic citation) must all agree.
CITATION.cff in particular is hand-maintained and has silently drifted behind
real releases before; this test fails the build when any of the three disagree.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _section_version(path: Path, section: str) -> str:
    """The ``version = "X"`` inside a TOML ``[section]`` (first match)."""
    text = path.read_text(encoding="utf-8")
    # Slice from the section header to the next top-level header.
    start = text.index(f"[{section}]")
    rest = text[start + len(section) + 2:]
    end = rest.find("\n[")
    block = rest if end == -1 else rest[:end]
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', block, re.MULTILINE)
    assert m, f"no version found in [{section}] of {path.name}"
    return m.group(1)


def _cff_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', text, re.MULTILINE)
    assert m, f"no version found in {path.name}"
    return m.group(1)


def test_versions_agree():
    cargo = _section_version(ROOT / "Cargo.toml", "package")
    pyproject = _section_version(ROOT / "pyproject.toml", "project")
    citation = _cff_version(ROOT / "CITATION.cff")
    versions = {"Cargo.toml": cargo, "pyproject.toml": pyproject, "CITATION.cff": citation}
    assert len(set(versions.values())) == 1, (
        "version drift across release files; bump them together: "
        + ", ".join(f"{f}={v}" for f, v in versions.items())
    )
