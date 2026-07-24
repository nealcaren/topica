"""Shared fixtures for the topica test suite."""

import pytest
from topica import Corpus


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "parity: slow CLI-parity check; builds the release binaries and "
        "compares binding output against the `train` CLI byte-for-byte.",
    )
    config.addinivalue_line(
        "markers",
        "slow: realistic-scale / high-iteration check (issue #271). Some bugs "
        "(e.g. #270's covariate-keyATM collapse) only appear at scale, so these "
        "fit tens of thousands of documents and are opt-in: run with "
        "`-m slow`; the default suite deselects them.",
    )
    config.addinivalue_line(
        "markers",
        "invariants: reference-free metamorphic / invariant check (issue #420). "
        "Asserts properties that must hold regardless of the correct numeric "
        "answer (finiteness, normalization, sampling-control invariance, SE "
        "stationarity) to cover the blind spot cross-implementation parity leaves.",
    )


# ---------------------------------------------------------------------------
# Two-cluster toy corpus: animal words vs space words (~15 docs each)
# ---------------------------------------------------------------------------

ANIMAL_WORDS = ["cat", "dog", "fish", "cat", "dog"]
SPACE_WORDS = ["planet", "star", "moon", "rocket", "planet"]

ANIMAL_DOCS: list[list[str]] = [list(ANIMAL_WORDS) for _ in range(15)]
SPACE_DOCS: list[list[str]] = [list(SPACE_WORDS) for _ in range(15)]
TOY_DOCS: list[list[str]] = ANIMAL_DOCS + SPACE_DOCS  # 30 docs total


@pytest.fixture(scope="session")
def toy_docs() -> list[list[str]]:
    """30-document two-cluster token lists (15 animal + 15 space)."""
    return TOY_DOCS


@pytest.fixture(scope="session")
def toy_corpus() -> Corpus:
    """Corpus built from toy_docs with no frequency filtering."""
    return Corpus.from_documents(TOY_DOCS)
