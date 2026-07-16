"""Exact MALLET state-ingest contracts for ``LDA.load_state``.

The sampler endpoint is stochastic across implementations, but MALLET's
``--output-state`` file is a concrete token-level state. Loading that file should
preserve token assignments, hyperparameters, and the smoothed topic-word formula
exactly.
"""

from __future__ import annotations

import gzip
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import topica


pytestmark = pytest.mark.parity


DOCS = [
    ["cat", "dog", "cat"],
    ["dog", "pet", "cat"],
    ["star", "moon", "star"],
    ["moon", "sky", "star"],
    ["tax", "vote", "tax"],
    ["law", "vote", "tax"],
]


def _parse_state(path: Path):
    alpha: list[float] | None = None
    beta: float | None = None
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#alpha"):
                alpha = [float(x) for x in line.split(":", 1)[1].split()]
            elif line.startswith("#beta"):
                beta = float(line.split(":", 1)[1])
            elif line.startswith("#"):
                continue
            else:
                doc, source, pos, typeindex, word, topic = line.split()
                rows.append((int(doc), source, int(pos), int(typeindex), word, int(topic)))
    assert alpha is not None and beta is not None
    return alpha, beta, rows


@pytest.fixture(scope="module")
def mallet_state_fixture(tmp_path_factory):
    mallet = shutil.which("mallet")
    if mallet is None:
        pytest.skip("mallet CLI not installed")

    d = tmp_path_factory.mktemp("mallet-state")
    txt = d / "docs.txt"
    mallet_file = d / "docs.mallet"
    state = d / "state.gz"
    diag = d / "diagnostics.xml"
    txt.write_text("\n".join(f"doc{i}\t{' '.join(doc)}" for i, doc in enumerate(DOCS)) + "\n")

    subprocess.run(
        [
            mallet,
            "import-file",
            "--input",
            str(txt),
            "--output",
            str(mallet_file),
            "--keep-sequence",
            "--token-regex",
            r"\S+",
            "--line-regex",
            r"^(\S+)\t(.*)$",
            "--name",
            "1",
            "--data",
            "2",
            "--label",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            mallet,
            "train-topics",
            "--input",
            str(mallet_file),
            "--num-topics",
            "3",
            "--num-iterations",
            "50",
            "--random-seed",
            "1",
            "--optimize-interval",
            "0",
            "--output-state",
            str(state),
            "--diagnostics-file",
            str(diag),
            "--show-topics-interval",
            "0",
            "--num-top-words",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"state": state, "diagnostics": diag}


def test_load_state_preserves_mallet_token_assignments(mallet_state_fixture, tmp_path) -> None:
    state = mallet_state_fixture["state"]
    alpha, beta, rows = _parse_state(state)

    model = topica.models.LDA.load_state(str(state))
    reemitted = tmp_path / "reemitted.gz"
    model.save_state(str(reemitted))
    alpha2, beta2, rows2 = _parse_state(reemitted)

    assert alpha2 == alpha
    assert beta2 == beta
    # The source column is not part of the mathematical state; topica assigns
    # synthetic doc names when reconstructing a MALLET state whose source is NA.
    state_without_source = [(d, p, typeindex, word, topic) for d, _, p, typeindex, word, topic in rows]
    reemitted_without_source = [
        (d, p, typeindex, word, topic) for d, _, p, typeindex, word, topic in rows2
    ]
    assert reemitted_without_source == state_without_source


def test_load_state_topic_word_matches_mallet_state_formula(mallet_state_fixture) -> None:
    state = mallet_state_fixture["state"]
    _, beta, rows = _parse_state(state)
    model = topica.models.LDA.load_state(str(state))

    vocab_by_id = [word for _, word in sorted({(typeindex, word) for _, _, _, typeindex, word, _ in rows})]
    assert model.vocabulary == vocab_by_id

    k = model.num_topics
    v = len(vocab_by_id)
    counts = np.zeros((k, v), dtype=float)
    tokens = np.zeros(k, dtype=float)
    for _, _, _, typeindex, _, topic in rows:
        counts[topic, typeindex] += 1
        tokens[topic] += 1
    expected = (counts + beta) / (tokens[:, None] + v * beta)
    np.testing.assert_allclose(model.topic_word, expected, rtol=0, atol=1e-12)


def test_load_state_diagnostics_overlap_mallet_xml(mallet_state_fixture) -> None:
    state = mallet_state_fixture["state"]
    diag_xml = mallet_state_fixture["diagnostics"]
    model = topica.models.LDA.load_state(str(state))
    got = {int(row["topic"]): row for row in model.diagnostics(n=5)}

    root = ET.parse(diag_xml).getroot()
    for topic in root.findall("topic"):
        topic_id = int(topic.attrib["id"])
        assert got[topic_id]["tokens"] == int(float(topic.attrib["tokens"]))
        assert math.isclose(
            got[topic_id]["document_entropy"],
            float(topic.attrib["document_entropy"]),
            abs_tol=5e-4,
        )


def test_mallet_diagnostics_xml_counts_match_state(mallet_state_fixture) -> None:
    """Pin MALLET's XML word counts/probabilities to its own token state.

    This guards the fixture and clarifies that topica's diagnostics are not a
    byte-for-byte MALLET XML clone for every field.
    """

    state = mallet_state_fixture["state"]
    diag_xml = mallet_state_fixture["diagnostics"]
    _, _, rows = _parse_state(state)
    topic_word_counts = Counter((topic, word) for _, _, _, _, word, topic in rows)
    topic_tokens = Counter(topic for _, _, _, _, _, topic in rows)

    root = ET.parse(diag_xml).getroot()
    for topic in root.findall("topic"):
        topic_id = int(topic.attrib["id"])
        tokens = topic_tokens[topic_id]
        for word in topic.findall("word"):
            text = word.text
            count = topic_word_counts[(topic_id, text)]
            assert int(word.attrib["count"]) == count
            assert math.isclose(float(word.attrib["prob"]), count / tokens, abs_tol=5e-6)

