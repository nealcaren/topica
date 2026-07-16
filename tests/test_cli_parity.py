"""CLI-parity regression test.

The Python binding and the bundled `train` CLI (`src/bin/train.rs`) share one
sampler — same PCG (`Pcg64Mcg`) seed, same initialize/sample/optimize/average
order, same TSV writers. So for an identical corpus and identical parameters the
binding must reproduce topica's own `train` CLI **byte-for-byte**. This test pins
that internal guarantee so the library and the CLI can't silently drift. (It is
not a parity check against MALLET or RustMallet, which use different RNGs and so
are not byte-identical to topica.)

It builds the release binaries once, runs `preprocess` + `train`, runs the
binding on the same corpus with matched parameters, and asserts the output
TSVs are identical. Skips cleanly when cargo or the sample data is unavailable
(e.g. an sdist-only checkout).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from topica.models import LDA
from topica import Corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = REPO_ROOT / "examples" / "sample-docs.txt"
RELEASE_DIR = REPO_ROOT / "target" / "release"

# Parameters used for BOTH pipelines. Defaults are aligned between LDA and
# train.rs, but we pin every value explicitly so the test documents exactly
# what must match — including the hyperparameter-optimization schedule, which
# is exercised here (iterations > burn_in, optimize_interval > 0).
SEED = 42
NUM_TOPICS = 5
ITERATIONS = 300
BURN_IN = 100
OPTIMIZE_INTERVAL = 50
NUM_SAMPLES = 3
SAMPLE_INTERVAL = 10
BETA = 0.01

pytestmark = pytest.mark.parity


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def cli_binaries() -> dict[str, Path]:
    """Build the release CLI binaries once; skip if cargo is unavailable."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo not available — cannot build CLI binaries for parity check")
    if not SAMPLE_DOCS.exists():
        pytest.skip(f"sample data missing: {SAMPLE_DOCS}")

    build = subprocess.run(
        ["cargo", "build", "--release", "--bins"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"cargo build failed:\n{build.stderr[-2000:]}")

    bins = {name: RELEASE_DIR / name for name in ("preprocess", "train")}
    for name, path in bins.items():
        if not path.exists():
            pytest.skip(f"expected binary not built: {path}")
    return bins


def test_python_matches_train_cli(cli_binaries: dict[str, Path], tmp_path: Path) -> None:
    """Binding output must be byte-identical to the `train` CLI."""
    corpus_path = tmp_path / "sample.corp"
    cli_tw = tmp_path / "cli_topic_word.tsv"
    cli_dt = tmp_path / "cli_doc_topic.tsv"
    py_tw = tmp_path / "py_topic_word.tsv"
    py_dt = tmp_path / "py_doc_topic.tsv"

    # 1. preprocess raw text -> binary corpus (the shared input to both sides).
    subprocess.run(
        [str(cli_binaries["preprocess"]),
         "--input", str(SAMPLE_DOCS), "--output", str(corpus_path)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    assert corpus_path.exists()

    # 2. upstream CLI training.
    subprocess.run(
        [str(cli_binaries["train"]),
         "--corpus", str(corpus_path),
         "--num-topics", str(NUM_TOPICS),
         "--iterations", str(ITERATIONS),
         "--burn-in", str(BURN_IN),
         "--optimize-interval", str(OPTIMIZE_INTERVAL),
         "--num-samples", str(NUM_SAMPLES),
         "--sample-interval", str(SAMPLE_INTERVAL),
         "--beta", str(BETA),
         "--seed", str(SEED),
         "--topic-word", str(cli_tw),
         "--doc-topic", str(cli_dt),
         "--show-topics-interval", "0"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )

    # 3. Python binding on the SAME corpus with matched parameters.
    corpus = Corpus.load(str(corpus_path))
    model = LDA(
        num_topics=NUM_TOPICS,
        beta=BETA,
        optimize_interval=OPTIMIZE_INTERVAL,
        burn_in=BURN_IN,
        seed=SEED,
    )
    model.fit(
        corpus,
        iters=ITERATIONS,
        num_samples=NUM_SAMPLES,
        sample_interval=SAMPLE_INTERVAL,
    )
    model.save_topic_word(str(py_tw))
    model.save_doc_topic(str(py_dt))

    # 4. byte-for-byte equality.
    assert _md5(py_tw) == _md5(cli_tw), "topic-word matrix differs from train CLI"
    assert _md5(py_dt) == _md5(cli_dt), "doc-topic matrix differs from train CLI"
    # Belt-and-suspenders: exact byte compare (also catches zero-length matches).
    assert py_tw.read_bytes() == cli_tw.read_bytes()
    assert py_dt.read_bytes() == cli_dt.read_bytes()
