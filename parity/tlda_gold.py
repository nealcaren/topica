"""Planted self-consistency gold for topica TensorLDA (Wave 2).
We fit topica TensorLDA once on a fixed-seed planted corpus and freeze its
topic-word + doc-topic matrices; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes validity invariants.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "tlda",
    "TensorLDA (Kangaslahti et al. 2026)",
    wave2.fit_tlda,
    block_fn=wave2.block_of_bw, top_n=5, purity_bar=0.7, coverage_bar=4,
    corpus_desc="planted K=4 block corpus (240 docs)",
)


def regenerate() -> None:
    wave2.regenerate(SPEC)


def run(verbose: bool = True) -> dict:
    return wave2.run(SPEC, verbose=verbose)


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
