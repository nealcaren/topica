"""Planted self-consistency gold for topica GSDMM (Yin & Wang 2014, short-text DMM) (issue #271, Wave 2).

GSDMM has no external reference; it auto-prunes empty clusters so the frozen topic count is what the fit settled on, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica gsdmm
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: short (3-token) K=4 planted-block corpus, 300 docs; GSDMM starts at 15 topics and collapses.

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/gsdmm_gold.py               # offline compare against committed gold
    python parity/gsdmm_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "gsdmm",
    "GSDMM (Yin & Wang 2014, short-text DMM)",
    wave2.fit_gsdmm,
    block_fn=wave2.block_of_bw, top_n=4, purity_bar=0.8, coverage_bar=4,
    corpus_desc="short (3-token) K=4 planted-block corpus, 300 docs; GSDMM starts at 15 topics and collapses",
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
