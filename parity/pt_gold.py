"""Planted self-consistency gold for topica PT (Pseudo-document Topic model / Zuo et al. 2016) (issue #271, Wave 2).

PT has no external reference in topica's stack, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica pt
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: short (3-token) K=4 planted-block corpus, 300 docs, 50 pseudo-documents.

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/pt_gold.py               # offline compare against committed gold
    python parity/pt_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "pt",
    "PT (Pseudo-document Topic model / Zuo et al. 2016)",
    wave2.fit_pt,
    block_fn=wave2.block_of_bw, top_n=4, purity_bar=0.9, coverage_bar=4,
    corpus_desc="short (3-token) K=4 planted-block corpus, 300 docs, 50 pseudo-documents",
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
