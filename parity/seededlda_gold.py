"""Planted self-consistency gold for topica SeededLDA (Jagarlamudi et al. 2012) (issue #271, Wave 2).

SeededLDA has no external reference; recovery checks each seeded topic locks onto its planted block, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica seededlda
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: planted K=4 block corpus, 300 docs, with one seed set per block (tests/test_model_invariants _block_keywords).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/seededlda_gold.py               # offline compare against committed gold
    python parity/seededlda_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "seededlda",
    "SeededLDA (Jagarlamudi et al. 2012)",
    wave2.fit_seededlda,
    block_fn=wave2.block_of_bw, top_n=5, purity_bar=0.9, coverage_bar=4,
    corpus_desc="planted K=4 block corpus, 300 docs, with one seed set per block (tests/test_model_invariants _block_keywords)",
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
