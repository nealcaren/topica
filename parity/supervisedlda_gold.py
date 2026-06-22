"""Planted self-consistency gold for topica SupervisedLDA (Blei & McAuliffe 2007) (issue #271, Wave 2).

SupervisedLDA has no external reference; recovery checks the two response-aligned blocks separate, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica supervisedlda
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: supervised 2-block corpus, 200 docs, continuous response driven by block-0 prevalence (tests/test_model_invariants _supervised_corpus).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/supervisedlda_gold.py               # offline compare against committed gold
    python parity/supervisedlda_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "supervisedlda",
    "SupervisedLDA (Blei & McAuliffe 2007)",
    wave2.fit_supervisedlda,
    block_fn=wave2.block_of_supervised, top_n=6, purity_bar=0.9, coverage_bar=2,
    corpus_desc="supervised 2-block corpus, 200 docs, continuous response driven by block-0 prevalence (tests/test_model_invariants _supervised_corpus)",
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
