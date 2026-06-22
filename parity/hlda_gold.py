"""Planted self-consistency gold for topica HLDA (Blei et al. 2003, nested CRP tree) (issue #271, Wave 2).

HLDA learns a topic TREE, not a fixed-K simplex; recovery is scored on the effective number of distinct doc paths, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica hlda
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: shared-stopword + K=4 leaf-block corpus, 300 docs, depth-2 nCRP tree (tests/test_model_invariants _fit_hlda).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/hlda_gold.py               # offline compare against committed gold
    python parity/hlda_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "hlda",
    "HLDA (Blei et al. 2003, nested CRP tree)",
    wave2.fit_hlda,
    is_hlda=True, eff_paths_bar=2.0,
    corpus_desc="shared-stopword + K=4 leaf-block corpus, 300 docs, depth-2 nCRP tree (tests/test_model_invariants _fit_hlda)",
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
