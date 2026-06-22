"""Planted self-consistency gold for topica PA (Pachinko Allocation, Li & McCallum 2006) (issue #271, Wave 2).

PA models a super/sub-topic DAG with no external reference; sub-topics legitimately mix the paired blocks so the purity bar is modest, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica pa
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: paired-block corpus, 160 docs, 4 sub-blocks under 2 super-topics (tests/test_model_invariants _fit_pa).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/pa_gold.py               # offline compare against committed gold
    python parity/pa_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "pa",
    "PA (Pachinko Allocation, Li & McCallum 2006)",
    wave2.fit_pa,
    block_fn=wave2.block_of_bw, top_n=5, purity_bar=0.55, coverage_bar=4,
    corpus_desc="paired-block corpus, 160 docs, 4 sub-blocks under 2 super-topics (tests/test_model_invariants _fit_pa)",
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
