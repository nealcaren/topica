"""Planted self-consistency gold for topica DETM (Dieng, Ruiz & Blei 2019, dynamic embedded) (issue #271, Wave 2).

DETM has no external reference; its softmax topic-word rows are near-flat so the shuffle check leans on jaccard, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica detm
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: planted K=4 block corpus (block=6, 240 docs, 4 time slices) + planted block-axis word embeddings (frozen in npz).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/detm_gold.py               # offline compare against committed gold
    python parity/detm_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "detm",
    "DETM (Dieng, Ruiz & Blei 2019, dynamic embedded)",
    wave2.fit_detm,
    block_fn=wave2.block_of_bw, top_n=5, purity_bar=0.6, coverage_bar=4,
    embeddings=wave2.emb_detm,
    corpus_desc="planted K=4 block corpus (block=6, 240 docs, 4 time slices) + planted block-axis word embeddings (frozen in npz)",
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
