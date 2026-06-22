"""Planted self-consistency gold for topica ETM (Dieng, Ruiz & Blei 2020, embedded topic model) (issue #271, Wave 2).

ETM (variational-EM here) has no external reference in topica's stack, so this is a PLANTED self-consistency / planted-recovery gold, NOT a
cross-implementation one (mirrors ``parity/sage_gold.py``). We fit topica etm
ONCE on a fixed-seed planted corpus and freeze its topic-word + doc-topic
matrices and the planted layout; the test then refits and asserts the fit
reproduces the frozen solution in cosine, recovers the planted structure, and
passes the Wave 0 validity invariants. A shuffled matrix fails the gate.

Corpus: planted K=4 block corpus (240 docs) + planted block-axis word embeddings (frozen in npz).

The fit contract is reused verbatim from ``tests/test_model_invariants.py`` (its
``FIT_ADAPTERS`` already encodes the right corpus / covariates / embeddings).

Two phases::

    python parity/etm_gold.py               # offline compare against committed gold
    python parity/etm_gold.py --regenerate  # fit topica once, write the gold
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wave2  # noqa: E402

SPEC = wave2.Spec(
    "etm",
    "ETM (Dieng, Ruiz & Blei 2020, embedded topic model)",
    wave2.fit_etm,
    block_fn=wave2.block_of_bw, top_n=5, purity_bar=0.7, coverage_bar=3,
    embeddings=wave2.emb_etm,
    corpus_desc="planted K=4 block corpus (240 docs) + planted block-axis word embeddings (frozen in npz)",
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
