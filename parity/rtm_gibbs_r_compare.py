"""Live R `lda` parity for topica's collapsed-Gibbs RTM (#424).

Human-readable end-to-end comparison against the installed R `lda` package's
`rtm.em` (the same collapsed-Gibbs algorithm as topica's `inference="gibbs"`
backend). The committed offline gold is built by `rtm_gibbs_gold.py`; this runs
both on the same fixed planted network and reports topica's aligned topic-word
cosine against R's own two-seed self-consistency floor.

Skips cleanly (exit 0) when Rscript / the `lda` package are unavailable.

    python parity/rtm_gibbs_r_compare.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import rtm_gibbs_gold as gold  # noqa: E402


def main() -> int:
    if not harness.r_available("lda"):
        print("SKIP: Rscript with the R `lda` package not available")
        return 0

    docs, edges, vocab, _groups = gold._planted()
    out = harness.run_rscript(
        gold._driver(),
        {
            "docs.txt": "\n".join(" ".join(d) for d in docs) + "\n",
            "vocab.txt": "\n".join(vocab) + "\n",
            "edges.csv": gold._edge_csv(edges),
        },
        ["phi1.csv", "phi2.csv"],
        timeout=1800,
    )
    phi1 = harness.read_r_beta_csv(out["phi1.csv"], vocab)
    phi2 = harness.read_r_beta_csv(out["phi2.csv"], vocab)
    r_self, _ = harness.align_cosine(phi1, phi2)

    topica_phi, eta = gold._fit_topica(docs, edges, vocab)
    cos, _ = harness.align_cosine(phi1, topica_phi)
    bar = r_self - gold.MARGIN

    print(f"corpus: {len(docs)} docs, {len(edges)} links, |V| = {len(vocab)}")
    print(f"topic-word phi cosine: topica {cos:.4f}  (R self {r_self:.4f}, bar {bar:.4f})")
    print(f"link coefficient (both negative, R's quirk): topica {np.round(eta, 3)}")
    print(f"verdict: {'PASS' if cos >= bar else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
