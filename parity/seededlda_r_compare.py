"""Live R `seededlda` parity for topica SeededLDA (#456).

Human-readable end-to-end comparison against the installed koheiw/`seededlda`
package (the committed offline gold is built by ``seededlda_gold.py``). It runs
R seededlda and topica on the same fixed poliblog subsample and reports:

  * the exact seed-mass match: topica's ``seed_prior_matrix`` vs R's
    ``seededlda:::tfm`` (`count * weight * 100`), and
  * the aligned seeded topic-word cosine of topica (``seed_prior="frequency"``)
    against R, relative to R's own two-seed noise floor.

Skips cleanly (exit 0) when Rscript / seededlda / quanteda are unavailable.

    python parity/seededlda_r_compare.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import seededlda_gold as gold  # noqa: E402


def main() -> int:
    if not harness.r_available("seededlda") or not harness.r_available("quanteda"):
        print("SKIP: Rscript with seededlda + quanteda not available")
        return 0

    docs, _rating, _day, vocab = harness.poliblog_corpus(
        n_docs=gold.N_DOCS, seed=gold.CORPUS_SEED
    )
    out = harness.run_rscript(
        gold._driver(),
        {"docs.txt": harness.docs_to_lines(docs)},
        ["phi1.csv", "phi2.csv", "tfm.csv"],
        timeout=1800,
    )
    phi1 = harness.read_r_beta_csv(out["phi1.csv"], vocab)
    phi2 = harness.read_r_beta_csv(out["phi2.csv"], vocab)
    r_tfm = gold._parse_tfm(out["tfm.csv"], vocab)

    kw = slice(0, gold.N_SEEDED)
    r_self, _ = harness.align_cosine(phi1[kw], phi2[kw])

    topica_phi, topica_seed_mat = gold._fit_topica(docs, vocab)
    cos, _ = harness.align_cosine(phi1[kw], topica_phi)
    tfm_max_abs = float(np.abs(topica_seed_mat - r_tfm).max())
    bar = r_self - gold.MARGIN

    print(f"corpus: {len(docs)} docs, |V| = {len(vocab)}")
    print(f"seed-mass (topica seed_prior_matrix vs R tfm): max |Δ| = {tfm_max_abs:.2e} "
          f"({'EXACT' if tfm_max_abs < 1e-6 else 'MISMATCH'})")
    print(f"seeded phi cosine: topica {cos:.4f}  (R self {r_self:.4f}, bar {bar:.4f})")
    print(f"verdict: {'PASS' if (tfm_max_abs < 1e-6 and cos >= bar) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
