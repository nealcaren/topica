"""Export a benchmark subset of the Congress speech corpus to a prepped CSV.

The Congress corpus (~96k already-tokenized speeches with party and congress-
session metadata) lives in the separate ECTM-paper project, not in this repo.
This script reads it and writes a seeded ``CONGRESS_N``-document subsample to
``benchmarks/congress_prepped.csv`` (gitignored, like poliblog5k_prepped.csv), so
the medium-corpus STM timing point (``speed_vs_size.py``) is reproducible without
vendoring a 200MB pickle.

It is a no-op unless the source pickle is present; point ``CONGRESS_PKL`` at it::

    CONGRESS_PKL=~/Documents/GitHub/ECTM-paper/analysis/congress/congress_prepped.pkl \\
      python benchmarks/export_congress.py            # default 25,000 docs
    CONGRESS_N=30000 python benchmarks/export_congress.py

Output columns: ``text`` (space-joined tokens), ``party`` (D/R), ``congress``
(session number, the continuous covariate for the day-style spline).
"""

from __future__ import annotations

import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKL = os.path.expanduser(
    "~/Documents/GitHub/ECTM-paper/analysis/congress/congress_prepped.pkl")
PKL = os.path.expanduser(os.environ.get("CONGRESS_PKL", DEFAULT_PKL))
OUT = os.environ.get("CONGRESS_CSV", os.path.join(HERE, "congress_prepped.csv"))
N = int(os.environ.get("CONGRESS_N", "25000"))
SEED = int(os.environ.get("CONGRESS_SEED", "0"))


def main() -> int:
    if not os.path.exists(PKL):
        print(f"SKIP: source corpus not found at {PKL} "
              "(set CONGRESS_PKL to the ECTM congress_prepped.pkl)")
        return 0
    import numpy as np

    with open(PKL, "rb") as f:
        d = pickle.load(f)
    docs, party, congress = d["docs"], d["party"], d["congress"]
    n_total = len(docs)
    rng = np.random.default_rng(SEED)
    n = min(N, n_total)
    idx = rng.choice(n_total, size=n, replace=False)
    idx.sort()  # keep input order for reproducibility / readability

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "party", "congress"])
        for i in idx:
            toks = docs[i]
            if not toks:
                continue
            w.writerow([" ".join(toks), party[i], congress[i]])
    sessions = sorted(set(congress[i] for i in idx))
    print(f"wrote {OUT}: {n} of {n_total} docs "
          f"(party D/R, congress sessions {sessions[0]}-{sessions[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
