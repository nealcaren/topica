"""Human-correlation check for `llm_coherence` against the Hoyle et al. (2021) human
topic-rating gold, replicating Stammbach et al. (2023).

Stammbach et al. show an LLM prompted with the crowd-worker instructions correlates
with human topic *ratings* better than NPMI / c_v. This script reproduces that on the
public Hoyle 2021 data (300 topics: wikitext + nytimes, each fit by mallet / dvae /
etm, 50 topics each, with mean human ratings and c_npmi already computed). For each
topic it runs `topica.llm.coherence` and reports the Spearman correlation of the LLM
ratings vs the mean human ratings, against the NPMI baseline.

Skips cleanly when the data cannot be fetched or no LLM backend is configured. The
LLM is `llm-bounded`, so this is a measurement (with model noise), not a bit-exact
gate; it is NOT run in CI. Configure the model/limit via the constants below.

Run:  VIRTUAL_ENV=.venv-dev .venv-dev/bin/python parity/llm_coherence_compare.py
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/ahoho/topics/dev/data/human/all_data/all_data.json"
CACHE = "/tmp/hoyle_human_all_data.json"
MODEL = os.environ.get("LLM_EVAL_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
N_WORDS = 10
PER_COMBO = int(os.environ.get("LLM_EVAL_LIMIT", "0")) or None  # None = all 50/combo


def available() -> bool:
    try:
        import topica  # noqa: F401
    except Exception:
        return False
    # an LLM backend must be reachable
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        return False
    return True


def _load():
    if not os.path.exists(CACHE):
        try:
            urllib.request.urlretrieve(DATA_URL, CACHE)
        except Exception:
            return None
    with open(CACHE) as f:
        return json.load(f)


def _spearman(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        # rank-correlation without scipy
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())) or 1.0
        return float((ra * rb).sum() / denom)


def run(verbose: bool = True) -> dict:
    import topica

    data = _load()
    if data is None:
        raise RuntimeError("could not fetch the Hoyle human data")
    call = topica.llm.backend(MODEL, temperature=0)

    llm_all, human_all, npmi_all = [], [], []
    rows = []
    for dataset in ("wikitext", "nytimes"):
        for model_name in ("mallet", "dvae", "etm"):
            entry = data[dataset][model_name]
            topics = entry["topics"]
            human = entry["metrics"]["ratings_scores_avg"]
            npmi = entry["metrics"]["c_npmi_10_full"]
            if PER_COMBO:
                topics, human, npmi = topics[:PER_COMBO], human[:PER_COMBO], npmi[:PER_COMBO]
            word_lists = [t[:N_WORDS] for t in topics]
            llm = topica.llm.coherence(word_lists, call=call, n_words=N_WORDS)
            ok = ~np.isnan(llm)
            llm, human, npmi = llm[ok], np.array(human)[ok], np.array(npmi)[ok]
            rows.append((f"{dataset}/{model_name}", _spearman(llm, human), _spearman(npmi, human), len(llm)))
            llm_all += list(llm); human_all += list(human); npmi_all += list(npmi)

    out = {
        "llm_spearman": _spearman(llm_all, human_all),
        "npmi_spearman": _spearman(npmi_all, human_all),
        "n_topics": len(llm_all),
        "per_combo": rows,
    }
    if verbose:
        print(f"llm_coherence vs human ratings ({out['n_topics']} topics, model={MODEL}):")
        print(f"  {'combo':18s} {'LLM':>7s} {'NPMI':>7s}  n")
        for name, ls, ns, n in rows:
            print(f"  {name:18s} {ls:7.3f} {ns:7.3f}  {n}")
        print(f"  {'CONCAT':18s} {out['llm_spearman']:7.3f} {out['npmi_spearman']:7.3f}  {out['n_topics']}")
        print(f"\n  llm_coherence Spearman {out['llm_spearman']:.3f} vs NPMI {out['npmi_spearman']:.3f} "
              f"(paper: LLM ~0.6 > NPMI ~0.45)")
    return out


if __name__ == "__main__":
    if not available():
        print("topica / an LLM API key not available; skipping llm_coherence parity.")
    else:
        r = run()
        assert r["llm_spearman"] > r["npmi_spearman"], (
            f"llm_coherence ({r['llm_spearman']:.3f}) did not beat NPMI "
            f"({r['npmi_spearman']:.3f})")
        print("\nOK: llm_coherence correlates with human ratings better than NPMI.")
