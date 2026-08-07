"""Live sanity check for `topica.llm.judge` (Zheng et al. 2025, App. G) against a real
LLM backend.

Fits a few models at different K on a bundled corpus and runs the pairwise topic
judge, printing the Bradley-Terry -> Elo leaderboard with bootstrap CIs. The judge is
`llm-bounded`, so this is a measurement (with model noise), not a bit-exact gate; it is
NOT run in CI. It skips cleanly when no LLM backend is configured.

Run:  VIRTUAL_ENV=.venv-dev .venv-dev/bin/python parity/llm_judge_live.py

Config via env: LLM_EVAL_MODEL (default an open model on openrouter),
LLM_JUDGE_COMPARISONS (default 30, small so a live run is cheap).
"""

from __future__ import annotations

import os

MODEL = os.environ.get("LLM_EVAL_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
N_COMPARISONS = int(os.environ.get("LLM_JUDGE_COMPARISONS", "30"))


def available() -> bool:
    try:
        import topica  # noqa: F401
    except Exception:
        return False
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def main() -> None:
    if not available():
        print("SKIP: no LLM backend configured (set OPENROUTER_API_KEY / OPENAI_API_KEY "
              "/ ANTHROPIC_API_KEY) or topica not importable.")
        return

    import topica

    # A small real corpus with clear structure. ng20 (5 groups) is bundled.
    bunch = topica.datasets.load_ng20_minilm()
    import re

    docs = [[w for w in re.split(r"[^a-z0-9]+", t.lower()) if w] for t in bunch.texts]
    docs = docs[:400]  # keep the live run cheap

    models = {}
    for k in (5, 10, 20):
        m = topica.LDA(k, seed=13)
        m.fit(docs, iters=200)
        models[f"lda_k{k}"] = m

    backend = topica.llm.backend(MODEL, temperature=0)
    result = topica.llm.judge(
        models, docs, backend=backend,
        n_comparisons=N_COMPARISONS, representation="summary", seed=13,
    )
    print(f"model={MODEL}  comparisons/pair={N_COMPARISONS}  docs={len(docs)}")
    print(result.summary())
    print("\nranking (best first):", result.ranking())
    # A weak, honest expectation: the K matched to the 5-group structure should not be
    # dead last. This is a smell test, not a gate (the judge is llm-bounded).
    if result.ranking()[-1] == "lda_k5":
        print("NOTE: lda_k5 ranked last — inspect result.comparisons for why.")


if __name__ == "__main__":
    main()
