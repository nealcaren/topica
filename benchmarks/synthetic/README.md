# Synthetic recovery benchmark

A suite of synthetic corpora with **known ground truth**, used to measure what
each topic model recovers. Because the latent structure is planted, we can score
recovery against the truth directly rather than only comparing models to each
other. The narrative walkthrough of the results is in the docs:
[Comparing the models](../../docs/guides/model-comparison.md).

## Design principle: separate the answer key from the prose

A deterministic, seeded sampler assigns every document its latent structure (topic
mixture, covariates, author, the topic of each sentence). That assignment is the
answer key, and no language model touches it. Language models do only the surface
work, turning each planted sentence-topic into fluent council-statement prose. The
corpus reads like real debate while staying fully labeled underneath.

## The corpora (all frozen under `data/`)

| Files | What it plants | Exercises |
|---|---|---|
| `latents.json`, `answer_key.npz`, `out/` | topics, region/party/year covariates, author ideal points, narrative framing | LDA, CTM, STM, DMR, SAGE, DTM, IdealPointTM, GSDMM |
| `out_es/`, `glossary.json` | Spanish translation + bilingual dictionary | InfoCTM (cross-lingual) |
| `mix_*` | documents with a planted 1..5 topic count | admixture vs mixture (θ recovery, wrong-K) |
| `narr_*` | 24-sentence docs with an ordered topic arc | NarrativeTM (trajectory) |
| `hier_*` | two-level domain/topic tree | HLDA (hierarchy) |

The realized text under `out/`, `out_es/`, `hier_out/`, `mix_out/`, `narr_out/`
is committed, so scoring runs with **no language model** required.

## Running it

Reproduce the whole scorecard (needs the dev environment with topica built):

```bash
python benchmarks/synthetic/run_all.py
```

Or run one axis:

```bash
python benchmarks/synthetic/score_mixedness.py     # admixture/mixture crossover
python benchmarks/synthetic/score_ksweep.py        # misspecified-K robustness
python benchmarks/synthetic/score_narrative.py     # NarrativeTM trajectory
```

Each `score_*.py` loads the frozen corpus from `data/`, fits the relevant models,
aligns recovered topics to the planted ones, and prints a scorecard.

## Regenerating a corpus (optional)

Regeneration is a three-step pipeline, only the middle step needs a language model:

1. `python sample_admixture.py` (etc.) writes the latents, answer key, and
   per-batch plan files into `data/`.
2. An LLM realizes each batch's planted sentence-topics into prose (the repo used
   Claude via a fan-out of batch agents; any capable model following the batch
   prompt works). Output goes to `data/out/`.
3. `python score_admixture.py` (etc.) scores recovery.

Only steps 1 and 3 are deterministic. The committed `out/` directories are a
frozen realization so the benchmark is reproducible without step 2.

## A note on interpretation

Synthetic data flatters models whose assumptions match the generator: topics are
well separated, covariate effects are large, noise is mild. Passing here is
necessary, not sufficient. The benchmark is a floor. A method that cannot recover
planted structure under these ideal conditions will not find it in real text,
which is what makes the `HDP` and `HLDA` results (fragile, hyperparameter-driven)
the informative ones.
