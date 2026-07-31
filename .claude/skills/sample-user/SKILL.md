---
name: sample-user
description: Run a two-agent "sample user" usability audit of a topica model's end-to-end analysis workflow, from the seat of a first-time domain-expert researcher. Use when a user wants to evaluate how usable a model is in practice ("do a sample-user run on LDA", "sample-user the new keyATM covariate path", "have sample users try STM"), or as the standard post-merge acceptance gate for any newly added model. Dispatches one Claude and one Gemini agent, each role-playing a first-time computational social scientist who runs the whole workflow (vocabulary -> choosing K -> fit -> robustness -> validation -> covariate effects -> reporting) on a bundled sample dataset, then synthesizes their friction into a ranked, verified GitHub issue. Complements add-topic-model (which ships a model); this audits whether a real user can actually use it.
---

# Sample-user usability audit

Find the friction a real researcher hits when they use a topica model to write a
paper — the confusing API, the silent footgun that puts a wrong number in a table,
the missing "start here", the crash on a reasonable mental model. The deliverable
is a **ranked, reproduced GitHub issue** of concrete fixes, plus a synthesis note.

This is a usability/acceptance gate, not a correctness review of the model math
(that is the dual-review gate in `add-topic-model`). Run it on any model — an
existing one on request, or every newly added model as the last acceptance step.

The value comes from **two independent first-time users from different model
families** (Claude + Gemini) hitting the same library blind, then triangulating.
Overlap = a real problem; divergence = surface area one user didn't reach.

## Why two agents, and the Gemini execution wrinkle

- **Claude sample user** runs as a `general-purpose` subagent with Bash — it
  *actually executes* every step and hits real errors live.
- **Gemini sample user** runs through the `antigravity` CLI (`agy`), which is
  **read-only in headless mode** — it can read the docs and *write* the workflow
  script, but cannot run `python`. So the pattern is: **Gemini writes the full
  workflow from the docs; you (the orchestrator) execute it** in a scratch dir.
  The friction is exactly where its reasonable, doc-grounded API guesses collide
  with reality — an authentic "does a competent user's mental model match topica"
  signal. Feed the execution results back to Gemini for its reactions + final report.

Keep them on **different bundled datasets** to diversify coverage.

## Setup facts (verified working)

- Run against the installed dev library — the genuine installed-user experience,
  no `PYTHONPATH`:
  `cd <repo> && VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python script.py`
  House import is `import topica` (no alias).
- Sample datasets (`topica.datasets.load_*`), all load offline/cached:
  - `load_gadarian()` — 341 immigration-survey responses; `treatment`, `pid_rep`; raw text (needs stopwords). Small; good for a fast covariate run.
  - `load_poliblog()` — 2000 political blogs; `rating` (Lib/Con), `day`, `blog`; text pre-tokenized+stemmed. Good for search_k + stability.
  - `load_dubois()` — 704 articles from The Crisis (1910–1934); `year`, `decade`, `author`, `subjects`; raw text. Good for prevalence-over-time.
  - `load_ng20_minilm()` — 20-Newsgroups with precomputed MiniLM vectors; for embedding/neural models (BERTopic/ProdLDA/FASTopic/Top2Vec).
- Give each agent an **isolated scratch dir** and forbid touching the repo tree.
- Compute is real: `search_k` (multi-seed) + a 5-run `ensemble` + STM can exceed a
  2-minute foreground limit. Run the executed script in the **background** and
  read its output file; probe uncertain return shapes with a fast tiny fit instead
  of re-running the whole heavy script per crash.

## Workflow the sample users must walk

Adapt the steps to the model family, but cover the whole arc. For a plain LDA/Gibbs
paper the canonical eight steps are:

1. **Getting started / loading** — import, load data, first-look discoverability
   (`dir(topica)`, `help`), did they know where to start?
2. **Corpus + vocabulary** — tokenize/stopword, choose vocab size
   (`min_doc_freq`/`max_doc_fraction`/`max_features`), inspect vocabulary and counts.
3. **Choosing K** — `search_k`; make a *reviewer-defensible* choice (coherence,
   frontier/elbow, stability). Did the output adjudicate, or just dump numbers?
4. **Fitting** — fit at chosen K; seed/reproducibility, speed, convergence signal.
5. **Robustness / stability** — refit across seeds; `ensemble`/`topic_stability`/
   bootstrap CIs — "are my topics real or seed noise?"
6. **Validation / interpretation** — `label_topics` (prob + FREX), `coherence`,
   `exclusivity`, top documents; can they build the journal's tables/figures?
7. **Covariate effects** — the paper-central question. Estimate prevalence by a
   covariate WITH uncertainty (`estimate_effect`/`predicted_prevalence`). Did the
   docs steer to STM when LDA would do? Was that clear?
8. **Reporting** — assemble a topic table; record provenance (`record_fit` →
   `AnalysisManifest`) for replication.

Family swaps: **STM/DMR/keyATM** → make covariates/prevalence the spine; **neural**
(ProdLDA/CTM/ETM/BERTopic) → use `load_ng20_minilm`, exercise embedding inputs and
`enable_experimental()` if gated; **dynamic** (DTM/dynamic keyATM) → time slices +
over-time prevalence.

## Run it

1. **Pick two datasets and two workflows** (usually the same 8 steps on different
   data). Create two scratch dirs.
2. **Launch the Claude sample user** (`Agent`, `general-purpose`, background) with a
   role (a named first-time computational social scientist), the rules (user not
   developer: rely on `help`/docstrings/docs, do **not** read the Rust/Python source
   to reverse-engineer behavior; never modify the repo), the dataset, the eight
   steps, and the deliverable: a `REPORT.md` = verdict + a severity-ranked friction
   log (each item = step, severity, what happened *with the code and real output*,
   what I expected, suggested fix) + what delighted + top-5 improvements.
3. **Launch the Gemini sample user** with `agy --add-dir="$PWD" --mode=plan
   --model=gemini-3.1-pro-high --print` (see the `antigravity` skill). Ask for
   exactly two sections: the complete `run.py`, and a pre-execution friction log
   (every place the docs left it guessing). It reads real docs under `docs/`.
4. **Execute Gemini's script** in its scratch dir (background, unbuffered `-u`).
   When it crashes on a guessed API, that IS a finding — record it, then either
   patch minimally and re-run, or (faster) probe the remaining uncertain calls'
   real signatures/return shapes with a small fast fit. Collect the full
   guess-vs-reality divergence set.
5. **Feed the execution results back to Gemini** (what worked, what crashed, and the
   real API shapes) and have it write its **final** `REPORT.md` reacting as the user.

## Synthesize, verify, file

- **Deduplicate** across both reports. Cross-agent overlap is your highest-signal
  finding; single-agent items are still real.
- **Rank by severity, correctness first:**
  - **Tier 1 — analytical traps**: a default or shape that leads to a *wrong
    published number* (e.g. an auto-intercept making `coef[0]` the intercept; a
    selector silently returning a grid endpoint; pruning that silently drops the
    corpus's defining words). These matter most — a usability audit's real payoff
    is catching the quiet ones.
  - **Tier 2 — broken/opaque APIs**: crashes, opaque errors, hidden signatures
    (e.g. PyO3 `help(Model.__init__)` printing nothing useful).
  - **Tier 3 — return-shape inconsistencies**: list-vs-dict, tuple-arity, one
    function's shape not matching its siblings.
  - **Tier 4 — discoverability / docs / defaults**: namespace with no "start here",
    param-name divergence from sklearn, under-documented capabilities, stale data
    docstrings, missing convergence signal.
- **Reproduce the top findings yourself** on `main` before writing them up — a
  sample-user report is a lead, not proof. Mark each verified item so the issue is
  credible (`✓verified`).
- **Also record what delighted them** — the things to keep and advertise (both
  users here praised LDA covariate effects with honest uncertainty *without* being
  forced into STM; per-topic ensemble stability CIs; the `record_fit` provenance
  card). Positive signal tells the maintainer what not to break.
- **File one GitHub issue** grouped by the four tiers, each item a checkbox with the
  concrete repro and a suggested fix. Title it as a sample-user audit of the model.
  End with the 🤖 Generated-with footer. (First reference audit: issue #647, the
  0.54.0 LDA run — a good template.)

## Notes / gotchas learned

- Buffered stdout is lost when a foreground run is killed on timeout — always run
  the executed script in the background with `python -u`, writing to a file.
- `label_topics` returns a **list** (not a dict); `find_thoughts` returns
  `(doc, score, top_words)` **triples**; `estimate_effect` results carry `.z` but no
  `.pvalue` and default `add_intercept=True` — these were the LDA run's real crashes,
  useful as a sanity check that a new run's harness is exercising the same surface.
- Do not let the agents "cheat" by reading the source to resolve an API — the whole
  point is what a doc-and-docstring-only user experiences. Reading source to *report*
  a fix location is fine; using it to avoid friction defeats the audit.
