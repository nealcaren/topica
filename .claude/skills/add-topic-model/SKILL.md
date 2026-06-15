---
name: add-topic-model
description: End-to-end workflow for adding a new topic-modeling algorithm to the topica library (a Rust + PyO3 + numpy package). Use when a user asks to add, port, or implement a topic model (e.g. "add CTM-2", "port BTM", "implement an anchored model") into topica. Covers grounding in the literature, finding and running the reference implementation to build a gold-standard result set, implementing a Rust core with Python bindings under topica's conventions, validating with an independent benchmark agent and an author-emulation review agent, and shipping via a GitHub issue + PR that also updates the README and docs.
---

# Add a topic model to topica

Port a new topic-modeling algorithm into topica as a faithful, fast, conventional,
and validated addition. The deliverable is a merged PR: a Rust-core model with
Python bindings, validated against its reference implementation, with the README
and docs updated.

This is a long, multi-phase task. Run the phases in order. Each builds on the
last; do not skip the validation phases (4 and 5) — faithfulness to the reference
is the whole point of a port.

## Orient first (read these once)

- `CONTRIBUTING-MODELS.md` — the deep implementer's playbook for the Rust/PyO3
  mechanics (the `Estimator` trait, the binding, the conformance checks, the
  add-a-model checklist). This skill orchestrates the *process*; that file is the
  *how-to* for Phase 3. Read it before implementing.
- `CLAUDE.md` — build/test commands, layout, conventions, branching. Authoritative.
- `docs/contributing/conventions.md` and `tests/test_naming_conventions.py` — the
  enforced cross-model naming/API contract. New models must pass that test.
- `references/conventions.md` (in this skill) — the distilled rules you will apply
  constantly: the fitted-model surface, naming + alias philosophy, the determinism
  guarantee, build/test gates.

Then pick a short kebab name for the model and its canonical class name early; it
threads through every phase.

## Phase 1 — Ground it in the literature

Goal: a written spec you can implement and check against, before any code.

1. Identify the algorithm and read its source paper(s). Capture, in a scratch note
   (`/private/tmp/<model>-spec.md`): the generative model, the inference method
   (collapsed Gibbs / variational EM / VAE / optimal transport / …), every
   hyperparameter and its default, and **what outputs define a correct fit**
   (topic-word, doc-topic, and any model-specific diagnostics, e.g. a covariate
   effect, a change-point, a keyword rate).
2. Decide where it sits in topica's family (count-based Gibbs, logistic-normal
   variational, embedding-based, …) — this determines which existing core file it
   most resembles (`src/lda.rs`, `src/ctm.rs`, `src/keyatm.rs`, `src/etm.rs`, …).
   Reuse the closest shared machinery (`src/sampler.rs`, `src/optimize.rs`,
   `src/coherence.rs`) rather than re-deriving it.
3. Fix the public name now: the canonical topica name follows our conventions, with
   aliases for the reference package's spelling. See `references/conventions.md`.

## Phase 2 — Reference implementation and gold standard

Goal: a frozen set of reference outputs the topica port must reproduce.

Read `references/reference-and-gold-standard.md` for the full procedure. In short:

1. Find the reference implementation (the paper's repo, an R/Python/C++ package).
2. **Check the license before reading or porting its code.** Permissive (MIT/BSD/
   Apache) — read freely to match algorithmic detail. Copyleft (GPL/AGPL) or
   unclear — do NOT copy code; implement from the paper and treat the reference as
   a black box. Record the license and your decision in the spec note.
3. Install and run it on a small, fixed-seed synthetic corpus. Capture a
   **gold-standard result set** (topic-word, doc-topic, diagnostics, settings,
   seed) to a parity fixture. This is the target the port must match.
4. Add a check under `parity/` that re-runs the reference when its tool is present
   and skips cleanly when it is not (mirror the existing `parity/` scripts).

## Phase 3 — Implement: Rust core + Python bindings

Goal: a working model that builds, passes the conventions test, and is
deterministic.

Follow `references/conventions.md` closely. In short:

1. Add `src/<model>.rs` with the fit/inference loop. Preserve topica's
   **bit-for-bit determinism**: a fixed seed (and thread count, for samplers) must
   reproduce exactly; parallel reductions must sum in a fixed (document) order.
2. Wire the PyO3 binding in `src/python.rs`; keep `python/topica/_topica.pyi` in
   sync with the binding signature. Expose the standard surface every model has:
   `fit(docs, …)`, then `topic_word`, `doc_topic`, `top_words(n)`, `save`/`load`.
3. Apply the naming contract: canonical argument names, reference-package aliases
   where helpful, `iters` for the iteration count, `seed=`, no deprecation cycles.
4. Build and gate after every change:
   ```bash
   VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/maturin develop --release --features python
   cargo test --lib
   VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/ -q
   ```
   Add a Rust unit test and a `tests/test_<model>.py`. The new model must pass
   `tests/test_naming_conventions.py`.

Do this implementation work on a short-lived branch off `main` (see `CLAUDE.md`
branching). Use `uv` for any Python env/package work.

## Phase 4 — Independent benchmark + evaluation agent

Goal: an unbiased measurement of fidelity and speed by an agent that did NOT write
the implementation.

Spawn a subagent (the **benchmark agent**) in an isolated git worktree. It runs the
new model against the Phase-2 gold standard and the live reference, reports aligned
parity (topic-aligned cosine / Jaccard / correlation) and fit speed across corpus
sizes, and renders a verdict. Use the exact prompt and rubric in
`references/evaluation-agents.md` (§Benchmark agent). Independence matters: this
agent must not have implemented the model.

Act on its findings before proceeding. Re-verify any fix yourself (rebuild, re-run
the parity check) — do not merge on the agent's word alone.

## Phase 5 — Author-emulation review agent

Goal: a faithfulness critique from the perspective of the method's originator.

Spawn a second subagent (the **author-emulation reviewer**) that role-plays the
original method's author. Give it the paper, the reference (or its description if
the license barred reading the code), and the topica implementation diff. It
critiques fidelity to the method as published, flags any silent deviation
(different default, dropped term, changed estimator), and judges whether this is a
faithful port. Use the exact prompt in `references/evaluation-agents.md`
(§Author-emulation reviewer).

Reconcile its critique with Phase 4. Genuine fidelity gaps are blockers; defensible
design choices (a topica-canonical name, a shared optimizer) are documented, not
"fixed."

## Phase 6 — Ship: issue, PR, README, docs

Goal: a merged PR that follows GitHub best practices and leaves the docs current.

Read `references/pr-and-docs.md` for the checklist. In short:

1. File a tracking **issue** describing the model, the reference, and the
   validation result.
2. Open a **PR** from the feature branch (squash-merge, delete branch on merge, per
   `CLAUDE.md`). The PR body summarizes the method, the parity result, and both
   agents' verdicts.
3. **Update the README and docs as part of the same PR**: add the model to the
   README model table and the `docs/guides/models.md` roster, and add a
   validation/replication note where the existing models keep theirs. (Updating the
   paper is out of scope — that is the maintainer's, not a contributor's, job.)
4. Confirm all gates green (`cargo test --lib`, `pytest`, `mkdocs build --strict`)
   before requesting merge.

## Bundled references

- `references/conventions.md` — fitted-model surface, naming + alias rules, the
  determinism guarantee, build/test gates, file layout.
- `references/reference-and-gold-standard.md` — finding/running the reference,
  license rules, capturing the gold-standard fixture, the `parity/` pattern.
- `references/evaluation-agents.md` — ready-to-use prompts for the Phase-4
  benchmark agent and the Phase-5 author-emulation reviewer, plus the parity rubric.
- `references/pr-and-docs.md` — the issue/PR template and the README + docs update
  checklist.

## Bundled scripts

- `scripts/compare_to_reference.py` — align a topica model's topic-word matrix to a
  saved reference matrix (Hungarian assignment) and report per-topic cosine, the
  mean, and doc-topic correlation. Use it in Phase 2/4 to score parity objectively.
  Run `python scripts/compare_to_reference.py --help`.
