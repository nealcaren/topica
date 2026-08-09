---
name: add-topic-model
description: End-to-end workflow for adding a new topic-modeling algorithm to the topica library (a Rust + PyO3 + numpy package). Use when a user asks to add, port, or implement a topic model (e.g. "add CTM-2", "port BTM", "implement an anchored model") into topica. Covers grounding in the literature, finding and running the reference implementation to build a gold-standard result set, implementing a Rust core with Python bindings under topica's conventions, a two-reviewer dual-review gate (Codex + Gemini, one faithful-parity and one adversarial) run both after the plan is approved and again before the PR, benchmarking against the reference, and shipping via a GitHub issue + PR that also updates the README and docs.
---

# Add a topic model to topica

Port a new topic-modeling algorithm into topica as a faithful, fast, conventional,
and validated addition. The deliverable is a merged PR: a Rust-core model with
Python bindings, validated against its reference implementation, with the README
and docs updated.

This is a long, multi-phase task. Run the phases in order. Each builds on the
last; do not skip the validation phases — faithfulness to the reference is the
whole point of a port.

**Two dual-review gates bracket the implementation.** A fixed pair of reviewers —
**Reviewer A (faithful / full parity, Codex)** and **Reviewer B (adversarial,
Gemini)**, with a Claude subagent as the per-slot fallback — reviews the work
**twice**: once **after the plan is approved and before any code** (Gate A, below
after Phase 2), and once **after implementation and before opening the PR** (Gate B,
Phase 5). The roles and their backing models stay fixed across both gates; do not
swap who is faithful-parity and who is adversarial. See
`references/evaluation-agents.md` for the mechanism, prompts, and synthesis rule.

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
   most resembles. **The core is split across two crates** (a Cargo workspace):
   - `topica-core/src/` holds the logistic-normal *structural* cluster and shared
     numerics — `ctm.rs` (CTM/STM/SAGE), `spectral.rs`, `cvb0.rs`, `estimator.rs`,
     `linalg.rs`, and the `variational/` kernels (L-BFGS, the Laplace E-step, the
     Σ/Γ M-step). This crate is dependency-light so downstream Rust (e.g. faSTM)
     can vendor it; touch it only for the structural-variational family.
   - `src/` holds every other model (`keyatm.rs`, `etm.rs`, `dmr.rs`, `hdp.rs`,
     `prodlda.rs`, the Gibbs/embedding/VAE models, …) plus the shared
     `sampler.rs`, `optimize.rs`, `coherence.rs`. `topica` re-exports `topica-core`,
     so `topica::ctm::*` still resolves. Most new models land here.
   Reuse the closest shared machinery rather than re-deriving it.
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

## Gate A — dual review of the plan (before any code)

Goal: catch a faithfulness or completeness problem in the *design* before you spend
a multi-phase implementation on it.

Once the Phase-1/2 spec is written and you (with the user) have approved the plan —
which core file, which estimator, the defaults, the public surface — hand it to the
**two fixed reviewers** before writing model code:

1. **Reviewer A (faithful / full parity — Codex)** via the `codex` skill. It reads
   the spec, the paper, and the reference code (where the license allows) and judges
   whether *this plan* reproduces the method faithfully and completely.
2. **Reviewer B (adversarial — Gemini)** via the `antigravity` skill. It attacks the
   plan: the single most likely way this design yields a plausible-but-wrong model,
   and what to test early to catch it.

If the `codex` CLI or Antigravity is unavailable, run that slot as a Claude subagent
(Agent tool, model `opus`) — same role, do not swap. Then **synthesize** both reports
into one findings list, resolve blockers with the user, and only then proceed to
Phase 3. Use the exact prompts and synthesis rule in `references/evaluation-agents.md`
(§Gate A, §Reviewer A/B prompts, §Synthesis).

## Phase 3 — Implement: Rust core + Python bindings

Goal: a working model that builds, passes the conventions test, and is
deterministic.

Follow `references/conventions.md` closely. In short:

0. **Scaffold the touchpoints first.** Run `just new-model <ModelName>` (or
   `python scripts/new_model.py --name <ModelName>`) to generate `src/<model>.rs`,
   `src/python/<model>.rs`, and `tests/test_<model>.py` from templates, each
   stamped with a `SCAFFOLD(<ModelName>)` marker, plus a printed checklist of the
   wiring steps. It wires nothing in (an un-finished model stays inert), and
   `tests/test_scaffold_guard.py` fails once the model is registered if any
   `SCAFFOLD` marker survives — so you can't ship a half-filled placeholder. Fill
   in the templates below; the steps that follow are that checklist in prose.
1. Implement `src/<model>.rs` with the fit/inference loop. Preserve topica's
   **bit-for-bit determinism**: a fixed seed (and thread count, for samplers) must
   reproduce exactly; parallel reductions must sum in a fixed (document) order.
2. Wire the PyO3 binding in `src/python/` (a directory module, not a single file):
   add `src/python/<model>.rs` with the pyclass and bring it in via `use super::*`
   (the established per-model recipe — see `nmf_lsa.rs`), and register it in
   `src/python/mod.rs`. Keep `python/topica/_topica.pyi` in
   sync with the binding signature. Expose the standard surface every model has:
   `fit(docs, …)`, then `topic_word`, `doc_topic`, `top_words(n)`, `save`/`load`.
3. Apply the naming contract: canonical argument names, reference-package aliases
   where helpful, `iters` for the iteration count, `seed=`, no deprecation cycles.
4. Build and gate after every change. The `just` runner wraps the release rebuild
   and the `VIRTUAL_ENV`/`.venv-dev` handshake:
   ```bash
   just build          # maturin develop --release --features python
   just test           # cargo test (core + feature-gated) + pytest
   ```
   The raw commands still work if `just` is unavailable:
   ```bash
   VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/maturin develop --release --features python
   cargo test --workspace --lib                                   # add --features embeddings,umap,tsne for an embedding model
   VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/ -q
   ```
   Add a Rust unit test and a `tests/test_<model>.py`. The new model must pass
   `tests/test_naming_conventions.py`.

Do this implementation work on a short-lived branch off `main` (see `CLAUDE.md`
branching). Use `uv` for any Python env/package work.

## Phase 4 — Benchmark against the reference

Goal: an unbiased measurement of fidelity and speed — the empirical substrate the
Gate-B reviewers judge against.

Measure the finished model against the Phase-2 gold standard and the live reference:
topic-aligned parity (cosine / Jaccard / doc-topic correlation with the noise floor),
model-specific diagnostics, determinism, and fit speed across corpus sizes. Do this
in an isolated git worktree with its own venv — either inline or via a Claude
benchmark subagent that did NOT write the model (independence matters). Use the
rubric in `references/evaluation-agents.md` (§Phase-4 benchmark).

"Fast" is part of the deliverable, so apply the **speed gate** (within ≈2-3x of the
reference on realistic-density inputs, or justify the gap). If it is missed, a
**performance-optimization iteration is a normal sub-phase**, not a failure —
optimize and re-benchmark before shipping; the first cut being slow is expected. To
amortize the multi-minute `--release` build + venv, **fold related checks into one
agent** (e.g. run the live-reference real-corpus parity inside the benchmark agent).

Re-verify any fix yourself (rebuild, re-run the parity check). Carry the parity table,
determinism result, and speed table into Gate B — the reviewers work from these.

## Phase 5 — Gate B: dual review before the PR

Goal: the same two-reviewer fidelity check as Gate A, now against the finished
artifact, before the PR is opened.

Hand the **implementation diff**, the spec, and the Phase-4 benchmark/parity results
to the **same two fixed reviewers**, in the **same roles** as Gate A:

1. **Reviewer A (faithful / full parity — Codex)** via the `codex` skill: does the
   diff faithfully and completely reproduce the method, and do the measured parity
   numbers actually support the "faithful" claim (no overclaim past the noise floor)?
2. **Reviewer B (adversarial — Gemini)** via the `antigravity` skill: silent
   deviations, correctness bugs, determinism holes, overclaimed parity/speed,
   untested edge cases.
3. **Reviewer C (comparative — Claude subagent, `opus`, isolated worktree)**:
   how the finished model stacks up against its same-family topica siblings on
   **accuracy, speed, and memory**, and whether the PR's comparative claims are
   honest. Gate B only (needs the built artifact to benchmark). Prompt in
   `references/evaluation-agents.md` (§Reviewer C).
4. **Reviewer D (sample-user on real data — Claude `general-purpose` subagent)**:
   a first-time computational social scientist ("a random sociologist") who takes the
   model through the whole paper workflow on a **real bundled dataset**, docs-only,
   and reports ranked friction (Tier-1 analytical traps first). Gate B only. Uses the
   `sample-user` skill; prompt in `references/evaluation-agents.md` (§Reviewer D).

Gate B is thus **four reviewers** (A+B+C+D); Gate A is two (A+B).

Fall back per-slot to a Claude subagent (model `opus`, isolated worktree,
background) only if the external model is unavailable — same role, do not swap.
Then **synthesize** both reports into one findings list. Genuine fidelity gaps,
broken determinism, failing gates, and overclaims are **blockers** — fix and
**re-verify yourself** before the PR. Defensible design choices (a topica-canonical
name, a shared optimizer) are documented, not "fixed." Use the exact prompts and
synthesis rule in `references/evaluation-agents.md` (§Gate B, §Synthesis).

## Phase 5.5 — Post-fix reference sanity check

Goal: confirm the Gate-B fixes didn't regress the model, by running the *finished*
artifact head-to-head against the live reference one more time.

The Phase-4 benchmark measured the model *before* the Gate-B fixes, and those fixes
change code — so re-verify against the reference after they land, on a **real
dataset** (not just the synthetic parity fixture — e.g. a 20 Newsgroups subset).
Fit both topica and the reference implementation on the same corpus and compare,
side by side:

- **Accuracy / fidelity** — the metric Phase 4 used (topic-aligned cosine, or a
  discovered-K-appropriate metric like cross-NMI / document-clustering agreement
  where the two land on different K), against the noise floor.
- **Speed** — wall-clock for both at a realistic corpus size, so the speed claim in
  the PR reflects the shipped code, not the pre-fix version.

A regression here — accuracy below the Phase-4 number, or a speed cliff — is a
**blocker**: the fixes broke something; diagnose and re-verify before the PR. If
both hold, record the head-to-head accuracy-and-speed numbers in the PR body next
to the parity result. This is the same comparison `benchmarks/full_model_run.py`
runs across the roster (reference-ceiling accuracy + timing ratio); add or refresh
the model's row there if it belongs on the table, capping the corpus where a fit is
otherwise intractable (as HDP/HLDA do at ~2k docs).

## Phase 6 — Ship: issue, PR, README, docs

Goal: a merged PR that follows GitHub best practices and leaves the docs current.

Read `references/pr-and-docs.md` for the checklist. In short:

1. **Create or reference** a tracking **issue** describing the model, the reference,
   and the validation result (one often already exists — reuse it, e.g. #178 for NMF).
2. Open a **PR** from the feature branch (squash-merge, delete branch on merge, per
   `CLAUDE.md`). The PR body summarizes the method, the parity result, and the
   synthesized verdicts from **both dual-review gates** (Gate A on the plan, Gate B
   on the diff) — naming the reviewer models used and any documented deviations.
3. **Update the docs as part of the same PR.** Both generated tables come from the
   registry (`python/topica/registry.py`), enforced by `test_registry.py`: add a
   `ModelInfo` to `REGISTRY` (drives the README/docs roster) **and** an `ImplInfo`
   to `IMPL` (drives the contributor `docs/contributing/model-map.md` — source
   file, binding, feature, parity tests; its paths are existence-checked in CI).
   Then run `scripts/gen_model_tables.py` (or `just gen-tables`) to regenerate both;
   do NOT hand-edit a generated table (it fights the generator). Your hand-written
   work is the per-model prose section (in `docs/guides/models.md`), a
   validation/replication note where the existing models keep theirs, and the
   README acknowledgement. (Updating the paper is out of scope — that is the
   maintainer's, not a contributor's, job.)
4. Confirm all gates green (`just test`, `mkdocs build --strict`)
   before requesting merge.

## Bundled references

- `references/conventions.md` — fitted-model surface, naming + alias rules, the
  determinism guarantee, build/test gates, file layout.
- `references/reference-and-gold-standard.md` — finding/running the reference,
  license rules, capturing the gold-standard fixture, the `parity/` pattern.
- `references/evaluation-agents.md` — the dual-review mechanism: the two fixed
  reviewers (faithful-parity Codex + adversarial Gemini, Claude fallback), the two
  gates (plan and pre-PR), ready-to-use prompts, the synthesis rule, and the
  Phase-4 benchmark rubric.
- `references/pr-and-docs.md` — the issue/PR template and the README + docs update
  checklist.

## Bundled scripts

- `scripts/compare_to_reference.py` — align a topica model's topic-word matrix to a
  saved reference matrix (Hungarian assignment) and report per-topic cosine, the
  mean, and doc-topic correlation. Use it in Phase 2/4 to score parity objectively.
  Run `python scripts/compare_to_reference.py --help`.
