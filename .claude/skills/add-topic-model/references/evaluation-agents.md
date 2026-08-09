# Evaluation agents — the dual-review gates

A port is reviewed by **two independent reviewers at two gates**. The reviewers,
their roles, and their backing models are **fixed across both gates** — do not swap
who is faithful-parity and who is adversarial between the plan review and the
pre-PR review. Consistency is the point: the same critic sees the plan and then
sees whether the finished code honored it.

## The two reviewers (fixed roles, fixed models)

| Slot | Role | Backend | Fallback |
|------|------|---------|----------|
| **Reviewer A** | **Faithful / full parity.** Checks the port against the method as published and as implemented in the reference, given the paper and (where the license allows) the reference code. Is the generative model, inference, defaults, and output set a true and *complete* reproduction? | **Codex** (`codex` skill) | Claude subagent (Agent tool, model `opus`) |
| **Reviewer B** | **Adversarial.** Assumes the port is subtly wrong and tries to prove it: silent deviations, dropped terms, off defaults, determinism holes, overclaimed parity, speed traps, untested edge cases. | **Gemini / Antigravity** (`antigravity` skill) | Claude subagent (Agent tool, model `opus`) |
| **Reviewer C** (**Gate B only**) | **Comparative.** Places the finished model against its topica siblings (the same-family models already in the roster) and evaluates it on **accuracy, speed, and memory** — is it competitive, where does it win/lose, and are the PR's comparative claims honest? Not a fidelity check (that is A/B); a "how does it stack up in the library" check. | Claude subagent (Agent tool, model `opus`, isolated worktree so it can build + run) | — |

Rules:

- **Prefer the external models.** Reviewer A is Codex; Reviewer B is Gemini via the
  `antigravity` skill. Independence from Claude (the implementer) is the value —
  two different model families catch different things.
- **Fall back per-slot, not by swapping.** If the `codex` CLI is unavailable, run
  Reviewer A as a Claude subagent — but it stays the faithful-parity reviewer. If
  Antigravity is unavailable, run Reviewer B as a Claude subagent — still
  adversarial. Never make Codex the adversarial one just because Gemini is down.
- **Same backend both gates.** Whoever played a role at Gate A plays it again at
  Gate B. (This mirrors the standing `pr-dual-review-merge-gate` /
  `pr-review-merge-workflow` practice: faithfulness + adversarial, one being
  Codex/Gemini, Claude subagents only when the external model is rate-limited.)

## The two gates

- **Gate A — plan review** (after the Phase-1/2 spec is written and the plan is
  approved, *before* writing model code). Inputs: the spec note
  (`/private/tmp/<model>-spec.md`), the paper, the reference code/description, and
  the intended topica design (which core file, estimator, defaults, public
  surface). No implementation exists yet, so this is a design review: will *this
  plan* reproduce the method faithfully and completely, and where is it most likely
  to go wrong?
- **Gate B — pre-PR review** (after implementation, benchmarking, and gates pass,
  *before* opening the PR). Inputs: the implementation diff, the spec, the
  Phase-4 benchmark/parity results, and the paper/reference. This is the
  fidelity-of-the-artifact review. **Gate B runs three reviewers**: A (faithful) +
  B (adversarial) + **C (comparative — accuracy/speed/memory vs the topica
  siblings)**. Reviewer C is Gate-B-only; Gate A is A+B (there is no artifact to
  benchmark yet).

At each gate: run both reviewers, **synthesize** into one reconciled findings list
(see below), act on blockers, then move forward. Do not proceed on a single
reviewer's say-so, and **re-verify every fix yourself** (rebuild, re-run the parity
check) — never merge on an agent's word.

## Synthesis (do this at each gate)

1. Collect both raw reports.
2. Merge into one list, de-duplicating overlapping findings. Tag each: **blocker**
   (a genuine fidelity gap, a broken determinism guarantee, a failing gate, an
   overclaim) or **defensible deviation** (a topica-canonical name with an alias, a
   shared numerical kernel, a faster-but-equivalent computation).
3. On disagreement, weight **Reviewer A** on questions of *method fidelity / parity*
   and **Reviewer B** on *where it breaks* — but a concrete, reproducible adversarial
   finding outranks a "looks fine." At Gate B, weight the measured benchmark numbers
   over either reviewer's intuition about behavior.
4. Fix blockers, document defensible deviations (docstring + PR body), re-verify,
   and record the synthesized verdict. At Gate A the verdict gates *starting to
   code*; at Gate B it gates *opening the PR*.

## Invoking the reviewers

- **Codex** — invoke the `codex` skill with the role prompt below and the materials
  (spec/diff, paper path, reference path or note).
- **Gemini** — invoke the `antigravity` skill with the adversarial prompt below.
- **Claude fallback** — spawn with the Agent tool, model `opus`. For Gate B (which
  needs a build) use `isolation: "worktree"` and `run_in_background: true`; for the
  benchmark run see the worktree notes at the end of this file. Gate A needs no
  build, so a plain subagent is fine.

Give both reviewers the **same materials** at a gate so their reports are
comparable.

### Reviewer A — faithful / full-parity prompt

Fill the `<…>` slots. At Gate A, "the port" is the *plan*; at Gate B it is the
*diff*.

> You are an independent reviewer checking a port of **`<method>`** (`<paper
> citation>`) into the `topica` topic-modeling library (Rust core + PyO3 + numpy).
> Your job is to judge whether the port is a **faithful and complete** reproduction
> of the method — not its style, its fidelity and full parity.
>
> Materials:
> - The paper / generative model + inference: `<path or summary>`.
> - The reference implementation: `<path if license-readable; else: "paper-derived,
>   review against the paper only — the reference code was not read for license
>   reasons">`.
> - The topica port: `<at Gate A: the spec note at /private/tmp/<model>-spec.md and
>   the intended design — core file, estimator, defaults, public surface. At Gate B:
>   the diff on branch <branch> — src/<model>.rs or topica-core/src/, the binding in
>   src/python/<model>.rs, the tests; plus the benchmark/parity results at <path>>`.
>
> Check, point by point:
> 1. **Generative model** — same latent variables, priors, likelihood. Any dropped
>    or added term?
> 2. **Inference** — the estimator, update equations, initialization, convergence
>    criterion match the method? Any silent substitution?
> 3. **Defaults** — priors, iteration counts, K handling are the ones the method
>    recommends? Any default that would change a user's result vs. the reference?
> 4. **Completeness** — is the *whole* method here, or a subset? Name anything the
>    paper specifies that the port omits or stubs.
> 5. **Outputs** — the port computes the outputs that define a correct fit, the way
>    the method defines them?
> 6. (Gate B only) **Measured parity** — do the reported topic-aligned cosine /
>    Jaccard / doc-topic correlation and the diagnostic checks actually support the
>    "faithful" claim, given the reference's own seed-to-seed noise floor? Is any
>    parity claim overstated?
>
> Deliverable: a verdict — faithful / faithful-with-documented-deviations /
> not-yet-faithful — with a numbered list of concrete issues, each marked **blocker**
> or **acceptable-with-a-note**. Separate genuine infidelities (must fix) from
> defensible engineering choices (document, do not fix), e.g. a shared optimizer or
> topica's canonical parameter names with aliases.

### Reviewer B — adversarial prompt

> You are an adversarial reviewer for a port of **`<method>`** (`<paper citation>`)
> into the `topica` topic-modeling library (Rust core + PyO3 + numpy). Assume the
> port is subtly wrong and your job is to **prove it**. Be skeptical; a concrete,
> reproducible failure is worth more than a general worry.
>
> Materials: `<same as Reviewer A — paper, reference or note, and at Gate A the
> spec/plan, at Gate B the diff on branch <branch> plus the benchmark results>`.
>
> Hunt for:
> 1. **Silent deviations** — a term dropped, a prior changed, an estimator swapped, a
>    default that quietly differs from the reference, dressed up as the same model.
> 2. **Correctness bugs** — indexing, normalization, log/exp domain, off-by-one in
>    the sampler/EM loop, NaN/inf from unguarded hyperparameters, edge cases (K=1,
>    empty docs, a single doc, a huge vocab) the tests miss.
> 3. **Determinism holes** (Gate B) — anywhere a fixed seed + thread count could fail
>    to reproduce (unordered parallel reductions, HashMap iteration, RNG reuse).
> 4. **Overclaimed parity / speed** — a parity number that rode on a degenerate noise
>    floor or a too-easy corpus; a speed claim measured on sparse input that hides a
>    dense-input blowup.
> 5. (Gate A) **Plan risks** — the single most likely way *this design* produces a
>    plausible-but-wrong model; what the plan should test to catch it early.
>
> Deliverable: a numbered list of specific findings, each with how to reproduce or
> where in the plan/code it lives, marked **blocker** or **nit**. Rank the blockers.
> If you genuinely cannot break it, say so and name what you probed.

### Reviewer C — comparative prompt (Gate B only)

Run as a Claude subagent (model `opus`, isolated worktree at a pinned SHA, background)
so it can build the model and run its own timings. Give it the diff, the spec, the
Phase-4 numbers, and the names of the same-family topica models already in the roster.

> You are a comparative reviewer for a newly added model **`<method>`** in the
> `topica` topic-modeling library. It is faithful to its reference (that is checked
> elsewhere). Your job: judge how it **stacks up against its topica siblings** —
> `<list the same-family models, e.g. LDA, LabeledLDA, KeyATM, SeededLDA>` — on three
> axes, and whether the PR's comparative claims are honest.
>
> Build the model (`references/conventions.md` has the worktree venv recipe) and,
> on a shared realistic corpus (e.g. a 20-Newsgroups subset, or the model's natural
> data), measure and compare against the most relevant 2-3 siblings:
> 1. **Accuracy / quality** — topic coherence, and task fit where one exists
>    (clustering agreement, held-out likelihood, the model's own diagnostic). Is it
>    competitive, better, or worse than the siblings, and *at what*?
> 2. **Speed** — fit wall-clock at small/medium/large sizes, single- and
>    multi-threaded where applicable. Where does it sit in the family's speed order?
> 3. **Memory** — peak RSS during fit at a realistic size vs the siblings.
>
> Deliverable: a compact table (model × {accuracy, speed, memory}) + a verdict:
> where this model wins, where it loses, and whether anything in the PR body
> overstates its standing. Flag any regression a user would hit by picking it over a
> sibling. Mark real problems **blocker**, tradeoffs **note**.

Reviewer C's findings feed the same synthesis. A genuine, reproducible regression
(e.g. 10x slower than the equivalent sibling with no upside, or a memory blowup) is a
**blocker**; an honest tradeoff (slower but more faithful, heavier but richer output)
is documented in the PR body, not "fixed."

## Phase-4 benchmark — the empirical substrate for Gate B

Gate B's faithful-parity reviewer works from *measured* results, not just a code
read. Produce them in Phase 4 before the reviewers run. You can measure inline or
spawn a benchmark subagent (Claude, `opus`, isolated worktree, background); either
way the numbers feed both reviewers.

**Parity rubric.** Topic models are stochastic and implementations differ in RNG and
update order, so the target is **topic-aligned similarity**, not bit equality:

1. Align the port's topics to the reference's by Hungarian assignment on the
   topic-word matrices (`scripts/compare_to_reference.py`).
2. Report per-topic cosine, the mean aligned cosine, top-word Jaccard, and the
   doc-topic correlation after alignment.
3. Calibrate against the **reference's own seed-to-seed variation**: run the
   reference twice with different seeds; the port should match the reference about
   as well as the reference matches itself. Landing inside that noise floor is a
   pass; clearly below it is a fidelity gap. **Two traps:** if the reference init is
   *deterministic* the floor is degenerate (zero variance ⇒ a 1.0000-exact bar) —
   use an init-perturbation / `init="random"` floor instead; and if the model is
   *non-convex / non-identified* (NMF, many LDA-ish), the bar is **objective parity**
   (reconstruction / held-out likelihood within tolerance), not reproducing the
   reference's specific decomposition.
4. Check that any model-specific diagnostic (a covariate effect's sign and rough
   magnitude, a recovered change-point, a keyword rate) agrees with the reference.

**Determinism.** Fit twice with the same seed (and, for a sampler, two thread
counts); confirm `topic_word`/`doc_topic` are bit-identical (`np.array_equal`).

**Speed (a gate, not just a measurement).** Time fit (fit only, excluding
import/build) at small/medium/large corpus sizes, single- and multi-threaded if
applicable, against the reference where a fair comparison exists. **Bar: within a
small constant (≈2-3x) of the reference on realistic-density inputs, or justify the
gap.** Use realistic-density inputs — a toy or all-sparse matrix can hide the gap
(NMF was ~1.3x on sparse text but ~1.6x on near-dense X). If the bar is missed, a
perf-optimization iteration is a normal, expected sub-phase — flag it, don't ship.

**Gates.** `cargo test --workspace --lib` and `pytest tests/ -q` must pass.

### Worktree notes (for any Claude subagent that builds)

An isolated worktree does NOT contain `CLAUDE.md` (git-excluded) or the dev venv.
Each building agent creates its own venv:

```
cd "$(git rev-parse --show-toplevel)"
uv venv .venv-agent
VIRTUAL_ENV="$PWD/.venv-agent" uv pip install maturin numpy pytest pandas scipy
VIRTUAL_ENV="$PWD/.venv-agent" .venv-agent/bin/maturin develop --release --features python
```

**Pin to an explicit commit SHA, not "the current branch."** `isolation:
"worktree"` bases off whatever the main tree has checked out *now*, which for a solo
dev may be other in-flight work mid-run (this bit us: the tree switched to
`feat/detm` mid-run and two worktrees cannot check out the same branch). Capture the
SHA up front (`git rev-parse HEAD`) and have the agent work from a **detached**
worktree at that SHA (`git worktree add --detach <sha>`). Detached worktrees at the
same commit do not conflict.
