# Evaluation agents

Two independent subagents validate a port. Spawn each with the Agent tool, model
`opus`, `isolation: "worktree"` (so its build does not collide with the main tree),
and `run_in_background: true` for the long benchmark agent. The agents return a raw
report as their final message; relay what matters and act on it. **Re-verify any
fix yourself** (rebuild, re-run the parity check) — do not merge on an agent's word.

Note: an isolated worktree does NOT contain `CLAUDE.md` (it is git-excluded) or the
dev venv. Each agent must create its own venv. Paste the build/test commands into
the prompt (they are included below).

**Concurrency: pin agents to an explicit commit SHA, not "the current branch."**
`isolation: "worktree"` bases off whatever the main tree has checked out *now*. For
a solo dev the main tree may be on other in-flight work mid-run (this bit us: the
main tree switched to `feat/detm` while the agents ran, and two worktrees cannot
check out the same branch). Capture the SHA up front (`git rev-parse HEAD` on the
finished model branch) and tell each agent to work from a **detached** worktree at
that SHA (`git worktree add --detach <sha>`). Detached worktrees at the same commit
do not conflict; never assume the feature branch is checked out anywhere.

## Parity rubric (shared by both agents)

Topic models are stochastic and implementations differ in RNG and update order, so
the target is **topic-aligned similarity**, not bit equality:

1. Align the port's topics to the reference's by Hungarian assignment on the
   topic-word matrices (`scripts/compare_to_reference.py`).
2. Report per-topic cosine, the mean aligned cosine, top-word Jaccard, and the
   doc-topic correlation after alignment.
3. Calibrate the bar against the **reference's own seed-to-seed variation**: run the
   reference twice with different seeds; the port should match the reference about
   as well as the reference matches itself. Landing inside that noise floor is a
   pass; landing clearly below it is a fidelity gap. **Two traps** (see
   `reference-and-gold-standard.md`): if the reference init is *deterministic* the
   floor is degenerate (zero variance ⇒ a 1.0000-exact bar) — use an
   init-perturbation / `init="random"` floor instead; and if the model is *non-convex
   / non-identified* (NMF, many LDA-ish), the bar is **objective parity**
   (reconstruction/held-out likelihood within tolerance), not reproducing the
   reference's specific decomposition, because equal-quality alternate solutions are
   legitimate.
4. Also check that any model-specific diagnostic (a covariate effect's sign and
   rough magnitude, a recovered change-point, a keyword rate) agrees with the
   reference.

## Benchmark agent (Phase 4)

Independence requirement: this agent must NOT have written the implementation. Give
it the branch with the finished model and the Phase-2 gold standard.

Prompt template (fill the `<…>` slots):

> You are an independent validation engineer for the `topica` topic-modeling
> library (Rust core + PyO3 + numpy). You are in an isolated git worktree on branch
> `<branch>`, which adds a new model: **`<Model>`** (`topica.<Model>`), a port of
> **`<reference name/paper>`**. You did not write it; your job is to measure how
> faithfully and how fast it reproduces the reference — skeptically.
>
> Set up your own environment (the shared `.venv-dev` is not in this worktree):
> ```
> cd "$(git rev-parse --show-toplevel)"
> uv venv .venv-agent
> VIRTUAL_ENV="$PWD/.venv-agent" uv pip install maturin numpy pytest pandas scipy
> VIRTUAL_ENV="$PWD/.venv-agent" .venv-agent/bin/maturin develop --release --features python
> ```
> The reference implementation is `<how to install/run it: R package, pip, binary>`.
> The gold-standard fixture from the port author is at `<path/description>`.
>
> Do the following and report findings, not just numbers:
> 1. **Fidelity.** On the gold-standard corpus (same seed/config), fit `topica.<Model>`
>    and compare to the reference using topic-aligned similarity: Hungarian-align the
>    topic-word matrices, report per-topic and mean cosine, top-word Jaccard, and
>    doc-topic correlation. Use `.claude/skills/add-topic-model/scripts/compare_to_reference.py`
>    if helpful. Establish the noise floor by running the reference twice with
>    different seeds, and say whether the port lands inside it.
> 2. **Diagnostics.** Check that `<model-specific outputs>` agree with the reference
>    in sign and rough magnitude.
> 3. **Determinism.** Fit twice with the same seed (and, for a sampler, two thread
>    counts) and confirm `topic_word`/`doc_topic` are bit-identical (`np.array_equal`).
> 4. **Speed (this is a gate, not just a measurement).** Time fit (fit only,
>    excluding import/build) at small/medium/large corpus sizes you choose (state
>    them), single- and multi-threaded if applicable, and against the reference where
>    a fair comparison exists. **Bar: within a small constant (≈2-3x) of the reference
>    on realistic-density inputs, or justify the gap.** "Fast" is in the deliverable;
>    a port that is 18-43x slower (the first NMF cut) is not done. Use
>    realistic-density inputs — a toy or all-sparse matrix can hide the gap (NMF was
>    ~1.3x on sparse text but ~1.6x on near-dense X). If the bar is missed, a
>    perf-optimization iteration is a normal, expected sub-phase — flag it, don't ship.
> 5. **Gates.** Run `cargo test --workspace --lib` and `pytest tests/ -q`; report pass/fail.
>
> Deliverable (your final message, a raw report): the parity table with the noise
> floor, the determinism result, the speed table, gate results, and a clear verdict
> — faithful / faithful-with-caveats / not-yet-faithful — with the specific gaps an
> implementer must close. Report negative results honestly.

## Author-emulation reviewer (Phase 5)

This agent role-plays the **originator of the method** reviewing an outside port of
their work. It reads the paper and the implementation, not to benchmark, but to
judge fidelity to the method as published. Give it the implementation diff and the
spec note; if the reference license barred reading the source, give it the paper and
say so (it reviews against the paper, not the code).

Prompt template (fill the `<…>` slots):

> You are **`<Author name>`**, the originator of **`<method>`** as published in
> `<paper citation>`. An outside contributor has ported your method into the
> `topica` library and wants your review. Adopt your perspective as the method's
> author: you care that the port is faithful to what you actually proposed, and you
> are quick to spot where someone has quietly changed the model and called it yours.
>
> Materials:
> - Your paper: `<path or summary of the generative model + inference>`.
> - The topica implementation: the diff on branch `<branch>` (the model core in
>   `src/<model>.rs` or `topica-core/src/` for the structural-variational family, the
>   binding in `src/python/<model>.rs`, the tests). Read it.
> - `<if the reference code was readable: the reference is at <path>; if not: note
>   that the port is paper-derived and you should review against the paper only>`.
>
> Review for **fidelity**, not style. Specifically:
> 1. Does the generative model match yours — same latent variables, priors, and
>    likelihood? Flag any dropped or added term.
> 2. Does the inference match the method (the estimator, the update equations, the
>    initialization, the convergence criterion)? Flag any silent substitution (e.g. a
>    different optimizer, a mean-field approximation where you used a full one).
> 3. Are the defaults (priors, iterations, K handling) the ones you recommend? Flag
>    any default that would change a user's result versus your implementation.
> 4. Are the outputs the ones that define a correct fit of your method, computed as
>    you define them?
> 5. Is your work credited correctly?
>
> Deliverable (your final message, a raw report): a fidelity verdict — faithful /
> faithful-with-documented-deviations / misrepresents the method — with a numbered
> list of concrete issues, each marked blocker or acceptable-with-a-note. Distinguish
> genuine infidelities (must fix) from defensible engineering choices (document, do
> not fix), e.g. reusing a shared optimizer or adopting topica's canonical parameter
> names with aliases for your spelling.

## Reconciling the two

- A **blocker** from either agent (a fidelity gap, a broken determinism guarantee, a
  failing gate) must be fixed and re-verified before the PR.
- A **defensible deviation** (canonical naming, a shared numerical kernel, a faster
  but equivalent computation) is kept and documented in the model docstring and PR.
- Where the two disagree, prefer the author-emulation reviewer on questions of
  *method fidelity* and the benchmark agent on questions of *measured behavior*.
