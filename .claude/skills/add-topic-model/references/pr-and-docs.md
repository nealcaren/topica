# Shipping: issue, PR, README, and docs

Follow the branching and merge conventions in `CLAUDE.md`. The deliverable is a
squash-merged PR that adds the model AND leaves the README and docs current.

## Issue first

Open a tracking issue with `gh issue create` that states:

- The model: name, what it does, and where it sits in the family.
- The reference: package/paper, version, license, and whether the port is
  code-derived (permissive license) or paper-derived (black-box oracle).
- The validation plan / result: the parity bar and the two agents' verdicts.

## Branch and PR

- Work on one short-lived branch off `main` (e.g. `feat/<model>`). Do not stack new
  work on a branch with an open PR.
- Open the PR with `gh pr create`. The body should cover:
  - **What**: the model and its method, in two or three sentences.
  - **Validation**: the parity table (topic-aligned cosine, Jaccard, doc-topic
    correlation) against the reference, the noise floor it was calibrated to, and
    the determinism check. Summarize the benchmark agent and author-emulation
    reviewer verdicts and how any blockers were resolved.
  - **Conventions**: confirm `tests/test_naming_conventions.py` passes and note any
    reference-package aliases added.
  - **Gates**: `cargo test --lib`, `pytest tests/ -q`, `mkdocs build --strict` all
    green.
  - End the PR body with the repo's standard generated-with trailer if the repo uses
    one (check recent merged PRs).
- Merge with `gh pr merge <n> --squash --delete-branch`, then sync and prune local
  `main`. Feature PRs do not bump the version; a release PR does that later.
- End commit messages with the `Co-Authored-By:` trailer the repo uses (see
  `CLAUDE.md`).

## README and docs updates (same PR)

These ship WITH the model, not in a follow-up:

1. **README model table.** Add a row for the new model in the appropriate section
   (count-based, structural/covariate, embedding-based, …), matching the existing
   rows' one-line description style. If the model is validated against a named
   reference, note it the way the other validated models do.
2. **README acknowledgements.** If you read a permissively-licensed reference to
   match details, credit it in the acknowledgements list alongside the existing
   references.
3. **`docs/guides/models.md`** (the model roster) — add the model with a short
   description and its key parameters, consistent with neighboring entries.
4. **Validation/replication note.** Where the existing models keep their validation
   evidence (e.g. a `docs/.../replications/` page or a validation section), add the
   parity result for the new model.
5. **A usage example.** If the model has a distinctive surface (a covariate, a
   keyword set, embeddings), add a short worked snippet to the relevant guide.

Do NOT update the paper (`paper/`). That is the maintainer's responsibility, not a
contributor's, and is intentionally out of scope for this workflow.

## Prose register (README, docs, docstrings)

User-facing prose follows the project's academic register: no em dashes, agent-led
"we", concrete over hedged, no LLM filler. Match the voice of the surrounding text.

## Final check before requesting merge

- `cargo test --lib` — green (includes the new model's Rust test).
- `pytest tests/ -q` — green (includes `tests/test_<model>.py` and the conventions
  test).
- `mkdocs build --strict` — clean (the docs edits build).
- The parity check under `parity/` passes when the reference is present and skips
  cleanly when it is not.
