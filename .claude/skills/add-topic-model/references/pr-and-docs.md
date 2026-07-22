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
  - **Gates**: `just test` (Rust core + feature-gated + pytest) and
    `mkdocs build --strict` all green.
  - End the PR body with the repo's standard generated-with trailer if the repo uses
    one (check recent merged PRs).
- Merge with `gh pr merge <n> --squash --delete-branch`, then sync and prune local
  `main`. Feature PRs do not bump the version; a release PR does that later.
- End commit messages with the `Co-Authored-By:` trailer the repo uses (see
  `CLAUDE.md`).

## README and docs updates (same PR)

These ship WITH the model, not in a follow-up:

1. **Registry entry (drives the generated tables).** Do NOT hand-edit the README
   or docs roster tables — they are generated from `python/topica/registry.py`
   (enforced by `test_registry.py`). Add a `ModelInfo` to `REGISTRY` (group,
   brings, inference, determinism, one-line summary) AND an `ImplInfo` to `IMPL`
   (source file, binding, shared machinery, Cargo feature, parity tests — its
   paths are existence-checked in CI), then run `scripts/gen_model_tables.py` (or
   `just gen-tables`) to regenerate the README roster and the contributor model
   map (`docs/contributing/model-map.md`).
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

- `cargo test --workspace --lib` — green (includes the new model's Rust test; add
  `--features embeddings,umap,tsne` for an embedding model).
- `pytest tests/ -q` — green (includes `tests/test_<model>.py`, the conventions
  test, and — once the model is registered — the scaffold-guard check that no
  `SCAFFOLD` marker remains).
- `mkdocs build --strict` — clean (the docs edits build).
- The parity check under `parity/` passes when the reference is present and skips
  cleanly when it is not.
