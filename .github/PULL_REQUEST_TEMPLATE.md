<!--
Thanks for contributing to topica. Keep PRs focused: one feature or fix per PR.
The title becomes the squash-merge commit, so write it as a conventional commit
(e.g. `feat(keyatm): ...`, `fix(stm): ...`, `docs: ...`, `perf(lda): ...`).
-->

## What

<!-- What does this PR change, and why? Link the issue it closes. -->

Closes #

## How

<!-- Key implementation notes a reviewer needs: the approach, any tradeoffs,
     and anything you deliberately left out of scope. -->

## Validation

<!-- How do you know it works? For a model, note the reference it was validated
     against and the parity result. For a fix, note the test that now passes. -->

- [ ] `cargo test --lib` passes
- [ ] `python -m pytest tests/ -q` passes
- [ ] `./scripts/preflight.sh` clean (fmt + clippy + generated-file sync)
- [ ] docs build clean (`mkdocs build --strict`) if docs or docstrings changed
- [ ] the `.pyi` stub is in sync if a binding signature changed

## Notes

<!-- Anything else: follow-up issues, benchmarks, screenshots. -->
