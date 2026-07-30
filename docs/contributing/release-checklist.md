# Release checklist

A topica release is a scientific-software claim as well as a package upload.
The automated release workflow is the hard gate; this checklist records the
small set of human checks that cannot be made cheap, deterministic CI jobs.

## Before creating a tag

- [ ] Start from a clean worktree and an up-to-date `main`.
- [ ] Choose the release version, update `pyproject.toml`, `Cargo.toml`, and
      `CITATION.cff` (and `Cargo.lock`), and add a dated changelog entry.
- [ ] Confirm that the tag will be exactly `v<version>`. CI (`test_version_consistency`)
      refuses a tag whose version disagrees with any of the three
      (`pyproject.toml`, `Cargo.toml`, `CITATION.cff`); `CITATION.cff` is the one
      that most often drifts.
- [ ] Read the open-issue triage: close delivered work, keep deliberate research
      deferrals visible, and do not turn an RFC into a release promise by accident.

## Required automated gates

The tag-triggered workflow must pass all of these before it can publish PyPI or
GitHub CLI artifacts:

- Rust formatting, Clippy, and the core test suite;
- Python integration tests on Linux, macOS, and Windows;
- strict documentation build;
- native-wheel install/import smoke tests;
- source-distribution build from the clean tagged checkout.

## Manual release dogfood

For a feature release that changes an estimator, numerical optimizer, sampler,
or covariate path, run one representative real-corpus workflow before tagging.
This is deliberately a manual check rather than pull-request CI: a useful corpus
is too large and too domain-specific for a stable hosted-test budget.

Record the command, corpus provenance and size, model configuration, seed,
runtime environment, convergence/stop reason, diagnostics, and the observed
result in the release PR or changelog notes. The purpose is not to maximize a
metric or to substitute an automated label for interpretation. It is to catch a
degenerate fit, implausible scale behavior, or an unusable public workflow before
users do.
