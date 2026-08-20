# Contributing to topica

Thanks for your interest in improving `topica`. Contributions of all kinds are
welcome: bug reports, documentation fixes, new diagnostics, and new topic models.

## Reporting bugs and asking for help

- **Bugs and feature requests:** open an issue on the
  [issue tracker](https://github.com/nealcaren/topica/issues). A minimal example
  that reproduces the problem (a small corpus, the model and settings, and what you
  expected) is the fastest path to a fix.
- **Questions and usage help:** open a
  [discussion](https://github.com/nealcaren/topica/discussions) or an issue
  labeled `question`.

## Proposing changes

1. Fork the repository and create a branch off `main`.
2. Make your change with tests and, where user-facing, documentation.
3. Run the checks below and open a pull request describing what changed and why.

`topica` is a Rust core (PyO3) with a thin Python layer. The development
virtualenv is `.venv-dev`. The `justfile` wraps the common commands:

```bash
just build      # maturin develop --release --features python
just test       # Rust unit tests + pytest
just lint       # cargo fmt --check + clippy
just docs       # mkdocs build --strict
just pre-pr     # the full pre-PR gate
```

Continuous integration runs the same checks on Linux, macOS, and Windows, so
please make sure `just pre-pr` passes locally before requesting review.

## Adding a topic model

Every model on the default surface is validated before it ships: against a
maintained reference implementation where one exists, and otherwise by recovering a
planted answer on a synthetic corpus with a known solution. New models must present
the shared topic-word and document-topic surface so that the existing diagnostic,
labeling, and covariate-effect tools apply unchanged. The contributor
documentation describes the details:

- [`docs/contributing/validation.md`](docs/contributing/validation.md) — the
  validation bar and how to meet it.
- [`docs/contributing/estimator-contract.md`](docs/contributing/estimator-contract.md)
  — the interface a model must implement.
- [`docs/contributing/conventions.md`](docs/contributing/conventions.md) — naming
  and API conventions.

## Code of conduct

By participating in this project you agree to abide by its
[Code of Conduct](CODE_OF_CONDUCT.md).
