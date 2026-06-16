# Contributing to topica

Thanks for your interest in topica. It is a Rust core (PyO3, maturin, abi3-py39)
with a thin Python layer, so contributing touches both languages.

## Development setup

topica builds with [maturin](https://www.maturin.rs/). Create a virtual
environment and build the extension in place:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release --features python
```

Always build with `--release`: the debug build is much slower for the Gibbs and
EM loops.

## Tests

```bash
cargo test --lib                 # Rust unit tests
python -m pytest tests/ -q       # Python tests
mkdocs build --strict            # docs must build clean
```

The `parity/` directory holds cross-implementation checks against R (`stm`,
`keyATM`) and Java MALLET. They skip cleanly when Rscript or the package is not
installed, so they are optional locally but run when the tooling is present.

## Conventions

- The house import is `import topica`, with no alias.
- One Rust file per model under `src/` (`keyatm.rs`, `stm.rs`, …); `python.rs`
  holds the PyO3 bindings. Keep the `python/topica/_topica.pyi` type stub in sync
  when you change a binding's signature.
- New gradients or samplers should ship with a test that checks them (finite
  differences for gradients, planted-data recovery for samplers), and where a
  reference implementation exists, a statistical-parity check under `parity/`.
- User-facing prose (README, docs, docstrings) is concrete and free of filler.

## Adding a model or a model feature

For the step-by-step version of either task, including the analysis-surface
contract a new model must satisfy and the testing and validation expectations,
see the implementer's playbook in
[`CONTRIBUTING-MODELS.md`](CONTRIBUTING-MODELS.md).

## Pull requests

- Branch from `main`, keep commits focused, and describe what the change does.
- Make sure the three test commands above pass before opening the PR.
- New models are validated against their reference implementation; describe the
  validation in the PR.
