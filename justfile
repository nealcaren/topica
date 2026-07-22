# topica development command runner (issue #384).
#
# `just` (https://just.systems) wraps the mixed Rust/PyO3 workflow so the
# release-mode rebuild and the VIRTUAL_ENV/.venv-dev handshake are not something
# a contributor has to remember. Install it with `cargo install just`,
# `brew install just`, or your package manager, then run `just` to list recipes.
#
# Platform assumption: these recipes assume a POSIX shell and a development
# virtualenv whose binaries live under `bin/` (macOS / Linux). On Windows, run
# under Git Bash or invoke the underlying commands directly; see
# .github/CONTRIBUTING.md for the raw command list.

# The dev virtualenv. maturin needs VIRTUAL_ENV set explicitly because the dev
# venv is `.venv-dev`, not `.venv`. Honor an already-active venv if there is one,
# else default to .venv-dev. Exported so every recipe (and maturin) sees it.
export VIRTUAL_ENV := env_var_or_default("VIRTUAL_ENV", justfile_directory() / ".venv-dev")

py := VIRTUAL_ENV / "bin" / "python"
maturin := VIRTUAL_ENV / "bin" / "maturin"

# List available recipes (default when you run bare `just`).
default:
    @just --list

# --- Format & lint ---------------------------------------------------------

# Rewrite Rust source to rustfmt's canonical form.
fmt:
    cargo fmt --all

# Check formatting without rewriting (the CI gate).
fmt-check:
    cargo fmt --all --check

# Clippy with warnings-as-errors, all targets, all features (the CI gate).
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# The full CI lint job: formatting check + clippy.
lint: fmt-check clippy

# --- Tests -----------------------------------------------------------------

# No `python` feature: pyo3 extension-module mode won't link a standalone test
# binary. `--workspace` so topica-core is covered too.
#
# Rust unit tests (core + topica-core).
test-rust:
    cargo test --workspace --lib

# The embeddings/umap/tsne cfg gates, which `test-rust` skips (issue #383). Not
# `--all-features`: that enables `python` and breaks linking.
#
# Rust tests behind the feature gates.
test-rust-features:
    cargo test --workspace --lib --features embeddings,umap,tsne

# Python test suite (needs a current dev build; run `just build` first).
test-py:
    {{py}} -m pytest tests/ -q

# Every test surface: Rust core, feature-gated Rust, and Python.
test: test-rust test-rust-features test-py

# --- Build & docs ----------------------------------------------------------

# Always --release: debug Gibbs/EM loops are an order of magnitude slower. Run
# after any Rust change.
#
# (Re)build the extension into the dev venv.
build:
    {{maturin}} develop --release --features python

# Strict docs build — a broken nav entry or cross-reference fails.
docs:
    {{py}} -m mkdocs build --strict

# --- Aggregate gates -------------------------------------------------------

# fmt + clippy + model-table + .pyi stub sync; also wired as the pre-push hook.
#
# Generated-file + lint preflight.
preflight:
    ./scripts/preflight.sh

# Rebuild, then lint + all tests + strict docs + preflight. Mirrors what CI
# enforces, in cheapest-useful order.
#
# Full pre-PR gate — run before opening a PR.
pre-pr: build lint test docs preflight
