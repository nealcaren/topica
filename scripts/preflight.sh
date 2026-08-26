#!/usr/bin/env bash
#
# Preflight checks — the local gate that mirrors the CI jobs that fail fast and
# often (lint + generated files), so a push cannot land a red CI-lint.
#
# Wired as the git pre-push hook via `.githooks/pre-push` (see the repo's
# `core.hooksPath = .githooks`). Run it by hand any time:  ./scripts/preflight.sh
# Bypass a push in a pinch with:  git push --no-verify
#
# It runs the EXACT commands CI runs, in order of cheapest-first:
#   1. cargo fmt --all --check
#   2. the Rust toolchain pin agrees (rust-toolchain.toml == the workflow env)
#   3. cargo clippy --workspace --all-targets --all-features -- -D warnings
#   4. generated model tables are in sync (scripts/gen_model_tables.py --check)
#   5. the .pyi type stub is in sync with the compiled extension (test_stub_sync.py)
#   6. the compat map and the agent cheat sheet are in sync (gen_compat.py /
#      gen_guide.py --check)
#
# Steps 3-4 need `topica` importable; they use $VIRTUAL_ENV or .venv-dev, and are
# skipped with a warning (not a hard failure) if no dev venv is present, so the
# fmt/clippy gate still runs anywhere.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

fail=0
step() { printf '\n\033[1m[preflight] %s\033[0m\n' "$1"; }

step "cargo fmt --all --check"
if ! cargo fmt --all --check; then
    echo "  -> run 'cargo fmt --all' to fix" >&2
    fail=1
fi

# The Rust toolchain is pinned in one place, `rust-toolchain.toml`, which drives
# local dev (rustup auto-selects it) AND CI (the workflows pass it via a
# RUST_TOOLCHAIN env). If those drift, CI's clippy differs from local clippy and a
# new stable can land a red-lint push (as happened twice in 2026-08). Fail on any
# mismatch so the pin stays a single source of truth. Bump by editing
# rust-toolchain.toml *and* the workflow env together, then `rustup update`.
step "rust toolchain pin agrees across rust-toolchain.toml and the workflows"
pin="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' rust-toolchain.toml | head -1)"
if [ -z "$pin" ]; then
    echo "  -> could not read channel from rust-toolchain.toml" >&2
    fail=1
else
    ok=1
    for wf in .github/workflows/CI.yml .github/workflows/docs.yml; do
        wf_pin="$(sed -n 's/.*RUST_TOOLCHAIN: *"\([^"]*\)".*/\1/p' "$wf" | head -1)"
        if [ "$wf_pin" != "$pin" ]; then
            echo "  -> $wf pins RUST_TOOLCHAIN='$wf_pin' but rust-toolchain.toml pins '$pin'" >&2
            ok=0
        fi
    done
    if command -v rustc >/dev/null 2>&1; then
        active="$(rustc --version 2>/dev/null | awk '{print $2}')"
        if [ "$active" != "$pin" ]; then
            echo "  -> active rustc is $active but the repo pins $pin; run 'rustup update'" >&2
            echo "     (rustup honors rust-toolchain.toml, so this usually just needs the" >&2
            echo "     pinned toolchain installed)." >&2
            ok=0
        fi
    fi
    if [ "$ok" = 1 ]; then
        echo "pinned at $pin, local and CI agree"
    else
        fail=1
    fi
fi

step "cargo clippy --workspace --all-targets --all-features -- -D warnings"
if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
    fail=1
fi

# #481: a bare `if X <= 0.0 { return Err(PyValueError...) }` positivity guard
# admits NaN/+inf (both compare false) and silently corrupts the fit. New
# hyperparameter checks must route through ensure_finite_pos / ensure_finite_nonneg
# (or an explicit `!x.is_finite()` clause). Flag the raw anti-pattern before it
# lands. (perl, not `grep -P`, so this runs on the macOS pre-push hook.)
step "no NaN-admitting '<= 0.0' hyperparameter guards in the bindings (#481)"
antipattern=$(perl -ne '
    if ($prev =~ /^\s*if\s+[A-Za-z_][\w.]*\s*<=?\s*0\.0\s*\{\s*$/ && /return\s+Err\(PyValueError/) {
        print "  $ARGV:", ($. - 1), ": ", $prev;
    }
    $prev = $_;
    # Reset per file so $. is the file-local line number and $prev cannot leak
    # across the *.rs glob boundary.
    if (eof) { close ARGV; $prev = ""; }
' src/python/*.rs || true)
if [ -n "$antipattern" ]; then
    echo "  a hyperparameter guard admits NaN/+inf; use ensure_finite_pos/_nonneg:" >&2
    printf '%s\n' "$antipattern" >&2
    fail=1
fi

# Python-side checks need the built extension; prefer an active venv, else .venv-dev.
VENV="${VIRTUAL_ENV:-$PWD/.venv-dev}"
PY="$VENV/bin/python"
if [ -x "$PY" ]; then
    step "generated model tables in sync (scripts/gen_model_tables.py --check)"
    if ! VIRTUAL_ENV="$VENV" "$PY" scripts/gen_model_tables.py --check; then
        echo "  -> run 'python scripts/gen_model_tables.py' to regenerate" >&2
        fail=1
    fi

    step "type stub in sync with the compiled extension (test_stub_sync.py)"
    if ! VIRTUAL_ENV="$VENV" "$PY" -m pytest tests/test_stub_sync.py -q -p no:cacheprovider; then
        echo "  -> update python/topica/_topica.pyi to match the binding" >&2
        fail=1
    fi

    step "compat map in sync with the namespaces (scripts/gen_compat.py --check)"
    if ! VIRTUAL_ENV="$VENV" "$PY" scripts/gen_compat.py --check; then
        echo "  -> run 'python scripts/gen_compat.py' to regenerate" >&2
        fail=1
    fi

    step "agent cheat sheet in sync with topica.guide (scripts/gen_guide.py --check)"
    if ! VIRTUAL_ENV="$VENV" "$PY" scripts/gen_guide.py --check; then
        echo "  -> run 'python scripts/gen_guide.py' to regenerate" >&2
        fail=1
    fi
else
    printf '\n\033[33m[preflight] no dev venv (%s); skipping the generated-file checks.\033[0m\n' "$VENV" >&2
    echo "  (fmt + clippy still ran; CI will catch table/stub drift)" >&2
fi

if [ "$fail" -ne 0 ]; then
    printf '\n\033[31m[preflight] FAILED — fix the above before pushing (or: git push --no-verify).\033[0m\n' >&2
    exit 1
fi
printf '\n\033[32m[preflight] all checks passed.\033[0m\n'
