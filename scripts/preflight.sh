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
#   2. cargo clippy --workspace --all-targets --all-features -- -D warnings
#   3. generated model tables are in sync (scripts/gen_model_tables.py --check)
#   4. the .pyi type stub is in sync with the compiled extension (test_stub_sync.py)
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
else
    printf '\n\033[33m[preflight] no dev venv (%s); skipping the generated-file checks.\033[0m\n' "$VENV" >&2
    echo "  (fmt + clippy still ran; CI will catch table/stub drift)" >&2
fi

if [ "$fail" -ne 0 ]; then
    printf '\n\033[31m[preflight] FAILED — fix the above before pushing (or: git push --no-verify).\033[0m\n' >&2
    exit 1
fi
printf '\n\033[32m[preflight] all checks passed.\033[0m\n'
