# topica conventions for a new model

The authoritative sources in the repo are `CLAUDE.md`,
`docs/contributing/conventions.md`, and `tests/test_naming_conventions.py` (which
fails CI if a new model breaks the contract). This file distils what you apply
while implementing. When this file and the repo disagree, the repo wins — re-read it.

## The fitted-model surface (every model has this)

After `fit`, a model exposes the same shape, so downstream tooling (diagnostics,
coherence, reporting, ensemble) works uniformly:

- `fit(docs, …)` — trains in place and returns `self`. `docs` is `list[list[str]]`
  or a `Corpus`.
- `topic_word` — the (K, V) topic-word matrix (φ).
- `doc_topic` — the (D, K) document-topic matrix (θ).
- `top_words(n)` — the top-n words per topic.
- `save(path)` / `load(path)` — round-trippable persistence.

Add model-specific outputs as named attributes (e.g. `feature_effects`,
`topic_count_history`, `keyword_rate`) — but never at the expense of the four above.

## Naming and aliases

- **Different things get different names; the same thing keeps one name.** Do not
  introduce a synonym for a concept that already has a canonical name in the family
  (the convention test enforces a forbidden-synonym list).
- **Canonical first, reference-package alias second.** The primary parameter name
  follows topica's family convention; accept the reference package's spelling as an
  alias so users migrating from that package can switch with minimal edits. Example:
  keyATM accepts `times=` (the topica-canonical temporal name, shared with DTM) with
  `timestamps=` as an alias.
- **`iters`** is the canonical name for the iteration count — not `n_iter`,
  `max_iter`, `num_iterations`, `sweeps`, or `em_its`.
- **`seed=`** for the RNG seed; **`num_threads=`** for the rayon pool where the
  model is parallel.
- Covariate models share `features` / `covariates`; temporal models share `times`.
  Reuse these; do not coin new ones. Check `docs/contributing/conventions.md` for
  the current canonical set and the per-model alias map.
- **No deprecation cycles.** topica has effectively one user; when a name changes,
  change it. Do not add deprecation shims unless asked.

## The determinism guarantee (do not break it)

topica fits are reproducible to the bit:

- The variational/EM models are **bit-for-bit identical regardless of thread
  count**. Any parallel reduction must combine partial results in a fixed order
  (ascending document order), never in thread-completion order. A change that
  reorders floating-point sums breaks this.
- The collapsed-Gibbs samplers are reproducible from a **fixed seed and thread
  count**. Use the project RNG (PCG) seeded from `seed`; partition documents
  deterministically across threads.

When in doubt, prove it: fit twice (and at two thread counts) with the same seed and
compare `topic_word`/`doc_topic` with `numpy.array_equal` (exact, not `allclose`).

## Build and test gates (run after every change)

```bash
# Build the extension into the dev venv (note: .venv-dev, and VIRTUAL_ENV must be set).
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/maturin develop --release --features python

cargo test --lib                                                  # Rust unit tests
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/ -q
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/mkdocs build --strict   # docs must build clean
```

Always build `--release` (debug Gibbs/EM loops are far too slow). The new model must
pass `tests/test_naming_conventions.py`. Add a focused `tests/test_<model>.py` and at
least one Rust `#[test]` in `src/<model>.rs`. Write scratch/benchmark files to
`/private/tmp`, never into the repo or `/tmp`.

## File layout

- `src/<model>.rs` — one file per model; the fit/inference loop and its unit tests.
  (The CTM/STM/SAGE structural-variational cluster and shared numeric kernels live
  in the `topica-core` workspace crate — `topica-core/src/ctm.rs`,
  `topica-core/src/variational/`, `spectral.rs`, `linalg.rs` — which `topica`
  re-exports; new models usually go in `src/`, not `topica-core`.)
- `src/python/` — the PyO3 bindings, a directory module: one `src/python/<model>.rs`
  per model/family wired in via `use super::*` and registered in `mod.rs` (shared
  helpers in `arrays.rs`/`error.rs`/`save.rs`). Keep the binding thin; logic lives
  in the model's core file.
- `python/topica/_topica.pyi` — the type stub. Update it to match any binding
  signature you add or change, or the stub drifts from reality.
- `python/topica/` — the thin Python layer, if the model needs Python-side helpers
  (frames, formulas, plotting) beyond the binding.
- `python/topica/__init__.py` — export the new class so `topica.<Model>` resolves.
- `tests/` — pytest. `parity/` — cross-implementation checks (skip cleanly when the
  reference tool is absent). `docs/` — mkdocs (Material).

## Conventions are mechanically checked

`tests/test_naming_conventions.py` introspects every model's signatures. Before
opening the PR, run it and read any failure: it will name the offending parameter
and the canonical alternative. Fix the model to match; do not weaken the test.
