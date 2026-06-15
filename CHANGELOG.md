# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

## [0.20.0] - 2026-06-15

### Added

- Five new models, each validated against its reference implementation:
  - **`NMF`** — non-negative matrix factorization (multiplicative updates,
    Frobenius or generalized-KL), validated against `sklearn.decomposition.NMF`.
  - **`LSA`** — latent semantic analysis (truncated SVD), identical to
    `sklearn.decomposition.TruncatedSVD` (cosine 1.000000).
  - **`CombinedTM`** / **`ZeroShotTM`** — contextualized neural topic models
    (Bianchi et al. 2021); ZeroShotTM is topica's first cross-lingual model.
  - **`DETM`** — Dynamic Embedded Topic Model (Dieng, Ruiz & Blei 2019):
    embedding-factored topics that drift across time slices, fit by structured
    amortized variational inference (hand-coded LSTM), validated on the paper's
    UN and ACL corpora at the reference's own seed-to-seed noise floor.
- **Model registry + purpose taxonomy** — `topica.list_models(group=, brings=,
  inference=, determinism=, tag=)` over a seven-group registry that is the single
  source of truth for the README and docs roster.
- **Opt-in VAE options** on the amortized-VAE models (ProdLDA, ETM-vae,
  CombinedTM, ZeroShotTM), off by default: `contrastive=` (CLNTM-style InfoNCE
  regularization) and `prior="dirichlet"` (a true Dirichlet prior via the Weibull
  reparameterization).
- **Two bundled skills** (shipped under `.claude/skills/`): `add-topic-model`
  (the developer workflow for porting a model) and `topica-analysis` (the user
  analysis guide, the canonical source for the generated `AGENTS.md`).

### Fixed

- Windows CI: repo text files (README, docs, generated tables) are now read and
  written as UTF-8, fixing a `UnicodeDecodeError` under the Windows cp1252 default.

## [0.19.0] - 2026-06-15

### Changed

- Faster LDA, STM/CTM, and keyATM fits, with bit-for-bit identical results
  (same seed and thread count) — the default path is unchanged:
  - **STM/CTM**: the L-BFGS E-step now evaluates value and gradient in one fused
    pass and exploits the symmetry of the Hessian and its inverse, cutting the
    dominant per-document O(K^3) work. Up to ~2.2x faster at K=200, ~1.5x at K=60.
  - **keyATM**: a word-major shadow of the topic-word counts makes the per-token
    s=0 candidate loop a contiguous cache-friendly read instead of a strided
    column walk. Up to ~1.6x faster on large, high-K/V corpora.
  - **LDA**: the SparseLDA per-document setup is built directly from the
    document's own token assignments instead of an O(K) zero-and-scan, helping
    short-document and high-K fits (~1.5-1.9x) at no cost to long documents.

### Added

- Opt-in "turbo" approximations that trade exactness for speed, off by default
  (the default path stays bit-identical):
  - `LDA.fit(turbo_merge_every=m)` runs `m` parallel sweeps against private count
    tables before reconciling, amortizing the per-sweep merge. Helps when the
    merge dominates (large, wide-vocabulary, many-thread fits); it can hurt
    smaller corpora, so it is documented as situational. `m=1` (default) is exact.
  - `KeyATM.fit(turbo_alpha_stride=s)` subsamples documents in the base alpha
    slice-sampler (an unbiased estimate of the data term at ~1/s the cost),
    ~2.3-2.5x faster at `s>=4` with topic coherence preserved and a moderate
    shift in per-document topic mixtures. `s=1` (default) is exact. Base model
    only, with `estimate_alpha`.

## [0.18.0] - 2026-06-15

### Changed

- `search_k(...).best_k()` no longer falls back to bare `coherence` when no
  held-out set is supplied. Mean UMass coherence is roughly monotone-decreasing
  in K, so that fallback silently returned the smallest K in the grid (#167). It
  now selects the `"frontier"`: the K maximizing `z(coherence) + z(exclusivity)`
  across the scanned grid — the knee `plot_search_k` draws. The held-out default
  (`heldout_loglik` / `perplexity`) is unchanged when a held-out set is supplied.
  This changes the K returned by `best_k()` for grids scored on coherence alone.

### Added

- `SearchKResult.best_k(metric="frontier")` selects the coherence/exclusivity
  frontier explicitly (requires at least two K values to z-score). Explicit
  `best_k(metric="coherence")` on a multi-K grid now warns that UMass coherence
  is roughly monotone in K and recommends `metric="frontier"` or `held_out=`.

## [0.17.0] - 2026-06-14

### Added

- `KeyATM.fit` accepts `times=` as the canonical per-document time index (the
  same name DTM uses), with `timestamps=` kept as an alias. Resolves the temporal
  naming split; `tests/test_naming_conventions.py` now enforces `times` on every
  temporal model, and the naming-drift burn-down (#155) is clear (the other
  candidates — `check_every`, `prior_variance` vs `sigma`/`sigma0`, `labels` vs
  `groups` — are decided as intentionally distinct, documented in the conventions
  guide).
- `GDMR` gains `metadata_names` (label the continuous dimensions, via
  `fit(..., metadata_names=[...])`) and a `feature_names` property that labels the
  derived Legendre basis terms (e.g. `year^2`) aligned with `feature_effects` —
  kept as two distinct names because they are two distinct things (the D input
  dimensions vs the basis columns). Both survive save/load (#157).
- `TopicEffect.to_frame()` returns a tidy DataFrame with one row per feature
  (columns `topic`, `feature`, `coef`, `se`, `z`, `ci_low`, `ci_high`,
  `r_squared`); concatenating the per-topic frames from `estimate_effect` gives a
  long table with one row per (topic, feature) and no special-casing (#151).
- `search_k` now returns a `SearchKResult` (still a list of per-K rows) that
  carries `.directions` (whether higher or lower is better per metric) and a
  `.best_k(metric=...)` selector, so auto-selecting K cannot sort the wrong way
  (coherence is negative; the maximum is best) (#153).
- `GDMR`, generalized DMR (g-DMR; Lee & Song 2020): DMR over one or more
  continuous metadata variables via a Legendre-polynomial basis with a decay
  prior, plus topic distribution functions `tdf` / `tdf_linspace` that read the
  fitted prevalence surface at arbitrary metadata values. Mirrors `DMR`'s
  interface (`features=`, with `covariates=`/`metadata=` aliases) and is
  validated against tomotopy's `GDMRModel` (#148).
- API conventions guide (`docs/contributing/conventions.md`) documenting the
  shared cross-model vocabulary, enforced by `tests/test_naming_conventions.py`
  (#155).
- `STM` / `CTM` take `variational="diagonal"` (default `"laplace"`) for a
  mean-field diagonal posterior covariance `ν = diag(1/H_ii)` instead of the full
  Laplace `ν = H⁻¹`. This skips the per-document Cholesky+inverse, a large E-step
  speedup at high K (~4.8× at K=60). The off-diagonal posterior covariance is
  dropped (`eta_cov` is diagonal), so `topic_correlation` and method-of-composition
  standard errors are less precise; the default Laplace path is unchanged
  (bit-for-bit identical). Note the diagonal objective is not a strict ELBO lower
  bound, so the per-iteration bound can drift slightly rather than increase
  monotonically (#163).
- `STM.fit` / `CTM.fit` take `num_threads=` to cap the rayon worker pool used by
  the variational fit (default `None` = all cores). Results are bit-for-bit
  identical regardless of the worker count, so this controls only resource use
  (#164).
- Speed: the STM/CTM Σ (topic-covariance) M-step update is now parallelized over
  the K-1 rows. Profiling found this `O(N·K²)` cross-term was a large serial tail
  (~37% of fit wall-clock at N=20k, K=60) that left cores idle while the
  already-parallel E-step finished — the main reason large fits ran near
  single-threaded. Parallelizing it cut a representative fit ~28% (and more as N
  grows). Each Σ row still sums over documents in order, so fits remain bit-for-bit
  identical regardless of thread count (#164).
- Memory: the STM/CTM E-step now reduces its per-document sufficient statistics
  in chunks instead of collecting all N documents' results first, bounding the
  fit-time transient peak from O(N·K²) (~11 GB at N≈395K, K=60) to ~128 MB. The
  reduction still sums in document order, so fits stay bit-for-bit identical
  regardless of thread count or chunk size (verified against the previous
  implementation on a multi-chunk fit). This closes the fit-time-peak gap that
  `keep_eta_cov=False` (#160) left open — that removed the post-fit copy; this
  removes the during-fit peak (#165).
- Memory: `STS` `fit` takes `keep_eta_cov=True` (matching STM/CTM): with
  `keep_eta_cov=False` the per-document variational covariance is not stored
  (O(N·eta_dim) fit), the fit is bit-identical, and the covariance is recomputed
  exactly on demand for method-of-composition uncertainty (#162). The small
  E-step snapshots needed for that recompute (a β/κ and Σ copy, not per-document)
  are now retained only on the `keep_eta_cov=False` path, so the default path of
  STM/CTM/STS carries no extra per-fit state.
- Memory: `STM`/`CTM` `fit` take `keep_eta_cov=True`. With `keep_eta_cov=False`,
  the per-document variational covariance is not stored, dropping the fit
  footprint from O(N·K²) to O(N·K) (no extra per-document state is retained). The
  fit is bit-identical either way; method-of-composition uncertainty
  (`posterior_theta_samples` / `estimate_effect` with draws / `standard_errors`)
  recomputes the covariance exactly on demand, and the `eta_cov` property raises
  an actionable error when it was not kept (#160, part of #158).
- Memory: the logistic-normal models (`STM`, `CTM`, `STS`) now store the
  per-document variational covariance `eta_cov` as float32 instead of float64,
  halving the dominant memory term (the `(num_docs, K-1, K-1)` array). The
  `eta_cov` property now returns a float32 array; consumers that need float64
  (e.g. `posterior_theta_samples`, `estimate_effect`) upcast internally. The
  on-disk save format is unchanged (covariances are still serialized as float64),
  so existing saved models load unchanged (#159, part of #158).
- The covariate-design helpers `spline` and `interaction` are now exported at the
  top level as `topica.spline` / `topica.interaction`, matching the `formulas`
  docstring and reflecting that they build design-matrix blocks usable by any
  covariate model (DMR, STM, STS, KeyATM), not only STM. The `topica.stm.spline` /
  `topica.stm.interaction` paths still work (#137 follow-up).

### Documentation

- Covariates guide gains a single end-to-end recipe (`from_dataframe` →
  `design_matrix` → `search_k`/`fit` → `estimate_effect` → `to_frame`) and a note
  that all design/effect helpers are canonically top-level `topica.*` (the
  `topica.stm.*` paths remain as compatibility aliases) (#149, #152).
- `design_matrix` and `from_dataframe` docstrings now name the optional
  `topica[formula]` extra so the requirement is visible before runtime (#150).
- `estimate_effect` documents the two ways to specify the design (`X` +
  `feature_names` vs `formula` + `data`, the latter previously undocumented) and
  the invariant that the effects design must match the one used in `fit`; the
  covariates guide gains the same note. `STM.fit`'s `prevalence=`/`covariates=`
  pair is intentional (R-`stm` native name plus the universal cross-model alias,
  per the conventions guide), so both are kept (#154).
- Softened several cross-implementation claims to match what the artifacts show
  (design-review #02/#05/#06): keyATM/seededlda "verified word-for-word" → topic
  agreement via the reproducible `parity/` harness; BERTopic/Top2Vec "matching
  assignments" → "comparable structure; exact cluster assignments differ" (own
  PCA/UMAP + HDBSCAN); gensim credited for the coherence-pipeline conventions (the
  measures are Röder et al. and Mimno et al.) and "computed in the Rust core" →
  "co-occurrence counting in the core." `estimate_effect` now states it propagates
  per-document θ posterior uncertainty but not global-parameter (β/Σ/γ)
  uncertainty, so its SEs run slightly smaller than R `stm`'s `estimateEffect`; the
  Gadarian vignette is hedged as a single fit (confirm with `searchK`,
  `select_model`, `permutation_test`). `topic_correlation`'s docstring notes it is
  the raw/simple estimate (matching `stm`'s `topicCorr` default) and points to the
  closure-corrected `viz.topic_correlation(method="clr")`; the c-TF-IDF
  row-normalization is labeled a surface-compatibility convenience, not a
  probability claim.
- Added `parity/coherence_gensim_compare.py`, a cross-implementation check of c_v
  against gensim's `CoherenceModel`. topica's c_v ranks topics as gensim does
  (Spearman ρ ≈ 0.998 on long-document corpora, ≈ 0.98 on short ones) with a small,
  documented offset that grows for documents shorter than the c_v window; absolute
  c_v is not comparable across implementations, but within-corpus ranking is
  (design-review #04.1). Confirmed the DTM variational bound (`src/dtm.rs`) is a
  verbatim transcription of gensim's `sslm.compute_bound` and added a citation
  comment so the formula is not mistakenly "corrected" into divergence
  (design-review #04.2, not a bug).
- Corrected the LDA/MALLET attribution in the README, docs, and paper. `LDA` is a
  port of David Mimno's RustMallet that uses its own RNG (PCG, vs RustMallet's
  ChaCha8), so it is **not** byte-identical to RustMallet, contrary to the previous
  "binds RustMallet … byte-for-byte" claim. The byte-for-byte guarantee that
  `tests/test_cli_parity.py` verifies is internal — the Python binding versus
  topica's own bundled `train` CLI. The Java MALLET cosine-1.000 result is a
  planted-corpus sanity check, now labeled as such (design-review #01).
- Corrected two model descriptions: `ETM` is a logistic-normal topic model (not
  "Generative LDA," which implies a Dirichlet prior), and `HDP` learns the topic
  count with concentrations held fixed by default (steered by `gamma`), rather than
  freely "inferring" it (design-review #03).

## [0.16.2] - 2026-06-13

### Fixed

- `BERTopic`/`Top2Vec` no longer panic when `min_cluster_size` (or `min_samples`)
  exceeds the number of documents: the degenerate regime now resolves to a clean
  `num_topics=0` with the usual "lower min_cluster_size / add data" warning,
  instead of letting a `petal-clustering` MST panic escape into Python (#122).
- `term_topic_browser(...).to_html(path)` now writes the interactive figure to the
  given path (via an `_InteractiveFigure` wrapper that also delegates the Plotly
  figure's own methods), instead of silently doing nothing (#135).

### Documentation

- Paper: the validation and availability sections now credit each artifact to the
  script that produces it. The Section 6 speed numbers point to the actual
  `benchmarks/` timing scripts (`bench_stm.py`, `bench.py`, `k_crossover.py`)
  rather than `speed_vs_r.py` alone, and the K-selection and clustered-SE
  discussion is credited to the worked example in the docs (not `replication.py`)
  (#111).
- Paper: the Sentiment-discourse (`STS`) validation now runs on the published
  political-blog fit (`Poliblogs_results.RDS`, K=5, the worked example's own
  corpus) instead of the small gadarian K=3 corpus. `parity/sts_r_compare.py`
  recovers the reference topics at a topic-word cosine of 0.93 (read at the mean
  sentiment, where `STS` parks the topic signal), against a 0.97 STM baseline and a
  0.96 same-ecosystem ceiling (#110).
- `paper/replication.py` now drives the STM content-covariate and STS parity
  checks, probes for the R packages each check needs so a missing `quanteda`/
  `jsonlite` reports a clean SKIP, guards the effect-figure step behind its
  matplotlib/pandas dependency, and `paper/README.md` lists the full reproduction
  toolchain (#111).

## [0.16.1] - 2026-06-12

### Fixed

- `plot_report`'s topic-correlation panel now masks the always-1.0 diagonal and
  scales to the off-diagonal range, instead of drawing the raw matrix on a
  saturated +/-1 scale where the diagonal swamped the real structure. (The
  original 0.12.1 fix had only reached the standalone `viz.TopicCorrelation`.)
- Save/load now round-trips the retained MCMC `theta_draws` for the remaining
  collapsed-Gibbs models (DMR, LabeledLDA, SAGE, KeyATM), so method-of-composition
  standard errors survive a save/load round-trip for every model, not just LDA and
  SeededLDA (#102).

### Changed

- `plot_report`'s "Prevalence by class" panel is now a connected-dot (dumbbell)
  plot for up to five classes: one dot per class per topic joined by a line, with
  topics ordered by the between-class gap, so the class differences read directly.
  It falls back to the heatmap for more than five classes.

## [0.16.0] - 2026-06-12

### Fixed

- `SeededLDA` save/load is no longer lossy: the seed topic names, seed words, and
  residual-topic count are now serialized, so a loaded model reports the correct
  `num_topics` and `transform()` works instead of panicking (#98).
- `predicted_prevalence` no longer crashes on categorical covariates passed
  through a formula (`at=`/`contrast=`) or on the 2-element-sequence `contrast=`
  form; the training `formulaic` model spec is reused for prediction so factor
  levels stay consistent (#99).
- `permutation_test` now threads the permuted covariate into each refit for
  covariate-aware models (STM, DMR, KeyATM), matching `stm::permutationTest`;
  p-values use the `(1 + count) / (1 + n)` convention and drop NaN null entries
  (#101).
- The `_topica.pyi` type stub is back in sync with the compiled module (missing
  `save`/`load`, `log_likelihood_history`, `doc_names`, several `fit()` keywords,
  and `Corpus.from_documents` parameters added; a bogus `HLDA.coherence` removed),
  and a parametrized test now guards against future drift (#108).

### Changed

- Covariate, feature, embedding, and timestamp matrices are now checked for
  non-finite values (NaN/inf) at the boundary and raise a clear `ValueError`
  naming the parameter, instead of panicking (KeyATM) or silently producing
  garbage estimates (STM, DMR) (#100).
- `top_words`/`top_documents` and related rankings sort with `f64::total_cmp`,
  so a stray NaN can no longer panic them into a `PanicException`. `BERTopic`
  and `Top2Vec` raise a clear `RuntimeError` from `transform`/`top_words` when
  clustering found no topics, rather than returning empty `(n, 0)` output.
  `u_mass` coherence against an external reference corpus no longer rewards a
  top word absent from the reference with a large positive score (#103).

### Performance

- keyATM's multithreaded sweep reconciles the topic-word counts with a sparse,
  parallel merge (`parallel_sweep_keyatm`), reducing the fixed per-sweep merge
  cost on many-thread fits (#84, #97).

### Changed

- API naming consistency (with backward-compatible aliases): the convergence
  tolerance is now `convergence_tol` in `fit()` for every iterative model
  (`em_tol` still works and warns; for the neural models `convergence_tol` is
  also accepted in `fit()` and overrides the constructor value). The topic-word
  prior is `beta` everywhere (`eta` kept as a deprecated alias on HDP/HLDA). A
  `covariates=` keyword is now accepted on every covariate model as an alias of
  the domain name (`prevalence=` for STM/STS, `features=` for DMR, which keep
  working; passing both raises a clear error). Verbosity is `progress_interval`
  (`report_interval` deprecated on HDP/GSDMM/KeyATM). `num_threads` is accepted
  in both the constructor and `fit()` (fit overrides) on LDA and KeyATM. SAGE's
  `burn_in` default is now 200, matching LDA/DMR. `alpha_sum` is unchanged (it
  is the sum over topics, intentionally distinct from a per-topic `alpha`). (#107)
- API consistency: `transform()` now takes `iters` (the canonical name used by
  `fit()`); the old `iterations=` keyword still works but raises a
  `DeprecationWarning` (#104). `SAGE.top_words` now matches every other model's
  shape, `top_words(n=10, *, topic=None, group=None)`, so `n` is the first
  positional argument and `topic=None` returns all topics; **breaking** for code
  that passed the topic index positionally (#105). The embedding models share one
  `transform(data, doc_embeddings=None)` signature and raise a clear `ValueError`
  when a required input is missing; **breaking** for `FASTopic.transform(emb)`
  called positionally, which now needs `transform(doc_embeddings=emb)` (#106).
- **Breaking (save format):** model files now carry an 8-byte header (magic,
  format version, model tag). Loading a file saved by an earlier version, or
  loading a file saved as the wrong model, now raises a clear error instead of
  panicking or silently misreading. Models saved before this release must be
  re-fit and re-saved. `LDA` and `SeededLDA` save/load also round-trip the
  retained MCMC `theta_draws` (so method-of-composition standard errors survive a
  round-trip) and the LDA sampler-backend flags (#98, #102).

### Added

- `CTM(...).fit(..., inference="svi")` adds a stochastic variational inference
  backend (online VB, Hoffman et al. 2013) for the logistic-normal core, for
  corpora too large to sweep in full each EM step. The global topics, mean, and
  covariance update from minibatches (`batch_size`, default 256) with a Robbins-
  Monro step `rho_t = (tau + t)^(-kappa)` (`tau` default 64, `kappa` default
  0.7); `iters` becomes the number of epochs. Each minibatch still runs STM's
  Laplace E-step per document, so the variational quality per token matches
  `"batch"`; the win is that one epoch touches every document with only
  minibatch-sized global state. It is deterministic for a seed. The full-batch
  variational EM remains the default (`inference="batch"`); SVI does not retain a
  per-iteration `bound`/`fit_history` trace and ignores `em_tol`.

- `KeyATM(..., sampler="cvb0")` adds a CVB0 backend for the base keyATM model:
  deterministic collapsed-variational inference over the (topic, keyword-switch)
  states, with a soft responsibility per (document, word) cell that mirrors the
  Gibbs conditional (token-weighting included). It is an **opt-in, non-R-parity**
  estimator (a different inference method, so it does not reproduce R keyATM),
  restricted to the base model — it errors with covariates, timestamps, or a
  prior_offset, which stay Gibbs-only — and produces no MCMC `theta_draws`. Use
  it when reproducibility/quality matters more than R-faithfulness. Default stays
  `"sparse"`.

- `LabeledLDA(..., sampler="cvb0")` runs the CVB0 backend with the per-document
  label set applied as a *mask* on the responsibilities (γ is zero off the
  allowed topics). This is the supervised model WarpLDA could not serve — its
  masked proposals would mix at a fraction of a percent — whereas masking is
  free in CVB0: it enforces the supervised constraint exactly (zero θ off the
  label set), deterministically, and tends to higher coherence. No MCMC
  `theta_draws`. Default stays `"sparse"`.

- `DMR(..., sampler="cvb0")` and `SeededLDA(..., sampler="cvb0")` extend the CVB0
  backend to those models — DMR with a per-document α (and the soft expected
  counts `E[n_dk]` feeding the λ optimizer directly, a cleaner fit than the
  hard-count sparse/warp paths), SeededLDA with the asymmetric seed β. Same
  deterministic, higher-coherence-at-larger-K, no-`theta_draws` trade as LDA's
  CVB0. Default stays `"sparse"`; the CVB0 SeededLDA path does not yet support
  `doc_topic_prior`.

- `LDA(..., sampler="cvb0")` adds collapsed variational Bayes, zeroth-order
  (Asuncion et al. 2009) as a deterministic, non-MCMC inference backend for the
  same LDA model. Each (document, word-type) cell keeps a soft topic
  responsibility updated from expected counts, so a fit is exactly reproducible
  for a seed and has no burn-in. It tends to give higher topic coherence than
  the samplers, increasingly so at larger K (on a 2,000-document poliblog
  subsample at K=100, mean c_v -68.5 vs -79.1 for `"sparse"`), at the cost of
  O(K)-per-token compute, so it is slower, not faster (~47s vs ~10s at K=100).
  Use it when topic quality matters more than fit time; it produces no MCMC
  theta draws (`theta_draws` is None). Default stays `"sparse"`.

- `SeededLDA(..., sampler="warp")` runs the WarpLDA backend (a seeded word phase:
  the word-proposal and its acceptance carry the asymmetric seed β
  `β_{k,w} = β + seed_weight·[w ∈ seeds_k]` and the per-topic normalizer
  `β_sum_k`). SeededLDA's default sparse sweep scores all K topics per token, so
  the win is even larger than for plain LDA: on a 2,000-document poliblog
  subsample at K=500 the warp path fits in ~2.6s against ~111s for `"sparse"`
  (~40x) at comparable coherence, and stays nearly flat in K. Default stays
  `"sparse"`; `"warp"` does not yet support `doc_topic_prior`.

- `DMR(..., sampler="warp")` runs the WarpLDA backend for DMR (a per-document-α
  doc phase: the doc-proposal and its acceptance use each document's
  `α_{d,k} = exp(λ_k · x_d)`, with the λ optimization loop unchanged). Same
  large-K win as LDA: on a 2,000-document poliblog subsample at K=500 it fits
  ~2.4x faster than the default `"sparse"` DMR sweep at comparable coherence,
  widening as K grows. Default stays `"sparse"`. Enabled by the shared per-doc
  WarpLDA doc phase, so the LDA hot path is untouched.

- `LDA(..., sampler="warp")` adds the WarpLDA cache-efficient two-pass
  Metropolis-Hastings sampler (Chen et al., 2016). Its per-sweep cost is flat in
  K (an O(1)-per-token MCEM scheme with delayed count updates), so it is the
  recommended sampler for large-K, fine-grained models. On a 2,000-document
  poliblog subsample at K=1,000 it fits ~4.7x faster than the default
  `"sparse"` sampler *and* reaches higher topic coherence (sparse is too slow to
  mix well at that K), and it dominates `"lightlda"` outright (several times
  faster and far higher coherence). At the topic counts typical of
  social-science work (K up to ~200) `"sparse"` remains the best
  quality-per-wall-clock choice and stays the default. The MH acceptance ratios
  were cross-checked against the reference C++ kernel (thu-ml/warplda).

- `LDA(..., init="spectral")` seeds the initial token-topic assignment from a
  deterministic anchor-word topic-word matrix (the same spectral recovery STM
  and CTM use) instead of a uniform random draw. It does not speed convergence,
  but it improves topic coherence at larger K (a robust +2 to +3 mean-coherence
  points across seeds at K=50 and K=100 on the poliblog corpus; a wash at small
  K), the fine-grained regime where the sparse sampler already pays off. It
  falls back to the random draw when the corpus is too small for anchor
  recovery. The default stays `init="random"`, so MALLET byte-parity and
  same-seed determinism are unchanged.

### Changed

- The collapsed-Gibbs samplers (LDA, DMR, LabeledLDA, SeededLDA, KeyATM, PA, PT,
  HDP, GSDMM, SAGE) now draw from a fast non-cryptographic PRNG (PCG) instead of
  ChaCha8. Gibbs sampling needs uniform draws, not cryptographic entropy, and PCG
  is faster: single-threaded, HDP is ~2x faster, LDA ~10% and keyATM ~9% (#67).
  Fits remain reproducible from a fixed seed, but **the random stream changed**,
  so a given seed now yields different (still-deterministic) topics than in
  0.15.0; pin a topica version if you need to reproduce an earlier fit exactly.
  The variational models (CTM, STM, STS, DTM, supervised LDA, ETM, ProdLDA,
  FASTopic) and the embedding-cluster models are unchanged.

### Fixed

- `HDP` no longer runs away to hundreds of topics on real corpora (#68). The
  concentration resampler was a positive-feedback loop: the Escobar-West update
  draws `gamma` from `Gamma(a + K, ...)`, whose mean grows with the topic count
  `K`, so more topics raised `gamma`, which created more topics, irreversibly
  (K reached 774 with gamma at 102 over 800 sweeps on a 3,500-document corpus).
  `resample_conc` now defaults to `False` (fixed concentrations give a stable,
  reproducible topic count; `gamma` sets the granularity directly), and the
  opt-in resampling path caps the concentrations so it stays bounded. Default
  concentrations remain `alpha=0.1`, `gamma=0.1` (the reference convention).

## [0.15.0] - 2026-06-10

This release completes the structural-topic-model and keyATM drop-in parity work
and rounds out the model-agnostic effect-estimation surface. It also moves the
heavy CI (wheels, sdist) to release tags and builds the test job optimized, so a
normal push runs only the fast test suite.

### Added

- `permutation_test(model, covariate, ...)` for a binary prevalence covariate: a
  distribution-free check on whether a topic's prevalence differs across the two
  groups, returning a `PermutationResult` per topic (#36).
- `select_model` / `plot_models`: fit N models at a fixed K under different seeds
  and pick the best by a held-out or coherence criterion, mirroring R `stm`'s
  `selectModel`; returns a `SelectModelResult` (#37).
- `prep_documents` / `plot_removed`: R `stm`-style preprocessing diagnostics that
  report how many documents, words, and tokens each vocabulary threshold removes,
  with metadata re-alignment via the `Corpus`'s kept indices (#41).
- A uniform convergence interface on every iterative model: `model.fit_history`
  (per-iteration `(iter, objective)`) and `model.converged`. The collapsed-Gibbs
  models gained an opt-in early stop (`convergence_tol` / `check_every`, default
  off so the full `iters` run is bit-for-bit unchanged); the variational models
  trace and early-stop on the ELBO (#46).
- `prevalence_ci(model, groups, ...)`: model-neutral per-group topic-prevalence
  credible bands read directly from a model's posterior theta draws (the
  draws-based companion to `by_strata`). `time_prevalence_ci(model, timestamps)`
  is the dynamic-keyATM wrapper that pins the period order to `time_labels`, so
  the dynamic time trend now carries the HMM posterior's own uncertainty rather
  than a generic ribbon (#42).
- Covariate-aware `stm.transform(model, docs, prevalence=/formula=/X=)`: held-out
  topic inference that builds each new document's prior from its covariates and
  the fitted `gamma` (`mu_d = X_d gamma`), matching R `stm`'s `fitNewDocuments`.
  A model-neutral `align_corpus(new_docs, model)` maps new tokens onto the fitted
  vocabulary (dropping out-of-vocabulary tokens) before transform (#39).
- `STM.fit(gamma_prior="pooled"|"l1", gamma_enet=...)`: an L1/elastic-net prior on
  the prevalence coefficients, fit by coordinate descent with an AIC-selected
  penalty, for high-dimensional prevalence designs (a factor with many levels).
  `"pooled"` (ridge, the default) is unchanged; `gamma_enet` is the elastic-net
  mix (R `stm`'s `gamma.enet`) (#40).

### Fixed

- `search_k(held_out=...)` now composes with a `make_heldout` split: it dispatches
  on the `Heldout` type and reports the held-out log-likelihood, instead of
  raising a `TypeError` from the legacy perplexity path (#55).

### Changed

- CI: `build-wheels` and `sdist` run only on release tags (`v*`) and manual
  dispatch, not on every push/PR; the wheels are consumed only by the release
  job. The 3-platform test job still runs on every push/PR and now builds with
  `--release` plus a cached Rust toolchain, cutting the test legs from roughly
  twenty minutes to a few. Committed tests no longer assume a macOS-only
  `/private/tmp`.

## [0.14.0] - 2026-06-10

This release makes the estimator interface uniform across the whole library and
adds two publication-grade quantities of interest. Every estimator now meets a
documented contract, checked in CI.

### Added

- `predicted_prevalence(model, ...)`: predicted topic prevalence at chosen
  covariate values, with difference contrasts and continuous prediction curves,
  and simulation-based confidence intervals. Model-agnostic (STM, CTM, the
  covariate keyATM, LDA, ...), built on the method-of-composition draws, so it is
  the same call regardless of model family. A `viz.predicted_prevalence_plot`
  renders the forest and curve figures (#35, #43).
- `make_heldout` / `eval_heldout`: R `stm`-style document-completion held-out
  log-likelihood, model-agnostic via each model's `transform`; `search_k` now
  reports a held-out metric for STM/CTM, not only LDA (#38).
- Estimator conformance facility: `topica.check_conformance(model)`, a
  registry-driven `tests/test_conformance.py`, and a contributor contract at
  `docs/contributing/estimator-contract.md`. New estimators that drop part of
  the contract fail CI.
- `theta_draws` and `doc_lengths` on the remaining Dirichlet models (DMR, SAGE,
  PA, PT, HDP, LabeledLDA, SupervisedLDA), so `composition_theta`,
  `standard_errors`, and `predicted_prevalence` work for them with no `corpus=`
  re-thread. SupervisedLDA draws from its variational Dirichlet posterior.
- Held-out `transform` on KeyATM, SeededLDA, SAGE, PA, and PT, so held-out
  perplexity, `eval_heldout`, and out-of-sample inference now work for the
  keyword, seeded, and anchored models.
- Settable `topic_names` on every estimator (default `["topic_0", ...]`).
- `coherence`, `save`/`load`, and `doc_names` on the neural and cluster models
  (ETM, FASTopic, ProdLDA, BERTopic, Top2Vec) where they were missing.

### Changed

- **Breaking:** the fit iteration count is the canonical keyword `iters` on every
  estimator (previously `iterations` for the collapsed-Gibbs models and
  `em_iters` for the variational ones); `search_k` likewise takes `iters`. No
  deprecation aliases.
- **Breaking:** ETM, ProdLDA, and FASTopic take the training length as
  `fit(iters=...)` rather than a constructor `epochs` / `em_iters` argument.

## [0.13.0] - 2026-06-10

### Added

- The Gibbs/Dirichlet models (`LDA`, `KeyATM` base/covariate/dynamic,
  `SeededLDA`) retain thinned post-burn-in MCMC document-topic draws as
  `model.theta_draws` (shape `(num_draws, num_docs, num_topics)`, f32). On by
  default (`keep_theta_draws=True`, `num_theta_draws=25`); pass
  `keep_theta_draws=False` to skip the store. `composition_theta` (and
  `standard_errors` / `estimate_effect` with `method="composition"`) prefers
  these real cross-sweep posterior samples over the within-document Dirichlet
  approximation, and needs no `corpus=` when they are present. Retention rides
  on sweeps that already run, so it adds negligible fit time (#31).
- The same models expose `model.doc_lengths` (per-document token counts, in
  `doc_topic` row order), so the Dirichlet-approximation fallback is also
  self-sufficient: `composition_theta(model)` works without re-threading the
  `Corpus`, even with `keep_theta_draws=False`. Passing `corpus=` still takes
  precedence (#32).

### Changed

- Standard errors for the Gibbs models now reflect genuine topic-estimation
  uncertainty (the cross-sweep posterior variance of theta), which grows when
  topics overlap and shrinks when the model is confident. Values therefore
  differ from 0.12.1, where the Dirichlet approximation added length-only
  `1/N_d` sampling noise regardless of identifiability; the new intervals can be
  wider or narrower depending on the corpus. Fit with `keep_theta_draws=False`
  to recover the prior approximation behavior.

## [0.12.1] - unreleased (rolled into 0.13.0)

### Fixed

- The `viz` topic-correlation panel (`TopicCorrelation`, and the correlation
  sub-panel of `plot_report`) masks its always-1.0 diagonal instead of drawing
  it. The self-correlation carried no information yet saturated the diverging
  color scale and visually dominated the panel; the diagonal now renders as a
  neutral background so the off-diagonal structure reads on a scale set by the
  strongest real correlation. `to_frame()` is unchanged and still reports the
  true diagonal.

### Docs

- Paper: added `ProdLDA` to the model-family table (count-based, with its
  bibliography entry), and a worked example that aligns two model families and
  compares their covariate effects with method-of-composition uncertainty.

## [0.12.0] - 2026-06-08

### Added

- `alpha` getter on the collapsed-Gibbs Dirichlet models that lacked it —
  `KeyATM`, `SeededLDA`, `LabeledLDA`, `SupervisedLDA`, `DMR`, `PA`, `PT`, and
  `SAGE` — returning the per-topic document-topic Dirichlet prior aligned with
  `doc_topic`'s columns (the estimated/asymmetric prior where one is fitted, the
  symmetric prior otherwise, and `exp(lambda_intercept)` for `DMR`'s
  per-document prior). This is what `effects.model_family` keys "dirichlet" off,
  so it is the mechanism behind the `composition_theta` fix below (#20, #21).

### Fixed

- `effects.model_family` misclassified every collapsed-Gibbs model except `LDA`
  and `HDP` as `"none"`, so `composition_theta` raised for them and `viz`
  effect/uncertainty panels silently fell back to point estimates. With `alpha`
  now exposed, `KeyATM`, `SeededLDA`, `LabeledLDA`, `SupervisedLDA`, `DMR`, `PA`,
  `PT`, and `SAGE` are correctly `"dirichlet"`; `GSDMM` stays `"none"` by design
  (a Dirichlet mixture, not an admixture) (#20, #21).
- `dirichlet_theta_samples` double-counted the symmetric prior on the `prior > 0`
  path, biasing draws toward uniform; the default `prior = 0` path is unchanged
  (#26).
- `find_thoughts` and `document_intrusion` (and `representative_docs` /
  `topic_info` through them) now raise on a `texts` / `doc_topic` length
  mismatch, the guard their siblings already had, so a document dropped by
  vocabulary pruning can no longer be returned in place of a real one;
  `plot_report`'s per-class panel gets the same alignment check (#24).
- Stopped swallowing exceptions that quietly degraded results: bootstrap refits
  and held-out `transform` now choose their call arity by inspecting the
  signature instead of treating any `TypeError` as an arity mismatch (which had
  re-run every resample at the default seed); `quality_frontier` warns when a
  windowed `coherence_type` is requested without `texts`; `plot_report` warns and
  names any panel it drops; the top-words fallback warns before discarding custom
  (e.g. FREX) weighting (#25).
- API-surface drift: the `DMR` type stub (copied from `STM`) now matches the real
  `fit(data, features, ...)` signature and exposes `feature_effects` (not the
  nonexistent `prevalence_effects`); `coherence` and the analysis surface work
  for `SAGE` via its group marginal and reject `DTM`'s time-sliced `topic_word`
  with a clear message; the `viz` capability descriptor marks `HLDA` and `DTM`
  (no usable `doc_topic`) as not soft-theta; `bootstrap_stability` accepts a
  `Corpus`, as its docstring promised (#27).

## [0.11.0] - 2026-06-07

### Added

- `topica.viz` — four more panels, continuing the toolkit's deferred roadmap:
  - `topic_health` — flags **dead** topics (expected mass share below
    `min_mass_frac`) and **near-duplicate** topics (φ-cosine above `dup_threshold`),
    off the same `topic_sizes` / topic-word surfaces the rest of the toolkit uses.
    Essential for honest reporting and for HDP, which returns many near-zero-mass
    topics by construction.
  - `prevalence_heatmap` — a groups × topics heatmap of mean topic prevalence
    (`by_strata`), with method-of-composition intervals in `.to_frame()` when a
    corpus and `nsims` are given.
  - `topics_over_time` — per-topic prevalence trajectories as small multiples (the
    readable replacement for a streamgraph), with optional method-of-composition CI
    ribbons.
  - `topic_correlation` — the honest, closure-corrected correlation layer
    (`clr` / `partial` / η-space `eta` / labeled-biased `raw`), drawn as a
    zero-centered diverging heatmap; refused for hard/degenerate-θ cluster models.
  - `dashboard()` now assembles these by introspection: topic-health always, the
    group heatmap with `groups=`, the time small-multiples with `timestamps=`, and
    the correlation layer for soft-θ models.
- `topica.project(data, n_components=2, method=...)` — a numpy-native projection
  primitive backed by topica's own Rust core: `"pca"` (default, deterministic,
  distance-faithful), `"umap"` (`umap-rs`), or `"tsne"` (new **`bhtsne`** Barnes-Hut
  reducer, pure Rust). UMAP and t-SNE warn that they are non-metric and not
  reproducible. No Python UMAP/sklearn dependency.
- `topica.viz.document_map` — the deferred 4th panel: a 2-D projection of the
  *document* cloud (a supplement figure). Coordinates come from the document
  embeddings you pass, or, for a count/soft-θ model, the clr-transformed θ simplex;
  a hard-θ cluster model with no embeddings is refused. PCA reports variance
  explained; UMAP/t-SNE carry the non-metric caveat and the seed. Density via alpha
  clouds / hexbin (never convex hulls), Okabe–Ito palette for small K else
  gray-all + `highlight_topic=`, a separate `-1` outlier layer, and stratified
  subsampling with a "showing N of D" badge. `dashboard(..., doc_embeddings=)` adds
  it.
- `topica.viz.document_inspector` — read one document the way the model read it: its
  θ mixture, its words shaded by attributed topic (`argmax_t p(t | w, d)` from θ and
  φ, so it needs no per-token assignments), and the `find_thoughts` neighbors of its
  dominant topic. Refused for hard/degenerate-θ cluster models.
- `topica.viz.content_covariate` — for an STM/SAGE content model, one topic's wording
  across covariate groups as a words × groups `p(w | topic, group)` heatmap (the
  union of each group's top words), surfacing the per-group distribution instead of a
  reference snapshot. `.contrast(...)` wraps the model's `word_contrast`. Refused for
  a model fit without a content covariate.
- `dashboard()` adds the content-wording panel for content models, and the inspector
  when `inspect_doc=` is given. The generic panels now collapse a content model's
  per-group (K, G, V) topic-word to its marginal, and the dashboard assembles every
  panel best-effort (a model that cannot support one is skipped, not fatal).

### Changed

- The interactive (`.to_html()`) backend is now **Plotly only**; the Altair
  dependency is dropped. `term_topic_browser` (a seriated heatmap plus a topic
  dropdown) and the dashboard report render with Plotly (WebGL), the same stack as
  the document map.
- Packaging simplified: the static `viz` and interactive `viz-interactive` extras
  are **merged into one `viz` extra** (matplotlib, pandas, scipy, plotly), and a new
  **`all`** extra installs everything in one shot. The base install stays
  `numpy`-only.
- `viz` design polish (from two independent expert reviews): the topic-similarity
  heatmap anchors its color scale at 0 (no contrast-stretch) and labels the colorbar
  `1 − <metric>`; the covariate effect plot drops sign-coded red/blue for a single
  neutral color (position already encodes sign); heatmaps share `SEQ`/`DIV` colormap
  constants; the coherence frontier gains a prevalence size legend; `topics_over_time`
  shares its y-axis by default; `search_k` is faceted (one metric per panel) instead
  of a triple twin-axis.

### Fixed

- CTM/STM expose `topic_covariance` (the fitted logistic-normal prior Σ over η,
  shape (K−1, K−1)), and `viz.topic_correlation(model, method="eta")` now uses it —
  the model's own covariance rather than an empirical re-correlation of η posterior
  means, which it had been mislabeling as "the model's covariance."
- `viz.term_barchart` FREX / relevance / score modes no longer crash on a SAGE
  content model (they now route through the group-averaged marginal, as `prob`/`lift`
  already did); the descriptor advertised these modes but they raised.
- `viz.dashboard` records skipped panels in `.skipped` and warns, instead of
  silently swallowing every failure (so a real error is visible, not indistinguishable
  from "not applicable").
- `find_thoughts` uses `argpartition` for the top-n (O(D)) rather than a full sort.
- The document map no longer prints a `seed=` for UMAP/t-SNE (neither fit is
  reproducible), and the docs no longer claim the interactive browser links a
  heatmap click to the barchart (it is a dropdown).

### Fixed

- Input validation hardened against adversarial edge cases:
  - Non-finite float hyperparameters (`NaN`/`Inf` for `beta`, `alpha`,
    `prior_variance`, `chain_variance`, `eta`, `alpha_sum`, and the rest) are now
    rejected at construction instead of silently producing a `NaN` fit.
  - A corpus with no words — all documents empty, or everything pruned by frequency
    filtering — is rejected at fit instead of yielding a degenerate `(K, 0)` model.
  - `coherence` / `topic_diversity` raise a clear error on a non-integer `topn` or a
    raw `topic_word` matrix, and `coherence` errors on an empty reference corpus
    instead of returning `NaN`.
  - `frex` rejects frequency weights outside `[0, 1]`.
- `coherence` / `topic_diversity` now accept any object satisfying the analysis
  contract (`topic_word` + `vocabulary`): top words are derived from the matrix when
  the model exposes no `top_words` method.

## [0.10.0] - 2026-06-06

### Added

- `AGENTS.md` — a working guide for LLM agents (Claude Code, Cursor, …) helping a
  social scientist run topica. It maps the API onto the text-analysis workflow
  (question → corpus → choose K → fit → validate → measure effects → report) with
  explicit handoffs, and draws the line on what the researcher owns (the question,
  K, topic labels, covariate choice, whether a result matters) versus what topica
  and the agent supply (mechanics, honest diagnostics, refusal to fabricate
  uncertainty).
- `topica.viz` — a manuscript-first visualization toolkit (the honest successor to
  pyLDAvis). Each view is a panel with `.to_frame()` (the numbers, always),
  `.to_png()` (matplotlib, for papers), and `.to_html()` (Altair, for the
  interactive subset). Panels read a per-model capability descriptor and switch
  their statistics/labels on it: c-TF-IDF `topic_word` disables the FREX/lift
  modes and is labeled as such, effect-plot CIs are refused where there is no θ
  posterior (and ghosted where the bootstrap flags a topic unreliable), and
  uncertainty is labeled for what it is. Panels: `coherence_frontier`, `search_k`,
  `effect_plot`, `term_barchart`, `topic_similarity` (a seriated K×K heatmap, the
  pyLDAvis replacement), `term_topic_browser` (linked interactive), and a
  `dashboard()` composite. New extras: `topica[viz]` (matplotlib/pandas/scipy) and
  `topica[viz-interactive]` (altair).

## [0.9.0] - 2026-06-06

### Added

- `topica.mmr(model, word_embeddings, diversity=...)` — maximal-marginal-relevance
  top words: rerank a topic's candidate words to cut redundant near-synonyms,
  balancing `topic_word` relevance against word-embedding similarity (BERTopic's
  `MaximalMarginalRelevance`). Accepts a model or a `(K, V)` matrix.
- `save` / `load` for the embedding-cluster models (`BERTopic`, `Top2Vec`), so a
  discovered fit can be frozen and reloaded — the way to keep a good (stochastic)
  UMAP discovery fit, since the prediction phase is deterministic. The loaded
  model's `transform` reproduces the original.
- `topica.add_ngrams(docs, ngram_range=(1, 2), min_df=...)` — expand pre-tokenized
  documents with contiguous n-grams (the mechanical analog of scikit-learn's
  `CountVectorizer(ngram_range=, min_df=)`), so an embedding model's c-TF-IDF topic
  words can include bigrams. Keeps every document, so it stays aligned with
  per-document embeddings. The exhaustive complement to `learn_phrases`.
- `reducer="umap"` now ships in the wheel for `BERTopic` / `Top2Vec` (opt-in at
  runtime, no special build). PCA stays the default. The UMAP discovery fit is not
  reproducible (the `umap-rs` optimizer's negative sampling is unseeded) and emits
  a warning saying so; following BERTopic's fit-vs-predict split, the prediction
  phase is deterministic regardless — `transform` never re-runs the reducer — so a
  fitted model still maps documents reproducibly. Use `reducer="pca"` for a fully
  reproducible fit, or `clusterer="kmeans"` to empty the `-1` bucket deterministically.
- `topica.diagnostics(model, texts)` — a one-call per-topic table (coherence,
  exclusivity, FREX, size, prevalence, top words, and optional bootstrap
  stability) as a pandas DataFrame, consolidating the scattered quality
  functions. It reads a model's analysis surface, so it works for every model
  and sidesteps the model-vs-matrix first-argument friction.
- `topica.perplexity(model, held_out)` — model-agnostic document-completion
  held-out perplexity (infer each held-out document's mixture from half its
  tokens, score the other half), a K-comparable signal for justifying a topic
  count across the generative models. (`LDA` keeps its rigorous left-to-right
  estimator as `LDA.perplexity` / `LDA.evaluate`.)
- `bootstrap_stability(..., reference=model)` measures stability of an
  already-fitted model's topics (matching resamples back to it) rather than a
  fresh full-corpus fit.

### Changed

- The post-hoc analysis module moved from `topica.diagnostics` to
  `topica.validation`, freeing the verb-like `diagnostics` name for the new
  one-call function. Its helpers stay importable (`from topica import
  validation`) and every function remains available top-level (`topica.frex`,
  `topica.coherence`, …).

## [0.8.0] - 2026-06-06

### Added

- `topica.standard_errors(model, corpus, of=..., method=...)` — one entry point
  for uncertainty on the quantities people publish (#15). `method="composition"`
  (default) auto-detects the model family, draws the right θ posterior
  (logistic-normal for STM/CTM, Dirichlet for the Gibbs models), and pools by
  Rubin's rules for `of="effect"`/`"prevalence"`. `method="bootstrap"` refits on
  resampled documents for `of="top_words"` and the embedding models, matching
  topics across refits and reporting `alignment_quality`/`alignment_margin` so it
  can flag and suppress SEs where the matching is unstable (split/merge or
  indistinct topics).
- `Corpus.doc_lengths` — per-document token counts in the pruned vocabulary,
  parallel to a model's `doc_topic` rows (needed by `dirichlet_theta_samples`).
- `estimate_effect` and `by_strata` now accept the fitted model directly and draw
  θ internally (with `corpus=`/`nsims=`), so the sampler no longer has to be
  wired by hand. `topica.model_family(model)` exposes the detection.

## [0.7.1] - 2026-06-06

### Added

- `BERTopic` and `Top2Vec` accept `clusterer="kmeans"` / `"agglomerative"` with
  `num_clusters=K`, a swappable alternative to the default HDBSCAN that assigns
  every document to a cluster (no `-1` noise bucket) (#7).
- `topica.report(model)` is now a callable one-line overview (an alias for
  `summary`), so the natural `report(model)` call works instead of raising
  `'module' object is not callable` (#12).
- A bundled `text -> llm_embed -> BERTopic` example and an `llm_embed`
  cross-reference in every embedding model's docstring (#5).

### Changed

- `Top2Vec.top_words()` now returns the centroid representation (vocabulary
  nearest the cluster centroid) by default when fit with `word_embeddings`, so
  its headline output is distinct from `BERTopic`'s shared c-TF-IDF; pass
  `representation="c-tf-idf"` for the shared view. `topic_neighbors` is now
  `(topic, *, n=10)`, so `topic_neighbors(0, n=8)` reads naturally (#8).
- `frex`, `label_topics`, `relevance`, `topic_correlation`, and `find_thoughts`
  now accept a fitted model or the raw matrix as the first argument (vocabulary
  derived from the model when omitted), matching `exclusivity` and the intrusion
  tests; a bare matrix with no vocabulary raises a clear message (#10).
- The model-neutral analysis surface moved from the `topica.report` module to
  `topica.analysis` (its functions remain available top-level, e.g.
  `topica.topic_info`, `topica.plot_report`), freeing the verb-like `report`
  name for the new callable (#12).

### Fixed

- A negative count (`num_topics`, `num_pseudo`, `num_super`, `num_sub`, `depth`)
  now raises a clean `ValueError` instead of leaking PyO3's
  `OverflowError: can't convert negative int to unsigned` (#13).

## [0.7.0] - 2026-06-06

### Added

- Embedding-based models: `BERTopic` and `Top2Vec` (embedding-clustering pipeline,
  class-based TF-IDF, `merge_topics` / `reduce_outliers`), `ETM` (per-document
  variational EM and an amortized VAE inference path via `inference="vae"`), and
  `FASTopic` (optimal transport, a hand-coded reverse-mode Sinkhorn).
- Model-neutral analysis surface (`topica.report`, `topica.effects`), including
  `plot_report` — a one-figure model overview — and `topic_info` /
  `topics_over_time` / `topics_per_class`.
- LLM topic labeling and embeddings as plumbing: `llm_topic_labels`,
  `topic_label_prompts`, `llm_backend`, and `llm_embed` (with caching via
  `save_embeddings` / `load_embeddings`). The core takes any callable; an optional
  `topica[llm]` extra adds the `llm` library and the ollama plugin.
- Polars support: `from_dataframe`, `align`, and `design_matrix` accept Polars
  frames alongside pandas.
- A Citing page collecting per-model references, a `LICENSE` file, `CITATION.cff`,
  `CONTRIBUTING.md`, and this changelog.

### Validated

- R-parity checks for the keyATM covariate and dynamic models and for `CTM`
  (as `stm` with no covariates), alongside the existing base keyATM and STM checks.
