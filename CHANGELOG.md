# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

## [0.53.0] - 2026-07-26

### Fixed

- **`STS` default now faithfully reproduces the reference model** (#560). topica's
  Structural Topic and Sentiment-Discourse model diverged from Chen & Mankad (2024)
  on its default fit, aligning to the authors' poliblog reference at a topic-word
  cosine of 0.85 where the reference expects the mid-0.90s. The reference-init path
  was not a faithful port: it approximated the prevalence latents from a phi
  log-ratio and built the prior covariance from the init-eta variance. R's `STS.R`
  seeds from `stm(max.em.its=0)`, which returns `eta = 0` and `invsigma = diag(1/20)`,
  so the prevalence latents start at 0 and the prior covariance is `diag(20)` across
  all latent dimensions. The default fit now uses that initialization and the
  reference's `lasso` kappa estimator (was topica-native `ridge`; ridge remains
  available via `kappa_estimation="ridge"`). The default `STS(...).fit(...)` now
  reproduces the reference at cosine 0.94, and on this corpus the lasso default is
  also faster than ridge.

### Fixed

- **UMAP reducer no longer collapses on real embeddings** (#555). The in-house
  UMAP became the default `reducer` for `BERTopic`/`Top2Vec` in 0.50.0, but on real
  sentence embeddings it collapsed to ~2 topics on the large majority of seeds for
  small-to-moderate corpora. The cause was `find_ab_params` under-converging: it fit
  the low-dimensional membership curve with plain gradient descent that stalled far
  from the least-squares optimum at the `min_dist=0.0` default (`a≈1.58` where the
  reference is `a≈1.93`), giving too-weak short-range attraction so the layout never
  opened the density valleys HDBSCAN needs. It now uses a damped Gauss-Newton solve
  matching SciPy `curve_fit` to four decimals across `min_dist`. Validated against
  the reference `umap-learn`/`bertopic` on 20 Newsgroups, congressional bills, and a
  synthetic corpus (collapse rate 88% to 0%; topic counts and ARI on par with the
  reference).

### Changed

- **`BERTopic.fit` is dramatically faster on medium and large corpora** (#557). The
  `approximate_distribution` step that builds the soft document-topic distribution
  (`doc_topic`) was `O(docs · windows · topics · vocabulary)` and single-threaded,
  dominating fit time — a 5000-document fit spent over an hour in it. It now keeps
  only each window's nonzero tokens (reducing the per-window cosine from the full
  vocabulary to the window size), precomputes each topic's norm once, and
  parallelizes across documents. The output `doc_topic` is bit-identical to before;
  the same 5000-document fit drops from ~72 minutes to a few seconds.

## [0.51.0] - 2026-07-25

### Added

- **Vocabulary control on `Corpus` for scikit-learn / gensim parity** (#553).
  `Corpus.from_documents` (and `from_dataframe`) gain `max_features`, capping the
  vocabulary to the N most frequent surviving terms (scikit-learn's
  `CountVectorizer(max_features=)`), and `vocabulary=`, pinning the vocabulary to a
  fixed, ordered term list (scikit-learn's `vocabulary=`). A new
  `Corpus.transform(documents)` vectorizes held-out documents against an existing
  corpus's vocabulary (scikit-learn's `vectorizer.transform` / gensim's `doc2bow`
  on new text), sharing the vocabulary at full width so a fitted model's
  `topic_word` columns stay aligned. Out-of-vocabulary tokens are dropped; a
  document left with no in-vocabulary token is dropped and recorded in
  `kept_indices`.

## [0.50.0] - 2026-07-25

### Changed

- **BERTopic and Top2Vec now default to `reducer="umap"`** (#549, #550), matching
  the upstream `bertopic` package and the original Top2Vec, both of which reduce
  with UMAP (`n_neighbors=15`, `n_components=5`, cosine). topica's in-house UMAP is
  seed-reproducible, so the default stays deterministic for a fixed `seed`. Pass
  `reducer="pca"` for the former linear default. This is a behaviour change: a
  default `BERTopic(...)` / `Top2Vec(...)` fit now uses UMAP.
- **CombinedTM: faithful `adapt_bert` encoder and raw bag-of-words input** (#503),
  matching the reference; the encoder concatenation and un-normalized BoW now match
  Bianchi et al. This changes fitted output for CombinedTM/ZeroShotTM.
- **FASTopic: the Sinkhorn solve matches the reference's stop criterion** (#500).
  The early stop now compares the L1 marginal error against `sinkhorn_tol` (default
  `5e-3`, the reference `stop_thr`) with `sinkhorn_iters=5000` (the reference
  `OT_max_iter`), so the topic-word plan converges instead of hitting the old
  50-iteration cap. Slightly changes fitted output and improves cross-implementation
  parity (aligned topic-word cosine 0.607 -> 0.616).
- **Spectral initialization: configurable projection threshold** (#542), default
  10000 to match R `stm`.
- **STM content-model refinements** (#531, #532, #534): `content_prior="l1"` is now
  a pure Laplace prior (not L1+L2), content_time periods are ordered numerically
  rather than lexically, and content-prior scales are guarded.
- **`estimate_effect` robust standard errors now use HC0, not HC1** (#533).
- **BTM: the background distribution `pw_b` is biterm-weighted**, not raw
  token-weighted (#492). **PT: restored the `(m_p + lambda)` popularity prior** and
  lgamma log-likelihood (#491).
- **`topica-core` bumped to 0.2.0** (the vendorable logistic-normal STM/CTM/SAGE +
  CVB0 core), which gained the shared residual-dispersion diagnostic (#543). topica
  re-exports it, so the Python API is unchanged.

- **GDMR prior is now faithful to tomotopy's `GDMRModel`** (#426), a breaking
  change to what `sigma`, `sigma0`, and `decay` mean. Previously `sigma` was the
  std on the intercept and `sigma0` on the higher-order terms — the *opposite* of
  tomotopy (and the paper's own `bab2min/g-dmr` code), where `sigma0` is the
  constant-term std and `sigma` the non-constant std. The decay also used an
  invented `decay^(total order)` form (a no-op at `decay=1`, and *growing*
  higher-order variance for `decay>1`); it now uses tomotopy's per-dimension
  `variance = sigma**2 / prod_d (p_d+1)**(2*decay)`, so any `decay>0` shrinks and
  `decay=0` recovers the paper's decay-free prior. Existing `sigma`/`sigma0`/`decay`
  configurations now produce different (correct) fits; a `decay>0`, `sigma!=sigma0`
  tomotopy parity leg was added to the gold. GDMR also gained an `alpha` parameter
  (default 0.1, tomotopy's default) that centers the constant-term prior at
  `log(alpha)` and sets the baseline Dirichlet concentration (smaller `alpha` ->
  sparser per-document topic mixtures), matching both reference implementations;
  `alpha=1` reproduces the previous zero-mean intercept. This is realized through a
  new optional `offset` argument on `DMR.fit` (a fixed `(num_docs, num_topics)`
  term added inside the exponent). Also: `tdf()` now uses a stable softmax, and the
  `alpha` property / `sigma`/`sigma0`/`decay` docstrings were corrected.

- **SAGE now uses its defining sparse prior by default** (#422). The κ content
  deviations were previously fit under a fixed Gaussian ridge, which is not
  canonical SAGE (Eisenstein, Ahmed & Xing 2011) — the model is *defined* by a
  sparsity-inducing prior. `SAGE(prior=...)` now selects `"laplace"` (default,
  canonical sparse SAGE, fit by adaptive reweighting), `"gaussian"` (the previous
  dense ridge / STM-style content model), or `"jeffreys"` (more aggressive sparse).
  New `content_kappa` getter exposes the fitted deviations. **This is a behaviour
  change:** a default SAGE fit now produces sparse content deviations `κ`, which
  change `β` and therefore the fitted topics themselves (strong group structure is
  still recovered, but the exact topic-word distributions and held-out
  `transform`/`doc_topic` differ). Pass `prior="gaussian"` to reproduce the pre-0.5
  behaviour exactly. A SAGE model saved before this change cannot be loaded (the
  save layout gained fields; topica's on-disk format is not migrated across schema
  changes) — re-fit it.

### Fixed

- **Model-review faithfulness sweep (#488–#509).** A faithfulness and correctness
  review of the remaining models, each traced against its reference, produced a
  batch of fixes: HLDA's depth>=4 subtree-deletion index panic (#496), TopicGPT's
  zero-doc-topic invariant break and brace-unsafe custom prompts (#509), DETM
  gradient coverage and corpus-scale ELBO reporting (#495), the ideal-point family's
  exact grid-path W gradient plus documented sign/rotation non-identifiability of
  the position outputs (#498, #499, #505, #506, #508), Wordfish and TBIP honest
  parity claims, PartyEmbeddings core panic guards and `+inf` rejection (#508),
  non-lossy EmbeddingLDA save/load (#502), InfoCTM save/load persistence (#504), and
  the remaining #481-class hyperparameter guards routed through finite-positive /
  finite-non-negative checks across the batch (#510, #517). No core sampler or
  gradient was found unfaithful; the value was in the periphery.
- **STM content-model ν recompute (#442):** for a content (covariate-word) STM,
  `_recompute_eta_cov` rebuilt every document's variational covariance ν against
  the group-averaged β instead of the document's own group β, so the recomputed ν
  (used for method-of-composition uncertainty when `keep_eta_cov=False`) did not
  match the fit's. It now rebuilds ν against each document's group β. A fresh fit
  (`keep_eta_cov=False`) carries a new `content_beta_estep` snapshot — the group β
  active during the final E-step, the content analogue of `beta_estep`, retained on
  both the core `CtmModel` and the Python `STM` object — and reproduces the stored
  ν exactly, in Laplace or diagonal mode. A model reloaded from disk does not carry
  the snapshot (like `beta_estep`/`sigma_estep`) and falls back to the persisted
  per-group `content_beta`: off by the final content M-step, still far closer than
  the group average. `keep_eta_cov=True` stores the per-group E-step ν directly and
  was never affected. Fit output (topics, θ, stored ν) is unchanged for every
  model, content or not.
  - **Save-format break (all STM models):** the per-document group index is now a
    persisted `StmState` field, appended to the positional bincode payload. Because
    that payload is not self-describing, `#[serde(default)]` does not migrate older
    saves — **any** STM model (content, prevalence-only, or plain) written by an
    earlier version now fails to load, surfacing as an invalid-file/EOF error rather
    than a clean version mismatch (the save-format-versioning fix is #443). Refit to
    load under this version.

- **SAGE correctness (#422):** the opt-in `convergence_tol` early stop tripped at
  iter 20 (before any content-deviation `κ` was learned), because with `κ=0` the
  monitored word log-likelihood is a corpus constant; it is now gated on a completed
  `κ` update. Added input validation (`alpha > 0`, `lbfgs_iters >= 1`,
  `num_samples >= 1`, finite non-negative `convergence_tol`, no duplicate
  `group_names`), a guard that skips (and warns on) a non-finite `κ` optimization
  step instead of corrupting the topics, and corrected the early-stop docstring
  (the monitored quantity is the word-emission log-likelihood, not a full collapsed
  model-fit likelihood). Default fitted output is unchanged. The faithfulness fix
  (SAGE's sparse prior vs. the current Gaussian ridge) is tracked separately.

### Added

- **Top2Vec reference surface** (#489, PRs #520 and #548): size-ordered topics
  (topic 0 largest) with a `topic_sizes` getter, `hierarchical_topic_reduction(n)`
  (merge the smallest topic into its nearest by topic-vector cosine, the reference's
  reduce-to-N driver), and the search API — `search_documents_by_topic`,
  `search_documents_by_keywords`, `search_words_by_vector`, `similar_words`,
  `search_topics`.
- **`GSDMM.transform()`** to soft-assign held-out short texts (#490); **per-document
  proportions** for PA (`doc_super`, #497) and DTM (`doc_topic`, #494); **InfoCTM
  save/load** (#504).
- **Shared residual-dispersion diagnostic** in `topica-core`, exposed through
  `inspect` (#543).
- **`RTM`, the relational topic model** ([Chang & Blei 2010](https://www.jstor.org/stable/27801582))
  (#414): joint model of document text and a link graph (citations, hyperlinks,
  co-sponsorship, adjacency). `fit(docs, links=edges)` on undirected document
  pairs; `predict_link`, `suggest_links` (link prediction from words for unseen
  documents), `eta`/`nu`/`phi_bar`. Variational EM with `logistic` (default) or
  `exponential` link functions and the paper's ρ regularization. Validated against
  a standalone NumPy implementation of the paper's variational equations
  (`parity/rtm_reference.py`); the R `lda` `rtm.em` is collapsed Gibbs, so it is a
  directional baseline only.

- **`model.seed` and `Corpus.preprocessing` getters** (#399): every model now
  exposes the `seed` it was constructed with, and a `Corpus` exposes the
  vocabulary-filtering parameters Topica applied — `min_doc_freq`,
  `max_doc_fraction`, `min_cf`, `rm_top` — as a dict (`None` for a corpus loaded
  from disk, where they are not persisted). The analysis manifest now records both
  the seed and the preprocessing, which it previously could not read back.

- **Content-addressed analysis bundle**: `AnalysisManifest.bundle(path, model=,
  corpus=)` packages the manifest and the saved artifacts into one `.zip` — each
  artifact file is named by its BLAKE2b digest. `bundle` refuses to package a
  model/corpus whose fingerprints do not match the manifest (the wrong fit), and
  `AnalysisManifest.load_bundle` checks every artifact against its content-addressed
  name and the manifest reference, raising on a corrupt bundle or artifacts that no
  longer match. This is an integrity/content-addressing guarantee, not authenticity
  (detecting a fully rewritten bundle needs a signature). `extract_bundle(path,
  dest)` recovers the artifacts for reloading. Bundling the corpus is opt-in and
  sensitive (it embeds raw tokens).
- **Built-in diagnostic capture** for the manifest:
  `record_fit(..., diagnostics=["coherence", "exclusivity"])` computes those
  topic-quality metrics and records each as computed evidence (mean over topics),
  or `value=None` with a note when the metric is not defined for the model.
- **Manifest comparison**: `AnalysisManifest.compare(other)` diffs two manifests
  directly (no corpus or model needed) and returns a `ManifestDiff` reporting per
  field `same` / `changed` / `only_in_a` / `only_in_b` / `incomparable` — so you
  can see whether two runs used the same corpus, model, and inputs, or what
  changed between them.
- **Analysis card rendering** for the manifest: `AnalysisManifest.render(path)`
  writes a self-contained HTML card and `.to_markdown()` returns a Markdown
  version (Quarto/notebook). The card renders only what the manifest recorded, so
  it cannot over-claim: decisions are shown as researcher-authored, diagnostics as
  computed evidence, fingerprints as verifiable identity, and a supplied
  `VerifyResult` renders as a colour-coded per-field table (never a single badge).
- **Analysis manifest** (`topica.record_fit`, `topica.AnalysisManifest`): a
  portable, privacy-aware JSON record of one fit. It captures the model settings,
  environment, fit arguments, researcher decisions, and canonical fingerprints of
  the corpus, design matrix, and model outputs, then `verify(corpus, model)`
  reports per field whether the fit's identity and replay conditions still hold
  (`exact` / `input_changed` / `artifact_changed` / `environment_changed` /
  `unverifiable`), never a single pass/fail. Privacy-by-default: `privacy="minimal"`
  records only coarse corpus counts, with an aggregate description and a sensitive
  content fingerprint as explicit opt-ins. The manifest composes with
  `Corpus.save` / `model.save` rather than replacing them.

### Removed

- **ECTM** (the Evolving Content Topic Model) is removed. Its one interpretive
  payoff — how a group words a topic over ordered time — is delivered more
  accurately, and with defensible inference, by the STM `content_time` path
  (`STM.fit(..., content=, content_time=, content_prior="l1", content_smooth=)`)
  plus the `topica.content.content_trajectory` / `content_divergence` readers with
  a design-preserving document/cluster bootstrap. The `topica.ECTM` class, the
  `topica.ectm` helper module, and the ECTM worked examples are gone; the party-
  platforms example is re-pointed to `examples/stm_content_time_platforms.py`.
  Save tag 27 is retired and will not be reused.

## [0.34.0] - 2026-07-13

### Changed

- The release pipeline now verifies that the tag and both package manifests agree,
  builds and smoke-tests native wheels, builds the source distribution from a
  clean checkout, and gates both PyPI and CLI publishing on lint, tests, and the
  strict documentation build.

### Documentation

- Added the contributor release checklist, including a manual real-corpus
  dogfood check for changes to estimators, samplers, optimizers, or covariate
  paths.
- Documented the default policy for Gibbs sampling: single-threaded fits are the
  exact, reference-comparable path; parallel Gibbs is an explicit approximate
  opt-in, and its likelihood-plateau stopping heuristic is disabled by default.
- Expanded acknowledgements with the project's inspiration and origin story.

## [0.33.0] - 2026-06-28

### Changed

- The experimental ideal-point family was consolidated under the `IdealPoint` stem,
  with the topic-model badge (`TM`) kept on both members:
  - `IdealPointLDA` is **removed** as a separate class and folded into `IdealPointTM`
    as its count representation. `IdealPointTM.fit` now takes `word_embeddings` and
    `vocabulary` as optional keyword arguments: omit them for the count representation
    (the former `IdealPointLDA`, "Wordfish with topics"); pass them for the
    word-embedding representation (the ETM form). The two are the same model — the
    embedding is a low-rank factorization of the same displaced topic-word matrix — so
    one class with a fit-time knob replaces two. A new `representation` property
    reports `"counts"` or `"word2vec"`. Migration: `IdealPointLDA(...)` →
    `IdealPointTM(...)`; `IdealPointTM(...).fit(docs, emb, vocab, ...)` →
    `IdealPointTM(...).fit(docs, word_embeddings=emb, vocabulary=vocab, ...)`
    (`word_embeddings`/`vocabulary` are now keyword-only).
  - `SentenceIdealTM` is **renamed** to `IdealPointSentenceTM` for prefix consistency;
    it remains the separate continuous (Gaussian-mixture, EM) model over sentence or
    document embeddings.
  - Save files are unchanged on disk: `IdealPointTM` reads both representations' save
    tags (31, 33) and `IdealPointSentenceTM` reads tag 34, so models saved before the
    rename still load.

### Fixed

- `IdealPointSentenceTM` is now deterministic from a fixed `seed` regardless of thread
  count. Two parallel `f64` reductions (the E-step log-likelihood and the M-step
  variance) summed in rayon work-stealing order, which is not associative, so two
  same-seed fits could differ by ULPs and diverge (the log-likelihood drives the EM
  early-stop). Both now collect in index order and sum sequentially, matching
  topica's determinism guarantee. Surfaced as an intermittent macOS CI failure of
  `test_sentence_ideal.py::test_determinism`.

### Added

- `Wordfish.fit(control=...)` — a categorical confound covariate for Wordfish. When a
  corpus has a dominant non-ideological axis (a chamber, government/opposition status,
  an era, a language) the latent position is otherwise hijacked by it; `control`
  absorbs that level-specific word usage into per-level word offsets
  (`log rate += delta[level, word]`, baseline level held at zero), and the
  initialization is residualized by level so `theta` does not start on the nuisance
  axis. Exposed via `control_names` / `control_word_offsets`. With no `control` the fit
  is exactly the historical Wordfish, bit-for-bit (quanteda parity preserved). On a
  planted contaminated corpus, ideology recovery rises from ~0.03 to ~0.8.
- `topica.polarization(positions, labels)` — a model-agnostic ideal-point diagnostic:
  the distance between named camps' centroids on any model's `author_positions` (1-D,
  or Euclidean for a multi-dimensional fit), so polarization can be traced over time
  (the Rheault & Cochrane 2020 party-distance metric). `normalize=True` gives an
  effect-size form comparable across corpora. Joins `bimodality` /
  `split_half_reliability` / `position_intervals` in `topica.scaling`.
- `PartyEmbeddings` (#303) — **party embeddings** (Rheault & Cochrane 2020), the
  corpus-trained word-embedding member of the `ideal-point` family. A PV-DM
  (distributed-memory paragraph-vector) model trained by negative sampling with
  party-period metadata tags; the ideological placement is the leading principal
  components of the learned party vectors, and because words share the space a
  party's language can be read off by proximity. Exposes the family surface
  (`author_positions`/`author_names`/`anchors=`) plus `nearest_words`,
  `guided_positions` (a custom lexicon axis), and `distance` (polarization).
  topica reimplements the PV-DM negative-sampling training in Rust (no
  word2vec/doc2vec engine existed) from Mikolov et al. (2013) and Le & Mikolov
  (2014); the reference package builds on gensim `Doc2Vec`, and the topica scale
  matches it at correlation 1.00 on a planted ordering
  (`parity/party_embeddings_compare.py`). Save tag 36.
- `KeyATM.feature_effect_se` (#316) — standard errors for the covariate keyATM's
  feature effects `λ`, mirroring `DMR.feature_effect_se` and completing the
  standard-errors roadmap (#309). Computed from the observed information of the
  penalized Dirichlet-multinomial in the standardized fit space (where `λ` is fit,
  z-scored and ±5-bounded per #270), then mapped to the original covariate scale by
  the standardization Jacobian (the intercept mixes the slopes, so it is not a
  per-column rescale). Entries whose standardized `λ` sat at the ±5 bound are
  returned as `NaN` (a constrained estimate has no valid asymptotic SE). Computed at
  fit time alongside `feature_effects`; `dmr::dmr_lambda_cov` now exposes the full
  covariance the Jacobian needs.

## [0.32.0] - 2026-06-27

### Changed

- New `ideal-point` model group ("Ideal point"), and `Wordfish` and `TBIP` graduate
  from the experimental tier into it. Both clear topica's bar for a validated model — a
  published reference (Slapin & Proksch 2008; Vafa, Naidu & Blei 2020) plus a
  reference-implementation parity check (`parity/wordfish_r_compare.py` against R
  `quanteda`, |r| = 1.00; `parity/tbip_parity.py` against a PyTorch reference) — so they
  no longer require `topica.enable_experimental()`. `IdealPointTM`, `IdealPointLDA`, and
  `SentenceIdealTM` remain experimental (original constructions with no external
  reference) and stay in their current groups.

## [0.31.0] - 2026-06-26

### Added

- `TBIP` (#299) — **Text-Based Ideal Points** (Vafa, Naidu & Blei 2020), experimental: a
  Poisson factorization in which an author's latent ideal point rescales a neutral
  topic-word intensity by a per-word ideological factor, fit by the paper's mean-field
  variational inference (reparameterized SVI, Adam). The official implementation is
  TensorFlow 1.14, so topica reimplements the published model and inference in the Rust core
  (hand-coded reverse-mode gradients, FD-checked); the VI minibatch is rayon-parallel with a
  fixed-chunk deterministic reduction. Validated by synthetic planted-position recovery and
  against a PyTorch reference. Joins the ideal-point family; gated behind
  `topica.enable_experimental()`. Save tag 35.
- Standard errors for ideal-point positions (#298): `topica.position_intervals(fit, group)`
  returns model-agnostic bootstrap standard errors and confidence intervals for any of the
  ideal-point models; `Wordfish.position_se` adds an analytic asymptotic standard error,
  validated equal to R `quanteda`'s `se.theta` (|r| = 1.00).
- Intrinsic ideal-point diagnostics (#297): `topica.bimodality` (a bimodality coefficient
  for the positions) and `topica.split_half_reliability` (refit on disjoint document halves
  and correlate), so an unsupervised scale can be certified without an external reference.

## [0.30.0] - 2026-06-25

### Added

- Three experimental **ideal-point comparison companions** to `IdealPointTM`, spanning
  the representation x structure grid (so the contribution of topics vs embeddings can be
  measured head to head):
  - `Wordfish` (#293) — Slapin & Proksch (2008) Poisson scaling: a one-dimensional
    unsupervised ideal point from word frequencies alone, no topics. Deterministic;
    validated bit-for-scale against R `quanteda.textmodels::textmodel_wordfish` (r = 1.00).
  - `IdealPointLDA` (#294) — the count-based twin of `IdealPointTM`: the same
    position-displaced topic-word model parameterized directly over the vocabulary (no
    embeddings). "Wordfish with topics."
  - `SentenceIdealTM` (#295) — a continuous ideal-point model over sentence/document
    embeddings: topics are Gaussian clusters whose centroids are displaced by a latent
    author position. Closed-form EM.
  - `examples/ideal_point_comparison.py` fits all four on one corpus and reports recovery
    and pairwise agreement.
  All gated behind `topica.enable_experimental()`.

## [0.29.0] - 2026-06-25

### Added

- `IdealPointTM` (experimental) — an embedded topic model with a latent ideal-point
  head. One unsupervised fit yields topics *and* a one-dimensional scaling of authors
  along a discovered latent axis: `author_positions` (the ideal points),
  `topic_discrimination` (which topics carry the cleavage), and `position_shift` (how
  word choice within a topic differs across the axis). Takes word embeddings like
  `ETM`, plus an optional `group=` (one position per author) and `anchors=` (orient the
  sign). Gated behind `topica.enable_experimental()`. Validated against DW-NOMINATE on
  U.S. congressional press releases (Pearson ~0.79-0.88 across the 115th-118th House);
  recovery depends strongly on genre, working best on expressive, author-grouped text.
  See `docs/guides/models.md#idealpointtm` and `examples/idealpoint.py`.

## [0.28.0] - 2026-06-22

### Added

- `topica.stop_reason(model)` — a model-neutral report of why a fit's training
  loop stopped: early convergence on `convergence_tol` (a floor) versus the
  iteration cap (a ceiling), with the last relative change. Answers the question
  `converged` alone leaves implicit (#267).
- Statistical-validity test coverage for every model (#271): universal invariant
  and metamorphic checks (no degenerate/collapsed θ, an effective-topics floor,
  finite parameters, "more iterations must not collapse the fit") with a coverage
  manifest that fails if a new model ships without them. This is the layer that
  catches the class of bug #270 was.
- Committed gold-fixture parity for 30 models, validating topica **offline** in CI
  (no reference toolchain installed): cross-implementation against the reference
  where one exists — R `stm` (STM/CTM/STS), R `keyATM`, Java MALLET (LDA/DMR/
  LabeledLDA), tomotopy (GDMR), gensim (DTM), scikit-learn (NMF/LSA), the AVITM
  PyTorch reference (ProdLDA/CombinedTM/ZeroShotTM/InfoCTM), BERTopic, and
  FASTopic — and against a frozen planted self-consistency baseline where no
  reference implementation exists (SAGE, HDP, PA, HLDA, ETM, DETM, ECTM,
  SupervisedLDA, SeededLDA, GSDMM, PT, EmbeddingLDA). Each fixture ships a JSON
  provenance log and a non-vacuous control. A shared `parity/harness.py` and a
  `scripts/ci_sim.py` offline-gold guard support it.

### Fixed

- Covariate keyATM (`fit(covariates=...)`) no longer silently collapses the
  document-topic matrix onto a single topic on a high-dimensional design at scale.
  Following R `keyATM`, the covariates are now standardized and the regression
  coefficients bounded under the N(0,1) prior, which keeps `α = exp(x·λ)` from
  running away; validated against R `keyATM`'s covariate model (#270).

### Documentation

- Every public binding parameter is now documented, with a doc-coverage lint
  (constructors included) so the runtime `help()` text and the API site can no
  longer drift behind the signatures (#268). `convergence_tol`'s units and scale
  are documented across the model `fit` methods (#267).
- Speed is reported **to convergence** — the time a user actually waits — across
  the README, the paper, and `docs/benchmarks.md`, replacing the earlier
  per-iteration framing (the per-iteration decomposition is noted, not headlined).

## [0.27.0] - 2026-06-21

### Added

- Survey weights and average marginal effects for covariate inference:
  `estimate_effect(weights=)` runs weighted least squares (composing with
  `cluster=` and `link=`), and `average_marginal_effects()` / `ame()` reports the
  average change in a topic's proportion per unit of a covariate (the average
  derivative for a continuous covariate, the average level-vs-reference contrast
  for a factor) — cleaner than raw coefficients under splines or interactions.
  `TopicEffect` now also exposes the full Rubin-pooled coefficient covariance
  `.vcov`. Validated bit-for-bit against faSTM's effect layer (#258).
- `semantic_coherence()` — stm's `semCoh1beta` semantic coherence, from the shared
  Rust core. The broader gensim-aligned `coherence()` suite is unchanged (#260).
- `frex()` and `label_topics()` accept `word_counts=` or `corpus=` to apply stm's
  James-Stein FREX shrinkage and exact lift; `Corpus.word_counts` exposes the
  per-term corpus frequencies (#260).
- `topica-core` gains `inspect` (FREX / lift / score / exclusivity / semantic
  coherence) and `effects` (`estimate_effect_topic`) modules plus
  `Corpus::from_texts` — the engine shared by faSTM (R) and stmata (Stata), so the
  stm-faithful diagnostics have one definition across languages (#259).

### Changed

- FREX, lift, score, exclusivity, and semantic coherence now all come from the
  single stm-faithful implementation in `topica-core`'s `inspect`, so they cannot
  drift between topica, faSTM, and stmata (#260). Two definitions changed as a
  result:
  - **`label_topics` lift** is now stm's lift, `log P(w|topic) − log P(w)`, not
    the previous `beta / mean_k beta` ratio (which was not lift). With
    `word_counts=`/`corpus=` it is exact; without, `P(w)` is estimated from the
    column marginal (same ranking, correct log scale).
  - **`exclusivity`** is now stm's FREX-summary over the top words (roughly
    `[0, n]`), not a mean exclusivity in `[0, 1]`. The model-selection frontier
    uses z-scores, so K selection is unaffected by the rescale.
- Spectral initialization now weights the Arora recovery by the pooled unigram
  frequency (stm's `wprob`), matching R `stm`'s spectral start exactly (per-topic
  cosine 1.0 on gadarian); previously it used the gram row sums. The converged
  STM fit is unchanged (0.996 cosine to the prior solution) (#265).

### Fixed

- Restore the `--features python` build: the `LoadOptions.lowercase` field added
  to `topica-core` had left the Python binding's struct literal incomplete; this
  also cleared two pre-existing `cargo fmt`/`clippy` lints (#262).

## [0.26.0] - 2026-06-19

### Fixed

- STM/CTM `gamma_prior="pooled"` is now empirical Bayes, a faithful port of R
  `stm`'s `vb.variational.reg` (`gamma.prior="Pooled"`): the prevalence
  regression estimates its coefficient and noise precisions from the data
  (adaptive shrinkage, intercept unpenalised) instead of applying a fixed 1e-6
  ridge. On a wide prevalence design (a `s(day)` spline, or many one-hot
  covariate levels) the old near-OLS coefficients refit freely each M-step, so
  EM took noticeably more iterations to converge than `stm`. The fit is
  unchanged (topic-word cosine vs R `stm` stays 0.958 on poliblog5k K=20), the
  bound is still monotone, and convergence is faster on covariate-rich designs.
  Golden-tested bit-close to stm 1.3.8 (#247).

### Added

- `parity/stm_spline_iters_247.py`: a skip-clean cross-engine EM-iteration anchor
  (topica vs R `stm`, with a QR-orthonormal control) documenting #247, plus a
  `docs/benchmarks.md` note to time covariate-rich STM to convergence rather than
  at a fixed iteration count.

## [0.25.0] - 2026-06-18

### Changed

- Internal refactor (no public API change): the logistic-normal structural-topic
  core — `corpus`, `spectral`, the shared `variational` kernels, `estimator`,
  `ctm` (CTM/STM/SAGE), `cvb0`, and `linalg` — now lives in a new Cargo workspace
  member, `topica-core`, which `topica` depends on and re-exports. `topica::ctm::*`
  and the entire Python/Rust API are unchanged. The split lets downstream Rust
  consumers (e.g. the faSTM R package, heading for CRAN) vendor a small,
  dependency-light crate (`rand`, `rand_chacha`, `rayon`, `regex`; serde opt-in)
  instead of all of `topica`. A golden parity test pins `fit_ctm` output so the two
  crates stay in lock-step (#242).

### Added

- Paper appendix: a side-by-side validation appendix that fits both the reference
  package and `topica` on the poliblog corpus for 15 models and shows the results
  together, with a master validation map covering every model (#241).

## [0.24.1] - 2026-06-18

### Added

- **`content_placebo` for ECTM** — `topica.ectm.content_placebo` is the
  content-side counterpart of `topica.permutation_test`: it shuffles the content
  group labels within each period, refits, and recomputes each topic's mean
  `content_divergence`, returning the observed divergence, the permutation null,
  the finite-sample floor, and a one-sided p-value per topic. This establishes that
  a between-group content divergence is real rather than a small-cell estimation
  artifact. (#230, #232)

- **SAGE content-model κ coefficients on `STM`** — `STM.content_kappa` returns the
  additive decomposition behind the per-group topic-word model as a dict: `m`
  (num_words), `kappa_topic` (num_topics × num_words), `kappa_cov` (num_groups ×
  num_words), and `kappa_interaction` (num_topics × num_groups × num_words), where
  the per-group log-probabilities are `m + kappa_topic + kappa_cov +
  kappa_interaction` (softmax over words). These were computed internally but
  previously discarded; they are the identifying parts R `stm`'s `sageLabels()` /
  `labelTopics()` rank words by (and cannot be recovered from the per-group β
  alone). Persisted across save/load. (#237)

## [0.24.0] - 2026-06-18

### Added

- **Minibatch / stochastic VI for ECTM** — `ECTM.fit(inference="svi",
  batch_size=, tau=, kappa=, content_every=)` fits the 10^5–10^6-document corpora
  ECTM is built for without subsampling. Each step samples `batch_size` documents,
  runs the warm-started Laplace E-step, scales the minibatch sufficient statistics
  to corpus size, and moves the globals with a Robbins-Monro rate
  `(tau + step)^(-kappa)`; the expensive content-κ M-step is amortized across
  `content_every` minibatches (default once per epoch). Validated against the batch
  fit on a 96k-speech congressional corpus (matched topic-word cosine 0.97,
  between-group divergence-spectrum correlation 0.97, in a few minutes vs ~22 for
  batch). SVI is **seed-reproducible** (not bit-exact); batch stays the default.
  A content-convergence guard (`content_converged`, `content_shift_history`) warns
  when a fit's content model has not settled, so an understated headline cannot
  pass silently. SVI is ECTM-only: a size-ladder benchmark showed plain STM/CTM
  batch EM converges in ~15–25 cheap iterations at every scale, so SVI offers no
  benefit there. (#231, #233)

- **`beta_init` warm-start hook** — `STM.fit(..., beta_init=)` and
  `CTM.fit(..., beta_init=)` accept a caller-supplied `(num_topics, num_words)`
  base topic-word matrix, overriding the spectral/random init. This lets an
  external front end (e.g. an R `stm`-compatible wrapper) inject a precomputed β
  and reproduce that fit deterministically. Batch only. (#234)

### Fixed

- **Spectral initialization now reproduces R `stm`'s `recoverL2`** — the
  anchor-word recovery (`recover()`) solved the per-word simplex problem with a
  fixed, too-large exponentiated-gradient step that diverged to arbitrary vertices
  instead of the constrained optimum (matching R `stm` at only ~0.37 cosine even
  on identical inputs). The step is now scale-adaptive (`1/(2L)`, L a Gershgorin
  bound on λmax) and runs to convergence, with anchor words set to exact unit
  vectors; it reaches R `stm`'s reference recovery at cosine **1.0** on gadarian
  (`parity/spectral_recover_stm.py`) in ~70 iterations. This corrects the spectral
  init under the whole logistic-normal family (CTM/STM/STS/ECTM/DTM). (#234)

## [0.23.1] - 2026-06-17

### Added

- **Bundled multilingual stopwords** — `topica.stopwords(lang)` returns a
  `frozenset` for any of **58 languages** (the [stopwords-iso](https://github.com/stopwords-iso/stopwords-iso)
  lists, MIT licensed, bundled in the wheel), accepting an ISO 639-1 code
  (`"fr"`) or an English name (`"french"`, case-insensitive) and raising with the
  available codes on an unknown language; `topica.stopword_languages()` lists
  them. This gives the cross-lingual models (`InfoCTM`, `ZeroShotTM`) and any
  non-English corpus a ready stoplist. `ENGLISH_STOPWORDS` is unchanged (the
  short, stable default); `stopwords("en")` is the larger stopwords-iso English
  list. (#225)

## [0.23.0] - 2026-06-17

### Added

- **Deterministic spectral init for ECTM** — ECTM's content model now starts
  from the same anchor-word spectral base as STM/CTM/STS (`init="spectral"`, the
  default), with the group×period content deviations starting at zero (as R `stm`
  does for κ). A random base β left the model multimodal — on the German
  Manifesto corpus (K=25) 4 of 10 seeds collapsed ~320,000 nats below the
  structured basin. The spectral base removes that scatter: the fit is now
  seed- and thread-independent (ECTM is `bit-exact`). `init="random"` keeps the
  old seeded behavior for multi-start. (#220)
- **`DTM(init="spectral")`** — an optional deterministic anchor-word seed for the
  dynamic topic model, for a single reproducible fit in the good basin. The
  default stays `init="random"` (a seeded static-LDA seed, matching gensim's
  `LdaSeqModel`); the rule across models is that the default init tracks each
  model's reference implementation. (#221)
- **`select_model` for the stochastic models** — `stm`'s `selectModel` (fit N
  seeds at fixed K, return the coherence/exclusivity frontier) now drives
  `model="prodlda" | "etm" | "fastopic" | "combinedtm" | "zeroshottm"` in
  addition to `"lda"`/`"stm"`, each with its required data argument
  (`word_embeddings`+`vocabulary` for ETM; `doc_embeddings` for the
  embedding/VAE models). The burn-in selector falls back to mean coherence for
  models with no scalar bound (FASTopic). These are the models whose fit depends
  on the seed; STM/CTM are deterministic, so multi-start there buys nothing. (#222)

### Fixed

- **STM content covariate used a random base β under `init="spectral"`** — the
  init guard `init_spectral && content.is_none()` routed content models to a
  random basis, so the SAGE content model started from a random basis and most
  seeds collapsed to flat/no-group content. STM content now uses the
  deterministic spectral base (verified against R `stm`'s `kappa.init`/`stm.init`:
  R zeros all content κ and uses the full deterministic Gram). (#216, #217)

### Changed

- **Determinism tags corrected for six models** — `DTM`, `ETM`, `FASTopic`,
  `CombinedTM`, `ZeroShotTM`, and `LSA` were tagged `bit-exact` but reach the RNG
  at initialization (random init, or randomized SVD for LSA), so their output
  depends on the seed. Re-tagged to `seed-reproducible`, which also matches their
  reference implementations (all random-init/stochastic). A registry-driven test
  now enforces every determinism tag so the claim and the behavior can't drift
  apart. (#219)
- **Paper roster + reproduction** — `paper/topica.tex` updated to the current
  model roster and 0.22.0 numbers, with a one-command reproduction harness
  (`paper/reproduce.py`). (#213, #215)
- **Parity scaffolding** — the ProdLDA-family AVITM torch reference is
  consolidated into `parity/refs/avitm.py`, with new CombinedTM/ZeroShotTM parity
  harnesses. (#218)
- **CI** — JavaScript actions bumped to Node 24 majors. (#214)

## [0.22.0] - 2026-06-16

### Added

- **`topica.datasets`** — bundled example datasets for quickstarts and worked
  examples: `load_gadarian` (vendored in the wheel, loads offline), `load_poliblog`
  and `load_dubois` (downloaded once from a pinned commit, SHA-256 verified, cached
  under `~/.cache/topica` or `TOPICA_DATA_HOME`). Each returns a pandas DataFrame
  ready for `from_dataframe`, or the cached CSV path with `return_path=True`. (#204)
- **InfoCTM** — topica's first cross-lingual topic model (Wu et al. 2023): two
  ProdLDA models aligned by a bilingual dictionary through a topic-alignment
  mutual-information term, so topics correspond across two languages. Brings
  `text, dictionary`; validated against the reference. (#200)
- **LLM-based topic evaluation** under the `topica.llm` namespace —
  `topica.llm.coherence` and `topica.llm.intrusion` (Stammbach et al. 2023: have an
  LLM rate a topic's word set or spot the intruder), `topica.llm.select_k` for
  LLM-guided K selection, the Tan & D'Souza metric suite, and a pluggable
  `topica.llm.backend` (OpenAI or local via ollama; the docs default to and
  showcase open models). (#201, #202)
- **ECTM content standard errors** — `topica.ectm.content_contrast_se` gives
  instant analytic per-word SEs for a group contrast (multinomial sampling
  variance from each cell's effective token count; conservative, ignores period
  pooling), and `content_trajectory_ci` gives a cluster-bootstrap confidence band
  (resample by source document/platform, not paragraph). On the platforms,
  `climate` is a z=5.7 Democratic word in 2016 and absent in 1964.
- **ECTM prevalence helpers** — `topica.ectm.prevalence_by_group` /
  `prevalence_contrast` give the descriptive "how often each group discusses a
  topic" view (mean `doc_topic` by group x period). The partisan examples now also
  fit a `prevalence=party*spline(year)` design and report the attention gap with
  method-of-composition standard errors via `topica.stm.predicted_prevalence`,
  alongside the content trajectories — both halves of the ECTM picture.
- **ECTM partisan examples** — `examples/ectm_platforms.py` (U.S. party platforms
  1948-2024, Dem vs Rep across 20 elections; corpus ships in-repo, rebuilt by
  `prep_platforms.py`) and `examples/ectm_speeches.py` (congressional speeches
  with Voteview party, 1948-2008; built by `prep_speeches.py`). The platforms demo
  shows `climate` entering the Democratic environment vocabulary while Republicans
  never adopt it; the speeches demo recovers the mid-1990s rise in the partisanship
  of congressional language.
- **`ECTM`** — the Evolving Content Topic Model: an STM whose content (topic-word)
  model carries a group-by-time interaction. The same stable topic is worded
  differently across a document group, and that difference drifts across discrete
  time periods, with a first-order random-walk prior tying adjacent periods so
  sparse cells borrow strength from their temporal neighbours. Reuses STM's
  logistic-normal variational E-step; the content M-step generalizes the SAGE
  content κ-regression to (group × period) cells with random-walk and shrinkage
  penalties (`η_kgtv = m_v + κT_k + κKP_kt + κKG_kg + κKGP_kgt`). Standard fitted
  surface plus `content_word_dist(group, period)`, the per-document logistic-normal
  posterior (`eta_mean`/`eta_cov`), and the `topica.ectm` interpretation helpers
  (`content_words`, `content_contrast`, `content_trajectory`, `content_divergence`).
  See `examples/ectm_poliblog.py`. **Experimental:** ECTM ships before a published
  paper and a reference-parity check, so it is gated — call
  `topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) before use, and
  expect it may change without a deprecation cycle.
- **`topica.enable_experimental()`** — opt into experimental, unvalidated models
  (currently `ECTM`). Such models are kept out of the validated roster and refuse
  to construct or load until enabled (also via the `TOPICA_EXPERIMENTAL`
  environment variable). `topica.experimental_enabled()` reports the current state.

### Changed

- **pandas is now a core dependency** (alongside NumPy). The DataFrame-first
  workflow (`from_dataframe`, `topica.datasets`, `topic_table`) is the default
  on-ramp, so `pip install topica` now ships pandas and the quickstart runs with
  no extras. Still no JVM and no PyTorch.
- **Repo root slimmed** — `CONTRIBUTING.md` and `CONTRIBUTING-MODELS.md` moved
  under `.github/` (GitHub still surfaces the contributing guide from there).
  Dataset CSVs are marked `binary` in a new `.gitattributes` so their bytes (and
  the dataset checksums) are identical across platforms. (#209)

### Documentation

- **CSV-first onboarding** — the README leads with a runnable `from_dataframe`
  example on a bundled dataset and collapses the model roster behind a "start
  here" teaser; the quickstart leads with the CSV path, the minimal `fit(corpus)`
  call, and a choosing-K section. New `topica.datasets` API page. (#205, #207)
- **STM prevalence parity corrected** — `docs/replications/stm.md` now leads with
  the Poliblog results (aligned topic-word cosine 0.97 at 2k/K=20, 0.84 at
  5k/K=15; prevalence effects track R at Pearson 0.84) instead of the mislabeled
  Gadarian K=3 stress case (0.51). No code change; the parity was always there.
  Adds the `parity/stm_poliblog5k_compare.py` harness. (#206)

## [0.21.0] - 2026-06-16

### Added

- **`TopicGPT`** — LLM-driven topic discovery (Pham, Hoyle, Sun & Iyyer 2024): a
  generate / refine / assign prompt pipeline that reads documents, induces a topic
  taxonomy with natural-language descriptions, and assigns each document with a
  supporting quote. Cluster-style (`llm-bounded`), with the standard fitted surface
  (`doc_topic`, a class-TF-IDF `topic_word` descriptor, `top_words`, `coherence`,
  `transform`, `save`/`load`) plus `topic_descriptions`. Bring your own backend
  callable or a model name via `topica.llm_backend`. The prompts are adapted from
  the published TopicGPT reference (chtmp223/topicGPT, MIT) — the bracketed
  `[level] Label: Description` format, few-shot structure, and rules — and are
  fully overridable (`prompts=` accepts a partial dict; `with_prompt(stage, …)`).
- **`topic_semantic_diversity`** (TSD; Wu, Nguyen & Luu 2024, Eq. 18) — a
  model-agnostic diagnostic counting unique top-word *pairs* across topics (a
  stricter, sense-aware companion to `topic_diversity`); no embeddings.
- **`prior="stick_breaking"`** on the amortized-VAE models (ProdLDA, CombinedTM,
  ZeroShotTM, `ETM(inference="vae")`) — the Gaussian stick-breaking construction
  (Miao, Grefenstette & Blunsom 2017; Nalisnick & Smyth 2017). Keeps the laplace
  Gaussian latent and KL, mapping onto the simplex by stick-breaking; a
  nonparametric-flavored prior that softens the fixed-`K` assumption. Completes the
  alternative-prior VAE family (`laplace` / `dirichlet` / `stick_breaking`). The
  laplace default stays bit-identical.

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
