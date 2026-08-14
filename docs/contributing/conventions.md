# API conventions: the shared vocabulary

topica has more than 40 models. They feel like one library only if the same
concept wears the same name everywhere: the iteration count is always `iters`,
the seed is always `seed`, a covariate design matrix is always reachable as
`covariates=`. This page records that vocabulary so a newly added model matches
by construction rather than by memory.

!!! note "The test is the source of truth"
    A hand-maintained style guide drifts. The authority is
    `tests/test_naming_conventions.py`, which introspects every model's
    `__init__`/`fit` signature and fails when a model breaks a rule. This page is
    the *explanation*; the test is the *contract*. When the two disagree, the
    test wins and this page is wrong. It pairs with the structural
    [estimator contract](estimator-contract.md) (which methods/attributes a model
    must expose) and the [implementer's playbook](https://github.com/nealcaren/topica/blob/main/.github/CONTRIBUTING-MODELS.md).

## Naming rules

| Concept | Canonical | Not |
|---|---|---|
| Iteration count | `iters` | `iterations`, `n_iter`, `max_iter`, `epochs` |
| Secondary iteration count | `<thing>_iters` (`var_iters`, `lbfgs_iters`) | — |
| RNG seed | `seed=13` | `random_state`, `random_seed` |
| Counts | `num_*` (`num_topics`, `num_samples`, `num_threads`, `num_theta_draws`) | `n_*`, `n` |
| Tolerance | `convergence_tol` (and `em_tol` for EM/variational models) | `tol` |
| Periodic cadence | `*_interval` (`optimize_interval`, `sample_interval`, `progress_interval`) | — |
| Human labels for a matrix/index arg | `<thing>_names` (`feature_names`, `prevalence_names`, `label_names`) | — |
| Dirichlet / regression priors | `alpha`, `beta`, `prior_variance` | — |

## Structural rules

1. **`num_topics` is the first positional argument** of the constructor;
   everything else in `__init__` is keyword-only (after `*`). `seed=13` is always
   present. The exceptions are principled and recorded in the test: models that
   discover K (`HDP`, `BERTopic`, `Top2Vec`), models whose leading required input
   is something else (`KeyATM` keywords, `SeededLDA` seed words, `LabeledLDA`
   labels), and the two-level `PA` (`num_super`, `num_sub`).
2. **`fit(self, data, <side-input>, *, ...)`** — `data` is first; the model's
   supervision/covariate input is a *positional* argument immediately after it,
   with its `<thing>_names` as the first keyword-only argument.
3. The Gibbs-sampler family shares a fixed keyword block, in this order:
   `iters, num_samples, sample_interval, progress, progress_interval,
   keep_theta_draws, num_theta_draws, convergence_tol, check_every`.
4. **Every model exposes `model.settings`** — its constructor configuration as a
   JSON-serialisable dict, keyword-named to match `__init__` (issue #400). It
   reports the *effective* values in force (e.g. `num_threads` after its `max(1)`
   floor; `sampler`/`init` as the canonical public string, not an internal flag),
   and omits data/guidance inputs (`keywords`, `seed_words`, covariate arrays,
   embeddings, LLM backends). `tests/test_model_settings.py` derives the expected
   keys from each constructor signature, so a new parameter that is not surfaced in
   `settings` fails the suite — there is no second inventory to maintain. The
   analysis manifest (`record_fit`) reads this surface directly.
5. **`top_words(n, topic=None)` returns `(word, prob)` pairs**, not bare words:
   `list[list[tuple[str, float]]]` for all topics, or `list[tuple[str, float]]`
   for one `topic=`. So `", ".join(model.top_words(n, topic=t))` raises — unpack
   the pairs first (`", ".join(w for w, _ in model.top_words(n, topic=t))`). The
   probabilities are the topic-word weights, descending. This is uniform across
   every model (issue #742).

## Threads and stopping rules

Reproducibility is the default. Gibbs samplers use `num_threads=1` by default:
this is the exact, reference-comparable path. Passing `num_threads > 1` is an
explicit opt-in to an approximate parallel sampler; the result remains
reproducible for the same seed and thread count, but it is not expected to equal
the single-threaded fit. Variational and matrix-factorization models may use all
available cores by default when their parallel reductions remain bit-exact.

Likewise, `convergence_tol=0.0` is the default for Gibbs samplers. It means
"run the requested number of sweeps," not "the chain has converged." A positive
tolerance is only a pragmatic early-stop heuristic on the log-likelihood trace;
it does not establish MCMC mixing. Use `stop_reason(model)` to report whether
the heuristic stopped a fit, and reserve MCMC-native diagnostics for retained
draws or multiple chains.

### The `fit` side-input vocabulary

The second positional argument names what the model conditions on. One concept,
one name:

| Concept | Name | Models |
|---|---|---|
| Document covariate design matrix | `covariates` (universal alias); native `features` (DMR/GDMR), `prevalence` (STM/STS) | DMR, GDMR, STM, STS, KeyATM |
| Content covariate (group → words) | `content` | STM |
| Time index | `times` | DTM |
| Document labels | `labels` | LabeledLDA |
| Document groups | `groups` | SAGE |
| Supervised response | `y` | SupervisedLDA |
| Document embeddings | `doc_embeddings` | BERTopic, FASTopic, Top2Vec |
| Word embeddings | `word_embeddings` | ETM |

## The alias policy

topica is a faithful drop-in for several reference packages (R `stm`/`keyATM`,
MALLET, tomotopy), and migrating users arrive with the reference package's
vocabulary in their fingers. The rule:

- **The topica canonical name is the default and what the docs use.**
- **A model may keep a native primary that matches its reference implementation**
  where that is the explicit goal — `STM.fit(data, prevalence=...)` mirrors R
  `stm`, and that fidelity is intentional, not drift.
- **`covariates=` is the one cross-model alias that must work on every covariate
  model**, so code written against one transfers to the others. The test enforces
  this.
- **Reference-package aliases are welcome as additive keyword aliases** to ease
  switching (for example `GDMR.fit` accepts `metadata=` for users coming from
  tomotopy's `GDMRModel`). Resolve the aliases to one value and raise if more than
  one is supplied; never let two spellings silently disagree.

## Reference: the covariate family

These signatures are the template. A new covariate model should look like one of
them, adding only its own model-specific knobs.

```python
DMR.__init__(num_topics, *, beta=0.01, optimize_interval=50, burn_in=200,
             seed=13, prior_variance=1.0, lbfgs_iters=20, sampler='sparse')
DMR.fit(data, features=None, *, feature_names=None, iters=1000,
        num_samples=5, sample_interval=25, progress=None, progress_interval=50,
        keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0,
        check_every=10, covariates=None)

STM.fit(data, prevalence=None, *, prevalence_names=None, content=None,
        content_names=None, iters=500, convergence_tol=1e-05, ..., covariates=None)
LabeledLDA.fit(data, labels, *, label_names=None, iters=1000, ...)
SupervisedLDA.fit(data, y, *, iters=25, var_iters=15, ...)
```

## Worked example: GDMR (outer shape shared, inner computation unique)

`GDMR` (generalized DMR) is the live test of these rules. It is *not* "DMR with
continuous features passed raw" — that would just be DMR. The g-DMR contribution
(Lee & Song 2020) is a Legendre-polynomial basis over continuous metadata plus a
decay prior that shrinks higher-order terms, giving a smooth topic-distribution
function. So GDMR follows the rules where they apply and takes the documented
"computation can be unique" carve-out where the model genuinely differs:

- **Outer shape matches DMR.** `num_topics` first positional; keyword-only rest;
  `seed=13`. `fit(data, features=None, *, iters=1000, ...)` with the same
  keyword block.
- **Cross-model alias honored.** `features` is the native primary; `covariates=`
  works (universal alias); `metadata=` works (tomotopy migration alias).
- **Model-specific params are namespaced, not forced into DMR's vocabulary.**
  `degrees`, `metadata_range` describe the basis; `sigma`, `sigma0`, `decay` are
  the structured prior (DMR's single `prior_variance` cannot express a
  per-degree decay, so reusing that name would mislead). New methods `tdf` /
  `tdf_linspace` read the fitted surface.

The principle: **share the skeleton, keep the organs.** A reader who knows DMR
can drive GDMR; a reader who knows g-DMR finds the parameters the literature
names.

## Resolved naming decisions

These were the candidate inconsistencies (#155). Each is now decided, so the
`KNOWN_DRIFT` list in `tests/test_naming_conventions.py` is empty.

- **LLM-based evaluation is a namespace, `topica.llm.*`, not flat `llm_*`.** The
  diagnostics that call an external model (`topica.llm.coherence`,
  `topica.llm.intrusion`, `topica.llm.select_k`, and the future
  `outlier`/`repetitiveness`/`diversity`/`alignment`) are grouped under
  `topica.llm` rather than spread as top-level `llm_*` functions. This is a
  deliberate carve-out from the flat-function rule: the suite is a coherent,
  growing family that shares one property the rest of the library does not —
  `llm-bounded` non-determinism — and the namespace signals that at the call site.
  `topica.llm.backend` is the same constructor as the top-level `topica.llm_backend`
  (kept flat because it is also the bring-your-own-model adapter for `TopicGPT` and
  `label_topics`, not only an eval metric).

- **Temporal index — `times`.** Canonical across models (DTM's positional arg).
  KeyATM now accepts `times=` and keeps `timestamps=` as an alias. A
  `test_temporal_models_accept_times` check enforces `times` on every temporal
  model; new temporal models must use it and not introduce a third name.
- **`check_every` is kept** for the convergence-check cadence. The `*_interval`
  pattern names how often we *do* something (sample, optimize, report);
  `check_every` names how often we *test* convergence, a deliberately distinct
  concept, so it is not renamed to `check_interval`.
- **Gaussian-prior naming follows each model's lineage.** `DMR` uses
  `prior_variance` (a variance, matching its MALLET-family lineage); `GDMR` uses
  `sigma`/`sigma0` (std-devs, matching the g-DMR literature) and additionally
  needs two scales plus `decay`, which a single `prior_variance` cannot express.
  These are different parameterizations of different priors, so they keep
  different names.
- **`labels` (LabeledLDA) vs `groups` (SAGE) are different things.** LabeledLDA's
  `labels` restrict which topics a document may use (a per-document topic set);
  SAGE's `groups` select a content group that reshapes the topic-word
  distribution. Different roles, so different names — the right outcome under the
  one-concept-one-name rule.
