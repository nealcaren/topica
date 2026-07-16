# Dynamic keyATM: U.S. Supreme Court opinions

**Source.** Eshima, S., Imai, K., & Sasaki, T. (2024). Keyword-Assisted Topic
Models. *American Journal of Political Science*, 68(2), 730–750.
[doi:10.1111/ajps.12779](https://doi.org/10.1111/ajps.12779). Replication data:
[Harvard Dataverse, `doi:10.7910/DVN/RKNNVL`](https://doi.org/10.7910/DVN/RKNNVL).

The paper's third application fits a dynamic keyATM to U.S. Supreme Court
opinions and shows that anchoring topics with keywords yields more interpretable
topics and better document classification than weighted LDA, while a hidden
Markov model tracks how topic prevalence shifts over time. We refit that analysis
with topica's `KeyATM` and compare against the paper's published tables.

The corpus is 17,245 opinions from 1946 to 2012, drawn from the Supreme Court
Database, with a human-coded primary topic per opinion and a keyword list per
topic. We use the same 14 topics, a 5-state HMM (the paper's choice), and the
default information-theory token weighting.

## Fit

```python
import topica

# `docs` are the tokenized opinions; `years` is the filing year of each opinion;
# `seeds` maps each of the 14 topic labels to its keyword list.
model = topica.models.KeyATM(seeds, num_topics=14, seed=2020)
model.fit(docs, timestamps=years, num_states=5, iters=1000, num_threads=8)
```

The documents are sorted by year internally, so `doc_topic` returns in the
original order and `time_prevalence` follows `time_labels` (1946 … 2012).

## Same data

Before comparing model output, we confirm we are on the same corpus. topica's
gold-label counts match the paper's Table 5 exactly, every topic:

| Topic | topica count | paper Table 5 |
|---|---:|---:|
| Criminal procedure | 4,268 | 4,268 |
| Economic activity | 3,062 | 3,062 |
| Civil rights | 2,855 | 2,855 |
| Judicial power | 1,964 | 1,964 |
| First amendment | 1,795 | 1,795 |
| Due process | 738 | 738 |
| Federalism | 720 | 720 |
| Unions | 664 | 664 |

## Topic interpretability (paper Table 6)

The estimated top words match the paper's keyATM column nearly word for word:

| Topic | topica | paper keyATM |
|---|---|---|
| Criminal procedure | trial jury defendant evidence criminal sentence petitioner conviction | trial jury defendant evidence criminal sentence petitioner right conviction counsel |
| First amendment | public amendment first speech government religious right may | public amendment first speech government may interest right political religious |
| Unions | employee union labor employer board agreement employment contract | employee union labor employer board agreement party contract employment bargaining |
| Federal taxation | tax property income pay payment amount fund benefit | tax property benefit income interest … |

Our Privacy topic (*search officer police arrest warrant amendment fourth
seizure*) is a clean Fourth-Amendment cluster, if anything tighter than the
paper's.

## Document classification (paper Figure 4)

The paper measures classification by the area under the ROC curve for each
topic's posterior topic proportion predicting the human label, and reports that
keyATM beats weighted LDA on every topic except Privacy. topica's per-topic AUROC
averages 0.79 (median 0.80), and Privacy is the lone weak topic at 0.57, the same
exception the paper singles out and attributes to uninformative Privacy keywords.

| Strong (AUROC ≥ 0.88) | Middle | Weak |
|---|---|---|
| Unions 0.98, Federal taxation 0.98, First amendment 0.95, Miscellaneous 0.93, Criminal procedure 0.90, Attorneys 0.89 | Judicial power 0.82, Civil rights 0.79, Interstate relations 0.75 | Due process 0.67, Federalism 0.66, Economic activity 0.62, Privacy 0.57 |

Single-label classification accuracy is 0.43, about six times the 0.07 chance
rate over 14 topics. (Private action has only three labeled opinions and carries
no signal, so its AUROC is not meaningful.)

## Direct comparison with R keyATM

Matching the published tables is one bar. A stronger one is matching the
reference software's own output. We fit R's `keyATM` and topica's `KeyATM` on the
same documents with the same settings (14 topics, 5 states, 300 sweeps, seed
2020, information-theory weighting), topica single-threaded so both run the exact
sequential sampler, and compare directly. Because keywords pin each topic to a
fixed label in both engines, topic *k* is the same topic in both, so we compare
them position by position.

The two are different samplers in different languages with different random
number streams, so they cannot agree bit for bit. They agree to the level that
matters:

- **Topic word distributions.** Mean cosine 0.93 across the 14 topics (median
  0.93, minimum 0.86). The estimated topics are the same.
- **Classification.** Mean AUROC 0.82 in R and 0.82 in topica, with a mean
  absolute per-topic difference of 0.017. The largest gap, 0.055, is on
  Interstate relations, which has only 119 labeled opinions.

This is what "topica reproduces keyATM" means in practice: hand both engines the
same corpus and they return the same topics and the same classification
performance.

## Time trend

The HMM partitions 1946–2012 into regimes and recovers the expected movements in
the docket: Judicial power falls steeply across the period, Economic activity and
Unions rise, and Criminal procedure and First amendment decline. `time_prevalence`
gives the smoothed proportion per year and `time_state` the regime each year
occupies.

To add credible bands to the time-trend figure, use `topica.time_prevalence_ci`,
which reads the credible interval directly from the retained MCMC theta draws
(pass `keep_theta_draws=True` at fit, which is the default):

```python
import topica

ci = topica.time_prevalence_ci(model, timestamps=years, ci=0.95)
# ci["labels"]   — list of period labels aligned with model.time_labels
# ci["mean"]     — (T, K) posterior mean prevalence per period
# ci["ci_low"]   — (T, K) lower 2.5th-percentile credible bound
# ci["ci_high"]  — (T, K) upper 97.5th-percentile credible bound
```

This differs from `topica.keyatm.by_strata`: `by_strata` widens a descriptive
normal interval using Rubin's rules; `time_prevalence_ci` reads the band
directly off the posterior draw stack, which gives the HMM's own uncertainty
without a normal approximation. Use `time_prevalence_ci` for ribbon plots that
claim to show posterior uncertainty.

## What differs

Two differences are worth stating plainly. First, this is a single chain of 1,000
sweeps, where the paper averages five chains of 3,000; the topics and the AUROC
ordering are stable, but the exact numbers will move at the third decimal across
seeds. Second, the change-point HMM makes prevalence piecewise-constant within a
regime, so a year-by-year curve has steps at the estimated change points rather
than the fully smooth path some readers may expect.

## Speed

Per-sweep cost on this corpus (22.4M weighted tokens, 14 topics, 5 states),
measured at steady state:

| Engine | s/sweep | vs R |
|---|---:|---:|
| R `keyATM` (single-thread) | 1.8 | 1.0× |
| topica (single-thread) | 1.7 | 1.05× faster |
| topica (8 threads) | 0.47 | **3.8× faster** |

topica matches R single-threaded and pulls ahead with threads. The single-thread
parity came from one change: the sampler had been probing a per-topic hash map
for every keyword topic on every token, hundreds of millions of lookups per
sweep, which an inverted word-to-topic index removes without changing the sampled
distribution. R's `keyATM` has no multi-threading, so the 8-thread column is the
margin available on a multi-core machine.
