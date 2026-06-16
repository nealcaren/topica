# Structural Topic Model: the `stm` vignette

**Source.** Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). stm: An R
Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
The `stm` package is the field standard for prevalence- and content-covariate
topic models in the social sciences.

topica's `STM` reimplements the same model: correlated topics with a prevalence
regression, a content (SAGE) covariate, spectral initialization, and effect
estimation by the method of composition. This page asks whether it produces the
same answers as R's `stm`.

## What "replicate" means for STM

STM is fit by variational EM, which is non-convex: the objective has many local
optima, and the solution depends on where the optimization starts. R's own `stm`
does not return one canonical answer. Fit it twice from different random seeds
and the two topic-word matrices agree only to a cosine of about 0.81. So the bar
is not bit-identical output. It is statistical: under a matched initialization,
topica should land in the same neighborhood of solutions that R lands in, and its
agreement with R should sit inside the spread of R's agreement with itself.

We feed identical integer-coded documents to both engines and align topics
one-to-one before comparing. The harnesses live in `parity/`:
[`stm_poliblog_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_poliblog_compare.py)
and [`stm_poliblog5k_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_poliblog5k_compare.py)
for the prevalence model on Poliblog,
[`stm_content_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_content_r_compare.py)
for the content model, and
[`stm_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_r_compare.py)
for the small Gadarian stress case.

## Content model: exact agreement

The content (SAGE) covariate is the deterministic part of STM: given the topic
assignments, the per-group word distributions follow in closed form. Here topica
and R agree exactly. On a bilingual corpus fit with `content = ~group`, K = 2,
the best-aligned cosine between R's and topica's per-group word distributions is
1.000 in both groups, and both engines separate the two topics rather than
collapsing them (topic-separation near 0 in each):

| Content group | topica–R cosine | both separated |
|---|---:|:--|
| `de` | 1.000 | yes |
| `en` | 1.000 | yes |

This is the path where a symmetric-initialization bug once collapsed all topics
to the background; the exact match against R is how we know it is fixed.

## Prevalence model: same neighborhood as R, and the same conclusions

For the prevalence model we compare topica's spectral fit to R's spectral fit on
the `stm` Poliblog vignette, against the floor of R's agreement with itself. On
the 2,000-document corpus at K = 20 the two engines' topic-word matrices align to
a cosine of 0.97; on the full 5,000-document corpus at K = 15 it is 0.84. Both
sit well above how closely R reproduces *itself* across initializations:

| Comparison (Poliblog 5k, K = 15) | aligned cosine |
|---|---:|
| R Spectral vs R Random (R's own basin spread) | 0.62 |
| R Random vs R Random (R's self-consistency) | 0.68 |
| **R Spectral vs topica Spectral** | **0.84** |

topica reproduces R's spectral solution more closely than R reproduces itself
from a different seed. Where the per-topic cosine dips (the 5k median is 0.98,
but a few topics fall lower) it is always a handful of genuinely bistable topics
that the two optimizers split differently, never a systematic offset — the
expected behavior of a non-convex model, where there is no single STM fit to
reproduce.

What replicates stably across optima is the substantive conclusion. Regressing
topic prevalence on ideology, topica recovers R's effect coefficients with a
Pearson correlation of 0.84, and the same sign and significance on 13 of 15
topics. The [Poliblog](../examples/poliblog.md) and
[Gadarian](../examples/gadarian.md) worked examples refit the canonical `stm`
vignettes end to end and recover the prevalence effects the package documents,
with honest standard errors from the method of composition.

The smaller Gadarian survey corpus (339 documents, K = 3) is a deliberately
harder, more multimodal case: with so few short open-ended responses, R itself
self-agrees only to a cosine of 0.81, and topica lands at 0.51 — still inside the
spread of R's own Spectral-versus-Random runs (0.62). It is the stress test, not
the headline; see [`stm_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_r_compare.py).

## Speed

On matched iterations from a spectral start, topica fits the same model 3–22×
faster than R `stm`, single-threaded, and more with multiple cores, since topica
parallelizes the variational E-step while `stm` is single-threaded. The full
table is on the [benchmarks page](../benchmarks.md).
