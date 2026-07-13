# TensorLDA: implementation validation

`TensorLDA` is topica's experimental Rust implementation of Online Tensor
Latent Dirichlet Allocation (TLDA), a method-of-moments estimator based on
second- and third-order cumulants. The underlying method is published in
Kangaslahti et al. (2026), [*Political Analysis* 34(1):53–77](https://doi.org/10.1017/pan.2025.10024).
The upstream implementation is [TensorLy TLDA](https://github.com/tensorly/tlda).

!!! warning "Experimental: use as a topic-recovery method, not a prevalence estimator"
    TensorLDA is gated behind `topica.enable_experimental()`. The implementation
    produces reproducible topic-word and document-topic matrices from a fixed
    seed, but its `weights` have not yet been validated as calibrated estimates
    of substantive topic prevalence. Treat the weights as model diagnostics, not
    as estimates for group or time comparisons.

## Scope

The published method targets large and online corpora. Topica implements its
count-based moment, whitening, factorization, and variational-prediction path in
Rust; it does not currently provide the upstream project's CUDA implementation.
The `n_eigenvec` option permits a whitening rank greater than the number of
topics. Topica returns exactly `num_topics` weights in either case. Its
column-norm recovery rule for this rectangular case is a Topica extension, not a
claim of byte-for-byte upstream compatibility.

```python
import topica

topica.enable_experimental()
model = topica.TensorLDA(
    num_topics=10, n_eigenvec=10, theta=1.0, seed=42,
)
model.fit(docs)

topic_words = model.topic_word
raw_factors = model.unwhitened_raw
```

## Current evidence

### Upstream comparison

On the two-category UCI 20 Newsgroups example, the square-rank configuration
with equivalent settings produced aligned topic-word cosines of 0.999997 and
0.999974. The exact upstream example configuration uses `n_topic=2` and
`n_eigenvec=10`; the upstream package returns ten weights for that two-topic
configuration, so its weight output is not a valid parity target. With that
configuration, Topica's aligned topic-word cosines were 0.999686 and 0.735117.

The fit-only median time on that small UCI corpus was 0.0538 seconds for Topica
and 0.1575 seconds for the upstream NumPy implementation (2.93× faster). This
is a small CPU benchmark, not a GPU comparison or a general performance claim.

### Known-truth simulations

[`parity/tlda_true_lda_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/tlda_true_lda_compare.py)
samples documents from an ordinary LDA generative process with known unequal
Dirichlet topic weights, then reports aligned topic-word cosine, weight rank
correlation, and document-topic recovery. It is intentionally a report rather
than a pass/fail gate: current results establish a baseline for improvement.

On its fixed simulation (2,500 documents, 100 tokens each, four topics), the
current implementation obtains mean aligned topic-word cosine 0.529 at
`n_eigenvec=4` and 0.544 at `n_eigenvec=8`. The corresponding weight-rank
correlations with the planted weights are 0.60 and 0.40. These are not yet
strong enough to validate the estimator as a reliable prevalence-measurement
tool.

A separate, more realistic synthetic gentrification corpus with template
sentences, role-specific language, and neighborhood-conditioned topic mixtures
is a stress test rather than an LDA-identification test. There, Topica obtains
mean aligned topic-word cosine about 0.68, but weight rank correlation with the
known aggregate prevalence is weak (0.16 at `R=K`, -0.39 at `R>K`). The upstream
implementation is similarly weak on prevalence in its supported square case.

## Interpretation and next steps

The evidence supports using TensorLDA experimentally to explore topic-word
patterns in large count corpora, with the same coherence, stability, intrusion,
and representative-document checks expected of every topic model. It does not
yet support interpreting `weights` as calibrated topic prevalence or using them
for substantive covariate effects.

Priorities before removing the experimental gate are:

1. Improve and validate recovery on the known-truth LDA simulator across corpus
   size, topic separation, and unequal prevalences.
2. Make the upstream comparison portable and automated for its valid square-rank
   configuration.
3. Derive and validate a calibrated prevalence estimator, potentially separate
   from the tensor factorization's raw component norms.
4. Add robustness results for overlapping vocabularies and covariate-structured
   corpora.

