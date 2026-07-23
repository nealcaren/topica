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

## Streaming / online fit

`fit` holds the whole count matrix in memory. The method's headline is that it
does not have to: `partial_fit(batch, batch_index)` builds the model one batch at
a time. The first time a `batch_index` is seen it updates the running mean and the
incremental whitening (a faithful port of `sklearn.IncrementalPCA`); every later
sighting whitens that batch with the current PCA and runs the third-order factor
SGD. So the usual driver makes one pass over the batches to build the whitening,
then several passes to train the factors, never materializing the full matrix.

```python
import topica

topica.enable_experimental()
model = topica.TensorLDA(num_topics=10, learning_rate=0.01, seed=42)

vocab = build_vocab(stream)          # fix the vocabulary up front
n_iter_train = 40
for _ in range(1 + n_iter_train):    # pass 0 builds whitening; the rest train
    for i, batch in enumerate(read_batches(stream)):   # each batch: list[list[str]]
        model.partial_fit(batch, i, vocabulary=vocab)
model.finalize()

topic_words = model.topic_word       # topic_word / weights / top_words / transform
doc_topics  = model.transform(some_docs)   # doc_topic is not retained for a stream
```

Fix the vocabulary once via `vocabulary=` (recommended for streaming) or let the
first batch define it; out-of-vocabulary tokens in later batches are dropped.
`pca_batch_size` bounds the incremental-PCA sub-batch (and so the internal
Gram-matrix size) independently of how large a batch you pass. `finalize()`
recovers the topic-word matrix and weights; it errors unless every batch was fed
at least twice (one whitening pass, then at least one training pass). `doc_topic`
is unavailable on a streamed model because the documents are not retained — call
`transform(docs)` for the documents whose topics you want.

Streaming reproduces the reference package's online `partial_fit` exactly: on a
shared corpus the aligned topic-word cosine between topica's streaming fit and the
reference's online fit is `> 0.9999`. Because the method-of-moments factorization
is init-sensitive, both implementations recover the same topics — including the
same partial recoveries on hard corpora. `parity/tlda_compare.py` runs the
comparison when the reference package is available.

## Running the upstream comparison

The reference ([TensorLy TLDA](https://github.com/tensorly/tlda)) is not on PyPI
and is not a topica dependency, so the parity script locates it at run time and
skips cleanly when it is absent. Obtain it once and point `TOPICA_TLDA_REF` at
the checkout:

```bash
git clone https://github.com/tensorly/tlda /path/to/tlda
export TOPICA_TLDA_REF=/path/to/tlda
python parity/tlda_compare.py     # runs the square-rank comparison; exits 0 if the ref is absent
```

`parity/tlda_ref.py` centralizes this lookup. Because the script exits 0 when
`TOPICA_TLDA_REF` is unset, it is safe to wire into CI or a scheduled integration
job: the job is a no-op wherever the reference is not installed, and a real
comparison wherever it is. No machine-specific path is baked into the repository.

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
correlation, and document-topic recovery.

The square-rank *topic-recovery* piece of that simulation is now an explicit,
CI-runnable acceptance gate in
[`tests/test_tlda_recovery.py`](https://github.com/nealcaren/topica/blob/main/tests/test_tlda_recovery.py):
at `n_eigenvec = num_topics` it asserts a mean aligned topic-word cosine floor
(0.50) and a per-topic floor (0.35), guarding against regression. Prevalence
calibration is deliberately kept as a *separate, ungated* question — the test
computes the planted-weight and document-topic diagnostics but does not assert on
them, because the `weights` are not yet a validated prevalence estimator. The
rectangular `n_eigenvec > num_topics` case is exercised as a Topica extension,
not an upstream-parity claim.

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

1. Raise the known-truth topic-recovery acceptance floor (now enforced at square
   rank in `tests/test_tlda_recovery.py`) across corpus size, topic separation,
   and unequal prevalences.
2. Broaden the now-portable upstream comparison (`parity/tlda_compare.py`,
   located via `TOPICA_TLDA_REF`) beyond the small UCI example and wire it into a
   scheduled integration job.
3. Derive and validate a calibrated prevalence estimator, potentially separate
   from the tensor factorization's raw component norms.
4. Add robustness results for overlapping vocabularies and covariate-structured
   corpora.

