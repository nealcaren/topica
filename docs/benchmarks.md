# Benchmarks

How fast topica is, measured honestly. Every number here is fit time only (model
construction and import excluded), on fixed-seed synthetic corpora, on one
machine (Apple M-series, 14 cores), and reproducible with the command shown.
Speed depends on corpus size, vocabulary, the number of topics, and hardware, so
read these as orders of magnitude, not guarantees.

## STM vs R `stm`

This is the comparison that matters for social scientists. R's `stm` is the
field standard, and a fit you wait minutes for in R runs in seconds in topica.
Both engines run the **same** number of EM iterations from a spectral
initialization (R with `emtol=0` so it does not stop early), so this measures
per-iteration cost, not time to convergence.

**All cores** (topica parallelizes the variational E-step; R `stm` is single-threaded):

| docs | vocab | K | topica | R `stm` | speedup |
|-----:|------:|---:|-------:|--------:|--------:|
| 1,000 | 500 | 10 | 0.10s | 3.21s | **32.2×** |
| 2,000 | 2,000 | 10 | 0.47s | 6.78s | **14.4×** |
| 5,000 | 5,000 | 20 | 2.35s | 26.6s | **11.3×** |

**Single core** (apples-to-apples, `RAYON_NUM_THREADS=1`):

| docs | vocab | K | topica | R `stm` | speedup |
|-----:|------:|---:|-------:|--------:|--------:|
| 1,000 | 500 | 10 | 0.30s | 3.13s | **10.4×** |
| 2,000 | 2,000 | 10 | 1.16s | 6.60s | **5.7×** |
| 5,000 | 5,000 | 20 | 6.95s | 27.5s | **3.9×** |

So topica is roughly **4 to 10 times faster single-threaded and 11 to 32 times on
all cores**, and it produces the same fit (the content and prevalence models are
[validated against R `stm`](publishing/validation.md)). Reproduce:

```bash
python benchmarks/bench_stm.py                      # all cores
RAYON_NUM_THREADS=1 python benchmarks/bench_stm.py  # single core
```

!!! note "What this is, and is not"
    Per-iteration fit time, not time to convergence (the two engines may need a
    different number of iterations to converge). One machine, synthetic corpora.
    R `stm` is single-threaded by design; the all-cores column is topica's
    automatic parallelism, which is the speed you actually get.

!!! warning "Time to convergence with a wide prevalence design"
    The number of EM iterations to convergence is not the same across engines, and
    a wide prevalence design (a `s(day)` spline, or many one-hot covariate levels)
    can cost topica somewhat more EM iterations than R `stm` even at the same
    `emtol`. On poliblog5k (5,000 docs, K=20, `~ rating + s(day)`) topica converges
    in roughly 32 iterations to R's 20; with `~ rating` alone both are near 20. The
    fit is the same (topics match R `stm`) and the bound increases monotonically —
    it just takes a few more steps near the optimum. So when you report wall-clock
    for a covariate-rich STM, time it **to convergence**, not at a fixed iteration
    count, or the comparison flatters whichever engine converges in fewer steps.
    `parity/stm_spline_iters_247.py` reproduces and explains this.

## LDA: MALLET's algorithm without the JVM

topica's LDA began as a port of RustMallet, David Mimno's Rust port of MALLET's
SparseLDA collapsed-Gibbs sampler, and follows its sampler and fixed-point
optimizer closely. It uses its own RNG (PCG) rather than RustMallet's, so it is
not byte-identical to RustMallet; against Java MALLET (also a different RNG) it
recovers the same topics on a planted corpus (cosine 1.000). On fit
time it is roughly at parity with Java MALLET — and adds no JVM startup, unlike
the JVM samplers or pure-Python gensim.

Against [tomotopy](https://github.com/bab2min/tomotopy), a C++/SIMD library,
which one is faster turns on the number of topics K, because the two make
opposite algorithmic bets. tomotopy computes a dense topic distribution for every
token and vectorizes it with Eigen, which is fastest when K is small; topica
(like MALLET) uses a sparse sampler that visits only the topics a word actually
occupies, which wins when K is large. On a 3,500-document, 2,632-word corpus (500
Gibbs iterations, single core, fit time only) the two cross between K=50 and
K=100:

| K | topica | tomotopy | faster |
|---:|-------:|---------:|--------|
| 20 | 20.2s | 13.4s | tomotopy 1.51× |
| 50 | 22.5s | 17.5s | tomotopy 1.29× |
| 100 | 25.0s | 28.3s | topica 1.13× |
| 200 | 30.3s | 47.5s | topica 1.57× |
| 400 | 39.1s | 90.5s | topica 2.31× |

topica's time barely moves as K grows, because the sparse sampler touches only
the topics each word occupies; tomotopy's rises with K, because it scores all K
every token. Fine-grained topic models, the large-K regime social scientists
often want, favor topica's sampler; small-K fits favor tomotopy's. Either way
topica also retains the MCMC posterior draws its uncertainty tooling needs
(`composition_theta`, `prevalence_ci`) at no measurable extra cost, which neither
MALLET nor tomotopy computes at all. Reproduce the sweep with:

```bash
python benchmarks/k_crossover.py
```

## Memory

topica holds the corpus and model state in compact native arrays, so it also fits
in a fraction of the reference tools' memory. Fitting the structural topic model
on the poliblog corpus, peak resident memory:

| docs | topica STM | R `stm` |
|-----:|-----------:|--------:|
| 2,000 | 297 MB | 1,399 MB |
| 3,500 | 392 MB | 1,388 MB |
| 5,000 | 484 MB | 1,506 MB |

Roughly a quarter to a third of R `stm`'s footprint.

## keyATM vs R `keyATM`

topica's keyATM reproduces the R package's keyword-assisted model and is
[validated against it](replications/keyatm-dynamic.md): the same keyword topics,
the same per-sweep asymmetric-α estimation, the same `model_fit` log-likelihood.
On speed it matches R's C++ sampler single-threaded and adds a
document-partitioned parallel sweep that R has no equivalent of. Same keywords,
same number of Gibbs sweeps, α learned each sweep on both sides; fit time only.

| docs | vocab | K | sweeps | topica (1 core) | topica (4 cores) | R `keyATM` |
|-----:|------:|---:|-------:|----------------:|-----------------:|-----------:|
| 2,000 | 2,632 | 10 | 1,000 | 23.3s | **13.5s** | 25.4s |

So topica is at parity with R single-threaded — a touch faster here — and about
**1.9× faster on four cores**. If you do not need the R-matching asymmetric prior, `estimate_alpha=False`
fixes a symmetric α and skips the per-sweep slice sampler for a further 15 to 20%
(more at larger K). This row, with the STM and LDA comparisons above, is
reproducible in one command:

```bash
python benchmarks/speed_vs_r.py
```

## Multithread scaling with corpus size

The tables above are a single 2,000-document corpus. For the approximate
parallel Gibbs samplers (LDA, keyATM), the multithread speedup **grows with
corpus size**: each sweep ends with a count-table merge whose cost is fixed
(independent of how many tokens moved), so a larger corpus amortizes it over more
sampling work and parallelizes better. Single-threaded, topica stays at parity
with MALLET at every size; the multithreaded gain is what widens. LDA, K=20,
1,000 Gibbs sweeps, eight cores, seeded subsamples of `poliblog5k`:

| docs | MALLET (1 core) | topica (1 core) | topica (8 cores) | topica vs MALLET, 8 cores | topica thread scaling |
|-----:|----------------:|----------------:|-----------------:|--------------------------:|----------------------:|
| 2,000 | 31.9s | 28.5s | 9.8s | 3.3× | 2.9× |
| 3,500 | 38.6s | 39.6s | 10.3s | 3.7× | 3.8× |
| 5,000 | 60.9s | 63.5s | 15.4s | 4.0× | 4.1× |

So the small-corpus multithreaded figures understate what users see on real,
larger corpora. keyATM parallelizes too but scales less cleanly here: its
per-worker sweep clones a dense topic-word table, a larger fixed cost than LDA's
sparse delta merge, so its thread scaling is more variable (tracked as
optimization headroom). Reproduce the curve with:

```bash
python benchmarks/speed_vs_size.py
```

## Across the family vs tomotopy

tomotopy implements much of the same count-based family in C++, so we can compare
topica model-for-model. The table below is single-threaded, K=20, 500 Gibbs
iterations (variational EM for CTM), on the 3,500-document poliblog corpus; ratio
is reference time over topica time, so above 1 means topica is faster. K=20 sits
in tomotopy's small-K sweet spot (see the LDA crossover above), so this is the
regime least favorable to topica's sparse samplers.

| model | topica | tomotopy | ratio |
|-------|-------:|---------:|------:|
| CTM | 33.1s | 116.9s | **3.5×** |
| DMR | 20.2s | 15.8s | 0.78× |
| LDA | 19.7s | 12.6s | 0.64× |
| LabeledLDA | 5.5s | 3.4s | 0.62× |
| PA | 204s | 89s | 0.44× |
| PT | 77s | 15s | 0.20× |

topica wins decisively on CTM (its structural-topic-model core uses a Laplace
E-step, faster than tomotopy's mean-field CTM), is within ~2× on the SparseLDA
models at this small K (and ahead at large K, per the crossover above), and lags
on PA and PT, which still recompute a dense per-token distribution rather than a
sparse one — tracked as optimization headroom. HDP and supervised LDA are omitted:
topica's HDP and tomotopy's infer different topic counts, and topica's supervised
LDA is variational where tomotopy's is Gibbs, so neither is a like-for-like speed
comparison.

## Inference backends: SparseLDA, WarpLDA, LightLDA, CVB0, and SVI

For most work, SparseLDA (the default `sampler="sparse"`) is the right choice:
its sparse buckets keep it fastest and highest-coherence up to roughly K = 200.
Its per-token cost grows with K, though, so at large K the picture flips.

WarpLDA (`sampler="warp"`) is a Metropolis-Hastings sampler whose per-sweep cost
is **flat in K** (an O(1)-per-token scheme that holds the count tables fixed
while every token samples, then updates them, so each pass touches a single
count matrix). On a 2,000-document, 2,632-word poliblog subsample, fit time and
mean topic coherence (`c_v`, top-10), single core:

| K | sparse | warp | warp vs sparse |
|---:|-------:|-----:|----------------|
| 100 | 9.3s, coh −79.1 | 5.4s, coh −80.6 | faster, ~equal coherence |
| 500 | 13.5s, coh −101.4 | 4.3s, coh −102.2 | **3× faster**, ~equal |
| 1,000 | 17.9s, coh −99.2 | 3.8s, coh **−96.4** | **4.7× faster and higher coherence** |

At K = 1,000 SparseLDA is too slow to mix well in a comparable budget, while
WarpLDA stays fast and mixes more, so it wins on both axes. WarpLDA also
dominates the older LightLDA alias-MH sampler (`sampler="lightlda"`) across the
large-K range — several times faster and markedly higher coherence, because
LightLDA mixes poorly at these topic counts. So: keep `"sparse"` for K up to a
couple hundred, switch to `"warp"` for fine-grained, large-K models
(K ≳ 500); `"lightlda"` is retained for compatibility but `"warp"` supersedes it.

CVB0 (`sampler="cvb0"`) sits on the other axis. It is collapsed variational
Bayes, zeroth-order ([Asuncion et al. 2009](https://arxiv.org/abs/1205.2662)): a
deterministic, non-sampling backend that keeps a soft topic responsibility per
(document, word-type) cell. It tends to give **higher topic coherence**,
increasingly so with K (on the same corpus, mean `c_v` −68.5 against −79.1 for
`"sparse"` at K = 100), but it costs `O(K)` per token, so it is **slower, not
faster** (≈47s vs ≈10s at K = 100) and produces no MCMC `theta_draws`. Use it
when you want the cleanest topics and fit time is not the constraint.

These backends are not LDA-only. The same machinery carries the per-document
prior, seed weighting, supervised label mask, or keyword-switch state across the
family, so the speed and quality choices follow the model:

| Model | default | speed backend | quality / scale backend |
|-------|---------|---------------|-------------------------|
| `LDA` | `sparse` | `warp` (flat in K), `lightlda` | `cvb0` (deterministic) |
| `DMR` | `sparse` | `warp` (per-doc α) | `cvb0` (soft counts feed the λ optimizer) |
| `SeededLDA` | `sparse` | `warp` (seeded word phase) | `cvb0` (asymmetric seed β) |
| `LabeledLDA` | `sparse` | — | `cvb0` (label mask is free; warp can't serve it) |
| `KeyATM` | `sparse` | — | `cvb0` (base model only; opt-in, non-R-parity) |
| `CTM` | `batch` | — | `svi` (online VB, for web-scale corpora) |

The Dirichlet-prior models gain a collapsible `cvb0` backend (zero off-mask
responsibilities make `LabeledLDA`'s supervision and `keyATM`'s keyword switch
exact and free, where a masked MH proposal would barely mix). The
logistic-normal models have no Dirichlet to collapse, so `CTM` instead gains
stochastic variational inference (`inference="svi"`): minibatch Laplace E-steps
with Robbins-Monro global updates, so one epoch touches every document while the
global state stays minibatch-sized — the backend for corpora too large to sweep
in full each EM step. `keyATM`'s and `CTM`'s alternates are opt-in: keyATM-CVB0
trades R-parity for determinism on the base model, and CTM-SVI trades the
per-iteration bound trace for scale.

## Coherence

`c_v` and the other windowed coherence measures are computed in the Rust core,
counting only the word pairs within a topic's top-N rather than a full
vocabulary-by-vocabulary matrix. A 500-topic `c_v` that took minutes in a
pure-Python loop now takes a fraction of a second, which is what makes coherence
practical for model selection at large K.
