# Issue #378 — options for speeding up the ProdLDA-family VAE

**Status:** exploration only. No library code changed on this branch; this is a
findings note plus a reproducible measurement harness
([`vae_gemm_probe/`](vae_gemm_probe/)). It extends the maintainer's earlier
profiling + rayon spike ([issue #378 comment](https://github.com/nealcaren/topica/issues/378#issuecomment-5048073906))
with a direct, apples-to-apples measurement of the recommended GEMM direction and a
bit-level determinism proof.

## The problem, briefly

The hand-coded dense-decoder VAE backbone shared by `ProdLDA`, `CombinedTM`,
`ZeroShotTM`, `InfoCTM`, and `Scholar` is single-threaded scalar Rust. On
realistic inputs it runs 4.7–6.8× slower than a CPU PyTorch reference at scale
because the decoder's three dense `O(N·K·V)` terms (and the encoder GEMMs) are
scalar triple-loops while PyTorch uses multithreaded BLAS. (`ETM(inference="vae")`
has its own *sparse* per-document decoder, not the dense `theta·beta`, so it is out
of scope for this change.)

The hard constraint is **determinism**: fits must be bit-for-bit identical
regardless of thread count (fixed-order reductions), with no PyTorch/autodiff
dependency.

The three decoder terms, verbatim from `src/prodlda.rs` (`beta` is flat row-major
`K×V`, `theta_do` is `N×K`):

```
forward : logit_raw = theta_do (N×K) · beta (K×V)
backward: dtheta_do = dlogit_raw (N×V) · betaᵀ (V×K)
backward: g.beta   += theta_doᵀ (K×N) · dlogit_raw (N×V)   ← cross-document reduction
```

The `g.beta` term is the one that blocks a pure rayon-over-batch approach: it is a
reduction *across* documents, so parallelizing it with per-thread accumulation would
reorder the sum and break bit-identity-across-threads. The maintainer's rayon spike
therefore capped at ~1.5–1.8× (Amdahl: only the per-document-independent ~40% is
parallelizable while `g.beta` stays serial).

## What this probe measured

`vae_gemm_probe` replicates the three decoder terms exactly and times the current
scalar triple-loop against a pure-Rust GEMM (`matrixmultiply`, BLAS-free, already a
transitive dependency via `ndarray`). All single-threaded, release build, 4-core
box.

### 1. GEMM is 2.7–6.8× faster than the scalar loop — single-threaded

Decoder forward+backward per pass, scalar vs GEMM:

| size (N / V / K) | scalar | GEMM | speedup |
|---|---|---|---|
| 500 / 500 / 10    | 3.96 ms   | 0.99 ms   | **4.0×** |
| 2000 / 2000 / 20  | 130.2 ms  | 35.3 ms   | **3.7×** |
| 4000 / 3000 / 30  | 579.1 ms  | 109.6 ms  | **5.3×** |
| 3000 / 3000 / 50  | 739.9 ms  | 125.9 ms  | **5.9×** |

K sweep at N=2000, V=2000 (the decoder cost grows with K):

| K | 10 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|
| speedup | 2.7× | 4.2× | 4.9× | 5.4× | 6.8× |

This is a **single-threaded** win — it improves the default path with no thread-count
caveat, and it speeds up the exact `g.beta` reduction that the rayon approach could
not touch (a GEMM's reduction order is fixed, so no determinism cost).

### 2. Numeric delta vs the current scalar path: ~1e-15 (a one-time re-baseline)

Relative max-abs difference between scalar and GEMM outputs was **3e-16 – 4e-15**
across every size — i.e. the last 1–2 ULPs. A blocked GEMM sums in a different order
than the naive `for j in 0..V` loop, so switching to GEMM **changes the exact bits**
of every VAE-family fit. That is not a correctness change (it is float-epsilon), but
it does mean a **one-time re-baseline** of any frozen golden fixtures, and the
FD-gradient + parity sweeps must be re-run. This is the main "cost" of the GEMM
route and is worth stating up front.

### 3. The threaded GEMM stays bit-identical across thread counts ✅

This is the decisive result for #378's hard constraint. Enabling
`matrixmultiply`'s `threading` feature and running the same three GEMMs under
`MATMUL_NUM_THREADS = 1, 2, 4` gave **identical bitwise checksums** for all three
outputs:

```
MATMUL_NUM_THREADS=1 | logit=f2ca8ed25e556779 dtheta=a44a2c4300db642f gbeta=7cf098632fdbce52
MATMUL_NUM_THREADS=2 | logit=f2ca8ed25e556779 dtheta=a44a2c4300db642f gbeta=7cf098632fdbce52
MATMUL_NUM_THREADS=4 | logit=f2ca8ed25e556779 dtheta=a44a2c4300db642f gbeta=7cf098632fdbce52
```

`matrixmultiply` parallelizes over **output blocks**, and each output element's
`k`-dimension reduction is still computed by a single thread in fixed order — so the
result is bit-identical regardless of thread count. This means the GEMM route can be
**multithreaded while keeping the strong "identical regardless of thread count"
guarantee** — no downgrade to the sampler-style "identical from fixed seed + thread
count", and no rayon needed for the matmul work at all.

## End-to-end expectation (Amdahl)

The decoder is only part of a fit iteration. From the maintainer's profile, at the
typical K (10–30) the per-iteration cost splits roughly half decoder-matmul/heads
and half `O(N·V)` elementwise (batchnorm apply, vocab softmax, reconstruction). So:

- **Decoder GEMM alone** ≈ 1.4–1.6× end-to-end at typical K, rising with K (the
  decoder share grows — at K=80 it is ~80% of the iteration, so the fit approaches
  the decoder's own 6–7×).
- To reach the issue's implied ~2–3× target across the board, also GEMM the
  **encoder** layer-2 (`N×H×H`) and head projections (`N×H×K`), and rayon the
  genuinely per-document-independent elementwise remainder (batchnorm apply, recon
  softmax, reparameterization — a fixed-order per-doc map, so bit-identical).

## Options, with trade-offs

| # | Option | Speedup | Determinism | Notes |
|---|---|---|---|---|
| **A** | **GEMM the 3 decoder terms** (`matrixmultiply`, single-threaded) | 2.7–6.8× on the decoder; ~1.4–1.6× end-to-end at typical K | **Strong** (fixed GEMM reduction order); one-time re-baseline vs today's bits | **Recommended first step.** Improves the default path, resolves the `g.beta` reduction the rayon route could not. |
| **B** | GEMM the encoder matmuls (layer-2, head projections) too | Addresses the other ~half at typical K | Strong (same as A) | Layer-1 is a *sparse* BoW matvec — leave it sparse; densifying would lose the sparsity win. |
| **C** | Enable `matrixmultiply` `threading` on top of A/B | Near-linear on the matmul work, on top of A/B | **Strong — proven bit-identical at 1/2/4 threads here** | Keeps the strong guarantee *while multithreaded*. No rayon needed for matmuls. |
| **D** | rayon over batch for the elementwise `O(N·V)` parts | ~1.5–1.8× on that slice (maintainer's spike) | Strong (per-doc map in fixed doc order) | Complements A–C; needed only if profiling still shows the elementwise half dominant. |
| **E** | rayon-parallel `g.beta` with thread-local accumulation | — | **Downgrade** to "seed + thread count" | ✗ Not recommended — the GEMM (A) resolves this deterministically instead. |

## Recommended sequencing

1. **A — GEMM the three decoder terms**, single-threaded first. Promote
   `matrixmultiply` to a direct dependency (one line; BLAS-free, no new transitive
   weight). Re-baseline the golden fixtures once and re-run the FD + parity sweeps.
   Prove `np.array_equal` for `topic_word`/`doc_topic` across two thread counts and
   two runs. Ship — this alone is a real single-threaded win for all five
   dense-decoder models (ProdLDA, CombinedTM, ZeroShotTM, InfoCTM, Scholar).
2. **B — GEMM the encoder** layer-2 + head projections; keep layer-1 sparse.
3. **C — turn on `matrixmultiply` threading** for free multi-core scaling that stays
   bit-identical across thread counts (verified here).
4. **D — rayon the elementwise remainder** only if it still dominates after A–C.

This keeps topica's strongest determinism guarantee intact (no downgrade for the VAE
family), adds no heavy dependency, and turns the current 4.7–6.8× deficit into a
single-threaded win first, then multi-core scaling — rather than the ~1.8× ceiling a
pure-rayon approach hits.

## Caveats

- Numbers are decoder-term microbenchmarks on one 4-core box; treat the end-to-end
  figures as Amdahl estimates, not measured full-fit speedups. The real gate is a
  full-fit benchmark (`benchmarks/`) after implementing A.
- The `matrixmultiply` transpose views used here (`betaᵀ`, `theta_doᵀ` via swapped
  strides) are what a real implementation would use; they add no copy.
- SCHOLAR's content deviation terms and the contrastive path are extra scalar work
  outside these three GEMMs and can stay as-is (they fire only when enabled).
