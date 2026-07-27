# VAE decoder GEMM probe (issue #378)

A throwaway measurement harness for issue #378 — "ProdLDA-family VAE fit is
single-threaded scalar Rust". It isolates the three `O(N·K·V)` decoder terms shared
by `ProdLDA`, `CombinedTM`, `ZeroShotTM`, `ETM(inference="vae")`, `InfoCTM`, and
`Scholar`, and compares the current **scalar triple-loop** against a pure-Rust
**GEMM** (`matrixmultiply`, BLAS-free — already a transitive dep via `ndarray`).

It is a **standalone crate, detached from the topica workspace** (see the empty
`[workspace]` in `Cargo.toml`), so it never enters the library build or CI. Run it
by hand:

```bash
cd benchmarks/vae_gemm_probe

# speed (scalar vs GEMM, single-threaded) + numeric delta vs the scalar path
cargo run --release

# bit-for-bit determinism of the threaded GEMM across thread counts
cargo build --release
MATMUL_NUM_THREADS=1 ./target/release/determinism
MATMUL_NUM_THREADS=2 ./target/release/determinism
MATMUL_NUM_THREADS=4 ./target/release/determinism   # identical checksums => bit-identical
```

The decoder loops mirrored here are verbatim from `src/prodlda.rs`
(`batch_forward` / `batch_backward`):

```
forward : logit_raw[i][j]  = sum_t theta_do[i][t] * beta[t*V+j]
backward: dtheta_do[i][t]  = sum_j dlogit_raw[i][j] * beta[t*V+j]
backward: g_beta[t*V+j]   += sum_i theta_do[i][t] * dlogit_raw[i][j]   (cross-doc reduction)
```

Findings and the recommended direction are written up in
[`../notes_issue_378_vae_gemm.md`](../notes_issue_378_vae_gemm.md).
