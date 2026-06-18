# topica paper: reproduction report

- Date: 2026-06-17 longleaf c151406.ll.unc.edu topica-0.23.1 rep=1
- topica: 0.23.1
- Machine: Linux-5.14.0-611.16.1.el9_7.x86_64-x86_64-with-glibc2.34, 384 cores (x86_64)
- Toolchains: Rscript=yes, R:stm=yes, R:keyATM=yes, java=yes, mallet=yes

> Section 6 (performance) is hardware-dependent; absolute timings reflect the machine above. Relative speedups are the portable claim.

## Step status

| Step | Section | Status | Time |
|---|---|---|---|
| bench | 6 | OK | 2110s |
| bench_stm_st | 6 | OK | 85s |
| bench_stm_mt | 6 | OK | 78s |
| speed_vs_r | 6 | OK | 191s |
| speed_vs_size | 6 | OK | 938s |
| bench_scaling | 6 | OK | 213s |
| k_crossover | 6 | OK | 211s |

## Paper claims to update

| § | Claim | Fresh value in |
|---|---|---|
| 5 | STM content (SAGE) per-group cosine = 1.000 (de, en) | log: stm_content |
| 5 | STM prevalence Poliblog aligned cosine vs R | log: stm_prevalence |
| 5 | keyATM agreement with R keyATM | log: keyatm |
| 5 | STS benchmarks vs the sts package | log: sts |
| 5 | LDA vs Java MALLET (cosine / Jaccard) | log: mallet_lda |
| 6 | STM 3-22x faster than R stm (single / multi-thread) | json: bench_results.json + speed_vs_r |
| 6 | LDA at parity with MALLET; multithread speedup grows with size | json: bench_results.json, speed_vs_size.json |
| 6 | keyATM ~2x multithreaded vs R keyATM | json: bench_results.json |
| 6 | K-scaling / crossover curves | json: scaling_results.json, k_crossover_results.json |
| 7 | Spanning comparison: per-model coherence/exclusivity on one corpus | log: worked_example |
| 7 | STM Poliblog covariate-effect z-values (fig_poliblog_effect) | log: worked_example |

## Benchmark outputs (Section 6)

### bench_results.json

```json
[
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 1.7978244019905105,
    "ref_time": 32.286,
    "topica_rss_mb": 178.0390625,
    "ref_rss_mb": 674.4453125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 35.29171066900017,
    "ref_time": 35.009,
    "topica_rss_mb": 91.49609375,
    "ref_rss_mb": 339.36328125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 26.496479810972232,
    "ref_time": 34.763,
    "topica_rss_mb": 103.4921875,
    "ref_rss_mb": 345.56640625
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 19.123689545027446,
    "ref_time": 34.141,
    "topica_rss_mb": 90.91015625,
    "ref_rss_mb": 347.6015625
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 16.056612316984683,
    "ref_time": 34.144,
    "topica_rss_mb": 99.52734375,
    "ref_rss_mb": 347.57421875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 25.135933610028587,
    "ref_time": 23.95907519198954,
    "topica_rss_mb": 91.484375,
    "ref_rss_mb": 707.0546875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 16.38738198397914,
    "ref_time": 17.59532315999968,
    "topica_rss_mb": 94.48046875,
    "ref_rss_mb": 721.08203125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 10.81708724598866,
    "ref_time": 12.676786130992696,
    "topica_rss_mb": 90.76953125,
    "ref_rss_mb": 709.27734375
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 8.292174210015219,
    "ref_time": 7.155101160984486,
    "topica_rss_mb": 91.48046875,
    "ref_rss_mb": 729.0546875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 2.834241345990449,
    "ref_time": 44.009,
    "topica_rss_mb": 205.4296875,
    "ref_rss_mb": 700.8515625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 64.47179259703262,
    "ref_time": 62.349,
    "topica_rss_mb": 135.03515625,
    "ref_rss_mb": 375.88671875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 49.12961907801218,
    "ref_time": 62.344,
    "topica_rss_mb": 158.28515625,
    "ref_rss_mb": 377.609375
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 35.602339458011556,
    "ref_time": 62.554,
    "topica_rss_mb": 161.91015625,
    "ref_rss_mb": 380.98828125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 32.13720007002121,
    "ref_time": 63.005,
    "topica_rss_mb": 136.03515625,
    "ref_rss_mb": 382.6328125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 47.54667285003234,
    "ref_time": 44.003225551045034,
    "topica_rss_mb": 128.2734375,
    "ref_rss_mb": 752.1015625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 29.74447171401698,
    "ref_time": 29.579828094982076,
    "topica_rss_mb": 139.63671875,
    "ref_rss_mb": 739.47265625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 18.04664173902711,
    "ref_time": 16.64892081398284,
    "topica_rss_mb": 134.2734375,
    "ref_rss_mb": 734.9921875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 11.670275347016286,
    "ref_time": 10.090182508982252,
    "topica_rss_mb": 133.0234375,
    "ref_rss_mb": 756.12109375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 3.8834161160048097,
    "ref_time": 54.416,
    "topica_rss_mb": 346.92578125,
    "ref_rss_mb": 688.4921875
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 93.11829062603647,
    "ref_time": 88.843,
    "topica_rss_mb": 175.7734375,
    "ref_rss_mb": 422.11328125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 67.23703386401758,
    "ref_time": 88.869,
    "topica_rss_mb": 216.421875,
    "ref_rss_mb": 421.3984375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 53.81028790201526,
    "ref_time": 89.068,
    "topica_rss_mb": 203.2734375,
    "ref_rss_mb": 422.328125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 46.18894917197758,
    "ref_time": 88.976,
    "topica_rss_mb": 236.265625,
    "ref_rss_mb": 419.90234375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 70.68935568601592,
    "ref_time": 64.12623026402434,
    "topica_rss_mb": 176.26171875,
    "ref_rss_mb": 764.98046875
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 42.00968011200894,
    "ref_time": 43.062275892007165,
    "topica_rss_mb": 182.25,
    "ref_rss_mb": 769.015625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 24.0463702440029,
    "ref_time": 22.305131588014774,
    "topica_rss_mb": 176.25390625,
    "ref_rss_mb": 749.13671875
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 15.96889125101734,
    "ref_time": 13.406808988016564,
    "topica_rss_mb": 176.25390625,
    "ref_rss_mb": 769.4375
  }
]
```

### speed_vs_size.json

```json
[
  {
    "n_docs": 2000,
    "vocab": 2632,
    "stm": {
      "r": 32.395,
      "tt1": 1.7777655830141157
    },
    "keyatm": {
      "r": 34.103,
      "tt1": 35.63771334895864,
      "ttN": 16.088588712969795
    },
    "lda": {
      "r": 23.69271386298351,
      "tt1": 25.215068218007218,
      "ttN": 7.82176113902824
    }
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "stm": {
      "r": 44.488,
      "tt1": 2.8007945219869725
    },
    "keyatm": {
      "r": 63.608,
      "tt1": 65.00868795602582,
      "ttN": 32.77126236399636
    },
    "lda": {
      "r": 43.47297952399822,
      "tt1": 47.34306761296466,
      "ttN": 11.317349033022765
    }
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "stm": {
      "r": 54.247,
      "tt1": 3.7877206649864092
    },
    "keyatm": {
      "r": 89.018,
      "tt1": 94.01338262797799,
      "ttN": 43.6797772180289
    },
    "lda": {
      "r": 64.16212131996872,
      "tt1": 70.53370616701432,
      "ttN": 15.697045790962875
    }
  }
]
```

### scaling_results.json

```json
{
  "n_docs": 5000,
  "vocab": 1000,
  "iters": 10,
  "ks": [
    20,
    50,
    100,
    200
  ],
  "variants": {
    "laplace_keep": {
      "time": [
        0.6365,
        1.8201,
        4.5966,
        20.5656
      ],
      "rss": [
        109.93359375,
        435.078125,
        842.09765625,
        2455.64453125
      ]
    },
    "laplace_nokeep": {
      "time": [
        0.6136,
        1.6522,
        4.2123,
        18.7541
      ],
      "rss": [
        111.0,
        326.5703125,
        266.5078125,
        295.17578125
      ]
    },
    "diagonal_keep": {
      "time": [
        0.5658,
        1.6954,
        4.0006,
        12.7265
      ],
      "rss": [
        123.375,
        368.9296875,
        826.8046875,
        2518.83203125
      ]
    }
  }
}
```

### k_crossover_results.json

```json
{
  "size": 2000,
  "iters": 500,
  "rows": [
    {
      "K": 20,
      "topica": 14.163120580022223,
      "tomotopy": 9.705408584035467,
      "faster": "tomotopy 1.46x"
    },
    {
      "K": 50,
      "topica": 16.03963124100119,
      "tomotopy": 13.154553326952737,
      "faster": "tomotopy 1.22x"
    },
    {
      "K": 100,
      "topica": 18.171268555975985,
      "tomotopy": 16.423152573988773,
      "faster": "tomotopy 1.11x"
    },
    {
      "K": 200,
      "topica": 22.237748876039404,
      "tomotopy": 24.5225828649709,
      "faster": "topica 1.10x"
    },
    {
      "K": 400,
      "topica": 29.889813259011135,
      "tomotopy": 42.92546360602137,
      "faster": "topica 1.44x"
    }
  ]
}
```

## Step output (tails)

### §6 bench — OK

```
== N~5000 (5000 docs, 2632 vocab) ==
  STM vs R stm ...
    topica 3.9s  R 54.4s  speedup 14.0x  topica RSS 347 MB  R RSS 688 MB
  keyATM vs R keyATM (thread sweep) ...
    threads=1  topica 93.1s  R 88.8s  speedup 1.0x  RSS 176 MB
    threads=2  topica 67.2s  R 88.9s  speedup 1.3x  RSS 216 MB
    threads=4  topica 53.8s  R 89.1s  speedup 1.7x  RSS 203 MB
    threads=8  topica 46.2s  R 89.0s  speedup 1.9x  RSS 236 MB
  LDA vs MALLET (matched thread sweep) ...
    threads=1  topica 70.7s  MALLET@1 64.1s  RSS 176 MB
    threads=2  topica 42.0s  MALLET@2 43.1s  RSS 182 MB
    threads=4  topica 24.0s  MALLET@4 22.3s  RSS 176 MB
    threads=8  topica 16.0s  MALLET@8 13.4s  RSS 176 MB
  BERTopic clustering stage ...
  BERTopic leg: skipped — bertopic/umap not importable
Results written to /work/users/n/c/ncaren/topica/benchmarks/bench_results.json
## topica benchmark: speed vs reference (wall-clock)
Corpus: poliblog5k subsampled to each size (seeded, reproducible). Speedup = reference time / topica single-thread time. All timings exclude model loading.
### STM
| n_docs | vocab | ref (s) | topica 1-thread (s) | 1-thread speedup | topica RSS (MB) | ref RSS (MB) |
|---|---|---|---|---|---|---|
| 2000 | 2632 | 32.3 | 1.8 | 18.0x | 178 | 674 |
| 3500 | 2632 | 44.0 | 2.8 | 15.5x | 205 | 701 |
| 5000 | 2632 | 54.4 | 3.9 | 14.0x | 347 | 688 |
### KEYATM
| n_docs | vocab | ref (s) | topica 1-thread (s) | 1-thread speedup | topica 2-thread (s) | 2-thread speedup | topica 4-thread (s) | 4-thread speedup | topica 8-thread (s) | 8-thread speedup | topica RSS (MB) | ref RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 2632 | 35.0 | 35.3 | 1.0x | 26.5 | 1.3x | 19.1 | 1.8x | 16.1 | 2.1x | 91 | 339 |
| 3500 | 2632 | 62.3 | 64.5 | 1.0x | 49.1 | 1.3x | 35.6 | 1.8x | 32.1 | 2.0x | 135 | 376 |
| 5000 | 2632 | 88.8 | 93.1 | 1.0x | 67.2 | 1.3x | 53.8 | 1.7x | 46.2 | 1.9x | 176 | 422 |
### LDA
| n_docs | vocab | ref (s) | topica 1-thread (s) | 1-thread speedup | topica 2-thread (s) | 2-thread speedup | topica 4-thread (s) | 4-thread speedup | topica 8-thread (s) | 8-thread speedup | topica RSS (MB) | ref RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 2632 | 24.0 | 25.1 | 1.0x | 16.4 | 1.1x | 10.8 | 1.2x | 8.3 | 0.9x | 91 | 707 |
| 3500 | 2632 | 44.0 | 47.5 | 0.9x | 29.7 | 1.0x | 18.0 | 0.9x | 11.7 | 0.9x | 128 | 752 |
| 5000 | 2632 | 64.1 | 70.7 | 0.9x | 42.0 | 1.0x | 24.0 | 0.9x | 16.0 | 0.8x | 176 | 765 |
Website table written to /work/users/n/c/ncaren/topica/benchmarks/website_table.md
Thread-scaling figure written to /work/users/n/c/ncaren/topica/paper/fig_thread_scaling.pdf
Memory figure written to /work/users/n/c/ncaren/topica/paper/fig_memory.pdf
All outputs written.
```

### §6 bench_stm_st — OK

```
topica threads: 1;  EM iterations: 30;  R stm: available
  docs  vocab   K |       topica      R stm  speedup
----------------------------------------------------
  1000    500  10 |       0.49s     4.74s     9.7x
  2000   2000  10 |       1.28s    10.32s     8.1x
  5000   5000  20 |       8.84s    37.90s     4.3x
```

### §6 bench_stm_mt — OK

```
topica threads: all (384 cores);  EM iterations: 30;  R stm: available
  docs  vocab   K |       topica      R stm  speedup
----------------------------------------------------
  1000    500  10 |       0.13s     4.67s    36.9x
  2000   2000  10 |       0.50s    10.36s    20.7x
  5000   5000  20 |       2.82s    37.93s    13.4x
```

### §6 speed_vs_r — OK

```
Benchmarking (this fits each model in R and topica)...
corpus: 2000 docs, 2632 vocab (poliblog)
| Model   | Reference | Settings                       |     ref |  topica | speedup | notes |
|---------|-----------|--------------------------------|---------|---------|---------|-------|
| STM     | R stm     | K=20, 30 EM its, spectral      |   34.3s |    1.8s |  19.1x | single-thread (variational EM) |
| LDA     | MALLET    | K=20, 1000 Gibbs its           |   24.2s |   25.8s |   0.9x | 1 thread; 4-thread: 9.8s (2.5x) |
| keyATM  | R keyATM  | K=10 (4 kw), 1000 sweeps       |   35.0s |   35.4s |   1.0x | 1 thread; 4-thread: 18.4s (1.9x) |
```

### §6 speed_vs_size — OK

```
poliblog5k: 5000 docs available | sizes=[2000, 3500, 5000] | threads=8
== N~2000  (2000 docs, 2632 vocab) ==
  STM    vs R stm   : 18.2x single-thread
  keyATM vs R keyATM: 1.0x ST, 2.1x MT (topica 2.2x thread scaling)
  LDA    vs MALLET  : 0.94x ST, 3.03x MT (topica 3.2x thread scaling)
== N~3500  (3500 docs, 2632 vocab) ==
  STM    vs R stm   : 15.9x single-thread
  keyATM vs R keyATM: 1.0x ST, 1.9x MT (topica 2.0x thread scaling)
  LDA    vs MALLET  : 0.92x ST, 3.84x MT (topica 4.2x thread scaling)
== N~5000  (5000 docs, 2632 vocab) ==
  STM    vs R stm   : 14.3x single-thread
  keyATM vs R keyATM: 0.9x ST, 2.0x MT (topica 2.2x thread scaling)
  LDA    vs MALLET  : 0.91x ST, 4.09x MT (topica 4.5x thread scaling)
Thread scaling (topica single / multi), by corpus size:
|   docs | LDA ST/MT | keyATM ST/MT |
|--------|-----------|--------------|
|   2000 |      3.2x |         2.2x |
|   3500 |      4.2x |         2.0x |
|   5000 |      4.5x |         2.2x |
Wrote /work/users/n/c/ncaren/topica/benchmarks/speed_vs_size.json
```

### §6 bench_scaling — OK

```
K-scaling sweep: N=5000, vocab=1000, iters=10, K in [20, 50, 100, 200]
  K=  20 laplace_keep    time=   0.64s rss=    110MB
  K=  50 laplace_keep    time=   1.82s rss=    435MB
  K= 100 laplace_keep    time=   4.60s rss=    842MB
  K= 200 laplace_keep    time=  20.57s rss=   2456MB
  K=  20 laplace_nokeep  time=   0.61s rss=    111MB
  K=  50 laplace_nokeep  time=   1.65s rss=    327MB
  K= 100 laplace_nokeep  time=   4.21s rss=    267MB
  K= 200 laplace_nokeep  time=  18.75s rss=    295MB
  K=  20 diagonal_keep   time=   0.57s rss=    123MB
  K=  50 diagonal_keep   time=   1.70s rss=    369MB
  K= 100 diagonal_keep   time=   4.00s rss=    827MB
  K= 200 diagonal_keep   time=  12.73s rss=   2519MB
Results written to /work/users/n/c/ncaren/topica/benchmarks/scaling_results.json
Figure written to /work/users/n/c/ncaren/topica/benchmarks/../paper/fig_scaling.pdf
```

### §6 k_crossover — OK

```
poliblog5k subsampled to 2000 docs (2632 vocab); iters=500; K sweep=[20, 50, 100, 200, 400]
  [k-crossover] K=20 ...
    topica 14.16s  tomotopy 9.71s  (tomotopy 1.46x)
  [k-crossover] K=50 ...
    topica 16.04s  tomotopy 13.15s  (tomotopy 1.22x)
  [k-crossover] K=100 ...
    topica 18.17s  tomotopy 16.42s  (tomotopy 1.11x)
  [k-crossover] K=200 ...
    topica 22.24s  tomotopy 24.52s  (topica 1.10x)
  [k-crossover] K=400 ...
    topica 29.89s  tomotopy 42.93s  (topica 1.44x)
| K | topica | tomotopy | faster |
|---|---|---|---|
| 20 | 14.2s | 9.7s | tomotopy 1.46× |
| 50 | 16.0s | 13.2s | tomotopy 1.22× |
| 100 | 18.2s | 16.4s | tomotopy 1.11× |
| 200 | 22.2s | 24.5s | **topica 1.10×** |
| 400 | 29.9s | 42.9s | **topica 1.44×** |
wrote /work/users/n/c/ncaren/topica/benchmarks/k_crossover_results.json and /work/users/n/c/ncaren/topica/benchmarks/k_crossover.md
```
