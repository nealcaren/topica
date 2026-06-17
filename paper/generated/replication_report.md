# topica paper: reproduction report

- Date: 2026-06-17 — machine-independent (§5 parity + §7), macOS 15 / Apple M3 Max, topica 0.23.1
- topica: 0.23.1
- Machine: macOS-26.5.1-arm64-arm-64bit-Mach-O, 14 cores (arm)
- Toolchains: Rscript=yes, R:stm=yes, R:keyATM=yes, java=yes, mallet=yes

> Section 6 (performance) is hardware-dependent; absolute timings reflect the machine above. Relative speedups are the portable claim.

## Step status

| Step | Section | Status | Time |
|---|---|---|---|
| worked_example | 7 | OK | 20s |
| stm_prevalence | 5 | OK | 433s |
| stm_content | 5 | OK | 2s |
| keyatm | 5 | OK | 113s |
| sts | 5 | OK | 327s |
| mallet_lda | 5 | OK | 2s |

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
    "topica_time": 1.7322174169821665,
    "ref_time": 21.452,
    "topica_rss_mb": 299.28125,
    "ref_rss_mb": 1476.953125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 22.513707582984352,
    "ref_time": 24.391,
    "topica_rss_mb": 104.109375,
    "ref_rss_mb": 421.109375
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 17.15082637500018,
    "ref_time": 23.003,
    "topica_rss_mb": 116.609375,
    "ref_rss_mb": 375.6875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 11.681865041988203,
    "ref_time": 23.014,
    "topica_rss_mb": 117.859375,
    "ref_rss_mb": 398.75
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 11.057358999998542,
    "ref_time": 23.342,
    "topica_rss_mb": 115.921875,
    "ref_rss_mb": 413.46875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 17.876091749989428,
    "ref_time": 18.918263915984426,
    "topica_rss_mb": 104.484375,
    "ref_rss_mb": 94.265625
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 11.125859375024447,
    "ref_time": 13.594176374987,
    "topica_rss_mb": 112.234375,
    "ref_rss_mb": 102.671875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 8.385870208003325,
    "ref_time": 8.084937165986048,
    "topica_rss_mb": 113.109375,
    "ref_rss_mb": 102.125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 5.723200000007637,
    "ref_time": 6.250574208010221,
    "topica_rss_mb": 108.0,
    "ref_rss_mb": 105.109375
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 2.7588249999971595,
    "ref_time": 29.493,
    "topica_rss_mb": 390.046875,
    "ref_rss_mb": 1551.859375
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 41.472926499991445,
    "ref_time": 42.082,
    "topica_rss_mb": 145.859375,
    "ref_rss_mb": 585.359375
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 30.621552583004814,
    "ref_time": 43.189,
    "topica_rss_mb": 172.328125,
    "ref_rss_mb": 543.046875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 25.898009625001578,
    "ref_time": 42.986,
    "topica_rss_mb": 183.234375,
    "ref_rss_mb": 544.03125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 21.606027624977287,
    "ref_time": 45.562,
    "topica_rss_mb": 175.109375,
    "ref_rss_mb": 571.453125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 34.070702291006455,
    "ref_time": 36.27563445799751,
    "topica_rss_mb": 146.609375,
    "ref_rss_mb": 97.796875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 22.886093165987404,
    "ref_time": 26.90651495900238,
    "topica_rss_mb": 160.609375,
    "ref_rss_mb": 103.125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 14.397758000006434,
    "ref_time": 15.45437354198657,
    "topica_rss_mb": 160.265625,
    "ref_rss_mb": 102.578125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 9.673319875000743,
    "ref_time": 9.94051041698549,
    "topica_rss_mb": 160.90625,
    "ref_rss_mb": 158.78125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 3.797926583007211,
    "ref_time": 37.684,
    "topica_rss_mb": 481.96875,
    "ref_rss_mb": 1616.203125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 59.32411062499159,
    "ref_time": 66.397,
    "topica_rss_mb": 189.421875,
    "ref_rss_mb": 641.890625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 43.984526209009346,
    "ref_time": 63.107,
    "topica_rss_mb": 225.21875,
    "ref_rss_mb": 645.234375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 34.38470445902203,
    "ref_time": 63.092,
    "topica_rss_mb": 246.4375,
    "ref_rss_mb": 628.734375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 27.370394833007595,
    "ref_time": 60.918,
    "topica_rss_mb": 242.09375,
    "ref_rss_mb": 625.0625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 50.42945162500837,
    "ref_time": 51.369629041000735,
    "topica_rss_mb": 202.1875,
    "ref_rss_mb": 101.171875
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 29.983983541984344,
    "ref_time": 35.693671334011015,
    "topica_rss_mb": 214.3125,
    "ref_rss_mb": 105.328125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 20.624890999984927,
    "ref_time": 19.69585795799503,
    "topica_rss_mb": 214.609375,
    "ref_rss_mb": 104.625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 14.239972958981525,
    "ref_time": 20.023035541002173,
    "topica_rss_mb": 215.734375,
    "ref_rss_mb": 159.328125
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
      "r": 21.881,
      "tt1": 1.8066414579807315
    },
    "keyatm": {
      "r": 25.065,
      "tt1": 23.117957000009483,
      "ttN": 10.940011042024707
    },
    "lda": {
      "r": 19.874371124984464,
      "tt1": 18.695788667013403,
      "ttN": 4.860455583984731
    }
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "stm": {
      "r": 30.433,
      "tt1": 2.7450121250003576
    },
    "keyatm": {
      "r": 43.204,
      "tt1": 41.88521199999377,
      "ttN": 18.318136124988087
    },
    "lda": {
      "r": 36.16577149997465,
      "tt1": 35.14725258297403,
      "ttN": 8.058429917000467
    }
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "stm": {
      "r": 37.507,
      "tt1": 3.6756450830143876
    },
    "keyatm": {
      "r": 62.405,
      "tt1": 60.11874970799545,
      "ttN": 27.492519000021275
    },
    "lda": {
      "r": 52.55875412499881,
      "tt1": 50.51132254197728,
      "ttN": 11.530239832994994
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
        1.3647,
        3.9059,
        13.794,
        91.6225
      ],
      "rss": [
        230.046875,
        618.4375,
        1415.34375,
        3987.40625
      ]
    },
    "laplace_nokeep": {
      "time": [
        0.7007,
        2.409,
        8.6724,
        58.8754
      ],
      "rss": [
        187.390625,
        362.78125,
        380.546875,
        411.203125
      ]
    },
    "diagonal_keep": {
      "time": [
        0.6035,
        1.6623,
        3.8724,
        12.8205
      ],
      "rss": [
        230.125,
        619.375,
        1053.1875,
        3735.40625
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
      "topica": 10.095441290992312,
      "tomotopy": 6.722977583005559,
      "faster": "tomotopy 1.50x"
    },
    {
      "K": 50,
      "topica": 11.429730625008233,
      "tomotopy": 9.362867082993034,
      "faster": "tomotopy 1.22x"
    },
    {
      "K": 100,
      "topica": 12.699719917000039,
      "tomotopy": 14.899222417006968,
      "faster": "topica 1.17x"
    },
    {
      "K": 200,
      "topica": 16.163348624977516,
      "tomotopy": 26.081858833000297,
      "faster": "topica 1.61x"
    },
    {
      "K": 400,
      "topica": 22.105549833009718,
      "tomotopy": 47.3077949580038,
      "faster": "topica 2.14x"
    }
  ]
}
```

### matrix_results.json

```json
{
  "matrix": [
    {
      "model": "LDA",
      "topica_time": 19.57133474999864,
      "tomo_time": 12.444013542000903,
      "topica_rss_mb": 145.3125,
      "tomo_rss_mb": 107.140625,
      "note": ""
    },
    {
      "model": "CTM",
      "topica_time": 33.092676542000845,
      "tomo_time": 116.93112266701064,
      "topica_rss_mb": 389.828125,
      "tomo_rss_mb": 110.875,
      "note": "topica: Laplace/STM E-step; tomotopy: mean-field CTM"
    },
    {
      "model": "DMR",
      "topica_time": 20.35676787501143,
      "tomo_time": 16.326084707994596,
      "topica_rss_mb": 146.859375,
      "tomo_rss_mb": 108.625,
      "note": "topica: numeric feature matrix; tomotopy: string metadata"
    },
    {
      "model": "HDP",
      "topica_time": 6.617218499988667,
      "tomo_time": 12.860053457989125,
      "topica_rss_mb": 130.328125,
      "tomo_rss_mb": 107.640625,
      "note": "K inferred: topica=3, tomotopy=2"
    },
    {
      "model": "PA",
      "topica_time": 205.4903187500022,
      "tomo_time": 89.29648087501118,
      "topica_rss_mb": 148.046875,
      "tomo_rss_mb": 112.015625,
      "note": "k1(super)=10, k2(sub)=20"
    },
    {
      "model": "PT",
      "topica_time": 76.82579324999824,
      "tomo_time": 15.219984041992575,
      "topica_rss_mb": 141.71875,
      "tomo_rss_mb": 108.015625,
      "note": "num_pseudo=100"
    },
    {
      "model": "SLDA",
      "topica_time": 681.5377478330047,
      "tomo_time": 26.43244941699959,
      "topica_rss_mb": 226.1875,
      "tomo_rss_mb": 108.78125,
      "note": "topica: variational EM; tomotopy: Gibbs"
    },
    {
      "model": "LabeledLDA",
      "topica_time": 5.858298708000802,
      "tomo_time": 3.3671141250088112,
      "topica_rss_mb": 131.0625,
      "tomo_rss_mb": 108.0625,
      "note": "2 labels (Liberal/Conservative); K fixed by label set"
    }
  ]
}
```

## Step output (tails)

### §7 worked_example — OK

```
========================================================================
Spanning comparison: different model families, one scoring loop
========================================================================
  fit LDA        K=15   11.0s
  fit CTM        K=15    1.0s
  fit STM        K=15    0.9s
  fit BERTopic   K=3     2.6s
  model       K   coherence(c_v)  exclusivity  diversity
  --------------------------------------------------------
  LDA        15        0.365        0.580     0.717
  CTM        15        0.351        0.446     0.579
  STM        15        0.350        0.435     0.565
  BERTopic   3         0.459        0.526     0.693
========================================================================
STM covariate-effect figure (fig_poliblog_effect.pdf)
========================================================================
  2000 docs, vocab 2612
  wrote /Users/nealcaren/Documents/GitHub/topica/paper/fig_poliblog_effect.pdf
  wrote /Users/nealcaren/Documents/GitHub/topica/paper/fig_poliblog_report.pdf
  Per-topic effect of conservative rating on prevalence:
    topic  7 mccain, sen, joe               coef=-0.0843  [-0.0943, -0.0744]
    topic  6 rove, tortur, administr        coef=-0.0652  [-0.0770, -0.0533]
    topic 13 hillari, clinton, deleg        coef=-0.0226  [-0.0339, -0.0113]
    topic 10 republican, parti, democrat    coef=-0.0115  [-0.0221, -0.0009]
    topic  9 iraqi, iraq, afghanistan       coef=-0.0083  [-0.0210, +0.0044]
    topic  0 poll, margin, percent          coef=-0.0062  [-0.0193, +0.0068]
    topic  5 billion, price, market         coef=-0.0028  [-0.0178, +0.0122]
    topic  2 school, abort, children        coef=-0.0020  [-0.0134, +0.0095]
    topic 14 romney, huckabe, reagan        coef=+0.0095  [+0.0028, +0.0161]
    topic 12 blagojevich, investig, governo coef=+0.0139  [+0.0015, +0.0264]
    topic 11 pentagon, russia, build        coef=+0.0170  [+0.0081, +0.0258]
    topic  8 ballot, immigr, franken        coef=+0.0179  [+0.0078, +0.0281]
    topic  4 media, stori, coverag          coef=+0.0431  [+0.0274, +0.0587]
    topic  3 wright, barack, ayer           coef=+0.0458  [+0.0344, +0.0571]
    topic  1 israel, isra, hama             coef=+0.0557  [+0.0437, +0.0677]
```

### §5 stm_prevalence — OK

```
corpus: 2000 docs, 2632 vocab, K=20
topica EM: converged after 40 iterations (em_tol=1e-5)
R-Spectral vs topica-Spectral cosine      : 0.974
R-Spectral vs R-Random (within-R basins)   : 0.611
R Random-vs-Random self-consistency        : 0.618
per-topic cosine: min 0.863  median 0.994  max 0.999
PASS — topica reproduces R's Spectral solution (cosine 0.974)
```

### §5 stm_content — OK

```
R levels ['de', 'en'] | topica groups ['de', 'en']
  de: R sep=0.032 tt sep=0.063 cosine=1.000
  en: R sep=0.037 tt sep=0.067 cosine=0.999
```

### §5 keyatm — OK

```
corpus: 2000 docs, 2632 vocab, 10 topics (4 keyword + 6 regular)
keyword topics  — R vs topica : 0.889   (R vs R: 0.902)
all topics      — R vs topica : 0.661   (R vs R: 0.704)
PASS — topica's keyword topics match R's as well as R matches itself across seeds (cosine 0.889)
```

### §5 sts — OK

```
  vocab regen=6366 fit=6365 shared=6365 overlap=1.0000
docs=13246  vocab=6365  K=5
topica-STS vs R-STS:  0.931  (chance 0.084, top-10 Jaccard 0.624)
topica-STM vs R-STM:  0.973  <- cross-implementation baseline
R-STS   vs R-STM:     0.959  <- same-ecosystem ceiling
topica-STS vs topica-STM: 0.935  <- STS extends STM
OK: topica STS recovers the published poliblog topics, as faithfully as STM does.
```

### §5 mallet_lda — OK

```
LDA        vs MALLET: Jaccard=1.000 cosine=1.000
LabeledLDA vs MALLET: cosine=1.000
DMR        vs MALLET: topic cosine=1.000 effect MALLET=+5.69 ours=+6.77
```
