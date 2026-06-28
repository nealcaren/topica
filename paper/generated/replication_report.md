# topica paper: reproduction report

- Date: 2026-06-28
- topica: 0.33.0
- Machine: macOS-26.5.1-arm64-arm-64bit-Mach-O, 14 cores (arm)
- Toolchains: Rscript=yes, R:stm=yes, R:keyATM=yes, java=yes, mallet=yes

> Section 6 (performance) is hardware-dependent; absolute timings reflect the machine above. Relative speedups are the portable claim.

## Step status

| Step | Section | Status | Time |
|---|---|---|---|
| worked_example | 7 | OK | 22s |
| stm_prevalence | 5 | OK | 449s |
| stm_content | 5 | OK | 2s |
| keyatm | 5 | OK | 126s |
| sts | 5 | OK | 516s |
| mallet_lda | 5 | OK | 2s |
| weights_ame | 5 | OK | 1s |
| validation_appendix | 5 | OK | 913s |

## Paper claims to update

| § | Claim | Fresh value in |
|---|---|---|
| 5 | STM content (SAGE) per-group cosine = 1.000 (de, en) | log: stm_content |
| 5 | STM prevalence Poliblog aligned cosine vs R | log: stm_prevalence |
| 5 | keyATM agreement with R keyATM | log: keyatm |
| 5 | STS benchmarks vs the sts package | log: sts |
| 5 | LDA vs Java MALLET (cosine / Jaccard) | log: mallet_lda |
| A | Side-by-side reference tables (Appendix A) | tex: generated/validation_appendix.tex + log: validation_appendix |
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
    "topica_time": 1.921349000011105,
    "ref_time": 20.717,
    "topica_rss_mb": 290.203125,
    "ref_rss_mb": 1453.078125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 22.53854133299319,
    "ref_time": 23.037,
    "topica_rss_mb": 104.484375,
    "ref_rss_mb": 414.15625
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 17.42804641701514,
    "ref_time": 23.447,
    "topica_rss_mb": 116.375,
    "ref_rss_mb": 408.203125
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 13.616511999978684,
    "ref_time": 23.514,
    "topica_rss_mb": 113.90625,
    "ref_rss_mb": 384.4375
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 10.9513549170224,
    "ref_time": 23.083,
    "topica_rss_mb": 115.8125,
    "ref_rss_mb": 396.140625
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 17.947071541973855,
    "ref_time": 19.069887332967483,
    "topica_rss_mb": 105.03125,
    "ref_rss_mb": 97.9375
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 11.314257959020324,
    "ref_time": 13.631589125026949,
    "topica_rss_mb": 111.15625,
    "ref_rss_mb": 99.421875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 7.671337541018147,
    "ref_time": 8.033928457996808,
    "topica_rss_mb": 112.40625,
    "ref_rss_mb": 102.875
  },
  {
    "n_docs": 2000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 5.71411016600905,
    "ref_time": 4.77178237499902,
    "topica_rss_mb": 111.953125,
    "ref_rss_mb": 105.1875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 2.8674497079919092,
    "ref_time": 29.482,
    "topica_rss_mb": 382.75,
    "ref_rss_mb": 1520.765625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 41.11587099998724,
    "ref_time": 43.292,
    "topica_rss_mb": 145.328125,
    "ref_rss_mb": 549.625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 32.25427745800698,
    "ref_time": 42.939,
    "topica_rss_mb": 171.140625,
    "ref_rss_mb": 536.640625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 24.82255283300765,
    "ref_time": 42.729,
    "topica_rss_mb": 175.5,
    "ref_rss_mb": 528.515625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 20.87278916698415,
    "ref_time": 42.082,
    "topica_rss_mb": 172.9375,
    "ref_rss_mb": 504.03125
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 34.029078709019814,
    "ref_time": 34.84574341698317,
    "topica_rss_mb": 150.15625,
    "ref_rss_mb": 96.546875
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 20.690476707997732,
    "ref_time": 25.35564358299598,
    "topica_rss_mb": 156.78125,
    "ref_rss_mb": 99.140625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 14.452566375024617,
    "ref_time": 15.098499416024424,
    "topica_rss_mb": 160.15625,
    "ref_rss_mb": 100.90625
  },
  {
    "n_docs": 3500,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 8.978368333016988,
    "ref_time": 8.348230708041228,
    "topica_rss_mb": 161.140625,
    "ref_rss_mb": 160.765625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "stm",
    "threads": 1,
    "topica_time": 3.807014291989617,
    "ref_time": 36.604,
    "topica_rss_mb": 476.375,
    "ref_rss_mb": 1613.328125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 1,
    "topica_time": 58.72746412502602,
    "ref_time": 61.45,
    "topica_rss_mb": 190.046875,
    "ref_rss_mb": 567.125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 2,
    "topica_time": 45.440015415952075,
    "ref_time": 60.835,
    "topica_rss_mb": 234.96875,
    "ref_rss_mb": 612.140625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 4,
    "topica_time": 35.95400870795129,
    "ref_time": 62.685,
    "topica_rss_mb": 237.765625,
    "ref_rss_mb": 589.671875
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "keyatm",
    "threads": 8,
    "topica_time": 30.366369833005592,
    "ref_time": 60.874,
    "topica_rss_mb": 226.15625,
    "ref_rss_mb": 652.234375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 1,
    "topica_time": 50.157592834031675,
    "ref_time": 52.266874750028364,
    "topica_rss_mb": 189.484375,
    "ref_rss_mb": 103.78125
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 2,
    "topica_time": 29.427663874987047,
    "ref_time": 35.8862297089654,
    "topica_rss_mb": 214.375,
    "ref_rss_mb": 108.390625
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 4,
    "topica_time": 19.722003375005443,
    "ref_time": 20.24327491701115,
    "topica_rss_mb": 210.21875,
    "ref_rss_mb": 105.375
  },
  {
    "n_docs": 5000,
    "vocab": 2632,
    "model": "lda",
    "threads": 8,
    "topica_time": 12.981792000005953,
    "ref_time": 21.236610917025246,
    "topica_rss_mb": 214.640625,
    "ref_rss_mb": 161.890625
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
        0.6927,
        2.1617,
        6.905,
        30.6251
      ],
      "rss": [
        230.21875,
        518.6875,
        1349.296875,
        3657.21875
      ]
    },
    "laplace_nokeep": {
      "time": [
        0.6574,
        2.1018,
        6.7972,
        29.6441
      ],
      "rss": [
        190.515625,
        374.890625,
        392.125,
        399.03125
      ]
    },
    "diagonal_keep": {
      "time": [
        0.6189,
        1.8386,
        5.0796,
        16.9477
      ],
      "rss": [
        214.625,
        622.484375,
        1357.6875,
        4242.40625
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
  fit LDA        K=15   12.2s
  fit CTM        K=15    1.3s
  fit STM        K=15    1.3s
  fit BERTopic   K=3     2.9s
  model       K   coherence(c_v)  exclusivity  diversity
  --------------------------------------------------------
  LDA        15        0.365        9.723     0.717
  CTM        15        0.365        9.476     0.589
  STM        15        0.366        9.474     0.581
  BERTopic   3         0.459        7.464     0.693
========================================================================
STM covariate-effect figure (fig_poliblog_effect.pdf)
========================================================================
  2000 docs, vocab 2612
  wrote /Users/nealcaren/Documents/GitHub/topica/paper/fig_poliblog_effect.pdf
  wrote /Users/nealcaren/Documents/GitHub/topica/paper/fig_poliblog_report.pdf
  Per-topic effect of conservative rating on prevalence:
    topic  7 sen, mccain, lieberman         coef=-0.0844  [-0.0941, -0.0747]
    topic  6 rove, tortur, cheney           coef=-0.0653  [-0.0771, -0.0534]
    topic 13 hillari, clinton, primari      coef=-0.0198  [-0.0309, -0.0088]
    topic 10 republican, parti, democrat    coef=-0.0119  [-0.0223, -0.0015]
    topic  0 poll, margin, percent          coef=-0.0071  [-0.0200, +0.0059]
    topic  9 iraqi, iraq, afghanistan       coef=-0.0057  [-0.0184, +0.0071]
    topic  5 billion, price, market         coef=-0.0030  [-0.0179, +0.0120]
    topic  2 school, abort, gay             coef=-0.0016  [-0.0130, +0.0097]
    topic 14 romney, huckabe, reagan        coef=+0.0086  [+0.0018, +0.0154]
    topic 12 blagojevich, investig, governo coef=+0.0143  [+0.0022, +0.0265]
    topic 11 pentagon, russia, build        coef=+0.0147  [+0.0056, +0.0237]
    topic  8 ballot, immigr, franken        coef=+0.0190  [+0.0090, +0.0289]
    topic  4 media, matthew, stori          coef=+0.0428  [+0.0274, +0.0582]
    topic  3 wright, barack, obama          coef=+0.0448  [+0.0336, +0.0560]
    topic  1 isra, israel, hama             coef=+0.0546  [+0.0427, +0.0664]
```

### §5 stm_prevalence — OK

```
corpus: 2000 docs, 2632 vocab, K=20
topica EM: converged after 36 iterations (em_tol=1e-5)
R-Spectral vs topica-Spectral cosine      : 0.975
R-Spectral vs R-Random (within-R basins)   : 0.611
R Random-vs-Random self-consistency        : 0.618
per-topic cosine: min 0.858  median 0.994  max 0.999
PASS — topica reproduces R's Spectral solution (cosine 0.975)
```

### §5 stm_content — OK

```
R levels ['de', 'en'] | topica groups ['de', 'en']
  de: R sep=0.032 tt sep=0.081 cosine=1.000
  en: R sep=0.037 tt sep=0.091 cosine=0.999
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
topica-STS vs R-STS:  0.930  (chance 0.083, top-10 Jaccard 0.624)
topica-STM vs R-STM:  0.976  <- cross-implementation baseline
R-STS   vs R-STM:     0.959  <- same-ecosystem ceiling
topica-STS vs topica-STM: 0.934  <- STS extends STM
OK: topica STS recovers the published poliblog topics, as faithfully as STM does.
```

### §5 mallet_lda — OK

```
LDA        vs MALLET: Jaccard=1.000 cosine=1.000
LabeledLDA vs MALLET: cosine=1.000
DMR        vs MALLET: topic cosine=1.000 effect MALLET=+5.69 ours=+6.77
```

### §5 weights_ame — OK

```
quantity               topica      faSTM (R)     |diff|
b_int              0.30201725     0.30201725   4.44e-16
b_year             0.06020481     0.06020481   7.63e-17
b_party            0.12474212     0.12474212   4.44e-16
se_year            0.00212265     0.00212265   5.20e-18
se_party           0.00389391     0.00389391   3.90e-18
sec_year           0.00219116     0.00219116   5.20e-18
sec_party          0.00329665     0.00329665   1.30e-18
ame_year           0.06020481     0.06020481   5.55e-17
ame_year_se        0.00212265     0.00212265   5.64e-18
ame_party          0.12474212     0.12474212   4.44e-16
ame_party_se       0.00389391     0.00389391   3.90e-18
max |diff| = 4.44e-16  (tol 1e-08)
OK: topica's weighted estimate_effect and AME match faSTM's formulas.
```

### §5 validation_appendix — OK

```
/Users/nealcaren/Documents/GitHub/topica/.venv-dev/lib/python3.13/site-packages/tomotopy/models.py:637: RuntimeWarning: The training result may differ even with fixed seed if `workers` != 1.
  return self._train(iterations, workers, parallel, freeze_topics, callback_interval, callback)
  lda: ok
  nmf: ok
  lsa: ok
  stm: ok
  ctm: ok
  content: ok
/Users/nealcaren/Documents/GitHub/topica/.venv-dev/lib/python3.13/site-packages/tomotopy/models.py:637: RuntimeWarning: The training result may differ even with fixed seed if `workers` != 1.
  return self._train(iterations, workers, parallel, freeze_topics, callback_interval, callback)
  dmr: ok
  gdmr: ok
  keyatm: ok
/Users/nealcaren/Documents/GitHub/topica/.venv-dev/lib/python3.13/site-packages/tomotopy/models.py:637: RuntimeWarning: The training result may differ even with fixed seed if `workers` != 1.
  return self._train(iterations, workers, parallel, freeze_topics, callback_interval, callback)
  slda: ok
  labeledlda: ok
  hdp: ok
  pa: ok
  dtm: ok
  sts: ok
wrote /Users/nealcaren/Documents/GitHub/topica/paper/generated/validation_appendix.tex
```
