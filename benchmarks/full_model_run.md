| model | reference | K | threads | accuracy | topica_s | ref_s | speedup | topica_MB / ref_MB | note |
|---|---|---|---|---|---|---|---|---|---|
| LDA | tomotopy | 50 | 1 | 0.686 | 75.79 | 70.92 | 0.94x | 204.6 / 447.4 |  |
| LDA | MALLET | 50 | 1 | 0.757 | 75.79 | 86.27 | 1.14x | 204.6 / 0.0 |  |
| LDA | tomotopy | 50 | 8 | 0.669 | 19.34 | 18.53 | 0.96x | 49.8 / 0.0 |  |
| DMR | tomotopy | 10 | 1 | 0.862 | 46.88 | 34.07 | 0.73x | 10.0 / 3.4 | covariate prior; at/above ref seed ceiling (#564) |
| DMR | tomotopy | 25 | 1 | 0.759 | 56.68 | 45.21 | 0.8x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| GDMR | tomotopy | 10 | 1 | 0.864 | 48.02 | 28.43 | 0.59x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| GDMR | tomotopy | 25 | 1 | 0.736 | 56.99 | 37.04 | 0.65x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| SupervisedLDA (var) | tomotopy | 10 | 1 | 0.86 | 50.32 | 65.61 | 1.3x | 0.0 / 0.0 | variational EM (Blei-McAuliffe original) |
| SupervisedLDA (var) | tomotopy | 25 | 1 | 0.764 | 94.33 | 86.99 | 0.92x | 0.0 / 0.0 | variational EM (Blei-McAuliffe original) |
| SupervisedLDA (gibbs) | tomotopy | 10 | 1 | 0.821 | 104.4 | 65.6 | 0.63x | 0.0 / 0.0 | collapsed Gibbs, matches tomotopy |
| SupervisedLDA (gibbs) | tomotopy | 25 | 1 | 0.764 | 224.36 | 85.91 | 0.38x | 0.0 / 0.0 | collapsed Gibbs, matches tomotopy |
| PA | tomotopy | 50 | 1 | 0.689 | 662.58 | 385.23 | 0.58x | 0.0 / 0.0 | 3 super over 50 sub; sub compared |
| DTM | tomotopy | 10 | 1 | 0.61 | 19.83 | 96.45 | 4.86x | 0.0 / 0.0 | 4 slices; final slice compared |
| DTM | tomotopy | 25 | 1 | 0.518 | 49.82 | 102.64 | 2.06x | 11.1 / 0.0 | 4 slices; final slice compared |
| LabeledLDA | tomotopy | 20 | 1 | 1.0 | 53.71 | 35.32 | 0.66x | 78.2 / 5.8 | one topic per newsgroup (20NG labels) |
| STM | R stm | 10 | 1 | 0.946 | 7.0 | 20.74 | 2.96x | 191.3 / 7.7 | prevalence ~rating+s(day); Spectral |
| STM | R stm | 25 | 1 | 0.948 | 15.58 | 44.31 | 2.84x | 110.5 / 0.0 | prevalence ~rating+s(day); Spectral |
| STM (content/SAGE) | R stm | 10 | 1 | 0.938 | 3.6 | 88.04 | 24.43x | 0.0 / 6.0 | poliblog content=~rating; marginal topic-word |
| STM (content/SAGE) | R stm | 25 | 1 | 0.891 | 8.6 | 97.65 | 11.35x | 53.1 / 0.0 | poliblog content=~rating; marginal topic-word |
| KeyATM | R keyATM | 10 | 1 | 0.956 | 99.94 | 104.0 | 1.04x | 0.0 / 0.0 | 4 keyword topics scored (of 10); no-keyword topics excluded (unseeded, seed-unstable like LDA free topics) |
| NMF | sklearn | 50 | 1 | 0.959 | 19.34 | 43.19 | 2.23x | 0.0 / 5372.0 |  |
| LSA | sklearn | 50 | 1 | 0.918 | 0.62 | 2.23 | 3.58x | 0.0 / 0.0 |  |
| BERTopic | bertopic | 5 | 1 | 0.395 | 2.79 | 17.25 | 6.18x | 0.0 / 0.0 | cross-NMI of doc assignments (20-News, UMAP+HDBSCAN) |
| Top2Vec | bertopic | 5 | 1 | 0.389 | 2.87 | 6.18 | 2.15x | 0.0 / 0.0 | cross-NMI of doc assignments (20-News, vs BERTopic) |
| SemanticSignalSeparation | turftopic | 10 | 1 | 0.312 | 0.17 | 2.43 | 13.99x | 0.0 / 0.0 | NOT algorithm speed: ref time/RSS is turftopic loading MiniLM + re-encoding the vocab (topica gets those embeddings from the caller). ICA also non-convergent here => cross-NMI is noise. Fair algorithm speed + fidelity are in s3_planted. |
| SemanticSignalSeparation (planted) | turftopic | 10 | 1 | 0.996 | 0.02 | 0.03 | 1.31x | 0.0 / 0.0 | fidelity = cross-NMI of dominant-axis doc assignments on converging planted axes |
| FASTopic | fastopic | 10 | 1 | 0.644 | 16.82 | 16.57 | 0.99x | 0.0 / 0.0 | accuracy = cross-NMI of doc assignments |
| ProdLDA | pytorch-avitm | 10 | 1 | 0.472 | 23.11 | 8.57 | 0.37x | 0.0 / 0.0 | accuracy = cross-NMI of doc assignments (seed 0) |
| OnlineLDA | gensim | 50 | 1 | 0.436 | 39.19 | 50.95 | 1.3x | 0.0 / 0.0 |  |
| CTM | tomotopy | 50 | 1 | 0.466 | 82.01 | 5764.49 | 70.29x | 0.0 / 0.0 | topica variational vs tomotopy Gibbs CTM (same model, diff. inference) |
| HDP | tomotopy | 0 | 1 | 0.434 | 9.37 | 61.05 | 6.52x | 0.0 / 0.0 | nonparametric; K discovered both sides on a 2000-doc subset; cross-NMI of doc assignments (topic-word cosine not comparable). topica estimates the DP concentrations by default (#617); tomotopy finds fewer topics via a simplified new-table weight, so the two clusterings agree only partially. |
| HLDA | tomotopy | 0 | 1 | 0.213 | 24.52 | 0.82 | 0.03x | 0.0 / 0.0 | depth 3 tree on 2000-doc subset; cross-NMI of doc assignments (discovered-K, cosine not comparable). At the sharp default beta=0.01 topica fits a far finer, higher-posterior tree than tomotopy (~100x more nodes), so it is slower per fit; raise beta for a compact reference-scale tree (#615). num_threads speeds the per-fit work (see the threaded row). |
| HLDA | tomotopy | 0 | 8 | 0.213 | 33.75 | 0.82 | 0.02x | 0.0 / 0.0 | depth 3 tree on 2000-doc subset; cross-NMI of doc assignments (discovered-K, cosine not comparable). At the sharp default beta=0.01 topica fits a far finer, higher-posterior tree than tomotopy (~100x more nodes), so it is slower per fit; raise beta for a compact reference-scale tree (#615). num_threads speeds the per-fit work (see the threaded row). |
| PT | tomotopy | 50 | 1 | 0.529 | 1860.55 | 103.0 | 0.06x | 0.0 / 0.0 | pseudo-document short-text LDA (p=50) |
| AnchorLDA | anchor-topic | 50 | 1 | 0.988 | 19.35 | 16.67 | 0.86x | 2720.7 / 6252.6 | Arora RecoverL2 matched (recover='l2', temper=1.0). The two libs use different greedy anchor selectors, so end-to-end cosine is below the recovery-given-anchors parity (=1.0, parity/anchor_compare.py). topica also pays a heavier O(V^2) Q-build, so it is slower here (#622). |
