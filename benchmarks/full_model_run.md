| model | reference | K | threads | accuracy | topica_s | ref_s | speedup | topica_MB / ref_MB | note |
|---|---|---|---|---|---|---|---|---|---|
| LDA | tomotopy | 50 | 1 | 0.686 | 81.16 | 72.43 | 0.89x | 667.8 / 447.0 |  |
| LDA | MALLET | 50 | 1 | 0.757 | 81.16 | 82.92 | 1.02x | 667.8 / 0.0 |  |
| LDA | tomotopy | 50 | 8 | 0.669 | 19.3 | 20.12 | 1.04x | 52.2 / 0.0 |  |
| DMR | tomotopy | 10 | 1 | 0.862 | 47.32 | 34.41 | 0.73x | 24.5 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| DMR | tomotopy | 25 | 1 | 0.759 | 56.06 | 46.14 | 0.82x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| GDMR | tomotopy | 10 | 1 | 0.864 | 47.68 | 27.16 | 0.57x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| GDMR | tomotopy | 25 | 1 | 0.736 | 58.51 | 37.44 | 0.64x | 0.0 / 0.0 | covariate prior; at/above ref seed ceiling (#564) |
| SupervisedLDA (var) | tomotopy | 10 | 1 | 0.86 | 50.8 | 64.16 | 1.26x | 0.0 / 0.0 | variational EM (Blei-McAuliffe original) |
| SupervisedLDA (var) | tomotopy | 25 | 1 | 0.764 | 96.69 | 83.55 | 0.86x | 88.2 / 0.0 | variational EM (Blei-McAuliffe original) |
| SupervisedLDA (gibbs) | tomotopy | 10 | 1 | 0.821 | 106.62 | 69.07 | 0.65x | 4.9 / 0.0 | collapsed Gibbs, matches tomotopy |
| SupervisedLDA (gibbs) | tomotopy | 25 | 1 | 0.764 | 231.71 | 96.0 | 0.41x | 0.0 / 0.0 | collapsed Gibbs, matches tomotopy |
| PA | tomotopy | 50 | 1 | 0.689 | 646.3 | 454.68 | 0.7x | 210.1 / 0.0 | 3 super over 50 sub; sub compared |
| DTM | tomotopy | 10 | 1 | 0.61 | 18.02 | 91.33 | 5.07x | 9.3 / 0.0 | 4 slices; final slice compared |
| DTM | tomotopy | 25 | 1 | 0.518 | 45.67 | 94.74 | 2.07x | 0.0 / 0.0 | 4 slices; final slice compared |
| LabeledLDA | tomotopy | 20 | 1 | 1.0 | 48.67 | 32.75 | 0.67x | 0.0 / 0.0 | one topic per newsgroup (20NG labels) |
| STM | R stm | 10 | 1 | 0.946 | 6.23 | 19.63 | 3.15x | 172.2 / 0.0 | prevalence ~rating+s(day); Spectral |
| STM | R stm | 25 | 1 | 0.948 | 13.79 | 42.95 | 3.12x | 130.8 / 0.0 | prevalence ~rating+s(day); Spectral |
| STM (content/SAGE) | R stm | 10 | 1 | 0.938 | 3.18 | 81.07 | 25.47x | 0.0 / 0.0 | poliblog content=~rating; marginal topic-word |
| STM (content/SAGE) | R stm | 25 | 1 | 0.891 | 7.67 | 91.23 | 11.89x | 14.1 / 0.0 | poliblog content=~rating; marginal topic-word |
| KeyATM | R keyATM | 10 | 1 | 0.956 | 87.67 | 98.25 | 1.12x | 10.9 / 0.0 | 4 keyword topics scored (of 10); no-keyword topics excluded (unseeded, seed-unstable like LDA free topics) |
| NMF | sklearn | 50 | 1 | 0.959 | 17.73 | 41.49 | 2.34x | 68.9 / 5162.2 |  |
| LSA | sklearn | 50 | 1 | 0.918 | 0.55 | 2.27 | 4.11x | 13.2 / 69.4 |  |
| BERTopic | bertopic | 5 | 1 | 0.395 | 9.59 | 16.41 | 1.71x | 464.2 / 0.0 | cross-NMI of doc assignments (20-News, UMAP+HDBSCAN) |
| Top2Vec | bertopic | 5 | 1 | 0.389 | 2.54 | 5.62 | 2.21x | 39.0 / 0.0 | cross-NMI of doc assignments (20-News, vs BERTopic) |
| SemanticSignalSeparation | turftopic | 10 | 1 | 0.312 | 0.16 | 2.62 | 16.48x | 0.4 / 0.0 | NOT algorithm speed: ref time/RSS is turftopic loading MiniLM + re-encoding the vocab (topica gets those embeddings from the caller). ICA also non-convergent here => cross-NMI is noise. Fair algorithm speed + fidelity are in s3_planted. |
| SemanticSignalSeparation (planted) | turftopic | 10 | 1 | 0.996 | 0.02 | 0.02 | 1.16x | 2.3 / 0.0 | fidelity = cross-NMI of dominant-axis doc assignments on converging planted axes |
| FASTopic | fastopic | 10 | 1 | 0.647 | 15.91 | 15.61 | 0.98x | 42.3 / 0.0 | accuracy = cross-NMI of doc assignments |
| ProdLDA | pytorch-avitm | 10 | 1 | 0.472 | 20.98 | 7.97 | 0.38x | 36.0 / 0.0 | accuracy = cross-NMI of doc assignments (seed 0) |
| OnlineLDA | gensim | 50 | 1 | 0.436 | 37.08 | 45.7 | 1.23x | 27.1 / 0.0 |  |
| CTM | tomotopy | 50 | 1 | 0.466 | 74.78 | 5221.28 | 69.82x | 887.7 / 0.0 | topica variational vs tomotopy Gibbs CTM (same model, diff. inference) |
| HDP | tomotopy | 0 | 1 | 0.434 | 10.64 | 60.95 | 5.73x | 520.0 / 832.7 | nonparametric; K discovered both sides on a 2000-doc subset; cross-NMI of doc assignments (topic-word cosine not comparable). topica estimates the DP concentrations by default (#617); tomotopy finds fewer topics via a simplified new-table weight, so the two clusterings agree only partially. |
| HLDA | tomotopy | 0 | 1 | 0.213 | 304.97 | 0.74 | 0.0x | 48.7 / 0.0 | depth 3 tree on 2000-doc subset; cross-NMI of doc assignments (discovered-K, cosine not comparable). At the sharp default beta=0.01 topica fits a far finer, higher-posterior tree than tomotopy (~100x more nodes), so it is slower per fit; raise beta for a compact reference-scale tree (#615). num_threads speeds the per-fit work (see the threaded row). |
| HLDA | tomotopy | 0 | 8 | 0.213 | 65.06 | 0.73 | 0.01x | 2.6 / 0.0 | depth 3 tree on 2000-doc subset; cross-NMI of doc assignments (discovered-K, cosine not comparable). At the sharp default beta=0.01 topica fits a far finer, higher-posterior tree than tomotopy (~100x more nodes), so it is slower per fit; raise beta for a compact reference-scale tree (#615). num_threads speeds the per-fit work (see the threaded row). |
