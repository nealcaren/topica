# Validated against reference (frozen parity gold)

21 models with an external cross-implementation reference; 0 validated by self-consistency (no reproducible external implementation). Agreement = topica-vs-reference topic-word cosine (or the metric named in each gold); ceiling = the reference's own seed-to-seed self cosine.

### Cross-implementation (external reference)

| Model | Reference | Corpus | docs | K | Agreement | Ref. ceiling |
|---|---|---|--:|--:|--:|--:|
| BERTopic (Grootendorst 2022): topica reimpl vs the bertopic package | bertopic 0.17.4 (UMAP+HDBSCAN) | synthetic planted clusters from parity/t | 320 |  | 1.0 | 0.4 |
| BTM (biterm topic model, Yan et al. 2013) | R BTM package (Wijffels, Apache-2.0) | planted biterm fixture (btm_compare) |  | 4 | 1.0 | 1.0 |
| CombinedTM (Bianchi, Terragni & Hovy 2021) | torch 2.13.0 (AVITM reference, parity/refs/avitm.py) | synthetic planted-block, 600 docs x 25 t | 600 | 5 | 0.985 | 0.98 |
| CTM (R stm with prevalence=NULL) | R 4.5.2 / stm 1.3.8 | poliblog (examples/poliblog.csv), fixed- | 2000 | 20 | 0.975 | 0.646 |
| DMR (Dirichlet-Multinomial Regression LDA) | Java MALLET /opt/homebrew/Cellar/mallet/202108/libexec (DMRDriver.java) | synthetic two-cluster fixture, 160 docs  | 160 | 2 | 1.0 | 1.0 |
| DTM (Blei & Lafferty 2006) | gensim 4.4.0 | synthetic smooth-drift, 4 time slices x  | 1200 | 2 | 1.0 | 0.9 |
| ETM (embedded topic model, Dieng et al. 2020) | embedded_topic_model 1.2.1 (amortized VAE) | 20NG 1200-doc subset (shared vocab + wor | 1200 | 20 | 0.275 |  |
| FASTopic (Wu et al. 2024) | fastopic 1.0.1 (all-MiniLM-L6-v2 embeddings) | 20-newsgroups 5-group subset (420-doc fi | 420 | 10 | 0.607 | 0.694 |
| GDMR (g-DMR, Lee & Song 2020) | tomotopy 0.14.0 | synthetic two-cluster, 300 docs x 10 tok | 300 | 2 | 1.0 |  |
| GSDMM (Yin & Wang 2014, short-text DMM) | rwalk/gsdmm MovieGroupProcess | 20NG 1500-doc subset | 1500 | 20 | 0.45 |  |
| InfoCTM (Wu et al. 2023, AAAI) | paper-derived torch InfoCTM (parity/refs) | matched bilingual planted-block |  | 5 | 1.0 | 0.8 |
| LabeledLDA (label-constrained LDA) | Java MALLET /opt/homebrew/Cellar/mallet/202108/libexec (LabeledLDADriver.java) | synthetic multi-label fixture, 200 docs, | 200 |  | 1.0 | 1.0 |
| LDA (collapsed Gibbs / SparseLDA) | Java MALLET /opt/homebrew/Cellar/mallet/202108/libexec | planted five-topic fixture, 250 docs x 1 | 250 | 5 | 1.0 | 1.0 |
| PolylingualLDA / PLTM (Mimno et al. 2009) | Java MALLET PolylingualTopicModel | planted trilingual parallel fixture (plt |  |  | 1.0 |  |
| RTM collapsed Gibbs (R lda rtm.em / rtm.collapsed.gibbs.sampler) | R lda 1.5.2 | planted network (90 docs, K=3, 479 links | 90 | 3 | 0.999 | 0.999 |
| SeededLDA (koheiw/seededlda) | R seededlda 1.4.4 | poliblog subsample (400 docs, seed 271) | 400 |  | 0.936 | 0.973 |
| STM | R 4.5.2 / stm 1.3.8 | poliblog (examples/poliblog.csv), fixed- | 2000 | 20 | 0.975 | 0.63 |
| STS reference profile (CRAN sts, kappaEstimation='adjusted') | R sts 1.4 | stm poliblog5k first 300 docs, prepDocum | 300 | 5 | 0.887 | 0.998 |
| Top2Vec (Angelov 2020) vs BERTopic (Grootendorst 2022) | bertopic 0.17.4 (UMAP+HDBSCAN) | synthetic planted clusters from parity/t | 320 |  | 1.0 | 0.4 |
| Wordfish (Slapin & Proksch 2008) | R quanteda.textmodels textmodel_wordfish | planted 1-D scaling fixture (wordfish_r_ |  |  | 1.0 | 0.993 |
| ZeroShotTM (Bianchi, Nozza & Hovy 2021) | torch 2.12.0 (AVITM reference, parity/refs/avitm.py) | synthetic planted-block, 600 docs x 25 t | 600 | 5 | 0.976 | 0.937 |

### Self-consistency (no reproducible external reference)

| Model | Reference | Corpus | docs | K | Agreement | Ref. ceiling |
|---|---|---|--:|--:|--:|--:|
