# Model implementation map

This page maps every shipped model to the files a change touches: its core
algorithm, its PyO3 binding, the shared machinery it builds on, the Cargo feature
it needs, and where its reference-parity tests live. Use it to find the surfaces
for a model without searching the tree.

It is **generated from the registry**, not hand-maintained here. The source of
truth is the `IMPL` map in
[`python/topica/registry.py`](https://github.com/nealcaren/topica/blob/main/python/topica/registry.py);
`scripts/gen_model_tables.py` renders it into the block below, and
`tests/test_registry.py` fails CI if the block is stale, if `IMPL` and the model
registry cover different models, or if any path or Cargo feature it names does
not exist. So this map cannot silently drift from the code.

To change a row, edit `registry.py` and run `python scripts/gen_model_tables.py`
(the `preflight` hook and CI reject a stale block); do not edit the generated
block directly.

Column meanings:

- **Source** — the core algorithm file. A Rust `src/*.rs` (or `topica-core/`)
  file, or a `python/topica/*.py` module for the pure-Python models.
- **Binding** — the PyO3 binding: `src/python/mod.rs`, an extracted
  `src/python/<model>.rs` module, or `—` for a pure-Python model.
- **Core / family** — the shared machinery the model builds on (the collapsed
  Gibbs core in `model.rs`/`sampler.rs`, the CTM/STM variational core in
  `topica-core`, the ProdLDA VAE, and so on).
- **Feature** — the Cargo feature required beyond the default build. `default`
  builds with a plain `cargo build`; `embeddings` is the clustering-model gate
  (`cargo test --features embeddings,umap,tsne` covers it).
- **Validation** — the reference-parity / gold artifacts for the model. Every
  registered model is additionally covered by the conformance suite
  (`tests/test_conformance.py`).

<!-- BEGIN MODEL MAP (generated from topica.registry IMPL; edit registry.py, not this block) -->

### General-purpose

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `LDA` | `src/model.rs` | `src/python/mod.rs` | SparseLDA collapsed Gibbs (model.rs, sampler.rs) | default | `parity/lda_gold.py`, `parity/mallet_parity.py` |
| `OnlineLDA` | `src/online_lda.rs` | `src/python/online_lda.rs` | online-VB SVI schedule (variational/svi.rs), Dirichlet mean-field E-step (optimize.rs digamma) | default | `parity/online_lda_gensim_compare.py` |
| `CTM` | `topica-core/src/ctm.rs` | `src/python/mod.rs` | CTM/STM variational core (topica-core) | default | `parity/ctm_gold.py`, `parity/ctm_r_compare.py` |
| `ProdLDA` | `src/prodlda.rs` | `src/python/neural.rs` | hand-coded batched VAE (prodlda.rs) | default | `parity/prodlda_gold.py`, `parity/prodlda_compare.py` |
| `HDP` | `src/hdp.rs` | `src/python/mod.rs` | collapsed Gibbs (model.rs, sampler.rs) | default | `parity/hdp_gold.py` |
| `NMF` | `src/nmf.rs` | `src/python/nmf_lsa.rs` | multiplicative-update matrix factorization | default | `parity/nmf_vs_sklearn.py` |
| `LSA` | `src/lsa.rs` | `src/python/nmf_lsa.rs` | truncated SVD (linalg) | default | `parity/lsa_vs_sklearn.py` |
| `AnchorLDA` | `python/topica/anchor.py` | — _(Python)_ | spectral anchor-word recovery (Python over Rust primitives) | default | `tests/test_anchor.py` |
| `TensorLDA` | `src/tlda.rs` | `src/python/tlda.rs` | method-of-moments cumulants (linalg, spectral) | default | `parity/tlda_gold.py`, `parity/tlda_compare.py` |
| `PolylingualLDA` | `src/pltm.rs` | `src/python/pltm.rs` | collapsed Gibbs (model.rs, sampler.rs) | default | `parity/pltm_compare.py` |
| `CorEx` | `src/cor_ex.rs` | `src/python/cor_ex.rs` | total-correlation info-theoretic optimizer | default | `parity/corex_gold.py` |
| `MGLDA` | `src/mg_lda.rs` | `src/python/mg_lda.rs` | two-grain collapsed Gibbs over sliding sentence windows | default | `parity/mglda_gold.py` |

### Covariates & structure

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `STM` | `src/sts.rs` | `src/python/mod.rs` | CTM/STM variational core (topica-core) | default | `parity/stm_gold.py`, `parity/stm_r_compare.py` |
| `STS` | `src/sts.rs` | `src/python/mod.rs` | CTM/STM variational core (topica-core) | default | `parity/sts_gold.py`, `parity/sts_r_compare.py` |
| `SAGE` | `src/sage.rs` | `src/python/mod.rs` | collapsed Gibbs + SAGE deviations | default | `parity/sage_gold.py` |
| `DMR` | `src/dmr.rs` | `src/python/mod.rs` | collapsed Gibbs + DMR prior (optimize.rs) | default | `parity/dmr_gold.py` |
| `GDMR` | `src/dmr.rs` | `src/python/mod.rs` | collapsed Gibbs + DMR prior (optimize.rs) | default | `parity/gdmr_gold.py`, `parity/test_gdmr_tomotopy.py` |
| `Scholar` | `src/scholar.rs` | `src/python/scholar.rs` | ProdLDA VAE + covariate prior (prodlda.rs) | default | `tests/test_scholar.py` |
| `NarrativeTM` | `python/topica/narrative.py` | — _(Python)_ | intra-document trajectory over Gibbs core (Python) | default | `tests/test_content_trajectory.py` |
| `RTM` | `src/rtm.rs` | `src/python/rtm.rs` | variational EM + link head (optimize.rs digamma) | default | `parity/rtm_compare.py`, `parity/rtm_reference.py`, `tests/test_rtm.py` |
| `FactorialLDA` | `src/factorial_lda.rs` | `src/python/factorial_lda.rs` | collapsed Gibbs over tuples + MCEM gradient ascent on log-linear priors | default | `parity/factorial_lda_compare.py`, `tests/test_factorial_lda.py` |
| `AuthorTopic` | `src/author_topic.rs` | `src/python/author_topic.rs` | collapsed Gibbs (author×topic + word×topic counts) | default | `parity/author_topic_gold.py` |

### Guided & supervised

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `KeyATM` | `src/keyatm.rs` | `src/python/mod.rs` | collapsed Gibbs + keyword index | default | `parity/keyatm_gold.py`, `parity/keyatm_r_compare.py` |
| `SeededLDA` | `src/seeded.rs` | `src/python/mod.rs` | collapsed Gibbs (model.rs, sampler.rs) | default | `parity/seededlda_gold.py` |
| `GuidedNMF` | `src/guided_nmf.rs` | `src/python/guided_nmf.rs` | supervised multiplicative-update matrix factorization | default | `parity/guidednmf_gold.py` |
| `LabeledLDA` | `src/labeled.rs` | `src/python/mod.rs` | collapsed Gibbs (model.rs, sampler.rs) | default | `parity/labeledlda_gold.py` |
| `SupervisedLDA` | `src/slda.rs` | `src/python/mod.rs` | variational EM + Gaussian response head | default | `parity/supervisedlda_gold.py` |
| `DiscLDA` | `src/disclda.rs` | `src/python/disclda.rs` | collapsed Gibbs + class transform | default | `parity/disclda_20ng.py`, `tests/test_disclda.py` |

### Short text

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `GSDMM` | `src/gsdmm.rs` | `src/python/mod.rs` | collapsed Gibbs mixture (one topic/doc) | default | `parity/gsdmm_gold.py` |
| `PT` | `src/pt.rs` | `src/python/mod.rs` | collapsed Gibbs over pseudo-documents | default | `parity/pt_gold.py` |
| `BTM` | `src/btm.rs` | `src/python/btm.rs` | collapsed Gibbs over biterms | default | `parity/btm_compare.py`, `tests/test_btm.py` |

### Dynamic & hierarchical

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `DTM` | `src/dtm.rs` | `src/python/mod.rs` | variational Kalman over time slices | default | `parity/dtm_gold.py` |
| `DETM` | `src/detm.rs` | `src/python/neural.rs` | embedding VAE + LSTM q(eta) (etm_vae.rs) | default | `parity/detm_gold.py` |
| `TopicsOverTime` | `src/topics_over_time.rs` | `src/python/topics_over_time.rs` | collapsed Gibbs (LDA + per-topic Beta time factor, method-of-moments psi) | default | `parity/tot_gold.py` |
| `HLDA` | `src/hlda.rs` | `src/python/hierarchical.rs` | nested-CRP collapsed Gibbs | default | `parity/hlda_gold.py` |
| `PA` | `src/pa.rs` | `src/python/hierarchical.rs` | collapsed Gibbs over a topic DAG | default | `parity/pa_gold.py` |

### Embedding-based

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `BERTopic` | `src/bertopic.rs` | `src/python/embedding_cluster.rs` | embedding clustering (cluster.rs, reduce.rs, represent.rs) | `embeddings` | `parity/bertopic_gold.py` |
| `Top2Vec` | `src/top2vec.rs` | `src/python/embedding_cluster.rs` | embedding clustering (cluster.rs, reduce.rs) | `embeddings` | `parity/top2vec_gold.py`, `parity/top2vec_compare.py` |
| `MechanisticLDA` | `python/topica/mtm.py` | — _(Python)_ | SAE featurization (Python) over the SparseLDA collapsed-Gibbs core | default | `tests/test_mtm.py` |
| `MechanisticBERTopic` | `python/topica/mtm.py` | — _(Python)_ | SAE featurization (Python) over the embedding-clustering core (cluster.rs, reduce.rs) | `embeddings` | `tests/test_mtm.py` |
| `SemanticSignalSeparation` | `src/semantic_signal_separation.rs` | `src/python/semantic_signal_separation.rs` | FastICA over document embeddings + vocabulary projection (reduce.rs) | `embeddings` | `parity/s3_compare.py`, `tests/test_semantic_signal_separation.py` |
| `ETM` | `src/etm.rs` | `src/python/neural.rs` | variational EM over word embeddings (ctm.rs) | default | `parity/etm_gold.py` |
| `GaussianLDA` | `src/gaussian_lda.rs` | `src/python/gaussian_lda.rs` | collapsed Gibbs (NIW-Gaussian topics, Student-t predictive, rank-1 Cholesky up/downdates) | default | `parity/gaussian_lda_gold.py` |
| `IdealPointTM` | `src/idealpoint.rs` | `src/python/idealpoint.rs` | variational EM + ideal-point head | default | `tests/test_idealpoint.py`, `tests/test_idealpoint_counts.py` |
| `IdealPointSentenceTM` | `src/sentence_ideal.rs` | `src/python/sentence_ideal.rs` | Gaussian-cluster EM over embeddings | default | `tests/test_sentence_ideal.py` |
| `FASTopic` | `src/fastopic.rs` | `src/python/mod.rs` | reverse-mode Sinkhorn optimal transport | default | `parity/fastopic_gold.py`, `parity/fastopic_compare.py` |
| `EmbeddingLDA` | `python/topica/embedding.py` | — _(Python)_ | seeded Gibbs + embedding NN expansion (Python) | default | `parity/embeddinglda_gold.py`, `tests/test_embedding_lda.py` |
| `CombinedTM` | `src/prodlda.rs` | `src/python/neural.rs` | contextualized ProdLDA VAE (prodlda.rs) | default | `parity/combinedtm_gold.py`, `parity/combinedtm_compare.py` |
| `ZeroShotTM` | `src/prodlda.rs` | `src/python/neural.rs` | contextualized ProdLDA VAE (prodlda.rs) | default | `parity/zeroshot_gold.py`, `parity/zeroshot_compare.py` |
| `InfoCTM` | `src/infoctm.rs` | `src/python/neural.rs` | two ProdLDA VAEs + TAMI alignment (prodlda.rs) | default | `parity/infoctm_gold.py`, `parity/infoctm_compare.py` |

### Ideal point

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `Wordfish` | `src/wordfish.rs` | `src/python/wordfish.rs` | Poisson-scaling EM | default | `parity/wordfish_r_compare.py`, `tests/test_wordfish.py` |
| `Wordshoal` | `src/wordshoal.rs` | `src/python/wordshoal.rs` | per-domain Wordfish + cross-domain linear factor EM | default | `parity/wordshoal_r_compare.py`, `tests/test_wordshoal.py` |
| `TBIP` | `src/tbip.rs` | `src/python/tbip.rs` | Poisson-factorization mean-field SVI | default | `parity/tbip_parity.py`, `tests/test_tbip.py` |
| `PartyEmbeddings` | `src/party_embeddings.rs` | `src/python/party_embeddings.rs` | PV-DM paragraph vectors (negative sampling) | default | `parity/party_embeddings_compare.py`, `tests/test_party_embeddings.py` |

### LLM-based

| Model | Source | Binding | Core / family | Feature | Validation |
|---|---|---|---|---|---|
| `TopicGPT` | `python/topica/llm.py` | — _(Python)_ | LLM prompting pipeline (Python) | default | `tests/test_topicgpt.py` |
<!-- END MODEL MAP -->
