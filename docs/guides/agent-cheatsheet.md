# Agent cheat sheet

A one-screen reference for an LLM agent or a first-time user who has just run
`import topica` and needs the canonical patterns without reading the whole
documentation set. Everything below is also available at the REPL:

```python
import topica

topica.guide()            # the one-screen essentials, printed
topica.guide("STM")       # one model: purpose, signatures, first calls
topica.guide(full=True)   # every validated model, grouped
```

`topica.guide()` renders live from the model registry and each model's real
signature, so it always matches the installed build. It ships in the wheel, so it
is available from a plain `pip install topica` with no repository checkout. The
page below is generated from the same builder; do not edit it by hand. Edit
`python/topica/_guide.py` and run `scripts/gen_guide.py`.

For task-shaped starting points you can transplant onto your own data (compare
prevalence across groups, compare how groups word a topic, track topics over
time), see the [task recipes](https://github.com/nealcaren/topica/tree/main/examples/recipes)
in `examples/recipes/`.

<!-- BEGIN GUIDE (generated from topica.guide via scripts/gen_guide.py; edit _guide.py, not this block) -->

## The one-screen guide

```text
topica - quick guide for agents and first-time users

THE WORKFLOW (a full LDA analysis in five lines)
    import topica
    corpus = topica.from_dataframe(df, text_col="text")      # build + prune vocab
    res    = topica.select.search_k(corpus, ks=[10, 20, 30])  # choose K; res.best_k()
    model  = topica.LDA(num_topics=res.best_k()).fit(corpus)
    topica.inspect.topic_table(model)                         # labelled topics
    # with metadata:
    topica.effects.estimate_effect(model, X=X, corpus=corpus) # covariate effects + CIs

EVERY FITTED MODEL EXPOSES THE SAME SURFACE
    model.topic_word                           (K, V) topic-word matrix; rows sum to 1 (generative models)
    model.doc_topic                            (D, K) document-topic matrix; rows sum to 1
    model.vocabulary                           V words, aligned to topic_word columns
    model.top_words(n)                         list[list[str]] top n per topic; weights=True for (word, prob)
    model.num_topics / .doc_names / .settings  K, row labels, and the fit config as a dict
    model.save(path) / Model.load(path)        round-trip a fitted model to disk

PICK A MODEL BY GOAL
  Common openings:
    Explore themes with no prior structure
        -> LDA (or NMF): `search_k()`, `topic_table()`
    Relate topics to metadata (author, date, party)
        -> STM (or DMR): `estimate_effect()`, `one_hot()`, `spline()`
    Measure concepts you can name in advance
        -> KeyATM (or SeededLDA): `KeyATM(keywords=…)`, `.keyword_rate`
    Very short documents: tweets, headlines, survey answers
        -> GSDMM (or PT): `fit()`
    Cluster by meaning using embeddings
        -> BERTopic (or ETM): `fit(docs, doc_embeddings=…)`
  Specialized (start here when your design calls for it):
    Topics shift over time slices
        -> DTM (or DETM): `fit(docs, times=…)`
    Documents linked in a network (citations, replies)
        -> RTM: `fit(docs, links=…)`
    Documents in more than one language
        -> PolylingualLDA: `fit(doc_tuples)`
    Place authors or actors on an ideological scale
        -> Wordfish (or TBIP): `fit(docs)`
    How tone or sentiment varies with metadata
        -> STS: `estimate_effect()`

HELPER NAMESPACES (topica.<stage>.*)
    select                             choosing K (search_k, select_model)
    inspect                            reading topics (label_topics, topic_table, frex, find_thoughts)
    evaluate                           validation (coherence, exclusivity, topic_stability, perplexity)
    effects                            covariate effects (estimate_effect, predicted_prevalence)
    design                             design matrices (one_hot, design_matrix, spline)
    data                               corpus + bundled datasets (from_dataframe, tokenize, datasets)
    compare / provenance / embeddings  two-fit drift, analysis manifest, embedding I/O
    -> guide("<name>") prints any helper's signature (e.g. guide("estimate_effect"))

GO DEEPER
    topica.guide("STM")            one model: signatures + first calls
    topica.guide("topic_stability") one helper: signature + purpose
    topica.guide(full=True)        every model, grouped
    help(topica.STM)               full constructor / fit docstrings
    topica.list_models()           the roster (list_models(group=...) to filter)
    docs: https://nealcaren.github.io/topica/
```

## Every validated model

```text
topica model reference (validated roster)

### General-purpose

LDA(num_topics, *, alpha_sum=None, beta=0.01, optimize_interval=50, burn_in=200, seed=13, num_threads=1, sampler='sparse', mh_steps=2, use_symmetric_alpha=False, init='random')
    Classic latent Dirichlet allocation via a fast SparseLDA collapsed-Gibbs sampler.
    .fit(data, *, iters=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, num_threads=None, turbo_merge_every=1)

OnlineLDA(num_topics, *, alpha_sum=None, beta=0.01, tau=1.0, kappa=0.7, batch_size=256, inner_iters=100, mean_change_tol=0.001, total_docs=None, seed=13)
    Online (streaming) variational-Bayes LDA (Hoffman et al. 2010): minibatch stochastic VB with a decaying learning rate and a streaming partial_fit; the gensim LdaModel analogue for very large or streaming corpora.
    .fit(data, *, iters=100, convergence_tol=0.0, progress=None)

CTM(num_topics, *, sigma_shrink=0.0, seed=13, init='spectral', variational='laplace')
    Correlated topic model: a logistic-normal prior that lets topics co-occur.
    .fit(data, *, iters=500, convergence_tol=1e-05, inference='batch', batch_size=256, tau=64.0, kappa=0.7, beta_init=None, em_tol=None, keep_eta_cov=True, num_threads=None, spectral_projection_threshold=Ellipsis, progress=None)

ProdLDA(num_topics, *, alpha=1.0, hidden_size=100, dropout=0.2, batch_size=200, lr=0.002, convergence_tol=0.0, seed=13, prior=Ellipsis, contrastive=False, contrastive_weight=0.5, contrastive_temp=0.5, em_tol=None)
    Product-of-experts LDA (AVITM) for sharper, more coherent topics; hand-coded VAE.
    .fit(data, *, iters=None, convergence_tol=None, progress=None)

HDP(*, alpha=0.1, gamma=0.1, beta=0.01, seed=13, resample_conc=True, concentration_max=1000000.0, eta=None)
    Hierarchical Dirichlet process: infers the number of topics from the data.
    .fit(data, *, iters=150, progress_interval=0, keep_theta_draws=True, num_theta_draws=25, report_interval=None)

NMF(num_topics, *, beta_loss='frobenius', init='nndsvd', weighting='tfidf', convergence_tol=0.0001, seed=13)
    Non-negative matrix factorization of the document-term matrix via multiplicative updates.
    .fit(data, *, iters=None, convergence_tol=None, num_threads=None)

LSA(num_topics, *, weighting='tfidf', seed=13)
    Latent semantic analysis: a truncated SVD of the weighted document-term matrix.
    .fit(data, *, num_threads=None)

AnchorLDA(num_topics: 'int', *, recover: 'str' = 'kl', min_count: 'int' = 5, seed: 'int' = 13, eta: 'float' = 1.0, convergence_tol: 'float' = 1e-05, frex_w: 'float' = 0.5, frequency_temper: 'float' = 0.5, anchor_min_doc_freq: 'float' = 0.01)
    Anchor-words spectral recovery (Arora et al. 2013): deterministic, Gibbs-free topics from the word co-occurrence matrix.
    .fit(data, *, iters=None, min_count=None)

PolylingualLDA(num_topics, *, alpha=None, beta=0.01, iters=1000, optimize_alpha=True, optimize_interval=10, optimize_burn_in=200, seed=13)
    Polylingual topic model (Mimno et al. 2009): aligned topics across languages from document tuples that share one topic distribution.
    .fit(data, *, iters=None, progress=None)

CorEx(num_topics=2, *, anchor_words=None, anchor_strength=1.0, count='binarize', convergence_tol=1e-05, seed_match='fixed', case_insensitive=False, seed=13)
    Correlation Explanation: information-theoretic topic model that maximizes total correlation; supports anchor words.
    .fit(data, *, iters=None, convergence_tol=None, num_threads=None)

MGLDA(num_global_topics, num_local_topics, *, window=3, alpha_global=0.1, alpha_local=0.1, alpha_mix_global=0.1, alpha_mix_local=0.1, beta_global=0.01, beta_local=0.01, gamma=0.1, seed=13)
    Multi-Grain LDA: global (document-level) + local (sliding-window aspect) topics with a per-token grain switch. For reviews / aspect extraction.
    .fit(data, *, iters=1000, progress=None)

TopicalNGrams(num_topics, *, alpha_sum=50.0, beta=0.01, gamma=0.01, delta1=1.0, delta2=1.0, min_count=1, seed=13)
    Topical N-Grams (Wang, McCallum & Wei 2007): an LDA extension that jointly discovers topics and topic-specific multiword phrases. A per-token bigram-status indicator, sampled with the topic, decides whether a token continues a phrase from the previous word given its topic, so phrase structure is learned during fitting rather than fixed beforehand. Exposes top_phrases alongside top_words.
    .fit(data, *, iters=1000, progress=None)

### Covariates & structure

STM(num_topics, *, sigma_shrink=0.0, seed=13, init='spectral', variational='laplace')
    Structural topic model: relate topic prevalence and content to covariates.
    .fit(corpus, prevalence=None, *, formula=None, data=None, prevalence_names=None, content=None, content_names=None, content_time=None, content_smooth=1.0, content_prior_var=0.5, content_prior='l2', iters=500, convergence_tol=1e-05, gamma_prior='pooled', gamma_enet=1.0, beta_init=None, em_tol=None, covariates=None, keep_eta_cov=True, num_threads=None, spectral_projection_threshold=10000, restarts=1, progress=None)

STS(num_topics, *, seed=13, init='spectral')
    Structural topic-and-sentiment model over document metadata.
    .fit(data, sentiment_seed, prevalence=None, *, prevalence_names=None, iters=30, convergence_tol=1e-05, kappa_estimation=None, kappa_ridge=0.001, em_tol=None, covariates=None, keep_eta_cov=True, reference='none', progress=None)

SAGE(num_topics, *, alpha=0.1, prior='laplace', prior_variance=1.0, optimize_interval=50, burn_in=200, seed=13, lbfgs_iters=20)
    Sparse additive generative model: the same topic worded differently across groups.
    .fit(data, groups, *, group_names=None, iters=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10)

DMR(num_topics, *, beta=0.01, optimize_interval=50, burn_in=200, seed=13, alpha=0.1, prior_variance=1.0, alpha_epsilon=1e-10, lbfgs_iters=20, sampler='sparse', num_threads=1)
    Dirichlet-multinomial regression: a document-metadata prior on topic proportions.
    .fit(data, features=None, *, feature_names=None, iters=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, covariates=None, offset=None, num_threads=None)

GDMR(num_topics: 'int', *, degrees: 'list[int]', beta: 'float' = 0.01, optimize_interval: 'int' = 50, burn_in: 'int' = 200, seed: 'int' = 13, sigma: 'float' = 1.0, sigma0: 'float' = 3.0, decay: 'float' = 0.0, alpha: 'float' = 0.1, metadata_range: 'list[tuple[float, float]] | None' = None, lbfgs_iters: 'int' = 20, sampler: 'str' = 'sparse', num_threads: 'int' = 1) -> 'None'
    Generalized DMR with a smooth (Legendre-basis) prior over continuous covariates.
    .fit(data: "'Corpus | Sequence[Sequence[str]]'", features=None, *, metadata_names=None, iters: 'int' = 1000, num_samples: 'int' = 5, sample_interval: 'int' = 25, keep_theta_draws: 'bool' = True, convergence_tol: 'float' = 0.0, check_every: 'int' = 10, covariates=None, metadata=None, progress=None) -> 'None'

Scholar(num_topics, *, covariates=None, covariate_names=None, content=None, content_names=None, interactions=False, alpha=1.0, hidden_size=100, dropout=0.2, batch_size=200, lr=0.002, l2_prior_reg=0.0, l1_content_reg=0.0, convergence_tol=0.0, seed=13)
    SCHOLAR (Card et al. 2018): a ProdLDA VAE with a covariate-shifted prevalence prior, an optional supervised label head, and optional content (topic-covariate) word deviations — neural STM prevalence + sLDA + SAGE.
    .fit(data, *, covariates=None, labels=None, content=None, iters=None, convergence_tol=None, progress=None)

RTM(num_topics, *, link=None, inference='variational', alpha=None, beta=0.1, rho=None, negative_ratio=1.0, ridge=1.0, seed=13)
    Relational topic model (Chang & Blei 2010): jointly models document text and a link graph (citations, hyperlinks, adjacency); predicts links from words and words from links.
    .fit(data, links, *, iters=50, e_sweeps=3, e_inner=5, progress=None)

FactorialLDA(factor_sizes, *, sigma_alpha=1.0, sigma_alpha_bias=1.0, sigma_omega=0.5, sigma_omega_bias=10.0, delta0=0.1, delta1=0.1, alpha_bias_init=Ellipsis, omega_bias_init=Ellipsis, step_alpha_doc=0.01, step_alpha_corpus=None, step_alpha_bias=None, step_omega=0.001, step_omega_bias=None, step_beta=0.001, block_freq=1, weight_burnin=100, word_priors=True, sparsity=True, symmetric_word_prior=False, seed=13)
    Factorial LDA (Paul & Dredze 2012): each token is a K-tuple of latent factors (e.g. topic x sentiment); structured word priors tie tuples sharing a component and a sparsity prior deactivates unsupported tuples.
    .fit(data, *, iters=2000, samples=100, eval_every=0, omega_priors=None, observed_factors=None, progress=None)

AuthorTopic(num_topics, *, alpha=None, beta=0.01, seed=13)
    Author-Topic Model: each author has a topic distribution; documents mix their authors. Answers what an author writes about.
    .fit(data, authors, *, iters=1000, progress=None)

AuthorRecipientTopic(num_topics: 'int', *, alpha: 'float | None' = None, beta: 'float | None' = 0.1, seed: 'int' = 13)
    Author-Recipient-Topic (McCallum et al. 2007): topics conditioned on the (sender, recipient) pair, for the language of a directed social network (who talks to whom about what). Realized over the AuthorTopic engine.
    .fit(docs: 'Sequence[Sequence[str]]', *, authors: 'Sequence', recipients: 'Sequence[Sequence]', iters: 'int' = 1000, progress=None)

### Guided & supervised

KeyATM(keywords, *, num_topics=None, alpha=None, beta=0.01, beta_keyword=0.1, gamma1=1.0, gamma2=1.0, seed=13, estimate_alpha=True, sampler='sparse', num_threads=1)
    Keyword-assisted topics: anchor named topics with a few seed words each.
    .fit(data, *, iters=1500, covariates=None, feature_names=None, times=None, timestamps=None, num_states=5, weights='information-theory', num_threads=None, optimize_interval=50, burn_in=200, prior_variance=1.0, lbfgs_iters=20, progress_interval=0, prior_offset=None, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, report_interval=None, turbo_alpha_stride=1, progress=None)

SeededLDA(seed_words, *, residual=0, alpha=0.5, beta=0.1, weight=0.01, seed=13, seed_prior='frequency', sampler='sparse', seed_match='fixed', case_insensitive=False, num_threads=1)
    Seeded LDA: steer named topics toward supplied seed words.
    .fit(data, *, iters=2000, doc_topic_prior=None, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, num_threads=None, progress=None)

GuidedNMF(num_topics, seed_words, *, guidance=3.0, lam=None, seed_weight=1.0, init='random', weighting='tfidf', convergence_tol=0.0, seed_match='fixed', case_insensitive=False, init_a=None, init_s=None, init_b=None, seed=13)
    Guided NMF: seed-word-guided semi-supervised NMF; the matrix-factorization analogue of SeededLDA.
    .fit(data, *, iters=None, convergence_tol=None, num_threads=None)

LabeledLDA(*, alpha=0.1, beta=0.01, seed=13, sampler='sparse', num_threads=1)
    Labeled LDA: each document label is a topic; tokens are restricted to its labels.
    .fit(data, labels, *, label_names=None, iters=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, num_threads=None)

SupervisedLDA(num_topics, *, alpha=0.1, seed=13, inference='variational')
    Supervised LDA: topics shaped to predict a per-document real-valued response.
    .fit(data, y, *, iters=25, var_iters=15, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=1, num_threads=None, progress=None)

DiscLDA(k_class, k_shared, *, alpha=None, beta=0.01, iters=1000, infer_sweeps=100, class_prior=None, seed=13)
    Discriminative LDA (Lacoste-Julien et al. 2008): topics split into per-class and shared blocks; reads how classes talk differently.
    .fit(data, y, *, iters=None, progress=None)

### Short text

GSDMM(num_topics, *, alpha=0.1, beta=0.1, seed=13, num_threads=1)
    Gibbs-sampling Dirichlet mixture: one topic per short document.
    .fit(data, *, iters=30, progress_interval=0, report_interval=None, num_threads=1, verbose=False, progress=None)

PT(num_topics, *, num_pseudo=100, alpha=0.1, beta=0.01, pseudo_doc_prior=0.1, seed=13, num_threads=1)
    Pseudo-document topic model: pool short texts into pseudo-documents.
    .fit(data, *, iters=1000, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, num_threads=None, progress=None)

BTM(num_topics, *, alpha=None, beta=0.01, iters=1000, window=15, background=False, seed=13, num_threads=1)
    Biterm topic model: learns topics from corpus-level word co-occurrence (biterms).
    .fit(data, *, iters=None, num_threads=None, progress=None)

### Dynamic & hierarchical

DTM(num_topics, *, alpha=0.01, chain_variance=0.005, obs_variance=0.5, seed=13, init='random')
    Dynamic topic model: a fixed topic set whose word distributions drift across time slices.
    .fit(data, times, *, iters=20, progress=None)

DETM(num_topics, *, delta=0.005, hidden_size=800, eta_hidden_size=200, eta_nlayers=3, batch_size=1000, lr=0.005, wdecay=1.2e-06, grad_clip=None, convergence_tol=0.0, seed=13)
    Dynamic embedded topic model: embedding-factored topics that drift across time slices, fit as an amortized VAE.
    .fit(data, word_embeddings, vocabulary, *, times=None, timestamps=None, iters=100, convergence_tol=None, progress=None)

TopicsOverTime(num_topics, *, alpha=None, beta=0.1, seed=13)
    Topics over Time: LDA with a per-topic Beta density over continuous timestamps; each topic has a temporal peak. Descriptive continuous-time prevalence (not vocabulary drift).
    .fit(data, times=None, *, timestamps=None, iters=1000, progress=None)

HLDA(*, depth=3, gamma=1.0, beta=0.01, alpha=None, level_prior='dirichlet', gem_mean=0.5, gem_scale=100.0, seed=13, eta=None)
    Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics.
    .fit(data, *, iters=500, num_threads=1, progress=None)

PA(num_super, num_sub, *, alpha=0.1, beta=0.01, seed=13, num_threads=1)
    Pachinko allocation: a DAG of super- and sub-topics.
    .fit(data, *, iters=1000, keep_theta_draws=True, num_theta_draws=25, convergence_tol=0.0, check_every=10, num_threads=None, progress=None)

### Embedding-based

KeyNMF(num_topics, *, top_n=25, metric=Ellipsis, seed=13)
    KeyNMF (Kristensen-McLachlan et al. 2024): NMF over an embedding-derived keyword-importance matrix. For each document it scores its words by the similarity between the document embedding and the word embedding, keeps the top-N positive, and factors that sparse doc-word matrix. The bridge between the count-based NMF family and the embedding backend; sparse, readable topics robust to short/noisy text.
    .fit(data, doc_embeddings, *, word_embeddings, vocabulary, iters=None, convergence_tol=None)

BERTopic(*, n_components=5, min_cluster_size=15, min_samples=None, nr_topics=None, window=4, stride=1, reducer='umap', n_neighbors=15, bm25=False, reduce_frequent=False, weighting='c-tf-idf', min_similarity=0.0, clusterer='hdbscan', num_clusters=None, resolution=1.0, knn_neighbors=15, diagnostics=True, min_dist=0.0, spread=1.0, n_epochs=0, negative_sample_rate=5, repulsion_strength=1.0, metric='cosine', seed=13)
    Cluster document embeddings; label topics by class-based TF-IDF.
    .fit(data, doc_embeddings)

Top2Vec(*, n_components=5, min_cluster_size=15, min_samples=None, reducer='umap', n_neighbors=15, clusterer='hdbscan', num_clusters=None, resolution=1.0, knn_neighbors=15, diagnostics=True, min_dist=0.0, spread=1.0, n_epochs=0, negative_sample_rate=5, repulsion_strength=1.0, metric='cosine', seed=13)
    Topics as dense regions in a joint document-word embedding space.
    .fit(data, doc_embeddings, *, word_embeddings=None, vocabulary=None)

SemanticSignalSeparation(num_topics, *, feature_importance=Ellipsis, iters=200, convergence_tol=0.0001, seed=13)
    Topics as independent axes of semantic space (S3, Kardos et al. 2025): FastICA over the document embeddings, with each word's importance read off by projecting the vocabulary embeddings onto each axis. Signed poles.
    .fit(data, doc_embeddings, vocab_embeddings, *, vocabulary=None)

ETM(num_topics, *, inference='em', convergence_tol=0.0001, sigma_shrink=0.0, prior_variance=1000000.0, max_inner=25, hidden_size=800, batch_size=1000, lr=0.005, wdecay=1.2e-06, seed=13, prior=Ellipsis, contrastive=False, contrastive_weight=0.5, contrastive_temp=0.5, em_tol=None)
    Embedded topic model: topic-word distributions factored through word embeddings.
    .fit(data, word_embeddings, vocabulary, *, iters=None, convergence_tol=None, progress=None)

GaussianLDA(num_topics, *, alpha=None, kappa=0.1, nu=None, psi_scale=3.0, init=Ellipsis, seed=13)
    Gaussian LDA (Das, Zaheer & Dyer 2015): each topic is a Gaussian over the word-embedding space (Normal-Inverse-Wishart prior), so topics generalize over semantically similar words. Collapsed Gibbs with a Student-t posterior predictive and rank-1 Cholesky up/downdates.
    .fit(data, word_embeddings, vocabulary, *, iters=None, progress=None)

FASTopic(num_topics, *, lr=0.002, dt_alpha=3.0, tw_alpha=2.0, theta_temp=1.0, convergence_tol=1e-06, sinkhorn_iters=5000, sinkhorn_tol=0.005, seed=13, em_tol=None)
    Topics from optimal-transport plans between document, topic, and word embeddings.
    .fit(data, doc_embeddings, *, iters=None, convergence_tol=None, progress=None)

CombinedTM(num_topics, *, alpha=1.0, hidden_size=100, dropout=0.2, batch_size=200, lr=0.002, convergence_tol=0.0, seed=13, prior=Ellipsis, contrastive=False, contrastive_weight=0.5, contrastive_temp=0.5)
    Contextualized ProdLDA: encoder reads the bag of words plus a document embedding.
    .fit(data, doc_embeddings, *, iters=None, convergence_tol=None)

ZeroShotTM(num_topics, *, alpha=1.0, hidden_size=100, dropout=0.2, batch_size=200, lr=0.002, convergence_tol=0.0, seed=13, prior=Ellipsis, contrastive=False, contrastive_weight=0.5, contrastive_temp=0.5)
    Contextualized ProdLDA: encoder reads the document embedding alone, enabling cross-lingual transfer.
    .fit(data, doc_embeddings, *, iters=None, convergence_tol=None)

InfoCTM(num_topics, *, mi_weight=30.0, mi_temperature=0.2, pos_threshold=0.4, hidden_size=100, dropout=0.0, lr=0.002, convergence_tol=0.0, seed=13, languages=None)
    Cross-lingual: two ProdLDA models aligned by a bilingual dictionary through a mutual-information term.
    .fit(data_a, data_b, *, dictionary, embeddings_a=None, embeddings_b=None, iters=None, batch_size=128, progress=None)

### Ideal point

Wordfish(*, beta_prior_sd=3.0, theta_prior_sd=1.0, min_count=1, convergence_tol=1e-06, seed=13)
    Poisson scaling (Slapin & Proksch 2008): an unsupervised one-dimensional ideal-point estimate from word frequencies alone, no topics. The word-frequency baseline companion to IdealPointTM.
    .fit(data, *, group=None, control=None, anchors=None, iters=None, convergence_tol=None, progress=None)

Wordshoal(*, theta_prior_sd=1.0, loading_prior_sd=0.5, intercept_prior_sd=0.5, tau_prior=1.0, min_count=1, convergence_tol=0.001, seed=13)
    Multi-domain scaling (Lauderdale & Herzog 2016): scales each debate/domain with Wordfish, then combines the within-domain positions into one cross-domain actor scale via a linear factor model. The multi-domain extension of Wordfish, for speeches carrying trusted debate labels.
    .fit(data, *, speakers, domains, anchors=None, iters=None, convergence_tol=None)

TBIP(num_topics, *, a_gamma=0.3, b_gamma=0.3, iters=7000, batch_size=512, learning_rate=0.05, min_count=1, seed=13)
    Text-Based Ideal Points (Vafa, Naidu & Blei 2020): a Poisson factorization whose neutral topic-word intensities are rescaled by a per-word ideological factor exp(x_s * eta_kv), with the author position x_s latent. Fit by the paper's mean-field variational inference (reparameterized SVI). Recovers ideological scales from unlabeled text.
    .fit(data, *, group=None, iters=None, batch_size=None, learning_rate=None, progress=None)

PartyEmbeddings(num_dims=2, *, vector_size=200, window=20, min_count=5, negative=5, sample=0.0001, learning_rate=0.025, seed=13)
    Party embeddings (Rheault & Cochrane 2020): a PV-DM paragraph-vector model trained by negative sampling with party-period metadata tags; the leading principal components of the learned party vectors give the ideological scale, and words share the space so a party's language can be read off by proximity. The corpus-trained word-embedding member of the ideal-point family.
    .fit(data, *, group, control=None, anchors=None, iters=5)

### LLM-based

TopicGPT(*, backend: 'Optional[Callable[[str], str]]' = None, model: 'Optional[str]' = None, hierarchical: 'bool' = False, assignment: 'str' = 'hard', sample: 'Optional[int]' = None, max_topics: 'Optional[int]' = None, min_topic_count: 'int' = 1, temperature: 'float' = 0.0, seed: 'int' = 13, prompts: 'Optional[dict]' = None) -> 'None'
    LLM-driven topic discovery: prompt a model to propose, refine, and assign a topic taxonomy with descriptions.
    .fit(data, *, metadata=None) -> "'TopicGPT'"

Experimental models are omitted; enable_experimental() then list_models(experimental=True) to see them.
```
<!-- END GUIDE -->
