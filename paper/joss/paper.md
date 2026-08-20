---
title: 'topica: Fast, all-in-one topic modeling for Python'
tags:
  - Python
  - Rust
  - topic modeling
  - natural language processing
  - text as data
  - computational social science
authors:
  - name: Neal Caren
    orcid: 0000-0003-1704-5494
    affiliation: 1
affiliations:
  - name: Department of Sociology, University of North Carolina at Chapel Hill, United States
    index: 1
date: 20 August 2026
bibliography: paper.bib
---

# Summary

Topic models discover recurring themes in a collection of documents and describe
each document as a mixture of those themes. They are a standard tool for
researchers who work with text at scale. `topica` is a topic-modeling library for
Python that brings fifty models on its validated default surface, classical and
embedding-based alike, behind one NumPy-native interface. Latent Dirichlet Allocation, the Structural
Topic Model, keyword-assisted models, dynamic and short-text models, and modern
clustering and neural models all present the same surface: a topic-word matrix and
a document-topic matrix. Because the surface is shared, one set of tools for
choosing the number of topics, reading topics, checking stability, validating
fits, and estimating covariate effects applies to every model, and a count-based
model and a clustering of sentence embeddings can be fit and compared on the same
corpus in a single session. A parallel Rust core makes the fits fast and
reproducible, and the package installs as a single wheel that needs only NumPy and
pandas, with no Java runtime and no deep-learning framework.

# Statement of need

The models a text-as-data researcher needs are scattered across incompatible
software. The Structural Topic Model lives in the R package `stm`
[@roberts2019stm]; the fastest collapsed-Gibbs samplers live in the Java toolkit
MALLET [@mccallum2002mallet]; keyword-assisted models live in R `keyATM` [@eshima2024keyatm];
and clustering, autoencoding, and embedding-based models live in separate Python
repositories such as BERTopic [@grootendorst2022bertopic] and Top2Vec
[@angelov2020top2vec]. Some libraries do collect several models, but each covers
one region of the space: gensim [@rehurek2010gensim] adds HDP, author-topic, and
dynamic variants to its variational LDA; tomotopy [@tomotopy] provides a fast C++
collapsed-Gibbs engine for a dozen classical models behind a Python API; Turftopic
[@kardos2025turftopic] unifies contextual, embedding-based models under a
scikit-learn [@pedregosa2011scikit] interface; and evaluation-oriented frameworks
such as OCTIS [@terragni2021octis] and TopMost [@wu2024topmost] assemble classical
and neural models for benchmarking, hyperparameter optimization, and metric-based
comparison. These frameworks target research *on* topic models. None of them spans
the classical, structural, keyword, and embedding families that applied work draws
on together, and none pairs the models with a shared diagnostic and
covariate-effect layer, so a study that wants to compare a structural model against
a clustering of embeddings, and to read how each topic varies with a document
attribute, must move data between ecosystems by hand and re-learn an interface for
each.

`topica` closes that gap by unifying the families behind a single, uniform
interface built for reuse. Every model that presents the shared topic-word and
document-topic surface inherits the entire stack of coherence, exclusivity,
stability, labeling, and covariate-effect tools without any of it being rewritten,
which is also what makes the library straightforward to extend. The design starts
from the premise that the researcher, not the software, owns the decisions that
make a topic-model study credible: what the corpus represents, how many topics is
the right granularity, and what each topic means. The software supplies the
mechanics and the diagnostics that support those decisions, and it matches the
uncertainty it reports to what each model can support, exposing standard errors
alongside covariate effects where the estimator earns them. Speed and
reproducibility come from the Rust core. Fit to convergence, the structural and
other variational models run roughly one and a half to three times faster than R
`stm` on a single core and five to seven times faster on a sixteen-core
workstation; `topica` is at parity with the compiled MALLET and `keyATM` samplers;
and it reproduces the variational models to the bit and the samplers exactly from a
fixed seed and thread count.

# Functionality

`topica` groups its roster into families that share the common surface:

- **Classical count-based models**: LDA [@blei2003lda] (with the MALLET SparseLDA
  [@yao2009sparselda], LightLDA [@yuan2015lightlda], and WarpLDA [@chen2016warplda]
  samplers), Labeled LDA [@ramage2009llda], SeededLDA [@jagarlamudi2012seeded], SAGE
  [@eisenstein2011sage], the Dirichlet-Multinomial Regression (DMR)
  [@mimno2008dmr] and generalized-DMR [@lee2020gdmr] covariate models, Pachinko
  allocation [@li2006pa], HDP [@teh2006hdp], and hierarchical LDA [@griffiths2004hlda].
- **Structural and keyword models**: the Structural Topic Model
  [@roberts2014structural], the Correlated Topic Model [@blei2007ctm], and the
  keyATM family (base, covariate, and dynamic) [@eshima2024keyatm].
- **Short-text, dynamic, and specialized models**: BTM [@yan2013biterm], GSDMM
  [@yin2014gsdmm], the Dynamic Topic Model [@blei2006dtm], Topics-over-Time
  [@wang2006topics], author-topic [@rosenzvi2004author] and supervised
  [@blei2008slda] models, and political-scaling models such as Wordfish
  [@slapin2008wordfish] and Wordshoal [@lauderdale2016measuring].
- **Neural and embedding-based models**: ProdLDA [@srivastava2017prodlda], ETM
  [@dieng2020etm], CombinedTM [@bianchi2021combined], ZeroShotTM [@bianchi2021zeroshot],
  FASTopic [@wu2024fastopic], BERTopic [@grootendorst2022bertopic], Top2Vec
  [@angelov2020top2vec], KeyNMF [@kristensenmclachlan2024keynmf], and Semantic Signal
  Separation [@mclachlan2024s3]. The embedding-based models consume document
  embeddings the user supplies from any encoder, so the package carries no embedder
  as a dependency.

All models share a common ingress (a `Corpus` built from a DataFrame or token
lists), and a common analysis workflow organized into stages: selecting the number
of topics, fitting, inspecting topics, evaluating robustness, and estimating
covariate effects. The library ships coherence and exclusivity metrics, stability
and reproducibility checks, optional large-language-model topic labeling, and an
analysis-provenance record so that a fitted result carries the settings that
produced it. Each model on the default surface is validated before it ships:
twenty-one of them are checked against an external reference implementation in R,
Java, or Python, and the rest by recovering a planted answer on a synthetic corpus
with a known solution.

# Target audience

`topica` is built for computational social scientists and other empirical
researchers who treat text as data, and for instructors who want a single
framework in which students can compare topic-modeling approaches rather than
learning a separate tool for each. The uniform surface, the covariate-effect
tools, and the honest reporting of uncertainty are aimed at the applied questions
that dominate this work: how a theme varies with a document attribute such as time,
author, party, or treatment condition, and how confident a researcher can be in
that variation. Worked examples and reproducible case studies on bundled datasets
are provided in the documentation.

# AI usage disclosure

`topica` was developed with the assistance of the Claude Code agent (Anthropic) for
implementation, refactoring, and documentation. Correctness was not taken on trust:
every model on the default surface is validated against a reference implementation
or by planted-recovery and invariance checks, the package carries an automated test
suite run in continuous integration, and a cross-implementation parity harness
compares results against R and Java references. This paper was drafted with the same
assistance and reviewed and edited by the author.

# References
