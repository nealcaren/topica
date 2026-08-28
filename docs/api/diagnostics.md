# Diagnostics

Model-agnostic quality, interpretation, and validation tools. They take any
fitted model's `topic_word` / `doc_topic` (or raw arrays), so they work the same
across every model family. They live in the `topica.evaluate` namespace
(`topica.evaluate.<name>`, the documented path below); every name is also
reachable bare at the top level (`topica.<name>`) as a compatibility alias.

## One-call table

::: topica.evaluate.diagnostics

::: topica.evaluate.perplexity

::: topica.evaluate.reply_completion

## Quality

::: topica.evaluate.coherence

::: topica.evaluate.coherence_ci

::: topica.evaluate.semantic_coherence

::: topica.evaluate.embedding_coherence

::: topica.evaluate.topic_diversity

::: topica.evaluate.topic_semantic_diversity

::: topica.evaluate.inverted_rbo

::: topica.evaluate.exclusivity

::: topica.evaluate.topic_significance

::: topica.evaluate.coherence_over_time

::: topica.evaluate.diversity_over_time

::: topica.select.quality_frontier

## External validation

When you have gold (or partially gold) labels for your documents, `agreement`
scores how well the discovered topics recover them — the check that actually
tracks recovery, where coherence can mislead.

::: topica.agreement

::: topica.evaluate.classification_quality

## Interpretation

::: topica.inspect.label_topics

::: topica.inspect.topics_for_term

::: topica.llm_topic_labels

::: topica.llm_backend

::: topica.topic_label_prompts

::: topica.inspect.frex

::: topica.inspect.mmr

::: topica.inspect.relevance

::: topica.inspect.find_thoughts

::: topica.inspect.find_thoughts_html

::: topica.inspect.topic_correlation

::: topica.inspect.prepare_pyldavis

## Validation

::: topica.evaluate.word_intrusion

::: topica.evaluate.document_intrusion

### LLM-based evaluation (`topica.llm`)

::: topica.llm.coherence

::: topica.llm.intrusion

::: topica.llm.select_k

::: topica.llm.outlier

::: topica.llm.repetitiveness

::: topica.llm.diversity

::: topica.llm.alignment

::: topica.llm.adversarial

::: topica.evaluate.bootstrap_stability

::: topica.select.search_k

::: topica.evaluate.check_residuals

::: topica.evaluate.document_residuals

::: topica.evaluate.flag_topics

::: topica.evaluate.topic_dendrogram

::: topica.evaluate.align_topics

::: topica.evaluate.topic_stability

::: topica.ensemble

::: topica.EnsembleResult

::: topica.cross_ensemble

## MCMC convergence (`topica.mcmc`)

Single-chain autocorrelation and effective sample size for the collapsed-Gibbs
models, computed from the retained log-likelihood trace and `theta_draws`. See
the [convergence section](../guides/diagnostics.md#has-the-chain-plateaued-or-mixed)
of the diagnostics guide.

::: topica.mcmc_diagnostics

::: topica.McmcDiagnostics

::: topica.effective_sample_size

::: topica.integrated_autocorr_time

::: topica.autocorrelation

Multi-chain Gelman-Rubin R-hat and cross-chain ESS, from several fits of the same
model at different seeds. See the
[R-hat section](../guides/diagnostics.md#do-independent-chains-agree-r-hat) of the
diagnostics guide.

::: topica.multichain_diagnostics

::: topica.MultiChainDiagnostics

::: topica.rhat

## Held-out likelihood

Build a within-corpus word-heldout set — the analogue of R `stm`'s
`make.heldout` — and score it under a fitted model to get document-completion
log-likelihood.

::: topica.evaluate.make_heldout

::: topica.evaluate.eval_heldout

## Estimator conformance

Check any fitted model or model class against the topica estimator contract;
returns a list of violation strings (empty means fully conformant).

::: topica.provenance.check_conformance

## Reporting

Model-neutral summaries that work on any fitted model.

::: topica.plot_report

::: topica.topic_info

::: topica.topics_over_time

::: topica.topics_per_class

::: topica.contrastive_topics
