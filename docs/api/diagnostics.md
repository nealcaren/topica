# Diagnostics

Model-agnostic quality, interpretation, and validation tools. They take any
fitted model's `topic_word` / `doc_topic` (or raw arrays), so they work the same
across every model family. All are available at the top level (`topica.<name>`)
and in the `topica.validation` module.

## One-call table

::: topica.diagnostics

::: topica.perplexity

## Quality

::: topica.coherence

::: topica.coherence_ci

::: topica.semantic_coherence

::: topica.topic_diversity

::: topica.topic_semantic_diversity

::: topica.exclusivity

::: topica.quality_frontier

## External validation

When you have gold (or partially gold) labels for your documents, `agreement`
scores how well the discovered topics recover them — the check that actually
tracks recovery, where coherence can mislead.

::: topica.agreement

## Interpretation

::: topica.label_topics

::: topica.topics_for_term

::: topica.llm_topic_labels

::: topica.llm_backend

::: topica.topic_label_prompts

::: topica.frex

::: topica.mmr

::: topica.relevance

::: topica.find_thoughts

::: topica.find_thoughts_html

::: topica.topic_correlation

::: topica.prepare_pyldavis

## Validation

::: topica.word_intrusion

::: topica.document_intrusion

### LLM-based evaluation (`topica.llm`)

::: topica.llm.coherence

::: topica.llm.intrusion

::: topica.llm.select_k

::: topica.llm.outlier

::: topica.llm.repetitiveness

::: topica.llm.diversity

::: topica.llm.alignment

::: topica.llm.adversarial

::: topica.bootstrap_stability

::: topica.search_k

::: topica.check_residuals

::: topica.document_residuals

::: topica.flag_topics

::: topica.topic_dendrogram

::: topica.align_topics

::: topica.topic_stability

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

::: topica.make_heldout

::: topica.eval_heldout

## Estimator conformance

Check any fitted model or model class against the topica estimator contract;
returns a list of violation strings (empty means fully conformant).

::: topica.check_conformance

## Reporting

Model-neutral summaries that work on any fitted model.

::: topica.plot_report

::: topica.topic_info

::: topica.topics_over_time

::: topica.topics_per_class

::: topica.contrastive_topics
