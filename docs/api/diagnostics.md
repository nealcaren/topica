# Diagnostics

Model-agnostic quality, interpretation, and validation tools. They take any
fitted model's `topic_word` / `doc_topic` (or raw arrays), so they work the same
across every model family. All are available at the top level (`topica.<name>`)
and in the `topica.validation` module.

## One-call table

::: topica.diagnostics.diagnostics

::: topica.diagnostics.perplexity

## Quality

::: topica.diagnostics.coherence

::: topica.diagnostics.coherence_ci

::: topica.diagnostics.semantic_coherence

::: topica.diagnostics.topic_diversity

::: topica.diagnostics.topic_semantic_diversity

::: topica.diagnostics.exclusivity

::: topica.diagnostics.quality_frontier

## Interpretation

::: topica.interpret.label_topics

::: topica.interpret.llm_topic_labels

::: topica.interpret.llm_backend

::: topica.interpret.topic_label_prompts

::: topica.interpret.frex

::: topica.interpret.mmr

::: topica.interpret.relevance

::: topica.interpret.find_thoughts

::: topica.interpret.find_thoughts_html

::: topica.interpret.topic_correlation

::: topica.viz.prepare_pyldavis

## Validation

::: topica.diagnostics.word_intrusion

::: topica.diagnostics.document_intrusion

### LLM-based evaluation (`topica.llm`)

::: topica.llm.coherence

::: topica.llm.intrusion

::: topica.llm.select_k

::: topica.llm.outlier

::: topica.llm.repetitiveness

::: topica.llm.diversity

::: topica.llm.alignment

::: topica.llm.adversarial

::: topica.diagnostics.bootstrap_stability

::: topica.select.search_k

::: topica.diagnostics.check_residuals

::: topica.diagnostics.document_residuals

::: topica.diagnostics.flag_topics

::: topica.interpret.topic_dendrogram

::: topica.diagnostics.align_topics

::: topica.diagnostics.topic_stability

::: topica.ensemble.ensemble

::: topica.ensemble.EnsembleResult

::: topica.ensemble.cross_ensemble

## MCMC convergence (`topica.mcmc`)

Single-chain autocorrelation and effective sample size for the collapsed-Gibbs
models, computed from the retained log-likelihood trace and `theta_draws`. See
the [convergence section](../guides/diagnostics.md#has-the-chain-plateaued-or-mixed)
of the diagnostics guide.

::: topica.diagnostics.mcmc_diagnostics

::: topica.diagnostics.McmcDiagnostics

::: topica.diagnostics.effective_sample_size

::: topica.diagnostics.integrated_autocorr_time

::: topica.diagnostics.autocorrelation

## Held-out likelihood

Build a within-corpus word-heldout set — the analogue of R `stm`'s
`make.heldout` — and score it under a fitted model to get document-completion
log-likelihood.

::: topica.diagnostics.make_heldout

::: topica.diagnostics.eval_heldout

## Estimator conformance

Check any fitted model or model class against the topica estimator contract;
returns a list of violation strings (empty means fully conformant).

::: topica.diagnostics.check_conformance

## Reporting

Model-neutral summaries that work on any fitted model.

::: topica.viz.plot_report

::: topica.interpret.topic_info

::: topica.effects.topics_over_time

::: topica.effects.topics_per_class

::: topica.interpret.contrastive_topics
