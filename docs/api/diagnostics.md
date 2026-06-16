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

::: topica.topic_diversity

::: topica.topic_semantic_diversity

::: topica.exclusivity

::: topica.quality_frontier

## Interpretation

::: topica.label_topics

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

::: topica.align_topics

::: topica.topic_stability

::: topica.ensemble

::: topica.EnsembleResult

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
