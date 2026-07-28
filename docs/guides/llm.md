# LLM-based topics

Most models in topica learn topics from word counts or embeddings. **TopicGPT**
(Pham et al. 2024) takes a different route: it *prompts a language model* to read
the documents, propose a topic taxonomy with plain-language descriptions, refine
it, and assign each document to a topic with a supporting quote. The headline
output is the set of topic *descriptions*, which the count-based clustering models
do not produce.

As with [LLM labeling](keywords.md) and `llm_embed`, topica assembles the prompts
and you bring the model. The backend is any callable `str -> str`, so your own
client, an `ollama` endpoint, a test fake, or the `topica[llm]` adapter all work
without topica taking an API dependency.

## TopicGPT

```python
import topica

# Bring a model: a callable, or a model name via llm_backend, e.g.
# model="gpt-4o-mini" / "claude-3-5-haiku-latest" / "gemini-2.5-flash", or a local
# model with base_url="http://localhost:11434/v1" (ollama). Any callable works.
model = topica.TopicGPT(backend=my_callable, assignment="hard")
model.fit(docs)                       # docs: a Corpus, raw strings, or token lists

model.num_topics                      # discovered count (a fitted attribute, like HDP)
model.topic_descriptions              # the LLM's one-line description per topic
model.top_words(8)                    # salient words per topic (see the caveat below)
model.assignments[0]                  # Assignment(topic_id=..., quote="...")
model.transform(new_docs)             # assign held-out docs to the taxonomy
```

`assignment="soft"` lets a document belong to several topics (row-normalized
`doc_topic`); `hierarchical=True` also induces a two-level super/sub grouping in
`model.hierarchy`. `sample=` restricts the generation stage to the first *n*
documents as a cost control (assignment still covers every document), and
`model.estimated_calls(docs)` reports the backend-call budget before you run.
`min_topic_count=` prunes topics evoked by fewer than *n* documents before
refinement (the reference's rare-topic removal; the default of 1 keeps every
topic), and `max_topics=` caps the count carried past refinement. When a backend
reply does not match the requested format, the unparsed lines are recorded in
`model.parse_drops` and a warning reports the count, so a malformed backend is
not silent.

### What composes, and the caveats

TopicGPT is a *cluster-style* model: the LLM does the clustering and labeling that
HDBSCAN + class-based TF-IDF do in [BERTopic](embedding.md). It therefore exposes
the standard fitted-model surface — `doc_topic`, `topic_word`, `top_words`,
`coherence`, `transform`, `save`/`load` — so `coherence`, `topic_diversity`,
`exclusivity`, `label_topics`, and `find_thoughts` all run on it.

Two caveats follow from there being **no generative posterior**:

- **`topic_word` is a descriptor, not a distribution.** It is synthesized after
  the fact by class-based TF-IDF over each topic's assigned documents (the same
  construction BERTopic uses), so it ranks salient words but is not a generative
  `P(w | topic)`. Read it as "words characteristic of this topic", not as model
  probabilities.
- **No confidence intervals.** `estimate_effect`, `posterior_theta_samples`, and
  `ensemble` are declined with an informative error. With no posterior over topic
  proportions there is nothing to put an interval around; use a generative
  model (LDA, STM, ...) when you need covariate effects with uncertainty.

### Determinism and cost

TopicGPT is tagged **`llm-bounded`**: with `temperature=0` and a backend `seed`
the output is *stable*, not bit-reproducible, because it depends on an external
model topica does not control. Within a single fit, responses are cached keyed by
`(prompt, model)`, which gives within-session reproducibility and avoids repeated
calls for repeated prompts. Cost scales with the number of documents (one
assignment call each); use `sample=` to bound the generation stage.

### Editable prompts

The generation, refinement, and assignment prompts *are* the method, so they are
exposed as editable assets in `topica.topicgpt.PROMPTS`. The defaults are adapted
from the published TopicGPT reference prompts
([chtmp223/topicGPT](https://github.com/chtmp223/topicGPT), MIT-licensed; Pham,
Hoyle, Sun & Iyyer 2024): the bracketed `[level] Label: Description` output
format, the few-shot demonstrations, and the rules (generalizable single topics;
never invent a topic or a quote). The two documented deviations are that
refinement asks for the full refined topic list rather than only the incremental
merge edits, and the assignment prompt carries topica's hard/soft `{n_label}`
phrasing.

Override one stage and keep the rest — a partial dict merges over the defaults:

```python
# Adapt just the generation stage to your domain (keep {taxonomy} and {document}):
model = topica.TopicGPT(backend=my_callable,
                        prompts={"generation": my_generation_template})

# Or, equivalently, the convenience method (chainable, before fit):
model = topica.TopicGPT(backend=my_callable).with_prompt("generation", my_generation_template)
```

A custom `generation`/`assignment` template must keep the `{taxonomy}` and
`{document}` placeholders (assignment also `{n_label}`); `refinement` keeps
`{taxonomy}`. Unknown keys and missing placeholders raise immediately rather than
failing later at format time. Auditing and adapting the prompts is part of using
the method responsibly: the researcher owns the theory the prompts encode.
