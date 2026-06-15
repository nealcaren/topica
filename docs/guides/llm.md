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

# Bring a model: a callable, or model="gpt-4o-mini" via the topica[llm] adapter.
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

### What composes, and the caveats

TopicGPT is a *cluster-style* model: the LLM does the clustering and labeling that
HDBSCAN + class-based TF-IDF do in [BERTopic](embedding.md). It therefore exposes
the standard fitted-model surface — `doc_topic`, `topic_word`, `top_words`,
`coherence`, `transform`, `save`/`load` — so `coherence`, `topic_diversity`,
`exclusivity`, `label_topics`, and `find_thoughts` all run on it.

Two honest caveats follow from there being **no generative posterior**:

- **`topic_word` is a descriptor, not a distribution.** It is synthesized after
  the fact by class-based TF-IDF over each topic's assigned documents (the same
  construction BERTopic uses), so it ranks salient words but is not a generative
  `P(w | topic)`. Read it as "words characteristic of this topic", not as model
  probabilities.
- **No confidence intervals.** `estimate_effect`, `posterior_theta_samples`, and
  `ensemble` are declined with an informative error. With no posterior over topic
  proportions there is nothing honest to put an interval around; use a generative
  model (LDA, STM, ...) when you need covariate effects with uncertainty.

### Determinism and cost

TopicGPT is tagged **`llm-bounded`**: with `temperature=0` and a backend `seed`
the output is *stable*, not bit-reproducible, because it depends on an external
model topica does not control. Within a single fit, responses are cached keyed by
`(prompt, model)`, which gives within-session reproducibility and avoids repeated
calls for repeated prompts. Cost scales with the number of documents (one
assignment call each); use `sample=` to bound the generation stage.

### Editable prompts

The generation, refinement, and assignment prompts are the method, so they are
exposed as editable assets in `topica.topicgpt.PROMPTS` and can be overridden
per-instance:

```python
from topica.topicgpt import PROMPTS

custom = dict(PROMPTS)
custom["assignment"] = my_assignment_template   # must keep the {taxonomy}, {document}, {n_label} fields
model = topica.TopicGPT(backend=my_callable, prompts=custom)
```

Auditing and adapting the prompts is part of using the method responsibly: the
researcher owns the theory the prompts encode.
