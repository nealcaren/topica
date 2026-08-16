# 4. Validate the topics

This is the step that separates a publishable analysis from a fishing
expedition. You must show, with evidence, that your topics are (a) **coherent**
to humans, (b) **reproducible**, and (c) **interpretable** as substantive themes.

## Human validation: intrusion tests

The field-standard test (Chang et al. 2009, *Reading Tea Leaves*) asks whether a
human can tell a topic apart from noise. topica builds both variants for
you, with an answer key.

**Word intrusion**: each topic's top words plus one intruder from another topic.

```python
import topica

tests = topica.evaluate.word_intrusion(model, n_words=5, seed=0)
for t in tests:
    print(f"Topic {t['topic']}: {t['words']}")
    # answer key for coding:
    # intruder '{t['intruder']}' at position {t['intruder_index']}
```

Show the shuffled `words` to coders (hide the key); a topic where coders reliably
spot the intruder is coherent.

**Document intrusion**: a topic's representative documents plus one where the
topic is nearly absent. This tests whether the topic captures real document
similarity.

```python
tests = topica.evaluate.document_intrusion(model, texts=texts, n_docs=3, seed=0)
```

Report intrusion accuracy (and, ideally, inter-coder agreement) in your
supplementary materials.

## Computational validation: coherence and exclusivity

Per-topic metrics complement the human tests. Report both together: a good
topic is coherent **and** exclusive.

```python
coherence   = model.coherence(10)               # UMass, per topic
cv          = topica.evaluate.coherence(model, texts, coherence_type="c_v", topn=10)
exclusivity = topica.evaluate.exclusivity(model, 10)          # per topic

frontier = topica.select.quality_frontier(model, n=10)      # tidy: coherence, exclusivity, prevalence
```

`c_v` correlates best with human judgement; UMass is a fast intrinsic check;
NPMI is the normalized middle ground. The coherence×exclusivity scatter is the
canonical STM quality plot: weak topics sit in the lower-left.

## External validation: agreement with gold labels

Coherence rates a topic's top words, not whether documents landed in the right
topic — and for embedding-based cluster models the two can diverge (a model can
keep tight, coherent top-words while the document partition drifts). If you have
hand-coded labels for your documents, or even a labeled subset, score the recovery
directly:

```python
pred   = model.labels                      # or model.doc_topic.argmax(1)
scores = topica.agreement(pred, gold)      # {ari, nmi, homogeneity, completeness, v_measure, purity}

# partially labeled corpus: score only the coded documents
scores = topica.agreement(pred[mask], gold[mask])
```

ARI is adjusted for chance (0 = random, 1 = a perfect match up to relabeling). Use
`noise="drop"` to exclude HDBSCAN's unassigned (`-1`) documents, or the default
`noise="keep"` to score them honestly as their own bucket. This is the number that
tracks recovery when coherence alone would reassure you a failed run "worked."

## Stability: answer the "fishing expedition" critique head-on

A topic that dissolves when you perturb the corpus is not a finding. Refit on
bootstrap resamples and report which topics are robust:

```python
boot = topica.evaluate.bootstrap_stability(docs, k=20, n_boot=50, iters=800)
for t, s in zip(boot["topic"], boot["stability"]):
    print(f"Topic {t}: stability {s:.2f}")     # mean top-word Jaccard across resamples
print("overall:", boot["mean"])
boot.to_frame()                                 # the same, as a tidy DataFrame
```

For a **covariate model** (STM, DMR, ...), bootstrap it *as itself* — pass the
same per-document design you fit with, and it is resampled in lock-step with the
documents so the covariate rows stay aligned:

```python
X, names = topica.design.one_hot(df["party"])
stm = topica.STM(num_topics=20, seed=13).fit(docs, prevalence=X, prevalence_names=names)
boot = topica.evaluate.bootstrap_stability(
    docs, reference=stm,
    model_factory=lambda s: topica.STM(num_topics=20, seed=s),
    prevalence=X, prevalence_names=names,      # design resampled with the docs
)
```

Any fit argument with one row per document (`prevalence` / `covariates`, a
`content` label vector, dynamic `timestamps`) is resampled with the draw;
per-feature arguments like `prevalence_names` are passed through.

Also check that the *same* topics emerge across **random seeds**, aligning topics
between fits and scoring their overlap:

```python
a = topica.LDA(num_topics=20, seed=1); a.fit(docs, iters=800)
b = topica.LDA(num_topics=20, seed=2); b.fit(docs, iters=800)
pairs = topica.evaluate.align_topics(a, b)                    # one-to-one matching (Hungarian)
print("stability across seeds:", topica.evaluate.topic_stability([a, b], topn=10))
```

Flag fragile topics explicitly. A paper that says "topics 4 and 11 were unstable
across resamples and we interpret them cautiously" is *more* credible, not less.

## Interpretation: close reading, not just top words

Distant reading (top words) and **close reading** (the actual documents) are both
required. Pull a topic's most representative documents and read them, with the
topic's words highlighted in your notebook:

```python
labels = topica.inspect.label_topics(model.topic_word, model.vocabulary, n=8)  # prob & FREX
html = topica.inspect.find_thoughts_html(model, texts, n_docs=3, n_words=8)    # highlighted quotes
```

Use **FREX** (frequent *and* exclusive) words, not just the most probable ones,
when labeling. They are far more evocative of what makes a topic distinctive.

!!! danger "What topics are NOT"
    Topics are statistical patterns of word co-occurrence. They are **not**
    necessarily coherent concepts, and **not** necessarily what a document is
    "about." Write *"this document has high probability for Topic 3"* and *"words
    associated with Topic 3 suggest…"* Never write *"the model discovered that…"* or
    *"this document is about Topic 3."*

## Common problems and what to do

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **Junk topics** (stopwords, numbers) | Under-pruned vocabulary | Add custom stopwords; raise `min_doc_freq`; `rm_top` |
| **Duplicate topics** | `K` too high, or duplicate documents | Lower `K`; dedupe; check `align_topics` / `topic_correlation` |
| **Uninterpretable topics** | Not every topic is meaningful | Document as "Mixed/Other"; consider lower `K` |
| **One dominant topic** | Corpus-wide vocabulary | May be legitimate; or treat as a background topic and `rm_top` |

→ Next: [Measure effects properly](effects.md).
