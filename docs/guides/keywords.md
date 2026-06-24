# Distinguishing words

Sometimes the question isn't "what are the topics" but "which words separate
these two groups." **Fighting Words** (Monroe, Colaresi & Quinn 2008) answers
that with statistical significance, and, unlike a raw log-odds ratio, it doesn't
let rare words dominate.

```python
import topica

conservative = [tokenize(t) for t in con_texts]
liberal      = [tokenize(t) for t in lib_texts]

scored = topica.fighting_words(conservative, liberal, prior=0.05)
# sorted by z-score: corpus-A markers at the top, corpus-B at the bottom
for word, z in scored[:10]:
    print(f"{word:20s} {z:+.1f}")     # |z| > 1.96 ~ significant at 95%
```

A large positive `z` marks a word distinctive of the first corpus; a large
negative `z`, the second. Because the estimator's variance grows for rare words,
the z-score already accounts for how much evidence each word carries.

## Top words per side

```python
top = topica.top_fighting_words(conservative, liberal, n=15)
print("conservative:", [w for w, _ in top["a"]])
print("liberal:     ", [w for w, _ in top["b"]])
```

## Informative prior

By default the prior is a symmetric pseudocount. Pass `informative=True` to scale
the prior by each word's overall frequency: Monroe et al.'s informative
Dirichlet prior, which pulls extreme estimates toward the corpus background:

```python
topica.fighting_words(conservative, liberal, prior=0.01, informative=True)
```

This pairs naturally with [SAGE / content STM](covariates.md), which find
group-distinguishing wording *within a fitted topic model*, where Fighting Words
works directly on two raw corpora with no model at all.

## Contrastive topics

Plain Fighting Words pools the whole corpus into two bags of words. Once you have
a fitted model, you can hold each topic fixed and ask how the two groups word it
differently. `topica.contrastive_topics` weights every document's word counts by
its responsibility for a topic, splits those weighted counts by group, and runs
the same z-score per topic:

```python
rows = topica.contrastive_topics(model, corpus, groups)  # groups: one label per doc
for r in rows[:3]:                                        # most contrastive first
    print(f"topic {r['topic']} {r['name']}  used more by {r['leans']}")
    print("  ", r['a_label'], "words:", [w for w, _ in r['a_words'][:6]])
    print("  ", r['b_label'], "words:", [w for w, _ in r['b_words'][:6]])
```

`corpus` is the same documents, in the same order, that produced
`model.doc_topic`. Each row carries two complementary signals, because a topic
both groups use equally can still split sharply on *how* they word it:

- `usage_diff` — mean `doc_topic` for group A minus group B: which topics one side
  simply talks about more. Rows are sorted by its magnitude.
- `vocab_shift` — the RMS within-topic z over the words it keeps: how much the two
  groups diverge in wording the topic.

On the poliblog corpus (liberal vs conservative blogs) the iraq/war topic is used
about equally by both sides (`usage_diff` near zero) but splits hard on
`vocab_shift`: liberals write `iraq / surge / bush`, conservatives `israel /
hamas / taliban`. Reporting both columns keeps that distinction visible. It works
on any model exposing `doc_topic` and `vocabulary` (LDA, STM, DMR, CTM, keyATM).
