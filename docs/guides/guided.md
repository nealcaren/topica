# Guided topics (seed words)

Plain LDA is unsupervised: you label the topics after the fact and have no
control over whether the themes you care about appear. **Guided** models let you
inject prior knowledge as a few **seed words per topic**, so a topic forms around
words you already believe belong together. You seed *topics with keywords*, not
*documents with labels*, so there is no hand-coding.

This is squarely a social-science tool: it improves [measurement validity and
reproducibility](../publishing/validation.md), the things reviewers push on.
topica has two, matching the two standard R packages.

## SeededLDA

Seed-word priors steer some topics; `residual` unseeded topics are learned
freely. Faithful to the `seededlda` package (Watanabe): a seed word gets a
`weight × 100` prior pseudocount in its topic, plus seeded initialization.

```python
import topica

model = topica.models.SeededLDA(
    {"economy": ["jobs", "wages", "tax"],
     "immigration": ["border", "visa", "deport"]},
    residual=3,           # 3 extra unseeded topics
    seed=1,
)
model.fit(docs, iters=2000)

model.topic_names         # ['economy', 'immigration', 'residual_1', ...]
for t in range(model.num_topics):
    print(model.topic_names[t], [w for w, _ in model.top_words(8, topic=t)])
```

## KeyATM

The Keyword-Assisted Topic Model (Eshima, Imai & Sasaki 2024) is the modern,
well-validated version. A token in a keyword topic comes either from a
distribution over only that topic's keywords or from the topic's full
distribution; the learned mix is the **keyword rate**.

```python
model = topica.models.KeyATM(
    {"economy": ["jobs", "wages", "tax"],
     "immigration": ["border", "visa", "deport"]},
    num_topics=10,        # 2 keyword topics + 8 regular topics
    seed=1,
)
model.fit(docs, iters=1500)

model.keyword_rate        # per-topic share drawn from the keyword distribution
```

By default keyATM applies information-theory token weighting (each token counts
by its word's surprisal in bits), which downweights frequent words and sharpens
topics. Set `weights="inv-freq"` or `weights="none"` to change it. On large
corpora, pass `num_threads=N` to sample document partitions in parallel
(approximate distributed Gibbs); both options apply to every variant below.

### Covariate keyATM

Pass `covariates` to let document metadata shape topic prevalence, the keyATM
covariate model. The document-topic prior becomes a Dirichlet-multinomial
regression, `α_{d,k} = exp(x_d · λ_k)` (Mimno & McCallum 2008, the same engine as
[`DMR`](models.md#dmr)), so you can ask whether a covariate moves a named topic.
An intercept is prepended; the learned coefficients are in `feature_effects`.

```python
import numpy as np
is_dem = np.array([...]).reshape(-1, 1)          # one row per document
model = topica.models.KeyATM(seeds, num_topics=2, seed=1)
model.fit(docs, covariates=is_dem, feature_names=["is_dem"], iters=1000)

model.feature_names       # ['intercept', 'is_dem']
model.feature_effects     # (num_topics, 2): coefficient of each covariate per topic
model.feature_effect_se   # same shape: standard error of each coefficient
z = model.feature_effects / model.feature_effect_se   # |z| > ~2 ⇒ notable
```

A larger `feature_effects[k, j]` means covariate `j` raises topic `k`'s
prevalence. `feature_effect_se` gives the standard error of each λ, so you can
tell a real effect from noise: it is the observed information of the same
penalized Dirichlet-multinomial that `DMR` uses, computed in the standardized fit
space (keyATM z-scores covariates internally, issue #270) and mapped back to the
original covariate scale, so it is exact (no bootstrap) and computed once at fit
time. An entry is `NaN` when its standardized coefficient hit keyATM's ±5 bound,
where the constrained estimate has no valid asymptotic standard error. For
uncertainty on the resulting topic prevalences, pair the fitted `doc_topic` with
[`estimate_effect`](covariates.md).

### Dynamic keyATM

Pass `timestamps` (one per document) to let topic prevalence shift over time.
This is the keyATM dynamic model, a Chib (1998) change-point hidden Markov model:
the timeline is split into `num_states` latent regimes, each with its own
document-topic prior, and the model estimates where prevalence changes. Following
the keyATM Supreme Court application (Eshima, Imai & Sasaki 2024, Section 3.3),
documents carry a year and the model recovers when each topic rises or falls.

```python
model = topica.models.KeyATM(seeds, num_topics=14, seed=1)
model.fit(docs, timestamps=years, num_states=5, iters=3000)

model.time_labels        # ['1946', '1947', ..., '2012']  (T distinct timestamps)
model.time_state         # [0, 0, 1, 1, ..., 4]  regime of each segment
model.time_prevalence    # (T, num_topics): smoothed prevalence path, rows sum to 1
model.transition_matrix  # (num_states, num_states), left-to-right
```

Documents may be passed in any order; they are sorted by timestamp internally and
`doc_topic` is returned in the original order. Plot a column of `time_prevalence`
against `time_labels` to see a topic's trajectory.

## Embedding-guided topics (EmbeddingLDA)

`SeededLDA` and `KeyATM` ask you to name the seed words. `EmbeddingLDA` instead
discovers them from a pre-trained embedding space: it clusters the vocabulary's
embeddings into `num_topics` semantic groups, seeds each topic with the words
nearest its cluster centroid, and fits a `SeededLDA` underneath. The embeddings
warm-start where topics form; the Gibbs sampler can still override any seed the
text contradicts, so this is a prior, not a constraint.

You supply the embeddings (topica does not call any model itself), aligned to
the vocabulary:

```python
from sentence_transformers import SentenceTransformer
import topica

vocab = sorted({w for d in docs for w in d})
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(vocab)

model = topica.models.EmbeddingLDA(num_topics=10, embeddings=emb, vocabulary=vocab,
                            top_m=20, weight=1.0)
model.fit(docs, iters=1000)
for i, words in enumerate(model.top_words(8)):
    print(f"Topic {i}:", ", ".join(w for w, _ in words))
```

`top_m` sets how many of each cluster's nearest words become seeds, and `weight`
how hard they anchor (a seed gets `weight * 100` prior pseudocounts; raise it to
hold topics closer to their semantic cluster, lower it to let the data lead).
The whole fitted-model surface (`topic_word`, `doc_topic`, `top_words`,
`coherence`, ...) is delegated to the underlying `SeededLDA`, and `model.seeds`
holds the embedding-derived seed sets. `topica.embed.embedding_seeds(...)` exposes just
the clustering step if you want to inspect or edit the seeds before fitting.

## Which to use

- **`KeyATM`** is the better-validated choice and the one with the political-
  science following; prefer it for new work.
- **`SeededLDA`** is simpler and maps directly onto the `seededlda` workflow.
- **`EmbeddingLDA`** when you have embeddings but no hand-picked seed list, and
  want the topic structure anchored to semantic similarity.

Both feed the same [diagnostics](diagnostics.md), [effects](../publishing/effects.md),
and [validation](../publishing/validation.md) as every other model.

!!! note "Faithful to the references"
    On a shared corpus with identical seeds, topica recovers seeded-topic
    vocabulary close to R's `seededlda` and keyword topics that align with R's
    `keyATM` as well as R aligns with itself across seeds. The comparison runs the
    reference packages through the reproducible harness in `parity/` (it needs a
    local R install with the packages); it measures topic-word agreement, not
    identical word lists.
