# Political blogs: validation and clustered errors

This worked example carries the **validation** and **effects** load of the
[publishing workflow](../publishing/index.md): choosing `K` with a scan,
validating topics, and the headline task of estimating covariate effects with
**clustered standard errors** for nested data. The corpus is `poliblog5k` from
the R `stm` package: 2,000 political-blog posts from the 2008 U.S. campaign, each
tagged with the blog's `rating` (Conservative / Liberal), the `day`, and the
`blog` it came from.

!!! info "Focus of this example"
    K selection · topic validation · **clustered SEs** · GLM links. For corpus
    cleaning and dynamic topics see [Du Bois](dubois.md); for STM prevalence by
    group and time see [Congress](congress.md).

    Data: [`examples/poliblog.csv`](https://github.com/nealcaren/topica/blob/main/examples/poliblog.csv)
    (reconstructed from `stm`'s preprocessed, stemmed `poliblog5k.docs`).

## 1–2. Corpus and model

The posts nest within six blogs, and we want to know how topic prevalence differs
by ideology. That covariate question makes the
[right model](../publishing/choosing-model.md) the STM.

```python
import csv, numpy as np, topica
from topica import Corpus, stm

rows = list(csv.DictReader(open("examples/poliblog.csv")))
docs = [r["text"].split() for r in rows]          # already tokenized + stemmed by stm
corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)
print(corpus.num_docs, "docs, vocab", corpus.num_words)
```

```
2000 docs, vocab 2612
```

## 3. Choose and justify K

```python
for r in topica.search_k(docs, ks=[10, 15, 20], iters=500):
    print(f"K={r['k']:>2}  coherence={r['coherence']:.1f}  exclusivity={r['exclusivity']:.3f}")
```

```
K=10  coherence=-51.1  exclusivity=0.579
K=15  coherence=-57.9  exclusivity=0.532
K=20  coherence=-59.8  exclusivity=0.556
```

Coherence (here UMass, so closer to zero is better) and exclusivity both favor the
smaller model. We still take `K = 15` for finer thematic resolution, exactly the
[trade-off the K guide warns against resolving by metric alone](../publishing/choosing-k.md),
and we confirm the substantive effects below survive `K ∈ {10, 20}`.

## 4. Validate

```python
conservative = np.array([r["rating"] == "Conservative" for r in rows], float).reshape(-1, 1)
model = topica.STM(num_topics=15, seed=1)
model.fit(docs, conservative, prevalence_names=["conservative"], iters=25)

labels = stm.label_topics(model.topic_word, model.vocabulary, n=6)
for t in range(15):
    print(f"T{t:>2}: " + ", ".join(w for w, _ in labels[t]["frex"]))
```

```
T 1: isra, israel, hama, iran, iranian, terrorist
T 2: school, abort, children, gay, god, parent
T 3: wright, barack, obama, ayer, chicago, team
T 6: rove, tortur, administr, cheney, bush, constitut
T 7: lieberman, mccain, joe, biden, sen, john
T 9: iraqi, iraq, afghanistan, troop, withdraw, saddam
T10: republican, parti, democrat, gop, conserv, pelosi
T13: hillari, clinton, primari, deleg, nomin, edward
```

The topics are readable: foreign policy, social issues, the Obama–Wright story,
the financial crisis, the primaries. Validate them with a human intrusion test
and with bootstrap stability:

```python
print(topica.word_intrusion(model, n_words=5, seed=0)[0])
# {'topic': 0, 'words': ['voter','mccain','poll','state','obama','investig'],
#  'intruder': 'investig', 'intruder_index': 5}

boot = topica.bootstrap_stability(docs, k=15, n_boot=20, iters=400)
print("mean topic stability:", round(boot["mean"], 2))   # 0.36
```

Stability of 0.36 (mean top-word Jaccard across resamples) is moderate for 15
topics on noisy blog text, with a per-topic spread from about 0.08 to 0.58.
Report the spread and treat the low-stability topics cautiously.

## 5. Effects — and why clustering changes the answer

The posts are **nested in six blogs**. Trusting ordinary standard errors here
treats 2,000 posts as 2,000 independent observations, which badly overstates
certainty. Estimate the ideology effect on each topic with the method of
composition, once naively and once clustered by blog:

```python
draws = stm.posterior_theta_samples(model, nsims=25, seed=0)
blog  = np.array([r["blog"] for r in rows])

iid       = stm.estimate_effect(draws, conservative, feature_names=["conservative"])
clustered = stm.estimate_effect(draws, conservative, feature_names=["conservative"],
                                cluster=blog)
```

| Topic (FREX) | coef | SE (iid) | SE (**clustered**) | z (clustered) |
|--------------|-----:|---------:|-------------------:|--------------:|
| isra, israel, hama | +0.055 | 0.006 | 0.010 | **+5.8** |
| wright, barack, obama | +0.044 | 0.006 | 0.020 | +2.2 |
| media, stori, matthew | +0.042 | 0.008 | 0.048 | +0.9 |
| hillari, clinton, primari | −0.022 | 0.006 | 0.041 | −0.5 |
| rove, tortur, administr | −0.065 | 0.006 | **0.034** | −1.9 |
| lieberman, mccain, joe | −0.084 | 0.005 | **0.029** | −2.9 |

Clustering inflates the standard errors three- to six-fold. The Rove/torture
topic looks overwhelmingly liberal under iid errors (z ≈ −10) but is **not
significant** once we account for all those posts coming from a handful of blogs
(z = −1.9). Conservatives reliably talk more about Israel/Iran and the
Obama–Wright story. The apparent torture effect does not survive well-calibrated
uncertainty.

!!! warning "Report the caveat"
    With only **six** clusters, cluster-robust inference is itself approximate
    (CR1 wants ~30+ clusters). Say so. A careful paper reports the clustered
    result and notes the small number of clusters. That is far more credible than
    using iid errors that assume 2,000 independent observations.

## 6. Report

Bounded inference (topic proportions live in `[0,1]`) via a fractional-logit
link, a topic table with FREX labels and prevalence, and the saved model. See
[Report and make reproducible](../publishing/reporting.md).

```python
glm = stm.estimate_effect(draws, conservative, feature_names=["conservative"],
                          cluster=blog, link="logit")
model.save("poliblog_stm.tt")
```

## A guided alternative: name the topics up front

The STM above discovers topics, which you then label. If you already know the
themes you want to measure — a common case in political communication — a
[guided model](../guides/guided.md) seeds them directly, so each topic
corresponds to a construct by construction (better validity and reproducibility).
Here `KeyATM` seeds four 2008-campaign themes and learns four more freely:

```python
seeds = {
    "foreign_policy": ["isra", "iran", "iraq", "troop", "afghanistan"],
    "economy":        ["economi", "market", "tax", "job", "financi"],
    "social":         ["abort", "gay", "marriag", "religi", "church"],
    "campaign":       ["poll", "vote", "campaign", "candid", "elect"],
}
ka = topica.KeyATM(seeds, num_topics=8, seed=1)
ka.fit(docs, iters=800)
for t in range(4):
    print(f"{ka.topic_names[t]:15s}", ka.top_words(7, topic=t))
```

```
foreign_policy  ['iraq', 'war', 'militari', 'iran', 'bush', 'forc', 'american']
economy         ['will', 'tax', 'american', 'govern', 'economi', 'year', 'econom']
social          ['american', 'america', 'peopl', 'will', 'women', 'countri', 'right']
campaign        ['obama', 'vote', 'democrat', 'republican', 'voter', 'will', 'poll']
```

The named topics land on their themes (the foreign-policy and campaign topics are
sharp; the social topic is more diffuse because that theme is weaker in this
corpus). `ka.keyword_rate` reports how much each topic leans on its keywords.
Feed the result to the same `estimate_effect` and validation steps as the STM —
the only thing that changed is that you specified the topics instead of labeling
them afterward.
