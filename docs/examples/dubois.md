# W.E.B. Du Bois in *The Crisis*

A worked analysis of **704 articles from *The Crisis***, the NAACP magazine Du
Bois edited from 1910 to 1934. This example covers corpus building, phrase
detection, LDA, dynamic topics over the decades, and the nonparametric `HDP`
check on `K`.

!!! info "Focus of this example"
    Corpus building & cleaning · **temporal / dynamic topics** · using `HDP` as a
    `K` sanity check. For topic validation and clustered errors see
    [Poliblog](poliblog.md); for STM prevalence by group and time see
    [Congress](congress.md).

    Data: [`examples/dubois_crisis.csv`](https://github.com/nealcaren/topica/blob/main/examples/dubois_crisis.csv) ·
    full notebook (adds STM decade-prevalence + held-out `transform`):
    [`dubois_tutorial.ipynb`](https://github.com/nealcaren/topica/blob/main/examples/dubois_tutorial.ipynb)

## 1. Build the corpus

The articles run 1910–1934, Du Bois's editorship. A few corpus-specific
stopwords (`crisis`, `negro`, `colored`) would otherwise dominate every topic.

```python
import csv, numpy as np, topica
from topica import Corpus, tokenize

rows = list(csv.DictReader(open("examples/dubois_crisis.csv")))
for r in rows:
    r["decade"] = int(r["decade"])

EXTRA = ["crisis", "negro", "negroes", "colored", "men", "man", "upon", "us",
         "one", "two", "shall", "may", "must", "said", "make", "made"]
stop = sorted(set(open("examples/english-stoplist.txt").read().split()) | set(EXTRA))
docs = [tokenize(r["text"], stopwords=stop, min_length=3) for r in rows]

by_decade = {}
for r in rows:
    by_decade[r["decade"]] = by_decade.get(r["decade"], 0) + 1
print(len(rows), "articles;", {d: by_decade[d] for d in sorted(by_decade)})
```

```
704 articles; {1910: 356, 1920: 276, 1930: 72}
```

## 2. Phrases, then a pruned corpus

```python
phrases = topica.learn_phrases(docs, min_count=8, threshold=12.0)
docs = topica.apply_phrases(docs, phrases)            # "jim crow" -> "jim_crow"
corpus = Corpus.from_documents(docs, min_doc_freq=10, rm_top=20)
print("vocab", corpus.num_words)                  # 3423
```

## 3. LDA

```python
lda = topica.LDA(num_topics=15, seed=1)
lda.fit(corpus, iters=400, num_samples=4, sample_interval=25)
for t in range(15):
    print(f"T{t:>2}: " + ", ".join(w.replace("_", " ") for w, _ in lda.top_words(8, topic=t)))

print("c_v:", round(float(np.mean(topica.coherence(lda, docs, coherence_type="c_v"))), 3),
      "diversity:", round(topica.topic_diversity(lda, topn=15), 3))
```

```
T 1: schools, education, children, public, segregation, training, college, teachers
T 3: lynching, murder, state, law, crime, case, mob, governor
T 6: labor, economic, industrial, industry, today, workers, capital, class
T 8: war, africa, races, england, peace, europe, civilization, government
T 9: votes, voters, party, political, states, election, state, disfranchisement
T13: church, committee, organization, conference, general, members, national, board
T14: officers, french, camp, regiment, general, division, troops, army
c_v: 0.612 diversity: 0.884
```

The topics are recognizable: education, lynching, labor and economics, war and
Africa, voting and disfranchisement, the NAACP's organizational work, and Black
WWI regiments.

## 4. Dynamic topics over the decades

The Dynamic Topic Model fixes the topics but lets their vocabulary drift across
ordered time slices. `chain_variance=0.05` lets real trends show; the default
0.005 is too stiff for three slices.

```python
decades = sorted(by_decade)
times = [decades.index(r["decade"]) for r in rows]
dtm = topica.DTM(num_topics=8, chain_variance=0.05, seed=1)
dtm.fit(corpus, times, iters=20)

vocab = list(dtm.vocabulary)
per_time = np.stack([dtm.topic_word(s) for s in range(dtm.num_times)])
print("           " + "   ".join(f"{d}s" for d in decades))
for w in ["war", "labor", "africa", "schools"]:
    wid = vocab.index(w)
    topic = int(per_time[:, :, wid].mean(0).argmax())
    evo = dtm.word_evolution(topic, w)
    print(f"  {w:8s}: " + "   ".join(f"{1000 * float(p):5.1f}" for p in evo))
```

```
           1910s   1920s   1930s
  war     :   7.3     6.5     6.3
  labor   :  13.1    15.4    18.6
  africa  :  15.9    21.7    23.9
  schools :  16.8    16.2    14.1
```

The trajectories track Du Bois's intellectual arc: **labor** and **Africa** rise
steadily across his editorship (his economic turn and the Pan-African
Congresses), while **war** recedes after the WWI 1910s and **schools** drifts
down.

`word_evolution` traces one word you already have in mind. To see *which* words
drive a topic's drift, use `word_drift`:

```python
labor_topic = int(per_time[:, :, vocab.index("labor")].mean(0).argmax())
drift = dtm.word_drift(labor_topic, n=6)      # first slice (1910s) vs last (1930s)
print("rising :", [w for w, _ in drift["rising"]])
print("falling:", [w for w, _ in drift["falling"]])
```

```
rising : ['industry', 'labor', 'workers', 'capital', 'communists', 'economic']
falling: ['business', 'pay', 'union', 'modern', 'service', 'movement']
```

The labor topic's vocabulary shifts from *business, pay, union* toward *industry,
workers, capital, communists*. The topic doesn't just grow; its language moves
from a reformist register to a Marxist one, which is Du Bois's own trajectory in
these years.

## 5. How many topics? Ask HDP

The Hierarchical Dirichlet Process infers the topic count rather than taking one.
It is a check on the `K = 15` chosen above.

```python
hdp = topica.HDP(eta=0.3, seed=1)
hdp.fit(corpus, iters=150)
print("HDP inferred K =", hdp.num_topics)          # 17
```

`HDP` lands on 17, close to the 15 used for LDA.

## 6. Guided topics — name themes up front

The models above *discover* topics, which you then label. When you already know
the themes you want to measure, a [guided model](../guides/guided.md) seeds them
by name, so each topic maps to a construct by construction. `KeyATM` names four
themes from Du Bois's program and learns four more freely:

```python
seeds = {
    "education": ["school", "schools", "education", "college", "children"],
    "labor":     ["labor", "wages", "industrial", "economic", "workers"],
    "voting":    ["vote", "votes", "ballot", "suffrage", "franchise"],
    "africa":    ["africa", "african", "congo", "liberia", "empire"],
}
ka = topica.KeyATM(seeds, num_topics=8, seed=1)
ka.fit(docs, iters=800)
for t in range(4):
    print(f"{ka.topic_names[t]:10s}", [w for w, _ in ka.top_words(7, topic=t)])
```

```
education  ['school', 'schools', 'work', 'education', 'south', 'children', 'people']
labor      ['white', 'american', 'black', 'labor', 'social', 'race', 'world']
voting     ['south', 'vote', 'state', 'women', 'united_states', 'lynching', 'southern']
africa     ['africa', 'world', 'war', 'great', 'church', 'america', 'england']
```

Education, voting, and Africa land cleanly on their seeds; labor stays diffuse
because Du Bois ties labor to race throughout the corpus — a substantive signal,
not a model failure. `ka.keyword_rate` reports how much each topic leans on its
seeds. The [full notebook](https://github.com/nealcaren/topica/blob/main/examples/dubois_tutorial.ipynb)
continues with STM decade-prevalence (labor rises, women's suffrage falls after
1920) and held-out `transform`, following the
[publishing workflow](../publishing/index.md) end to end.
