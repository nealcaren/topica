# Party platforms: how the words of a topic move

This worked example asks a question most topic models cannot: not just *which*
topics a group discusses, but *how it words them*, and how that wording **changes
over time**. The corpus is every Democratic and Republican national platform from
1948 to 2024. We treat the party as the content **group** and the election year
as an ordered **`content_time`** covariate, then read how each party's vocabulary
on a shared topic evolves across twenty elections.

The model is [`STM`](../guides/models.md#stm) with a `content_time` covariate.
STM's content model already lets a topic be worded differently across a group (the
SAGE content covariate); adding `content_time=` crosses that group with the
period into a base-by-period content design, tied across adjacent periods by a
first-order random walk (`content_smooth`) and regularized by a sparse L1 prior
(`content_prior="l1"`). The reading layer —
[`topica.content.content_trajectory`](../api/content.md) and `content_divergence`
— reads the smoothed surface back out, with a design-preserving document/cluster
bootstrap for confidence bands.

!!! info "Focus of this example"
    Content covariates that drift over ordered time · whole-distribution vs
    per-word contrasts · a design-preserving bootstrap that resamples platforms,
    not `(party, year)` cells. For a single topic's vocabulary drift without
    groups see the dynamic models ([`DTM`](../guides/models.md#dtm)).

    Data ships gzipped in
    [`examples/platforms_data/`](https://github.com/nealcaren/topica/tree/main/examples);
    rebuild it from the American Presidency Project with
    [`examples/prep_platforms.py`](https://github.com/nealcaren/topica/blob/main/examples/prep_platforms.py).
    The full script is
    [`examples/stm_content_time_platforms.py`](https://github.com/nealcaren/topica/blob/main/examples/stm_content_time_platforms.py).

## 1. The corpus

Each platform is a long document, so we split it into paragraphs. That gives many
documents per (party, year) cell, which the model needs, and it makes the unit of
text a single argument rather than a whole platform. The result is about 11,000
paragraph-documents, both parties present at all twenty elections.

```python
import gzip, json, re, numpy as np, topica
from collections import Counter
from topica.content import content_divergence, content_trajectory

rows = json.load(gzip.open("examples/platforms_data/platforms.json.gz", "rt"))
docs, party, year = [], [], []
for r in rows:
    toks = [w for w in re.findall(r"[a-z]{3,}", r["text"].lower()) if w not in STOP]
    docs.append(toks); party.append(r["party"]); year.append(int(r["year"]))
# drop very rare and ubiquitous words, and near-empty paragraphs (see the script)
```

```text
11353 platform paragraphs | parties D/R | 20 elections 1948-2024 | vocab 4963
```

## 2. Fit STM with a content_time covariate

Party is the content group; the election year is the ordered `content_time`
covariate. The prevalence design (party, a year spline, and their interaction)
is the *attention* half — how often each topic is discussed — and is optional
here; the content design is where the wording lives.

```python
party_col, pn = topica.design.one_hot(party)                 # indicator(Republican)
yr_basis, sn = topica.design.spline(np.asarray(year, float), df=4)
inter, _ = topica.design.interaction(party_col, yr_basis, name="party_year")
X = np.column_stack([party_col, yr_basis, inter])
names = list(pn) + list(sn) + [f"party_year_{i}" for i in range(inter.shape[1])]

model = topica.STM(num_topics=18, seed=1)
model.fit(docs, prevalence=X, prevalence_names=names,
          content=party, content_time=year,
          content_prior="l1", content_prior_var=1.0, content_smooth=6.0,
          iters=150)
```

`content_smooth` is the random-walk precision over periods (larger pools adjacent
periods more); `content_prior="l1"` keeps the content deviations sparse so a
period's wording departs from the baseline only where the data earn it. With
`content_time=None` the fit reduces to plain STM.

## 3. Which topics are worded most differently by party

`content_divergence` gives the whole-distribution distance between the two
groups' wording of a topic, one value per period. Averaging it ranks topics by
how partisan their language is.

!!! warning "Strip actor names before a content covariate that correlates with the actor"
    On a corpus of named authors (press releases, floor speeches), a person's
    surname appears only in that person's documents, so under `content=party` the
    `word_contrast` for a topic is topped by legislator names — speaker identity,
    not framing. Remove actor names from the vocabulary during preprocessing (add
    them to the stopword list) so the substantive framing contrast surfaces
    instead. This example's corpus is already name-light; a raw press-release
    corpus is not.

```python
groups = ("D", "R")

def avg_div(k):
    d = content_divergence(model, groups=groups, topic=k, measure="tv")
    return float(np.nanmean(d.divergence))

for k in sorted(range(model.num_topics), key=avg_div, reverse=True)[:4]:
    print(f"Topic #{k} (avgTV={avg_div(k):.3f}): "
          f"{' '.join(model.top_words(9, topic=k))}")
```

```text
Topic #17 (avgTV=0.216): policies president administration first policy help important both essential
Topic #4  (avgTV=0.150): administration percent million billion president spending now last inflation
Topic #13 (avgTV=0.146): act service improve veterans adequate living possible age disabled
Topic #12 (avgTV=0.142): rights right women equal discrimination civil religious oppose individual
```

## 4. One topic's wording over time

Find the environment topic (the one that weights `climate` most) and trace its
between-party wording across the twenty elections.

```python
vi = {w: i for i, w in enumerate(model.vocabulary)}
env = int(np.asarray(model.topic_word)[:, vi["climate"]].argmax())

div = content_divergence(model, groups=groups, topic=env, measure="tv")
for yr, d in zip(div.periods, div.divergence):
    print(f"  {yr}  {d:.2f}  {'#' * int(d * 40)}")
```

The party divergence on the environment is flat through the twentieth century and
climbs sharply after 2000:

```text
  1948  0.07  ##          1996  0.11  ####
  1964  0.10  ###         2000  0.13  #####
  1980  0.10  ####        2012  0.18  #######
  1988  0.08  ###         2020  0.22  ########
  1992  0.09  ###         2024  0.25  #########
```

`content_trajectory` reads the same surface at the word level — the signed
`p(w | D, period) - p(w | R, period)` contrast (positive = more Democratic):

```python
traj = content_trajectory(model, ["climate", "conservation", "clean", "pollution"],
                          groups=groups, topic=env)
```

```text
D-R word-probability contrast (x1000), every 4th election (positive = Democratic):
             1948  1964  1980  1996  2012
  climate     -0.2  -0.7  -0.9   0.2  12.0
  conservation -4.4   3.7   1.4   0.7  -1.4
  clean       -1.4  -0.0  -1.7   1.2  21.1
  pollution   -0.0  -0.2  -0.1   0.5   2.0
```

Inside a single stable topic, `climate` and `clean` enter the Democratic
vocabulary after 2000 while Republicans never adopt them, and `conservation` — the
older, shared framing — fades. This is evolution *within* the topic's wording, not
a change in how often the topic is discussed, and a fixed-vocabulary dynamic model
cannot represent it.

## 5. A design-preserving bootstrap

The point trajectory hides its own uncertainty. `content_divergence(..., ci=True)`
resamples whole **platforms** (the `(party, year)` clusters) with replacement,
refits STM, realigns the target topic by its anchor words in each replicate, and
returns percentile bands. Resampling the real sampling unit — the document, or a
cluster of documents — is what makes the interval honest; resampling `(group,
period)` cells would leave cells missing and target an ill-defined population.

```python
cluster = [f"{p}{y}" for p, y in zip(party, year)]
fit_kwargs = dict(num_topics=18, prevalence=X, prevalence_names=names,
                  content=party, content_time=year,
                  content_prior="l1", content_prior_var=1.0, content_smooth=6.0,
                  iters=80)
anchor = list(model.top_words(8, topic=env))
dci = content_divergence(model, groups=groups, anchor_words=anchor,
                         measure="hellinger", ci=True, corpus=docs,
                         fit_kwargs=fit_kwargs, cluster=cluster, B=40, seed=0)
```

The bands are widest at the first and last elections, where the random walk is
least constrained by neighboring periods — the honest uncertainty a point
trajectory hides. Pass `cluster=` any length-`num_docs` label (e.g. a speaker or
source document) to resample whole clusters instead of individual documents.
