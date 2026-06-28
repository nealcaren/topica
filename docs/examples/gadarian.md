# Gadarian: a survey experiment

This worked example is the canonical STM vignette (Roberts, Stewart & Tingley),
and it carries the **model-choice** and **experimental-effect** load of the
[publishing workflow](../publishing/index.md). The data are 341 open-ended
responses from a survey experiment (Gadarian & Albertson): half of respondents
were primed to feel **anxious** about immigration (`treatment = 1`), half not. The
question is whether the prime changed *what people wrote about*.

!!! info "Focus of this example"
    Model choice (an experiment ⟶ STM) · effect estimation by the method of
    composition. The randomized design means responses are independent, so no
    clustering is needed, unlike [Poliblog](poliblog.md). For validation and
    nested-data clustering see [Poliblog](poliblog.md); for corpus building see
    [Du Bois](dubois.md).

    Data: [`examples/gadarian.csv`](https://github.com/nealcaren/topica/blob/main/examples/gadarian.csv) ·
    full script: [`examples/stm_vignette.py`](https://github.com/nealcaren/topica/blob/main/examples/stm_vignette.py)

## Why STM, and fit it

The design is a randomized experiment with a single binary covariate. We want to
know how that covariate moves **topic prevalence**, with a valid hypothesis test.
That is precisely what the [Structural Topic Model](../publishing/choosing-model.md)
is for: prevalence regressed on `treatment`, plus the method of composition for
well-calibrated standard errors. A small `K` suits short responses and a handful of
theoretically motivated frames.

```python
import csv, numpy as np, topica
from topica import tokenize, stm

rows = list(csv.DictReader(open("examples/gadarian.csv")))
stop = set(open("examples/english-stoplist.txt").read().split())
docs = [tokenize(r["open.ended.response"], stopwords=stop, min_length=3) for r in rows]
treatment = np.array([float(r["treatment"]) for r in rows]).reshape(-1, 1)
print("treated:", int(treatment.sum()), "control:", int((1 - treatment).sum()))

model = topica.STM(num_topics=3, seed=1)
model.fit(docs, treatment, prevalence_names=["treatment"], iters=40)
```

```
treated: 171 control: 170
```

## The three frames

Read the topics with both highest-probability and FREX words:

```python
labels = stm.label_topics(model.topic_word, model.vocabulary, n=7)
for t in range(3):
    print(f"T{t}  prob: " + ", ".join(w for w, _ in labels[t]["prob"]))
    print(f"     frex: " + ", ".join(w for w, _ in labels[t]["frex"]))
```

```
T0  prob: citizens, illegals, way, free, benefits, services, crime
     frex: benefits, using, fact, never, issue, years, medical
T1  prob: illegal, border, welfare, coming, language, care, health
     frex: assimilate, help, society, wages, well, mexican, control
T2  prob: people, immigrants, immigration, jobs, country, think, english
     frex: difficult, away, process, low, looking, born, wage
```

## The treatment effect

```python
draws = stm.posterior_theta_samples(model, nsims=30, seed=0)
effects = stm.estimate_effect(draws, treatment, feature_names=["treatment"])
for t, e in enumerate(effects):
    d = e.as_dict()["treatment"]
    print(f"T{t}: coef={d['coef']:+.3f}  z={d['z']:+.1f}  ci=({d['ci'][0]:+.3f}, {d['ci'][1]:+.3f})")
```

```
T0: coef=+0.121  z=+3.9  ci=(+0.059, +0.182)
T1: coef=-0.044  z=-1.4  ci=(-0.106, +0.018)
T2: coef=-0.076  z=-2.4  ci=(-0.138, -0.014)
```

In this single 3-topic fit the anxiety prime is associated with higher prevalence
of the threat frame (T0: benefits, services, crime; z=+3.9) and lower prevalence
of the procedural frame (T2: process, born, wage; z=−2.4, borderline). That echoes
the direction of the original study, but read it as one fit, not a settled result:
before reporting, confirm `K` with `searchK`, the topics across several seeds with
`select_model`, and the effect with a `permutation_test`. Because treatment was
randomized and each respondent contributes one independent response, ordinary
method-of-composition standard errors are appropriate here; no clustering is
needed.

## Close-read the rising frame

Distant reading should always be checked against the documents. Pull the
responses most associated with the topic the prime raised:

```python
texts = [r["open.ended.response"] for r in rows]
for i, prop, txt in stm.find_thoughts(model.doc_topic, texts=texts, topic=0, n=2):
    print(f"doc {i} (θ={prop:.2f}): {txt[:80]}")
```

```
doc 160 (θ=0.95): i am most worried about the conception that forms in the relation of th
doc 284 (θ=0.94): the fact that congress doesn't have the balls to enforce the laws alread
```

The [full vignette script](https://github.com/nealcaren/topica/blob/main/examples/stm_vignette.py)
adds the rest of the reviewer-proof apparatus on this dataset: the
topic-correlation network and a `searchK` check. It is guarded by the test suite.
