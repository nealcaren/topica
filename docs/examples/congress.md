# Congressional press releases (STM by party and time)

A worked analysis of **3,120 U.S. House press releases, 2013–2024** (260 per year,
balanced by party). This is the Structural Topic Model in its home setting:
prevalence that depends on *who* is speaking (party) and *when* (year). We ask
which topics are distinctively Democratic or Republican, and which rose or fell
across the twelve years, each with honest uncertainty.

!!! info "Focus of this example"
    Building a corpus from **raw** web text · **STM with a group covariate and a
    time trend** · reading party and time effects with confidence intervals. For
    dynamic topics over time see [Du Bois](dubois.md); for K selection,
    validation, and clustered SEs see [Poliblog](poliblog.md).

    Data: [`topica.datasets.load_congress()`](../api/datasets.md#topica.datasets.load_congress)
    (a clean bundled sample) · full raw-to-result script:
    [`examples/congress_tutorial.py`](https://github.com/nealcaren/topica/blob/main/examples/congress_tutorial.py),
    which builds the frame from the raw JSONL of Derek Willis's
    [congress-press](https://github.com/dwillis/congress-press) (MIT). The archive
    runs 2001–present; the tutorial samples a slice and falls back to the bundled
    copy offline.

## 1. From raw text to a corpus

Press releases arrive with datelines, HTML, and URLs. `strip_html=True` removes
tags and `http`/`www` boilerplate before tokenizing, so `href`/`aspx` do not form
a junk topic; `from_dataframe` keeps the `party` and `year` covariates aligned to
the documents that survive pruning.

```python
import numpy as np
import topica

df = topica.datasets.load_congress()          # bundled clean sample (or build from raw)
corpus = topica.from_dataframe(
    df, text_col="text", strip_html=True,
    stopwords=topica.data.ENGLISH_STOPWORDS,
    min_doc_freq=10, max_doc_fraction=0.4,
)
```

## 2. Choose K, then fit the STM

We pick `K` at the coherence/exclusivity frontier, then fit with a hand-built
design: the party contrast (reference = Democrat) plus a centered linear year, so
the year coefficient reads directly as a per-year trend. For curvature, swap in
`topica.design.spline(year, df=4)`, which returns a `(basis, names)` pair you `hstack`
the same way.

```python
X_party, party_names = topica.design.one_hot(corpus.metadata["party"], reference="Democrat")
scan = topica.select.search_k(corpus, [10, 15, 20, 25], model="stm",
                       prevalence=X_party, iters=60, seed=13)
k = scan.best_k()                              # frontier knee; warns on a grid edge

year = corpus.metadata["year"].to_numpy(float)
X = np.hstack([X_party, (year - year.mean()).reshape(-1, 1)])
names = party_names + ["year"]
model = topica.STM(num_topics=k, seed=13)
model.fit(corpus, prevalence=X, prevalence_names=names, iters=200)
```

A representative `K = 15` FREX table (`label_topics(model, n=8)`):

| Topic | FREX label words |
|---|---|
| Guns / violence | gun, victims, violence, trafficking, sexual, police, crime |
| Veterans | veterans, va, affairs, veteran, homeless, disability, suicide |
| COVID relief | relief, coronavirus, covid, faa, loan, pandemic, cares |
| Foreign policy | iran, ukraine, hamas, syria, saudi, israel, gaza |
| Tax / budget | tax, debt, taxes, shutdown, cuts, republicans, democrats |
| Climate / EPA | emissions, epa, climate, environmental, wildlife, pollution |
| Opioids | opioid, addiction, drugs, prescription, drug, patients |

## 3. Party and time effects, with uncertainty

`estimate_effect` propagates topic-estimation uncertainty (method of composition).
Read every effect by **name** — never `coef[0]`, which is the intercept.

```python
effects = topica.effects.estimate_effect(model, X=X, feature_names=names, nsims=50, seed=0)

for eff in effects:
    party = eff.effect_of("Republican")   # positive = more Republican (baseline Democrat)
    trend = eff.effect_of("year")         # per-year change in the prevalence logit
```

The signs line up with substance. Opioids, COVID relief, and climate lean
Democratic; veterans and committee procedure lean Republican. Over 2013–2024
(scaling the per-year slope by the eleven-year span), climate and opioids each rise
about `+0.027` in prevalence logit with intervals that exclude zero
(`CI[+0.012, +0.041]`, `p < 1e-3`), while veterans and procedural topics decline.

Because the reference level is Democrat and the year term is centered, no effect is
read off the intercept, and the party and time stories are separately identified —
the pairing an STM makes cheap to report.

## Reproduce

```bash
.venv-dev/bin/python examples/congress_tutorial.py
```

`seed=13` throughout; widen `ks`, `iters`, and `nsims` for a publication run. The
bundled `load_congress()` is the quick path; the tutorial script shows the full
raw-JSONL ingestion the loader hides.
