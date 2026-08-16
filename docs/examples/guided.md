# Guided topics: naming the themes you're looking for

Unsupervised topic models decide for themselves what the topics are. Often you
already know several of the themes you care about and want the model to *find those*
— anchored to your vocabulary — while still discovering the rest. That is what a
keyword-guided model does. topica's [`KeyATM`](../api/keyatm.md) (and the
Gibbs [`SeededLDA`](../api/models.md)) let you seed named topics with a handful of
keywords each; the seeded topics stay anchored, the remaining topics are free.

!!! info "Focus of this example"
    **Keyword-guided / semi-supervised** topic models (`KeyATM`, `SeededLDA`) ·
    seeding named topics · reading the keyword switch rate. For fully unsupervised
    covariate modeling on the same corpus see [Congress](congress.md).

    Data: [`load_congress`](../api/datasets.md#topica.datasets.load_congress) —
    3,120 U.S. House press releases, 2013–2024.

## 1. Seed the themes, fit the model

We name five policy themes with a few keywords each and ask for ten topics, so the
model anchors those five and discovers five more on its own.

```python
import topica

df = topica.datasets.load_congress()
corpus = topica.from_dataframe(df, text_col="text", strip_html=True,
                               stopwords=topica.data.ENGLISH_STOPWORDS,
                               min_doc_freq=10, max_doc_fraction=0.4)

keywords = {
    "healthcare":  ["health", "care", "insurance", "medicare", "medicaid", "patients"],
    "defense":     ["military", "defense", "veterans", "troops", "security", "war"],
    "immigration": ["immigration", "border", "immigrants", "visa", "citizenship"],
    "economy":     ["tax", "jobs", "economy", "budget", "workers", "wages"],
    "climate":     ["climate", "energy", "environment", "emissions", "clean"],
}
model = topica.KeyATM(keywords, num_topics=10, seed=13).fit(corpus.documents(), iters=400)
```

## 2. The anchored topics stay on theme, and drift informatively

The five seeded topics come back recognizably on-theme, and *where they drift* is
itself a finding — the healthcare seed lands on the opioid crisis, the climate seed
pulls in agriculture and the EPA:

| Seeded topic | Top FREX words |
|---|---|
| healthcare | patients, opioid, substance, prescription, cdc, care, mental |
| defense | iran, sanctions, ndaa, korea, navy, afghanistan, servicemembers |
| immigration | cbp, daca, customs, visa, border (and Spanish-language releases) |
| economy | tax, cuts, loan, obamacare, income, shutdown |
| climate | farm, epa, conservation, usda, wildlife, drinking |

The five free topics discover structure you did not name: schools and colleges;
gun violence and elections; oversight letters and investigations; Black history and
voting rights; and rail, transit, and infrastructure.

## 3. The keyword switch rate says how much the seeds mattered

`KeyATM` reports, per keyword topic, the rate at which its tokens are drawn from the
seed set rather than the topic's full distribution — a direct read on how much the
guidance is doing.

```python
model.keyword_rate    # e.g. [0.074, 0.079, 0.071, 0.036, 0.043, 0, 0, 0, 0, 0]
```

The five seeded topics carry a positive switch rate; the five free topics sit at
zero (no keywords), so the number cleanly separates guided from discovered topics.

## When to reach for this

Seed a guided model when a literature or a codebook gives you the themes in
advance and you want measured, anchored versions of them rather than hoping an
unsupervised run surfaces them. For the Gibbs alternative with the same idea, swap
in `topica.SeededLDA(keywords, num_topics=10)`. When you have *no* prior themes,
use a plain [`LDA`](poliblog.md) or the covariate [`STM`](congress.md) instead.

## Reproduce

`seed=13`; raise `iters` (2,000+) for a publication run. The keyword dictionary is
the one knob that matters most — a few high-precision keywords per theme beat a long
noisy list.
