# Short text: when a document is one topic, not a mixture

LDA assumes every document is a *mixture* of topics. That assumption breaks on
short text — a headline, a tweet, a search query — where a handful of words support
at most one theme. Fitting LDA anyway spreads each short document thinly across
topics and blurs them. topica's short-text models make the honest assumption
instead: [`GSDMM`](../api/models.md) (the Dirichlet Mixture "movie-group" model)
assigns each document to a single cluster, and [`BTM`](../api/models.md) models
word co-occurrence (biterms) directly rather than per-document mixtures.

!!! info "Focus of this example"
    **Short-text models** (`GSDMM`, `BTM`) · one-topic-per-document clustering ·
    why LDA blurs on short docs. Uses the headline-length `title` field of the
    congress corpus (median 6 tokens).

    Data: [`load_congress`](../api/datasets.md#topica.datasets.load_congress) — we
    model the release **titles**, not the bodies.

## 1. Cluster short titles with GSDMM

`GSDMM` takes a cap on the number of clusters and prunes empty ones as it runs, so
it also estimates *how many* topics the short corpus supports. Each title lands in
one cluster.

```python
import topica

df = topica.datasets.load_congress()
corpus = topica.from_dataframe(df, text_col="title",     # the headline, not the body
                               stopwords=topica.data.ENGLISH_STOPWORDS,
                               min_doc_freq=5, max_doc_fraction=0.3)
model = topica.GSDMM(num_topics=20, seed=13).fit(corpus.documents(), iters=40)
```

The clusters are crisp and legible despite the median title being six tokens long:

| Size | Cluster (FREX) |
|---:|---|
| 287 | appropriations, security, homeland, defense, budget |
| 278 | introduce, introduces, mental, legislation, improve |
| 255 | passes, reform, pass, act, amendments |
| 253 | justice, warren, general, accountability, press |
| 193 | affordable, access, help, opportunities |
| 192 | supreme, court, ruling, scotus, debt |
| 168 | million, project, grant, projects, port |

## 2. The contrast: LDA blurs on the same titles

Fit a plain `LDA` on the identical short documents and the topics come back generic
and overlapping, because the mixture assumption has nothing to hold onto in six
words:

```python
lda = topica.LDA(num_topics=8, seed=13).fit(corpus)
```

> house, committee, passes, senate, appropriations · border, call, hearing,
> security, investigation · state, address, letter, office, union

Procedural boilerplate (`house`, `committee`, `senate`) dominates several topics,
and substantive themes bleed together. On full-length documents LDA is the right
tool; on titles it is the wrong assumption, and the short-text model recovers the
sharper structure.

## When to reach for this

Use `GSDMM` or `BTM` when documents are short enough that a per-document topic
*mixture* is over-parameterized — tweets, headlines, product names, log lines,
survey one-liners. `GSDMM` gives a hard one-topic-per-document assignment and infers
the cluster count; `BTM` is often steadier when even single documents are tiny,
since it pools co-occurrence across the whole corpus. For full-length text, prefer
[`LDA`](poliblog.md) or a covariate model.

## Reproduce

`seed=13`; `GSDMM` converges in a few dozen sweeps. Lower the `num_topics` cap to
merge fine clusters, or raise it to let more emerge — GSDMM prunes the ones that
stay empty.
