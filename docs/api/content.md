# Content-covariate diagnostics

Diagnostics for the content-covariate models (STM, STS, SAGE, ECTM), which learn
a *group-specific* topic-word tensor $\beta_{k,g,v}$ — how each topic is worded by
each group. The global topic-word average hides that variation; these read it
back out.

They answer a question that recurs with content models: does a group's
distinctive language land **within** a topic (one topic, worded differently by
group) or **fragment** into parallel, group-skewed topics? `topic_polarization`
measures the first; `split_topics` detects the second. The main lever that moves
a fit between the two is ECTM's `content_prior_var` (looser prior → more
within-topic variation).

All read the group tensor through one adapter, so they work across model
families — STM via `topic_word_by_group`, SAGE via its 3-D `topic_word`, ECTM
via `content_word_dist(group, period)` (period-averaged by default; pass
`period=` for a per-period trajectory). STS has a *continuous* sentiment axis
rather than discrete groups, so the adapter discretizes it — evaluating
`topic_word_at(level)` at the sentiment poles `-1`/`0`/`+1`
(negative/neutral/positive) by default; pass `levels=` to choose your own.

::: topica.content.topic_polarization

::: topica.content.group_exclusivity

::: topica.content.split_topics

::: topica.content.stratified_coherence

::: topica.content.diagnostics

::: topica.content.group_topic_word

## Wording over ordered time (STM `content_time`)

Fit STM with an ordered `content_time=` covariate and these readers trace **how
two groups word a topic across time**: the per-period, per-word contrast
`p(w | g1) - p(w | g2)`, and the per-period whole-vocabulary distance between the
groups. Both take `ci=True` for a **design-preserving bootstrap** that resamples
documents (or whole `cluster=`s), refits, realigns the topic by `anchor_words`, and
returns percentile bands — intervals widen where the random walk is least
constrained (the first and last period).

```python
from topica import content

stm = topica.STM(num_topics=20, seed=1)
stm.fit(docs, content=party, content_time=year, content_prior="l1")

tr = content.content_trajectory(stm, ["tax", "climate"],
                                groups=("Democrat", "Republican"), anchor_words=econ)
tr.to_frame()                       # word, period, estimate

dv = content.content_divergence(stm, groups=("Democrat", "Republican"),
                                anchor_words=econ, measure="hellinger",
                                ci=True, corpus=docs, fit_kwargs=fk)  # + bootstrap CI
```

::: topica.content.content_trajectory

::: topica.content.content_divergence

## Choosing K with group-stratified coherence

`topica.search_k` accepts a `"stratified_<type>"` coherence metric for
content models (`model="stm"` with a `content` covariate): it scores each group's
own top words against that group's subcorpus, and reports group-adjusted
exclusivity and mean polarization alongside.

```python
res = topica.search_k(
    corpus, ks=[5, 10, 15], model="stm",
    prevalence=X, content=source,
    coherence_type="stratified_c_npmi",
)
# each row: k, coherence (stratified), exclusivity (group-adjusted), polarization
```
