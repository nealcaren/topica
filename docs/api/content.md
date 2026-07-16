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
families — STM/STS via `topic_word_by_group`, SAGE via its 3-D `topic_word`, ECTM
via `content_word_dist(group, period)` (period-averaged by default; pass
`period=` for a per-period trajectory).

::: topica.content.topic_polarization

::: topica.content.group_exclusivity

::: topica.content.split_topics

::: topica.content.group_topic_word
