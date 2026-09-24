# Threaded conversations: ThreadTM

Forum comments are written in reply to something, and deep replies are often too short to
place on their own words. `ThreadTM` fits an ordinary topic model as the base (LDA by default,
or STM or CTM) and lets each document borrow topic mass from its reply context:

$$
\tilde\theta_d = \frac{n_d\,\theta_d + a_p\,\theta_{\mathrm{par}(d)} + a_t\,\theta_{\mathrm{thr}(d)}}
                      {n_d + a_p + a_t}
$$

Here $n_d$ is the document's in-vocabulary token count, $\theta_{\mathrm{par}(d)}$ its parent's
base topic mix, and $\theta_{\mathrm{thr}(d)}$ the token-weighted mean mix of every other
document in its thread except the parent. The two contexts do not overlap, so $a_p$ weights the
parent and $a_t$ weights everyone else in the thread. This is the posterior mean of $\theta_d$
under a Dirichlet prior centered on the reply context, with pseudo-counts $a_p$ and $a_t$. A
three-token reply is dominated by its context; a three-hundred-token reply barely moves.

!!! warning "Experimental"
    `ThreadTM` is an original construction validated by planted recovery (see
    `tests/test_threads.py`), with no published reference yet. Call
    `topica.enable_experimental()` first. It may change without a deprecation cycle.

## Fit

```python
import topica

topica.enable_experimental()
data = topica.datasets.load_threads()
docs, parents = data.documents, data.parents

model = topica.ThreadTM(20, seed=13)                 # base="lda"
model.fit(docs, parents, iters=1000, corpus_kwargs={"min_cf": 5})
print(model)
model.top_words(10)                                  # the base model's topics
theta = model.doc_topic                              # (D, K) smoothed, one row per document
```

For an STM base, pass the prevalence design with one row per input document; `ThreadTM` keeps
its rows aligned with the documents the corpus keeps and uses five best-bound restarts by
default:

```python
X, names = topica.one_hot(community)
model = topica.ThreadTM(20, base="stm", seed=13).fit(docs, parents, prevalence=X)
```

The pseudo-counts are estimated, not assumed. `fit` holds out half the tokens of eligible leaf
replies, fits the base on the rest, chooses $(a_p, a_t)$ by held-out log likelihood on
validation threads, and reports every number from separate test threads. It then fits the base
on the full corpus and smooths that fit.

## Read

| Attribute | Meaning |
|---|---|
| `alpha`, `alpha_ci` | the pseudo-counts $a_p$ and $a_t$, with 95% intervals |
| `parent_share`, `parent_share_ci` | $a_p / (a_p + a_t)$: how dyadic the conversation is |
| `edge_effect` | held-out gain of the true parent over a shuffled parent, nats per token |
| `completion` | held-out gain over the base, overall and by reply-length tercile |
| `alpha_by_group` | per-community pseudo-counts when `groups=` is passed |

`edge_effect` is the placebo-netted measure of what the *specific* parent adds. The placebo
permutes parent assignments within (thread, depth), which keeps every reply's depth and thread
and every parent's number of children, and changes only which comment each reply answers.

## Uncertainty

Intervals are thread-bootstrap percentiles: validation threads are resampled and the
pseudo-counts re-chosen for each draw, and test threads are resampled for the held-out gains.
The parent share is searched on a grid that is evenly spaced on the logit scale and stops just
short of 0 and 1, so the estimate and its interval stay inside the unit interval (a very weak
prior against the boundary). `parent_share_at_bound` flags an estimate at the outermost grid
value, where the data cannot tell the smaller pseudo-count from zero.

A single calibration holds the base fit and the held-out mask fixed. `fit(..., n_refit=R)`
repeats the calibration `R` more times with new masks and base seeds and pools the bootstrap
draws, so intervals also reflect masking and base-fit variation. Use it for any number you
report; `replicates` lists each calibration's point estimates. The held-out effects
(`completion`, `edge_effect`, `op_effect`) are then reported as the median over calibrations,
so an effect and its interval describe the same pooled distribution. The parameters (`alpha`,
`parent_share`, `rho`) stay the first calibration's, because those are the values the smoother
applies; their intervals pool every calibration. `fit` warns when the calibration rests on fewer than 200 evaluation
leaves or 2,000 held-out test tokens; `summary()["settings"]` has the counts.

`draws` is a dict of NumPy arrays (`alpha`, `parent_share`, `completion`, `edge_effect`, and
`rho` and `op_effect` when they apply). To compare two communities, fit each separately and
difference the draws:

```python
a = topica.ThreadTM(20, seed=13).fit(docs_a, parents_a, n_refit=4)
b = topica.ThreadTM(20, seed=13).fit(docs_b, parents_b, n_refit=4)
diff = a.draws["edge_effect"] - b.draws["edge_effect"]
np.percentile(diff, [2.5, 50, 97.5])
# parent_share draws are NaN where a draw chose no borrowing: drop those first.
```

The two fits have their own topics, so compare the tree quantities (shares, effects, `rho`),
not topic-level numbers.

## When only some replies inherit: `switch=True`

One pseudo-count per context borrows the same amount for every reply of a given length. When
some replies take up their context's topics and others turn to different ones, that single
weight helps the first group and hurts the second, and it can settle near zero even though
many replies inherit. `switch=True` lets each reply's own words decide whether it borrows:

```python
model = topica.ThreadTM(20, seed=13, switch=True).fit(docs, parents, iters=1000)
model.rho               # prior probability that a reply inherits, with rho_ci
model.inherit_weights   # (D,) approximate inherit probability per reply
```

Each reply's topic mix gets a two-component mixture prior. The *inherit* component is the
pooled shrinkage above: a Dirichlet centered on the context mix, weighted by the pseudo-counts.
The *new* component is centered on the corpus mean mix. The reply's inherit probability $w_d$
is the (approximate) posterior probability of the inherit component given its observed
tokens, and

$$
\tilde\theta_d = w_d\,\frac{n_d\theta_d + a_p\theta_{\mathrm{par}(d)} + a_t\theta_{\mathrm{thr}(d)}}
                            {n_d + a_p + a_t} + (1 - w_d)\,\theta_d .
$$

The component probabilities use an approximation to the Dirichlet-multinomial marginal
likelihood of the reply's tokens with the topics held fixed: a sequential Polya urn that
predicts each token from the ones before it, carrying soft topic counts. It is exact for two
tokens; beyond that it is an approximation and depends on token order (we use the reply's own
order, so results are reproducible). Because the reply's own fitted $\theta_d$ never enters,
it cannot vouch for the words it was fit to. (A likelihood ratio built from $\theta_d$ would
always favor "new".) Read `inherit_weights` as a ranking of replies and its mean as an
approximate inherit rate, not as calibrated posterior probabilities. The pseudo-counts, the
innovate concentration and $\rho$ are chosen by held-out log likelihood on validation threads,
and `completion`, `edge_effect` and `op_effect` are reported on test threads, as in the pooled
fit. Bootstrap draws re-choose the parameters on resampled validation threads; the held-out
gains are evaluated at the point estimate and resampled over test threads, so their intervals
are conditional on the calibration (use `n_refit` to widen them for masking and base-fit
variation). A pseudo-count at the top of its range (1000) means inheriting replies take their
context nearly wholesale. `alpha` and
`parent_share` describe the inherit component: among replies that inherit, how much comes from
the parent. The mean of `inherit_weights` over replies estimates the share that inherit at all.

### Reading the switch's numbers

| Quantity | What it answers |
|---|---|
| `rho`, `rho_ci` | The model's estimate of the share of replies that inherit their context: the prior weight of the inherit component, chosen by held-out fit. Report this, with its interval, as a model-based estimate rather than a count of replies. |
| `alpha`, `parent_share` | Among replies that inherit, how much of the borrowing comes from the parent (versus the thread or the original post)? A parent share of 0.98 with `rho` of 0.55 means about half the replies inherit, and those that do take up their parent. |
| `inherit_weights` | Which replies inherit? A ranking, NaN for roots and empty replies. A short reply carries little evidence, so its weight stays near `rho`; only longer replies are classified with confidence. |
| `inherit_rate` | The mean of `inherit_weights` over replies. It leans toward `rho` in communities of short comments and has no interval. |
| `strength_at_bound` | The total pseudo-count is at the top of its range (1000): inheriting replies take their context nearly wholesale. |

The context shares are searched on a coarse lattice, so equal pseudo-counts across contexts
(for example a parent share of exactly 1/3 with three contexts) usually mean "not
distinguishable at this resolution", not a finding of equality. The switch's solution can
move with the base fit: check it across `iters` and seeds, or use `n_refit`.

On the planted simulator in `tests/test_threads.py`, the pooled fit borrows nothing when half
the replies inherit, while the switch recovers a positive edge effect and a mean inherit
weight near one half. We use one inherit component over all contexts rather than one component
per context: when each reply must pick a single context, a long, precisely estimated original
post outcompetes a short, noisy parent even for replies that answer the parent, and the placebo
contrasts stop separating who a reply answers.

The switch costs roughly two to four times the pooled fit and does not yet support `groups=`.

## The original post as a context

`contexts=("parent", "op", "thread")` separates the original post (the thread root) from the
rest of the thread, with its own pseudo-count $a_o$:

$$
\tilde\theta_d = \frac{n_d\theta_d + a_p\theta_{\mathrm{par}(d)} + a_o\theta_{\mathrm{op}(d)}
  + a_t\theta_{\mathrm{thr}(d)}}{n_d + a_p + a_o + a_t}
$$

The thread mix then excludes the parent and the root. A top-level reply's parent *is* the root,
so it borrows from its parent and the rest of the thread, with no separate original-post term;
its $a_o$ is simply dropped, not passed to the other contexts. `op_effect` compares two
matched fits: in both, the root and one random other comment of the same thread are left out
of the thread mix; one uses the root as the original post and the other uses that comment. It
asks whether the original post predicts replies better than a same-thread comment in the same
slot. A reply whose root the vocabulary emptied has no original post in either arm. It works
with `switch=True` as well.

Read a positive `op_effect` with care. When the original post itself carries the thread's
topic, "replies follow the thread" and "replies answer the original post" produce the same
data, and a long post is simply the most precise estimate of that topic. The OP effect is
cleanly interpretable when the original post and the discussion can diverge.

In the pooled fit, the parent and original-post pseudo-counts compete when the two are
correlated: on one Ask community we tested, the pooled fit moved all the weight to the original
post and set $a_p$ to zero, while the switch kept both. Compare `alpha` with and without `"op"`
before reading a drop in $a_p$ as a finding.

## Strip quotes first

Quoted text makes a reply look like its parent for reasons that have nothing to do with topical
uptake. Strip it before fitting:

```python
from topica import threads

texts = [threads.strip_quotes(t) for t in raw_texts]        # ">" and "&gt;" lines, <i> spans
docs = [topica.tokenize(t, stopwords="english") for t in texts]
docs = threads.strip_copied_runs(docs, parents, n=5)        # unmarked verbatim copying
```

## Smoothing a model you already fit

`topica.threads.ThreadSmoother` is the layer underneath. Give it a factory
`base(corpus)` (or `base(corpus, seed)`) and it calibrates and smooths any model with
`doc_topic` and `topic_word`; `transform(model, corpus, parents)` applies fitted pseudo-counts
to another fit over the same documents.

## Relation to other threaded models

[`CSATM`](models.md#csatm) also smooths each comment toward its ancestors after fitting, with a
fixed distance decay. `ThreadTM` estimates how much to borrow on held-out replies, separates the
parent from the rest of the thread, and nets the parent's contribution against a placebo.
[`TreeFieldTM`](models.md#treefieldtm) builds the reply tree into a logistic-normal prior and
estimates it jointly; it was topica's earlier threaded model under the name `ThreadTM`.

## Limits

- Without `switch=True`, one pseudo-count per context borrows for every reply alike. When
  only some replies take up their parent's topics and the others turn to sharply different
  ones, borrowing can cost the second group as much as it helps the first, and the fit returns
  little or no borrowing. Read a small pseudo-count as "borrowing does not help on average",
  not "no reply follows its parent". `completion["by_length"]` shows where borrowing helps and
  hurts. A negative tercile among short or middle-length replies is a sign to try the switch;
  a negative longest tercile often persists under it (long replies barely borrow either way).
- Pseudo-counts are calibrated on leaf replies and applied to every document. For an internal
  comment, the thread mix includes its own replies.
- `doc_topic` is a topic measure. Reuse of the parent's own words is a separate construct and
  is not folded in.
