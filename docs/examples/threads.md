# Threaded conversations (ThreadTM)

A worked analysis of **two subreddits** where the unit of analysis is not a
document but a *reply tree*. [`ThreadTM`](../api/models.md) is a logistic-normal
topic model with a reply-tree structured prior: a comment's topic prior is coupled
to the comment it answers, so the model can express *persistence* — the degree to
which a reply stays on its parent's topic. This example shows what that buys, and,
just as important, when it does not.

!!! info "Focus of this example"
    Fitting a topic model on **threaded** data (posts plus nested replies) ·
    reading **persistence** with its identifiability gate · learning why reply-tree
    depth is not the same thing as persistence.

    Data: [`topica.datasets.load_threads()`](../api/datasets.md#topica.datasets.load_threads)
    — two subreddits from ConvoKit's `reddit-corpus-small` (Chang et al. 2020):
    `askscience` (technical Q&A) and `pokemontrades` (trade coordination).
    `ThreadTM` is experimental, so call `topica.enable_experimental()` first.

## 1. Load a threaded corpus

Threaded data cannot go through [`from_dataframe`](../api/keywords.md): a flat text
table discards the reply tree. `load_threads` returns a `Bunch` whose rows stay
aligned to a `parents` index — the 0-based row of the comment each row answers, or
`-1` for a thread root. `documents` (tokenized, one per row) and `parents` line up,
so the fit is turnkey.

```python
import topica

topica.enable_experimental()          # ThreadTM is experimental
b = topica.datasets.load_threads()

len(b.documents)                      # 5042 comments
sum(p < 0 for p in b.parents)         # 171 reply trees
set(b.subreddit)                      # {'askscience', 'pokemontrades'}
```

Every non-root `parent` index is smaller than its child's row, so the array is safe
to pass straight to `fit`. `b.texts` holds the raw comment text if you want a
different vocabulary; keep every row so `parents` stays valid.

## 2. Fit each community on its own reply trees

Reply persistence is a property of a *community*, so we fit each subreddit on its
own trees. Subsetting has to remap the parent indices to the subset's row numbers:

```python
def subset(b, name, k=5):
    keep = [i for i, s in enumerate(b.subreddit) if s == name]
    remap = {old: new for new, old in enumerate(keep)}
    docs = [b.documents[i] for i in keep]
    parents = [-1 if b.parents[i] < 0 else remap.get(b.parents[i], -1) for i in keep]
    return topica.ThreadTM(k, coupling="parent", seed=13).fit(
        docs, parents=parents, min_count=5
    )

asksci = subset(b, "askscience")
trades = subset(b, "pokemontrades")
```

`askscience` recovers clean physical-science topics:

```text
T2: pilot aircraft missile radar like plane time sleep
T3: air like heat water temperature skin lightning why
T4: earth orbit planet speed sun pressure velocity gravity
```

## 3. Read persistence — and its identifiability gate

`persistence()` fits an internal *no-tree* pass (a plain logistic-normal model, so
a parent and child are estimated **independently**) and regresses each reply's topic
mix on its coupling neighbor's. The slope is `observed_persistence`; the crucial
companion is `reliability`, the share of a comment's estimated topic mix that is
signal rather than posterior noise. **When `reliability <= 0` the per-comment topic
estimates are mostly noise and the persistence slope is not identifiable — the
number is not a structural claim, however tight its interval.**

```python
asksci.persistence()
# observed_persistence  +0.594   observed_ci [+0.534, +0.635]
# reliability           +0.467        -> identified

trades.persistence()
# observed_persistence  +0.427   observed_ci [+0.389, +0.463]
# reliability           -0.308        -> NOT identifiable
```

Both slopes are positive with intervals that exclude zero. If you stopped at the
slope you would report persistence in both communities. But only `askscience`
clears the identifiability gate. In `askscience`, replies genuinely answer their
parent, so a comment's topic mix carries real signal and the reply tree tracks it.
In `pokemontrades`, the topic mix is dominated by boilerplate ("added you on DS",
"ready when you are") — little recoverable per-comment signal — so the coupling
slope, though "significant", is not a structural persistence.

## 4. Depth is not persistence

The twist: `pokemontrades` has the **deepest reply trees in the entire source
corpus** — a median depth of 8 against `askscience`'s 4 (and chains up to 60 deep).
Long chains, but each turn coordinates a trade rather than developing the topic. Deep threads do not imply
that topics persist down them — you have to read `reliability`, not tree shape.
This is the honest boundary of the model: `ThreadTM`'s reply-tree prior pays off
where the conversation is *contingent* (replies respond on topic), and its own
diagnostic tells you when a community is not that.

!!! tip "Always read `reliability` before claiming persistence"
    A positive, tight `observed_persistence` is not enough. Report it as a
    structural finding only when `reliability > 0`. On small single-community fits
    the gate often fails; pool more data or raise `min_count` to sharpen the
    per-comment estimates, and if the gate still fails, the honest conclusion is
    that persistence is not recoverable in that corpus.

## What ThreadTM adds over LDA/STM

LDA gives you the topics; STM adds *who* talks about them (the `subreddit`
covariate here works exactly as in STM). ThreadTM adds one more axis — whether the
conversation actually *responds* on a topic — and, unlike a raw parent-child
correlation, it corrects that estimate for measurement error and gates it on
identifiability. On real forum data the benefit is genre-dependent: it is a
descriptive instrument for contingency, not a universal predictive win. Read
`persistence()` honestly and it will tell you which of your communities it applies
to.
