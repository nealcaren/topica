# Party platforms: how the words of a topic move

This worked example uses [`ECTM`](../guides/models.md#ectm), the Evolving Content
Topic Model, to ask a question most topic models cannot: not just *which* topics a
group discusses, but *how it words them*, and how that wording **changes over
time**. The corpus is every Democratic and Republican national platform from 1948
to 2024. We treat the party as the content **group** and the election year as the
**period**, then read how each party's vocabulary on a shared topic evolves across
twenty elections.

!!! warning "ECTM is experimental"
    ECTM has no published paper or reference-parity check yet, so it is gated.
    Call `topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) once
    before constructing it, as the fit cell below does. Read the results as
    provisional.

!!! info "Focus of this example"
    Content covariates that drift over time · the content vs prevalence split ·
    analytic and bootstrap uncertainty · why clustered resampling matters. For
    prevalence effects with clustered SEs see [Political blogs](poliblog.md); for
    a single topic's vocabulary drift without groups see the dynamic models
    ([`DTM`](../guides/models.md#dtm)).

    Data ships gzipped in
    [`examples/platforms_data/`](https://github.com/nealcaren/topica/tree/main/examples);
    rebuild it from the American Presidency Project with
    [`examples/prep_platforms.py`](https://github.com/nealcaren/topica/blob/main/examples/prep_platforms.py).
    The full script is
    [`examples/ectm_platforms.py`](https://github.com/nealcaren/topica/blob/main/examples/ectm_platforms.py).

## 1. The corpus

Each platform is a long document, so we split it into paragraphs. That gives many
documents per (party, year) cell, which the model needs, and it makes the unit of
text a single argument rather than a whole platform. The result is about 11,000
paragraph-documents, both parties present at all twenty elections.

```python
import gzip, json, re, numpy as np, topica
from collections import Counter

rows = json.load(gzip.open("examples/platforms_data/platforms.json.gz", "rt"))
docs   = [re.findall(r"[a-z]{3,}", r["text"].lower()) for r in rows]
party  = [r["party"] for r in rows]   # "D" / "R"
year   = [int(r["year"]) for r in rows]
print(len(docs), "paragraphs |", len(set(year)), "elections", min(year), "-", max(year))
```

```
11385 paragraphs | 20 elections 1948 - 2024
```

## 2. Fit ECTM

ECTM separates two outcomes, and answers both in one fit. The **content** side
(which words a party uses for a topic, and how that drifts) is the topic-word
model. The **prevalence** side (how *often* a party discusses a topic) is the
standard logistic-normal regression, which we drive with a `prevalence` design so
attention can depend on party and a smooth year trend. Pass the party as
`content`, the year as `times`, and the design as `prevalence`:

```python
party_col, pn = topica.one_hot(party)                       # indicator(Republican)
yr_basis,  sn = topica.spline(np.asarray(year, float), df=4)
inter,     _  = topica.interaction(party_col, yr_basis, name="party_year")
X = np.column_stack([party_col, yr_basis, inter])

topica.enable_experimental()   # ECTM is experimental and gated; opt in first
model = topica.models.ECTM(num_topics=18, seed=1)
model.fit(docs, times=year, content=party,
          prevalence=X, prevalence_names=list(pn) + list(sn) + ["party_year"],
          iters=150, period_smooth=6.0, interaction_shrink=1.2)
```

`period_smooth` is the random-walk precision that ties adjacent elections, so a
party's wording in 2016 borrows strength from 2012 and 2020 rather than being
estimated from one platform in isolation. `interaction_shrink` pulls the
group-by-time term toward zero unless the data earn it. After fitting,
`model.content_word_dist(group, period)` returns the topic-word matrix for any
cell, and the [`topica.ectm`](../api/models.md) helpers read it.

## 3. Topic one: the environment, where a word is born

The most partisan-worded topic about the land tells a clean story. In 1948 the two
parties speak the **same** language of stewardship; over seventy-six years the
Democratic vocabulary migrates while the Republican stays put.

```python
from topica.ectm import content_contrast
env = 15   # located by its climate mass; see the script
for yr in (1948, 1988, 2024):
    c = content_contrast(model, env, "D", "R", str(yr), n=5)
    print(yr, "D:", [w for w, _ in c["toward_D"]], " R:", [w for w, _ in c["toward_R"]])
```

```
1948 D: [resources, development, conservation, natural]  R: [veterans, service, federal, management]
1988 D: [environmental, water, protect, air, toxic]      R: [federal, veterans, trust, system]
2024 D: [climate, clean, energy, communities, pollution] R: [federal, service, veterans, programs]
```

Trace single words across the twenty elections with `content_trajectory`, on the
probability scale (the `D` minus `R` gap, per mille):

```
"climate"       D-R:  '48 +2   '84 +2   '96 +4   '08 +21   '20 +30
"conservation"  D-R:  '48 +11  ··················→  '84..'20 ~+1
"clean"         D-R:  '48 +3   ··················→  '20 +18
```

`conservation`, the shared 1948 word, fades to nothing. `climate`, absent and
non-partisan for fifty years, erupts after 2000 into a Democratic signature that
Republicans never adopt. Now the **other** half:

```python
from topica.ectm import prevalence_by_group
att = prevalence_by_group(model, party, year, topic=env) * 100   # % of platform
# D:  6  6  5  5  4  4  7      R:  5  5  4  4  5  5  (no 2024 GOP platform was issued in 2020)
```

Attention is flat and roughly equal the whole time, near five percent for both
parties. **Nobody started discussing the environment more.** They started
discussing it in incompatible words. A prevalence-only model (STM, keyATM, DMR)
sees only that flat line and would call the environment unpolarized. The entire
signal lives in the content.

## 4. Topic two: civil rights, where the words converge

The second topic is the mirror image, and it shows why you need both halves. Here
the distinctive words *converge* over time, so a content-only reading would say
the topic is de-polarizing. The real movement is in attention.

```
1948  D: [truman, equal, right]        R: [state, federal, local, governments]
1988  D: [discrimination, equal, rights] R: [family, state, oppose, right]
2024  D: [rights, freedom, equal, voting] R: [state, oppose, federal, law]

"equal"   D-R:  '48 +19  '60 +12  '84 +13  '96 +8  '20 +9     (the gap NARROWS)
Attention (% of platform):
   Democrats:  '48 14   '60 11   '72 12   '84 8   '96 3   '08 6   '20 7
   Republicans:'48  9   '60 11   '72  8   '84 8   '96 10  '08 11
```

The 1948 Democratic fingerprint is `truman` and `equal`, the year Truman
desegregated the military and split his party over civil rights: the model puts
the history right on the surface. Democrats own `equal` and `discrimination` from
the start, but the `equal` gap **erodes** from +19 to +9 as equality becomes
consensus language both sides claim. Meanwhile the Republican frame, `state,
federal, local, oppose, law`, barely moves in seventy-six years. And the attention
flips: Democrats foreground civil rights heavily in the Truman era (fourteen
percent of the platform), then it fades on their side while Republican attention
holds steady and ends higher. One cleavage lives in the words, the other in the
attention. ECTM is the model that holds both up at once.

## 5. What does ECTM add over a cross-tab?

A fair question: `content_trajectory` is close to "share of `climate` among this
party's tokens in this topic this year," which a `groupby` approximates. The
difference shows up exactly where it should, in the sparse early cells:

```
'climate' share for Democrats (per mille):  '48   '64   '80   '96   '12
   naive within-topic cross-tab:            0.0   0.0   0.0   2.6   21.3
   ECTM (random-walk pooled):               1.8   1.6   1.5   3.2   21.9
```

The raw cross-tab is `0.0` in the early decades: `climate` never appears, so the
contrast is undefined and a log-odds is impossible. ECTM partial-pools across
adjacent elections to a small, stable estimate, isolates the word within the
topic, and (next section) comes with uncertainty. It is the regularized,
inferential version of the cross-tab, agreeing with it where data exist.

## 6. Uncertainty, and why you must cluster

The content estimates are point values; put error bars on them two ways. The fast
analytic screen reports per-word standard errors from each cell's effective token
count:

```python
from topica.ectm import content_contrast_se
dl = [len(d) for d in docs]
for w, c, se in content_contrast_se(model, env, "D", "R", "2016", party, year, dl, n=4):
    print(f"{w:10} {1000*c:+6.1f} +/- {1000*se:4.1f}  z={c/se:+.1f}")
```

```
climate    +48.4 +/-  8.5  z=+5.7
change     +25.2 +/-  6.2  z=+4.1
clean      +21.7 +/-  5.8  z=+3.8
federal    -61.8 +/- 10.3  z=-6.0
```

That looks decisive, but it counts each **token** as independent. Our paragraphs
nest in platforms (about twenty per party, one per election), so the real number
of independent units is far smaller. `content_trajectory_ci` resamples whole
**platforms** and refits, giving the correct band:

```python
from topica.ectm import content_trajectory_ci
def refit(d, g, p):
    m = topica.models.ECTM(num_topics=18, seed=1)
    m.fit(d, times=p, content=g, iters=90, period_smooth=6.0, interaction_shrink=1.2)
    return m
anchor = [w for w, _ in model.top_words(20, topic=env)]
band = content_trajectory_ci(refit, docs, party, year, anchor_words=anchor,
                             word="climate", contrast=("D", "R"),
                             clusters=list(zip(party, year)), n_boot=20, seed=0)
```

```
'climate' D-R contrast (per mille), 95% CI:
              platform-clustered (correct)   paragraph-level (too tight)
   2000:       2.3  [-0.1,  6.8]             0.9  [-0.7,  4.7]
   2016:      14.9  [ 0.1, 44.1]             6.3  [-0.0, 29.0]
   2020:      18.4  [ 1.9, 40.5]             6.3  [-0.9, 22.8]
```

Two lessons. First, clustering by platform gives **much wider** bands than
resampling paragraphs: bootstrapping paragraphs treats correlated text as
independent and understates uncertainty. Second, under correct platform-level
resampling the `climate` gap is large and directionally robust but only
**separates from zero in the last decade**, not the 1990s. The analytic z-scores
above, which assume token independence, overstate the certainty for this corpus.
The arc is real; the precise confidence on any single election is wide, because
there are only about twenty platforms per party behind seventy-six years.

!!! warning "Effective sample size, not token count"
    With clustered text, report the cluster bootstrap, not the analytic
    per-word SE. `content_contrast_se` is a fast screen for *which* words separate
    the groups; it assumes token independence and so overstates precision when
    documents nest. The number of independent units here is the number of
    platforms, and the bands reflect it.

### Is a divergence real, or just the floor?

The cluster bootstrap puts a band on a single word. The headline ECTM quantity is
the whole-topic **content divergence** between two groups (the total-variation
distance between their word distributions, via `content_divergence`). Because each
(group, period) cell carries its own parameters, that divergence has a
finite-sample floor above zero even when the groups are identical: estimate two
distributions from finite text and they will differ a little by chance. The guard
is a permutation placebo. `content_placebo` shuffles the group labels within each
period (preserving each period's composition), refits, and recomputes the
divergence, building the null the floor is read from:

```python
from topica.ectm import content_placebo
res = content_placebo(model, docs, party, year, n_perm=200, seed=0)
for r in res.as_dict():
    print(f"{r['topic_name']:14} obs={r['observed']:.3f}  floor={r['floor']:.3f}  p={r['pvalue']:.3f}")
```

```
environment    obs=0.118  floor=0.041  p=0.005
civil_rights   obs=0.052  floor=0.039  p=0.180
```

The environment topic's divergence clears its floor by a wide margin (the two
parties really do word it differently); civil rights sits in the bulk of the
null, divergence indistinguishable from the finite-sample artifact. This is the
content-side counterpart of `topica.permutation_test`, which does the same shuffle
to test *prevalence* rather than wording. Use `content_placebo` to establish a
divergence is real before reading its trajectory.

## 7. Scaling to large corpora

The fit above is full-batch variational EM: every iteration touches every
document. That is fine for eleven thousand paragraphs, but it does not scale to
the hundreds of thousands or millions of documents in a congressional or
social-media corpus, where one batch iteration is already expensive and dozens are
needed. For those, pass `inference="svi"` to switch to minibatch stochastic
variational inference:

```python
model = topica.models.ECTM(num_topics=18, seed=1)
model.fit(docs, times=year, content=party, prevalence=X,
          iters=8,                 # epochs (passes over the corpus), not batch EM steps
          inference="svi", batch_size=2048, tau=64.0, kappa=0.7,
          period_smooth=6.0, interaction_shrink=1.2)
```

Each step samples `batch_size` documents, runs the same Laplace E-step on just
those, scales their sufficient statistics up to the full corpus, and nudges every
global parameter toward the minibatch estimate with a decaying step size
`(tau + step)^(-kappa)`. `iters` now counts **epochs** (passes over the corpus);
`tau` down-weights the noisy early minibatches and `kappa` in `(0.5, 1]` sets how
fast the step size decays. Larger `batch_size` gives steadier steps at higher
per-step cost.

ECTM's content (topic-word) M-step is far more expensive than the cheap
mean/covariance/prevalence updates, so it is re-solved only every `content_every`
minibatches (default `0` = once per epoch) while the cheap globals update every
step. Once per epoch keeps an SVI epoch about as cheap as a batch iteration;
lowering `content_every` re-solves the content model more often for better
per-epoch fidelity at more cost. On a 96,000-speech congressional corpus (party ×
congress, K=30, vocabulary 25,000) eight SVI epochs recover a full batch fit's
topics and between-party content divergences to about 0.97 (matched cosine and
divergence-spectrum correlation) in a few minutes, well under the batch fit's
runtime.

Two things to know. First, the SVI fit is **seed-reproducible but not bit-exact**:
it draws its minibatches from the model seed, so a fixed seed reproduces a run
exactly, but a different seed gives a different (statistically equivalent) fit,
unlike the deterministic spectral batch fit. Second, SVI removes the need to
subsample, which matters for inference: a smaller corpus has smaller (group,
period) cells, and smaller cells raise the finite-sample floor of the estimated
content divergence, so fitting the full data keeps that floor as low as the data
allow. Use the batch fit for small corpora where it is affordable (it is
deterministic and needs no step-size tuning) and SVI when the corpus is too large
to fit in one piece.

Because too few content solves can leave the between-group divergences understated,
an SVI fit reports whether its content model settled. Check `model.content_converged`
(and the per-solve trace `model.content_shift_history`); a fit that has not
converged also emits a warning at fit time. If it reports `False`, raise `iters`
(epochs) or lower `content_every` and refit.

## 8. What to claim, and what not to

The defensible findings are the **shapes**. The environment is a cleavage that
opened inside the vocabulary: shared `conservation` language in 1948 giving way to
a Democratic `climate` vocabulary the other side never adopts, while attention
stays equal. Civil rights is the opposite shape: an old cleavage whose distinctive
words converge while the movement passes to attention and framing. Both shapes,
their directions, and their rough magnitudes survive every estimator we tried.

What not to over-read: these are platform paragraphs, heavily correlated within
each platform, so single-election wiggles are mostly noise (trust the smoothed
arcs), and topic boundaries drift over seventy-six years, so the civil-rights
attention numbers are suggestive rather than nailed down. ECTM gives you the two
surfaces, content and prevalence, each with uncertainty. Reading them carefully is
the analysis.
