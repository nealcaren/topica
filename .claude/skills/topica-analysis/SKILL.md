---
name: topica-analysis
description: Working guide for an LLM agent helping a social scientist run a topic-modeling analysis with the topica Python library — building a defensible corpus, choosing and justifying K, fitting a model, validating it, estimating covariate effects with honest uncertainty, and reporting results. Use when the user wants to USE topica to analyze a text corpus (not extend the library). The companion developer skill for adding a model is add-topic-model.
---

# Using topica well

This is the working guide for an LLM agent (Claude Code, Cursor, or similar)
helping a social scientist run topic models with `topica`. It is a working guide,
not an API reference; the API lives at <https://nealcaren.github.io/topica/>. Read
it before you write code that fits a model or reports a result. If instead you are
*extending* topica (adding a model, a diagnostic, or a binding), this is the wrong
guide — use the `add-topic-model` skill and `CONTRIBUTING-MODELS.md`.

The one idea to carry through everything below: **topica and the agent supply the
mechanics and the honest diagnostics; the researcher owns the theory.** A topic
model is a measurement instrument, not a discovery oracle. The decisions that
make a topic-model paper credible are theoretical, and they are the researcher's:
what the corpus represents, how many topics is the right granularity for the
question, what each topic means, which covariates belong in the model, and
whether a result is substantively interesting. Your job is to make those
decisions cheap to act on and hard to get wrong, to surface uncertainty the
researcher might otherwise miss, and to refuse to manufacture certainty that the
data does not support. When one of those theory-owned decisions comes up, name it
and hand it back. Do not quietly pick `K` because a metric peaked there.

## What topica is, in one paragraph

`topica` is a Python topic-modeling library with a Rust core. It brings the
models that used to be split across R (`stm`, `keyATM`) and Java (MALLET) into
one place, runs them fast, and makes every fit reproducible from a fixed seed.
The house import is `import topica`, no alias. The core needs only NumPy. More
than a dozen models share one shape: `model.fit(docs, ...)`, then `topic_word`
(the φ matrix, topics × vocabulary), `doc_topic` (the θ matrix, documents ×
topics), `top_words(n)`, and `save` / `load`. Diagnostics are model-agnostic:
they take any fitted model's `topic_word` / `doc_topic`.

Because the count-based variational models (`CTM`, `STM`, `DTM`,
`SupervisedLDA`) are deterministic to the bit and the Gibbs models are
seed-reproducible, "run it again and see if the topics hold" is a real test here,
not a hope. Use that.

## How topica changes the standard advice

If you have internalized the usual computational-text-analysis guidance ("STM is
R-only", "use BERTopic for the neural approach in Python", "covariates mean you
have to leave Python"), update it. With topica:

- **STM, CTM, SAGE, STS, DMR, and keyATM all run in Python**, validated against
  their reference implementations. A covariate-aware topic model, including
  covariate-driven *sentiment* (`STS`, the sentiment-discourse extension of STM),
  no longer forces a switch to R.
- **Classical and neural live side by side.** `BERTopic`, `Top2Vec`, `ETM`, and
  `FASTopic` take embedding vectors you supply (no PyTorch in the wheel) and sit
  next to LDA and STM behind the same diagnostics. You can compare them on one
  corpus without leaving the session.
- **Effect estimates carry honest uncertainty.** `estimate_effect` and
  `standard_errors` give you method-of-composition and bootstrap intervals, and
  they refuse to fabricate intervals for models that have no posterior to
  compose. Trust the refusal.

The choice of model still follows the research question, not the convenience of
the language.

## The workflow

These phases mirror how a careful text-analysis paper is actually built. At each
**handoff** the researcher makes a call you should not make for them. Stop, show
what you have, and ask.

### Phase 0: research question and method

Before any code, get the question into one of three shapes, because they need
different tools:

- **Descriptive / exploratory** ("what themes are in this corpus?") → a topic
  model: `LDA` to start, `STM` if covariates matter, `HDP` if you genuinely do
  not want to fix `K`.
- **Measurement** ("how much does theme X vary by group/time?") → `STM` or
  `DMR`, with the grouping or time variable as a prevalence covariate, plus
  `estimate_effect`. If the question is about the *tone* of a theme rather than
  its prevalence (how positively or negatively a theme is discussed, and how that
  varies by covariate), `STS` adds a sentiment-discourse latent on top of STM.
- **Confirmatory** ("do documents express concept X?") → often *not* a topic
  model. A dictionary, `KeyATM` / `SeededLDA` (guided by seed words), or a
  classifier fits a known target better than unsupervised topics do.

Topic models are a poor fit for very short, formulaic, or highly technical text,
and for questions that need crisp category boundaries. Say so when you see one.
For short text (tweets, survey answers, headlines) reach for `GSDMM` or `PT`, not
LDA.

> **Handoff.** The question framing and the model choice are the researcher's.
> Lay out the options with their trade-offs; let them choose.

### Phase 1: build a defensible corpus

The corpus defines the scope of every claim that follows, so preprocessing is a
substantive decision, not a chore. Use `topica`'s preprocessing tools and
**write down every choice** (stopwords, frequency thresholds, phrases), because
the methods section has to report them.

```python
import topica

corpus = topica.Corpus.from_documents(docs)   # docs: list of token lists
# or build from raw strings with tokenize(), learn_phrases()/apply_phrases()
```

Defaults worth stating explicitly to the researcher: lowercase, drop stopwords,
drop terms in fewer than ~5-10 documents and terms in a very large share of
documents. Prefer lemmatization to stemming (stemming hurts interpretability).
Start minimal and add preprocessing only when topics show artifacts; do not
pile on cleaning preemptively. Inspect the corpus (document count, length
distribution, vocabulary size) and report it before modeling.

> **Handoff.** Stopword lists and frequency cutoffs change what topics can
> exist. Surface them; do not bury them in a default.

### Phase 2: choose and justify K

This is where agents most often overstep. **K is a research decision about
granularity, not a hyperparameter to maximize.** K = 10 gives broad themes,
K = 50 gives fine distinctions, and several values of K are usually defensible
for the same corpus.

Do not pick the K that maximizes coherence. A model with fewer topics will often
score higher on coherence while collapsing distinctions the researcher cares
about. Use the diagnostics as evidence, not as an optimizer:

```python
rows = topica.search_k(docs, [10, 15, 20, 25, 30], iters=1000)
# coherence, exclusivity, and held-out perplexity per K
```

The honest procedure: start from a theoretically plausible K, fit a small range
around it, and for each K look at coherence **and** exclusivity **and** whether
every topic can be labeled. Count the junk topics. Then let the researcher
choose, and report sensitivity to that choice. `topica.viz.search_k(rows)` and
`quality_frontier` make the trade-off legible.

`rows.best_k()` exists, but treat it as one input, not the decision: with a
`held_out=` set it defaults to the held-out metric, which is roughly monotone in
K and so tends to return the **largest K you scanned** (it warns when it does).
Without held-out it defaults to the coherence/exclusivity frontier (a knee). Read
it as "the K this criterion prefers," report the criterion, and still hand the
choice to the researcher — never present `best_k()` as the answer.

> **Handoff.** Present the curves and the labeled topics at two or three values
> of K. The researcher picks K; you report why.

### Phase 3: fit

Always pass a seed. Always report it. For STM, covariates enter as a design
matrix (build it with `one_hot`, and `stm.spline` / `stm.interaction` for
non-linear and interaction terms), not a formula string:

```python
X, names = topica.one_hot(party)
model = topica.STM(num_topics=20, seed=42)
model.fit(docs, prevalence=X, prevalence_names=names)   # content=... for SAGE
```

A plain LDA fit is `topica.LDA(num_topics=20, seed=42).fit(docs, iters=1000)`.
Check that the fit converged (the EM models expose a bound / `converged` flag;
the Gibbs models expose log-likelihood history). A model that did not converge is
not a result. Read topics off several surfaces together — top words alone
underdetermine what a topic is:

- `model.top_words(n)` returns per-topic `(word, weight)` tuples, not bare
  strings; take just the words with `[w for w, _ in row]`.
- `label_topics`, `frex`, `relevance`, `find_thoughts`, `topic_table` and
  `summary` are **module-level functions, not model methods**, and take the
  model's matrices as the first argument — e.g.
  `topica.label_topics(model.topic_word, model.vocabulary)` and
  `topica.find_thoughts(model.doc_topic, texts, topic=t)` (the highest-θ
  documents for a topic).
- `label_topics` has no `method=` selector: it reports probability and FREX
  side by side. For FREX- or relevance-ranked words specifically, call the
  separate `topica.frex(model.topic_word, model.vocabulary)` or
  `topica.relevance(model.topic_word, model.vocabulary)`.

### Phase 4: validate (not optional)

Algorithmic output is not ground truth. Before any topic becomes a finding:

- **Stability.** Refit under different seeds and check the topics persist:
  `bootstrap_stability`. topica's determinism makes this a clean test.
- **Consensus across runs.** `select_model` fits several seeds and keeps the best
  by a quality score; `ensemble` goes further and combines independent runs into
  a consensus set of topics more reliable than any single fit. Reach for these
  when a topic has to be stable to carry a claim.
- **Coherence and exclusivity.** `coherence` (`u_mass`, `c_v`, `c_uci`,
  `c_npmi`), `exclusivity`, `topic_diversity`.
- **Human validation.** `word_intrusion` and `document_intrusion` build the
  intrusion tests that show a topic is coherent to a human, not just to the
  metric. Offer to generate them; the researcher (or their coders) runs them.
- **Robustness.** Different K, different preprocessing, a subset by time or
  source. Compare to an alternative model if the claim is strong.

A one-call summary is available:

```python
topica.diagnostics(model, texts)   # coherence, exclusivity, diversity table
```

> **Handoff.** Whether a topic is "real enough" to interpret is a judgment.
> Give the diagnostics; let the researcher draw the line, and report where they
> drew it.

### Phase 5: measure effects with honest uncertainty

If the question is about variation (over time, across groups), this is the
results figure. Use the method of composition, which propagates topic-estimation
uncertainty into the effect, and report intervals:

```python
from topica import stm
draws   = stm.posterior_theta_samples(model, nsims=50, seed=0)
effects = stm.estimate_effect(draws, X, feature_names=names, cluster=source_id)
# or, model-agnostic with the uncertainty kind made explicit:
se = topica.standard_errors(model, corpus, of="effect", method="composition",
                            X=X, feature_names=names)
```

`topica.viz.effect_plot(model, corpus, X=...)` draws the per-topic forest plot
and, importantly, **refuses to draw confidence bands for models that have no
posterior** (cluster models like BERTopic), drawing point estimates and saying so
instead. When you see that refusal, do not paper over it with a bootstrap unless
the researcher asks; the absence of a band is information.

### Phase 6: report

Topica produces publication-ready tables and figures (`topic_table`, `summary`,
the `topica.viz` panels and `dashboard`). The methods section must state: model,
K and the rationale, preprocessing, covariates, software version, seed, and the
validation done. The results section gives labeled topics with top words,
prevalence, covariate effects with intervals, and representative documents.

The single most important reporting discipline is language. Topics are
statistical patterns of word co-occurrence. They are **not** necessarily coherent
concepts, and they are not what a document is "about." Write:

- "Document *d* has high probability on Topic 3," not "Document *d* is about
  Topic 3."
- "Words associated with Topic 3 suggest ...," not "The model discovered that ..."
- "One interpretation of this pattern is ...," not a claim that the topic is a
  fact.

Findings apply to the analyzed corpus, not to "language" or "discourse" in
general.

## What the agent owns vs. what the researcher owns

| The agent / topica owns | The researcher owns |
|---|---|
| Running models, picking sensible defaults | The research question |
| Computing coherence, exclusivity, stability | What counts as good enough |
| Surfacing the K trade-off curves | **The choice of K** |
| Listing top words, FREX, example documents | **What each topic means** (the labels) |
| Proposing covariate specifications | **Which covariates belong in the model** |
| Reporting effect sizes with honest intervals | **Whether a result matters substantively** |
| Refusing to fabricate uncertainty | Theoretical interpretation and the argument |

When you find yourself about to make a call from the right column, stop and hand
it back. The failure mode to avoid is an agent that runs the whole pipeline,
picks K off a metric, auto-labels the topics, and presents a finished result that
looks authoritative and is theoretically ungrounded. A topica session done well
leaves the researcher having made every decision that a reviewer will question,
with the diagnostics to defend each one.

## Quick reference

- **Build:** `Corpus.from_documents`, `tokenize`, `learn_phrases` / `apply_phrases`
- **Models:** the names below are *illustrative, not the full roster* — topica ships
  ~30 models and the list grows, so do not treat this as complete. The canonical,
  always-current list is the generated roster in the docs
  (<https://nealcaren.github.io/topica/guides/models/>, from `docs/guides/models.md`)
  and the README table; consult it rather than relying on any hardcoded list here.
  Common starting points: `LDA`, `STM`, `CTM`, `STS` (sentiment), `DMR`, `HDP`,
  `KeyATM`, `SeededLDA`, `BERTopic`, `GSDMM` (short text)
- **Read topics:** `model.top_words(n)` (method → `(word, weight)` tuples) plus
  the module functions `topica.label_topics`, `frex`, `relevance`,
  `find_thoughts`, `topic_table`, `summary` (first arg is the model's
  `topic_word` / `doc_topic` matrix, or the model)
- **Choose K:** `search_k`, `quality_frontier`, `topica.viz.search_k`
- **Validate:** `coherence`, `exclusivity`, `topic_diversity`,
  `bootstrap_stability`, `word_intrusion`, `document_intrusion`, `diagnostics`
- **Reliability across runs:** `select_model`, `ensemble`
- **Effects:** `estimate_effect`, `standard_errors`, `posterior_theta_samples`,
  `topica.viz.effect_plot`
- **Compare corpora:** `fighting_words`
- **Report:** `topica.viz.dashboard`, `summary`, `topic_table`

Everything is documented at <https://nealcaren.github.io/topica/>. When in doubt
about a signature, read the docs or the type stub (`python/topica/_topica.pyi`);
do not guess.
