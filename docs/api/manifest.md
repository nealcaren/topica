# Analysis manifest

A portable, privacy-aware record of one fit: the inputs and decisions needed to
inspect or replay it, plus a verifier that reports what actually holds rather than
a single pass/fail. The manifest is not a second model format. It composes with
`Corpus.save` and `model.save`, and records fingerprints and settings, not raw
content.

We write the manifest with an explicit `record_fit` call. The explicit boundary
is deliberate: a call cannot honestly see preprocessing performed before it, so it
records only what it observes and marks the rest.

```python
import topica

corpus = topica.Corpus.from_documents(tokens)
model = topica.STM(num_topics=20, seed=42)
model.fit(corpus, prevalence=X, iters=1000)

record = topica.record_fit(model, corpus, prevalence=X, prevalence_names=names, iters=1000)
record.add_decision("K", "Chose 20 after inspecting 15/20/25.")
record.save("analysis.topica.json")

# On another machine or session:
record = topica.AnalysisManifest.load("analysis.topica.json")
print(record.verify(corpus, model).summary())
```

## Privacy

The default `privacy="minimal"` records the model, the environment, and coarse
corpus counts (document count, total tokens) only. It carries no length
distribution, vocabulary size, preprocessing detail, or raw content.
`privacy="aggregate"` opts those descriptive fields in. A content fingerprint,
which lets `verify` prove corpus identity, is a separate opt-in
(`content_fingerprint=True`) and is **sensitive**: a hash is not anonymisation,
and a small or public corpus can be brute-forced. `privacy="full"` (raw values)
is intentionally not part of this version.

## What `verify` reports

`verify` re-derives fingerprints from the supplied corpus and model and reports a
status per field, never collapsing them into one flag: `exact`, `input_changed`,
`artifact_changed`, `environment_changed` (differs, but the model's determinism
class says it may still replay), or `unverifiable` (nothing recorded to check
against, a bounded component, or a fingerprint from a spec this build does not
recognise).

An `unverifiable` field is an **absence of evidence, not a mismatch**, and the two
are kept apart. `result.differences` lists the fields that actually changed;
`result.unverifiable` lists the ones nothing could be checked against. This matters
on the default path: at `privacy="minimal"` no content fingerprint is recorded, so a
perfect re-verification still leaves `corpus_fingerprint` unverifiable. `ok` stays
`False` there — nothing is claimed as verified that was not — but the summary says
plainly that no differences were found, rather than reporting a clean reproduction as
a failure:

```
verify: no differences found (1 field unverifiable)
  environment         exact
  corpus_counts       exact
  corpus_fingerprint  unverifiable
  model_topic_word    exact
  model_doc_topic     exact
```

`ManifestDiff` draws the same distinction: `incomparable` (different fingerprint
specs, or a keyed digest) is an inability to tell, reported apart from the genuine
`changed` / `only_in_a` / `only_in_b` differences.

The determinism the manifest records is **config-aware** (issue #401), not the
coarse per-class registry tag: a `cvb0` sampler or an `init="random"` fit is
recorded as `seed-reproducible` even when the model class is nominally `bit-exact`,
and `determinism_detail` carries the machine-readable `replay_requires` (the `seed`,
plus `num_threads` for the collapsed-Gibbs approximate parallel sampler) and any
caveats. Compute it directly for any model with `topica.effective_determinism`.

## Recording evidence

`record_fit(..., diagnostics=["coherence", "exclusivity"])` computes those
built-in topic-quality metrics and records each as *computed evidence* (its mean
over topics), or `value=None` with a note when a metric is not defined for the
model. You can also attach your own with `record.add_diagnostic(name, value)` and
interpretive notes with `record.add_decision(key, note)`; both are stored as
visibly researcher-authored, never as tool-verified conclusions.

## Embedding-based models

An embedding+cluster model such as `BERTopic` is determined by the document
embeddings it was fit on, but the fitted model does not retain them, so `record_fit`
cannot see them on its own. Pass them so the fit stays verifiable and reproducible:

```python
model = topica.BERTopic(num_clusters=20, seed=0).fit(tokens, embeddings)
record = topica.record_fit(model, corpus, embeddings=embeddings)   # fingerprints them
```

The embeddings are recorded as an order-sensitive fingerprint under
`inputs["embeddings"]` (a hash, not the vectors). If you omit them for an
embedding-based model, their absence is recorded honestly — the manifest marks that
the determining input was not captured, rather than reading as a complete record.

## Comparing two fits

`a.compare(b)` diffs two manifests directly, with no corpus or model needed, and
reports per field `same` / `changed` / `only_in_a` / `only_in_b` / `incomparable`.
It answers "did these two runs use the same corpus, model, and inputs, or did
something change, and what?" — for example, whether a collaborator changed the
corpus or only the seed.

```python
a = topica.AnalysisManifest.load("run-a.json")
b = topica.AnalysisManifest.load("run-b.json")
print(a.compare(b).summary())
```

`a.compare(b)` is a *provenance* diff (same corpus/model/inputs?). To compare the
**topics themselves** — which are stable, which drifted, which are new — without the
models, record each fit with its top words retained and pass the two manifests to
[`topica.compare`](../guides/model-comparison.md#comparing-two-manifests-without-the-models):

```python
ra = topica.record_fit(model_a, corpus_a, topic_words_n=25)   # opt-in; off by default
cmp = topica.compare(ra, rb)                                  # Jaccard top-word alignment
```

Retain generously (`topic_words_n≈25`, not 10): top-word *set* overlap is coarser
than the live cosine, and seed-varying fits churn their top-word lists, so too small
a window can mislabel a stable topic as vanished/appeared (`compare` flags such
near-misses).

Top words are retained only when you ask (`topic_words_n>0`): they are
human-readable tokens drawn from the corpus vocabulary, so — like
`content_fingerprint` — they are an explicit opt-in and are off at every privacy
level, keeping the default manifest fingerprints-not-content.

## Bundling the analysis

`record.bundle(path, model=model, corpus=corpus)` writes a self-contained,
**content-addressed** `.zip`: the manifest plus the saved artifacts, each file
named by its own hash, as one shareable, self-verifying unit.

```python
record.bundle("analysis.zip", model=model)                 # manifest + model
record.bundle("analysis.zip", model=model, corpus=corpus, include_corpus=True)
```

`bundle` refuses to package a model or corpus whose fingerprints do not match the
manifest (guarding against bundling the wrong fit). `AnalysisManifest.load_bundle(path)`
reloads the manifest and **checks every artifact's bytes** against its
content-addressed name and the manifest reference, raising on a corrupt bundle or
artifacts that no longer match. This is an integrity / content-addressing check,
not authenticity — detecting a *fully rewritten* bundle (artifact, digest, and
reference all changed together) needs a signature, which is tracked separately.
`AnalysisManifest.extract_bundle(path, dest)` writes the artifacts out so you can
reload them (`topica.LDA.load(...)`). Bundling the corpus is opt-in and
**sensitive** — it embeds the raw tokens.

## The analysis card

`record.render(path)` writes a self-contained HTML **analysis card**, and
`record.to_markdown()` returns a Markdown version for a notebook or Quarto. The
card shows only what the manifest recorded, so it cannot over-claim: researcher
decisions are labelled as authored (not tool-verified), diagnostics as computed
evidence, and fingerprints as verifiable identity rather than content. Pass a
`VerifyResult` to include the graded verification table.

```python
record.render("analysis-card.html", verification=record.verify(corpus, model))
```

## Reference

::: topica.record_fit

::: topica.effective_determinism

::: topica.manifest.AnalysisManifest

::: topica.manifest.VerifyResult

::: topica.manifest.ManifestDiff

::: topica.manifest.fingerprint_corpus

::: topica.manifest.fingerprint_array

::: topica.manifest.fingerprint_design
