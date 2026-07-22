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

## Recording evidence

`record_fit(..., diagnostics=["coherence", "exclusivity"])` computes those
built-in topic-quality metrics and records each as *computed evidence* (its mean
over topics), or `value=None` with a note when a metric is not defined for the
model. You can also attach your own with `record.add_diagnostic(name, value)` and
interpretive notes with `record.add_decision(key, note)`; both are stored as
visibly researcher-authored, never as tool-verified conclusions.

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

## Bundling the analysis

`record.bundle(path, model=model, corpus=corpus)` writes a self-contained,
**content-addressed** `.zip`: the manifest plus the saved artifacts, each file
named by its own hash, as one shareable, self-verifying unit.

```python
record.bundle("analysis.zip", model=model)                 # manifest + model
record.bundle("analysis.zip", model=model, corpus=corpus, include_corpus=True)
```

`AnalysisManifest.load_bundle(path)` reloads the manifest and **verifies every
artifact's bytes** against its content-addressed name and the manifest reference,
raising on a corrupt or tampered bundle. `AnalysisManifest.extract_bundle(path,
dest)` writes the artifacts out so you can reload them (`topica.LDA.load(...)`).
Bundling the corpus is opt-in and **sensitive** — it embeds the raw tokens.

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

::: topica.manifest.AnalysisManifest

::: topica.manifest.VerifyResult

::: topica.manifest.ManifestDiff

::: topica.manifest.fingerprint_corpus

::: topica.manifest.fingerprint_array

::: topica.manifest.fingerprint_design
