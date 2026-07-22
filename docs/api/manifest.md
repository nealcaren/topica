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

## Reference

::: topica.record_fit

::: topica.manifest.AnalysisManifest

::: topica.manifest.VerifyResult

::: topica.manifest.fingerprint_corpus

::: topica.manifest.fingerprint_array

::: topica.manifest.fingerprint_design
