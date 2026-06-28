# A best-practice workflow for publishable topic models

The rest of this site shows **what topica can do**. This section is about
what you **should** do if you want a topic-modeling analysis to survive peer
review in sociology, political science, communication, or a related field.

The single biggest critique of topic modeling in the social sciences is that it
can become a *fishing expedition*: run a model, read the topics, tell a story.
Reviewers know this, and they push back. A publishable analysis answers their
objections before they raise them. It treats the topic model as a **measurement
instrument** that must be specified, validated, and used with well-calibrated uncertainty.

This is a six-step workflow. Each step maps onto specific topica functions.

<div class="grid cards" markdown>

- :material-database: **[1. Build a defensible corpus](corpus.md)**

    Document the population, the unit of analysis, and every preprocessing
    choice. Split long documents; justify the vocabulary.

- :material-shape: **[2. Pick the right model](choosing-model.md)**

    Choosing a model is a substantive decision. Match it to your question and
    data; justify it and check robustness to alternatives.

- :material-numeric: **[3. Choose and justify K](choosing-k.md)**

    `K` is a research decision, not a tuning parameter. Anchor it in theory,
    scan a range with `search_k`, and report sensitivity.

- :material-check-decagram: **[4. Validate the topics](validation.md)**

    Prove the topics are real and reproducible: intrusion tests, the
    coherence–exclusivity frontier, bootstrap stability, and close reading.

- :material-chart-line: **[5. Measure effects properly](effects.md)**

    Relate topics to covariates with well-calibrated uncertainty: the method of
    composition, **clustered standard errors**, and bounded GLM links.

- :material-file-document-check: **[6. Report and make reproducible](reporting.md)**

    A methods section, a topic table, the right supplementary materials, and a
    seed-fixed, scriptable pipeline.

</div>

## Three worked examples, together covering it all

No single analysis shows every technique, but the [worked examples](../examples/dubois.md)
**together** demonstrate the whole workflow on real, redistributable data:

| Workflow step | [Du Bois](../examples/dubois.md) | [Gadarian](../examples/gadarian.md) | [Poliblog](../examples/poliblog.md) |
|---------------|:------:|:------:|:------:|
| Corpus building & cleaning | ●●● | ● | ●● |
| Model choice (why this one) | ●● | ●●● | ●● |
| Choosing & justifying K | ● | ● | ●●● |
| Topic validation | ● | ●● | ●●● |
| Effects (method of composition) | ●● | ●●● | ●●● |
| **Clustered SEs** (nested data) | – | – | ●●● |
| Temporal / dynamic topics | ●●● | – | ● |
| Reporting & reproducibility | ●● | ●● | ●● |

## The short version

!!! tip "A reviewer-proof checklist"

    - [ ] The corpus population, unit, and time span are stated, with counts.
    - [ ] Every preprocessing step (tokenization, stopwords, pruning, phrases,
          document splitting) is documented and motivated.
    - [ ] `K` is justified theoretically **and** by a `search_k` scan; results are
          shown to be robust to nearby `K`.
    - [ ] Topics are validated by humans (word/document **intrusion tests**) and
          by metrics (**coherence + exclusivity**).
    - [ ] Topic **stability** under resampling is reported; fragile topics are
          flagged, not hidden.
    - [ ] Every topic has a substantive **label** and **representative quotes**
          (close reading, not just top words).
    - [ ] Covariate effects use **proper uncertainty** (method of composition)
          and **clustered standard errors** when documents are nested.
    - [ ] The full pipeline is **seed-fixed** and shared; topic-word and
          document-topic matrices are in the replication archive.

Treat the model's output as *"this document has high probability for Topic 3"*,
never *"this document **is about** Topic 3."* Topics are statistical patterns of
word co-occurrence: sometimes coherent concepts, sometimes not. Your job is to
demonstrate which.
