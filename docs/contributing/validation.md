# Validation status and experimental graduation

Every model topica ships is validated before it enters the roster, but the
*evidence* behind that word varies. Some models are checked bit-for-bit against a
maintained reference implementation (MALLET, gensim, R `stm`, tomotopy, the
`bertopic`/`fastopic`/`corextopic` packages, a paper-faithful PyTorch or NumPy
port). Others have no runnable reference to check against, so they are validated
by *planted recovery*: we fit on a synthetic corpus with a known, identifiable
structure, freeze the solution, and assert the fit reproduces it and recovers the
planted topics. This page states the bar for graduating a model from the
experimental tier to the validated roster, and records where every model stands.

## The experimental tier

A model is **experimental** when it has not earned a place on the validated
roster, on either of two grounds:

- it is **unpublished** — a topica original with no paper, so it can only ever be
  validated by planted recovery, never against an independent yardstick; or
- it is a **published method whose benefit has not held up** — topica ships it,
  its inference is faithful, but it does not demonstrably beat a simpler baseline,
  so shipping it as validated would overstate what it buys a user.

A published method with faithful inference stays **validated** even when its
accuracy basis is planted-recovery, because for many methods no maintained
reference implementation exists to check against; planted-recovery is a
validation *basis*, not an experimental marker. So the experimental flag is about
whether the model has *earned the roster* (a paper and a demonstrated benefit),
not about code quality: an experimental model's inference can be faithful,
deterministic, and fully tested and still be experimental.

Experimental models are gated at construction (call
`topica.enable_experimental()`, or set `TOPICA_EXPERIMENTAL=1`) and listed apart
from the validated roster; they may change or be removed without a deprecation
cycle.

## The triple gate

A model graduates from experimental to validated only when it passes all three
gates. Each maps onto machinery topica already runs when a model is added.

1. **Accuracy gate.** Cross-implementation reference parity where a reference
   exists (the `parity/*_gold.py` / `*_compare.py` bar); planted-recovery plus the
   reference-free invariant suite (`tests/test_model_invariants.py`, issue #420)
   where none does, with the limitation documented. A planted gold that a fixed
   corpus cannot use to distinguish the model from a simpler baseline does not
   clear this gate on its own.
2. **Adversarial gate.** The `add-topic-model` dual review: one faithful-parity
   reviewer and one adversarial reviewer, run independently, with every finding
   in the code, the metadata, or the validation layer resolved.
3. **User gate.** The `sample-user` two-agent first-time-researcher usability
   audit, run on a real dataset end to end (vocabulary, choosing K, fit,
   robustness, validation, covariate effects, reporting), with the friction it
   surfaces filed and the blockers fixed.

All three green is the graduation criterion. A model that clears accuracy and
adversarial but not the user gate is validated-but-rough, not ready to graduate;
one that clears the user gate on a model whose planted gold cannot tell it apart
from a baseline has not cleared accuracy.

## Evidence levels

The audit below records each model on three axes.

- **Paper** — whether the method has a published paper. A *topica original* is a
  construction with no paper; it can only ever be validated by planted recovery,
  so it starts experimental by definition.
- **Validation basis** — how the accuracy gate is met:
    - *cross-impl* — checked against a maintained, runnable reference
      implementation.
    - *paper-oracle* — checked against a result computed directly from the
      paper's equations (a hand-written reference), not a maintained library.
    - *planted* — planted-recovery / self-consistency gold only; no external
      reference exists to benchmark against.
    - *behavioral* — exercised for correct orchestration but not numeric parity
      (the LLM-driven model, whose output is model-bounded).
- **Experimental** — whether the model is gated, and whether that status is
  justified by the first two columns.

*planted* is a defensible basis, not a defect: for most of the models that carry
it, no maintained or deterministic reference implementation exists (gensim,
tomotopy, and R do not implement SAGE, HDP as a fit-and-freeze target, HLDA, PA,
PT, SupervisedLDA, ETM, DETM, or Topics-over-Time in a form we can pin). The point
of the audit is honesty about the two evidence levels the single word "validated"
spans, not a claim that the planted-basis models are wrong; prior review deemed
their cores faithful.

## Per-model audit (topica 0.55.0)

Validated roster, by validation basis.

### Cross-implementation reference parity

| Model | Paper | Reference checked against |
|---|---|---|
| `LDA` | Blei et al. 2003 | Java MALLET |
| `OnlineLDA` | Hoffman et al. 2010 | gensim |
| `CTM` | Blei & Lafferty 2007 | R `stm` (as CTM) |
| `ProdLDA` | Srivastava & Sutton 2017 | PyTorch AVITM |
| `NMF` | Lee & Seung 1999 | scikit-learn |
| `LSA` | Deerwester et al. 1990 | scikit-learn |
| `AnchorLDA` | Arora et al. 2013 | `anchor-topic` (Q + given-anchor L2 recovery; the default KL + tempering is a topica extension, not reference-checked) |
| `PolylingualLDA` | Mimno et al. 2009 | topica cross-lingual parity harness |
| `STM` | Roberts et al. 2014 | R `stm` |
| `STS` | Chen & Mankad 2024 | R `stm` / `sts` |
| `DMR` | Mimno & McCallum 2008 | Java MALLET |
| `GDMR` | Terragni et al. 2020 | tomotopy |
| `RTM` | Chang & Blei 2010 | NumPy reference (variational + Gibbs) |
| `FactorialLDA` | Paul & Dredze 2012 | reference Java (finite-difference + planted recovery) |
| `KeyATM` | Eshima et al. 2024 | R `keyATM` |
| `SeededLDA` | Watanabe & Baturo 2024 | R `seededlda` |
| `GuidedNMF` | Vendrow et al. 2021 | `ssnmf` |
| `KeyNMF` | Kristensen-McLachlan 2024 | `turftopic` (method-level, corrected) |
| `CorEx` | Gallagher et al. 2017 | `corextopic` |
| `AuthorTopic` | Rosen-Zvi et al. 2004 | gensim |
| `MGLDA` | Titov & McDonald 2008 | tomotopy |
| `LabeledLDA` | Ramage et al. 2009 | Java MALLET |
| `BTM` | Yan et al. 2013 | reference BTM |
| `DTM` | Blei & Lafferty 2006 | gensim `LdaSeqModel` |
| `BERTopic` | Grootendorst 2022 | `bertopic` package |
| `Top2Vec` | Angelov 2020 | `bertopic` / `top2vec` |
| `SemanticSignalSeparation` | Kristensen-McLachlan et al. 2024 | `turftopic` |
| `GaussianLDA` | Das et al. 2015 | Java oracle |
| `Wordfish` | Slapin & Proksch 2008 | R |
| `TopicalNGrams` | Wang et al. 2007 | Java MALLET |
| `TBIP` | Vafa et al. 2020 | paper-derived reference |
| `PartyEmbeddings` | Rheault & Cochrane 2020 | comparison harness |
| `FASTopic` | Wu et al. 2024 | `fastopic` package |
| `CombinedTM` | Bianchi et al. 2021 | PyTorch AVITM |
| `ZeroShotTM` | Bianchi et al. 2021 | PyTorch AVITM |
| `InfoCTM` | Wu et al. 2023 | paper-derived PyTorch |

### Paper-oracle parity

| Model | Paper | Oracle |
|---|---|---|
| `DiscLDA` | Lacoste-Julien et al. 2008 | paper's 20-Newsgroups protocol, replicated |
| `Wordshoal` | Lauderdale & Herzog 2016 | from-paper R oracle (no installable reference) |

### Planted-recovery only (no external reference)

These are validated, but by planted recovery / self-consistency plus the invariant
suite, because no maintained reference implementation exists to benchmark against.
This is the two-evidence-levels gap issue #660 opened; it is recorded here rather
than hidden.

| Model | Paper | Note |
|---|---|---|
| `SAGE` | Eisenstein et al. 2011 | no gensim/tomotopy/R implementation |
| `HDP` | Teh et al. 2006 | Dirichlet-process mixture; concentration equations faithful to blei-lab/hdp |
| `HLDA` | Blei et al. 2003 | learns a topic tree; no fit-and-freeze reference |
| `PA` | Li & McCallum 2006 | super/sub-topic DAG |
| `PT` | Zuo et al. 2016 | pseudo-document short-text model |
| `GSDMM` | Yin & Wang 2014 | short-text DMM |
| `SupervisedLDA` | Blei & McAuliffe 2007 | supervised response head |
| `ETM` | Dieng et al. 2020 | embedded topic model (variational-EM path) |
| `DETM` | Dieng et al. 2019 | dynamic embedded topic model |
| `TopicsOverTime` | Wang & McCallum 2006 | per-topic Beta over time; validated by planted + NumPy method-of-moments + SciPy |
| `Scholar` | Card et al. 2018 | VAE core shares ProdLDA's AVITM validation; the label/covariate heads are planted-recovery only |

### Behavioral

| Model | Paper | Note |
|---|---|---|
| `TopicGPT` | Pham et al. 2024 | LLM-driven; validated for orchestration with a deterministic fake backend, not numeric parity (output is model-bounded) |

### Experimental tier

Gated behind `enable_experimental()`. The right-hand column records whether the
experimental status is justified and what would let the model graduate.

| Model | Paper | Basis | Status |
|---|---|---|---|
| `TensorLDA` | Kangaslahti et al. 2026 | planted gold + opt-in TensorLy compare | **Graduation candidate.** It has a paper and an opt-in cross-implementation compare (`parity/tlda_compare.py` against `tensorly/tlda`, run when `TOPICA_TLDA_REF` points at a checkout). Running that compare as the standing accuracy gate, plus the adversarial and user gates, would clear the triple gate. |
| `NarrativeTM` | topica original | planted / tests | Justified. No paper (an original GDMR-over-position construction), so planted-only by nature; cannot graduate under the current definition. |
| `IdealPointTM` | topica original | planted / tests | Justified. A Wordfish-with-topics construction with no dedicated paper; the word-topic variants are reliable, bare-scaling is not. |
| `IdealPointSentenceTM` | topica original | planted / tests | Justified. The sentence-embedding sibling of `IdealPointTM`, same no-paper status. |
| `EmbeddingLDA` | topica original | planted only | Justified (issue #660). A topica original with no paper and no external reference; its planted gold cannot distinguish it from plain LDA (on the planted corpus, plain LDA and shuffled/random embeddings score the same block purity), and on real labeled text its recovery sits below plain LDA. The `SeededLDA` core it delegates to is validated; the embedding-seeding benefit is what is unproven. |

## Following up

Running the four current experimental models through the full triple gate
(graduate or keep-gated per the result) is tracked in issue #660. `TensorLDA` is
the nearest candidate, since it is the only experimental model with both a paper
and a cross-implementation reference already in the tree.
