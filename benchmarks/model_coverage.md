# topica model coverage: reference & benchmark status

_Where each model stands on external validation and on the speed/accuracy
benchmark. Generated against the dev venv (`.venv-dev`) on 2026-07-28; toolchain
availability is machine-specific and re-checked by the probe in
`benchmarks/full_model_run.py`._

Two independent axes:

- **Ground truth** — is there an external reference implementation we validate
  against, and can we *run it locally right now* (needed for a fresh timing
  ratio) or only replay a **frozen gold** matrix committed under `parity/`
  (fidelity only, no fresh timing)?
- **In the table** — does the model appear in the `full_model_run.py`
  fidelity+speed table, which requires a locally runnable reference.

## Tier 1 — reference runs locally → in (or ready for) the speed/accuracy table

Fresh side-by-side fit is possible, so these carry both an agreement number and a
wall-clock speedup.

**Wired now (the table's current rows):** LDA, DMR, GDMR, PA, DTM, LabeledLDA,
OnlineLDA, KeyATM, STM, STM/content, SupervisedLDA (var), SupervisedLDA (gibbs),
NMF, LSA, FASTopic, ProdLDA, BERTopic, SemanticSignalSeparation.

**Being added (this pass):** CTM, HDP*, HLDA*, SeededLDA.  (*K discovered.)

**Runnable locally but not yet wired (candidates to add):** ETM, DETM, InfoCTM,
PT, RTM, Scholar, TBIP, Wordfish, EmbeddingLDA.

| Model | Reference | Local? | Frozen gold |
|---|---|:--:|:--:|
| LDA | tomotopy / MALLET | yes | yes |
| DMR | tomotopy | yes | yes |
| GDMR | tomotopy | yes | yes |
| PA | tomotopy | yes | yes |
| DTM | tomotopy | yes | yes |
| LabeledLDA | tomotopy | yes | yes |
| OnlineLDA | gensim | yes | yes |
| CTM | tomotopy | yes | yes |
| HDP | tomotopy | yes | yes |
| HLDA | tomotopy | yes | yes |
| PT | tomotopy | yes | yes |
| KeyATM | R `keyATM` | yes | yes |
| STM, STM/content | R `stm` | yes | yes |
| SeededLDA | R `seededlda` | yes | yes |
| RTM | R `lda` | yes | yes |
| Wordfish | R `quanteda.textmodels` | yes | — |
| NMF, LSA | scikit-learn | yes | yes |
| ProdLDA, ETM, DETM, FASTopic, InfoCTM, Scholar, TBIP | torch (hand-coded ref) | yes | yes |
| SemanticSignalSeparation | turftopic | yes | (planted) |
| AnchorLDA | anchor-topic (RecoverL2, Arora 2013) | yes | (parity) |
| BERTopic | bertopic | yes | yes |
| SupervisedLDA (var/gibbs) | planted oracle | yes | yes |
| EmbeddingLDA | gensim (as-LDA) | yes | yes |

## Tier 2 — reference exists but NOT runnable locally → omitted from the speed table

We have a **frozen gold** matrix, so **fidelity is still certified**, but the
reference toolchain is not installed on this machine, so there is **no fresh
timing** — hence no speed row. Install the toolchain (or run on a machine that
has it) to promote these to Tier 1.

| Model | Reference (missing locally) | Frozen gold |
|---|---|:--:|
| CombinedTM | `contextualized_topic_models` | yes |
| ZeroShotTM | `contextualized_topic_models` | yes |
| GSDMM | `gsdmm` | yes |
| BTM | `biterm` | yes |
| Top2Vec | `top2vec` (registered, but row SKIPs) | yes |
| STS | R `STS` (CRAN, uninstalled) | yes |

## Tier 3 — no runnable external reference → validated by internal certification

Nothing to be *faithful to*, or the only reference is non-reproducible. These are
validated by planted-recovery / finite-difference-gradient / paper-as-oracle
checks, not by parity, and are **out of scope** for a fidelity-vs-reference table
by construction.

| Model | Why no reference |
|---|---|
| FLDA / FactorialLDA | reference is GPL Java, non-seedable (no reproducible parity) |
| DiscLDA | paper-as-oracle only (no matching reference implementation) |
| SAGE | Gibbs+MAP *is* the model; no external implementation to match |
| IdealPointTM | embedding-native; novel to topica |
| IdealPointSentenceTM | embedding-native; novel to topica |
| PartyEmbeddings | novel to topica |
| NarrativeTM | novel to topica |
| TensorLDA | internal oracle only |
| PLTM | internal oracle only |
| PolylingualLDA | internal oracle only |
| TopicGPT | LLM-driven; no numeric reference |

## Direct answers

**(1) Omitted from the speed/accuracy table** = everything outside Tier-1-wired:
the Tier-1 not-yet-wired candidates (ETM, DETM, InfoCTM, PT, RTM, Scholar, TBIP,
Wordfish, EmbeddingLDA), all of Tier 2, and all of Tier 3.

**(2) Cannot run the ground truth locally** = Tier 2 (frozen gold only —
CombinedTM, ZeroShotTM, GSDMM, BTM, Top2Vec, STS) plus Tier 3 (no runnable
reference at all — FLDA/FactorialLDA, DiscLDA, SAGE, and the novel/embedding
models). Everything in Tier 1 can be fit against its reference on this machine.
