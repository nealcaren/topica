# Topic-model clusters as labels, embeddings as features

**Data.** The bundled `examples/ng20_minilm.npz`: 2,594 documents from five
20-Newsgroups groups (`comp.graphics`, `rec.sport.baseball`, `sci.med`,
`sci.space`, `talk.politics.guns`), each with a 384-d `all-MiniLM-L6-v2`
sentence embedding (L2-normalized) and the in-vocab token text.

**Idea.** Let an *unsupervised* topic model assign every document to a cluster,
then train a *supervised* classifier to reproduce those cluster labels from the
embeddings — cross-validated, so a document is always scored by a model that
never saw it. The two representations are independent: LDA sees only word
counts, the classifier sees only neural sentence embeddings. Where they
disagree, we ask whether the "different-cluster" documents are a meaningful set.
We keep the true newsgroup label out of the whole pipeline and use it only as an
external referee at the end.

Reproduce with `python experiments/ng20_topic_classify/run_experiment.py`.

## Pipeline

1. **Topic model → cluster.** `topica.LDA(num_topics=5, seed=13)` on the token
   texts; each document's argmax topic is its cluster `z`.
2. **Classify → cluster.** Multinomial logistic regression on the 384-d
   embeddings, 5-fold `cross_val_predict`, predicting `z`. Out-of-fold
   predictions `ẑ`.
3. **Study the disagreements** (`ẑ ≠ z`).

## What LDA found (K = 5)

| cluster | top words | maps to |
|--------:|-----------|---------|
| 0 | space nasa launch earth data research center orbit | sci.space |
| 1 | year game team good games hit runs baseball | rec.sport.baseball |
| 2 | edu image graphics com file software ftp available | comp.graphics |
| 3 | don just like people think know time good | sci.med* |
| 4 | gun guns file law people firearms weapons control | talk.politics.guns |

Cluster 3 is a generic "discussion" topic with no strong content signature; the
Hungarian mapping assigns it to `sci.med` by default, which is why it is the
noisiest of the five. Against the true newsgroups the LDA partition scores
ARI = 0.32, NMI = 0.47, best-map accuracy = 0.66 — a moderate, recognizable
recovery, with the generic cluster doing most of the damage.

## Are the clusters learnable from the embeddings? Yes.

| quantity | accuracy |
|---|---:|
| embeddings → **LDA cluster** (5-fold CV) | **0.850** |
| embeddings → **true newsgroup** (ceiling) | 0.899 |
| embeddings → **shuffled cluster** (floor) | 0.434 (base rate 0.453) |

Two independent views of the corpus agree on **85%** of documents — nearly the
90% the embeddings manage against the *real* labels, and far above the ~45%
base rate a shuffled-label control collapses to. The LDA clusters are genuine,
coherent regions of embedding space, not classifier artifacts.

## The disagreements are a meaningful set

390 documents (15.0%) land in a different cluster than LDA assigned. They are
not random:

**1. They sit where LDA was unsure.** Mean LDA topic entropy is 0.58 on
disagreements vs. 0.42 on agreements (max-θ 0.55 vs. 0.74). The classifier
overrides LDA precisely on the documents LDA itself assigned with low
confidence. See `fig_entropy.png`.

**2. When the two views disagree, the embedding view is right more often.**
Using the true newsgroup as referee, on the disagreement set the embedding
classifier's pick matches the true group **47.9%** of the time vs. **42.6%** for
the LDA cluster it overrode. The reassignments are net-corrective — the
embedding classifier fixes more bag-of-words boundary errors than it introduces.
(Absolute numbers are low because these are, by construction, the corpus's
hardest, most confusable documents.)

**3. They are interpretably ambiguous.** The confident overrides read as
documents whose word counts point one way and whose meaning points another —
exactly the boundary cases a bag-of-words model mishandles:

- *"huge good **mri** sets big cost … **scan** … gordon banks … pitt **edu**"* —
  LDA files it under **comp.graphics** (θ = 0.92) on `image/edu/file`; it is a
  **sci.med** post about MRI scanners. The embedding classifier moves it to
  medicine.
- *"cut **medical newsletter** … reported cases colorado increased …"* — LDA
  puts it in **sci.space** on `data/research`; the embedding reads it as
  **sci.med**, correctly.
- *"final votes creation newsgroup misc **health** …"* — a health-newsgroup
  vote landing under **comp.graphics** for LDA, **sci.med** for the embedding.

A minority of disagreements are genuinely degenerate posts (garbled or
near-empty text) that neither view places well — also a meaningful category,
just not an interesting one.

## Sensitivity to K

| K | embed→cluster recovery | disagreements | ARI to true |
|--:|---:|---:|---:|
| 5 | 0.850 | 15.0% | 0.323 |
| 8 | 0.779 | 22.1% | 0.240 |
| 10 | 0.754 | 24.6% | 0.192 |
| 15 | 0.702 | 29.8% | 0.202 |

Recoverability degrades gracefully as topics are split finer, and the
disagreement rate rises with it: the more finely LDA slices the corpus, the more
documents live near a boundary where the count view and the meaning view part
ways. K = 5 (matching the number of source groups) is both the most learnable
and the best-aligned to the true structure.

## Answer

**Yes — the different-cluster documents are a meaningful set.** They are the
genuinely ambiguous, boundary-straddling documents (plus a small tail of
degenerate posts), concentrated on LDA's low-confidence assignments, and when
the count view and the embedding view part ways the embedding view agrees with
the true newsgroup more often. Cross-checking an unsupervised topic assignment
against an embedding classifier is a practical way to *surface* a corpus's hard
cases: the disagreements are a short, readable list of the documents worth a
second look.

### Artifacts
- `results.json` — every metric above, machine-readable
- `disagreements.csv` — all 390 disagreements, most-confident-override first
- `fig_confusion.png` — LDA cluster × embedding-predicted cluster
- `fig_entropy.png` — LDA topic entropy, agreements vs. disagreements
