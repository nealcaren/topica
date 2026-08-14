# Matrix factorization vs. neural, against known labels

Not every corpus wants a Gibbs topic model. When the goal is a fast, deterministic
decomposition or a clustering of document embeddings, topica gives you
matrix-factorization models ([`NMF`](../api/models.md), [`LSA`](../api/models.md))
and neural / embedding models ([`BERTopic`](../api/embedding.md),
[`ProdLDA`](../api/embedding.md)) behind the same API. This example runs all three
families on a corpus with **known labels** and asks a question you can only answer
when the truth is available: which method recovers the real structure, and does
coherence tell you the same thing? (Spoiler: it does not.)

!!! info "Focus of this example"
    **Matrix factorization** (NMF, LSA) and a **neural / embedding** model
    (BERTopic) · validating against known classes · why coherence is not
    comparable across model families. For a generative-model workflow see
    [Du Bois](dubois.md) or [Poliblog](poliblog.md).

    Data: [`load_ng20_minilm`](../api/datasets.md#topica.datasets.load_ng20_minilm)
    — a 5-group 20-Newsgroups subset (2,594 documents) with MiniLM sentence
    embeddings precomputed, so the embedding model runs offline with no
    `sentence-transformers` install.

## 1. Build the corpus and choose the rank K

The five groups are `comp.graphics`, `rec.sport.baseball`, `sci.med`,
`sci.space`, and `talk.politics.guns`. We hold the labels out of every model and
use them only to score recovery afterward. `NMF` and `LSA` factor the document-term
matrix at a fixed rank K, so `search_k` reports a `reconstruction_error` scree —
read it like a scree plot and take the elbow.

```python
import numpy as np
import topica

b = topica.datasets.load_ng20_minilm()
docs = [t.split() for t in b.texts]
labels = np.array(b.labels)
corpus = topica.Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.4)

for family in ("nmf", "lsa"):
    scan = topica.search_k(corpus, [3, 4, 5, 6, 8], model=family)
    print(family, scan.best_k("reconstruction_error", rule="elbow"))
```

Both the NMF and the LSA scree elbow at **K = 5**, the true number of desks — a
reassuring sign the decomposition sees the same coarse structure the labels
encode.

## 2. Fit all three families at K = 5

```python
nmf = topica.NMF(num_topics=5, seed=13).fit(corpus)          # tf-idf, Frobenius
lsa = topica.LSA(num_topics=5).fit(corpus)                    # truncated SVD
bert = topica.BERTopic(clusterer="kmeans", num_clusters=5,
                       reduce_frequent=True, seed=1).fit(docs, b.doc_embeddings)
```

The FREX top words are legible for the factorization and the embedding model
alike:

| | NMF | BERTopic |
|---|---|---|
| Baseball | pitching, braves, cubs, scored | alomar, lopez, baerga, catcher |
| Graphics | files, image, format, ftp | jpeg, tiff, gif, vga, vesa |
| Guns | gun, guns, weapons, crime | firearms, firearm, militia, fbi |
| Space | space, launch, lunar, orbit | lunar, orbit, probe, shuttle |

One NMF topic collapses to Usenet signature noise (`geb, shameful, dsl, cadre`) —
the raw-newsgroup boilerplate that survives tokenization; on scraped text prefer
`from_dataframe(strip_html=True)` and a stronger stoplist.

## 3. Score recovery against the labels

Assign each document to its arg-max topic and compare that clustering to the true
groups with purity and the adjusted Rand index (ARI):

```python
from sklearn.metrics import adjusted_rand_score
from collections import Counter

def recovery(doc_topic, labels):
    z = np.asarray(doc_topic).argmax(1)
    purity = sum(Counter(labels[z == k]).most_common(1)[0][1]
                 for k in set(z)) / len(labels)
    ids = {l: i for i, l in enumerate(sorted(set(labels)))}
    return purity, adjusted_rand_score([ids[l] for l in labels], z)
```

| Model | Family | Purity | ARI |
|---|---|:---:|:---:|
| **BERTopic** (k-means on embeddings) | neural / embedding | **0.70** | **0.51** |
| **NMF** (tf-idf, Frobenius) | matrix factorization | 0.52 | 0.18 |
| **LSA** (truncated SVD) | matrix factorization | 0.34 | 0.03 |

The embedding model recovers the five desks far more cleanly than the bag-of-words
factorizations; LSA's signed loadings scatter documents worst. Method choice, not
just K, changes the answer.

## 4. Why coherence cannot arbitrate this

It is tempting to rank the models by topic coherence instead of labels. Do not.
LSA's `topic_word` rows are signed SVD loadings, not a word distribution, so
coherence computed on them is not on the same footing as NMF's — topica warns you:

```python
topica.coherence(lsa, corpus)   # UserWarning: topic_word has negative entries,
                                # not comparable across model families ...
```

Coherence ranks topics *within* a model; across families it can invert the
label-based verdict entirely. When you have ground truth, score against it; when
you do not, compare like with like and lean on held-out likelihood or stability.

## Reproduce

`seed=13`/`seed=1` as above; the embeddings ship with `load_ng20_minilm`, so the
whole comparison runs offline. For a finer-grained neural model on raw tokens (no
embeddings), swap in `topica.ProdLDA(num_topics=5)`.
