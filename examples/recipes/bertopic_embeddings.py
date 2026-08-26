"""Recipe: cluster documents by meaning using embeddings.

Task shape: "I have sentence embeddings for my documents (from a transformer) and
want topics that group by meaning, not by shared words." BERTopic reduces the
embeddings, clusters them, and labels each cluster with c-TF-IDF words. topica
takes the embeddings you supply, so there is no PyTorch in the wheel.

Watch out: this is clustering, not a fitted posterior. There is no doc-topic
distribution to compose, so covariate-effect estimation and topic-proportion
uncertainty behave differently than for LDA/STM. Some documents fall in no
cluster (an outlier "topic"); that is expected.

Data here is the bundled `ng20_minilm` sample: 20-Newsgroups documents with
precomputed MiniLM embeddings, fetched on first use. REPLACE it with your own
texts + doc_embeddings (one row per document, aligned) and the rest transfers.

Run:  python examples/recipes/bertopic_embeddings.py
"""
import numpy as np

import topica

# --- your data: texts + aligned embeddings ---------------------------------
data = topica.datasets.load_ng20_minilm()
texts = data.texts                 # list[str], one per document
embeddings = data.doc_embeddings   # (n_docs, dim) float array, same order

# The embeddings drive the clustering; the tokens only supply the c-TF-IDF topic
# labels. Tokenize in place (no pruning) so docs stay aligned to embedding rows.
docs = [topica.tokenize(t, stopwords=topica.stopwords("english"), min_length=3)
        for t in texts]

# --- fit: reduce -> cluster -> label ---------------------------------------
# min_cluster_size is the main knob: larger = fewer, bigger topics. reducer="pca"
# is used here for a stable, seed-reproducible layout; the default "umap" can
# collapse to very few topics on some corpora (widen min_cluster_size or switch
# reducer if that happens). For a fixed number of topics, pass
# clusterer="kmeans", num_clusters=K instead of density clustering.
model = topica.BERTopic(reducer="pca", min_cluster_size=25, seed=13).fit(docs, embeddings)

# --- read the discovered topics --------------------------------------------
# doc_topic is a soft assignment; column mass approximates each topic's size.
sizes = np.asarray(model.doc_topic).sum(axis=0)
order = np.argsort(sizes)[::-1]
print(f"BERTopic on {len(texts)} documents -> {model.num_topics} topics discovered.\n")
for t in order:
    print(f"  topic {t}  (~{sizes[t]:.0f} docs): {', '.join(model.top_words(8, topic=t))}")
