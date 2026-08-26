"""Recipe: topics for very short documents (tweets, headlines, survey answers).

Task shape: "My documents are one or two sentences each. Standard LDA gives mushy,
over-fragmented topics." Short text breaks the LDA assumption that a document
mixes several topics. GSDMM (a Dirichlet mixture) assigns ONE topic per document,
which fits short text. You set an upper bound on the number of clusters; where the
data supports fewer, GSDMM leaves clusters empty, so the retained count is a rough
read on how many topics the corpus supports (widen the bound and refit to check).

Data here is the bundled `gadarian` survey (short open-ended answers). REPLACE the
load + text column with your own short documents; the rest transfers.

Run:  python examples/recipes/gsdmm_short_text.py
"""
import numpy as np

import topica

# --- your (short) data -----------------------------------------------------
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"
K_MAX = 12       # an UPPER bound; GSDMM collapses unused clusters below this

# --- corpus ----------------------------------------------------------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL,
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=2,
)

# --- fit: one topic per document -------------------------------------------
model = topica.GSDMM(num_topics=K_MAX, seed=13).fit(corpus, iters=40)

# --- read the clusters the data actually supported -------------------------
# doc_topic is one-hot here (each doc in one cluster), so column sums are the
# cluster sizes; print only the populated clusters, largest first.
sizes = np.asarray(model.doc_topic).sum(axis=0)
populated = [t for t in np.argsort(sizes)[::-1] if sizes[t] >= 1]
print(f"GSDMM started from K_MAX={K_MAX}; {len(populated)} clusters retained.\n")
for t in populated:
    print(f"  cluster {t}  (n={int(sizes[t])} docs): {', '.join(model.top_words(8, topic=t))}")
