"""Recipe: explore themes with no prior structure, and justify K.

Task shape: "Here is a corpus. What is it about, and how many topics should I
use?" This is the default first pass. Scan a few values of K, let the metrics
inform (not decide) the choice, fit LDA at that K, and read labelled topics.
Choosing K is a judgment call the researcher owns; the scan makes it defensible.

Data here is the bundled `gadarian` survey; it ships with topica. REPLACE the
load + text column with your own DataFrame and the rest transfers unchanged.

Run:  python examples/recipes/lda_explore.py
"""
import numpy as np

import topica

# --- your data -------------------------------------------------------------
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"
KS = [3, 4, 5, 6, 8]        # the candidate topic counts to scan

# --- corpus ----------------------------------------------------------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL,
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)

# --- scan K (stm's searchK): coherence + exclusivity per candidate ---------
scan = topica.select.search_k(corpus, ks=KS, seed=13)
K = scan.best_k()
print(f"Scanned K in {KS}; frontier-suggested K = {K}")
print("(A suggestion, not a verdict: read the topics at a couple of K before committing.)\n")

# --- fit LDA at the chosen K -----------------------------------------------
model = topica.LDA(num_topics=K, seed=13).fit(corpus)

# --- read the topics -------------------------------------------------------
# topic_table gives publication-ready prob + FREX labels; here we print the top
# words per topic so the terminal output stays readable.
print(f"LDA topics at K={K}:")
for t, words in enumerate(model.top_words(8)):
    print(f"  topic {t}: {', '.join(words)}")

# --- how coherent are they? (a validation check, not a K selector) ---------
topics = model.top_words(10)
texts = [" ".join(doc) for doc in corpus.documents()]
per_topic = topica.evaluate.coherence(topics, texts, coherence_type="c_v")
print(f"\nMean c_v coherence: {float(np.mean(per_topic)):.3f}  (0-1; higher = more coherent)")
