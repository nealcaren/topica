"""Recipe: compare how two groups word the same topics.

Task shape: "Both groups discuss the same themes, but do they describe them
differently?" That is a *content* covariate, not a prevalence one: the STM keeps
one shared set of topics and learns a per-group word distribution for each, so
you can read topic 3 "in group A's words" vs "in group B's words".

Data here is the bundled `gadarian` survey; the content group is party id
(`pid_rep`). REPLACE the load + column names with your own DataFrame and the rest
transfers unchanged.

Run:  python examples/recipes/stm_content_groups.py
"""
import numpy as np

import topica

# --- your data -------------------------------------------------------------
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"
GROUP_COL = "pid_rep"           # the content group; any string/int labels work
LABEL = {0: "non-Rep", 1: "Republican"}   # 0/1 codes -> readable group names
K = 4

# --- corpus ----------------------------------------------------------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL, metadata_cols=[GROUP_COL],
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)
# Content groups are self-naming: pass readable string labels, one per document.
group = np.array([LABEL[v] for v in corpus.metadata[GROUP_COL].to_numpy(dtype=int)])

# --- fit with a content covariate ------------------------------------------
# `content=` makes the group shift word choice within shared topics (SAGE-style).
model = topica.STM(num_topics=K, seed=13).fit(corpus, content=group, iters=100)

# --- read each group's wording of each topic -------------------------------
# topic_word_by_group is (num_topics, num_groups, vocab); model.groups gives the
# group order. topic_polarization ranks topics by how much the groups diverge.
vocab = np.asarray(model.vocabulary)
by_group = model.topic_word_by_group
groups = list(model.groups)
polar = topica.content.topic_polarization(model)
order = np.argsort(polar)[::-1]       # most group-divergent topics first

print(f"STM with K={K}, content = ~{GROUP_COL}")
print("Topics ordered by how differently the groups word them.\n")
for t in order:
    print(f"Topic {t}  (group divergence {polar[t]:.3f})")
    for gi, g in enumerate(groups):
        top = vocab[np.argsort(by_group[t, gi])[::-1][:7]]
        print(f"    {str(g):>11}: {', '.join(top)}")
    print()
