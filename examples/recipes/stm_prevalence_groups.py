"""Recipe: compare topic prevalence across two groups.

Task shape: "Do these two groups (treatment vs control, party A vs B, before vs
after) talk about different things?" Fit an STM with the group as a prevalence
covariate, then read the per-topic effect with uncertainty. Positive effect =
the group uses that topic more.

Data here is the bundled `gadarian` survey (immigration open-ends, with an
experimental `treatment` and party id `pid_rep`); it ships with topica so this
runs with no download. REPLACE the load + column names with your own DataFrame
and the rest transfers unchanged.

Run:  python examples/recipes/stm_prevalence_groups.py
"""
import numpy as np

import topica

# --- your data -------------------------------------------------------------
# One row per document; a text column and one or more group/covariate columns.
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"
GROUP_COLS = ["treatment", "pid_rep"]   # numeric 0/1 here; the effect is read for each
FOCUS = "treatment"                     # the group contrast you want to report
K = 5

# --- corpus (metadata stays aligned to surviving documents) ----------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL, metadata_cols=GROUP_COLS,
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)
meta = corpus.metadata
# Prevalence design: the covariates, one column each. For a categorical group
# use topica.design.one_hot(meta["col"]); for a nonlinear trend, design.spline().
X = np.column_stack([meta[c].to_numpy(dtype=float) for c in GROUP_COLS])

# --- fit -------------------------------------------------------------------
model = topica.STM(num_topics=K, seed=13).fit(
    corpus, prevalence=X, prevalence_names=GROUP_COLS, iters=80,
)

# --- read topics + the group effect ---------------------------------------
print(f"STM with K={K}, prevalence = ~{' + '.join(GROUP_COLS)}\n")
labels = topica.inspect.label_topics(model.topic_word, model.vocabulary, n=7)
effects = topica.effects.estimate_effect(
    model.doc_topic, X, feature_names=GROUP_COLS,
)
fi = effects[0].feature_names.index(FOCUS)
print(f"Effect of '{FOCUS}' on each topic's prevalence (95% CI):")
for t in range(K):
    e = effects[t]
    lo, hi = e.ci_low[fi], e.ci_high[fi]
    flag = "*" if lo > 0 or hi < 0 else " "   # CI excludes zero
    top = ", ".join(w for w, _ in labels[t]["prob"][:5])
    print(f"{flag} topic {t} [{e.coef[fi]:+.3f}  ({lo:+.3f}, {hi:+.3f})]  {top}")
print(f"\n* = 95% CI excludes zero. Positive = higher '{FOCUS}' raises the topic.")
