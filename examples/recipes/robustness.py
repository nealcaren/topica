"""Recipe: are my topics real, or seed/sample noise?

Task shape: "Before I put this K-topic model in a paper, how do I know the topics
would survive a rerun or a slightly different sample?" Refit on bootstrap
resamples of the corpus and score how stably each topic reappears. Fragile topics
(low stability) are a signal that K is too high or the topic is an artifact.

This is the step a first-time user often can't find: for a covariate model like
STM you pass a `model_factory` (seed -> a fresh model) and thread the covariate
design through `fit_kwargs`; `bootstrap_stability` resamples the documents AND the
per-document covariates together, so the prevalence rows stay aligned.

Data here is the bundled `gadarian` survey. REPLACE the load + columns with your
own; the rest transfers.

Run:  python examples/recipes/robustness.py
"""
import numpy as np

import topica

# --- your data -------------------------------------------------------------
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"
GROUP_COLS = ["treatment", "pid_rep"]
K = 5

corpus = topica.from_dataframe(
    df, text_col=TEXT_COL, metadata_cols=GROUP_COLS,
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)
X = np.column_stack([corpus.metadata[c].to_numpy(dtype=float) for c in GROUP_COLS])

# --- bootstrap stability of an STM at K -----------------------------------
# model_factory(seed) returns a fresh model; fit_kwargs carries the covariate
# design (resampled in tandem with the documents on each bootstrap draw).
result = topica.evaluate.bootstrap_stability(
    corpus,
    n_boot=8,
    seed=13,
    model_factory=lambda seed: topica.STM(num_topics=K, seed=seed),
    fit_kwargs=dict(prevalence=X, prevalence_names=GROUP_COLS, iters=80),
)

print(f"Bootstrap topic stability (STM, K={K}, 8 resamples):")
print(f"  mean stability = {result['mean']:.2f}  (1.0 = topic reappears every resample)\n")
stab = result["stability"]
ref = result["reference"]
for t in np.argsort(stab):        # least stable first — the ones to worry about
    flag = "  <- fragile" if stab[t] < 0.5 else ""
    print(f"  topic {t}: stability {stab[t]:.2f}  [{', '.join(ref.top_words(5, topic=t))}]{flag}")
print("\nLow-stability topics argue for a smaller K or merging; report this, don't hide it.")
