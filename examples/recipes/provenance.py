"""Recipe: record provenance so the analysis is reproducible.

Task shape: "The reviewers (and future me) need to know exactly how this model was
fit — the data fingerprint, the settings, the seed — to reproduce the numbers in
the table." `record_fit` captures a fitted model plus its corpus and settings as
an AnalysisManifest, which you then serialize (JSON for the replication archive,
Markdown for the appendix).

Note the shape: `record_fit(model, corpus, ...)` — the second argument is the
CORPUS you fit on, not a path. It RETURNS a manifest object; you write it out
yourself. Data here is the bundled `gadarian` survey.

Run:  python examples/recipes/provenance.py
"""
import numpy as np

import topica

# --- fit something worth recording -----------------------------------------
df = topica.datasets.load_gadarian()
corpus = topica.from_dataframe(
    df, text_col="open.ended.response", metadata_cols=["treatment", "pid_rep"],
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)
X = np.column_stack([corpus.metadata[c].to_numpy(dtype=float) for c in ["treatment", "pid_rep"]])
model = topica.STM(num_topics=5, seed=13).fit(
    corpus, prevalence=X, prevalence_names=["treatment", "pid_rep"], iters=80,
)

# --- record it -------------------------------------------------------------
# Pass the same covariate design you fit with so the manifest records it too.
manifest = topica.provenance.record_fit(
    model, corpus, prevalence=X, prevalence_names=["treatment", "pid_rep"],
)

# --- serialize -------------------------------------------------------------
# JSON for the replication archive; Markdown for the paper's appendix.
manifest.save("gadarian_manifest.json")               # -> a JSON file on disk
print("Wrote gadarian_manifest.json")
print("\nProvenance card (Markdown, paste into an appendix):\n")
print(manifest.to_markdown())
