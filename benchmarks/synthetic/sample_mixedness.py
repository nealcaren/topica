"""Mixedness-controlled corpus: documents span a planted single->multi topic
spectrum (m = 1..5 topics per doc) so we can show the admixture/mixture CROSSOVER
-- GSDMM winning the single-topic end, LDA/CTM holding across the whole range.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

SEED = 909090
rng = np.random.default_rng(SEED)

TOPIC_CORE = {
    "housing":     ["zoning", "permits", "units", "landlord", "vacancy", "density"],
    "budget":      ["appropriation", "deficit", "reserve", "levy", "audit", "fiscal"],
    "transit":     ["station", "route", "transfer", "corridor", "timetable", "depot"],
    "policing":    ["precinct", "dispatch", "patrol", "warrant", "response", "beat"],
    "environment": ["watershed", "wetland", "canopy", "runoff", "habitat", "particulate"],
    "budgeting2":  ["surplus", "bond", "grant", "expenditure", "ledger", "quarterly"],
}
THEMES = list(TOPIC_CORE)
K = len(THEMES)
M_LEVELS = [1, 2, 3, 4, 5]
PER = 30
N_DOCS = len(M_LEVELS) * PER
SENT = 8   # sentences per doc, fixed so length isn't a confound

docs = []
theta = np.zeros((N_DOCS, K))
doc_m = np.zeros(N_DOCS, dtype=int)
d = 0
for m in M_LEVELS:
    for _ in range(PER):
        topics = list(rng.choice(K, size=m, replace=False))
        # mixing weights over the chosen topics (uniform-ish with mild noise)
        w = rng.dirichlet(np.ones(m) * 3.0)
        sent_topics = list(rng.choice(topics, size=SENT, p=w))
        counts = np.bincount(sent_topics, minlength=K).astype(float)
        theta[d] = counts / counts.sum()
        doc_m[d] = m
        docs.append({"id": d, "m": m,
                     "sentences": [{"topic": THEMES[t], "core_words": TOPIC_CORE[THEMES[t]]}
                                   for t in sent_topics]})
        d += 1

spec = {"topic_core": TOPIC_CORE, "themes": THEMES, "m_levels": M_LEVELS}
json.dump({"spec": spec, "docs": docs}, open(os.path.join(OUT, "mix_latents.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "mix_answer_key.npz"), theta=theta, doc_m=doc_m, themes=np.array(THEMES))

os.makedirs(os.path.join(OUT, "mix_batches"), exist_ok=True); os.makedirs(os.path.join(OUT, "mix_out"), exist_ok=True)
B, names = 15, []
for bi in range(0, N_DOCS, B):
    name = f"mbatch_{bi//B:02d}"
    json.dump({"docs": docs[bi:bi + B]}, open(os.path.join(OUT, "mix_batches", f"{name}.json"), "w"), indent=1)
    names.append(name)
json.dump(names, open(os.path.join(OUT, "mix_batch_names.json"), "w"))
p = np.clip(theta, 1e-12, None); keff = np.exp(-(p * np.log(p)).sum(1))
print(f"{N_DOCS} docs, {len(names)} batches; m per doc: {dict(zip(*np.unique(doc_m, return_counts=True)))}")
print("k_eff by m:", {int(m): round(float(keff[doc_m == m].mean()), 2) for m in M_LEVELS})
