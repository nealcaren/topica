"""Dedicated hierarchical corpus: near-single-topic docs with a planted 2-level
tree, so HLDA can recover domain (internal node) vs topic (leaf) structure.

Tree:  root (procedural/generic framing)
        |- urban        {housing, budget, transit}      shared domain vocab
        |- civic_order  {policing, environment}         shared domain vocab
Each doc is ABOUT one leaf topic; its sentences mix that leaf's TOPIC_CORE (leaf
level) + its DOMAIN_CORE (domain level) + light procedure framing (root level).
"""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

SEED = 424242
rng = np.random.default_rng(SEED)

TOPIC_CORE = {
    "housing":     ["zoning", "permits", "units", "landlord", "vacancy", "density"],
    "budget":      ["appropriation", "deficit", "reserve", "levy", "audit", "fiscal"],
    "transit":     ["station", "route", "transfer", "corridor", "timetable", "depot"],
    "policing":    ["precinct", "dispatch", "patrol", "warrant", "response", "beat"],
    "environment": ["watershed", "wetland", "canopy", "runoff", "habitat", "particulate"],
}
DOMAIN = {"housing": "urban", "budget": "urban", "transit": "urban",
          "policing": "civic_order", "environment": "civic_order"}
DOMAIN_CORE = {
    "urban":       ["district", "constituents", "ordinance", "neighborhood", "citywide", "parcel"],
    "civic_order": ["safety", "oversight", "protection", "standards", "wellbeing", "compliance"],
}
PROC = ["motion", "agenda", "quorum", "chair", "amendment", "record"]
LEAVES = list(TOPIC_CORE)
N_DOCS = 96

docs = []
doc_leaf = np.zeros(N_DOCS, dtype=int)
doc_domain = np.zeros(N_DOCS, dtype=int)
DOMAINS = ["urban", "civic_order"]

for d in range(N_DOCS):
    leaf = LEAVES[d % len(LEAVES)]
    dom = DOMAIN[leaf]
    L = int(rng.integers(6, 10))
    sents = []
    for pos in range(L):
        # edges get a procedural framing sentence; interior is the leaf topic
        is_proc = (pos == 0 and rng.random() < 0.6) or (pos == L - 1 and rng.random() < 0.5)
        if is_proc:
            sents.append({"topic": "procedure", "core_words": PROC})
        else:
            sents.append({
                "topic": leaf,
                "core_words": TOPIC_CORE[leaf],       # leaf-level vocabulary
                "domain_words": DOMAIN_CORE[dom],     # domain-level shared vocabulary
            })
    doc_leaf[d] = LEAVES.index(leaf)
    doc_domain[d] = DOMAINS.index(dom)
    docs.append({"id": d, "leaf": leaf, "domain": dom, "sentences": sents})

spec = {"topic_core": TOPIC_CORE, "domain_core": DOMAIN_CORE, "domain_map": DOMAIN,
        "leaves": LEAVES, "domains": DOMAINS, "proc": PROC}
json.dump({"spec": spec, "docs": docs}, open(os.path.join(OUT, "hier_latents.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "hier_answer_key.npz"), doc_leaf=doc_leaf, doc_domain=doc_domain,
         leaves=np.array(LEAVES), domains=np.array(DOMAINS))

# split into batches
os.makedirs(os.path.join(OUT, "hier_batches"), exist_ok=True)
os.makedirs(os.path.join(OUT, "hier_out"), exist_ok=True)
B, names = 16, []
for bi in range(0, N_DOCS, B):
    name = f"hbatch_{bi//B:02d}"
    json.dump({"docs": docs[bi:bi + B]}, open(os.path.join(OUT, "hier_batches", f"{name}.json"), "w"), indent=1)
    names.append(name)
json.dump(names, open(os.path.join(OUT, "hier_batch_names.json"), "w"))
print(f"{N_DOCS} docs, {len(names)} batches, leaves={LEAVES}")
print("docs per leaf:", {t: int((doc_leaf == i).sum()) for i, t in enumerate(LEAVES)})
