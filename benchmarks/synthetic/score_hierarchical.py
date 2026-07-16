"""Hierarchical recovery: fit HLDA on the dedicated domain-structured corpus and
check it recovers the planted 2-level tree (domains at internal nodes, topics at
leaves; docs routed by domain)."""
import json, glob, re, sys
import numpy as np
import topica
topica.enable_experimental()
try:
    from sklearn.metrics import adjusted_rand_score as ARI
except Exception:
    ARI = None

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOP = set("the a an and or of to in on for with is are was were be been by at as it this that we our "
           "i you they he she will would can could should must may from not have has had do does did but "
           "if then so than which who what when where how all one more most some any each also into about "
           "over under out up down their its".split())
def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

L = json.load(open(f"{HERE}/hier_latents.json"))
SPEC = L["spec"]
lat = {d["id"]: d for d in L["docs"]}
ak = np.load(f"{HERE}/hier_answer_key.npz", allow_pickle=True)
LEAVES = list(ak["leaves"]); DOMAINS = list(ak["domains"])
DOMAIN_CORE = SPEC["domain_core"]; TOPIC_CORE = SPEC["topic_core"]

real = {}
for fn in glob.glob(f"{HERE}/hier_out/hbatch_*.json"):
    for d in json.load(open(fn))["docs"]:
        real[d["id"]] = d["sentences"]
ids = sorted(i for i in lat if i in real)
docs_tok = [[w for s in real[i] for w in toks(s)] for i in ids]
doc_domain = np.array([ak["doc_domain"][i] for i in ids])
print(f"assembled {len(ids)} hier docs")

# HLDA's natural default: it recovers each domain's vocabulary at an internal node
# (metric A) but over-splits the tree, so document-by-domain routing (B/C) does not
# recover. Set HLDA_GAMMA to narrow the tree and trade A down for fewer nodes.
_g = os.environ.get("HLDA_GAMMA")
_kw = {"gamma": float(_g)} if _g else {}
m = topica.models.HLDA(depth=3, seed=1, **_kw)
m.fit(docs_tok, iters=400)
levels = np.asarray(m.node_levels)
print(f"HLDA: {m.num_nodes} nodes, levels seen: {sorted(set(levels.tolist()))}", file=sys.stderr)

def node_words(node, n=12):
    tw = m.top_words(node, n)
    return set(w for (w, _) in tw) if tw and isinstance(tw[0], (list, tuple)) else set(tw)

# --- metric A: internal (level-1) nodes capture domain vocabulary -----------
internal = [nd for nd in range(m.num_nodes) if levels[nd] == 1]
domains_found = 0
for dom in DOMAINS:
    dcore = set(DOMAIN_CORE[dom])
    best = max((len(node_words(nd) & dcore) for nd in internal), default=0)
    if best >= 2:
        domains_found += 1
metric_a = domains_found / len(DOMAINS)

# --- metric B: docs route by domain at the internal level -------------------
paths = [list(p) for p in m.doc_paths]
# level-1 node for each doc = the node on its path whose level == 1
def lvl1(path):
    for nd in path:
        if levels[nd] == 1:
            return nd
    return path[min(1, len(path) - 1)]
route = np.array([lvl1(paths[k]) for k in range(len(ids))])
ari = ARI(doc_domain, route) if ARI else float("nan")

# --- metric C: leaves capture topic (not domain) vocabulary -----------------
leafnodes = [nd for nd in range(m.num_nodes) if levels[nd] == max(levels)]
leaf_hits = 0
for nd in leafnodes:
    nw = node_words(nd)
    best_leaf = max(len(nw & set(TOPIC_CORE[t])) for t in LEAVES)
    if best_leaf >= 2:
        leaf_hits += 1
metric_c = leaf_hits / max(1, len(leafnodes))

print("\n" + "=" * 58)
print(f"{'HIERARCHICAL (HLDA) SCORECARD':^58}")
print("=" * 58)
rows = [
    ("domains at internal nodes (A)", metric_a, metric_a >= 1.0),
    ("docs route by domain, ARI (B)", ari, (ari > 0.3) if ARI else False),
    ("leaves capture topic vocab (C)", metric_c, metric_c >= 0.5),
]
for name, v, ok in rows:
    val = f"{v:+.3f}" if isinstance(v, (int, float)) else str(v)
    print(f"  {'PASS' if ok else 'fail'}  {name:<34} {val}")
print("=" * 58)
# show the tree for inspection
print("\nRecovered tree (node: level, top words):", file=sys.stderr)
for nd in range(m.num_nodes):
    print(f"  node {nd} L{levels[nd]}: {sorted(node_words(nd, 6))}", file=sys.stderr)
