"""NarrativeTM trajectory recovery on the long multi-phase corpus: does the learned
global position->topic trajectory reproduce the planted topic ORDER?"""
import json, glob, re
import numpy as np
import topica
topica.enable_experimental()
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOP = set("the a an and or of to in on for with is are was were be been by at as it this that we our i you "
           "they he she will would can could should must may from not have has had do does did but if then so than "
           "which who what when where how all one more most some any each also into about over under out up down their its".split())
def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

L = json.load(open(f"{HERE}/narr_latents.json"))
lat = {d["id"]: d for d in L["docs"]}
ak = np.load(f"{HERE}/narr_answer_key.npz", allow_pickle=True)
THEMES = list(ak["themes"]); ARC = list(ak["arc"]); K = len(THEMES)
real = {}
for fn in glob.glob(f"{HERE}/narr_out/nbatch_*.json"):
    for d in json.load(open(fn))["docs"]:
        real[d["id"]] = d["sentences"]
ids = sorted(i for i in lat if i in real)
print(f"assembled {len(ids)} docs")

# docs as flat token lists; ground-truth beta from labeled sentences
doc_tokens, sent_records = [], []
for i in ids:
    dd = []
    for pl, tx in zip(lat[i]["sentences"], real[i]):
        tk = toks(tx); dd.append(tk); sent_records.append((tk, pl["topic"]))
    doc_tokens.append([w for s in dd for w in s])
from collections import Counter
vc = Counter(w for d in doc_tokens for w in d)
vocab = [w for w, c in vc.items() if c >= 3]; vindex = {w: k for k, w in enumerate(vocab)}; V = len(vocab)
docs_v = [[w for w in d if w in vindex] for d in doc_tokens]
beta_true = np.ones((K, V)) * 1e-9
for tk, lab in sent_records:
    for w in tk:
        if w in vindex:
            beta_true[THEMES.index(lab), vindex[w]] += 1
beta_true /= beta_true.sum(1, keepdims=True)

nt = topica.models.NarrativeTM(num_topics=K, segment_by="chunk", chunk_size=15, degree=5, seed=1)
nt.fit(docs_v, iters=600)

# align NarrativeTM topics to themes
bw = np.asarray(nt.topic_word); mv = list(nt.vocabulary); mi = {w: k for k, w in enumerate(mv)}
proj = np.zeros((K, V))
for k, w in enumerate(vocab):
    if w in mi:
        proj[:, k] = bw[:, mi[w]]
proj /= (proj.sum(1, keepdims=True) + 1e-12)
bt = beta_true / (np.linalg.norm(beta_true, axis=1, keepdims=True) + 1e-12)
bf = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
r, c = linear_sum_assignment(-(bt @ bf.T))
mp = dict(zip(r, c))                              # theme -> narr topic

# trajectory: peak position of each ARC topic
grid = np.linspace(0, 1, 60)
traj = np.asarray(nt.global_trajectory(grid))     # (60, K)
peak_pos = {t: grid[int(np.argmax(traj[:, mp[THEMES.index(t)]]))] for t in ARC}

planted_order = list(range(len(ARC)))             # ARC is already in planted order
recovered = [peak_pos[t] for t in ARC]
rho, _ = spearmanr(planted_order, recovered)

print("\n" + "=" * 54)
print(f"{'NARRATIVE TRAJECTORY RECOVERY':^54}")
print("=" * 54)
print(f"  {'planted arc position':<22}{'topic':<13}{'recovered peak t'}")
for r_i, t in enumerate(ARC):
    print(f"  {r_i:<22}{t:<13}{peak_pos[t]:.2f}")
print("-" * 54)
print(f"  Spearman(planted order, recovered peak) = {rho:+.3f}")
print(f"  {'PASS' if rho > 0.8 else 'fail'}  (threshold 0.8)")
print("=" * 54)
