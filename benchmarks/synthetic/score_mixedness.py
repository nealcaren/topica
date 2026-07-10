"""Admixture/mixture CROSSOVER on the mixedness-controlled corpus: theta recovery
binned by the planted number of topics per document (m = 1..5)."""
import json, glob, re
import numpy as np
import topica
topica.enable_experimental()
from scipy.optimize import linear_sum_assignment

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOP = set("the a an and or of to in on for with is are was were be been by at as it this that we our i you "
           "they he she will would can could should must may from not have has had do does did but if then so "
           "than which who what when where how all one more most some any each also into about over under out up down their its".split())
def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

lat = {d["id"]: d for d in json.load(open(f"{HERE}/mix_latents.json"))["docs"]}
ak = np.load(f"{HERE}/mix_answer_key.npz", allow_pickle=True)
THEMES = list(ak["themes"]); K = len(THEMES)
real = {}
for fn in glob.glob(f"{HERE}/mix_out/mbatch_*.json"):
    for d in json.load(open(fn))["docs"]:
        real[d["id"]] = d["sentences"]
ids = sorted(i for i in lat if i in real)
theta_true = ak["theta"][ids]; doc_m = ak["doc_m"][ids]
print(f"assembled {len(ids)}/{len(lat)} docs")

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

def align(model):
    bw = np.asarray(model.topic_word); mv = list(model.vocabulary); mi = {w: k for k, w in enumerate(mv)}
    proj = np.zeros((bw.shape[0], V))
    for k, w in enumerate(vocab):
        if w in mi:
            proj[:, k] = bw[:, mi[w]]
    proj /= (proj.sum(1, keepdims=True) + 1e-12)
    bt = beta_true / (np.linalg.norm(beta_true, axis=1, keepdims=True) + 1e-12)
    bf = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    r, c = linear_sum_assignment(-(bt @ bf.T))
    return dict(zip(r, c))

def theta_cos(model):
    mp = align(model); dtp = np.asarray(model.doc_topic)
    rec = np.zeros((len(ids), K))
    for tk in range(K):
        rec[:, tk] = dtp[:, mp[tk]]
    rec /= (rec.sum(1, keepdims=True) + 1e-12)
    num = (rec * theta_true).sum(1)
    den = np.linalg.norm(rec, axis=1) * np.linalg.norm(theta_true, axis=1) + 1e-12
    return num / den

models = {
    "LDA": topica.LDA(num_topics=K, seed=1),
    "CTM": topica.CTM(num_topics=K, seed=1),
    "GSDMM": topica.GSDMM(num_topics=K, seed=1),
}
models["LDA"].fit(docs_v, iters=800)
models["CTM"].fit(docs_v, iters=200)
models["GSDMM"].fit(docs_v, iters=60)
cos = {n: theta_cos(m) for n, m in models.items()}

print("\n" + "=" * 52)
print(f"{'THETA RECOVERY BY DOCUMENT MIXEDNESS (m)':^52}")
print("=" * 52)
print(f"  {'m (planted #topics)':<20}" + "".join(f"{n:>9}" for n in cos))
for m in [1, 2, 3, 4, 5]:
    mask = doc_m == m
    print(f"  {('m='+str(m)+'  (n='+str(mask.sum())+')'):<20}" + "".join(f"{cos[n][mask].mean():>9.3f}" for n in cos))
print("=" * 52)
print("  CROSSOVER: GSDMM should lead at m=1 (single-topic) and fall")
print("  below the admixture models as m rises.")
