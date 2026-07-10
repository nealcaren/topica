"""Doc-topic (theta) mixture recovery: the metric that separates admixture models
(LDA/STM) from mixture models (GSDMM). Both are fit on the SAME full documents;
we align topics to truth, then compare each model's recovered doc_topic to the
planted theta_d -- overall and binned by document mixedness (effective #topics).
"""
import json, glob, re
import numpy as np
import topica
topica.enable_experimental()
from scipy.optimize import linear_sum_assignment

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOP = set("the a an and or of to in on for with is are was were be been by at as it this that we our i "
           "you they he she will would can could should must may from not have has had do does did but if "
           "then so than which who what when where how all one more most some any each also into about over "
           "under out up down their its".split())
def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

lat = {d["id"]: d for d in json.load(open(f"{HERE}/latents.json"))["docs"]}
ak = np.load(f"{HERE}/answer_key.npz", allow_pickle=True)
THEMES = list(ak["themes"]); K = len(THEMES)
real = {}
for fn in glob.glob(f"{HERE}/out/batch_*.json"):
    for d in json.load(open(fn))["docs"]:
        real[d["id"]] = d["sentences"]
ids = sorted(i for i in lat if i in real)
theta_true = ak["theta"][ids]                      # (D, K) planted doc-topic

# vocab + tokens + ground-truth beta from labeled sentences
doc_tokens, sent_records = [], []
for i in ids:
    dt = []
    for pl, tx in zip(lat[i]["sentences"], real[i]):
        tk = toks(tx); dt.append(tk); sent_records.append((tk, pl["topic"]))
    doc_tokens.append([w for s in dt for w in s])
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
    return dict(zip(r, c))   # truth_k -> model_k

def theta_cos(model):
    mp = align(model)
    dtp = np.asarray(model.doc_topic)
    rec = np.zeros((len(ids), K))
    for tk in range(K):
        rec[:, tk] = dtp[:, mp[tk]]
    rec /= (rec.sum(1, keepdims=True) + 1e-12)
    num = (rec * theta_true).sum(1)
    den = np.linalg.norm(rec, axis=1) * np.linalg.norm(theta_true, axis=1) + 1e-12
    return num / den            # per-doc cosine to planted theta

# effective number of topics per doc (1 = single-topic, K = uniform mixture)
p = np.clip(theta_true, 1e-12, None)
k_eff = np.exp(-(p * np.log(p)).sum(1))

lda = topica.LDA(num_topics=K, seed=1); lda.fit(docs_v, iters=800)
gsd = topica.GSDMM(num_topics=K, seed=1); gsd.fit(docs_v, iters=60)
ctm = topica.CTM(num_topics=K, seed=1); ctm.fit(docs_v, iters=200)

cos = {"LDA (admixture)": theta_cos(lda), "CTM (admixture)": theta_cos(ctm),
       "GSDMM (mixture)": theta_cos(gsd)}

print(f"{len(ids)} docs, effective-topics range {k_eff.min():.1f}-{k_eff.max():.1f} (median {np.median(k_eff):.1f})\n")
print("MEAN doc-topic (theta) cosine to planted mixture:")
for name, c in cos.items():
    print(f"  {name:18s} {c.mean():.3f}")

# crossover: bin docs by mixedness
edges = np.quantile(k_eff, [0, 1/3, 2/3, 1.0])
labels = [f"k_eff {edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(3)]
binidx = np.clip(np.digitize(k_eff, edges[1:-1]), 0, 2)
print("\nCROSSOVER  (theta cosine by document mixedness):")
print(f"  {'mixedness bin':<18}" + "".join(f"{n.split()[0]:>10}" for n in cos))
for b in range(3):
    m = binidx == b
    row = "".join(f"{cos[n][m].mean():>10.3f}" for n in cos)
    print(f"  {labels[b]:<18}{row}   (n={m.sum()})")
print("\nExpectation: admixture (LDA/STM) stays high as mixedness rises; GSDMM falls.")
