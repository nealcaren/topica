"""Misspecified-K robustness: fit models with fewer / more topics than the true
K=6 and watch recovery degrade. Reuses the mixedness corpus. Also asks HDP to
DISCOVER the count nonparametrically.

Metrics per fitted K':
  coverage  = mean over the 6 TRUE topics of best cosine to any fitted topic
              (drops when K' too small: true topics get merged)
  redundancy= fitted topics that are nobody's best match / K'  (junk when K' too big)
"""
import json, glob, re
import numpy as np
import topica
topica.enable_experimental()

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOP = set("the a an and or of to in on for with is are was were be been by at as it this that we our i you they "
           "he she will would can could should must may from not have has had do does did but if then so than which "
           "who what when where how all one more most some any each also into about over under out up down their its".split())
def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

lat = {d["id"]: d for d in json.load(open(f"{HERE}/mix_latents.json"))["docs"]}
ak = np.load(f"{HERE}/mix_answer_key.npz", allow_pickle=True)
THEMES = list(ak["themes"]); K_TRUE = len(THEMES)
real = {}
for fn in glob.glob(f"{HERE}/mix_out/mbatch_*.json"):
    for d in json.load(open(fn))["docs"]:
        real[d["id"]] = d["sentences"]
ids = sorted(i for i in lat if i in real)
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
beta_true = np.ones((K_TRUE, V)) * 1e-9
for tk, lab in sent_records:
    for w in tk:
        if w in vindex:
            beta_true[THEMES.index(lab), vindex[w]] += 1
beta_true /= beta_true.sum(1, keepdims=True)
bt = beta_true / (np.linalg.norm(beta_true, axis=1, keepdims=True) + 1e-12)

def proj_of(model):
    bw = np.asarray(model.topic_word); mv = list(model.vocabulary); mi = {w: k for k, w in enumerate(mv)}
    P = np.zeros((bw.shape[0], V))
    for k, w in enumerate(vocab):
        if w in mi:
            P[:, k] = bw[:, mi[w]]
    P /= (P.sum(1, keepdims=True) + 1e-12)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

def score(model):
    bf = proj_of(model)                     # (Kfit, V) normalized
    S = bt @ bf.T                           # (6 true, Kfit)
    coverage = S.max(1).mean()              # each true topic's best match
    best_for = set(S.argmax(1).tolist())    # fitted topics that win >=1 true topic
    redundancy = 1 - len(best_for) / bf.shape[0]
    return coverage, redundancy

print(f"true K = {K_TRUE}\n")
print(f"  {'fitted K':<10}{'coverage':>10}{'redundancy':>12}")
for kfit in [2, 3, 4, 5, 6, 8, 10, 12, 16]:
    m = topica.models.LDA(num_topics=kfit, seed=1); m.fit(docs_v, iters=600)
    cov, red = score(m)
    tag = "  <- true" if kfit == K_TRUE else ("  (too few)" if kfit < K_TRUE else "")
    print(f"  {kfit:<10}{cov:>10.3f}{red:>12.2f}{tag}")

# HDP discovers K nonparametrically
h = topica.models.HDP(seed=1); h.fit(docs_v, iters=150)
print(f"\n  HDP discovered K = {h.num_topics}  (true {K_TRUE})")
print("\n  Read: coverage falls when K too small (true topics merge);")
print("  redundancy rises when K too big (extra topics are nobody's best match).")
