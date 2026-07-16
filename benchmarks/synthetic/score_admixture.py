"""Recovery scorecard: fit a battery of topica models on the realized synthetic
corpus and score each planted axis against the answer key.

Run after the realization workflow has written out/batch_*.json.
"""
import json, glob, re, sys
import numpy as np

import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

import topica
topica.enable_experimental()
from scipy.optimize import linear_sum_assignment
try:
    from sklearn.metrics import adjusted_rand_score as ARI
except Exception:
    ARI = None

STOP = set("the a an and or of to in on for with is are was were be been by at as it this that "
           "we our i you they he she will would can could should must may our their its from not "
           "have has had do does did but if then so than which who whom what when where how all "
           "one more most some any each also into about over under out up down".split())

def toks(s):
    return [w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP]

# ---- assemble realized text with planted labels ---------------------------
_L = json.load(open(f"{HERE}/latents.json"))
lat = {d["id"]: d for d in _L["docs"]}
SPEC = _L["spec"]
REGIONS, PARTIES, YEARS = SPEC["regions"], SPEC["parties"], SPEC["years"]
TOPIC_CORE = SPEC["topic_core"]
ak = np.load(f"{HERE}/answer_key.npz", allow_pickle=True)
THEMES = list(ak["themes"])
K = len(THEMES)

realized = {}
for fn in glob.glob(f"{HERE}/out/batch_*.json"):
    for d in json.load(open(fn))["docs"]:
        realized[d["id"]] = d["sentences"]

ids = sorted(i for i in lat if i in realized)
print(f"assembled {len(ids)}/{len(lat)} docs")
missing = [i for i in lat if i not in realized]
if missing:
    print(f"  WARNING missing {len(missing)} docs: {missing[:10]}")

# per-doc token lists, and per-sentence (tokens,label) for beta_true / narrative
doc_tokens, sent_records = [], []
first_subst_sent, first_subst_label = [], []
for i in ids:
    plans = lat[i]["sentences"]
    texts = realized[i]
    n = min(len(plans), len(texts))
    dt = []
    picked = None
    for j in range(n):
        tk = toks(texts[j])
        label = plans[j]["topic"]
        dt.append(tk)
        sent_records.append((tk, label))
        if picked is None and label != "procedure" and tk:
            picked = (tk, label)
    doc_tokens.append([w for s in dt for w in s])
    if picked:
        first_subst_sent.append(picked[0]); first_subst_label.append(picked[1])

# vocabulary
from collections import Counter
vc = Counter(w for d in doc_tokens for w in d)
vocab = [w for w, c in vc.items() if c >= 3]
vindex = {w: k for k, w in enumerate(vocab)}
V = len(vocab)
print(f"vocab (min count 3): {V} types")

def to_ids(tklist):
    return [w for w in tklist if w in vindex]

docs_v = [to_ids(d) for d in doc_tokens]

# ground-truth beta from labeled sentences
beta_true = np.ones((K, V)) * 1e-9
for tk, label in sent_records:
    ki = THEMES.index(label)
    for w in tk:
        if w in vindex:
            beta_true[ki, vindex[w]] += 1
beta_true /= beta_true.sum(1, keepdims=True)

def match_cosine(bt, bf):
    bt = bt / (np.linalg.norm(bt, axis=1, keepdims=True) + 1e-12)
    bf = bf / (np.linalg.norm(bf, axis=1, keepdims=True) + 1e-12)
    S = bt @ bf.T                      # (K_true, K_fit)
    r, c = linear_sum_assignment(-S)
    return S, dict(zip(r, c)), S[r, c].mean()

def fit_beta_on_vocab(model):
    """Project a fitted model's topic_word onto our scoring vocab."""
    bw = np.asarray(model.topic_word)
    mv = list(model.vocabulary)
    mi = {w: k for k, w in enumerate(mv)}
    out = np.zeros((bw.shape[0], V))
    for k, w in enumerate(vocab):
        if w in mi:
            out[:, k] = bw[:, mi[w]]
    out /= (out.sum(1, keepdims=True) + 1e-12)
    return out

doc_year = ak["doc_year"]; doc_author = ak["doc_author"]
author_pos = ak["author_position"]
scores = {}

# ---- 1. LDA topic recovery -------------------------------------------------
try:
    m = topica.models.LDA(num_topics=K, seed=1)
    m.fit(docs_v, iters=800)
    bf = fit_beta_on_vocab(m)
    S, mp, mc = match_cosine(beta_true, bf)
    scores["LDA topic-word cosine"] = (mc, mc > 0.6)
except Exception as e:
    scores["LDA topic-word cosine"] = (f"ERR {e}", False)

# ---- 2. STM prevalence: housing rises with year ---------------------------
try:
    yr = doc_year.astype(float)
    X = np.column_stack([np.ones_like(yr), yr])
    sm = topica.models.STM(num_topics=K, seed=1)
    sm.fit(docs_v, prevalence=X, prevalence_names=["intercept", "year"], iters=200)
    bf = fit_beta_on_vocab(sm)
    _, mp, _ = match_cosine(beta_true, bf)
    housing_fit = mp[THEMES.index("housing")]
    th = np.asarray(sm.doc_topic)[:, housing_fit]
    slope = np.polyfit(yr, th, 1)[0]
    scores["STM housing-rises slope>0"] = (slope, slope > 0)
except Exception as e:
    scores["STM housing-rises slope>0"] = (f"ERR {e}", False)

# ---- 3. IdealPointTM: recover author positions ----------------------------
try:
    ip = topica.models.IdealPointTM(num_topics=K, seed=1)
    grp = [f"author_{int(a):02d}" for a in doc_author]   # strings; zero-pad -> sort==index
    ip.fit(docs_v, group=grp, iters=150)
    pos = np.asarray(ip.author_positions).ravel()
    if pos.shape[0] == author_pos.shape[0]:
        r = abs(np.corrcoef(pos, author_pos)[0, 1])
        scores["IdealPointTM |corr| positions"] = (r, r > 0.5)
    else:
        scores["IdealPointTM |corr| positions"] = (f"shape {pos.shape}", False)
except Exception as e:
    scores["IdealPointTM |corr| positions"] = (f"ERR {e}", False)

# ---- 4. NarrativeTM: procedure is U-shaped over position ------------------
try:
    nt = topica.models.NarrativeTM(num_topics=K, segment_by="chunk", chunk_size=6, seed=1)
    nt.fit(docs_v, iters=400)
    bf = fit_beta_on_vocab(nt)
    _, mp, _ = match_cosine(beta_true, bf)
    proc = mp[THEMES.index("procedure")]
    traj = nt.global_trajectory([0.0, 0.5, 1.0])[:, proc]
    u = (traj[0] + traj[2]) / 2 - traj[1]     # >0 means procedure peaks at edges
    scores["NarrativeTM procedure U-shape"] = (u, u > 0)
except Exception as e:
    scores["NarrativeTM procedure U-shape"] = (f"ERR {e}", False)

# ---- 5. GSDMM short-text: cluster one substantive sentence per doc --------
try:
    short = [to_ids(s) for s in first_subst_sent]
    keep = [k for k, s in enumerate(short) if s]
    short = [short[k] for k in keep]
    labels = [THEMES.index(first_subst_label[k]) for k in keep]
    g = topica.models.GSDMM(num_topics=K + 3, seed=1)
    g.fit(short, iters=40)
    pred = np.asarray(g.doc_topic).argmax(1)
    ari = ARI(labels, pred) if ARI else float("nan")
    scores["GSDMM short-text ARI"] = (ari, (ari > 0.2) if ARI else False)
except Exception as e:
    scores["GSDMM short-text ARI"] = (f"ERR {e}", False)

# ---- 6. SAGE content covariate: party-frame word separation ---------------
try:
    par_lbl = [PARTIES[int(p)] for p in ak["doc_party"]]   # "Civic"/"Labor"
    sg = topica.models.SAGE(num_topics=K, seed=1)
    sg.fit(docs_v, par_lbl, group_names=PARTIES, iters=300)
    tw = np.asarray(sg.topic_word)                        # (K, G, V_sage)
    mv = list(sg.vocabulary); mi = {w: k for k, w in enumerate(mv)}
    mm = np.asarray(sg.topic_word_marginal)               # (K, V_sage)
    proj = np.zeros((K, V))
    for k, w in enumerate(vocab):
        if w in mi:
            proj[:, k] = mm[:, mi[w]]
    proj /= (proj.sum(1, keepdims=True) + 1e-12)
    _, mp, _ = match_cosine(beta_true, proj)              # theme -> SAGE topic
    seps = []
    for theme, d in SPEC["party_frames"].items():
        kt = mp[THEMES.index(theme)]
        lr = np.log(tw[kt, 0] + 1e-12) - np.log(tw[kt, 1] + 1e-12)   # Civic - Labor
        ci = [mi[w] for w in d["Civic"] if w in mi]
        li = [mi[w] for w in d["Labor"] if w in mi]
        if ci and li:
            seps.append(lr[ci].mean() - lr[li].mean())
    sep = float(np.mean(seps))                            # >0: Civic frames itself, Labor itself
    scores["SAGE party-frame separation"] = (sep, sep > 0)
except Exception as e:
    scores["SAGE party-frame separation"] = (f"ERR {e}", False)

# ---- 7. DMR: region/party -> topic prevalence -----------------------------
try:
    reg, par = ak["doc_region"], ak["doc_party"]
    R, P = len(REGIONS), len(PARTIES)
    feats = np.zeros((len(ids), R + P))
    for n, (rr, pp) in enumerate(zip(reg, par)):
        feats[n, int(rr)] = 1.0
        feats[n, R + int(pp)] = 1.0
    feats = feats - feats.mean(0)     # center -> identifiable vs DMR's auto-intercept
    fnames = [f"region_{x}" for x in REGIONS] + [f"party_{x}" for x in PARTIES]
    dm = topica.models.DMR(num_topics=K, seed=1)
    dm.fit(docs_v, features=feats, feature_names=fnames, iters=400)
    bf = fit_beta_on_vocab(dm)
    _, mp, _ = match_cosine(beta_true, bf)
    fe = np.asarray(dm.feature_effects)
    if fe.shape[0] != K:
        fe = fe.T
    fn_actual = list(dm.feature_names)                    # DMR prepends 'intercept'
    reg_cols = [fn_actual.index(f"region_{r}") for r in REGIONS]
    par_cols = [fn_actual.index(f"party_{p}") for p in PARTIES]
    # "which covariate level maximizes this topic" — robust to the intercept.
    region_checks = [("transit", "Coast"), ("policing", "South"), ("environment", "North")]
    party_checks = [("budget", "Civic"), ("housing", "Labor")]
    hits = 0
    for th, lvl in region_checks:
        row = fe[mp[THEMES.index(th)], reg_cols]
        if REGIONS[int(np.argmax(row))] == lvl:
            hits += 1
    for th, lvl in party_checks:
        row = fe[mp[THEMES.index(th)], par_cols]
        if PARTIES[int(np.argmax(row))] == lvl:
            hits += 1
    rate = hits / (len(region_checks) + len(party_checks))
    scores["DMR covariate->topic hit-rate"] = (rate, rate >= 0.6)
except Exception as e:
    scores["DMR covariate->topic hit-rate"] = (f"ERR {e}", False)

# ---- 8. DTM: transit vocabulary drift (rail up, bus down) -----------------
try:
    yr_idx = ak["doc_year"].astype(int).tolist()
    dt = topica.models.DTM(num_topics=K, seed=1)
    dt.fit(docs_v, yr_idx, iters=20)
    # DTM topic_word is per-slice; identify the transit topic by top-word overlap.
    transit_markers = set(TOPIC_CORE["transit"]) | {"bus", "rail", "bike", "parking",
                                                     "lane", "electric", "charging"}
    n_slices = len(YEARS)
    def toplist(k):
        words = set()
        for tt in range(n_slices):
            words |= set(dt.top_words(k, tt, 20))
        return words
    tj = max(range(K), key=lambda k: len(toplist(k) & transit_markers))
    vset = set(w for d in docs_v for w in d)
    val, ok = "n/a", False
    if "rail" in vset and "bus" in vset:
        late = np.asarray(dt.word_evolution(tj, "rail"))
        early = np.asarray(dt.word_evolution(tj, "bus"))
        rise, fall = late[-1] - late[0], early[0] - early[-1]
        val = float(rise + fall); ok = rise > 0 and fall > 0
    scores["DTM transit drift rail>/bus<"] = (val, ok)
except Exception as e:
    scores["DTM transit drift rail>/bus<"] = (f"ERR {e}", False)

# ---- 9. InfoCTM cross-lingual topic alignment -----------------------------
try:
    def toks_es(s):
        return re.findall(r"[a-záéíóúñü]{2,}", s.lower())
    es = {}
    for fn in glob.glob(f"{HERE}/out_es/batch_*.json"):
        for d in json.load(open(fn))["docs"]:
            es[d["id"]] = d["sentences"]
    if not es:
        raise RuntimeError("no Spanish corpus in out_es/ yet")
    pair_ids = [i for i in ids if i in es]
    docs_en = [doc_tokens[ids.index(i)] for i in pair_ids]
    docs_es = [[w for s in es[i] for w in toks_es(s)] for i in pair_ids]
    gloss = json.load(open(f"{HERE}/glossary.json"))
    dictionary = [(en, esw) for en, esw in gloss.items()]
    ic = topica.models.InfoCTM(num_topics=K, seed=1, languages=("en", "es"),
                        hidden_size=64, lr=0.01)
    ic.fit(docs_en, docs_es, dictionary=dictionary, iters=250, batch_size=64)
    ta, tb = ic.top_words(10, lang="en"), ic.top_words(10, lang="es")
    core_en = SPEC["topic_core"]
    core_es = {t: [gloss.get(w, w) for w in ws] for t, ws in core_en.items()}
    def words_of(entry):
        return [w for (w, _) in entry] if isinstance(entry[0], (list, tuple)) else list(entry)
    def theme_of(words, coremap):
        sw = set(words)
        return max(coremap, key=lambda t: len(sw & set(coremap[t])))
    aligned = sum(theme_of(words_of(ta[t]), core_en) == theme_of(words_of(tb[t]), core_es)
                  for t in range(K))
    rate = aligned / K
    scores["InfoCTM cross-lingual align"] = (rate, rate >= 0.6)
except Exception as e:
    scores["InfoCTM cross-lingual align"] = (f"ERR {e}", False)

# ---- scorecard -------------------------------------------------------------
print("\n" + "=" * 58)
print(f"{'RECOVERY SCORECARD':^58}")
print("=" * 58)
for k, (v, ok) in scores.items():
    val = f"{v:+.3f}" if isinstance(v, (int, float)) else str(v)[:30]
    print(f"  {'PASS' if ok else 'fail'}  {k:<34} {val}")
print("=" * 58)
