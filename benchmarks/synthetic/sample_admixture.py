"""Deterministic, seeded latent sampler for the multi-axis synthetic corpus.

This is the ANSWER KEY. No LLM touches it. It plants orthogonal latent structure
that different topica model families should each recover, and writes:
  - latents.json   : per-document + per-sentence assignments for the prose agents
  - answer_key.npz : numeric ground-truth arrays for the recovery scorecard

Frame: municipal council floor statements. Each document is one member's remarks.
Axes planted (axis -> model family that should recover it):
  themes (topic-word)              -> LDA/NMF/CTM/HDP
  region -> prevalence             -> STM/DMR prevalence
  party  -> within-topic wording   -> SAGE / STM-content
  year   -> "transit" vocab drift  -> DTM/DETM
  year   -> "housing" prevalence up -> STM(spline)/keyATM-dynamic
  author ideal point -> word tilt  -> IdealPointTM/Wordfish
  sentence position -> topic       -> NarrativeTM (procedure at the edges)
  short (1-sentence) split         -> GSDMM
"""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

SEED = 20260710
rng = np.random.default_rng(SEED)

# ---- fixed schema ----------------------------------------------------------
THEMES = ["procedure", "housing", "budget", "transit", "policing", "environment"]
K = len(THEMES)
SUBSTANTIVE = [t for t in THEMES if t != "procedure"]  # procedure is the framing topic
REGIONS = ["North", "Coast", "South"]
PARTIES = ["Civic", "Labor"]
YEARS = [2015, 2018, 2021, 2024]
N_AUTHORS = 24
N_DOCS = 240  # 10 docs/author -> stronger author-position signal

# Planted content lexicons (party frames a topic differently). Ground truth for
# the content axis; the prose agent must use the listed frame words.
PARTY_FRAMES = {
    "housing":     {"Civic": ["development", "growth", "market", "supply"],
                    "Labor": ["affordability", "tenant", "rent", "eviction"]},
    "budget":      {"Civic": ["efficiency", "savings", "taxpayer", "restraint"],
                    "Labor": ["investment", "services", "equity", "funding"]},
    "transit":     {"Civic": ["congestion", "parking", "commute", "roads"],
                    "Labor": ["access", "fares", "ridership", "coverage"]},
    "policing":    {"Civic": ["enforcement", "safety", "patrol", "order"],
                    "Labor": ["accountability", "reform", "oversight", "community"]},
    "environment": {"Civic": ["cost", "compliance", "feasibility", "balance"],
                    "Labor": ["climate", "emissions", "green", "resilience"]},
    "procedure":   {"Civic": ["motion", "order", "chair", "record"],
                    "Labor": ["motion", "order", "chair", "record"]},
}

# Per-topic CORE vocabulary (topic identity, party- and position-neutral). Sharpens
# topic separation so count models recover distinct topics. Ground truth: none of
# these overlap across topics.
TOPIC_CORE = {
    "procedure":   ["agenda", "quorum", "amendment", "adjourn", "roll", "gavel"],
    "housing":     ["zoning", "permits", "units", "landlord", "vacancy", "density"],
    "budget":      ["appropriation", "deficit", "reserve", "levy", "audit", "fiscal"],
    "transit":     ["station", "route", "transfer", "corridor", "timetable", "depot"],
    "policing":    ["precinct", "dispatch", "patrol", "warrant", "response", "beat"],
    "environment": ["watershed", "wetland", "canopy", "runoff", "habitat", "particulate"],
}

# Author ideal-point tilt lexicon (orthogonal to party): high position leans
# "private/market", low leans "public/collective". Ground truth for the axis.
# Expanded and applied to EVERY substantive sentence so the axis is recoverable.
POSITION_LEX = {
    "high": ["private", "incentive", "voluntary", "competitive", "deregulate", "entrepreneurial"],
    "low":  ["public", "collective", "mandate", "universal", "subsidized", "cooperative"],
}

# Transit vocabulary drift by year (DTM ground truth): same topic, moving words.
TRANSIT_DRIFT = {
    2015: ["bus", "parking", "lane"],
    2018: ["bus", "bike", "lane"],
    2021: ["bike", "rail", "charging"],
    2024: ["rail", "electric", "charging"],
}

# ---- planted prevalence effects (logit-scale) ------------------------------
# base rate per substantive topic (procedure handled separately by position)
base = {t: 0.0 for t in SUBSTANTIVE}
region_effect = {  # region -> {topic: logit bump}
    "North": {"environment": 1.4},
    "Coast": {"transit": 1.4},
    "South": {"policing": 1.4},
}
party_effect = {
    "Civic": {"budget": 1.0, "policing": 0.7},
    "Labor": {"housing": 1.0, "environment": 0.7},
}
# housing prevalence rises across years (index 0->3)
year_effect = {y: {"housing": 0.6 * i} for i, y in enumerate(YEARS)}

# ---- authors ---------------------------------------------------------------
author_party = [PARTIES[i % 2] for i in range(N_AUTHORS)]
author_region = [REGIONS[i % 3] for i in range(N_AUTHORS)]
author_position = rng.normal(0.0, 1.0, size=N_AUTHORS)
author_position = (author_position - author_position.mean()) / author_position.std()


def subst_theta(region, party, year):
    """Logistic-normal mixture over SUBSTANTIVE topics from covariates."""
    logits = np.array([base[t] for t in SUBSTANTIVE], dtype=float)
    for src in (region_effect[region], party_effect[party], year_effect[year]):
        for t, v in src.items():
            logits[SUBSTANTIVE.index(t)] += v
    logits += rng.normal(0.0, 0.5, size=len(SUBSTANTIVE))  # doc-level noise
    e = np.exp(logits - logits.max())
    return e / e.sum()


docs = []
theta_full = np.zeros((N_DOCS, K))       # doc-topic ground truth (incl. procedure)
doc_region = np.zeros(N_DOCS, dtype=int)
doc_party = np.zeros(N_DOCS, dtype=int)
doc_year = np.zeros(N_DOCS, dtype=int)
doc_author = np.zeros(N_DOCS, dtype=int)

for d in range(N_DOCS):
    a = int(rng.integers(N_AUTHORS))
    region, party = author_region[a], author_party[a]
    year = int(rng.choice(YEARS))
    th = subst_theta(region, party, year)
    L = int(rng.integers(5, 10))  # 5..9 sentences

    sent_topics = []
    for pos in range(L):
        rel = pos / (L - 1) if L > 1 else 0.5
        # procedure concentrated at the edges -> plants the narrative arc
        p_proc = 0.8 if pos == 0 else (0.7 if pos == L - 1 else 0.08)
        if rng.random() < p_proc:
            sent_topics.append("procedure")
        else:
            sent_topics.append(str(rng.choice(SUBSTANTIVE, p=th)))

    tilt = "high" if author_position[a] > 0 else "low"
    sentences = []
    for pos, topic in enumerate(sent_topics):
        plan = {
            "position": round(pos / (L - 1), 3) if L > 1 else 0.5,
            "topic": topic,
            "core_words": TOPIC_CORE[topic],
            "frame_words": PARTY_FRAMES[topic][party],
            "tilt_words": POSITION_LEX[tilt],
            "require_tilt": topic != "procedure",
        }
        if topic == "transit":
            plan["drift_words"] = TRANSIT_DRIFT[year]
        sentences.append(plan)

    # record ground-truth doc-topic proportions (empirical over the schedule)
    counts = np.array([sent_topics.count(t) for t in THEMES], dtype=float)
    theta_full[d] = counts / counts.sum()
    doc_region[d], doc_party[d], doc_year[d], doc_author[d] = (
        REGIONS.index(region), PARTIES.index(party), YEARS.index(year), a)

    docs.append({
        "id": d, "author": a, "region": region, "party": party, "year": year,
        "author_position_sign": tilt, "sentences": sentences,
    })

spec = {
    "themes": THEMES, "regions": REGIONS, "parties": PARTIES, "years": YEARS,
    "n_authors": N_AUTHORS, "n_docs": N_DOCS, "seed": SEED,
    "topic_core": TOPIC_CORE,
    "party_frames": PARTY_FRAMES, "position_lex": POSITION_LEX,
    "transit_drift": TRANSIT_DRIFT,
    "planted": {
        "region_effect": region_effect, "party_effect": party_effect,
        "housing_rises_over_years": True,
        "transit_vocab_drifts": True,
        "procedure_at_document_edges": True,
    },
}

with open(os.path.join(OUT, "latents.json"), "w") as f:
    json.dump({"spec": spec, "docs": docs}, f, indent=1)

np.savez(os.path.join(OUT, "answer_key.npz"),
         theta=theta_full, themes=np.array(THEMES),
         doc_region=doc_region, doc_party=doc_party, doc_year=doc_year,
         doc_author=doc_author, author_position=author_position,
         author_party=np.array(author_party), author_region=np.array(author_region))

n_sent = sum(len(d["sentences"]) for d in docs)
print(f"wrote {N_DOCS} docs, {n_sent} sentences, K={K}, {N_AUTHORS} authors")
print("theta col means (procedure should be high from edges):")
for t, m in zip(THEMES, theta_full.mean(0)):
    print(f"  {t:12s} {m:.3f}")
# quick sanity: housing prevalence should rise with year
hi = THEMES.index("housing")
for i, y in enumerate(YEARS):
    mask = doc_year == i
    print(f"  housing@{y}: {theta_full[mask, hi].mean():.3f}  (n={mask.sum()})")
