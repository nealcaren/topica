"""Long-document NarrativeTM corpus: ~24-sentence docs with a genuine multi-phase
narrative arc. Topics appear in a fixed ORDERED sequence of regimes, so the global
position->topic trajectory has real structure to recover (not just edge-framing).
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

SEED = 20260711
rng = np.random.default_rng(SEED)

TOPIC_CORE = {
    "procedure":   ["agenda", "quorum", "amendment", "adjourn", "roll", "gavel"],
    "housing":     ["zoning", "permits", "units", "landlord", "vacancy", "density"],
    "budget":      ["appropriation", "deficit", "reserve", "levy", "audit", "fiscal"],
    "transit":     ["station", "route", "transfer", "corridor", "timetable", "depot"],
    "policing":    ["precinct", "dispatch", "patrol", "warrant", "response", "beat"],
    "environment": ["watershed", "wetland", "canopy", "runoff", "habitat", "particulate"],
}
# The planted narrative ORDER: regime r (0..5) is dominated by ARC[r].
ARC = ["procedure", "housing", "budget", "transit", "policing", "environment"]
THEMES = list(TOPIC_CORE)
K = len(THEMES)
N_DOCS = 60
SENT = 24
REGIME = SENT // len(ARC)     # 4 sentences per regime
NOISE = 0.15                  # prob a sentence is drawn off-regime

docs = []
for d in range(N_DOCS):
    sent_topics = []
    for pos in range(SENT):
        regime = min(pos // REGIME, len(ARC) - 1)
        if rng.random() < NOISE:
            topic = THEMES[int(rng.integers(K))]        # off-regime sprinkle
        else:
            topic = ARC[regime]
        sent_topics.append(topic)
    docs.append({"id": d,
                 "sentences": [{"position": round(p / (SENT - 1), 3),
                                "topic": t, "core_words": TOPIC_CORE[t]}
                               for p, t in enumerate(sent_topics)]})

# planted trajectory: for each topic, its regime index (peak position order)
peak_order = {t: (ARC.index(t) if t in ARC else -1) for t in THEMES}
spec = {"topic_core": TOPIC_CORE, "themes": THEMES, "arc": ARC, "sent_per_doc": SENT}
json.dump({"spec": spec, "docs": docs}, open(os.path.join(OUT, "narr_latents.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "narr_answer_key.npz"), themes=np.array(THEMES), arc=np.array(ARC),
         peak_order=np.array([peak_order[t] for t in THEMES]))

os.makedirs(os.path.join(OUT, "narr_batches"), exist_ok=True); os.makedirs(os.path.join(OUT, "narr_out"), exist_ok=True)
B, names = 10, []
for bi in range(0, N_DOCS, B):
    name = f"nbatch_{bi//B:02d}"
    json.dump({"docs": docs[bi:bi + B]}, open(os.path.join(OUT, "narr_batches", f"{name}.json"), "w"), indent=1)
    names.append(name)
json.dump(names, open(os.path.join(OUT, "narr_batch_names.json"), "w"))
print(f"{N_DOCS} docs x {SENT} sentences, {len(names)} batches; arc = {ARC}")
