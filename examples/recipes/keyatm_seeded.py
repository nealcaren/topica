"""Recipe: measure concepts you can name in advance (seeded topics).

Task shape: "I already know the themes I care about (say, economy, security,
culture). Anchor topics to those concepts with a few seed words each, and tell me
how much each anchored topic actually shows up." keyATM keeps your named topics
(seeded by keywords) and can add unseeded topics for whatever else is there.

Data here is the bundled `gadarian` immigration survey. REPLACE the load, the
text column, and the KEYWORDS with your own concepts; the rest transfers.

Run:  python examples/recipes/keyatm_seeded.py
"""
import topica

# --- your data + your named concepts ---------------------------------------
df = topica.datasets.load_gadarian()
TEXT_COL = "open.ended.response"

# One entry per concept: a label -> a few seed words that name it. Seed words
# that are too rare to matter are dropped with a warning, so over-supply is safe.
KEYWORDS = {
    "economy_jobs": ["jobs", "taxes", "welfare", "money", "pay"],
    "crime_security": ["crime", "border", "security", "illegal", "laws"],
    "language_culture": ["english", "language", "learn", "speak", "assimilate"],
}

# --- corpus ----------------------------------------------------------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL,
    stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
)

# --- fit keyATM (named topics + a couple of unseeded ones) -----------------
# num_topics >= len(KEYWORDS) adds unseeded topics for themes you did not name.
model = topica.KeyATM(KEYWORDS, num_topics=5, seed=13).fit(corpus, iters=500)

# --- read each named topic + how present it is -----------------------------
# keyword_rate[t] is the share of topic t's mass carried by its seed words:
# high = the anchor held; low = the concept is weak or the seeds were off.
names = list(KEYWORDS)
rate = model.keyword_rate
print("Seeded topics (top words; keyword_rate = share of mass on seed words):\n")
for t, name in enumerate(names):
    words = ", ".join(model.top_words(8, topic=t))
    print(f"  [{name}]  keyword_rate={rate[t]:.2f}")
    print(f"      {words}")
print("\nUnseeded topics (whatever else the corpus contains):")
for t in range(len(names), model.num_topics):
    print(f"  topic {t}: {', '.join(model.top_words(8, topic=t))}")
