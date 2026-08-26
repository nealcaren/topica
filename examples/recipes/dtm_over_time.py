"""Recipe: track how topics evolve over time.

Task shape: "The same topics run through the whole corpus, but their wording
shifts across years. What did topic 3 sound like in the first period vs the
last?" A Dynamic Topic Model (DTM) ties each period's topics to the previous
period's, so you can read a topic's drift instead of refitting per year.

Data here is the bundled `load_congress` press releases (2013-2024), fetched on
first use and cached. The time slice is the calendar year. REPLACE the load +
column names with your own DataFrame and the rest transfers unchanged.

Kept deliberately small (subsampled, few EM iters) so it runs in under a minute;
raise SAMPLE / ITERS / K for a real analysis.

Run:  python examples/recipes/dtm_over_time.py
"""
import numpy as np

import topica

# --- your data -------------------------------------------------------------
df = topica.datasets.load_congress()
TEXT_COL = "text"
TIME_COL = "year"       # any orderable period column: year, month index, wave
K = 6
SAMPLE = 800            # subsample for a quick demo; None uses all rows
ITERS = 15

if SAMPLE is not None and len(df) > SAMPLE:
    df = df.sample(n=SAMPLE, random_state=13).reset_index(drop=True)

# --- corpus + 0-based contiguous time-slice index --------------------------
corpus = topica.from_dataframe(
    df, text_col=TEXT_COL, metadata_cols=[TIME_COL], strip_html=True,
    stopwords=topica.stopwords("english"), min_length=4, min_doc_freq=5,
)
years = corpus.metadata[TIME_COL].to_numpy(dtype=int)
slices = sorted(set(years))
slice_of = {y: i for i, y in enumerate(slices)}       # year -> 0-based slice
times = np.array([slice_of[y] for y in years])

# --- fit -------------------------------------------------------------------
model = topica.DTM(num_topics=K, seed=13).fit(corpus, times, iters=ITERS)
first, last = 0, len(slices) - 1
print(f"DTM with K={K} over {len(slices)} yearly slices "
      f"({slices[first]}-{slices[last]}), {corpus.num_docs} docs\n")

# --- read a topic's wording at the first vs last period --------------------
# model.top_words(topic, time, n) reads that period's topic-word distribution;
# model.word_evolution(topic, word) returns the word's probability per slice.
for t in range(K):
    early = ", ".join(model.top_words(t, first, 6))
    late = ", ".join(model.top_words(t, last, 6))
    print(f"Topic {t}")
    print(f"    {slices[first]}: {early}")
    print(f"    {slices[last]}: {late}")

# --- track one word's trajectory inside a topic ----------------------------
topic, word = 0, model.top_words(0, last, 1)[0]
traj = model.word_evolution(topic, word)
line = "  ".join(f"{slices[i]}:{p:.3f}" for i, p in enumerate(traj))
print(f"\nWord '{word}' in topic {topic} over time:\n    {line}")
