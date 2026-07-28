"""End-to-end embedding topics: embed, model, label, report.

Embeds the corpus once (local sentence-transformers, cached to disk), fits
FASTopic, labels the topics with an LLM, and writes a plot_report figure.

    pip install "topica[openai,viz]" sentence-transformers
    python examples/llm_topics.py

The embedder is a local sentence-transformers model (offline, no key). The labeler
defaults to OpenAI gpt-4o-mini, whose key is read from OPENAI_API_KEY (or pass
key=... to llm_backend). To label with Claude or Gemini, pip install
topica[anthropic]/topica[gemini] and set LABEL_MODEL to "claude-3-5-haiku-latest"
or "gemini-2.5-flash"; for a local run set LABEL_MODEL to a pulled ollama model
and LABEL_BASE_URL to "http://localhost:11434/v1".
"""

import csv
import os

import topica

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRISIS = os.path.join(ROOT, "examples", "dubois_crisis.csv")
STOP = os.path.join(ROOT, "examples", "english-stoplist.txt")

EMBED_MODEL = "all-MiniLM-L6-v2"       # local sentence-transformers, offline
LABEL_MODEL = "gpt-4o-mini"            # OpenAI; key from OPENAI_API_KEY
LABEL_BASE_URL = None                  # set to an OpenAI-compatible URL for local models


def main():
    rows = list(csv.DictReader(open(CRISIS)))
    stop = list(open(STOP).read().split())
    texts = [r["text"] for r in rows]
    decade = [f"{r['decade']}s" for r in rows]
    docs = [topica.tokenize(t, stopwords=stop, min_length=4) for t in texts]

    # 1. Embed once, cached. Re-runs reload from disk instead of re-embedding.
    cache = os.path.join(HERE, "crisis_emb.npz")
    if os.path.exists(cache):
        doc_emb = topica.load_embeddings(cache)
    else:
        from sentence_transformers import SentenceTransformer

        doc_emb = SentenceTransformer(EMBED_MODEL).encode(texts)
        topica.save_embeddings(cache, doc_emb, texts=texts, model=EMBED_MODEL)

    # 2. Fit an embedding-native, mixed-membership model.
    model = topica.FASTopic(num_topics=10, seed=1)
    model.fit(docs, doc_emb, iters=200)

    # 3. Name the topics with an LLM (pin temperature for stable labels). This is
    # the one step that needs a labeling model; skip it gracefully if none is set
    # up, in which case the report keeps topica's default top-word labels.
    try:
        backend = topica.llm_backend(LABEL_MODEL, base_url=LABEL_BASE_URL, temperature=0)
        labels = topica.llm_topic_labels(model, texts, backend=backend, set_labels=True)
        for t, label in enumerate(labels):
            print(f"{t:2d}  {label}")
    except Exception as e:
        print(f"[skipping LLM labels: {e}]")
        print("Top words per topic:")
        for t in range(model.num_topics):
            print(f"{t:2d}  " + " ".join(w for w, _ in model.top_words(6, topic=t)))

    # 4. A one-figure report (with LLM labels if step 3 ran, default labels if not).
    fig = topica.plot_report(model, texts=docs, timestamps=decade, n=6,
                             title="Du Bois's Crisis essays")
    fig.savefig(os.path.join(HERE, "crisis_report.png"), dpi=110, bbox_inches="tight")
    print("wrote crisis_report.png")


if __name__ == "__main__":
    main()
