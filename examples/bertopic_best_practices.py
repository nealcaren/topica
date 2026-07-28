"""BERTopic's "Best Practices" tutorial, translated to topica.

The original (Maarten Grootendorst's BERTopic tutorial) runs the
embed -> reduce -> cluster -> represent pipeline on ArXiv ML abstracts. topica
fits the same pipeline with no PyTorch / UMAP / numba / sentence-transformers in
its wheel: you bring the embeddings, topica does the rest. This script mirrors the
tutorial step for step and notes where topica does things differently.

Run it (e.g. in Colab):

    pip install topica datasets sentence-transformers
    python bertopic_best_practices.py

What maps cleanly:
  - CountVectorizer(ngram_range, min_df)      -> topica.add_ngrams(docs, ...)
  - MaximalMarginalRelevance(diversity)       -> topica.mmr(model, word_embeddings, ...)
  - UMAP + HDBSCAN + class-based TF-IDF      -> topica.BERTopic(reducer="umap", ...)
  - get_topic_info()                          -> topica.diagnostics(model, texts)
  - get_topic(i)                              -> model.top_words(n, topic=i)
  - set_topic_labels(...)                     -> topica.set_topic_labels(model, ...)
  - approximate_distribution(window, stride)  -> model.approximate_distribution(...)
  - reduce_outliers(...)                      -> model.reduce_outliers()
  - save(...) / load(...)                      -> model.save(path) / BERTopic.load(path)
  - transform(new_docs)                       -> model.transform(new_docs)

Where topica differs, on purpose:
  - **Tokenization.** BERTopic tokenizes raw strings internally (CountVectorizer);
    topica takes pre-tokenized documents, so we call topica.tokenize first and
    topica.add_ngrams for the bigrams + min_df.
  - **Reproducibility.** UMAP has no random_state equivalent here; the discovery fit
    is stochastic (topica warns). The *prediction* phase (transform) is
    deterministic. Use reducer="pca" for a fully reproducible fit, or save the fit.
  - **Representations.** topica reports FREX (frequency-exclusivity) words — the
    defensible label for publication — plus topica.mmr for diversity-aware words and,
    with the optional `topica[llm]` extra, LLM labels via topica.llm_topic_labels.
    KeyBERTInspired / PartOfSpeech have no direct equivalent.
  - **Visualization.** topica ships a static composite figure, topica.plot_report,
    rather than the interactive plotly views.
"""

import numpy as np

import topica


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # --- Data -------------------------------------------------------------
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
    abstracts = dataset["abstract"]
    titles = dataset["title"]  # noqa: F841  (used for hover labels in the original)

    # --- Pre-calculate embeddings (the tutorial's all-MiniLM-L6-v2) -------
    # topica is bring-your-own-vectors. (topica.llm_embed(abstracts,
    # model="sentence-transformers/all-MiniLM-L6-v2") is the in-package route.)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

    # --- Tokenize + n-grams (CountVectorizer(stop_words, min_df=2, ngram_range=(1,2))) ---
    docs = [topica.tokenize(a, stopwords=topica.ENGLISH_STOPWORDS) for a in abstracts]
    docs = topica.add_ngrams(docs, ngram_range=(1, 2), min_df=2)

    # --- Reduce (UMAP) + cluster (HDBSCAN) + represent (c-TF-IDF) ---------
    topic_model = topica.BERTopic(
        reducer="umap", n_neighbors=15, n_components=5,  # UMAP(n_neighbors=15, n_components=5)
        min_cluster_size=150,                            # HDBSCAN(min_cluster_size=150)
        seed=42,
    )
    theta = topic_model.fit_transform(docs, embeddings)  # like topics, probs = fit_transform(...)
    print(f"discovered {topic_model.num_topics} topics; doc-topic matrix {theta.shape}")

    # --- get_topic_info() -> the one-call diagnostics table --------------
    table = topica.diagnostics(topic_model, abstracts)
    print(table.head(20).to_string())

    # --- get_topic(1) -> top words of a topic ----------------------------
    print("topic 1:", [w for w, _ in topic_model.top_words(10, topic=1)])

    # --- Representations --------------------------------------------------
    # FREX words are already a column in `table`. The tutorial's MMR
    # (MaximalMarginalRelevance) maps to topica.mmr: rerank to cut near-synonyms,
    # using word embeddings (embed the vocabulary with the same model).
    vocab = list(topic_model.vocabulary)
    word_embeddings = embedding_model.encode(vocab)
    diverse = topica.mmr(topic_model, word_embeddings, n=10, diversity=0.3)
    print("topic 1 (MMR):", [w for w, _ in diverse[1]])
    # For LLM labels (the tutorial's OpenAI representation), with the topica[llm] extra:
    #   topica.llm_topic_labels(topic_model, abstracts,
    #                           llm_model="gpt-4o-mini", set_labels=True)

    # --- Custom labels ----------------------------------------------------
    topica.set_topic_labels(topic_model, {1: "Space Travel", 7: "Religion"})

    # --- Topic-document distribution (window / stride) -------------------
    topic_distr = topic_model.approximate_distribution(docs, window=8, stride=4)
    abstract_id = 10
    print(f"\nabstract {abstract_id} topic distribution:",
          np.round(topic_distr[abstract_id], 3))

    # --- Outlier reduction ------------------------------------------------
    moved = topic_model.reduce_outliers()
    print(f"reduce_outliers reassigned {moved} documents")

    # --- Visualize: a static composite report ----------------------------
    fig = topica.plot_report(topic_model, texts=abstracts)
    fig.savefig("topica_report.png", dpi=150, bbox_inches="tight")
    print("wrote topica_report.png")

    # --- Serialization: freeze the (stochastic UMAP) discovery fit -------
    # save -> load -> predict, so a good fit is reusable without refitting.
    topic_model.save("my_topica_model.tt")
    loaded_model = topica.BERTopic.load("my_topica_model.tt")

    # --- Inference on new documents --------------------------------------
    new_theta = loaded_model.transform(docs[:100])
    print("inference doc-topic matrix:", new_theta.shape)


if __name__ == "__main__":
    main()
