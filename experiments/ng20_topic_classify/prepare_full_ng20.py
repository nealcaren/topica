"""Build the full 20-group counterpart of examples/ng20_minilm.npz.

Downloads the full 20 Newsgroups train split, applies the SAME preprocessing
recipe the bundled 5-group file documents, computes all-MiniLM-L6-v2 sentence
embeddings on the cleaned token text, and writes ng20_full_minilm.npz next to
this script. That .npz is a generated artifact (~20 MB) and is git-ignored;
regenerate it with this script.

Recipe (matches the bundled file's meta string):
  - remove headers/footers/quotes
  - CountVectorizer(min_df=10, max_df=0.4, stop_words="english",
                    token_pattern=r"(?u)\b[a-z]{3,}\b")  -> vocabulary
  - keep docs with >= 20 raw words, then >= 5 in-vocab tokens
  - texts = space-joined in-vocab tokens (lowercase)
  - embeddings = MiniLM(texts), L2-normalized, float16

    python experiments/ng20_topic_classify/prepare_full_ng20.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "ng20_full_minilm.npz"


def main() -> None:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    print("Fetching full 20 Newsgroups (train)…")
    ng = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    raw, y = ng.data, np.array([ng.target_names[t] for t in ng.target])
    print(f"  {len(raw)} raw docs, {len(ng.target_names)} groups")

    vec = CountVectorizer(min_df=10, max_df=0.4, stop_words="english",
                          token_pattern=r"(?u)\b[a-z]{3,}\b")
    vec.fit(raw)
    vocab = set(vec.get_feature_names_out())
    analyzer = vec.build_analyzer()
    print(f"  vocabulary: {len(vocab)} terms")

    texts, labels = [], []
    for doc, lab in zip(raw, y):
        if len(doc.split()) < 20:
            continue
        toks = [t for t in analyzer(doc) if t in vocab]
        if len(toks) < 5:
            continue
        texts.append(" ".join(toks))
        labels.append(lab)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=object)
    print(f"  kept {len(texts)} docs after filtering")

    print("Embedding with all-MiniLM-L6-v2 (CPU)…")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(list(texts), batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = emb.astype(np.float16)
    print(f"  embeddings: {emb.shape} {emb.dtype}")

    meta = ("20-Newsgroups (all 20 groups), subset=train, headers/footers/quotes "
            "removed, docs with >=20 words then >=5 in-vocab tokens kept. texts = "
            "space-joined in-vocab tokens (lowercase, min_df=10, max_df=0.4, "
            "[a-z]{3,}). Embeddings: all-MiniLM-L6-v2 on the token text, float16. "
            "Built by prepare_full_ng20.py.")
    np.savez_compressed(OUT, texts=texts, labels=labels, doc_embeddings=emb,
                        meta=np.array(meta, dtype=object))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
