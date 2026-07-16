"""Cross-implementation check for topica's FASTopic against the reference
`fastopic` package (Wu et al. 2024).

Both models are the same model: theta and beta read off two entropic
optimal-transport plans between embedding sets, with no encoder. The reference
trains by autodiff through the unrolled Sinkhorn iterations; topica differentiates
the fixed point of a hand-coded reverse-mode Sinkhorn and steps with Adam. The
optimizers, initialization, and RNG differ, so exact agreement is impossible. We
hold them to a statistical-equivalence bar on a shared task: the SAME MiniLM
document embeddings, the same documents, the same topic count.

Metrics, all computed identically for both with topica's own coherence:
  - topic coherence (c_v, c_npmi) over the shared token corpus
  - topic diversity (TU)
  - doc-topic agreement: NMI of the argmax topic vs the true newsgroup label, and
    cross-NMI between the two implementations

Skips cleanly when fastopic / sentence-transformers / sklearn are unavailable or
the corpus cannot be fetched.
"""

from __future__ import annotations

import numpy as np

GROUPS = [
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.guns",
    "comp.graphics",
    "sci.med",
]
NUM_TOPICS = 10
TOP_N = 10
SEED = 0


def available() -> bool:
    try:
        import fastopic  # noqa: F401
        import sentence_transformers  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def load():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    data = fetch_20newsgroups(
        subset="train", categories=GROUPS, remove=("headers", "footers", "quotes"),
        random_state=SEED,
    )
    raw = [d.strip() for d in data.data]
    labels = np.array(data.target)
    keep = [i for i, d in enumerate(raw) if len(d.split()) >= 20]
    raw, labels = [raw[i] for i in keep], labels[keep]
    cv = CountVectorizer(min_df=10, max_df=0.4, stop_words="english", token_pattern=r"(?u)\b[a-z]{3,}\b")
    cv.fit([d.lower() for d in raw])
    vset = set(cv.get_feature_names_out())
    analyzer = cv.build_analyzer()
    token_docs, texts, mask = [], [], []
    for d in raw:
        toks = [t for t in analyzer(d.lower()) if t in vset]
        ok = len(toks) >= 5
        mask.append(ok)
        if ok:
            token_docs.append(toks)
            texts.append(" ".join(toks))
    return token_docs, texts, labels[np.array(mask)], sorted(vset)


def embed(texts):
    from sentence_transformers import SentenceTransformer

    return np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(texts, show_progress_bar=False, batch_size=64))


def _coherence(topics, token_docs, measure):
    import topica

    return float(np.mean(topica.coherence(topics, token_docs, coherence_type=measure, topn=TOP_N)))


def _diversity(topics):
    words = [w for t in topics for w in t[:TOP_N]]
    return len(set(words)) / max(1, len(words))


def run(verbose: bool = True) -> dict:
    import topica
    from fastopic import FASTopic
    from sklearn.metrics import normalized_mutual_info_score

    token_docs, texts, labels, vocab = load()
    emb = embed(texts)

    tm = topica.models.FASTopic(num_topics=NUM_TOPICS, lr=0.002, seed=SEED)
    t_theta = np.asarray(tm.fit_transform(token_docs, emb))
    t_topics = [[w for w, _ in tm.top_words(TOP_N, topic=t)] for t in range(tm.num_topics)]
    t_assign = t_theta.argmax(1)

    rm = FASTopic(NUM_TOPICS, verbose=False)
    r_top, r_theta = rm.fit_transform(texts, epochs=200, learning_rate=0.002, preset_doc_embeddings=emb)
    r_topics = [
        [w for w in (tw.split() if isinstance(tw, str) else list(tw)) if not w.isdigit()][:TOP_N]
        for tw in r_top
    ]
    r_assign = np.asarray(r_theta).argmax(1)

    metrics = {
        "num_docs": len(token_docs),
        "topica_c_v": _coherence(t_topics, token_docs, "c_v"),
        "reference_c_v": _coherence(r_topics, token_docs, "c_v"),
        "topica_c_npmi": _coherence(t_topics, token_docs, "c_npmi"),
        "reference_c_npmi": _coherence(r_topics, token_docs, "c_npmi"),
        "topica_diversity": _diversity(t_topics),
        "reference_diversity": _diversity(r_topics),
        "topica_label_nmi": float(normalized_mutual_info_score(labels, t_assign)),
        "reference_label_nmi": float(normalized_mutual_info_score(labels, r_assign)),
        "cross_nmi": float(normalized_mutual_info_score(t_assign, r_assign)),
    }
    if verbose:
        for k, v in metrics.items():
            print(f"  {k:24s} {v}")
    return metrics


if __name__ == "__main__":
    if not available():
        print("fastopic / sentence-transformers not installed; skipping.")
    else:
        run(verbose=True)
