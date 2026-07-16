"""Reproduce the STM (Structural Topic Model) R-package vignette with topica.

Works through the same analysis as Roberts, Stewart & Tingley's `stm` vignette,
on the same data (`gadarian`: 341 open-ended survey responses about immigration,
with an experimental `treatment` and party-id `pid_rep`), using topica
instead of R. Maps each vignette step to its topica equivalent:

    textProcessor / prepDocuments  ->  tokenize + stopwords + frequency pruning
    stm(prevalence = ~treatment)   ->  STM(K).fit(docs, prevalence=X)
    labelTopics                    ->  stm.label_topics  (prob / FREX)
    estimateEffect(~treatment)     ->  stm.estimate_effect(model.doc_topic, X)
    topicCorr                      ->  STM.topic_correlation
    findThoughts                   ->  stm.find_thoughts

Data: gadarian, from the stm R package (GPL-3); Gadarian & Albertson (2014).

Run:  python examples/stm_vignette.py
"""

import csv
import os
from collections import Counter

import numpy as np

from topica.models import STM
from topica import tokenize, stm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_gadarian():
    rows = []
    with open(os.path.join(HERE, "gadarian.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    text = [r["open.ended.response"] for r in rows]
    treatment = np.array([float(r["treatment"]) for r in rows])
    pid_rep = np.array([float(r["pid_rep"]) for r in rows])
    return text, treatment, pid_rep


def preprocess(text, stopwords, min_doc_freq=3):
    """textProcessor + prepDocuments: tokenize, drop stopwords, prune rare words,
    drop emptied documents. Returns (token_docs, keep_mask) so covariates stay
    aligned to the surviving documents."""
    toks = [tokenize(t, stopwords=stopwords, min_length=3) for t in text]
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= min_doc_freq}
    toks = [[w for w in d if w in vocab] for d in toks]
    keep = np.array([len(d) > 0 for d in toks])
    docs = [d for d, k in zip(toks, keep) if k]
    return docs, keep


def main():
    print("=" * 70)
    print("STM vignette — gadarian immigration survey — in topica")
    print("=" * 70)

    text, treatment, pid_rep = load_gadarian()
    stopwords = open(os.path.join(HERE, "english-stoplist.txt")).read().split()
    print(f"\n[1] Loaded {len(text)} open-ended responses "
          f"(treatment: {int(treatment.sum())} treated / {int((1-treatment).sum())} control)")

    docs, keep = preprocess(text, stopwords)
    X = np.column_stack([treatment[keep], pid_rep[keep]])
    vocab_size = len({w for d in docs for w in d})
    print(f"[2] Preprocessed -> {len(docs)} documents, {vocab_size} word types "
          f"({(~keep).sum()} empty docs dropped, covariates kept aligned)")

    # [3] Fit STM with prevalence covariates (treatment + party id).
    K = 3
    model = STM(num_topics=K, seed=1)
    model.fit(docs, X, prevalence_names=["treatment", "pid_rep"], iters=80)
    print(f"[3] Fit STM with K={K}, prevalence = ~treatment + pid_rep")

    # [4] labelTopics: prob + FREX words per topic.
    print("\n[4] labelTopics (top words per topic):")
    labels = stm.label_topics(model.topic_word, model.vocabulary, n=7)
    for t in range(K):
        print(f"  Topic {t}")
        print(f"    Highest prob: {', '.join(w for w, _ in labels[t]['prob'])}")
        print(f"    FREX:         {', '.join(w for w, _ in labels[t]['frex'])}")

    # [5] estimateEffect: how the treatment shifts topic prevalence.
    print("\n[5] estimateEffect(~treatment + pid_rep) — effect of TREATMENT on topic prevalence:")
    effects = stm.estimate_effect(
        model.doc_topic, X, feature_names=["treatment", "pid_rep"]
    )
    for t in range(K):
        e = effects[t]
        ti = e.feature_names.index("treatment")
        sig = "*" if abs(e.z[ti]) > 1.96 else " "
        print(f"  Topic {t}: treatment coef = {e.coef[ti]:+.4f}  "
              f"(SE {e.se[ti]:.4f}, z {e.z[ti]:+.2f}) {sig}")
    print("    (* = |z| > 1.96; positive = treatment raises this topic's prevalence)")

    # [6] topicCorr: which topics co-occur.
    print("\n[6] topicCorr (topic correlation matrix):")
    corr = model.topic_correlation
    for i in range(K):
        print("    " + "  ".join(f"{corr[i, j]:+.2f}" for j in range(K)))

    # [7] findThoughts: most representative responses per topic.
    print("\n[7] findThoughts (most representative response per topic):")
    kept_text = [t for t, k in zip(text, keep) if k]
    for t in range(K):
        thoughts = stm.find_thoughts(model.doc_topic, kept_text, topic=t, n=1)
        idx, prop, doc = thoughts[0]
        print(f"  Topic {t} (theta={prop:.2f}): {doc[:150].strip()}...")

    print("\n" + "=" * 70)
    print("Vignette complete — full STM workflow reproduced in topica.")
    print("=" * 70)


if __name__ == "__main__":
    main()
