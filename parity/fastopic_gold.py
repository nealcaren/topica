"""Committed-gold parity for topica FASTopic vs the reference `fastopic` package (#271, Wave 1).

FASTopic (Wu et al. 2024) is the same model in both implementations: theta and beta
are read off two entropic optimal-transport plans between embedding sets, with no
encoder. The reference (`fastopic` package) trains by autodiff through the unrolled
Sinkhorn iterations; topica differentiates the fixed point of a hand-coded
reverse-mode Sinkhorn and steps with Adam. Optimizers, initialization, and RNG
differ, so exact agreement is impossible. We hold them to a statistical-equivalence
bar on a SHARED task: the SAME MiniLM document embeddings, the same documents, the
same topic count — exactly the design of the live script ``parity/fastopic_compare.py``.

The corpus + embeddings + config are taken verbatim from ``parity/fastopic_compare.py``
(the 20-newsgroups 5-group subset, MiniLM ``all-MiniLM-L6-v2`` document embeddings,
K=10), except that the documents are deterministically subsampled so the committed
``.npz`` stays small (the MiniLM embeddings are the size driver and must be frozen so
the offline refit reproduces them — we never re-run a sentence-transformer at test
time). Because the reference re-tokenizes the texts with its own vocabulary, the
two topic-word matrices are aligned onto the *intersection vocabulary* before the
Hungarian alignment, then compared by mean aligned cosine.

The live script reports metrics but computes no seed-to-seed noise floor, so — as in
the CombinedTM gold — we measure one here: the reference is fit twice (two seeds) and
its own aligned topic-word self-cosine becomes the bar (minus a small margin). The
pass-bar is therefore ``topica-vs-reference aligned cosine >= reference-self cosine -
margin``: benchmark against the reference's own reproducibility, not an invented
threshold.

Two phases (mirrors parity/combinedtm_gold.py):

  * ``--regenerate`` (needs fastopic + sentence-transformers + sklearn): loads and
    subsamples the corpus, embeds it once with MiniLM, fits the reference twice to
    measure its topic-word self-cosine floor, freezes one run's topic-word matrix +
    reference vocab, the frozen embeddings, and the token corpus, and writes the
    committed gold (``parity/fastopic_gold.npz`` + ``.json``).
  * default (no fastopic / torch / sentence-transformers): loads the committed gold,
    fits topica FASTopic on the same frozen corpus + embeddings, aligns onto the
    shared vocab, Hungarian-aligns, and checks the bar.

Run directly::

    python parity/fastopic_gold.py               # offline compare against committed gold
    python parity/fastopic_gold.py --regenerate  # run fastopic once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import fastopic_compare as fc  # noqa: E402

NAME = "fastopic"

# Corpus subsample (keeps the committed npz < ~1 MB; MiniLM embeddings dominate).
SUBSAMPLE_N = 420
SUBSAMPLE_SEED = 271

NUM_TOPICS = fc.NUM_TOPICS  # 10, verbatim from the live script
TOP_N = fc.TOP_N            # 10
EPOCHS = 200               # reference epochs, verbatim from the live script
LR = 0.002

GOLD_SEED = 0
FLOOR_SEED = 1
# Pass margin below the reference's own seed-to-seed topic-word cosine floor.
# Wider than the planted-block golds (e.g. CombinedTM's 0.10) because on real
# MiniLM embeddings the cross-implementation topic-word cosine (~0.61) sits a clear
# step below the reference's own seed-to-seed self cosine (~0.69) — the PCA/Adam vs
# autodiff-Sinkhorn gap, not seed noise. The bar must clear that real gap.
MARGIN = 0.15


# --------------------------------------------------------------------------- #
# Shared corpus + embeddings (verbatim from the live script, then subsampled)
# --------------------------------------------------------------------------- #
def _load_subsampled():
    """Run the live-script ``load()`` then take a fixed-seed document subsample so
    the committed gold stays small. The vocabulary is re-derived from the subsample
    (drop word types that vanish) so the frozen corpus is self-consistent."""
    token_docs, texts, labels, _vocab = fc.load()
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    n = min(SUBSAMPLE_N, len(token_docs))
    idx = np.sort(rng.choice(len(token_docs), size=n, replace=False))
    token_docs = [token_docs[i] for i in idx]
    texts = [texts[i] for i in idx]
    labels = labels[idx]
    vocab = sorted({w for d in token_docs for w in d})
    return token_docs, texts, labels, vocab


def _train_reference(texts, emb, k: int, seed: int):
    """Fit the reference fastopic on shared embeddings; return (topic_word_over_ref_vocab,
    ref_vocab)."""
    import torch
    from fastopic import FASTopic

    torch.manual_seed(seed)
    np.random.seed(seed)
    rm = FASTopic(k, verbose=False)
    rm.fit_transform(texts, epochs=EPOCHS, learning_rate=LR, preset_doc_embeddings=emb)
    ref_vocab = [str(w) for w in rm.vocab]
    beta = np.asarray(rm.get_beta(), dtype=np.float64)
    return beta, ref_vocab


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _topica_topic_word(token_docs, emb, seed):
    """Fit topica FASTopic; return (topic_word, vocabulary)."""
    import topica

    tm = topica.models.FASTopic(num_topics=NUM_TOPICS, lr=LR, seed=seed)
    tm.fit_transform(token_docs, emb)
    return np.asarray(tm.topic_word, dtype=np.float64), [str(w) for w in tm.vocabulary]


def _align_on_shared_vocab(beta_a, vocab_a, beta_b, vocab_b):
    """Restrict both topic-word matrices to the intersection of their vocabularies
    (so the cosine is over comparable columns), then mean Hungarian-aligned cosine."""
    shared = sorted(set(vocab_a) & set(vocab_b))
    ia = {w: i for i, w in enumerate(vocab_a)}
    ib = {w: i for i, w in enumerate(vocab_b)}
    a = beta_a[:, [ia[w] for w in shared]]
    b = beta_b[:, [ib[w] for w in shared]]
    cos, _ = harness.align_cosine(a, b)
    return cos, len(shared)


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not fc.available():
        print("fastopic / sentence-transformers / sklearn not available; cannot regenerate.")
        sys.exit(1)

    token_docs, texts, labels, vocab = _load_subsampled()
    emb = fc.embed(texts).astype(np.float32)  # MiniLM is float32; reference requires it

    beta_gold_ref, ref_vocab_gold = _train_reference(texts, emb, NUM_TOPICS, GOLD_SEED)
    beta_floor_ref, ref_vocab_floor = _train_reference(texts, emb, NUM_TOPICS, FLOOR_SEED)

    # Reference seed-to-seed self cosine (both on the reference's own vocab union).
    ref_self_cos, _ = _align_on_shared_vocab(
        beta_gold_ref, ref_vocab_gold, beta_floor_ref, ref_vocab_floor
    )

    # Freeze the reference gold topic-word matrix re-expressed onto the corpus vocab.
    gold_tw = _ref_topic_word_from_beta(beta_gold_ref, ref_vocab_gold, vocab)

    # topica summary at regenerate time for the provenance log.
    t_tw, t_vocab = _topica_topic_word(token_docs, emb, GOLD_SEED)
    topica_cos, n_shared = _align_on_shared_vocab(gold_tw, vocab, t_tw, t_vocab)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word": gold_tw,                       # K x |corpus vocab|
            "vocab": np.array(vocab, dtype=object),      # corpus vocab (gold_tw columns)
            "labels": labels.astype(np.int64),
            "embeddings": emb.astype(np.float32),        # frozen MiniLM embeddings
            "corpus": harness.docs_to_lines(token_docs),
        },
        meta={
            "reference": _fastopic_version(),
            "model": "FASTopic (Wu et al. 2024)",
            "corpus": (
                f"20-newsgroups 5-group subset ({SUBSAMPLE_N}-doc fixed-seed subsample of "
                "parity/fastopic_compare.py's load()), MiniLM all-MiniLM-L6-v2 document "
                "embeddings frozen into the npz"
            ),
            "groups": fc.GROUPS,
            "subsample_n": SUBSAMPLE_N,
            "subsample_seed": SUBSAMPLE_SEED,
            "num_docs": len(token_docs),
            "vocab_size": len(vocab),
            "emb_dim": int(emb.shape[1]),
            "num_topics": NUM_TOPICS,
            "epochs": EPOCHS,
            "lr": LR,
            "seeds": {"gold": GOLD_SEED, "noise_floor": FLOOR_SEED},
            "margin": MARGIN,
            "reference_self_cosine": ref_self_cos,
            "topica_vs_reference_cosine": topica_cos,
            "shared_vocab_size": n_shared,
            "cosine_bar": ref_self_cos - MARGIN,
            "date": datetime.date.today().isoformat(),
            "pass_bar": (
                "topica-vs-reference Hungarian-aligned topic-word cosine (over the "
                "shared vocabulary) >= reference seed-to-seed self cosine - margin"
            ),
            "kind": "cross-implementation (fastopic package reference)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  reference self cosine   : {ref_self_cos:.4f}")
    print(f"  topica vs reference cos : {topica_cos:.4f}  (bar {ref_self_cos - MARGIN:.4f}, "
          f"shared vocab {n_shared})")


def _ref_topic_word_from_beta(beta, ref_vocab, target_vocab):
    index = {w: i for i, w in enumerate(ref_vocab)}
    out = np.zeros((beta.shape[0], len(target_vocab)), dtype=np.float64)
    for j, w in enumerate(target_vocab):
        i = index.get(w)
        if i is not None:
            out[:, j] = beta[:, i]
    return out


def _fastopic_version() -> str:
    try:
        import importlib.metadata as _md
        return f"fastopic {_md.version('fastopic')} (all-MiniLM-L6-v2 embeddings)"
    except Exception:
        return "fastopic (version unknown)"


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    gold_tw = arrays["topic_word"].astype(np.float64)
    vocab = [str(w) for w in arrays["vocab"]]
    emb = arrays["embeddings"].astype(np.float64)
    token_docs = harness.lines_to_docs(str(arrays["corpus"]))
    bar = float(meta["cosine_bar"])
    self_cos = float(meta["reference_self_cosine"])

    t_tw, t_vocab = _topica_topic_word(token_docs, emb, GOLD_SEED)
    cos, n_shared = _align_on_shared_vocab(gold_tw, vocab, t_tw, t_vocab)

    result = {
        "cosine": cos,
        "reference_self_cosine": self_cos,
        "bar": bar,
        "shared_vocab_size": n_shared,
        "margin_over_bar": cos - bar,
        "passes": bool(cos >= bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica vs reference topic-word cosine : {cos:.4f} "
              f"(reference self {self_cos:.4f}, bar {bar:.4f}, shared vocab {n_shared})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
