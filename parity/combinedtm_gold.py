"""Committed-gold parity for topica CombinedTM vs a PyTorch AVITM reference (#271, Wave 1).

CombinedTM (Bianchi, Terragni & Hovy 2021) is ProdLDA / AVITM with one change: the
encoder reads the normalized bag of words *concatenated with* a per-document
embedding. topica's network is hand-coded in Rust (no autograd); the reference is
the shared PyTorch AVITM backbone in ``parity/refs/avitm.py`` plus the ProdLDA
decoder, exactly as in the live script ``parity/combinedtm_compare.py``. RNG,
initialization, and autograd-vs-hand-coded backward differ, so agreement is
*statistical*, on a SHARED task: the same planted-block token corpus, the same
per-document one-hot-plus-noise embeddings, the same K and optimizer schedule.

The corpus + embeddings + config are taken verbatim from
``parity/combinedtm_compare.py`` (a clean planted-block design: K word blocks, each
document drawn from one block and carrying an embedding that one-hot-encodes its
block). On this well-identified design the two implementations land on essentially
the same topic-word matrix (aligned cosine ~0.97 — as tight as the reference
reproduces itself across seeds), so the gold freezes the reference topic-word
matrix and the reference's own seed-to-seed aligned-cosine floor, and the bar is
``topica-vs-reference aligned cosine >= reference-self cosine - margin`` — the same
"benchmark against the reference's own reproducibility" logic the gdmr/keyatm gold
use, not an invented threshold.

Two phases (mirrors parity/gdmr_gold.py):

  * ``--regenerate`` (needs torch): fits the AVITM reference twice (two seeds) to
    measure its topic-word self-cosine floor, freezes one run's topic-word matrix,
    the planted vocab, and the shared token corpus, and writes the committed gold
    (``parity/combinedtm_gold.npz`` + ``.json``).
  * default (no torch): loads the committed gold, fits topica CombinedTM on the
    same corpus + embeddings, Hungarian-aligns its topic-word matrix to the
    reference's, and checks the bar.

Run directly::

    python parity/combinedtm_gold.py               # offline compare against committed gold
    python parity/combinedtm_gold.py --regenerate  # run torch once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "combinedtm"

# Corpus + config (verbatim from parity/combinedtm_compare.py).
NUM_BLOCKS = 5
BLOCK = 10
NUM_DOCS = 600
DOC_LEN = 25
EMB_DIM = NUM_BLOCKS
NUM_TOPICS = NUM_BLOCKS
TOP_N = 10

ALPHA = 1.0
HIDDEN = 100
DROPOUT = 0.2
EPOCHS = 150
BATCH = 200
LR = 0.002

GOLD_SEED = 0
FLOOR_SEED = 1
# Pass margin below the reference's own seed-to-seed topic-word cosine floor.
MARGIN = 0.10


# --------------------------------------------------------------------------- #
# Shared corpus + embeddings (verbatim from the live script)
# --------------------------------------------------------------------------- #
def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(NUM_BLOCKS) for i in range(BLOCK)]
    token_docs, embs, labels = [], [], []
    for d in range(NUM_DOCS):
        b = d % NUM_BLOCKS
        token_docs.append([f"b{b}w{int(rng.integers(BLOCK))}" for _ in range(DOC_LEN)])
        e = np.zeros(EMB_DIM)
        e[b] = 3.0
        e = e + rng.normal(0.0, 0.1, EMB_DIM)
        embs.append(e)
        labels.append(b)
    return token_docs, np.asarray(embs, dtype=np.float64), np.asarray(labels), vocab


def _counts_matrix(token_docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(token_docs), len(vocab)), dtype=np.float32)
    for d, toks in enumerate(token_docs):
        for t in toks:
            j = index.get(t)
            if j is not None:
                m[d, j] += 1.0
    return m


# --------------------------------------------------------------------------- #
# PyTorch AVITM reference (CombinedTM: encoder reads [norm_bow | emb])
# --------------------------------------------------------------------------- #
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _torch_version() -> str:
    try:
        import torch
        return f"torch {torch.__version__} (AVITM reference, parity/refs/avitm.py)"
    except Exception:
        return "torch (version unknown)"


def _train_reference(counts: np.ndarray, emb: np.ndarray, k: int, seed: int):
    import torch
    import torch.nn as nn

    from refs.avitm import (
        build_encoder,
        dirichlet_laplace_kl,
        laplace_prior,
        reparameterize,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    v = counts.shape[1]
    e = emb.shape[1]

    class CombinedTMRef(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = build_encoder(v + e, k, HIDDEN, dropout=DROPOUT)
            self.drop = nn.Dropout(DROPOUT)
            self.beta = nn.Linear(k, v, bias=False)
            self.bn_dec = nn.BatchNorm1d(v, affine=False, eps=1e-5, momentum=0.1)

        def encode(self, xn, em):
            return self.enc.encode(torch.cat([xn, em], dim=1))

        def forward(self, xn, em):
            mu, lv = self.encode(xn, em)
            z = reparameterize(mu, lv)
            theta = self.drop(torch.softmax(z, dim=1))
            logits = self.bn_dec(self.beta(theta))
            return torch.log_softmax(logits, dim=1), mu, lv

    model = CombinedTMRef()
    prior_mu, prior_var = laplace_prior(k, ALPHA)

    cnt = torch.tensor(counts, dtype=torch.float32)
    norm = cnt / cnt.sum(1, keepdim=True).clamp_min(1.0)
    em = torch.tensor(emb, dtype=torch.float32)
    n = cnt.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            recon, mu, lv = model(norm[idx], em[idx])
            rl = -(cnt[idx] * recon).sum(1)
            kl = dirichlet_laplace_kl(mu, lv, prior_mu, prior_var, k)
            (rl + kl).mean().backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        beta = model.beta.weight.detach().t()  # K x V
        topic_word = torch.softmax(beta, dim=1).numpy()
    return topic_word.astype(np.float64)


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _topica_topic_word(token_docs, embs, vocab, seed):
    import topica

    tm = topica.models.CombinedTM(
        num_topics=NUM_TOPICS, alpha=ALPHA, hidden_size=HIDDEN, dropout=DROPOUT,
        batch_size=BATCH, lr=LR, seed=seed,
    )
    tm.fit(token_docs, embs, iters=EPOCHS)
    tw = np.asarray(tm.topic_word, dtype=np.float64)
    order = [list(tm.vocabulary).index(w) for w in vocab]
    return tw[:, order]


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not _torch_available():
        print("torch not available; cannot regenerate.")
        sys.exit(1)

    token_docs, embs, labels, vocab = _make_data()
    counts = _counts_matrix(token_docs, vocab)

    tw_gold = _train_reference(counts, embs, NUM_TOPICS, GOLD_SEED)
    tw_floor = _train_reference(counts, embs, NUM_TOPICS, FLOOR_SEED)
    ref_self_cos, _ = harness.align_cosine(tw_gold, tw_floor)

    # topica summary at regenerate time for the provenance log.
    t_tw = _topica_topic_word(token_docs, embs, vocab, GOLD_SEED)
    topica_cos, _ = harness.align_cosine(tw_gold, t_tw)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word": tw_gold,
            "vocab": np.array(vocab, dtype=object),
            "labels": labels.astype(np.int64),
            "embeddings": embs.astype(np.float64),
            "corpus": harness.docs_to_lines(token_docs),
        },
        meta={
            "reference": _torch_version(),
            "model": "CombinedTM (Bianchi, Terragni & Hovy 2021)",
            "corpus": ("synthetic planted-block, 600 docs x 25 tokens, K=5 blocks of "
                       "10 words, per-doc one-hot+noise embedding (from "
                       "parity/combinedtm_compare.py)"),
            "num_docs": len(token_docs),
            "vocab_size": len(vocab),
            "emb_dim": EMB_DIM,
            "num_topics": NUM_TOPICS,
            "alpha": ALPHA,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "seeds": {"gold": GOLD_SEED, "noise_floor": FLOOR_SEED},
            "margin": MARGIN,
            "reference_self_cosine": ref_self_cos,
            "topica_vs_reference_cosine": topica_cos,
            "cosine_bar": ref_self_cos - MARGIN,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("topica-vs-reference Hungarian-aligned topic-word cosine >= "
                         "reference seed-to-seed self cosine - margin"),
            "kind": "cross-implementation (PyTorch AVITM reference)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  reference self cosine   : {ref_self_cos:.4f}")
    print(f"  topica vs reference cos : {topica_cos:.4f}  (bar {ref_self_cos - MARGIN:.4f})")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    gold_tw = arrays["topic_word"].astype(np.float64)
    vocab = [str(w) for w in arrays["vocab"]]
    embs = arrays["embeddings"].astype(np.float64)
    token_docs = harness.lines_to_docs(str(arrays["corpus"]))
    bar = float(meta["cosine_bar"])
    self_cos = float(meta["reference_self_cosine"])

    t_tw = _topica_topic_word(token_docs, embs, vocab, GOLD_SEED)
    cos, _ = harness.align_cosine(gold_tw, t_tw)

    result = {
        "cosine": cos,
        "reference_self_cosine": self_cos,
        "bar": bar,
        "margin_over_bar": cos - bar,
        "passes": bool(cos >= bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica vs reference topic-word cosine : {cos:.4f} "
              f"(reference self {self_cos:.4f}, bar {bar:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
