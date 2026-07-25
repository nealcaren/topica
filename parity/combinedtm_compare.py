"""Cross-implementation check for topica's CombinedTM against a faithful PyTorch
reference of the contextualized topic model (Bianchi, Terragni & Hovy 2021).

CombinedTM is ProdLDA (AVITM; Srivastava & Sutton 2017) with one change: the
encoder reads the raw bag of words *concatenated with* a caller-supplied document
embedding that is first projected into vocabulary space by a learned ``adapt_bert``
``Linear(E, V)`` layer (first encoder layer ``Linear(2V, hidden)``; issue #503).
The decoder is unchanged -- a product-of-experts
``softmax(beta . theta)`` reconstructing the bag of words over the V-vocabulary.
The embedding is only an encoder input; it is never reconstructed.

The reference here reuses the shared AVITM backbone in ``refs.avitm`` (softplus
encoder built ``fc1, fc2, mu, lv`` with affine-free batchnorm heads, the Laplace
approximation to the Dirichlet prior, the reparameterization trick, and the
Dirichlet-Laplace KL) and the same ProdLDA decoder + training loop as
``prodlda_compare.py``. The only difference from ProdLDA is the encoder input
width and what it is fed: ``build_encoder(v_in = 2V, ...)`` fed
``concat([raw_bow, adapt_bert(doc_emb)], dim=1)``. topica's network is hand-coded in Rust
with no autograd; PyTorch differentiates the same graph. Initialization, RNG, and
autograd-vs-hand-coded backward differ, so exact agreement is impossible -- we
hold them to a statistical-equivalence bar on a shared task: the SAME planted
token corpus, the SAME per-document embeddings, the same topic count and
optimizer schedule.

Metrics, computed identically for both with topica's own coherence:
  - topic coherence (c_v, c_npmi) over the shared token corpus
  - topic diversity (TU): fraction of distinct words across the top lists
  - doc-topic agreement: NMI of the argmax topic vs the true planted block, and
    cross-NMI between the two implementations

Skips cleanly when torch is unavailable. Run directly:

    python parity/combinedtm_compare.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refs.avitm import (  # noqa: E402
    build_encoder,
    dirichlet_laplace_kl,
    laplace_prior,
    reparameterize,
)

# --- Shared task and optimizer schedule --------------------------------------
#
# Planted-block synthetic corpus (mirrors tests/test_combinedtm.py): K word
# blocks; each document draws its tokens from one block and carries an embedding
# that one-hot-encodes its block (plus small noise), so the contextual signal is
# clean and CombinedTM can recover the planted structure.

NUM_BLOCKS = 5
BLOCK = 10           # words per block -> vocab = NUM_BLOCKS * BLOCK
NUM_DOCS = 600
DOC_LEN = 25
EMB_DIM = NUM_BLOCKS  # one-hot over blocks (+ noise)
NUM_TOPICS = NUM_BLOCKS
TOP_N = 10

# Shared optimizer schedule (kept modest so the parity run finishes in minutes).
ALPHA = 1.0
HIDDEN = 100
DROPOUT = 0.2
EPOCHS = 150
BATCH = 200
LR = 0.002

SEEDS = (0, 1, 2)


def available() -> bool:
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def load(seed: int = 0):
    """Build a planted-block token corpus and per-document embeddings. The
    embedding one-hot-encodes the document's dominant block (plus small Gaussian
    noise), so it cleanly carries the planted topic structure. Returns
    ``(token_docs, embeddings, labels, vocab)``; data is fixed across fit seeds so
    topica and the reference see exactly the same task each run."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(NUM_BLOCKS) for i in range(BLOCK)]
    token_docs, embs, labels = [], [], []
    for d in range(NUM_DOCS):
        b = d % NUM_BLOCKS
        token_docs.append(
            [f"b{b}w{int(rng.integers(BLOCK))}" for _ in range(DOC_LEN)]
        )
        e = np.zeros(EMB_DIM)
        e[b] = 3.0
        e = e + rng.normal(0.0, 0.1, EMB_DIM)
        embs.append(e)
        labels.append(b)
    return (
        token_docs,
        np.asarray(embs, dtype=np.float64),
        np.asarray(labels),
        vocab,
    )


# --- PyTorch reference: CombinedTM --------------------------------------------
#
# Identical to the ProdLDA reference (prodlda_compare.py) except the encoder is
# built at ``v_in = 2V`` and fed ``concat([raw_bow, adapt_bert(emb)])`` with
# ``adapt_bert = Linear(E, V)`` (Bianchi et al.'s CombinedInferenceNetwork, #503).
# The decoder, Laplace-Dirichlet prior, reparameterization, KL, and batchnorm are
# unchanged and come from ``refs.avitm``; nothing is re-implemented here.


def _build_reference(v: int, e: int, k: int):
    import torch
    import torch.nn as nn

    class CombinedTMRef(nn.Module):
        def __init__(self):
            super().__init__()
            # CombinedTM (Bianchi et al., #503): encoder input is
            # [raw_bow (V) | adapt_bert(doc_emb) (V)] with adapt_bert = Linear(E, V),
            # so the first encoder layer is Linear(2V, hidden).
            self.adapt_bert = nn.Linear(e, v)
            self.enc = build_encoder(v + v, k, HIDDEN, dropout=DROPOUT)
            self.drop = nn.Dropout(DROPOUT)
            self.beta = nn.Linear(k, v, bias=False)  # weight V x K; logits = theta @ W^T
            self.bn_dec = nn.BatchNorm1d(v, affine=False, eps=1e-5, momentum=0.1)

        def encode(self, x_raw, emb):
            return self.enc.encode(torch.cat([x_raw, self.adapt_bert(emb)], dim=1))

        def forward(self, x_raw, emb):
            mu, lv = self.encode(x_raw, emb)
            z = reparameterize(mu, lv)
            theta = self.drop(torch.softmax(z, dim=1))
            logits = self.bn_dec(self.beta(theta))
            return torch.log_softmax(logits, dim=1), mu, lv

    return CombinedTMRef()


def _train_reference(counts: np.ndarray, emb: np.ndarray, k: int, seed: int):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    v = counts.shape[1]
    e = emb.shape[1]
    model = _build_reference(v, e, k)
    prior_mu, prior_var = laplace_prior(k, ALPHA)

    cnt = torch.tensor(counts, dtype=torch.float32)
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
            # Encoder reads the RAW counts (not length-normalized), matching upstream.
            recon, mu, lv = model(cnt[idx], em[idx])
            rl = -(cnt[idx] * recon).sum(1)
            kl = dirichlet_laplace_kl(mu, lv, prior_mu, prior_var, k)
            (rl + kl).mean().backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        beta = model.beta.weight.detach().t()  # K x V
        topic_word = torch.softmax(beta, dim=1).numpy()
        mu, _ = model.encode(cnt, em)
        assign = torch.softmax(mu, dim=1).argmax(1).numpy()
    return topic_word, assign


# --- Shared metrics ----------------------------------------------------------


def _coherence(topics, token_docs, measure):
    import topica

    return float(
        np.mean(topica.coherence(topics, token_docs, coherence_type=measure, topn=TOP_N))
    )


def _diversity(topics):
    words = [w for t in topics for w in t[:TOP_N]]
    return len(set(words)) / max(1, len(words))


def _counts_matrix(token_docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(token_docs), len(vocab)), dtype=np.float32)
    for d, toks in enumerate(token_docs):
        for t in toks:
            j = index.get(t)
            if j is not None:
                m[d, j] += 1.0
    return m


def _topica_fit(token_docs, embs, seed):
    import topica

    tm = topica.CombinedTM(
        num_topics=NUM_TOPICS, alpha=ALPHA, hidden_size=HIDDEN, dropout=DROPOUT,
        batch_size=BATCH, lr=LR, seed=seed,
    )
    tm.fit(token_docs, embs, iters=EPOCHS)
    theta = np.asarray(tm.transform(token_docs, embs))
    topics = [[w for w, _ in tm.top_words(TOP_N, topic=t)] for t in range(tm.num_topics)]
    return topics, theta.argmax(1)


def _ref_fit(counts, embs, vocab, seed):
    tw, assign = _train_reference(counts, embs, NUM_TOPICS, seed)
    topics = [[vocab[j] for j in row.argsort()[::-1][:TOP_N]] for row in tw]
    return topics, assign


def run(verbose: bool = True) -> dict:
    from sklearn.metrics import normalized_mutual_info_score

    token_docs, embs, labels, vocab = load()
    counts = _counts_matrix(token_docs, vocab)

    # Bit-exact agreement is impossible across frameworks; a single seed is also
    # misleading (both models swing topic-to-topic). Average over seeds and report
    # the spread, so the verdict reflects the model, not one lucky/unlucky draw.
    rows = {k: [] for k in ("t_cv", "r_cv", "t_npmi", "r_npmi", "t_div", "r_div",
                            "t_nmi", "r_nmi", "cross_nmi")}
    for seed in SEEDS:
        t_topics, t_assign = _topica_fit(token_docs, embs, seed)
        r_topics, r_assign = _ref_fit(counts, embs, vocab, seed)
        rows["t_cv"].append(_coherence(t_topics, token_docs, "c_v"))
        rows["r_cv"].append(_coherence(r_topics, token_docs, "c_v"))
        rows["t_npmi"].append(_coherence(t_topics, token_docs, "c_npmi"))
        rows["r_npmi"].append(_coherence(r_topics, token_docs, "c_npmi"))
        rows["t_div"].append(_diversity(t_topics))
        rows["r_div"].append(_diversity(r_topics))
        rows["t_nmi"].append(float(normalized_mutual_info_score(labels, t_assign)))
        rows["r_nmi"].append(float(normalized_mutual_info_score(labels, r_assign)))
        rows["cross_nmi"].append(float(normalized_mutual_info_score(t_assign, r_assign)))

    def ms(key):
        a = np.array(rows[key])
        return float(a.mean()), float(a.std())

    metrics = {
        "num_docs": len(token_docs),
        "vocab": len(vocab),
        "emb_dim": EMB_DIM,
        "seeds": list(SEEDS),
        "topica_c_v": ms("t_cv"),
        "reference_c_v": ms("r_cv"),
        "topica_c_npmi": ms("t_npmi"),
        "reference_c_npmi": ms("r_npmi"),
        "topica_diversity": ms("t_div"),
        "reference_diversity": ms("r_div"),
        "topica_label_nmi": ms("t_nmi"),
        "reference_label_nmi": ms("r_nmi"),
        "cross_nmi": ms("cross_nmi"),
    }
    if verbose:
        print(f"  CombinedTM parity: {metrics['num_docs']} docs, "
              f"V={metrics['vocab']}, E={metrics['emb_dim']}, K={NUM_TOPICS}")
        print(f"  {'metric':22s} {'topica (mean±sd)':>22s} {'reference (mean±sd)':>22s}")
        for label, tk, rk in [
            ("c_v coherence", "topica_c_v", "reference_c_v"),
            ("c_npmi coherence", "topica_c_npmi", "reference_c_npmi"),
            ("diversity (TU)", "topica_diversity", "reference_diversity"),
            ("label NMI", "topica_label_nmi", "reference_label_nmi"),
        ]:
            tm_, ts = metrics[tk]
            rm_, rs = metrics[rk]
            print(f"  {label:22s} {tm_:>13.3f} ±{ts:.3f} {rm_:>13.3f} ±{rs:.3f}")
        cm, cs = metrics["cross_nmi"]
        print(f"  {'cross-NMI (topica<->ref)':22s} {cm:>13.3f} ±{cs:.3f}")
        gap = metrics["topica_c_npmi"][0] - metrics["reference_c_npmi"][0]
        print(
            f"\n  c_npmi gap {gap:+.4f} vs reference seed-to-seed sd "
            f"{metrics['reference_c_npmi'][1]:.4f}: "
            f"{'within noise' if abs(gap) <= 2 * metrics['reference_c_npmi'][1] else 'systematic'}"
        )
    return metrics


if __name__ == "__main__":
    if not available():
        print("SKIP: torch / sklearn not installed.")
    else:
        run(verbose=True)
