"""Cross-implementation check for topica's ProdLDA against a faithful PyTorch
reference of AVITM (Srivastava & Sutton 2017).

Both are the same model: an autoencoding-variational LDA whose word-level mixture
is a product of experts, ``softmax(beta . theta)`` with unnormalized beta. The
reference here is a compact PyTorch implementation built to the paper's recipe and
matching topica's exact formulation -- softplus encoder, affine-free batch
normalization on the mean/logvar heads and the decoder, the Laplace approximation
to the Dirichlet prior (eq. 6), and high-momentum Adam (beta1 = 0.99). topica's
network is hand-coded in Rust with no autograd; PyTorch differentiates the same
graph. Initialization, RNG, and the autograd vs hand-coded backward differ, so
exact agreement is impossible -- we hold them to a statistical-equivalence bar on
a shared task: the SAME tokenized corpus, the same vocabulary, the same topic
count and optimizer schedule.

Metrics, computed identically for both with topica's own coherence:
  - topic coherence (c_v, c_npmi) over the shared token corpus
  - topic diversity (TU): fraction of distinct words across the top lists
  - doc-topic agreement: NMI of the argmax topic vs the true newsgroup label, and
    cross-NMI between the two implementations

Skips cleanly when torch / sklearn are unavailable or the corpus cannot be
fetched. Run directly:

    python parity/prodlda_compare.py
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

# Shared optimizer schedule (kept modest so the parity run finishes in minutes).
ALPHA = 1.0
HIDDEN = 100
DROPOUT = 0.2
EPOCHS = 120
BATCH = 200
LR = 0.002


def available() -> bool:
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
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
    cv = CountVectorizer(
        min_df=10, max_df=0.4, stop_words="english", token_pattern=r"(?u)\b[a-z]{3,}\b"
    )
    cv.fit([d.lower() for d in raw])
    vset = set(cv.get_feature_names_out())
    analyzer = cv.build_analyzer()
    token_docs, mask = [], []
    for d in raw:
        toks = [t for t in analyzer(d.lower()) if t in vset]
        ok = len(toks) >= 5
        mask.append(ok)
        if ok:
            token_docs.append(toks)
    return token_docs, labels[np.array(mask)], sorted(vset)


# --- PyTorch reference: AVITM / ProdLDA --------------------------------------
#
# The encoder backbone (fc1, fc2, mu, lv + affine-free batchnorm heads), the
# Laplace-Dirichlet prior, the reparameterization trick, and the KL are shared
# with the InfoCTM parity reference via ``refs.avitm``. The decoder and training
# loop below are ProdLDA-specific. The encoder submodule is constructed before
# ``beta``, so the RNG draw order (fc1, fc2, mu, lv, beta) is unchanged.


def _build_reference(v: int, k: int):
    import torch
    import torch.nn as nn

    class ProdLDARef(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = build_encoder(v, k, HIDDEN, dropout=DROPOUT)
            self.drop = nn.Dropout(DROPOUT)
            self.beta = nn.Linear(k, v, bias=False)  # weight is V x K; logits = theta @ W^T
            self.bn_dec = nn.BatchNorm1d(v, affine=False, eps=1e-5, momentum=0.1)

        def encode(self, xn):
            return self.enc.encode(xn)

        def forward(self, xn):
            mu, lv = self.encode(xn)
            z = reparameterize(mu, lv)
            theta = self.drop(torch.softmax(z, dim=1))
            logits = self.bn_dec(self.beta(theta))
            return torch.log_softmax(logits, dim=1), mu, lv

    return ProdLDARef()


def _train_reference(counts: np.ndarray, k: int, seed: int):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    v = counts.shape[1]
    model = _build_reference(v, k)
    prior_mu, prior_var = laplace_prior(k, ALPHA)

    cnt = torch.tensor(counts, dtype=torch.float32)
    norm = cnt / cnt.sum(1, keepdim=True).clamp_min(1.0)
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
            recon, mu, lv = model(norm[idx])
            rl = -(cnt[idx] * recon).sum(1)
            kl = dirichlet_laplace_kl(mu, lv, prior_mu, prior_var, k)
            (rl + kl).mean().backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        beta = model.beta.weight.detach().t()  # K x V
        topic_word = torch.softmax(beta, dim=1).numpy()
        mu, _ = model.encode(norm)
        assign = torch.softmax(mu, dim=1).argmax(1).numpy()
    return topic_word, assign


# --- Shared metrics ----------------------------------------------------------


def _coherence(topics, token_docs, measure):
    import topica

    return float(np.mean(topica.coherence(topics, token_docs, coherence_type=measure, topn=TOP_N)))


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


SEEDS = (0, 1, 2)


def _topica_fit(token_docs, seed):
    import topica

    tm = topica.ProdLDA(
        num_topics=NUM_TOPICS, alpha=ALPHA, hidden_size=HIDDEN, dropout=DROPOUT,
        batch_size=BATCH, lr=LR, seed=seed,
    )
    tm.fit(token_docs, iters=EPOCHS)
    theta = np.asarray(tm.transform(token_docs))
    topics = [[w for w in tm.top_words(TOP_N, topic=t)] for t in range(tm.num_topics)]
    return topics, theta.argmax(1)


def _ref_fit(counts, vocab, seed):
    tw, assign = _train_reference(counts, NUM_TOPICS, seed)
    topics = [[vocab[j] for j in row.argsort()[::-1][:TOP_N]] for row in tw]
    return topics, assign


def run(verbose: bool = True) -> dict:
    from sklearn.metrics import normalized_mutual_info_score

    token_docs, labels, vocab = load()
    counts = _counts_matrix(token_docs, vocab)

    # Bit-exact agreement is impossible across frameworks; a single seed is also
    # misleading (both models swing topic-to-topic). Average over seeds and report
    # the spread, so the verdict reflects the model, not one lucky/unlucky draw.
    rows = {k: [] for k in ("t_cv", "r_cv", "t_npmi", "r_npmi", "t_div", "r_div",
                            "t_nmi", "r_nmi", "cross_nmi")}
    for seed in SEEDS:
        t_topics, t_assign = _topica_fit(token_docs, seed)
        r_topics, r_assign = _ref_fit(counts, vocab, seed)
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
        print(f"  {'cross-NMI (topica↔ref)':22s} {cm:>13.3f} ±{cs:.3f}")
        gap = metrics["topica_c_npmi"][0] - metrics["reference_c_npmi"][0]
        print(
            f"\n  c_npmi gap {gap:+.4f} vs reference seed-to-seed sd "
            f"{metrics['reference_c_npmi'][1]:.4f}: "
            f"{'within noise' if abs(gap) <= 2 * metrics['reference_c_npmi'][1] else 'systematic'}"
        )
    return metrics


if __name__ == "__main__":
    if not available():
        print("torch / sklearn not installed; skipping.")
    else:
        run(verbose=True)
