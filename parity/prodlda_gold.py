"""Committed-gold parity for topica ProdLDA vs a PyTorch AVITM reference (#271, Wave 1).

ProdLDA (Srivastava & Sutton 2017's AVITM) is an autoencoding-variational LDA whose
word-level mixture is a product of experts, ``softmax(beta . theta)``. topica's
network is hand-coded in Rust (no autograd); the reference is the shared PyTorch
AVITM backbone in ``parity/refs/avitm.py`` plus the ProdLDA decoder + training loop,
exactly as in the live script ``parity/prodlda_compare.py``. RNG, initialization,
and autograd-vs-hand-coded backward differ, so agreement is *statistical*, on a
SHARED real-text task (a 20-newsgroups subset, tokenized identically for both).

**Bar logic (objective/coherence parity, NOT topic-word cosine).** This is the bar
the live ``prodlda_compare.py`` already uses: on real text the two implementations
swing topic-to-topic and a topic-word cosine is not a faithful target, so the live
script holds them to coherence equivalence: topica must be no *worse* than the
reference, within the reference's own seed-to-seed noise band (topica being *more*
coherent is a pass, not a failure -- and in practice topica is sharper, landing
above the reference mean). This gold freezes the reference's mean c_npmi and its
seed-to-seed standard deviation over three seeds, the shared token corpus, and (for
a non-vacuous shuffle check + provenance) one reference run's topic-word matrix. The
offline test refits topica once, scores its c_npmi with topica's own coherence on
the frozen corpus, and asserts it clears the one-sided lower bar
``reference_mean - 2 * reference_sd``. Honest flag: this is an objective-parity bar,
not a topic recovery bar -- the two implementations are coherence-equivalent (topica
slightly sharper), not topic-identical, on real text.

To keep the committed gold small (< 1 MB) and the refit affordable, the corpus is a
three-group, ``min_df=20`` 20NG subset (~1.5k docs, V~1.1k) rather than the live
script's five-group corpus; the tokenization pipeline is otherwise identical.

Two phases (mirrors parity/combinedtm_gold.py):

  * ``--regenerate`` (needs torch + sklearn): fits the AVITM reference at three seeds
    for the c_npmi mean + sd, freezes one run's topic-word matrix, the vocab, and the
    shared token corpus, and writes the committed gold.
  * default (no torch): loads the committed gold, fits topica ProdLDA on the frozen
    corpus, and checks the coherence-parity bar.

Run directly::

    python parity/prodlda_gold.py               # offline compare against committed gold
    python parity/prodlda_gold.py --regenerate  # run torch once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "prodlda"

# Smaller-than-live 20NG subset so the committed gold stays < 1 MB.
GROUPS = ["rec.sport.baseball", "sci.space", "talk.politics.guns"]
MIN_DF = 20
NUM_TOPICS = 10
TOP_N = 10
DATA_SEED = 0

ALPHA = 1.0
HIDDEN = 100
DROPOUT = 0.2
EPOCHS = 120
BATCH = 200
LR = 0.002

# Reference seeds for the c_npmi noise floor; topica's own fit seed.
REF_SEEDS = (0, 1, 2)
TOPICA_SEED = 0
# topica must clear (reference_mean - SD_MULT * reference_sd): no worse than the
# reference within its own seed-to-seed noise. Being more coherent is a pass.
SD_MULT = 2.0


def _torch_available() -> bool:
    try:
        import sklearn  # noqa: F401
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


def _load_corpus():
    """Tokenize the 20NG subset exactly as the live script, smaller groups/min_df."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    data = fetch_20newsgroups(
        subset="train", categories=GROUPS, remove=("headers", "footers", "quotes"),
        random_state=DATA_SEED,
    )
    raw = [d.strip() for d in data.data]
    labels = np.array(data.target)
    keep = [i for i, d in enumerate(raw) if len(d.split()) >= 20]
    raw, labels = [raw[i] for i in keep], labels[keep]
    cv = CountVectorizer(
        min_df=MIN_DF, max_df=0.4, stop_words="english",
        token_pattern=r"(?u)\b[a-z]{3,}\b",
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


def _counts_matrix(token_docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(token_docs), len(vocab)), dtype=np.float32)
    for d, toks in enumerate(token_docs):
        for t in toks:
            j = index.get(t)
            if j is not None:
                m[d, j] += 1.0
    return m


def _coherence(topics, token_docs):
    import topica
    return float(np.mean(topica.coherence(topics, token_docs, coherence_type="c_npmi", topn=TOP_N)))


# --------------------------------------------------------------------------- #
# PyTorch AVITM reference
# --------------------------------------------------------------------------- #
def _train_reference(counts: np.ndarray, k: int, seed: int):
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

    class ProdLDARef(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = build_encoder(v, k, HIDDEN, dropout=DROPOUT)
            self.drop = nn.Dropout(DROPOUT)
            self.beta = nn.Linear(k, v, bias=False)
            self.bn_dec = nn.BatchNorm1d(v, affine=False, eps=1e-5, momentum=0.1)

        def forward(self, xn):
            mu, lv = self.enc.encode(xn)
            z = reparameterize(mu, lv)
            theta = self.drop(torch.softmax(z, dim=1))
            logits = self.bn_dec(self.beta(theta))
            return torch.log_softmax(logits, dim=1), mu, lv

    model = ProdLDARef()
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
    return topic_word.astype(np.float64)


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _topica_topics(token_docs, seed):
    import topica

    tm = topica.ProdLDA(
        num_topics=NUM_TOPICS, alpha=ALPHA, hidden_size=HIDDEN, dropout=DROPOUT,
        batch_size=BATCH, lr=LR, seed=seed,
    )
    tm.fit(token_docs, iters=EPOCHS)
    topics = [[w for w, _ in tm.top_words(TOP_N, topic=t)] for t in range(tm.num_topics)]
    return topics


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not _torch_available():
        print("torch / sklearn not available; cannot regenerate.")
        sys.exit(1)

    token_docs, labels, vocab = _load_corpus()
    counts = _counts_matrix(token_docs, vocab)

    ref_npmi = []
    tw_gold = None
    for s in REF_SEEDS:
        tw = _train_reference(counts, NUM_TOPICS, s)
        if tw_gold is None:
            tw_gold = tw
        topics = [[vocab[j] for j in row.argsort()[::-1][:TOP_N]] for row in tw]
        ref_npmi.append(_coherence(topics, token_docs))
    ref_mean = float(np.mean(ref_npmi))
    ref_sd = float(np.std(ref_npmi))

    # topica summary for the provenance log.
    t_topics = _topica_topics(token_docs, TOPICA_SEED)
    topica_npmi = _coherence(t_topics, token_docs)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word": tw_gold,
            "vocab": np.array(vocab, dtype=object),
            "labels": labels.astype(np.int64),
            "corpus": harness.docs_to_lines(token_docs),
        },
        meta={
            "reference": _torch_version(),
            "model": "ProdLDA / AVITM (Srivastava & Sutton 2017)",
            "corpus": (f"20-newsgroups subset {GROUPS}, min_df={MIN_DF}, headers/footers/"
                       "quotes removed, tokenized as parity/prodlda_compare.py"),
            "num_docs": len(token_docs),
            "vocab_size": len(vocab),
            "num_topics": NUM_TOPICS,
            "alpha": ALPHA,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "ref_seeds": list(REF_SEEDS),
            "topica_seed": TOPICA_SEED,
            "sd_mult": SD_MULT,
            "reference_c_npmi_mean": ref_mean,
            "reference_c_npmi_sd": ref_sd,
            "reference_c_npmi_per_seed": ref_npmi,
            "topica_c_npmi": topica_npmi,
            "lower_bar": ref_mean - SD_MULT * ref_sd,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("topica mean c_npmi >= reference_mean - sd_mult * reference_sd "
                         "(objective/coherence parity, one-sided: topica no worse than "
                         "the reference within its own seed noise; more coherent is a "
                         "pass -- NOT topic-word cosine, the live prodlda_compare bar)"),
            "kind": "cross-implementation (PyTorch AVITM reference); coherence-parity bar",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  reference c_npmi : {ref_mean:.4f} +/- {ref_sd:.4f} (per-seed {ref_npmi})")
    print(f"  topica   c_npmi : {topica_npmi:.4f}  "
          f"(lower bar {ref_mean - SD_MULT*ref_sd:.4f})")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    token_docs = harness.lines_to_docs(str(arrays["corpus"]))
    ref_mean = float(meta["reference_c_npmi_mean"])
    ref_sd = float(meta["reference_c_npmi_sd"])
    sd_mult = float(meta["sd_mult"])

    t_topics = _topica_topics(token_docs, TOPICA_SEED)
    topica_npmi = _coherence(t_topics, token_docs)

    lower_bar = ref_mean - sd_mult * ref_sd
    result = {
        "topica_c_npmi": topica_npmi,
        "reference_c_npmi_mean": ref_mean,
        "reference_c_npmi_sd": ref_sd,
        "lower_bar": lower_bar,
        "gap": topica_npmi - ref_mean,
        "passes": bool(topica_npmi >= lower_bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  reference c_npmi : {ref_mean:.4f} +/- {ref_sd:.4f}")
        print(f"  topica   c_npmi : {topica_npmi:.4f}  (lower bar {lower_bar:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(gap vs ref mean {result['gap']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
